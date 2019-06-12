#include "engine/mcts.h"

#include <algorithm>
#include <thread>

#include "absl/synchronization/mutex.h"
#include "glog/logging.h"

namespace zebra_go {
namespace {

// Parameters:
static const int kMaxSearchDepth = 2;
static const int kMaxRollouts = 100;

}  // namespace

struct MctsNode {
  enum NodeState {
    // A new node.
    STATE_NEW = 0,
    // The node is being scored by AsyncScorer. Don't score it again.
    STATE_SCORING = 1,
    // The node is scored.
    STATE_SCORED = 2,
    // AsyncScorer failed on this node. Don't score it again.
    STATE_FAILED = 3,
  };

  // Guard everything of this node.
  absl::Mutex mutex;
  NodeState state = STATE_NEW;
  MctsNode* parent = nullptr;          // Null if it is a root.
  int depth = 0;
  int visit_count = 0;

  std::unique_ptr<GoBoard> board;
  std::map<GoPosition, MctsNode*> children;
  // Evaluation result of the policy network.
  PolicyResult candidate_moves;
  // A combination of the value network's output and GoBoard.GetApproxPoints.
  ValueResult score;
  // Among all the leaf nodes under this node, the number of leaves where
  // black/white wins. 0: black, 1: white
  int win_count[2];

  MctsNode(std::unique_ptr<GoBoard> game_state, MctsNode* parent_node)
      : parent(parent_node),
        board(std::move(game_state)) {
    if (parent == nullptr) {
      depth = 0;
    } else {
      depth = parent->depth + 1;
    }
    win_count[0] = 0;
    win_count[1] = 0;
  }

  // Returns true if
  bool IsLeaf() {
    absl::MutexLock lock(&mutex);
    CHECK(state != STATE_NEW && state != STATE_SCORING);
    if (state == STATE_FAILED) return true;
    if (candidate_moves.empty()) return true;
    if (score.first) return true;  // current player should resign.
    return false;
  }

  // Current player should pass.
  bool ShouldPass() {
    absl::MutexLock lock(&mutex);
    CHECK(state != STATE_NEW && state != STATE_SCORING);
    return state == STATE_FAILED || candidate_moves.empty();
  }

  // Current player should resign.
  bool ShouldResign() {
    absl::MutexLock lock(&mutex);
    CHECK(state != STATE_NEW && state != STATE_SCORING);
    return state == STATE_SCORED && score.first;
  }

  void CollectRolloutResults() {
    {
      absl::MutexLock lock(&mutex);
      CHECK(state != STATE_NEW && state != STATE_SCORING);
    }
    if (children.empty()) {
      bool current_player_wins = true;
      if (ShouldResign()) {
        current_player_wins = false;
      }
      bool black_wins = (current_player_wins ==
                         (board->current_player() == COLOR_BLACK));
      win_count[0] = black_wins ? 1 : 0;
      win_count[1] = black_wins ? 0 : 1;
    } else {
      for (auto iter : children) {
        MctsNode* node = iter.second;
        node->CollectRolloutResults();
        win_count[0] += node->win_count[0];
        win_count[1] += node->win_count[1];
      }
    }
  }

  std::string DebugString(bool with_detail=false) const {
    std::string result;
    absl::StrAppend(&result, "state=", state, "\t");
    absl::StrAppend(&result, "#children=", children.size(), "\t");
    absl::StrAppend(&result, "depth=", depth, "\t");
    absl::StrAppend(&result, "#visits=", visit_count, "\t");
    absl::StrAppend(&result, "wins=", win_count[0], ", ", win_count[1], "\t");
    absl::StrAppend(&result, "score=", AsyncScorer::DebugString(score), "\t");
    if (with_detail) {
      absl::StrAppend(&result, "Candidate moves: ");
      for (const auto& move : candidate_moves) {
        absl::StrAppend(&result, ToString(move.first), ":", move.second, "; ");
      }
      absl::StrAppend(&result, "\tChildren: ");
      for (const auto& iter : children) {
        absl::StrAppend(&result, ToString(iter.first), ":",
                        AsyncScorer::DebugString(iter.second->score), "; ");
      }
    }
    return result;
  }
};

// The class is thread-safe.
class MctsStats {
 public:
  void LogEvent(const std::string& event);
  std::string DebugString();

private:
  absl::Mutex mutex_;
  std::map<std::string, int> counts_;
};

void MctsStats::LogEvent(const std::string& event) {
  absl::MutexLock lock(&mutex_);
  counts_[event] += 1;
}

std::string MctsStats::DebugString() {
  absl::MutexLock lock(&mutex_);
  std::string result;
  for (const auto iter : counts_) {
    absl::StrAppend(&result, iter.first, ": ", iter.second, "\n");
  }
  return result;
}

std::string MonteCarloSearchTree::SearchResult::DebugString() const {
  return "";
}

MonteCarloSearchTree::MonteCarloSearchTree(std::unique_ptr<GoBoard> board,
                                          int num_threads, AsyncScorer* scorer)
    :  num_threads_(num_threads),
       scorer_(scorer),
       search_stats_(new MctsStats()) {
  CHECK_GT(num_threads, 0);
  CHECK(scorer_ != nullptr);

  rollout_points_.reserve(400);
  root_ = new MctsNode(std::move(board), /*parent_node=*/nullptr);

  // setup search threads
  // for (int i = 0; i < num_threads; ++i) {
  // search_threads_.emplace_back(&MonteCarloSearchTree::SearchThread, this, i);
  // }
}

MonteCarloSearchTree::~MonteCarloSearchTree() {
}

void MonteCarloSearchTree::SyncScoreNode(MctsNode* node) {
  if (scorer_->SyncScoreGoState(*node->board,
                                &node->candidate_moves,
                                &node->score)) {
    node->state = MctsNode::STATE_SCORED;
  } else {
    node->state = MctsNode::STATE_FAILED;
  }
  if (node->IsLeaf()) {
    return;
  }

  for (const std::pair<GoPosition, float>& move : node->candidate_moves) {
    std::unique_ptr<GoBoard> state = node->board->Clone();
    std::vector<GoPosition> deads;
    if (!state->Move(move.first, /*estimate_territory=*/true, &deads)) {
      LOG(WARNING) << "The scorer returns an illegal move.";
      continue;
    }
    MctsNode* child = new MctsNode(std::move(state), node);
    node->children[move.first] = child;
  }
}

MonteCarloSearchTree::SearchResult MonteCarloSearchTree::Search(
    absl::Duration time_limit) {
  SearchResult result;

  // Search the first two levels synchronously
  // Level 0:
  LOG(INFO) << "Score root node:";
  SyncScoreNode(root_);
  LOG(INFO) << "root" << root_->DebugString();
  if (root_->ShouldPass()) {
    result.moves.push_back(std::make_pair(kMovePass, 0));
    return result;
  } else if (root_->ShouldResign()) {
    result.moves.push_back(std::make_pair(kMoveResign, 0));
    return result;
  }
  // Level 1:
  for (auto iter : root_->children) {
    SyncScoreNode(iter.second);
    LOG(INFO) << "Child " << iter.second->DebugString();
    for (auto child_iter : iter.second->children) {
      rollout_points_.push_back(child_iter.second);
    }
  }
  LOG(INFO) << "points " << rollout_points_.size();

  // Rollout and wait
  for (MctsNode* node : rollout_points_) {
    if (scorer_->SyncScoreGoState(*node->board,
                                  &node->candidate_moves,
                                  &node->score)) {
      node->state = MctsNode::STATE_SCORED;
    } else {
      node->state = MctsNode::STATE_FAILED;
    }
  }

  LOG(INFO) << "All leaf nodes are scored.";

  // std::this_thread::sleep_for(absl::ToChronoMilliseconds(time_limit));

  // Finalize:
  root_->CollectRolloutResults();
  LOG(INFO) << "Collect done.";
  for (MctsNode* node : rollout_points_) {
    if (node->IsLeaf()) {
      node->score.second = 1;  // TODO
    } else {
      float total = node->win_count[0] + node->win_count[1];
      float prob_black_wins = 0.5;
      if (total > 0) {
        prob_black_wins = node->win_count[0] / total;
      } else {
        LOG(WARNING) << "0 rollouts for this node";
      }
      if (node->board->current_player() == COLOR_BLACK) {
        node->score.second = prob_black_wins;
      } else {
        node->score.second = 1 - prob_black_wins;
      }
    }
  }

  // Select the best:
  LOG(INFO) << root_->DebugString(true);
  result.num_rollouts = root_->win_count[0] + root_->win_count[1];
  for (auto iter : root_->children) {
    MctsNode* node = iter.second;
    float max_child_score = -1;
    for (auto child_iter : node->children) {
      if (child_iter.second->score.second > max_child_score) {
        max_child_score = child_iter.second->score.second;
      }
    }
    node->score.second = 1 - max_child_score;
    result.moves.push_back(std::make_pair(iter.first, node->score.second));
  }
  LOG(INFO) << root_->DebugString(true);

  // Sort
  std::sort(result.moves.begin(), result.moves.end(),
            [](const std::pair<GoPosition, float>& a,
               const std::pair<GoPosition, float>& b) {
              return a.second > b.second;
            });

  return result;
}

}  // namespace zebra_go
