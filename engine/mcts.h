#ifndef ZEBRA_GO_ENGINE_MCTS_H_
#define ZEBRA_GO_ENGINE_MCTS_H_

#include "engine/go_game.h"

#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "engine/scorer.h"

namespace zebra_go {

struct MctsNode;
class MctsStats;

class MonteCarloSearchTree {
 public:
  MonteCarloSearchTree(std::unique_ptr<GoBoard> board, int num_threads,
                       AsyncScorer* scorer);
  ~MonteCarloSearchTree();

  struct SearchResult {
    std::vector<std::pair<GoPosition, float>> moves;
    int num_rollouts = 0;

    std::string DebugString() const;
  };

  SearchResult Search(absl::Duration time_limit);

 private:
  void SyncScoreNode(MctsNode* node);

  const int num_threads_;
  AsyncScorer* scorer_ = nullptr;

  MctsNode* root_ = nullptr;
  // Rollouts start from these nodes, the grand children of the root.
  std::vector<MctsNode*> rollout_points_;

  std::unique_ptr<MctsStats> search_stats_;
  std::vector<std::thread> search_threads_;
};

}  // namespace zebra_go

#endif  // ZEBRA_GO_ENGINE_MCTS_H_
