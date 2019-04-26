#include "engine/scorer.h"

#include <random>

#include "absl/memory/memory.h"
#include "absl/synchronization/notification.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "engine/utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(model, "", "Load model from this file.");
DEFINE_string(input_layer_name, "go_input_input", "");
DEFINE_string(output_layer_prefix, "go_output/0", "");

namespace zebra_go {
namespace {

// To run ScoreGoState's callbacks.
ThreadPool* GetScorerThreadPool() {
  static ThreadPool* g_thread_pool = new ThreadPool(3);
  return g_thread_pool;
}

void Normalize(PolicyResult* policy_result) {
  if (policy_result->empty()) return;
  float sum = 0.0f;
  for (const auto& p : *policy_result) {
    sum += p.second;
  }
  if (sum > 0.0f) {
    for (auto& p : *policy_result) {
      p.second = p.second / sum;
    }
  }
}

ValueResult SimpleEvaluate(const GoBoard& board) {
  const std::tuple<GoSizeT, GoSizeT, GoSizeT> points = board.GetApproxPoints();
  const int unknown = std::get<0>(points);
  const int black = std::get<1>(points);
  const int white = std::get<2>(points);
  float total_points = unknown + black + white;
  int current, opponent;
  if (board.current_player() == COLOR_BLACK) {
    current = black;
    opponent = white;
  } else {
    current = white;
    opponent = black;
  }
  bool should_resign = (black + white > 15 && current + unknown < opponent);
  float score = (current - opponent) / total_points;
  return std::make_pair(should_resign, score);
}

void ConvertToPolicyResult(const GoBoard& board,
                           const std::vector<float>&  policy_output,
                           PolicyResult* policy_result) {
  struct CompareSecond {
    bool operator()(const std::pair<GoPosition, float>& a,
                    const std::pair<GoPosition, float>& b) const {
      return a.second < b.second;
    }
  };

  // Selects top 20 legal positions with highest scores.
  TopK<std::pair<GoPosition, float>, CompareSecond> top_k(20);
  for (size_t i = 0; i < policy_output.size(); ++i) {
    GoPosition pos = board.Decode(i);
    if (board.IsLegalMove(pos)) {
      top_k.Insert(std::make_pair(pos, policy_output[i]));
    }
  }

  policy_result->clear();
  policy_result->reserve(top_k.elements().size());
  for (const auto& p : top_k.elements()) {
    policy_result->push_back(p);
  }
  if (policy_result->empty()) {
    LOG(WARNING) << "All moves in a policy output are illegal.";
  } else {
    Normalize(policy_result);
  }
}

ValueResult CombineValueResult(float value_output,
                               const ValueResult& fast_eval) {
  if (fast_eval.first) {
    // current player should resign.
    return fast_eval;
  } else {
    return std::make_pair(false, value_output);
  }
}

}  // namespace

std::string AsyncScorer::DebugString(const PolicyResult& policy_result) {
  std::vector<std::string> candidates;
  for (const auto& p : policy_result) {
    candidates.push_back(absl::StrCat("([", p.first.first, ",", p.first.second,
                                      "]: ", p.second, ")"));
  }
  return absl::StrJoin(candidates, ", ");
}

std::string AsyncScorer::DebugString(const ValueResult& p) {
  return absl::StrCat("(Score: ", p.second, ", ",
                      "should resign:", p.first, ")");
}

GoPosition AsyncScorer::SamplePolicy(const PolicyResult& policies) {
  static thread_local std::random_device g_random_device;
  static thread_local std::minstd_rand g_rng(g_random_device());

  CHECK(!policies.empty());
  float sum = 0.0f;
  for (const auto& p : policies) {
    sum += p.second;
  }
  std::uniform_real_distribution<> dice(0.0, sum);
  const float roll = dice(g_rng);
  float acc = 0.0f;
  for (const auto& p : policies) {
     acc += p.second;
     if (acc >= roll) {
       return p.first;
     }
   }
   return policies.back().first;
}

bool AsyncScorer::SyncScoreGoState(const GoBoard& board,
                                   PolicyResult* policy, ValueResult* value) {
  bool success;
  absl::Notification waiter;
  auto cb = [&waiter, &success, policy, value](
      bool scorer_ok, PolicyResult policy_result, ValueResult value_result) {
    success = scorer_ok;
    if (success) {
      policy->swap(policy_result);
      *value = value_result;
    }
    waiter.Notify();
  };
  ScoreGoState(board, cb);
  waiter.WaitForNotification();
  return success;
}

void SimpleScorer::ScoreGoState(const GoBoard& board, Callback cb) {
  PolicyResult* policy_result = new PolicyResult();
  for (GoSizeT i = 0; i < board.width(); ++i) {
    for (GoSizeT j = 0; j < board.height(); ++j) {
      // So if there is no legal move, x and y will remain COORD_PASS.
      if (board.IsLegalMove({i, j})) {
        policy_result->emplace_back(std::make_pair(std::make_pair(i, j), 1.0));
      }
    }
  }
  Normalize(policy_result);
  auto value_result = SimpleEvaluate(board);

  GetScorerThreadPool()->Schedule(
      [cb, policy_result, value_result]() {
        cb(true, std::move(*policy_result), value_result);
        delete policy_result;
      });
}

std::unique_ptr<TfScorer> TfScorer::CreateFromFlags() {
  // Create TensorFlow client:
  auto tf_client = TensorFlowClient::Create(
      FLAGS_model, FLAGS_input_layer_name,
      FLAGS_output_layer_prefix, /*num_outputs=*/2, /*batch_size=*/128,
      /*max_queue_delay*/absl::Milliseconds(10));
  CHECK(tf_client != nullptr);
  return absl::make_unique<TfScorer>(std::move(tf_client));
}

void TfScorer::ScoreGoState(const GoBoard& board, Callback cb) {
  ValueResult fast_eval = SimpleEvaluate(board);
  auto callback = [&board, fast_eval, cb](
      const tf::Status& status, std::vector<std::vector<float>> outputs) {
    PolicyResult policy_result;
    if (status.ok()) {
      CHECK_EQ(2, outputs.size());

      auto& policy_output = outputs[0];
      CHECK_EQ(board.width() * board.height(), policy_output.size());
      ConvertToPolicyResult(board, policy_output, &policy_result);

      const auto& value_output = outputs[1];
      CHECK_EQ(1, value_output.size());

      cb(true, policy_result, CombineValueResult(value_output[0], fast_eval));
    } else {
      LOG(ERROR) << "TensorFlow error: " << status;
      cb(false, policy_result, fast_eval);
    }
  };
  tf_client_->AddInferenceTask(board.GetFeatures().Clone(), callback);
}

}  // namespace zebra_go
