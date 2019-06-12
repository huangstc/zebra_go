#ifndef ZEBRA_GO_ENGINE_SCORER_H_
#define ZEBRA_GO_ENGINE_SCORER_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "engine/go_game.h"
#include "model/tf_client.h"

namespace zebra_go {

// Represents an array of candidate moves output from the policy network.
// GoPosition: candidate move.
// float: score/probability of the move.
typedef std::vector<std::pair<GoPosition, float>> PolicyResult;

// bool: true if current player should resign.
// float: score for current player.
typedef std::pair<bool, float> ValueResult;

class AsyncScorer {
 public:
  // Callback of asynchronous scoring. The first boolean argument is set
  // to false when the scorer fails.
  typedef std::function<void(bool, PolicyResult, ValueResult)> Callback;

  // Randomly samples a candidate move, weighted by their scores.
  static GoPosition SamplePolicy(const PolicyResult& policies);

  // Prints debug strings.
  static std::string DebugString(const PolicyResult& p);
  static std::string DebugString(const ValueResult& p);

  virtual ~AsyncScorer() {}

  // Asynchronously runs inference for current player. Caller will get an
  // output from the policy network and an output from the value network
  // through the callback.
  virtual void ScoreGoState(const GoBoard& board, Callback cb) = 0;

  // Synchronous version of ScoreGoState.
  bool SyncScoreGoState(const GoBoard& board,
                        PolicyResult* policy, ValueResult* value);
};

// A trivial implementation of AsyncScorer, mainly for testing.
class SimpleScorer : public AsyncScorer {
 public:
  SimpleScorer() {}
  ~SimpleScorer() override {}

  void ScoreGoState(const GoBoard& board, Callback cb) override;
};

// An implementation of AsyncScorer based on a trained model.
class TfScorer : public AsyncScorer {
 public:
  // Creates an instance from the following flags:
  //   --model: serialized model file.
  //   --input_layer_name: the input layer's name.
  //   --output_layer_prefix: the name prefix of the model's output layers.
  static std::unique_ptr<TfScorer> CreateFromFlags();

  explicit TfScorer(std::unique_ptr<TensorFlowClient> tf_client)
      : tf_client_(std::move(tf_client)) {}

  ~TfScorer() override {}

  void ScoreGoState(const GoBoard& board, Callback cb) override;

 private:
  std::unique_ptr<TensorFlowClient> tf_client_;
};

}  // namespace zebra_go

#endif  // ZEBRA_GO_ENGINE_SCORER_H_
