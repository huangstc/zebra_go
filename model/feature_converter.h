#ifndef ZEBRA_GO_MODEL_FEATURE_COMPUTER_H_
#define ZEBRA_GO_MODEL_FEATURE_COMPUTER_H_

#include <string>
#include <vector>

#include "engine/go_game.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace zebra_go {

tensorflow::Tensor GoFeatureSetToTensor(const GoFeatureSet& features);

tensorflow::Tensor BatchGoFeatureSetsToTensor(
    const std::vector<const GoFeatureSet*>& batch_feature);

// "Current player" is the one who is going to play the next move.
// "outcome" is the final result of the game, positive if current player wins.
// "note" is a short string that describes the example.
// Returns false if the conversion failed.
bool GoFeatureSetToExample(GoPosition next_move, float outcome,
                           const GoFeatureSet& features,
                           const std::string& note,
                           tensorflow::Example* example);

}  // namespace zebra_go

#endif  // ZEBRA_GO_MODEL_FEATURE_COMPUTER_H_
