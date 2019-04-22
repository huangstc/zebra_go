#include "model/feature_converter.h"

namespace zebra_go {

namespace tf = tensorflow;

tf::Tensor GoFeatureSetToTensor(const GoFeatureSet& features) {
  return BatchGoFeatureSetsToTensor({&features});
}

tensorflow::Tensor BatchGoFeatureSetsToTensor(
    const std::vector<const GoFeatureSet*>& feature_batch) {
  CHECK(!feature_batch.empty());
  const int num_examples = feature_batch.size();
  const GoSizeT height = feature_batch[0]->height();
  const GoSizeT width = feature_batch[0]->width();
  const auto num_channels = feature_batch[0]->num_planes();
  tf::TensorShape shape({num_examples, height, width, num_channels});
  tensorflow::Tensor result(tf::DT_FLOAT, shape);
  auto tensor = result.tensor<float, 4>();
  for (int idx = 0; idx < num_examples; ++idx) {
    CHECK_EQ(feature_batch[idx]->height(), height);
    CHECK_EQ(feature_batch[idx]->width(), width);
    CHECK_EQ(feature_batch[idx]->num_planes(), num_channels);
    for (int pid = 0; pid < num_channels; ++pid) {
      const auto& plane = feature_batch[idx]->plane(pid);
      for (GoSizeT y = 0; y < height; ++y) {
        for (GoSizeT x = 0; x < width; ++x) {
          tensor(idx, y, x, pid) = plane[y * width + x];
        }
      }
    }
  }
  return result;
}

namespace {

void MakeFloatsFeature(const std::vector<float>& ff,
                       tensorflow::Feature* feature) {
  auto* float_list = feature->mutable_float_list();
  for (const float f : ff) {
    float_list->add_value(f);
  }
}

}  // namespace

// "Current player" is the one who is going to play the next move.
// "outcome" is the final result of the game, positive if current player wins.
bool GoFeatureSetToExample(GoPosition next_move, float outcome,
                           const GoFeatureSet& go_feature_set,
                           const std::string& note,
                           tensorflow::Example* example) {
  if (next_move.first < 0 || next_move.first >= go_feature_set.width()) {
    LOG(WARNING) << "Bad example: x is out of boundary " << next_move.first;
    return false;
  }
  if (next_move.second < 0 || next_move.second >= go_feature_set.height()) {
    LOG(WARNING) << "Bad example: y is out of bounary " << next_move.second;
    return false;
  }

  example->Clear();
  auto& features = *example->mutable_features()->mutable_feature();

  // Next move, for training the policy network.
  const int64_t encoded_next_move = next_move.second * go_feature_set.width() +
                                    next_move.first;
  features["next"].mutable_int64_list()->add_value(encoded_next_move);
  features["next_xy"].mutable_int64_list()->add_value(next_move.first);
  features["next_xy"].mutable_int64_list()->add_value(next_move.second);

  // Final game result, for training the value network.
  features["outcome"].mutable_float_list()->add_value(outcome);

  // A shor note to help debugging.
  features["note"].mutable_bytes_list()->add_value(note);

  // Encode all the planes.
  const int feature_len = go_feature_set.width() * go_feature_set.height();
  for (int pid = 0; pid < go_feature_set.num_planes(); ++pid) {
    DCHECK_EQ(go_feature_set.plane(pid).size(), feature_len);
    auto& new_plane = features[go_feature_set.GetPlaneName(pid)];
    MakeFloatsFeature(go_feature_set.plane(pid), &new_plane);
  }

  // "outcome", "next", "next_xy", "note" and image features.
  DCHECK_EQ(4 + go_feature_set.num_planes(), features.size());
  return true;
}

}  // namespace zebra_go
