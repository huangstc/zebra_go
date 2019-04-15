#include "model/feature_converter.h"

#include <memory>

#include "absl/memory/memory.h"
#include "engine/go_game.h"
#include "engine/sgf_utils.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace zebra_go {
namespace {

namespace tf = tensorflow;

class FeatureConverterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static const char kSgf[] = R"((
      ;GM[1]FF[4]CA[UTF-8]AP[test]SZ[5]
      ;AB[ba][ab][cb][db][bd][cd][ed][ae][de]
      ;AW[ca][da][bb][eb][bc][dc][ad][dd][ce]
    ))";
    board_ = SgfToGoBoard(kSgf);
  }

  void CheckFloatFeature(const tf::Example& example,
                         const std::string& key,
                         const std::vector<float>& expect_values) {
    LOG(INFO) << "Checking " << key;
    const auto& feature_map = example.features().feature();
    const auto iter = feature_map.find(key);
    ASSERT_TRUE(iter != feature_map.end());
    const auto& values = iter->second.float_list().value();
    ASSERT_EQ(values.size(), expect_values.size());
    for (int i = 0; i < values.size(); ++i) {
      EXPECT_FLOAT_EQ(values.Get(i), expect_values[i]);
    }
  }
  std::unique_ptr<GoBoard> board_;
};

TEST_F(FeatureConverterTest, ToTensor) {
  auto tensor = GoFeatureSetToTensor(board_->GetFeatures());
  ASSERT_EQ(4, tensor.dims());
  EXPECT_EQ(1, tensor.dim_size(0));  // batch size.
  EXPECT_EQ(5, tensor.dim_size(1));  // x
  EXPECT_EQ(5, tensor.dim_size(2));  // y
  EXPECT_EQ(7, tensor.dim_size(3));  // number of planes
}

TEST_F(FeatureConverterTest, ToExample) {
  tf::Example example;
  GoFeatureSetToExample({1, 4}, 2.5, board_->GetFeatures(), "test", &example);
  CheckFloatFeature(example, "orig", {0,-1,1,1,0,-1,1,-1,-1,1,0,1,0,1,0,1,-1,-1,
                                      1,-1,-1,0,1,-1,0});

  const std::vector<float> all_zeros(25, 0.0);

  std::vector<float> b1 = all_zeros;
  b1[2] = b1[3] = b1[15] = b1[22] = 1;
  CheckFloatFeature(example, "b1", b1);

  std::vector<float> b2 = all_zeros;
  b2[6] = b2[9] = b2[11] = b2[13] = b2[18] = 1;
  CheckFloatFeature(example, "b2", b2);

  CheckFloatFeature(example, "b3", all_zeros);

  std::vector<float> w1 = all_zeros;
  w1[1] = w1[7] = w1[8] = w1[20] = w1[23] = 1;
  CheckFloatFeature(example, "w1", w1);

  std::vector<float> w2 = all_zeros;
  w2[5] = w2[16] = w2[17] = w2[19] = 1;
  CheckFloatFeature(example, "w2", w2);

  CheckFloatFeature(example, "w3", all_zeros);

  CheckFloatFeature(example, "outcome", {2.5});

  {  // Check "next"
    const auto& feature_map = example.features().feature();
    const auto iter = feature_map.find("next");
    ASSERT_TRUE(iter != feature_map.end());
    const auto& values = iter->second.int64_list().value();
    ASSERT_EQ(1, values.size());
    EXPECT_EQ(21, values.Get(0));
  }
}

}  // namespace
}  // namespace zebra_go
