#include "engine/scorer.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace zebra_go {
namespace {

TEST(SamplePolicyTest, Sample) {
  PolicyResult policy_result({{{1,1}, 0.5}, {{2,2}, 0.3}});
  auto pos = AsyncScorer::SamplePolicy(policy_result);
  EXPECT_TRUE(pos == GoPosition({1,1}) || pos == GoPosition({2,2}));
}

}  // namespace
}  // namespace zebra_go
