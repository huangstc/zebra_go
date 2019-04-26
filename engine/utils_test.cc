#include "engine/utils.h"

#include <algorithm>
#include <mutex>
#include <random>

#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zebra_go {
namespace {

class ThreadPoolTest : public ::testing::Test {};

TEST_F(ThreadPoolTest, Run) {
  std::mutex mu;
  int sum[3] = {0, 0, 0};

  {
    ThreadPool pool(3);
    for (int i = 0; i < 3; ++i) {
      pool.Schedule([&mu, &sum, i] {
        std::lock_guard<std::mutex> guard(mu);
        sum[i] = (i+1) * (i+1);
      });
    }
  }

  EXPECT_EQ(1, sum[0]);
  EXPECT_EQ(4, sum[1]);
  EXPECT_EQ(9, sum[2]);
}

TEST(TopKTest, InsertAndGet) {
  std::vector<int> data(100, 0);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i * i;
  }
  auto rng = std::default_random_engine {};
  std::shuffle(data.begin(), data.end(), rng);

  TopK<int> top(7);
  for (int d : data) {
    top.Insert(d);
  }

  EXPECT_THAT(top.elements(), ::testing::UnorderedElementsAreArray(
      {99*99, 98*98, 97*97, 96*96, 95*95, 94*94, 93*93}));
}

}  // namespace
}  // namespace zebra_go
