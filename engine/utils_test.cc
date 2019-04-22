#include "engine/utils.h"

#include <mutex>

#include "glog/logging.h"
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

}  // namespace
}  // namespace zebra_go
