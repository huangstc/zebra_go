#ifndef ZEBRA_GO_ENGINE_UTILS_H_
#define ZEBRA_GO_ENGINE_UTILS_H_

#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace zebra_go {

// A simple ThreadPool.
class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);
  ~ThreadPool();

  // Schedules a function to be run on a ThreadPool thread.
  void Schedule(std::function<void()> closure);

  // Disable copy/move constructors and copy operator.
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

 private:
  class SafeTaskQueue;

  void WorkLoop();

  // Tasks are pushed to this queue and consumed by worker threads.
  std::unique_ptr<SafeTaskQueue> queue_;
  // Worker threads.
  std::vector<std::thread> threads_;
};

class Histogram {
 public:
  Histogram(float lower_bound, float upper_bound, int num_buckets);
  void Count(float p);
  std::string ToString() const;

 private:
  const float min_, max_, bucket_len_;
  std::vector<int> buckets_;
};

}  // namespace zebra_go

#endif  // ZEBRA_GO_ENGINE_UTILS_H_
