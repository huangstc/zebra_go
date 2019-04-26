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

// Using Heap to get top K elements of a stream.
// Comaparator: cmp(a, b) is true if a < b.
template<typename T, typename Comparator = std::less<T>>
class TopK {
 public:
  explicit TopK(size_t k) : k_(k), cmp_() {}

  size_t size() const { return buffer_.size(); }

  // Returns top-K elements collected so far. Note that elements may not be
  // sorted.
  const std::vector<T>& elements() const { return buffer_; }

  // Returns true if the element is inserted. Returns false if the element is
  // rejected, which happens when the buffer is full and the item is smaller
  // than the smallest element in the buffer.
  bool Insert(const T& item) {
    if (buffer_.size() < k_) {
      // The buffer is not full yet, insert.
      buffer_.push_back(item);
      if (buffer_.size() == k_) {
        for (int i = (k_ - 2) / 2; i >= 0; i--) {
          Heapify(i);
        }
      }
      return true;
    } else if (cmp_(buffer_[0], item)) {
      buffer_[0] = item;
      Heapify(0);
      return true;
    } else {
      return false;
    }
  }

 private:
  void Heapify(size_t root) {
    if (k_ < buffer_.size()) {
      return;
    }

    size_t l = 2 * root + 1;
    size_t r = 2 * root + 2;
    size_t smallest = root;
    if (l < k_ && cmp_(buffer_[l], buffer_[smallest])) {
      smallest = l;
    }
    if (r < k_ && cmp_(buffer_[r], buffer_[smallest])) {
      smallest = r;
    }
    if (smallest != root) {
      T tmp = buffer_[root];
      buffer_[root] = buffer_[smallest];
      buffer_[smallest] = tmp;
      Heapify(smallest);
    }
  }

  const size_t k_;
  const Comparator cmp_;
  std::vector<T> buffer_;
};

}  // namespace zebra_go

#endif  // ZEBRA_GO_ENGINE_UTILS_H_
