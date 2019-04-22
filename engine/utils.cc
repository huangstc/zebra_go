#include "engine/utils.h"

#include <numeric>
#include <queue>

#include "absl/synchronization/mutex.h"
#include "glog/logging.h"

namespace zebra_go {
namespace {

// A thread-safe queue.
template <class T>
class SafeQueue {
 public:
  SafeQueue() {}
  ~SafeQueue() {}

  // Adds an element to the queue.
  void Enqueue(T t) {
    absl::MutexLock lock(&mu_);
    q_.push(std::move(t));
    c_.Signal();
  }

  size_t size() const {
    absl::MutexLock lock(&mu_);
    size_t sz = q_.size();
    return sz;
  }

  // Gets the front element. If the queue is empty, waits till an element is
  // available.
  T Dequeue() {
    mu_.Lock();
    while (q_.empty()) {
      // Release lock, wait and reaquire it when signaled.
      c_.Wait(&mu_);
    }
    T val = q_.front();
    q_.pop();
    mu_.Unlock();
    return val;
  }

  // Blocks current thread until the queue is empty.
  void WaitTillEmpty() {
    absl::MutexLock lock(&mu_);
    mu_.Await(absl::Condition(+[](std::queue<T>* qq) {
                                return qq->empty();
                              }, &q_));
  }

 private:
  mutable absl::Mutex mu_;
  absl::CondVar c_;
  std::queue<T> q_;
};

}  // namespace

class ThreadPool::SafeTaskQueue : public SafeQueue<std::function<void()>> {};

ThreadPool::ThreadPool(int num_threads) : queue_(new SafeTaskQueue()) {
  for (int i = 0; i < num_threads; ++i) {
    threads_.push_back(std::thread(&ThreadPool::WorkLoop, this));
  }
}

ThreadPool::~ThreadPool() {
  for (size_t i = 0; i < threads_.size(); i++) {
    queue_->Enqueue(nullptr);  // Shutdown signal.
  }
  for (auto &t : threads_) {
    t.join();
  }
}

void ThreadPool::Schedule(std::function<void()> func) {
  CHECK(func);
  queue_->Enqueue(std::move(func));
}

void ThreadPool::WorkLoop() {
  while (true) {
    std::function<void()> func = queue_->Dequeue();
    if (func == nullptr) {  // Shutdown signal.
      return;
    } else {
      func();
    }
  }
}

Histogram::Histogram(float lower, float upper, int num_buckets)
    : min_(lower), max_(upper), bucket_len_((upper-lower)/num_buckets) {
  buckets_.resize(num_buckets, 0);
}

void Histogram::Count(float p) {
  if (p <= min_) {
    buckets_[0] += 1;
  } else if (p >= max_) {
    buckets_[buckets_.size() - 1] += 1;
  } else {
    size_t bucket_id = static_cast<int>((p - min_) / bucket_len_);
    buckets_[bucket_id] += 1;
  }
}

std::string Histogram::ToString() const {
  return std::accumulate(buckets_.begin()+1, buckets_.end(),
                         std::to_string(buckets_[0]),
                         [](const std::string& a, int b){
                           return a + ',' + std::to_string(b);
                         });
}

}  // namespace zebra_go
