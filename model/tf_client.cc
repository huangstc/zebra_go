#include "model/tf_client.h"

#include <thread>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "engine/utils.h"
#include "model/feature_converter.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

namespace zebra_go {

using ::std::string;

namespace {

// Reads a model from the output of keras_to_tensorflow.py
// Returns a Session object that loads the model graph.
std::unique_ptr<tf::Session> LoadGraph(const string& graph_file_name) {
  LOG(INFO) << "Loading graph from " << graph_file_name;
  tf::GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(tf::Env::Default(), graph_file_name, &graph_def));
  for (const auto& node : graph_def.node()) {
    LOG(INFO) << "Node:" << node.name() << " op:" << node.op();
  }
  VLOG(1) << graph_def.DebugString();
  LOG(INFO) << "Creating TF session.";
  auto sess = absl::WrapUnique<tf::Session>(
      tf::NewSession(tf::SessionOptions()));
  TF_CHECK_OK(sess->Create(graph_def));
  LOG(INFO) << "TF session is created.";
  return sess;
}

ThreadPool* GetTfThreadPool() {
  static ThreadPool* g_thread_pool = new ThreadPool(3);
  return g_thread_pool;
}

}  // namespace

class TensorFlowClient::InferenceTask {
 public:
  InferenceTask(ModelInput input, InferenceCallback cb)
      : input_(std::move(input)), cb_(cb) {}

  std::unique_ptr<GoFeatureSet> input_;
  InferenceCallback cb_;
};

class TensorFlowClient::TaskQueue {
 public:
  typedef std::function<void(std::vector<TensorFlowClient::InferenceTask*>)>
      RunModelCallback;

  TaskQueue(int max_size, absl::Duration max_queue_delay)
      : max_size_(max_size), max_queue_delay_(max_queue_delay),
        stopping_(false) {}

  ~TaskQueue() {
    if (alarm_thread_ != nullptr) {
      stopping_ = true;
      alarm_thread_->join();
    }
    Flush();
  }

  void Start(RunModelCallback run_model_callback) {
    run_model_ = run_model_callback;

    alarm_thread_ = absl::make_unique<std::thread>([this]() {
        while (!this->stopping_) {
          absl::SleepFor(this->max_queue_delay_);
          this->Flush();
        }
      });
  }

  void Flush() {
    Enqueue(nullptr);
  }

  // Flush the buffer if t is null.
  void Enqueue(TensorFlowClient::InferenceTask* t) {
    std::vector<TensorFlowClient::InferenceTask*> tasks;

    mu_.lock();
    if (t != nullptr) {
      buffer_.push_back(t);
    }
    if (buffer_.size() >= max_size_ || t == nullptr) {
      tasks.swap(buffer_);
    }
    mu_.unlock();

    if (!tasks.empty()) {
      run_model_(std::move(tasks));
    }
  }

 private:
  const size_t max_size_;
  const absl::Duration max_queue_delay_;

  RunModelCallback run_model_;

  mutable std::mutex mu_;
  std::vector<TensorFlowClient::InferenceTask*> buffer_;

  bool stopping_ = false;
  std::unique_ptr<std::thread> alarm_thread_;
};

std::unique_ptr<TensorFlowClient> TensorFlowClient::Create(
    const string& model_file_path, const string& input_layer_name,
    const string& output_layer_name_prefix, int num_outputs,
    int batch_size, absl::Duration max_queue_delay) {
  auto* client = new TensorFlowClient(
      model_file_path, input_layer_name, output_layer_name_prefix, num_outputs,
      batch_size, max_queue_delay);
  return absl::WrapUnique<TensorFlowClient>(client);
}

TensorFlowClient::TensorFlowClient(
    const string& model_file_path,  const string& input_layer_name,
    const string& output_layer_name_prefix, int num_outputs,
    int batch_size, absl::Duration max_queue_delay)
    : task_queue_(new TaskQueue(batch_size, max_queue_delay)),
      input_layer_name_(input_layer_name) {
  // Load the model to a TF session.
  tf_session_ = LoadGraph(model_file_path);
  for (int i = 0; i < num_outputs; ++i) {
    output_layer_names_.push_back(absl::StrCat(output_layer_name_prefix, i));
  }

  // Starts the queue.
  task_queue_->Start([this](std::vector<InferenceTask*> tasks) {
      this->RunModel(std::move(tasks));
    });
}

TensorFlowClient::~TensorFlowClient() {
  // The queue must be deleted before the session object.
  task_queue_.reset();
  tf_session_.reset();
}

void TensorFlowClient::AddInferenceTask(
    ModelInput input, InferenceCallback cb) {
  task_queue_->Enqueue(new InferenceTask(std::move(input), cb));
}

tf::Tensor TensorFlowClient::ToBatchTensor(
    const std::vector<TensorFlowClient::InferenceTask*>& tasks) {
  std::vector<const GoFeatureSet*> feature_sets;
  feature_sets.reserve(tasks.size());
  for (const TensorFlowClient::InferenceTask* task : tasks) {
    feature_sets.push_back(task->input_.get());
  }
  return BatchGoFeatureSetsToTensor(feature_sets);
}

void TensorFlowClient::RunModel(
    std::vector<TensorFlowClient::InferenceTask*> tasks) {
  if (tasks.empty()) {
    return;
  }

  tf::Tensor input = ToBatchTensor(tasks);
  std::vector<tf::Tensor> outputs;
  auto status = tf_session_->Run({{input_layer_name_, input}},
                                 output_layer_names_, {}, &outputs);
  if (status.ok()) {
    // Run callback with outputs.
    for (size_t idx = 0; idx < tasks.size(); ++idx) {
      auto* task_outputs = new std::vector<std::vector<float>>();
      for (size_t i = 0; i < outputs.size(); ++i) {
        task_outputs->emplace_back(std::vector<float>());
        std::vector<float>& data = task_outputs->back();
        data.reserve(outputs[i].dim_size(1));  // the 1st dim is the batch size.
        const auto& matrix = outputs[i].matrix<float>();
        for (int j = 0; j < outputs[i].dim_size(1); ++j) {
          data.push_back(matrix(idx, j));
        }
      }

      // Run the client callback in the global thread pool.
      auto client_cb = tasks[idx]->cb_;
      GetTfThreadPool()->Schedule(
          [client_cb, status, task_outputs]() {
            client_cb(status, *task_outputs);
            delete task_outputs;
          });
    }
  } else {
    // Run callbacks with an error status.
    for (auto t : tasks) {
      auto client_cb = t->cb_;
      GetTfThreadPool()->Schedule(
          [client_cb, status]() {
            std::vector<std::vector<float>> dummy_output;
            client_cb(status, std::move(dummy_output));
          });
    }
  }

  // Delete tasks.
  for (auto t : tasks) {
    delete(t);
  }
}

}  // namespace zebra_go
