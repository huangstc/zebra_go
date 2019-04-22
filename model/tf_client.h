#ifndef ZEBRA_GO_MODEL_TF_CLIENT_H_
#define ZEBRA_GO_MODEL_TF_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "engine/go_game.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

namespace zebra_go {

namespace tf = tensorflow;

// A class for performing model inference using a trained model.
class TensorFlowClient {
 public:
  // Creates a client instance with a model loaded from the given file.
  // The client can hold input examples in a buffer, and runs model inference
  // when the buffer is full or max_queue_delay has elapsed after the first
  // input is buffered.
  static std::unique_ptr<TensorFlowClient> Create(
       const std::string& model_file_path, const std::string& input_layer_name,
       const std::string& output_layer_name_prefix, int num_outputs,
       int batch_size, absl::Duration max_queue_delay);

  // Type of model inputs.
  typedef std::unique_ptr<GoFeatureSet> ModelInput;

  // Type of model outputs. The size of the outer vector equals to the number
  // of model outputs, which is always 2 in our case. The first entry, a vector
  // of size (board_width * board_height), is the output of the policy network.
  // The second entry, a float ranging from 0 to 1, is the value network's
  // output.
  typedef std::vector<std::vector<float>> ModelOutput;

  // Callback type.
  typedef std::function<void(const tf::Status&, ModelOutput)> InferenceCallback;

  // Asynchronously runs model inference on the input.
  virtual void AddInferenceTask(ModelInput input, InferenceCallback cb);

  ~TensorFlowClient();
  TensorFlowClient() = delete;
  TensorFlowClient(const TensorFlowClient&) = delete;

 protected:
  TensorFlowClient(
      const std::string& model_file_path, const std::string& input_layer_name,
      const std::string& output_layer_name_prefix, int num_outputs,
      int batch_size, absl::Duration max_queue_delay);

 private:
  class InferenceTask;
  class TaskQueue;

  // Converts an input batch to a tensor.
  static tf::Tensor ToBatchTensor(const std::vector<InferenceTask*>& tasks);

  void RunModel(std::vector<InferenceTask*> tasks);

  std::unique_ptr<TaskQueue> task_queue_;
  // A Session object is thread-safe for Session.run() calls.
  std::unique_ptr<tf::Session> tf_session_;
  const std::string input_layer_name_;
  std::vector<std::string> output_layer_names_;
};

}  // namespace zebra_go

#endif  // ZEBRA_GO_MODEL_TF_CLIENT_H_
