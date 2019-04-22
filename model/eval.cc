// Evaluates a model's accuracy with a test dataset.
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/blocking_counter.h"
#include "engine/go_game.h"
#include "engine/sgf_utils.h"
#include "engine/utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/tf_client.h"

DEFINE_string(model, "", "Load model from this file.");
DEFINE_string(input_layer_name, "go_input", "");
DEFINE_string(output_layer_prefix, "go_output", "");
DEFINE_string(sgf_files, "", "Use the game from this file to test the model.");

namespace tf = tensorflow;

namespace zebra_go {

using ::sgf_parser::GameRecord;
using ::std::string;

static const int kBoardSize = 19;
static const int kBatchSize = 128;

struct ExampleWithResult {
  ExampleWithResult(const GoFeatureSet& features, GoPosition move,
                    float final_result)
      : raw_features(features.Clone()),
        next_move(move),
        game_result(final_result) {}

  std::unique_ptr<GoFeatureSet> raw_features;
  const GoPosition next_move;
  const float game_result;  // positive: current player wins.

  // Eval results:
  std::vector<float> raw_scores;       // Copied from its policy output.
  float value_output;                  // Copied from its value output.
  std::vector<GoPosition> top_moves;   // Top-K moves.
};

// Replays the game to get examples.
int GameRecordToExamples(
    const string& sgf, const string& filename,
    std::vector<std::unique_ptr<ExampleWithResult>>* examples) {
  int num_new_examples = 0;
  ReplayGame(
      sgf,
      [](const ReplayContext& ctx) {
        return ctx.board->width() == 19 && ctx.board->height() == 19;
      },
      [filename, examples, &num_new_examples](const ReplayContext& ctx) {
        float outcome = ctx.game_result;
        if (ctx.board->current_player() == COLOR_WHITE) {
          outcome = -outcome;
        }
        examples->emplace_back(absl::make_unique<ExampleWithResult>(
            ctx.board->GetFeatures(), ctx.next_move, outcome));
      },
      [](std::unique_ptr<GoBoard> board) {});
  return num_new_examples;
}

// Gets examples from SGF files specified by the flag --sgf_files.
std::vector<std::unique_ptr<ExampleWithResult>> LoadExamples() {
  std::vector<std::unique_ptr<ExampleWithResult>> examples;

  const auto filenames = Glob(FLAGS_sgf_files);
  LOG(INFO) << "Found " << filenames.size() << " SGF files.";

  for (const auto& name : filenames) {
    const string sgf = ReadFileToString(name);
    GameRecordToExamples(sgf, name, &examples);
  }
  LOG(INFO) << "Extracted " << examples.size() << " examples.";
  return examples;
}

// Performs model inference on the examples.
void ScoreExamples(TensorFlowClient* tf_client,
                   std::vector<std::unique_ptr<ExampleWithResult>>* examples) {
  absl::BlockingCounter blocker(examples->size());
  for (auto& ex : *examples) {
    ExampleWithResult* example = ex.get();
    tf_client->AddInferenceTask(
        // Don't read example->raw_features afterwards, as it has been moved.
        std::move(example->raw_features),
        [&blocker, example](const tf::Status& status,
                            std::vector<std::vector<float>> outputs) {
          if (status.ok()) {
            CHECK_EQ(2, outputs.size());

            auto& policy_output = outputs[0];
            CHECK_EQ(kBoardSize * kBoardSize, policy_output.size());
            example->raw_scores.swap(policy_output);

            const auto& value_output = outputs[1];
            CHECK_EQ(1, value_output.size());
            example->value_output = value_output[0];
          } else {
            LOG(ERROR) << "Inference error: " << status;
          }
          blocker.DecrementCount();
        });
  }
  blocker.Wait();
  LOG(INFO) << "Finished inferencing on all examples.";
}

void RunEval() {
  auto tf_client = TensorFlowClient::Create(
      FLAGS_model, FLAGS_input_layer_name,
      FLAGS_output_layer_prefix, /*num_outputs=*/2, /*batch_size=*/kBatchSize,
      /*max_queue_delay*/absl::Milliseconds(10));
  CHECK(tf_client != nullptr);

  // Load all examples in RAM.
  auto examples = LoadExamples();
  ScoreExamples(tf_client.get(), &examples);

  // Eval:
  // Where human's move is the 1st result predicted by the model.
  int num_0 = 0;
  // Where human's move is in the first 3 results from the model.
  int num_3 = 0;
  // Where human's move is in the first 10 results from the model.
  int num_10 = 0;
  // Distribution of the value network's outputs for the winners.
  Histogram win_scores(0, 1.0, 100);
  // Distribution of the value network's outputs for the losers.
  Histogram lose_scores(0, 1.0, 100);
  for (const auto& ex : examples) {
    int golden = ex->next_move.first + ex->next_move.second * kBoardSize;
    float gloden_score = ex->raw_scores[golden];
    int rank = 0;
    for (const float x : ex->raw_scores) {
      if (x > gloden_score) {
        rank++;
      }
    }
    if (rank == 0) num_0++;
    if (rank < 3) num_3++;
    if (rank < 10) num_10++;

    if (ex->game_result > 0) {  // current player wins
      win_scores.Count(ex->value_output);
    } else {
      lose_scores.Count(ex->value_output);
    }
  }
  LOG(INFO) << "Top: " << (num_0 * 100.0) / examples.size() << "%";
  LOG(INFO) << "Top 3: " << (num_3 * 100.0) / examples.size() << "%";
  LOG(INFO) << "Top 10: " << (num_10 * 100.0) / examples.size() << "%";
  LOG(INFO) << "Winner score distribution:" << win_scores.ToString();
  LOG(INFO) << "Loser score distribution:" << lose_scores.ToString();
}

}  // namespace zebra_go

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  zebra_go::RunEval();
  return 0;
}
