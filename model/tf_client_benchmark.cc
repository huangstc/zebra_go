#include "model/tf_client.h"

#include <string>

#include "benchmark/benchmark.h"
#include "engine/sgf_utils.h"
#include "sgf_parser/parser.h"

namespace zebra_go {

// Not really a good benchmark because TensorFlowClient runs asynchronously.
static void BM_Inference(benchmark::State& state) {
  const std::string model_file_path = "testdata/20190421.pb";
  LOG(INFO) << "Load TF model from " << model_file_path;

  auto tf_client = TensorFlowClient::Create(
      model_file_path, /*input_layer_name=*/"go_input",
      /*output_layer_name_prefix=*/"go_output/", /*num_outputs=*/2,
      /*batch_size=*/128, /*max_queue_delay*/absl::Milliseconds(10));

  const std::string sgf = ReadFileToString("testdata/cj_supermatch_1991.sgf");

  for (auto _ : state) {
    int num_inferences[2] = {0, 0};
    ReplayGame(
        sgf,
        [](const ReplayContext& ctx) {
          return ctx.board->width() == 19 && ctx.board->height() == 19;
        },
        [&tf_client, &num_inferences](const ReplayContext& ctx) {
          tf_client->AddInferenceTask(
              ctx.board->GetFeatures().Clone(),
              [&num_inferences](const tf::Status& status,
                                std::vector<std::vector<float>> outputs) {
                if (status.ok()) {
                  num_inferences[0]++;
                } else {
                  num_inferences[1]++;
                }
              });
        },
        [](std::unique_ptr<GoBoard> board) {});
  }
}

BENCHMARK(BM_Inference);

}  // namespace zebra_go

BENCHMARK_MAIN();
