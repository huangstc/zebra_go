// Generates training data set in TFRecord from SGF files.

#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "engine/go_game.h"
#include "engine/sgf_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/feature_converter.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

DEFINE_string(sgf_files, "", "Pattern of input SGF files.");
DEFINE_int32(examples_per_file, 100000, "Number of examples per output file.");
DEFINE_string(output, "/tmp/go_training",
             "Write the data set with this prefix.");

namespace zebra_go {
namespace {

namespace tf = tensorflow;

using std::string;
using tf::io::RecordWriter;
using tf::io::RecordWriterOptions;

string ShortFilePath(const string& full) {
  auto pos = full.rfind('/');
  if (pos == string::npos) {
    return full;
  } else {
    return full.substr(pos);
  }
}

int GameRecordToExamples(const string& sgf, const string& filename,
                         std::vector<tf::Example>* examples) {
  int num_new_examples = 0;
  ReplayGame(
      sgf,
      [](const ReplayContext& ctx) {
        return ctx.board->width() == 19 && ctx.board->height() == 19;
      },
      [filename, examples, &num_new_examples](const ReplayContext& ctx) {
        if (ctx.num_steps < 3) return;
        tf::Example new_example;
        float outcome = ctx.game_result > 0 ? 1 : 0;
        if (ctx.board->current_player() == COLOR_WHITE) {
          outcome = 1 - outcome;
        }
        if (GoFeatureSetToExample(ctx.next_move, outcome,
                                  ctx.board->GetFeatures(), filename,
                                  &new_example)) {
          examples->emplace_back(std::move(new_example));
          ++num_new_examples;
        }
      },
      [](std::unique_ptr<GoBoard> board) {});
  return num_new_examples;
}

void WriteExamples(const string& path,
                   const std::vector<tf::Example>& examples) {
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(path, &file));

  RecordWriterOptions options;
  options.compression_type = RecordWriterOptions::ZLIB_COMPRESSION;
  RecordWriter writer(file.get(), options);

  string data;
  for (const auto& example : examples) {
    example.SerializeToString(&data);
    TF_CHECK_OK(writer.WriteRecord(data));
  }
  TF_CHECK_OK(writer.Close());
  TF_CHECK_OK(file->Close());
}

void FlushExamples(const string& prefix, int serial,
                   std::vector<tf::Example> examples) {
  const string file = absl::StrFormat("%s-%04d.rio", prefix.c_str(), serial);
  LOG(INFO) << "Writing " << examples.size() << " examples to " << file;
  WriteExamples(file, examples);
}

void Run() {
  const auto filenames = Glob(FLAGS_sgf_files);
  LOG(INFO) << "Found " << filenames.size() << " sgf files.";

  int failed_files = 0;
  int record_number = 0;
  size_t num_examples = 0;

  std::vector<tf::Example> examples;
  for (const auto& name : filenames) {
    const string sgf = ReadFileToString(name);
    int new_examples = GameRecordToExamples(sgf, ShortFilePath(name),
                                            &examples);
    if (new_examples > 0) {
      LOG(INFO) << "Converted " << new_examples << " examples from " << name;
    } else {
      LOG(WARNING) << "Failed in processing " << name;
      ++failed_files;
    }
    if (static_cast<int32_t>(examples.size()) > FLAGS_examples_per_file) {
     num_examples += examples.size();
     FlushExamples(FLAGS_output, record_number++, std::move(examples));
    }
  }
  if (!examples.empty()) {  // Flush the remaining examples in the last batch.
    num_examples += examples.size();
    FlushExamples(FLAGS_output, record_number++, std::move(examples));
  }
  LOG(INFO) << "Total examples: " << num_examples << ", " << failed_files
            << " files failed out of " << filenames.size() << " files";
}

}  // namespace
}  // namespace zebra_go

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  CHECK(!FLAGS_sgf_files.empty());
  CHECK(!FLAGS_output.empty());
  zebra_go::Run();
  return 0;
}
