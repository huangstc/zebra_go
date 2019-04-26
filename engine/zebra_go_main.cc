#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "engine/go_engine.h"
#include "engine/go_game.h"
#include "glog/logging.h"
#include "gflags/gflags.h"

DECLARE_bool(logtostderr);

DEFINE_bool(simple_engine, true,
           "Run a trivial engine to test other parts of the system.");

namespace zebra_go {

using ::std::string;

class GtpServer {
 public:
  GtpServer() {
    if (FLAGS_simple_engine) {
      engine_.reset(new SimpleEngine());
      engine_->SetBoardSize(19);
    } else {
      LOG(FATAL) << "Not implemented.";
    }
    RegisterHandlers();
  }

  void Run(std::istream& in, std::ostream& out) {
    string command;
    while (std::getline(in, command)) {
      int id = kNoId;
      string op;
      std::vector<string> args;
      ParseGtpCommand(command, &id, &op, &args);

      if (op == "quit") {
        LOG(INFO) << "Bye.";
        return;
      }
      LOG(INFO) << "Parsed command: [" << id << "] " << op << " args=["
                << absl::StrJoin(args, " ") << "]";

      auto iter = handlers_.find(op);
      if (iter == handlers_.end()) {
        LOG(ERROR) << "Unknown command: " << command;
        continue;
      }
      string output;
      const bool success = (iter->second)(args, &output);
      LOG(INFO) << (success ? "OK. " : "Failed! " ) << output;

      out << (success ? "=" : "?");
      if (id != kNoId) {
        out << id;
      }
      if (!output.empty()) {
        out << output;
      }
      out << std::endl << std::endl;
      google::FlushLogFiles(google::GLOG_INFO);
    }
  }

 private:
  typedef std::function<bool(const std::vector<string>&, string*)> Handler;

  static const int kNoId = -1;

  static void ParseGtpCommand(
      string input, int* id, string* command_name, std::vector<string>* args) {
    absl::AsciiStrToLower(&input);
    std::vector<string> splits = absl::StrSplit(input, " ", absl::SkipEmpty());
    // Try to parse ID first:
    size_t offset = 0;
    if (absl::SimpleAtoi(splits[0], id)) {
      offset = 1;
    } else {
      *id = kNoId;
    }
    // Copy command_name:
    *command_name = splits[offset++];
    // Copy args:
    args->assign(splits.begin() + offset, splits.end());
  }

  void RegisterHandlers() {
    // "quit" is a special command that has no handler.
    supported_commands_.push_back("quit");

#define SIMPLE_HANDLER(op, fixed_output) do {                              \
    RegisterHandler(                                                       \
        (op), [this](const std::vector<string>& args, string* output) {    \
          *output = (fixed_output);                                        \
          return true;                                                     \
        });                                                                \
  } while (0)

    SIMPLE_HANDLER("name", "ZebraGo");
    SIMPLE_HANDLER("version", "0.1");
    SIMPLE_HANDLER("protocol_version", "2");
    SIMPLE_HANDLER("final_score", "0");  // Actually not implemented.

#undef SIMPLE_HANDLER

    RegisterHandler(
        "list_commands",
        [this](const std::vector<string>& args, string* output) -> bool {
          *output = absl::StrJoin(supported_commands_, "\n");
          return true;
        });
    RegisterHandler(
        "boardsize",
        [this](const std::vector<string>& args, string* output) -> bool {
          int size;
          if (args.size() == 1 && absl::SimpleAtoi(args[0], &size)) {
            engine_->SetBoardSize(static_cast<GoSizeT>(size));
            return true;
          } else {
            *output = "Failed in parsing board size.";
            return false;
          }
        });
    RegisterHandler(
        "clear_board",
        [this](const std::vector<string>& args, string* output) -> bool {
          engine_->ClearBoard();
          return true;
        });
    RegisterHandler(
        "komi",
        [this](const std::vector<string>& args, string* output) -> bool {
          double komi = 0.0;
          if (args.size() == 1 && absl::SimpleAtod(args[0], &komi)) {
            engine_->SetKomi(komi);
            return true;
          } else {
            *output = "Failed in parsing komi.";
            return false;
          }
        });
    RegisterHandler(
        "play",
        [this](const std::vector<string>& args, string* output) -> bool {
          // Example input: "play b q16"
          if (args.size() != 2) {
            *output = "Wrong number of arguments of play";
            return false;
          }
          const GoColor player = ColorFromString(args[0]);
          const GoPosition pos = PositionFromString(args[1]);
          if (player == COLOR_NONE || pos == kNPos) {
            *output = "Bad arguments for play";
            return false;
          }
          engine_->Play(player, pos);
          return true;
        });
    RegisterHandler(
        "genmove",
        // Example input: "genmove w"
        [this](const std::vector<string>& args, string* output) -> bool {
          if (args.size() != 1) {
            *output = "Wrong number of arguments of genmove";
            return false;
          }
          const GoColor player = ColorFromString(args[0]);
          LOG(INFO) << "Genmove for player: " << ToString(player);
          const GoPosition move = engine_->GenMove(player);
          *output = ToString(move);
          return true;
        });
  }

  void RegisterHandler(const string& name, Handler handler) {
    handlers_.insert(std::make_pair(name, std::move(handler)));
    supported_commands_.push_back(name);
  }

  std::unique_ptr<GoEngine> engine_;
  std::map<string, Handler> handlers_;
  std::vector<string> supported_commands_;
};

}  // namespace zebra_go

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  zebra_go::GtpServer server;
  server.Run(std::cin, std::cout);

  return 0;
}
