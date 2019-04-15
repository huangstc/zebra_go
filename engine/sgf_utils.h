#ifndef ZEBRA_GO_ENGINE_SGF_UTILS_H_
#define ZEBRA_GO_ENGINE_SGF_UTILS_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "engine/go_game.h"
#include "sgf_parser/parser.h"

namespace zebra_go {

// Loads a game record to a GoBoard instance.
// Example input:
//  static const char kSgf[] = R"((
//    ;GM[1]FF[4]CA[UTF-8]AP[test]SZ[9]
//    ;AB[ba][ab][cb][db][bd][cd][ed][ae][de]
//    ;AW[ca][da][bb][eb][bc][dc][ad][dd][ce]
//    ;B[cg];W[gc]
//  ))";
// It may return null in case of any error.
std::unique_ptr<GoBoard> SgfToGoBoard(const std::string& sgf);

// Replays the game parsed from "sgf". Runs "begin_cb" when the GoBoard is just
// created. Caller can terminate the replay at this step by returning false
// in "begin_cb". Then it runs "callback" at each step during the replay and
// runs "end_callback" when reaching the end of the game. Returns false on any
// error. In case of an error, "callback" or "end_callback" may not be called.
struct ReplayContext {
  const GoBoard* board = nullptr;
  int num_steps        = 0;
  GoPosition next_move = kNPos;
  float game_result    = 0.0f;
};
bool ReplayGame(const std::string& sgf,
                std::function<bool(const ReplayContext&)> begin_cb,
                std::function<void(const ReplayContext&)> callback,
                std::function<void(std::unique_ptr<GoBoard>)> end_callback);

// Reads the file content to a string.
std::string ReadFileToString(const std::string& filename);

// Finds files by pattern.
std::vector<std::string> Glob(const std::string& pattern);

}  // zebra_go

#endif  // ZEBRA_GO_ENGINE_SGF_UTILS_H_
