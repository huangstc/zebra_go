#include "engine/sgf_utils.h"

#include <fstream>
#include <glob.h>
#include <string.h>   // for memset

#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "sgf_parser/parser.h"

namespace zebra_go {

using ::sgf_parser::GameRecord;

namespace {

bool PresetStones(GoColor c, const std::vector<sgf_parser::GoPos>& stones,
                  GoBoard* board) {
  std::vector<GoPosition> dead_stones;
  if (board->current_player() != c) {
    board->Move(kMovePass, &dead_stones);
  }
  CHECK_EQ(c, board->current_player());
  for (const auto& pos : stones) {
    const GoPosition move = pos;
    if (!board->IsLegalMove(move)) {
      LOG(WARNING) << "Trying to replay an illegal move:" << ToString(move);
      return false;
    }
    board->Move(move, &dead_stones);
    if (!dead_stones.empty()) {
      LOG(WARNING) << "Captured stones in a preset step: " << ToString(move);
      return false;
    }
    board->Move(kMovePass, &dead_stones);
    CHECK(dead_stones.empty());
  }
  return true;
}

}  // namespace

bool ReplayGame(const std::string& sgf,
                std::function<bool(const ReplayContext&)> begin_cb,
                std::function<void(const ReplayContext&)> callback,
                std::function<void(std::unique_ptr<GoBoard>)> end_callback) {
  // Parse the SGF.
  GameRecord game;
  std::string errors;
  if (!sgf_parser::SimpleParseSgf(sgf, &game, nullptr, &errors)) {
    LOG(WARNING) << "Failed in parsing SGF: " << errors;
    return false;
  }
  if (game.board_width <= 0 || game.board_width >= 27 ||
      game.board_height <= 0 || game.board_height >= 27) {
    LOG(WARNING) << "Bad board size: " << game.board_width
                 << "," << game.board_height;
    return false;
  }

  std::unique_ptr<GoBoard> board(
      new GoBoard(game.board_width, game.board_height));
  ReplayContext context;
  context.board = board.get();
  context.num_steps = 0;
  context.next_move = kNPos;
  context.game_result = game.result;

  if (!begin_cb(context)) {
    return false;
  }

  if (!PresetStones(COLOR_BLACK, game.black_stones, board.get()) ||
      !PresetStones(COLOR_WHITE, game.white_stones, board.get())) {
    return false;
  }

  for (const auto& m : game.moves) {
    GoColor player = (m.player == sgf_parser::GoMove::BLACK ? COLOR_BLACK
                                                            : COLOR_WHITE);
    if (player != board->current_player()) {
      board->Move(kMovePass, false, nullptr);
    }
    context.next_move = (m.pass ? kMovePass : m.move);
    context.num_steps += 1;
    callback(context);
    if (!board->Move(context.next_move, false, nullptr)) {
      LOG(WARNING) << "Illegal move #" << context.num_steps
                   << ": " << ToString(context.next_move);
      return false;
    }
  }

  end_callback(std::move(board));
  return true;
}

std::unique_ptr<GoBoard> SgfToGoBoard(const std::string& sgf) {
  std::unique_ptr<GoBoard> result;
  ReplayGame(sgf,
             [](const ReplayContext&) { return true; },
             [](const ReplayContext&) {},
             [&result](std::unique_ptr<GoBoard> board) {
               result = std::move(board);
             });
  return result;
}

std::string ReadFileToString(const std::string& filename) {
  std::string sgf;
  std::string line;
  std::ifstream myfile(filename);
  if (myfile.is_open()) {
    while (std::getline (myfile, line)) {
      absl::StrAppend(&sgf, line, "\n");
    }
    myfile.close();
  }
  return sgf;
}

std::vector<std::string> Glob(const std::string& pattern) {
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (return_value != 0) {
    globfree(&glob_result);
    LOG(FATAL) << "glob() failed with error code: " << return_value;
  }

  std::vector<std::string> filenames;
  for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
    filenames.push_back(std::string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);
  return filenames;
}

}  // namespace zebra_go
