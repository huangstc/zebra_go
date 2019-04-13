#include "engine/sgf_utils.h"

#include <fstream>

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

bool PlayGameRecord(const GameRecord& game, GoBoard* board) {
  CHECK_EQ(game.board_width, board->width());
  CHECK_EQ(game.board_height, board->height());

  if (!PresetStones(COLOR_BLACK, game.black_stones, board) ||
      !PresetStones(COLOR_WHITE, game.white_stones, board)) {
    return false;
  }

  int num_steps = 0;
  std::vector<GoPosition> dead_stones;
  for (const auto& m : game.moves) {
    GoColor player = (m.player == sgf_parser::GoMove::BLACK ? COLOR_BLACK
                                                            : COLOR_WHITE);
    if (player != board->current_player()) {
      board->Move(kMovePass, &dead_stones);
    }
    CHECK_EQ(player, board->current_player());
    GoPosition move = m.pass ? kMovePass : m.move;
    if (!board->Move(move, &dead_stones)) {
      LOG(WARNING) << "Illegal move #" << num_steps << ": " << ToString(move);
      return false;
    }
    num_steps++;
  }
  return true;
}

}  // namespace

std::unique_ptr<GoBoard> SgfToGoBoard(const std::string& sgf) {
  GameRecord game;
  std::string errors;
  if (!sgf_parser::SimpleParseSgf(sgf, &game, nullptr, &errors)) {
    LOG(WARNING) << "Failed in parsing SGF: " << errors;
    return nullptr;
  }
  std::unique_ptr<GoBoard> board(
      new GoBoard(game.board_width, game.board_height));
  if (!PlayGameRecord(game, board.get())) {
    return nullptr;
  } else {
    return board;
  }
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

}  // namespace zebra_go
