#include "engine/go_game.h"

#include "benchmark/benchmark.h"
#include "engine/sgf_utils.h"
#include "sgf_parser/parser.h"

namespace zebra_go {

int ReplayGame(const sgf_parser::GameRecord& game, bool estimate_territory) {
  int signature = 0;
  std::unique_ptr<GoBoard> board(
      new GoBoard(game.board_width, game.board_height));
  std::vector<GoPosition> dead_stones;
  int num_steps = 1;
  for (const auto& m : game.moves) {
    GoColor player = (m.player == sgf_parser::GoMove::BLACK ? COLOR_BLACK
                                                            : COLOR_WHITE);
    if (player != board->current_player()) {
      board->Move(kMovePass, estimate_territory, &dead_stones);
    }
    CHECK_EQ(player, board->current_player());
    GoPosition move = m.pass ? kMovePass : m.move;
    CHECK(board->Move(move, estimate_territory, &dead_stones))
        << "Illegal move #" << num_steps << ": " << ToString(move);
    // During the game "testdata/cj_supermatch_1991.sgf", at these steps
    // some stones are captured:
    // #48: D3, #64: C14, #83: S4, #90: R7, #97: P8,O8, #175: E12
    signature += (num_steps * dead_stones.size());
    num_steps++;
  }
  return signature;
}

/** 2019-04-12
Run on (12 X 4600 MHz CPU s)
CPU Caches:
  L1 Data 32K (x6)
  L1 Instruction 32K (x6)
  L2 Unified 256K (x6)
  L3 Unified 12288K (x1)
Load Average: 0.25, 0.22, 0.19
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may
be noisy and will incur extra overhead.
--------------------------------------------------------
Benchmark              Time             CPU   Iterations
--------------------------------------------------------
BM_ReplayGame/0     368830 ns       368823 ns         1888
BM_ReplayGame/1    1579381 ns      1579348 ns          434
*/
static void BM_ReplayGame(benchmark::State& state) {
  const std::string sgf = ReadFileToString("testdata/cj_supermatch_1991.sgf");
  sgf_parser::GameRecord game;
  std::string errors;
  if (!sgf_parser::SimpleParseSgf(sgf, &game, nullptr, &errors)) {
    LOG(FATAL) << "Failed in parsing SGF: " << errors;
  }

  for (auto _ : state) {
    CHECK_EQ(654, ReplayGame(game, state.range(0)));
  }
}

// 0: don't estimate territory; 1: estimate territory at every step.
BENCHMARK(BM_ReplayGame)->Arg(0)->Arg(1);

}  // namespace zebra_go

BENCHMARK_MAIN();
