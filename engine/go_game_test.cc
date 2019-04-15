#include "engine/go_game.h"

#include "engine/sgf_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zebra_go {
namespace {

using ::testing::UnorderedElementsAreArray;

class GoBoardTest : public ::testing::Test {};

TEST_F(GoBoardTest, Basic) {
  /*
        A B C D E
    05| X + O X +|05
    04| O X X O X|04
    03| + O + O +|03
    02| X O X X O|02
    01| + X O O +|01
        A B C D E
  */
  static const char kSgf[] = R"((
    ;GM[1]FF[4]CA[UTF-8]AP[test]SZ[5]
    ;AB[ba][ab][cb][db][bd][cd][ed][ae][de]
    ;AW[ca][da][bb][eb][bc][dc][ad][dd][ce]
  ))";

  auto original = SgfToGoBoard(kSgf);
  ASSERT_TRUE(original != nullptr);
  original->Move(kMovePass, true, nullptr);  // To estimate territory.
  if (original->current_player() != COLOR_BLACK) {
    original->Move(kMovePass, true, nullptr);
  }
  LOG(INFO) << original->DebugString(true);
  EXPECT_EQ(std::make_tuple(4, 11, 10), original->GetApproxPoints());

  std::vector<GoPosition> deads;

  // Case 1: cature two stones.
  {
    std::unique_ptr<GoBoard> b = original->Clone();
    ASSERT_TRUE(b->Move({4, 0}, &deads));
    const GoPosition expected1[] = {{2,0}, {3,0}};
    EXPECT_THAT(deads, UnorderedElementsAreArray(expected1));

    EXPECT_TRUE(b->IsLegalMove({3, 0}));
    ASSERT_TRUE(b->Move({3, 0}, &deads));
    const GoPosition expected2[] = {{4, 0}};
    EXPECT_THAT(deads, UnorderedElementsAreArray(expected2));
  }

  // Case 2: forbidden move.
  {
    std::unique_ptr<GoBoard> b = original->Clone();
    ASSERT_TRUE(b->Move({2, 2}, &deads));
    EXPECT_TRUE(deads.empty());

    EXPECT_FALSE(b->IsLegalMove({0,2}));

    ASSERT_TRUE(b->Move({1, 4}, &deads));
    const GoPosition expected[] = {{0,4}, {1,3}, {2,3}, {2,2}, {2,1}, {3,1}};
    EXPECT_THAT(deads, UnorderedElementsAreArray(expected));
  }

  // Case 3: Ko
  {
    std::unique_ptr<GoBoard> b = original->Clone();
    ASSERT_TRUE(b->Move({2, 2}, &deads));
    EXPECT_TRUE(deads.empty());

    ASSERT_TRUE(b->Move({4, 4}, &deads));
    ASSERT_EQ(1, deads.size());
    const GoPosition ko = {3,4};
    EXPECT_EQ(ko, deads[0]);

    // {3,4} is a Ko for Black.
    EXPECT_FALSE(b->IsLegalMove(ko));

    // {1,4} and {4,2} are suicide moves for Black.
    EXPECT_FALSE(b->IsLegalMove({1,4}));
    EXPECT_FALSE(b->IsLegalMove({4,2}));

    LOG(INFO) << b->DebugString(true);
  }

  // Case 4: a new chain is created with no initial liberties.
  {
    std::unique_ptr<GoBoard> b = original->Clone();
    ASSERT_TRUE(b->Move({4, 2}, &deads));
    EXPECT_TRUE(deads.empty());
    ASSERT_TRUE(b->Move({0, 2}, &deads));
    EXPECT_TRUE(deads.empty());

    ASSERT_TRUE(b->Move({4, 0}, &deads));
    ASSERT_EQ(3, deads.size());
    const GoPosition expected[] = {{2,0}, {3,0}, {4,1}};
    EXPECT_THAT(deads, UnorderedElementsAreArray(expected));
  }
}

// https://www.101weiqi.com/book/1000/73/1344/
TEST_F(GoBoardTest, KoTest) {
  static const char kSgf[] = R"((
    ;GM[1]FF[4]CA[UTF-8]AP[test]SZ[11]
    ;AB[ba][gb][ac][bc][cc][gc][cd][dd][ed][fd][ce]
    ;AW[cb][fb][dc][ec][fc][hc][bd][gd][hd][be][fe][cf][df][gf]
  ))";
  auto board = SgfToGoBoard(kSgf);
  ASSERT_TRUE(board != nullptr);
  if (board->current_player() != COLOR_BLACK) {
    board->Move(kMovePass, nullptr);
  }
  LOG(INFO) << board->DebugString(true);

  std::vector<GoPosition> deads;
  ASSERT_TRUE(board->Move({3, 1}, &deads));
  EXPECT_TRUE(deads.empty());
  ASSERT_TRUE(board->Move({3, 0}, &deads));
  EXPECT_TRUE(deads.empty());
  ASSERT_TRUE(board->Move({4, 0}, &deads));
  EXPECT_TRUE(deads.empty());
  ASSERT_TRUE(board->Move({4, 1}, &deads));
  ASSERT_EQ(1, deads.size());
  EXPECT_EQ(GoPosition({3, 1}), deads[0]);
  ASSERT_TRUE(board->Move({2, 0}, &deads));
  EXPECT_TRUE(deads.empty());
  ASSERT_TRUE(board->Move({5, 0}, &deads));
  ASSERT_EQ(1, deads.size());
  EXPECT_EQ(GoPosition({4, 0}), deads[0]);
  ASSERT_TRUE(board->Move({1, 1}, &deads));
  EXPECT_TRUE(deads.empty());
  ASSERT_TRUE(board->Move({7, 1}, &deads));
  EXPECT_TRUE(deads.empty());

  // Ko1
  ASSERT_TRUE(board->Move({3, 1}, &deads));
  const GoPosition ko1 = {2, 1};
  ASSERT_EQ(1, deads.size());
  EXPECT_EQ(ko1, deads[0]);
  EXPECT_FALSE(board->IsLegalMove(ko1));

  ASSERT_TRUE(board->Move({6, 0}, &deads));
  const GoPosition expected[] = {{6,1}, {6,2}};
  EXPECT_THAT(deads, UnorderedElementsAreArray(expected));

  // Ko2:
  ASSERT_TRUE(board->Move({4, 0}, &deads));
  const GoPosition ko2 = {3, 0};
  ASSERT_EQ(1, deads.size());
  EXPECT_EQ(ko2, deads[0]);
  EXPECT_FALSE(board->IsLegalMove(ko2));
}

TEST_F(GoBoardTest, ForbiddenMoves1) {
  static const char kSgf[] = R"((
    ;GM[1]FF[4]CA[UTF-8]AP[test]SZ[5]
    ;AB[ca][cb][cc][bc]
    ;AW[ba][bb][ab]
  ))";
  auto board = SgfToGoBoard(kSgf);
  ASSERT_TRUE(board != nullptr);
  if (board->current_player() != COLOR_BLACK) {
    board->Move(kMovePass, nullptr);
  }
  LOG(INFO) << board->DebugString(true);
  board->Move({0, 2}, nullptr);
  EXPECT_FALSE(board->IsLegalMove({0, 0}));
}

TEST_F(GoBoardTest, ForbiddenMoves2) {
  GoBoard board(9, 9);
  ASSERT_TRUE(board.Move({2, 0}, nullptr));
  ASSERT_TRUE(board.Move({1, 0}, nullptr));
  ASSERT_TRUE(board.Move({3, 0}, nullptr));
  ASSERT_TRUE(board.Move({2, 1}, nullptr));
  ASSERT_TRUE(board.Move({8, 8}, nullptr));  // Black's dummy move.
  ASSERT_TRUE(board.Move({3, 1}, nullptr));
  ASSERT_TRUE(board.Move({5, 0}, nullptr));
  ASSERT_TRUE(board.Move({5, 1}, nullptr));
  ASSERT_TRUE(board.Move({6, 0}, nullptr));
  ASSERT_TRUE(board.Move({6, 1}, nullptr));
  ASSERT_EQ(COLOR_BLACK, board.current_player());
  EXPECT_TRUE(board.IsLegalMove({4,0}));
  ASSERT_TRUE(board.Move({8, 7}, nullptr));  // Black's dummy move.
  ASSERT_TRUE(board.Move({7, 0}, nullptr));
  ASSERT_EQ(COLOR_BLACK, board.current_player());
  EXPECT_TRUE(board.IsLegalMove({4,0}));
  ASSERT_TRUE(board.Move({7, 8}, nullptr));  // Black's dummy move.
  ASSERT_TRUE(board.Move({4, 1}, nullptr));
  ASSERT_EQ(COLOR_BLACK, board.current_player());
  EXPECT_FALSE(board.IsLegalMove({4,0}));
}

// Test the function ReplayGame in sgf_utils.
TEST_F(GoBoardTest, ReplayGame) {
  const std::string sgf = ReadFileToString("testdata/shusai_19000415.sgf");
  ReplayGame(
      sgf,
      [](const ReplayContext& ctx) {
        return ctx.board->width() == 19 && ctx.board->height() == 19;
      },
      [](const ReplayContext& ctx) {
        LOG(INFO) << ctx.num_steps;
      },
      [](std::unique_ptr<GoBoard> board) {});
}

}  // namespace
}  // namespace zebra_go
