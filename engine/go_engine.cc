#include "engine/go_engine.h"

#include <tuple>

#include "absl/memory/memory.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_bool(simple_scorer, false, "Use SimpleScorer or TfScorer.");

namespace zebra_go {

void GoEngine::SetBoardSize(GoSizeT size) {
  board_.reset(new GoBoard(size, size));
}

void GoEngine::SetKomi(float komi) {
  LOG(WARNING) << "Not implemented.";
}

void GoEngine::ClearBoard() {
  if (board_ != nullptr) {
    const GoSizeT width = board_->width();
    const GoSizeT height = board_->height();
    board_.reset(new GoBoard(width, height));
  }
}

void GoEngine::Play(GoColor player, GoPosition move) {
  CHECK(board_ != nullptr);
  if (player != board_->current_player()) {
    LOG(ERROR) << "Error: not the turn of " << player;
    return;
  }
  VLOG(1) << "Before:\n " << board_->DebugString(/*output_chains=*/true);
  std::vector<GoPosition> deads;
  board_->Move(move, /*estimate_territory=*/true, &deads);
  LOG(INFO) << "Player " << player << " plays at [" << ToString(move)
            << "]. Captured [" << ToString(deads) << "].";
  VLOG(1) << "After: \n " << board_->DebugString(/*output_chains=*/true);
}

SimpleEngine::SimpleEngine() {
  if (FLAGS_simple_scorer) {
    LOG(INFO) << "Use the simplest scorer in SimpleEngine.";
    scorer_ = absl::make_unique<SimpleScorer>();
  } else {
    LOG(INFO) << "Use the DNN scorer in SimpleEngine.";
    scorer_ = TfScorer::CreateFromFlags();
  }
}

GoPosition SimpleEngine::GenMove(GoColor player) {
  CHECK(board_ != nullptr);
  CHECK(scorer_ != nullptr);
  if (player != board_->current_player()) {
    LOG(ERROR) << "Wrong state: not the turn of " << player << ", terminate.";
    return kMoveResign;
  }

  GoPosition next_move = kMovePass;  // Default to PASS.
  PolicyResult policy;
  ValueResult value;
  if (scorer_->SyncScoreGoState(*board_, &policy, &value)) {
    if (value.first) {  // Current player should resign.
      return kMoveResign;
    }
    next_move = AsyncScorer::SamplePolicy(policy);
  } else {
    LOG(ERROR) << "Scoring failed. Pass.";
  }

  std::vector<GoPosition> deads;
  board_->Move(next_move, /*estimate_territory=*/true, &deads);
  LOG(INFO) << "Player " << player << " plays at [" << ToString(next_move)
            << "]. Captured [" << ToString(deads) << "].";

  auto points = board_->GetApproxPoints();
  LOG(INFO) << "Estimated points: black=" << std::get<1>(points) << ", white="
             << std::get<2>(points) << ", unknown=" << std::get<0>(points);
  return next_move;
}

}  // namespace zebra_go
