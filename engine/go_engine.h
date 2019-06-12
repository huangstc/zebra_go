#ifndef ZEBRA_GO_ENGINE_GO_ENGINE_H_
#define ZEBRA_GO_ENGINE_GO_ENGINE_H_

#include "engine/go_game.h"
#include "engine/scorer.h"

namespace zebra_go {

// A Go engine maintains the state of a Go game and responds to commands of GTP
// (Go Text Protocol).
class GoEngine {
 public:
  GoEngine() {}
  virtual ~GoEngine() {}

  // Game settings.
  virtual void SetBoardSize(GoSizeT size);
  virtual void SetKomi(float komi);
  virtual void ClearBoard();

  // Updates the Go board by placing the player's stone at the given position.
  virtual void Play(GoColor player, GoPosition move);

  // Gets the next move for current player. May return kMovePass or kMoveResign
  // if current player wants to pass or resign.
  virtual GoPosition GenMove(GoColor player) = 0;

 protected:
  std::unique_ptr<GoBoard> board_;
};

// A simple implementation of GoEngine for testing.
class SimpleEngine : public GoEngine {
 public:
  SimpleEngine();
  ~SimpleEngine() override {}

  GoPosition GenMove(GoColor player) override;

 private:
  std::unique_ptr<AsyncScorer> scorer_;
};

// An engine using Monte Carlo tree search.
class MctsEngine : public GoEngine {
 public:
  MctsEngine();
  ~MctsEngine();

  GoPosition GenMove(GoColor player) override;

 private:
  std::unique_ptr<AsyncScorer> scorer_;
};

}  // namespace zebra_go

#endif  // ZEBRA_GO_ENGINE_GO_ENGINE_H_
