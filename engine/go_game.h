#ifndef ZEBRA_GO_ENGINE_GO_GAME_H_
#define ZEBRA_GO_ENGINE_GO_GAME_H_

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glog/logging.h>

namespace zebra_go {

// Color of a player, a stone or an area.
enum GoColor {
  COLOR_NONE  = 0,
  COLOR_BLACK = 1,
  COLOR_WHITE = 2,
};

// Integer type for the coordinate system.
typedef int16_t GoSizeT;

// Position.
typedef std::pair<GoSizeT, GoSizeT> GoPosition;

constexpr GoPosition kNPos = {-1, -1};        // Not a valid position.
constexpr GoPosition kMovePass = {-2, -2};    // A player passes.
constexpr GoPosition kMoveResign = {-3, -3};  // A player resigns.

// A1 to T19 in a 19*19 game. A1: lower left; T19: upper right.
std::string ToString(GoPosition pos);

// Gets the color of the opponent of player "s".
GoColor GetOpponent(GoColor s);

class GoFeatureSet;

class GoBoard {
 public:
  explicit GoBoard(GoSizeT size) : GoBoard(size, size) {}
  GoBoard(GoSizeT width, GoSizeT height);
  ~GoBoard() {}

  // Deep copy.
  std::unique_ptr<GoBoard> Clone() const;

  // Gets the board size.
  GoSizeT width() const  { return width_;  }
  GoSizeT height() const { return height_; }

  // The player who is going to play the next move.
  GoColor current_player() const { return current_player_; }

  // Gets the stone color of a position.
  GoColor GetStone(GoPosition pos) const {
    return stones_[Encode(pos)];
  }

  // Checks if the move is legal for current player.
  bool IsLegalMove(GoPosition move) const;

  // Current player plays at the position. Dead stones of the opponent caused
  // by this move will be put to "captured_stones", if it is not null.
  // estimate_territory is optional, run it only when necessary because
  // it is slow.
  bool Move(GoPosition move, bool estimate_territory,
            std::vector<GoPosition>* captured_stones);
  bool Move(GoPosition move, std::vector<GoPosition>* captured_stones) {
    return Move(move, false, captured_stones);
  }

  // Gets current feature set, which will be used by machine learning models
  // to compute the next move for current player.
  const GoFeatureSet& GetFeatures() const { return *features_; }

  // Gets estimated territory of each player. 0: shared; 1: black; 2: white.
  std::tuple<GoSizeT, GoSizeT, GoSizeT> GetApproxPoints() const {
    return std::make_tuple(approx_territory_[0], approx_territory_[1],
                           approx_territory_[2]);
  }

  // Encodes a board coordinate to an integer in [0, width * height).
  GoSizeT Encode(GoPosition pos) const {
    DCHECK(IsValidPosition(pos));
    return pos.second * width() + pos.first;
  }

  // Inverse operation of Encode.
  GoPosition Decode(GoSizeT s) const {
    GoPosition p = std::make_pair<GoSizeT, GoSizeT>(s % width(), s / width());
    DCHECK(IsValidPosition(p));
    return p;
  }

  // Prints a readable string for debugging.
  std::string DebugString(bool output_chains) const;

 private:
  GoBoard() = delete;

  struct GoChain {
    GoChain(GoColor chain_color, int16_t id) : color(chain_color),
                                               chain_id(id) {}
    ~GoChain() {}

    // Gets the first liberty.
    GoPosition FirstLiberty() const {
      DCHECK(!liberties.empty());
      return *liberties.begin();
    }

    // Returns a deep copy.
    std::unique_ptr<GoChain> Clone() const;

    // Returns a readable string for debugging.
    std::string DebugString() const;

    const GoColor color;
    const int16_t chain_id;
    std::vector<GoPosition> stones;
    std::set<GoPosition> liberties;
  };

  // Iterator class to loop over a position's neighbor coordinates.
  struct NeighorIterator {
    NeighorIterator(const GoBoard* b, GoPosition c);

    // Moves to the next position.
    NeighorIterator& operator++();

    GoPosition operator* () const {
      CHECK(!end()) << "Trying to read after reaching the end.";
      return current;
    }

    bool end() const { return offset >= 4; }

    const GoBoard* board;
    GoPosition center;
    GoPosition current;
    int offset;
  };

  // Boundary check.
  bool IsValidPosition(GoPosition move) const {
    return (move.first >= 0 && move.first < width() &&
            move.second >= 0 && move.second < height());
  }

  void SetStone(GoPosition pos, GoColor color) {
    stones_[Encode(pos)] = color;
  }
  void SetChainId(GoPosition pos, int16_t chain_id) {
    chains_[Encode(pos)] = chain_id;
  }

  // Gets the chain that occupies the position. Returns nullptr if there is no
  // stone on the position.
  GoChain* GetChain(GoPosition p) const;

  // Gets the chains and liberties (unoccupied coordinates) that are adjacent
  // to the position. Chains of current player will be returned in "neighbors"
  // and chains of the opponent will be returned in "opponents". "liberties" is
  // nullable, if caller doesn't care about liberties.
  void GetAdjacentChains(GoPosition pos,
                         std::set<GoChain*>* neighbors,
                         std::set<GoChain*>* opponents,
                         std::vector<GoPosition>* liberties) const;

  // Removes the chain from the board and puts its stones in "deads".
  void RemoveChain(const GoChain* c, std::vector<GoPosition>* deads);

  // Creates a new chain with the stone which has the given liberties.
  GoChain* CreateNewChain(GoPosition stone,
                          const std::vector<GoPosition>& liberties);

  // Merges the chains into one chain, where these chains must be able to be
  // joined together by the stone. "liberties" are the joint stone's liberties.
  void MergeChains(GoPosition joint, const std::vector<GoPosition>& liberties,
                   std::set<GoChain*>* chains);

  // Computes forbidden positions, a.k.a. suicide positions, for current player.
  void UpdateForbiddenPositions();

  // Computes the feature planes for current player.
  void UpdateFeatureSet();

  // Estimates terriotory for each player. Results can be accessed through
  // GetApproxPoints.
  void EstimateTerritory();

  const GoSizeT width_, height_;

  // The player who is going to play the next move.
  GoColor current_player_;

  std::vector<GoColor> stones_;   // coordinate-to-stone map.
  std::vector<int16_t> chains_;   // coordinate-to-chain-id map.

  // Map from chain ID to chain.
  std::unordered_map<int16_t, std::unique_ptr<GoChain>> chain_map_;

  // Next unused chain ID.
  int16_t next_chain_id_;

  // It is set when there is a ko situation, where current player is prohibited
  // from capturing the opponent's stone at this position. If there is no ko on
  // the board, it is set to kNPos.
  GoPosition ko_;

  // Suicide positions. If current player places a stone on one of such
  // positions, it will end up with a chain with no liberties without capturing
  // any opposing stones. Therefore, such move is prohibited.
  std::set<GoPosition> forbidden_positions_;

  // Feature set for current player.
  std::unique_ptr<GoFeatureSet> features_;

  // Approximate points of each party. The size of this vector is always 3,
  // corresponding to unknown, black and white, respectively.
  std::vector<GoSizeT> approx_territory_;
};

class GoFeatureSet {
 public:
  GoFeatureSet(GoSizeT width, GoSizeT height);

  // Accessors.
  int num_planes() const  { return planes_.size(); }
  GoSizeT width() const  { return width_; }
  GoSizeT height() const { return height_; }
  const std::vector<float>& plane(int idx) const { return planes_[idx]; }
  std::string GetPlaneName(int idx) const;

  // Deep copy.
  void CopyFrom(const GoFeatureSet& other);
  
  std::unique_ptr<GoFeatureSet> Clone() const {
    std::unique_ptr<GoFeatureSet> copy(new GoFeatureSet(width_, height_));
    copy->CopyFrom(*this);
    return copy;
  }

  // Sets the value at (x,y) of a plane.
  void Set(int plane_id, GoSizeT x, GoSizeT y, float value) {
    planes_[plane_id][y * width_ + x] = value;
  }

  // Resets all values to 0.
  void Reset();

 private:
  const GoSizeT width_, height_;
  std::vector<std::vector<float>> planes_;
};

}  // namespace zebra_go

#endif  // ZEBRA_GO_ENGINE_GO_GAME_H_
