#include "engine/go_game.h"

#include <algorithm>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace zebra_go {

using absl::StrAppend;

const int16_t INVALID_ID = 0;
const int16_t MIN_CHAIN_ID = 1000;
const int16_t MAX_CHAIN_ID = 3000;
const int16_t MIN_AREA_ID = 10000;
const int16_t MAX_AREA_ID = 20000;

std::string ToString(GoPosition pos) {
  if (pos.first == kNPos.first) return "unset";
  if (pos.first == kMovePass.first) return "pass";
  if (pos.first == kMoveResign.first) return "resign";
  char label = (pos.first >= 8) ? 'A' + pos.first + 1 : 'A' + pos.first;
  return absl::StrCat(std::string(1, label), pos.second + 1);
}

GoColor GetOpponent(GoColor s) {
  if (s == COLOR_BLACK) return COLOR_WHITE;
  if (s == COLOR_WHITE) return COLOR_BLACK;
  return COLOR_NONE;
}

static const size_t kNumFeaturePlanes = 7;

std::string GoFeatureSet::GetPlaneName(int idx) const {
  static const char* kPlaneNames[] = {
    "orig", "b1", "b2", "b3", "w1", "w2", "w3",
  };
  return kPlaneNames[idx];
}

GoFeatureSet::GoFeatureSet(GoSizeT width, GoSizeT height)
    : width_(width), height_(height), planes_(kNumFeaturePlanes) {
  for (size_t i = 0; i < planes_.size(); ++i) {
    planes_[i].resize(width_ * height_);
  }
}

void GoFeatureSet::CopyFrom(const GoFeatureSet& other) {
  CHECK_EQ(width_, other.width_);
  CHECK_EQ(height_, other.height_);
  CHECK_EQ(planes_.size(), other.planes_.size());
  for (size_t i = 0; i < other.planes_.size(); ++i) {
    planes_[i] = other.planes_[i];
  }
}

void GoFeatureSet::Reset() {
  for (size_t i = 0; i < planes_.size(); ++i) {
    std::fill(planes_[i].begin(), planes_[i].end(), 0.0f);
  }
}

GoBoard::NeighorIterator::NeighorIterator(const GoBoard* b, GoPosition c)
    : board(b), center(c), offset(-1) {
  ++(*this);
};

GoBoard::NeighorIterator& GoBoard::NeighorIterator::operator++() {
  static const std::pair<GoSizeT, GoSizeT> kDeltas[] = {
      {0, -1}, {0, 1}, {-1, 0}, {1, 0}
  };
  CHECK(!end()) << "Trying to advance after reaching the end.";
  ++offset;
  for (; offset < 4; ++offset) {
    current = std::make_pair(center.first + kDeltas[offset].first,
                             center.second + kDeltas[offset].second);
    if (board->IsValidPosition(current)) {
      return *this;
    }
  }
  return *this;
}

GoBoard::GoBoard(GoSizeT width, GoSizeT height)
    : width_(width), height_(height), current_player_(COLOR_BLACK),
      next_chain_id_(INVALID_ID + 1), ko_(kNPos) {
  CHECK_GT(width_, 0);
  CHECK_GT(height_, 0);
  stones_.resize(width_ * height_);
  std::fill(stones_.begin(), stones_.end(), COLOR_NONE);

  chains_.resize(width_ * height_);
  std::fill(chains_.begin(), chains_.end(), INVALID_ID);

  features_ = absl::make_unique<GoFeatureSet>(width, height);
  UpdateFeatureSet();

  approx_territory_.resize(3);
  approx_territory_[0] = width_ * height_;
  approx_territory_[1] = 0;
  approx_territory_[2] = 0;
}

std::unique_ptr<GoBoard> GoBoard::Clone() const {
  auto c = absl::make_unique<GoBoard>(width_, height_);
  c->current_player_ = current_player_;
  c->stones_ = stones_;
  c->chains_ = chains_;
  for (const auto& iter : chain_map_) {
    c->chain_map_.insert(std::make_pair(iter.first, iter.second->Clone()));
  }
  c->next_chain_id_ = next_chain_id_;
  c->ko_ = ko_;
  c->forbidden_positions_ = forbidden_positions_;
  c->features_->CopyFrom(*features_);
  c->approx_territory_ = approx_territory_;
  return c;
}

bool GoBoard::IsLegalMove(GoPosition move) const {
  if (move == kMovePass || move == kMoveResign) {
    return true;
  }
  if (!IsValidPosition(move)) {  // Out of the board's boundary.
    return false;
  }
  if (GetStone(move) != COLOR_NONE) {  // The new position is occupied.
    return false;
  }
  if (ko_ != kNPos && move == ko_) {
    return false;
  }
  if (forbidden_positions_.find(move) != forbidden_positions_.end()) {
    return false;
  }
  return true;
}

// Dead stones of the opponent will be put to "dead".
bool GoBoard::Move(GoPosition move, bool estimate_territory,
                   std::vector<GoPosition>* captured_stones) {
  if (!IsLegalMove(move)) {
    return false;
  }
  if (move == kMoveResign) {
    return true;
  }
  if (move == kMovePass) {
    current_player_ = GetOpponent(current_player_);
    UpdateForbiddenPositions();
    UpdateFeatureSet();
    if (estimate_territory) {
      EstimateTerritory();
    }
    return true;
  }

  // We need to store dead stones when updating the board. So keep them in
  // this vector even if the caller doesn't need them.
  std::vector<GoPosition> deads;
  if (captured_stones == nullptr) {
    captured_stones = &deads;
  }
  captured_stones->clear();

  // Reset Ko
  ko_ = kNPos;

  // Remove captured chains.
  std::set<GoChain*> neighbors;  // Adjacent chains of the same color.
  std::set<GoChain*> opponents;  // Adjacent opponent chains.
  std::vector<GoPosition> liberties;
  GetAdjacentChains(move, &neighbors, &opponents, &liberties);

  // Move.
  SetStone(move, current_player_);

  // Remove captured chains.
  for (GoChain* chain : opponents) {
    if (chain->liberties.size() == 1 && move == chain->FirstLiberty()) {
      // This move captures the chain.
      RemoveChain(chain, captured_stones);
    } else {  // Update the liberties of the other opponents.
      chain->liberties.erase(move);
    }
  }
  opponents.clear();

  // Merge neighbors or create a new chain.
  GoChain* new_chain = nullptr;
  if (neighbors.empty()) {
    new_chain = CreateNewChain(move, liberties);
  } else {
    MergeChains(move, liberties, &neighbors);
  }

  std::set<GoChain*> dummy;
  for (GoPosition removed_stone : *captured_stones) {
    std::set<GoChain*> touched_chains;
    GetAdjacentChains(removed_stone, &touched_chains, &dummy, nullptr);
    DCHECK(dummy.empty());
    for (GoChain* chain : touched_chains) {
      chain->liberties.insert(removed_stone);
    }
  }

  // It is a ko if the ko position is the only liberty of the new stone.
  if (captured_stones->size() == 1 && new_chain != nullptr &&
      new_chain->liberties.size() == 1 &&
      new_chain->FirstLiberty() == (*captured_stones)[0]) {
    ko_ = (*captured_stones)[0];
  }

  // Done.
  current_player_ = GetOpponent(current_player_);
  UpdateForbiddenPositions();
  UpdateFeatureSet();
  if (estimate_territory) {
    EstimateTerritory();
  } else {
    std::fill(approx_territory_.begin(), approx_territory_.end(), 0);
  }
  return true;
}

GoBoard::GoChain* GoBoard::GetChain(GoPosition p) const {
  DCHECK(IsValidPosition(p));
  const int16_t chain_id = chains_[Encode(p)];
  if (chain_id == INVALID_ID) return nullptr;
  const auto iter = chain_map_.find(chain_id);
  DCHECK(iter != chain_map_.end());
  return iter->second.get();
}

void GoBoard::GetAdjacentChains(GoPosition pos,
                                std::set<GoChain*>* neighbors,
                                std::set<GoChain*>* opponents,
                                std::vector<GoPosition>* liberties) const {
  for (NeighorIterator iter(this, pos); !iter.end(); ++iter) {
    GoPosition cur = *iter;
    if (GetStone(cur) == COLOR_NONE) {
      if (liberties != nullptr) {
        liberties->push_back(cur);
      }
    } else {
      GoChain* chain = GetChain(cur);
      CHECK(chain != nullptr);
      if (chain->color == this->current_player()) {
        neighbors->insert(chain);
      } else {
        opponents->insert(chain);
      }
    }
  }
}

void GoBoard::RemoveChain(const GoChain* c,
                          std::vector<GoPosition>* dead_stones) {
  for (GoPosition stone : c->stones) {
    dead_stones->emplace_back(stone);
    SetChainId(stone, INVALID_ID);
    SetStone(stone, COLOR_NONE);
  }
  chain_map_.erase(c->chain_id);
}

GoBoard::GoChain* GoBoard::CreateNewChain(
    GoPosition stone, const std::vector<GoPosition>& liberties) {
  auto chain = absl::make_unique<GoBoard::GoChain>(
      current_player_, next_chain_id_++);
  chain->stones.push_back(stone);
  chain->liberties.insert(liberties.begin(), liberties.end());
  SetChainId(stone, chain->chain_id);
  auto* result = chain.get();
  chain_map_.insert(std::make_pair(chain->chain_id, std::move(chain)));
  return result;
}

void GoBoard::MergeChains(
    GoPosition joint, const std::vector<GoPosition>& liberties,
    std::set<GoChain*>* chains) {
  DCHECK(!chains->empty());
  std::set<GoChain*>::const_iterator iter = chains->begin();
  GoChain* merged = *iter;
  const int16_t cid = merged->chain_id;

  for (++iter; iter != chains->end(); ++iter) {
    GoChain* from = *iter;
    for (GoPosition stone : from->stones) {
      SetChainId(stone, cid);
      merged->stones.push_back(stone);
    }
    merged->liberties.insert(from->liberties.begin(), from->liberties.end());
    chain_map_.erase(from->chain_id);
  }
  merged->stones.push_back(joint);
  SetChainId(joint, cid);
  merged->liberties.erase(joint);
  merged->liberties.insert(liberties.begin(), liberties.end());
}

void GoBoard::UpdateForbiddenPositions() {
  forbidden_positions_.clear();
  // A position is forbidden if it is the only liberty of a chain of the
  // current player and:
  //  * it cannot extend the chain to get more liberties;
  //  * it cannot capture an opponent chain;
  //  * it cannot connect to a chain of current player to form a live chain.
  for (const auto& iter : chain_map_) {
    const GoChain* current_chain = iter.second.get();
    if (current_chain->color != current_player()) continue;
    if (current_chain->liberties.size() >= 2) continue;
    GoPosition only_lib = current_chain->FirstLiberty();  // first and only.
    bool is_forbidden = true;
    for (NeighorIterator iter(this, only_lib); !iter.end(); ++iter) {
      GoPosition p = *iter;
      if (GetStone(p) == COLOR_NONE) {
        is_forbidden = false;
        break;
      }
      GoChain* chain = GetChain(p);
      DCHECK(chain != nullptr);
      if (chain == current_chain) continue;
      if (chain->color != current_player() &&
          chain->liberties.size() == 1) {
        // Terminate the loop because this is an opponent chain and can
        // be captured by this move.
        is_forbidden = false;
        break;
      }
      if (chain->color == current_player() &&
          chain->liberties.size() >= 2) {
        // Terminate the loop because this is a friend chain and the
        // merged chain will have at least one liberty.
        is_forbidden = false;
        break;
      }
    }
    if (is_forbidden) {
      forbidden_positions_.insert(only_lib);
    }
  }
}

void GoBoard::UpdateFeatureSet() {
  features_->Reset();
  const bool flip = (current_player() == COLOR_WHITE);
  int pid = 0;  // orig
  for (GoSizeT x = 0; x < width(); ++x) {
    for (GoSizeT y = 0; y < height(); ++y) {
      const GoColor color = GetStone({x, y});
      float value = 0.0f;
      if (color == COLOR_BLACK) {
        value = flip ? -1.0 : 1.0;
      } else if (color == COLOR_WHITE) {
        value = flip ? 1.0 : -1.0;
      }
      features_->Set(pid, x, y, value);
    }
  }

  for (const auto& iter : chain_map_) {
    const auto& chain = *iter.second;
    if (chain.liberties.size() > 3) {
      continue;
    }
    int pid = chain.liberties.size();   // b1, b2 or b3
    if (chain.color != current_player()) {
      pid += 3;  // w1, w2 or w3
    }
    for (const auto& stone : chain.stones) {
      features_->Set(pid, stone.first, stone.second, 1);
    }
  }
}

void GoBoard::EstimateTerritory() {
  std::fill(approx_territory_.begin(), approx_territory_.end(), 0);
  GoSizeT& unknown = approx_territory_[0];
  GoSizeT& black = approx_territory_[1];
  GoSizeT& white = approx_territory_[2];

  for (size_t i = 0; i < stones_.size(); ++i) {
    if (stones_[i] == COLOR_BLACK) {
      black += 1;
    } else if (stones_[i] == COLOR_WHITE) {
      white += 1;
    }
  }
  // Heuristic: don't run when the game just begins.
  if (black + white < 11) {
    unknown = height_ * width_ - black - white;
    return;
  }

  // 0: unvisited; [3, max): region id.
  std::vector<GoSizeT> region_id(width_ * height_, 0);
  GoSizeT next_region_id = 3;
  std::vector<GoPosition> stack;
  for (GoSizeT x = 0; x < width(); ++x) {
    for (GoSizeT y = 0; y < height(); ++y) {
      if (GetStone({x, y}) != COLOR_NONE) {
        // occupied by a stone.
        continue;
      }
      if (region_id[Encode({x, y})] != 0) {
        // already visited.
        continue;
      }

      // Start a new region.
      const GoSizeT current_region_id = next_region_id++;
      region_id[Encode({x, y})] = current_region_id;
      int num_black_neighbors = 0;
      int num_white_neighbors = 0;
      int num_visited = 0;
      stack.clear();
      stack.push_back(std::make_pair(x,y));
      num_visited++;

      // Floodfill from current coordinate.
      while (!stack.empty()) {
        GoPosition cur = stack.back();
        stack.pop_back();
        for (NeighorIterator iter(this, cur); !iter.end(); ++iter) {
          const GoPosition neighbor = *iter;
          const auto color = GetStone(neighbor);
          if (color == COLOR_BLACK) {  // Existing black stone.
            ++num_black_neighbors;
          } else if (color == COLOR_WHITE) {
            // Existing white stone.
            ++num_white_neighbors;
          } else {
            DCHECK_EQ(COLOR_NONE, color);
            if (region_id[Encode(neighbor)] == 0) {  // Unvisited coordinate.
              stack.push_back(neighbor);
              region_id[Encode(neighbor)] = current_region_id;
              num_visited++;
            }
          }
        }
      }

      if ((num_black_neighbors != 0 && num_white_neighbors != 0) ||
          (num_black_neighbors == 0 && num_white_neighbors == 0)) {
        unknown += num_visited;
      } else if (num_black_neighbors != 0 && num_white_neighbors == 0) {
        black += num_visited;
      } else if (num_black_neighbors == 0 && num_white_neighbors != 0) {
        white += num_visited;
      }
    }  // for y
  }  // for x
}

std::string GoBoard::DebugString(bool output_chains) const {
  std::string ascii;

  // Print basic information:
  StrAppend(&ascii, "Current player: ", current_player());

  if (ko_ != kNPos) {
    StrAppend(&ascii, ", Ko: ", ToString(ko_), "\n");
  } else {
    StrAppend(&ascii, "\n");
  }

  // Print x axis on the top.
  std::string x_axis("    ");
  for (GoSizeT i = 0; i < width(); ++i) {
    char label = (i >= 8) ? 'A' + i + 1 : 'A' + i;
    StrAppend(&x_axis, std::string(1, label), " ");
  }
  StrAppend(&ascii, "\n", x_axis, "\n");
  static const char* kSyms[] = {" +", " X", " O"};
  for (GoSizeT j = height() - 1; j >= 0; --j) {
    StrAppend(&ascii, absl::Dec(j+1, absl::kZeroPad2), "|");
    for (GoSizeT i = 0; i < width(); ++i) {
      StrAppend(&ascii, kSyms[GetStone({i, j})]);
    }
    StrAppend(&ascii, "|", absl::Dec(j+1, absl::kZeroPad2), "\n");
  }
  // Print x axis at the bottom.
  StrAppend(&ascii, x_axis, "\n");

  // Print chains if requested.
  if (output_chains) {
    for (const auto& iter : chain_map_) {
      StrAppend(&ascii, iter.second->DebugString());
    }
  }

  // Print forbidden positions:
  if (!forbidden_positions_.empty()) {
    StrAppend(&ascii, "Forbidden: ");
    for (GoPosition pos : forbidden_positions_) {
      StrAppend(&ascii, ToString(pos), ", ");
    }
    StrAppend(&ascii, "\n");
  }
  return ascii;
}

std::unique_ptr<GoBoard::GoChain> GoBoard::GoChain::Clone() const {
  auto copy = absl::make_unique<GoBoard::GoChain>(color, chain_id);
  copy->stones = stones;
  copy->liberties = liberties;
  return copy;
}

std::string GoBoard::GoChain::DebugString() const {
  std::string ascii;
  StrAppend(&ascii, (color == COLOR_BLACK ? "Black" : "White"),
            " Chain #", chain_id, ", #lib=", liberties.size(), ", Stones: ");
  for (const auto& s : stones) {
    StrAppend(&ascii, ToString(s), ", ");
  }
  StrAppend(&ascii, "\n");
  return ascii;
}

}  // namespace zebra_go
