// Copyright 2026 PDH-Zero contributors. Project-local research use.
// NOTE: Numerical projection only; Python remains the rules/action oracle.
#pragma once
#include <array>
#include <bitset>
#include <cstdint>
#include <vector>

namespace pdh {
constexpr int kPackedState = 1243;
constexpr int kObservation = 622;
constexpr int kActionFeatures = 74;
constexpr int kPackedAction = 7;
using Counts = std::array<int, 34>;
using Routes = std::array<double, 8>;
using Qualifications = std::array<double, 4>;
struct Meld { int kind; int type; };
struct Context {
  Counts own{};
  std::bitset<136> own_ids;
  std::bitset<136> revealed_ids;
  std::bitset<136> known_ids;
  std::array<Counts, 4> discards{}, exposed{}, revealed{};
  std::array<int, 4> discard_size{}, reveal_size{};
  std::array<std::vector<Meld>, 4> melds;
  std::array<double, 34> known{}, risk{};
  Routes before{};
  Qualifications qualifications{};
  int player;
};
// Validated legal projection: no other player's concealed cards or future wall.
Context Parse(const std::int32_t* packed);
Routes Potentials(const Counts& concealed, const std::vector<Meld>& melds);
Qualifications Qualify(const std::vector<Meld>& melds);
// Outputs preserve rc3's fixed observation/action ordering and teacher tie order.
void Observation(const std::int32_t* packed, const Context& ctx, double* out);
double ActionFeatures(const Context& ctx, const std::int32_t* packed,
                      const std::int32_t* action, double* out);
}  // namespace pdh
