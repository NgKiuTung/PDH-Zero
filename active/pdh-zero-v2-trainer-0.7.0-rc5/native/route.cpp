// Copyright 2026 PDH-Zero contributors. Project-local research use.
// NOTE: Port of strategy.route_potentials, including reference arithmetic order.
#include "features.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace pdh {
namespace {
bool Orphan(int t) { return t >= 27 || t % 9 == 0 || t % 9 == 8; }
double Clamp(double x) { return std::max(-1.0, std::min(1.0, x)); }
}  // namespace
Qualifications Qualify(const std::vector<Meld>& melds) {
  bool closed = true;
  for (const auto& m : melds) closed = closed && m.kind == 2;
  double empty = melds.empty() ? 1.0 : 0.0;
  return {empty, empty, closed ? 1.0 : 0.0, empty};
}
Routes Potentials(const Counts& concealed, const std::vector<Meld>& melds) {
  Counts counts = concealed;
  for (const auto& m : melds) counts[m.type] += 3;
  int pairs=0, triplets=0, quads=0, honors=0, terminals=0, other=0, unique=0;
  bool orphan_pair=false;
  std::array<int, 3> suits{};
  for (int t=0; t<34; ++t) {
    int n=counts[t];
    pairs += n/2; triplets += n/3; quads += (n==4);
    if (t<27) suits[t/9] += n; else honors += n;
    if (Orphan(t)) {
      terminals += n; unique += (n>0); orphan_pair = orphan_pair || n>=2;
    } else other += n;
  }
  int dominant = *std::max_element(suits.begin(), suits.end());
  int off = std::accumulate(suits.begin(), suits.end(), 0) - dominant;
  double seven=-1, orphans=-1, dragons=0, winds=0;
  if (melds.empty()) {
    seven=Clamp((pairs + 0.35*quads)/7.0);
    orphans=Clamp((unique/13.0)*0.9 + (orphan_pair ? 1.0 : 0.0)*0.1);
  }
  for (int t=31; t<34; ++t) dragons += std::min(counts[t],3)/3.0;
  for (int t=27; t<31; ++t) winds += std::min(counts[t],3)/3.0;
  return {seven, Clamp((std::min(triplets,4)/4.0)*0.85 + (pairs>0 ? 1.0 : 0.0)*0.15),
    Clamp(std::tanh((dominant+honors-2.0*off)/8.0)),
    Clamp(std::tanh((dominant-2.0*off-2.0*honors)/8.0)),
    Clamp(std::tanh((terminals-2.0*other)/8.0)), orphans,
    Clamp(dragons/3.0), Clamp(winds/4.0)};
}
}  // namespace pdh
