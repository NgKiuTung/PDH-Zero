// Copyright 2026 PDH-Zero contributors. Project-local research use.
// NOTE: Heuristic arithmetic preserves rc3, not an improved or alternate policy.
#include "features.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace pdh {
namespace {
double Potential(Counts counts,const std::vector<Meld>& melds) {
  for (const auto& m:melds) counts[m.type]+=3;
  int pairs=0,triplets=static_cast<int>(melds.size()),quads=0,honors=0,terminals=0;
  std::array<int,3> suits{};
  for (int t=0;t<34;++t) {
    int n=counts[t]; pairs+=n/2; triplets+=n/3; quads+=(n==4);
    if(t<27) suits[t/9]+=n; else honors+=n;
    if(t>=27 || t%9==0 || t%9==8) terminals+=n;
  }
  int dominant=*std::max_element(suits.begin(),suits.end());
  int off=std::accumulate(suits.begin(),suits.end(),0)-dominant;
  int other=std::accumulate(counts.begin(),counts.end(),0)-terminals;
  // NOTE: declared_groups is intentionally also counted above, matching Python.
  return std::max({melds.empty() ? pairs*4.0+quads*2.0 : -1000.0,
    triplets*5.0+pairs*1.2, dominant*1.8+honors*1.4-off*3.0,
    dominant*2.0-off*4.0-honors*2.5, terminals*2.2-other*4.0});
}
}  // namespace

double ActionFeatures(const Context& c,const std::int32_t* state,
                      const std::int32_t* a,double* out) {
  const int sizes[7]={1,2,3,4,1,0,0};
  int kind=a[0], player=a[1], index=a[2];
  if(kind<0 || kind>6 || player!=c.player || index < -1 || index>3)
    throw std::invalid_argument("native.input.action");
  if((kind==4)!=(index>=0)) throw std::invalid_argument("native.input.action_index");
  int n=sizes[kind];
  std::fill(out,out+kActionFeatures,0.0); out[kind]=1.0;
  Counts cards=c.own; auto melds=c.melds[c.player];
  int revealed=0;
  std::bitset<136> used;
  for(int i=0;i<n;++i) {
    int id=a[3+i];
    if(id<0 || id>135 || !c.own_ids[id] || used[id])
      throw std::invalid_argument("native.input.action_tile");
    used.set(id); --cards[id/4]; out[7+id/4]+=.25;
    revealed+=c.revealed_ids[id];
  }
  out[41]=static_cast<double>(revealed)/std::max(1,n);
  if(index>=0) out[42+index]=1.0;
  if(kind==1 || kind==2) {
    if(state[4]<0) throw std::invalid_argument("native.input.pending_discard");
    melds.push_back({kind==1 ? 0 : 1,state[4]/4});
  } else if(kind==3) melds.push_back({2,a[3]/4});
  else if(kind==4) {
    if(index>=static_cast<int>(melds.size()) || melds[index].kind!=0)
      throw std::invalid_argument("native.input.added_kong");
    melds[index].kind=3;
  }
  auto after=Potentials(cards,melds);
  auto q=Qualify(melds);
  int cursor=46;
  for(double x:after) out[cursor++]=x;
  for(int i=0;i<8;++i) out[cursor++]=after[i]-c.before[i];
  for(double x:q) out[cursor++]=x;
  double qloss=0;
  for(int i=0;i<4;++i) {
    double loss=std::max(0.0,c.qualifications[i]-q[i]); out[cursor++]=loss; qloss+=loss;
  }
  int selected=n ? a[3]/4 : -1;
  double known=selected<0 ? 0 : c.known[selected];
  double risk=selected<0 ? 0 : c.risk[selected];
  std::sort(after.begin(),after.end(),std::greater<double>());
  double option=std::max(-1.0,std::min(1.0,(after[0]+.35*after[1]+.15*after[2])/1.5));
  out[cursor++]=known; out[cursor++]=risk;
  out[cursor++]=selected>=0 && (selected>=27 || selected%9==0 || selected%9==8);
  out[cursor++]=option;
  if(cursor!=kActionFeatures) throw std::logic_error("native.output.action_size");
  if(kind==5) return 1000000.0;
  double potential=Potential(cards,melds);
  if(kind==6) return potential;
  if(kind==0) return potential+(revealed ? .2 : 0.0)+.08*option+.08*known-.35*risk;
  if(kind==1 || kind==2) return potential+.5-.12*qloss;
  if(kind==3) return potential+.1;
  return potential+.2;
}
}  // namespace pdh
