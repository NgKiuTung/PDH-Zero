// Copyright 2026 PDH-Zero contributors. Project-local research use.
// NOTE: Pack validation is repeated in native code to avoid unsafe extension reads.
#include "features.hpp"
#include <algorithm>
#include <stdexcept>

namespace pdh {
namespace {
int Range(int value, int lo, int hi) {
  if (value<lo || value>hi) throw std::invalid_argument("native.input.range");
  return value;
}
int Card(int value) { return Range(value,0,135); }
void OneHot(double* out, int index, int size) {
  if (index>=0) out[Range(index,0,size-1)] = 1.0;
}
}  // namespace
Context Parse(const std::int32_t* p) {
  Context c;
  c.player=Range(p[0],0,3);
  Range(p[1],0,3); Range(p[2],0,2); Range(p[3],-1,3);
  Range(p[4],-1,135); Range(p[5],-1,135); Range(p[6],0,136);
  Range(p[7],0,16); Range(p[8],0,1); Range(p[9],0,1); Range(p[10],0,2);
  int own_n=Range(p[16],0,14);
  for (int i=0; i<own_n; ++i) {
    int id=Card(p[17+i]);
    if (c.own_ids[id]) throw std::invalid_argument("native.input.duplicate");
    c.own_ids.set(id); c.known_ids.set(id); c.own[id/4]++;
  }
  for (int seat=0; seat<4; ++seat) {
    const auto* s=p+31+303*seat;
    c.discard_size[seat]=Range(s[0],0,136);
    c.reveal_size[seat]=Range(s[137],0,136);
    for (int j=0; j<c.discard_size[seat]; ++j) {
      int id=Card(s[1+j]); c.discards[seat][id/4]++; c.known_ids.set(id);
    }
    for (int j=0; j<c.reveal_size[seat]; ++j) {
      int id=Card(s[138+j]); c.revealed[seat][id/4]++; c.known_ids.set(id);
      if (seat==c.player) c.revealed_ids.set(id);
    }
    int n=Range(s[274],0,4);
    for (int j=0; j<n; ++j) {
      const auto* m=s+275+7*j;
      int kind=Range(m[0],0,3), size=Range(m[2],3,4), type=Card(m[3])/4;
      if (size != (kind==0 ? 3 : 4)) throw std::invalid_argument("native.input.meld");
      for (int k=0; k<size; ++k) {
        int id=Card(m[3+k]);
        if (id/4!=type) throw std::invalid_argument("native.input.meld_type");
        c.exposed[seat][type]++; c.known_ids.set(id);
      }
      c.melds[seat].push_back({kind,type});
    }
  }
  for (int id=0; id<136; ++id) if (c.known_ids[id]) c.known[id/4] += .25;
  c.before=Potentials(c.own,c.melds[c.player]);
  c.qualifications=Qualify(c.melds[c.player]);
  for (int t=0; t<34; ++t) {
    int suit=t<27 ? t/9 : 3;
    double threat=0;
    for (int s=0; s<4; ++s) {
      if (s==c.player || c.melds[s].empty()) continue;
      int aligned=0;
      for (const auto& m:c.melds[s]) aligned += (m.type<27 ? m.type/9 : 3)==suit;
      double n=static_cast<double>(c.melds[s].size());
      threat=std::max(threat,(n/4.0)*(.5+.5*(aligned/n)));
    }
    c.risk[t]=std::max(0.0,std::min(1.0,(1.0-c.known[t])*threat));
  }
  return c;
}
void Observation(const std::int32_t* p,const Context& c,double* o) {
  std::fill(o,o+kObservation,0.0);
  int cursor=0;
  for (int n:c.own) o[cursor++]=n/4.0;
  Counts own_revealed{};
  for (int id=0;id<136;++id) if(c.own_ids[id] && c.revealed_ids[id]) own_revealed[id/4]++;
  for (int n:own_revealed) o[cursor++]=n/4.0;
  for (int rel=0;rel<4;++rel) {
    int s=(c.player+rel)%4;
    for (int n:c.discards[s]) o[cursor++]=n/4.0;
    for (int n:c.exposed[s]) o[cursor++]=n/4.0;
    for (int n:c.revealed[s]) o[cursor++]=n/4.0;
  }
  OneHot(o+cursor,(p[1]-c.player+4)%4,4); cursor+=4;
  OneHot(o+cursor,p[2],3); cursor+=3;
  OneHot(o+cursor,p[3]<0 ? -1 : (p[3]-c.player+4)%4,4); cursor+=4;
  OneHot(o+cursor,p[4]<0 ? -1 : p[4]/4,34); cursor+=34;
  OneHot(o+cursor,p[5]<0 ? -1 : p[5]/4,34); cursor+=34;
  o[cursor++]=p[6]/136.0; o[cursor++]=p[7]/16.0;
  o[cursor++]=p[8]; o[cursor++]=p[9];
  o[cursor++]=(p[10]==0); o[cursor++]=(p[10]==1);
  o[cursor++]=c.melds[c.player].size()/4.0;
  o[cursor++]=c.discard_size[c.player]/30.0;
  o[cursor++]=c.reveal_size[c.player]/14.0;
  for(int rel=0;rel<4;++rel) {
    int s=(c.player+rel)%4;
    o[cursor++]=c.melds[s].size()/4.0; o[cursor++]=c.discard_size[s]/30.0;
  }
  for (double x:c.before) o[cursor++]=x;
  for (double x:c.qualifications) o[cursor++]=x;
  for (double x:c.known) o[cursor++]=x;
  for(int rel=0;rel<4;++rel) {
    int s=(c.player+rel)%4;
    o[cursor++]=rel==0 ? 0 : std::min(1.0,.8*(c.melds[s].size()/4.0)
                           +.2*std::min(1.0,c.reveal_size[s]/6.0));
  }
  if(cursor!=kObservation) throw std::logic_error("native.output.observation_size");
}
}  // namespace pdh
