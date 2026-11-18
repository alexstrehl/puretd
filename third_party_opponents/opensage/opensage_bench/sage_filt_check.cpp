// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Alex Strehl
//
// Adapted from Open Sage (bgsage v2.0.20260907) cpp/src/mobile.cpp,
// Copyright (C) 2026 Mark Higgins; this file is under the Mozilla Public
// License 2.0 (https://mozilla.org/MPL/2.0/).
//
// sage_filt_check.cpp -- Open Sage cubeful 1-ply on the fixed cubeful pool:
// the full-width search (every candidate scored, as sage_bench.cpp kind 0)
// against its pruned chooser MultiPlyStrategy::best_move_index_cubeful
// (0-ply cubeful screen 5 / 0.08, then 1-ply cubeful rescore; sage_bench.cpp
// kind 3). Caches are cleared before each search.
//
// Build: cmake target sage_filt_check (CMakeLists.txt here), e.g. after
// build.sh: cmake --build build/cmake --target sage_filt_check
//
// Usage: sage_filt_check ITEMS MODELDIR BEAROFF START COUNT
//   ITEMS     one fixedpool_bench.sage_items() tuple per line, flattened:
//             26 board ints (mover's frame), 2 dice, cube value, cube owner
//             (0 centred, 1 mover, 2 opponent), separated by whitespace
//   MODELDIR  bgsage's _assets/models; loads the production model stage11
//             (backgame_pair_phased, 24 nets, as bgsage.weights.
//             default_weights() and sage_bench.cpp's sb_create use it)
//   BEAROFF   bgsage.weights.bearoff_db_path()
//   START COUNT  item range (split the pool across processes)
// Output, one line per item: idx n_cands full_idx filt_idx loss t_full t_filt
//   loss = full-width equity of the full-width pick minus that of the
//   pruned pick (0 when they agree); times in seconds.

#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"
#include "bgbot/moves.h"
#include "bgbot/multipy.h"
#include "bgbot/neural_net.h"
#include "bgbot/pubeval.h"
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
using namespace bgbot;
static double now_s(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static double cl_eq(double w,double gw,double bw,double gl,double bl){return 2*w-1+gw-gl+bw-bl;}
int main(int argc,char**argv){
  if(argc!=6){fprintf(stderr,"usage: sage_filt_check ITEMS MODELDIR BEAROFF START COUNT\n");return 2;}
  std::string items=argv[1], md=argv[2], bo=argv[3]; int start=atoi(argv[4]), cnt=atoi(argv[5]);
  const char* names[]={"sl_s9_purerace","sl_s9_race_race","sl_s9_race_att","sl_s9_race_prim","sl_s9_race_anch","sl_s9_att_race","sl_s9_att_att","sl_s9_att_prim","sl_s9_att_anch","sl_s9_prim_race","sl_s9_prim_att","sl_s9_prim_anch","sl_s9_prim_anch","sl_s9_anch_race","sl_s9_anch_att","sl_s9_prim_anch","sl_s9_prim_anch",
    "sl_s11_bg_deep","sl_s11_bg_middle","sl_s11_bg_double","sl_s11_bg_p3","sl_s11_bg_containment","sl_s11_bg_snake","sl_s11_bg_massive"};
  const int n_nets=sizeof(names)/sizeof(names[0]);
  std::vector<std::string> p; std::vector<int> h;
  for(int i=0;i<n_nets;++i){p.push_back(md+"/"+names[i]+".weights.best"); h.push_back(i==0?100:400);}
  BearoffDB db; if(!db.load(bo)){fprintf(stderr,"bearoff\n");return 1;}
  auto mk=[&](){return std::shared_ptr<Strategy>(std::make_shared<BackgameAwarePairStrategy>(p,h,true));};
  auto strategy=std::make_shared<BearoffStrategy>(mk(),&db);
  const MoveFilter filter{5,0.08f};
  auto pubeval=std::make_shared<PubEval>();
  auto cl1=std::make_shared<MultiPlyStrategy>(mk(),2,filter,false,false,0);
  cl1->set_move_prefilter(pubeval); cl1->set_bearoff_db(&db);
  auto mpf=std::make_shared<MultiPlyStrategy>(strategy,2,filter,false,false,0);
  mpf->set_move_prefilter(pubeval); mpf->set_bearoff_db(&db);
  std::ifstream f(items); std::string line; int ln=0; std::vector<Board> c;
  while(std::getline(f,line)){
    if(ln++<start) continue; if(ln>start+cnt) break;
    std::istringstream ss(line); Board b; for(int i=0;i<26;++i) ss>>b[i];
    int d1,d2,cv,co; ss>>d1>>d2>>cv>>co;
    CubeInfo ci{cv,static_cast<CubeOwner>(co),{0,0,false},-1.0f,true,false};
    // full-width
    cl1->clear_cache(); mpf->clear_cache();
    double t0=now_s();
    possible_boards(b,d1,d2,c);
    int best=0; double be=0,bc=0; std::vector<double> eqs(c.size());
    for(int i=0;i<(int)c.size();++i){
      auto v=cubeful_probs_and_equity_nply(flip(c[i]),flip_cube_perspective(ci),*strategy,2,filter,1,nullptr,&b);
      double eq=-v.equity, cle=cl_eq(1.0-v.probs[0],v.probs[3],v.probs[4],v.probs[1],v.probs[2]);
      eqs[i]=eq;
      if(i==0||eq>be||(eq==be&&(cle>bc||(cle==bc&&c[i]<c[best])))){best=i;be=eq;bc=cle;}
    }
    double tf=now_s()-t0;
    // pruned (rollout-style cube_x)
    cl1->clear_cache(); mpf->clear_cache();
    t0=now_s();
    std::vector<Board> c2; possible_boards(b,d1,d2,c2);
    std::array<float,NUM_OUTPUTS> dummy{}; auto [pp,op]=pip_counts(b);
    float cx=cube_efficiency(dummy,is_race(b),pp,op);
    int k=mpf->best_move_index_cubeful(c2,b,ci,cx);
    double tq=now_s()-t0;
    int kk=-1; for(int i=0;i<(int)c.size();++i) if(c[i]==c2[k]){kk=i;break;}
    printf("%d %zu %d %d %.6f %.6e %.6e\n",ln-1,c.size(),best,kk,kk>=0?eqs[best]-eqs[kk]:-1.0,tf,tq);
    fflush(stdout);
  }
}
