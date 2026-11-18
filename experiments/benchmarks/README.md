# Per-decision compute cost

Single-core cost per decision of PureTD, gnubg 1.08.003, Open Sage and
HedgeHog, each through its own production code path (move generation,
evaluation, move filter, cube logic), on a fixed pool of decision points
that every engine replays: checker decisions (board and dice, plus the cube
state when cubeful) and pre-roll cube decisions (board and cube state).

PureTD runs through its C API ([c_api/](../../c_api/README.md)), one row
per forward backend: `torch` (libtorch/ATen `addmm` on MKL; the paper's
main tables) and `simd` (hand-written AVX2/AVX-512 kernels; the paper's
optimized-kernel appendix).

## Protocol

| | Zen 3 | Zen 5 |
|---|---|---|
| CPU | AMD Ryzen 9 5950X | AMD Ryzen 5 9600X |
| pinned core | 4 (SMT sibling 20 idle) | 2 (SMT sibling 8 idle) |
| kernel | 6.8.0-149-generic | 6.8.0-147-generic |
| cpufreq | acpi-cpufreq, schedutil, boost on | amd-pstate-epp, EPP performance |
| results | [results/zen3_5950x_20261113/](results/zen3_5950x_20261113/) | [results/zen5_9600x_20261113/](results/zen5_9600x_20261113/) |

Both: Ubuntu 24.04, gcc 13.3, glibc 2.39, Python 3.12, torch 2.5.1+cpu
(MKL 2024.2), run 2026-11-13. One pinned core for every engine, box
otherwise idle (`suite.sh` checks before each group). `env.txt` in each
results directory has the full environment and binary md5s.
[results/zen3_5900x_20261113/](results/zen3_5900x_20261113/) is a
comparison host (Ryzen 9 5900X), not in the paper. Every raw file is
stamped with the commit of the code that produced it.

- Two groups (`suite.sh fast|slow`), each run as eight sweeps; a sweep runs
  every job of its group once, each benchmark in a fresh process, one at a
  time, with 90 s idle between sweeps (`SWEEP_GAP`).
- `fast`: every 0-ply row, gnubg 1-ply, Open Sage 1-ply cubeless and
  HedgeHog 0-ply.
- `slow`: the other 1-ply rows (including Open Sage 1-ply pruned) and
  gnubg 2-ply.
- `cap` (network sizes, a `fast` job): run as its own eight sweeps after
  `fast` and `slow`.
- Every row times the full pools: 8,000 cubeless checker items; 10,000
  checker and 6,000 cube items cubeful.
- Every timed region follows a 5 s busy-loop clock ramp (`BENCH_WARMUP_S`,
  set by `suite.sh`; the scripts' own default is 3 s) and 20 untimed
  warm-up items.
- A row is the mean of its eight sweeps with a 95% t-interval over them
  (t(7) = 2.365, `aggregate_sweeps.py`); each results directory has the
  aggregate of every job in `aggregate.txt`.
- PureTD 1-ply rows (cubeful and prob5, both backends) use the C API root
  move filter at 0.08 (`P1_ROOT_FILTER`; see Fairness).
- gnubg: threads-disabled build (`--disable-threads`), timed through an
  additive 185-line patch to its Python module (`gnubgmodule.c`, the
  `benchpool` function) that calls `FindnSaveBestMoves` /
  `GeneralCubeDecisionE` per item in C; no engine code is modified.
- Open Sage: fixed pool only, timed by the loops in
  [third_party_opponents/opensage/opensage_bench/sage_bench.cpp](../../third_party_opponents/opensage/opensage_bench/sage_bench.cpp), which
  call its decision functions directly (see Fairness).
- HedgeHog: fixed pool only, timed by the `hh_bench_*` loops in the shim
  (engine `e8847f4`).
- MKL shim: off (`fakeintel: off` in `env.txt`). The scripts preload
  `c_engine/libfakeintel.so` only when `FAKEINTEL` is set.

Pools are fixed, so every sweep of a row times the same items, on both
hosts. The t-interval is repeatability.
The per-sweep bootstrap CI on each raw line is the sampling uncertainty of
the fixed sample of items, which more sweeps do not reduce.

## Fairness

Every engine runs its production defaults:

- gnubg: its pruning nets, the Normal move filter and its evaluation cache.
- Open Sage: cubeless 1-ply uses its default move filter (the top 5 moves
  within 0.08 at 0-ply). Its full-width cubeful 1-ply (the C API) has no
  root filter: checker play scores every move and every reply. Wherever
  Open Sage prunes replies (cubeless 1-ply, 1-ply cube decisions, the
  pruned row), a roll with more than 16 legal replies has them ranked by
  PubEval and the top 15 scored with the net. Cubeless rows attach its
  bearoff database, as its analyzer does.
- Open Sage 1-ply pruned (cubeful checker only, engine `sage-filt`; row
  label `Open Sage 1-ply filtered` in the raw files):
  `MultiPlyStrategy::best_move_index_cubeful` (SageBench kind 3), the
  chooser its rollouts use, with the library default filter (the top 5
  moves within 0.08 by 0-ply cubeful equity, then 1-ply rescoring) and
  its rollout cube-efficiency setting (`cube_x`). The row to use is
  `slow/sagewarm`; the same-named row in the committed `slow/bench` cleared
  caches before every item. The shipped loop keeps caches, so a rerun of
  `slow` gives the warm row directly. On every checker item of the
  cubeful pool it picks the full-width move 97.98% of the time (97.50% of
  the 8,073 items with a choice), losing 0.215 mEq per decision
  ([sage_filt_check.cpp](../../third_party_opponents/opensage/opensage_bench/sage_filt_check.cpp),
  [results/sage_filt_check_20261117.txt](results/sage_filt_check_20261117.txt)).
- HedgeHog: full width; at 0-ply the successors of a decision are
  evaluated in one batch with `eval_ensemble_successors`, which HedgeHog
  ships but its own search does not call. Cubeful rows run with its Jacoby
  rule on, as the pool's games and the other engines.
- PureTD: full width at 0-ply; at 1-ply the benchmark's root filter, 0.08
  (`P1_ROOT_FILTER`; the C API's default is full width; rows say
  `root-filter 0.08`): every legal move is
  screened at 0-ply, those within 0.08 of the best (and game-ending moves)
  are expanded with every reply scored, and cube decisions are full 1-ply.
  On every item of both committed pools the filtered choice
  equals the full-width choice on 100% of the prob5 and 99.91% of the
  cubeful decisions (7 of 8,073; 0.010 mEq/decision), expanding 5.7 and
  4.7 of about 20.5 candidates ([rootf_validate.py](rootf_validate.py),
  [results/rootf_validate_20261011.txt](results/rootf_validate_20261011.txt)).

Caches:

- gnubg keeps its evaluation cache across items.
- Open Sage keeps its caches across items, except in its full-width
  cubeful 1-ply rows: each checker item starts on cleared caches and the
  caches are cleared after each cube decision, both outside the timer, as
  its C API clears them after every 1-ply cube decision. The pruned row
  clears them once before the timed items (untimed), then keeps them, as
  within one of its rollouts.
- PureTD has no evaluation cache.

Single thread everywhere (gnubg `--disable-threads`, MKL/OpenMP/ATen 1
thread, Open Sage `parallel_evaluate=False`). No engine batches across
decisions. Fixed-pool items are converted to each engine's native format
before timing; every engine's items are then timed inside native code
(`CLOCK_MONOTONIC`) around one full decision, with no Python call in the
timed region. The pools include dances and forced moves, timed as each
engine handles them. Open Sage (bgsage 2.0.20260907): cubeful checker and cube
decisions are the functions its C API (`bgsage_mobile_analyze`) runs,
without the C API's JSON reply and without the second 1-ply evaluation it
makes after a cube decision for that reply; cubeless decisions are its move
generation and `best_move_index` (the call behind its Python package's
`best_move_index`). On every item of both pools, at 0-ply and 1-ply,
cubeful picks and cube decisions equal the C API's and cubeless picks equal
`best_move_index`'s with the bearoff database attached
([sage_pick_check.py](sage_pick_check.py),
[results/sage_pick_check_20261113.txt](results/sage_pick_check_20261113.txt)).

| engine | timed per item: checker / cube |
|---|---|
| PureTD | `bgc_best_move` (move generation, encoding, forward, selection) / `bgc_offer_double` (both evaluations + decision) |
| gnubg | `FindnSaveBestMoves` + `RefreshMoveList` / `GeneralCubeDecisionE` + `FindCubeDecision` (`gnubg.benchpool`, [third_party_opponents/gnubg/gnubg_bench/](../../third_party_opponents/gnubg/gnubg_bench/)) |
| Open Sage | `possible_boards` + candidate scoring (cubeful), `possible_boards` + `best_move_index_cubeful` (pruned), `possible_boards` + `best_move_index` (cubeless) / `cube_decision_1ply`, `cube_decision_nply` ([third_party_opponents/opensage/opensage_bench/](../../third_party_opponents/opensage/opensage_bench/sage_bench.cpp)) |
| HedgeHog | `hh_best_move` (cubeful), `hh_best_move_cubeless` / `hh_double_action` (`hh_bench_*` loops in the shim) |

Build flags: PureTD and the HedgeHog shim `-O3 -march=native`, Open Sage
its wheel's `-O3 -ffast-math -funroll-loops -mavx2 -mfma`, gnubg its
configure flags with `--enable-simd=fma`.

## Inputs

- PureTD cubeful:
  [best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt](../../best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt)
  (md5 `6afda0f8`). The scripts load `best_models/cubeful_money_512_512_256_256.pt`,
  which holds a later net of the same shape; copy the history file there to
  time the committed net.
- PureTD cubeless (prob5):
  [best_models/history/cubeless_prob5_512_512_256_256_ac95a983.pt](../../best_models/history/cubeless_prob5_512_512_256_256_ac95a983.pt)
  (md5 `ac95a983`), likewise for `best_models/cubeless_prob5_512_512_256_256.pt`.
- Network-size job: the 16 checkpoints of the network-size table, found by name in
  `best_models/`, `best_models/experimental/` and
  `best_models/experimental/small_models/` and pinned by md5
  ([capacity_costbench.py](capacity_costbench.py) `MANIFEST`).
- PureTD C API: `VENV=<python with torch> bash c_api/build.sh torch simd`
  (`torch` links that interpreter's libtorch).
- gnubg 1.08.003 built `--disable-threads --enable-simd=fma` with the
  `gnubg.benchpool` patch: `bash third_party_opponents/gnubg/gnubg_bench/build.sh
  $HOME/gnubg-bench` (needs flex, bison, glib, python3 headers; without
  texinfo it prints `makeinfo: command not found`, which is harmless). A
  threaded build polls at 10 ms and reads ~99 moves/s at every ply; an
  unpatched build times the fixed pool per Python call and says so.
- [Open Sage](https://github.com/markbgsage/bgsage) `v2.0.20260907` exactly
  (bgsage 1.x gives different rows, and releases before 1.5 are
  AGPL-3.0-or-later; commit `4b10c69318c3b80e2a7618232540b10a76217d20`),
  production model `stage11`; the fixed-pool loops built against its engine
  library with the wheel's flags (`opensage_bench/build.sh`, needs cmake), and its C API
  (`opensage_mobile_build.sh`), which the agreement checks and
  `tests/test_sage_bench.py` need.
- HedgeHog engine `e8847f4` (optional): shim (with the `hh_bench_*` loops)
  and `.ogxf` models per
  [third_party_opponents/hedgehog/README.md](../../third_party_opponents/hedgehog/README.md).
- Pools: [fixedpool_positions.json](fixedpool_positions.json), 8,000
  cubeless checker points (board and dice) from seeded gnubg-nn (GNU Backgammon's nets) 0-ply self-play;
  [fixedpool_cubeful_positions.json](fixedpool_cubeful_positions.json), the
  first 10,000 checker points (board, dice, cube state) and 6,000 cube points
  of seeded PureTD cubeful self-play (seed 1, Jacoby on), generated by an
  earlier 4L cubeful checkpoint (md5 `77a3205c`). `fixedpool_bench.py --mode
  cubeful --regen-pool` overwrites it with a pool from the current cubeful
  model.

Setup, from the repo root. The committed rows used torch 2.5.1+cpu;
`requirements.txt` pins 2.13.0, which also works.

```bash
python3 -m venv venv
venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install -r requirements.txt bgsage==2.0.20260907   # includes pytest
VENV=$PWD/venv/bin/python bash c_api/build.sh torch simd
bash third_party_opponents/gnubg/gnubg_bench/build.sh $HOME/gnubg-bench
bash third_party_opponents/opensage/opensage_bench/build.sh      # libsage_bench.so
bash third_party_opponents/opensage/opensage_mobile_build.sh     # libbgsage_mobile.so
venv/bin/python -m pytest -q tests/test_sage_bench.py
```

## Commands

The whole suite, from the repo root. `suite.sh` runs one group as eight
sweeps, one job at a time, and resumes a group from its finished sweep
files (use a fresh `<run>` for a new run). `VENV` is a python with torch and
bgsage. `ROWS` selects rows: `gnubg`, `puretd`, `sage` (the `bench.sh`
rows), `hh` (HedgeHog) and `cap` (network sizes); default `all`, which
runs `cap` inside the `fast` sweeps. Without HedgeHog, drop `hh`.

```bash
export BENCH_CORE=<core> VENV=$PWD/venv/bin/python GNUBG=$HOME/gnubg-bench/bin/gnubg
ROWS=gnubg,puretd,sage,hh bash experiments/benchmarks/suite.sh ~/bench_results/<run> fast   # bench.sh fast tier, HedgeHog 0-ply
ROWS=gnubg,puretd,sage bash experiments/benchmarks/suite.sh ~/bench_results/<run> slow      # bench.sh slow tier
ROWS=cap bash experiments/benchmarks/suite.sh ~/bench_results/<run> fast                    # network sizes, own sweeps
python experiments/benchmarks/aggregate_sweeps.py ~/bench_results/<run>/fast/bench_sweep*.txt   # one job at a time
```

The committed run is these three commands, in this order; in its raw
files, `rows=,all,` in the `bench.sh` headers means `gnubg,puretd,sage`.
The Open Sage pruned row (`slow/sagewarm`) is the `fixedpool_bench.py
--mode cubeful --engines sage-filt --plies 1` command run alone afterwards,
eight sweeps with the same gaps. One sweep takes about 5.5 min (`fast`),
46 min (`slow`) and 6 min (`cap`) on the 5950X, and 5, 33 and 5.5 min on
the 9600X; eight sweeps of `fast` and `slow`, gaps included, take about
7.3 h and 5.3 h.

Smoke tests: `SMOKE=1` (`suite.sh` only) times 30 items per row, one sweep,
no idle check; `aggregate_sweeps.py` needs at least two sweeps, so add
`NSWEEPS=2 SWEEP_GAP=0` to test the aggregation. `bench.sh` ignores
`SMOKE`; its smoke setting is `FP_CAP=30`. `bench.sh` alone runs both tiers
(`TIER=all`) as one sweep (`SWEEPS=1`).

The jobs of one sweep (`bench.sh`, then the extra `suite.sh` jobs). The
gnubg script runs inside the gnubg binary's own interpreter
(`GNUBG_FP_CAP=30` for a smoke run). PureTD rows are labelled
`prob5c[<b>]` (cubeless) and `PureTD-C[<b>]` (cubeful) for backend `<b>`
(`CAPI_BACKENDS`, default `torch simd`).

```bash
export BENCH_CORE=<core>
G=$HOME/gnubg-bench/bin/gnubg

# gnubg: cubeless and cubeful checker, cube; 0/1-ply (fast), 2-ply (slow)
GNUBG_FP_MODE=both GNUBG_FP_PLIES=0,1 $G -t -q -p third_party_opponents/gnubg/fixedpool_bench_gnubg.py
GNUBG_FP_MODE=both GNUBG_FP_PLIES=2 $G -t -q -p third_party_opponents/gnubg/fixedpool_bench_gnubg.py

# fast: full pools
python experiments/benchmarks/fixedpool_bench.py --mode cubeless --configs prob5c-0ply,sage-0ply,sage-1ply --backends torch,simd
python experiments/benchmarks/fixedpool_bench.py --mode cubeful --engines puretdc,sage0 --backends torch,simd
# slow (sage-filt: Open Sage 1-ply pruned)
python experiments/benchmarks/fixedpool_bench.py --mode cubeless --configs prob5c-1ply --backends torch,simd --root-filter 0.08
python experiments/benchmarks/fixedpool_bench.py --mode cubeful --engines puretdc,sage,sage-filt --plies 1 --backends torch,simd --root-filter 0.08

# HedgeHog 0-ply (fast; suite.sh job hh0)
python experiments/benchmarks/fixedpool_bench.py --mode cubeless --configs prob5c-0ply,hh-fox-0ply,hh-aureus-0ply --backends torch,simd
python experiments/benchmarks/fixedpool_bench.py --mode cubeful --engines puretdc,hh-fox0,hh-aureus0 --backends torch,simd

# network sizes (fast; job cap): 0-ply checker (and cube, cubeful nets) of each checkpoint
python experiments/benchmarks/capacity_costbench.py --models all --backends torch,simd
```

`capacity_sweeps.sh` runs the network-size job alone as eight sweeps
(`OUTDIR`, default `~/bench_results/capacity`).

Agreement checks (each prints one line per row: items, agree, over the C
API's 1 s budget, differ). The C API's budget is wall time, so each worker
gets its own core: `N` at most the physical core count minus one.

```bash
python experiments/benchmarks/sage_pick_check.py --workers N --cores 1-N         # loops vs C API / best_move_index
python experiments/benchmarks/sage_capi_vs_analyzer.py --workers N --cores 1-N   # C API vs BgBotAnalyzer
```

The pruned-vs-full-width check is a C++ tool built next to
`libsage_bench.so`; its header gives the item format and arguments:

```bash
cmake --build third_party_opponents/opensage/opensage_bench/build/cmake --target sage_filt_check
python -c "import sys; sys.path.insert(0, 'experiments/benchmarks'); import fixedpool_bench as fb
for b, d, c in fb.sage_items(fb.load_cubeful_pool()[0]): print(*b, *d, *c)" > items.txt   # md5 e3c9b107
MD=$(python -c "import bgsage, os; print(os.path.join(os.path.dirname(bgsage.__file__), '_assets', 'models'))")
BO=$(python -c "import bgsage.weights as w; print(w.bearoff_db_path())")
third_party_opponents/opensage/opensage_bench/build/cmake/sage_filt_check items.txt "$MD" "$BO" 0 10000 > sfc.txt
```

`START COUNT` (here `0 10000`) is an item range, to split the pool across
processes; the columns `idx` to `loss` match the per-item lines of
[results/sage_filt_check_20261117.txt](results/sage_filt_check_20261117.txt).

## Results

Decisions/s, mean of eight sweeps [95% t-interval]; in parentheses the
per-decision cost relative to PureTD `torch` 0-ply on the same benchmark
and host (its rate over the row's). Zen 3 = Ryzen 9 5950X, Zen 5 = Ryzen 5
9600X. Raw sweeps: `results/<host>/{fast,slow}/<job>_sweep<i>.txt`; aggregates:
`results/<host>/aggregate.txt`. PureTD 1-ply = root filter 0.08.

### Cubeless money, fixed pool (checker)

| engine | depth | Zen 5 | Zen 3 |
|---|---|---|---|
| PureTD torch | 0-ply | 5,094 [5,060, 5,127] (1.00×) | 4,548 [4,536, 4,560] (1.00×) |
| PureTD simd | 0-ply | 36,835 [36,670, 37,000] (0.14×) | 15,496 [15,459, 15,533] (0.29×) |
| gnubg | 0-ply | 42,728 [42,573, 42,884] (0.12×) | 34,619 [34,476, 34,762] (0.13×) |
| Open Sage | 0-ply | 19,861 [19,796, 19,925] (0.26×) | 16,933 [16,849, 17,016] (0.27×) |
| PureTD torch | 1-ply | 80.0 [80.0, 80.0] (64×) | 77.4 [77.3, 77.5] (59×) |
| PureTD simd | 1-ply | 557 [556, 558] (9.1×) | 222 [221, 223] (20×) |
| gnubg | 1-ply | 1,338 [1,335, 1,341] (3.8×) | 1,004 [1,001, 1,008] (4.5×) |
| Open Sage | 1-ply | 1,087 [1,083, 1,091] (4.7×) | 824 [816, 832] (5.5×) |
| gnubg | 2-ply | 72.4 [72.3, 72.5] (70×) | 53.5 [53.4, 53.6] (85×) |

### Cubeful money, fixed pool

Open Sage pruned: checker only, from `slow/sagewarm`.

| engine | depth | checker Zen 5 | checker Zen 3 | cube Zen 5 | cube Zen 3 |
|---|---|---|---|---|---|
| PureTD torch | 0-ply | 5,638 [5,633, 5,643] (1.00×) | 5,055 [5,045, 5,065] (1.00×) | 24,513 [24,255, 24,771] (1.00×) | 12,957 [12,874, 13,041] (1.00×) |
| PureTD simd | 0-ply | 43,442 [43,148, 43,736] (0.13×) | 18,006 [17,955, 18,057] (0.28×) | 264,045 [262,902, 265,188] (0.09×) | 111,245 [110,780, 111,710] (0.12×) |
| gnubg | 0-ply | 41,607 [41,344, 41,870] (0.14×) | 34,171 [33,896, 34,447] (0.15×) | 714,856 [712,357, 717,355] (0.03×) | 604,875 [597,551, 612,200] (0.02×) |
| Open Sage | 0-ply | 20,265 [20,186, 20,344] (0.28×) | 16,727 [16,643, 16,810] (0.30×) | 349,126 [347,644, 350,609] (0.07×) | 276,195 [273,823, 278,567] (0.05×) |
| PureTD torch | 1-ply | 63.5 [63.4, 63.5] (89×) | 62.0 [61.7, 62.2] (82×) | 136 [136, 136] (180×) | 134 [134, 134] (97×) |
| PureTD simd | 1-ply | 475 [474, 476] (11.9×) | 185 [185, 186] (27×) | 1,052 [1,050, 1,055] (23×) | 420 [418, 421] (31×) |
| gnubg | 1-ply | 1,436 [1,428, 1,444] (3.9×) | 1,094 [1,091, 1,097] (4.6×) | 4,107 [4,083, 4,130] (6.0×) | 3,173 [3,165, 3,181] (4.1×) |
| Open Sage | 1-ply | 52.0 [51.5, 52.4] (108×) | 40.2 [39.7, 40.7] (126×) | 1,873 [1,863, 1,883] (13.1×) | 1,465 [1,451, 1,479] (8.8×) |
| Open Sage pruned | 1-ply | 1,197 [1,184, 1,209] (4.7×) | 910 [904, 915] (5.6×) | – | – |
| gnubg | 2-ply | 75.3 [75.1, 75.4] (75×) | 56.5 [56.4, 56.6] (89×) | 248 [248, 249] (99×) | 188 [188, 188] (69×) |

### Network sizes, fixed pool

The one- to six-layer cubeful and prob5 nets of
[../capacity/README.md](../capacity/README.md), timed by
[capacity_costbench.py](capacity_costbench.py) (job `cap`, its own eight
sweeps; standalone: [capacity_sweeps.sh](capacity_sweeps.sh)). 0-ply
checker decisions/s; factors against the production net of the same
family, backend and host. Sweeps: `results/<host>_20261113/fast/capacity_sweep*.txt`;
cube rows in `aggregate.txt`. Table:
`python experiments/capacity/network_size_table.py --bench`.

| model | hidden | params | Zen 5 torch | Zen 3 torch | Zen 5 simd | Zen 3 simd |
|---|---|---|---|---|---|---|
| cubeful 1-layer | [512] | 103,425 | 28,284 [28,074, 28,494] (0.20×) | 24,440 [24,347, 24,534] (0.21×) | 189,388 [188,808, 189,968] (0.23×) | 112,868 [112,431, 113,305] (0.16×) |
| cubeful 2-layer | [512,512] | 366,081 | 8,581 [8,570, 8,591] (0.66×) | 7,561 [7,546, 7,576] (0.66×) | 64,810 [64,573, 65,047] (0.67×) | 27,626 [27,551, 27,701] (0.65×) |
| cubeful 3-layer | [512,512,256] | 497,153 | 6,386 [6,380, 6,392] (0.88×) | 5,660 [5,641, 5,678] (0.89×) | 46,574 [46,428, 46,719] (0.94×) | 19,482 [19,448, 19,516] (0.92×) |
| cubeful 4-layer (reference) | [512,512,256,256] | 562,945 | 5,646 [5,641, 5,652] (1.00×) | 5,018 [5,008, 5,028] (1.00×) | 43,550 [43,452, 43,647] (1.00×) | 17,931 [17,901, 17,960] (1.00×) |
| cubeful 5-layer | [512,512,256,256,256] | 628,737 | 5,062 [5,057, 5,068] (1.12×) | 4,509 [4,501, 4,517] (1.11×) | 37,750 [37,632, 37,867] (1.15×) | 15,485 [15,462, 15,508] (1.16×) |
| cubeful 5-layer wide | [768,512,256,256,256] | 811,265 | 3,972 [3,967, 3,977] (1.42×) | 3,538 [3,530, 3,546] (1.42×) | 27,665 [27,509, 27,821] (1.57×) | 11,032 [11,010, 11,054] (1.63×) |
| cubeful 6-layer wide | [768,512,256,256,256,256] | 877,057 | 3,673 [3,668, 3,678] (1.54×) | 3,274 [3,268, 3,280] (1.53×) | 24,309 [24,178, 24,440] (1.79×) | 9,678 [9,650, 9,706] (1.85×) |
| cubeful 5-layer 768x768 | [768,768,256,256,256] | 1,073,665 | 3,080 [3,078, 3,083] (1.83×) | 2,739 [2,733, 2,744] (1.83×) | 19,768 [19,604, 19,931] (2.20×) | 7,765 [7,748, 7,782] (2.31×) |
| prob5 1-layer | [512] | 103,429 | 22,737 [22,700, 22,774] (0.23×) | 18,620 [18,561, 18,678] (0.24×) | 162,613 [162,310, 162,916] (0.23×) | 96,122 [95,595, 96,649] (0.16×) |
| prob5 2-layer | [512,512] | 366,085 | 7,559 [7,550, 7,568] (0.68×) | 6,514 [6,504, 6,524] (0.69×) | 52,973 [52,782, 53,163] (0.70×) | 22,746 [22,693, 22,799] (0.67×) |
| prob5 3-layer | [512,512,256] | 496,133 | 5,786 [5,778, 5,794] (0.89×) | 5,072 [5,063, 5,080] (0.88×) | 41,728 [41,504, 41,951] (0.88×) | 17,554 [17,464, 17,643] (0.87×) |
| prob5 4-layer (reference) | [512,512,256,256] | 561,925 | 5,125 [5,119, 5,130] (1.00×) | 4,484 [4,463, 4,504] (1.00×) | 36,848 [36,704, 36,993] (1.00×) | 15,328 [15,213, 15,444] (1.00×) |
| prob5 5-layer | [512,512,256,256,256] | 627,717 | 4,599 [4,595, 4,604] (1.11×) | 4,048 [4,043, 4,054] (1.11×) | 33,490 [33,350, 33,630] (1.10×) | 13,906 [13,844, 13,967] (1.10×) |
| prob5 5-layer wide | [768,512,256,256,256] | 809,221 | 3,623 [3,620, 3,627] (1.41×) | 3,197 [3,187, 3,207] (1.40×) | 26,829 [26,776, 26,881] (1.37×) | 10,796 [10,678, 10,914] (1.42×) |
| prob5 6-layer wide | [768,512,256,256,256,256] | 875,013 | 3,357 [3,354, 3,360] (1.53×) | 2,964 [2,958, 2,970] (1.51×) | 25,394 [25,320, 25,467] (1.45×) | 10,213 [10,150, 10,275] (1.50×) |
| prob5 5-layer 768x768 | [768,768,256,256,256] | 1,071,621 | 2,814 [2,811, 2,817] (1.82×) | 2,483 [2,476, 2,491] (1.81×) | 20,315 [20,258, 20,371] (1.81×) | 8,060 [7,998, 8,121] (1.90×) |

### HedgeHog, fixed pool

Engine `e8847f4`, 0-ply, same hosts and runs (`fast/hh0`). Factors
against the PureTD `torch` 0-ply rows of the tables above (the HedgeHog
0-ply job re-times them within 1.6%).

Cubeless checker:

| model | Zen 5 | Zen 3 |
|---|---|---|
| Fox | 23,818 [23,665, 23,971] (0.21×) | 21,803 [21,613, 21,992] (0.21×) |
| Aureus | 14,580 [13,988, 15,171] (0.35×) | 7,717 [7,448, 7,985] (0.59×) |

Cubeful checker:

| model | Zen 5 | Zen 3 |
|---|---|---|
| Fox | 25,260 [25,135, 25,386] (0.22×) | 23,012 [22,749, 23,275] (0.22×) |
| Aureus | 14,676 [14,319, 15,034] (0.38×) | 8,214 [7,935, 8,492] (0.62×) |

Cube:

| model | Zen 5 | Zen 3 |
|---|---|---|
| Fox | 322,780 [316,268, 329,293] (0.08×) | 295,332 [292,289, 298,375] (0.04×) |
| Aureus | 207,682 [198,646, 216,719] (0.12×) | 122,675 [118,702, 126,648] (0.11×) |
