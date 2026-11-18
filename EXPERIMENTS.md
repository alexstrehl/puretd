# PureTD paper: tables to code

The paper's tables mapped to code, commands and committed results. Links are repo-relative; line anchors are pinned to this release. Tables are numbered in the order of their `table` environments, appendix included.

## Prerequisites

In build order.

1. Python 3.10–3.13 with torch: [experiments/head_to_head/README.md#L11-L35](experiments/head_to_head/README.md?plain=1#L11-L35), which also builds the C engine (`bash c_engine/build_unix.sh`).
2. C API (Tables 4, 5, 7, 9, 11, 12, 13, 21, 26, 27): `VENV=/path/to/venv/bin/python bash c_api/build.sh torch simd`, [c_api/README.md](c_api/README.md).
3. gnubg 1.08.003: the packaged binary (Ubuntu 24.04 ships 1.07.001) or the benchmark build below, as `GNUBG_CMD`, for head-to-head ([experiments/head_to_head/README.md#L18-L25](experiments/head_to_head/README.md?plain=1#L18-L25)); for the benchmarks a `--disable-threads` build with the `gnubg.benchpool` patch, `bash third_party_opponents/gnubg/gnubg_bench/build.sh $HOME/gnubg-bench` (needs flex, bison, glib, python3 headers).
4. Open Sage: `pip install bgsage==2.0.20260907`, exactly this version (bgsage 1.x gives different rows, and releases before 1.5 are AGPL-3.0-or-later); native C API `bash third_party_opponents/opensage/opensage_mobile_build.sh`, benchmark loops `bash third_party_opponents/opensage/opensage_bench/build.sh` ([experiments/head_to_head/OPENSAGE.md](experiments/head_to_head/OPENSAGE.md)).
5. HedgeHog (Tables 26, 27): engine, shim and models per [third_party_opponents/hedgehog/README.md](third_party_opponents/hedgehog/README.md). Without it, run the benchmark suite with `ROWS=gnubg,puretd,sage,cap`.

Host-specific paths and defaults in the commands (the repo `.venv`, `$HOME/gnubg-bench`, `--device cuda`, `--workers 16` to `64`) are placeholders: set `--workers` to the local core count; `--device cpu` works.

`play_models.py` and `run.py` play checkpoint and `janowski:` sides through the C API by default (built on demand). The committed rows of Tables 2 and 3 were played on the Python path; `--combine` refuses to pool across paths, so add `--python-api` to extend them.

Abbreviations: mEq, thousandths of a point (equity) per game; PR, XG++ Performance Rating, 500 × equity lost per decision (lower is better); DMP, double match point, the last game of a match, where only winning counts.

## Table 2: Cubeless-money strength

- Code: [experiments/head_to_head/run.py](experiments/head_to_head/run.py)
- Command: [experiments/head_to_head/README.md#L72-L90](experiments/head_to_head/README.md?plain=1#L72-L90)
- Results: [experiments/head_to_head/results/h2h_cubeless_summary.csv](experiments/head_to_head/results/h2h_cubeless_summary.csv), [experiments/head_to_head/results/prob5_vs_opensage0_cbl_10M.txt](experiments/head_to_head/results/prob5_vs_opensage0_cbl_10M.txt), [experiments/head_to_head/results/prob5_vs_opensage1_cbl_10M.txt](experiments/head_to_head/results/prob5_vs_opensage1_cbl_10M.txt) (pooled Open Sage outputs)

## Table 3: Cubeful-money strength

XG++ PR column: no exports committed.

- Code: [experiments/head_to_head/run.py](experiments/head_to_head/run.py), [tools/analyze_xg_results.py](tools/analyze_xg_results.py)
- Command: [experiments/head_to_head/README.md#L92-L119](experiments/head_to_head/README.md?plain=1#L92-L119)
- Results: [experiments/head_to_head/results/h2h_cubeful_summary.csv](experiments/head_to_head/results/h2h_cubeful_summary.csv), XG++ per-game export not committed

## Table 4: Cubeless-money cost versus strength

Zen 3 = Ryzen 9 5950X, Zen 5 = Ryzen 5 9600X, run 2026-11-13. Rates: sections `fast/bench` (0-ply, gnubg 1-ply, Open Sage 1-ply cubeless) and `slow/bench` (the other 1-ply rows, gnubg 2-ply) of each `aggregate.txt`.

- Code: [experiments/benchmarks/fixedpool_bench.py](experiments/benchmarks/fixedpool_bench.py), [third_party_opponents/gnubg/fixedpool_bench_gnubg.py](third_party_opponents/gnubg/fixedpool_bench_gnubg.py), [third_party_opponents/opensage/opensage_bench/sage_bench.cpp](third_party_opponents/opensage/opensage_bench/sage_bench.cpp), [experiments/head_to_head/run.py](experiments/head_to_head/run.py)
- Command: [experiments/benchmarks/README.md#L211-L217](experiments/benchmarks/README.md?plain=1#L211-L217) (suite), [experiments/benchmarks/README.md#L240-L261](experiments/benchmarks/README.md?plain=1#L240-L261) (per job), [experiments/head_to_head/README.md#L72-L90](experiments/head_to_head/README.md?plain=1#L72-L90) (equity cells)
- Results: [experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt](experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt), [experiments/head_to_head/results/h2h_cubeless_summary.csv](experiments/head_to_head/results/h2h_cubeless_summary.csv) (equity cells)

## Table 5: Cubeful-money cost versus strength

Sections as Table 4. Open Sage, pruned: section `slow/sagewarm` (engine `sage-filt`, caches kept, eight sweeps of its own), not the same-named row in `slow/bench`, which cleared caches before every item.

- Code: [experiments/benchmarks/fixedpool_bench.py](experiments/benchmarks/fixedpool_bench.py), [third_party_opponents/gnubg/fixedpool_bench_gnubg.py](third_party_opponents/gnubg/fixedpool_bench_gnubg.py), [third_party_opponents/opensage/opensage_bench/sage_bench.cpp](third_party_opponents/opensage/opensage_bench/sage_bench.cpp), [experiments/head_to_head/run.py](experiments/head_to_head/run.py)
- Command: [experiments/benchmarks/README.md#L211-L217](experiments/benchmarks/README.md?plain=1#L211-L217) (suite), [experiments/benchmarks/README.md#L240-L261](experiments/benchmarks/README.md?plain=1#L240-L261) (per job), [experiments/head_to_head/README.md#L92-L119](experiments/head_to_head/README.md?plain=1#L92-L119) (equity cells)
- Results: [experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt](experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt), [experiments/head_to_head/results/h2h_cubeful_summary.csv](experiments/head_to_head/results/h2h_cubeful_summary.csv) (equity cells), [experiments/benchmarks/results/sage_pick_check_20261113.txt](experiments/benchmarks/results/sage_pick_check_20261113.txt) (loops and C API agreement), [experiments/benchmarks/results/sage_filt_check_20261117.txt](experiments/benchmarks/results/sage_filt_check_20261117.txt) (pruned vs full width: 97.98% of 10,000 picks agree, 0.215 mEq/decision)
- Agreement checks: [experiments/benchmarks/sage_pick_check.py](experiments/benchmarks/sage_pick_check.py), [experiments/benchmarks/sage_capi_vs_analyzer.py](experiments/benchmarks/sage_capi_vs_analyzer.py) (Open Sage loops vs its C API and `best_move_index` on every item of both pools; C API vs BgBotAnalyzer on every cubeful item; 0- and 1-ply), [third_party_opponents/opensage/opensage_bench/sage_filt_check.cpp](third_party_opponents/opensage/opensage_bench/sage_filt_check.cpp) (pruned vs full-width 1-ply picks); commands [experiments/benchmarks/README.md#L270-L285](experiments/benchmarks/README.md?plain=1#L270-L285)

## Table 7: Learned versus heuristic cube handling

Both sides play 0-ply. Changes from the paper ([experiments/janowski/README.md#L25-L31](experiments/janowski/README.md?plain=1#L25-L31)): no bearoff tables on either side; Prob5Jan uses a fixed cube-life index x = 0.76 instead of the paper's adaptive index; results are luck-adjusted. Head-to-head: +15.3 [+14.8, +15.7] mEq/game, 1.2M games (paper +14.6). XG++ PR: the paper's values, not re-measured; no exports committed.

- Code: [janowski.py](janowski.py), [janowski_agent.py](janowski_agent.py), [c_api/puretd_janowski.c](c_api/puretd_janowski.c), [experiments/head_to_head/variance.py](experiments/head_to_head/variance.py), [tools/analyze_xg_results.py](tools/analyze_xg_results.py)
- Command: [experiments/janowski/README.md#L70-L87](experiments/janowski/README.md?plain=1#L70-L87) (head-to-head), [experiments/janowski/README.md#L96-L101](experiments/janowski/README.md?plain=1#L96-L101) (XG++ PR)
- Results: [experiments/janowski/results/table7_vr_1.2M.json](experiments/janowski/results/table7_vr_1.2M.json), [experiments/janowski/results/janowski_h2h.csv](experiments/janowski/results/janowski_h2h.csv), [experiments/janowski/README.md#L56-L60](experiments/janowski/README.md?plain=1#L56-L60) (head-to-head table), XG++ per-game exports not committed; PR values in [experiments/janowski/README.md#L105-L108](experiments/janowski/README.md?plain=1#L105-L108) (XG++ PR table)

## Table 9 and Figure 3: Strength across network sizes

Strength of the one- to six-layer cubeful and prob5 nets head-to-head against the production net of the family, and their 0-ply cost per decision on the fixed pools (job `cap`, section `fast/capacity`, eight sweeps of its own); the figure plots strength against Zen 5 `torch` checker decisions/s.

- Code: [experiments/benchmarks/capacity_costbench.py](experiments/benchmarks/capacity_costbench.py), [experiments/benchmarks/capacity_sweeps.sh](experiments/benchmarks/capacity_sweeps.sh), [experiments/capacity/network_size_table.py](experiments/capacity/network_size_table.py), [experiments/head_to_head/run.py](experiments/head_to_head/run.py)
- Command: [experiments/capacity/README.md#L15-L25](experiments/capacity/README.md?plain=1#L15-L25) (head-to-head), [experiments/capacity/README.md#L38-L43](experiments/capacity/README.md?plain=1#L38-L43) (decisions/s and table), [experiments/capacity/README.md#L91-L93](experiments/capacity/README.md?plain=1#L91-L93) (figure data)
- Results: [experiments/head_to_head/results/h2h_experimental_vs_prod.csv](experiments/head_to_head/results/h2h_experimental_vs_prod.csv), [experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt](experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt) (`fast/capacity` sections), [experiments/capacity/README.md#L54-L71](experiments/capacity/README.md?plain=1#L54-L71) (table), [experiments/benchmarks/README.md#L341-L358](experiments/benchmarks/README.md?plain=1#L341-L358) (decisions/s with intervals), [experiments/capacity/results/capacity_cubeful.dat](experiments/capacity/results/capacity_cubeful.dat), [experiments/capacity/results/capacity_prob5.dat](experiments/capacity/results/capacity_prob5.dat) (figure data)

## Table 11: Cubeless-money throughput

PureTD: C API `torch` rows; the `simd` rows are Table 13's.

- Code: [experiments/benchmarks/bench.sh](experiments/benchmarks/bench.sh), [experiments/benchmarks/aggregate_sweeps.py](experiments/benchmarks/aggregate_sweeps.py)
- Command: [experiments/benchmarks/README.md#L211-L217](experiments/benchmarks/README.md?plain=1#L211-L217) (suite), [experiments/benchmarks/README.md#L240-L261](experiments/benchmarks/README.md?plain=1#L240-L261) (per job)
- Results: [experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt](experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt)

## Table 12: Cubeful-money throughput

PureTD: C API `torch` rows; the `simd` rows are Table 13's. Open Sage, pruned: section `slow/sagewarm`, as Table 5.

- Code: [experiments/benchmarks/bench.sh](experiments/benchmarks/bench.sh), [experiments/benchmarks/aggregate_sweeps.py](experiments/benchmarks/aggregate_sweeps.py), [experiments/benchmarks/fixedpool_bench.py](experiments/benchmarks/fixedpool_bench.py) (engine `sage-filt`), [third_party_opponents/opensage/opensage_bench/sage_bench.cpp](third_party_opponents/opensage/opensage_bench/sage_bench.cpp) (the pruned 1-ply loop)
- Command: [experiments/benchmarks/README.md#L211-L217](experiments/benchmarks/README.md?plain=1#L211-L217) (suite), [experiments/benchmarks/README.md#L240-L261](experiments/benchmarks/README.md?plain=1#L240-L261) (per job)
- Results: [experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt](experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt)

## Table 13: PyTorch versus the optimized kernel

PureTD C API, `torch` and `simd` backends, in the same sweeps as Tables 11 and 12. 1-ply rows: root filter 0.08.

- Code: [c_api/backend_torch.cpp](c_api/backend_torch.cpp), [c_api/backend_simd.c](c_api/backend_simd.c), [c_api/README.md#root-filter](c_api/README.md#root-filter) (root filter), [experiments/benchmarks/rootf_validate.py](experiments/benchmarks/rootf_validate.py)
- Command: [experiments/benchmarks/README.md#L211-L217](experiments/benchmarks/README.md?plain=1#L211-L217) (suite), [experiments/benchmarks/README.md#L240-L261](experiments/benchmarks/README.md?plain=1#L240-L261) (per job)
- Results: [experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt](experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt), [experiments/benchmarks/README.md#L299-L328](experiments/benchmarks/README.md?plain=1#L299-L328) (`PureTD torch` and `PureTD simd` rows, with intervals), [experiments/benchmarks/results/rootf_validate_20261011.txt](experiments/benchmarks/results/rootf_validate_20261011.txt) (root filter agreement)

## Table 21: Cube-handling comparison with a larger training budget for prob5

Table 7's comparison with the stopping-rule nets: cubeful cycle 3 against prob5 cycle 5 (`janowski:`), two runs; nets in [best_models/history/](best_models/history/) (md5 in the name). Same code, changes from the paper and protocol as Table 7. Head-to-head, 1.2M games each: run 1 +16.5 [+16.0, +16.9] (paper +15.9), run 2 +16.8 [+16.4, +17.3] (paper +15.1). XG++ PR, run 1 (the paper's games, not re-measured): PureTD 0.77 [0.73, 0.82]; Prob5Jan 1.02 [0.97, 1.09] (the paper's adaptive index).

- Code: as Table 7
- Command: [experiments/janowski/README.md#L70-L87](experiments/janowski/README.md?plain=1#L70-L87) (head-to-head), [experiments/janowski/README.md#L96-L101](experiments/janowski/README.md?plain=1#L96-L101) (XG++ PR)
- Results: [experiments/janowski/results/table21_run1_vr_1.2M.json](experiments/janowski/results/table21_run1_vr_1.2M.json), [experiments/janowski/results/table21_run2_vr_1.2M.json](experiments/janowski/results/table21_run2_vr_1.2M.json), [experiments/janowski/results/janowski_h2h.csv](experiments/janowski/results/janowski_h2h.csv), [experiments/janowski/README.md#L56-L60](experiments/janowski/README.md?plain=1#L56-L60) (head-to-head table), [experiments/janowski/README.md#L113-L116](experiments/janowski/README.md?plain=1#L113-L116) (XG++ PR table); XG++ per-game exports not committed

## Table 26: Head-to-head strength against HedgeHog

HedgeHog engine, shim binary and models are third-party and not committed: [third_party_opponents/hedgehog/README.md#L21-L45](third_party_opponents/hedgehog/README.md?plain=1#L21-L45). HedgeHog engine `e8847f4`. Cubeful cells: the cube-capped (64, both sides) rows. The paper gives HedgeHog's equity: signs flipped against the CSVs.

- Code: [third_party_opponents/hedgehog/hedgehog_agent.py](third_party_opponents/hedgehog/hedgehog_agent.py), [third_party_opponents/hedgehog/shim/hh_shim.cpp](third_party_opponents/hedgehog/shim/hh_shim.cpp), [play_models.py#L479-L489](play_models.py#L479-L489), [third_party_opponents/gnubg/gnubg_eval.py#L291-L294](third_party_opponents/gnubg/gnubg_eval.py#L291-L294) (cube cap)
- Command: [experiments/head_to_head/HEDGEHOG.md#L14-L37](experiments/head_to_head/HEDGEHOG.md?plain=1#L14-L37)
- Results: [experiments/head_to_head/results/h2h_cubeless_summary.csv](experiments/head_to_head/results/h2h_cubeless_summary.csv), [experiments/head_to_head/results/h2h_cubeful_cap64_summary.csv](experiments/head_to_head/results/h2h_cubeful_cap64_summary.csv), [experiments/head_to_head/HEDGEHOG.md#L39-L62](experiments/head_to_head/HEDGEHOG.md?plain=1#L39-L62)

## Table 27: Fixed-position throughput against HedgeHog

HedgeHog engine `e8847f4`, as Table 26, 0-ply: the `fast` group (job `hh0`), eight sweeps per host. Factors: per-decision cost over the PureTD `torch` 0-ply rows of the same host; gnubg rows as Tables 11 and 12.

- Code: [experiments/benchmarks/fixedpool_bench.py](experiments/benchmarks/fixedpool_bench.py), [third_party_opponents/gnubg/fixedpool_bench_gnubg.py](third_party_opponents/gnubg/fixedpool_bench_gnubg.py), [third_party_opponents/hedgehog/shim/hh_shim.cpp](third_party_opponents/hedgehog/shim/hh_shim.cpp), [third_party_opponents/hedgehog/validate_costpath.py](third_party_opponents/hedgehog/validate_costpath.py)
- Command: [experiments/benchmarks/README.md#L255-L257](experiments/benchmarks/README.md?plain=1#L255-L257) (HedgeHog), [experiments/benchmarks/README.md#L211-L217](experiments/benchmarks/README.md?plain=1#L211-L217) (suite), [experiments/benchmarks/README.md#L240-L261](experiments/benchmarks/README.md?plain=1#L240-L261) (per job: gnubg, PureTD)
- Results: [experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt](experiments/benchmarks/results/zen5_9600x_20261113/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261113/aggregate.txt) (sections `fast/hh0`; PureTD reference rows: `fast/bench`, which `fast/hh0` re-times within 1.6%), [experiments/benchmarks/README.md#L360-L385](experiments/benchmarks/README.md?plain=1#L360-L385)
