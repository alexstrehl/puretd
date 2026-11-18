# Network size

Eight cubeful and eight prob5 nets of one to six hidden layers: strength
head-to-head at 0-ply against the production 4-layer net of the family,
and cost per decision on the fixed pools of
[../benchmarks/README.md](../benchmarks/README.md). One- to three-layer
nets: [../../best_models/experimental/small_models/README.md](../../best_models/experimental/small_models/README.md);
five- and six-layer nets: [../../best_models/experimental/README.md](../../best_models/experimental/README.md).

## Head-to-head against production

The variant as side 1. 10M games per row, chunks pooled with
`run.py --combine`. Cubeful rows: Jacoby on, capped mean (±128).

```bash
python experiments/head_to_head/run.py \
    --side1 best_models/experimental/small_models/cubeful_money_512_512_256.pt \
    --side2 best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt \
    --game-mode cubeful-money --jacoby --games 250000 --workers 48 --save DIR_CF
python experiments/head_to_head/run.py \
    --side1 best_models/experimental/cubeless_prob5_768_512_256_256_256.pt \
    --side2 best_models/history/cubeless_prob5_512_512_256_256_ac95a983.pt \
    --game-mode cubeless-money --games 250000 --workers 48 --save DIR_P5
python experiments/head_to_head/run.py --combine DIR_CF   # likewise DIR_P5
```

Cells: [../head_to_head/results/h2h_experimental_vs_prod.csv](../head_to_head/results/h2h_experimental_vs_prod.csv);
pooled outputs of four rows: `../head_to_head/results/*_vs_prod_10M.txt`.

## Decisions per second

The `cap` job of the benchmark suite's `fast` group
([../benchmarks/README.md](../benchmarks/README.md), network sizes):
[capacity_costbench.py](../benchmarks/capacity_costbench.py), eight
sweeps per host, checkpoints pinned by md5, every net through the C API
(`torch` and `simd` rows), 0-ply checker and cube decisions.

```bash
bash c_api/build.sh torch simd
BENCH_CORE=<core> bash experiments/benchmarks/capacity_sweeps.sh   # or ROWS=cap suite.sh <out> fast
python experiments/benchmarks/aggregate_sweeps.py $HOME/bench_results/capacity/capacity_sweep*.txt
python experiments/capacity/network_size_table.py   # the table below
```

## Results

Checker decisions/s at 0-ply, mean of eight sweeps; in parentheses the
per-decision cost relative to the production net on the same backend and
host. Zen 5 = Ryzen 5 9600X, Zen 3 = Ryzen 9 5950X. Sweeps and
aggregates, cube rows included: `../benchmarks/results/<host>_20261113/`
(`fast/capacity_sweep*.txt`, `aggregate.txt`); intervals in
[../benchmarks/README.md](../benchmarks/README.md) (network sizes).

| model | file | hidden | params | md5 | vs production, mEq [95% CI] | games | Zen 5 torch | Zen 3 torch | Zen 5 simd | Zen 3 simd |
|---|---|---|---|---|---|---|---|---|---|---|
| cubeful 1-layer | [`cubeful_money_512.pt`](../../best_models/experimental/small_models/cubeful_money_512.pt) | [512] | 103,425 | `39d75783` | -157.5 [-159.6, -155.3] | 10M | 28,284 (0.20×) | 24,440 (0.21×) | 189,388 (0.23×) | 112,868 (0.16×) |
| cubeful 2-layer | [`cubeful_money_512_512.pt`](../../best_models/experimental/small_models/cubeful_money_512_512.pt) | [512,512] | 366,081 | `1ddbb775` | -25.1 [-26.9, -23.3] | 10M | 8,581 (0.66×) | 7,561 (0.66×) | 64,810 (0.67×) | 27,626 (0.65×) |
| cubeful 3-layer | [`cubeful_money_512_512_256.pt`](../../best_models/experimental/small_models/cubeful_money_512_512_256.pt) | [512,512,256] | 497,153 | `27a728cf` | -4.5 [-6.3, -2.7] | 10M | 6,386 (0.88×) | 5,660 (0.89×) | 46,574 (0.94×) | 19,482 (0.92×) |
| cubeful 4-layer (reference) | [`cubeful_money_512_512_256_256_6afda0f8.pt`](../../best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt) | [512,512,256,256] | 562,945 | `6afda0f8` | --- |  | 5,646 (1.00×) | 5,018 (1.00×) | 43,550 (1.00×) | 17,931 (1.00×) |
| cubeful 5-layer | [`cubeful_money_512_512_256_256_256.pt`](../../best_models/experimental/cubeful_money_512_512_256_256_256.pt) | [512,512,256,256,256] | 628,737 | `37e41475` | +5.0 [+3.3, +6.8] | 10M | 5,062 (1.12×) | 4,509 (1.11×) | 37,750 (1.15×) | 15,485 (1.16×) |
| cubeful 5-layer wide | [`cubeful_money_768_512_256_256_256.pt`](../../best_models/experimental/cubeful_money_768_512_256_256_256.pt) | [768,512,256,256,256] | 811,265 | `f65c49fe` | +8.4 [+6.6, +10.2] | 10M | 3,972 (1.42×) | 3,538 (1.42×) | 27,665 (1.57×) | 11,032 (1.63×) |
| cubeful 6-layer wide | [`cubeful_money_768_512_256_256_256_256.pt`](../../best_models/experimental/cubeful_money_768_512_256_256_256_256.pt) | [768,512,256,256,256,256] | 877,057 | `4f86345a` | +11.1 [+9.3, +12.9] | 10M | 3,673 (1.54×) | 3,274 (1.53×) | 24,309 (1.79×) | 9,678 (1.85×) |
| cubeful 5-layer 768x768 | [`cubeful_money_768_768_256_256_256.pt`](../../best_models/experimental/cubeful_money_768_768_256_256_256.pt) | [768,768,256,256,256] | 1,073,665 | `de9e6e84` | +16.4 [+14.7, +18.2] | 10M | 3,080 (1.83×) | 2,739 (1.83×) | 19,768 (2.20×) | 7,765 (2.31×) |
| prob5 1-layer | [`cubeless_prob5_512.pt`](../../best_models/experimental/small_models/cubeless_prob5_512.pt) | [512] | 103,429 | `fa2c6935` | -82.6 [-83.5, -81.8] | 10M | 22,737 (0.23×) | 18,620 (0.24×) | 162,613 (0.23×) | 96,122 (0.16×) |
| prob5 2-layer | [`cubeless_prob5_512_512.pt`](../../best_models/experimental/small_models/cubeless_prob5_512_512.pt) | [512,512] | 366,085 | `e8c0c760` | -12.1 [-13.0, -11.3] | 10M | 7,559 (0.68×) | 6,514 (0.69×) | 52,973 (0.70×) | 22,746 (0.67×) |
| prob5 3-layer | [`cubeless_prob5_512_512_256.pt`](../../best_models/experimental/small_models/cubeless_prob5_512_512_256.pt) | [512,512,256] | 496,133 | `3da8e498` | -3.7 [-4.5, -2.8] | 10M | 5,786 (0.89×) | 5,072 (0.88×) | 41,728 (0.88×) | 17,554 (0.87×) |
| prob5 4-layer (reference) | [`cubeless_prob5_512_512_256_256_ac95a983.pt`](../../best_models/history/cubeless_prob5_512_512_256_256_ac95a983.pt) | [512,512,256,256] | 561,925 | `ac95a983` | --- |  | 5,125 (1.00×) | 4,484 (1.00×) | 36,848 (1.00×) | 15,328 (1.00×) |
| prob5 5-layer | [`cubeless_prob5_512_512_256_256_256.pt`](../../best_models/experimental/cubeless_prob5_512_512_256_256_256.pt) | [512,512,256,256,256] | 627,717 | `bade7d51` | +1.9 [+1.1, +2.8] | 10M | 4,599 (1.11×) | 4,048 (1.11×) | 33,490 (1.10×) | 13,906 (1.10×) |
| prob5 5-layer wide | [`cubeless_prob5_768_512_256_256_256.pt`](../../best_models/experimental/cubeless_prob5_768_512_256_256_256.pt) | [768,512,256,256,256] | 809,221 | `bd1a7931` | +5.2 [+4.3, +6.0] | 10M | 3,623 (1.41×) | 3,197 (1.40×) | 26,829 (1.37×) | 10,796 (1.42×) |
| prob5 6-layer wide | [`cubeless_prob5_768_512_256_256_256_256.pt`](../../best_models/experimental/cubeless_prob5_768_512_256_256_256_256.pt) | [768,512,256,256,256,256] | 875,013 | `a9df2ffc` | +6.7 [+5.8, +7.6] | 10M | 3,357 (1.53×) | 2,964 (1.51×) | 25,394 (1.45×) | 10,213 (1.50×) |
| prob5 5-layer 768x768 | [`cubeless_prob5_768_768_256_256_256.pt`](../../best_models/experimental/cubeless_prob5_768_768_256_256_256.pt) | [768,768,256,256,256] | 1,071,621 | `b3fad4eb` | +9.7 [+8.9, +10.6] | 10M | 2,814 (1.82×) | 2,483 (1.81×) | 20,315 (1.81×) | 8,060 (1.90×) |

The two 768x768 files in `best_models/experimental/` hold continuations
of the measured nets, same shape and cost: cubeful `16039ea7` and prob5
`b8d4306c`. Luck-adjusted over 200k games they score +19.8 [+18.9, +20.7]
and +10.1 [+9.7, +10.5] against the 4-layer net of their family, and +106.8
and +55.6 against gnubg 0-ply. The measured nets score +16.4 and +9.7
(table above) and +102.8 and +54.2 against gnubg 0-ply, all over 10M plain
games. Their rows:
[../../best_models/experimental/README.md](../../best_models/experimental/README.md)
and the [top README](../../README.md).

## Figure data

[results/capacity_cubeful.dat](results/capacity_cubeful.dat),
[results/capacity_prob5.dat](results/capacity_prob5.dat): one row per net,
`params` (thousands), `edge` with error-bar lengths `ehi`, `elo` (mEq),
`label`, `zen5` (Zen 5 `torch` checker decisions/s, the x-axis); `anc` and
`anc2` place the label.

```bash
python experiments/capacity/network_size_table.py --dat
```
