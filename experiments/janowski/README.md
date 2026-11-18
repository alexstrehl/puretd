# Learned versus heuristic cube handling

PureTD's cubeful money net against Prob5Jan: the prob5 cubeless net with
an analytic Janowski cube layer (R. Janowski, 1993, in GNU Backgammon's formulation). `janowski.py` converts the five
probabilities into cubeful equity, `(1 - x) * dead + x * live`, with the
piecewise-linear live-cube curve (knots at the live take and cash points)
and a fixed cube-life index x = 0.76 (`janowski.DEFAULT_X`; env
`JANOWSKI_X=<x>` overrides); `janowski_agent.py` plays cubeful money with
it. Checker play ranks candidates by the cubeful equity of the resulting
position with the opponent's cube decision folded in: the opponent
doubles when double/take and double/pass both beat no double, then takes
or passes, whichever is worse for it; exact ties go to the better
cubeless equity. Cube actions follow the repo's money rule
(`janowski.money_cube_action`: PureTD's `TDAgent._money_cube_decision`,
take iff DT <= 1). 1-ply has the structure of PureTD's cubeful 1-ply:
every candidate is expanded, and per roll each cube branch takes its own
best reply (the paper plays Prob5Jan at 0-ply only). Neither side uses
bearoff tables: the nets evaluate every position. Prob5Jan zeroes the
gammon and backgammon probabilities the board rules out
(`prob_agent.prob5_sanity`; env `JANOWSKI_BOARD_SANITY=0` turns it off).
Plain prob5 does not (`PROB5_BOARD_SANITY=1` turns it on). The code is
`JANOWSKI_REV` 2026-11-17. Saved head-to-heads record the revision, x and
the sanity setting; runs that differ do not pool.

## Changes from the paper

- No bearoff tables on either side.
- Prob5Jan uses a fixed cube-life index, x = 0.76 in every position. The
  paper used an adaptive index (constant in contact, linear in the pip
  count in races).
- Results are luck-adjusted (variance-reduced, below).

## Inputs

- PureTD, the cubeful checkpoint:
  [best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt](../../best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt)
  (md5 `6afda0f85b039a186f018753120ed74c`).
- Prob5Jan, side `janowski:`:
  [best_models/history/cubeless_prob5_512_512_256_256_ac95a983.pt](../../best_models/history/cubeless_prob5_512_512_256_256_ac95a983.pt)
  (md5 `ac95a983a114c269257225e0b84a5d9f`).
- Table 21 nets: [best_models/history/](../../best_models/history/) (`cubeful_run*_c3_*`, `prob5_run*_c5_*`)
  (cubeful cycle 3 against prob5 cycle 5 of each stopping-rule run; md5 in
  the name).
- XG++ for the PR.

## Head-to-head

Cubeful money, Jacoby, both sides 0-ply in the C API, independent dice,
luck-adjusted with
[variance.py](../head_to_head/variance.py) (beta 1, luck reference
`6afda0f8`; [../head_to_head/README.md](../head_to_head/README.md),
"Luck-adjusted scoring"). 95% CIs over game pairs. Positive = PureTD
ahead, mEq/game. Capped: each game's payoff clipped to ±128 before the
luck is subtracted.

| | PureTD | Prob5Jan | games | capped | uncapped | paper |
|---|---|---|---|---|---|---|
| Table 7 | `6afda0f8` | `ac95a983` | 1.2M | +15.3 [+14.8, +15.7] | +15.3 [+14.8, +15.7] | +14.6 |
| Table 21, run 1 | `7677895c` | `44da01ee` | 1.2M | +16.5 [+16.0, +16.9] | +16.5 [+16.0, +16.9] | +15.9 |
| Table 21, run 2 | `0ff8457f` | `98c16095` | 1.2M | +16.8 [+16.4, +17.3] | +16.8 [+16.4, +17.3] | +15.1 |

No game exceeded ±128, so capped and uncapped agree. The luck adjustment
cuts the variance about 120-fold. Pooled summaries:
[results/table7_vr_1.2M.json](results/table7_vr_1.2M.json),
[results/table21_run1_vr_1.2M.json](results/table21_run1_vr_1.2M.json),
[results/table21_run2_vr_1.2M.json](results/table21_run2_vr_1.2M.json);
rows in [results/janowski_h2h.csv](results/janowski_h2h.csv). Per-game
data available on request.

```bash
# Table 7: twelve chunks of 50k pairs (100k games), k = 0..11; one DIR per run
OMP_NUM_THREADS=1 python experiments/head_to_head/variance.py --game-mode cubeful-money \
    --side1 best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt \
    --side2 janowski:best_models/history/cubeless_prob5_512_512_256_256_ac95a983.pt \
    --reference best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt \
    --dice independent --seed 20261116 --pairs 50000 --pair-start $((k * 50000)) \
    --workers 48 --out DIR/c$k.npz
# Table 21, run 1 (seed 20261117; run 2: cubeful_run2_c3_0ff8457f.pt,
# prob5_run2_c5_98c16095.pt, seed 20261118)
OMP_NUM_THREADS=1 python experiments/head_to_head/variance.py --game-mode cubeful-money \
    --side1 best_models/history/cubeful_run1_c3_7677895c.pt \
    --side2 janowski:best_models/history/prob5_run1_c5_44da01ee.pt \
    --reference best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt \
    --dice independent --seed 20261117 --pairs 50000 --pair-start $((k * 50000)) \
    --workers 48 --out DIR/c$k.npz
python experiments/head_to_head/variance.py --summarize DIR/c*.npz   # pool one run's chunks
```

## XG++ PR

1,000 self-play cubeful money games per side, analysed in XG++; the PR is
decision-weighted over both players of each game, with a game-level
bootstrap 95% interval. The XG++ exports are not committed. Measured for
the paper, with bearoff tables; not re-measured with this code.

```bash
# per side: Prob5Jan (below) or PureTD (--model1/--model2 <cubeful.pt>)
python play_models.py --model1 janowski:<prob5.pt> --model2 janowski:<prob5.pt> \
    --game-mode cubeful-money --jacoby --games 100 --save-games DIR   # x10, then XG++
python tools/analyze_xg_results.py <export.txt> --rows-per-game 2
```

Table 7:

| | PR | checker | cube |
|---|---|---|---|
| PureTD 0-ply | 0.80 [0.76, 0.84] | 0.85 [0.80, 0.89] | 0.56 [0.46, 0.67] |
| Prob5Jan 0-ply, adaptive index (paper) | 1.13 [1.06, 1.22] | 1.05 [0.97, 1.14] | 1.58 [1.36, 1.83] |

Table 21, run 1 (PureTD `7677895c`, Prob5Jan `44da01ee`; the paper's
games):

| | PR | checker | cube |
|---|---|---|---|
| PureTD 0-ply | 0.77 [0.73, 0.82] | 0.84 [0.79, 0.89] | 0.44 [0.35, 0.53] |
| Prob5Jan 0-ply, adaptive index (paper) | 1.02 [0.97, 1.09] | 0.97 [0.91, 1.02] | 1.35 [1.16, 1.56] |
