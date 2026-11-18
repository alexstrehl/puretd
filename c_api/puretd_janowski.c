/*
 * puretd_janowski.c -- Prob5Jan: a cubeless prob5 net under the Janowski
 * money cube layer (BGC_KIND_PROB5JAN). Formulas: R. Janowski (1993) in
 * GNU Backgammon's formulation, as janowski.py.
 *
 * The Python modules are the reference, followed function for function:
 *   janowski.py         bgc_jan_equity, _folded, _cubeless_equity, _fold,
 *                       _money_cube_action
 *   janowski_agent.py   JanowskiAgent at its defaults (cube_fold,
 *                       fold_0ply, board_sanity per bgc_set_jan): _succ_values
 *                       (row_leaf), _pick, _screen_0ply, _oneply,
 *                       _value_oneply, _static_cube, _cube_values,
 *                       offer_double, respond_to_double,
 *                       choose_checker_action_cubeful.
 *
 * Numerics: double throughout, every expression in Python's evaluation
 * order (left to right; a * b / c is (a * b) / c), no FMA contraction.
 * Sums are sequential. Net probabilities are fp32 (sanity
 * mask, then the prob5 clamps) widened to double.
 */

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "puretd.h"
#include "puretd_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define JAN_EPS 1e-7            /* janowski._EPS */
#define JAN_DEFAULT_X 0.76      /* janowski.DEFAULT_X */

/* ── conversion (janowski.py) ──────────────────────────────────── */

double bgc_jan_cubeless_equity(const double p[5]) {
    return (2.0 * p[0] + p[1] + p[2] - p[3] - p[4]) - 1.0;
}

/* Segment of _piecewise_linear: y0 + (y1 - y0) * (p - x0) / (x1 - x0). */
static double seg(double p, double x0, double y0, double x1, double y1) {
    return y0 + (y1 - y0) * (p - x0) / (x1 - x0);
}

/* The first segment with p < its right knot, else the last. */
static double pwl(double p, const double *kx, const double *ky, int nk) {
    for (int i = 0; i < nk - 2; i++)
        if (p < kx[i + 1]) return seg(p, kx[i], ky[i], kx[i + 1], ky[i + 1]);
    return seg(p, kx[nk - 2], ky[nk - 2], kx[nk - 1], ky[nk - 1]);
}

double bgc_jan_equity(const double p[5], int persp, double x, int jacoby) {
    if (persp < PERSP_CENTERED || persp > PERSP_THEIRS) return NAN;
    const double p0 = p[0];
    const double win = p0 > JAN_EPS ? p0 : JAN_EPS;
    const double lose = 1.0 - p0 > JAN_EPS ? 1.0 - p0 : JAN_EPS;
    const double W = 1.0 + (p[1] + p[2]) / win;
    const double L = 1.0 + (p[3] + p[4]) / lose;
    const double dead = jacoby && persp == PERSP_CENTERED ? 2.0 * p0 - 1.0
                                                          : bgc_jan_cubeless_equity(p);
    if (p0 <= JAN_EPS || p0 >= 1.0 - JAN_EPS) return dead;
    const double denom = W + L + 0.5;
    const double tp = (L - 0.5) / denom;
    const double cp = (L + 1.0) / denom;
    double live;
    if (persp == PERSP_CENTERED) {
        const double kx[4] = {0.0, tp, cp, 1.0};
        const double ky[4] = {jacoby ? -1.0 : -L, -1.0, 1.0, jacoby ? 1.0 : W};
        live = pwl(p0, kx, ky, 4);
    } else if (persp == PERSP_MINE) {
        const double kx[3] = {0.0, cp, 1.0}, ky[3] = {-L, 1.0, W};
        live = pwl(p0, kx, ky, 3);
    } else {
        const double kx[3] = {0.0, tp, 1.0}, ky[3] = {-L, -1.0, W};
        live = pwl(p0, kx, ky, 3);
    }
    return (1.0 - x) * dead + x * live;
}

double bgc_jan_fold(double nd, double dt) {
    return (dt >= nd && 1.0 >= nd) ? (dt < 1.0 ? dt : 1.0) : nd;
}

double bgc_jan_equity_folded(const double p[5], int persp, double x, int jacoby) {
    const double nd = bgc_jan_equity(p, persp, x, jacoby);
    if (persp != PERSP_CENTERED && persp != PERSP_MINE) return nd;
    return bgc_jan_fold(nd, 2.0 * bgc_jan_equity(p, PERSP_THEIRS, x, jacoby));
}

/* TDAgent._money_cube_decision */
static int money_double(double nd, double dt, int persp, int jacoby) {
    if (dt > 1.0) {
        if (jacoby && persp == PERSP_CENTERED) return 1;
        return 1.0 >= nd;
    }
    return dt > nd;
}

int bgc_jan_money_cube_action(double nd, double dt, int persp, int jacoby, int *take) {
    if (take) *take = dt <= 1.0;
    return money_double(nd, dt, persp, jacoby);
}

/* ── the engine's state ────────────────────────────────────────── */

/* A leaf (successor row or terminal reply), the leaf player's view:
   value per branch, cubeless equity, the probabilities used. */
typedef struct { double v[2], sec, p[5]; } leaf;

struct bgc_jan {
    double x;
    int board_sanity;
    leaf *lf;                       /* per batch row */
    int cap_lf;
    leaf *c0;                       /* 0-ply leaves of the candidates, MAX_PLAYS */
    unsigned char *ex0;             /* candidate is terminal */
    double *sc, *ss;                /* 1-ply (score, sec) per candidate */
};

struct bgc_jan *bgc_i_jan_new(void) {
    struct bgc_jan *j = calloc(1, sizeof(*j));
    if (!j) return NULL;
    j->x = JAN_DEFAULT_X;
    j->board_sanity = 1;
    j->c0 = malloc(MAX_PLAYS * sizeof(leaf));
    j->ex0 = malloc(MAX_PLAYS);
    j->sc = malloc(MAX_PLAYS * sizeof(double));
    j->ss = malloc(MAX_PLAYS * sizeof(double));
    if (!j->c0 || !j->ex0 || !j->sc || !j->ss) { bgc_i_jan_free(j); return NULL; }
    return j;
}

void bgc_i_jan_free(struct bgc_jan *j) {
    if (!j) return;
    free(j->lf); free(j->c0); free(j->ex0); free(j->sc); free(j->ss);
    free(j);
}

int bgc_set_jan(bgc_engine *e, const bgc_jan_opts *o) {
    if (!e || bgc_kind(e) != BGC_KIND_PROB5JAN) return -1;
    struct bgc_jan *j = bgc_i_jan(e);
    const double x = o ? o->x : JAN_DEFAULT_X;
    if (!isfinite(x)) return -1;
    j->x = x;
    j->board_sanity = o ? o->board_sanity != 0 : 1;
    return 0;
}

/* Branch cubes: the leaf player's cube perspective, jacoby and whether
   gammons count, per root cube state. */
typedef struct { int n, persp[2], jac[2], gc[2]; } branches;

static void branch_set(branches *br, int b, const bgc_cube *c, int leaf_player) {
    br->persp[b] = cube_persp(c->owner, leaf_player);
    br->jac[b] = c->jacoby != 0;
    br->gc[b] = gammons_count(c);
}

/* MatchState.after_take(doubler). */
static bgc_cube after_take(const bgc_cube *c, int doubler) {
    bgc_cube t = *c;
    t.owner = (1 - doubler) + 1;
    t.value = c->value * 2;
    return t;
}

/* The terminal leaf: the player on roll there lost result r. */
static void leaf_terminal(const branches *br, int r, leaf *L) {
    L->p[0] = L->p[1] = L->p[2] = 0.0;
    L->p[3] = r >= 2 ? 1.0 : 0.0;
    L->p[4] = r >= 3 ? 1.0 : 0.0;
    for (int b = 0; b < br->n; b++) L->v[b] = -(br->gc[b] ? (double)r : 1.0);
    L->sec = bgc_jan_cubeless_equity(L->p);
}

/* The net's probabilities of a row: prob5_impossible_mask zeroing (when
   board_sanity), the prob5 clamps, widened to double. */
static void row_probs(const struct bgc_jan *j, const float *x, const float *y, double p[5]) {
    float q[5], c[5];
    memcpy(q, y, sizeof q);
    if (j->board_sanity) {
        unsigned char m[4];
        if (prob5_impossible_mask(x, 1, m))
            for (int k = 0; k < 4; k++) if (m[k]) q[1 + k] = 0.0f;
    }
    bgc_i_prob5_clamp(q, c);
    for (int k = 0; k < 5; k++) p[k] = (double)c[k];
}

/* _succ_values for one non-terminal row (fold on): the net's
   probabilities through the folded conversion. */
static void row_leaf(struct bgc_jan *j, const branches *br, const float *x, const float *y,
                     leaf *L) {
    row_probs(j, x, y, L->p);
    for (int k = 0; k < br->n; k++)
        L->v[k] = bgc_jan_equity_folded(L->p, br->persp[k], j->x, br->jac[k]);
    L->sec = bgc_jan_cubeless_equity(L->p);
}

static int ensure_lf(struct bgc_jan *j, int n) {
    if (n <= j->cap_lf) return 0;
    int cap = j->cap_lf ? j->cap_lf : 4096;
    while (cap < n) cap *= 2;
    leaf *l = realloc(j->lf, (size_t)cap * sizeof(leaf));
    if (!l) return -1;
    j->lf = l;
    j->cap_lf = cap;
    return 0;
}

/* _pick's order: (value, sec) compared in float32, lexicographic; the
   first of equals. Returns 1 when (v, s) beats the best so far. */
static inline int pick_better(double v, double s, int have, float *bv, float *bs) {
    const float fv = (float)v, fs = (float)s;
    if (have && !(fv < *bv || (fv == *bv && fs < *bs))) return 0;
    *bv = fv;
    *bs = fs;
    return 1;
}

/* _oneply(S, branch cubes) with S non-terminal: v[b] (roller's view, per
   unit cube) and the roller's cubeless equity of branch 0's averaged
   probabilities. 0, -1 on allocation failure. */
static int jan_oneply(bgc_engine *e, const BoardState *S, const bgc_cube *cube, int nb,
                      double v[2], double *sec) {
    struct bgc_jan *j = bgc_i_jan(e);
    const int roller = S->turn, leaf_player = 1 - roller;
    const bgc_cube taken = after_take(cube, roller);
    branches br;
    br.n = nb;
    branch_set(&br, 0, cube, leaf_player);
    if (nb > 1) branch_set(&br, 1, &taken, leaf_player);
    int nrow = 0, nrep = 0;
    if (bgc_i_expand21(e, S, 0, -1, &nrow, &nrep) || ensure_lf(j, nrow)) return -1;
    bgc_i_forward_succ(e, 0, nrow, S, 0, 0);
    bgc_batch bt;
    bgc_i_batch(e, &bt);
    for (int r = 0; r < nrow; r++)
        row_leaf(j, &br, bt.x + (size_t)r * bt.ldx, bt.y + (size_t)r * bt.out_dim, &j->lf[r]);
    double acc[2] = {0.0, 0.0}, pacc[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    int k = 0;
    for (int d1 = 1; d1 <= 6; d1++) {
        for (int d2 = d1; d2 <= 6; d2++, k++) {
            const double w = d1 == d2 ? 1.0 : 2.0;
            for (int b = 0; b < nb; b++) {
                leaf best = {{0.0, 0.0}, 0.0, {0.0}}, t;
                float bv = 0.0f, bs = 0.0f;
                int have = 0;
                for (int r = bt.cdoff[k]; r < bt.cdoff[k + 1]; r++) {
                    const leaf *L;
                    if (bt.rep_row[r] >= 0) {
                        L = &j->lf[bt.rep_row[r]];
                    } else {
                        leaf_terminal(&br, bt.rep_gr[r], &t);
                        L = &t;
                    }
                    if (pick_better(L->v[b], L->sec, have, &bv, &bs)) { best = *L; have = 1; }
                }
                acc[b] += w * best.v[b];
                if (b == 0)
                    for (int c = 0; c < 5; c++) pacc[c] += w * best.p[c];
            }
        }
    }
    for (int b = 0; b < nb; b++) v[b] = -acc[b] / 36.0;
    double p[5], rp[5];
    for (int c = 0; c < 5; c++) p[c] = pacc[c] / 36.0;
    rp[0] = 1.0 - p[0];
    rp[1] = p[3];
    rp[2] = p[4];
    rp[3] = p[1];
    rp[4] = p[2];
    *sec = bgc_jan_cubeless_equity(rp);
    return 0;
}

/* _value_oneply(S, cube, fold=True): (value, sec), S->turn's view. */
static int jan_value_oneply(bgc_engine *e, const BoardState *S, const bgc_cube *cube,
                            double *val, double *sec) {
    double v[2];
    if (can_offer(cube->owner, S->turn)) {
        if (jan_oneply(e, S, cube, 2, v, sec)) return -1;
        *val = bgc_jan_fold(v[0], 2.0 * v[1]);
    } else {
        if (jan_oneply(e, S, cube, 1, v, sec)) return -1;
        *val = v[0];
    }
    return 0;
}

double bgc_i_jan_value1(bgc_engine *e, const BoardState *s, const bgc_cube *cube, int *err) {
    *err = 0;
    if (!cube || board_is_game_over(s)) return NAN;
    double v, sec;
    if (jan_value_oneply(e, s, cube, &v, &sec)) { *err = 1; return NAN; }
    return v;
}

/* choose_checker_action_cubeful over the n candidates in the batch's
   plays: 0-ply screen (opponent's view, folded), then at 1-ply every
   candidate (n > 1), game-ending ones keeping their 0-ply (value, sec). cscore
   = mover view = -value. Returns the pick, -2 on error. */
int bgc_i_jan_choose(bgc_engine *e, const BoardState *s, int n, const bgc_cube *cube,
                     int plies) {
    if (!cube) return -2;
    struct bgc_jan *j = bgc_i_jan(e);
    bgc_batch bt;
    bgc_i_batch(e, &bt);
    branches br;
    br.n = 1;
    branch_set(&br, 0, cube, 1 - s->turn);
    for (int i = 0; i < n; i++) bgc_i_encode_base(e, i, &bt.plays[i].resulting_state);
    bgc_i_forward_succ(e, 0, n, s, 0, 0);
    bgc_i_batch(e, &bt);
    for (int i = 0; i < n; i++) {
        const BoardState *rs = &bt.plays[i].resulting_state;
        if (board_is_game_over(rs)) {
            leaf_terminal(&br, bgc_i_game_result(rs), &j->c0[i]);
            j->ex0[i] = 1;
        } else {
            row_leaf(j, &br, bt.x + (size_t)i * bt.ldx, bt.y + (size_t)i * bt.out_dim,
                     &j->c0[i]);
            j->ex0[i] = 0;
        }
        j->sc[i] = j->c0[i].v[0];
        j->ss[i] = j->c0[i].sec;
    }
    if (plies == 1 && n > 1) {
        for (int i = 0; i < n; i++) {
            if (j->ex0[i]) continue;
            const BoardState S = bt.plays[i].resulting_state;
            if (jan_value_oneply(e, &S, cube, &j->sc[i], &j->ss[i])) return -2;
        }
    }
    int best = 0;
    float bv = 0.0f, bs = 0.0f;
    for (int i = 0; i < n; i++) {
        if (pick_better(j->sc[i], j->ss[i], i > 0, &bv, &bs)) best = i;
        bt.cscore[i] = -j->sc[i];
    }
    return best;
}

/* The root's net probabilities (plain forward of row 0). */
static void root_probs(bgc_engine *e, const BoardState *s, double p[5]) {
    bgc_i_encode_base(e, 0, s);
    bgc_i_forward(e, 0, 1);
    bgc_batch bt;
    bgc_i_batch(e, &bt);
    row_probs(bgc_i_jan(e), bt.x, bt.y, p);
}

/* _cube_values at plies (0: _static_cube): (nd, dt) of s->turn. 0, -1
   on allocation failure. */
static int jan_cube_values(bgc_engine *e, const BoardState *s, const bgc_cube *cube,
                           int plies, double *nd, double *dt) {
    struct bgc_jan *j = bgc_i_jan(e);
    const int persp = cube_persp(cube->owner, s->turn);
    if (plies == 1) {
        double v[2], sec;
        if (jan_oneply(e, s, cube, 2, v, &sec)) return -1;
        *nd = v[0];
        *dt = 2.0 * v[1];
        return 0;
    }
    double p[5];
    root_probs(e, s, p);
    *nd = bgc_jan_equity(p, persp, j->x, cube->jacoby);
    *dt = 2.0 * bgc_jan_equity(p, PERSP_THEIRS, j->x, cube->jacoby);
    return 0;
}

float bgc_i_jan_value(bgc_engine *e, const BoardState *s, const bgc_cube *cube) {
    double nd, dt;
    if (!cube || jan_cube_values(e, s, cube, 0, &nd, &dt)) return NAN;
    return (float)nd;
}

int bgc_i_jan_value_prob5(bgc_engine *e, const BoardState *s, float probs[5], float *eq) {
    bgc_i_encode_base(e, 0, s);
    bgc_i_forward(e, 0, 1);
    bgc_batch bt;
    bgc_i_batch(e, &bt);
    float q[5], c[5];
    memcpy(q, bt.y, sizeof q);
    if (bgc_i_jan(e)->board_sanity) {
        unsigned char m[4];
        if (prob5_impossible_mask(bt.x, 1, m))
            for (int k = 0; k < 4; k++) if (m[k]) q[1 + k] = 0.0f;
    }
    const float v = bgc_i_prob5_clamp(q, c);
    if (probs) memcpy(probs, c, sizeof c);
    if (eq) *eq = v;
    return 0;
}

int bgc_i_jan_offer(bgc_engine *e, const BoardState *s, const bgc_cube *cube, int plies,
                    bgc_cube_eval *out) {
    if (!cube || (plies != 0 && plies != 1)) return -1;
    const int persp = cube_persp(cube->owner, s->turn);
    if (persp == PERSP_THEIRS) return 0;
    double nd, dt;
    if (jan_cube_values(e, s, cube, plies, &nd, &dt)) return -1;
    int take;
    const int dbl = bgc_jan_money_cube_action(nd, dt, persp, cube->jacoby, &take);
    if (out) { out->v_no_double = (float)nd; out->v_double_take = (float)dt; out->take = take; }
    return dbl;
}

int bgc_i_jan_respond(bgc_engine *e, const BoardState *s, const bgc_cube *cube, int plies,
                      const bgc_cube_eval *hint) {
    if (!cube || (plies != 0 && plies != 1)) return -1;
    if (hint && hint->plies == plies && !isnan(hint->v_double_take)) return hint->take != 0;
    double nd, dt;
    if (jan_cube_values(e, s, cube, plies, &nd, &dt)) return -1;
    return dt <= 1.0;
}
