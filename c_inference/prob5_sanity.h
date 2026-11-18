/* prob5_sanity.h -- board-sanity rules for prob5 outputs.
 *
 * Given the perspective196 row of the player on roll, decides which of
 * P(win gammon), P(win backgammon), P(lose gammon), P(lose backgammon)
 * the board rules out. The caller zeroes those outputs; the nesting
 * clamps (P(wg) <= P(win), P(lg) <= 1 - P(win), P(wbg) <= P(wg),
 * P(lbg) <= P(lg)) are applied afterwards by the caller.
 *
 * Per side, from that side's own direction of play:
 *   on_board   checkers on the board, the bar included (15 = none off);
 *   rearmost   point - 1 of the side's rearmost checker (0..23), 24 when
 *              a checker is on the bar, 0 with no checker on the board.
 * A race: rearmost(me) + rearmost(opponent) < 24, i.e. the two sides
 * have passed each other and no checker can be hit again.
 *
 * Rules (me = the player on roll):
 *   1. Once a side has borne off a checker (on_board < 15) it cannot lose
 *      a gammon or a backgammon: the opponent bearing off zeroes P(wg)
 *      and P(wbg), my bearing off zeroes P(lg) and P(lbg).
 *   2. In a race, a side with no checker in the winner's home board or
 *      on the bar (rearmost < 18) cannot lose a backgammon: P(wbg) = 0
 *      when rearmost(opponent) < 18, P(lbg) = 0 when rearmost(me) < 18.
 * Both rules match impossibility cases that GNU Backgammon's evaluator
 * also zeroes (SanityCheck in its eval.c).
 *
 * Decoding a row: my block is features 0..97, the opponent's 98..195.
 * Point k of a block (k = 0..23, four features each) holds a
 * thermometer, count = f0 + f1 + f2 + 2 * f3; the feature at 96 of a
 * block is bar / 2. Point k of my block is my point k + 1; the
 * opponent's block is mirrored, its point k is the opponent's point
 * 24 - k.
 *
 * One implementation for C, included by c_inference/nn_eval.c and
 * c_engine/bg_engine.c; prob_agent._prob5_impossible_np is the same
 * rules in numpy (parity: tests/test_prob5_sanity.py).
 */
#ifndef PROB5_SANITY_H
#define PROB5_SANITY_H

typedef struct {
    int on_board, rearmost;
} prob5_side;

/* The counts of one block of a perspective196 row. `mirrored`: the
 * opponent's block. */
static inline prob5_side prob5_side_counts(const float *block, int mirrored) {
    prob5_side c = {0, 0};
    for (int k = 0; k < 24; k++) {
        const float *f = block + 4 * k;
        if (f[0] == 0.0f) continue;                       /* empty point */
        int n = (int)(f[0] + f[1] + f[2] + 2.0f * f[3] + 0.5f);
        int pt = mirrored ? 23 - k : k;                   /* owner's point - 1 */
        c.on_board += n;
        if (pt > c.rearmost) c.rearmost = pt;
    }
    int bar = (int)(2.0f * block[96] + 0.5f);
    if (bar) {
        c.on_board += bar;
        c.rearmost = 24;
    }
    return c;
}

/* mask[0..3] = 1 where P(wg), P(wbg), P(lg), P(lbg) is ruled out.
 * Returns 1 if any is. */
static inline int prob5_ruled_out(const float *x, unsigned char mask[4]) {
    prob5_side me = prob5_side_counts(x, 0);
    prob5_side opp = prob5_side_counts(x + 98, 1);
    int race = me.rearmost + opp.rearmost < 24;
    int opp_off = opp.on_board < 15, me_off = me.on_board < 15;
    mask[0] = opp_off;
    mask[1] = opp_off || (race && opp.rearmost < 18);
    mask[2] = me_off;
    mask[3] = me_off || (race && me.rearmost < 18);
    return mask[0] | mask[1] | mask[2] | mask[3];
}

#endif
