"""prob_agent.prob5_sanity zeroes the gammons / backgammons the board
rules out (rules: c_inference/prob5_sanity.h). On for Prob5Jan
(JanowskiAgent), off by default for plain prob5 (ProbAgent board_sanity).

Expected masks come from the rules (`_rules`, on a side-by-side
view of the board) and from hand-built positions, one or more per rule.
`TestCParity`: the C implementation (prob5_sanity.h, through
bg_engine.c and nn_eval.c) gives prob_agent's masks and values.
"""

import os
import random
import shutil
import subprocess

import numpy as np
import pytest
import torch

from backgammon_engine import (BoardState, WHITE, BLACK, get_legal_plays,
                               switch_turn, opening_roll, is_race)
from encoding import get_encoder
from model import prob5_postprocess
import prob_agent
from prob_agent import ProbAgent, prob5_sanity

ENC = get_encoder("perspective196")


# ── the rules ─────────────────────────────────────────
#
# A board as two sides, (me = on roll, opponent): each a list of 25
# counts, index p = the owner's point - 1 (0..23), 24 = the bar.


def _sides(state):
    """(me, opponent) from a BoardState."""
    out = []
    for color in (state.turn, 1 - state.turn):
        a = [0] * 25
        for p in range(24):
            v = state.points[p] if color == WHITE else -state.points[23 - p]
            if v > 0:
                a[p] = v
        a[24] = state.bar[color]
        out.append(a)
    return tuple(out)


def _board(me, opp, turn):
    """BoardState with `turn` on roll (None if the sides share a point)."""
    pts = [0] * 24
    bar = [0, 0]
    off = [0, 0]
    for side, color in ((me, turn), (opp, 1 - turn)):
        for p in range(24):
            n = side[p]
            if not n:
                continue
            j = p if color == WHITE else 23 - p
            if pts[j]:
                return None
            pts[j] = n if color == WHITE else -n
        bar[color] = side[24]
        off[color] = 15 - sum(side)
    return BoardState(points=pts, bar=bar, off=off, turn=turn)


def _side_counts(side):
    """on_board, rearmost (prob5_sanity.h)."""
    on_board = sum(side)
    rearmost = 24 if side[24] else max([p for p in range(24) if side[p]] + [0])
    return on_board, rearmost


def _rules(me, opp):
    """Ruled-out (wg, wbg, lg, lbg) for the player on roll."""
    m_on, m_rear = _side_counts(me)
    o_on, o_rear = _side_counts(opp)
    race = m_rear + o_rear < 24
    opp_bore_off, i_bore_off = o_on < 15, m_on < 15
    return (opp_bore_off,
            opp_bore_off or (race and o_rear < 18),
            i_bore_off,
            i_bore_off or (race and m_rear < 18))


def _expected(me, opp, out):
    """The rules' zeroing, then the nesting clamps (float32)."""
    o = np.array(out, dtype=np.float32)
    for k, ruled_out in enumerate(_rules(me, opp), start=1):
        if ruled_out:
            o[k] = 0.0
    o[1] = min(o[1], o[0])
    o[3] = min(o[3], np.float32(1.0) - o[0])
    o[2] = min(o[2], o[1])
    o[4] = min(o[4], o[3])
    return o


def _ours(states, probs):
    x = torch.from_numpy(np.stack([ENC.encode(s) for s in states]))
    return prob5_sanity(torch.as_tensor(probs, dtype=torch.float32), x).numpy()


def _mask_of(state):
    x = ENC.encode(state)[None].astype(np.float32)
    m = prob_agent._prob5_impossible_np(x)
    return (False,) * 4 if m is None else tuple(bool(v) for v in m[0])


def _pts(**kw):
    """A side from p<index>=count keywords (p24 = the bar)."""
    a = [0] * 25
    for k, v in kw.items():
        a[int(k[1:])] = v
    return a


# ── position generators ──────────────────────────────────────────────


def _playout_states(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        state, dice = opening_roll(rng)
        while not state.is_game_over():
            plays = get_legal_plays(state, dice)
            state = rng.choice(plays)[1] if plays else switch_turn(state)
            if not state.is_game_over():
                out.append(state)
            dice = (rng.randint(1, 6), rng.randint(1, 6))
    return out[:n]


def _spread(rng, n, lo, hi, side):
    for _ in range(n):
        side[rng.randint(lo, hi)] += 1


def _random_board(rng, kind):
    """kind: 'race' (no contact), 'contact', 'bar'. Random off counts."""
    for _ in range(1000):
        me, opp = [0] * 25, [0] * 25
        n = [rng.choice([15, 15, rng.randint(1, 15)]) for _ in range(2)]
        if kind == "race":
            a = rng.randint(0, 23)          # my rearmost bound
            b = rng.randint(0, 23 - a)      # the opponent's: a + b < 24
            _spread(rng, n[0], 0, a, me)
            _spread(rng, n[1], 0, b, opp)
        else:
            _spread(rng, n[0], 0, 23, me)
            _spread(rng, n[1], 0, 23, opp)
            if kind == "bar":
                for side, k in ((me, n[0]), (opp, n[1])):
                    for _ in range(min(rng.randint(0, 3), k)):
                        occ = [p for p in range(24) if side[p]]
                        side[rng.choice(occ)] -= 1
                        side[24] += 1
        st = _board(me, opp, rng.choice([WHITE, BLACK]))
        if st is not None and _sides(st) == (me, opp):
            return st
    raise RuntimeError("no board")


# Hand-built positions, one or more per rule: (me, opponent, expected
# ruled-out (wg, wbg, lg, lbg) for the player on roll, why).
HOME15 = _pts(p0=3, p1=3, p2=3, p3=3, p4=3)     # 15 on my points 1-5
CASES = [
    (_pts(p6=4), HOME15, (False, True, True, True),
     "race; I have borne off: I cannot lose a gammon; no opponent checker "
     "in my home board: no backgammon; my gammon stays possible"),
    (_pts(p0=15), _pts(p0=4), (True, True, False, True),
     "the opponent has borne off: it cannot lose a gammon; race, my "
     "rearmost is my 1-point: no backgammon, a gammon stays possible"),
    (_pts(p6=4), _pts(p0=14), (True, True, True, True),
     "both sides have borne off: neither can lose a gammon"),
    (_pts(p18=15), _pts(p0=1, p1=1), (True, True, False, False),
     "race; all my checkers in the opponent's home board (my 19-point): it "
     "can still gammon and backgammon me"),
    (_pts(p17=15), _pts(p0=1, p1=1), (True, True, False, True),
     "race; my rearmost is my 18-point, outside the opponent's home board: "
     "no backgammon, a gammon stays possible"),
    (_pts(p0=15), _pts(p20=1, p22=14), (False, False, False, True),
     "race; the opponent's checkers in my home board: I can gammon and "
     "backgammon it; mine are all on my 1-point: no backgammon against me, "
     "a gammon stays possible"),
    (_pts(p24=1, p0=14), _pts(p0=15), (False, False, False, False),
     "my checker on the bar: contact (rearmost 24), nobody has borne off: "
     "nothing is ruled out"),
    (_pts(p0=2, p1=2, p2=2, p3=2, p4=2, p5=2, p23=2),
     _pts(p12=3, p13=3, p14=3, p15=3, p24=1), (True, True, True, True),
     "contact (the bar, my back checkers), but both have borne off"),
    (HOME15, _pts(p0=3, p1=3, p2=3, p3=3, p5=3), (False, True, False, True),
     "bearoff, nobody off: no backgammons; both gammons stay possible"),
]


def _hand_built():
    """CASES as BoardStates, both turns."""
    states = []
    for me, opp, _, _ in CASES:
        for turn in (WHITE, BLACK):
            st = _board(me, opp, turn)
            assert st is not None and _sides(st) == (me, opp)
            states.append(st)
    return states


def _all_boards():
    rng = random.Random(7)
    boards = _playout_states(1500, seed=3)
    for kind, k in (("race", 3000), ("contact", 600), ("bar", 600)):
        boards += [_random_board(rng, kind) for _ in range(k)]
    boards += _hand_built()
    return boards


@pytest.fixture(scope="module")
def boards():
    return _all_boards()


# ── hand-built cases ─────────────────────────────────────────────────


@pytest.mark.parametrize("case", range(len(CASES)),
                         ids=[c[3][:40] for c in CASES])
def test_hand_built_case(case):
    me, opp, expected, why = CASES[case]
    assert _rules(me, opp) == expected, why
    for turn in (WHITE, BLACK):
        assert _mask_of(_board(me, opp, turn)) == expected, (turn, why)


# ── the implementation vs the rules ──────────────────────────────────


class TestRules:

    @pytest.mark.parametrize("impl", ["c", "numpy"])
    def test_random_probs(self, boards, impl, monkeypatch):
        if impl == "numpy":
            monkeypatch.setattr(prob_agent, "_BG_FAST_AVAILABLE", False)
        elif not prob_agent._BG_FAST_AVAILABLE:
            pytest.skip("bg_fast not built")
        rng = np.random.default_rng(0)
        probs = rng.random((len(boards), 5)).astype(np.float32)
        ours = _ours(boards, probs)
        for s, p, o in zip(boards, probs, ours):
            np.testing.assert_array_equal(o, _expected(*_sides(s), p))

    def test_c_encoder_features(self, boards):
        bg_fast = pytest.importorskip("bg_fast")
        rng = np.random.default_rng(1)
        probs = torch.from_numpy(rng.random((len(boards), 5)).astype(np.float32))
        x_py = torch.from_numpy(np.stack([ENC.encode(s) for s in boards]))
        x_c = torch.from_numpy(np.stack([bg_fast.encode_state(s) for s in boards]))
        assert torch.equal(prob5_sanity(probs, x_py), prob5_sanity(probs, x_c))

    def test_race_matches_is_race(self, boards):
        for s in boards:
            me, opp = _sides(s)
            assert (_side_counts(me)[1] + _side_counts(opp)[1] < 24) == is_race(s)

    def test_coverage(self, boards):
        """Every rule fires and stays silent somewhere; rule 2 at the
        18-point boundary from both sides."""
        hits = {k: set() for k in ("opp_off", "my_off", "wbg", "lbg")}
        edge_w, edge_l = set(), set()
        for s in boards:
            me, opp = _sides(s)
            m_on, m_rear = _side_counts(me)
            o_on, o_rear = _side_counts(opp)
            race = m_rear + o_rear < 24
            hits["opp_off"].add(o_on < 15)
            hits["my_off"].add(m_on < 15)
            if race and o_on == 15:
                hits["wbg"].add(o_rear < 18)
                edge_w.add(o_rear)
            if race and m_on == 15:
                hits["lbg"].add(m_rear < 18)
                edge_l.add(m_rear)
        for k, v in hits.items():
            assert v == {True, False}, k
        assert {17, 18} <= edge_w and {17, 18} <= edge_l
        assert any(s.bar[0] or s.bar[1] for s in boards)
        assert {s.turn for s in boards} == {WHITE, BLACK}

    def test_untouched_when_nothing_ruled_out(self, boards):
        rng = np.random.default_rng(2)
        probs = torch.from_numpy(rng.random((len(boards), 5)).astype(np.float32))
        ours = _ours(boards, probs)
        plain = prob5_postprocess(probs).numpy()
        for s, o, p in zip(boards, ours, plain):
            if not any(_rules(*_sides(s))):
                np.testing.assert_array_equal(o, p)


# ── the C implementation vs prob_agent ───────────────────────────────

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_C_HARNESS = r"""
#include "nn_eval.c"
int prob5_impossible_mask(const float *x, int n, unsigned char *mask);  /* bg_engine.c */
void h_engine_mask(const float *x, int n, unsigned char *mask) { prob5_impossible_mask(x, n, mask); }
void h_nn_mask(const float *x, int n, unsigned char *mask) {
    for (int r = 0; r < n; r++) {
        float p[5] = {1, 1, 1, 1, 1};
        prob5_sanity_zero(x + 196 * r, p);
        for (int j = 0; j < 4; j++) mask[4 * r + j] = p[j + 1] == 0.0f;
    }
}
void h_nn_reduce(const float *x, const float *raw, int n, float *out) {
    for (int r = 0; r < n; r++) prob5_reduce(x + 196 * r, raw + 5 * r, 1, out + 5 * r);
}
"""


def _edge_boards():
    """Point-by-point sweeps of the feature decoding: n = 1..15 checkers of
    one side on one point or the bar (the rest off) against 15 checkers on
    the other side's 1-point, both sides and turns."""
    out = []
    for p in range(25):
        for n in range(1, 16):
            one = _pts(**{f"p{p}": n})
            for me, opp in ((one, _pts(p0=15)), (_pts(p0=15), one)):
                for turn in (WHITE, BLACK):
                    st = _board(me, opp, turn)
                    if st is not None:
                        out.append(st)
    return out


@pytest.fixture(scope="module")
def sanity_lib(tmp_path_factory):
    """_C_HARNESS + bg_engine.c, built from source."""
    import ctypes
    d = tmp_path_factory.mktemp("sanity_c")
    src = d / "h.c"
    src.write_text(_C_HARNESS)
    so = d / "h.so"
    r = subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(src),
         os.path.join(_REPO, "c_engine", "bg_engine.c"),
         "-I", os.path.join(_REPO, "c_inference"), "-lm"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lib = ctypes.CDLL(str(so))
    vp, ci = ctypes.c_void_p, ctypes.c_int
    for fn in (lib.h_engine_mask, lib.h_nn_mask):
        fn.argtypes, fn.restype = [vp, ci, vp], None
    lib.h_nn_reduce.argtypes, lib.h_nn_reduce.restype = [vp, vp, ci, vp], None
    return lib


@pytest.fixture(scope="module")
def sanity_x(boards):
    return np.ascontiguousarray(
        np.stack([ENC.encode(s) for s in boards + _edge_boards()]), dtype=np.float32)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
class TestCParity:
    """prob5_sanity.h through bg_engine.c prob5_impossible_mask and
    nn_eval.c prob5_sanity_zero / prob5_reduce (built from source) against
    prob_agent: identical masks, and prob5_reduce(sanity) bitwise
    prob5_sanity."""

    @staticmethod
    def _mask(fn, x):
        m = np.zeros((len(x), 4), dtype=np.uint8)
        fn(x.ctypes.data, len(x), m.ctypes.data)
        return m.astype(bool)

    def test_masks_equal(self, sanity_lib, sanity_x):
        lib, x = sanity_lib, sanity_x
        py = prob_agent._prob5_impossible_np(x)
        py = np.zeros((len(x), 4), bool) if py is None else py
        eng = self._mask(lib.h_engine_mask, x)
        nn = self._mask(lib.h_nn_mask, x)
        np.testing.assert_array_equal(eng, py)
        np.testing.assert_array_equal(nn, py)
        assert py.any(axis=0).all() and (~py).any(axis=0).all()

    def test_reduce_equals_prob5_sanity(self, sanity_lib, sanity_x):
        lib, x = sanity_lib, sanity_x
        rng = np.random.default_rng(5)
        raw = rng.random((len(x), 5)).astype(np.float32)
        raw[::7] = 1.0
        raw[1::7] = 0.0
        raw[2::7, 1:] = 1e-7
        out = np.empty_like(raw)
        lib.h_nn_reduce(x.ctypes.data, raw.ctypes.data, len(x), out.ctypes.data)
        py = prob5_sanity(torch.from_numpy(raw), torch.from_numpy(x)).numpy()
        np.testing.assert_array_equal(out, py)


# ── agents ───────────────────────────────────────────────────────────


def _barely_trained_prob5(seed):
    from model import ProbNetwork
    torch.manual_seed(seed)
    net = ProbNetwork(hidden_sizes=[8], encoder_name="perspective196")
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.5)
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    x, y = torch.rand(64, net.input_size), torch.rand(64, 5)
    for _ in range(5):
        opt.zero_grad()
        ((net(x) - y) ** 2).mean().backward()
        opt.step()
    return net.eval()


def _decode(x):
    """(me, opponent) sides from one perspective196 row."""
    sides = []
    for base, bar, mirrored in ((0, 96, False), (98, 194, True)):
        side = [0] * 25
        for k in range(24):
            f = x[base + 4 * k: base + 4 * k + 4]
            side[23 - k if mirrored else k] = int(round(f[0] + f[1] + f[2] + 2 * f[3]))
        side[24] = int(round(2 * x[bar]))
        sides.append(side)
    return sides


class _Spy:
    """Counts the rows the net evaluates and records every _probs output."""

    def __init__(self, agent, monkeypatch):
        self.net_rows = 0
        self.rows = []
        agent.network.register_forward_hook(self._hook)
        orig = agent._probs

        def probs(x):
            p = orig(x)
            self.rows.append((x.detach().cpu().numpy(),
                              p.detach().cpu().numpy()))
            return p
        monkeypatch.setattr(agent, "_probs", probs)

    def _hook(self, _m, inp, _out):
        self.net_rows += inp[0].shape[0]

    def check(self):
        n = 0
        for xs, ps in self.rows:
            for x, p in zip(xs, ps):
                np.testing.assert_array_equal(p, _expected(*_decode(x), p))
                n += 1
        assert n == self.net_rows > 0
        return n


def _game_positions(n, seed):
    """Positions with dice, biased to races and bear-offs."""
    states = _playout_states(max(4 * n, 600), seed)
    rng = random.Random(seed)
    late = [s for s in states if is_race(s) or s.off[0] or s.off[1]]
    pick = rng.sample(late, n // 2) + rng.sample(states, n // 2)
    return [(s, (rng.randint(1, 6), rng.randint(1, 6))) for s in pick]


def _drive_prob_agent(agent, plies):
    for s, dice in _game_positions(3 if plies else 200, seed=11):
        plays = get_legal_plays(s, dice)
        if plays:
            agent.choose_checker_action(s, dice, plays)
            agent.choose_next_state(s, dice)
    agent.evaluate_batch([s for s, _ in _game_positions(50, seed=12)])
    agent.evaluate(_hand_built()[0])


@pytest.mark.parametrize("plies", [0, 1])
def test_prob_agent_opt_in_obeys_rules(plies, monkeypatch):
    agent = ProbAgent(_barely_trained_prob5(1), plies=plies, board_sanity=True)
    spy = _Spy(agent, monkeypatch)
    _drive_prob_agent(agent, plies)
    assert spy.check() > 500


@pytest.mark.parametrize("plies", [0, 1])
@pytest.mark.parametrize("fast", [True, False])
def test_prob_agent_default_is_clamp_only(plies, fast, monkeypatch):
    """Plain prob5 by default: every _probs output is exactly
    prob5_postprocess of the net, on a sample where prob5_sanity would
    change many rows."""
    agent = ProbAgent(_barely_trained_prob5(1), plies=plies,
                      use_fast_engine=fast)
    assert agent.board_sanity is False
    spy = _Spy(agent, monkeypatch)
    _drive_prob_agent(agent, plies)
    net_rows = spy.net_rows
    n = changed = 0
    for xs, ps in spy.rows:
        x = torch.from_numpy(xs)
        with torch.no_grad():
            raw = agent.network(x)
        assert torch.equal(torch.from_numpy(ps), prob5_postprocess(raw))
        changed += int((prob5_sanity(raw, x).numpy() != ps).any(axis=1).sum())
        n += len(xs)
    assert n == net_rows > 500 and changed > 20


def test_prob_agent_default_decisions_match_clamp_only():
    """Default ProbAgent decisions equal those of an agent whose _probs
    is the clamp-only form; the opt-in agent's differ somewhere."""
    net = _barely_trained_prob5(4)
    plain, ref, opt = (ProbAgent(net), ProbAgent(net),
                       ProbAgent(net, board_sanity=True))

    def clamp_only(x):
        with torch.no_grad():
            return prob5_postprocess(ref.network(x))
    ref._probs = clamp_only
    differ = 0
    for s, dice in _game_positions(150, seed=21):
        a, b = plain.choose_next_state(s, dice), ref.choose_next_state(s, dice)
        c = opt.choose_next_state(s, dice)
        key = lambda st: None if st is None else (tuple(st.points),
                                                  tuple(st.bar), tuple(st.off))
        assert key(a) == key(b)
        differ += key(a) != key(c)
    assert differ > 0


def test_board_sanity_settings():
    from janowski_agent import JanowskiAgent
    net = _barely_trained_prob5(5)
    assert JanowskiAgent(net).board_sanity is True
    assert JanowskiAgent(net, board_sanity=False).board_sanity is False
    assert ProbAgent(net, board_sanity=True).board_sanity is True
    bs = prob_agent.board_sanity_from_env
    assert bs("V", False, {}) is False and bs("V", True, {}) is True
    assert bs("V", True, {"V": "0"}) is False
    assert bs("V", False, {"V": "1"}) is True
    with pytest.raises(ValueError, match="V must be 0 or 1"):
        bs("V", False, {"V": "yes"})


@pytest.mark.parametrize("oneply", [False, True])
def test_janowski_agent_obeys_rules(oneply, monkeypatch):
    from janowski_agent import JanowskiAgent
    from modes import MatchState
    agent = JanowskiAgent(_barely_trained_prob5(2), oneply=oneply)
    spy = _Spy(agent, monkeypatch)
    ms = MatchState()
    for s, dice in _game_positions(4 if oneply else 120, seed=13):
        if get_legal_plays(s, dice):
            agent.choose_checker_action_cubeful(s, dice, ms)
        agent.offer_double(s, ms)
        agent.respond_to_double(s, ms)
    assert spy.check() > 300


def test_zeroing_moves_values():
    """The rules reach the opt-in agent's values: a race where the net's
    gammon outputs are impossible scores as plain prob5 would not."""
    agent = ProbAgent(_barely_trained_prob5(3), board_sanity=True)
    st = _hand_built()[0]
    x = agent._encode_batch([st])
    with torch.no_grad():
        raw = agent.network(x)
    plain = float((prob5_postprocess(raw) * torch.tensor(
        [2.0, 1, 1, -1, -1])).sum() - 1)
    assert agent.evaluate(st) != pytest.approx(plain)
    assert ProbAgent(agent.network).evaluate(st) == pytest.approx(plain)
