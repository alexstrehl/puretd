#!/usr/bin/env python3
"""Full-pool agreement of Open Sage's C API (bgsage_mobile_analyze, the
opensage-native agent) with its Python BgBotAnalyzer (opensage-analyzer) on
cubeful money decisions: every checker and cube item of the committed cubeful
pool, at 0-ply and 1-ply (sage_plies 1, 2). Checker: the first legal move of
each ranking; cube: (double, take). Items the C API does not answer within its
1 s budget are counted, not compared. The budget is wall time, so give each
worker its own physical core: N at most the physical core count minus one.

  python experiments/benchmarks/sage_capi_vs_analyzer.py --workers N --cores 1-N \
      [--part I/K] > out.txt     # --part: this host's share of the chunks

Needs bgsage 2.0.20260907 and libbgsage_mobile.so (BGSAGE_MOBILE_LIB)."""
import argparse
import json
import multiprocessing as mp
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "..")]

_W = {}


def _init(counter=None, cores=()):
    """Per worker: pin to its own core from `cores` (the C API's 1 s budget is
    wall time, so oversubscribed workers would time out), load engines."""
    if cores:
        with counter.get_lock():
            k = counter.value
            counter.value += 1
        os.sched_setaffinity(0, {cores[k % len(cores)]})
    os.environ["OMP_NUM_THREADS"] = "1"
    import fixedpool_bench as fb
    from third_party_opponents.opensage.opensage_agents import OpenSageAgent, OpenSageAnalyzerAgent
    from third_party_opponents.opensage.opensage_mobile import OpenSageMobileAgent
    _W["ck"], _W["cu"] = fb.load_cubeful_pool()
    _W["nat"] = {p: OpenSageMobileAgent(sage_plies=p + 1) for p in (0, 1)}
    _W["an"] = {p: OpenSageAnalyzerAgent(sage_plies=p + 1) for p in (0, 1)}
    _W["board"] = OpenSageAgent._to_sage_board


def _first_legal(moves, legal):
    for b in moves:
        if tuple(b) in legal:
            return tuple(b)
    return None


def _task(t):
    """(kind, plies, lo, hi) -> (kind, plies, n, agree, n_over_budget, differing indices, over-budget indices)."""
    from backgammon_engine import get_legal_plays
    kind, plies, lo, hi = t
    nat, an = _W["nat"][plies], _W["an"][plies]
    agree = over = 0
    bad, overs = [], []
    if kind == "checker":
        for j, (s, d, ms) in enumerate(_W["ck"][lo:hi]):
            plays = get_legal_plays(s, d)
            if not plays:
                agree += 1
                continue
            legal = {tuple(_W["board"](ns, s.turn)) for _p, ns in plays}
            r = nat._eng.analyze(0, die1=d[0], die2=d[1], **nat._kw(s, ms))
            if r is None:
                over += 1
                overs.append(lo + j)
                continue
            a = _first_legal([m["board"] for m in r["moves"]], legal)
            ns, _ = an.choose_checker_action_cubeful(s, d, ms)
            b = tuple(_W["board"](ns, s.turn))
            agree += a == b
            if a != b:
                bad.append(lo + j)
    else:
        for j, (s, ms) in enumerate(_W["cu"][lo:hi]):
            r = nat._eng.analyze(1, **nat._kw(s, ms))
            if r is None:
                over += 1
                overs.append(lo + j)
                continue
            x = an._cube_action(s, ms)
            same = (bool(r["should_double"]), bool(r["should_take"])) == \
                   (bool(x.should_double), bool(x.should_take))
            agree += same
            if not same:
                bad.append(lo + j)
    return kind, plies, hi - lo, agree, over, bad, overs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32,
                    help="worker processes (default 32); give each its own physical core with --cores")
    ap.add_argument("--chunk", type=int, default=50, help="pool items per task (default 50)")
    ap.add_argument("--only", default="", help='e.g. "1-ply checker": run only that row')
    ap.add_argument("--cores", default="", help="comma list / a-b ranges: one pinned core per worker")
    ap.add_argument("--part", default="1/1", help="I/N: run every N-th chunk starting at I")
    a = ap.parse_args()
    i, n = map(int, a.part.split("/"))
    import fixedpool_bench as fb
    ck, cu = fb.load_cubeful_pool()
    tasks = [(k, p, lo, min(lo + a.chunk, size)) for p in (0, 1)
             for k, size in (("checker", len(ck)), ("cube", len(cu)))
             for lo in range(0, size, a.chunk)
             if not a.only or a.only == f"{p}-ply {k}"]
    tasks = tasks[i - 1::n]
    tot = {}
    cores = [c for part in a.cores.split(",") if part
             for c in (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1)
                       if "-" in part else [int(part)])]
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_init, initargs=(ctx.Value("i", 0), cores)) as pool:
        for kind, plies, m, ag, ov, bad, ovs in pool.imap_unordered(_task, tasks):
            t = tot.setdefault(f"{plies}-ply {kind}", [0, 0, 0, [], []])
            t[0] += m; t[1] += ag; t[2] += ov; t[3] += bad; t[4] += ovs
    print("Open Sage C API vs BgBotAnalyzer, cubeful money, fixed pool"
          f" (part {a.part})")
    for key in sorted(tot):
        m, ag, ov, bad, ovs = tot[key]
        print(f"{key:<15} items {m:>6}  agree {ag:>6}  over 1 s budget {ov:>3}  differ {len(bad):>3}"
              + (f"  (items {sorted(bad)[:20]})" if bad else "")
              + (f"  over-budget items {sorted(ovs)}" if ovs else ""))
    print("JSON " + json.dumps({k: v[:3] + [sorted(v[3]), sorted(v[4])] for k, v in tot.items()}))


if __name__ == "__main__":
    main()
