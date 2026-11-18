#!/usr/bin/env python3
"""Full-pool agreement of the Open Sage fixed-pool loops (SageBench) with the
paths they replace, at 0-ply and 1-ply: cubeful checker and cube against the
C API (bgsage_mobile_analyze), cubeless checker against best_move_index with
the bearoff database. Every item of both committed pools; items the C API
does not answer within its 1 s budget are counted, not compared. The budget
is wall time, so give each worker its own physical core: N at most the
physical core count minus one.

  python experiments/benchmarks/sage_pick_check.py --workers N --cores 1-N > out.txt

Needs bgsage 2.0.20260907, libsage_bench.so (SAGE_BENCH_LIB) and
libbgsage_mobile.so (BGSAGE_MOBILE_LIB) built by the same compiler."""
import argparse
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
    import bgbot_cpp
    from bgsage.weights import bearoff_db_path
    from third_party_opponents.opensage.opensage_agents import sage_multipy
    from third_party_opponents.opensage.opensage_mobile import SageMobile, OpenSageMobileAgent
    _W["fb"], _W["sb"], _W["eng"] = fb, fb.SageBench(), SageMobile()
    _W["ag"] = {}
    for plies in (0, 1):
        ag = OpenSageMobileAgent.__new__(OpenSageMobileAgent)
        ag.sage_plies, ag.beaver = plies + 1, False
        _W["ag"][plies] = ag
    _W["ck"], _W["cu"] = fb.load_cubeful_pool()
    from bench_common import load_fixedpool
    _W["cl"] = load_fixedpool()
    db = bgbot_cpp.BearoffDB()
    assert db.load(str(bearoff_db_path()))
    _W["st"] = {}
    for plies in (0, 1):
        st = sage_multipy(plies + 1)
        bgbot_cpp.multipy_set_bearoff_db(st, db)   # as BgBotAnalyzer and SageBench do
        _W["st"][plies] = (st, db)


def _task(t):
    """(kind, plies, lo, hi) -> (kind, plies, n, agree, n_over_budget, differing indices, over-budget indices)."""
    kind, plies, lo, hi = t
    fb, sb, eng, ag = _W["fb"], _W["sb"], _W["eng"], _W["ag"][plies]
    agree = over = 0
    bad, overs = [], []
    if kind == "checker":
        pts = _W["ck"][lo:hi]
        for j, (s, d, ms) in enumerate(pts):
            _, pick = sb.bench(0, fb.sage_items([(s, d, ms)]), plies)
            r = eng.analyze(0, die1=d[0], die2=d[1], **ag._kw(s, ms))
            if r is None:
                over += 1
                overs.append(lo + j)
            elif pick[0] == tuple(r["moves"][0]["board"]):
                agree += 1
            else:
                bad.append(lo + j)
    elif kind == "cube":
        pts = _W["cu"][lo:hi]
        for j, (s, ms) in enumerate(pts):
            _, cd = sb.bench(1, fb.sage_items([(s, ms)]), plies)
            r = eng.analyze(1, **ag._kw(s, ms))
            if r is None:
                over += 1
                overs.append(lo + j)
            elif cd[0] == (r["should_double"], r["should_take"]):
                agree += 1
            else:
                bad.append(lo + j)
    else:
        from bgsage.board import possible_moves
        st, _ = _W["st"][plies]
        items = fb.sage_items(_W["cl"][lo:hi])
        _, picks = sb.bench(2, items, plies)
        for j, ((b, d, _), pick) in enumerate(zip(items, picks)):
            c = possible_moves(b, d[0], d[1])
            if pick == tuple(c[st.best_move_index(c, b)]):
                agree += 1
            else:
                bad.append(lo + j)
    return kind, plies, hi - lo, agree, over, bad, overs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32,
                    help="worker processes (default 32); give each its own physical core with --cores")
    ap.add_argument("--chunk", type=int, default=50, help="pool items per task (default 50)")
    ap.add_argument("--only", default="", help='e.g. "1-ply checker": run only that row')
    ap.add_argument("--cores", default="", help="comma list / a-b ranges: one pinned core per worker")
    a = ap.parse_args()
    _init()
    sizes = {"checker": len(_W["ck"]), "cube": len(_W["cu"]), "cubeless": len(_W["cl"])}
    tasks = [(k, p, lo, min(lo + a.chunk, n)) for p in (0, 1)
             for k, n in sizes.items() for lo in range(0, n, a.chunk)
             if not a.only or a.only == f"{p}-ply {k}"]
    tot = {}
    cores = [c for part in a.cores.split(",") if part
             for c in (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1)
                       if "-" in part else [int(part)])]
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_init, initargs=(ctx.Value("i", 0), cores)) as pool:
        for kind, plies, n, ag, ov, bad, ovs in pool.imap_unordered(_task, tasks):
            t = tot.setdefault((kind, plies), [0, 0, 0, [], []])
            t[0] += n; t[1] += ag; t[2] += ov; t[3] += bad; t[4] += ovs
    ref = {"checker": "C API", "cube": "C API", "cubeless": "best_move_index + bearoff DB"}
    print("Open Sage fixed-pool loops vs the paths they replace, every pool item")
    for (kind, plies), (n, ag, ov, bad, ovs) in sorted(tot.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"{plies}-ply {kind:<8} vs {ref[kind]:<29} items {n:>6}  agree {ag:>6}  "
              f"over 1 s budget {ov:>3}  differ {len(bad):>3}"
              + (f"  (items {sorted(bad)[:20]})" if bad else "")
              + (f"  over-budget items {sorted(ovs)}" if ovs else ""))


if __name__ == "__main__":
    main()
