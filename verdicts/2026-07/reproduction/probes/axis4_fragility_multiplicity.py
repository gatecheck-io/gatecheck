"""AXIS 4 kill-panel probe — parameter fragility & honest multiplicity for tw_fool_month_10pct.

(a) Neighbor grid: window {19,21,23} x threshold {0.08..0.12} x horizon {189,252,315},
    benched IDENTICALLY to the survivor (same price series, same bench_posted_claim call,
    n_trials=20, cost 5e-4, seed 0). Report pass fraction + survivor rank.
(b) n_trials sweep on the survivor's exact config: at what declared search breadth does
    DSR fall below the 0.95 gate?

Run through reproduction/run_probes.py. Prices are cached in a fresh run directory.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


from posted_claims import PostedClaim, bench_posted_claim
from bench_engine import EdgeSpec, run_edge_test

SCRATCH = os.environ["GATECHECK_RUN_DIR"]
CACHE = os.path.join(SCRATCH, "spy_prices_axis4.npz")

LOOKBACK_DAYS = 252 * 16  # the bench default ('16y' config -> ~11y calendar, known lookback bug)


def get_prices() -> np.ndarray:
    if os.path.exists(CACHE):
        return np.load(CACHE)["p"]
    from snapshot_source import now_epoch
    from snapshot_source import SnapshotSource as YahooSource
    src = YahooSource()
    end = now_epoch()
    start = end - LOOKBACK_DAYS * 86400.0
    obs = src.fetch(["SPY.close"], start, end)
    pts = sorted((o.t_event, o.value) for o in obs if o.key == "SPY.close")
    p = np.array([v for _, v in pts], dtype=float)
    np.savez(CACHE, p=p)
    return p


def make_claim(window: int, threshold: float, horizon: int, n_trials: int = 20) -> PostedClaim:
    return PostedClaim(
        claim_id=f"w{window}_t{threshold:.2f}_h{horizon}",
        text="grid variant of tw_fool_month_10pct",
        source="Motley Fool 2026-05-05",
        asset="SPY", indicator="momentum", window=window, rule="long_above",
        threshold=threshold, horizon=horizon, direction=1, n_trials=n_trials,
    )


def main() -> None:
    p = get_prices()
    print(f"prices: n={p.size} bars, first={p[0]:.2f}, last={p[-1]:.2f}")

    # ---- baseline reproduction of the survivor -------------------------------------
    base = bench_posted_claim(make_claim(21, 0.10, 252), p)
    print("\n[BASELINE reproduction of the survivor]")
    print(f"  net Sharpe {base.net_sharpe:+.3f}  b&h {base.buy_hold_sharpe:+.3f}  "
          f"DSR {base.dsr:.3f}  OOS {base.oos_mean_sharpe:+.3f}  verdict {base.verdict}")

    # ---- (a) neighbor grid -----------------------------------------------------------
    windows = [19, 21, 23]
    thresholds = [0.08, 0.09, 0.10, 0.11, 0.12]
    horizons = [189, 252, 315]
    rows = []
    for w in windows:
        for t in thresholds:
            for h in horizons:
                v = bench_posted_claim(make_claim(w, t, h), p)
                rows.append(dict(window=w, threshold=t, horizon=h,
                                 net_sharpe=v.net_sharpe, bh=v.buy_hold_sharpe,
                                 dsr=v.dsr, discounted=v.discounted_sharpe,
                                 oos=v.oos_mean_sharpe, passed=bool(v.passed),
                                 verdict=v.verdict))
                print(f"  w={w} thr={t:.2f} h={h}: net {v.net_sharpe:+.3f} "
                      f"DSR {v.dsr:.3f} OOS {v.oos_mean_sharpe:+.3f} -> {v.verdict}")

    n_pass = sum(r["passed"] for r in rows)
    print(f"\n[GRID] {n_pass}/{len(rows)} neighbors PASS the identical gate "
          f"({n_pass/len(rows):.1%})")

    # survivor rank within the grid (by net Sharpe, and by discounted Sharpe)
    surv = next(r for r in rows if r["window"] == 21 and abs(r["threshold"] - 0.10) < 1e-9
                and r["horizon"] == 252)
    by_net = sorted(rows, key=lambda r: -r["net_sharpe"])
    by_disc = sorted(rows, key=lambda r: -r["discounted"])
    rank_net = 1 + by_net.index(surv)
    rank_disc = 1 + by_disc.index(surv)
    print(f"[GRID] survivor rank: {rank_net}/{len(rows)} by net Sharpe, "
          f"{rank_disc}/{len(rows)} by discounted Sharpe")
    print(f"[GRID] net-Sharpe range across grid: "
          f"[{min(r['net_sharpe'] for r in rows):+.3f}, {max(r['net_sharpe'] for r in rows):+.3f}]")

    # which gate leg fails for the failing neighbors?
    fail_beats = sum(1 for r in rows if not r["passed"] and r["net_sharpe"] <= r["bh"])
    fail_dsr = sum(1 for r in rows if not r["passed"] and r["dsr"] < 0.95)
    fail_oos = sum(1 for r in rows if not r["passed"] and r["oos"] <= 0)
    print(f"[GRID] failing-neighbor autopsy: {len(rows)-n_pass} fail; "
          f"of those {fail_beats} lose to b&h, {fail_dsr} fail DSR<0.95, {fail_oos} OOS<=0")

    # ---- (b) honest n_trials sweep on the survivor's exact config --------------------
    print("\n[N_TRIALS sweep on the exact survivor config w=21 thr=0.10 h=252]")
    sweep = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000]
    ntr_rows = []
    crossing = None
    for n in sweep:
        v = bench_posted_claim(make_claim(21, 0.10, 252, n_trials=n), p)
        ntr_rows.append(dict(n_trials=n, dsr=v.dsr, passed=bool(v.passed)))
        print(f"  n_trials={n:>5}: DSR {v.dsr:.4f}  passed={v.passed}")

    # bisect the exact crossing where DSR drops below 0.95
    lo, hi = 1, None
    for r in ntr_rows:
        if r["dsr"] >= 0.95:
            lo = max(lo, r["n_trials"])
        elif hi is None or r["n_trials"] < hi:
            hi = r["n_trials"]
    if hi is not None:
        a, b = lo, hi
        while b - a > 1:
            m = (a + b) // 2
            v = bench_posted_claim(make_claim(21, 0.10, 252, n_trials=m), p)
            if v.dsr >= 0.95:
                a = m
            else:
                b = m
        crossing = b
        print(f"  -> DSR first drops below 0.95 at n_trials = {crossing} "
              f"(last passing n_trials = {a})")
    else:
        print("  -> DSR never dropped below 0.95 in the sweep range")

    out = dict(baseline=dict(net=base.net_sharpe, bh=base.buy_hold_sharpe, dsr=base.dsr,
                             oos=base.oos_mean_sharpe, verdict=base.verdict),
               grid=rows, n_pass=n_pass, rank_net=rank_net, rank_disc=rank_disc,
               ntrials_sweep=ntr_rows, dsr_crossing=crossing)
    with open(os.path.join(SCRATCH, "axis4_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nresults written to axis4_results.json")


if __name__ == "__main__":
    main()
