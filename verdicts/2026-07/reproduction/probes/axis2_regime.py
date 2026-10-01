"""AXIS 2 — regime dependence probe for tw_fool_month_10pct.

Claim: SPY, indicator=momentum, window=21, rule=long_above, threshold=0.10,
horizon=252, direction=1, n_trials=20. Bench (repaired engine, ~11y): net Sharpe
+1.19 vs B&H +0.73, DSR 0.98, OOS fold Sharpe +1.32, TRADABLE.

Variants:
  (0) reproduce the bench baseline on the exact ~11y window (lookback 252*16 DAYS bug -> ~11y)
  (a) bench window minus COVID trigger cluster (drop entries Mar-2020..Aug-2020)
  (b) pre-2020 only (bench-window start..2019-12-31, and 2000..2019 extended)
  (c) 2022-01-01 onward
  (d) extended lookback: 2000-01-01..now, incl. per-episode forward returns (2008-09)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np


from snapshot_source import SnapshotSource as YahooSource
from snapshot_source import now_epoch
from bench_engine import (
    EdgeSpec, INDICATORS, RULES, run_edge_test, evaluate_positions, score_positions,
)

SPEC_KW = dict(indicator="momentum", window=21, rule="long_above", threshold=0.10,
               horizon=252, direction=1, n_trials=20)
COST = 5e-4


def fetch_spy_full():
    src = YahooSource()
    obs = src.fetch(["SPY.close"], start=0.0, end=now_epoch())
    pts = sorted((o.t_event, o.value) for o in obs if o.key == "SPY.close")
    ts = np.array([t for t, _ in pts], dtype=float)
    px = np.array([v for _, v in pts], dtype=float)
    ok = np.isfinite(px)
    return ts[ok], px[ok]


def d(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def epoch(y, m, day):
    return datetime(y, m, day, tzinfo=timezone.utc).timestamp()


def spec(name):
    return EdgeSpec(name=name, asset="SPY", source="motley fool 2026-05-05", **SPEC_KW)


def report(tag, v, n_triggers, first, last):
    print(f"\n--- {tag} ---")
    print(f"  bars={v.n_samples}  window {first}..{last}  raw-trigger-days={n_triggers}")
    print(f"  net Sharpe {v.net_sharpe:+.3f} | B&H {v.buy_hold_sharpe:+.3f} | "
          f"timing {v.timing_sharpe:+.3f} | DSR {v.dsr:.3f} | disc {v.discounted_sharpe:+.3f}")
    print(f"  OOS mean {v.oos_mean_sharpe:+.3f} CI [{v.oos_ci_lo:+.3f},{v.oos_ci_hi:+.3f}] "
          f"excl0={v.oos_excludes_zero} | p={v.p_value:.3f}")
    print(f"  beats B&H={v.beats_buy_hold}  dsr_ok={v.dsr_ok}  PASSED={v.passed}")


def triggers(px):
    ind = INDICATORS["momentum"](px, 21)
    raw = np.where(np.isfinite(ind) & (ind > 0.10), 1.0, 0.0)
    return raw


def episodes(ts, raw, gap_days=42):
    """Cluster raw trigger days into episodes (gap > gap_days trading... calendar days)."""
    idx = np.where(raw > 0)[0]
    if idx.size == 0:
        return []
    eps = [[idx[0]]]
    for i in idx[1:]:
        if ts[i] - ts[eps[-1][-1]] > gap_days * 86400:
            eps.append([i])
        else:
            eps[-1].append(i)
    return eps


def run_window(ts, px, t0, t1, tag):
    m = (ts >= t0) & (ts <= t1)
    tw, pw = ts[m], px[m]
    if pw.size < 300:
        print(f"\n--- {tag} --- too few bars ({pw.size})")
        return None
    v = run_edge_test(spec(tag), pw, cost_per_turnover=COST)
    raw = triggers(pw)
    report(tag, v, int(raw.sum()), d(tw[0]), d(tw[-1]))
    return v, tw, pw, raw


def main():
    ts, px = fetch_spy_full()
    print(f"SPY full history: {px.size} bars, {d(ts[0])} .. {d(ts[-1])}")

    end = now_epoch()

    # (0) baseline: the bench's exact window — lookback_days = 252*16 calendar days
    t0 = end - 252 * 16 * 86400.0
    base = run_window(ts, px, t0, end, "(0) BASELINE bench window (lookback bug ~11y)")
    if base is None:
        return
    v0, tw, pw, raw = base

    # trigger episodes inside the bench window
    print("\nTrigger episodes in the bench window (gap>42cal-days splits):")
    for ep in episodes(tw, raw):
        i0, i1 = ep[0], ep[-1]
        j = min(i0 + 252, pw.size - 1)
        fwd = pw[j] / pw[i0] - 1.0
        full = "" if (i0 + 252) < pw.size else " (truncated fwd window)"
        print(f"  {d(tw[i0])} .. {d(tw[i1])}  ({len(ep)} trigger days)  "
              f"fwd-252b from first trigger: {fwd:+.1%}{full}")

    # (a) bench window MINUS entries originating Mar-2020..Aug-2020
    lo, hi = epoch(2020, 3, 1), epoch(2020, 9, 1)
    ind = INDICATORS["momentum"](pw, 21)
    raw_masked = np.where(np.isfinite(ind) & (ind > 0.10), 1.0, 0.0)
    covid_mask = (tw >= lo) & (tw < hi)
    n_dropped = int(raw_masked[covid_mask].sum())
    raw_masked[covid_mask] = 0.0
    # horizon forward-fill (same loop as evaluate_positions)
    held, cur = 0, 0.0
    tgt = np.zeros_like(raw_masked)
    for i in range(raw_masked.size):
        if raw_masked[i] != 0.0:
            cur, held = raw_masked[i], 252
        if held > 0:
            tgt[i], held = cur, held - 1
    pos = tgt[:-1]
    ret = pw[1:] / pw[:-1] - 1.0
    va = score_positions(pos, ret, name="(a) minus COVID cluster", n_trials=20,
                         cost_per_turnover=COST)
    print(f"\n[dropped {n_dropped} trigger days in Mar..Aug-2020]")
    report("(a) bench window MINUS Mar-Aug-2020 entries", va, int(raw_masked.sum()),
           d(tw[0]), d(tw[-1]))
    frac_long = float(np.mean(pos != 0))
    print(f"  time-in-market: {frac_long:.1%} (baseline "
          f"{float(np.mean(evaluate_positions(spec('x'), pw)[:-1] != 0)):.1%})")

    # (b) pre-2020
    run_window(ts, px, t0, epoch(2019, 12, 31), "(b1) bench-window start .. 2019-12-31")
    run_window(ts, px, epoch(2000, 1, 1), epoch(2019, 12, 31), "(b2) 2000-01-01 .. 2019-12-31")

    # (c) 2022 onward
    run_window(ts, px, epoch(2022, 1, 1), end, "(c) 2022-01-01 .. now")

    # (d) extended lookback 2000..now
    ext = run_window(ts, px, epoch(2000, 1, 1), end, "(d) 2000-01-01 .. now (extended)")
    if ext is not None:
        _, tw2, pw2, raw2 = ext
        print("\nAll trigger episodes 2000..now, fwd-252-bar return from first trigger day:")
        wins = 0
        n_full = 0
        fwds = []
        for ep in episodes(tw2, raw2):
            i0, i1 = ep[0], ep[-1]
            if i0 + 252 < pw2.size:
                fwd = pw2[i0 + 252] / pw2[i0] - 1.0
                n_full += 1
                wins += fwd > 0
                fwds.append(fwd)
                trunc = ""
            else:
                fwd = pw2[-1] / pw2[i0] - 1.0
                trunc = " (TRUNCATED)"
            print(f"  {d(tw2[i0])} .. {d(tw2[i1])}  ({len(ep)}d)  fwd: {fwd:+.1%}{trunc}")
        if n_full:
            print(f"  full-window episodes: {n_full}, win rate {wins}/{n_full}, "
                  f"mean fwd {np.mean(fwds):+.1%}, min {np.min(fwds):+.1%}")

        # (d2) 2000..2012 window — the 2008-09 stress test isolated
        run_window(ts, px, epoch(2000, 1, 1), epoch(2012, 12, 31), "(d2) 2000-01-01 .. 2012-12-31")


if __name__ == "__main__":
    main()
