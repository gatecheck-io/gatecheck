"""AXIS 3 follow-up — leave-one-episode-out robustness of the exposure-matched null result.

Only 6 in-market episodes exist. For each episode k: drop block k from the strategy, rebuild the
exposure-matched null with the remaining 5 blocks, and ask whether the reduced strategy still
beats its null. Also map episode start/end dates for interpretability.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np


from posted_claims import PostedClaim
from bench_engine import evaluate_positions, _sharpe
from snapshot_source import now_epoch
from snapshot_source import SnapshotSource as YahooSource

RNG = np.random.default_rng(7)
COST = 5e-4
PPY = 252

src = YahooSource()
end = now_epoch()
start = end - 252 * 16 * 86400.0
obs = src.fetch(["SPY.close"], start, end)
pts = sorted((o.t_event, o.value) for o in obs if o.key == "SPY.close")
ts = np.array([t for t, _ in pts])
p = np.array([v for _, v in pts], dtype=float)

claim = PostedClaim("tw_fool_month_10pct", "", "", asset="SPY", indicator="momentum",
                    window=21, rule="long_above", threshold=0.10, horizon=252,
                    direction=1, n_trials=20)
spec = claim.to_spec()
ret = p[1:] / p[:-1] - 1.0
pos = evaluate_positions(spec, p)[:-1]
n = ret.size
in_mkt = pos != 0.0

d = np.diff(in_mkt.astype(int), prepend=0, append=0)
starts = np.flatnonzero(d == 1)
ends = np.flatnonzero(d == -1)


def dt(i):
    return datetime.fromtimestamp(ts[min(i, ts.size - 1)], tz=timezone.utc).date()


print("episodes:")
for k, (s, e) in enumerate(zip(starts, ends)):
    blk = ret[s:e]
    print(f"  ep{k}: {dt(s)} -> {dt(e)}  len {e - s:4d}  SPY over hold {np.prod(1 + blk) - 1:+.1%}")


def net_sharpe_of(pos_arr):
    turnover = np.abs(np.diff(pos_arr, prepend=0.0))
    return _sharpe(pos_arr * ret - COST * turnover, PPY)


def random_block_positions(lens, n, rng):
    lens = list(lens)
    rng.shuffle(lens)
    slack = n - sum(lens)
    cuts = np.sort(rng.integers(0, slack + 1, size=len(lens)))
    gaps = np.diff(np.concatenate(([0], cuts, [slack])))
    out = np.zeros(n)
    idx = 0
    for g, L in zip(gaps[:-1], lens):
        idx += g
        out[idx:idx + L] = 1.0
        idx += L
    return out


N_DRAWS = 1000
print("\nleave-one-episode-out (strategy minus block k vs exposure-matched null of remaining blocks):")
for k in range(starts.size):
    pos_k = pos.copy()
    pos_k[starts[k]:ends[k]] = 0.0
    sh_k = net_sharpe_of(pos_k)
    lens_k = [(e - s) for j, (s, e) in enumerate(zip(starts, ends)) if j != k]
    null_k = np.array([net_sharpe_of(random_block_positions(lens_k, n, RNG)) for _ in range(N_DRAWS)])
    p_ge = (null_k >= sh_k).mean()
    print(f"  drop ep{k} ({dt(starts[k])}): net Sharpe {sh_k:+.3f} | null mean {null_k.mean():+.3f} "
          f"sd {null_k.std():.3f} | P(null >= strat) = {p_ge:.3f}")

# also: in-hold mean advantage leave-one-out
print("\nleave-one-episode-out in-hold mean vs exposure-matched null in-hold mean:")
for k in range(starts.size):
    mask = in_mkt.copy()
    mask[starts[k]:ends[k]] = False
    mu_k = ret[mask].mean()
    lens_k = [(e - s) for j, (s, e) in enumerate(zip(starts, ends)) if j != k]
    null_mu = np.array([ret[random_block_positions(lens_k, n, RNG) != 0].mean()
                        for _ in range(N_DRAWS)])
    p_mu = (null_mu >= mu_k).mean()
    print(f"  drop ep{k}: in-hold mean {mu_k * 1e4:+.2f} bp | P(null >= actual) = {p_mu:.3f}")
