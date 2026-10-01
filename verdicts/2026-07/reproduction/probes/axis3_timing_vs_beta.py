"""AXIS 3 — timing vs beta kill-probe for claim tw_fool_month_10pct.

Claim: SPY, momentum(21) > 0.10 -> long 252 bars. Bench said net Sharpe +1.19 vs B&H +0.73,
DSR 0.98, TRADABLE. This probe asks whether the TRIGGER adds anything beyond being long SPY
a large fraction of a bull sample.

(a) exposure-matched null: same episode count + same block lengths, placed uniformly at random
    (non-overlapping, order shuffled), 1000 draws -> where does the strategy's net Sharpe sit?
(b) de-tilt / timing: engine timing_sharpe; in-hold vs out-of-hold mean daily return;
    permutation test on the in-hold mean.
(c) framing check: unconditional 12-month (252-bar) forward win rate on this sample vs the
    conditional post-trigger win rate the Fool cites (83%, +16.3%).
"""
from __future__ import annotations

import sys

import numpy as np


from posted_claims import PostedClaim, bench_posted_claim, _fetch_prices
from bench_engine import evaluate_positions, run_edge_test, _sharpe
from snapshot_source import SnapshotSource as YahooSource

RNG = np.random.default_rng(20260702)
COST = 5e-4
PPY = 252

claim = PostedClaim(
    claim_id="tw_fool_month_10pct",
    text="monthly S&P500 gain of 10%+ -> forward 12m win rate 83%, avg +16.3%",
    source="Motley Fool 2026-05-05",
    asset="SPY", indicator="momentum", window=21, rule="long_above",
    threshold=0.10, horizon=252, direction=1, n_trials=20,
)

print("fetching SPY exactly as the bench does (lookback_days=252*16)...")
prices = _fetch_prices("SPY", YahooSource(), 252 * 16)
print(f"n bars = {prices.size}")

# ---- 0. replicate the bench verdict ------------------------------------------------------
v = bench_posted_claim(claim, prices, cost_per_turnover=COST, seed=0)
print("\n=== bench replication ===")
print(f"net Sharpe {v.net_sharpe:+.3f} | B&H {v.buy_hold_sharpe:+.3f} | DSR {v.dsr:.3f} | "
      f"OOS {v.oos_mean_sharpe:+.3f} | verdict {v.verdict}")

# full verdict for the timing sharpe
spec = claim.to_spec()
full = run_edge_test(spec, prices, cost_per_turnover=COST, seed=0)
print(f"engine timing Sharpe (de-tilted) = {full.timing_sharpe:+.3f}")
print(f"circ-shift p = {full.p_value:.3f}, IC = {full.ic:+.4f}")

# ---- build pos/ret exactly as run_edge_test does -----------------------------------------
p = np.asarray(prices, float)
ret = p[1:] / p[:-1] - 1.0
target = evaluate_positions(spec, p)
pos = target[:-1]
n = ret.size

in_mkt = pos != 0.0
exposure = in_mkt.mean()
print(f"\nexposure: {exposure:.1%} of {n} bars in-market")

# contiguous in-market blocks (episodes)
d = np.diff(in_mkt.astype(int), prepend=0, append=0)
starts = np.flatnonzero(d == 1)
ends = np.flatnonzero(d == -1)
block_lens = (ends - starts).tolist()
print(f"episodes: {len(block_lens)} blocks, lengths = {block_lens}")

# raw trigger days (indicator crossing, pre-horizon-fill)
mom = np.full(p.shape, np.nan)
mom[21:] = p[21:] / p[:-21] - 1.0
trig = np.flatnonzero(np.nan_to_num(mom, nan=-9) > 0.10)
print(f"raw trigger days (mom21 > 10%): {trig.size}")


def net_sharpe_of(pos_arr: np.ndarray) -> float:
    turnover = np.abs(np.diff(pos_arr, prepend=0.0))
    pnl = pos_arr * ret - COST * turnover
    return _sharpe(pnl, PPY)


strat_sh = net_sharpe_of(pos)
print(f"probe-recomputed net Sharpe = {strat_sh:+.3f} (should match bench)")

# ---- (a) exposure-matched null: 1000 draws -----------------------------------------------
def random_block_positions(lens: list[int], n: int, rng) -> np.ndarray:
    """Place blocks of the given lengths uniformly at random, non-overlapping, order shuffled."""
    lens = list(lens)
    rng.shuffle(lens)
    S = sum(lens)
    slack = n - S
    if slack < 0:
        raise ValueError("blocks exceed series")
    # uniform composition of slack into len(lens)+1 gaps
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
null_sh = np.empty(N_DRAWS)
for i in range(N_DRAWS):
    null_sh[i] = net_sharpe_of(random_block_positions(block_lens, n, RNG))

pct = (null_sh < strat_sh).mean()
p_ge = (null_sh >= strat_sh).mean()
print("\n=== (a) exposure-matched null (episode-preserving, 1000 draws) ===")
print(f"null net Sharpe: mean {null_sh.mean():+.3f}, sd {null_sh.std():.3f}, "
      f"5-50-95 pct = {np.percentile(null_sh,5):+.3f} / {np.percentile(null_sh,50):+.3f} / "
      f"{np.percentile(null_sh,95):+.3f}")
print(f"strategy {strat_sh:+.3f} sits at the {pct:.1%} percentile; P(null >= strat) = {p_ge:.3f}")

# variant: K random trigger dates with 252-bar forward fill (merging overlaps), K = raw entry count
n_entries = len(block_lens)
null2 = np.empty(N_DRAWS)
for i in range(N_DRAWS):
    pos2 = np.zeros(n)
    for s in RNG.integers(0, n, size=n_entries):
        pos2[s:s + 252] = 1.0
    null2[i] = net_sharpe_of(pos2)
p_ge2 = (null2 >= strat_sh).mean()
print(f"variant (K={n_entries} random 252-bar holds, overlaps merge): mean {null2.mean():+.3f}, "
      f"sd {null2.std():.3f}, P(null >= strat) = {p_ge2:.3f}")

# ---- (b) de-tilt: does the trigger pick better bars? --------------------------------------
mu_in = ret[in_mkt].mean()
mu_out = ret[~in_mkt].mean() if (~in_mkt).any() else float("nan")
mu_all = ret.mean()
print("\n=== (b) de-tilt / timing ===")
print(f"mean daily ret: in-hold {mu_in*1e4:+.2f} bp | out-of-hold {mu_out*1e4:+.2f} bp | "
      f"all {mu_all*1e4:+.2f} bp")
print(f"annualized: in-hold {mu_in*252:+.2%} | out {mu_out*252:+.2%} | all {mu_all*252:+.2%}")
sd_in, sd_out = ret[in_mkt].std(), ret[~in_mkt].std()
print(f"daily sd: in-hold {sd_in*1e4:.1f} bp | out-of-hold {sd_out*1e4:.1f} bp")
print(f"Sharpe of SPY restricted to in-hold bars   = {_sharpe(ret[in_mkt], PPY):+.3f}")
print(f"Sharpe of SPY restricted to out-hold bars  = {_sharpe(ret[~in_mkt], PPY):+.3f}")
print(f"Sharpe of always-long SPY (all bars)       = {_sharpe(ret, PPY):+.3f}")

# permutation test on the in-hold mean using the SAME exposure-matched placement null
null_mu = np.empty(N_DRAWS)
for i in range(N_DRAWS):
    m = random_block_positions(block_lens, n, RNG) != 0
    null_mu[i] = ret[m].mean()
p_mu = (null_mu >= mu_in).mean()
print(f"P(exposure-matched null in-hold mean >= actual in-hold mean) = {p_mu:.3f}")
print(f"engine timing Sharpe (expanding de-tilt, net) = {full.timing_sharpe:+.3f}")

# ---- (c) framing check: conditional vs unconditional 12m win rate --------------------------
H = 252
fwd = p[H:] / p[:-H] - 1.0  # fwd[t] = 12m return starting at bar t
uncond_win = (fwd > 0).mean()
uncond_avg = fwd.mean()
trig_valid = trig[trig < fwd.size]
cond_win = (fwd[trig_valid] > 0).mean() if trig_valid.size else float("nan")
cond_avg = fwd[trig_valid].mean() if trig_valid.size else float("nan")
# episode-level (first trigger of each block) — the Fool counts 13 EVENTS, not days
ep_starts = starts[starts < fwd.size]
ep_win = (fwd[ep_starts] > 0).mean() if ep_starts.size else float("nan")
ep_avg = fwd[ep_starts].mean() if ep_starts.size else float("nan")
print("\n=== (c) framing check: 12-month forward returns on this sample ===")
print(f"UNCONDITIONAL: win rate {uncond_win:.1%}, avg {uncond_avg:+.1%}  "
      f"(all {fwd.size} rolling 252-bar windows)")
print(f"CONDITIONAL on trigger day (n={trig_valid.size} days): win {cond_win:.1%}, avg {cond_avg:+.1%}")
print(f"CONDITIONAL on episode start (n={ep_starts.size} events): win {ep_win:.1%}, avg {ep_avg:+.1%}")
print(f"Fool claim: 83% win, +16.3% avg (50y). Unconditional here: {uncond_win:.0%}, {uncond_avg:+.1%}")
