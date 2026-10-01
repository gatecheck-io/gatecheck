"""AXIS 1 probe 2: episode-level nulls for tw_fool_month_10pct.

The strategy is long-only and matches B&H while in-market; its edge vs B&H is earned
ONLY on flat bars. So the honest episode-unit tests are:
  (A) full EdgeVerdict (timing_sharpe = de-tilted skill, circular-shift p)
  (B) flat-block inference: per flat-stretch avoided market return, t on n_flat_blocks
  (C) block-placement bootstrap: keep the SAME 6 block lengths, place them uniformly at
      random (non-overlapping) in the sample -> null distribution of net Sharpe and of
      mean episode return. Where do the actual numbers fall?
"""
import sys, datetime
import numpy as np


from posted_claims import PostedClaim
from bench_engine import evaluate_positions, run_edge_test, _sharpe
from snapshot_source import SnapshotSource as YahooSource
from snapshot_source import now_epoch

PPY = 252.0
COST = 5e-4
rng = np.random.default_rng(7)

lookback_days = 252 * 16
end = now_epoch()
start = end - lookback_days * 86400.0
src = YahooSource()
obs = src.fetch(["SPY.close"], start, end)
pts = sorted((o.t_event, o.value) for o in obs if o.key == "SPY.close")
dates = [datetime.datetime.fromtimestamp(t, datetime.timezone.utc).date() for t, _ in pts]
p = np.array([v for _, v in pts], dtype=float)
mask = np.isfinite(p)
p = p[mask]
dates = [d for d, m in zip(dates, mask) if m]
print(f"bars={p.size}  window {dates[0]} .. {dates[-1]}")

claim = PostedClaim(
    "tw_fool_month_10pct", "10%+ monthly gain -> 12m hold", "Motley Fool 2026-05-05",
    asset="SPY", indicator="momentum", window=21, rule="long_above", threshold=0.10,
    horizon=252, direction=1, n_trials=20)
spec = claim.to_spec()

ret = p[1:] / p[:-1] - 1.0
target = evaluate_positions(spec, p)
pos = target[:-1]
inmkt = pos != 0.0
rdates = dates[1:]

# ---- (A) full EdgeVerdict ----
v = run_edge_test(spec, p)
print("\n[A] FULL EDGE VERDICT")
print(f"    net Sharpe {v.net_sharpe:+.3f}   timing (de-tilted) {v.timing_sharpe:+.3f}")
print(f"    b&h {v.buy_hold_sharpe:+.3f}   IC {v.ic:+.3f}   circ-shift p {v.p_value:.3f}   "
      f"significant={v.significant}")
print(f"    OOS mean {v.oos_mean_sharpe:+.3f}  CI [{v.oos_ci_lo:+.3f},{v.oos_ci_hi:+.3f}]")

# ---- blocks ----
def contiguous_blocks(b):
    out, i = [], 0
    while i < b.size:
        if b[i]:
            j = i
            while j + 1 < b.size and b[j + 1]:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out

blocks = contiguous_blocks(inmkt)
flat = contiguous_blocks(~inmkt)
lens = [e - s + 1 for s, e in blocks]
print(f"\n    {len(blocks)} in-market blocks (lens {lens}), {len(flat)} flat blocks")

# ---- (B) flat-block inference: the edge vs B&H is -sum(ret) on each flat stretch ----
print("\n[B] FLAT-BLOCK (avoided-return) INFERENCE — the actual source of the edge vs B&H")
avoided = []
for s, e in flat:
    a = -float(np.sum(ret[s:e + 1]))   # relative PnL vs B&H while flat
    avoided.append(a)
    print(f"    flat {rdates[s]} .. {rdates[e]}  len={e-s+1:>4}  avoided (rel PnL) = {a:+.4f}")
av = np.array(avoided)
n = av.size
m, sd = av.mean(), av.std(ddof=1)
t = m / (sd / np.sqrt(n))
from scipy import stats
pval = 2 * stats.t.sf(abs(t), n - 1)
tcrit = stats.t.ppf(0.975, n - 1)
print(f"    mean avoided = {m:+.4f}  sd {sd:.4f}  t({n-1}) = {t:.2f}  p = {pval:.3f}")
print(f"    95% CI [{m - tcrit*sd/np.sqrt(n):+.4f}, {m + tcrit*sd/np.sqrt(n):+.4f}]")
print(f"    total rel PnL vs B&H = {av.sum():+.4f}; largest single flat block share = "
      f"{np.max(np.abs(av))/abs(av.sum()):.0%}")

# ---- (C) block-placement bootstrap ----
print("\n[C] BLOCK-PLACEMENT BOOTSTRAP (same 6 block lengths, random non-overlapping placement)")
N = 5000
n_bars = ret.size

def random_placement(lens, n_bars, rng):
    """Place blocks of the given lengths uniformly at random without overlap.
    Gap-sampling: distribute the total slack across len(lens)+1 gaps uniformly."""
    total = sum(lens)
    slack = n_bars - total
    # sample gap sizes: composition of slack into k+1 parts, uniform
    cuts = np.sort(rng.integers(0, slack + 1, size=len(lens)))
    gaps = np.diff(np.concatenate([[0], cuts, [slack]]))
    order = rng.permutation(len(lens))
    posv = np.zeros(n_bars)
    cur = 0
    for gi, bi in enumerate(order):
        cur += gaps[gi]
        L = lens[bi]
        posv[cur:cur + L] = 1.0
        cur += L
    return posv

actual_pnl = pos * ret - COST * np.abs(np.diff(pos, prepend=0.0))
actual_sh = _sharpe(actual_pnl, PPY)
actual_ep_mean = np.mean([np.sum(actual_pnl[s:e + 1]) for s, e in blocks])
actual_tot = float(np.sum(actual_pnl))

null_sh, null_ep, null_tot = [], [], []
for _ in range(N):
    pv = random_placement(lens, n_bars, rng)
    pnl = pv * ret - COST * np.abs(np.diff(pv, prepend=0.0))
    null_sh.append(_sharpe(pnl, PPY))
    blks = contiguous_blocks(pv != 0)
    null_ep.append(np.mean([np.sum(pnl[s:e + 1]) for s, e in blks]))
    null_tot.append(float(np.sum(pnl)))
null_sh = np.array(null_sh); null_ep = np.array(null_ep); null_tot = np.array(null_tot)

p_sh = float(np.mean(null_sh >= actual_sh))
p_ep = float(np.mean(null_ep >= actual_ep_mean))
p_tot = float(np.mean(null_tot >= actual_tot))
print(f"    actual net Sharpe {actual_sh:+.3f}  vs null mean {null_sh.mean():+.3f} "
      f"(sd {null_sh.std():.3f})  -> p(null >= actual) = {p_sh:.3f}")
print(f"    actual mean episode pnl {actual_ep_mean:+.4f} vs null mean {null_ep.mean():+.4f} "
      f"(sd {null_ep.std():.4f}) -> p = {p_ep:.3f}")
print(f"    actual total pnl {actual_tot:+.4f} vs null mean {null_tot.mean():+.4f} "
      f"(sd {null_tot.std():.4f}) -> p = {p_tot:.3f}")
print(f"    null Sharpe quantiles: 50% {np.quantile(null_sh,0.5):+.2f}  "
      f"90% {np.quantile(null_sh,0.9):+.2f}  95% {np.quantile(null_sh,0.95):+.2f}  "
      f"99% {np.quantile(null_sh,0.99):+.2f}")

# how often does a random placement of the same lengths BEAT buy&hold?
bh_sh = _sharpe(ret, PPY)
print(f"    b&h Sharpe {bh_sh:+.3f}; fraction of random placements with Sharpe > b&h: "
      f"{float(np.mean(null_sh > bh_sh)):.3f}")

# ---- effective sample summary ----
print("\n[SUMMARY]")
print(f"    distinct trigger episodes (gap>21d): 10; contiguous holding blocks: {len(blocks)}")
print(f"    in-market {100*inmkt.mean():.1f}% of bars; edge vs B&H earned on {len(flat)} flat stretches")
print(f"    flat-block t({n-1}) = {t:.2f} p = {pval:.3f}; placement-null p(Sharpe) = {p_sh:.3f}")
