"""AXIS 1 kill-probe: effective sample size & overlap for tw_fool_month_10pct.

Claim: SPY momentum(21) > 0.10 -> long, horizon=252 (forward-fill hold), n_trials=20.
Bench: net Sharpe +1.19 vs B&H +0.73, DSR 0.98, OOS fold Sharpe +1.32, TRADABLE.

Probe:
 (a) distinct trigger episodes in the bench window (cluster consecutive trigger days;
     also gap-merged variants)
 (b) fraction of bars in-market
 (c) episode-level inference: per-episode net return, t on n_episodes
 (d) fold anatomy: per-fold Sharpe, in-market fraction, which episodes dominate,
     share of fold PnL from the single largest episode.
"""
import sys, datetime
import numpy as np


from posted_claims import PostedClaim, bench_posted_claim
from bench_engine import EdgeSpec, evaluate_positions, _sharpe, _momentum
from gatecheck.cv import purged_walk_forward
from snapshot_source import SnapshotSource as YahooSource
from snapshot_source import now_epoch

PPY = 252.0
COST = 5e-4

# ---- fetch exactly like bench_posted_claims (lookback_days = 252*16 calendar days) ----
lookback_days = 252 * 16
end = now_epoch()
start = end - lookback_days * 86400.0
src = YahooSource()
obs = src.fetch(["SPY.close"], start, end)
pts = sorted((o.t_event, o.value) for o in obs if o.key == "SPY.close")
dates = [datetime.datetime.utcfromtimestamp(t).date() for t, _ in pts]
p = np.array([v for _, v in pts], dtype=float)
mask = np.isfinite(p)
p = p[mask]
dates = [d for d, m in zip(dates, mask) if m]
print(f"bars={p.size}  window {dates[0]} .. {dates[-1]}  "
      f"({(dates[-1]-dates[0]).days/365.25:.2f} calendar years)")

claim = PostedClaim(
    "tw_fool_month_10pct",
    "monthly S&P gain of 10%+ -> forward 12m win rate 83%, avg +16.3%",
    "Motley Fool 2026-05-05", asset="SPY", indicator="momentum", window=21,
    rule="long_above", threshold=0.10, horizon=252, direction=1, n_trials=20)

# ---- 0. reproduce the bench verdict on this exact array ----
v = bench_posted_claim(claim, p)
print("\n[0] BENCH REPRODUCTION")
print(f"    net Sharpe {v.net_sharpe:+.3f}  b&h {v.buy_hold_sharpe:+.3f}  DSR {v.dsr:.3f}  "
      f"OOS {v.oos_mean_sharpe:+.3f}  verdict {v.verdict}")

spec = claim.to_spec()
ret = p[1:] / p[:-1] - 1.0
target = evaluate_positions(spec, p)
pos = target[:-1]
turnover = np.abs(np.diff(pos, prepend=0.0))
net_pnl = pos * ret - COST * turnover
rdates = dates[1:]  # ret[i] = bar i -> i+1; stamp with the bar EARNED (i+1)

# ---- (a) trigger episodes ----
ind = _momentum(p, spec.window)
trig = np.isfinite(ind) & (ind > spec.threshold)
trig_idx = np.flatnonzero(trig)

def cluster(idx, gap):
    """Cluster trigger days: new episode when gap between consecutive triggers > gap bars."""
    if idx.size == 0:
        return []
    eps, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            eps.append((s, prev)); s = i
        prev = i
    eps.append((s, prev))
    return eps

print(f"\n[a] TRIGGER DAYS: {trig_idx.size} of {p.size} bars ({100*trig_idx.size/p.size:.1f}%)")
for gap in (1, 5, 21):
    eps = cluster(trig_idx, gap)
    print(f"    gap<= {gap:>2} bars -> {len(eps)} trigger episodes:")
    for s, e in eps:
        print(f"       {dates[s]} .. {dates[e]}  ({e-s+1} bars of triggers, "
              f"first mom={ind[s]:+.3f} max mom={np.nanmax(ind[s:e+1]):+.3f})")

# ---- (b) in-market fraction + position blocks ----
inmkt = pos != 0.0
frac = inmkt.mean()
print(f"\n[b] IN-MARKET: {inmkt.sum()} of {pos.size} bars = {100*frac:.1f}%")

# contiguous in-market blocks = the actual holding episodes (trigger clusters merged
# by the 252-bar reset-on-retrigger hold)
blocks = []
i = 0
while i < pos.size:
    if inmkt[i]:
        j = i
        while j + 1 < pos.size and inmkt[j + 1]:
            j += 1
        blocks.append((i, j))
        i = j + 1
    else:
        i += 1
print(f"    {len(blocks)} contiguous holding blocks (episodes):")
ep_rets, ep_lens = [], []
for s, e in blocks:
    r = net_pnl[s:e + 1]
    tot = float(np.sum(r))                  # additive net pnl
    cmp_ = float(np.prod(1 + r) - 1)        # compounded
    sh = _sharpe(r, PPY)
    ep_rets.append(tot); ep_lens.append(e - s + 1)
    print(f"       {rdates[s]} .. {rdates[e]}  len={e-s+1:>4} bars  "
          f"sum-pnl={tot:+.4f}  comp={cmp_:+.4f}  blockSharpe={sh:+.2f}")

# share of total strategy PnL from the single largest block
tot_pnl = float(np.sum(net_pnl))
print(f"    total strategy net PnL (sum) = {tot_pnl:+.4f}")
for k, (s, e) in enumerate(blocks):
    print(f"       block {k} share of total PnL: {np.sum(net_pnl[s:e+1])/tot_pnl:+.1%}")

# ---- (c) episode-level inference ----
ep = np.array(ep_rets)
n = ep.size
print(f"\n[c] EPISODE-LEVEL INFERENCE (unit = contiguous holding block, n={n})")
if n >= 2:
    m, sd = ep.mean(), ep.std(ddof=1)
    t = m / (sd / np.sqrt(n))
    from scipy import stats  # may not exist; fallback below
    print("scipy ok")
else:
    t = float("nan")
print(f"    mean episode net return = {ep.mean():+.4f}   sd = {ep.std(ddof=1) if n>1 else float('nan'):.4f}")
if n >= 2:
    se = ep.std(ddof=1) / np.sqrt(n)
    tcrit = None
    try:
        from scipy import stats
        tcrit = stats.t.ppf(0.975, n - 1)
        pval = 2 * stats.t.sf(abs(t), n - 1)
        print(f"    t({n-1}) = {t:.2f}   p = {pval:.3f}")
    except Exception:
        # hardcoded two-sided 97.5% t critical values
        tc = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
        tcrit = tc.get(n - 1, 2.0)
        print(f"    t({n-1}) = {t:.2f}   (crit 2-sided 5% = {tcrit:.2f})")
    print(f"    95% CI on mean episode return: [{ep.mean()-tcrit*se:+.4f}, {ep.mean()+tcrit*se:+.4f}]")

# bar-level (naive) t for contrast
sh_bar = _sharpe(net_pnl, PPY)
t_bar = net_pnl.mean() / net_pnl.std() * np.sqrt(net_pnl.size)
print(f"    contrast: bar-level Sharpe {sh_bar:+.2f} -> naive bar t = {t_bar:.2f} "
      f"(treats {net_pnl.size} overlapping bars as independent)")

# effective N sanity: one 21d momentum signal + 252d hold => ~ p.size/252 independent obs
print(f"    overlap arithmetic: {p.size} bars / 252-bar hold = {p.size/252:.1f} "
      f"non-overlapping hold-lengths in the whole window")

# ---- (d) fold anatomy ----
print(f"\n[d] FOLD ANATOMY (purged_walk_forward n={ret.size}, n_splits=5, embargo=5)")
folds = purged_walk_forward(ret.size, n_splits=5, embargo=5)
for f in folds:
    te = f.test
    r = net_pnl[te]
    sh = _sharpe(r, PPY)
    im = inmkt[te].mean()
    d0, d1 = rdates[te[0]], rdates[te[-1]]
    # which blocks intersect this fold, and the largest block's share of fold PnL
    inter = []
    for k, (s, e) in enumerate(blocks):
        lo, hi = max(s, te[0]), min(e, te[-1])
        if lo <= hi:
            share = float(np.sum(net_pnl[lo:hi + 1]))
            inter.append((k, hi - lo + 1, share))
    fold_pnl = float(np.sum(r))
    dom = max(inter, key=lambda x: abs(x[2])) if inter else None
    dom_str = (f"block {dom[0]} contributes {dom[2]:+.4f} of fold PnL {fold_pnl:+.4f} "
               f"({dom[2]/fold_pnl:+.0%})" if inter and fold_pnl != 0 else "flat / none")
    print(f"    fold {f.index}: {d0}..{d1}  Sharpe {sh:+.2f}  in-mkt {100*im:.0f}%  "
          f"blocks {[k for k,_,_ in inter]}  {dom_str}")
fold_sh = [_sharpe(net_pnl[f.test], PPY) for f in folds if f.test.size >= 2]
print(f"    fold Sharpes: {[f'{s:+.2f}' for s in fold_sh]}")
print(f"    NOTE: fold 0 has no train (never emitted) -> the OOS mean is over the "
      f"{len(fold_sh)} emitted folds")

# ---- bonus: distinct 12m outcomes the ARTICLE's stat would count in this window ----
print(f"\n[e] ARTICLE-STYLE COUNT: distinct (gap>21d) trigger episodes with a full 252-bar "
      f"forward window:")
eps21 = cluster(trig_idx, 21)
for s, e in eps21:
    if s + 252 < p.size:
        fwd = p[s + 252] / p[s] - 1.0
        print(f"       trigger {dates[s]}: fwd 252-bar return {fwd:+.1%}")
    else:
        print(f"       trigger {dates[s]}: forward window INCOMPLETE (right edge)")
