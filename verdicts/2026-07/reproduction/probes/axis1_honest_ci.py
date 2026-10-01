"""AXIS 1 probe 3: overlap-honest CIs on the numbers the TRADABLE verdict rests on.

 (1) differential (strategy - B&H) daily PnL: annualized IR + naive t + circular
     block-bootstrap CI (L=252, the hold length) on both net Sharpe and the IR.
 (2) DSR recomputed at honest effective N (block-bootstrap sigma_sr instead of 1/sqrt(2774)).
 (3) leave-one-episode-out: neutralize each flat stretch (set long) / each holding block
     (set flat) and recompute net Sharpe vs B&H -> which single episode carries the pass?
"""
import sys, datetime
import numpy as np


from posted_claims import PostedClaim
from bench_engine import evaluate_positions, _sharpe, _skew_kurt
from gatecheck.deflation import deflated_sharpe_ratio
from snapshot_source import SnapshotSource as YahooSource
from snapshot_source import now_epoch

PPY = 252
COST = 5e-4
rng = np.random.default_rng(11)

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

claim = PostedClaim(
    "tw_fool_month_10pct", "", "", asset="SPY", indicator="momentum", window=21,
    rule="long_above", threshold=0.10, horizon=252, direction=1, n_trials=20)
spec = claim.to_spec()
ret = p[1:] / p[:-1] - 1.0
pos = evaluate_positions(spec, p)[:-1]
inmkt = pos != 0.0
rdates = dates[1:]

turnover = np.abs(np.diff(pos, prepend=0.0))
net_pnl = pos * ret - COST * turnover
diff_pnl = net_pnl - ret            # strategy minus buy&hold, per bar

sh_net = _sharpe(net_pnl, PPY)
sh_bh = _sharpe(ret, PPY)
sh_diff = _sharpe(diff_pnl, PPY)    # information ratio vs B&H
t_diff = diff_pnl.mean() / diff_pnl.std() * np.sqrt(diff_pnl.size)
print(f"[1] net Sharpe {sh_net:+.3f}  B&H {sh_bh:+.3f}  diff (IR vs B&H) {sh_diff:+.3f}  "
      f"naive bar t on diff = {t_diff:+.2f}")

# circular block bootstrap, block length = 252 (the hold horizon)
def cbb(x, L, n_draws, rng):
    n = x.size
    nblk = int(np.ceil(n / L))
    out = np.empty(n_draws)
    for d in range(n_draws):
        starts = rng.integers(0, n, size=nblk)
        idx = (starts[:, None] + np.arange(L)[None, :]).ravel() % n
        xb = x[idx][:n]
        out[d] = _sharpe(xb, PPY)
    return out

for L in (21, 126, 252):
    bs_net = cbb(net_pnl, L, 2000, rng)
    bs_diff = cbb(diff_pnl, L, 2000, rng)
    lo_n, hi_n = np.quantile(bs_net, [0.025, 0.975])
    lo_d, hi_d = np.quantile(bs_diff, [0.025, 0.975])
    p_d = float(np.mean(bs_diff <= 0.0))
    print(f"    CBB L={L:>3}: net Sharpe 95% CI [{lo_n:+.2f},{hi_n:+.2f}] "
          f"(sd {bs_net.std():.2f});  diff IR CI [{lo_d:+.2f},{hi_d:+.2f}] "
          f"(sd {bs_diff.std():.2f})  P(diff<=0)={p_d:.3f}")

# ---- (2) DSR at honest sigma_sr ----
print("\n[2] DSR: bench uses sigma_sr = 1/sqrt(2774) = %.4f" % (1/np.sqrt(net_pnl.size)))
sr_pp = sh_net / np.sqrt(PPY)
sk, ku = _skew_kurt(net_pnl)
for label, n_eff in [("bench n=2774", net_pnl.size), ("n_eff=11 (non-overlap holds)", 11),
                     ("n_eff=10 (trigger episodes)", 10), ("n_eff=6 (holding blocks)", 6)]:
    sig = 1.0 / np.sqrt(max(n_eff, 2))
    d = deflated_sharpe_ratio(sr_pp, n_trials=20, n_samples=net_pnl.size,
                              skew=sk, kurtosis=ku, sigma_sr=sig)
    print(f"    sigma_sr from {label:<32} -> DSR = {d:.3f}  "
          f"({'PASSES' if d >= 0.95 else 'FAILS'} the 0.95 bar)")
# also: DSR with both n_samples and sigma_sr at block-bootstrap scale
bs = cbb(net_pnl, 252, 2000, np.random.default_rng(3)) / np.sqrt(PPY)
sig_emp = float(bs.std())
d_emp = deflated_sharpe_ratio(sr_pp, n_trials=20, n_samples=net_pnl.size,
                              skew=sk, kurtosis=ku, sigma_sr=sig_emp)
print(f"    empirical sigma_sr from CBB(L=252) = {sig_emp:.4f} -> DSR = {d_emp:.3f}  "
      f"({'PASSES' if d_emp >= 0.95 else 'FAILS'})")

# ---- (3) leave-one-episode-out ----
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

print("\n[3a] LEAVE-ONE-FLAT-OUT: force long through each flat stretch (neutralize that one")
print("     timing decision); does the strategy still beat B&H?")
for k, (s, e) in enumerate(flat):
    pv = pos.copy(); pv[s:e + 1] = 1.0
    pnl = pv * ret - COST * np.abs(np.diff(pv, prepend=0.0))
    sh = _sharpe(pnl, PPY)
    print(f"    flat {k} ({rdates[s]}..{rdates[e]}, len {e-s+1:>4}) long-through -> "
          f"net Sharpe {sh:+.3f} vs B&H {sh_bh:+.3f}  "
          f"{'still beats' if sh > sh_bh else '*** NO LONGER BEATS B&H ***'}")

print("\n[3b] LEAVE-ONE-HOLD-OUT: flatten each holding block; Sharpe without it")
for k, (s, e) in enumerate(blocks):
    pv = pos.copy(); pv[s:e + 1] = 0.0
    pnl = pv * ret - COST * np.abs(np.diff(pv, prepend=0.0))
    sh = _sharpe(pnl, PPY)
    print(f"    block {k} ({rdates[s]}..{rdates[e]}) removed -> net Sharpe {sh:+.3f}  "
          f"{'still beats' if sh > sh_bh else 'no longer beats'} B&H")

# ---- extra: OOS CI honesty — the 4 fold Sharpes treated as iid ----
from gatecheck.cv import purged_walk_forward
folds = purged_walk_forward(ret.size, n_splits=5, embargo=5)
fold_sh = [_sharpe(net_pnl[f.test], PPY) for f in folds if f.test.size >= 2]
fold_diff = [_sharpe(diff_pnl[f.test], PPY) for f in folds if f.test.size >= 2]
print(f"\n[3c] folds: net {['%+.2f'%s for s in fold_sh]}  diff-vs-B&H {['%+.2f'%s for s in fold_diff]}")
fd = np.array(fold_diff)
t_fd = fd.mean() / (fd.std(ddof=1) / np.sqrt(fd.size))
from scipy import stats
print(f"    fold-level diff-IR: mean {fd.mean():+.2f}  t({fd.size-1}) = {t_fd:.2f}  "
      f"p = {2*stats.t.sf(abs(t_fd), fd.size-1):.3f}")
