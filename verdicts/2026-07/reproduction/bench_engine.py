"""July benchmark replay engine. Preserved research behavior, not a validated trade recommendation.

SMA ratio is centered: price/SMA - 1. Legacy claims used threshold 1;
the separately labeled mapping correction changes SMA thresholds to 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# -- safe indicator primitives (pure, causal; nan where history is insufficient) -----------


def _sma(p: np.ndarray, w: int) -> np.ndarray:
    out = np.full(p.shape, np.nan)
    if w < 1:
        return out
    csum = np.cumsum(np.insert(p, 0, 0.0))
    out[w - 1:] = (csum[w:] - csum[:-w]) / w
    return out


def _momentum(p: np.ndarray, w: int) -> np.ndarray:
    out = np.full(p.shape, np.nan)
    if w >= 1 and p.size > w:
        out[w:] = p[w:] / p[:-w] - 1.0
    return out


def _sma_ratio(p: np.ndarray, w: int) -> np.ndarray:
    sma = _sma(p, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sma > 0, p / sma - 1.0, np.nan)


def _zscore(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if w < 2:
        return out
    for t in range(w - 1, x.size):
        window = x[t - w + 1:t + 1]
        sd = window.std()
        out[t] = (x[t] - window.mean()) / sd if sd > 0 else 0.0
    return out


def _rsi(p: np.ndarray, w: int) -> np.ndarray:
    out = np.full(p.shape, np.nan)
    if w < 1 or p.size <= w:
        return out
    delta = np.diff(p)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = gain[:w].mean()
    avg_loss = loss[:w].mean()
    for i in range(w, p.size):
        g, l = gain[i - 1], loss[i - 1]
        avg_gain = (avg_gain * (w - 1) + g) / w
        avg_loss = (avg_loss * (w - 1) + l) / w
        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _pctrank(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if w < 2:
        return out
    for t in range(w - 1, x.size):
        window = x[t - w + 1:t + 1]
        out[t] = float(np.mean(window <= x[t]))
    return out


def _raw(x: np.ndarray, w: int) -> np.ndarray:
    return np.asarray(x, dtype=float).copy()


#: Whitelisted indicators: ``name -> f(series, window) -> array`` (nan where undefined).
INDICATORS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "momentum": _momentum,     # trailing % change over window
    "sma_ratio": _sma_ratio,   # price / SMA(window) - 1
    "zscore": _zscore,         # rolling z-score
    "rsi": _rsi,               # Wilder RSI(window), 0..100
    "pctrank": _pctrank,       # trailing percentile rank, 0..1
    "raw": _raw,               # use the series itself (external feature: VIX, put/call, ...)
}

#: Whitelisted rules: ``name -> f(indicator, threshold) -> target position in {-1,0,+1}``.
RULES: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "long_above": lambda ind, thr: np.where(ind > thr, 1.0, 0.0),
    "long_below": lambda ind, thr: np.where(ind < thr, 1.0, 0.0),
    "short_above": lambda ind, thr: np.where(ind > thr, -1.0, 0.0),
    "short_below": lambda ind, thr: np.where(ind < thr, -1.0, 0.0),
    "long_short_above": lambda ind, thr: np.where(ind > thr, 1.0, np.where(ind < thr, -1.0, 0.0)),
    "long_short_below": lambda ind, thr: np.where(ind < thr, 1.0, np.where(ind > thr, -1.0, 0.0)),
}


@dataclass(frozen=True)
class EdgeSpec:
    """A declarative trading-edge claim (data, never code).

    ``indicator``/``rule`` must be keys of :data:`INDICATORS` / :data:`RULES`. ``n_trials`` is
    the *declared* number of variants the edge was selected from (>1 raises the DSR bar) — set
    it honestly; a single-shot claim posted after visible grid-search is not ``n_trials=1``.
    """

    name: str
    indicator: str
    window: int
    rule: str
    threshold: float
    horizon: int = 1                 # minimum holding in bars (entries forward-fill h bars)
    direction: int = 1               # +1 as stated, -1 to invert the whole rule
    asset: str = "SPY"
    source: str = ""                 # citation / URL
    claimed_sharpe: Optional[float] = None
    n_trials: int = 1                # declared search breadth, for the DSR multiplicity haircut

    def validate(self) -> None:
        if self.indicator not in INDICATORS:
            raise ValueError(f"unknown indicator {self.indicator!r}; pick from {sorted(INDICATORS)}")
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; pick from {sorted(RULES)}")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")


def evaluate_positions(spec: EdgeSpec, series: np.ndarray) -> np.ndarray:
    """Turn a spec into a causal per-bar target position over ``series`` (the signal series).

    ``series`` is the price series for price indicators, or the external feature for ``raw``.
    Returns a length-``len(series)`` array in ``{-1, 0, +1}``; non-finite indicator bars (and
    bars before the window fills) map to flat (0). ``horizon`` forward-fills each entry.
    """
    spec.validate()
    x = np.asarray(series, dtype=float)
    ind = INDICATORS[spec.indicator](x, spec.window)
    target = RULES[spec.rule](ind, float(spec.threshold))
    target = np.where(np.isfinite(ind), target, 0.0) * float(spec.direction)

    if spec.horizon > 1:
        held = 0
        cur = 0.0
        out = np.zeros_like(target)
        for i in range(target.size):
            if target[i] != 0.0:
                cur, held = target[i], spec.horizon
            if held > 0:
                out[i], held = cur, held - 1
            else:
                out[i] = 0.0
        target = out
    return target


def _sharpe(pnl: np.ndarray, ppy: int) -> float:
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size < 2 or pnl.std() == 0.0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(ppy))


def _skew_kurt(pnl: np.ndarray) -> tuple[float, float]:
    x = pnl[np.isfinite(pnl)]
    if x.size < 3 or x.std() == 0.0:
        return 0.0, 3.0
    z = (x - x.mean()) / x.std()
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


@dataclass
class EdgeVerdict:
    """The honest verdict for one posted edge."""

    name: str
    source: str
    n_samples: int
    gross_sharpe: float          # raw net-of-cost annualized Sharpe of the strategy
    net_sharpe: float            # alias kept for clarity == gross_sharpe (after costs)
    timing_sharpe: float         # de-tilted (causal) Sharpe — skill beyond the directional tilt
    buy_hold_sharpe: float       # passive-long benchmark (is this beta or alpha?)
    best_random_sharpe: float    # best of multi-seed random-sign nulls
    dsr: float                   # Deflated Sharpe Ratio in [0,1] (multiplicity-corrected)
    discounted_sharpe: float     # net_sharpe * DSR (the shrunk, honest number)
    oos_mean_sharpe: float       # mean test-block Sharpe across purged walk-forward folds
    oos_ci_lo: float
    oos_ci_hi: float
    oos_excludes_zero: bool
    ic: float                    # IC(position, same-bar return)
    p_value: float               # autocorr-aware circular-shift p-value
    n_trials: int
    claimed_sharpe: Optional[float]
    cost_per_turnover: float
    beats_buy_hold: bool
    beats_random: bool
    significant: bool
    dsr_ok: bool
    passed: bool

    def summary(self) -> str:
        def mark(b: bool) -> str:
            return "PASS" if b else "fail"
        claim = "n/a" if self.claimed_sharpe is None else f"{self.claimed_sharpe:+.2f}"
        lines = [
            "=" * 66,
            f"TEST-EDGE — {self.name}   [{'PASS' if self.passed else 'FAIL'}]",
            "=" * 66,
            f"source             : {self.source or '(none given)'}",
            f"samples            : {self.n_samples}   declared n_trials : {self.n_trials}",
            f"claimed Sharpe     : {claim}   ->  realized net : {self.net_sharpe:+.2f}",
            "-" * 66,
            f"net Sharpe (cost)  : {self.net_sharpe:+.3f}",
            f"timing Sharpe      : {self.timing_sharpe:+.3f}  (de-tilted: skill beyond drift)",
            f"buy&hold Sharpe    : {self.buy_hold_sharpe:+.3f}   "
            f"[{mark(self.beats_buy_hold)}: beats passive long]",
            f"best random-sign   : {self.best_random_sharpe:+.3f}   "
            f"[{mark(self.beats_random)}: beats random]",
            f"Deflated Sharpe    : {self.dsr:.3f}  -> discounted {self.discounted_sharpe:+.3f}   "
            f"[{mark(self.dsr_ok)}: survives multiplicity]",
            f"OOS mean Sharpe    : {self.oos_mean_sharpe:+.3f}  "
            f"95% CI [{self.oos_ci_lo:+.3f}, {self.oos_ci_hi:+.3f}]   "
            f"[{mark(self.oos_excludes_zero)}: OOS-stable]",
            f"IC / circ-shift p  : {self.ic:+.3f} / {self.p_value:.3f}   "
            f"[{mark(self.significant)}: significant]",
            "-" * 66,
            f"VERDICT: {'PASS — survives the skeptic bench' if self.passed else 'FAIL — does not clear the bench'}",
            "=" * 66,
        ]
        return "\n".join(lines)


def run_edge_test(
    spec: EdgeSpec,
    prices: np.ndarray,
    *,
    signal_series: Optional[np.ndarray] = None,
    cost_per_turnover: float = 5e-4,
    periods_per_year: int = 252,
    n_splits: int = 5,
    embargo: int = 5,
    n_random_seeds: int = 25,
    dsr_threshold: float = 0.95,
    n_perm: int = 2000,
    seed: int = 0,
) -> EdgeVerdict:
    """Run ``spec`` through the full skeptic bench against ``prices`` and return the verdict.

    ``prices`` drives the traded asset's returns. By default the signal is computed on
    ``prices`` too; pass ``signal_series`` (aligned, same length) to drive the signal off an
    external feature (e.g. a VIX or put/call series) while trading ``prices`` — common for
    posted edges. ``passed`` requires the conjunction: beats buy&hold, survives the Deflated
    Sharpe multiplicity bar (``dsr >= dsr_threshold``), and is OOS-stable (test-fold Sharpe CI
    excludes zero). A FAIL is the expected, valid outcome for most posted edges.
    """
    from bench_support import baseline_pnl, baseline_positions
    from gatecheck.cv import purged_walk_forward
    from bench_support import net_timing_pnl
    from gatecheck.significance import (
        circular_shift_pvalue, information_coefficient, multiseed_sharpe,
        sharpe_ci_excludes_zero)
    from gatecheck.deflation import deflated_sharpe_ratio

    spec.validate()
    p = np.asarray(prices, dtype=float)
    sig = p if signal_series is None else np.asarray(signal_series, dtype=float)
    if sig.shape != p.shape:
        raise ValueError("signal_series must match prices in length")
    if p.size < max(spec.window + 2, n_splits + embargo + 2):
        raise ValueError("price series too short for this spec / CV configuration")

    ret = p[1:] / p[:-1] - 1.0                       # ret[i]: bar i -> i+1
    target = evaluate_positions(spec, sig)           # decided at close of bar i
    pos = target[:-1]                                # pos[i] held into ret[i] (no look-ahead)
    return score_positions(
        pos, ret, name=spec.name, source=spec.source, n_trials=spec.n_trials,
        claimed_sharpe=spec.claimed_sharpe, cost_per_turnover=cost_per_turnover,
        periods_per_year=periods_per_year, n_splits=n_splits, embargo=embargo,
        n_random_seeds=n_random_seeds, dsr_threshold=dsr_threshold, n_perm=n_perm, seed=seed)


def score_positions(
    pos: np.ndarray,
    ret: np.ndarray,
    *,
    name: str,
    source: str = "",
    n_trials: int = 1,
    claimed_sharpe: Optional[float] = None,
    cost_per_turnover: float = 5e-4,
    periods_per_year: int = 252,
    n_splits: int = 5,
    embargo: int = 5,
    n_random_seeds: int = 25,
    dsr_threshold: float = 0.95,
    n_perm: int = 2000,
    seed: int = 0,
) -> EdgeVerdict:
    """Score an arbitrary PIT-aligned per-bar position series through the skeptic bench (the core).

    ``pos[i]`` is the position held into ``ret[i]`` — already aligned with no look-ahead — and may be
    **any real number** (e.g. a continuous vol-managed exposure), not only ``{-1, 0, +1}``. Computes
    the net-of-cost Sharpe, the Deflated-Sharpe multiplicity haircut (``n_trials`` = the search
    breadth), purged walk-forward OOS stability, and the buy&hold / random-sign baselines; ``passed``
    = beats buy&hold AND survives the DSR bar AND is OOS-stable. This is the shared core of
    :func:`run_edge_test` and the continuous-position (vol-managed) benches.
    """
    from bench_support import baseline_pnl, baseline_positions
    from gatecheck.cv import purged_walk_forward
    from bench_support import net_timing_pnl
    from gatecheck.significance import (
        circular_shift_pvalue, information_coefficient, multiseed_sharpe,
        sharpe_ci_excludes_zero)
    from gatecheck.deflation import deflated_sharpe_ratio

    pos = np.asarray(pos, dtype=float)
    ret = np.asarray(ret, dtype=float)
    if pos.shape != ret.shape:
        raise ValueError("pos and ret must be the same length (pos[i] held into ret[i])")

    # raw net-of-cost P&L (the strategy as actually stated/traded)
    turnover = np.abs(np.diff(pos, prepend=0.0))
    net_pnl = pos * ret - cost_per_turnover * turnover
    net_sharpe = _sharpe(net_pnl, periods_per_year)

    # de-tilted timing P&L (skill beyond the unconditional directional tilt)
    timing_sharpe = _sharpe(net_timing_pnl(pos, ret, cost_per_turnover=cost_per_turnover),
                            periods_per_year)

    # benchmarks on the same returns
    buy_hold_sharpe = _sharpe(baseline_pnl(baseline_positions("buy_hold", ret.size), ret),
                              periods_per_year)
    rand = [_sharpe(baseline_pnl(baseline_positions("random_sign", ret.size, seed=s), ret)
                    - cost_per_turnover * np.abs(np.diff(
                        baseline_positions("random_sign", ret.size, seed=s), prepend=0.0)),
                    periods_per_year)
            for s in range(n_random_seeds)]
    best_random_sharpe = max(rand) if rand else 0.0

    # Deflated Sharpe Ratio (multiplicity haircut) on the per-OBSERVATION net Sharpe.
    # sigma_sr is the cross-trial SR dispersion; with only a declared n_trials (not the trial
    # SRs themselves) the principled default is the null per-observation SR-estimator
    # dispersion ~ 1/sqrt(n) — i.e. deflate against the best of n_trials pure-luck strategies.
    sr_pp = net_sharpe / np.sqrt(periods_per_year)
    sk, ku = _skew_kurt(net_pnl)
    sigma_sr = 1.0 / np.sqrt(max(int(ret.size), 2))
    dsr = deflated_sharpe_ratio(sr_pp, n_trials=int(n_trials), n_samples=int(ret.size),
                                skew=sk, kurtosis=ku, sigma_sr=sigma_sr)
    discounted = net_sharpe * dsr

    # purged walk-forward OOS: Sharpe on each test block (fixed rule -> time-robustness)
    folds = purged_walk_forward(ret.size, n_splits=n_splits, embargo=embargo)
    fold_sharpes = [_sharpe(net_pnl[f.test], periods_per_year) for f in folds
                    if f.test.size >= 2]
    agg = multiseed_sharpe(fold_sharpes) if len(fold_sharpes) >= 2 else None
    oos_mean = agg.mean if agg else (fold_sharpes[0] if fold_sharpes else 0.0)
    oos_lo = agg.lo if agg else oos_mean
    oos_hi = agg.hi if agg else oos_mean
    oos_excl = bool(len(fold_sharpes) >= 2 and sharpe_ci_excludes_zero(fold_sharpes)
                    and oos_mean > 0)

    ic = information_coefficient(pos, ret, method="spearman")
    p_value = circular_shift_pvalue(pos, ret, n_perm=n_perm, seed=seed, two_sided=True).p_value

    beats_bh = net_sharpe > buy_hold_sharpe
    beats_rand = net_sharpe > best_random_sharpe
    significant = p_value < 0.05 and np.isfinite(ic)
    dsr_ok = dsr >= dsr_threshold and discounted > 0.0
    passed = bool(beats_bh and dsr_ok and oos_excl)

    return EdgeVerdict(
        name=name, source=source, n_samples=int(ret.size),
        gross_sharpe=net_sharpe, net_sharpe=net_sharpe, timing_sharpe=timing_sharpe,
        buy_hold_sharpe=buy_hold_sharpe, best_random_sharpe=best_random_sharpe,
        dsr=dsr, discounted_sharpe=discounted,
        oos_mean_sharpe=oos_mean, oos_ci_lo=oos_lo, oos_ci_hi=oos_hi,
        oos_excludes_zero=oos_excl, ic=ic, p_value=p_value,
        n_trials=int(n_trials), claimed_sharpe=claimed_sharpe,
        cost_per_turnover=cost_per_turnover,
        beats_buy_hold=beats_bh, beats_random=beats_rand, significant=significant,
        dsr_ok=dsr_ok, passed=passed)


#: A couple of canonical "posted edge" archetypes for the CLI demo / docs.
EDGE_EXAMPLES: dict[str, EdgeSpec] = {
    "rsi2_meanrev": EdgeSpec(
        name="RSI(2) mean-reversion (long oversold)", indicator="rsi", window=2,
        rule="long_below", threshold=10.0, horizon=3, asset="SPY",
        source="classic r/algotrading staple", claimed_sharpe=1.5, n_trials=20),
    "sma200_trend": EdgeSpec(
        name="200-day trend filter (long above)", indicator="sma_ratio", window=200,
        rule="long_above", threshold=0.0, horizon=1, asset="SPY",
        source="classic trend-following", claimed_sharpe=0.8, n_trials=5),
    "mom12_1": EdgeSpec(
        name="12-1 momentum (long when positive)", indicator="momentum", window=120,
        rule="long_above", threshold=0.0, horizon=20, asset="SPY",
        source="academic momentum", claimed_sharpe=0.6, n_trials=10),
}
