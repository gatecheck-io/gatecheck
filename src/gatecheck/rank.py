"""Rank-space partial-correlation machinery (NaN-aware; numpy only, no scipy).

Lineage: extracted 2026-07-02 from the market_state research program
(``market_os/research/forward_vol_state.py`` — ``_rank`` /
``_spearman_partial_pvalue``; ``market_os/flow/forced_flow_probe.py`` —
``residualize``), post-audit.

Why rank-space partials, not OLS-then-Spearman
----------------------------------------------
The design here exists to prevent a specific, silently wrong pipeline: **residualize
on LEVELS, then rank-correlate the residuals**. For a binary or discrete carrier
(e.g. a ±1 sign signal), OLS-residualizing the carrier's *levels* on continuous
controls and then taking a Spearman/rank IC of the residuals can spuriously
**SIGN-FLIP** the measured association — the upstream program measured a carrier at
rank-IC +0.22 through the buggy pipeline whose true partial was −0.16. The reason:
after a levels-OLS projection, the *ordering* of the residuals of a two-valued
carrier is dominated by the controls, not by the carrier, so ranking the residuals
re-injects control structure with an arbitrary sign.

The correct construction — implemented by :func:`spearman_partial` — is a **true
Spearman partial**: rank the carrier, the target, and every control FIRST, then
residualize and correlate entirely in rank space. Monotone information is preserved,
and binary/discrete carriers keep their true sign.

The permutation null is autocorrelation-aware: it circularly shifts the *ranked
carrier* (preserving its serial correlation) rather than i.i.d.-permuting it, which
for a persistent carrier would yield an over-optimistic p-value.

Public API
----------
* :func:`rank_average` — tie-aware average ranks over finite entries (NaN preserved).
* :func:`residualize` — OLS residual of a target after projecting out controls
  (with intercept), on the pairwise-finite subset, NaN elsewhere.
* :func:`spearman_partial` — Spearman partial correlation + circular-shift p-value.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["rank_average", "residualize", "spearman_partial"]


def rank_average(x: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks over finite entries (NaN preserved); numpy-only, no scipy.

    Finite entries receive 1-based average ranks (ties share the mean of their rank
    positions — the standard Spearman convention); non-finite entries stay NaN, so the
    output is on the SAME index grid as the input.
    """
    x = np.asarray(x, dtype=float)
    r = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    if not np.any(m):
        return r
    v = x[m].astype(float)
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(v.size, dtype=float)
    ranks[order] = np.arange(1, v.size + 1, dtype=float)
    i = 0                                   # average ranks within tie groups
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0   # mean of 1-based ranks i+1..j+1
        i = j + 1
    r[m] = ranks
    return r


def residualize(target: np.ndarray, controls: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    r"""OLS residual of ``target`` after projecting out one or more ``controls``.

    Returns ``target - X @ beta_hat`` (with an intercept column) on the pairwise-finite
    subset, NaN elsewhere — so the residual is on the SAME index grid as ``target`` and can
    be fed straight into PIT-aligned IC machinery. The typical incremental-information
    question: compare a raw signal's IC to the IC of the signal residualized on a candidate
    explanation. If the residual IC survives, the signal carries information the control
    does not.

    The OLS fit is full-sample (a descriptive incremental measure, not a forecast): no
    forward quantity enters the fit, so it introduces no look-ahead into a signal→return
    pairing, which should still go through a PIT alignment helper.

    NOTE the sign-flip hazard in the module docstring: to measure a *rank-space* partial
    correlation, do NOT residualize levels and then rank — use :func:`spearman_partial`,
    which ranks first and residualizes in rank space.
    """
    y = np.asarray(target, dtype=float).ravel()
    if isinstance(controls, np.ndarray) and controls.ndim == 1:
        cols = [np.asarray(controls, dtype=float).ravel()]
    elif isinstance(controls, np.ndarray):
        cols = [controls[:, j].astype(float) for j in range(controls.shape[1])]
    else:
        cols = [np.asarray(c, dtype=float).ravel() for c in controls]
    n = min([len(y)] + [len(c) for c in cols])
    y = y[:n]
    cols = [c[:n] for c in cols]
    out = np.full(n, np.nan, dtype=float)
    mask = np.isfinite(y)
    for c in cols:
        mask &= np.isfinite(c)
    if mask.sum() < len(cols) + 2:
        return out
    X = np.column_stack([np.ones(int(mask.sum()))] + [c[mask] for c in cols])
    yy = y[mask]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    out[mask] = yy - X @ beta
    return out


def spearman_partial(sig: np.ndarray, target: np.ndarray,
                     controls: Sequence[np.ndarray], *, n_perm: int = 1000,
                     seed: int = 0) -> tuple[float, float]:
    """Spearman partial correlation of ``sig`` with ``target`` controlling for ``controls``,
    plus an autocorrelation-aware p-value (circular-shift of the ranked carrier, recomputing
    the partial).

    Ranks carrier, target, and each control, then partials in rank space (a true Spearman
    partial, unlike OLS-residualize-on-levels-then-rank, which sign-flips binary carriers —
    see the module docstring). The null preserves the carrier's autocorrelation by circularly
    shifting its rank series; the target+controls stay fixed and aligned. Returns
    ``(nan, 1.0)`` if under-powered (< 30 aligned finite observations) or degenerate.
    """
    rsig = rank_average(np.asarray(sig, dtype=float))
    rtar = rank_average(np.asarray(target, dtype=float))
    rctrl = [rank_average(np.asarray(c, dtype=float)) for c in controls]
    m = np.isfinite(rsig) & np.isfinite(rtar)
    for rc in rctrl:
        m &= np.isfinite(rc)
    if int(np.sum(m)) < 30:
        return float("nan"), 1.0
    Xc = [rc[m] for rc in rctrl]
    res_t = residualize(rtar[m], Xc)

    def _partial(rs_full: np.ndarray) -> float:
        res_s = residualize(rs_full[m], Xc)
        mm = np.isfinite(res_s) & np.isfinite(res_t)
        if int(np.sum(mm)) < 30:
            return float("nan")
        a, b = res_s[mm], res_t[mm]
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    obs = _partial(rsig)
    if not np.isfinite(obs):
        return float("nan"), 1.0
    rng = np.random.default_rng(seed)
    n = rsig.size
    cnt = tot = 0
    for _ in range(int(n_perm)):
        k = int(rng.integers(1, n))
        pk = _partial(np.roll(rsig, k))
        if np.isfinite(pk):
            tot += 1
            if abs(pk) >= abs(obs) - 1e-12:
                cnt += 1
    p = (cnt + 1) / (tot + 1) if tot > 0 else 1.0
    return obs, float(p)
