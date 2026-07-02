"""Statistical-significance primitives: IC, permutation nulls, multi-seed Sharpe CIs.

Lineage: extracted 2026-07-02 from the market_state research program
(``market_os/evaluation/significance.py``, post-audit), including the
2026-07-02 small-n Student-t CI repair (see change log at the bottom of this
docstring). These are the small, dependency-light statistics that turn a
backtest harness from *descriptive* into *weakly falsifiable*:

* an out-of-sample *information coefficient* (IC) of a point-in-time (PIT)
  estimate at bar ``t`` against the realized **next-bar** return, plus a
  fixed-seed permutation p-value;
* an autocorrelation-aware **circular-shift** permutation null for persistent
  signals;
* multi-seed Sharpe aggregation and the ``CI-excludes-zero`` / ``p10 > 0``
  primitives that acceptance gates assert on.

Design invariants (these are *hard* — a violation means the result is wrong):

* **PIT / no look-ahead.** The IC must compare a signal known at ``t`` with the
  return realized over ``(t, t+1]``. :func:`align_next_return` makes that shift
  explicit (drop the last signal, drop the first return) so the off-by-one
  alignment leak — the classic way an IC is silently inflated — is visible at
  the call site rather than buried in a helper.
* **Correct null.** The permutation test permutes the **signal**, not the
  forward return. Shuffling the signal destroys the signal→future link while
  *preserving the return series' own autocorrelation*, which is exactly the null
  "this signal carries no information about the next bar" without pretending the
  returns are i.i.d.
* **Determinism.** All randomness flows through a single seeded
  ``numpy.random.default_rng(seed)``. Same seed ⇒ byte-identical
  :class:`ICTest`.
* **Add-one estimator.** ``p = (1 + #{perm_stat >= obs_stat}) / (n_perm + 1)``
  so the p-value lives in ``[1/(n_perm+1), 1]`` and is never exactly 0.

Pure ``numpy`` + stdlib. No scipy: Spearman is Pearson on ranks, ranks via
``argsort``.

**Change log — 2026-07-02 (upstream repair R4; audit findings S7 / RA-21).**
:func:`multiseed_sharpe` previously applied the normal critical value ``z = 1.96``
at *every* ``n``, including ``n = 4–5`` fold/seed Sharpes (where the Student-t
requires 3.182 / 2.776) — an anti-conservative CI, ~30–60% too narrow at small
``n``. It now uses the two-sided 95% t-quantile with ``n-1`` dof (hardcoded
lookup for dof 1..30, :func:`t_quantile_95`; falls back to ``z`` beyond 30).
This makes every ``CI-excludes-zero`` gate STRICTER. Pass ``ci="z"`` for the
legacy pre-fix behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "information_coefficient",
    "align_next_return",
    "ICTest",
    "permutation_pvalue",
    "circular_shift_pvalue",
    "SeedSharpe",
    "multiseed_sharpe",
    "sharpe_ci_excludes_zero",
    "t_quantile_95",
]


# --------------------------------------------------------------------------- #
# Rank / correlation helpers (numpy only — no scipy).
# --------------------------------------------------------------------------- #
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks of ``x`` (ties share the mean of their rank positions).

    Average-rank handling matters for Spearman: with ties, naive ``argsort``
    ranks break ties arbitrarily and bias the correlation. We compute the
    competition-free *average* rank, which is the standard Spearman definition.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")  # stable sort
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)

    # Resolve ties to their average rank. Walk the sorted values in runs.
    sorted_x = x[order]
    n = len(x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = (i + j - 1) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns 0.0 if either side has zero variance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    r = float(np.sum(a * b) / denom)
    if not np.isfinite(r):
        return 0.0
    # Clamp tiny floating-point overshoot.
    return max(-1.0, min(1.0, r))


def _finite_pairs(
    signal: np.ndarray, fwd_return: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pairwise-finite subset of ``(signal, fwd_return)``."""
    s = np.asarray(signal, dtype=float).ravel()
    f = np.asarray(fwd_return, dtype=float).ravel()
    n = min(len(s), len(f))
    s = s[:n]
    f = f[:n]
    mask = np.isfinite(s) & np.isfinite(f)
    return s[mask], f[mask]


# --------------------------------------------------------------------------- #
# Information coefficient.
# --------------------------------------------------------------------------- #
def information_coefficient(
    signal: np.ndarray, fwd_return: np.ndarray, *, method: str = "spearman"
) -> float:
    """Information coefficient between ``signal`` and ``fwd_return``.

    NaN-safe and pairwise: non-finite entries on either side are dropped before
    correlating. ``method='spearman'`` (default) correlates the average-ranks of
    the two series (Spearman = Pearson on ranks); ``method='pearson'`` correlates
    the raw values.

    Returns ``0.0`` (a deliberately conservative, falsifiable null) when there
    are fewer than 3 finite pairs or when either series has zero variance.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    s, f = _finite_pairs(signal, fwd_return)
    if len(s) < 3:
        return 0.0

    if method == "spearman":
        # Zero variance ⇒ ranks are constant ⇒ _pearson returns 0.0 anyway,
        # but guard explicitly so the contract is obvious.
        if np.all(s == s[0]) or np.all(f == f[0]):
            return 0.0
        return _pearson(_rankdata(s), _rankdata(f))

    return _pearson(s, f)


# --------------------------------------------------------------------------- #
# PIT alignment helper.
# --------------------------------------------------------------------------- #
def align_next_return(
    signal_t: np.ndarray, realized_ret: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align ``signal[t]`` with ``return[t+1]`` (point-in-time, no look-ahead).

    Drops the **last** signal (it has no observed next return) and the **first**
    return (no prior signal predicts it), so the returned arrays pair
    ``signal[t]`` against ``ret[t+1]``. Both outputs have length ``n - 1`` for an
    input length ``n``; an input shorter than 2 yields empty arrays.

    Making the shift an explicit helper — called *at the harness site*, not
    hidden — keeps the PIT discipline auditable: the off-by-one IC leak is the
    single most common way a backtest fools itself.
    """
    s = np.asarray(signal_t, dtype=float).ravel()
    r = np.asarray(realized_ret, dtype=float).ravel()
    n = min(len(s), len(r))
    if n < 2:
        empty = np.asarray([], dtype=float)
        return empty, empty
    s = s[:n]
    r = r[:n]
    return s[:-1], r[1:]


# --------------------------------------------------------------------------- #
# Permutation p-value.
# --------------------------------------------------------------------------- #
@dataclass
class ICTest:
    """Result of a permutation IC test.

    Attributes
    ----------
    ic:
        Observed information coefficient on the aligned, finite pairs.
    p_value:
        Add-one permutation p-value in ``[1/(n_perm+1), 1]``.
    n:
        Number of finite pairs the statistic was computed on.
    n_perm:
        Number of permutations drawn.
    seed:
        Seed used for the permutation RNG (for reproducibility).
    """

    ic: float
    p_value: float
    n: int
    n_perm: int
    seed: int


def permutation_pvalue(
    signal: np.ndarray,
    fwd_return: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int = 0,
    method: str = "spearman",
    two_sided: bool = True,
) -> ICTest:
    """Permutation test for the IC between ``signal`` and ``fwd_return``.

    The **signal** is permuted (not the return) under a fixed
    ``np.random.default_rng(seed)``: this breaks any signal→future relationship
    while preserving the return series' own structure, which is the correct null
    for "the signal carries no next-bar information".

    The p-value uses the add-one estimator
    ``(1 + #{stat_perm >= stat_obs}) / (n_perm + 1)`` so it is bounded below by
    ``1/(n_perm+1)`` and never reported as exactly 0.

    ``two_sided=True`` compares ``|ic|`` (tests for *any* association);
    ``two_sided=False`` compares the signed ``ic`` (tests for *positive*
    association only — a negative observed IC then yields a large p-value).
    """
    s, f = _finite_pairs(signal, fwd_return)
    n = len(s)

    obs = information_coefficient(s, f, method=method)
    stat_obs = abs(obs) if two_sided else obs

    if n < 3 or n_perm <= 0:
        # No power: the add-one estimator degenerates to the most conservative
        # possible p-value (1.0), which is the honest answer.
        return ICTest(ic=obs, p_value=1.0, n=n, n_perm=max(n_perm, 0), seed=seed)

    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(s)
        stat = information_coefficient(perm, f, method=method)
        stat_perm = abs(stat) if two_sided else stat
        if stat_perm >= stat_obs:
            count_ge += 1

    p_value = (1 + count_ge) / (n_perm + 1)
    return ICTest(ic=obs, p_value=p_value, n=n, n_perm=n_perm, seed=seed)


# --------------------------------------------------------------------------- #
# Autocorrelation-aware permutation null (circular shift).
# --------------------------------------------------------------------------- #
def circular_shift_pvalue(
    signal: np.ndarray,
    fwd_return: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int = 0,
    method: str = "spearman",
    two_sided: bool = True,
    min_shift: int = 1,
) -> ICTest:
    r"""Autocorrelation-aware permutation test via **circular shifts** of the signal.

    The i.i.d. signal permutation (:func:`permutation_pvalue`) destroys *all* of the
    signal's own structure. For a highly autocorrelated signal (e.g. a positioning
    level with lag-1 autocorr ~0.8) that yields an over-optimistic (too-small)
    p-value, because the permuted nulls are far smoother-free than the real signal
    and so rarely reach the observed |IC|. The correct null here **preserves the
    signal's autocorrelation** by only *rolling* it (a circular / cyclic shift)
    relative to the forward return, which breaks the signal→future alignment while
    leaving the signal's serial correlation intact.

    A shift ``k`` is drawn for each permutation from ``[min_shift, n - min_shift]`` under a
    fixed ``np.random.default_rng(seed)``; the signal is ``np.roll``-ed by ``k`` and the IC
    recomputed on the same finite-pair mask logic. Add-one estimator as in
    :func:`permutation_pvalue`.

    ``two_sided=True`` compares ``|ic|``; ``two_sided=False`` the signed ``ic``.
    """
    s, f = _finite_pairs(signal, fwd_return)
    n = len(s)

    obs = information_coefficient(s, f, method=method)
    stat_obs = abs(obs) if two_sided else obs

    # Need enough room for distinct non-trivial shifts.
    if n < 3 or n_perm <= 0 or n <= 2 * max(min_shift, 1):
        return ICTest(ic=obs, p_value=1.0, n=n, n_perm=max(n_perm, 0), seed=seed)

    rng = np.random.default_rng(seed)
    lo = max(min_shift, 1)
    hi = n - lo  # inclusive upper handled by integers(lo, hi+1)
    count_ge = 0
    for _ in range(n_perm):
        k = int(rng.integers(lo, hi + 1))
        rolled = np.roll(s, k)
        stat = information_coefficient(rolled, f, method=method)
        stat_perm = abs(stat) if two_sided else stat
        if stat_perm >= stat_obs:
            count_ge += 1

    p_value = (1 + count_ge) / (n_perm + 1)
    return ICTest(ic=obs, p_value=p_value, n=n, n_perm=n_perm, seed=seed)


# --------------------------------------------------------------------------- #
# Multi-seed Sharpe aggregation.
# --------------------------------------------------------------------------- #
#: Two-sided 95% Student-t critical values ``t_{0.975, dof}`` for dof 1..30
#: (Abramowitz & Stegun table 26.10; no scipy). Beyond dof 30 the normal 1.96
#: is within ~2% and is used as the fallback.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_quantile_95(dof: int, *, fallback: float = 1.96) -> float:
    """Two-sided 95% Student-t critical value for ``dof`` degrees of freedom.

    Hardcoded lookup for dof 1..30 (e.g. ``t_quantile_95(3) == 3.182``); returns
    ``fallback`` (default the normal 1.96) for dof > 30, where the t and normal
    quantiles differ by < ~2%. Added 2026-07-02 (upstream repair R4; audit
    findings S7 / RA-21).
    """
    d = int(dof)
    if d < 1:
        raise ValueError("dof must be >= 1")
    return _T95[d] if d <= 30 else float(fallback)


@dataclass
class SeedSharpe:
    """Aggregate of a set of per-seed Sharpe ratios.

    Attributes
    ----------
    mean, median:
        Central tendency across seeds.
    lo, hi:
        95% CI for the mean: ``mean ± crit·sd/sqrt(n)`` (sample sd, ``ddof=1``),
        where ``crit`` is the Student-t quantile with ``n-1`` dof (the default
        since 2026-07-02; previously the normal ``z``). With a single seed the
        CI collapses to the point.
    pass_rate:
        Fraction of seeds with Sharpe strictly greater than ``threshold``.
    p10:
        10th percentile of the seed Sharpes (a dispersion floor: the
        10th-percentile seed must still profit).
    n:
        Number of finite seed Sharpes.
    """

    mean: float
    median: float
    lo: float
    hi: float
    pass_rate: float
    p10: float
    n: int


def multiseed_sharpe(
    sharpes: Sequence[float], *, threshold: float = 0.0, z: float = 1.96,
    ci: str = "t",
) -> SeedSharpe:
    """Aggregate per-seed Sharpe ratios into a :class:`SeedSharpe`.

    Pure aggregation — no randomness. Non-finite entries are dropped first. The
    CI is ``mean ± crit·sd/sqrt(n)`` using the *sample* standard deviation
    (``ddof=1``); with ``n == 1`` the standard error is 0 and the interval is
    the point estimate.

    ``ci`` selects the critical value ``crit`` (versioned behavior):

    * ``"t"`` (the default since 2026-07-02 — upstream repair R4; audit findings
      S7 / RA-21): the two-sided 95% Student-t quantile with ``n-1`` dof via
      :func:`t_quantile_95` (e.g. 3.182 at n=4 fold Sharpes); ``z`` is used only
      as the dof>30 fallback.
    * ``"z"`` — the legacy pre-2026-07-02 behavior, ``crit = z`` (1.96) at
      every ``n``: anti-conservative for small ``n``. Any recorded PASS that
      depended on a small-n CI-excludes-zero leg under ``"z"`` needs re-check;
      recorded FAILs remain FAILs a fortiori under ``"t"`` (the t CI is
      strictly wider).

    This single function backs both ``lo > 0`` (CI excludes zero) and
    ``p10 > 0`` (the 10th-percentile seed still profits) style gates. The
    aggregation is not Sharpe-specific: any per-seed scalar statistic can be
    fed through it.
    """
    if ci not in ("t", "z"):
        raise ValueError("ci must be 't' or 'z'")
    arr = np.asarray(list(sharpes), dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)

    if n == 0:
        nan = float("nan")
        return SeedSharpe(
            mean=nan, median=nan, lo=nan, hi=nan, pass_rate=0.0, p10=nan, n=0
        )

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    p10 = float(np.percentile(arr, 10))
    pass_rate = float(np.mean(arr > threshold))

    if n == 1:
        se = 0.0
    else:
        crit = t_quantile_95(n - 1, fallback=z) if ci == "t" else z
        sd = float(np.std(arr, ddof=1))
        se = crit * sd / np.sqrt(n)

    lo = mean - se
    hi = mean + se
    return SeedSharpe(
        mean=mean, median=median, lo=lo, hi=hi, pass_rate=pass_rate, p10=p10, n=n
    )


def sharpe_ci_excludes_zero(sharpes: Sequence[float]) -> bool:
    """True iff the mean-Sharpe CI lies strictly above zero.

    Uses the :func:`multiseed_sharpe` defaults — i.e. the Student-t CI since
    2026-07-02 (upstream repair R4), strictly stricter than the pre-fix z CI at
    n <= 31.
    """
    agg = multiseed_sharpe(sharpes)
    return bool(np.isfinite(agg.lo) and agg.lo > 0.0)
