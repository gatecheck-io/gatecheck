"""Overfitting defenses: the Deflated Sharpe Ratio and PBO.

Lineage: extracted 2026-07-02 from the source research program
(``market_os/strategy/deflation.py``, post-audit), where it implemented the
explicit defenses of López de Prado's Chapter 15 §15.5. Strategy discovery is
where overfitting is most lethal: searching over archetypes × states ×
parameters generates enormous multiplicity. A nominally significant Sharpe
found after many trials is, under the null, very likely a fluke. This module
implements (Bailey & López de Prado 2014; López de Prado 2018):

* the **Deflated Sharpe Ratio (DSR)**, which discounts an observed Sharpe for the number of
  trials, the non-normality (skew/kurtosis) of returns, and the sample length;
* a convenience :func:`deflate` that shrinks a Sharpe by its DSR probability — a single
  number an engine can use to penalize a conditional Sharpe by how likely it is spurious;
* a light **probability of backtest overfitting (PBO)** proxy (full combinatorial purged CV
  is out of scope here).

**Units.** All Sharpe inputs are **per-period** (the same period as the ``n_samples``
observations), *not* annualized — the DSR's ``sqrt(n_samples - 1)`` term presumes the SR is
in the units of the underlying sample. Annualize *after* deflation if desired. Critically,
``sigma_sr`` (the cross-trial SR dispersion that scales ``E[max SR]``) must be on the **same
per-period scale** as ``sharpe``: a per-period SR is small (e.g. a daily SR ≈ 0.05), so a
realistic ``sigma_sr`` is likewise small (e.g. 0.02). The default ``sigma_sr = 1.0`` is the
standardized-SR convention; pass the empirical per-period trial-SR std for calibrated DSRs.

The DSR (Bailey & López de Prado 2014, eqs 7–9) is

.. math::

    \\widehat{SR_0} = E[\\max_N SR]
        \\approx \\sigma_{SR}\\,\\Big[(1-\\gamma)\\,Z^{-1}\\!\\big(1 - \\tfrac{1}{N}\\big)
            + \\gamma\\,Z^{-1}\\!\\big(1 - \\tfrac{1}{N e}\\big)\\Big],

    \\mathrm{DSR} = \\Phi\\!\\left(
        \\frac{(SR - \\widehat{SR_0})\\,\\sqrt{n - 1}}
             {\\sqrt{1 - \\hat\\gamma_3 SR + \\frac{\\hat\\gamma_4 - 1}{4} SR^2}}
    \\right),

with ``gamma`` the Euler–Mascheroni constant, ``N = n_trials``, ``n = n_samples``,
``Z^{-1}`` the standard-normal inverse CDF, ``Phi`` its CDF, ``\\hat\\gamma_3`` the skew and
``\\hat\\gamma_4`` the kurtosis of returns. ``E[max_N SR]`` is the expected maximum Sharpe
across ``N`` independent trials under the null SR = 0; the DSR is the probability the
observed SR exceeds that benchmark, accounting for non-normality and sample length.

Conventions: numpy + stdlib only; deterministic; no scipy (``Phi`` via ``math.erf``,
``Z^{-1}`` via the Acklam rational approximation refined by one Halley step).
References:
Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio." *JPM* 40(5).
López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "normal_cdf",
    "normal_ppf",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "deflate",
    "probability_backtest_overfitting",
    "pbo_from_cv_ranks",
]

# Euler–Mascheroni constant.
_GAMMA = 0.5772156649015329


def normal_cdf(x: float) -> float:
    """Standard normal CDF ``Phi(x)`` via ``math.erf``. ``Phi(0)=0.5``, ``Phi(1.96)≈0.975``."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Standard normal inverse CDF ``Z^{-1}(p)`` (quantile), ``p in (0, 1)``.

    Peter Acklam's rational approximation, refined by a single Halley step against
    :func:`normal_cdf` for near-machine precision. Endpoints map to ``±inf``.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    if p == 0.5:
        return 0.0

    # Coefficients for Acklam's approximation.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    # One Halley refinement step.
    e = normal_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return x


def expected_max_sharpe(n_trials: int, *, sigma_sr: float = 1.0) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent null trials (SR=0).

    Bailey & López de Prado (2014), the order-statistics benchmark

    .. math::

        E[\\max_N SR] \\approx \\sigma_{SR}\\,\\Big[(1-\\gamma)\\,Z^{-1}\\!\\big(1-\\tfrac1N\\big)
            + \\gamma\\,Z^{-1}\\!\\big(1-\\tfrac1{Ne}\\big)\\Big],

    with ``gamma`` Euler–Mascheroni and ``sigma_sr`` the cross-trial SR dispersion (in
    SR-units; ``1.0`` is the convention when trial SRs are standardized). Strictly
    increasing in ``n_trials``: more trials raise the bar an honest strategy must clear.
    ``n_trials = 1`` yields ``0.0`` (no multiplicity).
    """
    n = int(n_trials)
    if n < 1:
        raise ValueError("n_trials must be >= 1")
    if n == 1:
        return 0.0
    term1 = (1.0 - _GAMMA) * normal_ppf(1.0 - 1.0 / n)
    term2 = _GAMMA * normal_ppf(1.0 - 1.0 / (n * math.e))
    return float(sigma_sr) * (term1 + term2)


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_trials: int,
    n_samples: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sigma_sr: float = 1.0,
) -> float:
    """Deflated Sharpe Ratio in ``[0, 1]`` (Bailey & López de Prado 2014).

    The probability that a *per-period* observed ``sharpe`` exceeds the expected maximum
    Sharpe from ``n_trials`` independent null trials, correcting for non-normality and the
    finite sample length ``n_samples``:

    .. math::

        \\mathrm{DSR} = \\Phi\\!\\left(
            \\frac{(SR - E[\\max_N SR])\\,\\sqrt{n - 1}}
                 {\\sqrt{1 - \\gamma_3 SR + \\frac{\\gamma_4 - 1}{4} SR^2}}\\right).

    ``skew`` is ``gamma_3`` and ``kurtosis`` is ``gamma_4`` (non-excess; normal = 3.0).
    The DSR **decreases** as ``n_trials`` grows (a higher benchmark to clear). More
    evidence (larger ``n_samples``) **sharpens** the test: it pushes the DSR toward 1
    when the observed ``sharpe`` exceeds the multiplicity benchmark ``E[max SR]`` and
    toward 0 when it falls short (more confidently a fluke). Guards ``n_samples < 2``
    and ``n_trials < 1`` by raising :class:`ValueError`.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    sr = float(sharpe)
    sr0 = expected_max_sharpe(n_trials, sigma_sr=sigma_sr)
    # Variance of the SR estimator under non-normality (denominator of the z-statistic).
    denom_var = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if denom_var <= 0.0:
        # Degenerate variance estimate: fall back to the normal-returns denominator.
        denom_var = 1.0
    z = (sr - sr0) * math.sqrt(n_samples - 1) / math.sqrt(denom_var)
    return normal_cdf(z)


def deflate(
    sharpe: float,
    *,
    n_trials: int,
    n_samples: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sigma_sr: float = 1.0,
) -> float:
    """Return a *discounted* Sharpe: ``sharpe * DSR``.

    Scales the per-period ``sharpe`` by its Deflated-Sharpe probability — a single-number
    shrinkage of a conditional Sharpe by how likely it is a fluke given the number of
    strategies/states searched (``n_trials``). Because the DSR lies in ``[0, 1]``,
    the result has the same sign as and magnitude no larger than ``|sharpe|``: more trials
    or shorter samples shrink it harder.
    """
    dsr = deflated_sharpe_ratio(
        sharpe,
        n_trials=n_trials,
        n_samples=n_samples,
        skew=skew,
        kurtosis=kurtosis,
        sigma_sr=sigma_sr,
    )
    return float(sharpe) * dsr


def probability_backtest_overfitting(in_sample_ranks) -> float:
    """Simplified probability of backtest overfitting (PBO) proxy in ``[0, 1]``.

    Given an array of the **out-of-sample rank fraction** achieved by the in-sample-best
    strategy across CV splits (``0`` = worst OOS, ``1`` = best OOS, ``0.5`` = OOS median),
    return the fraction of splits in which the IS-best *underperformed* the OOS median —
    i.e. ``rank < 0.5``. A high value means the in-sample winner routinely fails out of
    sample, the signature of overfitting.

    This is a light proxy for the logit-of-rank PBO statistic of the full combinatorial
    purged CV procedure (López de Prado 2018), which is out of scope here. An empty input
    returns the neutral ``0.5``.
    """
    ranks = np.asarray(in_sample_ranks, dtype=float).reshape(-1)
    ranks = ranks[np.isfinite(ranks)]
    if ranks.size == 0:
        return 0.5
    return float(np.mean(ranks < 0.5))


def pbo_from_cv_ranks(oos_rank_fractions) -> float:
    """PBO from per-fold OOS rank fractions of the in-sample-best strategy.

    Thin producer-facing wrapper over :func:`probability_backtest_overfitting`. The input
    is the vector of out-of-sample rank fractions (``0`` = worst OOS, ``1`` = best OOS)
    achieved by the *in-sample-best* strategy on each purged/embargoed walk-forward CV fold
    (:mod:`gatecheck.cv`). Forwards verbatim (no math change), so an empty input returns
    the neutral ``0.5``. Numpy + stdlib only.
    """
    return probability_backtest_overfitting(oos_rank_fractions)
