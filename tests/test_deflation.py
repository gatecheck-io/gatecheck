"""Tests for overfitting defenses: DSR and PBO (gatecheck.deflation).

Ported 2026-07-02 from the source program ``tests/test_strategy_deflation.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gatecheck.deflation import (
    deflate,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    normal_cdf,
    normal_ppf,
    pbo_from_cv_ranks,
    probability_backtest_overfitting,
)


# --- Phi / Z^{-1} ---------------------------------------------------------


def test_normal_cdf_known_values() -> None:
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert normal_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_normal_ppf_known_values() -> None:
    assert normal_ppf(0.5) == pytest.approx(0.0)
    assert normal_ppf(0.975) == pytest.approx(1.96, abs=1e-3)
    assert normal_ppf(0.025) == pytest.approx(-1.96, abs=1e-3)


def test_ppf_cdf_round_trip() -> None:
    for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-9)


def test_ppf_endpoints() -> None:
    assert normal_ppf(0.0) == -math.inf
    assert normal_ppf(1.0) == math.inf


# --- expected_max_sharpe --------------------------------------------------


def test_expected_max_sharpe_increases_with_trials() -> None:
    vals = [expected_max_sharpe(n) for n in (2, 5, 10, 100, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_expected_max_sharpe_single_trial_zero() -> None:
    assert expected_max_sharpe(1) == 0.0


def test_expected_max_sharpe_rejects_zero_trials() -> None:
    with pytest.raises(ValueError):
        expected_max_sharpe(0)


# --- deflated_sharpe_ratio ------------------------------------------------


# Per-period Sharpe and cross-trial SR dispersion (sigma_sr) are kept on a consistent
# per-period scale, so the DSR operates in its sensitive (non-saturated) range.


def test_dsr_in_unit_interval() -> None:
    for n_trials in (1, 5, 50, 500):
        for n_samples in (10, 100, 1000):
            dsr = deflated_sharpe_ratio(
                0.10, n_trials=n_trials, n_samples=n_samples, sigma_sr=0.02
            )
            assert 0.0 <= dsr <= 1.0


def test_dsr_decreases_with_more_trials() -> None:
    vals = [
        deflated_sharpe_ratio(0.10, n_trials=n, n_samples=250, sigma_sr=0.02)
        for n in (1, 5, 25, 100, 1000)
    ]
    assert all(b < a for a, b in zip(vals, vals[1:]))


def test_dsr_increases_with_more_samples() -> None:
    vals = [
        deflated_sharpe_ratio(0.08, n_trials=10, n_samples=n, sigma_sr=0.02)
        for n in (20, 100, 500, 2000)
    ]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_dsr_non_normality_lowers_significance() -> None:
    # Negative skew + fat tails inflate the SR-estimator variance -> lower DSR.
    base = deflated_sharpe_ratio(0.10, n_trials=10, n_samples=250, sigma_sr=0.02)
    fat = deflated_sharpe_ratio(
        0.10, n_trials=10, n_samples=250, sigma_sr=0.02, skew=-1.0, kurtosis=6.0
    )
    assert fat < base


def test_dsr_guards() -> None:
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.5, n_trials=0, n_samples=100)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.5, n_trials=10, n_samples=1)


# --- deflate --------------------------------------------------------------


def test_deflate_shrinks_sharpe() -> None:
    sr = 0.10
    d = deflate(sr, n_trials=50, n_samples=250, sigma_sr=0.02)
    assert 0.0 <= d <= sr
    assert d < sr  # multiplicity strictly shrinks a finite Sharpe


def test_deflate_shrinks_harder_with_more_trials() -> None:
    few = deflate(0.10, n_trials=5, n_samples=250, sigma_sr=0.02)
    many = deflate(0.10, n_trials=500, n_samples=250, sigma_sr=0.02)
    assert many < few


# --- probability_backtest_overfitting -------------------------------------


def test_pbo_hand_example() -> None:
    # 3 of 5 splits have OOS rank below the median -> 0.6.
    ranks = [0.2, 0.4, 0.45, 0.7, 0.9]
    assert probability_backtest_overfitting(ranks) == pytest.approx(0.6)


def test_pbo_all_above_median() -> None:
    assert probability_backtest_overfitting([0.6, 0.8, 0.99]) == 0.0


def test_pbo_all_below_median() -> None:
    assert probability_backtest_overfitting([0.1, 0.2, 0.3]) == 1.0


def test_pbo_empty_is_neutral() -> None:
    assert probability_backtest_overfitting([]) == 0.5


def test_pbo_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    p = probability_backtest_overfitting(rng.random(50))
    assert 0.0 <= p <= 1.0


# --- DSR runtime calibration regression at the deflation boundary ----------


def test_dsr_runtime_calibration_lands_in_sensitive_band() -> None:
    """Per-period SR + a small calibrated sigma_sr lands the DSR in (0.05,0.95),
    while the mis-scaled shape (annualized SR ~1.0 + sigma_sr=0.5) saturates. Pins the
    units contract so the mis-calibration bug class cannot silently return."""
    n_trials, n_samples = 27, 250
    # Realistic synthetic annualized conditional Sharpe; per-period = /sqrt(252).
    ann_sr = 0.6
    pp_sr = ann_sr / math.sqrt(252)
    sigma_sr_pp = 0.02  # a calibrated per-period dispersion

    new = deflated_sharpe_ratio(pp_sr, n_trials=n_trials, n_samples=n_samples,
                                sigma_sr=sigma_sr_pp)
    assert 0.05 < new < 0.95, new

    old = deflated_sharpe_ratio(ann_sr, n_trials=n_trials, n_samples=n_samples,
                                sigma_sr=0.5)
    assert old < 0.05 or old > 0.95, old


# --- pbo_from_cv_ranks wires the PBO proxy ----------------------------------


def test_pbo_from_cv_ranks_forwards_to_proxy() -> None:
    """pbo_from_cv_ranks must equal probability_backtest_overfitting on the same input."""
    ranks = [0.2, 0.4, 0.45, 0.7, 0.9]
    assert pbo_from_cv_ranks(ranks) == pytest.approx(
        probability_backtest_overfitting(ranks))
    assert pbo_from_cv_ranks(ranks) == pytest.approx(0.6)
    # Empty input -> neutral 0.5 (delegated to the proxy).
    assert pbo_from_cv_ranks([]) == 0.5
