"""Tests for gatecheck.significance (IC, permutation nulls, multi-seed Sharpe CIs).

Ported 2026-07-02 from market_state ``tests/test_evaluation_significance.py``, plus
tests for the small-n Student-t CI (upstream repair R4) and the circular-shift null.
"""

from __future__ import annotations

import numpy as np
import pytest

from gatecheck.significance import (
    ICTest,
    SeedSharpe,
    align_next_return,
    circular_shift_pvalue,
    information_coefficient,
    multiseed_sharpe,
    permutation_pvalue,
    sharpe_ci_excludes_zero,
    t_quantile_95,
)


# --------------------------------------------------------------------------- #
# information_coefficient
# --------------------------------------------------------------------------- #
def test_ic_perfect_signal():
    rng = np.random.default_rng(7)
    fwd = rng.normal(size=200)

    # signal == fwd_return  ->  IC ~ +1
    ic_pos = information_coefficient(fwd.copy(), fwd, method="spearman")
    assert ic_pos == pytest.approx(1.0, abs=1e-9)

    # signal == -fwd_return ->  IC ~ -1
    ic_neg = information_coefficient(-fwd, fwd, method="spearman")
    assert ic_neg == pytest.approx(-1.0, abs=1e-9)

    # independent noise -> IC ~ 0 (loose band; finite sample)
    noise = rng.normal(size=200)
    ic_noise = information_coefficient(noise, fwd, method="spearman")
    assert abs(ic_noise) < 0.2

    # pearson path on a perfectly linear relation -> +1
    ic_pearson = information_coefficient(2.0 * fwd + 3.0, fwd, method="pearson")
    assert ic_pearson == pytest.approx(1.0, abs=1e-9)


def test_ic_nan_safe():
    s = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.inf, 7.0])
    f = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0])
    # Pairwise-finite indices: 0,1,4,6 -> monotone increasing -> IC == 1.
    ic = information_coefficient(s, f, method="spearman")
    assert ic == pytest.approx(1.0, abs=1e-9)

    # Fewer than 3 finite pairs -> 0.0
    s2 = np.array([1.0, np.nan, np.nan])
    f2 = np.array([1.0, 2.0, 3.0])
    assert information_coefficient(s2, f2) == 0.0

    # Zero variance on one side -> 0.0
    s3 = np.array([5.0, 5.0, 5.0, 5.0])
    f3 = np.array([1.0, 2.0, 3.0, 4.0])
    assert information_coefficient(s3, f3) == 0.0

    # All-NaN -> 0.0, no crash.
    nan_arr = np.full(5, np.nan)
    assert information_coefficient(nan_arr, nan_arr) == 0.0


def test_ic_invalid_method():
    with pytest.raises(ValueError):
        information_coefficient(np.arange(5.0), np.arange(5.0), method="kendall")


def test_ic_handles_ties():
    # Spearman with ties: average-rank handling. Two identical "blocks" that are
    # monotone with the target should still give a high positive IC.
    s = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    f = np.array([10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
    ic = information_coefficient(s, f, method="spearman")
    assert ic > 0.9


# --------------------------------------------------------------------------- #
# align_next_return
# --------------------------------------------------------------------------- #
def test_align_next_return_shapes():
    sig = np.array([10.0, 20.0, 30.0, 40.0])
    ret = np.array([0.1, 0.2, 0.3, 0.4])
    s_al, r_al = align_next_return(sig, ret)

    # len-1 outputs.
    assert len(s_al) == 3
    assert len(r_al) == 3

    # PIT shift verified by hand: signal[t] pairs with ret[t+1].
    np.testing.assert_array_equal(s_al, np.array([10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(r_al, np.array([0.2, 0.3, 0.4]))

    # The last signal (40.0) and the first return (0.1) are dropped.
    assert 40.0 not in s_al
    assert 0.1 not in r_al


def test_align_next_return_degenerate():
    empty_s, empty_r = align_next_return(np.array([1.0]), np.array([1.0]))
    assert len(empty_s) == 0 and len(empty_r) == 0

    empty_s2, empty_r2 = align_next_return(np.array([]), np.array([]))
    assert len(empty_s2) == 0 and len(empty_r2) == 0


def test_align_then_ic_is_pit():
    # A signal that equals the *contemporaneous* return looks predictive only if
    # you cheat (no shift). After the PIT shift it predicts the next bar, which
    # is unrelated for i.i.d. returns -> IC near zero.
    rng = np.random.default_rng(3)
    ret = rng.normal(size=300)
    signal = ret.copy()  # signal[t] == ret[t]  (contemporaneous)
    s_al, r_al = align_next_return(signal, ret)
    ic = information_coefficient(s_al, r_al)
    assert abs(ic) < 0.2  # no genuine next-bar information


# --------------------------------------------------------------------------- #
# permutation_pvalue
# --------------------------------------------------------------------------- #
def test_permutation_deterministic():
    rng = np.random.default_rng(11)
    fwd = rng.normal(size=120)
    sig = 0.6 * fwd + 0.4 * rng.normal(size=120)

    a = permutation_pvalue(sig, fwd, n_perm=300, seed=42)
    b = permutation_pvalue(sig, fwd, n_perm=300, seed=42)
    assert isinstance(a, ICTest)
    assert a == b  # dataclass equality: identical seed -> identical result
    assert a.ic == b.ic and a.p_value == b.p_value and a.n == b.n

    # Different seed should (generally) move the p-value but keep the IC fixed.
    c = permutation_pvalue(sig, fwd, n_perm=300, seed=99)
    assert c.ic == pytest.approx(a.ic, abs=1e-12)


def test_permutation_low_p_for_real_signal_high_for_noise():
    rng = np.random.default_rng(5)
    fwd = rng.normal(size=250)
    real = 0.7 * fwd + 0.3 * rng.normal(size=250)  # genuinely informative
    noise = rng.normal(size=250)  # independent

    real_test = permutation_pvalue(real, fwd, n_perm=500, seed=0)
    noise_test = permutation_pvalue(noise, fwd, n_perm=500, seed=0)

    assert real_test.ic > 0.3
    assert real_test.p_value < 0.05  # real signal rejects the null
    assert noise_test.p_value > 0.05  # noise does not


def test_p_in_range():
    rng = np.random.default_rng(1)
    fwd = rng.normal(size=80)
    sig = rng.normal(size=80)
    n_perm = 200
    t = permutation_pvalue(sig, fwd, n_perm=n_perm, seed=0)
    lo = 1.0 / (n_perm + 1)
    assert lo <= t.p_value <= 1.0
    assert t.n_perm == n_perm
    assert t.n == 80

    # A perfectly informative signal hits the floor p-value.
    perfect = permutation_pvalue(fwd.copy(), fwd, n_perm=n_perm, seed=0)
    assert perfect.p_value == pytest.approx(lo, abs=1e-12)

    # Degenerate (too few pairs) -> conservative p == 1.0, no crash.
    deg = permutation_pvalue(np.array([1.0, np.nan]), np.array([1.0, 2.0]),
                             n_perm=100, seed=0)
    assert deg.p_value == 1.0
    assert deg.n < 3


def test_two_sided_vs_one_sided_differ_on_negative_ic():
    rng = np.random.default_rng(2)
    fwd = rng.normal(size=300)
    # Strongly *negatively* related signal.
    neg = -0.8 * fwd + 0.2 * rng.normal(size=300)

    one = permutation_pvalue(neg, fwd, n_perm=500, seed=0, two_sided=False)
    two = permutation_pvalue(neg, fwd, n_perm=500, seed=0, two_sided=True)

    assert one.ic < 0  # negative association
    # Two-sided detects the (strong) |association| -> small p.
    assert two.p_value < 0.05
    # One-sided (positive-only) cannot reject for a negative IC -> large p.
    assert one.p_value > 0.5
    assert one.p_value > two.p_value


# --------------------------------------------------------------------------- #
# circular_shift_pvalue — the autocorrelation-aware null
# --------------------------------------------------------------------------- #
def test_circular_shift_stricter_than_iid_for_persistent_null_pair():
    """Two INDEPENDENT persistent (AR(1) phi=0.95) series — a null with the nuisance
    geometry that fools the i.i.d. permutation test. The i.i.d. null destroys the
    signal's autocorrelation, understates the null |IC| dispersion of two persistent
    series, and falsely rejects; the circular-shift null preserves the autocorrelation
    and stays silent."""
    def ar1(n, phi, rng):
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.standard_normal()
        return x

    n = 400
    false_rejects_iid = 0
    for s in (0, 1, 2, 5, 6):
        rng = np.random.default_rng(s)
        sig = ar1(n, 0.95, rng)
        fwd = ar1(n, 0.95, rng)  # independent of sig, but also persistent

        iid = permutation_pvalue(sig, fwd, n_perm=400, seed=0)
        shift = circular_shift_pvalue(sig, fwd, n_perm=400, seed=0)
        assert shift.ic == pytest.approx(iid.ic, abs=1e-12)  # same observed statistic
        assert shift.p_value > iid.p_value                   # shift null is stricter here
        assert shift.p_value > 0.05                          # and never rejects the null
        false_rejects_iid += iid.p_value < 0.05

    # The i.i.d. null demonstrably over-rejects on this geometry (the bug class the
    # circular-shift null exists for).
    assert false_rejects_iid >= 3


def test_circular_shift_detects_real_signal():
    rng = np.random.default_rng(6)
    fwd = rng.normal(size=300)
    real = 0.7 * fwd + 0.3 * rng.normal(size=300)
    t = circular_shift_pvalue(real, fwd, n_perm=400, seed=0)
    assert t.p_value < 0.05


def test_circular_shift_degenerate_returns_conservative_p():
    t = circular_shift_pvalue(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                              n_perm=100, seed=0)
    assert t.p_value == 1.0


# --------------------------------------------------------------------------- #
# t_quantile_95 — the small-n repair (upstream R4)
# --------------------------------------------------------------------------- #
def test_t_quantile_known_values():
    assert t_quantile_95(1) == pytest.approx(12.706)
    assert t_quantile_95(3) == pytest.approx(3.182)
    assert t_quantile_95(4) == pytest.approx(2.776)
    assert t_quantile_95(30) == pytest.approx(2.042)
    assert t_quantile_95(31) == pytest.approx(1.96)     # fallback beyond the table
    assert t_quantile_95(100, fallback=2.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        t_quantile_95(0)


def test_small_n_t_ci_is_wider_than_z_ci():
    """The repair's contract: at small n the default t CI is strictly wider than the
    legacy z CI, so CI-excludes-zero gates got stricter, never looser."""
    sharpes = [0.2, 0.5, 0.3, 0.4]  # n=4 -> dof=3 -> crit 3.182 vs 1.96
    t_agg = multiseed_sharpe(sharpes, ci="t")
    z_agg = multiseed_sharpe(sharpes, ci="z")
    assert t_agg.mean == pytest.approx(z_agg.mean)
    assert t_agg.lo < z_agg.lo
    assert t_agg.hi > z_agg.hi
    # Ratio of half-widths equals the critical-value ratio.
    assert (t_agg.hi - t_agg.mean) / (z_agg.hi - z_agg.mean) == pytest.approx(
        3.182 / 1.96, rel=1e-9)


def test_multiseed_rejects_bad_ci_mode():
    with pytest.raises(ValueError):
        multiseed_sharpe([0.1, 0.2], ci="bootstrap")


# --------------------------------------------------------------------------- #
# multiseed_sharpe / sharpe_ci_excludes_zero
# --------------------------------------------------------------------------- #
def test_multiseed_ci_excludes_zero_when_strong():
    # Tight cluster well above zero -> CI excludes 0.
    strong = [0.55, 0.60, 0.62, 0.58, 0.65, 0.61, 0.59, 0.57, 0.63, 0.60]
    agg = multiseed_sharpe(strong)
    assert isinstance(agg, SeedSharpe)
    assert agg.lo > 0.0
    assert sharpe_ci_excludes_zero(strong) is True

    # A cluster straddling zero -> CI includes 0.
    weak = [-0.3, 0.4, -0.2, 0.5, -0.4, 0.3, 0.1, -0.1, 0.2, -0.05]
    assert sharpe_ci_excludes_zero(weak) is False
    assert multiseed_sharpe(weak).lo <= 0.0

    # Empty input -> not significant, no crash.
    assert sharpe_ci_excludes_zero([]) is False
    empty_agg = multiseed_sharpe([])
    assert empty_agg.n == 0
    assert np.isnan(empty_agg.mean)


def test_multiseed_p10_passrate_hand_example():
    # Hand-computable: 0.1, 0.2, ..., 1.0 (ten values).
    sharpes = [round(0.1 * k, 10) for k in range(1, 11)]
    agg = multiseed_sharpe(sharpes, threshold=0.0)

    assert agg.n == 10
    assert agg.mean == pytest.approx(0.55, abs=1e-9)
    assert agg.median == pytest.approx(0.55, abs=1e-9)
    # numpy linear percentile: p10 of 0.1..1.0 = 0.19.
    assert agg.p10 == pytest.approx(0.19, abs=1e-9)
    # All ten > 0 -> pass_rate 1.0.
    assert agg.pass_rate == pytest.approx(1.0, abs=1e-12)

    # pass_rate with a higher threshold: values strictly > 0.5 are 0.6..1.0 (5/10).
    agg2 = multiseed_sharpe(sharpes, threshold=0.5)
    assert agg2.pass_rate == pytest.approx(0.5, abs=1e-12)

    # p10 floor: 10th-percentile seed still profitable here.
    assert agg.p10 > 0.0

    # Single seed: CI collapses to the point.
    one = multiseed_sharpe([0.42])
    assert one.n == 1
    assert one.lo == pytest.approx(0.42, abs=1e-12)
    assert one.hi == pytest.approx(0.42, abs=1e-12)
    assert one.mean == pytest.approx(0.42, abs=1e-12)


def test_multiseed_drops_nonfinite():
    vals = [0.5, np.nan, 0.6, np.inf, 0.55, -np.inf]
    agg = multiseed_sharpe(vals)
    assert agg.n == 3
    assert agg.mean == pytest.approx((0.5 + 0.6 + 0.55) / 3.0, abs=1e-12)
