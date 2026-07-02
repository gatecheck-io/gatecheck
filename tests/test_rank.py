"""Tests for the rank-space partial machinery (gatecheck.rank).

New tests written 2026-07-02 for the extraction; the sign-preservation test pins the
OLS-levels-then-Spearman sign-flip bug class this module exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from gatecheck.rank import rank_average, residualize, spearman_partial


# --------------------------------------------------------------------------- #
# rank_average
# --------------------------------------------------------------------------- #
def test_rank_average_basic_and_ties():
    r = rank_average(np.array([10.0, 30.0, 20.0]))
    assert r.tolist() == [1.0, 3.0, 2.0]
    # ties share the mean of their 1-based rank positions
    r2 = rank_average(np.array([5.0, 1.0, 5.0]))
    assert r2.tolist() == [2.5, 1.0, 2.5]


def test_rank_average_preserves_nan_grid():
    x = np.array([3.0, np.nan, 1.0, 2.0, np.inf])
    r = rank_average(x)
    assert np.isnan(r[1]) and np.isnan(r[4])
    assert r[2] == 1.0 and r[3] == 2.0 and r[0] == 3.0
    # all-NaN input -> all-NaN output, no crash
    assert np.all(np.isnan(rank_average(np.full(4, np.nan))))


# --------------------------------------------------------------------------- #
# residualize
# --------------------------------------------------------------------------- #
def test_residualize_removes_linear_control():
    rng = np.random.default_rng(0)
    c = rng.normal(size=300)
    y = 2.0 * c + rng.normal(size=300)
    res = residualize(y, [c])
    m = np.isfinite(res)
    assert np.corrcoef(res[m], c[m])[0, 1] == pytest.approx(0.0, abs=1e-10)
    assert abs(res[m].mean()) < 1e-10  # intercept included


def test_residualize_nan_grid_and_underpowered():
    rng = np.random.default_rng(1)
    c = rng.normal(size=50)
    y = c + rng.normal(size=50)
    y[5] = np.nan
    c[7] = np.nan
    res = residualize(y, [c])
    assert res.shape == y.shape
    assert np.isnan(res[5]) and np.isnan(res[7])
    # too few finite pairs -> all NaN
    tiny = residualize(np.array([1.0, np.nan, 2.0]), [np.array([1.0, 2.0, 3.0])])
    assert np.all(np.isnan(tiny))


def test_residualize_accepts_2d_controls_matrix():
    rng = np.random.default_rng(2)
    C = rng.normal(size=(200, 2))
    y = C @ np.array([1.0, -2.0]) + rng.normal(size=200)
    res = residualize(y, C)
    m = np.isfinite(res)
    for j in range(2):
        assert abs(np.corrcoef(res[m], C[m, j])[0, 1]) < 1e-8


# --------------------------------------------------------------------------- #
# spearman_partial
# --------------------------------------------------------------------------- #
def test_spearman_partial_recovers_planted_partial():
    rng = np.random.default_rng(3)
    n = 800
    ctrl = rng.normal(size=n)
    target = ctrl + rng.normal(size=n)
    # signal shares the target's residual (given ctrl) with weight ~0.4
    resid = target - ctrl
    sig = 0.4 * resid + np.sqrt(1 - 0.16) * rng.normal(size=n)
    ic, p = spearman_partial(sig, target, [ctrl], n_perm=300, seed=0)
    assert 0.15 <= ic <= 0.55
    assert p < 0.05


def test_spearman_partial_silent_on_control_only_signal():
    """A signal that is a pure (noisy) function of the control must show ~no partial."""
    rng = np.random.default_rng(4)
    n = 800
    ctrl = rng.normal(size=n)
    target = ctrl + rng.normal(size=n)
    sig = 0.9 * ctrl + 0.1 * rng.normal(size=n)   # control-only content
    ic, p = spearman_partial(sig, target, [ctrl], n_perm=300, seed=0)
    assert abs(ic) < 0.1
    assert p > 0.05


def test_spearman_partial_keeps_binary_carrier_sign():
    """The sign-flip regression: a binary carrier whose TRUE partial association with the
    target (given a continuous control it is strongly correlated with) is NEGATIVE must
    read NEGATIVE. The buggy OLS-residualize-on-levels-then-rank pipeline flipped this
    class of carrier (upstream: read +0.22, true -0.16)."""
    rng = np.random.default_rng(5)
    n = 1000
    ctrl = rng.normal(size=n)
    sig = (ctrl + 0.8 * rng.normal(size=n) > 0).astype(float)  # binary, control-correlated
    # target rises with the control but FALLS with the carrier given the control
    target = 1.0 * ctrl - 0.8 * sig + 0.5 * rng.normal(size=n)
    ic, p = spearman_partial(sig, target, [ctrl], n_perm=300, seed=0)
    assert ic < -0.05, f"binary carrier sign not preserved: ic={ic:+.3f}"
    assert p < 0.05


def test_spearman_partial_underpowered_returns_nan():
    ic, p = spearman_partial(np.arange(10.0), np.arange(10.0), [np.arange(10.0)],
                             n_perm=50, seed=0)
    assert np.isnan(ic) and p == 1.0


def test_spearman_partial_deterministic():
    rng = np.random.default_rng(6)
    n = 300
    ctrl = rng.normal(size=n)
    target = ctrl + rng.normal(size=n)
    sig = rng.normal(size=n)
    a = spearman_partial(sig, target, [ctrl], n_perm=200, seed=42)
    b = spearman_partial(sig, target, [ctrl], n_perm=200, seed=42)
    assert a == b
