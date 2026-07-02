"""Tests for incremental OOS R² + oracle floors (gatecheck.incremental).

Ported 2026-07-02 from market_state ``tests/test_evaluation_incremental.py`` (the pure
helper-math arm; the upstream end-to-end pipeline smoke tests do not apply here), plus
tests for the oracle power floor.
"""

from __future__ import annotations

import numpy as np

from gatecheck.cv import purged_walk_forward
from gatecheck.incremental import (
    ORACLE_POWER_MIN,
    IncrementalVerdict,
    incremental_gate,
    materiality_floor,
    oracle_floor_r2,
    seed_incremental_r2,
)


def _folds(n: int):
    return purged_walk_forward(n, n_splits=5, embargo=10)


# --------------------------------------------------------------------------- #
# seed_incremental_r2 — informative > 0, noise ~ 0, shuffled ~ 0.
# --------------------------------------------------------------------------- #
def test_informative_added_features_have_positive_increment():
    rng = np.random.default_rng(0)
    n = 600
    base = rng.normal(size=(n, 3))
    extra = rng.normal(size=(n, 2))
    # target depends on BOTH base and extra -> extra adds genuine OOS power over base
    y = base @ np.array([1.0, -1.0, 0.5]) + extra @ np.array([2.0, 1.5]) \
        + 0.2 * rng.normal(size=n)
    inc = seed_incremental_r2(base, extra, y, _folds(n), l2=1.0)
    assert inc > 0.05


def test_noise_added_features_have_near_zero_increment():
    rng = np.random.default_rng(1)
    n = 600
    base = rng.normal(size=(n, 3))
    y = base @ np.array([1.0, -1.0, 0.5]) + 0.2 * rng.normal(size=n)
    noise = rng.normal(size=(n, 8))
    inc = seed_incremental_r2(base, noise, y, _folds(n), l2=1.0)
    assert abs(inc) < 0.05            # no real lift, and overfit control keeps it from sinking


def test_label_shuffle_destroys_increment():
    """A PIT/edge sanity: if the added features' relationship to y is shuffled away, their
    incremental power collapses to ~0 even though they were informative unshuffled."""
    rng = np.random.default_rng(2)
    n = 600
    base = rng.normal(size=(n, 2))
    extra = rng.normal(size=(n, 2))
    y = base @ np.array([1.0, -1.0]) + extra @ np.array([2.0, 1.0]) + 0.2 * rng.normal(size=n)
    perm = rng.permutation(n)
    inc_shuf = seed_incremental_r2(base, extra[perm], y, _folds(n), l2=1.0)
    assert abs(inc_shuf) < 0.05


# --------------------------------------------------------------------------- #
# oracle_floor_r2 + materiality_floor — the power check.
# --------------------------------------------------------------------------- #
def test_oracle_floor_is_large_on_a_forecastable_target():
    """The leaked oracle (y itself as the added feature) must recover most of the target's
    variance over a weak baseline — the recoverable ceiling."""
    rng = np.random.default_rng(3)
    n = 600
    base = rng.normal(size=(n, 2))            # weakly related baseline
    y = 0.3 * base[:, 0] + rng.normal(size=n)
    oracle = oracle_floor_r2(base, y, _folds(n), l2=1e-3)
    assert oracle > ORACLE_POWER_MIN
    assert oracle > 0.5                        # the oracle is nearly perfect by construction


def test_oracle_floor_is_a_ceiling_for_real_candidates():
    """No honest candidate beats the leaked oracle on the same folds."""
    rng = np.random.default_rng(4)
    n = 600
    base = rng.normal(size=(n, 2))
    extra = rng.normal(size=(n, 2))
    y = base @ np.array([0.5, -0.5]) + extra @ np.array([1.0, 0.8]) + 0.5 * rng.normal(size=n)
    folds = _folds(n)
    cand = seed_incremental_r2(base, extra, y, folds, l2=1e-3)
    oracle = oracle_floor_r2(base, y, folds, l2=1e-3)
    assert oracle > cand > 0.0


def test_materiality_floor_is_ten_percent_of_oracle_with_fallback():
    import pytest

    assert materiality_floor(0.40) == pytest.approx(0.04)
    assert materiality_floor(0.40, fraction=0.2) == pytest.approx(0.08)
    # non-positive / non-finite oracle -> fallback
    assert materiality_floor(0.0) == 0.005
    assert materiality_floor(float("nan")) == 0.005
    assert materiality_floor(-0.1, fallback=0.01) == 0.01


# --------------------------------------------------------------------------- #
# incremental_gate — decision logic + pass-through flag.
# --------------------------------------------------------------------------- #
def test_gate_passes_on_reliably_positive_base_increments():
    base_incs = [0.10, 0.12, 0.09, 0.11, 0.10]
    obs = [0.08, 0.09, 0.07, 0.08, 0.085]
    v = incremental_gate(base_incs, obs)
    assert isinstance(v, IncrementalVerdict)
    assert v.passed is True
    assert v.ci_lo > 0.0
    assert v.pass_through is False        # also beats the full-observable baseline


def test_gate_fails_on_zero_centered_increments():
    base_incs = [0.10, -0.12, 0.03, -0.08, -0.01]
    v = incremental_gate(base_incs, base_incs)
    assert v.passed is False
    assert v.ci_lo <= 0.0 <= v.ci_hi


def test_pass_through_flag_when_beats_base_but_not_observables():
    """Extras beat the narrow baseline but NOT the full observable set -> a hollow
    pass-through pass."""
    base_incs = [0.10, 0.12, 0.09, 0.11, 0.10]  # clearly > 0
    obs = [-0.01, 0.00, -0.02, 0.01, -0.01]     # straddles 0
    v = incremental_gate(base_incs, obs)
    assert v.passed is True
    assert v.pass_through is True
    assert "pass-through" in v.summary()


def test_summary_band_distinguishes_decisive_below_from_spanning():
    """A FAIL whose CI is entirely below the bar reads as a decisive 'below bar'."""
    decisive = incremental_gate([-0.05, -0.06, -0.055, -0.052, -0.058], [0.0] * 5)
    assert decisive.passed is False and decisive.ci_hi < 0.0
    assert "below" in decisive.summary()
    spanning = incremental_gate([0.10, -0.12, 0.03, -0.08, -0.01], [0.0] * 5)
    assert "spans" in spanning.summary()


def test_gate_echoes_target_and_threshold():
    v = incremental_gate([0.1, 0.2], [0.0, 0.1], target="next_realized_vol",
                         threshold=0.01)
    assert v.target == "next_realized_vol"
    assert v.threshold == 0.01
    assert "next_realized_vol" in v.summary()
