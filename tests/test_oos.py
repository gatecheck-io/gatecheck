"""Tests for the shared OOS primitives (gatecheck.oos).

Ported 2026-07-02 from market_state ``tests/test_evaluation_oos.py``.

The load-bearing properties: train-only normalization (no test stats leak in), an honest
train-mean-referenced OOS R² (can go negative), a fold-size-independent ridge strength (so
adding uninformative features does NOT crater OOS R² through overfit), and a proper Brier.
"""

from __future__ import annotations

import numpy as np
import pytest

from gatecheck.oos import (
    brier_score,
    logistic_oos_predict,
    oos_r2,
    ridge_oos_predict,
    standardize_train_test,
)


# --------------------------------------------------------------------------- #
# standardize_train_test — train-only stats; constant columns are safe.
# --------------------------------------------------------------------------- #
def test_standardize_uses_train_mean_std_only():
    Xtr = np.array([[0.0], [2.0], [4.0]])      # mean 2, std sqrt(8/3)
    Xte = np.array([[2.0], [6.0]])
    ztr, zte = standardize_train_test(Xtr, Xte)
    mu, sd = Xtr.mean(), Xtr.std()
    assert np.allclose(zte.ravel(), (Xte.ravel() - mu) / sd)
    assert np.isclose(ztr.mean(), 0.0)


def test_standardize_constant_column_maps_to_zero():
    Xtr = np.full((5, 1), 3.0)
    Xte = np.array([[3.0], [9.0]])             # even a different test value -> 0 (no train var)
    _, zte = standardize_train_test(Xtr, Xte)
    assert np.allclose(zte, 0.0)


# --------------------------------------------------------------------------- #
# ridge_oos_predict + oos_r2.
# --------------------------------------------------------------------------- #
def test_ridge_recovers_a_clean_linear_signal_oos():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    beta = np.array([1.5, -2.0, 0.5])
    y = X @ beta + 0.05 * rng.normal(size=400)
    pred, ymean = ridge_oos_predict(X[:300], y[:300], X[300:], l2=1e-3)
    assert oos_r2(y[300:], pred, ymean) > 0.95


def test_oos_r2_is_zero_for_train_mean_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full_like(y, 2.5)
    assert oos_r2(y, pred, y_train_mean=2.5) == pytest.approx(0.0)


def test_oos_r2_goes_negative_when_worse_than_train_mean():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([5.0, 5.0, 5.0])           # far from both y and its mean
    assert oos_r2(y, pred, y_train_mean=2.0) < 0.0


def test_ridge_noise_features_do_not_crater_oos_r2():
    """The overfit control the incremental gate depends on: with the fold-size-independent
    penalty, adding pure-noise columns keeps OOS R² near the no-feature baseline, not
    wildly negative."""
    rng = np.random.default_rng(1)
    n = 500
    signal = rng.normal(size=n)
    y = 2.0 * signal + 0.1 * rng.normal(size=n)
    X_sig = signal[:, None]
    X_wide = np.column_stack([signal, rng.normal(size=(n, 12))])  # 12 noise columns
    tr, te = slice(0, 350), slice(350, n)
    p1, m1 = ridge_oos_predict(X_sig[tr], y[tr], X_sig[te], l2=1.0)
    p2, m2 = ridge_oos_predict(X_wide[tr], y[tr], X_wide[te], l2=1.0)
    r2_sig = oos_r2(y[te], p1, m1)
    r2_wide = oos_r2(y[te], p2, m2)
    # noise must not destroy the fit: within a small tolerance of the signal-only model
    assert r2_wide > r2_sig - 0.1


# --------------------------------------------------------------------------- #
# logistic_oos_predict + brier_score.
# --------------------------------------------------------------------------- #
def test_logistic_predicts_separable_labels_oos():
    rng = np.random.default_rng(2)
    x = rng.normal(size=600)
    y = (x > 0).astype(float)
    X = x[:, None]
    prob = logistic_oos_predict(X[:400], y[:400], X[400:], l2=1e-2)
    yte = y[400:]
    # probabilities are valid and discriminate the two classes OOS
    assert np.all((prob > 0.0) & (prob < 1.0))
    assert prob[yte == 1].mean() > prob[yte == 0].mean() + 0.3


def test_brier_score_basics():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y, y) == pytest.approx(0.0)                 # perfect
    assert brier_score(np.full(4, 0.5), y) == pytest.approx(0.25)  # base-rate
    with pytest.raises(ValueError):
        brier_score(np.zeros(3), np.zeros(4))


def test_negative_l2_rejected():
    with pytest.raises(ValueError):
        ridge_oos_predict(np.ones((4, 1)), np.ones(4), np.ones((2, 1)), l2=-1.0)
    with pytest.raises(ValueError):
        logistic_oos_predict(np.ones((4, 1)), np.ones(4), np.ones((2, 1)), l2=-1.0)
