"""Pure OOS fit-on-train / score-on-test primitives.

Lineage: extracted 2026-07-02 from the source research program
(``market_os/evaluation/oos.py``, post-audit), where these were the shared leaf
math for the incremental-OOS-R² and regime-transition-Brier falsification gates.

Every routine here is **fit-on-train / score-on-test with TRAIN-ONLY normalization**, so a
fold's test score never sees its own statistics — the off-by-one leak these gates exist to
guard against cannot enter through the estimator. Invariant: ``numpy`` + stdlib only (no
``scipy``); this module imports nothing from the rest of the package.

The two estimators are deliberately the simplest honest choice:

* **Ridge regression** (closed-form normal equations, intercept unpenalized) for a
  continuous target. Regularization keeps the fit stable when a fold's train block is
  short or features collinear; an unpenalized intercept keeps the train-mean null nested.
* **Ridge logistic regression** (Newton/IRLS, intercept unpenalized) for a binary
  event, scored by Brier. A few Newton steps converge on these low-dimensional,
  well-conditioned designs; the L2 term tames the separable-fold case.

A falsification gate does not need DeepSurv or a GNN to be honest — it needs a leak-free
OOS comparison the verdict can stand on. These primitives are that, and no more.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "standardize_train_test",
    "ridge_oos_predict",
    "oos_r2",
    "logistic_oos_predict",
    "brier_score",
]


def _as2d(X: np.ndarray) -> np.ndarray:
    """Coerce to a 2-D ``(n, p)`` float design (a 1-D vector becomes one column)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    elif X.ndim != 2:
        raise ValueError(f"design must be 1-D or 2-D, got {X.ndim}-D")
    return X


def standardize_train_test(
    Xtr: np.ndarray, Xte: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """z-score ``Xte`` using **train** column mean/std (PIT: no test stats leak in).

    Columns whose train std is ~0 are mapped to all-zeros (a constant feature carries no
    information and must not blow up). Both inputs are coerced to 2-D ``(n, p)``.
    """
    Xtr = _as2d(Xtr)
    Xte = _as2d(Xte)
    if Xtr.shape[1] != Xte.shape[1]:
        raise ValueError(
            f"train/test feature counts differ: {Xtr.shape[1]} vs {Xte.shape[1]}"
        )
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    safe = sd >= 1e-12
    sd_safe = np.where(safe, sd, 1.0)
    ztr = np.where(safe, (Xtr - mu) / sd_safe, 0.0)
    zte = np.where(safe, (Xte - mu) / sd_safe, 0.0)
    return ztr, zte


def _ridge_coef(Xz: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Ridge normal-equations solve on a standardized design, intercept **unpenalized**.

    ``Xz`` carries no intercept column; one is prepended here and excluded from the L2
    penalty (so the train-mean null is nested and a zero-signal fit collapses to ``ȳ``).
    """
    n = Xz.shape[0]
    Xd = np.hstack([np.ones((n, 1)), Xz])
    p1 = Xd.shape[1]
    pen = np.eye(p1)
    pen[0, 0] = 0.0  # do not penalize the intercept
    A = Xd.T @ Xd + float(l2) * pen
    return np.linalg.solve(A, Xd.T @ np.asarray(y, dtype=float))


def ridge_oos_predict(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, *, l2: float = 1.0
) -> tuple[np.ndarray, float]:
    """Fit ridge on (``Xtr``, ``ytr``); return ``(test_predictions, train_mean_of_y)``.

    Standardization uses train stats only; the returned ``train_mean_of_y`` is the null
    predictor :func:`oos_r2` scores against (so OOS :math:`R^2` is honestly referenced to
    information available at train time, never the test mean).
    """
    if l2 < 0.0:
        raise ValueError(f"l2 must be >= 0, got {l2}")
    ytr = np.asarray(ytr, dtype=float)
    ztr, zte = standardize_train_test(Xtr, Xte)
    # Scale the penalty by n_train so ``l2`` is a fold-size-independent ridge *strength*:
    # the standardized Gram diagonal grows like n, so an absolute penalty would vanish on a
    # large fold and dominate on a small one — making per-fold increments incomparable and
    # letting wide designs overfit on long folds. ``l2 * n`` shrinks each coefficient by a
    # factor ~1/(1+l2) regardless of n (the overfit control the incremental-R² gate
    # depends on).
    w = _ridge_coef(ztr, ytr, l2 * ztr.shape[0])
    Xd_te = np.hstack([np.ones((zte.shape[0], 1)), zte])
    return Xd_te @ w, float(ytr.mean())


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: float) -> float:
    """Out-of-sample :math:`R^2` against the **train mean** as the null predictor.

    ``1 - SS_res / SS_tot`` with ``SS_tot = Σ(y_true - y_train_mean)²``. Can be negative
    (a model worse than predicting the train mean) — that is a meaningful, honest outcome,
    not an error. Returns 0.0 when the test target is constant at the train mean (no
    variance to explain).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(y_train_mean)) ** 2))
    if ss_tot < 1e-18:
        return 0.0
    return 1.0 - ss_res / ss_tot


def logistic_oos_predict(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    l2: float = 1.0,
    iters: int = 50,
) -> np.ndarray:
    """Ridge logistic (Newton/IRLS) fit on train; return predicted **probabilities** on test.

    Intercept unpenalized; train-only standardization. The L2 term (``l2 > 0``) keeps the
    Hessian well-conditioned and the step finite even when a train fold is perfectly
    separable (every label 0 or 1), which is common in short regime-stratified blocks.
    Probabilities are returned in the open interval (clamped away from {0,1}) so a
    downstream log-loss is finite; Brier needs no clamp but inherits it harmlessly.
    """
    if l2 < 0.0:
        raise ValueError(f"l2 must be >= 0, got {l2}")
    ytr = np.asarray(ytr, dtype=float)
    ztr, zte = standardize_train_test(Xtr, Xte)
    n, p = ztr.shape
    Xd = np.hstack([np.ones((n, 1)), ztr])
    p1 = p + 1
    pen = np.eye(p1)
    pen[0, 0] = 0.0
    l2n = float(l2) * n  # fold-size-independent ridge strength (see ridge_oos_predict)
    w = np.zeros(p1)
    for _ in range(int(iters)):
        eta = np.clip(Xd @ w, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        Wd = np.clip(mu * (1.0 - mu), 1e-6, None)
        H = Xd.T @ (Wd[:, None] * Xd) + l2n * pen
        grad = Xd.T @ (ytr - mu) - l2n * (pen @ w)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    Xd_te = np.hstack([np.ones((zte.shape[0], 1)), zte])
    eta_te = np.clip(Xd_te @ w, -30.0, 30.0)
    prob = 1.0 / (1.0 + np.exp(-eta_te))
    return np.clip(prob, 1e-6, 1.0 - 1e-6)


def brier_score(prob: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of a probabilistic prediction: ``mean((prob - y)²)``.

    A strictly proper score for a binary outcome (lower is better; 0 is perfect). It
    rewards both calibration and discrimination and, unlike a rank-IC, penalizes an
    over-confident wrong probability.
    """
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=float)
    if prob.shape != y.shape:
        raise ValueError(f"prob/y shape mismatch: {prob.shape} vs {y.shape}")
    if prob.size == 0:
        return float("nan")
    return float(np.mean((prob - y) ** 2))
