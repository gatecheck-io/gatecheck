"""Incremental OOS R² of added features over a baseline, with oracle power floors.

Lineage: extracted 2026-07-02 from the market_state research program
(``market_os/evaluation/incremental.py`` plus the oracle-floor convention of its
``real_g3`` / ``curve_residual_g3`` harnesses), post-audit.

What this measures
------------------
**Hypothesis.** A block of candidate features ``X_extra`` predicts a target ``y``
beyond a baseline feature block ``X_base`` — i.e. it adds OOS :math:`R^2` over the
baseline, fold by fold, through purged walk-forward CV (:mod:`gatecheck.cv`).

Two honesty devices ship with the increment:

* **The oracle power floor** (:func:`oracle_floor_r2`). Before reading a null
  increment as a FAIL, plant the target itself as the added feature (a deliberately
  leaked *oracle*) and run it through the IDENTICAL purged folds. The oracle's
  increment is the recoverable-R² ceiling: if even the oracle adds ~nothing, the DV
  is unforecastable on this panel and a candidate null is NOT_MEASURABLE, not a
  FAIL. The conventional materiality floor is **10% of the oracle increment**
  (:func:`materiality_floor`): a candidate must recover at least a tenth of what a
  perfect feature recovers.
* **The pass-through diagnostic** (:func:`incremental_gate`). Report the increment
  over TWO nested baselines — a narrow one (the claim) and the full observable set
  (the diagnostic). If the candidate beats the narrow baseline only because it
  smuggles in other observables, it adds nothing over the full set — a *hollow*
  pass, flagged instead of hidden behind a green light.

Pure adjudication math — numpy/stdlib + sibling modules (:mod:`.cv`, :mod:`.oos`,
:mod:`.significance`). PIT alignment and any forward shift of ``y`` are the
caller's responsibility; these functions only slice by fold index arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cv import Fold
from .oos import oos_r2, ridge_oos_predict
from .significance import multiseed_sharpe

__all__ = [
    "seed_incremental_r2",
    "oracle_floor_r2",
    "materiality_floor",
    "ORACLE_POWER_MIN",
    "IncrementalVerdict",
    "incremental_gate",
]

#: Minimum oracle incremental R² for the target to count as forecastable-in-principle
#: on a panel (below it, a candidate null is NOT_MEASURABLE rather than a FAIL).
ORACLE_POWER_MIN = 0.05


def seed_incremental_r2(
    X_base: np.ndarray,
    X_extra: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    *,
    l2: float = 1.0,
) -> float:
    """Mean over folds of the added features' incremental OOS :math:`R^2` for target ``y``.

    For each fold: fit ridge on the **baseline** features and on **baseline + extras**
    (both train-only), score OOS :math:`R^2` on the test block against the train-mean null,
    and take the difference. The per-seed statistic is the mean of those per-fold
    increments. Folds with a test or train block too small to fit are skipped; an empty
    result is ``nan`` (a seed with no usable fold contributes no information).

    ``X_base`` is ``(n, p_base)``, ``X_extra`` is ``(n, p_extra)``, ``y`` is ``(n,)``
    already aligned to the *target* (e.g. realized vol shifted one bar forward).
    PIT alignment and the forward shift are the caller's responsibility; this function only
    slices by the fold index arrays, which are pure integer math.
    """
    X_base = np.atleast_2d(np.asarray(X_base, dtype=float))
    X_extra = np.atleast_2d(np.asarray(X_extra, dtype=float))
    y = np.asarray(y, dtype=float)
    X_full = np.hstack([X_base, X_extra])

    incs: list[float] = []
    for fold in folds:
        tr, te = fold.train, fold.test
        if tr.size < 3 or te.size < 2:
            continue
        if not np.isfinite(y[tr]).all() or not np.isfinite(y[te]).all():
            continue
        pred_b, mean_b = ridge_oos_predict(X_base[tr], y[tr], X_base[te], l2=l2)
        pred_f, mean_f = ridge_oos_predict(X_full[tr], y[tr], X_full[te], l2=l2)
        r2_b = oos_r2(y[te], pred_b, mean_b)
        r2_f = oos_r2(y[te], pred_f, mean_f)
        incs.append(r2_f - r2_b)

    return float(np.mean(incs)) if incs else float("nan")


def oracle_floor_r2(
    X_base: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    *,
    l2: float = 1.0,
) -> float:
    """Leaked-oracle incremental OOS :math:`R^2`: the recoverable ceiling / power floor.

    Runs :func:`seed_incremental_r2` with the target ``y`` ITSELF as the added feature
    (deliberately leaked), through the identical purged folds and ridge. This is the
    mandatory power check before reading a candidate's null increment as a FAIL:

    * oracle increment ``>= ORACLE_POWER_MIN`` — the DV is forecastable-in-principle on
      this panel; a candidate null is a genuine FAIL.
    * oracle increment ``~ 0`` — even a perfect feature recovers nothing (target too
      noisy / folds too short): the candidate verdict is NOT_MEASURABLE, not a FAIL.
    """
    y = np.asarray(y, dtype=float)
    return seed_incremental_r2(X_base, y[:, None], y, folds, l2=l2)


def materiality_floor(oracle_inc: float, *, fraction: float = 0.1,
                      fallback: float = 0.005) -> float:
    """The conventional materiality floor: ``fraction`` (default 10%) of the oracle increment.

    A candidate increment below this floor is immaterial even if statistically nonzero —
    it recovers less than a tenth of what a perfect (leaked) feature recovers. Falls back
    to ``fallback`` when the oracle increment is non-finite or non-positive.
    """
    if np.isfinite(oracle_inc) and oracle_inc > 0:
        return float(fraction) * float(oracle_inc)
    return float(fallback)


@dataclass
class IncrementalVerdict:
    """The incremental-OOS-power verdict over two nested baselines.

    Attributes
    ----------
    passed:
        ``True`` iff the across-seed mean incremental OOS :math:`R^2` **over the narrow
        baseline** has a CI strictly above ``threshold``. Permitted to be False: a red
        verdict is a valid research outcome, not a defect.
    n_seeds:
        Number of finite per-seed increments aggregated.
    target:
        The predicted quantity, echoed so a stored verdict is self-describing.
    mean_incr_base, ci_lo, ci_hi, pass_rate:
        Central tendency / CI / per-seed pass-rate of the increment **over the narrow
        baseline** — the leg the gate turns on.
    mean_incr_obs, obs_ci_lo, obs_ci_hi:
        The increment **over the full observable set** (the pass-through diagnostic). When
        this straddles/under-shoots 0 while the narrow leg passes, the candidate beats the
        narrow baseline only by carrying other observables — a hollow, pass-through pass.
    threshold:
        The decision bar the narrow-baseline leg had to clear.
    """

    passed: bool
    n_seeds: int
    target: str
    mean_incr_base: float
    ci_lo: float
    ci_hi: float
    pass_rate: float
    mean_incr_obs: float
    obs_ci_lo: float
    obs_ci_hi: float
    threshold: float

    @property
    def pass_through(self) -> bool:
        """True when the gate passes over the narrow baseline but NOT over the full
        observable set — the candidate adds nothing beyond the observables it carries."""
        return bool(self.passed and not (np.isfinite(self.obs_ci_lo)
                                          and self.obs_ci_lo > self.threshold))

    @property
    def _band(self) -> str:
        """Where the narrow-baseline CI sits relative to the bar — distinguishing a decisive
        fail (CI entirely below) from an inconclusive one (CI straddles the bar)."""
        if self.passed:
            return "(excludes bar, above)"
        if np.isfinite(self.ci_hi) and self.ci_hi <= self.threshold:
            return "(excludes bar, below)"
        return "(spans bar)"

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            "=" * 64,
            f"INCREMENTAL OOS POWER — extras beat the baseline?  [{verdict}]",
            "=" * 64,
            f"target              : {self.target}",
            f"seeds               : {self.n_seeds}",
            f"threshold           : {self.threshold:+.4f}",
            f"incr R² vs baseline : {self.mean_incr_base:+.4f}  "
            f"95% CI [{self.ci_lo:+.4f}, {self.ci_hi:+.4f}]  {self._band}",
            f"pass-rate (seeds)   : {self.pass_rate:.0%}",
            f"incr R² vs all obs  : {self.mean_incr_obs:+.4f}  "
            f"95% CI [{self.obs_ci_lo:+.4f}, {self.obs_ci_hi:+.4f}]   (pass-through check)",
        ]
        if self.pass_through:
            lines.append("! pass-through: beats the narrow baseline but NOT the full "
                         "observable set —")
            lines.append("  the extras add no fusion value beyond the raw observables.")
        lines.append("=" * 64)
        return "\n".join(lines)


def incremental_gate(
    seed_incr_base,
    seed_incr_obs,
    *,
    threshold: float = 0.0,
    z: float = 1.96,
    target: str = "target",
) -> IncrementalVerdict:
    """Aggregate per-seed incremental OOS :math:`R^2` into an :class:`IncrementalVerdict`.

    ``seed_incr_base`` / ``seed_incr_obs`` are one scalar per seed: that seed's mean
    over OOS folds of the extras' incremental :math:`R^2` over the narrow baseline and
    over the full observable set, respectively (computed via
    :func:`seed_incremental_r2`). Cross-seed aggregation reuses the generic
    :func:`.significance.multiseed_sharpe` (mean / CI / pass-rate — it is not
    Sharpe-specific). The gate **passes** iff the narrow-baseline CI lower bound lies
    strictly above ``threshold``.

    Pure and deterministic: no pipeline, no randomness.
    """
    agg_b = multiseed_sharpe(seed_incr_base, threshold=threshold, z=z)
    agg_o = multiseed_sharpe(seed_incr_obs, threshold=threshold, z=z)
    passed = bool(np.isfinite(agg_b.lo) and agg_b.lo > threshold)
    return IncrementalVerdict(
        passed=passed,
        n_seeds=agg_b.n,
        target=target,
        mean_incr_base=agg_b.mean,
        ci_lo=agg_b.lo,
        ci_hi=agg_b.hi,
        pass_rate=agg_b.pass_rate,
        mean_incr_obs=agg_o.mean,
        obs_ci_lo=agg_o.lo,
        obs_ci_hi=agg_o.hi,
        threshold=float(threshold),
    )
