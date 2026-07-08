"""gatecheck — calibrated statistical gates for research adjudication.

Every statistical test in this package ships with (or can be run through)
**planted-truth / planted-null certification**: seeded synthetic worlds where the
answer is KNOWN, so you can measure whether your gate actually fires on truth and
stays silent on structured nulls — *tests for the tests*.

Provenance
----------
Extracted 2026-07-02 from the source research program (post-audit): the
audit-verified statistical spine of a ~160-PR falsification-first market
research program that ran to an honest descriptive-not-operational terminus.
These are the instruments that survived two adversarial audits of the program's
own methodology, including the repairs those audits forced (small-n Student-t
CIs, rank-space partials, autocorrelation-aware permutation nulls, gate
calibration itself).

* Audit summary: ``docs/AUDIT_SUMMARY.md`` in the repository.
* Calibration certificates: ``docs/CALIBRATION_CERTIFICATES.md``.
* Full adversarial audit reports of the source program: available on request.

Name confirmed 2026-07-02; the PyPI name ``gatecheck`` is available. Not yet
published (publication is an owner-gated launch step).

Modules
-------
* :mod:`gatecheck.calibration` — the headline: plant truth, plant null, certify any
  boolean gate (``certify``, world generators, ``Certificate``); includes one
  built-in PASS exhibit (``certificate_spearman_partial``).
* :mod:`gatecheck.deflation` — Deflated Sharpe Ratio, expected-max-Sharpe
  multiplicity benchmark, PBO proxy (Bailey & López de Prado).
* :mod:`gatecheck.cv` — purged + embargoed walk-forward cross-validation (pure,
  data-free index math).
* :mod:`gatecheck.significance` — PIT-aligned information coefficient, permutation
  and circular-shift p-values, multi-seed Sharpe aggregation with small-n
  Student-t CIs.
* :mod:`gatecheck.oos` — fit-on-train / score-on-test primitives with train-only
  standardization (ridge, ridge-logistic, OOS R², Brier).
* :mod:`gatecheck.incremental` — incremental OOS R² of added features over a
  baseline, with leaked-oracle power floors.
* :mod:`gatecheck.rank` — rank-space partial correlations (the machinery that
  prevents the OLS-levels-then-Spearman sign-flip bug).

Dependencies: numpy only. Everything is deterministic (seeded) and fully offline.
"""

from __future__ import annotations

from .calibration import (
    Certificate,
    CertifyResult,
    WorldFireRate,
    add_one_rate_ci,
    certificate_spearman_partial,
    certify,
    content_world,
    forward_vol,
    plant_edge_world,
    plant_vol_world,
    static_coupling_world,
    trailing_vol,
)
from .cv import Fold, apply_fold, purged_walk_forward
from .deflation import (
    deflate,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    normal_cdf,
    normal_ppf,
    pbo_from_cv_ranks,
    probability_backtest_overfitting,
)
from .incremental import (
    ORACLE_POWER_MIN,
    IncrementalVerdict,
    incremental_gate,
    materiality_floor,
    oracle_floor_r2,
    seed_incremental_r2,
)
from .oos import (
    brier_score,
    logistic_oos_predict,
    oos_r2,
    ridge_oos_predict,
    standardize_train_test,
)
from .rank import rank_average, residualize, spearman_partial
from .significance import (
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

__version__ = "0.1.0"

__all__ = [
    # calibration (the headline)
    "certify",
    "Certificate",
    "CertifyResult",
    "WorldFireRate",
    "add_one_rate_ci",
    "certificate_spearman_partial",
    "plant_vol_world",
    "static_coupling_world",
    "content_world",
    "plant_edge_world",
    "trailing_vol",
    "forward_vol",
    # deflation
    "deflated_sharpe_ratio",
    "deflate",
    "expected_max_sharpe",
    "normal_cdf",
    "normal_ppf",
    "probability_backtest_overfitting",
    "pbo_from_cv_ranks",
    # cv
    "Fold",
    "purged_walk_forward",
    "apply_fold",
    # significance
    "information_coefficient",
    "align_next_return",
    "ICTest",
    "permutation_pvalue",
    "circular_shift_pvalue",
    "SeedSharpe",
    "multiseed_sharpe",
    "sharpe_ci_excludes_zero",
    "t_quantile_95",
    # oos
    "standardize_train_test",
    "ridge_oos_predict",
    "oos_r2",
    "logistic_oos_predict",
    "brier_score",
    # incremental
    "seed_incremental_r2",
    "oracle_floor_r2",
    "materiality_floor",
    "ORACLE_POWER_MIN",
    "IncrementalVerdict",
    "incremental_gate",
    # rank
    "rank_average",
    "spearman_partial",
    "residualize",
    # meta
    "__version__",
]
