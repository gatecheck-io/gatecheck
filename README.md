# gatecheck

Calibrated statistical gates for research adjudication — **every test ships with
planted-truth / planted-null certification**. Tests for the tests.

> Working title: `gatecheck` — PyPI name availability has not been checked.

## Why

Backtest and signal-research pipelines are full of *gates*: "the Sharpe CI excludes
zero", "the partial IC is significant", "the feature adds OOS R²". A gate that has
never been run against a world where the truth is KNOWN is an uncalibrated
instrument — you don't know its power (does it fire on planted truth?) or its size
(does it stay silent on structured nulls that share the carrier's persistence,
coupling, and smoothing geometry?).

`gatecheck` bundles:

- **`gatecheck.calibration`** — the headline. Seeded, offline world generators with
  plantable truth (`plant_vol_world`, `content_world`, `plant_edge_world`) and
  structured nulls (`static_coupling_world`), plus a generic `certify()` runner:
  bring your gate function, get a `Certificate` with per-world fire rates and
  binomial CIs. One built-in PASS exhibit: `certificate_spearman_partial`.
- **`gatecheck.deflation`** — Deflated Sharpe Ratio, expected-max-Sharpe
  multiplicity benchmark, PBO proxy (Bailey & López de Prado 2014; López de
  Prado 2018). Per-period units, calibrated `sigma_sr`.
- **`gatecheck.cv`** — purged + embargoed walk-forward cross-validation. Pure,
  data-free integer index math: the splitter *cannot* leak.
- **`gatecheck.significance`** — PIT-aligned information coefficient, permutation
  and autocorrelation-aware circular-shift p-values, multi-seed Sharpe aggregation
  with small-n Student-t CIs (not the anti-conservative z at n=4).
- **`gatecheck.oos`** — fit-on-train / score-on-test primitives with train-only
  standardization (ridge, ridge-logistic, train-mean-referenced OOS R², Brier).
- **`gatecheck.incremental`** — incremental OOS R² of added features over a
  baseline, with a leaked-oracle power floor (if even the oracle recovers nothing,
  a null is NOT_MEASURABLE, not a FAIL) and a pass-through diagnostic.
- **`gatecheck.rank`** — rank-space partial correlations. Ranks FIRST, then
  residualizes in rank space — preventing the OLS-levels-then-Spearman pipeline
  that spuriously sign-flips binary/discrete carriers.

## Install

```bash
pip install -e .          # from a source checkout
python -m pytest tests -q # offline, deterministic
```

Dependencies: `numpy` only. Python >= 3.10. No network, ever.

## Quick start: certify your own gate

```python
import numpy as np
from gatecheck import certify, plant_vol_world, spearman_partial

def my_gate(world) -> bool:
    ic, p = spearman_partial(world["sig"], world["fv"],
                             [world["ret"], world["tv"]],
                             n_perm=500, seed=world["seed"])
    return bool(np.isfinite(ic) and abs(ic) >= 0.05 and p < 0.05)

res = certify(
    my_gate,
    worlds={
        "truth": lambda s: plant_vol_world(s, 0.25),   # planted signal
        "null":  lambda s: plant_vol_world(s, 0.0),    # planted null
    },
    n_seeds=25,
)
print(f"power {res.fire_rate_on_truth:.0%}  size {res.fire_rate_on_null:.0%}")
```

Decide your pass rule (e.g. truth >= 80%, every null <= 5%) **before** looking at
the numbers. Or run the built-in exhibit:

```bash
python -m gatecheck.calibration --fast
```

## Provenance

Extracted 2026-07-02 from the market_state research program (post-audit): the
audit-verified statistical spine of a falsification-first market research program.
Upstream, this harness certified four production gates and honestly FAILED one of
them (a zero-power content gate, quarantined as a result) — which is the point.

- Upstream program: *link placeholder*
- Audit reports: *link placeholder*

## License

MIT. Author placeholder: OWNER.
