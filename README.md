# gatecheck

Your backtest pipeline is full of gates — "the Sharpe CI excludes zero", "the
partial IC is significant", "the feature adds OOS R²" — and almost certainly none
of them has ever been run against a world where the answer is known. An untested
gate is an uncalibrated instrument: you don't know its power (does it fire when a
real effect is planted?) or its size (does it stay silent on structured nulls that
share your data's persistence, coupling, and smoothing?). `gatecheck` ships the
gates *with* their calibration certificates: measured fire rates on planted truths
and planted nulls, with binomial confidence intervals, reproducible from a seed.
Tests for the tests.

> Not yet on PyPI — install from source (`pip install -e .`). A PyPI release is planned.

## Track record

This library was not designed on a whiteboard. It is the extracted statistical
spine of a falsification-first market research program (the source program, ~160 PRs,
run to an honest "descriptive, not operational" terminus), and its track record is
mostly a record of things it killed:

- **30 posted trading claims benched, 0 survived.** The source program mined
  publicly posted edges (fintwit, Reddit, SSRN/arXiv q-fin, Substacks) and ran
  each through the deflation + purged-CV + significance stack in this package.
  None survived costs and multiplicity. The one apparent survivor was
  itself killed as a selection artifact (p = 0.68 under the proper null).
- **It killed its own +2.17-Sharpe result.** The program's single best-looking
  finding — a +2.17-Sharpe variance strategy — was a strike/tenor-mismatch false
  positive. An inline tenor-matched control (the kind of structured null this
  package institutionalizes) exposed it as carry-beta. The gate that killed the
  program's best number is in this box.
- **It was adversarially audited twice — and the audits found gates that could
  not fail.** Both audits surfaced tests whose construction made a negative
  verdict effectively impossible: nulls too weak for the carrier's persistence,
  CIs anti-conservative at small n, a residualize-then-rank pipeline that
  silently flipped signs. The repairs those audits forced (small-n Student-t
  CIs, rank-space partials, autocorrelation-aware circular-shift nulls, and gate
  calibration itself) are what this package is. A gate that cannot fail is
  worse than no gate — it launders overfitting into confidence. That failure
  mode is exactly why this library exists.
- **The honest admission: the source program's own content gate FAILED
  certification and was quarantined.** When the calibration harness in this
  package was run over the program's four production gates, one of them — a
  "content gate" meant to separate genuine forward information from mechanical
  persistence — had zero power in its live regime. It was quarantined, not
  patched around. (See [`docs/CALIBRATION_CERTIFICATES.md`](docs/CALIBRATION_CERTIFICATES.md) for the measured fire-rates, and [`docs/AUDIT_SUMMARY.md`](docs/AUDIT_SUMMARY.md).)

None of this proves the gates here are correct. It does mean each one has been
fired at planted truth and planted null at least once, and the misses are
documented rather than hidden.

## Quickstart

```bash
pip install -e .            # from a source checkout; numpy is the only dependency
python -m pytest tests -q   # 94 tests, offline, deterministic
```

Certify your own gate in ~10 lines — a gate is any function
`gate(world_dict) -> bool`:

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
        "truth": lambda s: plant_vol_world(s, 0.25),          # planted signal
        "null":  lambda s: plant_vol_world(s + 10_000, 0.0),  # planted null
    },
    n_seeds=25,
)
print(f"power {res.fire_rate_on_truth:.0%}   size {res.fire_rate_on_null:.0%}")
```

Output (deterministic): `power 96%   size 0%`, in ~4 s. Decide your pass rule
(e.g. truth ≥ 80%, every null ≤ 5%) **before** looking at the numbers.

The built-in exhibit certificate runs from the command line:

```bash
python -m gatecheck.calibration          # full scale (~9 s)
python -m gatecheck.calibration --fast   # test scale (~1 s)
```

Its generated numbers (seeds, fire rates, CIs, runtimes) are checked in at
[`docs/CERTIFICATE.md`](docs/CERTIFICATE.md).

## What's included

- **`gatecheck.calibration`** — the headline. Seeded, offline world generators
  with plantable truth (`plant_vol_world`, `content_world`, `plant_edge_world`)
  and structured nulls (`static_coupling_world` — persistence + contemporaneous
  coupling, zero anticipation: the geometry that fools i.i.d. permutation nulls),
  plus the generic `certify()` runner and `Certificate` dataclass. One built-in
  PASS exhibit: `certificate_spearman_partial`.
- **`gatecheck.deflation`** — Deflated Sharpe Ratio, expected-max-Sharpe
  multiplicity benchmark, and a PBO proxy (Bailey & López de Prado 2014; López
  de Prado 2018). All Sharpe inputs are per-period, and `sigma_sr` (the
  cross-trial SR dispersion) must be on the same per-period scale — the
  docstrings spell out the units contract because getting it wrong silently
  changes verdicts.
- **`gatecheck.cv`** — purged + embargoed walk-forward cross-validation. Pure,
  data-free integer index math: no values, labels, or RNG ever enter the split
  logic, so the splitter *cannot* leak.
- **`gatecheck.significance`** — PIT-aligned information coefficient with an
  explicit next-bar alignment helper, permutation and autocorrelation-aware
  circular-shift p-values, and multi-seed Sharpe aggregation with small-n
  Student-t CIs (the normal z = 1.96 at n = 4 is anti-conservative; that was an
  audit finding, and the fix is versioned in the API).
- **`gatecheck.oos`** — fit-on-train / score-on-test primitives with train-only
  standardization: closed-form ridge, ridge-logistic (IRLS), train-mean-referenced
  OOS R², Brier score.
- **`gatecheck.incremental`** — incremental OOS R² of added features over a
  baseline through purged folds, with two honesty devices: a **leaked-oracle
  power floor** (plant the target itself as a feature; if even the oracle
  recovers ~nothing, a null candidate is NOT_MEASURABLE, not a FAIL) and a
  **pass-through diagnostic** (re-run against the full observable set to catch
  hollow passes that only beat a narrow baseline).
- **`gatecheck.rank`** — rank-space partial correlations, and the reason they
  matter: the obvious pipeline (OLS-residualize the carrier's *levels* on the
  controls, then Spearman the residuals) can spuriously **sign-flip** a binary or
  discrete carrier. After a levels projection, the *ordering* of a two-valued
  carrier's residuals is dominated by the controls, so ranking them re-injects
  control structure with an arbitrary sign — the source program measured a
  carrier at rank-IC +0.22 through that pipeline whose true partial was −0.16.
  `spearman_partial` ranks everything *first*, then residualizes and correlates
  entirely in rank space; a regression test pins the binary-carrier sign.

## The certificate concept

A gate is a boolean function of a world. A **certificate** is the measured answer
to two questions:

1. **Power** — over `n_seeds` independent draws of a world with the effect
   *planted at known strength*, what fraction of the time does the gate fire?
   Pass rule of the built-in exhibit: **fire on truth ≥ 80%**.
2. **Size** — over the same number of draws of worlds where the effect is
   *absent by construction*, what fraction of the time does it fire anyway?
   Pass rule: **silent on every null, ≤ 5%**.

Both rates are reported with add-one (Laplace) binomial 95% CIs, which behave
sanely at 0/n and n/n. The nulls that matter are **structured**: not white noise,
but worlds that share the live carrier's nuisance geometry — persistence,
contemporaneous coupling, smoothing — with the causal link severed. A gate that
passes only a white null has been tested against an opponent that doesn't exist.

Verdicts are `PASS`, `FAIL`, or `CHARACTERIZED` — the third for properties that
aren't pass/fail, like a detection floor (sweep `plant_edge_world` over a Sharpe
grid and report where the gate crosses 50%/80% fire; upstream this disclosed a
DSR leg structurally blind below ~1.0 planted gross Sharpe at its own sample
geometry — a property worth disclosing, not hiding behind a green light).

## Limitations

Read these before trusting a certificate.

- **The worlds are stylized daily-bar synthetics.** AR(1) log-vol returns,
  AR(1) carriers, known coupling. Passing on them is *necessary, not
  sufficient*: a gate can be well-calibrated on every world here and still fail
  on a nuisance geometry you didn't plant. The harness measures the gate against
  the opponents you constructed — construct more.
- **numpy only, simplest-honest estimators.** Closed-form ridge and IRLS
  logistic; no scipy, sklearn, pandas, or GPU anything. That is a feature for
  auditability and determinism, but this is not a modeling library.
- **Not a backtester.** No prices, fills, costs, slippage, or portfolio
  accounting anywhere. `gatecheck` adjudicates statistical claims your backtester
  produces; it does not produce them.
- **Default CIs are wide.** At `n_seeds = 25`, a 96% observed power reads
  92.6% [82.7%, 100.0%] add-one. Raise `n_seeds` if the decision is close.
- **Alpha software.** Extracted and re-tested (94 offline deterministic tests),
  but the package boundary is days old. API may move before 0.2.

## Provenance

Extracted 2026-07-02 from the source research program, post-audit: the
audit-verified statistical spine of a ~160-PR falsification-first market research
program that ran to an honest descriptive-not-operational terminus. Upstream,
this harness certified four production gates, honestly FAILED one of them (the
zero-power content gate, quarantined as a result), and CHARACTERIZED the
undisclosed detection floor of another. Lineage of every module is recorded in
its docstring; no upstream code is imported.

- How this library relates to the audits: [`docs/AUDIT_SUMMARY.md`](docs/AUDIT_SUMMARY.md)
- Calibration certificates (the four production gates, including the content gate that FAILED and was quarantined): [`docs/CALIBRATION_CERTIFICATES.md`](docs/CALIBRATION_CERTIFICATES.md)
- Generated example certificate for this library: [`docs/CERTIFICATE.md`](docs/CERTIFICATE.md)
- Full adversarial audit reports of the source program: available on request.

## License

MIT, © 2026 gatecheck. (The legal copyright holder becomes the operating entity's
name at incorporation; see the launch playbook.)
