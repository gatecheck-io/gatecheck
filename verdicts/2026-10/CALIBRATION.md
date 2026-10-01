# The first monthly-event test failed calibration

Completed October 1, 2026. **FAIL_ALPHA_GATE.** Four of six zero-effect worlds
failed the prespecified false-alarm bound. This candidate is not cleared for
a market p-value. The result concerns the test method, not whether the source
claim is true or false. The source's 10/12 annual positive outcomes and matched
479/616 baseline remain descriptive observations.

The candidate and simulation plan were committed and pushed before calibration
at [`aa9aa9e`](https://github.com/gatecheck-io/gatecheck/tree/aa9aa9e6c2e5658e12015fa609a8783984423d8d/verdicts/2026-10).
There were 1,000 independent histories in each of ten worlds, 628 months per
history, and 999 stationary-bootstrap draws for each available test. All
settings, guards and worlds stayed fixed. No market p-value was evaluated.

## False alarms in worlds with exactly zero population rate increment

Entries show the observed rejection rate and its two-sided 95% exact
Clopper-Pearson interval. The allowed upper bound is 7.5% in both denominators,
for every declared zero-effect world. Availability is shown separately:
an unavailable test is not a valid negative result.

| World | Available tests | False alarms / all attempts, 95% CI | False alarms / available tests, 95% CI | Gate |
| --- | ---: | --- | --- | --- |
| iid_zero | 604/1000 | 3.70% [2.62%, 5.06%] | 6.13% [4.35%, 8.35%] | Fail |
| iid_drift | 882/1000 | 5.00% [3.73%, 6.54%] | 5.67% [4.24%, 7.41%] | Pass |
| iid_high_drift | 902/1000 | 2.90% [1.95%, 4.14%] | 3.22% [2.16%, 4.58%] | Pass |
| iid_heavy_tail | 911/1000 | 6.50% [5.05%, 8.21%] | 7.14% [5.55%, 9.00%] | Fail |
| serial_vol_90 | 804/1000 | 7.00% [5.50%, 8.76%] | 8.71% [6.85%, 10.87%] | Fail |
| serial_vol_98 | 598/1000 | 4.90% [3.65%, 6.43%] | 8.19% [6.12%, 10.69%] | Fail |

`iid_zero` and `serial_vol_98` pass with all attempts in the denominator but
fail once unavailable tests are disclosed and removed from that denominator.
`iid_heavy_tail` and `serial_vol_90` fail in both denominators. A gate failure
means the candidate did not meet its chosen uncertainty bound; it does not
establish that every population false-alarm probability exceeds 7.5%.
The closest high-drift Gaussian world passes; selecting that favorable world
after seeing the rest would change the declared certification scope.

## Detection of planted forward content

The interventions add known monthly log drift for twelve months after each
generated trigger. The probability increments below are separately seeded,
250,000-month population approximations, not exact effect-size targets or
market estimates. Detection intervals describe independent finite-history
simulation counts, not uncertainty in those long-world approximations.

| Added monthly log drift | Approximated annual positive-rate increment | Available tests | Detection / all attempts, 95% CI |
| ---: | ---: | ---: | --- |
| 0.002 | 3.93 pp | 865/1000 | 7.90% [6.30%, 9.75%] |
| 0.005 | 9.94 pp | 795/1000 | 16.30% [14.06%, 18.74%] |
| 0.010 | 17.18 pp | 562/1000 | 22.40% [19.85%, 25.11%] |

No declared alternative demonstrates an 80% detection rate. Even the strongest
planted intervention is detected in only 224/1000 histories. Its approximately
17.18-percentage-point increment is larger than the descriptive market increment
of approximately 5.57 points, but the worlds are not interchangeable with the
market. This does not quantify the power for the actual observed market effect.

In the strongest alternative, 438 histories are unavailable. Sparse samples
and the declared boundary guards contribute to detection limitations; all-positive
event samples are unavailable, not evidence of no effect. The rejection rate
among available histories is 39.86%, still below useful 80% detection. The candidate
does not support inferring absence from a nonsignificant or unavailable result.

## Drift sensitivity

The alternating-drift world produces 94/1000 rejections (9.40%, 95% CI
7.66%–11.38%). Its long-world conditional increment is approximately 5.77 points:
events select the high-drift regime even though no trigger intervention exists.
It is not a true zero-contrast null and was correctly excluded from false-alarm
certification. A conditional association alone would not establish causal
trigger content or a deployable trading edge.

## Verification and reproducibility

- All 10,000 draw identities, event/outcome counts and recorded rejection rules
  were checked against regenerated inputs using independent twelve-return
  products, rather than the candidate's cumulative-log outcome helper.
- All world-level intervals and gate decisions were recomputed directly from
  `draws.csv`; ten selected p-values were recomputed with a scalar stationary
  bootstrap rather than the vectorized index helper.
- All four long-world rate contrasts were independently checked. The generator
  itself is reused as the selected input source. This is not an independent
  recomputation of every bootstrap p-value or a universal method certification.
- The first verification implementation overflowed a cumulative price in the
  250,000-month check. It was repaired to multiply only the twelve forward
  gross returns; a regression check covers that boundary. The frozen candidate
  and all calibration results were unchanged.
- The project test suite passes all 122 tests after that verification repair.

Inspect [raw draws](evidence/draws.csv), [calibration summary](evidence/calibration.json),
[independent verification](evidence/verification.json), [frozen method](METHOD.md),
[plan](calibration_plan.json), and [hashes](evidence/provenance.json).
Use the commands in [README.md](README.md) to rerun from the frozen commit.

## Next research step

Retain this failed version. A replacement must address the rare, discrete
event-rate estimator and sparse boundary samples, rather than weakening the
false-alarm threshold or selecting only favorable worlds. Freeze the replacement
method and a new seed namespace before its certification run. Use this failed
suite as development evidence, not as independent certification of its successor.

Until a replacement passes its declared gate and discloses adequate detection
limits, Gatecheck reports descriptive agreement and uncertainty; it issues no
new significance or absence-of-effect verdict for the monthly claim. No new
Substack or X publication was made during this method-calibration step.
