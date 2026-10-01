# V2 passed the declared false-alarm gate; detection remains limited

Completed October 1, 2026. **PASS_ALPHA_GATE** in all six declared true-null
worlds, with 1,000 fresh histories per world. This qualifies the candidate's
false-alarm behavior within its stated model envelope. It does not establish
adequate power, market-model validity or a pass/fail verdict on the source claim.

The code and plan were committed and pushed before calibration as
[`12e7a80`](https://github.com/gatecheck-io/gatecheck/tree/12e7a80afabb514bec85aa33302ed6d85535b40d/verdicts/2026-10/v2).
The seed namespace was 20261002; V1's certification draws were not reused as
V2 certification data. Eleven worlds produced 11,000 histories. Each history
contained 628 months after burn-in; positive-contrast tests used 999 randomized
sign vectors and the whole drift confidence interval. No method, threshold or
simulation count was changed after certification began. V1's [failed result](../CALIBRATION.md)
remains in the record.

## False alarms and center-interval coverage

Rates show two-sided 95% exact Clopper-Pearson intervals across independent
histories. The unchanged rule requires each upper bound <=7.5%, using all
attempts and available tests. Coverage counts refer to the >=99.5% common-center
interval under the true-null model. Coverage is not a confidence rating for the
market or a test of its assumptions.

| Null world | Available | False alarms / all attempts, 95% CI | False alarms / available tests, 95% CI | Center covered | Gate |
| --- | ---: | --- | --- | ---: | --- |
| iid_zero | 1000/1000 | 1.80% [1.07%, 2.83%] | 1.80% [1.07%, 2.83%] | 997/1000 | Pass |
| iid_drift | 1000/1000 | 1.00% [0.48%, 1.83%] | 1.00% [0.48%, 1.83%] | 996/1000 | Pass |
| iid_high_drift | 1000/1000 | 0.70% [0.28%, 1.44%] | 0.70% [0.28%, 1.44%] | 994/1000 | Pass |
| iid_heavy_tail | 1000/1000 | 1.90% [1.15%, 2.95%] | 1.90% [1.15%, 2.95%] | 997/1000 | Pass |
| serial_vol_90 | 1000/1000 | 1.70% [0.99%, 2.71%] | 1.70% [0.99%, 2.71%] | 997/1000 | Pass |
| serial_vol_98 | 988/1000 | 0.90% [0.41%, 1.70%] | 0.91% [0.42%, 1.72%] | 998/1000 | Pass |

Only `serial_vol_98` has unavailable draws: 12 histories contain no completed
triggers. They retain missing results rather than p=1. All-positive event
outcome histories are accepted by this method. The false-alarm intervals are
per world; they do not constitute a simultaneous guarantee over all possible
financial processes.

## Power and detection limits

The intervention adds known monthly log drift for the twelve months following
a generated trigger. Probability increments are separately seeded, 250,000-month
approximations, not exact targets or estimates of the market effect.

| Added monthly log drift | Approximated annual positive-rate increment | Available | Detection / all attempts, 95% CI |
| ---: | ---: | ---: | --- |
| 0.002 | 3.62 pp | 1000/1000 | 2.70% [1.79%, 3.90%] |
| 0.005 | 9.44 pp | 1000/1000 | 5.50% [4.17%, 7.10%] |
| 0.010 | 15.83 pp | 1000/1000 | 20.10% [17.66%, 22.72%] |
| 0.020 | 21.59 pp | 1000/1000 | 67.00% [63.99%, 69.91%] |

None of these alternatives demonstrates the declared 80% detection benchmark.
The strongest intervention has an approximately 21.59-point rate increment and
is detected in 670/1000 histories, 67.00% [63.99%, 69.91%]. Smaller effects are
much harder to detect. These stylized worlds are not interchangeable with the
market, so their rates do not give an exact power estimate for the observed
5.57-point market contrast. They do rule out claiming reliable detection or
absence of a modest effect on the strength of this calibration.

The method's confidence-set maximization is conservative: uncertain drift is
not fitted away to improve a verdict. A lower false-alarm rate alone is not
evidence that a test is useful at every effect size. V1 and V2 use different
inferential assumptions and independent seeds; their raw power rates should
not be treated as a controlled demonstration of universal method superiority.

## Out-of-model drift sensitivity

The alternating-drift world produces 3.10% [2.12%, 4.37%]
rejections across all attempts. Its approximated positive-rate increment is
3.97 points even with no
trigger intervention. It violates constant drift and is not counted as a true
zero-effect null. This result does not certify changing regimes, asymmetric
innovations, signed serial dependence or leverage dynamics.

## Review and evidence

All 11,000 draw definitions and count arithmetic, center intervals and decisions,
world rejection intervals, and five long-world rate contrasts were independently
checked. One full scalar sweep of actual event/outcome transitions per world
agreed with the vectorized drift supremum. The generator is reused as the selected
input source; not every randomization p-value has a separate implementation.
Small exhaustive rational-path tests also cover the sweep and tied knots.
All 132 project tests pass.

Inspect [raw draws](evidence/draws.csv), [summary](evidence/calibration.json),
[independent verification](evidence/verification.json), [method](METHOD.md),
[frozen plan](calibration_plan.json) and [hashes](provenance.json).
The following [market evaluation](MARKET_RESULT.md) uses this exact frozen method.
The method's qualification is distinct from the source claim's unresolved
predictive value. No new Substack or X post was published in this step.
