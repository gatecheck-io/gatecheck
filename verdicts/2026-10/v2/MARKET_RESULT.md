# The monthly statistic reproduced; predictive advantage remains unresolved

Evaluated October 1, 2026, using the [frozen V2 method](METHOD.md) after its
[false-alarm calibration passed](CALIBRATION.md). **Descriptive source agreement;
no rejection of the declared model; no proof of absence.**

The displayed historical pattern is reproduced: 13 trigger months, with 12
completed twelve-month outcomes and 10 positive outcomes. The April 2026 trigger
is censored at the source publication cutoff, rather than counted as a loss.
The matched unconditional comparator has 479 positive outcomes among 616.

| Observation | Count | Rate |
| --- | ---: | ---: |
| Triggered, completed annual outcomes | 10/12 positive | 83.33% |
| Matched unconditional annual outcomes | 479/616 positive | 77.76% |
| Conditional minus unconditional rate | Same calendar eligibility and quotes | +5.57 percentage points |

## Model-conditional result

The unchanged one-sided test gives **p=0.219**, above alpha .05.
It does not reject the constant-drift, conditional-sign-invariance model.
The maximum Monte Carlo tail count is 213 of 999 sign vectors: the add-one
p-value is .214, and the frozen nuisance error budget adds .005. The whole
drift confidence interval was evaluated; no favorable drift point was selected.
The randomization resolution is 1/1000, with its seed fixed before evaluation.

This is not a distribution-free test of every zero-contrast time series.
The null assumes a constant common monthly log drift and sign invariance
conditional on the full centered magnitude vector. Signed serial dependence,
asymmetry, changing drift and leverage effects can violate it. Calibration on
synthetic worlds does not establish those assumptions for the S&P 500.

**The historical count can be correct while its predictive interpretation
remains unestablished.** This evaluation does not confirm a repeatable advantage
over the already-positive unconditional baseline. It also does not show that
the advantage is absent. Calibration has limited power at modest planted
increments and no declared alternative reaches the 80% detection benchmark.
Non-rejection here cannot support a blanket failed-claim verdict.

## Overlap, secondary outcome and uncertainty

The event windows share monthly increments in three pairs:

- 1974-10 and 1975-01: 9 shared monthly increments.
- 1982-08 and 1982-10: 10 shared monthly increments.
- 2020-04 and 2020-11: 5 shared monthly increments.

All events remain in the count. These pairs are not converted into a claimed
number of independent trials, and the test does not use independent binomial
annual-outcome uncertainty. The common-center confidence interval spans monthly
log drift 0.005818 to 0.015894, with
coverage >=99.5% under its null assumptions. It is a nuisance-parameter interval,
not a confidence interval for the conditional-rate effect.

The mean twelve-month return is 16.35% after triggers versus 10.17% for the
matched baseline, a descriptive difference of 6.19 points. The mean-return
contrast is secondary and was not separately significance-tested. No calibrated
effect-size confidence interval is supplied; no independence-based interval is
substituted for the overlapping data. Report these limits beside the result.

## Source, data and independent review

The [source article and displayed table](https://www.fool.com/investing/2026/05/05/sp-500-did-this-13-time-50-years-what-happens-next/)
define the historical question. The source does not need to be interpreted as
a trading strategy. Data use S&P 500 quote closes, final observed trading close
per complete calendar month, January 1974 through April 2026, and outcomes
available at the May 5, 2026 publication cutoff.

The frozen historical daily CSV SHA-256 is
`cd18f96fbac06771c70bc2c614667116aea5f0de4e63fe4dfb454b03a58a2a02`.
It was acquired October 1, 2026 from Yahoo Finance; it is not the missing original
July snapshot or a point-in-time vendor vintage. Historical revisions remain a
limit. Raw vendor prices are not redistributed in this package.

Independent review recomputed every calendar predicate from raw quote closes,
all 628 derived ledger rows, all 13 event identities, the 12 completed outcomes,
baseline and mean-return arithmetic, the center interval, and the whole market
p-value with a separate scalar sweep. The result agrees. Its independent
math.log vector hash is recorded separately; the data identity is pinned by the
raw CSV hash. Verification does not certify the market's symmetry assumptions.

Inspect [event table](market/events.csv), [full eligibility/outcome ledger](market/monthly_window_ledger.csv),
[market result](market/market.json), [independent verification](market/verification.json),
[reproduction commands](README.md), and [hashes](provenance.json).

## Research disposition

Retain this case as a reproduced descriptive statistic with a model-conditional
non-rejection and disclosed detection limits. Do not label it a proven edge,
an absent edge, or one more failed source claim. A confirmatory advantage claim
would need new evidence and a method with demonstrated useful detection at the
target effect size. The completed case is ready to explain publicly with these
limits; further parameter tuning on this observed history would not provide
independent confirmation. No new social publication was made during this run.
