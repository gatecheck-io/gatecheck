# Next study: calendar events and overlapping outcomes

Protocol fixed October 1, 2026, after the July source audit. This follows an
exploratory result we have already seen; it is not a blind preregistration.
No new inferential result is reported here.

## Question and source

Does a calendar-month S&P 500 gain of at least 10% provide evidence of a higher
probability of a positive return over the next twelve calendar months than
the unconditional monthly baseline?

The [source article and displayed table](https://www.fool.com/investing/2026/05/05/sp-500-did-this-13-time-50-years-what-happens-next/)
report a historical pattern. The July audit reproduced its event identities
and rounded annual rate. This follow-up asks a separate inferential question;
it does not assume the article proposed a trading strategy.

## Fixed definitions

- S&P 500 index quote close; last observed trading close of each complete
  calendar month. No ETF substitution or adjusted-close mixture.
- Event months: January 1974 through April 2026, matching the displayed table
  era. Outcomes use only observations available by May 5, 2026.
- Trigger: month-end close divided by prior month-end close minus one >= .10.
- Outcome: close twelve calendar months later divided by trigger close minus
  one; positive means strictly greater than zero. Incomplete horizons remain
  censored and never enter the denominator as losses.
- Comparator: every complete monthly twelve-month outcome on the same index,
  date eligibility and price convention. Primary contrast: event positive-rate
  minus baseline positive-rate. Mean-return difference is a secondary outcome.
- Costs and fills: not applicable to this descriptive event question. No
  tradability verdict, Sharpe hurdle or hypothetical holding timer is introduced.
- Preserve every trigger and show its forward window; do not choose a favorable
  non-overlap subset after seeing returns. Show denominators and overlap.

## Method gate before inference

The twelve-month outcomes share returns. A binomial test or independent-row
bootstrap is therefore not an accepted default. Before evaluating a market
p-value, freeze a proposed dependence-aware resampling method and demonstrate
its behavior on synthetic monthly series with known no-predictability and
planted effects. Generate triggers from each synthetic price series using the
same calendar rule, rather than imposing fixed event labels without explanation.

Record the null assumptions, alternative, test statistic, seeds, simulation
counts, effect sizes, missing/empty states and decision criterion before that
calibration run. Include serial dependence and drift sensitivities; quantify
false-positive uncertainty and detection limits. If the calibration fails,
publish that result and retain descriptive market figures without a rejection
claim. A subsequent market evaluation must keep the calibrated method unchanged.

## Delivery conditions

Publish the source question, frozen data/hash, event table, matched comparator,
method/calibration evidence, uncertainty, limits and runnable code. Distinguish
descriptive agreement, model-conditional inference and deployable trading
performance. Do not promote an exploratory result as out-of-sample validation.
