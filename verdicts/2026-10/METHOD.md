# Frozen candidate: monthly positive-rate contrast

Candidate `monthly-rate-stationary-pairs-v1`, fixed October 1, 2026 before its
first calibration run. The source result and data geometry were already known.
This is an exploratory method-development record, not a blind preregistration.
The exact settings and all worlds are in `calibration_plan.json`. A Git commit
and SHA-256 manifest pin the implementation before calibration; results must
name that commit. No result-dependent tuning or substitution is permitted.

## Estimand and candidate test

For monthly simple returns, E[t] = 1(return[t] >= .10). Y[t] = 1(the product
of 1 + return[t+1:t+13] > 1). Only complete forward windows enter either rate.
T = sum(E*Y)/sum(E) - mean(Y). The comparator includes event months, just as
the established matched unconditional baseline does. The final twelve months
are censored, not treated as losses. At 628 generated months there are 616
complete outcome rows. Triggers are generated from each simulated price path;
labels and outcomes are never separately planted or shuffled.

Resample the complete, aligned (E,Y) rows using the circular stationary
bootstrap: the first index is uniform; each subsequent index restarts at a
uniform index with probability 1/24, otherwise advances one index, wrapping
at the end. Use 999 draws. Each bootstrap draw recomputes the ratio contrast
with its own event denominator. Resampling aligned derived rows retains
observed event/outcome dependence within blocks. It is an estimator bootstrap,
not a newly assembled price path: recomputing forward returns across artificial
block joins would create a different method. Dependence is only approximately
retained, and the fixed 24-month mean block size is itself an assumption.

The proposed boundary-null distribution is T* - T. The one-sided candidate
p-value is (1 + count(T* - T >= T)) / (1 + valid bootstrap draws). Reject only
if T > 0 and p <= .05. This recentering is an asymptotic proposal, not an exact
randomization test. Rare, discrete events may make it inaccurate; calibration
is required before it is used to support a market significance claim.

The observed test is unavailable with fewer than 10 complete events, fewer
than two positive event outcomes, or fewer than two nonpositive event outcomes.
These prespecified guards avoid estimating tail uncertainty from a boundary
or nearly empty event sample. They also limit detection of very strong effects:
an all-positive sample is unavailable, never evidence against an effect.
Bootstrap samples with zero events are recorded and omitted. If more than 1%
are empty, the test is unavailable. Other bootstrap boundary counts are retained
as part of the estimator's sampling distribution. No missing state returns a
nominal nonsignificant p-value.

## Synthetic worlds and known answers

All generators operate on monthly log returns and form positive price paths by
exponentiating their cumulative sum. A 2,048-month burn-in is discarded.

- Three IID Gaussian nulls have monthly log drift 0, .006 and .010, and log
  return SD .045. The current return is independent of future returns, so the
  population conditional positive-rate contrast is exactly zero for each drift.
- One IID Student-t(5) null has drift .006 and SD .045. It tests heavy tails;
  independence again gives a zero population contrast.
- Two serial-volatility nulls have zero drift and independent symmetric Gaussian
  innovations. An independent Gaussian AR(1) log-volatility process has phi .90
  or .98 and stationary SD .5; the scale is normalized to unconditional return
  RMS .045. Volatility is serially dependent, but future return signs remain
  symmetric given the volatility path, including after a positive trigger.
  Thus both conditional and unconditional annual positive probabilities are .5.
  These are not nulls with predictable signs or positive-drift volatility regimes.
- Three alternatives add log drift .002, .005 or .010 in every one of the next
  twelve months after a generated trigger. A new trigger resets the active
  window; effects do not stack. The current trigger uses the actual return,
  including previously active drift. This creates forward content causally in
  the generator. The intervention strength is known; its resulting conditional
  probability increment is not an assumed exact number. Report a separately
  seeded 250,000-month population approximation for that increment and baseline.
- A deterministic alternating-drift sensitivity uses 120-month regimes with
  monthly drift 0 or .012. Conditional association may be nonzero because
  triggers select regimes. It is explicitly not counted as a true zero-effect
  null, even though no trigger intervention exists.

No world reproduces every property of the S&P 500. In particular, predictable
signs, leverage effects, structural breaks, and alternative block sizes remain
outside the declared zero-effect calibration envelope.

## Calibration and decision

Use 1,000 independent data and bootstrap seeds per world. Seeds derive from
NumPy SeedSequence([20261001, world index, replicate, stream]); streams 0, 1 and
2 designate generated data, bootstrap and population approximation respectively.
Parallel scheduling does not change seeds. Counts are fixed, not extended to
cross a desired threshold. All world draws and unavailable reasons are saved.

For every true null, report false rejections over all 1,000 attempts AND over
available tests. Unavailable tests are not quietly counted as valid negatives.
For both denominators, use two-sided 95% exact Clopper-Pearson intervals over
independent simulation replicates. Each declared null must have at least one
available test and the upper bound must be <= .075 in both denominators. These
are per-world intervals, not a simultaneous guarantee over the whole suite.

Report alternative detection rates, exact intervals, available-test fractions,
and the independently approximated effects. The largest tested effect with a
95% lower detection-rate bound >= .80 identifies demonstrated useful power;
if none reaches it, state that no such range was demonstrated. Do not interpolate
a detection floor or treat low power as evidence of no effect. Power uses all
attempts, with availability shown separately.

If any null gate fails, retain the failure and do not evaluate the market p-value
with this candidate. The descriptive 10/12 versus 479/616 result remains. A next
candidate needs a new version and a new frozen plan; this run cannot be used as
its independent certification set. If the gate passes, any market evaluation
must use the pinned code/settings unchanged and retain the power limitations.

## References and scope of support

The [Politis and Romano stationary-bootstrap paper](https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf)
supports block resampling for weakly dependent stationary observations. It does
not certify this rare-event ratio, chosen block size or finite-sample p-value.
[SciPy 1.14.1 exact proportion intervals](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)
document the Clopper-Pearson intervals used for simulation counts; they are not
binomial intervals applied to overlapping market outcomes.
