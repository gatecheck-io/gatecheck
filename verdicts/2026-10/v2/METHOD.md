# V2: preserve magnitudes, randomize signs, maximize over drift uncertainty

Candidate `monthly-sign-invariance-drift-supremum-v2`. The source figures and
V1 failure were known during development. This is an exploratory method record,
not blind preregistration or independent confirmation of the source pattern.
The frozen commit must precede calibration. Fresh namespace 20261002 replaces
20261001; no certification seeds are used for development or tuning.

## Question and exact scope

Keep the [original question and data definitions](../PROTOCOL.md): a calendar
month gain >=10%, strictly following twelve calendar months, complete outcomes,
and the conditional positive-rate minus the matched unconditional rate. T=w/k-b/n.
All triggers and overlapping forward windows remain. No outcome-based minimum
or non-overlap subset is selected. With no completed events the result is
unavailable. An all-positive event sample is accepted. A nonpositive contrast
cannot reject this one-sided question and receives p=1.

**The inferential null is explicitly model conditional:** for some constant
monthly log drift mu, the centered return vector is invariant under independent
sign changes, conditional on its entire vector of absolute magnitudes. This
allows dependent volatility and arbitrary tail magnitudes; it excludes signed
serial dependence, asymmetric innovations, changing drift and leverage dynamics
that make magnitudes reveal earlier signs. It is stronger than marginal symmetry
or a generic martingale condition. It is not the unrestricted null that the
conditional-rate increment equals zero for every possible time-series process.

The six declared zero-effect worlds meet this invariance null. They have zero
population conditional-rate increments for the reasons given in V1's method.
The full invariance family can also contain volatility-driven conditional-rate
differences with nonzero drift. Rejecting its upper-tail test would mean that
the positive observed contrast is unusual under the stated no-signed-content
model; it would not identify a causal trigger effect or universally prove a
population rate increment. Non-rejection would not prove no predictive content.
These limits must accompany the market result if the gate passes.

## Return-path randomization

For each drift mu, draw 999 independent sign vectors S and use
R*[t](mu) = mu + S[t]*(R[t]-mu). Recompute triggers and strictly forward
outcomes from that monthly path. Chronology and the full centered magnitude
vector remain, including volatility clustering. Shared twelve-month return
increments therefore remain shared. No binomial independence assumption is made
for event outcomes. Use the same generated signs at every drift value so the
empirical p-value can be maximized over a continuous interval.

At a known true mu, independent sign invariance makes the observed path and
randomized paths exchangeable conditional on centered magnitudes. The candidate
Monte Carlo p-value is (1 + number(T* >= T)) / 1000, with ties included. A
randomized path with no completed triggers gets statistic 0. That extension
cannot beat a positive observed contrast, but the path is retained in the null
denominator. It is not dropped and redrawn until an event appears.

## Unknown drift: a confidence-set maximum

Use nuisance-error budget beta=.005. For m=628 monthly observations, choose
the largest zero-based order index j satisfying 2*BinomCDF(j;m,.5) <= beta.
The closed interval [R_sorted[j],R_sorted[m-j-1]] has coverage >=99.5% for the
common center under the stated continuous conditional-sign model. If no finite
rank interval exists, the test is unavailable. The binomial distribution here
describes monthly centered signs under the null, not overlapping annual outcomes.

Take the supremum of the empirical Monte Carlo p-value over that entire interval
and add beta, capped at 1. Reject only at the unchanged one-sided alpha=.05.
If the true center lies in the confidence set, the resulting p is at least the
known-center valid p plus beta. A rejection then requires p_true <= alpha-beta;
adding the at-most-beta confidence-set miss probability gives error <= alpha
within the stated model. This is the Berger-Boos confidence-set construction.
Do not replace the supremum with a fitted mean, median, favorable grid point,
or assumption that drift is zero.

The supremum is calculated by a finite sweep, not interpolation. R*[t](mu) is
affine and nondecreasing in mu. Trigger and positive-outcome indicators each
change only once. Their conjunction changes at the later knot. The algorithm
tracks event counts, wins and baseline wins across all knots, including both
the inclusive-trigger state at a knot and the strict-outcome state just after
it. It groups tied changes before evaluating tail counts, then maximizes the
empirical count across the closed interval. Integer cross multiplication
compares contrasts, avoiding rounding of small rate differences. Independent
small-path exhaustive rational sweeps must agree before freezing the code.

## Calibration fixed before the run

Same six true-null worlds and three original alternatives as V1; add a fourth
forward intervention of .020 monthly log drift to characterize a stronger
detection range. The added strength is declared before its simulated outcomes.
The causal generator resets twelve subsequent active months at a trigger; it
does not stack interventions. Record separately seeded 250,000-month effect
approximations. The alternating-drift world remains an out-of-model sensitivity,
not a true zero-effect null. Its conditional association may be real regime
selection even with no trigger intervention.

Use 1,000 independent histories per world, 628 monthly returns after a 2,048-month
burn-in, and 999 sign vectors for every positive-contrast available test.
SeedSequence([20261002,world index,replicate,stream]) uses streams 0 for data,
1 for signs and 2 for population approximation. Scheduling cannot change seeds.
Record every attempted draw, unavailable reason, center interval, p-value and
decision. Also report the center interval's empirical coverage in true nulls.

Unchanged certification rule: each true-null world's two-sided 95% exact
Clopper-Pearson false-alarm upper bound must be <=7.5%, using both all attempts
and available tests. At least one test must be available. These are per-world
intervals. Power is reported over all attempts with availability separately.
An 80% detection range is demonstrated only where the exact lower bound reaches
80%; do not claim absence when effects are poorly detected. Passing the false-alarm
gate alone does not demonstrate adequate power for the market contrast.

If any null fails, publish the failure and do not evaluate the market p-value.
If all pass, use this exact frozen code and settings for the market evaluation.
Its sign seed [20261002,100,0,1], daily snapshot hash, calendar cutoffs and expected
source counts are pinned in advance. Compare the data hash before use. Publish
the event table and derived full eligibility/outcome ledger, keeping incomplete
outcomes censored. The historical vendor acquisition is from October 1, not
the missing original July snapshot. No threshold or resampling count is changed
after seeing the market p-value. No new social publication is part of this run.

## References

- [Berger and Boos (1994), P Values Maximized Over a Confidence Set for the Nuisance Parameter](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476836):
  confidence-set maximization with an error-budget adjustment. It does not certify
  this implementation or the market assumptions.
- [Geyer and Jones, exact location intervals](https://www.stat.umn.edu/geyer/3011/examp/conf.html):
  binomial order-statistic intervals. Their coverage is applied here only under
  the explicitly stronger independent conditional-sign assumption.
- [Chiu, Sharp and Bloem-Reddy, Randomization Tests for Conditional Group Symmetry](https://arxiv.org/abs/2412.14391):
  the finite-sample group-randomization framework. This study uses independent
  sign changes and its own discrete contrast; it is not their kernel test.
