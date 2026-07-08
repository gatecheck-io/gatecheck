# Audit summary — why this library ships calibration certificates

`gatecheck` is the extracted statistical spine of a falsification-first research
program that spent its life adjudicating trading-edge claims and closed at an
honest negative ("no deployable edge at the free-data frontier"). Before that
program closed, it commissioned **two independent adversarial audits of its own
methodology** — auditors instructed to assume every result was wrong and to
break it.

This document is the short version. The full audit reports (several hundred
findings, adversarially re-verified with runnable probes) are available on
request; what matters for *this library* is the one lesson that produced it.

## What the audits found (the load-bearing lesson)

The audits' central finding was not a wrong number. It was that **several of the
program's own gates had never been tested against worlds where the right answer
was known** — so some could not fail, one could not pass, and one known
estimator bug had survived under a green test suite. The specific catches that
shaped this library:

- A **content gate** with **zero statistical power** in the regime it was
  actually used in — it returned the same verdict whether or not real
  information was present. Every demotion it had issued was therefore
  unmeasured, not measured. It was **quarantined**, not patched around.
- A **detection floor** that was never stated: the deployable-edge test could
  not, by construction, detect an effect below roughly a 1.0 gross Sharpe — so
  "we found no edge" needed the qualifier "above this floor," which had never
  been quantified.
- A live **rank-space vs. levels-OLS** estimator bug that silently flipped the
  sign of a discrete carrier — fixed, with the fix strengthening (not weakening)
  the underlying result.
- Small-sample confidence intervals using a normal quantile where a Student-t
  quantile was correct (anti-conservative), and a deflated-Sharpe call with a
  scale bug that made one "tradable" branch unreachable.

## What this library is

Every one of those is a case of **an adjudicating instrument nobody had
adjudicated.** The fix the audits forced — and the thing `gatecheck` packages —
is the discipline of **certifying every gate against planted-truth and
planted-null worlds before trusting it**: measure whether it fires when a real
effect is planted, and whether it stays silent on structured nulls that share
your data's persistence and coupling. Tests for the tests.

The concrete results of running that certification over the source program's own
production gates are in [`CALIBRATION_CERTIFICATES.md`](CALIBRATION_CERTIFICATES.md):
the core rank-space partial **PASSED** (fires ~96% on planted truth, ~0–4% on
nulls), the content gate **FAILED** (0/15 on genuine content — the honest
admission above), and the edge and capacity gates were **CHARACTERIZED** (their
detection floors and biases measured rather than assumed).

The Student-t fix, the rank-space partial, the autocorrelation-aware permutation
nulls, and the certification harness itself all ship in this package. The point
of the library is not that its authors are clever; it is that **a gate you have
not calibrated is an instrument you cannot trust — including your own.**
