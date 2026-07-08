# Gate-calibration certificates — planted-truth / planted-null certification of the adjudicating instruments

**Repair R5 (2026-07-02).** Audit refs: `audit/RESEARCH_AUDIT_TWO_PASS.md` (lesson L8 — the audits' central
lesson: the program certified its claims against nulls but never certified its *gates* against worlds where
the truth is known), appendix clusters **V01** (content gate), **V06** (capacity null), **V07** (edge bench
detection floor), **V14** (static-coupling world), and `RESEARCH_AUDIT_REPORT.md` **RA-01**.

Harness: `market_os/evaluation/calibration.py`. Fast versions run as tests in
`tests/test_calibration_harness.py` (offline, ~4 s). The full certificates below regenerate with:

```
python -m market_os.evaluation.calibration
```

All worlds are seeded numpy simulations — fully offline, fully deterministic. Fire rates are reported as raw
fractions with **add-one (Laplace) 95% binomial CIs** in brackets. Recorded run: 2026-07-02, seeds
`base_seed=0` throughout, total runtime **34.6 s** (Windows, CPython + numpy).

| # | Gate | Verdict |
|---|------|---------|
| 1 | `forward_vol_state._spearman_partial_pvalue` (the program's load-bearing partial-IC gate) | **PASS** |
| 2 | `content_gate.joint_shift_content_p` — derived-DV branch (the implied-up legs) | **FAIL** (quarantined) |
| 3 | `edge.score_positions` — the DSR leg (the edge bench) | **CHARACTERIZED** (floor ~1.0 gross Sharpe) |
| 4 | `capacity.tercile_random_ranking_sharpe` — max-of-50 same-tercile null | **CHARACTERIZED** (~98% kill-bias) |

---

## 1. `forward_vol_state._spearman_partial_pvalue` — **PASS**

**What was planted.** An AR(1) log-vol daily world (`plant_vol_world`, n=1500, w=5). Truth: a signal mixing a
persistent AR(1) carrier with the standardized residual of forward vol after the gate's own controls
`[ret_t, TV_t]`, at planted levels-partial **0.25** (realized rank-space Spearman partial ~0.20 — the scale of
the program's own recorded headline effect, holdout h5 IC −0.277). Null #1: the same world at
`signal_strength=0` (pure persistent carrier). Null #2: the audit's **V14 static-coupling world** (the
construction pinned in `tests/test_vrp_mechanism.py` — AR(1) gamma, iid returns, `vix = level + RW − 5γ`,
zero anticipation anywhere), gamma as the carrier. Fire = the module's own materiality: `|partial IC| ≥ 0.05`
and circular-shift `p < .05` (`n_perm=1000`; the live verdict additionally requires the pre-registered sign,
which only makes the null stricter).

| world | fired | raw | add-one 95% CI |
|-------|-------|-----|----------------|
| truth (planted ~0.25) | 24/25 | **96.0%** | 92.6% [82.7%, 100.0%] |
| null (strength 0) | 0/25 | **0.0%** | 3.7% [0.0%, 10.8%] |
| null — V14 static coupling | 1/25 | **4.0%** | 7.4% [0.0%, 17.3%] |

**Verdict: PASS** (rule: truth ≥ 80% fire AND both nulls ≤ 5%, raw fractions). The program's core
adjudicating statistic detects a headline-sized planted forward signal ~96% of the time and stays at/below
nominal on both a white null and the zero-anticipation structured null. Runtime 9.1 s, seeds 0–24 per world.

---

## 2. `content_gate.joint_shift_content_p` — derived-DV branch — **FAIL** (certified zero-power; gate quarantined)

**What was planted.** The audit's **V01** world (`content_world`, n=1600, w=21): an implied-vol-like signal
0.70-mixed with its own trailing-vol control. Truth: **genuine, returns-coupled forward content** — an
overshoot-and-correct state driven by the underlier's own |ret| shocks (jumps on shock days, then decays), in
the **attenuation regime** (partial |IC| < marginal |IC|) that every live input to this branch occupied.
Null (slow): the control mixed with an independent slow AR(1) — pure bounded persistence, content-free.
Null (smoothed fast-gap): a pure 5-bar smoothing of the control itself — the audit's false-pass geometry.
Fire = the module's own rule (`impliedup_has_content`): `|IC| ≥ MATERIAL_IC (0.05)` and
`content-p < CONTENT_ALPHA (0.05)`; `n_perm=300, n_surr=400`.

| world | fired | raw | add-one 95% CI |
|-------|-------|-----|----------------|
| truth (genuine content, attenuation regime) | 0/15 | **0.0%** | 5.9% [0.0%, 17.1%] |
| null (slow persistence) | 1/15 | 6.7% | 11.8% [0.0%, 27.1%] |
| null (smoothed fast-gap) | 1/15 | 6.7% | 11.8% [0.0%, 27.1%] |

Diagnostics on the truth worlds: **median content-p = 1.000** (pinned); the leg was material (|IC| ≥ 0.05,
so content-p was actually computed) in 14/15 seeds, and standard-significant+material in 1/15 — i.e. even a
leg the standard null certifies is pinned at content-p ~ 1.0, exactly the audit's V01 signature.

**Verdict: FAIL — recorded honestly.** Genuine control-correlated content cannot fire this branch (zero
power in the attenuation regime), while content-free geometries can false-fire: the branch's error direction
depends on conditional-vs-marginal geometry, not on content. Root cause (audit V01): surrogates roll the
signal and re-derive the DV but never decouple the controls, so each surrogate's partial IC collapses to its
marginal IC. **Consequence:** the recorded "content-p = 1.00" cells (docs/CONTENT_GATE.md's six implied-up
legs; `crypto_vrp_state`'s identical construction) are construction artifacts, not graded evidence. The gate
is **QUARANTINED** (prominent warning at the top of `market_os/research/content_gate.py`, dated 2026-07-02,
audit ref V01/RA-01): *do not adjudicate with this gate pending redesign*. The gate code itself is
unmodified (redesign out of scope for R5); the non-derived branch (`derive_window=None`) is unaffected and
remains valid. Runtime 16.3 s.

---

## 3. `edge.score_positions` — the DSR leg — **CHARACTERIZED** (detection floor)

**What was planted.** `plant_edge_world`: persistent ±1 positions (flip prob 0.05/bar) and
`ret = μ·pos + σ·z` with μ set so the strategy's **gross** annualized Sharpe equals the grid value. Sample
geometry matches the posted-edge bench as actually run (audit V07): **n=2774 bars (~11y), n_trials=10**.
Fire = `dsr_ok` (DSR ≥ 0.95 and discounted > 0) — the DSR leg alone; the full conjunction `passed` (also
requires beats-buy&hold + OOS-CI) is recorded alongside and can only be lower. 20 seeds per grid point.

| planted gross Sharpe | DSR-leg fire | add-one 95% CI | full-gate `passed` |
|---------------------|--------------|----------------|--------------------|
| 0.00 (null) | 0/20 | 4.5% [0.0%, 13.2%] | 0/20 |
| 0.25 | 0/20 | 4.5% [0.0%, 13.2%] | 0/20 |
| 0.50 | 4/20 | 22.7% [5.2%, 40.2%] | 3/20 |
| 0.75 | 1/20 | 9.1% [0.0%, 21.1%] | 1/20 |
| 1.00 | 10/20 | 50.0% [29.1%, 70.9%] | 8/20 |
| 1.25 | 12/20 | 59.1% [38.5%, 79.6%] | 7/20 |
| 1.50 | 20/20 | 95.5% [86.8%, 100.0%] | 15/20 |
| 2.00 | 20/20 | 95.5% [86.8%, 100.0%] | 19/20 |

* **Detection floor (50% fire): planted gross Sharpe ~1.00.**
* **Detection floor (80% fire): planted gross Sharpe ~1.38.**
* **Null false-fire rate: 0/20.**
* Cost drag ~0.08 Sharpe (5 bp × ~0.10 turnover/bar), so net ≈ gross − 0.08; the non-monotone 0.50→0.75 dip
  is 20-seed sampling noise (independent seeds per grid point), not structure.

**Verdict: CHARACTERIZED.** This is the quantitative statement of the audit's V07 finding: at the bench's
own sample the DSR leg is blind below roughly **1 gross Sharpe** (50% coin-flip at ~1.0, reliable detection
only ≥ ~1.4), and the full conjunction is stricter still. The bench never false-fires on a planted null. The
program's "no deployable $0 edge anywhere tested" is therefore, from the bench alone, "no edge above roughly
a 1-Sharpe-equivalent conjunction bar" — exactly as the audit re-scoped it. Runtime 7.1 s.

---

## 4. `capacity.tercile_random_ranking_sharpe` — max-of-50 same-tercile null — **CHARACTERIZED** (kill-bias)

**What was planted.** A TRUE-NULL book: iid Gaussian forward returns (T=1350 bars — EXP-1's ~5.4y panel
scale — A=16 names per tercile, k=5, horizon=5, costs 0 so the order-statistic property is exact). Candidate
= ONE random long-k/short-k same-tercile book (the identical construction, independent seed). The kill
condition "fails its null" fires when the candidate's Sharpe ≤ the **max-of-50** random-book bar the program
used. 30 trials.

| world | fired | raw | add-one 95% CI |
|-------|-------|-----|----------------|
| planted-null book ("fails its null" fires) | 30/30 | **100.0%** | 96.9% [90.8%, 100.0%] |

* Exchangeability prediction: 50/51 ≈ **98.0%** (the bar is a ~98th-percentile order statistic of the null
  Sharpe distribution — measured 30/30, consistent).
* A nominal 5%-level "signal beats random" test would fire on 95.0% of null books.
* **Implied bar height: mean +0.94 / median +0.92 annualized Sharpe** — this is what a true signal must beat
  to survive.

**Verdict: CHARACTERIZED.** The same-tercile null is a kill-biased apparatus: on a book with NO signal it
declares "fails its null" essentially always, and its implicit bar (~+0.9 Sharpe) means a genuine weak
positive would also routinely "fail". Per audit V06 this was **not** decisive for the recorded EXP-1 kill
(the bottom-tercile GROSS was ~0-to-negative, and the recorded max-of-50 nulls were themselves negative net),
but it is a real bias that must be re-designed before any future reuse of this null. Runtime 2.0 s.

---

## Reading the four together

The harness certifies the program's verdict hierarchy about as the audits predicted: the **descriptive
partial-IC instrument is sound** (PASS at its own materiality, silent under zero-anticipation structure);
the **content gate's derived branch is broken as an adjudicator** (zero power where its live inputs lived —
quarantined); and the two **negative-verdict engines are honest but blunt** — the edge bench cannot see
below ~1 gross Sharpe and the capacity null kills ~everything, so their FAILs bound the *tested claim class*,
not the world. No recorded program number is changed by this repair; what changes is that each instrument now
carries a measured operating characteristic instead of an assumed one.

Seeds: `base_seed=0`; per-world offsets documented in `calibration.py` (`+100_000` / `+200_000` for nulls,
`+1000·j` per edge grid point, `+999_983` / `+10_000·(i+1)` for capacity candidate/bar). Regenerate:
`python -m market_os.evaluation.calibration` (~35 s, offline).
