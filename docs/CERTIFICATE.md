# Calibration certificate — built-in exhibit

Generated output of the package's built-in example certificate:
`gatecheck.calibration.certificate_spearman_partial`, which certifies
`gatecheck.rank.spearman_partial` (at the |IC| ≥ 0.05 AND p < .05 materiality)
on planted truth, a white planted null, and a structured static-coupling null.

Everything below is reproducible byte-for-byte: all randomness flows through
`numpy.random.default_rng(seed)` with the seed schedule listed here.

## Environment

| item | value |
|---|---|
| date generated | 2026-07-02 |
| gatecheck | 0.1.0 (commit at generation: `35a0df6` + docs) |
| python | 3.12.6 (Windows 11) |
| numpy | 1.26.4 |
| command | `python -m gatecheck.calibration` |

## Configuration (full scale — the defaults)

| parameter | value | meaning |
|---|---|---|
| `n_seeds` | 25 | independent world draws per world |
| `n` | 1500 | bars per world (~6 trading years daily) |
| `w` | 5 | forward/trailing vol window |
| `signal_strength` | 0.25 | planted levels-partial corr of signal vs forward vol given `[ret_t, TV_t]` |
| `n_perm` | 1000 | circular-shift permutations per p-value |
| `base_seed` | 0 | first seed of each schedule |

**Seed schedule** (deterministic, from `base_seed = 0`):

- `truth` — `plant_vol_world(seed, 0.25)`, seeds `0 .. 24`
- `null` — `plant_vol_world(seed + 100_000, 0.0)`, seeds `100000 .. 100024`
- `null_static` — `static_coupling_world(seed + 200_000)`, seeds `200000 .. 200024`

**Pass rule (fixed before generation, in code):** fire on truth ≥ 80% AND fire on
*both* nulls ≤ 5% (raw fractions; add-one CIs recorded alongside).

## Result — full scale

```
================================================================================================
CALIBRATION CERTIFICATE — rank.spearman_partial (|IC|>=0.05 & p<.05 materiality)   [PASS]
================================================================================================
planted : AR(1)-vol world, planted forward partial corr ~+0.25 (truth) / 0 (null) / static-coupling carrier (structured null); n=1500, w=5
  truth            fired  24/25  raw 96.0%   add-one 92.6% [82.7%, 100.0%]
  null             fired   0/25  raw  0.0%   add-one  3.7% [ 0.0%, 10.8%]
  null_static      fired   1/25  raw  4.0%   add-one  7.4% [ 0.0%, 17.3%]
n_perm : 1000
pass_rule : truth >= 80% fire AND both nulls <= 5% fire (raw fractions)
runtime : 9.0s
================================================================================================
```

| world | fired | raw rate | add-one rate [95% CI] | pass bound | met |
|---|---|---|---|---|---|
| truth (planted +0.25) | 24/25 | 96.0% | 92.6% [82.7%, 100.0%] | ≥ 80% | yes |
| null (planted 0) | 0/25 | 0.0% | 3.7% [0.0%, 10.8%] | ≤ 5% | yes |
| null_static (structured) | 1/25 | 4.0% | 7.4% [0.0%, 17.3%] | ≤ 5% | yes |

**Verdict: PASS.** Runtime 9.0 s (single core; wall time on the generation
machine — runtimes are hardware-dependent, fire counts are not).

## Realized effect size on truth

The planted quantity is a *levels* partial correlation of 0.25; the gate reads a
rank-space Spearman partial, which is attenuated. Across the 25 truth seeds
(seeds `0..24`, `n_perm=0` point estimates):

- median realized rank-space partial IC: **+0.193**
- range: **+0.114 .. +0.328**

So the certificate demonstrates ≥ 96% power at a realized rank-IC of roughly
0.19 at n = 1500 — it says nothing about power at weaker effects or shorter
samples. To characterize the detection floor, sweep `signal_strength` the way
`plant_edge_world` sweeps planted Sharpe.

## Result — fast scale (`--fast`, the test-suite arm)

Configuration: `n_seeds=6, n=1200, n_perm=200` (other parameters at defaults;
same seed schedule, truncated to 6 seeds).

```
================================================================================================
CALIBRATION CERTIFICATE — rank.spearman_partial (|IC|>=0.05 & p<.05 materiality)   [PASS]
================================================================================================
planted : AR(1)-vol world, planted forward partial corr ~+0.25 (truth) / 0 (null) / static-coupling carrier (structured null); n=1200, w=5
  truth            fired   6/6   raw 100.0%   add-one 87.5% [64.6%, 100.0%]
  null             fired   0/6   raw  0.0%   add-one 12.5% [ 0.0%, 35.4%]
  null_static      fired   0/6   raw  0.0%   add-one 12.5% [ 0.0%, 35.4%]
n_perm : 200
pass_rule : truth >= 80% fire AND both nulls <= 5% fire (raw fractions)
runtime : 0.6s
================================================================================================
```

## Reading notes (the skeptical ones)

- **The 1/25 fire on `null_static` is consistent with the design.** The gate's
  significance leg targets a 5% size; one fire in 25 structured-null draws is
  the expected order of magnitude, and the pass rule (raw ≤ 5%) is met exactly
  at its boundary. At `n_seeds=25` the add-one CI on that null reaches 17.3% —
  if a ≤ 5% size matters to your decision, run more seeds.
- **The pass rule was fixed in code before the numbers were generated** — it is
  the same rule this construction was held to upstream. Deciding thresholds
  after seeing fire rates would make the certificate circular.
- **This is a certificate for THIS gate on THESE worlds.** It transfers to your
  gate and your data only to the extent your nuisance geometry (persistence,
  coupling, smoothing) is represented by these worlds. Certify your own gate:
  see the Quickstart in the README.

## Reproduce

```bash
python -m gatecheck.calibration          # full scale, ~9 s
python -m gatecheck.calibration --fast   # fast arm, ~1 s
```

Identical seeds produce identical fire counts on any platform with the same
numpy generation stream (PCG64 default).
