# Changelog

All notable changes to `gatecheck` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/) (pre-1.0: minor bumps may break API).

## [0.1.0] — 2026-07-02

Initial extraction. The statistical spine of the `market_state` research program
(post-audit), repackaged as a standalone, numpy-only, pip-installable library.
No upstream code is imported; lineage is recorded per-module in docstrings.

### Added

- `gatecheck.calibration` — planted-truth / planted-null gate certification:
  world generators (`plant_vol_world`, `static_coupling_world`, `content_world`,
  `plant_edge_world`), the generic `certify()` runner with add-one binomial CIs,
  the `Certificate` dataclass, and the built-in PASS exhibit
  `certificate_spearman_partial` (runnable via `python -m gatecheck.calibration
  [--fast]`; generated numbers in `docs/CERTIFICATE.md`).
- `gatecheck.deflation` — Deflated Sharpe Ratio with an explicit per-period
  `sigma_sr` units contract, expected-max-Sharpe multiplicity benchmark, PBO
  proxy.
- `gatecheck.cv` — purged + embargoed walk-forward CV as pure, data-free integer
  index math.
- `gatecheck.significance` — PIT-aligned IC with explicit next-bar alignment,
  permutation and autocorrelation-aware circular-shift p-values, multi-seed
  Sharpe aggregation with the small-n Student-t CI repair (`t_quantile_95`;
  versioned `ci='t'`/`'z'` behavior).
- `gatecheck.oos` — train-only-standardized ridge / ridge-logistic OOS
  primitives, train-mean-referenced OOS R², Brier score.
- `gatecheck.incremental` — incremental OOS R² over a baseline with the
  leaked-oracle power floor (`oracle_floor_r2`, `materiality_floor`,
  `ORACLE_POWER_MIN`) and the pass-through diagnostic (`incremental_gate`,
  genericized `IncrementalVerdict`).
- `gatecheck.rank` — rank-space partial correlations (`rank_average`,
  `residualize`, `spearman_partial`), designed to prevent the
  OLS-levels-then-Spearman sign-flip on binary/discrete carriers; includes a
  binary-carrier sign-preservation regression test.
- Test suite: 94 offline, deterministic tests (no network, no data files).
- Docs: README for a skeptical quant/dev audience; `docs/CERTIFICATE.md` with
  the generated exhibit certificate (seeds, fire rates, CIs, runtimes); MIT
  license.

### Known limitations (see README "Limitations")

- Worlds are stylized daily-bar synthetics; certification on them is necessary,
  not sufficient.
- Name confirmed `gatecheck` (2026-07-02); PyPI name available; not yet published
  (owner-gated launch step).
- Provenance links resolve once the GitHub org is created at launch.
