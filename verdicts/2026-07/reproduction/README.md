# July receipt recovery and replay

Recovered October 1, 2026. Start with [the final claim audit](CLAIM_AUDIT.md) and
[the correction review](REVIEW.md). This
directory supplies the missing benchmark runner and seven recovered exploratory
review scripts. It is research tooling under the July receipts; the library's
public API is unchanged.

## What can be reproduced

- **Legacy mapping:** all 14 archived claim definitions, the recorded costs and
  gate, on a newly acquired and explicitly dated historical snapshot.
- **SMA mapping correction:** the same run with the five centered SMA-ratio
  thresholds changed from 1 to 0. Other proxy assumptions remain unchanged.
- **Window sensitivity:** a separate 16-calendar-year run, rather than silently
  treating `252 * 16` calendar days as 16 years.
- **Technical correction:** all 14 common-window proxy definitions with the
  SMA thresholds and Wilder RSI seed fixed, separately labeled.
- **Source-directed audit:** all 14 entries with source availability, stated
  objectives, calendar-event definitions, actual exit rules where specified,
  explicit execution/price sensitivities and unfinished-outcome censoring.
  Missing source rules remain gaps; no aggregate source failure count is made.
- **Exploratory review:** all seven located July probes against the selected
  snapshot, with a fixed July 2 clock and saved stdout/stderr.
- **Frozen SPY regression:** if the original undated SPY NPZ is available,
  axis4 can replay the archived 45-cell grid and trial-count sweep. The local
  recovered cache matched every one of 398 saved numeric values exactly.

The original QQQ/NVDA snapshots, original SPY timestamps and an eighth script
mentioned in old notes have not been found. The full original July run is not
independently reproducible from the recovered assets. Vendor price files remain
outside Git; result JSON, hashes, code and logs are in [evidence](evidence/).
Historical vendor revisions can make a new download differ from the October
snapshot. Verify hashes before calling two inputs identical.

## Run from the repository root

Use a dedicated Python environment. The recorded run used Python 3.12.6,
NumPy 1.26.4 and SciPy 1.14.1. NumPy is sufficient for the claim replay; some
archived uncertainty probes also require SciPy.

```sh
python -m pip install -e ".[test]"
python -m pip install -r verdicts/2026-07/reproduction/requirements.txt
python -m pytest -q

# This is the only command that accesses the price provider.
python verdicts/2026-07/reproduction/fetch_snapshot.py --output verdicts/2026-07/reproduction/data/snapshot

python verdicts/2026-07/reproduction/replay.py --snapshot-dir verdicts/2026-07/reproduction/data/snapshot --output verdicts/2026-07/reproduction/runs/legacy_close.json
python verdicts/2026-07/reproduction/replay.py --snapshot-dir verdicts/2026-07/reproduction/data/snapshot --sma-threshold-correction --output verdicts/2026-07/reproduction/runs/corrected_close.json
python verdicts/2026-07/reproduction/replay.py --snapshot-dir verdicts/2026-07/reproduction/data/snapshot --start 2010-07-02 --sma-threshold-correction --output verdicts/2026-07/reproduction/runs/corrected_16_calendar_years.json

python verdicts/2026-07/reproduction/run_probes.py --snapshot-dir verdicts/2026-07/reproduction/data/snapshot --output verdicts/2026-07/reproduction/runs/probes
```

Output names/directories must be new, so a rerun cannot overwrite prior evidence.
The fetcher stores quote close and adjusted close separately, dated CSVs,
provider responses and SHA-256 metadata. The offline reader checks hashes,
date/timestamp alignment, order, duplicates, coverage and positive finite prices.
It does not silently drop bad bars. Replay output uses `null` for an undefined
metric, such as a flat position's correlation.

For an additional adjusted-close sensitivity of the legacy proxy harness, add
`--price-field adjclose` and choose a new output name. The initial recovery
comparisons use quote close. The separate source audit below includes both
price fields for trading rules; index event studies use quote close.

## Source audit and independent verification

Use new output folders. These commands acquire the longer histories needed by
the source questions; the following analysis commands work offline.

```sh
python verdicts/2026-07/reproduction/fetch_snapshot.py --start 1993-01-01 --assets SPY QQQ NVDA --output verdicts/2026-07/reproduction/data/audit_equities
python verdicts/2026-07/reproduction/fetch_snapshot.py --start 1945-01-01 --assets ^GSPC --output verdicts/2026-07/reproduction/data/audit_index
python verdicts/2026-07/reproduction/replay.py --snapshot-dir verdicts/2026-07/reproduction/data/snapshot --sma-threshold-correction --rsi-seed-correction --output verdicts/2026-07/reproduction/runs/technical_corrections.json
python verdicts/2026-07/reproduction/audited_batch.py --equity-snapshot-dir verdicts/2026-07/reproduction/data/audit_equities --index-snapshot-dir verdicts/2026-07/reproduction/data/audit_index --output verdicts/2026-07/reproduction/runs/audited_batch.json
python verdicts/2026-07/reproduction/independent_verification.py --equity-snapshot-dir verdicts/2026-07/reproduction/data/audit_equities --index-snapshot-dir verdicts/2026-07/reproduction/data/audit_index --batch verdicts/2026-07/reproduction/runs/audited_batch.json --output verdicts/2026-07/reproduction/runs/independent_verification.json
```

Quote `"^GSPC"` when using a shell that treats a caret specially. An acquisition
made later can differ from the hashes recorded here. The paper uses adjusted
SPY log returns and its June cutoff. The monthly study caps observations at
its May publication date. The source audit uses prior history for warmup and
charges both books their initial entry cost; this differs deliberately from
the archived harness accounting described below. RSI5 and ranked-rally rules
are explicitly labeled variants where the source leaves details undefined.

To check an original SPY cache held locally:

```sh
python verdicts/2026-07/reproduction/run_probes.py --frozen-spy path/to/spy_prices_axis4.npz --output verdicts/2026-07/reproduction/runs/frozen
```

This executes axis4 only and compares its output with
[the original numeric archive](archive/axis4_results.json). It never attaches
invented dates or newly downloaded QQQ/NVDA bars to the original SPY cache.

## Accounting and gate conventions

- Signal at close of bar `i`; position held into return `p[i+1]/p[i]-1`.
  Same-close fills are a simplifying execution assumption.
- Cost: `0.0005 * abs(position change)`; entering a unit position costs 5 bp,
  switching from -1 to +1 costs 10 bp. Cash earns zero. Borrow, financing,
  slippage, taxes and capacity are absent.
- The archived passive Sharpe uses asset returns without an entry-cost charge;
  strategy turnover includes the initial entry. This convention is preserved.
- Quote close excludes the adjusted-close dividend treatment. Both fields are
  retained; they must not be mixed within a run.
- SMA ratio is `price/SMA - 1`. The legacy threshold of 1 means
  `price > 2*SMA`. The separately labeled corrected threshold is 0.
- A nonzero signal resets the holding timer. Repeated triggers can merge into
  contiguous positions longer than the nominal horizon.
- Annualization 252; five CV splits with embargo 5; seed 0; 2,000 circular shifts;
  25 random-sign baselines; per-claim trial counts from `claims.json`.
- The initial gate requires Sharpe above passive, DSR at least .95, positive
  discounted Sharpe, and a positive fold-Sharpe confidence interval. The
  circular-shift p-value and random baseline are diagnostics, not pass conditions.
  Fixed-rule CV checks time robustness; it does not fit a strategy or establish
  independent one-year episodes.

These conventions reproduce a harness. A gate pass does not establish that a
publisher's original claim, investment objective or trading implementation was
replicated correctly. Several recorded definitions are explicit proxies.

## Files and provenance

[provenance.json](provenance.json) records hashes of recovered source files,
original and adapted probes, missing assets and portability changes. Original
files are retained locally without modification. The adaptations replace
source-program imports, owner-specific paths and live downloads with local
modules and an offline snapshot adapter. The archived probes' numerical logic
is preserved, including exploratory limitations and older date-boundary
conventions. They are evidence, not a newly calibrated test suite.

[bench_engine.py](bench_engine.py) and [posted_claims.py](posted_claims.py) hold
the recovered engine and schema; [bench_support.py](bench_support.py) supplies
the small baseline/accounting helpers. The statistical primitives come from
this checkout's `gatecheck`. [Tests](../../../tests/test_july_reproduction.py)
check the SMA mapping, next-bar alignment, turnover, timer resets, immutable
input hashes and input rejection with independent small vectors.
[Audit tests](../../../tests/test_july_audit_methods.py) also check the RSI seed,
stateful exits, calendar horizons, execution delay, censoring and pre-1970 dates.
