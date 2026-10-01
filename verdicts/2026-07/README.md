# Verdict Series #1 — July 2026 receipts

**Recovery review, October 1, 2026:** five moving-average definitions used an
incorrect threshold. The July “13 of 14 fail” headline cannot stand as a result
for the rules described in its table. The unmatched base-rate comparison and
the attribution of the 118% figure also need correction. See the complete
[review record](reproduction/REVIEW.md) before citing those conclusions.
The completed [source audit](reproduction/CLAIM_AUDIT.md) accounts for all
fourteen entries. It supports no aggregate rejected-source-claim count;
historical descriptions, drawdown strategies and statistical tests ask
different questions. The monthly statistic largely reproduces on its own
calendar/index definition, with a matched baseline of 77.8% rather than 85.8%.

The [original receipts](reproduction/archive/original_receipts.md) and
[original numeric results](reproduction/archive/axis4_results.json) are retained
as historical evidence. Their preservation does not endorse their conclusions.
The [dated correction text](reproduction/REVIEW.md) explains these errors and
links this research record. The historical publication date should be preserved.

## The recorded claim set

- 61 mined claims were recorded in July.
- [14 definitions](claims.json) were mapped to the benchmark vocabulary.
- [47 other claims](not_benchable.json) were recorded with reasons they could
  not be mapped. They remain part of the denominator.

The original JSON definitions remain unchanged. Some definitions are proxies
for fuller trading rules or calendar conventions; the runner does not establish
that every original publisher's claim was faithfully replicated.

## Newly acquired data and corrected reruns

All these runs use quote-close data acquired October 1, with a July 2, 2026
cutoff. They are distinct from the original July downloads.

| Run | Date window | Initial gate passes |
| --- | --- | ---: |
| [Legacy definitions](reproduction/evidence/legacy_close.json) | 2015-06-19–2026-07-02 | 1/14 |
| [Five SMA thresholds corrected](reproduction/evidence/corrected_close.json) | 2015-06-19–2026-07-02 | 3/14 |
| [SMA correction plus 16 calendar years](reproduction/evidence/corrected_16_calendar_years.json) | 2010-07-02–2026-07-02 | 2/14 |
| [SMA and RSI seed corrections](reproduction/evidence/technical_corrections_common_window.json) | 2015-06-19–2026-07-02 | 3/14 |

An initial pass is a candidate under this harness's assumptions, not a final
trading verdict. The two newly passing SMA rules in the shorter sample have
not undergone the rolling-momentum survivor's exploratory review.
The [source-directed batch](reproduction/evidence/audited_batch.json) is
separate: it measures source outcomes and labeled sensitivities, rather than
assigning all entries a Sharpe-gate verdict. Representative
[independent calculations](reproduction/evidence/independent_verification.json)
passed. Exact original source replications remain partial as documented.

## Reproduction status

The [replay package](reproduction/README.md) contains a standalone acquisition
command, a strict offline reader, the recovered benchmark, seven adapted review
scripts, recorded logs/results, hashes and small-vector verification tests.
The library's statistical API is unchanged.

The original undated SPY cache reproduced all 398 saved axis4 numbers exactly.
Original QQQ/NVDA snapshots, original SPY timestamps and an eighth script
mentioned in old notes remain missing. Consequently, the full original July
run cannot be claimed as independently reproduced. Vendor files stay local;
the package documents acquisition and input hashes for new reruns.
