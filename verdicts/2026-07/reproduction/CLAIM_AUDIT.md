# July claim audit and final local rerun

Prepared October 1, 2026. **The July conclusion needs revision.** All 14 stored
entries were audited and rerun against explicit definitions. Verification of
the original sources is partial: some bodies are unavailable and several rules
are incompletely specified. No aggregate count of rejected source claims is
supported. This package records the corrected research evidence and its limits.

## What changes the conclusion

1. **Five moving-average thresholds tested the wrong rule.** The recovered
   indicator is `price/SMA - 1`; threshold 1 requires price above twice its
   average. Threshold 0 implements price above its average. A second defect
   counted the last RSI seed change twice. Both corrections are isolated from
   the preserved legacy code. The corrected common-window harness has three
   initial passes, but that count still concerns proxy definitions rather than
   fourteen faithfully reproduced source claims.
2. **The monthly example's descriptive statistic largely reproduces.** Using
   the S&P 500 index, calendar months and the source table's era, all 13 event
   identities match. Of 12 completed one-year outcomes, 10 gained: **83.3%**, with
   a mean gain of **16.35%**. The matched unconditional comparison is 479 gains
   in 616 completed monthly one-year windows: **77.8%**, mean **10.17%**. Outcomes
   are limited to information available on May 5, 2026. The source's displayed
   table begins in 1974, despite its “50 years” prose. These small, overlapping
   samples establish descriptive agreement, not statistical significance or
   a deployable trading edge. The old 83% versus 85.8% comparison does not reject
   [this claim](https://www.fool.com/investing/2026/05/05/sp-500-did-this-13-time-50-years-what-happens-next/).
3. **Risk reduction and statistical tests need their own outcomes.** Under the
   stored Reddit SMA200 protocol, SPY maximum drawdown is 24.0% versus passive
   56.5%, using quote close, a next-close execution delay and 3 bp turnover cost
   over 2005–July 2026. Return is lower. This supports the direction of a risk
   reduction claim under those assumptions, although the removed original body
   prevents exact verification. Separately, the
   [paper's statistical example](https://arxiv.org/html/2606.29591v1)
   reproduces its rounded lag-one correlation and sign-test numbers. A trading
   rule failing a Sharpe gate neither replicates that test nor proves its null.
4. **The exploratory panel is evidence with limits.** Its frozen SPY replay
   matches all 398 archived numbers. That verifies recovered computations,
   while leaving source-rule fidelity, calibration of the rejection instrument,
   independence of event outcomes and the publishers' search history unresolved.

## Every stored entry accounted for

“Partial” below means an explicit diagnostic was completed, with the stated
source or implementation gap retained. It does not mean the source was refuted.
The [registry](audit_registry.json) records source availability and individual
mapping defects; the [full batch](evidence/audited_batch.json) records every
variant, date, input hash, completed denominator and censored outcome.

| Stored entry | Source question and diagnostic | Audit conclusion / remaining gap |
| --- | --- | --- |
| `rd_boring_trend_spy` | SMA200 return and drawdown, 2005–July 2026; 3 bp, next close; quote/adjusted close | Drawdown falls from 56.5% to 24.0% on quote close; return falls. Original Reddit body is removed, so exact source replication is unavailable. |
| `rd_boring_trend_qqq` | Same protocol for QQQ | Quote-close drawdown falls from 53.6% to 23.0%; return falls. Same source gap. |
| `rd_boring_trend_nvda` | Same protocol for NVDA | Quote-close drawdown falls from 85.1% to 52.6%; return increases in this selected stock. Same source gap; no general stock-selection inference. |
| `rd_boring_mom6m_spy` | Explicit SPY 126-bar momentum benchmark | The stored source record does not assert this SPY performance result. Treat it as a benchmark extension, not a failed source claim. |
| `rd_boring_bband20_spy` | Explicit SPY 20-bar population-SD band benchmark | The stored record names other assets' results. A SPY extension cannot refute those results; the source's band convention is unverified. |
| `pa_spy_lag1_reversal_null_fri` | Adjusted SPY log returns, 1993–June 2026; source sign coding and normal approximation | 8,403 returns; correlation −.08059, correlation z −7.387, sign p .1112. Rounded source statistics reproduce. This is neither a known-ground-truth calibration nor proof of no predictability. |
| `pa_spy_sma20_fast_timing` | SMA20 versus SMA50 above SMA200, 2006–2025; declared price/fill sensitivities | Slow comparator compounds more in these variants. Full SSRN paper is inaccessible; exact source implementation and costs are unverified. |
| `tw_detrick_42d_surge_195` | Index ≥19.5% rolling 42-bar gain since 1945; calendar forward returns; all triggers and a declared clustering sensitivity | Original event separation/table is unavailable. Threshold days and a chosen cluster rule cannot establish the quoted historical sample. |
| `tw_fool_quarter_10pct` | Index calendar-quarter ≥10% gains, trailing decade; each next quarter must gain | Seven completed events, six successes (85.7%); June 2026 is an eighth, censored event. Matched baseline 22/38 (57.9%). Completed rate agrees with the article; its upstream NYT event table and total-count interpretation remain unverified. |
| `tw_fool_month_10pct` | Index calendar-month ≥10% gains; source table era; matched unconditional baseline | All 13 identities match; one-year result 10/12 (83.3%) versus 479/616 (77.8%). Three-month completed result is 9/12 (75%), rather than the stored 69%. Partial descriptive replication; no significance claim. |
| `tw_qqq_ma225_cross` | QQQ SMA225, January 2000–February 2025; both price fields and same/next-close fills, adopted 5 bp | Return/drawdown improve over passive in these variants, but the exact published 1,061% return and 28.6% drawdown do not reproduce. Unknown source fills, costs, adjustment and exact sessions prevent a settled verdict. |
| `tw_spy_rsi2_below10` | Correct Wilder RSI2; cross below 10, exit above 80; both fields/fill timings | Quote/same-close variant: CAGR 7.99%, drawdown 34.96%; next-close: 5.18%, 42.05%. The original five-day hold did not implement the exit rule. Source sample/execution remains incomplete. |
| `yt_qe_6wk_rally_252d` | Index retrospective top-20, non-overlapping 30-return rallies since 1950; explicitly declared greedy ranking | 17/19 completed one-year outcomes gain (89.5%), mean 19.17%, under this selection. Source's exact non-overlap algorithm is unstated; this differs from its reported rate and cannot settle its claim. Retrospective selection is not a live trading rule. |
| `yt_qs_rsi5_lt30_spy` | Correct RSI5, exit above 50; explicit one-day-rising SMA200 variant without unspecified confirmation | Adjusted/same-close variant: CAGR 5.14%, drawdown 14.43%, 199 completed blocks. Source confirmation, trend slope, dates and fills are unspecified. This is a labeled variant, not an exact replication. |

Primary source bodies were read for the paper, monthly/quarterly articles,
QQQ225, ranked rallies and RSI5. The RSI2 site's access check prevented reading
its original body; the publisher's
[alternate statement](https://www.linkedin.com/posts/quantifiedstrategies_tradingstrategy-activity-7444337121932267521-sCf6)
supplied the crossing/exit rules. Detrick's available coverage is secondary.
The removed Reddit post and inaccessible SSRN full text are explicit gaps.
Source availability is not evidence that a source's claim is false.

Other source links:
[quarter](https://www.fool.com/investing/2026/07/01/the-sp-500-posted-its-best-quarterly-performance-s/),
[QQQ225](https://www.financialwisdomtv.com/post/qqq-trading-strategy-that-beats-the-market-proven-backtest-results),
[ranked rallies](https://quantifiableedges.com/a-historical-look-at-the-top-20-6-week-spx-rallies-since-1950/),
[RSI5](https://quantifiedstrategies.substack.com/p/rsi-30-50-strategy-for-beginners).

## Verification and reproducibility

The original 14 definitions and archived receipts are unchanged. The new source
audit is separate from the legacy harness. It uses pre-sample indicator warmup,
explicit costs, both initial entry costs, separate quote/adjusted prices,
same-close/next-close sensitivities where source execution is unknown, and
censoring of unfinished forward windows and open holding blocks. Cash earns
zero; borrow, financing, slippage, taxes and capacity are outside scope.

The newly acquired snapshots contain 8,413 SPY, 6,871 QQQ, 6,903 NVDA and 20,493
index rows, ending July 2, 2026. Hashes, dates, positive finite prices, timestamp
alignment and uniqueness are checked by the offline reader; no bad bars are
silently discarded. Price-field adjustments and historical vendor revisions
remain provider limitations. Original QQQ/NVDA snapshots and original SPY dates
are still missing. Vendor files stay local; exact reruns require the recorded
snapshot hashes or an explicitly labeled new acquisition.

[Independent verification](evidence/independent_verification.json), calculated
directly from raw CSV without the audit helpers, passed the monthly event
identities and annual counts/means, paper correlation/sign figures, and SPY200
and QQQ225 quote/next-close return and drawdown. This is representative numerical
verification, not independent recomputation of every sensitivity. The full
test suite passed **113 tests**. Seven recovered exploratory scripts completed;
the frozen axis4 regression matched 398 numbers exactly.

The [common-window technical correction](evidence/technical_corrections_common_window.json)
retains the July proxy vocabulary and fixes SMA/RSI only. It still has three
initial passes. The [source-directed batch](evidence/audited_batch.json) accounts
for all fourteen entries and intentionally records no aggregate failure count.
See [run instructions](README.md) and [provenance](provenance.json).
The scoped [validation record](VALIDATION.md) distinguishes complete batch
coverage from partial exact-source and independent numerical verification.

## Required editorial action

The historical July article needs a dated correction of the “13 of 14 fail”
headline, the monthly baseline comparison, and the 118% attribution. Its 118%
calculation concerns avoided returns during a 2020 cash interval, not gains
from a 2020–21 long holding block. Preserve the historical record and link the
corrected methods. Exact source gaps should remain visible. A proposed note
is saved in [the correction review](REVIEW.md); it has not been published.

The next worked example can use the fully inspectable calendar-month table,
with the question, comparator and dependence treatment fixed before inference.
Do not promote the rerun as fourteen settled claim verdicts.
