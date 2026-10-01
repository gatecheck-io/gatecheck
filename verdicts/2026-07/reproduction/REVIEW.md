# Recovery findings and proposed July corrections

Prepared October 1, 2026. This review records the evidence supporting a dated
Substack correction. Publication does not establish a new trading verdict.

**Final source audit:** [CLAIM_AUDIT.md](CLAIM_AUDIT.md) accounts for every stored
entry and supersedes the preliminary source-verification discussion below.
The full source-directed rerun and representative independent checks are done.
The original “13 of 14 fail” headline is unsupported; three corrected proxy
gate passes are not a replacement source-verdict count. Original source access
and unspecified rules still limit exact replications.

## The headline needs correction

Five archived SMA definitions use `threshold=1`, while the engine calculates
`price/SMA - 1`. The published descriptions say above the moving average;
the code instead requires price to exceed twice the moving average.
Consequently, the near-zero SPY/QQQ trend results do not evaluate the rules
described in the July table.

On newly acquired quote-close bars from June 19, 2015 through July 2, 2026,
the unchanged legacy mapping yields one initial gate pass. Correcting those
five thresholds yields three initial passes: QQQ SMA200, NVDA SMA200 and the
SPY rolling-21-day momentum proxy. These are harness results under the recorded
assumptions, not exact replications of the publishers' full trading rules.
The corrected SMA rules have not undergone the survivor's exploratory panel.

| Affected mapping | Legacy net Sharpe | Corrected net Sharpe | Passive Sharpe | Corrected initial gate |
| --- | ---: | ---: | ---: | --- |
| SPY SMA200 | 0.000 | 0.719 | 0.730 | fail |
| QQQ SMA200 | 0.000 | 0.991 | 0.876 | pass |
| NVDA SMA200 | -0.264 | 1.413 | 1.335 | pass |
| SPY SMA20 | 0.000 | 0.561 | 0.730 | fail |
| QQQ SMA225 | 0.000 | 0.867 | 0.876 | fail |

See the complete [legacy result](evidence/legacy_close.json) and
[corrected result](evidence/corrected_close.json), including exposure,
turnover, drawdown, dates, gate components, environment and snapshot hashes.
The original claim definitions remain untouched to preserve the record.

## Separate the time window from the mapping correction

The old fetcher used `252 * 16` as calendar days: 4,032 days, roughly eleven
years. It did not request sixteen years of trading bars. The default replay
preserves the observed June 19, 2015 start. A separately labeled
[16-calendar-year result](evidence/corrected_16_calendar_years.json) uses
July 2, 2010 through July 2, 2026 and the corrected SMA thresholds. Indicators
start from the selected interval in both cases; there is no prior-window warmup.
Neither rerun reproduces an original publisher's different starting date.
The corrected sixteen-year run has two initial passes (NVDA SMA200 and the
SPY rolling-month proxy); QQQ SMA200 no longer beats passive in that sample.

## The 83% versus 85.8% comparison is not a matched test

The [publisher's source article](https://www.fool.com/investing/2026/05/05/sp-500-did-this-13-time-50-years-what-happens-next/)
describes calendar-month S&P 500 gains over a much longer history. The bench
uses overlapping rolling 21-trading-day SPY gains over roughly eleven years.
Different instruments, event definitions and samples cannot establish that
the source's statistic carries no information.

The recovered timing probe reports 85.8% positive forward returns for all
2,523 complete rolling 252-bar windows in the shorter sample. Under its own
rolling trigger definition, all 42 complete trigger-day forward returns are
positive; five later triggers have incomplete forward windows. Those
observations overlap. These descriptive numbers do not validate either an
independent-event inference or the publisher's calendar-month claim.
The completed source-directed check now matches all 13 source-table event
identities. Ten of twelve completed one-year outcomes gain (83.3%), mean 16.35%,
versus a matched 479/616 (77.8%), mean 10.17%, unconditional baseline. Observations
stop at the May 5 publication date. These dependent, small event samples do not
establish significance; they invalidate the old unmatched refutation.

## The 118% attribution was misstated

The recovered avoided-return calculation attributes approximately 118% of
gross additive excess to the **flat stretch during February–April 2020**.
It does not attribute that number to the 2020–21 long rebound holding block.
The denominator is a sum of daily return differences, rather than compounded
wealth or a percentage of Sharpe; that qualification belongs with the number.
Forcing long through that flat stretch reduces net Sharpe from about 1.186
to .877, still above passive .730 in this sample.

The archived [episode-null log](evidence/probes/axis1_episode_null/stdout.txt)
and [honest-CI log](evidence/probes/axis1_honest_ci/stdout.txt) retain both
calculations. The six-block avoided-return interval is approximately
[-.076, .115], with two-sided p=.627; other exploratory episode or bootstrap
calculations answer different questions and should not be interchanged.

## What the recovered panel supports

All seven located scripts ran successfully on the October snapshot with the
July cutoff. The longer-window rolling proxy remains fragile: the recovered
regime probe reports DSR about .948 after dropping March–August 2020 triggers
and .687 on 2000–July 2026. Its archived numeric grid and trial-count sweep
were also replayed against the original SPY cache: all 398 recorded numbers
matched exactly, including 7/45 grid passes and the DSR crossing at 49 trials.

This establishes that recovered code reproduces that saved evidence. It does
not certify the exploratory panel as a calibrated rejection instrument,
prove a specific publisher's hidden search breadth, or settle a source claim
that was translated into a different rule. Treat the old final verdict as
historical and under review.

## Remaining reproduction limits

- Seven scripts were located; the old notes mention eight. No script is
  invented to complete that count.
- The original SPY cache has 2,775 prices and no timestamps. Original QQQ/NVDA
  downloads are missing. The October SPY slice has the same length but differs
  by up to about .02 per share; all-asset reruns are new evidence.
- The initial recovery snapshots contain 6,664 rows per asset, January 3, 2000 through
  July 2, 2026. Quote close and adjusted close are stored separately; the
  recorded runs use quote close, not a dividend-adjusted total-return series.
- RSI entries omit some source exit/trend rules; calendar-quarter and
  calendar-month descriptions become rolling-day proxies. A Sharpe hurdle
  also does not by itself evaluate a source's drawdown objective. New output
  includes drawdown on the matched sample but does not recreate source periods.
- Cash return, costs, same-close fills, holding resets, passive entry cost,
  CV interpretation and trial-count assumptions are explicit in the
  [runner documentation](README.md). They are not changed retroactively.
- A second audit acquisition supplies longer histories and both price-field
  sensitivities for trading rules. Correct RSI seed/stateful exits and calendar
  event checks are in the source-directed batch. Source bodies, execution,
  confirmation and selection gaps are listed individually in the final audit.

## Dated correction text for the article

> October 1, 2026: I found errors in the July benchmark's moving-average and RSI
> implementations, along with mismatches between several source questions and
> the rules I tested. The “13 of 14 fail” headline is withdrawn. A source-directed
> rerun reproduces the monthly example at 83.3%, against a matched 77.8% baseline;
> the previous 85.8% comparison used a different sample and event definition.
> These descriptive rates do not prove a trading edge. The 118% attribution also
> concerns a 2020 cash interval, rather than the rebound holding block. The
> original results remain as a historical record alongside corrected code,
> independent checks and explicit source-verification limits.

The archived receipts preserve the original record; the correction explains
why its aggregate verdict and unmatched comparison are withdrawn.
