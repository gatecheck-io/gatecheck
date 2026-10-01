# Final local validation

**Share the research record with caveats; July's original verdict is withdrawn.**
Coverage of the fourteen-entry ledger and batch is complete. Verification of
exact original sources and independent recomputation of every sensitivity is
partial. This validation covers the research record and linked evidence.

Start with [the claim audit](CLAIM_AUDIT.md). The counts below describe the
current corrected package, rather than certifying every historical claim.
Totals are scoped inventory counts, not complete verification. `0 / N` means no
remaining demonstrated defect observed within that category; it does not mean
all N items received complete independent verification. Categories overlap.

## Research record quality

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Research usefulness and completeness | 0 / 3 | Recovery, per-source questions and supported editorial changes are covered. Exact original replication remains partial. |
| Analytical clarity | 0 / 4 | Four authored review documents distinguish corrected results, proxy-gate counts, historical evidence and source gaps. |
| Visual and interaction consistency | N/A | Plain Markdown record; no dashboard controls or quantitative charts. |

## Analytical correctness and robustness

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Source authority and confidence | 0 / 14 | All entries trace to retained or accessed sources with gaps disclosed. Exact source-definition checks remain incomplete on 12 entries; the paper statistics and monthly table received specified-source checks. |
| SQL/value accuracy | 0 / 14 | All entries ran. Independent raw-CSV recomputation covers monthly annual figures, paper figures and selected SPY200/QQQ225 metrics. The other 10 entries and remaining sensitivities are not independently recomputed. |
| Within-chart agreement | N/A | No quantitative charts. |
| Complete source details | 0 / 14 | Each entry records source availability and definitions; the batch includes sample, price-field and snapshot provenance. Unavailable source details are named gaps. |
| Cross-artifact consistency | 0 / 4 | Authored document links resolve; unchanged claim hashes and the independent-check batch hash match the final evidence. |
| Data-quality controls | 0 / 4 | All four actual snapshots pass hash, dates/order, row-count, timestamp and positive-price checks. Adjusted prices are present; provider revisions remain a limitation. |
| Conclusion support | 0 / 14 | Current per-entry conclusions are qualified diagnostics, with no aggregate rejected-source-claim count. Publication interfaces are outside this analytical validation. |

## Prioritized problems and remedies

1. **July headline and source questions (P0).** Wrong SMA thresholds and a
   common trading gate failed to evaluate several stated objectives.
   **Fixed locally:** isolate the corrected threshold and RSI seed, account
   for each source question, and remove the aggregate source-failure count.
   **Published correction:** the dated note in [REVIEW.md](REVIEW.md) is live,
   with links in [PUBLICATION.md](PUBLICATION.md).
2. **Monthly comparison and 118% attribution (P1).** The earlier baseline used
   a different instrument, period and event convention; the attribution named
   a different episode. **Fixed locally:** the calendar/index result is 10/12
   (83.3%) against 479/616 (77.8%); the 118% figure is identified as avoided
   additive returns during the 2020 cash stretch. Neither is a causal or
   significance result.
3. **Source availability and underspecified rules (P1 limitation).** The Reddit
   body, SSRN full text and some upstream event data cannot be inspected;
   other sources omit execution or selection details. **Remaining evidence
   needed:** original bodies/tables and precise rules before exact verdicts.
   Declared variants remain diagnostics. The audit does not infer source
   falsehood from missing evidence.
4. **Inference and original-data reproduction (P1 limitation).** Original
   QQQ/NVDA snapshots and SPY dates are missing; calendar outcomes overlap and
   retrospective rank selection is not an ex-ante strategy. **Remaining work
   for a new inference:** freeze a specific question and dependence treatment,
   then evaluate it as a separate study. Preserved panel results and current
   instrument calibration do not remove these limitations.

The coverage helper checked the accounting of the retained local review record.
It does not certify scientific truth. Independent numerical coverage and
reproduction limits are recorded in [CLAIM_AUDIT.md](CLAIM_AUDIT.md),
[provenance.json](provenance.json) and the linked evidence.
