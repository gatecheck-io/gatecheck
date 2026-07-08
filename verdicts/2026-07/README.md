# Verdict Series #1 — receipts (July 2026)

The full record behind the July 2026 issue. Everything here is regenerable: the
gate is the open-source `gatecheck` library plus the formalized claims in
[`claims.json`](claims.json); prices are daily bars from a public vendor.

- **61 claims mined** from what was posted on fintwit, Reddit, SSRN/arXiv, and
  YouTube in the weeks before publication.
- **14 were formalizable** and benched ([`claims.json`](claims.json)).
- **47 were not formalizable** in the gate's vocabulary and are recorded as
  NOT_BENCHABLE with reasons ([`not_benchable.json`](not_benchable.json)) — the
  denominator is the full mined set, never just the benched subset.

## The bench (14 formalized claims)

Net Sharpe is after transaction costs; "b&h" is buy-and-hold on the same asset
over the same window; DSR is the deflated Sharpe with a per-claim multiplicity
haircut. Window ≈ 11.0 years of daily bars.

| claim | source | rule | net Sh | b&h | DSR | verdict |
|---|---|---|---|---|---|---|
| rd_boring_trend_spy | r/quant | SPY > 200d SMA | +0.00 | +0.73 | 0.06 | NOT_TRADABLE |
| rd_boring_trend_qqq | r/quant | QQQ > 200d SMA | +0.00 | +0.88 | 0.06 | NOT_TRADABLE |
| rd_boring_trend_nvda | r/quant | NVDA > 200d SMA | −0.26 | +1.34 | 0.00 | NOT_TRADABLE |
| rd_boring_mom6m_spy | r/quant | 6-mo momentum > 0 | +0.63 | +0.73 | 0.70 | NOT_TRADABLE |
| rd_boring_bband20_spy | r/quant | 20d z-score < −2 | +0.59 | +0.73 | 0.67 | NOT_TRADABLE |
| pa_spy_lag1_reversal | arXiv | lag-1 reversal (author predicts NULL) | −0.39 | +0.73 | 0.01 | NOT_TRADABLE ✓ |
| pa_spy_sma20_fast | SSRN | SPY > 20d SMA | +0.00 | +0.73 | 0.06 | NOT_TRADABLE |
| tw_detrick_42d_surge | media | 42d momentum > 19.5%, hold 12m | +0.83 | +0.73 | 0.71 | NOT_TRADABLE |
| tw_fool_quarter_10pct | media | 63d momentum > 10%, hold 2q | +0.47 | +0.73 | 0.37 | NOT_TRADABLE |
| **tw_fool_month_10pct** | media | 21d momentum > 10%, hold 12m | **+1.19** | +0.73 | **0.98** | **TRADABLE → killed by panel** |
| tw_qqq_ma225_cross | blog | QQQ > 225d SMA | +0.00 | +0.88 | 0.03 | NOT_TRADABLE |
| tw_spy_rsi2_below10 | blog | RSI(2) < 10 | +0.51 | +0.73 | 0.41 | NOT_TRADABLE |
| yt_qe_6wk_rally | YouTube | 30d momentum > 16%, hold 12m | +0.82 | +0.73 | 0.93 | NOT_TRADABLE |
| yt_qs_rsi5_lt30 | YouTube | RSI(5) < 30 | +0.26 | +0.73 | 0.15 | NOT_TRADABLE |

13 of 14 fail the gate outright. The arXiv entry (✓) is a claim whose own author
predicted it would not trade — and it benched negative, as predicted.

## The survivor's autopsy — tw_fool_month_10pct

*Long SPY for 12 months whenever the trailing 21 trading days gained >10%.* Bench:
net Sharpe +1.19 vs buy-hold +0.73, DSR 0.977, out-of-sample positive → TRADABLE.
A survivor is a candidate, not a conclusion; four independent kill-panel probes
then attacked it.

1. **Effective sample — KILLED.** 47 trigger days collapse to ~10 episodes and,
   because the hold resets on re-trigger, just **6 contiguous holding blocks**
   (68% of bars in-market). The single 2020–21 COVID-rebound block is **118% of
   the entire excess** over buy-and-hold. Episode-level inference on the
   differential: 95% CI [−0.33, +0.33], P(≤0) = 0.32.
2. **Regime — KILLED.** Drop the Mar–Aug 2020 triggers → DSR 0.948, fail.
   **2000–2019: net +0.15 vs passive +0.30 — below buy-and-hold for two decades.**
   Extended 2000–now: DSR 0.687, fail. The killing triggers are bull-trap rallies
   in secular bears (Mar 2000 −25% fwd, Apr 2001 −13%, Oct 2001 −18%). The pass
   exists only because the ~11y bench window contains exclusively V-recoveries.
3. **Timing vs. exposure — WEAKENED (the honest counterweight).** The trigger
   *does* beat exposure-matched random gating (P ≈ 0.001) — real regime-placement
   information. But the article's headline **83% forward win rate is BELOW the
   sample's 85.8% unconditional 12-month win rate**: the framing statistic carries
   no information, and the de-tilted timing return alone (+0.68) does not beat
   passive (+0.73).
4. **Fragility & multiplicity — KILLED.** Not threshold-shopped (rank 5/45 in its
   neighbor grid), but knife-edged: 0.027 DSR headroom, 0/30 neighbors pass at
   ±25% hold horizon, and the pass dies at n_trials = 49 — while the same
   publisher ran **≥44 "the market did X, here's what happens next" articles in
   18 months.** At any honest family size, DSR 0.90–0.94, not tradable.

**Final verdict: NOT_TRADABLE.** The descriptive drift is real (15/18 wins,
mean +14.7% since 2000 — the publisher's numbers replicate as *description*);
what dies is the tradable-edge claim, and specifically its evidentiary basis: one
regime, six blocks, one publisher's unlabeled search over a hundred-odd stats.

## Reproducing this

The gate is [`gatecheck`](../../README.md) (this repo). The formalized claims are
[`claims.json`](claims.json) in the posted-edge schema. Prices are public daily
bars. The verdicts follow from costs + deflation + out-of-sample above the
gate's certified detection floor (see
[`../../docs/CALIBRATION_CERTIFICATES.md`](../../docs/CALIBRATION_CERTIFICATES.md)).
