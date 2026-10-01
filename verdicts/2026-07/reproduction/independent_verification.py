"""Recompute consequential results directly from CSV, without the audited helpers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def read_csv(directory, asset):
    body = (directory / f'{asset}.csv').read_bytes()
    return list(csv.DictReader(body.decode().splitlines())), hashlib.sha256(body).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equity-snapshot-dir', type=Path, required=True)
    parser.add_argument('--index-snapshot-dir', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output filename')
    batch = json.loads(args.batch.read_text())
    expected = {r['claim_id']: r for r in batch['results']}
    rows, index_hash = read_csv(args.index_snapshot_dir, '^GSPC')
    # Retain the final raw CSV close per calendar month, at the publication cutoff.
    monthly = {}
    for row in rows:
        if row['date'] <= '2026-04-30':
            year, month = map(int, row['date'][:7].split('-'))
            monthly[year * 12 + month - 1] = float(row['close'])
    first, last = 1974 * 12, 2026 * 12 + 3
    candidates = [k for k in sorted(monthly) if first <= k <= last and k - 1 in monthly]
    event_keys = [k for k in candidates if monthly[k] / monthly[k - 1] - 1 >= .1]
    complete = [k for k in event_keys if k + 12 in monthly]
    gains = [monthly[k + 12] / monthly[k] - 1 for k in complete]
    baseline_keys = [k for k in candidates if k + 12 in monthly]
    baseline_gains = [monthly[k + 12] / monthly[k] - 1 for k in baseline_keys]
    stats = dict(events=len(event_keys), complete=len(gains), wins=sum(g > 0 for g in gains),
                 win_rate=sum(g > 0 for g in gains) / len(gains), mean_return=statistics.mean(gains),
                 baseline_complete=len(baseline_gains), baseline_wins=sum(g > 0 for g in baseline_gains),
                 baseline_win_rate=sum(g > 0 for g in baseline_gains) / len(baseline_gains))
    result = expected['tw_fool_month_10pct']['diagnostics'][0]
    assert stats['events'] == result['event_count']
    for key in ['complete', 'wins', 'win_rate', 'mean_return']:
        assert math.isclose(stats[key], result['conditional']['12'][key], abs_tol=1e-12)
    assert stats['baseline_complete'] == result['baseline']['12']['complete']
    assert stats['baseline_wins'] == result['baseline']['12']['wins']
    # Event identities are compared to the primary article's displayed table.
    published_months = {'1974-10', '1975-01', '1976-01', '1980-11', '1982-08', '1982-10',
                        '1984-08', '1987-01', '1991-12', '2011-10', '2020-04', '2020-11', '2026-04'}
    event_months = {f'{k // 12:04d}-{k % 12 + 1:02d}' for k in event_keys}
    assert event_months == published_months

    spy, spy_hash = read_csv(args.equity_snapshot_dir, 'SPY')
    prices = [float(r['adjclose']) for r in spy if r['date'] <= '2026-06-19']
    returns = [math.log(b / a) for a, b in zip(prices, prices[1:])]
    average = statistics.mean(returns)
    centered = [r - average for r in returns]
    rho = sum(a * b for a, b in zip(centered, centered[1:])) / sum(r * r for r in centered)
    same = sum((a > 0) == (b > 0) for a, b in zip(returns, returns[1:]))
    frequency = same / (len(returns) - 1)
    z = (2 * frequency - 1) * math.sqrt(len(returns) - 1)
    sign = dict(observations=len(returns), lag1_acf=rho, acf_z=rho * math.sqrt(len(returns)),
                sign_z=z, sign_p=math.erfc(abs(z) / math.sqrt(2)))
    reference = expected['pa_spy_lag1_reversal_null_fri']['diagnostics'][0]
    assert sign['observations'] == reference['log_return_observations']
    assert math.isclose(sign['lag1_acf'], reference['lag1_autocorrelation'], abs_tol=1e-10)
    assert math.isclose(sign['sign_p'], reference['sign_two_sided_normal_p'], abs_tol=1e-12)

    # Verify selected return/drawdown metrics from raw close ratios and rolling sums,
    # rather than calling the trading or indicator helpers under review.
    equity_checks = []
    for asset, window, cost, start, end, lag, cid in [
        ('SPY', 200, .0003, '2005-01-01', '2026-07-02', 1, 'rd_boring_trend_spy'),
        ('QQQ', 225, .0005, '2000-01-01', '2025-02-28', 1, 'tw_qqq_ma225_cross')]:
        raw, sha = read_csv(args.equity_snapshot_dir, asset)
        px = [float(r['close']) for r in raw]
        signal = [0.] * len(raw)
        for i in range(window - 1, len(raw)):
            signal[i] = float(px[i] > sum(px[i - window + 1:i + 1]) / window)
        selected = [i for i in range(len(raw) - 1) if start <= raw[i + 1]['date'] <= end]
        wealth = peak = 1.
        worst, previous = 0., 0.
        for i in selected:
            position = signal[i - lag] if i >= lag else 0.
            daily = position * (px[i + 1] / px[i] - 1) - cost * abs(position - previous)
            wealth *= 1 + daily
            peak = max(peak, wealth)
            worst = min(worst, wealth / peak - 1)
            previous = position
        comparison = next(d for d in expected[cid]['diagnostics']
                          if d['price_field'] == 'close' and d['execution_lag_closes'] == lag)
        assert math.isclose(wealth - 1, comparison['strategy']['total_return'], abs_tol=1e-9)
        assert math.isclose(worst, comparison['strategy']['maximum_drawdown'], abs_tol=1e-9)
        equity_checks.append(dict(claim_id=cid, csv_sha256=sha, total_return=wealth - 1,
                                  maximum_drawdown=worst, matched=True))
    output = dict(independent_checks_passed=True, monthly_calendar=stats,
                  monthly_source_event_identities_match=True, paper_sign=sign,
                  selected_trading_metrics=equity_checks,
                  checks_scope='Full monthly event identities and 12-month counts; paper lag-1/sign metrics; SPY200 and QQQ225 close/next-close return and drawdown; not every sensitivity',
                  csv_sha256={'^GSPC': index_hash, 'SPY': spy_hash},
                  batch_sha256=hashlib.sha256(args.batch.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
