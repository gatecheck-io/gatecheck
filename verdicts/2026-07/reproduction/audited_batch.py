"""Run all 14 audited entries; emit scoped diagnostics, not invented source verdicts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from audited_methods import (calendar_study, daily_forward, paper_sign_diagnostic,
                             rsi_exit_positions, select_ranked_nonoverlap, trading_summary)
from bench_engine import EdgeSpec, _sma, evaluate_positions
from replay import json_safe
from snapshot_source import SnapshotSource

ROOT = Path(__file__).resolve().parent


def arrays(source, asset, field='close'):
    rows, meta = source.load(asset)
    if any(r[field] is None for r in rows) or meta['last_date'] != '2026-07-02':
        raise ValueError('Missing price field or wrong audit cutoff')
    return [r['date'].isoformat() for r in rows], np.array([r[field] for r in rows]), meta


def run(equities, index):
    body = (ROOT / 'audit_registry.json').read_bytes()
    registry = json.loads(body)
    archived = {c['claim_id']: c for c in json.loads((ROOT.parent / 'claims.json').read_text())}
    outputs, input_meta = [], {}
    for entry in registry['claims']:
        cid = entry['claim_id']
        result = dict(**entry, source=registry['source_groups'][entry['source_group']])
        asset = archived[cid]['asset']
        diagnostics = []
        if cid.startswith('rd_') or cid in {'pa_spy_sma20_fast_timing', 'tw_qqq_ma225_cross',
                                           'tw_spy_rsi2_below10', 'yt_qs_rsi5_lt30_spy'}:
            for field in ('close', 'adjclose'):
                days, p, meta = arrays(equities, asset, field)
                input_meta[asset] = meta
                kw = archived[cid]
                if cid.startswith('rd_'):
                    spec = EdgeSpec(cid, kw['indicator'], kw['window'], kw['rule'],
                                    0 if kw['indicator'] == 'sma_ratio' else kw['threshold'])
                    target = evaluate_positions(spec, p)
                    start, end, cost, lags = '2005-01-01', '2026-07-02', .0003, [1]
                    variants = [(kw['indicator'], target)]
                elif cid == 'pa_spy_sma20_fast_timing':
                    target = evaluate_positions(EdgeSpec(cid, 'sma_ratio', 20, 'long_above', 0), p)
                    slow = np.where(_sma(p, 50) > _sma(p, 200), 1., 0.)
                    variants = [('SMA20', target), ('SMA50_above_SMA200', slow)]
                    start, end, cost, lags = '2006-01-01', '2025-12-31', .0005, [0, 1]
                elif cid == 'tw_qqq_ma225_cross':
                    target = evaluate_positions(EdgeSpec(cid, 'sma_ratio', 225, 'long_above', 0), p)
                    variants = [('SMA225', target)]
                    start, end, cost, lags = '2000-01-01', '2025-02-28', .0005, [0, 1]
                elif cid == 'tw_spy_rsi2_below10':
                    target = rsi_exit_positions(p, 2, 10, 80, crossing=True)
                    variants = [('RSI2_cross_below10_exit_above80', target)]
                    start, end, cost, lags = '1993-01-01', '2026-07-02', .0005, [0, 1]
                else:
                    target = rsi_exit_positions(p, 5, 30, 50, trend=True)
                    variants = [('RSI5_below30_exit50_one_day_SMA200_rise_no_confirmation', target)]
                    start, end, cost, lags = '1993-01-01', '2026-07-02', .0005, [0, 1]
                for name, target in variants:
                    for lag in lags:
                        diagnostics.append(dict(variant=name, price_field=field,
                                                requested_sample=[start, end],
                                                **trading_summary(days, p, target, start, end,
                                                                  cost=cost, execution_lag=lag)))
        elif cid == 'pa_spy_lag1_reversal_null_fri':
            days, p, meta = arrays(equities, 'SPY', 'adjclose')
            input_meta['SPY'] = meta
            mask = np.array([d <= '2026-06-19' for d in days])
            diagnostics.append(dict(price_field='adjclose', first_date=days[0], last_date=np.array(days)[mask][-1],
                                    requested_sample=['1993-01-01', '2026-06-19'],
                                    **paper_sign_diagnostic(p[mask])))
        else:
            days, p, meta = arrays(index, '^GSPC')
            input_meta['^GSPC'] = meta
            if cid == 'tw_fool_month_10pct':
                # Use only outcomes available when the source article was published.
                n = sum(d <= '2026-05-05' for d in days)
                diagnostics.append(dict(observation_cutoff='2026-05-05',
                                        **calendar_study(days[:n], p[:n], '1974-01-01', '2026-04-30')))
            elif cid == 'tw_fool_quarter_10pct':
                diagnostics.append(calendar_study(days, p, '2016-07-01', '2026-06-30', quarters=True))
            elif cid == 'yt_qe_6wk_rally_252d':
                eligible = [i for i, d in enumerate(days) if '1950-01-01' <= d <= '2026-05-15']
                selected = select_ranked_nonoverlap(p, 30, eligible)
                detail = daily_forward(days, p, selected)
                for row, i in zip(detail['events'], selected):
                    row['rally_start'] = days[i - 30]
                    row['rally_gain'] = float(p[i] / p[i - 30] - 1)
                diagnostics.append(dict(selection='retrospective top-20 greedy descending return, non-overlapping 30-return intervals',
                                        selection_tie_break='earlier date first', **detail))
            else:
                gains = np.r_[np.full(42, np.nan), p[42:] / p[:-42] - 1]
                triggers = [i for i, d in enumerate(days) if d >= '1945-01-01' and gains[i] >= .195]
                cluster = [i for k, i in enumerate(triggers) if k == 0 or i - triggers[k - 1] > 42]
                diagnostics.append(dict(selection='all daily >=19.5% rolling 42-bar gains',
                                        horizons={str(h): daily_forward(days, p, triggers, h) for h in [3, 6, 12]}))
                diagnostics.append(dict(selection='declared sensitivity: first trigger after a gap >42 trading bars',
                                        horizons={str(h): daily_forward(days, p, cluster, h) for h in [3, 6, 12]}))
        result['diagnostics'] = diagnostics
        outputs.append(result)
    if set(archived) != {r['claim_id'] for r in outputs}:
        raise ValueError('Audit must account for every original entry exactly once')
    return dict(reviewed_on='2026-10-01', registry_sha256=hashlib.sha256(body).hexdigest(),
                status='Needs revision: claim fidelity and exact original replication remain limited as specified per entry',
                original_entries_accounted_for=len(outputs),
                source_claim_failure_count=None,
                count_note='No aggregate rejected-claim count: descriptions, benchmark extensions, statistical nulls and incompletely specified strategies are different questions',
                environment=dict(python=platform.python_version(), numpy=np.__version__),
                snapshots={a: {k: m[k] for k in ['asset', 'first_date', 'last_date', 'rows', 'csv_sha256', 'acquired_utc', 'provenance']} for a, m in input_meta.items()},
                protocol_notes=['No source verdict inferred from a Sharpe failure',
                                'Trade signal prehistory is used for indicator warmup; initial book starts from the pre-sample signal',
                                'Next-close lag means yesterday close signal trades at today close and first earns the following return',
                                'Unknown fills and adjustments use predeclared sensitivities; results are not selected for attractiveness',
                                'Passive and strategy both pay the adopted initial entry cost; no forced final liquidation',
                                'Open holding blocks and unfinished forward windows are censored from completed-trade/event summaries',
                                'Event studies use index quote close; retrospective ranking is not an ex-ante strategy',
                                'No inferential significance claimed for overlapping calendar event outcomes'], results=outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equity-snapshot-dir', type=Path, required=True)
    parser.add_argument('--index-snapshot-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output filename')
    result = json_safe(run(SnapshotSource(args.equity_snapshot_dir), SnapshotSource(args.index_snapshot_dir)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for row in result['results']:
        print(f"{row['claim_id']}: {len(row['diagnostics'])} diagnostic(s); {row['source_verdict']}")


if __name__ == '__main__':
    main()
