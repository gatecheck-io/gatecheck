"""Replay the recorded July mappings or an explicitly labeled SMA mapping correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, fields, replace
from datetime import date
from pathlib import Path

import numpy as np

from bench_engine import evaluate_positions, run_edge_test
from posted_claims import PostedClaim
from snapshot_source import SnapshotSource

ROOT = Path(__file__).resolve().parent


def drawdown(pnl) -> float:
    equity = np.r_[1., np.cumprod(1 + np.asarray(pnl))]
    return float(np.min(equity / np.maximum.accumulate(equity) - 1))


def json_safe(value):
    """Undefined correlations become JSON null; retain finite computed values."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def replay(source: SnapshotSource, start: date, end: date, *, field='close', corrected=False, rsi_corrected=False):
    claims_body = (ROOT.parent / 'claims.json').read_bytes()
    definitions = json.loads(claims_body)
    allowed = {f.name for f in fields(PostedClaim)}
    outputs, snapshots = [], {}
    for definition in definitions:
        claim = PostedClaim(**{k: v for k, v in definition.items() if k in allowed})
        original_threshold = claim.threshold
        if corrected and claim.indicator == 'sma_ratio':
            if claim.threshold != 1:
                raise ValueError('Unexpected archived SMA threshold')
            claim.threshold = 0.0
        days, prices, meta = source.series(claim.asset, start, end, field)
        snapshots[claim.asset] = {k: meta[k] for k in (
            'provider', 'acquired_utc', 'csv_sha256', 'first_date', 'last_date', 'rows', 'provenance')}
        spec = claim.to_spec()
        signal = prices
        if rsi_corrected and spec.indicator == 'rsi':
            from audited_methods import wilder_rsi
            signal = wilder_rsi(prices, spec.window)
            spec = replace(spec, indicator='raw')
        v = run_edge_test(spec, prices, signal_series=signal, seed=0)
        pos = evaluate_positions(spec, signal)[:-1]
        ret = prices[1:] / prices[:-1] - 1
        pnl = pos * ret - 5e-4 * np.abs(np.diff(pos, prepend=0.))
        outputs.append(dict(claim_id=claim.claim_id, asset=claim.asset,
                            archived_threshold=original_threshold, tested_threshold=claim.threshold,
                            first_date=days[0], last_date=days[-1], price_bars=len(prices),
                            exposure=float(np.mean(pos != 0)),
                            turnover_units=float(np.abs(np.diff(pos, prepend=0.)).sum()),
                            strategy_max_drawdown=drawdown(pnl), buy_hold_max_drawdown=drawdown(ret),
                            **asdict(v)))
    return dict(mode=('technical_corrections_common_window' if rsi_corrected else
                      'sma_threshold_corrected' if corrected else 'legacy_mapping'),
                corrections=dict(sma_threshold=corrected, wilder_rsi_seed=rsi_corrected),
                status='New rerun; not an exact reproduction of all original July downloads',
                requested_start=start.isoformat(), requested_end_inclusive=end.isoformat(),
                price_field=field, claims_sha256=hashlib.sha256(claims_body).hexdigest(),
                environment=dict(python=platform.python_version(), numpy=np.__version__),
                settings=dict(cost_per_turnover=5e-4, cash_return=0, periods_per_year=252,
                              n_splits=5, embargo=5, seed=0, n_perm=2000, random_sign_seeds=25,
                              gate='beats buy-hold AND DSR >= .95 AND positive fold-Sharpe CI'),
                snapshots=snapshots, gate_passes=sum(v['passed'] for v in outputs), results=outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--start', type=date.fromisoformat, default=date(2015, 6, 19))
    parser.add_argument('--end', type=date.fromisoformat, default=date(2026, 7, 2))
    parser.add_argument('--price-field', choices=['close', 'adjclose'], default='close')
    parser.add_argument('--sma-threshold-correction', action='store_true')
    parser.add_argument('--rsi-seed-correction', action='store_true',
                        help='Correct RSI seed only; fixed-horizon mappings remain explicitly proxies')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new name to preserve prior evidence')
    result = replay(SnapshotSource(args.snapshot_dir), args.start, args.end,
                    field=args.price_field, corrected=args.sma_threshold_correction,
                    rsi_corrected=args.rsi_seed_correction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"{result['mode']}: {result['gate_passes']}/14 pass on {result['price_field']}")
    for row in result['results']:
        print(f"{row['claim_id']:33} net={row['net_sharpe']:+.3f} bh={row['buy_hold_sharpe']:+.3f} "
              f"DSR={row['dsr']:.3f} pass={row['passed']}")


if __name__ == '__main__':
    main()
