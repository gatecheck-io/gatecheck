"""Independent small-vector checks for the recovered receipts, not market verdict tests."""
import csv
import hashlib
import io
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

REPLAY = Path(__file__).resolve().parents[1] / 'verdicts' / '2026-07' / 'reproduction'
sys.path.insert(0, str(REPLAY))
from bench_engine import EdgeSpec, evaluate_positions, run_edge_test, _sma_ratio, _sharpe
from replay import drawdown, json_safe
from snapshot_source import SnapshotSource


def test_centered_sma_threshold_changes_the_claim():
    prices = np.array([100., 102., 104., 102.])
    expected = np.array([np.nan, 102 / 101 - 1, 104 / 103 - 1, 102 / 103 - 1])
    np.testing.assert_allclose(_sma_ratio(prices, 2), expected)
    legacy = EdgeSpec('legacy', 'sma_ratio', 2, 'long_above', 1)
    fixed = EdgeSpec('fixed', 'sma_ratio', 2, 'long_above', 0)
    np.testing.assert_array_equal(evaluate_positions(legacy, prices), [0, 0, 0, 0])
    np.testing.assert_array_equal(evaluate_positions(fixed, prices), [0, 1, 1, 0])


def test_next_bar_execution_and_two_units_of_turnover():
    # External signal jumps before a loss. Trading the contemporaneous gain
    # would be lookahead; the correct position pays the following loss.
    prices = np.array([100., 110., 99., 99., 100., 102., 101., 103., 102.])
    signals = np.array([0., 2., 0., 2., 0., 2., 0., 2., 0.])
    spec = EdgeSpec('alignment', 'raw', 1, 'long_short_above', 1)
    held = np.array([-1., 1., -1., 1., -1., 1., -1., 1.])
    returns = prices[1:] / prices[:-1] - 1
    expected = held * returns - .001 * np.array([1, 2, 2, 2, 2, 2, 2, 2])
    result = run_edge_test(spec, prices, signal_series=signals, cost_per_turnover=.001,
                           n_splits=2, embargo=0, n_perm=10, n_random_seeds=0)
    assert result.net_sharpe == pytest.approx(_sharpe(expected, 252))


def test_retrigger_extends_the_archived_hold():
    spec = EdgeSpec('reset', 'raw', 1, 'long_above', 1, horizon=3)
    # Triggers at bars 0 and 2 produce one five-bar hold, rather than two trades.
    np.testing.assert_array_equal(evaluate_positions(spec, np.array([2, 0, 2, 0, 0, 0.])),
                                  [1, 1, 1, 1, 1, 0])


def write_snapshot(directory, days, closes):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['date', 'timestamp', 'close', 'adjclose'])
    for day, close in zip(days, closes):
        ts = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())
        writer.writerow([day, ts, close, close])
    body = buffer.getvalue().encode()
    (directory / 'SPY.csv').write_bytes(body)
    (directory / 'SPY.metadata.json').write_text(json.dumps(dict(
        asset='SPY', csv_sha256=hashlib.sha256(body).hexdigest(), rows=len(days),
        first_date=days[0], last_date=days[-1], requested_start=days[0])))


@pytest.mark.parametrize('days,closes', [
    (['2026-07-01', '2026-07-01'], [100, 101]),
    (['2026-07-02', '2026-07-01'], [100, 101]),
    (['2026-07-01', '2026-07-02'], [100, 'nan']),
    (['2026-07-01', '2026-07-02'], [100, 0]),
])
def test_snapshot_rejects_invalid_bars(tmp_path, days, closes):
    write_snapshot(tmp_path, days, closes)
    with pytest.raises(ValueError):
        SnapshotSource(tmp_path).load('SPY')


def test_snapshot_hash_and_explicit_coverage(tmp_path):
    write_snapshot(tmp_path, ['2026-07-01', '2026-07-02'], [100, 101])
    reader = SnapshotSource(tmp_path)
    days, prices, _ = reader.series('SPY', date(2026, 7, 1), date(2026, 7, 2))
    assert days == ['2026-07-01', '2026-07-02']
    np.testing.assert_array_equal(prices, [100, 101])
    with pytest.raises(ValueError, match='coverage'):
        reader.series('SPY', date(2026, 6, 30), date(2026, 7, 2))
    with pytest.raises(ValueError, match='coverage'):
        reader.series('SPY', date(2026, 7, 1), date(2026, 7, 3))
    with (tmp_path / 'SPY.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='hash'):
        SnapshotSource(tmp_path).load('SPY')


def test_drawdown_includes_the_initial_capital():
    assert drawdown([-.2, .1, .25, -.1]) == pytest.approx(-.2)


def test_undefined_metrics_remain_explicit_in_saved_results():
    value = json_safe(dict(passed=np.bool_(True), ic=np.nan, dsr=np.float64(.9)))
    assert json.loads(json.dumps(value, allow_nan=False)) == dict(passed=True, ic=None, dsr=.9)
