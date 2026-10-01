"""Small independent examples for the post-audit definitions and their boundaries."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'verdicts/2026-07/reproduction'))
from audited_methods import (calendar_study, lag_positions, paper_sign_diagnostic,
                             rsi_exit_positions, select_ranked_nonoverlap,
                             trading_summary, wilder_rsi)
from snapshot_source import SnapshotSource


def test_wilder_seed_then_only_new_changes():
    # First gain/loss means are 5 and 2.5. The next means are 6 and 1.25.
    result = wilder_rsi([100, 110, 105, 112, 110], 2)
    np.testing.assert_allclose(result, [np.nan, np.nan, 100 * 5 / 7.5,
                                        100 * 6 / 7.25, 100 * 3 / 4.625])


def test_rsi_exit_holds_until_threshold_instead_of_resetting_a_timer():
    # RSI at the last six bars: 50, 25, 12.5, 6.25, 68.75, 81.25.
    target = rsi_exit_positions([100, 110, 100, 90, 80, 70, 90, 100], 2, 10, 80, crossing=True)
    np.testing.assert_array_equal(target, [0, 0, 0, 0, 0, 1, 1, 0])


def test_two_positive_quarters_is_different_from_positive_combined_return():
    days = ['2019-12-31', '2020-03-31', '2020-06-30', '2020-09-30', '2020-12-31']
    result = calendar_study(days, [100, 120, 130, 125, 135], '2020-03-31', '2020-03-31', quarters=True)
    assert result['events'][0]['forward_returns']['2'] > 0
    assert result['events'][0]['next_two_quarters_both_positive'] is False


def test_calendar_months_exclude_partial_current_month_and_censor_forward_year():
    days = ['2020-12-31', '2021-01-29', '2021-02-26', '2021-03-31',
            '2021-04-30', '2021-05-28', '2021-06-30', '2021-07-02']
    result = calendar_study(days, [100, 112, 113, 115, 120, 121, 122, 150],
                            '2021-01-01', '2021-07-31')
    assert result['event_count'] == 1
    assert result['events'][0]['date'] == '2021-01-29'
    assert result['events'][0]['forward_returns']['3'] == pytest.approx(120 / 112 - 1)
    assert result['conditional']['12'] == dict(complete=0, censored=1, wins=0, win_rate=None, mean_return=None)


def test_ranked_nonoverlap_does_not_take_every_threshold_day():
    assert select_ranked_nonoverlap([100, 120, 150, 151, 180, 190], 2, range(2, 6), 2) == [2, 5]


def test_next_close_execution_has_one_additional_bar_delay():
    np.testing.assert_array_equal(lag_positions([0, 1, 1, 0], 1), [0, 0, 1])


def test_both_books_pay_the_same_initial_entry_cost():
    result = trading_summary(['2021-01-01', '2021-01-04', '2021-01-05', '2021-01-06'],
                             [100, 100, 100, 100], [1, 1, 1, 1],
                             '2021-01-01', '2021-01-06', cost=.001, execution_lag=0)
    assert result['strategy']['total_return'] == pytest.approx(-.001)
    assert result['passive']['total_return'] == pytest.approx(-.001)
    assert result['right_censored_open_block'] is True
    assert result['completed_holding_blocks'] == 0


def test_source_sign_statistic_on_alternating_returns():
    result = paper_sign_diagnostic([100, 110, 99, 108.9, 98.01])
    assert result['continuation_frequency'] == 0
    assert result['sign_z'] == pytest.approx(-math.sqrt(3))
    assert result['sign_two_sided_normal_p'] == pytest.approx(math.erfc(math.sqrt(1.5)))


def test_pre_1970_index_dates_work_on_windows(tmp_path):
    body = b'date,timestamp,close,adjclose\n1969-12-30,-172800,100,100\n1969-12-31,-86400,101,101\n'
    (tmp_path / '^GSPC.csv').write_bytes(body)
    (tmp_path / '^GSPC.metadata.json').write_text(json.dumps(dict(
        asset='^GSPC', csv_sha256=hashlib.sha256(body).hexdigest(), rows=2,
        first_date='1969-12-30', last_date='1969-12-31', requested_start='1969-12-30')))
    rows, _ = SnapshotSource(tmp_path).load('^GSPC')
    assert len(rows) == 2
