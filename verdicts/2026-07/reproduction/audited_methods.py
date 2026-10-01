"""Explicit diagnostic definitions after the source audit; legacy engine stays intact."""
from __future__ import annotations

import calendar
import math
from datetime import date

import numpy as np

from bench_engine import _sma, _sharpe
from replay import drawdown


def wilder_rsi(prices, window):
    """Seed once from the first window changes, then smooth only new changes."""
    p = np.asarray(prices, dtype=float)
    if window < 2 or p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError('RSI needs positive finite prices and a window >= 2')
    out = np.full(p.shape, np.nan)
    if p.size <= window:
        return out
    changes = np.diff(p)
    gains, losses = np.maximum(changes, 0), np.maximum(-changes, 0)
    gain, loss = gains[:window].mean(), losses[:window].mean()
    for i in range(window, p.size):
        if i > window:
            gain = (gain * (window - 1) + gains[i - 1]) / window
            loss = (loss * (window - 1) + losses[i - 1]) / window
        total = gain + loss
        out[i] = 100 * gain / total if total > 0 else 0.
    return out


def rsi_exit_positions(prices, window, entry, exit_level, *, crossing=False, trend=False):
    p = np.asarray(prices, float)
    rsi = wilder_rsi(p, window)
    sma = _sma(p, 200) if trend else None
    out, held = np.zeros(p.size), False
    for i in range(1, p.size):
        if not np.isfinite(rsi[i]):
            continue
        if held and (rsi[i] > exit_level if crossing else rsi[i] >= exit_level):
            held = False
        elif not held:
            enter = rsi[i] < entry
            if crossing:
                enter = enter and np.isfinite(rsi[i - 1]) and rsi[i - 1] >= entry
            if trend:
                # One-day SMA rise is a declared variant: source does not specify slope length.
                enter = enter and np.isfinite(sma[i - 1]) and p[i] > sma[i] and sma[i] > sma[i - 1]
            held = bool(enter)
        out[i] = float(held)
    return out


def lag_positions(target, execution_lag):
    if execution_lag not in (0, 1):
        raise ValueError('Execution lag must be 0 or 1 closes')
    raw = np.asarray(target, float)[:-1]
    return raw.copy() if execution_lag == 0 else np.r_[0., raw[:-1]]


def trading_summary(days, prices, target, start, end, *, cost, execution_lag):
    p = np.asarray(prices, float)
    if len(days) != p.size or np.asarray(target).shape != p.shape:
        raise ValueError('Dates, prices and targets must align')
    all_ret = p[1:] / p[:-1] - 1
    mask = np.array([start <= day <= end for day in days[1:]])
    ret, held = all_ret[mask], lag_positions(target, execution_lag)[mask]
    if ret.size < 2 or cost < 0 or not np.all(np.isfinite(held)):
        raise ValueError('Invalid trading sample')
    chosen = np.flatnonzero(mask)
    net = held * ret - cost * np.abs(np.diff(held, prepend=0.))
    passive = ret.copy()
    passive[0] -= cost  # charge the same initial unit cost to both books
    years = (date.fromisoformat(days[chosen[-1] + 1]) - date.fromisoformat(days[chosen[0]])).days / 365.2425

    def metrics(pnl):
        wealth = float(np.prod(1 + pnl))
        return dict(total_return=wealth - 1, cagr=wealth ** (1 / years) - 1,
                    maximum_drawdown=drawdown(pnl), sharpe=_sharpe(pnl, 252))

    completed, beginning = [], None
    for i, position in enumerate(held):
        if position != 0 and beginning is None:
            beginning = i
        elif position == 0 and beginning is not None:
            completed.append(float(np.prod(1 + net[beginning:i + 1]) - 1))
            beginning = None
    return dict(price_anchor_date=days[chosen[0]], first_return_date=days[chosen[0] + 1],
                last_return_date=days[chosen[-1] + 1], return_bars=int(ret.size),
                cost_per_unit_turnover=cost, execution_lag_closes=execution_lag, cash_return=0,
                exposure=float(np.mean(held != 0)), strategy=metrics(net), passive=metrics(passive),
                completed_holding_blocks=len(completed), right_censored_open_block=beginning is not None,
                completed_block_win_rate=float(np.mean(np.array(completed) > 0)) if completed else None,
                average_completed_block_return=float(np.mean(completed)) if completed else None)


def add_months(day, months):
    total = day.year * 12 + day.month - 1 + months
    year, month0 = divmod(total, 12)
    month = month0 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def calendar_endpoints(days, prices, *, quarters=False):
    dates = [date.fromisoformat(d) for d in days]
    indices = {}
    for i, day in enumerate(dates):
        key = day.year * 12 + day.month - 1
        if quarters:
            key = day.year * 4 + (day.month - 1) // 3
        indices[key] = i
    result = []
    for key, i in sorted(indices.items()):
        months = 3 if quarters else 1
        year, unit0 = divmod(key, 4 if quarters else 12)
        month = (unit0 + 1) * months
        calendar_end = date(year, month, calendar.monthrange(year, month)[1])
        if calendar_end <= dates[-1]:
            result.append(dict(key=key, date=days[i], calendar_end=calendar_end.isoformat(),
                               price=float(prices[i])))
    return result


def summarize_returns(values):
    completed = [v for v in values if v is not None]
    return dict(complete=len(completed), censored=len(values) - len(completed),
                wins=sum(v > 0 for v in completed),
                win_rate=float(np.mean(np.array(completed) > 0)) if completed else None,
                mean_return=float(np.mean(completed)) if completed else None)


def calendar_study(days, prices, first, last, *, quarters=False, threshold=.1):
    points = calendar_endpoints(days, prices, quarters=quarters)
    by_key = {p['key']: p for p in points}
    horizons = [1, 2] if quarters else [3, 6, 12]
    events, baseline = [], []
    for point in points:
        previous = by_key.get(point['key'] - 1)
        if previous is None or not first <= point['calendar_end'] <= last:
            continue
        forward = {str(h): (by_key[point['key'] + h]['price'] / point['price'] - 1
                            if point['key'] + h in by_key else None) for h in horizons}
        row = dict(date=point['date'], calendar_end=point['calendar_end'],
                   gain=point['price'] / previous['price'] - 1, forward_returns=forward)
        if quarters:
            q1, q2 = by_key.get(point['key'] + 1), by_key.get(point['key'] + 2)
            row['next_two_quarters_both_positive'] = (
                q1['price'] > point['price'] and q2['price'] > q1['price']) if q1 and q2 else None
        baseline.append(row)
        if row['gain'] >= threshold:
            events.append(row)
    result = dict(definition='calendar_quarter' if quarters else 'calendar_month',
                  event_window=[first, last], threshold=threshold, events=events,
                  event_count=len(events), baseline_periods=len(baseline),
                  conditional={str(h): summarize_returns([r['forward_returns'][str(h)] for r in events]) for h in horizons},
                  baseline={str(h): summarize_returns([r['forward_returns'][str(h)] for r in baseline]) for h in horizons})
    if quarters:
        for name, rows in [('conditional', events), ('baseline', baseline)]:
            vals = [r['next_two_quarters_both_positive'] for r in rows]
            observed = [v for v in vals if v is not None]
            result[name]['next_two_quarters_both_positive'] = dict(
                complete=len(observed), censored=len(vals) - len(observed),
                wins=sum(observed), win_rate=float(np.mean(observed)) if observed else None)
    return result


def daily_forward(days, prices, indices, months=12):
    dates = [date.fromisoformat(day) for day in days]
    rows = []
    for i in indices:
        target = add_months(dates[i], months)
        j = int(np.searchsorted(dates, target, side='right') - 1)
        complete = target <= dates[-1]
        rows.append(dict(date=days[i], target_calendar_date=target.isoformat(),
                         forward_end_date=days[j] if complete else None,
                         forward_return=float(prices[j] / prices[i] - 1) if complete else None))
    return dict(events=rows, summary=summarize_returns([r['forward_return'] for r in rows]))


def select_ranked_nonoverlap(prices, window, eligible, count=20):
    """Retrospective ranking sensitivity, not an ex-ante trade selection rule."""
    p = np.asarray(prices, float)
    candidates = [i for i in eligible if i >= window]
    candidates.sort(key=lambda i: (-(p[i] / p[i - window] - 1), i))
    selected = []
    for i in candidates:
        if all(not (i - window < j and i > j - window) for j in selected):
            selected.append(i)
        if len(selected) == count:
            break
    return sorted(selected)


def paper_sign_diagnostic(prices):
    # Source definition: log returns; sign=1 for r>0, including zeros in sign=0.
    r = np.diff(np.log(np.asarray(prices, float)))
    if r.size < 3:
        raise ValueError('Insufficient returns')
    centered = r - r.mean()
    rho = float(np.dot(centered[1:], centered[:-1]) / np.dot(centered, centered))
    signs = r > 0
    continuation = float(np.mean(signs[1:] == signs[:-1]))
    z = (2 * continuation - 1) * math.sqrt(r.size - 1)
    return dict(log_return_observations=int(r.size), lag1_autocorrelation=rho,
                lag1_acf_z=rho * math.sqrt(r.size), sign_pairs=int(r.size - 1),
                continuation_frequency=continuation, sign_z=z,
                sign_two_sided_normal_p=math.erfc(abs(z) / math.sqrt(2)),
                zero_returns=int(np.sum(r == 0)),
                null='continuation frequency .5 under the source coin-flip assumption',
                interpretation='Descriptive partial replication; failure to reject does not prove no edge or calibrate a test')
