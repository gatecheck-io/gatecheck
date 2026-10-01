"""Small benchmark helpers preserving the July accounting conventions."""
from __future__ import annotations
import numpy as np


def baseline_positions(name: str, n: int, *, seed: int = 0) -> np.ndarray:
    if n < 0:
        raise ValueError('n must be nonnegative')
    if name == 'buy_hold':
        return np.ones(n)
    if name == 'constant_short':
        return -np.ones(n)
    if name == 'random_sign':
        return np.random.default_rng(seed).choice([-1., 1.], size=n)
    raise ValueError(f'Unknown baseline {name}')


def baseline_pnl(positions, returns):
    pos, ret = np.asarray(positions, float), np.asarray(returns, float)
    if pos.shape != ret.shape:
        raise ValueError('Mismatched position/return shapes')
    return pos * ret


def net_timing_pnl(raw_positions, returns, *, cost_per_turnover: float, causal: bool = True):
    raw, ret = np.asarray(raw_positions, float), np.asarray(returns, float)
    if raw.shape != ret.shape or cost_per_turnover < 0:
        raise ValueError('Invalid position/return shapes or cost')
    if not causal:
        raise ValueError('This replay supports causal timing only')
    baseline = np.cumsum(raw) / np.arange(1, raw.size + 1)
    return (raw - baseline) * ret - cost_per_turnover * np.abs(np.diff(raw, prepend=0.))
