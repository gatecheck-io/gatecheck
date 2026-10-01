"""Strict offline daily-bar reader. The archived probes use a fixed July cutoff."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

CUTOFF = datetime(2026, 7, 2, 23, 59, 59, tzinfo=timezone.utc)


def now_epoch() -> float:
    return CUTOFF.timestamp()


@dataclass(frozen=True)
class Observation:
    key: str
    t_event: float
    value: float


class SnapshotSource:
    """Read explicit, hashed snapshots; never download or silently discard bars."""

    def __init__(self, directory: Path | str | None = None):
        selected = directory or os.environ.get('GATECHECK_SNAPSHOT_DIR')
        if not selected:
            raise ValueError('Supply a snapshot directory or GATECHECK_SNAPSHOT_DIR')
        self.directory = Path(selected)
        self._cache: dict[str, tuple[list[dict], dict]] = {}

    def load(self, asset: str) -> tuple[list[dict], dict]:
        if asset not in {'SPY', 'QQQ', 'NVDA', '^GSPC'}:
            raise ValueError(f'Unsupported asset: {asset}')
        if asset in self._cache:
            return self._cache[asset]
        body = (self.directory / f'{asset}.csv').read_bytes()
        meta = json.loads((self.directory / f'{asset}.metadata.json').read_text(encoding='utf-8'))
        if meta['asset'] != asset or hashlib.sha256(body).hexdigest() != meta['csv_sha256']:
            raise ValueError(f'{asset}: snapshot hash or asset mismatch')
        reader = csv.DictReader(body.decode('utf-8').splitlines())
        if reader.fieldnames != ['date', 'timestamp', 'close', 'adjclose']:
            raise ValueError(f'{asset}: unexpected CSV columns')
        rows = []
        for raw in reader:
            day, ts = date.fromisoformat(raw['date']), int(raw['timestamp'])
            if (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ts)).date() != day:
                raise ValueError(f'{asset}: timestamp/date mismatch at {day}')
            values = {}
            for field in ('close', 'adjclose'):
                value = float(raw[field]) if raw[field] else None
                if value is not None and (not math.isfinite(value) or value <= 0):
                    raise ValueError(f'{asset}: invalid {field} at {day}')
                values[field] = value
            if values['close'] is None:
                raise ValueError(f'{asset}: missing close at {day}')
            if rows and (day <= rows[-1]['date'] or ts <= rows[-1]['timestamp']):
                raise ValueError(f'{asset}: duplicate or unordered bars')
            rows.append(dict(date=day, timestamp=ts, **values))
        if not rows or len(rows) != meta['rows']:
            raise ValueError(f'{asset}: empty snapshot or row-count mismatch')
        if (rows[0]['date'].isoformat() != meta['first_date'] or
                rows[-1]['date'].isoformat() != meta['last_date']):
            raise ValueError(f'{asset}: metadata date mismatch')
        self._cache[asset] = rows, meta
        return rows, meta

    def series(self, asset: str, start: date, end: date, field: str = 'close'):
        if start >= end or field not in {'close', 'adjclose'}:
            raise ValueError('Invalid date interval or price field')
        rows, meta = self.load(asset)
        # Weekends are allowed at the lower boundary, but not an absent end bar.
        if start < date.fromisoformat(meta['requested_start']) or end > rows[-1]['date']:
            raise ValueError(f'{asset}: requested interval exceeds snapshot coverage')
        chosen = [r for r in rows if start <= r['date'] <= end]
        if len(chosen) < 2 or chosen[-1]['date'] != end:
            raise ValueError(f'{asset}: interval lacks required final bar')
        if any(r[field] is None for r in chosen):
            raise ValueError(f'{asset}: missing {field}; no silent filtering')
        return ([r['date'].isoformat() for r in chosen],
                np.array([r[field] for r in chosen], dtype=float), meta)

    def fetch(self, keys: list[str], start: float, end: float) -> list[Observation]:
        result = []
        for key in keys:
            asset, field = key.split('.')
            if field not in {'close', 'adjclose'}:
                raise ValueError(f'Unsupported field: {field}')
            rows, _ = self.load(asset)
            chosen = [r for r in rows if start <= r['timestamp'] <= end]
            if not chosen or any(r[field] is None for r in chosen):
                raise ValueError(f'{key}: empty interval or missing values')
            result.extend(Observation(key, r['timestamp'], r[field]) for r in chosen)
        return result
