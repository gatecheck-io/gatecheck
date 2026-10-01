"""Acquire explicitly dated daily bars; replay never fetches or refreshes data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


def download(asset: str, start: date, end: date, output: Path) -> dict:
    if any((output / f'{asset}{suffix}').exists() for suffix in ('.csv', '.yahoo.json', '.metadata.json')):
        raise FileExistsError(f'{asset}: existing snapshot files; use a new output folder')
    params = dict(period1=int(datetime.combine(start, datetime.min.time(), timezone.utc).timestamp()),
                  period2=int(datetime.combine(end + timedelta(days=1), datetime.min.time(), timezone.utc).timestamp()),
                  interval='1d', events='div,splits')
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(asset, safe="")}?{urlencode(params)}'
    with urlopen(Request(url, headers={'User-Agent': 'gatecheck-reproduction/0.1'}), timeout=20) as response:
        raw = response.read()
    payload = json.loads(raw)
    if payload['chart'].get('error') or not payload['chart'].get('result'):
        raise ValueError(f'{asset}: vendor returned no result')
    result = payload['chart']['result'][0]
    timestamps = result['timestamp']
    close = result['indicators']['quote'][0]['close']
    adj = result['indicators'].get('adjclose', [{}])[0].get('adjclose', [None] * len(timestamps))
    if len(close) != len(timestamps) or len(adj) != len(timestamps):
        raise ValueError(f'{asset}: misaligned vendor arrays')
    rows = []
    for ts, c, a in zip(timestamps, close, adj):
        day = (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ts)).date()
        if not start <= day <= end:
            continue
        if c is None or not math.isfinite(c) or c <= 0:
            raise ValueError(f'{asset}: missing/nonpositive close at {day}; not silently dropped')
        if a is not None and (not math.isfinite(a) or a <= 0):
            raise ValueError(f'{asset}: invalid adjusted close at {day}')
        rows.append([day.isoformat(), ts, repr(c), '' if a is None else repr(a)])
    days = [r[0] for r in rows]
    if not rows or days != sorted(set(days)):
        raise ValueError(f'{asset}: empty, duplicate or unordered dates')
    if days[-1] != end.isoformat():
        raise ValueError(f'{asset}: last available date {days[-1]} does not reach cutoff {end}')
    output.mkdir(parents=True, exist_ok=True)
    csv_text = io.StringIO(newline='')
    writer = csv.writer(csv_text, lineterminator='\n')
    writer.writerow(['date', 'timestamp', 'close', 'adjclose'])
    writer.writerows(rows)
    body = csv_text.getvalue().encode('utf-8')
    csv_path = output / f'{asset}.csv'
    if csv_path.exists() and csv_path.read_bytes() != body:
        raise FileExistsError(f'{csv_path} differs: use a new output folder for a new snapshot')
    csv_path.write_bytes(body)
    raw_path = output / f'{asset}.yahoo.json'
    if not raw_path.exists():
        raw_path.write_bytes(raw)
    metadata = dict(asset=asset, provider='Yahoo Finance chart endpoint', requested_url=url,
                    acquired_utc=datetime.now(timezone.utc).isoformat(),
                    requested_start=start.isoformat(), requested_end_inclusive=end.isoformat(),
                    first_date=days[0], last_date=days[-1], rows=len(rows),
                    csv_sha256=hashlib.sha256(body).hexdigest(),
                    response_sha256=hashlib.sha256(raw).hexdigest(),
                    provenance='New acquisition of historical bars; not the original July download',
                    fields='Quote close and vendor adjusted close kept separately')
    (output / f'{asset}.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument('--end', type=date.fromisoformat, default=date(2026, 7, 2))
    parser.add_argument('--assets', nargs='+', choices=['SPY', 'QQQ', 'NVDA', '^GSPC'], default=['SPY', 'QQQ', 'NVDA'])
    args = parser.parse_args()
    if args.start >= args.end:
        parser.error('start must precede end')
    for asset in args.assets:
        result = download(asset, args.start, args.end, args.output)
        print(json.dumps({k: result[k] for k in ['asset', 'rows', 'first_date', 'last_date', 'csv_sha256']}))


if __name__ == '__main__':
    main()
