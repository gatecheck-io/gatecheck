"""Independent snapshot predicates, overlap ledger, and scalar market p-value."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from scipy.stats import binom

from verify_results import scalar_supremum

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--market',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Use a new verification filename')
    plan=json.loads((ROOT/'calibration_plan.json').read_text())
    market=json.loads((args.market/'market.json').read_text())
    body=args.snapshot.read_bytes()
    assert hashlib.sha256(body).hexdigest()==plan['market_daily_csv_sha256']==market['daily_snapshot_sha256']
    rows=list(csv.DictReader(body.decode().splitlines()))
    points={}
    for row in rows:
        if row['date']<='2026-04-30': points[row['date'][:7]]=row
    keys=[f'{year:04d}-{month:02d}' for year in range(1974,2027) for month in range(1,13)
          if f'{year:04d}-{month:02d}'<='2026-04']
    closes=[float(points['1973-12']['close'])]+[float(points[key]['close']) for key in keys]
    logs=np.array([math.log(b/a) for a,b in zip(closes,closes[1:])])
    events=[];baseline=[];complete_events=[]
    expected_flags=[]
    for i,key in enumerate(keys):
        trigger=closes[i+1]/closes[i]-1>=.1
        complete=i+12<len(keys)
        forward=closes[i+13]/closes[i+1]-1 if complete else None
        expected_flags.append((key,trigger,complete,forward>0 if complete else None))
        if complete: baseline.append(forward)
        if trigger:
            events.append((key,forward))
            if complete: complete_events.append((i,key,forward))
    n,k,w,b=len(baseline),len(complete_events),sum(r[2]>0 for r in complete_events),sum(r>0 for r in baseline)
    assert (len(events),k,w,n,b)==(13,12,10,616,479)
    assert market['event_rate']==w/k and market['baseline_rate']==b/n
    assert math.isclose(market['contrast'],w/k-b/n,abs_tol=1e-12)
    assert math.isclose(market['mean_return_contrast'],statistics.mean(r[2] for r in complete_events)-statistics.mean(baseline),abs_tol=1e-12)
    with (args.market/'monthly_window_ledger.csv').open(newline='') as f: ledger=list(csv.DictReader(f))
    assert len(ledger)==len(expected_flags)
    for row,(month,trigger,complete,positive) in zip(ledger,expected_flags):
        assert row['month']==month and row['trigger']==str(trigger) and row['complete']==str(complete)
        assert row['positive_outcome']==('' if positive is None else str(positive))
    with (args.market/'events.csv').open(newline='') as f: saved_events=list(csv.DictReader(f))
    assert [(r['month'],None if r['forward_return']=='' else float(r['forward_return'])) for r in saved_events]==events
    rank=max(j for j in range(len(logs)//2) if 2*binom.cdf(j,len(logs),.5)<=plan['nuisance_error_budget'])
    ordered=sorted(logs)
    low,high=ordered[rank],ordered[len(logs)-rank-1]
    assert math.isclose(low,market['location_ci_low'],abs_tol=1e-14) and math.isclose(high,market['location_ci_high'],abs_tol=1e-14)
    signs=np.random.default_rng(np.random.SeedSequence(plan['market_randomization_stream'])).integers(
        0,2,(plan['randomization_draws'],len(logs)),dtype=np.int8)*2-1
    maximum=scalar_supremum(logs,signs,12,.1,low,high,(k,w,b))
    p=min(1.,(1+maximum)/(1+plan['randomization_draws'])+plan['nuisance_error_budget'])
    assert maximum==market['tail_count'] and p==market['p_value']
    overlap=[]
    for index,(i,left,_) in enumerate(complete_events):
        for j,right,_ in complete_events[index+1:]:
            common=max(0,min(i+12,j+12)-max(i+1,j+1)+1)
            if common: overlap.append(dict(first_trigger=left,second_trigger=right,shared_monthly_returns=common))
    verification=dict(verified_utc=datetime.now(timezone.utc).isoformat(),verdict='CHECKS_PASSED',
        scope='All raw calendar predicates, 628 derived ledger rows, 13 source event identities, 12 completed outcomes, '
              'matched baseline and mean-return contrast, center interval and full independent scalar drift supremum.',
        daily_snapshot_sha256=market['daily_snapshot_sha256'],
        independent_log_returns_float64le_sha256=hashlib.sha256(logs.astype('<f8').tobytes()).hexdigest(),
        market_json_sha256=hashlib.sha256((args.market/'market.json').read_bytes()).hexdigest(),
        events=13,completed_events=k,event_wins=w,baseline_complete=n,baseline_wins=b,
        event_mean_return=statistics.mean(r[2] for r in complete_events),
        baseline_mean_return=statistics.mean(baseline),p_value=p,tail_count=maximum,
        overlapping_event_pairs=overlap,
        caveat='The p-value is conditional on the stated symmetry model and includes drift uncertainty; '
               'verification does not establish those assumptions for actual market data or remove historical vendor-revision risk.')
    args.output.write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__': main()
