"""Evaluate the frozen market snapshot only after V2's null gate passes."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from sign_randomization import ROOT,digest,randomization_test,validate_freeze


def monthly_snapshot(path,plan):
    if digest(path)!=plan['market_daily_csv_sha256']:
        raise ValueError('Market snapshot differs from the frozen data hash')
    with Path(path).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if [r['date'] for r in rows]!=sorted(set(r['date'] for r in rows)):
        raise ValueError('Dates must be unique and ordered')
    monthly={}
    for row in rows:
        if row['date']<='2026-04-30':
            key=int(row['date'][:4])*12+int(row['date'][5:7])-1
            price=float(row['close'])
            if not np.isfinite(price) or price<=0: raise ValueError('Invalid quote close')
            monthly[key]=(row['date'],price)
    first,last=1974*12,2026*12+3
    keys=list(range(first,last+1))
    if not all(k in monthly for k in range(first-1,last+1)):
        raise ValueError('A calendar month is missing; do not bridge it')
    logs=np.array([np.log(monthly[k][1]/monthly[k-1][1]) for k in keys])
    ledger=[]
    for i,k in enumerate(keys):
        complete=k+12 in monthly
        forward=monthly[k+12][1]/monthly[k][1]-1 if complete else None
        ledger.append(dict(month=f'{k//12:04d}-{k%12+1:02d}',quote_date=monthly[k][0],
                           trigger=bool(monthly[k][1]/monthly[k-1][1]-1>=.1),
                           complete=complete,forward_end_month=f'{(k+12)//12:04d}-{(k+12)%12+1:02d}' if complete else None,
                           positive_outcome=bool(forward>0) if complete else None,
                           monthly_gain=float(np.expm1(logs[i])),forward_return=forward))
    return logs,ledger


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    commit=validate_freeze()
    plan=json.loads((ROOT/'calibration_plan.json').read_text())
    calibration=json.loads(args.calibration.read_text())
    if calibration['method_id']!=plan['method_id'] or calibration['verdict']!='PASS_ALPHA_GATE':
        parser.error('V2 calibration has not passed; do not evaluate a market p-value')
    if calibration['frozen_manifest_sha256']!=digest(ROOT/'frozen_manifest.json'):
        parser.error('Calibration was produced by a different frozen method')
    logs,ledger=monthly_snapshot(args.snapshot,plan)
    result=randomization_test(logs,plan,np.random.SeedSequence(plan['market_randomization_stream']))
    assert result['total_events']==13 and result['completed_events']==12
    assert result['event_wins']==10 and result['baseline_wins']==479 and result['complete_rows']==616
    args.output.mkdir(parents=True)
    result.update(evaluated_utc=datetime.now(timezone.utc).isoformat(),frozen_commit=commit,
                  method_id=plan['method_id'],daily_snapshot_sha256=digest(args.snapshot),
                  calibration_sha256=digest(args.calibration),
                  data_provenance='October 1 acquisition of historical quote closes, not the original July snapshot',
                  inference_scope='Constant drift and conditional sign invariance given all centered return magnitudes; '
                                  'not a distribution-free zero-contrast test, causal validation, or proof of absence.')
    (args.output/'market.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    events=[r for r in ledger if r['trigger']]
    with (args.output/'events.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(events[0]));writer.writeheader();writer.writerows(events)
    # Derived eligibility/outcome flags; raw vendor prices are not redistributed.
    fields=['month','quote_date','trigger','complete','forward_end_month','positive_outcome']
    with (args.output/'monthly_window_ledger.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in ledger)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
