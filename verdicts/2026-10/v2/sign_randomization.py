"""V2: sign-invariant monthly-return null, with drift uncertainty maximized.

A model-conditional randomization test, not a distribution-free test of every
zero-contrast process. No event-outcome boundary is removed from the test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from monthly_calibration import generate_world, exact_rate, rate_summary


def location_interval(returns,budget=.005):
    """Sign-count interval for a common continuous center of symmetry."""
    r=np.sort(np.asarray(returns,float))
    n=len(r)
    k=int(binom.ppf(budget/2,n,.5))
    while k>=0 and binom.cdf(k,n,.5)>budget/2:
        k-=1
    if k<0:
        return dict(low=None,high=None,lower_rank=None,coverage_lower_bound=1.)
    return dict(low=float(r[k]),high=float(r[n-k-1]),lower_rank=k+1,
                coverage_lower_bound=float(1-2*binom.cdf(k,n,.5)))


def tail_flags(k,w,b,observed,n):
    """Integer cross multiplication: w/k-b/n >= observed contrast."""
    ko,wo,bo=observed
    k,w,b=np.asarray(k,np.int64),np.asarray(w,np.int64),np.asarray(b,np.int64)
    return (k>0)&(n*ko*w >= k*(n*wo+ko*(b-bo)))


def supremum_tail(returns,signs,horizon,threshold,low,high,observed):
    """Exact empirical supremum across all drift knots, including isolated ties.

Counts are piecewise constant in drift. Inclusive trigger changes occur at a
knot; strict-positive outcome changes occur immediately after it. Both states
are included. This maximizes the tail count across the whole closed interval,
not across a selected grid or a single fitted drift.
"""
    r=np.asarray(returns,float)
    signs=np.asarray(signs,np.int8)
    draws,m=signs.shape
    n=m-horizon
    if len(r)!=m or not low<=high or np.any((signs!=1)&(signs!=-1)):
        raise ValueError('Invalid signs or drift interval')
    a=signs*r
    minus=signs==-1
    ce=np.where(minus[:,:n],(np.log1p(threshold)+r[:n])/2,
                np.where(r[:n]>=np.log1p(threshold),-np.inf,np.inf))
    forward_a=np.zeros((draws,n))
    forward_b=np.zeros((draws,n),dtype=np.int16)
    for offset in range(1,horizon+1):
        forward_a+=a[:,offset:offset+n]
        forward_b+=2*minus[:,offset:offset+n]
    cy=np.full((draws,n),np.inf)
    np.divide(-forward_a,forward_b,out=cy,where=forward_b>0)
    cy[(forward_b==0)&(forward_a>0)]=-np.inf
    cw=np.maximum(ce,cy)
    # E(mu) is inclusive; Y(mu) is strict. E*Y follows the later change.
    inclusive=np.concatenate([np.ones_like(ce,dtype=bool),np.zeros_like(cy,dtype=bool),ce>cy],axis=1)
    cuts=np.concatenate([ce,cy,cw],axis=1)
    phase=(~inclusive).astype(np.int8)
    initial=(cuts<low)|((cuts==low)&inclusive)
    k0=initial[:,:n].sum(axis=1,dtype=np.int32)
    b0=initial[:,n:2*n].sum(axis=1,dtype=np.int32)
    w0=initial[:,2*n:].sum(axis=1,dtype=np.int32)
    initial_tail=tail_flags(k0,w0,b0,observed,n)
    valid=(cuts>=low)&(cuts<=high)&~((cuts==low)&inclusive)&~((cuts==high)&~inclusive)
    ordered_cuts=np.where(valid,cuts,np.inf)
    order=np.lexsort((phase,ordered_cuts),axis=1)
    channels=np.broadcast_to(np.repeat(np.arange(3,dtype=np.int8),n),(draws,3*n))
    sorted_channel=np.take_along_axis(channels,order,axis=1)
    sorted_valid=np.take_along_axis(valid,order,axis=1)
    k=k0[:,None]+np.cumsum(sorted_valid&(sorted_channel==0),axis=1,dtype=np.int32)
    b=b0[:,None]+np.cumsum(sorted_valid&(sorted_channel==1),axis=1,dtype=np.int32)
    w=w0[:,None]+np.cumsum(sorted_valid&(sorted_channel==2),axis=1,dtype=np.int32)
    tails=tail_flags(k,w,b,observed,n).astype(np.int8)
    delta=np.diff(tails,axis=1,prepend=initial_tail[:,None].astype(np.int8))
    row,col=np.nonzero(delta)
    original_col=order[row,col]
    change_cut=cuts[row,original_col]
    change_phase=phase[row,original_col]
    changes=delta[row,col]
    initial_count=int(initial_tail.sum())
    maximum=initial_count
    worst=low
    worst_phase='at_lower_endpoint'
    if len(changes):
        ordering=np.lexsort((change_phase,change_cut))
        x,p,d=change_cut[ordering],change_phase[ordering],changes[ordering]
        counts=initial_count+np.cumsum(d,dtype=np.int32)
        # Never evaluate an artificial intermediate state within a tied group.
        group_end=np.r_[(x[1:]!=x[:-1])|(p[1:]!=p[:-1]),True]
        ends=np.flatnonzero(group_end)
        best=ends[np.argmax(counts[ends])]
        if counts[best]>maximum:
            maximum=int(counts[best])
            worst=float(x[best])
            worst_phase='at_knot' if p[best]==0 else 'right_of_knot'
    if not 0<=maximum<=draws:
        raise AssertionError('Tail count left its valid range')
    return dict(tail_count=maximum,draws=draws,worst_drift=worst,worst_phase=worst_phase,
                randomization_p_supremum=(1+maximum)/(1+draws))


def randomization_test(returns,plan,seed):
    r=np.asarray(returns,float)
    summary=rate_summary(r,plan['forward_months'],plan['trigger_simple_return'])
    ci=location_interval(r,plan['nuisance_error_budget'])
    result=dict(**summary,available=False,p_value=None,rejected=False,
                unavailable_reason=None,location_ci_low=ci['low'],location_ci_high=ci['high'],
                location_coverage_lower_bound=ci['coverage_lower_bound'],
                randomization_p_supremum=None,tail_count=None,worst_drift=None,worst_phase=None)
    k,w,b=summary['completed_events'],summary['event_wins'],summary['baseline_wins']
    if not k:
        result['unavailable_reason']='no_completed_events'
        return result
    if ci['low'] is None:
        result['unavailable_reason']='unbounded_location_interval'
        return result
    if summary['complete_rows']*w<=k*b:
        result.update(available=True,p_value=1.)
        return result
    rng=np.random.default_rng(seed)
    signs=rng.integers(0,2,(plan['randomization_draws'],len(r)),dtype=np.int8)*2-1
    tail=supremum_tail(r,signs,plan['forward_months'],plan['trigger_simple_return'],
                        ci['low'],ci['high'],(k,w,b))
    p=min(1.,tail['randomization_p_supremum']+plan['nuisance_error_budget'])
    tail.pop('draws')
    result.update(tail,available=True,p_value=p,rejected=bool(p<=plan['alpha']))
    return result


def seed(plan,index,replicate,stream):
    return np.random.SeedSequence([plan['seed_namespace'],index,replicate,stream])


def run_world(index,world,plan):
    draws=[]
    for rep in range(plan['replicates_per_world']):
        r=generate_world(world,plan['monthly_observations'],seed(plan,index,rep,0),
                         burn_in=plan['burn_in_months'],horizon=plan['forward_months'],
                         threshold=plan['trigger_simple_return'])
        result=randomization_test(r,plan,seed(plan,index,rep,1))
        result['location_covers_true_center']=(result['location_ci_low']<=world['mu']<=result['location_ci_high']) if world['kind']=='null' else None
        draws.append(dict(world=world['id'],replicate=rep,**result))
        if (rep+1)%100==0:
            print(f"{world['id']}: {rep+1}/{plan['replicates_per_world']}",flush=True)
    available=sum(d['available'] for d in draws)
    fires=sum(d['rejected'] for d in draws)
    all_rate=exact_rate(fires,len(draws))
    available_rate=exact_rate(fires,available)
    null_pass=None
    if world['kind']=='null':
        null_pass=bool(available and all_rate['ci_high']<=plan['maximum_null_ci_upper']
                       and available_rate['ci_high']<=plan['maximum_null_ci_upper'])
    oracle=None
    if world['kind']!='null':
        r=generate_world(world,plan['oracle_months'],seed(plan,index,0,2),
                         burn_in=plan['burn_in_months'],horizon=plan['forward_months'],
                         threshold=plan['trigger_simple_return'])
        oracle=rate_summary(r,plan['forward_months'],plan['trigger_simple_return'])
    summary=dict(world=world,attempted=len(draws),available=available,
                  unavailable_reasons=dict(Counter(d['unavailable_reason'] for d in draws if not d['available'])),
                  rejections_all_attempts=all_rate,rejections_available=available_rate,
                  null_gate_pass=null_pass,oracle_population_approximation=oracle,
                  location_coverage=exact_rate(sum(d['location_covers_true_center'] for d in draws),len(draws))
                    if world['kind']=='null' else None,
                  median_events=float(np.median([d['completed_events'] for d in draws])))
    return summary,draws


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_freeze():
    manifest=json.loads((ROOT/'frozen_manifest.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if digest(path)!=expected:
            raise ValueError(f'Frozen file changed: {name}')
        relative=path.resolve().relative_to(ROOT.parents[2]).as_posix()
        blob=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=expected:
            raise ValueError(f'Current Git commit does not pin frozen bytes: {name}')
    blob=subprocess.check_output(['git','show',f'{commit}:verdicts/2026-10/v2/frozen_manifest.json'],cwd=ROOT)
    if hashlib.sha256(blob).hexdigest()!=digest(ROOT/'frozen_manifest.json'):
        raise ValueError('Freeze manifest is not committed')
    return commit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.freeze:
        path=ROOT/'frozen_manifest.json'
        if path.exists(): parser.error('Existing freeze cannot be replaced')
        names=['METHOD.md','calibration_plan.json','sign_randomization.py','market_evaluation.py',
               'requirements.txt','../monthly_calibration.py']
        path.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
                            files={name:digest(ROOT/name) for name in names}),indent=2)+'\n',encoding='utf-8')
        print('Freeze saved; commit before calibration.')
        return
    if args.output is None or args.output.exists(): parser.error('Use a new output directory')
    commit=validate_freeze()
    plan=json.loads((ROOT/'calibration_plan.json').read_text())
    args.output.mkdir(parents=True)
    summaries,draws={},[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs={executor.submit(run_world,i,w,plan):i for i,w in enumerate(plan['worlds'])}
        for future in as_completed(jobs):
            summary,rows=future.result()
            summaries[jobs[future]]=summary
            draws.extend(rows)
            (args.output/f"{summary['world']['id']}.json").write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
            print(f"COMPLETED {summary['world']['id']}",flush=True)
    worlds=[summaries[i] for i in range(len(plan['worlds']))]
    passed=all(w['null_gate_pass'] for w in worlds if w['world']['kind']=='null')
    power=[w['world']['id'] for w in worlds if w['world']['kind']=='alternative'
           and w['rejections_all_attempts']['ci_low']>=.8]
    draws.sort(key=lambda d:(d['world'],d['replicate']))
    csv_path=args.output/'draws.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(draws[0]))
        writer.writeheader(); writer.writerows(draws)
    result=dict(method_id=plan['method_id'],frozen_commit=commit,
                 completed_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
                 numpy=np.__version__,scipy=scipy.__version__,draws_sha256=digest(csv_path),
                 frozen_manifest_sha256=digest(ROOT/'frozen_manifest.json'),
                 verdict='PASS_ALPHA_GATE' if passed else 'FAIL_ALPHA_GATE',
                 model_conditional_market_evaluation_permitted=passed,
                 demonstrated_80pct_power_worlds=power,worlds=worlds)
    (args.output/'calibration.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f"{result['verdict']}; demonstrated >=80% power: {power}",flush=True)


if __name__=='__main__': main()
