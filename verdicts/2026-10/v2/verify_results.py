"""Independent count, interval, and scalar event-sweep checks of V2 receipts.

The frozen generator supplies synthetic inputs. This verifier does not call the
candidate's event, interval, or supremum helpers in its independent checks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta,binom

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from monthly_calibration import generate_world


def direct_rows(ret,h=12,threshold=.1):
    gross=np.exp(ret)
    n=len(ret)-h
    forward=np.ones(n)
    for offset in range(1,h+1): forward*=gross[offset:offset+n]
    return gross[:n]-1>=threshold,forward>1,int(np.sum(gross-1>=threshold))


def interval(k,n):
    if not n: return None,None
    return float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.


def scalar_supremum(ret,signs,h,threshold,low,high,observed):
    """Direct E/Y transitions, updating wins from actual membership each time."""
    draws,m=signs.shape
    n=m-h
    trigger_cut=np.empty((draws,n))
    outcome_cut=np.empty((draws,n))
    for i,s in enumerate(signs):
        trigger_cut[i]=np.where(s[:n]<0,(np.log1p(threshold)+ret[:n])/2,
                               np.where(ret[:n]>=np.log1p(threshold),-np.inf,np.inf))
        sums=np.convolve(s*ret,np.ones(h),mode='valid')[1:]
        coeff=2*np.convolve((s<0).astype(int),np.ones(h,dtype=int),mode='valid')[1:]
        outcome_cut[i]=np.where(coeff>0,np.divide(-sums,coeff,out=np.zeros(n),where=coeff>0),
                               np.where(sums>0,-np.inf,np.inf))
    e=trigger_cut<=low
    y=outcome_cut<low
    k=e.sum(axis=1).astype(int)
    b=y.sum(axis=1).astype(int)
    w=(e&y).sum(axis=1).astype(int)
    ko,wo,bo=observed
    def flagged(i):
        return bool(k[i]>0 and n*ko*w[i]>=k[i]*(n*wo+ko*(b[i]-bo)))
    flags=np.array([flagged(i) for i in range(draws)])
    count=int(sum(flags));maximum=count
    er,ec=np.nonzero((trigger_cut>low)&(trigger_cut<=high))
    yr,yc=np.nonzero((outcome_cut>=low)&(outcome_cut<high))
    cuts=np.r_[trigger_cut[er,ec],outcome_cut[yr,yc]]
    phases=np.r_[np.zeros(len(er),int),np.ones(len(yr),int)]
    rows=np.r_[er,yr];cols=np.r_[ec,yc]
    order=np.lexsort((phases,cuts))
    for position,j in enumerate(order):
        i,t=int(rows[j]),int(cols[j])
        previous=bool(flags[i])
        if phases[j]==0:
            e[i,t]=True;k[i]+=1;w[i]+=int(y[i,t])
        else:
            y[i,t]=True;b[i]+=1;w[i]+=int(e[i,t])
        current=flagged(i);flags[i]=current
        count+=int(current)-int(previous)
        if position==len(order)-1 or cuts[j]!=cuts[order[position+1]] or phases[j]!=phases[order[position+1]]:
            maximum=max(maximum,count)
    return maximum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Use a new verification output')
    plan=json.loads((ROOT/'calibration_plan.json').read_text())
    result=json.loads((args.evidence/'calibration.json').read_text())
    body=(args.evidence/'draws.csv').read_bytes()
    assert hashlib.sha256(body).hexdigest()==result['draws_sha256']
    draws=list(csv.DictReader(body.decode().splitlines()))
    assert len(draws)==plan['replicates_per_world']*len(plan['worlds'])
    rank=max(j for j in range(plan['monthly_observations']//2)
             if 2*binom.cdf(j,plan['monthly_observations'],.5)<=plan['nuisance_error_budget'])
    summaries,spots=[],[]
    for i,world in enumerate(plan['worlds']):
        subset=[d for d in draws if d['world']==world['id']]
        assert sorted(int(d['replicate']) for d in subset)==list(range(plan['replicates_per_world']))
        spot_done=False
        coverage=0
        for row in subset:
            rep=int(row['replicate'])
            ret=generate_world(world,plan['monthly_observations'],
                               np.random.SeedSequence([plan['seed_namespace'],i,rep,0]),
                               burn_in=plan['burn_in_months'],horizon=plan['forward_months'],
                               threshold=plan['trigger_simple_return'])
            e,y,total=direct_rows(ret,plan['forward_months'],plan['trigger_simple_return'])
            n,k,w,b=len(y),int(sum(e)),int(sum(e&y)),int(sum(y))
            assert (n,k,w,b,total)==tuple(int(row[f]) for f in
                ['complete_rows','completed_events','event_wins','baseline_wins','total_events'])
            assert total-k==int(row['censored_events'])
            if k: assert math.isclose(w/k-b/n,float(row['contrast']),abs_tol=1e-12)
            sorted_ret=sorted(ret)
            low,high=sorted_ret[rank],sorted_ret[len(ret)-rank-1]
            assert low==float(row['location_ci_low']) and high==float(row['location_ci_high'])
            if world['kind']=='null':
                covers=low<=world['mu']<=high
                assert str(covers)==row['location_covers_true_center']
                coverage+=covers
            if k:
                assert row['available']=='True' and row['unavailable_reason']==''
                p=float(row['p_value'])
                assert row['rejected']==str(p<=plan['alpha'] and n*w>k*b)
                if n*w<=k*b:
                    assert p==1. and row['randomization_p_supremum']==''
                else:
                    assert p==min(1.,float(row['randomization_p_supremum'])+plan['nuisance_error_budget'])
                    assert float(row['randomization_p_supremum'])==(1+int(row['tail_count']))/(1+plan['randomization_draws'])
                    if not spot_done:
                        signs=np.random.default_rng(np.random.SeedSequence([plan['seed_namespace'],i,rep,1])).integers(
                            0,2,(plan['randomization_draws'],len(ret)),dtype=np.int8)*2-1
                        tail=scalar_supremum(ret,signs,plan['forward_months'],plan['trigger_simple_return'],low,high,(k,w,b))
                        assert tail==int(row['tail_count']),(world['id'],tail,row['tail_count'])
                        spots.append(dict(world=world['id'],replicate=rep,tail_count=tail,p_value=p))
                        spot_done=True
            else:
                assert row['available']=='False' and row['p_value']=='' and row['unavailable_reason']=='no_completed_events'
        available=sum(r['available']=='True' for r in subset)
        fires=sum(r['rejected']=='True' for r in subset)
        saved=result['worlds'][i]
        assert saved['world']==world and available==saved['available']
        assert dict(Counter(r['unavailable_reason'] for r in subset if r['available']=='False'))==saved['unavailable_reasons']
        for field,denom in [('rejections_all_attempts',len(subset)),('rejections_available',available)]:
            lo,hi=interval(fires,denom)
            assert saved[field]['count']==fires and saved[field]['denominator']==denom
            assert math.isclose(lo,saved[field]['ci_low'],abs_tol=1e-10) and math.isclose(hi,saved[field]['ci_high'],abs_tol=1e-10)
        passed=None
        if world['kind']=='null':
            passed=bool(available and interval(fires,len(subset))[1]<=plan['maximum_null_ci_upper']
                        and interval(fires,available)[1]<=plan['maximum_null_ci_upper'])
            assert passed==saved['null_gate_pass']
            assert coverage==saved['location_coverage']['count']
        oracle=saved['oracle_population_approximation']
        if oracle:
            ret=generate_world(world,plan['oracle_months'],
                np.random.SeedSequence([plan['seed_namespace'],i,0,2]),
                burn_in=plan['burn_in_months'],horizon=plan['forward_months'],threshold=plan['trigger_simple_return'])
            e,y,total=direct_rows(ret,plan['forward_months'],plan['trigger_simple_return'])
            assert sum(e)==oracle['completed_events'] and sum(e&y)==oracle['event_wins'] and sum(y)==oracle['baseline_wins']
            assert math.isclose(sum(e&y)/sum(e)-sum(y)/len(y),oracle['contrast'],abs_tol=1e-12)
        summaries.append(dict(world=world['id'],draws_checked=len(subset),available=available,
                              rejections=fires,null_gate_pass=passed))
        print(f"Verified {world['id']}: {len(subset)} draw definitions/counts and one full scalar supremum",flush=True)
    assert all(s['null_gate_pass'] for s in summaries if s['null_gate_pass'] is not None)==result['model_conditional_market_evaluation_permitted']
    verification=dict(verified_utc=datetime.now(timezone.utc).isoformat(),verdict='CHECKS_PASSED',
        frozen_commit=result['frozen_commit'],draws_sha256=result['draws_sha256'],
        calibration_sha256=hashlib.sha256((args.evidence/'calibration.json').read_bytes()).hexdigest(),
        scope='All 11000 draw definitions/counts, center intervals, decisions and world rejection intervals; '
              'all five long-world rate contrasts; one complete independent scalar drift supremum per world. '
              'Generators reused as selected inputs; not every randomization p-value independently recomputed.',
        worlds=summaries,scalar_supremum_spot_checks=spots)
    args.output.write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print('CHECKS_PASSED',flush=True)


if __name__=='__main__': main()
