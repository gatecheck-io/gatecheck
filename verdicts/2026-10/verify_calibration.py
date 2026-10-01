"""Independent arithmetic and selected scalar resampling checks of saved draws.

Uses the frozen generator as the selected input source; does not use its event,
outcome, bootstrap-index, p-value, or interval helpers in the recomputations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta

from monthly_calibration import generate_world

ROOT = Path(__file__).resolve().parent


def interval(k,n):
    if not n:
        return None,None
    return (float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.)


def direct_rows(ret,plan):
    # Multiply only twelve gross returns. A 250,000-month cumulative price
    # overflows; the frozen implementation avoids that with log-price differences.
    gross = np.exp(ret)
    h = plan['forward_months']
    n = len(ret)-h
    event = gross[:n]-1 >= plan['trigger_simple_return']
    product = np.ones(n)
    for offset in range(1,h+1):
        product *= gross[offset:offset+n]
    positive = product > 1
    total = int(np.sum(gross-1 >= plan['trigger_simple_return']))
    return event,positive,total


def scalar_p(e,y,plan,seed):
    rng = np.random.default_rng(seed)
    b,n = plan['bootstrap_draws'],len(e)
    restart = rng.random((b,n)) < 1/plan['mean_block_months']
    restart[:,0] = True
    starts = rng.integers(0,n,size=(b,n),dtype=np.int32)
    observed = sum(bool(a and c) for a,c in zip(e,y))/sum(e)-sum(y)/n
    boot = []
    for j in range(b):
        pos = int(starts[j,0])
        events,wins,baseline = 0,0,0
        for t in range(n):
            if restart[j,t]:
                pos = int(starts[j,t])
            elif t:
                pos = (pos+1)%n
            events += int(e[pos])
            wins += int(e[pos] and y[pos])
            baseline += int(y[pos])
        if events:
            boot.append(wins/events-baseline/n)
    return (1+sum(t-observed >= observed for t in boot))/(1+len(boot)),b-len(boot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new verification output')
    plan = json.loads((ROOT/'calibration_plan.json').read_text())
    result = json.loads((args.evidence/'calibration.json').read_text())
    body = (args.evidence/'draws.csv').read_bytes()
    assert hashlib.sha256(body).hexdigest()==result['draws_sha256']
    rows = list(csv.DictReader(body.decode().splitlines()))
    assert len(rows)==len(plan['worlds'])*plan['replicates_per_world']
    checks,selected = [],[]
    for index,world in enumerate(plan['worlds']):
        subset = [r for r in rows if r['world']==world['id']]
        assert sorted(int(r['replicate']) for r in subset)==list(range(plan['replicates_per_world']))
        spot_done = False
        for row in subset:
            rep = int(row['replicate'])
            seed = np.random.SeedSequence([plan['seed_namespace'],index,rep,0])
            ret = generate_world(world,plan['monthly_observations'],seed,
                                 burn_in=plan['burn_in_months'],horizon=plan['forward_months'],
                                 threshold=plan['trigger_simple_return'])
            e,y,total = direct_rows(ret,plan)
            k,w = int(sum(e)),int(sum(e&y))
            assert k==int(row['completed_events']) and w==int(row['event_wins'])
            assert total==int(row['total_events']) and total-k==int(row['censored_events'])
            assert sum(y)==int(row['baseline_wins']) and len(y)==int(row['complete_rows'])
            if k:
                contrast = w/k-sum(y)/len(y)
                assert math.isclose(contrast,float(row['contrast']),abs_tol=1e-12)
            availability = row['available']=='True'
            boundary = k<plan['minimum_events'] or min(w,k-w)<plan['minimum_each_event_outcome']
            if availability:
                assert not boundary and row['unavailable_reason']==''
                p = float(row['p_value'])
                assert row['rejected']==str(contrast>0 and p<=plan['alpha'])
                if not spot_done:
                    p_direct,empty = scalar_p(e,y,plan,
                        np.random.SeedSequence([plan['seed_namespace'],index,rep,1]))
                    assert p_direct==p and empty==int(row['empty_bootstrap_draws'])
                    selected.append(dict(world=world['id'],replicate=rep,p_value=p,
                                         empty_bootstrap_draws=empty))
                    spot_done = True
            else:
                assert row['p_value']=='' and row['rejected']=='False'
                assert boundary or row['unavailable_reason']=='too_many_empty_bootstrap_draws'
        available = sum(r['available']=='True' for r in subset)
        fires = sum(r['rejected']=='True' for r in subset)
        saved = result['worlds'][index]
        assert saved['world']==world and available==saved['available']
        reasons = Counter(r['unavailable_reason'] for r in subset if r['available']!='True')
        assert dict(reasons)==saved['unavailable_reasons']
        for field,n in [('rejections_all_attempts',len(subset)),('rejections_available',available)]:
            observed = saved[field]
            lo,hi = interval(fires,n)
            assert observed['count']==fires and observed['denominator']==n
            assert math.isclose(observed['ci_low'],lo,abs_tol=1e-10)
            assert math.isclose(observed['ci_high'],hi,abs_tol=1e-10)
        null_pass = None
        if world['kind']=='null':
            null_pass = bool(available and interval(fires,len(subset))[1]<=plan['maximum_null_ci_upper']
                             and interval(fires,available)[1]<=plan['maximum_null_ci_upper'])
            assert null_pass==saved['null_gate_pass']
        oracle = saved['oracle_population_approximation']
        if oracle:
            ret = generate_world(world,plan['oracle_months'],
                np.random.SeedSequence([plan['seed_namespace'],index,0,2]),
                burn_in=plan['burn_in_months'],horizon=plan['forward_months'],
                threshold=plan['trigger_simple_return'])
            e,y,total = direct_rows(ret,plan)
            assert int(sum(e))==oracle['completed_events'] and int(sum(e&y))==oracle['event_wins']
            assert int(sum(y))==oracle['baseline_wins']
            assert math.isclose(sum(e&y)/sum(e)-sum(y)/len(y),oracle['contrast'],abs_tol=1e-12)
        checks.append(dict(world=world['id'],draws_checked=len(subset),available=available,
                           rejections=fires,null_gate_pass=null_pass))
        print(f"Verified {world['id']}: {len(subset)} draw definitions/counts",flush=True)
    all_pass = all(c['null_gate_pass'] for c in checks if c['null_gate_pass'] is not None)
    assert result['market_inference_permitted']==all_pass
    verification = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        verdict='CHECKS_PASSED', frozen_commit=result['frozen_commit'],
        draws_sha256=result['draws_sha256'],
        calibration_sha256=hashlib.sha256((args.evidence/'calibration.json').read_bytes()).hexdigest(),
        scope='All 10000 draw definitions and count arithmetic; all world intervals and gate decisions; '
              'all four long-world rate contrasts; one independent scalar bootstrap per world. '
              'The frozen generator is reused as the input source; not every bootstrap p-value is independently recomputed.',
        worlds=checks, scalar_bootstrap_spot_checks=selected)
    args.output.write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print('CHECKS_PASSED',flush=True)


if __name__=='__main__':
    main()
