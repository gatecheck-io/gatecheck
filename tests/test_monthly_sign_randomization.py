"""Independent exhaustive finite-path checks of the V2 drift supremum."""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

ROOT=Path(__file__).resolve().parents[1]/'verdicts/2026-10/v2'
sys.path.insert(0,str(ROOT))
from sign_randomization import location_interval,randomization_test,supremum_tail,tail_flags


def rational_supremum(r,signs,h,threshold,low,high,observed):
    r=list(map(lambda x:F(float(x)),r))
    cutoff=F(float(np.log1p(threshold)))
    low,high=F(float(low)),F(float(high))
    n=len(r)-h
    knots={low,high}
    for s in signs:
        for t in range(n):
            if s[t]<0:
                knots.add((cutoff+r[t])/2)
            a=sum(F(int(s[j]))*r[j] for j in range(t+1,t+h+1))
            b=sum(1-int(s[j]) for j in range(t+1,t+h+1))
            if b:
                knots.add(-a/b)
    knots=sorted(x for x in knots if low<=x<=high)
    candidates=knots+[(a+b)/2 for a,b in zip(knots,knots[1:])]
    ko,wo,bo=observed
    obs=F(wo,ko)-F(bo,n)
    maximum=0
    for mu in candidates:
        count=0
        for s in signs:
            path=[F(int(a))*b+(1-int(a))*mu for a,b in zip(s,r)]
            e=[path[t]>=cutoff for t in range(n)]
            y=[sum(path[t+1:t+h+1])>0 for t in range(n)]
            k=sum(e)
            stat=F(sum(a and b for a,b in zip(e,y)),k)-F(sum(y),n) if k else F(0)
            count+=stat>=obs
        maximum=max(maximum,count)
    return maximum


@pytest.mark.parametrize('seed',[17,41,58])
def test_entire_drift_sweep_agrees_with_exhaustive_rational_paths(seed):
    rng=np.random.default_rng(seed)
    r=rng.normal(.01,.15,20)
    s=rng.integers(0,2,(7,20),dtype=np.int8)*2-1
    expected=rational_supremum(r,s,3,.1,-.05,.07,(3,2,9))
    actual=supremum_tail(r,s,3,.1,-.05,.07,(3,2,9))
    assert actual['tail_count']==expected


def test_inclusive_and_strict_tied_knots_are_both_evaluated():
    cutoff=np.log1p(.1)
    r=np.array([-cutoff,0,cutoff,-cutoff,0,.2,-.1,0,.1])
    s=np.array([[-1]*9,[1,-1,1,-1,1,-1,1,-1,1]],dtype=np.int8)
    expected=rational_supremum(r,s,1,.1,-.02,.03,(3,2,3))
    assert supremum_tail(r,s,1,.1,-.02,.03,(3,2,3))['tail_count']==expected


def test_sign_count_interval_covers_at_least_declared_probability():
    r=np.arange(628)/10000
    ci=location_interval(r,.005)
    j=ci['lower_rank']-1
    assert 2*binom.cdf(j,628,.5)<=.005
    assert 2*binom.cdf(j+1,628,.5)>.005
    assert ci['low']==r[j] and ci['high']==r[627-j]
    assert ci['coverage_lower_bound']>=.995
    assert location_interval([0,1],.005)['low'] is None


def test_integer_comparison_preserves_exact_rate_ties():
    np.testing.assert_array_equal(tail_flags([3,0,3],[2,0,1],[9,9,9],(3,2,9),17),[True,False,False])


def test_all_positive_event_samples_are_kept_and_empty_events_unavailable():
    cfg=json.loads((ROOT/'calibration_plan.json').read_text())
    cfg=dict(cfg,randomization_draws=5)
    r=np.full(628,.001)
    r[::30]=.12
    # A losing window outside the triggered forward windows lowers the baseline.
    r[25::30]=-.04
    result=randomization_test(r,cfg,19)
    assert result['event_wins']==result['completed_events']
    assert result['available'] and result['p_value'] is not None
    missing=randomization_test(np.zeros(628),cfg,19)
    assert not missing['available'] and missing['p_value'] is None


def test_nuisance_adjustment_is_applied_and_seeds_are_deterministic():
    cfg=json.loads((ROOT/'calibration_plan.json').read_text())
    cfg=dict(cfg,randomization_draws=19)
    r=np.random.default_rng(15).normal(.006,.05,628)
    a=randomization_test(r,cfg,93)
    b=randomization_test(r,cfg,93)
    assert a==b
    if a['randomization_p_supremum'] is not None:
        assert a['p_value']==min(1,a['randomization_p_supremum']+.005)
