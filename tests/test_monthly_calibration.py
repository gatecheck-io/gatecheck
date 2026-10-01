"""Hand-computed and independent boundary checks of the October candidate."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

STUDY = Path(__file__).resolve().parents[1]/'verdicts/2026-10'
sys.path.insert(0,str(STUDY))
from monthly_calibration import (bootstrap_test, exact_rate, generate_world,
                                 outcome_rows, rate_summary, seeded, stationary_indices)


def plan():
    return json.loads((STUDY/'calibration_plan.json').read_text())


def test_strictly_forward_window_excludes_trigger_and_censors_tail():
    returns = np.array([.1, -.2, .2, .1, -.1])
    e,y,f = outcome_rows(np.log1p(returns),horizon=2,threshold=.099)
    np.testing.assert_array_equal(e,[True,False,True])
    np.testing.assert_array_equal(y,[False,True,False])
    np.testing.assert_allclose(np.expm1(f),[-.04,.32,-.01])
    s = rate_summary(np.log1p(returns),horizon=2,threshold=.099)
    assert s['completed_events']==2 and s['censored_events']==1
    assert s['contrast']==pytest.approx(-1/3)


def test_random_small_paths_agree_with_direct_price_ratios():
    logs = np.random.default_rng(14).normal(.006,.045,100)
    e,y,_ = outcome_rows(logs)
    price = 100*np.exp(np.r_[0,np.cumsum(logs)])
    events = [price[t+1]/price[t]-1 >= .1 for t in range(88)]
    wins = [price[t+13]/price[t+1]-1 > 0 for t in range(88)]
    np.testing.assert_array_equal(e,events)
    np.testing.assert_array_equal(y,wins)


def test_bootstrap_long_run_is_circular_and_single_month_blocks_are_uniform():
    index = stationary_indices(7,5,1e100,np.random.default_rng(9))
    np.testing.assert_array_equal((np.diff(index,axis=1))%7,np.ones((5,6)))
    index = stationary_indices(7,10000,1,np.random.default_rng(9))
    freq = np.bincount(index.ravel(),minlength=7)/index.size
    assert np.max(np.abs(freq-1/7)) < .005


def test_absent_and_boundary_events_are_unavailable_not_p_one():
    cfg = plan()
    absent = bootstrap_test(np.zeros(628),cfg,2)
    assert absent['p_value'] is None and not absent['available']
    assert absent['unavailable_reason']=='too_few_events'
    boundary = bootstrap_test(np.full(628,.11),cfg,2)
    assert boundary['p_value'] is None and not boundary['rejected']
    assert boundary['unavailable_reason']=='event_outcome_boundary'


def test_candidate_p_value_against_independent_scalar_bootstrap():
    # Frequent artificial triggers and varied outcomes: independent scalar recomputation.
    logs = np.random.default_rng(120).normal(0,.1,110)
    cfg = dict(plan(),bootstrap_draws=29,minimum_events=2,minimum_each_event_outcome=1)
    result = bootstrap_test(logs,cfg,55)
    assert result['available']
    idx = stationary_indices(98,29,24,np.random.default_rng(55))
    price = np.exp(np.r_[0,np.cumsum(logs)])
    e = [price[t+1]/price[t]-1 >= .1 for t in range(98)]
    y = [price[t+13]/price[t+1] > 1 for t in range(98)]
    observed = sum(a*b for a,b in zip(e,y))/sum(e)-sum(y)/len(y)
    boot = []
    for row in idx:
        events = [e[i] for i in row]
        outcomes = [y[i] for i in row]
        if sum(events):
            boot.append(sum(a*b for a,b in zip(events,outcomes))/sum(events)
                        -sum(outcomes)/len(outcomes))
    expected = (1+sum(t-observed >= observed for t in boot))/(1+len(boot))
    assert result['p_value']==expected


def test_exact_interval_endpoint_is_not_add_one_normal_approximation():
    r = exact_rate(0,1000)
    assert r['rate']==0 and r['ci_low']==0
    assert r['ci_high']==pytest.approx(1-.025**(1/1000))
    assert exact_rate(0,0)['ci_high'] is None


def test_world_seeds_are_schedule_independent_and_plant_is_forward_only():
    cfg = plan()
    world = {'model':'plant','mu':0.,'sigma':.045,'monthly_added_log_drift':.002}
    planted = generate_world(world,120,seeded(cfg,0,5,0),burn_in=0)
    unplanted = generate_world(dict(world,monthly_added_log_drift=0),120,
                              seeded(cfg,0,5,0),burn_in=0)
    first = next(i for i,x in enumerate(unplanted) if x >= math.log1p(.1))
    np.testing.assert_array_equal(planted[:first+1],unplanted[:first+1])
    assert planted[first+1]==pytest.approx(unplanted[first+1]+.002)
    np.testing.assert_array_equal(planted,generate_world(world,120,seeded(cfg,0,5,0),burn_in=0))


def test_invalid_series_is_not_silently_cleaned():
    with pytest.raises(ValueError):
        outcome_rows([0]*15+[float('nan')])
