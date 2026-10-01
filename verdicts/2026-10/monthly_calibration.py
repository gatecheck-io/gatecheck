"""Frozen monthly rare-event candidate and seeded offline calibration.

This study module is separate from the library API and the July receipts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome_rows(log_returns, horizon=12, threshold=.1):
    """Regenerate triggers and strictly forward outcomes from a price path."""
    r = np.asarray(log_returns, dtype=float)
    if r.ndim != 1 or len(r) <= horizon or not np.all(np.isfinite(r)):
        raise ValueError('Need finite monthly log returns and a complete horizon')
    # Anchored log prices avoid under/overflow in long population approximations.
    log_price = np.r_[0., np.cumsum(r)]
    n = len(r) - horizon
    trigger = np.expm1(r[:n]) >= threshold
    forward_log = log_price[horizon + 1:] - log_price[1:n + 1]
    return trigger, forward_log > 0, forward_log


def rate_summary(log_returns, horizon=12, threshold=.1):
    e, y, forward = outcome_rows(log_returns, horizon, threshold)
    count, wins = int(e.sum()), int(y[e].sum())
    total_triggers = int(np.sum(np.expm1(log_returns) >= threshold))
    return dict(complete_rows=len(y), completed_events=count, event_wins=wins,
                total_events=total_triggers, censored_events=total_triggers-count,
                event_rate=wins/count if count else None,
                baseline_wins=int(y.sum()), baseline_rate=float(y.mean()),
                contrast=float(y[e].mean()-y.mean()) if count else None,
                mean_return_contrast=float(np.expm1(forward[e]).mean()-
                                           np.expm1(forward).mean()) if count else None)


def stationary_indices(n, draws, mean_block, rng):
    """Uniform restarts with geometrically distributed circular run lengths."""
    if n < 1 or draws < 1 or mean_block < 1:
        raise ValueError('Invalid stationary bootstrap dimensions')
    positions = np.arange(n, dtype=np.int32)
    restart = rng.random((draws, n)) < 1/mean_block
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(draws, n), dtype=np.int32)
    anchor = np.maximum.accumulate(np.where(restart, positions, 0), axis=1)
    start = np.take_along_axis(starts, anchor, axis=1)
    return (start + positions - anchor) % n


def bootstrap_test(log_returns, plan, seed):
    summary = rate_summary(log_returns, plan['forward_months'], plan['trigger_simple_return'])
    result = dict(**summary, available=False, p_value=None, rejected=False,
                  empty_bootstrap_draws=0, unavailable_reason=None)
    k, w = summary['completed_events'], summary['event_wins']
    if k < plan['minimum_events']:
        result['unavailable_reason'] = 'too_few_events'
        return result
    if min(w, k-w) < plan['minimum_each_event_outcome']:
        result['unavailable_reason'] = 'event_outcome_boundary'
        return result
    e, y, _ = outcome_rows(log_returns, plan['forward_months'], plan['trigger_simple_return'])
    index = stationary_indices(len(e), plan['bootstrap_draws'], plan['mean_block_months'],
                               np.random.default_rng(seed))
    es, ys = e[index], y[index]
    counts = es.sum(axis=1)
    valid = counts > 0
    empty = int((~valid).sum())
    result['empty_bootstrap_draws'] = empty
    if empty / plan['bootstrap_draws'] > plan['maximum_empty_bootstrap_fraction']:
        result['unavailable_reason'] = 'too_many_empty_bootstrap_draws'
        return result
    boot = (es[valid] & ys[valid]).sum(axis=1)/counts[valid] - ys[valid].mean(axis=1)
    observed = summary['contrast']
    p = float((1+np.sum(boot-observed >= observed))/(1+len(boot)))
    result.update(available=True, p_value=p,
                  rejected=bool(observed > 0 and p <= plan['alpha']))
    return result


def seeded(plan, world_index, replicate, stream):
    return np.random.SeedSequence([plan['seed_namespace'], world_index, replicate, stream])


def generate_world(world, n, seed, *, burn_in=2048, horizon=12, threshold=.1):
    rng = np.random.default_rng(seed)
    size = n + burn_in
    mu, sigma = world['mu'], world['sigma']
    model = world['model']
    if model == 'normal':
        ret = mu + sigma*rng.standard_normal(size)
    elif model == 'student':
        df = world['df']
        ret = mu + sigma*np.sqrt((df-2)/df)*rng.standard_t(df, size)
    elif model == 'log_vol':
        phi, sd = world['phi'], world['log_vol_sd']
        h = np.empty(size)
        h[0] = sd*rng.standard_normal()
        innovation = sd*np.sqrt(1-phi*phi)*rng.standard_normal(size)
        for t in range(1, size):
            h[t] = phi*h[t-1] + innovation[t]
        ret = mu + sigma*np.exp(h-sd*sd)*rng.standard_normal(size)
    elif model == 'plant':
        ret = mu + sigma*rng.standard_normal(size)
        remaining = 0
        trigger_log = np.log1p(threshold)
        for t in range(size):
            if remaining:
                ret[t] += world['monthly_added_log_drift']
                remaining -= 1
            if ret[t] >= trigger_log:
                remaining = horizon
    elif model == 'regime':
        phase = (np.arange(size)//world['regime_months']) % 2
        drift = mu + world['drift_amplitude']*(2*phase-1)
        ret = drift + sigma*rng.standard_normal(size)
    else:
        raise ValueError(f'Unknown world model: {model}')
    return ret[burn_in:]


def exact_rate(k, n, confidence=.95):
    if n == 0:
        return dict(count=k, denominator=n, rate=None, ci_low=None, ci_high=None)
    interval = binomtest(k, n).proportion_ci(confidence_level=confidence, method='exact')
    return dict(count=k, denominator=n, rate=k/n,
                ci_low=float(interval.low), ci_high=float(interval.high))


def run_world(index, world, plan):
    draws = []
    for replicate in range(plan['replicates_per_world']):
        ret = generate_world(world, plan['monthly_observations'], seeded(plan,index,replicate,0),
                             burn_in=plan['burn_in_months'], horizon=plan['forward_months'],
                             threshold=plan['trigger_simple_return'])
        draws.append(dict(world=world['id'], replicate=replicate,
                          **bootstrap_test(ret, plan, seeded(plan,index,replicate,1))))
        if (replicate+1) % 250 == 0:
            print(f"{world['id']}: {replicate+1}/{plan['replicates_per_world']}", flush=True)
    available = sum(d['available'] for d in draws)
    fires = sum(d['rejected'] for d in draws)
    all_rate = exact_rate(fires, len(draws), plan['confidence_level'])
    available_rate = exact_rate(fires, available, plan['confidence_level'])
    null_pass = None
    if world['kind'] == 'null':
        null_pass = bool(available and all_rate['ci_high'] <= plan['maximum_null_ci_upper']
                         and available_rate['ci_high'] <= plan['maximum_null_ci_upper'])
    oracle = None
    if world['kind'] != 'null':
        ret = generate_world(world, plan['oracle_months'], seeded(plan,index,0,2),
                             burn_in=plan['burn_in_months'], horizon=plan['forward_months'],
                             threshold=plan['trigger_simple_return'])
        oracle = rate_summary(ret, plan['forward_months'], plan['trigger_simple_return'])
    summary = dict(world=world, attempted=len(draws), available=available,
                   unavailable_reasons=dict(Counter(d['unavailable_reason'] for d in draws
                                                    if not d['available'])),
                   rejections_all_attempts=all_rate, rejections_available=available_rate,
                   null_gate_pass=null_pass, oracle_population_approximation=oracle,
                   median_events=float(np.median([d['completed_events'] for d in draws])),
                   median_baseline_rate=float(np.median([d['baseline_rate'] for d in draws])))
    return summary, draws


def freeze_manifest():
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                files={name:sha256(ROOT/name) for name in
                       ['PROTOCOL.md','METHOD.md','calibration_plan.json','monthly_calibration.py',
                        'requirements.txt']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest_path = ROOT/'frozen_manifest.json'
    if args.freeze:
        if manifest_path.exists():
            parser.error('An existing freeze cannot be replaced')
        manifest_path.write_text(json.dumps(freeze_manifest(),indent=2)+'\n',encoding='utf-8')
        print('Saved frozen_manifest.json; commit it before calibration.')
        return
    if args.output is None or args.output.exists():
        parser.error('A new --output directory is required')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        if sha256(ROOT/name) != expected:
            parser.error(f'Frozen file changed: {name}')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in list(manifest['files'])+['frozen_manifest.json']:
        blob = subprocess.check_output(['git','show',f'{commit}:verdicts/2026-10/{name}'],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != sha256(ROOT/name):
            parser.error(f'Commit does not pin exact frozen bytes: {name}')
    plan = json.loads((ROOT/'calibration_plan.json').read_text())
    args.output.mkdir(parents=True)
    summaries, draws = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(run_world,i,world,plan):i for i,world in enumerate(plan['worlds'])}
        for future in as_completed(jobs):
            summary, rows = future.result()
            index = jobs[future]
            summaries[index] = summary
            draws.extend(rows)
            (args.output/f"{summary['world']['id']}.json").write_text(
                json.dumps(summary,indent=2)+'\n',encoding='utf-8')
            print(f"COMPLETED {summary['world']['id']}",flush=True)
    worlds = [summaries[i] for i in range(len(plan['worlds']))]
    passed = all(w['null_gate_pass'] for w in worlds if w['world']['kind']=='null')
    powered = [w['world']['id'] for w in worlds if w['world']['kind']=='alternative'
               and w['rejections_all_attempts']['ci_low'] >= .8]
    result = dict(schema_version=1, method_id=plan['method_id'], frozen_commit=commit,
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  frozen_manifest_sha256=sha256(manifest_path),
                  verdict='PASS_ALPHA_GATE' if passed else 'FAIL_ALPHA_GATE',
                  market_inference_permitted=passed, demonstrated_80pct_power_worlds=powered,
                  worlds=worlds)
    draw_path = args.output/'draws.csv'
    draws.sort(key=lambda d:(d['world'],d['replicate']))
    with draw_path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(draws[0]))
        writer.writeheader()
        writer.writerows(draws)
    result['draws_sha256'] = sha256(draw_path)
    (args.output/'calibration.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f"{result['verdict']}; demonstrated >=80% power: {powered}",flush=True)


if __name__ == '__main__':
    main()
