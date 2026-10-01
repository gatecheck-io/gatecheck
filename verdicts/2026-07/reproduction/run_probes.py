"""Run the seven recovered exploratory probes against one explicit offline snapshot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path)
    parser.add_argument('--frozen-spy', type=Path, help='Original undated NPZ: axis4 only, never fabricate dates')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.snapshot_dir) == bool(args.frozen_spy):
        parser.error('Choose exactly one of --snapshot-dir or --frozen-spy')
    if args.output.exists():
        parser.error('Output already exists; use a new directory')
    if args.snapshot_dir and importlib.util.find_spec('scipy') is None:
        parser.error('Archived uncertainty probes require SciPy: pip install scipy==1.14.1')
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + str(ROOT.parents[2] / 'src')
    env['PYTHONIOENCODING'] = 'utf-8'
    if args.snapshot_dir:
        env['GATECHECK_SNAPSHOT_DIR'] = str(args.snapshot_dir.resolve())
        probes = sorted((ROOT / 'probes').glob('axis*.py'))
    else:
        probes = [ROOT / 'probes' / 'axis4_fragility_multiplicity.py']
    rows = []
    for probe in probes:
        run_dir = args.output / probe.stem
        run_dir.mkdir()
        env['GATECHECK_RUN_DIR'] = str(run_dir.resolve())
        if args.frozen_spy:
            shutil.copyfile(args.frozen_spy, run_dir / 'spy_prices_axis4.npz')
        result = subprocess.run([sys.executable, str(probe)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        (run_dir / 'stdout.txt').write_bytes(result.stdout)
        (run_dir / 'stderr.txt').write_bytes(result.stderr)
        rows.append(dict(probe=probe.name, sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
                         returncode=result.returncode))
        print(f'{probe.name}: exit {result.returncode}', flush=True)
    manifest = dict(input_mode='frozen_undated_SPY_only' if args.frozen_spy else 'reacquired_dated_snapshot',
                    clock='2026-07-02T23:59:59Z', probes=rows)
    if args.frozen_spy:
        manifest['input_sha256'] = hashlib.sha256(args.frozen_spy.read_bytes()).hexdigest()
        from verify_archive import compare
        manifest['archive_comparison'] = compare(
            args.output / 'axis4_fragility_multiplicity' / 'axis4_results.json')
    else:
        manifest['snapshots'] = {p.stem: json.loads(p.read_text(encoding='utf-8'))
                                 for p in args.snapshot_dir.glob('*.metadata.json')}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    if any(row['returncode'] != 0 for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
