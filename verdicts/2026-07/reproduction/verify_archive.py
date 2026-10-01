"""Compare every archived axis4 value, without turning new data into original data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def compare(path: Path, tolerance=1e-9):
    archived = json.loads((ROOT / 'archive' / 'axis4_results.json').read_text(encoding='utf-8'))
    actual = json.loads(path.read_text(encoding='utf-8'))
    differences, maximum, count = [], 0., 0

    def walk(a, b, label):
        nonlocal maximum, count
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                differences.append(label + ': keys differ')
                return
            for key in a:
                walk(a[key], b[key], f'{label}.{key}')
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                differences.append(label + ': list length differs')
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{label}[{i}]')
        elif isinstance(a, (float, int)) and not isinstance(a, bool):
            count += 1
            delta = abs(a - b)
            maximum = max(maximum, delta)
            if not math.isfinite(delta) or delta > tolerance:
                differences.append(f'{label}: expected {a}, received {b}')
        elif a != b:
            differences.append(f'{label}: expected {a}, received {b}')

    walk(archived, actual, 'axis4')
    result = dict(matched=not differences, tolerance=tolerance, numeric_values_compared=count,
                  maximum_absolute_difference=maximum, differences=differences)
    if differences:
        raise ValueError(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    print(json.dumps(compare(parser.parse_args().results), indent=2))
