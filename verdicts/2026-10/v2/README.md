# V2 monthly-event inference candidate

This replaces the [failed stationary-bootstrap candidate](../CALIBRATION.md)
with [sign randomization and a maximum over drift uncertainty](METHOD.md).
The original question and event definitions remain. The inferential model is
explicitly conditional on constant drift and sign invariance given centered
magnitudes; it is not a generic zero-contrast test for all dependent processes.

The [plan](calibration_plan.json) pins fresh seeds, all eleven worlds, simulation
counts, null gate, and the subsequent market-data hash and evaluation seed.
The implementation is committed before any certification run. Failed versions
and weak detection limits are retained without changing their thresholds.

## Reproduce

From the repository root, on the frozen commit named by the calibration result:

```text
python -m pip install -r verdicts/2026-10/v2/requirements.txt
python -m pytest tests/test_monthly_sign_randomization.py -q
python verdicts/2026-10/v2/sign_randomization.py --output verdicts/2026-10/v2/replication --workers 4
```

Use a new output directory. The runner verifies exact frozen file and commit
hashes. Do not replace the committed manifest. The generator is the unchanged,
frozen V1 generator, with newly declared certification seeds and one additional
strong intervention. V1's certification data are development evidence only.

## Conditional market evaluation

Only after all true-null gates pass and independent checks agree:

```text
python verdicts/2026-10/v2/market_evaluation.py --snapshot verdicts/2026-07/reproduction/data/audit_index_2026-10-01/^GSPC.csv --calibration verdicts/2026-10/v2/evidence/calibration.json --output verdicts/2026-10/v2/market-replication
```

The raw historical vendor data remain local; the frozen hash and retrieval
instructions are in the prior reproduction package. Evaluation publishes
derived event and eligibility/outcome ledgers, with the original calendar
definitions, censoring, source-count checks and model limits. No new inference
should be described as proof of absence, causality or a trading edge.
