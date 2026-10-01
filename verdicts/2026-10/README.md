# Calendar-month follow-up: method calibration

The [study protocol](PROTOCOL.md) fixes the question separately from the
[July source reproduction](../2026-07/reproduction/CLAIM_AUDIT.md).
[METHOD.md](METHOD.md) and [calibration_plan.json](calibration_plan.json) fix the
first candidate's assumptions, seeds, counts and rejection rule.

The next operation is an offline synthetic calibration, not a new market verdict.
Its frozen commit and exact file hashes will be retained with the results.

## Reproduce the calibration

From the repository root, with Python 3.12.6:

```text
python -m pip install -r verdicts/2026-10/requirements.txt
python -m pytest tests/test_monthly_calibration.py -q
python verdicts/2026-10/monthly_calibration.py --output verdicts/2026-10/replication --workers 4
```

Use a new output directory. The runner refuses an existing output, a changed
frozen input, or an implementation whose exact bytes do not match its named Git
commit. The saved `frozen_manifest.json` was created before calibration; do not
regenerate or replace it. For results reproduction from a later documentation
commit, check out the `frozen_commit` named in `evidence/calibration.json` first.

Each of ten worlds gets 1,000 independent draws, with 999 bootstrap draws for
each available test. Null false-alarm rates are reported using all attempts and
available-test denominators. Alternative rates include unavailable attempts,
with availability and intervention strengths reported separately. Results do
not certify assumptions outside the declared worlds.
