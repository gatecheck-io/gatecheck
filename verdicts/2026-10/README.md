# Calendar-month follow-up: method calibration

The [study protocol](PROTOCOL.md) fixes the question separately from the
[July source reproduction](../2026-07/reproduction/CLAIM_AUDIT.md).
[METHOD.md](METHOD.md) and [calibration_plan.json](calibration_plan.json) fix the
first candidate's assumptions, seeds, counts and rejection rule.

The first candidate [failed calibration](CALIBRATION.md). Four of six declared
zero-effect worlds failed the false-alarm gate, and no tested alternative
demonstrated 80% detection. No market p-value was evaluated. The frozen commit,
raw draws, independently checked results and exact file hashes are retained in
[evidence](evidence/).

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

## Verify the published results

On the current checkout:

```text
python verdicts/2026-10/verify_calibration.py --evidence verdicts/2026-10/evidence --output verification-replication.json
python -m pytest -q
```

The verifier recomputes all draw definitions and count arithmetic, confidence
intervals and decisions, all four long-world contrasts, and one selected scalar
bootstrap p-value per world. It reuses the frozen generator as its input source;
its scope does not include an independent rewrite of every simulation generator
or every p-value. The final suite has 122 passing tests.
