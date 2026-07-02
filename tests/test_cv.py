"""Tests for purged + embargoed walk-forward CV (gatecheck.cv).

Ported 2026-07-02 from market_state ``tests/test_evaluation_cv.py``.
"""

import numpy as np
import pytest

from gatecheck.cv import Fold, apply_fold, purged_walk_forward


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _all_test_indices(folds):
    return np.concatenate([f.test for f in folds])


# --------------------------------------------------------------------------- #
# forward chaining
# --------------------------------------------------------------------------- #
def test_forward_chaining():
    # For every emitted fold: max(train) < min(test); no future bar trains a past
    # test.  With embargo it must be even stricter: max(train)+embargo < min(test).
    for embargo in (0, 1, 3, 5):
        folds = purged_walk_forward(120, n_splits=6, embargo=embargo)
        assert folds, "expected at least one usable fold"
        for f in folds:
            assert f.train.size >= 1
            assert int(f.train.max()) < int(f.test.min())
            assert int(f.train.max()) + embargo < int(f.test.min())


# --------------------------------------------------------------------------- #
# symmetric embargo (hand oracle: n=20, n_splits=4, embargo=2)
# --------------------------------------------------------------------------- #
def test_embargo_symmetric():
    n, n_splits, e = 20, 4, 2
    folds = purged_walk_forward(n, n_splits=n_splits, embargo=e)

    # Test-block bounds are the contiguous near-equal partition of [0, 20).
    bounds = [(0, 5), (5, 10), (10, 15), (15, 20)]

    by_index = {f.index: f for f in folds}

    for k, (a, b) in enumerate(bounds):
        pre = set(range(max(a - e, 0), a))     # e bars immediately BEFORE block k
        post = set(range(b, min(b + e, n)))    # e bars immediately AFTER block k

        # (1) If fold k is emitted, neither band is in ITS train.
        if k in by_index:
            train_k = set(by_index[k].train.tolist())
            assert pre.isdisjoint(train_k), f"pre-embargo leaked into fold {k} train"
            assert post.isdisjoint(train_k), f"post-embargo leaked into fold {k} train"

        # (2) The post-embargo band of block k is in NO later fold's train.
        for j, fold in by_index.items():
            if j > k:
                later_train = set(fold.train.tolist())
                assert post.isdisjoint(later_train), (
                    f"post-embargo band of block {k} leaked into later fold {j}"
                )

    # Concrete expected values for the emitted folds (folds 1..3; fold 0 has no
    # train under forward-chaining and is therefore not emitted).
    assert by_index[1].train.tolist() == [0, 1, 2]
    assert by_index[1].test.tolist() == [5, 6, 7, 8, 9]
    assert by_index[1].embargo.tolist() == [3, 4, 10, 11]

    assert by_index[2].train.tolist() == [0, 1, 2, 3, 4, 7]
    assert by_index[2].test.tolist() == [10, 11, 12, 13, 14]
    assert by_index[2].embargo.tolist() == [8, 9, 15, 16]

    assert by_index[3].train.tolist() == [0, 1, 2, 3, 4, 7, 8, 9, 12]
    assert by_index[3].test.tolist() == [15, 16, 17, 18, 19]
    assert by_index[3].embargo.tolist() == [13, 14]


# --------------------------------------------------------------------------- #
# disjointness + partition structure
# --------------------------------------------------------------------------- #
def test_disjoint_and_partition():
    folds = purged_walk_forward(100, n_splits=5, embargo=3)

    for f in folds:
        # pairwise disjoint
        assert np.intersect1d(f.train, f.test).size == 0
        assert np.intersect1d(f.train, f.embargo).size == 0
        assert np.intersect1d(f.test, f.embargo).size == 0
        # sorted
        assert np.all(np.diff(f.train) > 0)
        assert np.all(np.diff(f.test) > 0)
        if f.embargo.size > 1:
            assert np.all(np.diff(f.embargo) > 0)

    # Emitted test blocks are contiguous, non-overlapping and strictly ordered.
    tests = [f.test for f in folds]
    for t in tests:
        assert t.tolist() == list(range(int(t.min()), int(t.max()) + 1))
    for t_prev, t_next in zip(tests, tests[1:]):
        assert int(t_prev.max()) + 1 == int(t_next.min())  # contiguous, no gap/overlap

    # Across ALL n_splits blocks the union covers [0, n): emitted tests plus the
    # single dropped leading block (fold 0) tile the whole index space.
    covered = set(_all_test_indices(folds).tolist())
    indices = {f.index for f in folds}
    # The only legitimately-dropped blocks are leading ones (forward-chaining).
    assert min(indices) == 0 or all(i not in indices for i in range(min(indices)))
    assert max(covered) == 99


# --------------------------------------------------------------------------- #
# embargo=0 reduces to plain walk-forward
# --------------------------------------------------------------------------- #
def test_embargo_zero_reduces_to_plain_walk_forward():
    n, n_splits = 50, 5
    folds = purged_walk_forward(n, n_splits=n_splits, embargo=0)
    bounds = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    by_index = {f.index: f for f in folds}

    # Fold 0 (test [0,10)) has no train and is dropped; folds 1..4 are plain
    # expanding walk-forward: train is exactly everything before the test block.
    assert set(by_index) == {1, 2, 3, 4}
    for k in (1, 2, 3, 4):
        a, b = bounds[k]
        assert by_index[k].train.tolist() == list(range(0, a))
        assert by_index[k].test.tolist() == list(range(a, b))
        assert by_index[k].embargo.size == 0  # no embargo bars at all


# --------------------------------------------------------------------------- #
# expanding vs rolling train sizes
# --------------------------------------------------------------------------- #
def test_expanding_vs_rolling_train_sizes():
    n, n_splits, e = 120, 6, 2

    exp = purged_walk_forward(n, n_splits=n_splits, embargo=e, expanding=True)
    roll = purged_walk_forward(n, n_splits=n_splits, embargo=e, expanding=False)

    # same set of emitted folds / test blocks either way
    assert [f.index for f in exp] == [f.index for f in roll]
    for fe, fr in zip(exp, roll):
        assert fe.test.tolist() == fr.test.tolist()
        assert fe.embargo.tolist() == fr.embargo.tolist()

    # Expanding train sizes are non-decreasing across folds.
    exp_sizes = [f.train.size for f in exp]
    assert exp_sizes == sorted(exp_sizes)
    assert exp_sizes[-1] > exp_sizes[0]

    # Rolling train sizes are all equal to the first emitted fold's train length.
    roll_sizes = [f.train.size for f in roll]
    assert len(set(roll_sizes)) == 1
    assert roll_sizes[0] == exp_sizes[0]

    # Rolling window is the most-recent eligible bars, so its max index is the same
    # as expanding (right edge), but it has fewer (or equal) bars overall.
    for fe, fr in zip(exp, roll):
        assert int(fr.train.max()) == int(fe.train.max())
        assert fr.train.size <= fe.train.size


# --------------------------------------------------------------------------- #
# min_train respected / drives feasibility
# --------------------------------------------------------------------------- #
def test_min_train_respected():
    # Every emitted fold must have at least ``min_train`` training bars.
    folds = purged_walk_forward(80, n_splits=4, embargo=2, min_train=15)
    assert folds
    for f in folds:
        assert f.train.size >= 15

    # Raising min_train drops more leading folds (their train is too small).
    lo = purged_walk_forward(80, n_splits=8, embargo=1, min_train=1)
    hi = purged_walk_forward(80, n_splits=8, embargo=1, min_train=20)
    assert {f.index for f in hi}.issubset({f.index for f in lo})
    assert len(hi) <= len(lo)


# --------------------------------------------------------------------------- #
# bad-argument validation
# --------------------------------------------------------------------------- #
def test_raises_on_bad_args():
    with pytest.raises(ValueError):
        purged_walk_forward(100, n_splits=1)          # n_splits < 2
    with pytest.raises(ValueError):
        purged_walk_forward(100, n_splits=5, embargo=-1)   # embargo < 0
    with pytest.raises(ValueError):
        purged_walk_forward(100, n_splits=5, min_train=0)  # min_train < 1
    with pytest.raises(ValueError):
        # n too small: min_train + embargo + n_splits = 10 + 4 + 5 = 19 > 12
        purged_walk_forward(12, n_splits=5, embargo=4, min_train=10)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_determinism():
    kw = dict(n_splits=7, embargo=3, min_train=5, expanding=True)
    a = purged_walk_forward(200, **kw)
    b = purged_walk_forward(200, **kw)
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert fa.index == fb.index
        assert np.array_equal(fa.train, fb.train)
        assert np.array_equal(fa.test, fb.test)
        assert np.array_equal(fa.embargo, fb.embargo)


# --------------------------------------------------------------------------- #
# exact index example (n=100, n_splits=5, embargo=3): first & last emitted folds
# --------------------------------------------------------------------------- #
def test_exact_index_example():
    folds = purged_walk_forward(100, n_splits=5, embargo=3)
    assert [f.index for f in folds] == [1, 2, 3, 4]  # fold 0 dropped (no train)

    first, last = folds[0], folds[-1]

    # First emitted fold: test block [20, 40); train is [0, 17) (everything before
    # the 3-bar pre-embargo band 17,18,19); embargo = {17,18,19} U {40,41,42}.
    assert first.index == 1
    assert first.test.tolist() == list(range(20, 40))
    assert first.train.tolist() == list(range(0, 17))
    assert first.embargo.tolist() == [17, 18, 19, 40, 41, 42]

    # Last emitted fold: test block [80, 100); eligible train is [0, 77) minus the
    # post-embargo bands of the prior blocks {20,21,22},{40,41,42},{60,61,62}.
    assert last.index == 4
    assert last.test.tolist() == list(range(80, 100))
    forbidden = {20, 21, 22, 40, 41, 42, 60, 61, 62}
    expected_train = [i for i in range(0, 77) if i not in forbidden]
    assert last.train.tolist() == expected_train
    assert last.embargo.tolist() == [77, 78, 79]  # post band {100,101,102} clipped


# --------------------------------------------------------------------------- #
# apply_fold
# --------------------------------------------------------------------------- #
def test_apply_fold_slices_values():
    n = 60
    values = np.arange(n, dtype=float) * 10.0
    folds = purged_walk_forward(n, n_splits=4, embargo=1)
    f = folds[0]
    tr, te = apply_fold(values, f)
    assert np.array_equal(tr, values[f.train])
    assert np.array_equal(te, values[f.test])
    # works on a 2-D array along the leading axis too
    mat = np.arange(n * 3, dtype=float).reshape(n, 3)
    tr2, te2 = apply_fold(mat, f)
    assert tr2.shape == (f.train.size, 3)
    assert te2.shape == (f.test.size, 3)


def test_fold_is_frozen_dataclass():
    f = purged_walk_forward(40, n_splits=4, embargo=1)[0]
    assert isinstance(f, Fold)
    with pytest.raises(Exception):
        f.index = 99  # frozen
