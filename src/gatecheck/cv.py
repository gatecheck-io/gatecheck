"""Purged + embargoed walk-forward cross-validation over an ordered bar index.

Lineage: extracted 2026-07-02 from the market_state research program
(``market_os/evaluation/cv.py``, post-audit).

This module is intentionally *data-free*: every routine takes only ``n`` (a bar
count) and integer hyper-parameters and returns integer index arrays.  Because no
values, returns, labels, timestamps or RNG ever enter the split logic, the splitter
*cannot* leak information -- it is pure, deterministic index math.

Forward-chaining only
---------------------
The half-open interval ``[0, n)`` is partitioned into ``n_splits`` contiguous,
non-overlapping, *ordered* test blocks of near-equal size.  Each fold trains only on
bars that lie strictly before its test block::

    max(train) < min(test)        # no future bar ever trains a past test

Symmetric embargo
-----------------
Forward-looking labels straddle the train/test boundary on *both* sides, so the
embargo is applied symmetrically (both bands are purged, not only the pre-test
band):

* The ``embargo`` bars immediately **before** a test block are dropped from *that*
  fold's train (their forward-horizon labels overlap the test block).
* The ``embargo`` bars immediately **after** a test block are excluded from the
  train of every *later* fold (those bars' labels reach back into the now-past test
  block, and in an expanding scheme they would otherwise be eligible).

Concretely, for fold ``k`` with test block ``[a, b)``::

    train = [ i for i in eligible
              if i < a - embargo
              and i not in any prior test's post-embargo band ]

where a prior test block ``[a_j, b_j)`` (j < k) contributes the post-embargo band
``[b_j, b_j + embargo)``.

``expanding=True`` lets the train grow fold-over-fold; ``expanding=False`` uses a
rolling window whose length equals the *first* fold's train length.

No randomness is used anywhere -- the result is fully determined by the arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Fold", "purged_walk_forward", "apply_fold"]


@dataclass(frozen=True)
class Fold:
    """One walk-forward split.

    Attributes
    ----------
    index:
        0-based ordinal of the fold (folds are ordered by test-block position).
    train:
        Sorted int index array of training bars (may be empty only if it would
        violate ``min_train`` -- which ``purged_walk_forward`` forbids, so in
        practice non-empty).
    test:
        Sorted int index array of the contiguous test block.
    embargo:
        Sorted int index array of the embargoed bars for this fold: the ``embargo``
        bars immediately before *and* after this fold's test block (clipped to
        ``[0, n)`` and excluding the test bars themselves).

    The three arrays ``train``, ``test``, ``embargo`` are pairwise disjoint.
    """

    index: int
    train: np.ndarray
    test: np.ndarray
    embargo: np.ndarray


def _test_block_bounds(n: int, n_splits: int) -> list[tuple[int, int]]:
    """Return ``n_splits`` (start, stop) half-open bounds partitioning ``[0, n)``.

    Blocks are contiguous, non-overlapping, ordered, and of near-equal size.  The
    first ``n % n_splits`` blocks are one element larger (numpy-style splitting), so
    every block is non-empty whenever ``n >= n_splits``.
    """
    base = n // n_splits
    rem = n % n_splits
    bounds: list[tuple[int, int]] = []
    start = 0
    for k in range(n_splits):
        size = base + (1 if k < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def purged_walk_forward(
    n: int,
    *,
    n_splits: int = 5,
    embargo: int = 0,
    min_train: int = 1,
    expanding: bool = True,
) -> list[Fold]:
    """Build purged + embargoed forward-chaining walk-forward folds.

    Parameters
    ----------
    n:
        Number of ordered bars; the index space is ``[0, n)``.
    n_splits:
        Number of test blocks (>= 2).
    embargo:
        Symmetric embargo width in bars (>= 0).
    min_train:
        Minimum number of training bars required in *every* fold (>= 1).
    expanding:
        If ``True`` the training window grows across folds; if ``False`` a rolling
        window of fixed length (the first fold's train length) is used.

    Returns
    -------
    list[Fold]
        One :class:`Fold` per test block, ordered by test-block position.

    Raises
    ------
    ValueError
        If ``n_splits < 2``, ``embargo < 0``, ``min_train < 1``, or ``n`` is too
        small to honor ``min_train`` training bars plus ``n_splits`` test blocks
        plus the embargo band of the first test block.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    if min_train < 1:
        raise ValueError(f"min_train must be >= 1, got {min_train}")

    # Global feasibility.  Forward-chaining means the very first test block (which
    # starts at index 0) can never have a training set, so it -- and possibly the
    # next few blocks while the train is still below ``min_train`` -- are simply not
    # emitted.  For the configuration to yield *any* usable fold, the index space
    # must hold at least ``min_train`` training bars, the ``embargo`` pre-test band,
    # and the ``n_splits`` test blocks (each >= 1 bar):
    #     n >= min_train + embargo + n_splits
    # This is the binding constraint for the last (largest-train) fold to clear
    # ``min_train``; smaller ``n`` cannot honor the request and is an error.
    min_n = min_train + embargo + n_splits
    if n < min_n:
        raise ValueError(
            f"n={n} too small: need at least min_train({min_train}) + "
            f"embargo({embargo}) + n_splits({n_splits}) = {min_n} bars."
        )

    bounds = _test_block_bounds(n, n_splits)

    # Rolling-window length: the train length of the *first emitted* fold (the
    # earliest fold whose eligible train reaches ``min_train``).  Computed below as
    # we sweep folds in order; the first qualifying train fixes the window.
    rolling_len: int | None = None

    folds: list[Fold] = []
    for k, (a, b) in enumerate(bounds):
        test = np.arange(a, b, dtype=np.int64)

        # Post-embargo bands of all *prior* test blocks are forbidden for the train.
        forbidden = np.zeros(n, dtype=bool)
        for j in range(k):
            bj = bounds[j][1]
            lo = bj
            hi = min(bj + embargo, n)
            if hi > lo:
                forbidden[lo:hi] = True

        # Eligible training pool: strictly before the pre-test embargo band.
        train_hi = a - embargo  # exclusive
        candidate = np.arange(0, max(train_hi, 0), dtype=np.int64)
        if candidate.size:
            candidate = candidate[~forbidden[candidate]]

        # Forward-chaining: a fold whose eligible train cannot reach ``min_train`` is
        # genuinely unusable (nothing precedes it).  Skip it rather than fabricate a
        # leaky train.  The global guard above ensures the last fold always clears.
        if candidate.size < min_train:
            continue

        if not expanding:
            # Rolling window of fixed length (= first emitted fold's train length),
            # anchored to the right edge of the eligible pool (most recent bars).
            if rolling_len is None:
                rolling_len = int(candidate.size)
            if candidate.size > rolling_len:
                candidate = candidate[candidate.size - rolling_len:]

        train = np.asarray(candidate, dtype=np.int64)

        # Symmetric embargo set for *this* fold: the bands immediately before and
        # after this test block, clipped and excluding the test bars themselves.
        emb_lo_pre = max(a - embargo, 0)
        emb_pre = np.arange(emb_lo_pre, a, dtype=np.int64)
        emb_hi_post = min(b + embargo, n)
        emb_post = np.arange(b, emb_hi_post, dtype=np.int64)
        embargo_idx = np.concatenate([emb_pre, emb_post]).astype(np.int64)
        embargo_idx = np.sort(embargo_idx)

        # ---- hard, in-code invariants (cheap; guard against regressions) --------
        if train.size:
            assert int(train.max()) + embargo < int(test.min()), (
                f"forward-chaining violated in fold {k}: "
                f"max(train)+embargo={int(train.max()) + embargo} "
                f">= min(test)={int(test.min())}"
            )
        # pairwise disjoint
        assert np.intersect1d(train, test).size == 0, f"train/test overlap fold {k}"
        assert np.intersect1d(train, embargo_idx).size == 0, (
            f"train/embargo overlap fold {k}"
        )
        assert np.intersect1d(test, embargo_idx).size == 0, (
            f"test/embargo overlap fold {k}"
        )
        # sortedness
        assert np.all(np.diff(train) > 0) if train.size > 1 else True
        assert np.all(np.diff(test) > 0) if test.size > 1 else True

        folds.append(Fold(index=k, train=train, test=test, embargo=embargo_idx))

    return folds


def apply_fold(values: np.ndarray, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Slice ``values`` into ``(values[fold.train], values[fold.test])``.

    Parameters
    ----------
    values:
        Array whose leading axis is indexed by bar position (length ``n``).
    fold:
        A :class:`Fold` produced by :func:`purged_walk_forward`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(train_values, test_values)`` selected along the leading axis.
    """
    values = np.asarray(values)
    return values[fold.train], values[fold.test]
