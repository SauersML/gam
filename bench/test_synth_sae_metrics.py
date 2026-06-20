#!/usr/bin/env python3
"""NumPy-only tests for bench/_synth_sae_metrics.py (no gamfit/torch needed).

Run directly (``python3 bench/test_synth_sae_metrics.py``) or under pytest.
These are the direction-matrix cases that the original #1432 "verification"
could not catch, because it hard-coded the assignment count instead of feeding
real directions. The decisive point: at equal width, the old width-ratio metric
``min(L, T) / max(L, T)`` is identically 1.0 for perfect, all-duplicate, and
random dictionaries alike. The quality-aware metrics here separate them.
"""

from __future__ import annotations

import numpy as np

from _synth_sae_metrics import (
    feature_uniqueness,
    match_directions,
    recovery_scores,
)


def _orthonormal(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, n)))
    return q.T[:n]


def _old_width_ratio(learned: np.ndarray, truth: np.ndarray) -> float:
    """The metric proposed in PR #1432, reduced to what it actually computes."""
    rows, _cols, _ = match_directions(learned, truth)
    return float(rows.shape[0] / max(learned.shape[0], truth.shape[0]))


def test_assignment_count_is_always_min_dim() -> None:
    # Documents the root cause: matchers always fill min(L, T) pairs.
    truth = _orthonormal(10, 16, seed=1)
    for n_learned in (4, 10, 25):
        learned = _orthonormal(n_learned, 16, seed=n_learned)
        rows, cols, _ = match_directions(learned, truth)
        assert rows.shape[0] == min(n_learned, 10)
        assert cols.shape[0] == rows.shape[0]


def test_perfect_recovery_scores_one() -> None:
    truth = _orthonormal(10, 16, seed=2)
    learned = truth.copy()
    assert feature_uniqueness(learned, truth) == 1.0
    rec = recovery_scores(learned, truth)
    assert rec.f1 > 0.999
    assert rec.precision > 0.999
    assert rec.recall > 0.999
    assert rec.mcc > 0.999


def test_equal_width_duplicates_penalized() -> None:
    truth = _orthonormal(4, 16, seed=3)
    learned = np.repeat(truth[:1], 4, axis=0)  # four copies of one feature
    # Old metric: tautologically 1.0 at equal width.
    assert _old_width_ratio(learned, truth) == 1.0
    # New metrics expose the collapse.
    assert feature_uniqueness(learned, truth) == 0.25
    assert recovery_scores(learned, truth).f1 < 0.3


def test_equal_width_random_penalized_by_recovery() -> None:
    truth = _orthonormal(16, 64, seed=4)
    rng = np.random.default_rng(5)
    learned = rng.standard_normal((16, 64))
    learned /= np.linalg.norm(learned, axis=1, keepdims=True)
    assert _old_width_ratio(learned, truth) == 1.0  # old metric blind to quality
    # Recovery F1 collapses for random directions in a high-dim space.
    assert recovery_scores(learned, truth).f1 < 0.5


def test_overcomplete_excess_lowers_precision() -> None:
    truth = _orthonormal(10, 32, seed=6)
    junk = _orthonormal(6, 32, seed=99)
    learned = np.vstack([truth, junk])  # 10 perfect + 6 junk
    rec = recovery_scores(learned, truth)
    assert rec.recall > 0.99  # every truth feature still recovered
    assert rec.precision < 0.7  # but excess learned dirs drag precision down
    assert rec.precision < rec.recall


def test_undercomplete_lowers_recall() -> None:
    truth = _orthonormal(10, 32, seed=7)
    learned = truth[:6].copy()  # only 6 of 10 truth features learned
    rec = recovery_scores(learned, truth)
    assert rec.precision > 0.99  # the 6 it has are perfect
    assert rec.recall < 0.7  # but 4 truth features are unrecovered
    assert abs(rec.recall - 0.6) < 1e-6


def test_empty_learned_is_zero() -> None:
    truth = _orthonormal(10, 16, seed=8)
    empty = np.zeros((0, 16))
    assert feature_uniqueness(empty, truth) == 0.0
    rec = recovery_scores(empty, truth)
    assert (rec.mcc, rec.precision, rec.recall, rec.f1) == (0.0, 0.0, 0.0, 0.0)
    assert rec.rows.size == 0


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
