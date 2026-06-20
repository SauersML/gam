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

from itertools import permutations

from _synth_sae_metrics import (
    _hungarian_max_numpy,
    feature_uniqueness,
    firing_latent_mask,
    match_directions,
    n_firing_latents,
    recovery_scores,
)


def _orthonormal(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, n)))
    return q.T[:n]


def _brute_force_max_mass(sim: np.ndarray) -> float:
    """Optimal one-to-one matched mass by exhaustive search (small matrices)."""
    n, m = sim.shape
    if min(n, m) == 0:
        return 0.0
    if n <= m:
        best = 0.0
        for cols in permutations(range(m), n):
            best = max(best, float(sum(sim[r, c] for r, c in enumerate(cols))))
        return best
    return _brute_force_max_mass(sim.T)


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


def test_exact_hungarian_beats_greedy() -> None:
    # The old greedy fallback took 0.9 then was forced to 0.0 (mass 0.9); the
    # optimal assignment takes both 0.8s (mass 1.6). The exact matcher must find
    # the optimum.
    sim = np.array([[0.9, 0.8], [0.8, 0.0]])
    rows, cols = _hungarian_max_numpy(sim)
    assert abs(sim[rows, cols].sum() - 1.6) < 1e-12


def test_hungarian_matches_brute_force() -> None:
    rng = np.random.default_rng(11)
    for _ in range(40):
        n = int(rng.integers(1, 6))
        m = int(rng.integers(1, 6))
        sim = rng.random((n, m))
        rows, cols = _hungarian_max_numpy(sim)
        assert rows.shape[0] == min(n, m)
        assert np.unique(cols).size == cols.size  # one-to-one
        got = float(sim[rows, cols].sum())
        assert abs(got - _brute_force_max_mass(sim)) < 1e-9


def test_jaccard_one_only_for_perfect_bijection() -> None:
    truth = _orthonormal(10, 16, seed=20)
    assert recovery_scores(truth.copy(), truth).jaccard > 0.999
    # Equal width but all-duplicate: Jaccard collapses well below 1.
    dup = np.repeat(truth[:1], 10, axis=0)
    assert recovery_scores(dup, truth).jaccard < 0.2


def test_dead_slots_penalized_via_total_count() -> None:
    # 10 perfect + 6 dead (zero-norm) decoder slots. Matching is over the 10
    # live dirs (mass ~10), but counting all 16 slots in L penalizes the dead
    # capacity, exactly as a width-16 SAE recovering 10 truths should be scored.
    truth = _orthonormal(10, 32, seed=21)
    live = truth.copy()
    rec_live = recovery_scores(live, truth)  # default L = live count
    rec_total = recovery_scores(live, truth, n_learned_total=16)
    assert rec_live.precision > 0.999  # ignoring dead capacity -> perfect
    assert abs(rec_total.precision - 10.0 / 16.0) < 1e-6  # dead capacity penalized
    assert abs(rec_total.cardinality_ceiling - 10.0 / 16.0) < 1e-6


def test_empty_learned_is_zero() -> None:
    truth = _orthonormal(10, 16, seed=8)
    empty = np.zeros((0, 16))
    assert feature_uniqueness(empty, truth) == 0.0
    rec = recovery_scores(empty, truth)
    assert (rec.mcc, rec.precision, rec.recall, rec.f1) == (0.0, 0.0, 0.0, 0.0)
    assert rec.rows.size == 0


def test_functional_dead_latents_differ_from_geometric() -> None:
    # #1435: a latent with a nonzero decoder direction that NEVER fires on the
    # eval set is functionally dead, even though it is geometrically live.
    # 4 latents x 6 samples; latent index 1 never exceeds the 1e-8 threshold.
    activations = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.3, 0.0],
            [2.0, 0.0, 0.0, 1.1],
            [0.7, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.2, 0.8],
            [1.3, 0.0, 0.6, 0.4],
        ],
        dtype=float,
    )
    mask = firing_latent_mask(activations)
    assert mask.tolist() == [True, False, True, True]
    assert n_firing_latents(activations) == 3
    # functional dead count = 4 slots - 3 firing
    assert 4 - n_firing_latents(activations) == 1


def test_firing_threshold_respected() -> None:
    # A latent firing only below the threshold is treated as dead at that
    # threshold (consistency with the learned_l0 |a|>1e-8 convention).
    activations = np.array([[1e-9, 1.0], [2e-9, 0.0]], dtype=float)
    assert n_firing_latents(activations, threshold=1e-8) == 1
    assert n_firing_latents(activations, threshold=0.0) == 2


def test_firing_empty_and_1d() -> None:
    assert n_firing_latents(np.zeros((5, 0))) == 0
    assert firing_latent_mask(np.zeros((5, 0))).size == 0
    # 1-D treated as a single latent.
    assert n_firing_latents(np.zeros(5)) == 0
    assert n_firing_latents(np.array([0.0, 0.0, 0.5])) == 1


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
