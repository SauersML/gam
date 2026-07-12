"""Pins for the two-tier harvest contract and the cross-node tree reducer.

Three properties the frontier-distribution / harvest-economics design must hold:

1. **Graceful degradation.** A factor-LESS row still yields a valid metric — the
   Euclidean / identity fallback — so the recon-only majority of rows is usable
   without paying the per-row Fisher-factor cost.
2. **Deterministic importance subsample.** The designed importance subsample is
   bit-reproducible across two calls with the same seed (and order-independent of
   how importance was computed).
3. **Bit-reproducible cross-node reduction.** The fixed-topology
   :class:`TreeReducer` returns identical results regardless of the order
   partials arrive, and resumes from a checkpoint to the bit.

Numpy-level throughout; no torch build, no cluster, no clock entropy required.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gamfit.torch.distributed_reduce import (  # noqa: E402
    TreeReducer,
    build_reduction_tree,
)
from gamfit.torch.harvest_contract import (  # noqa: E402
    HarvestedRow,
    MetricConsumerContract,
    metric_quadratic_form,
    row_metric,
    select_importance_subsample,
    stream_harvest_with_selectivity,
)


# ---------------------------------------------------------------------------
# (1) Factor-less rows degrade to a valid (Euclidean) metric.
# ---------------------------------------------------------------------------


def test_factorless_row_degrades_to_identity_metric() -> None:
    p = 5
    rng = np.random.default_rng(0)
    act = rng.standard_normal(p)

    bare = HarvestedRow(activation=act, factor=None)
    assert not bare.has_factor
    assert bare.rank == 0

    W = row_metric(bare)
    # Euclidean fallback: W_n == I_p exactly.
    np.testing.assert_array_equal(W, np.eye(p))

    # Matrix-free quadratic form agrees with ‖v‖² under the identity metric.
    v = rng.standard_normal(p)
    assert metric_quadratic_form(bare, v) == float(v @ v)

    contract = MetricConsumerContract()
    assert contract.is_degraded(bare)
    np.testing.assert_array_equal(contract.dense_metric(bare), np.eye(p))


def test_row_with_factor_uses_output_fisher_metric() -> None:
    p, r = 6, 2
    rng = np.random.default_rng(1)
    act = rng.standard_normal(p)
    U = rng.standard_normal((p, r))

    row = HarvestedRow(activation=act, factor=U, mass_residual=0.3)
    assert row.has_factor
    assert row.rank == r

    # Present factor ⇒ W_n = U Uᵀ (the rank-r output-Fisher pullback).
    np.testing.assert_allclose(row_metric(row), U @ U.T)

    v = rng.standard_normal(p)
    # Matrix-free quad form == vᵀ (U Uᵀ) v.
    expected = float(v @ (U @ U.T) @ v)
    assert abs(metric_quadratic_form(row, v) - expected) < 1e-10

    contract = MetricConsumerContract()
    assert not contract.is_degraded(row)


# ---------------------------------------------------------------------------
# (2) Importance subsample is deterministic across calls with the same seed.
# ---------------------------------------------------------------------------


def test_importance_subsample_deterministic_same_seed() -> None:
    rng = np.random.default_rng(7)
    importance = rng.random(200)
    n_select = 32

    a = select_importance_subsample(importance, n_select, seed=123)
    b = select_importance_subsample(importance, n_select, seed=123)

    assert a.shape == (n_select,)
    np.testing.assert_array_equal(a, b)
    # Sorted, unique indices within range.
    assert np.all(a[:-1] < a[1:])
    assert a.min() >= 0 and a.max() < importance.shape[0]


def test_importance_subsample_order_independent_via_row_keys() -> None:
    # Same population, presented in two different orders but with stable row
    # keys: the selected GLOBAL keys must match (order independence).
    rng = np.random.default_rng(8)
    n = 100
    importance = rng.random(n)
    keys = list(range(n))

    perm = rng.permutation(n)
    importance_perm = importance[perm]
    keys_perm = [keys[i] for i in perm]

    sel = select_importance_subsample(importance, 20, seed=42, row_keys=keys)
    sel_perm = select_importance_subsample(
        importance_perm, 20, seed=42, row_keys=keys_perm
    )

    # Map both selections back to global keys; the chosen SETS must be equal.
    chosen = {keys[i] for i in sel}
    chosen_perm = {keys_perm[i] for i in sel_perm}
    assert chosen == chosen_perm


def test_importance_subsample_prefers_heavy_rows() -> None:
    # A handful of very heavy rows should almost always be selected.
    importance = np.full(50, 1e-6)
    heavy = [3, 17, 28, 41]
    for i in heavy:
        importance[i] = 1.0e6
    sel = set(int(i) for i in select_importance_subsample(importance, 4, seed=5))
    assert sel == set(heavy)


def test_select_clamps_and_empty() -> None:
    importance = np.array([0.5, 0.2, 0.9])
    # n_select larger than population clamps to all.
    full = select_importance_subsample(importance, 99, seed=0)
    np.testing.assert_array_equal(full, np.array([0, 1, 2]))
    # n_select == 0 → empty.
    empty = select_importance_subsample(importance, 0, seed=0)
    assert empty.shape == (0,)


# ---------------------------------------------------------------------------
# Streaming-with-selectivity: only selected rows compute factors.
# ---------------------------------------------------------------------------


def test_stream_computes_factors_only_for_selected() -> None:
    p, r = 4, 2
    n = 30
    rng = np.random.default_rng(11)
    importance = rng.random(n)
    activations = rng.standard_normal((n, p))
    n_select = 6

    expected_selected = set(
        int(i) for i in select_importance_subsample(importance, n_select, seed=99)
    )

    computed_for: list[int] = []

    def compute_factor(idx: int, act: np.ndarray) -> tuple[np.ndarray, float]:
        computed_for.append(idx)
        # Deterministic factor from the index (content irrelevant to the pin).
        U = np.full((p, r), float(idx) + 1.0)
        return U, 0.1 * idx

    # Stream rows in a SHUFFLED order to prove selection is order-independent.
    order = list(rng.permutation(n))
    stream = ((i, activations[i]) for i in order)

    rows = list(
        stream_harvest_with_selectivity(
            stream,
            importance=importance,
            n_select=n_select,
            compute_factor=compute_factor,
            seed=99,
        )
    )

    assert len(rows) == n
    # Exactly the selected rows computed a factor.
    assert set(computed_for) == expected_selected
    with_factor = {
        int(row.row_key) for row in rows if row.has_factor and row.row_key is not None
    }
    assert with_factor == expected_selected

    # Every row carries an activation; factor-less rows degrade to identity.
    for row in rows:
        assert row.activation.shape == (p,)
        W = row_metric(row)
        assert W.shape == (p, p)
        if not row.has_factor:
            np.testing.assert_array_equal(W, np.eye(p))


# ---------------------------------------------------------------------------
# (3) Cross-node TreeReducer: identical under permuted arrival order.
# ---------------------------------------------------------------------------


def _make_partials(n_ranks: int, dim: int, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Wide dynamic range so non-associative float summation would actually
    # differ under a different fold order if the topology were not fixed.
    scales = np.array([10.0 ** (k - n_ranks // 2) for k in range(n_ranks)])
    return {
        rank: (rng.standard_normal(dim) * scales[rank]) for rank in range(n_ranks)
    }


def test_tree_reducer_permutation_invariant() -> None:
    n_ranks, dim = 7, 8
    partials = _make_partials(n_ranks, dim, seed=3)

    reducer = TreeReducer(range(n_ranks))
    ref = reducer.reduce(partials)

    rng = np.random.default_rng(123)
    for _ in range(8):
        order = list(rng.permutation(n_ranks))
        permuted = {rank: partials[rank] for rank in order}
        red = TreeReducer(range(n_ranks))
        got = red.reduce(permuted)
        # BIT-identical regardless of arrival order (fixed fold topology).
        assert np.array_equal(got, ref)


def test_tree_reducer_matches_fixed_fold_order() -> None:
    # Cross-check against an explicit evaluation of the same fixed tree.
    n_ranks, dim = 5, 4
    partials = _make_partials(n_ranks, dim, seed=4)
    tree = build_reduction_tree(range(n_ranks))

    def eval_tree(t) -> np.ndarray:
        if t.is_leaf():
            return np.array(partials[t.rank_id], dtype=np.float64)
        return eval_tree(t.left) + eval_tree(t.right)

    ref = eval_tree(tree)
    got = TreeReducer(range(n_ranks)).reduce(partials)
    assert np.array_equal(got, ref)


def test_tree_reducer_repeated_calls_never_reuse_previous_input() -> None:
    reducer = TreeReducer([0, 1])
    first = reducer.reduce({0: np.array([1.0]), 1: np.array([2.0])})
    second = reducer.reduce({0: np.array([10.0]), 1: np.array([20.0])})

    np.testing.assert_array_equal(first, np.array([3.0]))
    np.testing.assert_array_equal(second, np.array([30.0]))


def test_tree_reducer_resumes_from_checkpoint_bitexact() -> None:
    n_ranks, dim = 6, 5
    partials = _make_partials(n_ranks, dim, seed=9)

    full = TreeReducer(range(n_ranks))
    ref = full.reduce(partials)

    # Checkpoint state belongs to an explicit immutable-input session, never to
    # the reusable TreeReducer topology object.
    first = TreeReducer(range(n_ranks)).start_session(partials, job_id="fit-9")
    first.reduce()
    ckpt = json.loads(json.dumps(first.checkpoint()))

    resumed = TreeReducer(range(n_ranks)).start_session(partials, job_id="fit-9")
    resumed.restore(ckpt)
    got = resumed.reduce()
    assert np.array_equal(got, ref)

    # Topology guard: a checkpoint for a different rank set is rejected.
    other_partials = {**partials, n_ranks: np.zeros(dim)}
    other = TreeReducer(range(n_ranks + 1)).start_session(
        other_partials, job_id="fit-9"
    )
    try:
        other.restore(ckpt)
    except ValueError:
        pass
    else:  # pragma: no cover - guard must trip
        raise AssertionError("expected a topology-mismatch ValueError on restore")


def test_tree_reducer_checkpoint_rejects_cross_job_and_cross_input_restore() -> None:
    reducer = TreeReducer([0, 1])
    partials = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}
    session = reducer.start_session(partials, job_id="job-a")
    session.reduce()
    checkpoint = session.checkpoint()

    different_job = reducer.start_session(partials, job_id="job-b")
    with pytest.raises(ValueError, match="job mismatch"):
        different_job.restore(checkpoint)

    different_partials = {0: np.array([10.0, 20.0]), 1: np.array([30.0, 40.0])}
    different_input = reducer.start_session(different_partials, job_id="job-a")
    with pytest.raises(ValueError, match="input fingerprint mismatch"):
        different_input.restore(checkpoint)


def test_tree_reducer_combiner_default_is_sum() -> None:
    partials = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0]), 2: np.array([5.0, 6.0])}
    got = TreeReducer([0, 1, 2]).reduce(partials)
    np.testing.assert_array_equal(got, np.array([9.0, 12.0]))


@pytest.mark.parametrize("bad_rank", [True, 1.0, 1.5, "1"])
def test_tree_reducer_rejects_lossy_rank_coercions(bad_rank: object) -> None:
    with pytest.raises(TypeError, match="rank ids must be integers"):
        TreeReducer([0, bad_rank])  # type: ignore[list-item]

    reducer = TreeReducer([0, 1])
    with pytest.raises(TypeError, match="rank ids must be integers"):
        reducer.reduce({bad_rank: np.array([1.0]), 1: np.array([2.0])})  # type: ignore[dict-item]


def test_tree_reducer_accepts_numpy_integral_rank_ids() -> None:
    reducer = TreeReducer([np.int64(0), np.int32(1)])
    got = reducer.reduce(
        {np.int16(0): np.array([1.0]), np.int64(1): np.array([2.0])}
    )
    np.testing.assert_array_equal(got, np.array([3.0]))


def test_tree_reducer_checkpoint_is_little_endian_and_nan_payload_exact() -> None:
    bits = np.array(
        [0x7FF8000000000042, 0x8000000000000000, 0x7FF0000000000000],
        dtype=np.uint64,
    )
    little = bits.view(np.float64)
    big = little.astype(np.dtype(">f8"))

    little_session = TreeReducer([0]).start_session({0: little}, job_id="bits")
    big_session = TreeReducer([0]).start_session({0: big}, job_id="bits")
    assert little_session.input_fingerprint == big_session.input_fingerprint

    result = little_session.reduce()
    checkpoint = little_session.checkpoint()
    assert checkpoint["entries"][0]["dtype"] == "<f8"
    assert result.view(np.uint64).tolist() == bits.tolist()

    resumed = TreeReducer([0]).start_session({0: little}, job_id="bits")
    resumed.restore(json.loads(json.dumps(checkpoint)))
    assert resumed.reduce().view(np.uint64).tolist() == bits.tolist()

    different_payload = bits.copy()
    different_payload[0] = np.uint64(0x7FF8000000000043)
    different = TreeReducer([0]).start_session(
        {0: different_payload.view(np.float64)}, job_id="bits"
    )
    assert different.input_fingerprint != little_session.input_fingerprint


def test_tree_reducer_isolates_mutating_and_aliased_combiner_outputs() -> None:
    retained_arguments: list[np.ndarray] = []

    def destructive_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left += right
        retained_arguments.append(left)
        return left

    reducer = TreeReducer(
        [0, 1, 2], combine=destructive_add, combine_id="destructive-add.v1"
    )
    partials = {
        0: np.array([1.0]),
        1: np.array([2.0]),
        2: np.array([3.0]),
    }
    session = reducer.start_session(partials, job_id="owned-cache")
    np.testing.assert_array_equal(session.reduce(), np.array([6.0]))

    # User code retained the exact arrays it mutated and returned. Those must
    # be working copies, never arrays owned by the session cache.
    for retained in retained_arguments:
        retained.fill(999.0)
    np.testing.assert_array_equal(session.reduce(), np.array([6.0]))

    resumed = reducer.start_session(partials, job_id="owned-cache")
    resumed.restore(json.loads(json.dumps(session.checkpoint())))
    np.testing.assert_array_equal(resumed.reduce(), np.array([6.0]))


def test_tree_reducer_restore_rejects_semantically_forged_internal_cache() -> None:
    reducer = TreeReducer([0, 1])
    partials = {0: np.array([1.0]), 1: np.array([2.0])}
    source = reducer.start_session(partials, job_id="semantic-validation")
    source.reduce()

    # Produce a self-consistent SHA-256 checkpoint around an incorrect root.
    # The checksum alone cannot establish that a subtree is the result of the
    # declared combiner.
    source._cache[source.tree.signature()] = np.array([1234.0])
    forged = source.checkpoint()

    target = reducer.start_session(partials, job_id="semantic-validation")
    np.testing.assert_array_equal(target.reduce(), np.array([3.0]))
    before = target.checkpoint()
    with pytest.raises(ValueError, match="does not match deterministic reduction"):
        target.restore(forged)
    # Restore is transactional: a failed attempt cannot poison existing state.
    assert target.checkpoint() == before
    np.testing.assert_array_equal(target.reduce(), np.array([3.0]))


def test_tree_reducer_restore_enforces_strict_json_checkpoint_shape() -> None:
    reducer = TreeReducer([0])
    partials = {0: np.array([1.0])}
    source = reducer.start_session(partials, job_id="strict-json")
    source.reduce()
    checkpoint = source.checkpoint()

    extra_top_level = dict(checkpoint)
    extra_top_level["ignored"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        reducer.start_session(partials, job_id="strict-json").restore(extra_top_level)

    malformed_entry = json.loads(json.dumps(checkpoint))
    malformed_entry["entries"][0]["shape"] = [True]
    with pytest.raises(ValueError, match="shape must be a list of integers"):
        reducer.start_session(partials, job_id="strict-json").restore(malformed_entry)

    wrong_dtype = json.loads(json.dumps(checkpoint))
    wrong_dtype["entries"][0]["dtype"] = ">f8"
    with pytest.raises(ValueError, match="dtype must be '<f8'"):
        reducer.start_session(partials, job_id="strict-json").restore(wrong_dtype)

    payload = json.dumps(checkpoint)
    duplicate_key_payload = payload.replace(
        '"schema":', '"schema": "duplicate", "schema":', 1
    )
    with pytest.raises(ValueError, match="malformed reduction checkpoint JSON"):
        reducer.start_session(partials, job_id="strict-json").restore_json(
            duplicate_key_payload
        )
