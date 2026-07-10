from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "audit_sae"
    / "fixed_budget_probe_2251.py"
)
SPEC = importlib.util.spec_from_file_location("fixed_budget_probe_2251", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)

CANDIDATE_SCRIPT = SCRIPT.with_name("build_fixed_budget_candidate_2251.py")
CANDIDATE_SPEC = importlib.util.spec_from_file_location(
    "build_fixed_budget_candidate_2251", CANDIDATE_SCRIPT
)
assert CANDIDATE_SPEC is not None and CANDIDATE_SPEC.loader is not None
candidate_builder = importlib.util.module_from_spec(CANDIDATE_SPEC)
sys.modules[CANDIDATE_SPEC.name] = candidate_builder
CANDIDATE_SPEC.loader.exec_module(candidate_builder)


def fixture_archive(path: Path, *, active: int = 64) -> None:
    rng = np.random.default_rng(2251)
    n_documents, n_classes, width = 90, 3, 128
    labels = np.repeat(np.arange(n_classes), n_documents // n_classes)
    train = np.concatenate([np.arange(c * 30, c * 30 + 20) for c in range(n_classes)])
    test = np.concatenate([np.arange(c * 30 + 20, c * 30 + 30) for c in range(n_classes)])

    scalar = rng.normal(scale=1.0, size=(n_documents, width))
    candidate = rng.normal(scale=0.25, size=(n_documents, width))
    raw = rng.normal(scale=0.25, size=(n_documents, 12))
    for row, label in enumerate(labels):
        scalar[row, label] += 1.5
        candidate[row, label] += 4.0
        raw[row, label] += 4.0

    tokens_per_document = 2
    n_tokens = n_documents * tokens_per_document
    offsets = np.arange(0, n_tokens + 1, tokens_per_document, dtype=np.int64)
    scalar_indices = np.empty((n_tokens, 64), dtype=np.uint32)
    for row in range(n_tokens):
        scalar_indices[row] = (np.arange(64) + row) % width
    scalar_values = np.ones_like(scalar_indices, dtype=np.float32)

    block_indices = np.empty((n_tokens, 16), dtype=np.uint32)
    for row in range(n_tokens):
        block_indices[row] = (np.arange(16) + row) % (width // 4)
    block_values = np.ones((n_tokens, 16, 4), dtype=np.float32)

    np.savez(
        path,
        labels=labels,
        train_indices=train,
        test_indices=test,
        document_offsets=offsets,
        reps__raw=raw,
        reps__scalar=scalar,
        reps__block=candidate,
        active__scalar=np.asarray([active]),
        n_atoms__scalar=np.asarray([width]),
        block_size__scalar=np.asarray([1]),
        route_indices__scalar=scalar_indices,
        route_values__scalar=scalar_values,
        active__block=np.asarray([active]),
        n_atoms__block=np.asarray([width]),
        block_size__block=np.asarray([4]),
        route_indices__block=block_indices,
        route_values__block=block_values,
    )


def test_fixed_budget_archive_and_route_diagnostics_pin_64_scalars(tmp_path: Path) -> None:
    archive = tmp_path / "probe.npz"
    fixture_archive(archive)
    labels, train, test, _offsets, arms = probe.load_arms(archive)
    by_name = {arm.name: arm for arm in arms}
    block = by_name["block"]
    scalar = by_name["scalar"]

    assert block.n_atoms == scalar.n_atoms == 128
    assert block.active == scalar.active == 64
    assert block.route_indices.shape[1] * block.block_size == 64
    assert scalar.route_indices.shape[1] * scalar.block_size == 64

    primary, selected = probe.evaluate_once(block, labels, train, test, None, seed=0)
    diagnostics = probe.route_diagnostics(block, selected)
    assert primary["accuracy"] >= 0.95
    assert primary["macro_auc"] >= 0.99
    assert diagnostics["structural_scalar_budget"] == 64
    assert diagnostics["max_nonzero_scalar_coordinates"] == 64
    assert diagnostics["mean_nonzero_scalar_coordinates"] == 64.0
    assert len(primary["per_class"]) == 3


def test_archive_refuses_budget_rounding(tmp_path: Path) -> None:
    archive = tmp_path / "bad_probe.npz"
    fixture_archive(archive, active=63)
    with pytest.raises(ValueError, match="requires exactly 64"):
        probe.load_arms(archive)


def test_sparse_block_pooling_preserves_signed_coordinates() -> None:
    blocks = np.asarray([[0], [1], [0]], dtype=np.uint32)
    codes = np.asarray([[[2.0, -1.0]], [[3.0, 4.0]], [[-2.0, 1.0]]], dtype=np.float32)
    offsets = np.asarray([0, 2, 3], dtype=np.int64)
    pooled = candidate_builder.mean_pool_sparse_block_route(
        blocks, codes, offsets, n_atoms=4, block_size=2
    )
    np.testing.assert_allclose(pooled[0], [1.0, -0.5, 1.5, 2.0])
    np.testing.assert_allclose(pooled[1], [-2.0, 1.0, 0.0, 0.0])
