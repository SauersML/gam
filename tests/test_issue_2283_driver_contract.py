"""Contract tests for the provenance-complete #2283 measurement driver."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_driver():
    experiment_dir = Path(__file__).resolve().parents[1] / "experiments" / "1026_close"
    sys.path.insert(0, str(experiment_dir))
    spec = importlib.util.spec_from_file_location(
        "issue_2283_driver", experiment_dir / "driver_1026_arms.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_fit_is_fail_closed_and_reports_every_route(monkeypatch):
    driver = _load_driver()
    calls = []

    @dataclasses.dataclass(frozen=True)
    class Convergence:
        inner_ev_residual: float = 0.0
        inner_tolerance: float = 1.0e-6
        decoder_residual: float = 0.0
        decoder_tolerance: float = 1.0e-6
        routing_residual: float = 0.0
        routing_tolerance: float = 1.0e-6
        outer_rho_residual: float = 0.0
        outer_tolerance: float = 1.0e-6
        selected_rho: float = 1.0
        outer_iterations: int = 2

    class Fit:
        decoder = np.eye(2, dtype=np.float32)
        explained_variance = 1.0
        score_route_stats = {"device_minibatches": 7, "cpu_minibatches": 0}
        convergence = Convergence()

        def transform(self, values, *, score_mode):
            calls.append(("transform", score_mode, np.asarray(values).shape))
            rows = np.asarray(values).shape[0]
            return types.SimpleNamespace(
                indices=np.zeros((rows, 1), dtype=np.uint32),
                codes=np.ones((rows, 1), dtype=np.float32),
                score_route_stats={"device_minibatches": 1, "cpu_minibatches": 0},
            )

        def reconstruct(self, indices, codes):
            return np.zeros((indices.shape[0], 2), dtype=np.float32)

    def sparse_dictionary_fit(values, k, **kwargs):
        calls.append(("fit", np.asarray(values).shape, k, kwargs))
        return Fit()

    fake_gamfit = types.ModuleType("gamfit")
    fake_gamfit.sparse_dictionary_fit = sparse_dictionary_fit
    monkeypatch.setitem(sys.modules, "gamfit", fake_gamfit)

    train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    test = np.array([[1.0, 1.0]], dtype=np.float32)
    collect = {}
    driver.fit_gam_flat(
        train,
        test,
        train.mean(axis=0),
        K=2,
        top_k=1,
        minibatch=8192,
        score_mode="required",
        max_epochs=30,
        collect=collect,
    )

    assert calls[0] == (
        "fit",
        (2, 2),
        2,
        {
            "active": 1,
            "minibatch": 8192,
            "max_epochs": 30,
            "score_mode": "required",
        },
    )
    assert calls[1] == ("transform", "required", (1, 2))
    assert collect["sparse_route_stats"] == {
        "fit": {"device_minibatches": 7, "cpu_minibatches": 0},
        "held_out": {"device_minibatches": 1, "cpu_minibatches": 0},
    }
    assert collect["sparse_convergence"] == dataclasses.asdict(Convergence())


def test_row_identity_hashes_follow_the_exact_split():
    driver = _load_driver()
    values = np.arange(24, dtype=np.float32).reshape(12, 2)
    row_ids = np.arange(100, 112, dtype=np.int64)

    train, test, train_ids, test_ids = driver.make_split(values, row_ids, 0.25, 7)

    lookup = {tuple(row): int(row_id) for row, row_id in zip(values, row_ids, strict=True)}
    assert [lookup[tuple(row)] for row in train] == train_ids.tolist()
    assert [lookup[tuple(row)] for row in test] == test_ids.tolist()
    assert driver._array_sha256(train_ids) != driver._array_sha256(test_ids)
    assert driver._array_sha256(train_ids) == driver._array_sha256(train_ids.copy())


def test_measurement_identity_rejects_abbreviated_or_noncanonical_digests():
    driver = _load_driver()
    good_git = "a" * 40
    good_wheel = "b" * 64
    driver._validate_measurement_identity("issue2283-seed0", good_git, good_wheel)

    for run_id, git_sha, wheel_sha in (
        ("", good_git, good_wheel),
        ("run", "a" * 12, good_wheel),
        ("run", good_git.upper(), good_wheel),
        ("run", good_git, "b" * 63),
    ):
        with np.testing.assert_raises(ValueError):
            driver._validate_measurement_identity(run_id, git_sha, wheel_sha)


def test_flat_checkpoint_round_trip_is_manifested_and_pair_bound(tmp_path):
    driver = _load_driver()
    arrays = {
        "decoder": np.eye(2, dtype=np.float32),
        "train_indices": np.zeros((2, 1), dtype=np.uint32),
        "train_codes": np.ones((2, 1), dtype=np.float32),
        "held_out_indices": np.ones((1, 1), dtype=np.uint32),
        "held_out_codes": -np.ones((1, 1), dtype=np.float32),
        "train_reconstruction": np.eye(2, dtype=np.float32),
        "held_out_reconstruction": np.ones((1, 2), dtype=np.float32),
    }
    pair = {"schema": driver.PAIR_SCHEMA, "run_id": "issue2283-seed0"}
    config = {"K_flat": 2, "score_mode": "required"}
    metadata = {
        "pair_identity": pair,
        "flat_config": config,
        "route_stats": {},
        "convergence": {},
        "explained_variance": 1.0,
        "held_out_ev": 1.0,
    }
    path = tmp_path / "flat.npz"

    digest = driver._write_flat_checkpoint(str(path), arrays, metadata)
    loaded, loaded_metadata, loaded_digest = driver._read_flat_checkpoint(
        str(path), pair, config
    )

    assert digest == loaded_digest == driver._file_sha256(path)
    assert loaded_metadata["schema"] == driver.FLAT_CHECKPOINT_SCHEMA
    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded[name], expected)
    with np.testing.assert_raises_regex(ValueError, "configuration"):
        driver._read_flat_checkpoint(str(path), pair, {"K_flat": 3})


def test_required_route_certificate_rejects_any_cpu_minibatch():
    driver = _load_driver()
    driver._assert_required_device_routes(
        {
            "fit": {
                "minibatches": 3,
                "admitted_minibatches": 3,
                "device_minibatches": 3,
                "cpu_minibatches": 0,
            }
        }
    )
    with np.testing.assert_raises_regex(RuntimeError, "not wholly device-resident"):
        driver._assert_required_device_routes(
            {
                "fit": {
                    "minibatches": 3,
                    "admitted_minibatches": 3,
                    "device_minibatches": 2,
                    "cpu_minibatches": 1,
                }
            }
        )
