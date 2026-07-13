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
