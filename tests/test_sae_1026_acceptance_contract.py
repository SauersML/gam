from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_driver_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "sae_ev_vs_k_olmo.py"
    spec = importlib.util.spec_from_file_location("sae_ev_vs_k_olmo", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_1026_cpu_partial_ladder_cannot_close():
    driver = _load_driver_module()
    rows = [
        {"K": 8, "hybrid_ev_out": 0.90, "linear_ev_out": 0.55},
    ]
    report = driver._acceptance_report(
        rows,
        ladder=[1, 2, 4, 8],
        cpu_partial=True,
        official_reference_ev=driver.OFFICIAL_QWEN_W32K_EV,
    )

    assert report["closure_allowed"] is False
    assert "CPU partial" in report["blocker"]


def test_1026_full_ladder_must_clear_official_w32k_reference():
    driver = _load_driver_module()
    rows = [
        {"K": 8, "hybrid_ev_out": 0.40, "linear_ev_out": 0.41},
        {"K": 32, "hybrid_ev_out": 0.49, "linear_ev_out": 0.50},
        {"K": 128, "hybrid_ev_out": 0.521, "linear_ev_out": 0.522},
        {"K": 512, "hybrid_ev_out": 0.5229, "linear_ev_out": 0.522},
    ]

    report = driver._acceptance_report(
        rows,
        ladder=[8, 32, 128, 512],
        cpu_partial=False,
        official_reference_ev=driver.OFFICIAL_QWEN_W32K_EV,
    )

    assert report["closure_allowed"] is False
    assert "below parity bar" in report["blocker"]

    rows[-1]["hybrid_ev_out"] = 0.524
    report = driver._acceptance_report(
        rows,
        ladder=[8, 32, 128, 512],
        cpu_partial=False,
        official_reference_ev=driver.OFFICIAL_QWEN_W32K_EV,
    )

    assert report["closure_allowed"] is True
    assert report["blocker"] is None


def test_2267_route_selection_switches_at_realized_pca_width():
    driver = _load_driver_module()

    dense_curved = driver._arm_route(32, 32, "curved", 3)
    dense_linear = driver._arm_route(32, 32, "linear", 3)
    assert dense_curved["lane"] == driver.DENSE_MANIFOLD_LANE
    assert dense_linear["lane"] == driver.DENSE_MANIFOLD_LANE
    assert dense_curved["assignment"] == "softmax"
    assert dense_linear["assignment"] == "softmax"
    assert dense_curved["active_support"] is None
    assert dense_linear["active_support"] is None

    curved = driver._arm_route(128, 32, "curved", 3)
    linear = driver._arm_route(128, 32, "linear", 3)
    assert curved == {
        "arm": "curved",
        "lane": driver.CURVED_TOPK_LANE,
        "model_type": "ManifoldSAE",
        "certificate_type": "ManifoldSAE.certificates+termination",
        "assignment": "topk",
        "active_support": 3,
        "support_contract": "exact_selected_support_values_per_row",
        "precision": "float64",
    }
    assert linear == {
        "arm": "linear",
        "lane": driver.LINEAR_SPARSE_LANE,
        "model_type": "SparseDictionaryFit",
        "certificate_type": "SparseDictionaryConvergence",
        "assignment": "active_set_least_squares",
        "active_support": 3,
        "support_contract": "exact_selected_nonzero_codes_per_row",
        "precision": "float32",
    }


def test_2267_route_selection_never_clamps_active_support():
    driver = _load_driver_module()

    with np.testing.assert_raises_regex(
        ValueError, "active support s=9 cannot exceed dictionary width K=8"
    ):
        driver._arm_route(8, 32, "curved", 9)


def test_2267_overcomplete_curved_worker_passes_exact_topk(monkeypatch):
    driver = _load_driver_module()
    calls = []

    class FakeManifoldSAE:
        chosen_k = 3
        atom_topologies = ["circle"] * 3
        hybrid_split = {
            "curved_atom_count": np.int64(3),
            "linear_atom_count": np.int64(0),
        }
        certificates = {"fit": "certified"}
        termination = {"verdict": "converged"}

        def converged_latents(self, x):
            x = np.asarray(x, dtype=float)
            indices = np.tile(np.array([[0, 1]], dtype=np.uint32), (x.shape[0], 1))
            values = np.full((x.shape[0], 2), 0.5, dtype=float)
            return {
                "fitted": x.copy(),
                "support_indices": indices,
                "support_values": values,
            }

    def fake_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeManifoldSAE()

    fake_gamfit = types.ModuleType("gamfit")
    fake_gamfit.sae_manifold_fit = fake_fit
    monkeypatch.setitem(sys.modules, "gamfit", fake_gamfit)

    train = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]], dtype=float)
    test = np.array([[2.0, 1.0], [-2.0, 3.0]], dtype=float)
    result = driver._manifold_fit_worker(
        train,
        test,
        3,
        "curved",
        "topk",
        2,
        42,
        7,
        60.0,
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["K"] == 3
    assert kwargs["atom_topology"] == "circle"
    assert kwargs["assignment"] == "topk"
    assert kwargs["top_k"] == 2
    assert kwargs["n_iter"] == 7
    assert kwargs["run_structure_search"] is False
    assert kwargs["structured_residual_passes"] == 0
    assert result["test_ev"] == 1.0
    assert "assignments" not in FakeManifoldSAE().converged_latents(train)
    assert result["support_evidence"]["train"]["nonzero_min"] == 2
    assert result["support_evidence"]["held_out"]["nonzero_max"] == 2
    assert result["certificate"]["type"] == "ManifoldSAE.certificates+termination"
    assert result["hybrid_split"]["curved_atom_count"] == 3


def test_2267_overcomplete_linear_worker_uses_sparse_transform_then_decode(monkeypatch):
    driver = _load_driver_module()
    calls = []

    convergence_values = {
        "inner_ev_residual": 0.1,
        "inner_tolerance": 0.2,
        "decoder_residual": 0.1,
        "decoder_tolerance": 0.2,
        "routing_residual": 0.1,
        "routing_tolerance": 0.2,
        "outer_rho_residual": 0.1,
        "outer_tolerance": 0.2,
        "selected_rho": 1.0,
        "outer_iterations": 2,
    }

    class SparseDictionaryFit:
        decoder = np.zeros((3, 2), dtype=np.float32)
        active = 2
        score_route_stats = {"cpu_minibatches": 1}
        convergence = types.SimpleNamespace(**convergence_values)

        def transform(self, x, active, *, score_mode):
            calls.append(("transform", np.asarray(x).shape, active, score_mode))
            rows = np.asarray(x).shape[0]
            return types.SimpleNamespace(
                indices=np.tile(np.array([[0, 1]], dtype=np.uint32), (rows, 1)),
                codes=np.ones((rows, 2), dtype=np.float32),
                score_route_stats={"cpu_minibatches": 1},
            )

        def reconstruct(self, indices, codes):
            calls.append(("reconstruct", indices.shape, codes.shape))
            return np.zeros((indices.shape[0], 2), dtype=np.float32)

    def fake_fit(x, **kwargs):
        calls.append(("fit", np.asarray(x).shape, kwargs))
        return SparseDictionaryFit()

    fake_gamfit = types.ModuleType("gamfit")
    fake_gamfit.sparse_dictionary_fit = fake_fit
    monkeypatch.setitem(sys.modules, "gamfit", fake_gamfit)

    train = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]], dtype=float)
    test = np.array([[2.0, 1.0], [-2.0, 3.0]], dtype=float)
    result = driver._linear_sparse_fit_worker(train, test, 3, 2, 5, "off")

    assert calls[0] == (
        "fit",
        (3, 2),
        {"K": 3, "active": 2, "max_epochs": 5, "score_mode": "off"},
    )
    assert calls[1:3] == [
        ("transform", (3, 2), 2, "off"),
        ("transform", (2, 2), 2, "off"),
    ]
    assert calls[3:] == [
        ("reconstruct", (3, 2), (3, 2)),
        ("reconstruct", (2, 2), (2, 2)),
    ]
    assert np.isfinite(result["train_ev"])
    assert np.isfinite(result["test_ev"])
    assert result["support_evidence"]["train"]["selected_min"] == 2
    assert result["support_evidence"]["held_out"]["nonzero_max"] == 2
    assert result["certificate"] == {
        "type": "SparseDictionaryConvergence",
        "convergence": convergence_values,
    }


def test_2267_sparse_support_evidence_rejects_padded_zero_code():
    driver = _load_driver_module()

    with np.testing.assert_raises_regex(RuntimeError, "violated exact sparse support s=2"):
        driver._exact_sparse_support_summary(
            np.array([[0, 0]], dtype=np.uint32),
            np.array([[1.0, 0.0]], dtype=np.float32),
            rows=1,
            k=3,
            active_support=2,
            label="test",
        )
