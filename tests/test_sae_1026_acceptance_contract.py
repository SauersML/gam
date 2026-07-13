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


def test_1026_fit_ev_reports_jsonable_hybrid_split(monkeypatch):
    driver = _load_driver_module()

    class FakeModel:
        atom_topologies = ["circle", "linear"]
        hybrid_split = {
            "curved_atom_count": np.int64(1),
            "linear_atom_count": np.int64(1),
            "atoms": [
                {
                    "atom": "atom_0",
                    "kept_curved": True,
                    "fitted_turning": np.float64(6.28),
                    "train_loao_delta_ev": np.float64(0.12),
                    "linear_image": {
                        "atom_idx": np.int64(0),
                        "b0": np.array([1.0, 0.0]),
                        "b1": np.array([0.0, 1.0]),
                        "t_bar": np.float64(0.5),
                    },
                }
            ],
        }

        def reconstruct(self, x):
            return np.asarray(x, dtype=float)

    fake_gamfit = types.ModuleType("gamfit")
    fake_gamfit.sae_manifold_fit = lambda *args, **kwargs: FakeModel()
    monkeypatch.setitem(sys.modules, "gamfit", fake_gamfit)

    x = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 1.0]], dtype=float)
    ev, fit_seconds, recon_seconds, split, topologies = driver._fit_ev(
        x,
        x,
        2,
        "circle",
        42,
        1,
        60.0,
    )

    assert ev == 1.0
    assert fit_seconds >= 0.0
    assert recon_seconds >= 0.0
    assert topologies == ["circle", "linear"]
    assert split["curved_atom_count"] == 1
    assert split["atoms"][0]["fitted_turning"] == 6.28
    assert split["atoms"][0]["linear_image"]["b0"] == [1.0, 0.0]
