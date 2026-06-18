from __future__ import annotations

import importlib.util
from pathlib import Path


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
        {"K": 8, "curved_ev_out": 0.90, "hybrid_ev_out": 0.90, "linear_ev_out": 0.55},
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
        {"K": 8, "curved_ev_out": 0.40, "hybrid_ev_out": 0.40, "linear_ev_out": 0.41},
        {"K": 32, "curved_ev_out": 0.49, "hybrid_ev_out": 0.49, "linear_ev_out": 0.50},
        {"K": 128, "curved_ev_out": 0.521, "hybrid_ev_out": 0.521, "linear_ev_out": 0.522},
        {"K": 512, "curved_ev_out": 0.5229, "hybrid_ev_out": 0.5229, "linear_ev_out": 0.522},
    ]

    report = driver._acceptance_report(
        rows,
        ladder=[8, 32, 128, 512],
        cpu_partial=False,
        official_reference_ev=driver.OFFICIAL_QWEN_W32K_EV,
    )

    assert report["closure_allowed"] is False
    assert "below parity bar" in report["blocker"]

    rows[-1]["curved_ev_out"] = 0.524
    rows[-1]["hybrid_ev_out"] = 0.524
    report = driver._acceptance_report(
        rows,
        ladder=[8, 32, 128, 512],
        cpu_partial=False,
        official_reference_ev=driver.OFFICIAL_QWEN_W32K_EV,
    )

    assert report["closure_allowed"] is True
    assert report["blocker"] is None
