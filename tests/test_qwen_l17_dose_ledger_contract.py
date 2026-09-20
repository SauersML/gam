"""Numpy-only contract gates for the #2263 gate-2 frozen dose ledger driver."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DRIVER = _load("qwen_l17_dose_ledger", "scripts/qwen_l17_dose_ledger.py")
SCORER = _load("qwen_l17_exact_dose_replay", "scripts/qwen_l17_exact_dose_replay.py")


def test_the_frozen_settings_are_the_historical_protocol() -> None:
    protocol = DRIVER.FROZEN_PROTOCOL
    assert protocol["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert protocol["layer"] == 17
    assert (protocol["model_dtype"], protocol["harvest_dtype"]) == ("bfloat16", "float32")
    assert protocol["floor_repetitions"] == 5
    # The historical DOSE_FLOOR_MULT=30 multiplied repeated-forward KLs that are exactly
    # 0 on a deterministic model; each record is floored at its derived band instead.
    assert "floor_multiplier" not in protocol
    assert (protocol["max_templates"], protocol["bases"], protocol["fit_iterations"]) == (6, 10, 40)
    fractions = np.asarray(protocol["fractions"])
    assert fractions.size == 10
    assert fractions[0] == pytest.approx(0.01, rel=1e-12)
    assert fractions[-1] == pytest.approx(0.6, rel=1e-12)
    ratios = fractions[1:] / fractions[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)


def test_every_feature_fills_its_fit_cloud_and_base_pool() -> None:
    protocol = DRIVER.FROZEN_PROTOCOL
    tasks = DRIVER.gate2_tasks(protocol["features"])
    assert [task.name for task in tasks] == ["weekday", "month", "color"]
    fit_rows = {
        task.name: len(task.labels) * min(len(task.train_templates), protocol["max_templates"])
        for task in tasks
    }
    # The historical harvest caches were weekday n42, month n72 and color n48.
    assert fit_rows == {"weekday": 42, "month": 72, "color": 48}
    for task in tasks:
        assert len(task.train_templates) >= protocol["max_templates"]
        assert len(task.eval_templates) * len(task.labels) >= protocol["bases"]
        assert not set(task.train_templates) & set(task.eval_templates)
        for template in task.train_templates + task.eval_templates:
            assert template.count("{label}") == 1
            assert " {label}" in template


def test_the_base_split_is_seeded_distinct_and_half_calibration() -> None:
    first = DRIVER.base_split(21, 10, 0)
    assert first == DRIVER.base_split(21, 10, 0)
    indices = [index for index, _ in first]
    assert len(set(indices)) == 10
    assert all(0 <= index < 21 for index in indices)
    assert [split for _, split in first] == ["calibration"] * 5 + ["heldout"] * 5
    with pytest.raises(ValueError):
        DRIVER.base_split(7, 10, 0)


def test_every_ladder_target_is_requested_and_controls_only_raise_floors() -> None:
    fractions = (0.001, 0.005, 0.01)
    assert DRIVER.dose_ladder(1e-2, fractions) == [(k, 1e-2 * f) for k, f in enumerate(fractions)]
    # Deterministic repeats measure 0, or a KL negative by roundoff: no evidence above
    # the band. Stochastic repeats raise floors by their largest KL.
    assert DRIVER.control_evidence_nats([0.0] * 5) == 0.0
    assert DRIVER.control_evidence_nats([0.0, 2e-6, -1e-18]) == 2e-6
    with pytest.raises(ValueError):
        DRIVER.control_evidence_nats([])
    with pytest.raises(ValueError):
        DRIVER.control_evidence_nats([0.0, float("nan")])


def test_router_topk_changes_count_token_rows_whose_expert_set_moved() -> None:
    base = [np.asarray([[1, 2], [3, 4]]), np.asarray([[5, 6], [7, 8]])]
    # Call 0: row 0 keeps {1, 2} in another order, row 1 trades expert 4 for 5.
    # Call 1: both rows keep their sets. One row in total changed its set.
    patched = [np.asarray([[2, 1], [3, 5]]), np.asarray([[5, 6], [8, 7]])]
    assert DRIVER.router_topk_changes(base, patched) == 1
    assert DRIVER.router_topk_changes(base, base) == 0
    assert DRIVER.router_topk_changes([], []) == 0
    with pytest.raises(ValueError, match="router calls"):
        DRIVER.router_topk_changes(base, patched[:1])
    with pytest.raises(ValueError, match="shapes differ"):
        DRIVER.router_topk_changes(base, [patched[0][:1], patched[1]])


def _plan(predicted: float, measured: float) -> dict[str, object]:
    return {
        "predicted_nats": predicted,
        "predicted_nats_kind": "exact_directional",
        "exact_directional_nats": predicted,
        "measured_nats": measured,
        "effective_delta": [predicted, -predicted],
        "resident_metric_nats": 0.4 * predicted,
        "resident_metric_nats_kind": "uncertified_approximation",
        "iterations": 3,
        "displacement": 0.1,
    }


def _observation(measured: float) -> dict[str, object]:
    return {
        "router_topk_changes": 0,
        "logit_format": "bfloat16",
        "logit_max_abs": 30.0,
        "logit_max_abs_change": 0.25,
        "evaluation_band_nats": 1e-12,
        "measurement_band_nats": 0.25 * measured,
        "floor_nats": 0.25 * measured,
    }


def test_the_assembled_ledger_is_what_the_scorer_accepts() -> None:
    rows = {
        feature: [
            DRIVER.ledger_row(
                _plan(0.01 * (k + 1), 0.0101 * (k + 1)),
                feature=feature,
                atom=atom,
                base_prompt_id=f"{feature}-e{b}-l0",
                split=split,
                fraction_index=k,
                target_nats=0.01 * (k + 1),
                observation=_observation(0.0101 * (k + 1)),
            )
            for b, split in enumerate(("calibration", "heldout"))
            for k in range(3)
        ]
        for atom, feature in enumerate(("weekday", "month"))
    }
    protocol = {field: 0 for field in SCORER._PROTOCOL_FIELDS if field != "row_count"}
    protocol["features"] = ["weekday", "month"]
    ledger = DRIVER.assemble_ledger(protocol, rows)
    assert ledger["protocol"]["row_count"] == 12
    validated = SCORER._validate_ledger(ledger)
    assert len({row["intervention_id"] for row in validated}) == 12
    assert {row["predicted_nats_kind"] for row in validated} == {"exact_directional"}


def test_snapshot_identity_follows_the_blob_names(tmp_path: Path) -> None:
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "aaa").write_bytes(b"weights")
    (blobs / "bbb").write_bytes(b"weights")
    snapshot = tmp_path / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    os.symlink(blobs / "aaa", snapshot / "model.safetensors")
    revision, first = DRIVER.snapshot_identity(snapshot)
    assert revision == "rev1"
    assert DRIVER.snapshot_identity(snapshot)[1] == first
    (snapshot / "model.safetensors").unlink()
    os.symlink(blobs / "bbb", snapshot / "model.safetensors")
    assert DRIVER.snapshot_identity(snapshot)[1] != first
