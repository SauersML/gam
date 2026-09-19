from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen_l17_exact_dose_replay.py"
SPEC = importlib.util.spec_from_file_location("qwen_l17_exact_dose_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def _protocol(row_count: int) -> dict[str, object]:
    return {
        "model": "Qwen3.6-35B-A3B",
        "model_revision": "frozen-revision",
        "model_sha256": "model-hash",
        "layer": 17,
        "hook_module": "model.layers.17",
        "steering_mode": "on_chart_amplitude_normalized",
        "model_dtype": "bfloat16",
        "harvest_dtype": "float32",
        "gam_sha": "gam-sha",
        "wheel_sha256": "wheel-hash",
        "driver_sha256": "driver-hash",
        "prompt_bank_sha256": "prompt-hash",
        "harvest_cache_sha256": "cache-hash",
        "seed": 0,
        "fractions": [0.01, 0.02, 0.1],
        "floor_multiplier": 30,
        "floor_repetitions": 5,
        "max_templates": 6,
        "bases": 10,
        "fit_iterations": 40,
        "row_count": row_count,
    }


def _row(
    identifier: str,
    *,
    atom: int,
    prompt: str,
    split: str,
    predicted: float,
    measured: float,
) -> dict[str, object]:
    return {
        "intervention_id": identifier,
        "atom": atom,
        "base_prompt_id": prompt,
        "split": split,
        "predicted_nats": predicted,
        "predicted_nats_kind": "exact_directional",
        "exact_directional_nats": predicted,
        "measured_nats": measured,
        "effective_delta": [predicted, -predicted],
        "resident_metric_nats": predicted * 0.4,
        "resident_metric_nats_kind": "uncertified_approximation",
    }


def _ledger() -> dict[str, object]:
    rows = [
        _row("c0", atom=0, prompt="c-a", split="calibration", predicted=0.005, measured=0.00505),
        _row("c1", atom=0, prompt="c-b", split="calibration", predicted=0.01, measured=0.0102),
        _row("c2", atom=0, prompt="c-e", split="calibration", predicted=0.02, measured=0.024),
        _row("c3", atom=1, prompt="c-c", split="calibration", predicted=0.01, measured=0.0101),
        _row("c4", atom=1, prompt="c-f", split="calibration", predicted=0.02, measured=0.0198),
        _row("c5", atom=1, prompt="c-d", split="calibration", predicted=0.04, measured=0.06),
        _row("h0", atom=0, prompt="same-index", split="heldout", predicted=0.008, measured=0.008),
        _row("h1", atom=0, prompt="other", split="heldout", predicted=0.01, measured=0.0101),
        _row("h2", atom=1, prompt="same-index", split="heldout", predicted=0.015, measured=0.0149),
        _row("h3", atom=1, prompt="third", split="heldout", predicted=0.02, measured=0.0201),
        _row("h4", atom=0, prompt="outside", split="heldout", predicted=0.015, measured=0.02),
        _row("h5", atom=0, prompt="below", split="heldout", predicted=0.002, measured=0.02),
        _row("h6", atom=1, prompt="below", split="heldout", predicted=0.004, measured=0.04),
    ]
    return {"protocol": _protocol(len(rows)), "rows": rows}


def test_acceptance_uses_calibration_radius_and_stable_atom_prompt_clusters() -> None:
    report = replay.acceptance_report(
        _ledger(), readout_tol_rel=0.1, bootstrap_draws=200, seed=2249
    )
    assert report["readout_region_nats_by_atom"] == {"0": [0.005, 0.01], "1": [0.01, 0.02]}
    assert report["row_counts"] == {
        "total": 13,
        "calibration": 6,
        "heldout": 7,
        "heldout_in_calibrated_region": 4,
        "heldout_outside_calibrated_region": 3,
    }
    assert report["included_intervention_ids"] == ["h0", "h1", "h2", "h3"]
    assert report["excluded_intervention_ids"] == ["h4", "h5", "h6"]
    assert report["r2_through_origin"] > 0.99
    # The repeated prompt token belongs to two atoms and therefore two clusters;
    # the old base-index-only key would collapse these into one.
    ci = replay._cluster_bootstrap_slope_ci(
        [row for row in replay._validate_ledger(_ledger()) if row["split"] == "heldout"],
        draws=20,
        seed=7,
    )
    assert len(ci) == 2


def test_acceptance_rejects_non_public_or_unstable_rows() -> None:
    ledger = _ledger()
    ledger["rows"][0]["predicted_nats_kind"] = "uncertified_approximation"
    with pytest.raises(ValueError, match="is not exact"):
        replay.acceptance_report(
            ledger, readout_tol_rel=0.1, bootstrap_draws=20, seed=1
        )

    ledger = _ledger()
    ledger["rows"][1]["intervention_id"] = ledger["rows"][0]["intervention_id"]
    with pytest.raises(ValueError, match="duplicate intervention_id"):
        replay.acceptance_report(
            ledger, readout_tol_rel=0.1, bootstrap_draws=20, seed=1
        )


def test_readout_radius_never_uses_heldout_measurements() -> None:
    ledger = _ledger()
    first = replay.acceptance_report(
        ledger, readout_tol_rel=0.1, bootstrap_draws=20, seed=1
    )
    ledger["rows"][-1]["measured_nats"] = 1000.0
    second = replay.acceptance_report(
        ledger, readout_tol_rel=0.1, bootstrap_draws=20, seed=1
    )
    assert first["readout_region_nats_by_atom"] == second["readout_region_nats_by_atom"]
    assert first["included_intervention_ids"] == second["included_intervention_ids"]


def test_heldout_rows_below_the_calibrated_region_are_not_certified() -> None:
    # h5 and h6 sit below every calibration dose of their atom, where no
    # calibration row measured the readout, and their measurements are ten times
    # their predictions. Scoring them as certified would let an unmeasured dose
    # range into the acceptance; they must be reported outside the region and must
    # not move the certified score by any amount.
    ledger = _ledger()
    report = replay.acceptance_report(ledger, readout_tol_rel=0.1, bootstrap_draws=50, seed=3)
    assert {"h5", "h6"} <= set(report["excluded_intervention_ids"])
    assert not {"h5", "h6"} & set(report["included_intervention_ids"])

    ledger["rows"] = [row for row in ledger["rows"] if row["intervention_id"] not in {"h5", "h6"}]
    ledger["protocol"]["row_count"] = len(ledger["rows"])
    without = replay.acceptance_report(ledger, readout_tol_rel=0.1, bootstrap_draws=50, seed=3)
    for key in ("slope_through_origin", "r2_through_origin", "slope_cluster_bootstrap_95_ci"):
        assert report[key] == without[key]


def test_one_failure_blocks_the_entire_equal_dose_calibration_stratum() -> None:
    ledger = _ledger()
    rows = ledger["rows"]
    rows.insert(
        1,
        _row(
            "c0-fail",
            atom=0,
            prompt="c-fail",
            split="calibration",
            predicted=0.005,
            measured=0.01,
        ),
    )
    ledger["protocol"]["row_count"] = len(rows)
    with pytest.raises(ValueError, match="atom 0 has no contiguous calibration dose"):
        replay.acceptance_report(
            ledger, readout_tol_rel=0.1, bootstrap_draws=20, seed=1
        )
