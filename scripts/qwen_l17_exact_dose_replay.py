#!/usr/bin/env python3
"""Score a frozen Qwen layer-17 ledger emitted by the public steering API.

The producer must call ``ManifoldSAE.steer`` / ``steer_to_target`` and persist
the returned plan together with its plan-aware applied-dose observation. This
scorer never re-computes or overwrites ``predicted_nats``. It accepts only an
``exact_full`` or ``exact_directional`` public prediction, learns the readout-KL
radius exclusively from calibration prompts, freezes that radius, and scores
held-out prompts below it.

The historical private-driver monkeypatch is intentionally gone. A ledger that
does not carry strict public-plan values, stable intervention identifiers, and
complete frozen-protocol provenance is not accepted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np


_EXACT_KINDS = frozenset({"exact_full", "exact_directional"})
_RESIDENT_KINDS = frozenset(
    {"exact_full", "certified_psd_lower_bound", "uncertified_approximation"}
)
_PROTOCOL_FIELDS = (
    "model",
    "model_revision",
    "model_sha256",
    "layer",
    "hook_module",
    "steering_mode",
    "model_dtype",
    "harvest_dtype",
    "gam_sha",
    "wheel_sha256",
    "driver_sha256",
    "prompt_bank_sha256",
    "harvest_cache_sha256",
    "seed",
    "fractions",
    "floor_multiplier",
    "floor_repetitions",
    "max_templates",
    "bases",
    "fit_iterations",
    "row_count",
)


def _finite_nonnegative(value: Any, *, field: str, intervention_id: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"intervention {intervention_id!r} field {field!r} must be finite and "
            f"non-negative; got {value!r}"
        )
    return number


def _validate_ledger(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = ledger.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("ledger must contain a protocol mapping")
    missing_protocol = [field for field in _PROTOCOL_FIELDS if field not in protocol]
    if missing_protocol:
        raise ValueError(f"protocol is missing required fields {missing_protocol}")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ledger rows must be a non-empty list")
    if int(protocol["row_count"]) != len(rows):
        raise ValueError(
            f"protocol row_count={protocol['row_count']} does not match {len(rows)} rows"
        )

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("every ledger row must be a mapping")
        intervention_id = str(raw.get("intervention_id", ""))
        if not intervention_id:
            raise ValueError("every ledger row needs a non-empty intervention_id")
        if intervention_id in seen:
            raise ValueError(f"duplicate intervention_id {intervention_id!r}")
        seen.add(intervention_id)

        split = raw.get("split")
        if split not in ("calibration", "heldout"):
            raise ValueError(
                f"intervention {intervention_id!r} split must be 'calibration' or 'heldout'"
            )
        atom = int(raw["atom"])
        base_prompt_id = str(raw.get("base_prompt_id", ""))
        if not base_prompt_id:
            raise ValueError(
                f"intervention {intervention_id!r} needs a stable base_prompt_id"
            )
        kind = str(raw.get("predicted_nats_kind", ""))
        if kind not in _EXACT_KINDS:
            raise ValueError(
                f"intervention {intervention_id!r} public prediction kind {kind!r} is not exact"
            )
        resident_kind = str(raw.get("resident_metric_nats_kind", kind))
        if resident_kind not in _RESIDENT_KINDS:
            raise ValueError(
                f"intervention {intervention_id!r} has invalid resident factor kind "
                f"{resident_kind!r}"
            )
        predicted = _finite_nonnegative(
            raw["predicted_nats"], field="predicted_nats", intervention_id=intervention_id
        )
        measured = _finite_nonnegative(
            raw["measured_nats"], field="measured_nats", intervention_id=intervention_id
        )
        exact = _finite_nonnegative(
            raw.get("exact_directional_nats", predicted),
            field="exact_directional_nats",
            intervention_id=intervention_id,
        )
        if kind == "exact_directional" and predicted != exact:
            raise ValueError(
                f"intervention {intervention_id!r} predicted_nats must be the public "
                "exact_directional_nats value"
            )
        effective_delta = raw.get("effective_delta")
        if not isinstance(effective_delta, list) or not effective_delta:
            raise ValueError(
                f"intervention {intervention_id!r} needs a non-empty effective_delta list"
            )
        if not all(math.isfinite(float(value)) for value in effective_delta):
            raise ValueError(
                f"intervention {intervention_id!r} effective_delta must be finite"
            )
        row = dict(raw)
        row.update(
            intervention_id=intervention_id,
            split=split,
            atom=atom,
            base_prompt_id=base_prompt_id,
            predicted_nats=predicted,
            measured_nats=measured,
            exact_directional_nats=exact,
            predicted_nats_kind=kind,
            resident_metric_nats_kind=resident_kind,
        )
        validated.append(row)
    return validated


def _relative_readout_error(row: dict[str, Any]) -> float:
    predicted = float(row["predicted_nats"])
    measured = float(row["measured_nats"])
    if predicted == 0.0:
        return 0.0 if measured == 0.0 else math.inf
    return abs(measured - predicted) / predicted


def _calibrate_readout_radii(
    rows: list[dict[str, Any]], *, tolerance: float
) -> dict[int, float]:
    if not math.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("readout tolerance must be a finite fraction in [0,1)")
    calibration = [row for row in rows if row["split"] == "calibration"]
    if not calibration:
        raise ValueError("readout-radius calibration needs calibration rows")
    atoms = sorted({int(row["atom"]) for row in rows})
    radii: dict[int, float] = {}
    for atom in atoms:
        ordered = sorted(
            (row for row in calibration if int(row["atom"]) == atom),
            key=lambda row: (float(row["predicted_nats"]), row["intervention_id"]),
        )
        radius: float | None = None
        cursor = 0
        while cursor < len(ordered):
            dose = float(ordered[cursor]["predicted_nats"])
            end = cursor + 1
            while end < len(ordered) and float(ordered[end]["predicted_nats"]) == dose:
                end += 1
            # A radius certifies the whole calibration stratum at this dose. One
            # failing prompt prevents another prompt at the same dose from
            # extending the boundary by sort-order accident.
            if any(
                _relative_readout_error(row) > tolerance
                for row in ordered[cursor:end]
            ):
                break
            radius = dose
            cursor = end
        if radius is None:
            raise ValueError(
                f"atom {atom} has no contiguous calibration dose inside readout tolerance"
            )
        radii[atom] = radius
    return radii


def _score(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("dose score needs at least one row")
    predicted = np.asarray([row["predicted_nats"] for row in rows], dtype=np.float64)
    measured = np.asarray([row["measured_nats"] for row in rows], dtype=np.float64)
    x2 = float(predicted @ predicted)
    y2 = float(measured @ measured)
    if not (x2 > 0.0 and y2 > 0.0):
        raise ValueError("dose score needs non-zero prediction and measurement")
    slope = float((predicted @ measured) / x2)
    residual = measured - slope * predicted
    return {
        "n": len(rows),
        "slope_through_origin": slope,
        "r2_through_origin": float(1.0 - (residual @ residual) / y2),
    }


def _cluster_bootstrap_slope_ci(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> list[float]:
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    clusters: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["atom"]), str(row["base_prompt_id"]))
        clusters.setdefault(key, []).append(row)
    keys = sorted(clusters)
    if len(keys) < 2:
        raise ValueError("cluster bootstrap needs at least two atom/prompt clusters")
    rng = np.random.Generator(np.random.PCG64(seed))
    slopes = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(keys), size=len(keys))
        replay = [row for index in sampled for row in clusters[keys[int(index)]]]
        slopes[draw] = float(_score(replay)["slope_through_origin"])
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return [float(lo), float(hi)]


def _group_report(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return {
        name: _score(group) if len(group) >= 3 else {"n": len(group)}
        for name, group in sorted(groups.items())
    }


def acceptance_report(
    ledger: dict[str, Any], *, readout_tol_rel: float, bootstrap_draws: int, seed: int
) -> dict[str, Any]:
    rows = _validate_ledger(ledger)
    radii = _calibrate_readout_radii(rows, tolerance=readout_tol_rel)
    heldout = [row for row in rows if row["split"] == "heldout"]
    included = [
        row
        for row in heldout
        if float(row["predicted_nats"]) <= radii[int(row["atom"])]
    ]
    excluded = [row for row in heldout if row not in included]
    report: dict[str, Any] = dict(_score(included))
    ci = _cluster_bootstrap_slope_ci(included, draws=bootstrap_draws, seed=seed)
    report.update(
        slope_cluster_bootstrap_95_ci=ci,
        readout_tol_rel=readout_tol_rel,
        readout_radius_nats_by_atom={str(key): value for key, value in radii.items()},
        row_counts={
            "total": len(rows),
            "calibration": sum(row["split"] == "calibration" for row in rows),
            "heldout": len(heldout),
            "heldout_in_readout_radius": len(included),
            "heldout_outside_readout_radius": len(excluded),
        },
        included_intervention_ids=[row["intervention_id"] for row in included],
        excluded_intervention_ids=[row["intervention_id"] for row in excluded],
        resident_metric_kind_counts=dict(
            sorted(Counter(row["resident_metric_nats_kind"] for row in rows).items())
        ),
        by_atom=_group_report(included, lambda row: f"atom_{int(row['atom'])}"),
        out_of_readout_radius=(
            _score(excluded) if len(excluded) >= 3 else {"n": len(excluded)}
        ),
        protocol=ledger["protocol"],
        prediction="public exact_full/exact_directional Fisher dose of effective_delta",
        measurement="KL(p_base || p_patched) for the same effective_delta",
        validity_rule=(
            "readout radii learned only from split=calibration, then frozen; "
            "heldout predicted_nats <= the matching atom radius"
        ),
    )
    report["acceptance"] = {
        "slope": 0.9 <= report["slope_through_origin"] <= 1.1,
        "r2": report["r2_through_origin"] >= 0.9,
        "ci_excludes_0_54": not (ci[0] <= 0.54 <= ci[1]),
        "ci_excludes_4_30": not (ci[0] <= 4.30 <= ci[1]),
    }
    report["accepted"] = all(report["acceptance"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--readout-tol-rel", type=float, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2249)
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text())
    report = acceptance_report(
        ledger,
        readout_tol_rel=args.readout_tol_rel,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
