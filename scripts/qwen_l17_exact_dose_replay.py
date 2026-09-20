#!/usr/bin/env python3
"""Score a frozen Qwen layer-17 ledger emitted by the public steering API.

The producer must call ``ManifoldSAE.steer`` / ``steer_to_target`` and persist
the returned plan together with its plan-aware applied-dose observation. This
scorer never re-computes or overwrites ``predicted_nats``. It accepts only an
``exact_full`` or ``exact_directional`` public prediction.

What the meter is calibrated against. The prediction is the Fisher dose
``q = ½ (Jδ)ᵀ F (Jδ)`` of the applied move, with ``F`` the softmax Fisher of the
base output. The measurement is ``m = KL(p_base || p_patched)`` of the same move
over the full vocabulary. That KL is the centred cumulant generating function of
the logit change ``Δ`` under ``p_base`` evaluated at 1, ``½ Var(Δ) + κ₃(Δ)/6 + …``,
and ``Δ = Jδ + O(|δ|²)``, so ``m = q + O(|δ|³)``. Along one chord of the fitted
atom this is ``m / q = 1 + γ √q + O(q)``: the meter is exact in the limit of a
small dose, and its relative error grows like the square root of the dose.

The acceptance tests that limit and uses no tolerance. Each held-out chord's two
smallest resolved doses ``q₁ < q₂`` give the straight line in ``√q`` through
``(√q₁, m₁/q₁)`` and ``(√q₂, m₂/q₂)``; its value at zero dose, ``r₀``, cancels the
``γ √q`` term and leaves ``−β √(q₁ q₂)`` of the ``O(q)`` term, which is reported
from a third dose where one exists. The verdict reads the chord-level bootstrap
95% interval ``[lo, hi]`` of the mean ``r₀``:

* ``FAIL`` when it excludes 1;
* ``UNRESOLVED`` when it contains 1 and also a ratio one of #2249's miscalibrated
  predictors measured, so the ledger cannot tell the meter from that defect and
  needs more rows; it never passes;
* ``PASS`` when it contains 1 and excludes every such ratio.

``r₀`` is linear in the measurements and the bootstrap draws do not depend on them,
so a meter mis-scaled by ``c`` moves the interval to exactly ``c`` times itself; the
smallest mis-scale the gate refuses is therefore ``1/lo − 1`` upward and
``1 − 1/hi`` downward, and the report publishes both.

The coverage it certifies. The expansion needs the logits to be three times
differentiable in the activation. On a mixture-of-experts model they are not
where a router's top-k expert set changes, so a chord's resolved doses stop at its
first patched forward whose top-k sets differ from the base forward's
(``router_topk_changes > 0``). A dose resolves only when its KL exceeds its own
measurement floor (``measurement_floor_nats``), which the Rust owner derives
from the record's logit format and extents and raises by stochastic control
evidence; a row at or under it is kept, reported and left out, never clamped, and
a KL below minus its float64 evaluation band is refused, because roundoff cannot
produce it. Held-out doses count only inside the region the resolved calibration
rows occupy for the same atom, from their lowest dose to their highest;
calibration chords publish the error law ``γ`` over that region.

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
from typing import Any

import numpy as np


# The measured/predicted ratios of the three miscalibrated predictors #2249 was
# filed on: the through-origin slopes of its table over 51 rows, tangent quadratic
# form 4.30, path-integrated 0.541, tangent first-order 0.545. They are measurements
# of known defects, not tolerances. The meter's ratio is ruled out as defect ``X``
# exactly when ``X`` lies outside the interval, so an interval that covers one of
# them cannot tell the meter from that defect.
_KNOWN_FAILURE_RATIOS = {
    "tangent_quadratic_2249": 4.30,
    "path_integrated_2249": 0.541,
    "tangent_first_order_2249": 0.545,
}
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
    "floor_repetitions",
    "max_templates",
    "bases",
    "fit_iterations",
    "router_modules",
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
        evaluation_band = _finite_nonnegative(
            raw["measurement_evaluation_nats"],
            field="measurement_evaluation_nats",
            intervention_id=intervention_id,
        )
        floor = float(raw["measurement_floor_nats"])
        if not (math.isfinite(floor) and floor > 0.0):
            raise ValueError(
                f"intervention {intervention_id!r} needs a finite positive measurement_floor_nats; "
                f"got {floor!r}"
            )
        # The measured KL is scored as computed, never clamped. A true KL is
        # non-negative, so only roundoff within the float64 evaluation band can make a
        # computed one negative; below that it is a sign error.
        measured = float(raw["measured_nats"])
        if not (math.isfinite(measured) and measured >= -evaluation_band):
            raise ValueError(
                f"intervention {intervention_id!r} measured_nats {measured!r} is below minus its "
                f"float64 evaluation band {evaluation_band!r}, which roundoff cannot produce"
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
        router_topk_changes = raw.get("router_topk_changes")
        if (
            not isinstance(router_topk_changes, int)
            or isinstance(router_topk_changes, bool)
            or router_topk_changes < 0
        ):
            raise ValueError(
                f"intervention {intervention_id!r} needs router_topk_changes, the count of "
                "token rows whose router top-k set differs from the base forward's; got "
                f"{router_topk_changes!r}"
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
            router_topk_changes=router_topk_changes,
            measurement_evaluation_nats=evaluation_band,
            measurement_floor_nats=floor,
        )
        validated.append(row)
    return validated


def _calibrated_regions(rows: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    """Each atom's calibrated dose region, ``(lowest, highest)`` resolved calibration dose.

    A calibration certifies only the doses its rows occupy and resolve, so a
    held-out dose below the lowest or above the highest calibration dose of its
    atom that cleared its own measurement floor is not scored.
    """
    calibration = [row for row in rows if row["split"] == "calibration" and _resolved(row)]
    regions: dict[int, tuple[float, float]] = {}
    for atom in sorted({int(row["atom"]) for row in rows}):
        doses = [float(row["predicted_nats"]) for row in calibration if int(row["atom"]) == atom]
        if not doses:
            raise ValueError(f"atom {atom} has no resolved calibration rows to certify a dose region")
        regions[atom] = (min(doses), max(doses))
    return regions


def _chord_window(chord: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The chord's doses in increasing order, up to its first router top-k change.

    Past a change of any router's expert set the logits are not smooth in the
    activation along the chord, so the small-dose expansion no longer describes
    that dose or any larger one.
    """
    ordered = sorted(chord, key=lambda row: (float(row["predicted_nats"]), row["intervention_id"]))
    window: list[dict[str, Any]] = []
    for row in ordered:
        if row["router_topk_changes"] > 0:
            break
        window.append(row)
    return window


def _chord_extrapolation(window: list[dict[str, Any]]) -> dict[str, Any] | None:
    """``r₀``, ``γ`` and the truncation estimate of one chord's resolved doses.

    ``r = m/q`` is extrapolated to zero dose along the straight line in ``x = √q``
    through the two smallest doses. With ``r = 1 + γ x + β x²``, that line's value
    at zero is ``1 − β x₁ x₂``; a third dose gives ``β`` by the second divided
    difference, and ``−β x₁ x₂`` is reported as the truncation estimate.
    """
    points = [
        (math.sqrt(float(row["predicted_nats"])), float(row["measured_nats"]) / float(row["predicted_nats"]))
        for row in window
        if float(row["predicted_nats"]) > 0.0
    ]
    if len(points) < 2 or not points[0][0] < points[1][0]:
        return None
    (x1, r1), (x2, r2) = points[0], points[1]
    slope = (r2 - r1) / (x2 - x1)
    truncation = None
    if len(points) >= 3 and points[1][0] < points[2][0]:
        x3, r3 = points[2]
        curvature = ((r3 - r2) / (x3 - x2) - slope) / (x3 - x1)
        truncation = -curvature * x1 * x2
    return {
        "r0": r1 - x1 * slope,
        "gamma": slope,
        "r0_truncation": truncation,
        "doses_nats": [x1 * x1, x2 * x2],
    }


def _resolved(row: dict[str, Any]) -> bool:
    """A measurement resolves its dose only strictly above its own floor."""
    return float(row["measured_nats"]) > float(row["measurement_floor_nats"])


def _chords(
    rows: list[dict[str, Any]], regions: dict[int, tuple[float, float]], split: str
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, list[str]]]:
    """Per-chord extrapolations for one split, plus the rows and chords each rule set aside.

    The router window is cut on the whole chord before the calibrated region and the
    measurement floor are applied: a top-k change at a dose outside the region, or
    one the forward does not resolve, still ends the smooth part of the chord for
    every larger dose. A dose at or under its own floor is kept and set aside, never
    clamped, and the chord's next resolved doses take its place.
    """
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["split"] == split:
            grouped.setdefault((int(row["atom"]), str(row["base_prompt_id"])), []).append(row)
    chords: dict[tuple[int, str], dict[str, Any]] = {}
    set_aside: dict[str, list[str]] = {
        "outside_calibrated_region": [],
        "past_router_change": [],
        "at_or_under_measurement_floor": [],
        "unresolved_chords": [],
    }
    for key in sorted(grouped):
        window = _chord_window(grouped[key])
        in_window = {row["intervention_id"] for row in window}
        set_aside["past_router_change"].extend(
            row["intervention_id"] for row in grouped[key] if row["intervention_id"] not in in_window
        )
        lo, hi = regions[key[0]]
        resolved = []
        for row in window:
            if not lo <= float(row["predicted_nats"]) <= hi:
                set_aside["outside_calibrated_region"].append(row["intervention_id"])
            elif not _resolved(row):
                set_aside["at_or_under_measurement_floor"].append(row["intervention_id"])
            else:
                resolved.append(row)
        extrapolation = _chord_extrapolation(resolved)
        if extrapolation is None:
            set_aside["unresolved_chords"].append(f"{key[0]}:{key[1]}")
            continue
        chords[key] = extrapolation
    for name in ("outside_calibrated_region", "past_router_change", "at_or_under_measurement_floor"):
        set_aside[name].sort()
    return chords, set_aside


def _bootstrap_mean_ci(values: np.ndarray, *, draws: int, seed: int) -> list[float]:
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    if values.size < 2:
        raise ValueError("the r0 bootstrap needs at least two resolved held-out chords")
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        means[draw] = float(np.mean(values[rng.integers(0, values.size, size=values.size)]))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def acceptance_report(
    ledger: dict[str, Any], *, bootstrap_draws: int, seed: int
) -> dict[str, Any]:
    rows = _validate_ledger(ledger)
    regions = _calibrated_regions(rows)
    heldout, heldout_set_aside = _chords(rows, regions, "heldout")
    calibration, calibration_set_aside = _chords(rows, regions, "calibration")
    r0 = np.asarray([chord["r0"] for chord in heldout.values()], dtype=np.float64)
    lo, hi = _bootstrap_mean_ci(r0, draws=bootstrap_draws, seed=seed)
    truncation = [chord["r0_truncation"] for chord in heldout.values() if chord["r0_truncation"] is not None]
    by_atom: dict[str, dict[str, Any]] = {}
    for atom in sorted(regions):
        gammas = [chord["gamma"] for key, chord in calibration.items() if key[0] == atom]
        atom_r0 = [chord["r0"] for key, chord in heldout.items() if key[0] == atom]
        by_atom[f"atom_{atom}"] = {
            "calibrated_region_nats": list(regions[atom]),
            "calibration_chords": len(gammas),
            "gamma_mean": float(np.mean(gammas)) if gammas else None,
            "heldout_chords": len(atom_r0),
            "r0_mean": float(np.mean(atom_r0)) if atom_r0 else None,
        }
    report: dict[str, Any] = {
        "r0_mean": float(np.mean(r0)),
        "r0_chord_bootstrap_95_ci": [lo, hi],
        "r0_ci_half_width": 0.5 * (hi - lo),
        "r0_resolution": {
            "upward_mis_scale": 1.0 / lo - 1.0 if lo > 0.0 else math.inf,
            "downward_mis_scale": 1.0 - 1.0 / hi if hi > 0.0 else math.inf,
        },
        "r0_truncation_mean": float(np.mean(truncation)) if truncation else None,
        "r0_truncation_chords": len(truncation),
        "heldout_chords": {f"{key[0]}:{key[1]}": chord for key, chord in heldout.items()},
        "by_atom": by_atom,
        "row_counts": {
            "total": len(rows),
            "calibration": sum(row["split"] == "calibration" for row in rows),
            "heldout": sum(row["split"] == "heldout" for row in rows),
            "heldout_outside_calibrated_region": len(heldout_set_aside["outside_calibrated_region"]),
            "heldout_past_router_change": len(heldout_set_aside["past_router_change"]),
            "heldout_at_or_under_measurement_floor": len(
                heldout_set_aside["at_or_under_measurement_floor"]
            ),
            "calibration_past_router_change": len(calibration_set_aside["past_router_change"]),
            "calibration_at_or_under_measurement_floor": len(
                calibration_set_aside["at_or_under_measurement_floor"]
            ),
        },
        "heldout_outside_calibrated_region": heldout_set_aside["outside_calibrated_region"],
        "heldout_past_router_change": heldout_set_aside["past_router_change"],
        "heldout_at_or_under_measurement_floor": heldout_set_aside["at_or_under_measurement_floor"],
        "heldout_chords_unresolved": heldout_set_aside["unresolved_chords"],
        "resident_metric_kind_counts": dict(
            sorted(Counter(row["resident_metric_nats_kind"] for row in rows).items())
        ),
        "protocol": ledger["protocol"],
        "prediction": "public exact_full/exact_directional Fisher dose of effective_delta",
        "measurement": "KL(p_base || p_patched) for the same effective_delta",
        "validity_rule": (
            "held-out doses inside the atom's calibrated region, before the chord's first "
            "router top-k change, and measured strictly above their own measurement floor; "
            "r0 from the two smallest, extrapolated linearly in sqrt(dose)"
        ),
    }
    defects_in_interval = sorted(
        name for name, ratio in _KNOWN_FAILURE_RATIOS.items() if lo <= ratio <= hi
    )
    if not lo <= 1.0 <= hi:
        verdict = "FAIL"
    elif defects_in_interval:
        verdict = "UNRESOLVED"
    else:
        verdict = "PASS"
    report["verdict"] = verdict
    report["unresolved_against"] = defects_in_interval
    report["accepted"] = verdict == "PASS"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2249)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text())
    report = acceptance_report(
        ledger,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
