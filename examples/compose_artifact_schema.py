"""Strict JSON contracts for compose/control/eval readiness artifacts.

These helpers are intentionally fit-free. They exist so cluster driver scripts can
fail on schema/version/baseline mistakes before launching GPU or long CPU jobs.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

COMPOSE_SCHEMA = "gam.compose_per_atom"
NULL_CONTROL_SCHEMA = "gam.null_control"
SCHEMA_VERSION = 1
MIN_GAMFIT_VERSION = "0.1.259"
EV_BASELINE = "train_mean"
EV_TSS_MEAN_SOURCE = "train_split_column_mean"


class ArtifactSchemaError(ValueError):
    """Raised when a compose/control artifact is not publishable."""


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(version).split("+", 1)[0])
    if not parts:
        return ()
    return tuple(int(p) for p in parts[:4])


def require_gamfit_version(
    actual: str | None = None, *, min_version: str = MIN_GAMFIT_VERSION
) -> str:
    """Fail fast when the running wheel predates the compose-readiness cutover."""
    if actual is None:
        import gamfit

        actual = getattr(gamfit, "__version__", "0.0.0+unknown")
    got = _version_tuple(str(actual))
    need = _version_tuple(str(min_version))
    if not got or got < need:
        raise RuntimeError(
            f"gamfit>={min_version} is required for compose readiness artifacts; "
            f"running {actual!r}"
        )
    return str(actual)


def explained_variance_train_mean(
    x: Any, recon: Any, train_mean: Any
) -> float:
    """EV = 1 - SSE/TSS with TSS about the train-split column mean."""
    x_arr = np.asarray(x, dtype=np.float64)
    recon_arr = np.asarray(recon, dtype=np.float64)
    mean_arr = np.asarray(train_mean, dtype=np.float64).reshape(1, -1)
    if x_arr.shape != recon_arr.shape:
        raise ValueError(f"x and recon shapes differ: {x_arr.shape} vs {recon_arr.shape}")
    if mean_arr.shape[1] != x_arr.shape[1]:
        raise ValueError(
            f"train_mean width {mean_arr.shape[1]} does not match x width {x_arr.shape[1]}"
        )
    rss = float(np.sum((x_arr - recon_arr) ** 2))
    tss = float(np.sum((x_arr - mean_arr) ** 2))
    return 1.0 - rss / tss if tss > 0.0 else 0.0


def ev_contract(eval_split: str, *, heldout_subsample_n: int | None = None) -> dict[str, Any]:
    """Canonical EV contract block shared by compose and frontier artifacts."""
    return {
        "baseline": EV_BASELINE,
        "definition": "1 - SSE_recon/TSS",
        "tss_mean_source": EV_TSS_MEAN_SOURCE,
        "eval_split": str(eval_split),
        "heldout_subsample_n": heldout_subsample_n,
    }


def make_compose_per_atom_artifact(
    *,
    gamfit_version: str,
    random_state: int,
    min_effect_ev: float,
    operating_point: dict[str, Any],
    atoms: list[dict[str, Any]],
    births: list[dict[str, Any]],
    eval_split: str = "heldout",
    collapse_events_total: int = 0,
    grown_vs_joint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a strict `compose_per_atom.json` payload and validate it."""
    payload: dict[str, Any] = {
        "schema": COMPOSE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "producer": {
            "gamfit_version": require_gamfit_version(gamfit_version),
            "min_gamfit_version": MIN_GAMFIT_VERSION,
        },
        "run": {"random_state": int(random_state)},
        "ev_contract": ev_contract(
            eval_split, heldout_subsample_n=operating_point.get("heldout_subsample_n")
        ),
        "min_effect_ev": float(min_effect_ev),
        "salience_floor": float(min_effect_ev),
        "operating_point": dict(operating_point),
        "collapse_events_total": int(collapse_events_total),
        "atoms": list(atoms),
        "births": list(births),
    }
    if grown_vs_joint is not None:
        payload["grown_vs_joint"] = dict(grown_vs_joint)
    validate_compose_per_atom_artifact(payload, expected_random_state=int(random_state))
    return payload


def make_null_control_artifact(
    *,
    salience_floor: float,
    real_reference: dict[str, Any],
    gaussian_matched: dict[str, Any],
    shuffled: dict[str, Any],
    harmonic_null: dict[str, Any] | None = None,
    theta_accept: float | None = None,
) -> dict[str, Any]:
    """Build a strict `null_control.json` payload and validate it.

    ``theta_accept`` is the pre-registered null-calibrated curved-atom acceptance threshold
    (Amendment 1(1) of prereg_35b.md): the q99 of the turning Θ the matched-Gaussian null
    manufactures through the identical composed pipeline. EVAL reads it as the operative A2
    acceptance threshold; when absent, A2/I1/G_band/G_util stay PENDING (never Θ>1)."""
    payload = {
        "schema": NULL_CONTROL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "salience_floor": float(salience_floor),
        "real_reference": dict(real_reference),
        "gaussian_matched": dict(gaussian_matched),
        "shuffled": dict(shuffled),
        "harmonic_null": dict(harmonic_null or {}),
    }
    if theta_accept is not None:
        payload["theta_accept"] = float(theta_accept)
    validate_null_control_artifact(payload, compose_min_effect_ev=float(salience_floor))
    return payload


def validate_compose_per_atom_artifact(
    payload: dict[str, Any], *, expected_random_state: int | None = None
) -> None:
    _require_schema(payload, COMPOSE_SCHEMA)
    producer = _require_dict(payload, "producer")
    require_gamfit_version(_require_str(producer, "gamfit_version"))
    run = _require_dict(payload, "run")
    random_state = _require_int(run, "random_state")
    if expected_random_state is not None and random_state != int(expected_random_state):
        raise ArtifactSchemaError(
            f"run.random_state must be {expected_random_state}; got {random_state}"
        )
    ev = _require_dict(payload, "ev_contract")
    if _require_str(ev, "baseline") != EV_BASELINE:
        raise ArtifactSchemaError("ev_contract.baseline must be 'train_mean'")
    if _require_str(ev, "tss_mean_source") != EV_TSS_MEAN_SOURCE:
        raise ArtifactSchemaError(
            "ev_contract.tss_mean_source must be 'train_split_column_mean'"
        )
    if _require_str(ev, "eval_split") != "heldout":
        raise ArtifactSchemaError("compose artifacts must report held-out EV")
    min_effect = _require_number(payload, "min_effect_ev")
    salience = _require_number(payload, "salience_floor")
    if abs(min_effect - salience) > 1e-12:
        raise ArtifactSchemaError("min_effect_ev and salience_floor must match")
    op = _require_dict(payload, "operating_point")
    for key in ("total_actives", "heldout_ev", "linear_only_heldout_ev"):
        _require_number(op, key)
    _require_int(op, "heldout_subsample_n")
    atoms = _require_list(payload, "atoms")
    if not atoms:
        raise ArtifactSchemaError("atoms must be non-empty")
    for i, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            raise ArtifactSchemaError(f"atoms[{i}] must be an object")
        _require_str(atom, "topology")
        _require_number(atom, "theta")
        _require_number(atom, "delta_ev")
        if _require_str(atom, "delta_ev_source") != "heldout_loao":
            raise ArtifactSchemaError(
                f"atoms[{i}].delta_ev_source must be 'heldout_loao'"
            )
    births = _require_list(payload, "births")
    prev_ev: float | None = None
    for i, birth in enumerate(births):
        if not isinstance(birth, dict):
            raise ArtifactSchemaError(f"births[{i}] must be an object")
        ev_after = _require_number(birth, "ev")
        _require_int(birth, "collapse_events")
        if prev_ev is not None and ev_after + 1e-12 < prev_ev:
            raise ArtifactSchemaError("birth EV trace must be monotone non-decreasing")
        prev_ev = ev_after
    _require_int(payload, "collapse_events_total")


def validate_null_control_artifact(
    payload: dict[str, Any], *, compose_min_effect_ev: float | None = None
) -> None:
    _require_schema(payload, NULL_CONTROL_SCHEMA)
    floor = _require_number(payload, "salience_floor")
    if compose_min_effect_ev is not None and abs(floor - float(compose_min_effect_ev)) > 1e-12:
        raise ArtifactSchemaError("null_control.salience_floor must match compose min_effect_ev")
    for arm in ("real_reference", "gaussian_matched", "shuffled"):
        block = _require_dict(payload, arm)
        _require_int(block, "n_curved_accepted")
        _require_number(block, "mean_theta")
        for j, point in enumerate(block.get("scatter_points", [])):
            if not isinstance(point, dict):
                raise ArtifactSchemaError(f"{arm}.scatter_points[{j}] must be an object")
            _require_number(point, "theta")
            _require_number(point, "delta_ev")
    # Amendment 1(1): optional null-calibrated acceptance threshold; if present must be a
    # positive turning (radians). Absent → EVAL leaves curved-atom acceptance uncalibrated.
    if "theta_accept" in payload:
        ta = payload["theta_accept"]
        if not isinstance(ta, (int, float)) or isinstance(ta, bool) or float(ta) <= 0.0:
            raise ArtifactSchemaError("theta_accept must be a positive number (radians)")


def _require_schema(payload: dict[str, Any], schema: str) -> None:
    if payload.get("schema") != schema:
        raise ArtifactSchemaError(f"schema must be {schema!r}; got {payload.get('schema')!r}")
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ArtifactSchemaError(
            f"schema_version must be {SCHEMA_VERSION}; got {version!r}"
        )


def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ArtifactSchemaError(f"{key} must be an object")
    return value


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ArtifactSchemaError(f"{key} must be a list")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ArtifactSchemaError(f"{key} must be a non-empty string")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactSchemaError(f"{key} must be an integer")
    return value


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactSchemaError(f"{key} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ArtifactSchemaError(f"{key} must be a finite number")
    return out
