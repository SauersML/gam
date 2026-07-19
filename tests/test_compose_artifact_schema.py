from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import compose_artifact_schema as cas  # noqa: E402


def _atom(theta: float | None = 1.2, delta_ev_source: str = "heldout_loao") -> dict:
    return {
        "idx": 0,
        "topology": "circle",
        "theta": theta,
        "delta_ev": 0.02,
        "delta_ev_source": delta_ev_source,
        "d_atom": 1,
    }


def _compose_payload() -> dict:
    return cas.make_compose_per_atom_artifact(
        gamfit_version=cas.MIN_GAMFIT_VERSION,
        random_state=0,
        min_effect_ev=0.005,
        operating_point={
            "total_actives": 40,
            "heldout_ev": 0.9,
            "linear_only_heldout_ev": 0.86,
            "heldout_subsample_n": 50_000,
        },
        atoms=[_atom()],
        births=[{"ev": 0.86, "collapse_events": 0}, {"ev": 0.9, "collapse_events": 0}],
    )


def test_version_guard_accepts_current_wheel() -> None:
    cas.require_gamfit_version("0.1.258")


def test_explained_variance_uses_train_mean_not_heldout_mean() -> None:
    x = np.array([[10.0, 0.0], [12.0, 0.0]])
    recon = np.array([[10.0, 0.0], [10.0, 0.0]])
    train_mean = np.array([0.0, 0.0])
    heldout_mean = x.mean(axis=0)
    honest = cas.explained_variance_train_mean(x, recon, train_mean)
    leaked = cas.explained_variance_train_mean(x, recon, heldout_mean)
    assert honest > leaked


def test_compose_schema_requires_theta_and_heldout_delta_ev() -> None:
    payload = _compose_payload()
    cas.validate_compose_per_atom_artifact(payload, expected_random_state=0)

    missing_theta = _compose_payload()
    missing_theta["atoms"][0]["theta"] = None
    with pytest.raises(cas.ArtifactSchemaError, match="theta"):
        cas.validate_compose_per_atom_artifact(missing_theta)

    in_sample = _compose_payload()
    in_sample["atoms"][0]["delta_ev_source"] = "birth_in_sample"
    with pytest.raises(cas.ArtifactSchemaError, match="delta_ev_source"):
        cas.validate_compose_per_atom_artifact(in_sample)


def test_compose_schema_requires_seed2_provenance() -> None:
    payload = _compose_payload()
    payload["run"]["random_state"] = 0
    with pytest.raises(cas.ArtifactSchemaError, match="random_state"):
        cas.validate_compose_per_atom_artifact(payload, expected_random_state=2)


def test_null_control_schema_rejects_theta_none_and_floor_mismatch() -> None:
    payload = cas.make_null_control_artifact(
        salience_floor=0.005,
        real_reference={"n_curved_accepted": 7, "mean_theta": 1.8},
        gaussian_matched={
            "n_curved_accepted": 0,
            "mean_theta": 0.1,
            "scatter_points": [{"theta": 0.1, "delta_ev": 0.001}],
        },
        shuffled={"n_curved_accepted": 1, "mean_theta": 0.2},
        harmonic_null={"higher_modes_on_first_harmonic_plus_noise": False},
    )
    cas.validate_null_control_artifact(payload, compose_min_effect_ev=0.005)

    bad_theta = dict(payload)
    bad_theta["gaussian_matched"] = dict(payload["gaussian_matched"])
    bad_theta["gaussian_matched"]["mean_theta"] = None
    with pytest.raises(cas.ArtifactSchemaError, match="mean_theta"):
        cas.validate_null_control_artifact(bad_theta, compose_min_effect_ev=0.005)

    with pytest.raises(cas.ArtifactSchemaError, match="salience_floor"):
        cas.validate_null_control_artifact(payload, compose_min_effect_ev=0.010)
