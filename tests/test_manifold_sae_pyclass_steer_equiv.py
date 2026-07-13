"""Native ManifoldSAE ownership and resident-Fisher contracts (#2091)."""
from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gamfit import ManifoldSAE
from gamfit import _sae_manifold as facade


GOLDEN = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "manifold_sae"
    / "golden_full.json"
)


def _model() -> ManifoldSAE:
    return ManifoldSAE.from_dict(json.loads(GOLDEN.read_text()))


def _analytic_topology_model(
    basis: str,
    topology: str,
    latent_dim: int,
    basis_size: int,
    n_harmonics: int,
) -> ManifoldSAE:
    """Replace atom zero with one exact analytic persisted topology.

    The remaining two atoms keep the golden artifact's valid routing and Fisher
    state.  Only the persisted atom schema is changed, so these tests exercise
    the public model's frozen-state rebuild rather than any fit path.
    """
    payload = deepcopy(json.loads(GOLDEN.read_text()))
    n_rows = len(payload["coords"][0])
    p_out = len(payload["decoder_blocks"][0][0])
    coords = [
        [((row + 1) * (axis + 2)) / (n_rows + 7) for axis in range(latent_dim)]
        for row in range(n_rows)
    ]
    decoder = [
        [((basis_row + 1) * (column + 2)) / (basis_size + p_out) for column in range(p_out)]
        for basis_row in range(basis_size)
    ]

    payload["basis_kinds"][0] = basis
    payload["basis_specs"][0] = f"{basis}:persisted"
    payload["atom_topologies"][0] = topology
    payload["atom_topology"] = (
        topology if all(value == topology for value in payload["atom_topologies"]) else "mixed"
    )
    payload["atom_dims"][0] = latent_dim
    payload["basis_sizes"][0] = basis_size
    payload["n_harmonics"][0] = n_harmonics
    payload["coords"][0] = coords
    payload["decoder_blocks"][0] = decoder
    payload["duchon_centers"][0] = None

    atom = payload["atoms"][0]
    atom["basis"] = basis
    atom["active_dim"] = latent_dim
    atom["coords"] = coords
    atom["coords_u_arc"] = None
    atom["decoder_coefficients"] = decoder
    atom["decoder_covariance_channel_factors"] = None
    atom["shape_band_coords"] = None
    atom["shape_band_mean"] = None
    atom["shape_band_sd"] = None
    return ManifoldSAE.from_dict(payload)


def test_public_type_is_the_pyo3_model() -> None:
    assert ManifoldSAE is facade.rust_module().ManifoldSAE
    model = _model()
    assert type(model) is ManifoldSAE
    assert not hasattr(facade.rust_module(), "ManifoldSaeCore")


def test_fit_has_one_native_return_and_no_sparse_variant() -> None:
    source = inspect.getsource(facade.sae_manifold_fit)
    assert "return rust_module().sae_manifold_from_fit_payload(" in source
    assert "return sparse_dictionary_fit(" not in source
    assert "return block_sparse_dictionary_fit(" not in source


def test_steer_reuses_the_resident_metric() -> None:
    model = _model()
    assert model.fisher_metric_build_count == 1
    before = model.steer(
        0,
        0,
        1.0,
        np.array([0.0], dtype=np.float64),
        np.array([0.25], dtype=np.float64),
    )
    assert before["predicted_nats"] is not None
    assert model.fisher_metric_build_count == 1
    again = model.steer(
        0,
        0,
        1.0,
        np.array([0.0], dtype=np.float64),
        np.array([0.25], dtype=np.float64),
    )
    assert again == before
    assert model.fisher_metric_build_count == 1


def test_target_dose_probe_is_wired_through_the_public_model() -> None:
    model = _model()
    t_from = np.array([0.0], dtype=np.float64)
    t_to = np.array([0.25], dtype=np.float64)
    unit = model.steer(0, 0, 1.0, t_from, t_to)
    target = 0.5 * float(unit["predicted_nats"])
    probe_calls: list[dict[str, object]] = []

    def patched_forward_kl(steer_plan: dict[str, object]) -> dict[str, object]:
        probe_calls.append(steer_plan)
        exact = float(steer_plan["predicted_nats"])
        return {
            "effective_delta": list(steer_plan["delta"]),
            "exact_directional_nats": exact,
            "measured_nats": exact,
        }

    plan = model.steer_to_target(
        {
            "atom_k": 0,
            "metric_row": 0,
            "target_nats": target,
            "t_from": t_from,
            "t_to": t_to,
            "tol_rel": 1.0e-12,
            "max_iter": 4,
            "readout_tol_rel": 0.1,
        },
        patched_forward_kl,
    )

    assert plan["validation"] == "applied_dose_probe"
    assert plan["iterations"] == 1
    assert len(probe_calls) == 1
    assert probe_calls[0]["amplitude"] == plan["amplitude"]
    assert plan["measured_nats"] == pytest.approx(target, rel=1.0e-12)
    assert plan["predicted_nats"] == pytest.approx(target, rel=1.0e-12)
    assert plan["predicted_nats_kind"] == "exact_directional"
    assert plan["resident_metric_nats_kind"] == "uncertified_approximation"
    np.testing.assert_allclose(
        plan["delta"],
        model.steer(0, 0, plan["amplitude"], t_from, t_to)["delta"],
        rtol=0.0,
        atol=0.0,
    )
    assert model.fisher_metric_build_count == 1


def test_target_dose_rejects_scalar_and_malformed_probe_results() -> None:
    model = _model()
    t_from = np.array([0.0], dtype=np.float64)
    t_to = np.array([0.25], dtype=np.float64)
    target = 0.5 * float(model.steer(0, 0, 1.0, t_from, t_to)["predicted_nats"])
    request = {
        "atom_k": 0,
        "metric_row": 0,
        "target_nats": target,
        "t_from": t_from,
        "t_to": t_to,
        "tol_rel": 1.0e-12,
        "max_iter": 4,
        "readout_tol_rel": 0.1,
    }

    with pytest.raises(ValueError, match="must return a mapping"):
        model.steer_to_target(request, lambda _plan: target)

    def missing_measurement(plan: dict[str, object]) -> dict[str, object]:
        return {
            "effective_delta": list(plan["delta"]),
            "exact_directional_nats": float(plan["predicted_nats"]),
        }

    with pytest.raises(ValueError, match="missing.*measured_nats"):
        model.steer_to_target(request, missing_measurement)

    def wrong_effective_shape(plan: dict[str, object]) -> dict[str, object]:
        exact = float(plan["predicted_nats"])
        return {
            "effective_delta": [0.0],
            "exact_directional_nats": exact,
            "measured_nats": exact,
        }

    with pytest.raises(ValueError, match="effective_delta length"):
        model.steer_to_target(request, wrong_effective_shape)


@pytest.mark.parametrize(
    ("basis", "topology", "latent_dim", "basis_size", "n_harmonics", "t_from", "t_to"),
    [
        ("euclidean", "euclidean", 2, 3, 0, [0.2, -0.3], [0.45, -0.1]),
        ("torus", "torus", 2, 9, 1, [0.99, 0.3], [0.01, 0.55]),
        ("cylinder", "cylinder", 2, 6, 1, [0.99, 0.25], [0.01, 0.4]),
        ("mobius", "mobius", 2, 3, 1, [0.2, 0.4], [0.45, -0.2]),
    ],
)
def test_public_steer_rebuilds_every_analytic_coordinate_action(
    basis: str,
    topology: str,
    latent_dim: int,
    basis_size: int,
    n_harmonics: int,
    t_from: list[float],
    t_to: list[float],
) -> None:
    model = _analytic_topology_model(
        basis, topology, latent_dim, basis_size, n_harmonics
    )
    source = np.asarray(t_from, dtype=np.float64)
    target = np.asarray(t_to, dtype=np.float64)

    zero = model.steer(0, 0, 1.0, source, source)
    np.testing.assert_array_equal(zero["delta"], np.zeros(4, dtype=np.float64))

    moved = model.steer(0, 0, 1.0, source, target)
    assert np.linalg.norm(np.asarray(moved["delta"], dtype=np.float64)) > 0.0


@pytest.mark.parametrize(
    ("basis", "topology", "basis_size", "period", "t_from", "t_wrapped", "t_unwrapped"),
    [
        ("torus", "torus", 9, 1.0, [0.99, 0.3], [0.01, 0.3], [1.01, 0.3]),
        (
            "cylinder",
            "cylinder",
            6,
            1.0,
            [0.99, 0.25],
            [0.01, 0.25],
            [1.01, 0.25],
        ),
    ],
)
def test_public_steer_respects_periodic_seams_in_product_manifolds(
    basis: str,
    topology: str,
    basis_size: int,
    period: float,
    t_from: list[float],
    t_wrapped: list[float],
    t_unwrapped: list[float],
) -> None:
    model = _analytic_topology_model(basis, topology, 2, basis_size, 1)
    source = np.asarray(t_from, dtype=np.float64)
    wrapped = model.steer(0, 0, 1.0, source, np.asarray(t_wrapped, dtype=np.float64))
    unwrapped = model.steer(
        0, 0, 1.0, source, np.asarray(t_unwrapped, dtype=np.float64)
    )
    assert period == 1.0
    np.testing.assert_allclose(wrapped["delta"], unwrapped["delta"], rtol=0.0, atol=1e-12)


def test_public_mobius_steer_identifies_deck_twins() -> None:
    model = _analytic_topology_model("mobius", "mobius", 2, 3, 1)
    source = np.asarray([0.2, 0.4], dtype=np.float64)
    deck_twin = np.asarray([1.2, -0.4], dtype=np.float64)
    plan = model.steer(0, 0, 1.0, source, deck_twin)
    np.testing.assert_allclose(plan["delta"], np.zeros(4), rtol=0.0, atol=1e-12)


def test_public_target_dose_rebuilds_cylinder_metadata() -> None:
    model = _analytic_topology_model("cylinder", "cylinder", 2, 6, 1)
    t_from = np.asarray([0.15, -0.2], dtype=np.float64)
    t_to = np.asarray([0.18, -0.1], dtype=np.float64)
    unit = model.steer(0, 0, 1.0, t_from, t_to)
    target = 0.5 * float(unit["predicted_nats"])

    def patched_forward_kl(steer_plan: dict[str, object]) -> dict[str, object]:
        exact = float(steer_plan["predicted_nats"])
        return {
            "effective_delta": list(steer_plan["delta"]),
            "exact_directional_nats": exact,
            "measured_nats": exact,
        }

    plan = model.steer_to_target(
        {
            "atom_k": 0,
            "metric_row": 0,
            "target_nats": target,
            "t_from": t_from,
            "t_to": t_to,
            "tol_rel": 1.0e-12,
            "max_iter": 4,
            "readout_tol_rel": 0.1,
        },
        patched_forward_kl,
    )
    assert plan["validation"] == "applied_dose_probe"
    assert plan["measured_nats"] == pytest.approx(target, rel=1e-12)


def test_attach_fisher_is_atomic_and_builds_once() -> None:
    model = _model()
    original = model.to_dict()
    original_count = model.fisher_metric_build_count

    # Shape failure occurs while constructing the candidate RowMetric.  No
    # serialized field or resident metric is mutated.
    bad = np.zeros((1, 2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="must be \\(n, p, rank\\)"):
        model.attach_fisher(bad, "output_fisher", "uncertified_approximation")
    assert model.to_dict() == original
    assert model.fisher_metric_build_count == original_count

    valid = np.asarray(original["fisher_factors"], dtype=np.float64) * 0.5
    mass = np.asarray(original["fisher_mass_residual"], dtype=np.float64)
    model.attach_fisher(
        valid, "output_fisher", "uncertified_approximation", mass
    )
    assert model.fisher_metric_build_count == original_count + 1
    assert model.metric_provenance == "OutputFisher"
    assert model.fisher_provenance == "output_fisher"
    assert model.fisher_factor_kind == "uncertified_approximation"
    np.testing.assert_array_equal(model.fisher_factors, valid)
    np.testing.assert_array_equal(model.fisher_mass_residual, mass)


def test_detach_is_explicit_not_attach_none() -> None:
    model = _model()
    with pytest.raises(TypeError):
        model.attach_fisher(None, "output_fisher", "uncertified_approximation")
    model.detach_fisher()
    assert model.fisher_factors is None
    assert model.fisher_provenance is None
    assert model.fisher_factor_kind is None
    assert model.fisher_mass_residual is None
    assert model.metric_provenance == "Euclidean"
