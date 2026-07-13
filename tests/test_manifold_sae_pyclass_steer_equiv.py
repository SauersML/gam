"""Native ManifoldSAE ownership and resident-Fisher contracts (#2091)."""
from __future__ import annotations

import inspect
import json
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
    probe_calls: list[float] = []

    def patched_forward_kl(amplitude: float) -> float:
        probe_calls.append(amplitude)
        return float(model.steer(0, 0, amplitude, t_from, t_to)["predicted_nats"])

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

    assert plan["validation"] == "patched_forward"
    assert plan["iterations"] == 1
    assert probe_calls == [plan["amplitude"]]
    assert plan["measured_nats"] == pytest.approx(target, rel=1.0e-12)
    assert plan["predicted_nats"] == pytest.approx(target, rel=1.0e-12)
    np.testing.assert_allclose(
        plan["delta"],
        model.steer(0, 0, plan["amplitude"], t_from, t_to)["delta"],
        rtol=0.0,
        atol=0.0,
    )
    assert model.fisher_metric_build_count == 1


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
