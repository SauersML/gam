"""Contract for the Rust-owned `ManifoldSAE` #[pyclass] (issue #2091 cutover).

The pyclass owns the fitted-model state in Rust (serde `ManifoldSaePayload`) and
exposes the flat attribute surface consumers read plus a `to_dict` that round-
trips through the same serde schema as `sae_manifold_payload_roundtrip`. This
suite constructs the pyclass from the golden fixture and checks:
  - to_dict is a value-for-value fixed point (the load-bearing invariant);
  - the dense-array, scalar, list, and report-block getters return the right
    shapes/types/values the Python dataclass exposed.

This is the getter analogue of the round-trip contract in
`tests/test_manifold_sae_golden_roundtrip.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gamfit._sae_manifold import rust_module

GOLDEN_FULL = (
    Path(__file__).resolve().parent / "fixtures" / "manifold_sae" / "golden_full.json"
)


ManifoldSAE = rust_module().ManifoldSAE


def _golden() -> dict:
    return json.loads(GOLDEN_FULL.read_text())


def test_to_dict_is_a_fixed_point() -> None:
    core = ManifoldSAE(_golden())
    again = core.to_dict()
    golden = _golden()
    if again != golden:
        mismatched = sorted(
            k for k in set(golden) & set(again) if golden[k] != again[k]
        )
        raise AssertionError(f"pyclass to_dict drift on keys: {mismatched}")


def test_dense_array_getters() -> None:
    golden = _golden()
    core = ManifoldSAE(golden)
    fitted = core.fitted
    assert isinstance(fitted, np.ndarray)
    assert fitted.shape == np.asarray(golden["fitted"]).shape
    np.testing.assert_array_equal(fitted, np.asarray(golden["fitted"]))
    assert core.assignments.shape == np.asarray(golden["assignments"]).shape
    assert core.low_level_logits.shape == np.asarray(golden["logits"]).shape
    np.testing.assert_array_equal(
        core.training_mean, np.asarray(golden["training_mean"])
    )
    np.testing.assert_array_equal(core.tier0_scale, np.asarray(golden["tier0_scale"]))
    # Numeric state stays array-valued; geometry is one validated mapping list.
    assert isinstance(core.coords, list) and len(core.coords) == len(golden["coords"])
    np.testing.assert_array_equal(core.coords[0], np.asarray(golden["coords"][0]))
    assert len(core.decoder_blocks) == len(golden["decoder_blocks"])
    assert core.geometry_plans == golden["geometry_plans"]


def test_fisher_and_selected_getters() -> None:
    golden = _golden()
    core = ManifoldSAE(golden)
    np.testing.assert_array_equal(
        core.fisher_factors, np.asarray(golden["fisher_factors"])
    )
    assert core.fisher_factors.shape == np.asarray(golden["fisher_factors"]).shape
    np.testing.assert_array_equal(
        core.fisher_mass_residual, np.asarray(golden["fisher_mass_residual"])
    )
    np.testing.assert_array_equal(
        core.selected_log_lambda_smooth,
        np.asarray(golden["selected_log_lambda_smooth"]),
    )
    ard = core.selected_log_ard
    assert isinstance(ard, list) and len(ard) == len(golden["selected_log_ard"])
    np.testing.assert_array_equal(ard[1], np.asarray(golden["selected_log_ard"][1]))


def test_scalar_and_list_getters() -> None:
    golden = _golden()
    core = ManifoldSAE(golden)
    assert core.atom_topology == "mixed"
    assert core.assignment == golden["assignment"]
    assert core.assignment_label == golden["assignment_label"]
    assert core.metric_provenance == golden["metric_provenance"]
    assert core.fisher_provenance == golden["fisher_provenance"]
    assert core.fisher_factor_kind == golden["fisher_factor_kind"]
    assert core.top_k == golden["top_k"]
    assert core.reconstruction_r2 == golden["reconstruction_r2"]
    assert core.penalized_loss_score == golden["penalized_loss_score"]
    assert core.chosen_k == len(golden["atoms"])
    assert core.selected_log_lambda_sparse == golden["selected_log_lambda_sparse"]
    assert core.atom_topologies == ["circle", "linear", "euclidean"]
    assert core.basis_kinds == ["periodic", "linear", "duchon"]
    assert core.atom_dims == [plan["latent_dim"] for plan in golden["geometry_plans"]]
    assert core.basis_sizes == [len(block) for block in golden["decoder_blocks"]]


def test_atoms_is_an_object_surface() -> None:
    """model.atoms is a list of AtomCore handles read by attribute (the
    Rust-owned AtomCore surface), NOT a list of dicts."""
    golden = _golden()
    core = ManifoldSAE(golden)
    atoms = core.atoms
    assert isinstance(atoms, list) and len(atoms) == len(golden["atoms"])
    a0 = atoms[0]
    g0 = golden["atoms"][0]
    assert not isinstance(a0, dict)  # object surface, not a mapping
    assert a0.basis == golden["geometry_plans"][0]["kind"]
    assert a0.active_dim == g0["active_dim"]
    assert a0.evidence == g0["evidence"]
    np.testing.assert_array_equal(
        a0.decoder_coefficients, np.asarray(g0["decoder_coefficients"])
    )
    np.testing.assert_array_equal(a0.coords, np.asarray(g0["coords"]))
    # atom 0 (periodic, d=1) carries the arc coordinate + shape band.
    np.testing.assert_array_equal(a0.coords_u_arc, np.asarray(g0["coords_u_arc"]))
    np.testing.assert_array_equal(a0.shape_band_mean, np.asarray(g0["shape_band_mean"]))
    # atom 1 (euclidean, d=2) carries neither.
    assert atoms[1].coords_u_arc is None
    assert atoms[1].shape_band_mean is None
    # No atom in the fixture carries a covariance factor -> dense cov is None.
    assert all(a.decoder_covariance is None for a in atoms)


def test_atom_dense_covariance_is_reconstructed() -> None:
    """When a compact per-channel factor is present, atom.decoder_covariance is
    the dense (M_k*p, M_k*p) block-diagonal matrix (not the compact factor)."""
    golden = _golden()
    # Plant a compact (p, M_k, M_k) factor on atom 0: M_k rows in decoder, p cols.
    coeffs = np.asarray(golden["atoms"][0]["decoder_coefficients"])
    m_k, p = coeffs.shape
    factor = np.zeros((p, m_k, m_k))
    for c in range(p):
        factor[c] = np.eye(m_k) * (c + 1.0)
    golden["atoms"][0]["decoder_covariance_channel_factors"] = factor.tolist()
    atom0 = ManifoldSAE(golden).atoms[0]
    cov = atom0.decoder_covariance
    assert cov is not None and cov.shape == (m_k * p, m_k * p)
    # Same-channel diagonal blocks restored; cross-channel entries zero.
    for c in range(p):
        for b1 in range(m_k):
            for b2 in range(m_k):
                assert cov[b1 * p + c, b2 * p + c] == factor[c, b1, b2]


def test_report_block_getters() -> None:
    golden = _golden()
    core = ManifoldSAE(golden)
    assert core.diagnostics == golden["diagnostics"]
    assert core.penalized_quasi_laplace_criterion == pytest.approx(
        golden["penalized_quasi_laplace_criterion"]
    )
    assert core.solver_plan == golden["solver_plan"]
    assert core.hybrid_split == golden["hybrid_split"]
    assert core.certificates == golden["certificates"]
    assert core.structure_certificate_json == golden["structure_certificate"]
    # A None report block reads back as Python None.
    payload = dict(golden)
    payload["cotrain"] = None
    assert ManifoldSAE(payload).cotrain is None


def test_native_summary_and_description_length_use_the_fitted_artifact() -> None:
    golden = _golden()
    core = ManifoldSAE(golden)

    summary = core.summary()
    assert summary["K"] == len(golden["atoms"])
    assert summary["atom_topology"] == "mixed"
    assert summary["penalized_loss_score"] == golden["penalized_loss_score"]
    assert summary["penalized_quasi_laplace_criterion"] == pytest.approx(
        golden["penalized_quasi_laplace_criterion"]
    )
    assert "reml_score" not in summary
    assert "evidence" not in summary

    description = core.description_length()
    assert description is not None
    assert description["n_tokens"] == len(golden["fitted"])
    assert description["g_dict"] == len(golden["atoms"])
    assert description["n_params"] == sum(
        len(row)
        for decoder in golden["decoder_blocks"]
        for row in decoder
    )
    assert summary["bits_per_token"] == pytest.approx(
        description["bits_per_token"]
    )
    assert summary["description_length"] == description
