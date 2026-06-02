"""Regression tests for the SAE manifold facade audit (#605-#609).

These exercise the pure-Python validation / payload / metadata logic of
``gamfit._sae_manifold`` directly, so they do not require the compiled Rust
extension. The few places that would touch Rust (the row-block penalty
capability probe) are monkeypatched to a no-op; any test that needs a *real*
fit is guarded and ``xfail``-skipped when the extension is unavailable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import gamfit._sae_manifold as sae


def _no_row_block_probe(monkeypatch):
    """Disable the Rust-backed row-block capability probe for payload tests."""
    monkeypatch.setattr(sae, "_require_sae_row_block_penalty", lambda kind, kwarg: None)


# ---------------------------------------------------------------------------
# #605 — isometry_weight must flow into the analytic-penalty payload, with a
# distinct descriptor per weight, and be omitted entirely when <= 0.
# ---------------------------------------------------------------------------
def test_isometry_weight_changes_payload(monkeypatch):
    _no_row_block_probe(monkeypatch)
    common = dict(
        ard_per_atom=False,
        decoder_feature_sparsity_groups=None,
        block_orthogonality_weight=0.0,
        d_max=2,
        p_out=3,
    )
    payload_a = sae._build_analytic_penalties_payload(isometry_weight=1.0, **common)
    payload_b = sae._build_analytic_penalties_payload(isometry_weight=2.5, **common)

    assert payload_a is not None and payload_b is not None
    assert payload_a != payload_b, "differing isometry_weight must yield differing payloads"

    items_a = json.loads(payload_a)
    items_b = json.loads(payload_b)
    iso_a = [it for it in items_a if it["kind"] == "isometry"]
    iso_b = [it for it in items_b if it["kind"] == "isometry"]
    assert len(iso_a) == 1 and len(iso_b) == 1
    assert iso_a[0]["target"] == "t"
    assert iso_a[0]["weight"] == 1.0
    assert iso_b[0]["weight"] == 2.5


def test_isometry_weight_zero_omits_descriptor(monkeypatch):
    _no_row_block_probe(monkeypatch)
    payload = sae._build_analytic_penalties_payload(
        isometry_weight=0.0,
        ard_per_atom=False,
        decoder_feature_sparsity_groups=None,
        block_orthogonality_weight=0.0,
        d_max=2,
        p_out=3,
    )
    # No active knobs => no payload at all.
    assert payload is None


# ---------------------------------------------------------------------------
# #606 — X and Z are aliases; differing arrays must raise, identical arrays are
# accepted, and neither-supplied still raises the original TypeError.
# ---------------------------------------------------------------------------
def test_x_neq_z_raises(monkeypatch):
    # Short-circuit before any Rust call: the alias check is the first thing
    # sae_manifold_fit does.
    x = np.arange(12, dtype=float).reshape(6, 2)
    z = x.copy()
    z[0, 0] += 1.0  # make them differ
    with pytest.raises(ValueError, match="X and Z are aliases"):
        sae.sae_manifold_fit(X=x, Z=z, K=2)


def test_x_neq_z_shape_mismatch_raises():
    x = np.zeros((6, 2))
    z = np.zeros((6, 3))
    with pytest.raises(ValueError, match="X and Z are aliases"):
        sae.sae_manifold_fit(X=x, Z=z, K=2)


def test_neither_x_nor_z_raises():
    with pytest.raises(TypeError, match=r"requires Z= \(or X=\)"):
        sae.sae_manifold_fit(K=2)


# ---------------------------------------------------------------------------
# #607 — "ibp" and "ibp_map" canonicalize to the same kind; summary thresholds
# are mode-specific on the canonical kind.
# ---------------------------------------------------------------------------
def test_ibp_aliases_share_canonical_kind():
    a = sae._canonical_assignment("ibp", "assignment")
    b = sae._canonical_assignment("ibp_map", "assignment")
    assert a == b == "ibp_map"


def test_gated_aliases_to_jumprelu():
    assert sae._canonical_assignment("gated", "assignment") == "jumprelu"
    assert sae._canonical_assignment("jumprelu", "assignment") == "jumprelu"


class _StubModule:
    """Captures the threshold passed to the Rust assignment summary."""

    def __init__(self):
        self.threshold = None

    def sae_manifold_assignment_summary(self, assignments, threshold):
        self.threshold = float(threshold)
        return (0.0, 0.0)


def _make_fit(kind: str, n_atoms: int = 4) -> sae.ManifoldSAE:
    """Build a minimal ManifoldSAE with a known canonical kind, no Rust fit."""
    atoms = [
        sae.SaeManifoldAtomFit(
            basis="periodic",
            decoder_coefficients=np.zeros((1, 1)),
            assignments=np.zeros((1,)),
            coords=np.zeros((1, 1)),
            evidence=0.0,
            active_dim=1,
        )
        for _ in range(n_atoms)
    ]
    return sae.ManifoldSAE(
        atoms=atoms,
        atom_topology="circle",
        atom_topologies=["circle"] * n_atoms,
        assignment=kind,
        assignment_label=kind,
        primitive_names=[],
        fitted=np.zeros((1, 1)),
        assignments=np.zeros((1, n_atoms)),
        coords=[np.zeros((1, 1))],
        decoder_blocks=[np.zeros((1, 1))],
        basis_specs=["periodic"] * n_atoms,
        reml_score=0.0,
        reconstruction_r2=0.0,
        training_mean=np.zeros(1),
        training_data=np.zeros((1, 1)),
        low_level=sae.SaeManifoldFitResult(
            atoms, n_atoms, {}, {}, np.zeros((1, 1)), np.zeros((1, n_atoms)), [], 0.0
        ),
        low_level_logits=np.zeros((1, n_atoms)),
        _basis_kinds=["periodic"] * n_atoms,
        _atom_dims=[1] * n_atoms,
        _basis_sizes=[1] * n_atoms,
        _n_harmonics=[0] * n_atoms,
        _duchon_centers=[None] * n_atoms,
    )


@pytest.mark.parametrize(
    "kind,expected_threshold",
    [
        ("softmax", 0.25),   # 1/K with K=4
        ("ibp_map", 0.5),
        ("jumprelu", 0.0),
    ],
)
def test_summary_threshold_mode_specific(monkeypatch, kind, expected_threshold):
    stub = _StubModule()
    monkeypatch.setattr(sae, "rust_module", lambda: stub)
    fit = _make_fit(kind, n_atoms=4)
    fit.summary()
    assert stub.threshold == pytest.approx(expected_threshold)


def test_summary_canonical_kind_for_ibp_label(monkeypatch):
    """A raw 'ibp' label still drives the canonical ibp_map 0.5 threshold."""
    stub = _StubModule()
    monkeypatch.setattr(sae, "rust_module", lambda: stub)
    # Even if the stored label is the raw alias, the canonical field governs.
    fit = _make_fit("ibp_map", n_atoms=4)
    fit.assignment_label = "ibp"  # raw user label
    fit.summary()
    assert stub.threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# #608 — mixed-topology fits report "mixed" and expose per-atom topologies.
# ---------------------------------------------------------------------------
def test_single_topology_collapses():
    assert sae._topology_for_bases(["periodic", "periodic"]) == "circle"
    assert sae._topology_for_bases(["sphere", "sphere", "sphere"]) == "sphere"


def test_mixed_topology_reports_mixed():
    assert sae._topology_for_bases(["periodic", "sphere"]) == "mixed"
    assert sae._topology_for_bases(["sphere", "torus", "duchon"]) == "mixed"


def test_per_atom_topologies_preserved():
    per_atom = sae._topologies_for_bases(["periodic", "sphere", "duchon"])
    assert per_atom == ["circle", "sphere", "euclidean"]


# ---------------------------------------------------------------------------
# #609 — top_k must be within [1, k_atoms]; 0/None disable it.
# ---------------------------------------------------------------------------
def test_top_k_too_large_raises(monkeypatch):
    # Force the analytic-penalty payload (which would touch Rust) to a no-op so
    # the top_k validation downstream is the thing that trips.
    _no_row_block_probe(monkeypatch)
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError, match=r"top_k must be in \[1, K=2\]"):
        sae.sae_manifold_fit(Z=x, K=2, top_k=5)


def test_top_k_negative_raises(monkeypatch):
    _no_row_block_probe(monkeypatch)
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError, match=r"top_k must be in \[1, K=2\]"):
        sae.sae_manifold_fit(Z=x, K=2, top_k=-3)


# ---------------------------------------------------------------------------
# Real-fit guards: only run when the compiled extension is importable. These
# assert the end-to-end metadata wiring (#607 canonical kind, #608 mixed list).
# ---------------------------------------------------------------------------
def _rust_available() -> bool:
    try:
        from gamfit._binding import rust_module

        rust_module()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _rust_available(), reason="compiled gamfit._rust unavailable")
def test_ibp_aliases_same_metadata_e2e():
    x = np.random.default_rng(1).standard_normal((40, 4))
    fit_ibp = sae.sae_manifold_fit(Z=x, K=3, d_atom=2, assignment="ibp", n_iter=2)
    fit_map = sae.sae_manifold_fit(Z=x, K=3, d_atom=2, assignment="ibp_map", n_iter=2)
    assert fit_ibp.assignment == fit_map.assignment == "ibp_map"


@pytest.mark.skipif(not _rust_available(), reason="compiled gamfit._rust unavailable")
def test_mixed_basis_topology_e2e():
    x = np.random.default_rng(2).standard_normal((40, 4))
    fit = sae.sae_manifold_fit(
        Z=x, K=2, d_atom=2, atom_basis=["periodic", "sphere"], n_iter=2
    )
    assert fit.atom_topology == "mixed"
    assert fit.atom_topologies == ["circle", "sphere"]
