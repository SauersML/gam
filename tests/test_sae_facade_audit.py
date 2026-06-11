"""Regression tests for the SAE manifold facade audit (#605-#609).

These exercise the pure-Python validation / payload / metadata logic of
``gamfit._sae_manifold`` directly, so they do not require the compiled Rust
extension. The few places that would touch Rust are monkeypatched to a fake
module so these tests cover Python metadata wiring deterministically.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import gamfit
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
        gate_sparsity="l1",
        sparsity_weight=0.0,
        scad_mcp_gamma=3.7,
        decoder_feature_sparsity_groups=None,
        block_orthogonality_weight=0.0,
        nuclear_norm_weight=0.0,
        nuclear_norm_max_rank=None,
        decoder_incoherence_weight=0.0,
        k_atoms=1,
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
        gate_sparsity="l1",
        sparsity_weight=0.0,
        scad_mcp_gamma=3.7,
        decoder_feature_sparsity_groups=None,
        block_orthogonality_weight=0.0,
        nuclear_norm_weight=0.0,
        nuclear_norm_max_rank=None,
        decoder_incoherence_weight=0.0,
        k_atoms=1,
        d_max=2,
        p_out=3,
    )
    # No active knobs => no payload at all.
    assert payload is None


def test_decoder_incoherence_payload_builder_default_is_on_for_multi_atom(monkeypatch):
    _no_row_block_probe(monkeypatch)
    payload = sae._build_analytic_penalties_payload(
        isometry_weight=0.0,
        ard_per_atom=False,
        gate_sparsity="l1",
        sparsity_weight=0.0,
        scad_mcp_gamma=3.7,
        decoder_feature_sparsity_groups=None,
        block_orthogonality_weight=0.0,
        nuclear_norm_weight=0.0,
        nuclear_norm_max_rank=None,
        k_atoms=2,
        d_max=2,
        p_out=3,
    )
    assert payload is not None
    assert json.loads(payload) == [
        {
            "kind": "decoder_incoherence",
            "target": "beta",
            "block_sizes": [1, 1],
            "p_out": 3,
            "weight": 1.0,
        }
    ]


def test_missing_x_raises():
    with pytest.raises(TypeError, match=r"requires X input array"):
        sae.sae_manifold_fit(K=2)


# ---------------------------------------------------------------------------
# #607 — assignment summary thresholds are mode-specific on the canonical kind.
# ---------------------------------------------------------------------------
def test_ibp_alias_is_rejected():
    with pytest.raises(ValueError, match="not a recognized assignment kind"):
        sae._canonical_assignment("ibp", "assignment")


def test_jumprelu_assignment_is_canonical():
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
    diagnostics = _diagnostics(n_atoms)
    return sae.ManifoldSAE(
        atoms=atoms,
        atom_topology="circle",
        atom_topologies=["circle"] * n_atoms,
        assignment=kind,
        assignment_label=kind,
        primitive_names=[],
        fitted=np.zeros((1, 1)),
        assignments=np.zeros((1, n_atoms)),
        coords=[np.zeros((1, 1)) for _ in range(n_atoms)],
        decoder_blocks=[np.zeros((1, 1)) for _ in range(n_atoms)],
        basis_specs=["periodic"] * n_atoms,
        reml_score=0.0,
        reconstruction_r2=0.0,
        training_mean=np.zeros(1),
        training_data=np.zeros((1, 1)),
        low_level=sae.SaeManifoldFitResult(
            atoms, n_atoms, {}, {}, np.zeros((1, 1)), np.zeros((1, n_atoms)), [], 0.0
        ),
        low_level_logits=np.zeros((1, n_atoms)),
        diagnostics=diagnostics,
        _basis_kinds=["periodic"] * n_atoms,
        _atom_dims=[1] * n_atoms,
        _basis_sizes=[1] * n_atoms,
        _n_harmonics=[0] * n_atoms,
        _duchon_centers=[None] * n_atoms,
    )


def _diagnostics(n_atoms: int, trust: list[float] | None = None) -> dict[str, object]:
    scores = np.ones(n_atoms, dtype=float) if trust is None else np.asarray(trust, dtype=float)
    return {
        "atom_trust": scores,
        "atoms": [
            {
                "trust_score": float(scores[k]),
                "sigma_min_tangent": 1.0,
                "sigma_max_tangent": 1.0,
                "tangent_condition_score": 1.0,
                "coverage": 1.0,
                "activation_frequency": 1.0,
                "untyped": False,
                "active_token_count": 1,
            }
            for k in range(n_atoms)
        ],
    }


class _FakeRustModule:
    def build_info(self) -> dict[str, list[str]]:
        return {
            "sae_row_block_penalties": [
                "ard",
                "isometry",
                "block_orthogonality",
                "scad_mcp",
            ],
        }

    def sae_manifold_reconstruction_r2(self, observed, fitted) -> float:
        observed = np.asarray(observed, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        ss_res = float(np.sum((observed - fitted) ** 2))
        ss_tot = float(np.sum((observed - observed.mean(axis=0, keepdims=True)) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1.0e-12)

    def sae_manifold_fit_minimal(
        self,
        z,
        atom_basis,
        atom_dim,
        alpha,
        tau,
        learnable_alpha,
        assignment_kind,
        *,
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        gumbel_schedule,
        analytic_penalties,
        random_state,
        top_k,
        initial_logits,
        initial_coords,
        jumprelu_threshold,
        native_ard_enabled,
    ):
        self.last_native_ard_enabled = bool(native_ard_enabled)
        z = np.asarray(z, dtype=float)
        n_obs, p_out = z.shape
        k_atoms = len(atom_basis)
        logits = np.zeros((n_obs, k_atoms), dtype=float)
        assignments = np.full((n_obs, k_atoms), 1.0 / float(k_atoms), dtype=float)
        atoms = []
        for atom_k, basis in enumerate(atom_basis):
            dim = int(atom_dim[atom_k])
            atoms.append({
                "basis_kind": str(basis),
                "decoder_B": np.zeros((max(1, dim + 1), p_out), dtype=float),
                "assignments_z": assignments[:, atom_k],
                "on_atom_coords_t": np.zeros((n_obs, dim), dtype=float),
                "active_dim": dim,
            })
        return {
            "atoms": atoms,
            "atom_plans": [
                {
                    "kind": str(atom_basis[atom_k]),
                    "latent_dim": int(atom_dim[atom_k]),
                    "basis_size": int(atoms[atom_k]["decoder_B"].shape[0]),
                    "n_harmonics": 0,
                    "duchon_centers": None,
                }
                for atom_k in range(k_atoms)
            ],
            "assignments_z": assignments,
            "logits": logits,
            "fitted": np.zeros_like(z),
            "reml_score": -1.0,
            "chosen_k": k_atoms,
            "dispersion": 1.0,
            "oos_projection_top1": False,
            "diagnostics": _diagnostics(k_atoms),
        }


def test_new_sae_helpers_are_importable_and_defaults_are_research_objective():
    assert gamfit.sae_fit is sae.fit
    assert gamfit.align is sae.align
    assert gamfit.plot is sae.plot
    assert callable(gamfit.sae_trust_diagnostics)
    assert callable(gamfit.atom_trust_scores)

    signature = inspect.signature(sae.sae_manifold_fit)
    assert signature.parameters["gate_sparsity"].default == "scad"
    assert signature.parameters["nuclear_norm_weight"].default == 1.0
    assert signature.parameters["decoder_incoherence_weight"].default == 1.0


def test_trust_diagnostics_normalize_and_round_trip():
    diagnostics = _diagnostics(2, trust=[0.25, 0.75])
    normalized = gamfit.sae_trust_diagnostics({"diagnostics": diagnostics})
    np.testing.assert_allclose(gamfit.atom_trust_scores(normalized), [0.25, 0.75])
    assert normalized["atoms"][0]["trust_score"] == pytest.approx(0.25)
    assert normalized["atoms"][1]["trust_score"] == pytest.approx(0.75)
    assert set(normalized) == {"atom_trust", "atoms"}
    deleted = {
        "mean_neighbor_coherence",
        "coherence_score",
        "topology_evidence_margin",
        "topology_margin_score",
        "coverage_score",
        "typed_reconstruction_mse",
        "level0_reference_mse",
        "level0_residual_ratio",
        "level0_score",
    }
    assert deleted.isdisjoint(normalized["atoms"][0])
    stale = _diagnostics(1)
    stale["atoms"][0]["level0_score"] = 1.0
    with pytest.raises(ValueError, match="extra=\\['level0_score'\\]"):
        gamfit.sae_trust_diagnostics({"diagnostics": stale})


def test_small_sae_fit_trust_diagnostics_round_trip_new_schema():
    x = np.random.default_rng(1005).normal(size=(10, 3))
    fit = sae.sae_manifold_fit(
        X=x,
        K=1,
        d_atom=1,
        atom_basis="periodic",
        assignment="softmax",
        n_iter=1,
        random_state=0,
        isometry_weight=0.0,
        ard_per_atom=False,
        gate_sparsity="l1",
        nuclear_norm_weight=0.0,
        decoder_incoherence_weight=0.0,
    )
    expected_atom_keys = {
        "trust_score",
        "sigma_min_tangent",
        "sigma_max_tangent",
        "tangent_condition_score",
        "coverage",
        "activation_frequency",
        "untyped",
        "active_token_count",
    }
    assert set(fit.diagnostics) == {"atom_trust", "atoms"}
    assert set(fit.diagnostics["atoms"][0]) == expected_atom_keys
    restored = sae.ManifoldSAE.from_dict(fit.to_dict())
    assert set(restored.diagnostics) == {"atom_trust", "atoms"}
    assert set(restored.diagnostics["atoms"][0]) == expected_atom_keys
    np.testing.assert_allclose(
        gamfit.atom_trust_scores(restored.diagnostics),
        gamfit.atom_trust_scores(fit.diagnostics),
    )


def test_research_trust_scores_are_assignment_weighted():
    fit = _make_fit("softmax", K=2)
    fit.training_data = np.zeros((3, 1))
    fit.assignments = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    fit.diagnostics = _diagnostics(2, trust=[0.2, 0.8])

    trust = sae._trust_scores(fit)
    np.testing.assert_allclose(trust["atom"], [0.2, 0.8])
    np.testing.assert_allclose(trust["row"], [0.2, 0.8, 0.5])
    assert trust["per_atom"].shape == (3, 2)


def test_alignment_public_api_uses_rich_result():
    fit_a = _make_fit("softmax", K=2)
    fit_b = _make_fit("softmax", K=2)
    fit_a.decoder_blocks = [np.eye(2), np.fliplr(np.eye(2))]
    fit_b.decoder_blocks = [np.fliplr(np.eye(2)), np.eye(2)]
    for atom, block in zip(fit_a.atoms, fit_a.decoder_blocks):
        atom.decoder_coefficients = block
    for atom, block in zip(fit_b.atoms, fit_b.decoder_blocks):
        atom.decoder_coefficients = block

    aligned = gamfit.align(fit_a, fit_b)
    assert aligned.assignment == [(0, 1), (1, 0)]
    payload = aligned.to_dict()
    assert payload["assignment"] == [[0, 1], [1, 0]]
    assert "mean_grassmann_distance" in payload["summary"]


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
    fit = _make_fit(kind, K=4)
    fit.summary()
    assert stub.threshold == pytest.approx(expected_threshold)


def test_summary_canonical_kind_for_ibp_map_label(monkeypatch):
    """The canonical ibp_map label drives the 0.5 threshold."""
    stub = _StubModule()
    monkeypatch.setattr(sae, "rust_module", lambda: stub)
    # Even if the stored label is the raw alias, the canonical field governs.
    fit = _make_fit("ibp_map", n_atoms=4)
    fit.assignment_label = "ibp_map"
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
        sae.sae_manifold_fit(X=x, K=2, top_k=5)


def test_top_k_negative_raises(monkeypatch):
    _no_row_block_probe(monkeypatch)
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError, match=r"top_k must be in \[1, K=2\]"):
        sae.sae_manifold_fit(X=x, K=2, top_k=-3)


# ---------------------------------------------------------------------------
# Metadata wiring tests use the fake Rust module above so they are deterministic
# and do not depend on the current convergence state of the compiled solver.
# ---------------------------------------------------------------------------

def test_ibp_map_metadata_e2e(monkeypatch):
    fake = _FakeRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    x = np.random.default_rng(1).standard_normal((40, 4))
    baseline = dict(
        isometry_weight=0.0,
        ard_per_atom=False,
        gate_sparsity="l1",
        nuclear_norm_weight=0.0,
        decoder_incoherence_weight=0.0,
    )
    fit_map = sae.sae_manifold_fit(
        X=x, K=3, d_atom=2, assignment="ibp_map", n_iter=2, **baseline
    )
    assert fit_map.assignment == "ibp_map"


def test_mixed_basis_topology_e2e(monkeypatch):
    fake = _FakeRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    x = np.random.default_rng(2).standard_normal((40, 4))
    fit = sae.sae_manifold_fit(
        X=x,
        K=2,
        d_atom=2,
        atom_basis=["periodic", "sphere"],
        n_iter=2,
        isometry_weight=0.0,
        ard_per_atom=False,
        gate_sparsity="l1",
        nuclear_norm_weight=0.0,
        decoder_incoherence_weight=0.0,
    )
    assert fit.atom_topology == "mixed"
    assert fit.atom_topologies == ["circle", "sphere"]


def test_ard_per_atom_controls_native_ard_plumbing(monkeypatch):
    fake = _FakeRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    x = np.random.default_rng(3).standard_normal((24, 3))

    sae.sae_manifold_fit(
        X=x,
        K=2,
        d_atom=1,
        n_iter=2,
        isometry_weight=0.0,
        ard_per_atom=False,
        gate_sparsity="l1",
        nuclear_norm_weight=0.0,
        decoder_incoherence_weight=0.0,
    )
    assert fake.last_native_ard_enabled is False

    sae.sae_manifold_fit(
        X=x,
        K=2,
        d_atom=1,
        n_iter=2,
        isometry_weight=0.0,
        ard_per_atom=True,
        gate_sparsity="l1",
        nuclear_norm_weight=0.0,
        decoder_incoherence_weight=0.0,
    )
    assert fake.last_native_ard_enabled is True
