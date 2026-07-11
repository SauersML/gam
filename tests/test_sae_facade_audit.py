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
        coord_sparsity="l1",
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
        coord_sparsity="l1",
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
        coord_sparsity="l1",
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
def test_assignment_schema_accepts_only_canonical_tokens():
    for token in ("softmax", "ordered_beta_bernoulli", "threshold_gate", "topk"):
        assert sae._canonical_assignment(token, "assignment") == token
    for alias in ("ibp", "top-k", "gated", "jumprelu"):
        with pytest.raises(ValueError):
            sae._canonical_assignment(alias, "assignment")


def test_threshold_gate_assignment_is_canonical():
    assert sae._canonical_assignment("threshold_gate", "assignment") == "threshold_gate"


class _StubModule:
    """Captures the threshold passed to the Rust assignment summary."""

    def __init__(self):
        self.threshold = None

    def sae_manifold_assignment_summary(self, assignments, threshold):
        self.threshold = float(threshold)
        return (0.0, 0.0)

    def sae_canonical_assignment_kind(self, assignment):
        return _CANONICAL_ASSIGNMENT(assignment)


_MANIFOLD_SAE_CORE = sae.rust_module().ManifoldSaeCore
_TOPOLOGY_FOR_BASIS = sae.rust_module().sae_topology_for_basis
_CANONICAL_ASSIGNMENT = sae.rust_module().sae_canonical_assignment_kind


def _make_fit(
    kind: str,
    n_atoms: int = 4,
    *,
    assignments: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
    basis_kinds: list[str] | None = None,
    decoder_blocks: list[np.ndarray] | None = None,
    functional_evidence: list[dict[str, object] | None] | None = None,
    topology_persistence: dict[str, object] | None = None,
) -> sae.ManifoldSAE:
    """Build a real Rust-owned model handle without running a fit."""
    kinds = ["euclidean"] * n_atoms if basis_kinds is None else list(basis_kinds)
    if len(kinds) != n_atoms:
        raise ValueError("basis_kinds length must equal n_atoms")
    blocks = (
        [np.zeros((1, 1), dtype=float) for _ in range(n_atoms)]
        if decoder_blocks is None
        else [np.asarray(block, dtype=float) for block in decoder_blocks]
    )
    if len(blocks) != n_atoms:
        raise ValueError("decoder_blocks length must equal n_atoms")
    n_rows = 1 if assignments is None else int(np.asarray(assignments).shape[0])
    p_out = 1 if not blocks else int(blocks[0].shape[1])
    assignment_values = (
        np.zeros((n_rows, n_atoms), dtype=float)
        if assignments is None
        else np.asarray(assignments, dtype=float)
    )
    if assignment_values.shape != (n_rows, n_atoms):
        raise ValueError("assignments must be (n_rows, n_atoms)")
    dims = [1] * n_atoms
    coords = [np.zeros((n_rows, 1), dtype=float) for _ in range(n_atoms)]
    evidence = [None] * n_atoms if functional_evidence is None else functional_evidence
    atoms = [
        {
            "basis": kinds[k],
            "decoder_coefficients": blocks[k].tolist(),
            "assignments": assignment_values[:, k].tolist(),
            "coords": coords[k].tolist(),
            "coords_u_arc": None,
            "evidence": 0.0,
            "active_dim": 1,
            "decoder_covariance_channel_factors": None,
            "shape_band_coords": None,
            "shape_band_mean": None,
            "shape_band_sd": None,
            "functional_evidence": evidence[k],
        }
        for k in range(n_atoms)
    ]
    topologies = [str(_TOPOLOGY_FOR_BASIS(name)) for name in kinds]
    payload = {
        "schema": "gamfit.ManifoldSAE/v2",
        "atom_topology": topologies[0] if topologies and len(set(topologies)) == 1 else "mixed",
        "atom_topologies": topologies,
        "assignment": kind,
        "assignment_label": kind,
        "alpha": 1.0,
        "learnable_alpha": False,
        "tau": 0.5,
        "sparsity_strength": 1.0,
        "smoothness": 1.0,
        "learning_rate": 1.0,
        "max_iter": 1,
        "random_state": 0,
        "top_k": None,
        "threshold_gate_threshold": 0.0,
        "oos_projection_top1": False,
        "dispersion": 1.0,
        "penalized_loss_score": 0.0,
        "reml_score": 0.0,
        "reconstruction_r2": 0.0,
        "primitive_names": [],
        "basis_specs": kinds,
        "basis_kinds": kinds,
        "atom_dims": dims,
        "basis_sizes": [int(block.shape[0]) for block in blocks],
        "n_harmonics": [0] * n_atoms,
        "training_mean": [0.0] * p_out,
        "training_data": None,
        "training_data_retained": False,
        "fitted": np.zeros((n_rows, p_out), dtype=float).tolist(),
        "assignments": assignment_values.tolist(),
        "logits": np.zeros((n_rows, n_atoms), dtype=float).tolist(),
        "coords": [coord.tolist() for coord in coords],
        "decoder_blocks": [block.tolist() for block in blocks],
        "duchon_centers": [None] * n_atoms,
        "crosscoder": None,
        "atoms": atoms,
        "diagnostics": sae._json_ready(
            _diagnostics(n_atoms) if diagnostics is None else diagnostics
        ),
        "top_k_projection": None,
        "pre_topk": None,
        "solver_plan": None,
        "atom_two_lens": None,
        "residual_gauge": None,
        "incoherence_report": None,
        "curvature_report": None,
        "coordinate_fidelity": None,
        "topology_persistence": topology_persistence,
        "atom_inference": None,
        "certificates": None,
        "structure_certificate": None,
        "cotrain": None,
        "hybrid_split": None,
        "fisher_factors": None,
        "fisher_provenance": None,
        "metric_provenance": "Euclidean",
        "fisher_mass_residual": None,
        "selected_log_lambda_sparse": 0.0,
        "selected_log_lambda_smooth": [0.0] * n_atoms,
        "selected_log_ard": [[] for _ in range(n_atoms)],
    }
    return sae.ManifoldSAE(
        _MANIFOLD_SAE_CORE(payload),
        training_data=np.zeros((n_rows, p_out), dtype=float),
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

    def basis_with_jet(self, kind, coords, params):
        """Minimal real basis plus its coordinate jet.

        For ``periodic`` atoms this is the Fourier design (width 2H+1) exercised
        by the `_periodic_shape_band` round-trip. For any other manifold kind
        (e.g. ``sphere`` in the mixed-topology fit) a generic affine design
        ``[1, t_1, ..., t_d]`` (width 1+d) plus its identity jet is returned; the
        exact analytic basis is not required to keep this fake fit's surface
        contract deterministic.
        """
        coords = np.asarray(coords, dtype=float)
        if str(kind) == "periodic":
            theta = 2.0 * np.pi * coords[:, 0]
            h = int(params["n_harmonics"])
            cols = [np.ones_like(theta)]
            jet_cols = [np.zeros_like(theta)]
            for m in range(1, h + 1):
                cols.append(np.cos(m * theta))
                cols.append(np.sin(m * theta))
                jet_cols.append(-2.0 * np.pi * m * np.sin(m * theta))
                jet_cols.append(2.0 * np.pi * m * np.cos(m * theta))
            phi = np.stack(cols, axis=1)
            jet = np.stack(jet_cols, axis=1)[:, :, None]
            penalty = np.zeros((phi.shape[1], phi.shape[1]), dtype=float)
            return phi, jet, penalty
        n_obs, dim = coords.shape
        phi = np.concatenate([np.ones((n_obs, 1)), coords], axis=1)
        jet = np.zeros((n_obs, dim + 1, dim), dtype=float)
        for axis in range(dim):
            jet[:, axis + 1, axis] = 1.0
        penalty = np.zeros((phi.shape[1], phi.shape[1]), dtype=float)
        return phi, jet, penalty

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
        random_state,
        top_k,
        gumbel_schedule=None,
        analytic_penalties=None,
        initial_logits=None,
        initial_coords=None,
        threshold_gate_threshold=0.0,
        native_ard_enabled=True,
        **_forward_compat_kwargs,
    ):
        # #240: `ard_per_atom` controls ARD via the dedicated `native_ard_enabled`
        # FFI flag (the switch that sizes/drops each atom's `log_ard` precisions),
        # NOT via a registry `{"kind": "ard"}` descriptor. The registry `ard`
        # penalty is deliberately skipped on every SAE objective path, so plumbing
        # the flag through the descriptor was a silent no-op (issue #240). Read the
        # real flag the facade now forwards.
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
            "penalized_loss_score": -1.0,
            "chosen_k": k_atoms,
            "dispersion": 1.0,
            "oos_projection_top1": False,
            "diagnostics": _diagnostics(k_atoms),
        }


def test_oos_payload_threads_trained_basis_sizes(monkeypatch):
    class _OosFake:
        def __init__(self):
            self.basis_sizes = None

        def sae_canonical_assignment_kind(self, assignment):
            return _CANONICAL_ASSIGNMENT(assignment)

        def sae_manifold_predict_oos(self, *args, **kwargs):
            x_new = np.asarray(args[0], dtype=float)
            decoder_blocks = args[3]
            self.basis_sizes = list(args[6])
            return {
                "assignments_z": np.ones((x_new.shape[0], 1), dtype=float),
                "on_atom_coords_t": [np.zeros((x_new.shape[0], 1), dtype=float)],
                "logits": np.zeros((x_new.shape[0], 1), dtype=float),
                "fitted": np.zeros((x_new.shape[0], decoder_blocks[0].shape[1]), dtype=float),
            }

    fit = _make_fit(
        "softmax",
        n_atoms=1,
        basis_kinds=["euclidean"],
        decoder_blocks=[np.zeros((1, 2), dtype=float)],
    )
    fake = _OosFake()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)

    reconstructed = fit.reconstruct(np.ones((2, 2), dtype=float))

    assert fake.basis_sizes == [1]
    assert reconstructed.shape == (2, 2)


def test_new_sae_helpers_are_importable_and_defaults_are_research_objective():
    assert gamfit.sae_fit is sae.fit
    assert not hasattr(gamfit, "featurize")
    assert gamfit.plot is sae.plot
    assert callable(gamfit.sae_trust_diagnostics)
    assert callable(gamfit.atom_trust_scores)

    signature = inspect.signature(sae.sae_manifold_fit)
    assert signature.parameters["coord_sparsity"].default == "scad"
    assert "gate_sparsity" not in signature.parameters
    assert "n_atoms" not in signature.parameters
    assert signature.parameters["nuclear_norm_weight"].default == 1.0
    assert signature.parameters["decoder_incoherence_weight"].default == 1.0

    stagewise_signature = inspect.signature(sae.sae_manifold_fit_stagewise)
    assert stagewise_signature.parameters["assignment"].default == "softmax"


def test_research_fit_returns_each_explicit_model_handle(monkeypatch):
    first_model = object()
    second_model = object()
    fitted = iter((first_model, second_model))

    def fake_fit(_x, **_config):
        return next(fitted)

    monkeypatch.setattr(sae, "sae_manifold_fit", fake_fit)
    first = sae.fit(np.zeros((3, 1)), {"K": 1})
    second = sae.fit(np.ones((3, 1)), {"K": 1})

    assert first is first_model
    assert second is second_model
    assert not hasattr(sae, "_LAST_RESEARCH_LOOP_MODEL")
    assert not hasattr(sae, "featurize")


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
        coord_sparsity="l1",
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
    assignments = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    fit = _make_fit(
        "softmax",
        n_atoms=2,
        assignments=assignments,
        diagnostics=_diagnostics(2, trust=[0.2, 0.8]),
    )

    trust = fit.trust_scores()
    np.testing.assert_allclose(trust["atom"], [0.2, 0.8])
    np.testing.assert_allclose(trust["row"], [0.2, 0.8, 0.5])
    assert trust["per_atom"].shape == (3, 2)


@pytest.mark.parametrize(
    "kind,expected_threshold",
    [
        ("softmax", 0.25),   # 1/K with K=4
        # #1547: the ordered_beta_bernoulli summary threshold is a small responsibility-mass
        # epsilon (1e-8), not a 0.5 gate bar — `assignments_z` are normalized
        # reconstruction responsibilities that cannot reach 0.5 once K>=2.
        ("ordered_beta_bernoulli", 1.0e-8),
        ("jumprelu", 0.0),
    ],
)
def test_summary_threshold_mode_specific(monkeypatch, kind, expected_threshold):
    stub = _StubModule()
    monkeypatch.setattr(sae, "rust_module", lambda: stub)
    fit = _make_fit(kind, n_atoms=4)
    fit.summary()
    assert stub.threshold == pytest.approx(expected_threshold)


def test_summary_canonical_kind_for_ordered_beta_bernoulli_label(monkeypatch):
    """The canonical ordered_beta_bernoulli label drives the responsibility-mass threshold."""
    stub = _StubModule()
    monkeypatch.setattr(sae, "rust_module", lambda: stub)
    fit = _make_fit("ordered_beta_bernoulli", n_atoms=4)
    fit.summary()
    assert stub.threshold == pytest.approx(1.0e-8)


def test_summary_and_roundtrip_surface_atom_functional_evidence():
    evidence = {
        "source": "riesz",
        "marginal_slope": {"estimate": [0.5], "se": [0.1], "norm": 0.5},
        "average_derivative": {"estimate": [[0.5]], "se": [[0.1]], "norm": 0.5},
        "peak_contrast": {
            "estimate": [1.2],
            "se": [0.2],
            "norm": 1.2,
            "from_coord": [0.0],
            "to_coord": [1.0],
        },
    }
    fit = _make_fit("softmax", n_atoms=1, functional_evidence=[evidence])

    summary = fit.summary()
    assert summary["atom_functionals"] == [evidence]

    payload = fit.to_dict()
    assert payload["atoms"][0]["functional_evidence"] == evidence

    restored = sae.ManifoldSAE.from_dict(payload)
    assert restored.atoms[0].functional_evidence == evidence


def test_topology_persistence_report_surfaces_covering_side():
    persistence = {
        "atoms": [
            {
                "atom": 0,
                "raced_kind": "periodic",
                "support_size": 48,
                "landmark_count": 48,
                "stability_band": "below_landmark_cap",
                "covering_side": "at_or_above_covering_number",
                "measured_betti": {"b0": 1, "b1": 1, "b2": None},
                "expected_betti": {"b0": 1, "b1": 1, "b2": None},
                "inferred_kind": "loop",
                "contested": False,
            }
        ]
    }
    fit = _make_fit("softmax", n_atoms=1, topology_persistence=persistence)

    row = fit.topology_persistence_report()[0]
    assert row["covering_side"] == "at_or_above_covering_number"
    assert row["measured_betti"] == {"b0": 1, "b1": 1, "b2": None}
    assert row["expected_betti"] == {"b0": 1, "b1": 1, "b2": None}


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
    _scalar, per_atom = sae.rust_module().sae_atom_topologies(
        ["periodic", "sphere", "duchon"]
    )
    assert per_atom == ["circle", "sphere", "euclidean"]


# ---------------------------------------------------------------------------
# #609 — top_k must be within [1, k_atoms]; None disables it.
# ---------------------------------------------------------------------------
def test_top_k_too_large_raises(monkeypatch):
    # Force the analytic-penalty payload (which would touch Rust) to a no-op so
    # the top_k validation downstream is the thing that trips.
    _no_row_block_probe(monkeypatch)
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError, match=r"top_k must be in \[1, K=2\]"):
        sae.sae_manifold_fit(X=x, K=2, top_k=5)


@pytest.mark.parametrize("invalid_top_k", [-3, 0])
def test_top_k_nonpositive_raises(monkeypatch, invalid_top_k):
    _no_row_block_probe(monkeypatch)
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError, match=r"top_k must be in \[1, K=2\]"):
        sae.sae_manifold_fit(X=x, K=2, top_k=invalid_top_k)


# ---------------------------------------------------------------------------
# Metadata wiring tests use the fake Rust module above so they are deterministic
# and do not depend on the current convergence state of the compiled solver.
# ---------------------------------------------------------------------------

def test_ordered_beta_bernoulli_metadata_e2e(monkeypatch):
    fake = _FakeRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    x = np.random.default_rng(1).standard_normal((40, 4))
    baseline = dict(
        isometry_weight=0.0,
        ard_per_atom=False,
        coord_sparsity="l1",
        nuclear_norm_weight=0.0,
        decoder_incoherence_weight=0.0,
    )
    fit_map = sae.sae_manifold_fit(
        X=x, K=3, d_atom=2, assignment="ordered_beta_bernoulli", n_iter=2, **baseline
    )
    assert fit_map.assignment == "ordered_beta_bernoulli"


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
        coord_sparsity="l1",
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
        coord_sparsity="l1",
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
        coord_sparsity="l1",
        nuclear_norm_weight=0.0,
        decoder_incoherence_weight=0.0,
    )
    assert fake.last_native_ard_enabled is True


# ---------------------------------------------------------------------------
# #977 — VARIABLE-K discovery boundary. The structure search may GROW K via
# evidence-gated births; the Rust producer re-derives every per-atom field from
# the post-search dictionary so each has length == discovered K. A born atom
# (index >= seed K) carries NO posterior shape-band keys (its uncertainty came
# from the pre-search Schur factor, which is indexed by the seed dictionary), so
# `from_payload` must read those bands as optional and return `None` rather than
# panic, and the scalar `atom_topology` must collapse to "mixed" when births
# made the dictionary heterogeneous.
# ---------------------------------------------------------------------------
def _grown_k_payload(n_obs: int, p_out: int) -> dict[str, object]:
    """Seed = 2 periodic atoms; discovered = 3 (a born euclidean atom).

    The born atom (index 2) deliberately omits every `shape_band_*` /
    `decoder_covariance` key, mirroring the Rust producer's omission for an atom
    with no entry in the seed-indexed `shape_uncertainty`. All length-K fields
    (`atoms`, `atom_plans`, `assignments_z`, `logits`, `chosen_k`) carry the
    DISCOVERED K = 3.
    """
    rng = np.random.default_rng(977)
    k = 3
    assignments = rng.random((n_obs, k))
    assignments /= assignments.sum(axis=1, keepdims=True)
    logits = np.log(np.clip(assignments, 1e-9, None))
    kinds = ["periodic", "periodic", "euclidean"]
    dims = [1, 1, 2]
    sizes = [3, 3, 4]
    nharm = [1, 1, 0]
    atoms = []
    for idx in range(k):
        atom = {
            "basis_kind": kinds[idx],
            "decoder_B": rng.standard_normal((sizes[idx], p_out)),
            "assignments_z": assignments[:, idx],
            "on_atom_coords_t": rng.standard_normal((n_obs, dims[idx])),
            "active_dim": dims[idx],
        }
        # Only the two SEED atoms carry posterior bands; the born atom omits them.
        if idx < 2:
            grid = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
            atom["shape_band_coords"] = grid
            atom["shape_band_mean"] = rng.standard_normal((grid.shape[0], p_out))
            atom["shape_band_sd"] = np.abs(rng.standard_normal((grid.shape[0], p_out)))
        atoms.append(atom)
    return {
        "atoms": atoms,
        "atom_plans": [
            {
                "kind": kinds[idx],
                "latent_dim": dims[idx],
                "basis_size": sizes[idx],
                "n_harmonics": nharm[idx],
                "duchon_centers": None,
            }
            for idx in range(k)
        ],
        "assignments_z": assignments,
        "logits": logits,
        "fitted": rng.standard_normal((n_obs, p_out)),
        "reml_score": -2.0,
        "penalized_loss_score": -2.0,
        "chosen_k": k,
        "dispersion": 1.0,
        "oos_projection_top1": False,
        "diagnostics": _diagnostics(k),
    }


def test_grown_k_payload_round_trips_through_from_payload(monkeypatch):
    monkeypatch.setattr(sae, "rust_module", lambda: _FakeRustModule())
    n_obs, p_out = 16, 3
    x = np.random.default_rng(1).standard_normal((n_obs, p_out))
    payload = _grown_k_payload(n_obs, p_out)
    # The seed dictionary was 2 periodic atoms; `topology` is the SEED scalar.
    model = sae.ManifoldSAE.from_payload(
        x, payload, topology="circle", assignment="softmax", penalties=[]
    )
    # Discovered K (= 3) threads through every per-atom surface.
    assert len(model.atoms) == 3
    assert model.low_level.chosen_k == 3
    assert model.assignments.shape == (n_obs, 3)
    assert model.low_level_logits.shape == (n_obs, 3)
    assert len(model.coords) == 3
    assert len(model.decoder_blocks) == 3
    assert model.basis_specs == ["periodic", "periodic", "euclidean"]
    # The scalar topology is re-derived from the POST-search kinds: a born
    # euclidean atom makes the dictionary heterogeneous, so the honest scalar
    # collapses to "mixed" even though the seed argument said "circle".
    assert model.atom_topologies == ["circle", "circle", "euclidean"]
    assert model.atom_topology == "mixed"
    # The two seed atoms carry their posterior bands; the born atom (index 2) has
    # `None` bands rather than a stale read or a panic.
    assert model.atoms[0].shape_band_coords is not None
    assert model.atoms[1].shape_band_mean is not None
    assert model.atoms[2].shape_band_coords is None
    assert model.atoms[2].shape_band_mean is None
    assert model.atoms[2].shape_band_sd is None
    assert model.atoms[2].decoder_covariance is None


def test_from_payload_rejects_plans_atoms_length_mismatch(monkeypatch):
    monkeypatch.setattr(sae, "rust_module", lambda: _FakeRustModule())
    x = np.random.default_rng(2).standard_normal((16, 3))
    payload = _grown_k_payload(16, 3)
    # Drop one plan so atom_plans (2) disagrees with atoms (3) — the producer-side
    # contract violation a grown K could surface as an opaque IndexError.
    payload["atom_plans"] = payload["atom_plans"][:2]
    with pytest.raises(ValueError, match="atom_plans"):
        sae.ManifoldSAE.from_payload(
            x, payload, topology="circle", assignment="softmax", penalties=[]
        )


def test_from_payload_rejects_chosen_k_mismatch(monkeypatch):
    monkeypatch.setattr(sae, "rust_module", lambda: _FakeRustModule())
    x = np.random.default_rng(3).standard_normal((16, 3))
    payload = _grown_k_payload(16, 3)
    payload["chosen_k"] = 2  # stale seed K vs 3 emitted atoms
    with pytest.raises(ValueError, match="chosen_k"):
        sae.ManifoldSAE.from_payload(
            x, payload, topology="circle", assignment="softmax", penalties=[]
        )
