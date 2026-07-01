"""Recent SAE-manifold facade surface tests.

These tests keep the Python-facing wiring runnable without depending on a
long, converged SAE solve. Rust-facing calls use a fake module so the surface
contract is deterministic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
import gamfit._sae_manifold as sae


def _toy_matrix(n: int = 12, p: int = 4) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, n, dtype=float)
    cols = [grid, grid**2, np.sin(np.pi * grid), np.cos(np.pi * grid)]
    return np.column_stack(cols[:p])


class _CapturingRustModule:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def build_info(self) -> dict[str, object]:
        return {
            "sae_row_block_penalties": [
                "ard",
                "block_orthogonality",
                "isometry",
                "scad_mcp",
            ]
        }

    def sae_manifold_reconstruction_r2(self, observed, fitted) -> float:
        observed = np.asarray(observed, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        ss_res = float(np.sum((observed - fitted) ** 2))
        ss_tot = float(np.sum((observed - observed.mean(axis=0, keepdims=True)) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1.0e-12)

    def periodic_basis_with_jet(self, t, n_harmonics):
        x = np.asarray(t, dtype=float)
        columns = [np.ones_like(x)]
        jacobian_columns = [np.zeros_like(x)]
        penalties = [0.0]
        for harmonic in range(1, int(n_harmonics) + 1):
            omega = 2.0 * np.pi * harmonic
            columns.extend([np.cos(omega * x), np.sin(omega * x)])
            jacobian_columns.extend([-omega * np.sin(omega * x), omega * np.cos(omega * x)])
            penalties.extend([omega**4, omega**4])
        phi = np.stack(columns, axis=1)
        jet = np.stack(jacobian_columns, axis=1)[:, :, None]
        penalty = np.diag(penalties)
        return phi, jet, penalty

    def basis_with_jet(self, kind, coords, params=None):
        params = params or {}
        n_harmonics = int(params.get("n_harmonics", 2))
        t = np.asarray(coords, dtype=float).reshape(-1)
        return self.periodic_basis_with_jet(t, n_harmonics)

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
        jumprelu_threshold=0.0,
        **_forward_compat_kwargs,
    ):
        z = np.asarray(z, dtype=float)
        k_atoms = len(atom_basis)
        n, p = z.shape
        self.calls.append(
            {
                "atom_basis": list(atom_basis),
                "atom_dim": list(atom_dim),
                "alpha": float(alpha),
                "tau": float(tau),
                "learnable_alpha": bool(learnable_alpha),
                "assignment_kind": str(assignment_kind),
                "sparsity_strength": float(sparsity_strength),
                "smoothness": float(smoothness),
                "max_iter": int(max_iter),
                "learning_rate": float(learning_rate),
                "gumbel_schedule": gumbel_schedule,
                "analytic_penalties": analytic_penalties,
                "random_state": int(random_state),
                "top_k": top_k,
                "initial_logits": initial_logits,
                "initial_coords": initial_coords,
                "jumprelu_threshold": float(jumprelu_threshold),
            }
        )

        logits = np.full((n, k_atoms), 0.2, dtype=float)
        if assignment_kind == "softmax":
            assignments = np.full((n, k_atoms), 1.0 / float(k_atoms), dtype=float)
        else:
            assignments = 1.0 / (1.0 + np.exp(-logits / max(float(tau), 1.0e-6)))

        atoms = []
        for atom_k, dim in enumerate(atom_dim):
            dim = int(dim)
            coords = np.tile(
                np.linspace(0.1, 0.9, n, dtype=float)[:, None],
                (1, dim),
            )
            decoder = np.full((3, p), 0.05 * float(atom_k + 1), dtype=float)
            # Periodic shape bands carry a single 1-D phase coordinate column
            # regardless of the atom's latent dim (from_payload's
            # `_periodic_shape_band` requires width 1).
            band_coords = np.linspace(0.1, 0.9, 5, dtype=float)[:, None]
            band_mean = np.full((band_coords.shape[0], p), 0.1 * float(atom_k + 1))
            band_sd = np.full((band_coords.shape[0], p), 0.01 * float(atom_k + 1))
            atoms.append(
                {
                    "basis_kind": str(atom_basis[atom_k]),
                    "decoder_B": decoder,
                    "assignments_z": assignments[:, atom_k],
                    "on_atom_coords_t": coords,
                    "active_dim": dim,
                    "decoder_covariance": np.eye(decoder.size),
                    "shape_band_coords": band_coords,
                    "shape_band_mean": band_mean,
                    "shape_band_sd": band_sd,
                }
            )

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
            "diagnostics": {
                "atom_trust": np.ones(k_atoms),
                "atoms": [
                    {
                        "trust_score": 1.0,
                        "sigma_min_tangent": 1.0,
                        "sigma_max_tangent": 1.0,
                        "tangent_condition_score": 1.0,
                        "coverage": 1.0,
                        "activation_frequency": 1.0,
                        "untyped": False,
                        "active_token_count": n,
                    }
                    for _ in range(k_atoms)
                ],
            },
            "curvature_report": {
                "note": "fake SAE curvature report",
                "atoms": [
                    {
                        "atom": atom_k,
                        "kappa_hat": 0.1 * float(atom_k + 1),
                    }
                    for atom_k in range(k_atoms)
                ],
            },
        }


def _captured_penalties(fake: _CapturingRustModule) -> list[dict[str, object]]:
    raw = fake.calls[-1]["analytic_penalties"]
    if raw is None:
        return []
    return json.loads(str(raw))


def test_recent_penalty_knobs_emit_expected_analytic_descriptors(monkeypatch):
    fake = _CapturingRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    # Every fit now routes through the `sae_manifold_fit_minimal` FFI (the former
    # numpy closed-form fast paths were removed), so the `analytic_penalties`
    # payload is always observed on the captured call.
    x = _toy_matrix(n=14, p=4)

    fit = gamfit.sae_manifold_fit(
        X=x,
        K=2,
        d_atom=2,
        atom_topology="circle",
        assignment="ibp_map",
        isometry_weight=0.0,
        ard_per_atom=False,
        sparsity_weight=0.3,
        gate_sparsity="scad",
        scad_mcp_gamma=4.2,
        nuclear_norm_weight=0.4,
        nuclear_norm_max_rank=1,
        decoder_incoherence_weight=0.7,
        n_iter=1,
        random_state=11,
    )

    assert fit.assignment == "ibp_map"
    assert {
        "ScadMcpPenalty",
        "NuclearNormPenalty",
        "DecoderIncoherencePenalty",
    }.issubset(set(fit.primitive_names))

    items = _captured_penalties(fake)
    by_kind = {str(item["kind"]): item for item in items}
    assert set(by_kind) == {"scad_mcp", "nuclear_norm", "decoder_incoherence"}

    assert by_kind["scad_mcp"] == {
        "kind": "scad_mcp",
        "target": "t",
        "variant": "scad",
        "gamma": 4.2,
        "weight": 0.3,
    }
    assert by_kind["nuclear_norm"] == {
        "kind": "nuclear_norm",
        "target": "beta",
        "weight": 0.4,
        "max_rank": 1,
    }
    assert by_kind["decoder_incoherence"] == {
        "kind": "decoder_incoherence",
        "target": "beta",
        "block_sizes": [1, 1],
        "p_out": x.shape[1],
        "weight": 0.7,
    }


@pytest.mark.parametrize(
    "gate_sparsity,gamma,expected_descriptor",
    [
        ("l1", None, None),
        (
            "scad",
            3.9,
            {
                "kind": "scad_mcp",
                "target": "t",
                "variant": "scad",
                "gamma": 3.9,
                "weight": 0.25,
            },
        ),
        (
            "mcp",
            1.8,
            {
                "kind": "scad_mcp",
                "target": "t",
                "variant": "mcp",
                "gamma": 1.8,
                "weight": 0.25,
            },
        ),
    ],
)
def test_gate_sparsity_variants_are_accepted_and_described(
    monkeypatch,
    gate_sparsity,
    gamma,
    expected_descriptor,
):
    fake = _CapturingRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)

    fit = gamfit.sae_manifold_fit(
        X=_toy_matrix(),
        K=1,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        isometry_weight=0.0,
        ard_per_atom=False,
        sparsity_weight=0.25,
        gate_sparsity=gate_sparsity,
        scad_mcp_gamma=gamma,
        decoder_incoherence_weight=0.0,
        n_iter=1,
    )

    items = _captured_penalties(fake)
    scad_mcp_items = [item for item in items if item["kind"] == "scad_mcp"]
    assert fit.assignments.shape == (_toy_matrix().shape[0], 1)
    if expected_descriptor is None:
        assert scad_mcp_items == []
    else:
        assert scad_mcp_items == [expected_descriptor]


@pytest.mark.parametrize(
    "assignment,expected_kind",
    [
        ("ibp_map", "ibp_map"),
        ("softmax", "softmax"),
        # #1777 — the hard-sigmoid gate's canonical token is now "threshold_gate";
        # "jumprelu" is retained as a deprecated alias (the raw spelling survives
        # only as `assignment_label`).
        ("jumprelu", "threshold_gate"),
    ],
)
def test_assignment_kinds_run_through_facade(monkeypatch, assignment, expected_kind):
    fake = _CapturingRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    x = _toy_matrix(n=10, p=3)

    fit = gamfit.sae_manifold_fit(
        X=x,
        K=2,
        d_atom=1,
        atom_topology="circle",
        assignment=assignment,
        isometry_weight=0.0,
        ard_per_atom=False,
        decoder_incoherence_weight=0.0,
        n_iter=1,
        jumprelu_threshold=0.15,
    )

    assert fake.calls[-1]["assignment_kind"] == expected_kind
    assert fake.calls[-1]["jumprelu_threshold"] == pytest.approx(0.15)
    assert fit.assignment == expected_kind
    assert fit.assignment_label == assignment
    assert fit.assignments.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(fit.assignments))


def test_sae_curvature_report_is_user_reachable_and_round_trips(monkeypatch):
    fake = _CapturingRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)

    fit = gamfit.sae_manifold_fit(
        X=_toy_matrix(n=10, p=3),
        K=2,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        isometry_weight=0.0,
        ard_per_atom=False,
        decoder_incoherence_weight=0.0,
        n_iter=1,
    )

    rows = fit.curvature()
    assert len(rows) == 2
    assert rows[0]["kappa_hat"] == pytest.approx(0.1)
    assert rows[1]["kappa_hat"] == pytest.approx(0.2)
    for row in rows:
        # The SAE curvature report carries the plug-in sup-norm bound only — a
        # curvature bound is not an estimand, so no SE/CI/flatness/verdict keys
        # (#1099 rescoped under #1115).
        assert set(row) == {"atom", "kappa_hat"}

    assert fit.atom_curvature(1)["kappa_hat"] == pytest.approx(0.2)
    restored = gamfit.ManifoldSAE.from_dict(fit.to_dict())
    assert restored.curvature() == rows


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"gate_sparsity": "elastic"}, "must be one of 'l1'"),
        (
            {"gate_sparsity": "scad", "scad_mcp_gamma": 2.0},
            "scad_mcp_gamma must be finite and > 2",
        ),
        (
            {"gate_sparsity": "mcp", "scad_mcp_gamma": 1.0},
            "scad_mcp_gamma must be finite and > 1",
        ),
        (
            {"nuclear_norm_weight": -0.1},
            "nuclear_norm_weight must be finite and non-negative",
        ),
        (
            {"nuclear_norm_max_rank": 0},
            "nuclear_norm_max_rank must be >= 1",
        ),
        (
            {"decoder_incoherence_weight": -0.1},
            "decoder_incoherence_weight must be finite and non-negative",
        ),
    ],
)
def test_recent_penalty_knobs_validate_parameters_eagerly(kwargs, match):
    with pytest.raises(ValueError, match=match):
        gamfit.sae_manifold_fit(
            X=_toy_matrix(),
            K=2,
            d_atom=1,
            atom_topology="circle",
            n_iter=1,
            **kwargs,
        )


def _two_atom_circle_data(n: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    clean = np.column_stack([np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t)])
    return clean + 0.03 * rng.standard_normal(clean.shape)


def _multi_atom_surface_fit(x: np.ndarray, monkeypatch) -> gamfit.ManifoldSAE:
    fake = _CapturingRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    return gamfit.sae_manifold_fit(
        X=x,
        K=2,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        isometry_weight=0.0,
        ard_per_atom=False,
        sparsity_weight=0.01,
        smoothness_weight=0.01,
        decoder_incoherence_weight=0.0,
        n_iter=20,
        learning_rate=1.0,
        random_state=0,
    )


def test_multi_atom_fresh_fit_populates_uncertainty_shape_band_api(monkeypatch):
    x = _two_atom_circle_data()
    fit = _multi_atom_surface_fit(x, monkeypatch)
    p = x.shape[1]

    assert len(fit.atoms) == 2
    for atom_k, atom in enumerate(fit.atoms):
        d_k = fit.coords[atom_k].shape[1]
        m_k = atom.decoder_coefficients.shape[0]

        if (
            atom.decoder_covariance is None
            or atom.shape_band_coords is None
            or atom.shape_band_mean is None
            or atom.shape_band_sd is None
        ):
            raise AssertionError(
                "fake SAE surface payload must populate posterior shape "
                f"uncertainty for atom={atom_k}"
            )

        assert atom.decoder_covariance.shape == (m_k * p, m_k * p)
        assert atom.shape_band_coords.shape[1] == d_k
        assert atom.shape_band_mean.shape == atom.shape_band_sd.shape
        assert atom.shape_band_mean.shape[1] == p
        assert atom.shape_band_mean.shape[0] == atom.shape_band_coords.shape[0]
        assert np.all(np.isfinite(atom.decoder_covariance))
        assert np.all(np.isfinite(atom.shape_band_mean))
        assert np.all(np.isfinite(atom.shape_band_sd))
        assert np.all(atom.shape_band_sd >= 0.0)

        band = fit.shape_uncertainty(atom=atom_k, n_sd=2.0)
        assert set(band) == {"coords", "mean", "sd", "lower", "upper"}
        assert band["coords"].shape == atom.shape_band_coords.shape
        assert band["mean"].shape == band["sd"].shape == band["lower"].shape
        assert band["upper"].shape == band["mean"].shape
        np.testing.assert_allclose(band["lower"], band["mean"] - 2.0 * band["sd"])
        np.testing.assert_allclose(band["upper"], band["mean"] + 2.0 * band["sd"])

        assert band["coords"].ndim == 2
        assert band["coords"].shape[1] == d_k
        assert band["mean"].shape[1] == p
        assert np.all(np.isfinite(band["mean"]))
        assert np.all(np.isfinite(band["sd"]))
