"""#1784 — the ibp_map assignment must default its concentration to the K-aware
``default_ibp_concentration_for_k_atoms(K)`` when the caller leaves ``alpha``
unset, instead of the historical fixed ``alpha=1.0`` that collapsed the ordered
stick-breaking prior to a near-hard mask past the first ~3 atoms (which made the
manifold SAE underfit an equal-K linear dictionary and left the K=128 joint
Hessian rank-deficient → ``RemlConvergenceError``).

These tests pin the Python-side *wiring* (which alpha the facade forwards to the
Rust ``sae_manifold_fit_minimal`` kernel) without a real fit, using the same
fake-rust-module pattern as ``test_sae_manifold_ibp_refresh``. The reconstruction
*quality* invariant (manifold EV ≥ linear EV at equal K) is pinned in the Rust
``manifold::tests_ibp_capacity_1784`` unit test against the real solver.
"""
import math

import numpy as np
import pytest

import gamfit
import gamfit._sae_manifold as sae


# Self-contained fake-rust module (#2020 test-suite skew). This file previously
# did `from test_sae_manifold_ibp_refresh import _FakeRustModule`, a bare
# cross-test-module import that FAILS collection under the repo's pytest
# `importlib` import mode (test dirs are not on sys.path), aborting every
# `pytest tests/ -k …` run that does not pass `--continue-on-collection-errors`.
# The other four `_FakeRustModule` users each define their own copy for exactly
# this reason; do the same here so the module is self-contained. Kept in sync
# with `test_sae_manifold_ibp_refresh._FakeRustModule`.
def _diagnostics(k_atoms: int, n_obs: int) -> dict[str, object]:
    return {
        "atom_trust": np.ones(k_atoms, dtype=float),
        "atoms": [
            {
                "trust_score": 1.0,
                "sigma_min_tangent": 1.0,
                "sigma_max_tangent": 1.0,
                "tangent_condition_score": 1.0,
                "coverage": 1.0,
                "activation_frequency": 1.0,
                "untyped": False,
                "active_token_count": int(n_obs),
            }
            for _ in range(k_atoms)
        ],
    }


class _FakeRustModule:
    def __init__(self):
        self.basis_snapshots = []

    def sae_manifold_reconstruction_r2(self, observed, fitted):
        observed = np.asarray(observed, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        ss_res = float(np.sum((observed - fitted) ** 2))
        mean = observed.mean(axis=0, keepdims=True)
        ss_tot = float(np.sum((observed - mean) ** 2))
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

    def build_info(self):
        return {
            "sae_row_block_penalties": [
                "ard", "top_k_activation", "jumprelu", "sparsity",
                "row_precision_prior", "parametric_row_precision_prior",
                "scad_mcp", "block_orthogonality", "isometry",
            ]
        }

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
        assert assignment_kind == "ibp_map"
        n = int(z.shape[0])
        coords_seed = np.linspace(0.0, 1.0, n, dtype=float)
        for step in range(int(max_iter)):
            t = coords_seed + 0.07 * float(step)
            phi, _jet, _pen = self.periodic_basis_with_jet(t, n_harmonics=2)
            self.basis_snapshots.append(np.array(phi, copy=True))
        K = len(atom_basis)
        logits = np.full((n, K), 0.1, dtype=float)
        coords = np.full((K, n, max(int(d) for d in atom_dim)), 0.1, dtype=float)
        assignments = 1.0 / (1.0 + np.exp(-logits / float(tau)))
        atoms = []
        for atom, dim in enumerate(atom_dim):
            decoder = np.full((5, z.shape[1]), 0.2, dtype=float)
            atoms.append(
                {
                    "decoder_B": decoder,
                    "basis_kind": atom_basis[atom],
                    "basis_centers": None,
                    "on_atom_coords_t": coords[atom, :, :dim],
                    "assignments_z": assignments[:, atom],
                    "active_dim": int(dim),
                }
            )
        return {
            "atoms": atoms,
            "atom_plans": [
                {
                    "kind": str(atom_basis[atom]),
                    "latent_dim": int(atom_dim[atom]),
                    "basis_size": int(atoms[atom]["decoder_B"].shape[0]),
                    "n_harmonics": 0,
                    "duchon_centers": None,
                }
                for atom in range(K)
            ],
            "assignments_z": assignments,
            "logits": logits,
            "atom_active_mask": [True for _ in atom_dim],
            "fitted": np.zeros_like(z),
            "reml_score": -1.0,
            "penalized_loss_score": -1.0,
            "chosen_k": K,
            "dispersion": 1.0,
            "oos_projection_top1": False,
            "diagnostics": _diagnostics(K, n),
            "log_alpha": np.log(alpha),
            "log_lambda_smooth": np.log(smoothness),
            "log_ard": [np.zeros(dim) for dim in atom_dim],
            "assignment_prior": "ibp_map",
        }


class _AlphaCapturingRust(_FakeRustModule):
    """Records the ``alpha`` (and assignment kind) the facade forwards."""

    def __init__(self):
        super().__init__()
        self.captured_alpha = None
        self.captured_kind = None
        self.captured_learnable = None

    def sae_manifold_fit_minimal(self, z, atom_basis, atom_dim, alpha, tau,
                                 learnable_alpha, assignment_kind, **kw):
        self.captured_alpha = float(alpha)
        self.captured_kind = str(assignment_kind)
        self.captured_learnable = bool(learnable_alpha)
        return super().sae_manifold_fit_minimal(
            z, atom_basis, atom_dim, alpha, tau, learnable_alpha,
            assignment_kind, **kw)


def _x(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 2))


def _fit(monkeypatch, *, K, **kwargs):
    fake = _AlphaCapturingRust()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    model = sae.sae_manifold_fit(
        _x(3 * K + 1), K=K, atom_topology="circle", d_atom=1,
        assignment="ibp_map", n_iter=1, **kwargs)
    return fake, model


def test_default_ibp_concentration_formula_matches_rust():
    # α = 1/(exp(1/K) − 1), floored at 1.0. Spot-check the boundary + span.
    assert sae._default_ibp_concentration_for_k_atoms(1) == pytest.approx(1.0)
    for K in (2, 8, 64, 128, 512):
        alpha = sae._default_ibp_concentration_for_k_atoms(K)
        # last atom retains prior mass π_{K-1} = (α/(α+1))^K ≈ e^{-1}.
        pi_last = (alpha / (alpha + 1.0)) ** K
        assert pi_last == pytest.approx(math.exp(-1.0), rel=1e-9)
        assert alpha >= 1.0


@pytest.mark.parametrize("K", [64, 128])
def test_ibp_unset_alpha_defaults_to_k_aware(monkeypatch, K):
    fake, model = _fit(monkeypatch, K=K)  # alpha left unset
    expected = sae._default_ibp_concentration_for_k_atoms(K)
    assert fake.captured_kind == "ibp_map"
    assert fake.captured_alpha == pytest.approx(expected)
    assert float(model.alpha) == pytest.approx(expected)
    # The old fixed default would have been 1.0; the K-aware value must be larger.
    assert fake.captured_alpha > 1.0


def test_ibp_explicit_alpha_is_respected(monkeypatch):
    fake, model = _fit(monkeypatch, K=64, alpha=2.5)
    assert fake.captured_alpha == pytest.approx(2.5)
    assert float(model.alpha) == pytest.approx(2.5)


def test_ibp_alpha_override_leaves_base_at_one(monkeypatch):
    # A per-fit ibp_alpha override drives the concentration in Rust, so the base
    # alpha the facade forwards stays the historical 1.0 (no K-aware bump).
    fake, _ = _fit(monkeypatch, K=64, ibp_alpha=3.0)
    assert fake.captured_alpha == pytest.approx(1.0)


def test_ibp_auto_alpha_starts_at_one(monkeypatch):
    # Learnable alpha ("auto") seeds at 1.0 and is refined by the solver; the
    # K-aware default only moves the fixed seed.
    fake, model = _fit(monkeypatch, K=128, alpha="auto")
    assert fake.captured_learnable is True
    assert fake.captured_alpha == pytest.approx(1.0)
    assert bool(model.learnable_alpha) is True
