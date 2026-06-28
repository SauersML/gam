import numpy as np
import pytest

import gamfit._sae_manifold as sae


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
    ):
        assert assignment_kind == "ibp_map"
        # Production Rust iterates internally; emulate per-iteration basis
        # refresh by recording one snapshot per inner step. Each snapshot
        # uses perturbed input so the recovered coordinates drift between
        # iterations.
        n = int(z.shape[0])
        coords_seed = np.linspace(0.0, 1.0, n, dtype=float)
        for step in range(int(max_iter)):
            t = coords_seed + 0.07 * float(step)
            phi, _jet, _pen = self.periodic_basis_with_jet(t, n_harmonics=2)
            self.basis_snapshots.append(np.array(phi, copy=True))
        K = len(atom_basis)
        logits = np.full((n, K), 0.1, dtype=float)
        # Final coordinates: nonzero so the test's `np.linalg.norm > 0` holds.
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
            "chosen_k": K,
            "dispersion": 1.0,
            "oos_projection_top1": False,
            "diagnostics": _diagnostics(K, n),
            "log_alpha": np.log(alpha),
            "log_lambda_smooth": np.log(smoothness),
            "log_ard": [np.zeros(dim) for dim in atom_dim],
            "assignment_prior": "ibp_map",
        }


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage: _FakeRustModule is incomplete vs the refactored "
    "sae_manifold_fit FFI — the IBP driver now calls build_info() (and "
    "basis_with_jet) which this fake does not stub, so the fit fails with "
    "AttributeError: build_info. Rebuild the fake against the current FFI "
    "surface to re-enable.",
)
def test_ibp_driver_refreshes_basis_between_rust_steps(monkeypatch):
    fake = _FakeRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    z = np.array([[0.0, 0.2], [0.3, -0.1], [0.8, 0.5], [1.1, 0.9]])

    fit = sae.sae_manifold_fit(
        z,
        K=1,
        atom_topology="circle",
        d_atom=1,
        assignment="ibp_map",
        alpha=1.0,
        schedule=sae.gumbel_linear_schedule(tau_start=0.7, tau_min=0.7, steps=1),
        n_iter=2,
        learning_rate=0.1,
        random_state=4,
    )

    assert len(fake.basis_snapshots) == 2
    assert not np.allclose(fake.basis_snapshots[0], fake.basis_snapshots[1])
    assert np.isfinite(fit.reml_score)
    assert np.all(np.isfinite(fit.assignments))
    assert np.linalg.norm(fit.coords[0]) > 0.0


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
