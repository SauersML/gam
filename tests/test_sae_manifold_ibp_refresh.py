import numpy as np

import gamfit._sae_manifold as sae


class _FakeRustModule:
    def __init__(self):
        self.basis_snapshots = []

    def sae_manifold_fit_ibp(
        self,
        z,
        atom_basis,
        atom_dim,
        basis_values,
        basis_jacobian,
        basis_sizes,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        alpha,
        tau,
        learnable_alpha,
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        *,
        gumbel_schedule=None,
        analytic_penalties=None,
    ):
        assert max_iter == 1
        self.basis_snapshots.append(np.array(basis_values, copy=True))
        logits = np.array(initial_logits, copy=True) + 0.05
        coords = np.array(initial_coords, copy=True)
        coords[:, :, : max(atom_dim)] += 0.1
        assignments = 1.0 / (1.0 + np.exp(-logits / tau))
        atoms = []
        for atom, dim in enumerate(atom_dim):
            m = basis_sizes[atom]
            decoder = np.array(decoder_coefficients[atom, :m, :], copy=True)
            if not np.any(decoder):
                decoder[:, :] = 0.2
            atoms.append(
                {
                    "decoder_B": decoder,
                    "basis_kind": atom_basis[atom],
                    "basis_centers": None,
                    "on_atom_coords_t": coords[atom, :, :dim],
                    "assignments_z": assignments[:, atom],
                    "active_dim": dim,
                }
            )
        return {
            "atoms": atoms,
            "assignments_z": assignments,
            "logits": logits,
            "atom_active_mask": [True for _ in atom_dim],
            "fitted": np.zeros_like(z),
            "reml_score": -1.0,
            "log_alpha": np.log(alpha),
            "log_lambda_smooth": np.log(smoothness),
            "log_ard": [np.zeros(dim) for dim in atom_dim],
            "assignment_prior": "ibp_map",
        }


def test_ibp_driver_refreshes_basis_between_rust_steps(monkeypatch):
    fake = _FakeRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    z = np.array([[0.0, 0.2], [0.3, -0.1], [0.8, 0.5], [1.1, 0.9]])

    fit = sae.sae_manifold_fit(
        z,
        n_atoms=1,
        atom_basis="periodic",
        atom_dim=1,
        assignment_prior="ibp_map",
        alpha=1.0,
        tau=0.7,
        max_iter=2,
        learning_rate=0.1,
        random_state=4,
    )

    assert len(fake.basis_snapshots) == 2
    assert not np.allclose(fake.basis_snapshots[0], fake.basis_snapshots[1])
    assert np.isfinite(fit.reml_score)
    assert np.all(np.isfinite(fit.assignments))
    assert np.linalg.norm(fit.coords[0]) > 0.0
