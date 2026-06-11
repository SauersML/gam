import numpy as np

import gamfit._sae_manifold as sae


def test_sae_dictionary_atom_count_matches_python_and_rust_payload(monkeypatch):
    class _FakeRust:
        def sae_manifold_fit_minimal(self, z, k_atoms, atom_basis, atom_dim, assignment_kind, alpha, tau, learnable_alpha, sparsity_strength, smoothness, max_iter, learning_rate, random_state, top_k, *, gumbel_schedule=None):
            atoms = []
            for i in range(k_atoms):
                atoms.append({"decoder_B": np.ones((2, z.shape[1])) * (i + 1), "basis_kind": atom_basis[i], "on_atom_coords_t": np.zeros((z.shape[0], atom_dim[i])), "assignments_z": np.ones(z.shape[0]), "active_dim": atom_dim[i]})
            return {
                "atoms": atoms,
                "atom_plans": [
                    {"kind": str(atom_basis[i]), "latent_dim": int(atom_dim[i]), "basis_size": 2, "n_harmonics": 0, "duchon_centers": None}
                    for i in range(k_atoms)
                ],
                "assignments_z": np.ones((z.shape[0], k_atoms)),
                "logits": np.zeros((z.shape[0], k_atoms)),
                "fitted": np.zeros_like(z),
                "reml_score": -1.0,
                "chosen_k": 1,
                "dispersion": 1.0,
                "oos_projection_top1": False,
                "diagnostics": _diagnostics(k_atoms, z.shape[0]),
            }

        def sae_manifold_reconstruction_r2(self, observed, fitted):
            return 0.0

    monkeypatch.setattr(sae, "rust_module", lambda: _FakeRust())

    fit = sae.sae_manifold_fit(np.random.default_rng(1).normal(size=(5, 3)), K=3, n_iter=1)

    assert fit.low_level.chosen_k == 1, "The chosen K reported by Rust should be preserved; Python should not silently overwrite it with len(atoms)."


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
