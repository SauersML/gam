import numpy as np

import gamfit._sae_manifold as sae


class _FakeRustModule:
    def __init__(self):
        self.assignment_kinds = []

    def sae_manifold_fit_minimal(
        self,
        z,
        k_atoms,
        atom_basis,
        atom_dim,
        assignment_kind,
        alpha,
        tau,
        learnable_alpha,
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        random_state,
        top_k,
        *,
        gumbel_schedule=None,
    ):
        assert k_atoms == 2
        assert assignment_kind == "softmax"
        assert learnable_alpha is False
        assert max_iter == 1
        assert random_state == 4
        assert top_k is None
        assert gumbel_schedule is None
        self.assignment_kinds.append(assignment_kind)
        logits = np.array([[0.2, -0.1], [0.4, 0.0], [-0.2, 0.5], [0.1, 0.3]], dtype=float)
        weights = np.exp(logits / float(tau))
        assignments = weights / weights.sum(axis=1, keepdims=True)
        atoms = []
        for atom, dim in enumerate(atom_dim):
            decoder = np.full((2, z.shape[1]), 0.1 * float(atom + 1))
            atoms.append(
                {
                    "decoder_B": decoder,
                    "basis_kind": atom_basis[atom],
                    "on_atom_coords_t": np.full((z.shape[0], dim), 0.1 * float(atom + 1)),
                    "assignments_z": assignments[:, atom],
                    "active_dim": int(dim),
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
            "assignment_prior": "softmax",
        }


def test_softmax_fixed_k_dispatches_to_rust(monkeypatch):
    fake = _FakeRustModule()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    z = np.array([[0.0, 0.2], [0.3, -0.1], [0.8, 0.5], [1.1, 0.9]])

    fit = sae.sae_manifold_fit(
        z,
        K=2,
        atom_topology="periodic",
        d_atom=1,
        assignment="softmax",
        alpha=1.0,
        tau=0.7,
        max_iter=1,
        learning_rate=0.1,
        random_state=4,
    )

    assert fake.assignment_kinds == ["softmax"]
    assert fit.low_level.chosen_k == 2
    assert fit.assignments.shape == (z.shape[0], 2)
