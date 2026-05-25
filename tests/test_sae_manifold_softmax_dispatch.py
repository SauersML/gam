import numpy as np

import gamfit._sae_manifold as sae


class _FakeRustModule:
    def __init__(self):
        self.assignment_kinds = []

    def periodic_basis_with_jet(self, t, n_harmonics):
        x = np.mod(np.asarray(t, dtype=float), 1.0)
        cols = [np.ones_like(x)]
        dcols = [np.zeros_like(x)]
        penalty_diag = [1e-8]
        for h in range(1, int(n_harmonics) + 1):
            angle = 2.0 * np.pi * h * x
            cols.extend([np.sin(angle), np.cos(angle)])
            dcols.extend([2.0 * np.pi * h * np.cos(angle), -2.0 * np.pi * h * np.sin(angle)])
            penalty_diag.extend([float(h**4), float(h**4)])
        return np.stack(cols, axis=1), np.stack(dcols, axis=1)[:, :, None], np.diag(penalty_diag)

    def sae_manifold_fit(
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
        assignment_kind=None,
        gumbel_schedule=None,
        analytic_penalties=None,
    ):
        assert assignment_kind == "softmax"
        assert max_iter == 1
        self.assignment_kinds.append(assignment_kind)
        logits = np.asarray(initial_logits, dtype=float)
        weights = np.exp(logits / float(tau))
        assignments = weights / weights.sum(axis=1, keepdims=True)
        atoms = []
        for atom, dim in enumerate(atom_dim):
            m = int(basis_sizes[atom])
            decoder = np.asarray(decoder_coefficients[atom, :m, :], dtype=float).copy()
            if not np.any(decoder):
                decoder[:, :] = 0.1 * float(atom + 1)
            atoms.append(
                {
                    "decoder_B": decoder,
                    "basis_kind": atom_basis[atom],
                    "on_atom_coords_t": np.asarray(initial_coords[atom, :, :dim], dtype=float).copy(),
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

    fit = sae._fit_fixed_k(
        z,
        2,
        "periodic",
        1,
        1.0,
        1.0,
        "softmax",
        1.0,
        False,
        0.7,
        None,
        max_iter=1,
        learning_rate=0.1,
        random_state=4,
        penalties=None,
    )

    assert fake.assignment_kinds == ["softmax"]
    assert fit.chosen_k == 2
    assert fit.assignments.shape == (z.shape[0], 2)
