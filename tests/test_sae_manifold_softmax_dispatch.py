import numpy as np

import gamfit._sae_manifold as sae


class _FakeRustModule:
    def __init__(self):
        self.assignment_kinds = []
        self.calls = 0

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
        # #1512: the SAE driver gates row-block penalties on
        # build_info()["sae_row_block_penalties"]; advertise the kinds the real
        # extension reports so the fit accepts the test's configuration.
        return {
            "sae_row_block_penalties": [
                "ard", "top_k_activation", "jumprelu", "sparsity",
                "row_precision_prior", "parametric_row_precision_prior",
                "scad_mcp", "block_orthogonality", "isometry",
            ]
        }

    def basis_with_jet(self, kind, coords, params=None):
        # #1512: refactored FFI signature basis_with_jet(kind, coords, params)
        # -> (phi, jet, penalty). Delegate to the periodic stub; coords is
        # (n, dim), flattened to the 1-D parameter the periodic basis expects.
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
        **_forward_compat_kwargs,
    ):
        # #1512: the FFI positional convention is now (z, bases, dims,
        # alpha, tau, learnable_alpha, kind, **keyword), and the schedule /
        # analytic_penalties / fisher_* / row_loss_weights ride as keywords
        # (absorbed by **_forward_compat_kwargs). k_atoms is the basis count.
        k_atoms = len(atom_basis)
        assert len(atom_dim) == int(k_atoms) == 2
        assert assignment_kind == "softmax"
        assert learnable_alpha is False
        assert max_iter == 1
        assert gumbel_schedule is None
        self.assignment_kinds.append(assignment_kind)
        self.calls += 1
        logits = np.array([[0.2, -0.1], [0.4, 0.0], [-0.2, 0.5], [0.1, 0.3]], dtype=float)
        weights = np.exp(logits / float(tau))
        assignments = weights / weights.sum(axis=1, keepdims=True)
        atoms = []
        # Synthetic basis size: matches the periodic basis the production
        # Rust kernel would have chosen given n_harmonics derived from
        # closed-form defaults — using 3 here is enough to exercise the
        # decoder-shape contract.
        basis_size = 3
        for atom, dim in enumerate(atom_dim):
            decoder = np.full((basis_size, z.shape[1]), 0.1 * float(atom + 1))
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
            "atom_plans": [
                {
                    "kind": str(atom_basis[atom]),
                    "latent_dim": int(atom_dim[atom]),
                    "basis_size": int(basis_size),
                    "n_harmonics": 0,
                    "duchon_centers": None,
                }
                for atom in range(int(k_atoms))
            ],
            "assignments_z": assignments,
            "logits": logits,
            "atom_active_mask": [True for _ in atom_dim],
            "fitted": np.zeros_like(z),
            "reml_score": -1.0,
            "penalized_loss_score": -1.0,
            "chosen_k": int(k_atoms),
            "dispersion": 1.0,
            "oos_projection_top1": False,
            "diagnostics": _diagnostics(int(k_atoms), z.shape[0]),
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
        n_iter=1,
        learning_rate=0.1,
        random_state=4,
    )

    assert fake.assignment_kinds == ["softmax"]
    assert fake.calls == 1
    assert fit.low_level.chosen_k == 2
    assert fit.assignments.shape == (z.shape[0], 2)


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
