import numpy as np

import gamfit._sae_manifold as sae


def test_sae_assignment_mode_and_schedule_survive_python_to_rust_bridge(monkeypatch):
    captured = {}

    class _FakeRust:
        def build_info(self):
            # #1512: sae_manifold_fit now validates the requested assignment mode
            # against rust_module().build_info()["sae_row_block_penalties"]
            # (kept in lockstep with the Rust AssignmentMode set). The orphaned
            # fake predates that lookup; advertise the row-block penalties the
            # real module exposes so the 'jumprelu' assignment under test is
            # accepted instead of failing with AttributeError: build_info.
            return {
                "sae_row_block_penalties": [
                    "ard",
                    "top_k_activation",
                    "jumprelu",
                    "sparsity",
                    "row_precision_prior",
                    "parametric_row_precision_prior",
                    "scad_mcp",
                    "block_orthogonality",
                    "isometry",
                ],
            }

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

        def sae_manifold_fit_minimal(self, z, atom_basis, atom_dim, alpha, tau, learnable_alpha, assignment_kind, *, gumbel_schedule=None, **_extra):
            # #1512: the FFI positional convention is now (x, bases, dims,
            # alpha_value, tau, learnable_alpha, kind, **keyword) — the bridge
            # passes the assignment mode (`kind`) as the 7th positional and the
            # schedule / analytic_penalties / random_state / top_k / fisher_* /
            # row_loss_weights as keywords. The orphaned fake used the old
            # (z, k_atoms, atom_basis, atom_dim, assignment_kind, alpha, tau,
            # learnable_alpha, ...) order and a fixed positional tail; match the
            # current convention and absorb new keywords with **_extra so the
            # mode/temperature/schedule capture below stays valid as the FFI grows.
            captured["assignment_kind"] = assignment_kind
            captured["tau"] = tau
            captured["gumbel_schedule"] = dict(gumbel_schedule or {})
            atoms = [{"decoder_B": np.ones((2, z.shape[1])), "basis_kind": atom_basis[0], "on_atom_coords_t": np.zeros((z.shape[0], atom_dim[0])), "assignments_z": np.ones(z.shape[0]), "active_dim": atom_dim[0]}]
            return {
                "atoms": atoms,
                "atom_plans": [{"kind": str(atom_basis[0]), "latent_dim": int(atom_dim[0]), "basis_size": 2, "n_harmonics": 0, "duchon_centers": None}],
                "assignments_z": np.ones((z.shape[0], 1)),
                "logits": np.zeros((z.shape[0], 1)),
                "fitted": np.zeros_like(z),
                "reml_score": -1.0,
                "penalized_loss_score": -1.0,
                "chosen_k": 1,
                "dispersion": 1.0,
                "oos_projection_top1": False,
                "diagnostics": _diagnostics(1, z.shape[0]),
            }

        def sae_manifold_reconstruction_r2(self, observed, fitted):
            return 0.0

    monkeypatch.setattr(sae, "rust_module", lambda: _FakeRust())

    schedule = sae.gumbel_linear_schedule(tau_start=0.9, tau_min=0.2, steps=7, iter_count=3)
    sae.sae_manifold_fit(np.random.default_rng(0).normal(size=(4, 2)), K=1, assignment="jumprelu", tau=0.7, schedule=schedule, n_iter=1)

    # #1777 — the legacy "jumprelu" assignment spelling canonicalizes to the
    # primary "threshold_gate" token (mapping to Rust AssignmentMode::ThresholdGate)
    # before it crosses the FFI bridge.
    assert captured["assignment_kind"] == "threshold_gate", "Python AssignmentMode legacy 'jumprelu' alias should canonicalize to 'threshold_gate' (Rust AssignmentMode::ThresholdGate), not any other mode."
    assert captured["tau"] == 0.7 and captured["gumbel_schedule"] == schedule.to_rust_descriptor(), "Temperature scalar and schedule descriptor should survive the Python-to-Rust FFI bridge unchanged."


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
