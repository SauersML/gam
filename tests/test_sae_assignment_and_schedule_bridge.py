import numpy as np

import gamfit._sae_manifold as sae


def test_sae_assignment_mode_and_schedule_survive_python_to_rust_bridge(monkeypatch):
    captured = {}

    class _FakeRust:
        def sae_manifold_fit_minimal(self, z, k_atoms, atom_basis, atom_dim, assignment_kind, alpha, tau, learnable_alpha, sparsity_strength, smoothness, max_iter, learning_rate, random_state, top_k, *, gumbel_schedule=None):
            captured["assignment_kind"] = assignment_kind
            captured["tau"] = tau
            captured["gumbel_schedule"] = dict(gumbel_schedule or {})
            atoms = [{"decoder_B": np.ones((2, z.shape[1])), "basis_kind": atom_basis[0], "on_atom_coords_t": np.zeros((z.shape[0], atom_dim[0])), "assignments_z": np.ones(z.shape[0]), "active_dim": atom_dim[0]}]
            return {"atoms": atoms, "assignments_z": np.ones((z.shape[0], 1)), "fitted": np.zeros_like(z), "reml_score": -1.0}

        def sae_manifold_reconstruction_r2(self, observed, fitted):
            return 0.0

    monkeypatch.setattr(sae, "rust_module", lambda: _FakeRust())

    schedule = sae.gumbel_linear_schedule(tau_start=0.9, tau_min=0.2, steps=7, iter_count=3)
    sae.sae_manifold_fit(np.random.default_rng(0).normal(size=(4, 2)), K=1, assignment="gated", tau=0.7, schedule=schedule, n_iter=1)

    assert captured["assignment_kind"] == "jumprelu", "Python AssignmentMode 'gated' should map to Rust AssignmentMode::JumpReLU, not any other mode."
    assert captured["tau"] == 0.7 and captured["gumbel_schedule"] == schedule.to_rust_descriptor(), "Temperature scalar and schedule descriptor should survive the Python-to-Rust FFI bridge unchanged."
