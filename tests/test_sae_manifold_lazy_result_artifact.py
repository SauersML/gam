from __future__ import annotations

import numpy as np
import pytest

sae = pytest.importorskip("gamfit._sae_manifold")


def _diagnostics(n_atoms: int = 1) -> dict[str, object]:
    return {
        "atom_trust": [1.0] * n_atoms,
        "atoms": [
            {
                "trust_score": 1.0,
                "sigma_min_tangent": 1.0,
                "sigma_max_tangent": 1.0,
                "tangent_condition_score": 1.0,
                "coverage": 1.0,
                "activation_frequency": 1.0,
                "untyped": False,
                "active_token_count": 3,
            }
            for _ in range(n_atoms)
        ],
    }


class _BasisOnlyRust:
    def basis_with_jet(self, kind, coords, params):
        assert kind == "periodic"
        assert params["n_harmonics"] == 0
        n_rows = np.asarray(coords).shape[0]
        return (
            np.ones((n_rows, 1), dtype=float),
            np.zeros((n_rows, 1, 1), dtype=float),
            np.zeros((1, 1), dtype=float),
        )

    def sae_manifold_reconstruct_ffi(
        self, atom_basis, atom_dim, decoder_blocks, coords, assignments, p_out
    ):
        assert atom_basis == ["periodic"]
        assert atom_dim == [1]
        assert int(p_out) == 1
        phi, _jet, _pen = self.basis_with_jet("periodic", coords[0], {"n_harmonics": 0})
        return np.asarray(assignments)[:, 0:1] * (phi @ np.asarray(decoder_blocks[0]))


def test_sae_result_does_not_retain_training_data_and_reconstructs_lazily(monkeypatch):
    monkeypatch.setattr(sae, "rust_module", lambda: _BasisOnlyRust())
    x = np.arange(3.0).reshape(3, 1)
    fitted = np.full((3, 1), 2.0)
    payload = {
        "atom_plans": [
            {
                "kind": "periodic",
                "latent_dim": 1,
                "basis_size": 1,
                "n_harmonics": 0,
                "duchon_centers": None,
            }
        ],
        "atoms": [
            {
                "basis_kind": "periodic",
                "decoder_B": np.array([[2.0]]),
                "assignments_z": np.ones(3),
                "on_atom_coords_t": np.zeros((3, 1)),
                "active_dim": 1,
            }
        ],
        "chosen_k": 1,
        "assignments_z": np.ones((3, 1)),
        "logits": np.zeros((3, 1)),
        "fitted": fitted,
        "reconstruction_r2": 1.0,
        "penalized_loss_score": 0.0,
        "oos_projection_top1": False,
        "dispersion": 1.0,
        "diagnostics": _diagnostics(),
    }

    model = sae.ManifoldSAE.from_payload(
        x,
        payload,
        topology="circle",
        assignment="softmax",
        penalties=[],
    )

    assert model.training_data is not x
    assert model.training_data.nbytes == 0
    with pytest.raises(ValueError, match="training_data is not retained"):
        np.asarray(model.training_data)
    np.testing.assert_allclose(model.reconstruct_training(), fitted)


def _old_numpy_reconstruct(
    atom_basis: list[str],
    atom_dims: list[int],
    decoder_blocks: list[np.ndarray],
    coords: list[np.ndarray],
    assignments: np.ndarray,
) -> np.ndarray:
    p_out = int(decoder_blocks[0].shape[1])
    out = np.zeros((assignments.shape[0], p_out), dtype=np.float64)
    for atom_idx, decoder in enumerate(decoder_blocks):
        phi, _jet, _penalty = sae._basis_with_jet_for_atom(
            atom_basis[atom_idx],
            coords[atom_idx],
            int(decoder.shape[0]),
            int(atom_dims[atom_idx]),
        )
        out += assignments[:, atom_idx : atom_idx + 1] * (phi @ decoder)
    return out


def test_native_sae_manifold_reconstruct_matches_numpy_loop_and_shapes():
    rust = sae.rust_module()

    theta = np.array([[0.0], [0.25], [0.5], [0.75]], dtype=np.float64)
    sphere_coords = np.array(
        [[0.0, 0.0], [0.2, 0.4], [-0.3, 0.7], [0.5, -0.2]],
        dtype=np.float64,
    )
    atom_basis = ["periodic", "sphere"]
    atom_dims = [1, 2]
    decoder_blocks = [
        np.array(
            [[0.3, -0.2], [1.1, 0.4], [-0.7, 0.9]],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.2, -0.1],
                [0.4, 0.3],
                [-0.6, 0.5],
                [0.7, -0.8],
                [0.1, 0.6],
                [-0.2, 0.9],
                [0.5, -0.4],
            ],
            dtype=np.float64,
        ),
    ]
    coords = [theta, sphere_coords]
    assignments = np.array(
        [[0.9, 0.1], [0.25, 0.75], [0.4, 0.6], [0.8, 0.2]],
        dtype=np.float64,
    )
    expected = _old_numpy_reconstruct(
        atom_basis, atom_dims, decoder_blocks, coords, assignments
    )
    native = np.asarray(
        rust.sae_manifold_reconstruct_ffi(
            atom_basis, atom_dims, decoder_blocks, coords, assignments, 2
        ),
        dtype=np.float64,
    )
    max_err = float(np.max(np.abs(native - expected)))
    assert max_err <= 1.0e-10

    payload = {
        "atom_plans": [
            {
                "kind": "periodic",
                "latent_dim": 1,
                "basis_size": 3,
                "n_harmonics": 1,
                "duchon_centers": None,
            },
            {
                "kind": "sphere",
                "latent_dim": 2,
                "basis_size": 7,
                "n_harmonics": 0,
                "duchon_centers": None,
            },
        ],
        "atoms": [
            {
                "basis_kind": "periodic",
                "decoder_B": decoder_blocks[0],
                "assignments_z": assignments[:, 0],
                "on_atom_coords_t": coords[0],
                "active_dim": 1,
            },
            {
                "basis_kind": "sphere",
                "decoder_B": decoder_blocks[1],
                "assignments_z": assignments[:, 1],
                "on_atom_coords_t": coords[1],
                "active_dim": 2,
            },
        ],
        "chosen_k": 2,
        "assignments_z": assignments,
        "logits": np.zeros_like(assignments),
        "fitted": expected,
        "reconstruction_r2": 1.0,
        "penalized_loss_score": 0.0,
        "oos_projection_top1": False,
        "dispersion": 1.0,
        "diagnostics": _diagnostics(2),
    }
    model = sae.ManifoldSAE.from_payload(
        np.zeros_like(expected),
        payload,
        topology="circle",
        assignment="softmax",
        penalties=[],
    )
    reconstructed = model.reconstruct_training()
    assert reconstructed.shape == expected.shape
    np.testing.assert_allclose(reconstructed, expected, rtol=0.0, atol=1.0e-10)

    atoms = [
        sae.StagewiseAtom(
            decoder=decoder_blocks[0],
            coords=coords[0],
            assignments=assignments[:, 0],
            topology="circle",
            latent_dim=1,
            delta_ev=None,
        ),
        sae.StagewiseAtom(
            decoder=decoder_blocks[1],
            coords=coords[1],
            assignments=assignments[:, 1],
            topology="sphere",
            latent_dim=2,
            delta_ev=0.1,
        ),
    ]
    stagewise = sae.StagewiseSAE(
        atoms=atoms,
        logits=np.zeros_like(assignments),
        ev_trace=np.array([1.0]),
        backfit_ev_trace=np.array([1.0]),
        births_accepted=1,
        births_rejected=0,
        stopped_reason="test",
        terminal_joint_penalized_laml=0.0,
        terminal_data_fit=0.0,
        birth_records=[],
        collapse_events=[],
        log_lambda_sparse=0.0,
        log_lambda_smooth=np.zeros(2),
        log_ard=[np.zeros(1), np.zeros(2)],
        assignment="softmax",
        seed=model,
        training_data=np.zeros_like(expected),
    )
    fitted = stagewise.fitted
    assert fitted.shape == expected.shape
    np.testing.assert_allclose(fitted, expected, rtol=0.0, atol=1.0e-10)
