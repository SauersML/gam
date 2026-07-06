from __future__ import annotations

import numpy as np
import pytest

sae = pytest.importorskip("gamfit._sae_manifold")


def _diagnostics() -> dict[str, object]:
    return {
        "atom_trust": [1.0],
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
