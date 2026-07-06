import numpy as np
import pytest


pytest.importorskip("gamfit._rust")


def test_audit_sae_loads_npy_checkpoint_and_returns_report(tmp_path):
    import gamfit

    decoder = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    activations = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    checkpoint = tmp_path / "decoder.npy"
    np.save(checkpoint, decoder)

    report = gamfit.audit_sae(checkpoint, activations, active=1, score_mode="off")

    assert report["api"] == "gamfit.audit_sae"
    assert report["checkpoint"]["format"] == "npy"
    assert report["decoder_shape"] == (2, 3)
    assert report["dual_certificate"]["n_rows"] == activations.shape[0]
    assert "dark_matter_fraction" in report["routability"]
    assert report["topology"]["summary"]["n_atoms"] == decoder.shape[0]
