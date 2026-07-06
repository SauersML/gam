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
    assert "covering_side" in report["topology"]["atoms"][0]


def test_audit_sae_surfaces_atlas_nerve_covering_side_next_to_betti():
    import gamfit

    decoder = np.eye(4, dtype=np.float32)
    codes = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    activations = codes.copy()

    report = gamfit.audit_sae(
        decoder,
        activations,
        codes=codes,
        block_size=2,
        block_topk=2,
        active=2,
        score_mode="off",
    )

    atlas_nerve = report["atlas_nerve"]
    assert atlas_nerve["computed"] is True
    assert "betti" in atlas_nerve
    assert "covering_side" in atlas_nerve
