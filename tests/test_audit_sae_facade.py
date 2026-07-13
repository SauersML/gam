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
    donor_indices = np.asarray([[1], [0], [1], [0]], dtype=np.uint32)
    donor_values = np.asarray(
        [[[0.8]], [[0.7]], [[0.3]], [[0.9]]], dtype=np.float32
    )

    report = gamfit.audit_sae(
        checkpoint,
        activations,
        random_weight_codes=(donor_indices, donor_values),
        active=1,
        score_mode="off",
    )

    assert report["api"] == "gamfit.audit_sae"
    assert report["checkpoint"]["format"] == "npy"
    assert report["decoder_shape"] == (2, 3)
    assert report["dual_certificate"]["n_rows"] == activations.shape[0]
    assert "dark_matter_fraction" in report["routability"]
    assert report["topology"]["summary"]["n_atoms"] == 0


def test_audit_sae_surfaces_atlas_nerve_covering_side_next_to_betti():
    import gamfit

    decoder = np.eye(4, dtype=np.float32)
    dense_codes = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.25, 0.75],
        ],
        dtype=np.float32,
    )
    activations = dense_codes.copy()
    random_weight_dense = np.asarray(
        [
            [0.3, 0.1, 0.4, 0.2],
            [0.2, 0.6, 0.1, 0.7],
            [0.8, 0.4, 0.5, 0.3],
            [0.6, 0.9, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    route_indices = np.tile(np.asarray([[0, 1]], dtype=np.uint32), (4, 1))
    route_values = dense_codes.reshape(4, 2, 2)
    donor_values = random_weight_dense.reshape(4, 2, 2)

    report = gamfit.audit_sae(
        decoder,
        activations,
        codes=(route_indices, route_values),
        random_weight_codes=(route_indices, donor_values),
        block_size=2,
        active=2,
        score_mode="off",
    )

    atlas_nerve = report["atlas_nerve"]
    assert atlas_nerve["computed"] is True
    assert "betti" in atlas_nerve
    assert "covering_side" in atlas_nerve
    assert "euler_characteristic" not in atlas_nerve
    assert isinstance(atlas_nerve["nerve_euler_characteristic"], int)
    assert atlas_nerve["certified_euler_characteristic"] is None
    assert atlas_nerve["holonomy_status"] == "not_analyzed"
    assert atlas_nerve["certified_orientability"] is None
    assert atlas_nerve["topology_promotion"]["certified"] is False
    assert "null_pvalue" in atlas_nerve
    assert "spikein_power" in atlas_nerve
