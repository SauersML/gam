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
    n_rows = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n_rows, endpoint=False, dtype=np.float32)
    circle = np.stack((np.cos(theta), np.sin(theta)), axis=1).astype(np.float32)
    dense_codes = np.concatenate((circle, circle), axis=1)
    activations = dense_codes.copy()
    donor_circle = np.stack((np.cos(3.0 * theta), np.sin(3.0 * theta)), axis=1)
    random_weight_dense = np.concatenate((donor_circle, donor_circle), axis=1).astype(
        np.float32
    )
    route_indices = np.tile(np.asarray([[0, 1]], dtype=np.uint32), (n_rows, 1))
    route_values = dense_codes.reshape(n_rows, 2, 2)
    donor_values = random_weight_dense.reshape(n_rows, 2, 2)

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
    assert atlas_nerve["holonomy_status"] == "analyzed_refused"
    assert atlas_nerve["holonomy_provenance"] == "gaussian_pca_plugin"
    assert "population_spectrum_uncertified" in atlas_nerve["holonomy_refusal_codes"]
    assert (
        "population_cross_gram_margin_uncertified"
        in atlas_nerve["holonomy_refusal_codes"]
    )
    assert atlas_nerve["holonomy_analysis"]["covariance_authority"] == (
        "asymptotic_plugin"
    )
    assert atlas_nerve["holonomy_analysis"]["orientation_refusals"]
    prescriptions = atlas_nerve["holonomy_analysis"]["sample_prescription"]
    assert len(prescriptions) == 2
    assert {entry["current_pilot_rows"] for entry in prescriptions} == {32}
    assert {entry["current_inference_rows"] for entry in prescriptions} == {16}
    assert {entry["pilot_requirement"] for entry in prescriptions} == {
        "exact_capture_no_sampling_requirement"
    }
    assert {entry["inference_requirement"] for entry in prescriptions} == {
        "population_tail_inputs_required"
    }
    assert all(entry["required_inference_rows"] is None for entry in prescriptions)
    assert atlas_nerve["certified_orientability"] is None
    assert atlas_nerve["topology_promotion"]["certified"] is False
    assert "null_pvalue" in atlas_nerve
    assert "spikein_power" in atlas_nerve
