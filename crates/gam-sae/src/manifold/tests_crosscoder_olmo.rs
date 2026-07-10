//! #2231 Inc D fixture gate: the typed crosscoder entry on REAL row-aligned
//! OLMo-3-32B activations (layers 18 and 19, same 635 prompts, per-layer
//! PCA-64 — see `tests/data/README_olmo_fixture.md`).
//!
//! This is the promotion evidence the issue asked for: the unified-engine
//! schedule (`run_auto_sae_crosscoder_fit`) fits one shared chart across two
//! real consecutive layers, prices per-block relevance, and produces a
//! MEASURED cross-layer drift report (equal ambient widths) with honest-unit
//! reconstructions on both layers.

use super::tests_olmo::{olmo_fixture_path, read_npy_f32_2d};
use super::*;

fn center_columns(mut x: Array2<f64>) -> Array2<f64> {
    let means = x.mean_axis(ndarray::Axis(0)).expect("nonempty fixture");
    for mut row in x.rows_mut() {
        row -= &means;
    }
    x
}

#[test]
fn olmo_l18_l19_pair_crosscoder_fits_with_measured_drift() {
    let anchor = center_columns(read_npy_f32_2d(&olmo_fixture_path(
        "olmo_l18_pair_pca64_635.npy",
    )));
    let block = center_columns(read_npy_f32_2d(&olmo_fixture_path(
        "olmo_l19_pair_pca64_635.npy",
    )));
    assert_eq!(anchor.dim(), (635, 64), "row-aligned single-layer tell");
    assert_eq!(block.dim(), (635, 64));

    let mut config = SaeCrosscoderAutoFitConfig::standard(4, 3);
    config.max_iter = 30;
    // Fixed-rho keeps the gate fast; the outer search is exercised by the
    // synthetic crosscoder tests and the full E3 driver.
    config.run_outer_rho_search = false;
    let report = run_auto_sae_crosscoder_fit(SaeCrosscoderAutoFitRequest {
        anchor_label: "L18".to_string(),
        anchor,
        blocks: vec![NamedCrosscoderTarget {
            label: "L19".to_string(),
            target: block,
        }],
        config,
        cancel: None,
    })
    .expect("real-pair crosscoder fit must complete");

    assert_eq!(report.layers.len(), 2);
    assert_eq!(report.layers[0].label, "L18");
    assert_eq!(report.layers[1].label, "L19");
    for layer in &report.layers {
        assert!(
            layer.reconstruction_r2.is_finite(),
            "{}: R2 must be finite, got {}",
            layer.label,
            layer.reconstruction_r2
        );
        assert!(
            layer.reconstruction_r2 > 0.0,
            "{}: shared-chart reconstruction must beat the column-mean baseline, got {}",
            layer.label,
            layer.reconstruction_r2
        );
    }
    match &report.drift {
        CrosscoderDriftStatus::Measured(drift) => {
            assert_eq!(drift.num_atoms, 4);
            assert!(
                drift.mean_drift().is_finite() && drift.mean_drift() >= 0.0,
                "mean drift must be a finite non-negative angle statistic"
            );
        }
        CrosscoderDriftStatus::Undefined { reason } => {
            panic!("equal-width layers must have measured drift; got undefined: {reason}")
        }
    }

    // The wire report is the FFI/CLI contract — it must materialize (with
    // transport measured between the two real layers) without error.
    let wire = report
        .wire_report(SaeCrosscoderEvaluationConfig {
            transport_grid_resolution: Some(64),
            law_gap_tolerance: Some(0.05),
        })
        .expect("wire report on the real pair");
    assert_eq!(wire.layout.anchor_dim, 64);
    assert_eq!(wire.layout.block_dims, vec![64]);
    assert_eq!(wire.transport.len(), 4, "one anchor->block report per atom");
    for transport in &wire.transport {
        assert!(transport.phase_r2.is_finite());
        assert!(transport.smooth_r2.is_finite());
        assert_eq!(transport.law_holds.is_some(), true);
    }
}
