//! #2015 real-data acceptance gate: the unified behavior-anchored entry
//! (`run_auto_sae_behavior_fit`) on REAL row-aligned Qwen3.5-9B data — layer-21
//! residual-stream activations (PCA-64, EVR 0.549) paired with the model's TRUE
//! next-token distributions (global top-63 tokens + renormalized tail bucket)
//! at the same 2000 WikiText token positions.
//!
//! Provenance: `tests/data/README_qwen35_behavior_fixture.md`; harvested by
//! `scripts/qwen_joint_behavior_harvest.py`; the full-width archive
//! `qwen35_9b_joint_behavior.npz` lives in the MSI project home.

use super::tests_olmo::{olmo_fixture_path, read_npy_f32_2d};
use super::*;

/// Rows loaded through the f32 fixture must be exact probability vectors again
/// before the sphere-tangent embedding sees them.
fn renormalize_rows(mut probs: Array2<f64>) -> Array2<f64> {
    for mut row in probs.rows_mut() {
        let sum: f64 = row.iter().sum();
        assert!(
            sum > 0.99 && sum < 1.01,
            "fixture row is not near-simplex: {sum}"
        );
        row /= sum;
    }
    probs
}

#[test]
fn qwen_real_activation_behavior_fit_selects_identifiable_lambda_y() {
    let activation = read_npy_f32_2d(&olmo_fixture_path("qwen35_9b_actsL21_pca64_2000.npy"));
    let probabilities = renormalize_rows(read_npy_f32_2d(&olmo_fixture_path(
        "qwen35_9b_behavior_probs64_2000.npy",
    )));
    assert_eq!(activation.dim(), (2000, 64));
    assert_eq!(probabilities.dim(), (2000, 64));

    let mut config = SaeCrosscoderAutoFitConfig::standard(4, 3);
    config.max_iter = 30;
    // Fixed-rho keeps the gate fast; lambda_y selection through the outer REML
    // walk is exercised by the synthetic behavior tests and the full driver.
    config.run_outer_rho_search = false;
    let report = run_auto_sae_behavior_fit(SaeBehaviorAutoFitRequest {
        activation,
        probabilities,
        config,
        cancel: None,
    })
    .expect("real Qwen activation/behavior fit must complete");

    assert_eq!(report.crosscoder.layers.len(), 2);
    for layer in &report.crosscoder.layers {
        assert!(
            layer.reconstruction_r2.is_finite() && layer.reconstruction_r2 > 0.0,
            "{}: shared-chart reconstruction must beat the column-mean baseline, got {}",
            layer.label,
            layer.reconstruction_r2
        );
    }

    let ident = &report.weight_identifiability;
    assert!(
        ident.identifiable,
        "real data leaves residual variance in BOTH blocks; got {ident:?}"
    );
    assert!(ident.activation_residual_variance > 0.0);
    assert!(ident.behavior_residual_variance > 0.0);
    assert!(ident.log_lambda_curvature > 0.0);
    assert!(report.behavior_block.log_lambda_y.is_finite());

    // The fitted behavior decodes to exact distributions with finite KL on
    // every row — the honest summary must not hide an infinite row.
    assert_eq!(
        report.kl.infinite_rows, 0,
        "no fitted row may decode off-simplex"
    );
    assert_eq!(report.kl.finite_rows, 2000);
    let mean_kl = report
        .kl
        .mean_kl_nats
        .expect("all-finite KL implies a mean");
    assert!(mean_kl.is_finite() && mean_kl >= 0.0);

    // Per-atom isometry certificates and the binding-neutral wire report are
    // the FFI/CLI contract — they must materialize on real data.
    assert_eq!(report.isometry.len(), 4);
    let wire = report.wire_report().expect("behavior wire report");
    assert!(wire.lambda_y > 0.0);
    assert_eq!(wire.target_probabilities.len(), 2000);
    assert_eq!(wire.fitted_probabilities.len(), 2000);
    serde_json::to_string(&wire).expect("wire report must serialize");
}
