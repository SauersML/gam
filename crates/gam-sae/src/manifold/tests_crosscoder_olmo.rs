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
        // OBJECTIVE (#2234): the K=4 shared chart must RECONSTRUCT the real
        // activations, not merely beat the column-mean baseline (> 0). The
        // single-layer manifold-SAE reaches held-out ΔEV ≈ 0.27 on this OLMo
        // corpus (fixture README); the two-layer shared chart must clear a
        // quarter of each layer's centered PCA-64 variance. Measured at the
        // last converged fit (98c0b8bd5): L18 = 0.479, L19 = 0.490.
        assert!(
            layer.reconstruction_r2 > 0.25,
            "{}: shared-chart reconstruction must explain > 25% of centered \
             variance (a genuine fit, cf. single-layer ΔEV ≈ 0.27), got {}",
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
        })
        .expect("wire report on the real pair");
    assert_eq!(wire.layout.anchor_dim, 64);
    assert_eq!(wire.layout.block_dims, vec![64]);
    assert_eq!(wire.transport.len(), 4, "one anchor->block report per atom");
    // OBJECTIVE transport-law measurement on the REAL L18 -> L19 pair (#2234).
    //
    // The phase-shift law (transport `t -> s·t + φ`, `transport_law.rs`) is
    // MEASURED here but NOT asserted to hold on every atom: it does not on real
    // OLMo L18/L19 at K=4 (atom 2's `phase_r2` measured -0.641). That is a
    // genuine NEGATIVE RESULT (see #2234): layer-to-layer transport is not a
    // clean phase shift for every atom, and asserting it would be a known-red
    // XFAIL in disguise. The invariants with teeth are that every atom's
    // measurement is defined over the requested grid, and that the most
    // phase-like atom explains a strong majority of its transport with a pure
    // phase shift (measured phase_r2 = 0.940; the recovered phase is a near
    // half-turn, φ ≈ ±0.49, for every atom).
    let mut best_phase_r2 = f64::NEG_INFINITY;
    for transport in &wire.transport {
        assert!(
            transport.phase_r2.is_finite(),
            "atom {}: phase-law circular R^2 must be finite, got {}",
            transport.atom,
            transport.phase_r2
        );
        assert_eq!(
            transport.transport_grid.len(),
            64,
            "atom {}: one transport sample per requested grid point",
            transport.atom
        );
        assert!(
            transport.deviation_locus.is_some(),
            "atom {}: a non-empty grid must name the phase law's worst locus",
            transport.atom
        );
        best_phase_r2 = best_phase_r2.max(transport.phase_r2);
    }
    assert!(
        best_phase_r2 > 0.8,
        "at least one atom's transport should be strongly phase-like: best phase_r2 = {}",
        best_phase_r2
    );
}
