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
            law_gap_tolerance: Some(0.05),
        })
        .expect("wire report on the real pair");
    assert_eq!(wire.layout.anchor_dim, 64);
    assert_eq!(wire.layout.block_dims, vec![64]);
    assert_eq!(wire.transport.len(), 4, "one anchor->block report per atom");
    // OBJECTIVE transport-law measurement on the REAL L18 -> L19 pair (#2234).
    //
    // Derivation from the #2231 Inc-D crosscoder + transport-law contract
    // (`transport_law.rs`):
    //
    //  * ON-MANIFOLD, NOT NOISE (asserted): the empirical transport t -> t' of
    //    every atom must be captured by a SMOOTH few-harmonic map to at least a
    //    MAJORITY of its circular variance. `smooth_r2` is exactly that circular
    //    R^2 (the alternative hypothesis). The synthetic nonlinear arm holds
    //    `smooth_r2 > 0.9` for a clean planted reparam; > 0.5 (more signal than
    //    noise) is the honest real-data floor. Measured: {0.969, 0.993, 0.671,
    //    0.963}.
    //
    //  * THE PHASE-TRANSPORT LAW ITSELF IS MEASURED BUT **NOT** ASSERTED TRUE.
    //    The law (transport is a PURE phase shift, `law_gap = smooth_r2 -
    //    phase_r2 <= tol`) does NOT hold on real OLMo L18/L19 at K=4: `law_holds`
    //    is `Some(false)` for all four atoms (measured `law_gap` = {0.378, 0.053,
    //    1.312, 0.270}; the best atom, 0.053, sits just above the 0.05 tolerance;
    //    atom 2's `phase_r2` is even negative, -0.641). This is a genuine
    //    NEGATIVE RESULT (see #2234): the atoms live on smooth manifolds but
    //    their layer-to-layer transport is not a clean phase shift. Asserting
    //    `law_holds == true` here would be a known-red XFAIL in disguise, so we
    //    assert the honest structural invariants with teeth instead.
    let mut best_law_gap = f64::INFINITY;
    let mut best_phase_r2 = f64::NEG_INFINITY;
    for transport in &wire.transport {
        // The law verdict must be DEFINED on every atom: finite diagnostics and
        // a materialized verdict (the wire config requested a gap tolerance).
        assert!(
            transport.phase_r2.is_finite()
                && transport.smooth_r2.is_finite()
                && transport.law_gap.is_finite(),
            "atom {}: transport-law diagnostics must be finite \
             (phase_r2 = {}, smooth_r2 = {}, law_gap = {})",
            transport.atom, transport.phase_r2, transport.smooth_r2, transport.law_gap
        );
        assert!(
            transport.law_holds.is_some(),
            "atom {}: law verdict must materialize when a gap tolerance is configured",
            transport.atom
        );
        // Structural nesting invariant: the smooth alternative NESTS the phase
        // law, so `smooth_r2 >= phase_r2` up to the chordal-metric roundoff (a
        // small negative `law_gap` is honest numerics per `transport_law.rs`; a
        // large negative gap would mean the measurement is broken).
        assert!(
            transport.law_gap > -1e-6,
            "atom {}: smooth alternative must nest the phase law (law_gap = {})",
            transport.atom, transport.law_gap
        );
        // OBJECTIVE: the atom is on-manifold, not noise.
        assert!(
            transport.smooth_r2 > 0.5,
            "atom {}: transport must be a genuine smooth map; smooth_r2 = {} <= 0.5 \
             would be noise, not manifold structure",
            transport.atom, transport.smooth_r2
        );
        best_law_gap = best_law_gap.min(transport.law_gap);
        best_phase_r2 = best_phase_r2.max(transport.phase_r2);
    }
    // The phase-transport law is PARTIALLY present (not clean): at least the
    // best atom is on the LINEAR side of the synthetic linear/nonlinear boundary.
    // The synthetic planted-NONLINEAR fixture clears `law_gap > 5 * 0.02 = 0.10`;
    // the best real atom must sit BELOW that boundary (measured 0.053), and the
    // most phase-like atom must explain a strong majority of its transport with a
    // pure phase shift (measured phase_r2 = 0.940 — the recovered phase is a near
    // half-turn, φ ≈ ±0.49, for every atom).
    assert!(
        best_law_gap < 0.10,
        "the phase law should be at least partially present: best-atom law_gap = {} \
         (>= 0.10 would mean even the best atom is as nonlinear as the planted reparam)",
        best_law_gap
    );
    assert!(
        best_phase_r2 > 0.8,
        "at least one atom's transport should be strongly phase-like: best phase_r2 = {}",
        best_phase_r2
    );
}
