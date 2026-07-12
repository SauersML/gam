//! #2015 — column-equilibration primitive test.
//!
//! HISTORY: unit-RMS data equilibration was wired into the crosscoder fit
//! path (`run_auto_sae_crosscoder_fit`/`run_sae_crosscoder_fit`) and REVERTED
//! the same night: for a homoscedastic reconstruction objective, dividing
//! columns by their RMS is not a reparametrization — it changes the estimand
//! (noise-dominated columns are amplified to unit RMS and the fit spends
//! capacity explaining them). Measured on MSI 13021686 at the wired commit:
//! `zz2015_tiny_inner_crawl_terminates` refused at the co-collapse floor
//! (EV 0.4566 vs null 0.4583) and the planted transport-law verdicts
//! collapsed (phase R² 0.139, smooth R² 0.837). The κ≈1e8 conditioning fix
//! must live in the inner solver's linear algebra (preconditioning), not in
//! the data frame. The primitive below is kept (spec'd + unit-tested) for a
//! future solver-level design; the fit path passes a unit `column_scale`.

use ndarray::Array2;

use crate::manifold::equilibrate_crosscoder_columns;

/// Direct unit check of the equilibration primitive itself: every non-empty
/// column of the mutated target has unit RMS, and un-scaling by the returned
/// factor exactly reproduces the original (a diagonal, lossless change of
/// variables).
#[test]
fn equilibrate_crosscoder_columns_normalizes_and_round_trips() {
    let mut target = Array2::<f64>::zeros((5, 3));
    // Column scale ratio 1e4: column 0 ~ O(1e2), column 1 ~ O(1), column 2 ~ O(1e-2).
    for i in 0..5 {
        let t = i as f64;
        target[[i, 0]] = 100.0 * (t + 1.0);
        target[[i, 1]] = 1.0 * (t + 1.0);
        target[[i, 2]] = 0.01 * (t + 1.0);
    }
    let original = target.clone();
    let scale = equilibrate_crosscoder_columns(&mut target);
    assert_eq!(scale.len(), 3);
    for col in 0..3 {
        let rms = (target.column(col).iter().map(|v| v * v).sum::<f64>() / 5.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-9,
            "column {col} must be unit-RMS after equilibration, got {rms}"
        );
    }
    // Undo: multiplying each column back by its scale must exactly reproduce
    // the original (equilibration is a lossless per-column rescale).
    for i in 0..5 {
        for j in 0..3 {
            let restored = target[[i, j]] * scale[j];
            assert!(
                (restored - original[[i, j]]).abs() <= 1e-9 * original[[i, j]].abs().max(1.0),
                "round trip mismatch at ({i},{j}): {restored} vs {}",
                original[[i, j]]
            );
        }
    }
    // The ratio of the largest to smallest scale reflects the planted 1e4 spread.
    let scale_max = scale.iter().cloned().fold(0.0_f64, f64::max);
    let scale_min = scale.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        scale_max / scale_min > 1e3,
        "the equilibration scale must reflect the planted column-scale spread, got ratio {}",
        scale_max / scale_min
    );
}
