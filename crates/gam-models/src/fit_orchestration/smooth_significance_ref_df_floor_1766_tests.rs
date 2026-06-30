//! Rust-layer regression for the `smooth_significance()` LR reference-d.f.
//! collapse (#1766), driving the real `smooth_term_lr_inference_forspec`
//! pipeline (the same entry the Python `Model.smooth_significance` FFI calls).
//!
//! The whole-term likelihood-ratio test drops the smooth (its unpenalized
//! linear null space included), so its χ² reference d.f. must be at least the
//! dimension the term spans when present — never below 1. The bug was a
//! degenerate Wood truncation `tr(F)²/tr(F²)` on the NON-symmetric coefficient
//! influence `F`: as REML shrinks a flat smooth onto its null space the
//! off-diagonal coupling blows up, `tr(F²)` runs away, and `tr(F)²/tr(F²)`
//! collapsed toward `1e-12`. Referencing a tiny positive `W ~ 1e-4` against
//! `χ²_{~0}` then reported a shrunk-to-flat term as MAXIMALLY significant
//! (`p ~ 1e-12`) — a Type-I error decided by the reference d.f., not the data.
//!
//! The fix floors `ref_df` at `max(edf, null_dim, 1)`, which binds only on the
//! degenerate collapse. These guards assert the post-fix contract directly on
//! the report: a planted-null flat smooth gets `ref_df >= 1` and a large
//! (non-significant) p-value, while a genuinely wiggly smooth is still flagged
//! and the floor never inflates its reference d.f.

use super::entry::materialize;
use super::request::{FitConfig, FitRequest};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Fit `y ~ s(x)` from raw `(x, y)` columns and return the single smooth term's
/// LR report `(ref_df, statistic_lr, p_value_corrected)`.
fn smooth_lr_report(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let headers: Vec<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| StringRecord::from(vec![xi.to_string(), yi.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let mat = materialize("y ~ s(x)", &ds, &cfg).expect("materialize y ~ s(x)");
    let request = match mat.request {
        FitRequest::Standard(request) => request,
        _ => panic!("expected a Standard fit request for y ~ s(x)"),
    };
    let reports = super::drivers::smooth_term_lr_inference_forspec(
        request.data.view(),
        request.y.view(),
        request.weights.view(),
        request.offset.view(),
        &request.spec,
        request.family.clone(),
        &request.options,
    )
    .expect("smooth-term LR inference");
    assert_eq!(reports.len(), 1, "exactly one smooth term expected");
    let r = &reports[0];
    (r.ref_df, r.statistic_lr, r.p_value_corrected)
}

#[test]
fn flat_null_smooth_ref_df_floored_and_not_significant_1766() {
    // Essentially-constant response: REML shrinks s(x) onto its 1-d.f. linear
    // null space (edf -> 1.0), the exact regime where the degenerate Wood
    // truncation used to crash ref_df to ~1e-12 and report p ~ 1e-12.
    let n = 200usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let mut offenders: Vec<String> = Vec::new();
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, 0.01).unwrap();
        let y: Vec<f64> = (0..n).map(|_| noise.sample(&mut rng)).collect();
        let (ref_df, w, p) = smooth_lr_report(&x, &y);
        // ref_df must be at least 1 (the whole-term test spans >= its null-space
        // dimension); a flat term with W ~ 1e-4 must read as the LEAST
        // significant possible, not the most.
        if ref_df < 1.0 - 1e-6 || !(p > 0.5) {
            offenders.push(format!(
                "(seed={seed}, ref_df={ref_df:.3e}, W={w:.3e}, p={p:.3e})"
            ));
        }
    }
    assert!(
        offenders.is_empty(),
        "flat-null smooth mis-scaled: a shrunk-to-flat s(x) reported a collapsed \
         reference d.f. (< 1) or a (near-)significant p-value (<= 0.5) — the #1766 \
         ref_df -> ~0 collapse is back. Offenders: {}",
        offenders.join("; ")
    );
}

#[test]
fn strong_signal_smooth_still_flagged_1766() {
    // Power control: the ref_df floor must not inflate a genuinely wiggly
    // smooth's reference d.f. into non-significance.
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(100);
    let noise = Normal::new(0.0, 0.3).unwrap();
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (8.0 * xi).sin() + noise.sample(&mut rng))
        .collect();
    let (ref_df, w, p) = smooth_lr_report(&x, &y);
    assert!(
        ref_df > 1.0,
        "a wiggly s(x) should carry reference d.f. well above 1 (got {ref_df:.3}); \
         the floor must not be capping a real signal"
    );
    assert!(
        p < 1e-3,
        "power control: a strong wiggly s(x) was not flagged (ref_df={ref_df:.3}, \
         W={w:.3e}, p={p:.3e})"
    );
}
