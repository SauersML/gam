#![cfg(test)]
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
//! The original fix FLOORED `ref_df` at `max(edf, null_dim, 1)`. #2672 removed
//! that floor, on the ground that it was a patch on the wrong shape rather than
//! on a missing quantity: the reference is now the statistic's own null spectrum
//! `w = eig(2F_jj − F_jj²)` and `ref_df = Σ_j w_j` is its first moment, so as
//! REML shrinks a term the weights and the statistic collapse TOGETHER and the
//! `χ²_{d→0}` degeneracy cannot arise. Measured on this fixture: `ref_df = 0.096`
//! with `W = 2.5e-3` gives `p = 0.544`, not `1e-12`.
//!
//! So the `ref_df >= 1` clause here is gone — it required the floor, and would
//! have rejected its removal — and what replaces it is the ANALYTIC bound the
//! floor was standing in for, which catches #1766 by eleven orders rather than
//! by one:
//!
//! ```text
//!     edf  <=  ref_df  <=  2 * edf        (Wood's band, since w_j = λ_j(2 − λ_j))
//!     nu = (Sum w)^2 / Sum w^2  >=  1     (Cauchy-Schwarz on non-negative w)
//! ```
//!
//! The #1766 assembly reported `ref_df ~ 1e-12` against `edf ~ 0.09` — it fails
//! `ref_df >= edf` outright. The p-value clause stays exactly as it was.

//! Declared only as `#[cfg(test)] mod smooth_significance_ref_df_floor_1766_tests;`
//! in `fit_orchestration.rs`; the inner attribute states that scope in the file
//! itself so the compiler enforces it rather than a naming convention.
#![cfg(test)]

use super::entry::materialize;
use super::request::{FitConfig, FitRequest};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Fit `y ~ s(x)` from raw `(x, y)` columns and return the single smooth term's
/// LR report.
fn smooth_lr_report(x: &[f64], y: &[f64]) -> super::drivers::SmoothTermLrInference {
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
    reports.into_iter().next().expect("one smooth term")
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
        let report = smooth_lr_report(&x, &y);
        let (ref_df, w, p) = (
            report.ref_df,
            report.statistic_lr,
            report.p_value,
        );
        let provenance = &report.ref_df_provenance;
        // `ref_df` must sit inside Wood's analytic band `[edf, 2·edf]` and the
        // reference's shape must not degenerate. Both are identities of the
        // spectral assembly rather than floors, and the #1766 collapse
        // (`ref_df ~ 1e-12` against `edf ~ 0.09`) violates the first by eleven
        // orders. A slack of one part in `1e6` of `edf` covers the assembly's
        // own roundoff and nothing else.
        let slack = 1e-6 * provenance.edf.abs().max(1.0);
        let banded = ref_df >= provenance.edf - slack && ref_df <= 2.0 * provenance.edf + slack;
        let shaped = provenance.chi_square_df >= 1.0 - 1e-9;
        //
        // The p-value bar is a COLLAPSE floor, not a largeness requirement.
        // `y` here carries no x-signal at all, so a correctly calibrated test
        // returns p ~ Uniform(0,1) on this fixture and ANY fixed "p must be
        // large" bar fails on correct code at a rate the bar itself sets — the
        // old `p > 0.5` was a coin flip per seed over eight seeds. #1766
        // reported p ~ 4e-12 on every seed, so 1e-4 separates the defect from a
        // calibrated null by eight orders of magnitude while a true uniform
        // clears all eight seeds with probability ~99.9%.
        if !banded || !shaped || !(p > 1e-4) {
            offenders.push(format!(
                "(seed={seed}, ref_df={ref_df:.3e}, edf={:.3e}, nu={:.3e}, \
                 W={w:.3e}, p={p:.3e})",
                provenance.edf, provenance.chi_square_df
            ));
        }
    }
    assert!(
        offenders.is_empty(),
        "flat-null smooth mis-scaled: a shrunk-to-flat s(x) reported a reference \
         d.f. outside Wood's analytic band [edf, 2·edf], a degenerate shape \
         (nu < 1), or a collapsed p-value (<= 1e-4) — the #1766 ref_df -> ~0 \
         collapse is back. A p-value merely on the small side of uniform is NOT \
         this defect and must not be read as one, and `ref_df < 1` is not it \
         either: #2672 removed that floor and the spectral reference is a \
         first moment, free to be small when the term is. Offenders: {}",
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
    let report = smooth_lr_report(&x, &y);
    let (ref_df, w, p) = (
        report.ref_df,
        report.statistic_lr,
        report.p_value,
    );
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
