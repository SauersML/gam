//! Integration tests for the subsample-outer pipeline.
//!
//! Subsample-outer (the bam-style principled fix for biobank-scale GAMs)
//! is composed of multiple shipped pieces:
//!   - Phase 1 (`c7691932`): `OuterScoreSubsample` type + helpers
//!   - Phase 2 (`75a22867`): `log_likelihood_only_with_options` in margslope families
//!   - Phase 3 (`66c9a846`): `sigma_exact_joint_psi_*_with_options` helpers
//!   - Phase 4 (`ab9f2523`): auto-injection at workflow.rs wrappers
//!   - Phase 5 (`24df6f51`): workspace bridge so `_with_options` activates
//!   - Magic K (`3af67ae6`): `auto_outer_subsample_k` derives K from n
//!     (ApproxKind: StatisticalApproximation — subsampled outer score).
//!
//! These tests exercise the seams BETWEEN those pieces through the
//! public surface, catching integration gaps that unit tests within each
//! crate module would miss.
//!
//! All tests run on synthetic small-n data so the suite stays fast (<1s).

use gam::families::marginal_slope_shared::{
    BIOBANK_OUTER_SUBSAMPLE_K_MAX, BIOBANK_OUTER_SUBSAMPLE_K_MIN,
    BIOBANK_OUTER_SUBSAMPLE_THRESHOLD, auto_outer_subsample_k, build_outer_score_subsample,
    inject_biobank_outer_subsample, inject_biobank_outer_subsample_from_arrays, outer_row_indices,
    outer_score_scale,
};

#[test]
fn auto_k_anchors_match_documented_values() {
    // The documented anchors in the function's doc-comment must match
    // the implementation. Regression catch: changing the heuristic
    // without updating the docs (or vice versa).
    assert_eq!(auto_outer_subsample_k(50_000), 4_000);
    assert_eq!(auto_outer_subsample_k(100_000), 6_250);
    assert_eq!(auto_outer_subsample_k(320_000), 20_000);
    assert_eq!(auto_outer_subsample_k(1_000_000), 40_000);
}

#[test]
fn auto_k_clamps_at_floor_and_ceiling() {
    // n below floor*16 → K = floor (statistical adequacy guard).
    assert_eq!(auto_outer_subsample_k(0), BIOBANK_OUTER_SUBSAMPLE_K_MIN);
    assert_eq!(
        auto_outer_subsample_k(60_000),
        BIOBANK_OUTER_SUBSAMPLE_K_MIN
    );
    // n above ceiling*16 → K = ceiling (memory + diminishing returns).
    assert_eq!(
        auto_outer_subsample_k(640_000),
        BIOBANK_OUTER_SUBSAMPLE_K_MAX
    );
    assert_eq!(
        auto_outer_subsample_k(50_000_000),
        BIOBANK_OUTER_SUBSAMPLE_K_MAX
    );
}

#[test]
fn inject_biobank_outer_subsample_skips_below_threshold() {
    use gam::families::custom_family::BlockwiseFitOptions;
    let n = BIOBANK_OUTER_SUBSAMPLE_THRESHOLD; // exactly at threshold → no subsample
    let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let secondary: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
    let mut opts = BlockwiseFitOptions::default();
    let installed = inject_biobank_outer_subsample(&mut opts, &z, &secondary);
    assert!(!installed, "inject must skip at exactly the threshold");
    assert!(opts.outer_score_subsample.is_none());
}

#[test]
fn inject_biobank_outer_subsample_fires_above_threshold_with_auto_k() {
    use gam::families::custom_family::BlockwiseFitOptions;
    let n = BIOBANK_OUTER_SUBSAMPLE_THRESHOLD + 10_000; // 60_000
    let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let secondary: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
    let mut opts = BlockwiseFitOptions::default();
    let installed = inject_biobank_outer_subsample(&mut opts, &z, &secondary);
    assert!(installed, "inject must fire above threshold");
    let s = opts
        .outer_score_subsample
        .as_ref()
        .expect("subsample installed");
    let expected_k = auto_outer_subsample_k(n);
    // Stratification can overshoot K slightly (per-stratum ceil rounding).
    assert!(
        s.mask.len() >= expected_k && s.mask.len() <= expected_k + 200,
        "subsample size {} not within [K, K+200] of auto K = {}",
        s.mask.len(),
        expected_k,
    );
    assert_eq!(s.n_full, n);
    let expected_scale = n as f64 / s.mask.len() as f64;
    assert!((s.weight_scale - expected_scale).abs() < 1e-12);
}

#[test]
fn inject_preserves_caller_supplied_subsample() {
    use gam::families::custom_family::BlockwiseFitOptions;
    use gam::families::marginal_slope_shared::OuterScoreSubsample;
    use std::sync::Arc;
    let n = BIOBANK_OUTER_SUBSAMPLE_THRESHOLD + 50_000;
    let z: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let secondary: Vec<u8> = vec![0u8; n];
    let mut opts = BlockwiseFitOptions::default();
    // Caller pre-installs a subsample with a recognizable seed.
    let preset = OuterScoreSubsample::new(vec![0, 1, 2, 3], n, 0xBEEF);
    opts.outer_score_subsample = Some(Arc::new(preset));
    let installed = inject_biobank_outer_subsample(&mut opts, &z, &secondary);
    assert!(
        !installed,
        "inject must not overwrite caller-supplied subsample"
    );
    let s = opts.outer_score_subsample.as_ref().unwrap();
    assert_eq!(s.seed, 0xBEEF);
    assert_eq!(s.mask.len(), 4);
}

#[test]
fn inject_from_arrays_handles_event_indicator_correctly() {
    use gam::families::custom_family::BlockwiseFitOptions;
    let n = BIOBANK_OUTER_SUBSAMPLE_THRESHOLD + 10_000;
    let z: Vec<f64> = (0..n).map(|i| (i % 1000) as f64).collect();
    // Event rate ~5% — typical biobank survival rate. Mix of 0.0 / 1.0 / NaN
    // (nan-safe behavior: nan is treated as nonzero per the doc).
    let secondary_f64: Vec<f64> = (0..n)
        .map(|i| if i % 20 == 0 { 1.0 } else { 0.0 })
        .collect();
    let mut opts = BlockwiseFitOptions::default();
    let installed = inject_biobank_outer_subsample_from_arrays(&mut opts, &z, &secondary_f64);
    assert!(installed);
    let s = opts.outer_score_subsample.as_ref().unwrap();
    // Stratification key = (z-decile × event), so events get represented.
    // With 5% event rate, expect ≥1 event row in the mask.
    let event_rows: usize = s.mask.iter().filter(|&&i| secondary_f64[i] == 1.0).count();
    assert!(
        event_rows > 0,
        "subsample mask has no event rows out of {} masked",
        s.mask.len()
    );
}

#[test]
fn inject_rejects_mismatched_lengths() {
    use gam::families::custom_family::BlockwiseFitOptions;
    let n = BIOBANK_OUTER_SUBSAMPLE_THRESHOLD + 10_000;
    let z: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let secondary_short: Vec<u8> = vec![0u8; n - 1];
    let mut opts = BlockwiseFitOptions::default();
    let installed = inject_biobank_outer_subsample(&mut opts, &z, &secondary_short);
    assert!(!installed, "inject must defensively reject length mismatch");
    assert!(opts.outer_score_subsample.is_none());
}

#[test]
fn build_outer_score_subsample_is_deterministic_per_seed() {
    let n = 5000;
    let z: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let secondary: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
    let s1 = build_outer_score_subsample(&z, &secondary, 1000, 0x123456789ABCDEF);
    let s2 = build_outer_score_subsample(&z, &secondary, 1000, 0x123456789ABCDEF);
    assert_eq!(s1.mask, s2.mask, "same seed must produce identical mask");
    let s3 = build_outer_score_subsample(&z, &secondary, 1000, 0xDEADBEEF);
    assert_ne!(
        s1.mask, s3.mask,
        "different seeds must produce different masks"
    );
}

#[test]
fn outer_row_indices_and_scale_round_trip_to_full_n_when_no_subsample() {
    use gam::families::custom_family::BlockwiseFitOptions;
    let opts = BlockwiseFitOptions::default();
    let n = 100;
    let scale = outer_score_scale(&opts, n);
    assert!(
        (scale - 1.0).abs() < 1e-12,
        "no-subsample scale must be 1.0"
    );
    let indices = outer_row_indices(&opts, n).to_vec();
    assert_eq!(indices.len(), n);
    let expected: Vec<usize> = (0..n).collect();
    assert_eq!(indices, expected);
}

#[test]
fn subsample_mask_arc_is_pinned_after_inject() {
    // Critical invariant for the path #1 design plan: the mask installed
    // by `inject_biobank_outer_subsample` must remain stable across
    // subsequent reads via the cloned options. The outer optimizer's
    // BFGS line search assumes constant cost noise within a bracket, so
    // the mask must NOT vary between outer iterations.
    use gam::families::custom_family::BlockwiseFitOptions;
    use std::sync::Arc;
    let n = BIOBANK_OUTER_SUBSAMPLE_THRESHOLD + 50_000;
    let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let secondary: Vec<u8> = (0..n).map(|i| (i % 4 == 0) as u8).collect();
    let mut opts = BlockwiseFitOptions::default();
    let installed = inject_biobank_outer_subsample(&mut opts, &z, &secondary);
    assert!(installed);
    let original_arc = opts
        .outer_score_subsample
        .as_ref()
        .expect("subsample installed")
        .clone();
    let original_ptr = Arc::as_ptr(&original_arc);

    for iter in 0..10 {
        let cloned = opts.clone();
        let cloned_arc = cloned
            .outer_score_subsample
            .as_ref()
            .expect("subsample preserved across clone")
            .clone();
        assert_eq!(
            Arc::as_ptr(&cloned_arc),
            original_ptr,
            "iter {iter}: subsample Arc identity must be preserved across BlockwiseFitOptions::clone()",
        );
    }
}

#[test]
fn auto_k_monotone_non_decreasing_in_n() {
    let mut prev = 0usize;
    for n in (0..2_000_000).step_by(7919) {
        let k = auto_outer_subsample_k(n);
        assert!(
            k >= prev,
            "auto_outer_subsample_k regressed at n={n}: prev={prev} k={k}"
        );
        prev = k;
    }
}
