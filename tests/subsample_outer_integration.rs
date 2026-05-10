//! Integration tests for the outer-score subsample pipeline.
//!
//! These tests exercise the public Horvitz-Thompson semantics of the
//! stratified outer-score subsample without any automatic workflow wiring.
//! The invariant under test is the per-row weighting
//! contract: `OuterScoreSubsample::rows[i].weight = N_h / k_h` for the
//! row's stratum, and `outer_row_weights_by_index` exposes those weights
//! to per-row score / Hessian sums.

use gam::families::marginal_slope_shared::{
    OuterScoreSubsample, WeightedOuterRow, build_outer_score_subsample, outer_row_indices,
    outer_row_weights_by_index, outer_score_scale,
};

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
fn outer_row_indices_and_weights_round_trip_to_full_n_when_no_subsample() {
    use gam::families::custom_family::BlockwiseFitOptions;
    let opts = BlockwiseFitOptions::default();
    let n = 100;
    // Full-data path: legacy global scale is 1.0 and per-row HT weights
    // are all 1.0 (no subsample).
    let scale = outer_score_scale(&opts, n);
    assert!(
        (scale - 1.0).abs() < 1e-12,
        "no-subsample scale must be 1.0"
    );
    let weights = outer_row_weights_by_index(&opts, n);
    assert_eq!(weights.len(), n);
    for (i, w) in weights.iter().enumerate() {
        assert!(
            (w - 1.0).abs() < 1e-12,
            "row {i} weight should be 1.0 in full-data path, got {w}"
        );
    }
    let indices = outer_row_indices(&opts, n).to_vec();
    assert_eq!(indices.len(), n);
    let expected: Vec<usize> = (0..n).collect();
    assert_eq!(indices, expected);
}

// ---------------------------------------------------------------------------
// Per-row Horvitz-Thompson semantics.
//
// `OuterScoreSubsample` carries per-row inverse-inclusion weights w_i =
// N_h / k_h for the row's stratum. The stratified builder uses
// ceil(k * N_h / n).max(1), so per-stratum sampling fractions are unequal:
//   - the "max(1)" boost produces large w for a one-row sample of a rare
//     stratum, exactly the design knob that lets rare classes contribute
//     to the outer score
//   - the "ceil" boost makes most strata sample slightly above the global
//     rate, so per-row HT weights lie below n/m in the bulk and above n/m
//     in the rare tail
// A single global rescale by mean(w) under-weights rare strata and over-
// weights common strata; that bias is exactly what the per-row weights
// recover.
// ---------------------------------------------------------------------------

#[test]
fn per_row_ht_weight_recovers_full_data_sum_under_constant_within_stratum() {
    // Construct a tiny dataset where every row in a given stratum has the
    // same per-row contribution `c_h`, so the full-data sum is exactly
    // sum_h N_h * c_h. The HT estimator should reproduce that exactly:
    //   sum_{i in mask} w_i * c_{h(i)} = sum_h k_h * (N_h/k_h) * c_h
    //                                  = sum_h N_h * c_h
    // The deliberately-unequal stratum sizes (1 row in stratum A, many
    // in stratum B) force the builder into the "max(1) for the rare
    // stratum, ceil for the bulk" regime, which is the exact case where
    // the legacy global-scale would give the wrong answer.
    use std::sync::Arc;
    let n_full = 100usize;
    // Stratum A: a single rare row at index 0; per-row contribution c_A = 7.0.
    // Stratum B: rows 1..n_full; per-row contribution c_B = 1.0.
    let c_at = |i: usize| -> f64 { if i == 0 { 7.0 } else { 1.0 } };
    let true_sum: f64 = (0..n_full).map(c_at).sum();
    assert!((true_sum - (7.0 + 99.0)).abs() < 1e-12);

    // Subsample: row 0 (the rare A row at k_A=1, N_A=1, w_A=1) plus rows
    // 1..3 from the bulk (k_B=2, N_B=99, w_B=49.5). The legacy global
    // weight_scale would be n/m = 100/3 ~ 33.33 and would *bias* both
    // contributions; the correct HT estimator multiplies row 0 by 1.0 and
    // rows 1..3 by N_B/k_B = 49.5.
    let weighted_rows = vec![
        WeightedOuterRow {
            index: 0,
            weight: 1.0,
            stratum: 0,
        },
        WeightedOuterRow {
            index: 1,
            weight: 49.5,
            stratum: 1,
        },
        WeightedOuterRow {
            index: 2,
            weight: 49.5,
            stratum: 1,
        },
    ];
    let s = OuterScoreSubsample::from_weighted_rows(weighted_rows, n_full, 0);

    // Per-row HT estimator (the new contract): sum w_i * c_i over masked rows.
    let ht_sum: f64 = s.rows.iter().map(|r| r.weight * c_at(r.index)).sum();
    // = 1*7 + 49.5*1 + 49.5*1 = 7 + 99 = 106 = true_sum
    assert!(
        (ht_sum - true_sum).abs() < 1e-9,
        "HT estimator must recover full-data sum: got {ht_sum}, want {true_sum}",
    );

    // The legacy global rescale (n_full / m = 33.33) would give a
    // demonstrably biased estimate.
    let m = s.mask.len() as f64;
    let masked_sum: f64 = s.mask.iter().map(|&i| c_at(i)).sum();
    let legacy_global = (n_full as f64 / m) * masked_sum;
    // legacy = 33.33 * (7 + 1 + 1) = 33.33 * 9 = 300, far from 106.
    assert!(
        (legacy_global - true_sum).abs() > 50.0,
        "legacy global rescale must be biased here; got {legacy_global} vs true {true_sum}",
    );

    // Routing through `outer_row_weights_by_index` gives the same row-
    // indexed weights an outer-loop hot path would consume.
    use gam::families::custom_family::BlockwiseFitOptions;
    let mut opts = BlockwiseFitOptions::default();
    opts.outer_score_subsample = Some(Arc::new(s));
    let weights = outer_row_weights_by_index(&opts, n_full);
    assert_eq!(weights.len(), n_full);
    assert!((weights[0] - 1.0).abs() < 1e-12);
    assert!((weights[1] - 49.5).abs() < 1e-12);
    assert!((weights[2] - 49.5).abs() < 1e-12);
    // Unmasked rows default to 1.0 (irrelevant for a subsample iter).
    assert!((weights[50] - 1.0).abs() < 1e-12);
}

#[test]
fn stratified_builder_assigns_unequal_per_row_weights() {
    // The stratified builder produces a real subsample with per-row
    // weights that are NOT all equal (which is what motivated the HT fix
    // in the first place). Verify against the canonical 100-z-decile x
    // {0,1}-event design: rare strata get max(1)-boosted to k=1 while
    // bulk strata get ceil()-boosted to k > 1, so weight ratios across
    // strata exceed 1.
    let n = 50_000usize;
    // Heavily imbalanced secondary class (rate ~ 0.5%) so most strata in
    // class 1 collapse to k=1 with N=~250 (w ~ 250) while class-0 strata
    // run at k_h ~ ceil(K * N_h/n) (w ~ n/k_h ~ 12.5 at K=4_000).
    let z: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3).collect();
    let secondary: Vec<u8> = (0..n)
        .map(|i| if i % 200 == 0 { 1u8 } else { 0u8 })
        .collect();
    let k = 4_000usize;
    let s = build_outer_score_subsample(&z, &secondary, k, 0xC0FFEE_5EED);

    let weights: Vec<f64> = s.rows.iter().map(|r| r.weight).collect();
    let min_w = weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_w / min_w > 2.0,
        "stratified weights must vary across strata; got min={min_w} max={max_w}",
    );

    // The mean weight is the diagnostic `weight_scale`. It should be
    // *strictly between* min and max, demonstrating that any single
    // global rescale is wrong for at least some strata.
    let mean_w: f64 = weights.iter().sum::<f64>() / weights.len() as f64;
    assert!(
        mean_w > min_w && mean_w < max_w,
        "mean weight {mean_w} must lie strictly between min {min_w} and max {max_w}",
    );
    assert!((s.weight_scale - mean_w).abs() < 1e-9);

    // HT identity: sum of per-row weights equals n_full to within the
    // ceil()/max(1) overshoot bound (one extra unit per stratum that
    // boosted past its proportional allocation).
    let total_weight: f64 = weights.iter().sum();
    assert!(
        total_weight >= n as f64 - 200.0 && total_weight <= n as f64 + 200.0,
        "Σ w_i = {total_weight} must approximate n_full = {n} within stratum-count bound",
    );
}
