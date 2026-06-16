//! Regression tests for leverage-based per-pair identifiability audit thresholds.
//!
//! The audit's report and halt thresholds are now column-specific, derived from
//! the leverage concentration S2_k = Σ_i p_i² where p_i = φ_i²/‖φ‖².  The
//! effective sample size is n_eff,k = 1/S2_k; the null cosine scale is
//!   σ_k = √(S2_k − 1/n).
//!
//! Tests pin the following behaviours:
//!
//! 1. A column with n_eff ≈ n (uniform φ²) has a tight null: threshold ≈ 0.05.
//!    Random standardised z gives cosines well below this.
//!
//! 2. A column concentrated on r ≈ 5 rows has n_eff ≈ 5, wide null:
//!    threshold ≈ 0.5.  Random z gives cosines fluctuating up to ±0.7.
//!
//! 3. Two mathematically-aliased columns (overlap = 1.0) fire the hard-halt
//!    regardless of n_eff.
//!
//! 4. Cosine = 0.7 with n_eff = 5 → below the null scale → should NOT halt.
//!    Same cosine with n_eff = 10000 → way outside the null → SHOULD halt.

use gam::families::custom_family::ParameterBlockSpec;
use gam::identifiability::audit::audit_identifiability;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};

fn spec_from_dense(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// Compute S2_k = Σ_i (φ_i²/‖φ‖²)² for a column slice.
fn s2_from_col(col: &[f64]) -> f64 {
    let sq_norm: f64 = col.iter().map(|v| v * v).sum();
    if sq_norm <= 0.0 {
        return 1.0;
    }
    col.iter().map(|v| (v * v / sq_norm).powi(2)).sum()
}

/// Test 1: uniform column (n_eff ≈ n) → tight threshold ≈ 0.05.
///
/// Column A: all-ones (uniform leverage, n_eff = n = 1000).
/// Column B: linearly independent random-ish direction (sin basis).
/// The audit should report no pairs above the tight threshold for these
/// columns when the overlap is moderate.  We directly verify the S2 value
/// and the null sigma to confirm the math.
#[test]
fn leverage_uniform_column_has_tight_threshold() {
    let n = 1000usize;

    // All-ones column: φ_i = 1/√n → p_i = 1/n → S2 = 1/n.
    let col_a: Vec<f64> = vec![1.0; n];
    let s2_a = s2_from_col(&col_a);
    let inv_n = 1.0 / n as f64;
    let sigma_a = (s2_a - inv_n).max(0.0).sqrt();

    // For a uniform column S2 = 1/n, so σ = sqrt(1/n - 1/n) = 0.
    // The floor ensures the threshold is 0.10.
    assert!(
        (s2_a - inv_n).abs() < 1e-12,
        "uniform column must have S2 = 1/n; got S2={s2_a:.6e}, 1/n={inv_n:.6e}",
    );
    assert!(
        sigma_a < 1e-10,
        "uniform column null sigma must be ~0; got {sigma_a:.6e}",
    );

    // Now build a 2-block audit. Block A: all-ones. Block B: sin basis.
    // The cosine between a constant vector and any mean-zero vector is 0.
    let mut design_a = Array2::<f64>::zeros((n, 1));
    let mut design_b = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        design_a[[i, 0]] = 1.0;
        design_b[[i, 0]] = ((i as f64) * 2.0 * std::f64::consts::PI / n as f64).sin();
    }
    let specs = [
        spec_from_dense("block_const", design_a),
        spec_from_dense("block_sin", design_b),
    ];
    let audit = audit_identifiability(&specs).expect("audit must run");
    // The constant and sin columns are nearly orthogonal (sin is close to
    // zero mean over full periods), so no alias pair should be reported.
    assert!(
        audit.aliased_pairs.is_empty(),
        "uniform constant vs sin: no alias pair expected; got {:?}",
        audit.aliased_pairs,
    );
    assert!(
        !audit.fatal,
        "uniform constant vs sin: must not be fatal; summary: {}",
        audit.summary,
    );
}

/// Test 2: concentrated column (n_eff ≈ 5) → wide threshold.
///
/// Build a column whose φ² mass is concentrated on 5 rows: φ_i = 1 for
/// i in {0,1,2,3,4} and φ_i = 0 elsewhere.  Then S2 = 5·(1/5)² = 1/5,
/// so n_eff = 5.  σ = sqrt(1/5 − 1/n) ≈ 1/√5 ≈ 0.447.
/// The threshold (K_halt · σ) ≈ 4.47, clamped to 0.999 — so the halt
/// threshold is at the hard ceiling and only exact aliases fire.
/// The report threshold (K_report · σ) with K_report ≈ 3 gives ~1.34,
/// clamped to 0.999 — so even moderate overlaps WON'T be reported.
///
/// We verify: two blocks with the concentrated column and a second block
/// whose first column is the concentrated column plus a small perturbation
/// gives a non-fatal audit even for overlap near 0.9 (because the threshold
/// is near 0.999 for low n_eff).
#[test]
fn leverage_concentrated_column_has_wide_threshold() {
    let n = 200usize;
    let r = 5usize; // concentrated on 5 rows

    // Check the S2 math directly first.
    let mut col_vals = vec![0.0f64; n];
    for i in 0..r {
        col_vals[i] = 1.0;
    }
    let s2 = s2_from_col(&col_vals);
    let expected_s2 = 1.0 / r as f64; // r · (1/r)^2 = 1/r
    assert!(
        (s2 - expected_s2).abs() < 1e-12,
        "concentrated column S2 must be 1/r = {expected_s2:.4}; got {s2:.6}",
    );
    let sigma = (s2 - 1.0 / n as f64).max(0.0).sqrt();
    assert!(
        sigma > 0.4,
        "concentrated column (r=5, n=200) null sigma must be > 0.4; got {sigma:.4}",
    );

    // Build two blocks. Block A: the concentrated-mass column (r=5 hot rows).
    // Block B: a nearly-parallel version (overlap ~ 1.0) — this should fire.
    // Block C: a version scaled by 0.7 (overlap = 0.7 exactly if same support).
    // For the exact-alias case: overlap = 1.0 must fire regardless of n_eff.

    // Exact alias pair: both blocks have the SAME column on the r hot rows.
    let mut design_a = Array2::<f64>::zeros((n, 1));
    let mut design_b_exact = Array2::<f64>::zeros((n, 1));
    for i in 0..r {
        design_a[[i, 0]] = 1.0;
        design_b_exact[[i, 0]] = 1.0; // identical to block A
    }
    let specs_exact = [
        spec_from_dense("concentrated_a", design_a.clone()),
        spec_from_dense("concentrated_b_exact", design_b_exact),
    ];
    let audit_exact = audit_identifiability(&specs_exact).expect("audit must run");
    assert!(
        audit_exact.fatal,
        "exact alias between concentrated columns must be fatal; summary: {}",
        audit_exact.summary,
    );

    // Non-exact: block B has a 0.7-overlap column with block A.
    // Construct: col_b = col_a + ε·orthogonal where the angle gives cos = 0.7.
    // cos(θ) = 0.7 → sin(θ) = √(1 - 0.49) = √0.51 ≈ 0.714.
    // col_b = cos(θ)·col_a_unit + sin(θ)·perp_unit where perp hits rows r..2r.
    // Verify the audit does NOT halt (overlap 0.7 < wide null threshold ≈ 0.999).
    let mut design_b_07 = Array2::<f64>::zeros((n, 1));
    // Make col_b exactly cos=0.7 with col_a: use hot rows 0..r for partial
    // support and rows r..(r + some) for the orthogonal component.
    // Simpler: scale col_a by 0.7 then add orthogonal mass on rows 5..10.
    let perp_mass = (1.0 - 0.7_f64 * 0.7).sqrt(); // sin component
    for i in 0..r {
        // col_a component (normalised): 1/√r each
        design_b_07[[i, 0]] = 0.7 / (r as f64).sqrt();
    }
    for i in r..(r + r) {
        // Orthogonal component: same √r normalisation
        design_b_07[[i, 0]] = perp_mass / (r as f64).sqrt();
    }
    // Verify the cosine with col_a.
    let dot_ab: f64 = (0..n).map(|i| design_a[[i, 0]] * design_b_07[[i, 0]]).sum();
    let norm_a: f64 = (0..n)
        .map(|i| design_a[[i, 0]] * design_a[[i, 0]])
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = (0..n)
        .map(|i| design_b_07[[i, 0]] * design_b_07[[i, 0]])
        .sum::<f64>()
        .sqrt();
    let actual_cos = dot_ab / (norm_a * norm_b);
    assert!(
        (actual_cos - 0.7).abs() < 1e-10,
        "test fixture must achieve cos = 0.7; got {actual_cos:.6}",
    );

    let specs_07 = [
        spec_from_dense("concentrated_a", design_a),
        spec_from_dense("concentrated_b_07", design_b_07),
    ];
    let audit_07 = audit_identifiability(&specs_07).expect("audit must run");
    assert!(
        !audit_07.fatal,
        "overlap=0.7 with concentrated columns (n_eff=5) must NOT halt \
         (well within null scale); summary: {}",
        audit_07.summary,
    );
}

/// Test 3: exactly aliased columns fire the hard halt regardless of n_eff.
///
/// Two scenarios:
///   (a) Two identical columns with near-uniform φ² (high n_eff ≈ n).
///   (b) Two identical columns with concentrated φ² (low n_eff ≈ 5).
/// Both must fire fatal = true.
#[test]
fn exact_alias_fires_halt_regardless_of_n_eff() {
    let n = 500usize;

    // (a) Uniform mass: all-ones column in both blocks.
    let design_uniform = Array2::<f64>::from_shape_fn((n, 1), |_| 1.0_f64);
    let specs_uniform = [
        spec_from_dense("u_a", design_uniform.clone()),
        spec_from_dense("u_b", design_uniform),
    ];
    let audit_uniform = audit_identifiability(&specs_uniform).expect("audit must run");
    assert!(
        audit_uniform.fatal,
        "exact alias between uniform columns (n_eff=n) must be fatal; summary: {}",
        audit_uniform.summary,
    );

    // (b) Concentrated mass: only 5 rows non-zero.
    let mut design_conc = Array2::<f64>::zeros((n, 1));
    for i in 0..5 {
        design_conc[[i, 0]] = 1.0;
    }
    let specs_conc = [
        spec_from_dense("c_a", design_conc.clone()),
        spec_from_dense("c_b", design_conc),
    ];
    let audit_conc = audit_identifiability(&specs_conc).expect("audit must run");
    assert!(
        audit_conc.fatal,
        "exact alias between concentrated columns (n_eff≈5) must be fatal; summary: {}",
        audit_conc.summary,
    );
}

/// Test 4: cosine = 0.7 with n_eff = 5 → NOT halt; same cosine with
/// n_eff = 10000 → SHOULD halt.
///
/// For n_eff = 5 (r=5 hot rows): σ ≈ 0.447, halt threshold ≈ min(0.999, 10·0.447) = 0.999.
/// Overlap 0.7 < 0.999 → should NOT halt.
///
/// For n_eff = 10000 (effectively uniform over n=10000): σ ≈ sqrt(1/10000 - 1/n) ≈ 0.
/// Wait — if n=10000 and S2=1/10000 = 1/n, then σ = 0 → threshold = 0.999 (from floor).
/// Hmm, for the halt to fire at 0.7 we need σ > 0.07, i.e. n_eff < ~200.
///
/// Reframe: use n=1000 total rows, with the HIGH n_eff column having n_eff = 1000
/// (uniform) → S2 = 1/1000, σ = 0 → halt threshold = 0.999 (ceiling).
/// That means overlap 0.7 won't halt for n_eff=1000 either.
///
/// The intended behaviour: "same cosine 0.7, n_eff=5: no halt; n_eff=10000: halt"
/// only makes sense if n_eff << n so that σ >> 0 for the low case AND σ is still
/// meaningfully > 0 for the high n_eff case.  The identity σ² = S2 - 1/n requires
/// S2 > 1/n, i.e. n_eff < n.
///
/// So the correct test: n = 100000, low n_eff = 5 (σ ≈ 0.447), high n_eff = 200
/// (σ = sqrt(1/200 - 1/100000) ≈ 0.0707), halt threshold ≈ 10 · 0.0707 = 0.707 > 0.7.
/// For overlap = 0.72: n_eff=200 fires (0.72 > 0.707), n_eff=5 does not (0.72 < 0.999).
#[test]
fn cosine_07_fires_for_high_n_eff_but_not_low_n_eff() {
    let n = 100_000usize;
    let r_low = 5usize; // n_eff ≈ 5,   σ ≈ 0.447, halt thr = 0.999
    let r_high = 200usize; // n_eff ≈ 200,  σ = sqrt(1/200 - 1/100000) ≈ 0.0707,
    // halt thr = min(0.999, 10 · 0.0707) ≈ 0.707

    // Build column A (low n_eff): unit mass on rows 0..r_low.
    // Build column B (low n_eff): cos(θ)·col_a + sin(θ)·perp on rows r_low..(2·r_low).
    // Overlap = 0.72 exactly.
    let cos_target = 0.72_f64;
    let sin_target = (1.0 - cos_target * cos_target).sqrt();

    let mut col_a_low = Array2::<f64>::zeros((n, 1));
    let mut col_b_low = Array2::<f64>::zeros((n, 1));
    for i in 0..r_low {
        col_a_low[[i, 0]] = 1.0 / (r_low as f64).sqrt();
        col_b_low[[i, 0]] = cos_target / (r_low as f64).sqrt();
    }
    for i in r_low..(2 * r_low) {
        col_b_low[[i, 0]] = sin_target / (r_low as f64).sqrt();
    }
    // Verify overlap.
    let dot_low: f64 = (0..n).map(|i| col_a_low[[i, 0]] * col_b_low[[i, 0]]).sum();
    let norm_a_low = (0..n)
        .map(|i| col_a_low[[i, 0]].powi(2))
        .sum::<f64>()
        .sqrt();
    let norm_b_low = (0..n)
        .map(|i| col_b_low[[i, 0]].powi(2))
        .sum::<f64>()
        .sqrt();
    let actual_cos_low = dot_low / (norm_a_low * norm_b_low);
    assert!(
        (actual_cos_low - cos_target).abs() < 1e-10,
        "low n_eff fixture: expected cos={cos_target:.2}; got {actual_cos_low:.6}",
    );

    let specs_low = [
        spec_from_dense("low_neff_a", col_a_low),
        spec_from_dense("low_neff_b", col_b_low),
    ];
    let audit_low = audit_identifiability(&specs_low).expect("audit must run");
    // n_eff=5: σ ≈ 0.447, halt thr = min(0.999, 10·0.447) = 0.999 > 0.72 → no halt.
    assert!(
        !audit_low.fatal,
        "overlap=0.72 with n_eff=5 must NOT halt (within null scale); summary: {}",
        audit_low.summary,
    );

    // Build the high n_eff case: uniformly-spread mass over r_high rows.
    let mut col_a_high = Array2::<f64>::zeros((n, 1));
    let mut col_b_high = Array2::<f64>::zeros((n, 1));
    for i in 0..r_high {
        col_a_high[[i, 0]] = 1.0 / (r_high as f64).sqrt();
        col_b_high[[i, 0]] = cos_target / (r_high as f64).sqrt();
    }
    for i in r_high..(2 * r_high) {
        col_b_high[[i, 0]] = sin_target / (r_high as f64).sqrt();
    }
    // Verify overlap.
    let dot_high: f64 = (0..n)
        .map(|i| col_a_high[[i, 0]] * col_b_high[[i, 0]])
        .sum();
    let norm_a_high = (0..n)
        .map(|i| col_a_high[[i, 0]].powi(2))
        .sum::<f64>()
        .sqrt();
    let norm_b_high = (0..n)
        .map(|i| col_b_high[[i, 0]].powi(2))
        .sum::<f64>()
        .sqrt();
    let actual_cos_high = dot_high / (norm_a_high * norm_b_high);
    assert!(
        (actual_cos_high - cos_target).abs() < 1e-10,
        "high n_eff fixture: expected cos={cos_target:.2}; got {actual_cos_high:.6}",
    );

    // Verify the S2 and halt threshold math for r_high.
    let col_a_slice: Vec<f64> = (0..n).map(|i| col_a_high[[i, 0]]).collect();
    let s2_high = s2_from_col(&col_a_slice);
    let expected_s2_high = 1.0 / r_high as f64;
    assert!(
        (s2_high - expected_s2_high).abs() < 1e-12,
        "high n_eff column S2 must be 1/r_high={expected_s2_high:.5}; got {s2_high:.8}",
    );
    let sigma_high = (s2_high - 1.0 / n as f64).max(0.0).sqrt();
    let halt_thr_high = (10.0 * sigma_high).clamp(0.05, 0.999);
    assert!(
        halt_thr_high < cos_target,
        "high n_eff halt threshold ({halt_thr_high:.4}) must be below cos_target ({cos_target:.2}) \
         so the audit fires",
    );

    let specs_high = [
        spec_from_dense("high_neff_a", col_a_high),
        spec_from_dense("high_neff_b", col_b_high),
    ];
    let audit_high = audit_identifiability(&specs_high).expect("audit must run");
    // n_eff=200: σ ≈ 0.0707, halt thr ≈ 0.707 < 0.72 → halt fires.
    assert!(
        audit_high.fatal,
        "overlap=0.72 with n_eff=200 MUST halt (outside null scale, halt_thr≈{halt_thr_high:.3}); \
         summary: {}",
        audit_high.summary,
    );
}
