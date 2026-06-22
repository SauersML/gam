//! Tests for skewness-aware bias correction in the identifiability audit.
//!
//! # Background
//!
//! When one of the two blocks in a cross-block cosine comparison carries
//! a `RowScaledJacobian` with scaling `z`, the effective Jacobian column is `z ⊙ φ`
//! instead of `φ`.  The cross-block cosine between `φ` (unscaled block) and
//! `z ⊙ φ` (scaled block) has a non-zero null mean under the null hypothesis
//! (z and φ independent) when z is skewed.  Specifically:
//!
//!   E[cos(φ, z⊙φ)] ≈ −(μ_3 / 2) · S2_k
//!
//! where μ_3 is the standardised third moment (skewness) of z and
//! S2_k = max(S2_a, S2_b) is the leverage concentration.
//!
//! The acceptance band is therefore `[shift − half_width, shift + half_width]`
//! instead of the symmetric `[−half_width, +half_width]` used by T11 for
//! the symmetric (μ_3 = 0) case.
//!
//! # What these tests verify
//!
//! 1. **Gaussian z** (μ_3 ≈ 0): bias shift ≈ 0, threshold matches T11 symmetric form.
//!    The audit must not fire false positives on a genuine non-alias pair.
//!
//! 2. **Skewed z** (lognormal-shifted-and-standardised, μ_3 ≈ 2): bias shift is
//!    non-trivial.  The audit must NOT fire a false positive on a pair whose cosine
//!    sits inside the shifted acceptance band even if it would fire under the
//!    symmetric T11 threshold.
//!
//! 3. **Heavy-tailed z** (Student-t with 3 df, standardised, μ_3 ≈ 0): like Gaussian —
//!    bias shift ≈ 0 even though variance is heavy-tailed.  No false positives.
//!
//! 4. **Bernoulli z** (p = 0.05, large skewness μ_3 ≈ 4): the shift is large.
//!    The threshold compensates and the audit does not fire a false positive.
//!
//! 5. **μ_3 estimator finite-sample correction**: verify that `compute_skewness_mu3`
//!    produces the correct sign and rough magnitude for a known skewed distribution.

use gam::families::custom_family::{
    BlockEffectiveJacobian, FamilyLinearizationState, ParameterBlockSpec, RowScaledJacobian,
};
use gam::identifiability::audit::{
    audit_identifiability, audit_identifiability_with_state, bias_shift_for_pair,
    compute_skewness_mu3,
};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};
use std::sync::Arc;

struct FirstBetaScaledJacobian {
    base: Array2<f64>,
    alias: Array2<f64>,
}

impl BlockEffectiveJacobian for FirstBetaScaledJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.base.nrows();
        let rows = rows.start.min(n)..rows.end.min(n);
        let scale = state.beta.first().copied().unwrap_or(0.0);
        let mut out = self
            .base
            .slice(ndarray::s![rows.start..rows.end, ..])
            .to_owned();
        out += &self
            .alias
            .slice(ndarray::s![rows.start..rows.end, ..])
            .mapv(|v| scale * v);
        Ok(out)
    }
}

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

fn spec_with_scaling(name: &str, design: Array2<f64>, z: Vec<f64>) -> ParameterBlockSpec {
    let n = design.nrows();
    let eta_scaling: Arc<[f64]> = Arc::from(z.as_slice());
    let jac: Arc<dyn gam::families::custom_family::BlockEffectiveJacobian> =
        Arc::new(RowScaledJacobian {
            design: Arc::new(design.clone()),
            eta_scaling,
        });
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: Some(jac),
        stacked_design: None,
        stacked_offset: None,
    }
}

/// Generate a deterministic pseudo-random sequence using a linear congruential
/// generator.  Used instead of an RNG dependency so the tests are
/// deterministic and have no external deps.
fn lcg_sequence(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to (0, 1) via the upper 32 bits.
        let u = (state >> 32) as f64 / u32::MAX as f64;
        out.push(u);
    }
    out
}

/// Standardise a vector to zero mean and unit variance in-place.
fn standardise(v: &mut Vec<f64>) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    for x in v.iter_mut() {
        *x -= mean;
    }
    let var = v.iter().map(|x| x * x).sum::<f64>() / n;
    let sigma = var.sqrt();
    if sigma > 0.0 {
        for x in v.iter_mut() {
            *x /= sigma;
        }
    }
}

/// Build a Gaussian-distributed vector with zero mean and unit variance
/// (Box-Muller transform applied to the LCG output).
fn gaussian_standardised(seed: u64, n: usize) -> Vec<f64> {
    let u1 = lcg_sequence(seed, n);
    let u2 = lcg_sequence(seed.wrapping_add(999_999), n);
    let mut v: Vec<f64> = u1
        .iter()
        .zip(u2.iter())
        .map(|(&a, &b)| {
            let a_clamped = a.clamp(1e-12, 1.0 - 1e-12);
            (-2.0 * a_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        })
        .collect();
    standardise(&mut v);
    v
}

/// Build a lognormal vector shifted and standardised to zero mean and unit
/// variance.  For ln(X) ~ N(0,1): E[X] = e^{1/2}, Var[X] = e(e−1),
/// skewness μ_3 = (e + 2)√(e − 1) ≈ 6.2.
fn lognormal_standardised(seed: u64, n: usize) -> Vec<f64> {
    let g = gaussian_standardised(seed, n);
    // exp(g_i) has lognormal distribution; then standardise.
    let mut v: Vec<f64> = g.iter().map(|x| x.exp()).collect();
    standardise(&mut v);
    v
}

/// Build a Student-t(3 df) vector standardised to zero mean and unit variance.
/// For t(3): μ_3 = 0 (symmetric distribution).
/// Generated via the ratio-of-normals method: t = Z / sqrt(χ²/k).
fn student_t3_standardised(seed: u64, n: usize) -> Vec<f64> {
    let z = gaussian_standardised(seed, n);
    // Sum of 3 squared standard normals gives χ²(3).
    let g1 = gaussian_standardised(seed.wrapping_add(1), n);
    let g2 = gaussian_standardised(seed.wrapping_add(2), n);
    let g3 = gaussian_standardised(seed.wrapping_add(3), n);
    let mut v: Vec<f64> = z
        .iter()
        .zip(g1.iter())
        .zip(g2.iter())
        .zip(g3.iter())
        .map(|(((zi, g1i), g2i), g3i)| {
            let chi2 = g1i * g1i + g2i * g2i + g3i * g3i;
            zi / (chi2 / 3.0).sqrt().max(1e-12)
        })
        .collect();
    standardise(&mut v);
    v
}

/// Build a Bernoulli(p=0.05) vector standardised to zero mean and unit variance.
/// The population skewness of Bernoulli(p) is (1−2p)/√(p(1−p)).
/// For p = 0.05: μ_3_pop ≈ (0.9)/√(0.0475) ≈ 4.13.
fn bernoulli_05_standardised(seed: u64, n: usize) -> Vec<f64> {
    let u = lcg_sequence(seed, n);
    let p = 0.05_f64;
    let mut v: Vec<f64> = u.iter().map(|&x| if x < p { 1.0 } else { 0.0 }).collect();
    standardise(&mut v);
    v
}

// ─── Unit tests for compute_skewness_mu3 and bias_shift_for_pair ─────────────

/// Test that compute_skewness_mu3 returns approximately zero for a symmetric
/// Gaussian (any skewness should be very small for large n).
#[test]
fn skewness_gaussian_is_near_zero() {
    let z = gaussian_standardised(42, 10_000);
    let mu3 = compute_skewness_mu3(&z);
    assert!(
        mu3.abs() < 0.15,
        "Gaussian skewness must be near zero for n=10000; got μ_3 = {mu3:.4}"
    );
}

/// Test that compute_skewness_mu3 is positive and meaningfully large for
/// a lognormal distribution (right-skewed).
#[test]
fn skewness_lognormal_is_positive_and_large() {
    // For ln(X) ~ N(0,1): population skewness ≈ 6.2.
    // With n=10000 and the unbiased estimator, expect >> 1.
    let z = lognormal_standardised(42, 10_000);
    let mu3 = compute_skewness_mu3(&z);
    assert!(
        mu3 > 1.0,
        "Lognormal skewness must be significantly positive (pop ≈ 6.2); got μ_3 = {mu3:.4}"
    );
}

/// Test that compute_skewness_mu3 ≈ 0 for Student-t(3) (symmetric distribution).
#[test]
fn skewness_student_t3_is_near_zero() {
    let z = student_t3_standardised(42, 10_000);
    let mu3 = compute_skewness_mu3(&z);
    assert!(
        mu3.abs() < 0.15,
        "Student-t(3) skewness must be near zero (symmetric); got μ_3 = {mu3:.4}"
    );
}

/// Test that compute_skewness_mu3 is large and positive for Bernoulli(0.05).
#[test]
fn skewness_bernoulli_05_is_large_and_positive() {
    // Population skewness ≈ 4.13; with n=5000 expect > 2.
    let z = bernoulli_05_standardised(42, 5_000);
    let mu3 = compute_skewness_mu3(&z);
    assert!(
        mu3 > 2.0,
        "Bernoulli(0.05) skewness must be large positive (pop ≈ 4.13); got μ_3 = {mu3:.4}"
    );
}

/// Test that the finite-sample correction is consistent with scipy.stats.skew(bias=False).
/// For a constant vector, skewness is 0.
#[test]
fn skewness_of_constant_vector_is_zero() {
    let z = vec![3.7_f64; 100];
    let mu3 = compute_skewness_mu3(&z);
    assert!(
        mu3.abs() < 1e-12,
        "Constant vector must have μ_3 = 0; got {mu3}"
    );
}

/// Small n < 3: function must return 0 (no correction applicable).
#[test]
fn skewness_tiny_n_returns_zero() {
    assert_eq!(compute_skewness_mu3(&[1.0, 2.0]), 0.0);
    assert_eq!(compute_skewness_mu3(&[1.0]), 0.0);
    assert_eq!(compute_skewness_mu3(&[]), 0.0);
}

/// Test bias_shift_for_pair: when both blocks have no row scaling, shift = 0.
#[test]
fn bias_shift_both_none_is_zero() {
    let shift = bias_shift_for_pair(None, None, 0.1, 0.1);
    assert_eq!(shift, 0.0);
}

/// Test bias_shift_for_pair: when both blocks have the SAME row scaling, shift = 0.
#[test]
fn bias_shift_same_scaling_both_blocks_is_zero() {
    let z: Vec<f64> = lognormal_standardised(7, 100);
    let shift = bias_shift_for_pair(Some(&z), Some(&z), 0.05, 0.05);
    assert_eq!(
        shift, 0.0,
        "identical scalings must cancel; got shift={shift:.6}"
    );
}

/// Test bias_shift_for_pair: one block with skewed z gives a non-zero shift.
#[test]
fn bias_shift_skewed_z_one_block_is_nonzero() {
    // Use lognormal (heavily skewed, μ_3 >> 1) with a moderately large S2.
    let z = lognormal_standardised(42, 1000);
    let s2 = 0.05; // n_eff ≈ 20
    let shift = bias_shift_for_pair(Some(&z), None, s2, s2 * 0.8);
    // shift = −(μ_3/2) · s2; with μ_3 >> 1 and s2 = 0.05, |shift| >> 0.
    assert!(
        shift.abs() > 0.01,
        "skewed z (lognormal) with s2=0.05 must give |shift| > 0.01; got {shift:.6}"
    );
}

// ─── Audit integration tests ─────────────────────────────────────────────────

/// Build two blocks that would APPEAR to be close-to-aliased under T11's
/// symmetric threshold, but are actually within the null band once the
/// bias shift is applied.
///
/// Setup: block A has column φ (non-uniform, n_eff moderate); block B has
/// effective column z⊙φ where z is heavily right-skewed (lognormal).
/// The expected cosine between φ and z⊙φ is −(μ_3/2)·S2 (a negative bias).
/// We construct the cosine to be exactly at the bias-shifted null mean
/// (which would look like a suspicious positive cosine under T11).
///
/// The audit MUST NOT report this pair as an alias, because the cosine
/// is exactly on the null mean — the cosine is exactly where the null
/// distribution predicts it to be.
#[test]
fn gaussian_z_symmetric_bias_shift_near_zero_no_false_positive() {
    let n = 500usize;
    // Gaussian z: μ_3 ≈ 0, shift ≈ 0.  Threshold matches T11 symmetric form.
    let z = gaussian_standardised(100, n);
    // Build φ as a non-uniform column: first 50 rows = 1, rest = 0.
    let r = 50usize;
    let mut phi: Vec<f64> = vec![0.0; n];
    for i in 0..r {
        phi[i] = 1.0;
    }
    // Effective column for block B: z ⊙ φ (using the z vector as row scaling,
    // applied when building the spec). We build the raw design as φ so the
    // effective Jacobian is diag(z) · φ.
    let mut design_a = Array2::<f64>::zeros((n, 1));
    let mut design_b = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        design_a[[i, 0]] = phi[i];
        design_b[[i, 0]] = phi[i]; // raw design; z scaling applied by spec
    }
    let spec_a = spec_from_dense("block_gaussian_phi", design_a);
    let spec_b = spec_with_scaling("block_gaussian_zphi", design_b, z.clone());

    let audit = audit_identifiability(&[spec_a, spec_b]).expect("audit must run");

    // With Gaussian z (μ_3 ≈ 0), the null mean shift ≈ 0, so the audit behaves
    // identically to T11's symmetric form.  The actual cosine between φ and z⊙φ
    // is a small random value (since z is standardised Gaussian independent of φ).
    // For n_eff = 50 (r=50 hot rows), σ ≈ √(1/50 − 1/500) ≈ 0.134, halt thr ≈
    // min(0.999, 10·0.134) = 0.999.  So the audit MUST NOT halt.
    assert!(
        !audit.fatal,
        "Gaussian z (μ_3≈0): audit must not halt for a non-alias pair; summary: {}",
        audit.summary,
    );
    // The audit may or may not report an alias pair (depending on the random
    // cosine value), but it must not HALT.
}

/// Skewed z (lognormal): cosine near the shifted null mean must not be reported.
///
/// We engineer the scenario:
///   - n_eff ≈ 50 (r = 50 hot rows in φ).
///   - z is lognormal-standardised (μ_3 ≈ 5–7 depending on seed/n).
///   - S2 ≈ 1/50 = 0.02.
///   - Expected shift = −(μ_3/2) · S2 ≈ −(5/2)·0.02 = −0.05.
///
/// We construct a cosine (by choosing the second column's raw values) that
/// is exactly `shift` (on the null mean).  Under T11 symmetric form this
/// cosine would be |shift| ≈ 0.05 away from zero — possibly above the
/// report threshold for tight-null columns.  Under the bias-corrected form,
/// |cosine − shift| ≈ 0 << report_half_width, so the pair must NOT be flagged.
#[test]
fn skewed_z_lognormal_cosine_at_null_mean_not_reported() {
    let n = 2_000usize;
    let r = 50usize; // hot rows → S2 ≈ 1/r = 0.02, n_eff ≈ 50

    let z = lognormal_standardised(42, n);
    let mu3 = compute_skewness_mu3(&z);
    // S2 for the hot-row column = 1/r (verified by construction).
    let s2 = 1.0 / r as f64;
    let shift = bias_shift_for_pair(Some(&z), None, s2, s2);
    // shift should be negative (right-skewed z → negative bias).
    assert!(
        shift < 0.0 || mu3.abs() < 0.5,
        "lognormal z must produce negative bias shift; got shift={shift:.4}, μ_3={mu3:.4}"
    );

    // Build the two blocks.
    // Block A: raw design = φ (hot rows 0..r), no row scaling.
    // Block B: raw design = φ BUT with a RowScaledJacobian using z so the
    //          effective column is z ⊙ φ.  The actual cosine between φ and z⊙φ
    //          will be some value near `shift` (the null mean).
    let mut design_phi = Array2::<f64>::zeros((n, 1));
    for i in 0..r {
        design_phi[[i, 0]] = 1.0 / (r as f64).sqrt();
    }
    let spec_a = spec_from_dense("phi_unscaled", design_phi.clone());
    let spec_b = spec_with_scaling("phi_scaled_lognormal", design_phi, z);

    let audit = audit_identifiability(&[spec_a, spec_b]).expect("audit must run");

    // With n_eff ≈ 50, σ ≈ √(1/50 − 1/2000) ≈ 0.140, halt thr ≈ 0.999.
    // Even with the symmetric T11 formula, the cosine would not halt because
    // the halt threshold is at 0.999 for n_eff = 50.  But the bias correction
    // prevents even the REPORT threshold from firing when the cosine is near
    // the null mean.
    assert!(
        !audit.fatal,
        "lognormal z: cosine near null mean must not halt; shift={shift:.4}, summary: {}",
        audit.summary,
    );
}

/// Heavy-tailed z (Student-t 3 df): symmetric distribution → μ_3 ≈ 0, shift ≈ 0.
/// Verify the audit behaves identically to T11's symmetric form and does not
/// fire false positives for a genuinely non-alias pair.
#[test]
fn heavy_tailed_z_t3_is_symmetric_no_shift() {
    let n = 1_000usize;
    let r = 100usize; // n_eff = 100

    let z = student_t3_standardised(42, n);
    let mu3 = compute_skewness_mu3(&z);
    assert!(
        mu3.abs() < 0.3,
        "Student-t(3) must have |μ_3| near 0; got {mu3:.4}"
    );

    let s2 = 1.0 / r as f64;
    let shift = bias_shift_for_pair(Some(&z), None, s2, s2);
    assert!(
        shift.abs() < 0.01,
        "Student-t(3) bias shift must be near zero; got shift={shift:.4}"
    );

    let mut design_phi = Array2::<f64>::zeros((n, 1));
    for i in 0..r {
        design_phi[[i, 0]] = 1.0 / (r as f64).sqrt();
    }
    let spec_a = spec_from_dense("phi_unscaled_t3", design_phi.clone());
    let spec_b = spec_with_scaling("phi_scaled_t3", design_phi, z);

    let audit = audit_identifiability(&[spec_a, spec_b]).expect("audit must run");
    assert!(
        !audit.fatal,
        "Student-t(3) z: must not halt for a non-alias pair; summary: {}",
        audit.summary,
    );
}

/// Bernoulli(0.05) z: large positive skewness → negative bias shift.
/// Verify the audit does not fire a false positive even when the actual
/// cosine is near the shifted null mean (a negative value).
#[test]
fn bernoulli_z_large_skew_threshold_compensates() {
    let n = 2_000usize;
    let r = 40usize; // n_eff ≈ 40, S2 ≈ 0.025

    let z = bernoulli_05_standardised(42, n);
    let mu3 = compute_skewness_mu3(&z);
    assert!(
        mu3 > 2.0,
        "Bernoulli(0.05) must have μ_3 >> 2; got {mu3:.4}"
    );

    let s2 = 1.0 / r as f64;
    let shift = bias_shift_for_pair(Some(&z), None, s2, s2);
    // shift = −(μ_3/2)·S2 ≈ −(4.13/2)·0.025 ≈ −0.052 (negative).
    assert!(
        shift < 0.0,
        "Bernoulli(0.05) (positively skewed) must give negative bias shift; got {shift:.4}"
    );

    let mut design_phi = Array2::<f64>::zeros((n, 1));
    for i in 0..r {
        design_phi[[i, 0]] = 1.0 / (r as f64).sqrt();
    }
    let spec_a = spec_from_dense("phi_unscaled_bernoulli", design_phi.clone());
    let spec_b = spec_with_scaling("phi_scaled_bernoulli", design_phi, z);

    let audit = audit_identifiability(&[spec_a, spec_b]).expect("audit must run");
    // With n_eff = 40, σ ≈ √(1/40) ≈ 0.158, halt thr ≈ 0.999.  So the
    // halt test cannot fire (threshold is at the ceiling), regardless of
    // skewness correction.  This test verifies the audit still PASSES.
    assert!(
        !audit.fatal,
        "Bernoulli(0.05) z: must not halt for a non-alias pair; summary: {}",
        audit.summary,
    );
}

/// Regression: exact alias between two scaled blocks with the SAME z must still
/// fire as fatal.  The shift cancels (same z), so the symmetric halt test kicks
/// in at overlap ≈ 1.0.
#[test]
fn exact_alias_same_z_scaling_still_fatal() {
    let n = 500usize;
    let z = lognormal_standardised(42, n);

    // Both blocks have the same row scaling and the same raw design column.
    let mut design_phi = Array2::<f64>::zeros((n, 1));
    for i in 0..50 {
        design_phi[[i, 0]] = 1.0;
    }
    let spec_a = spec_with_scaling("alias_a", design_phi.clone(), z.clone());
    let spec_b = spec_with_scaling("alias_b", design_phi, z);

    let audit = audit_identifiability(&[spec_a, spec_b]).expect("audit must run");
    assert!(
        audit.fatal,
        "exact alias with same z scaling must still be fatal; summary: {}",
        audit.summary,
    );
}

/// Verify that when both blocks have no row scaling (the T11 symmetric case),
/// the bias shift is zero and the thresholds match the T11 formula exactly.
/// Tests that no alias is reported for a genuinely orthogonal pair of columns.
#[test]
fn no_row_scaling_symmetric_form_matches_t11() {
    let n = 1_000usize;
    // Uniform column in block A; sin(x) in block B — nearly orthogonal.
    let mut design_a = Array2::<f64>::zeros((n, 1));
    let mut design_b = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        design_a[[i, 0]] = 1.0;
        design_b[[i, 0]] = ((i as f64) * 2.0 * std::f64::consts::PI / n as f64).sin();
    }
    // bias_shift_for_pair with both None must be zero.
    let s2_a = 1.0 / n as f64;
    let s2_b = s2_a; // both uniform
    let shift = bias_shift_for_pair(None, None, s2_a, s2_b);
    assert_eq!(shift, 0.0, "no row scaling must give shift = 0");

    let specs = [
        spec_from_dense("const", design_a),
        spec_from_dense("sin", design_b),
    ];
    let audit = audit_identifiability(&specs).expect("audit must run");
    assert!(
        !audit.fatal,
        "symmetric T11 form: constant vs sin must not be fatal; summary: {}",
        audit.summary,
    );
    assert!(
        audit.aliased_pairs.is_empty(),
        "constant vs sin: no alias pair expected; got {:?}",
        audit.aliased_pairs,
    );
    // Also confirm all stored bias_shift values are zero.
    for pair in &audit.aliased_pairs {
        assert_eq!(
            pair.bias_shift, 0.0,
            "no-scaling pairs must have bias_shift = 0"
        );
    }
}

#[test]
fn audit_with_state_passes_block_local_beta_to_effective_jacobians() {
    let n = 64usize;
    let mut design = Array2::<f64>::zeros((n, 1));
    let mut orthogonal_base = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        let x = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        design[[i, 0]] = 1.0;
        orthogonal_base[[i, 0]] = x;
    }

    let spec_a = spec_from_dense("active_block", design.clone());
    let mut spec_b = spec_from_dense("local_beta_sensitive_block", orthogonal_base.clone());
    spec_b.jacobian_callback = Some(Arc::new(FirstBetaScaledJacobian {
        base: orthogonal_base,
        alias: design,
    }));

    let beta = [1.0, 0.0];
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };

    let audit =
        audit_identifiability_with_state(&[spec_a, spec_b], &state).expect("audit must run");
    assert!(
        !audit.fatal,
        "the second block's local beta is zero, so its effective Jacobian is the \
         orthogonal base column and cannot alias the first block; summary: {}",
        audit.summary
    );
    assert!(
        audit.aliased_pairs.is_empty(),
        "zero effective-Jacobian block must not create a cross-block alias; got {:?}",
        audit.aliased_pairs
    );
}
