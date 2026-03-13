//! Gradient Diagnostic Strategies for LAML/REML Optimization
//!
//! This module implements four diagnostic strategies to identify root causes of
//! gradient calculation mismatches between analytic and finite-difference gradients:
//!
//! 1. KKT Audit (Envelope Theorem Check): Detects violations of the stationarity
//!    assumption used in implicit differentiation.
//!
//! 2. Component-wise Finite Difference: Breaks down the total cost into components
//!    (D_p, log|H|, log|S|) and checks each gradient term separately.
//!
//! 3. Spectral Bleed Trace: Detects when truncated eigenspace corrections are
//!    inconsistent with the penalty's energy in that subspace.
//!
//! 4. Dual-Ridge Consistency Check: Verifies that the ridge used by the inner
//!    solver (PIRLS) matches what the outer gradient calculation assumes.

use ndarray::{Array1, Array2, ArrayView2};
use std::fmt;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

// =============================================================================
// Rate-Limited Diagnostic Output
// =============================================================================
// These helpers prevent diagnostic spam while ensuring important messages are seen.
// Pattern: show first occurrence, then every Nth occurrence, with count indicator.

/// Rate-limited diagnostic counters for gradient calculations
pub static GRAD_DIAG_BETA_COLLAPSE_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static GRAD_DIAG_DELTA_ZERO_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static GRAD_DIAG_LOGH_CLAMPED_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static GRAD_DIAG_KKT_SKIP_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Rate-limited diagnostic for Hessian minimum eigenvalue warnings
pub static H_MIN_EIG_LOG_BUCKET: AtomicI32 = AtomicI32::new(i32::MIN);
pub static H_MIN_EIG_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
pub const MIN_EIG_DIAG_EVERY: usize = 200;
pub const MIN_EIG_DIAG_THRESHOLD: f64 = 1e-4;

/// Rate-limited check for Hessian minimum eigenvalue diagnostics.
/// Returns true if this eigenvalue warrants a diagnostic message.
pub fn should_emit_h_min_eig_diag(min_eig: f64) -> bool {
    if !min_eig.is_finite() || min_eig <= 0.0 {
        return true;
    }
    if min_eig >= MIN_EIG_DIAG_THRESHOLD {
        return false;
    }
    let bucket = if min_eig.is_finite() && min_eig > 0.0 {
        min_eig.log10().floor() as i32
    } else {
        i32::MIN
    };
    let last = H_MIN_EIG_LOG_BUCKET.load(Ordering::Relaxed);
    let count = H_MIN_EIG_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
    if bucket != last || count.is_multiple_of(MIN_EIG_DIAG_EVERY) {
        H_MIN_EIG_LOG_BUCKET.store(bucket, Ordering::Relaxed);
        true
    } else {
        false
    }
}

// =============================================================================
// Formatting Utilities for Diagnostic Output
// =============================================================================

/// Configuration for gradient diagnostics
#[derive(Clone, Debug)]
pub struct DiagnosticConfig {
    /// Tolerance for KKT residual norm (envelope theorem violation)
    pub kkt_tolerance: f64,
    /// Step size for finite difference calculations
    pub fd_step_size: f64,
    /// Relative error threshold for flagging issues
    pub rel_error_threshold: f64,
    /// Whether to emit warnings to stderr
    pub emitwarnings: bool,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            kkt_tolerance: 1e-4,
            fd_step_size: 1e-5,
            rel_error_threshold: 0.1,
            emitwarnings: true,
        }
    }
}

/// Result of envelope theorem (KKT) audit
#[derive(Clone, Debug)]
pub struct EnvelopeAudit {
    /// Norm of the inner KKT residual ∇_β L(β*, ρ)
    pub kkt_residual_norm: f64,
    /// Ridge used by the inner solver
    pub innerridge: f64,
    /// Ridge assumed by the outer gradient calculation
    pub outerridge: f64,
    /// Whether the envelope theorem is violated
    pub isviolated: bool,
    /// Human-readable diagnostic message
    pub message: String,
}

impl fmt::Display for EnvelopeAudit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Result of spectral bleed trace diagnostic
#[derive(Clone, Debug)]
pub struct SpectralBleedResult {
    pub penalty_k: usize,
    /// Energy of penalty S_k in the truncated subspace: trace(U_⊥' S_k U_⊥)
    pub truncated_energy: f64,
    /// Correction term actually applied in the gradient
    pub applied_correction: f64,
    /// Whether there's a spectral bleed issue
    pub has_bleed: bool,
    /// Human-readable diagnostic message
    pub message: String,
}

impl fmt::Display for SpectralBleedResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Result of dual-ridge consistency check
#[derive(Clone, Debug)]
pub struct DualRidgeResult {
    /// Ridge used during P-IRLS optimization
    pub pirlsridge: f64,
    /// Ridge used in LAML cost function
    pub costridge: f64,
    /// Ridge used in gradient calculation
    pub gradientridge: f64,
    /// Effective ridge impact: ||ridge * β||
    pub ridge_impact: f64,
    /// Phantom penalty contribution: 0.5 * ridge * ||β||²
    pub phantom_penalty: f64,
    /// Whether there's a ridge mismatch
    pub has_mismatch: bool,
    /// Human-readable diagnostic message
    pub message: String,
}

impl fmt::Display for DualRidgeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Complete diagnostic report for a gradient evaluation
#[derive(Clone, Debug, Default)]
pub struct GradientDiagnosticReport {
    /// Envelope theorem audit results
    pub envelopeaudit: Option<EnvelopeAudit>,
    /// Spectral bleed results for each penalty
    pub spectral_bleed: Vec<SpectralBleedResult>,
    /// Dual-ridge consistency result
    pub dualridge: Option<DualRidgeResult>,
    /// Total analytic gradient
    pub analytic_gradient: Option<Array1<f64>>,
    /// Total numeric gradient (FD)
    pub numericgradient: Option<Array1<f64>>,
    /// Per-component relative L2 error
    pub component_rel_errors: Option<Array1<f64>>,
}

impl GradientDiagnosticReport {
    /// Create an empty report
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any diagnostics detected issues
    pub fn has_issues(&self) -> bool {
        let envelope_issue = self.envelopeaudit.as_ref().is_some_and(|a| a.isviolated);
        let bleed_issue = self.spectral_bleed.iter().any(|s| s.has_bleed);
        let ridge_issue = self.dualridge.as_ref().is_some_and(|r| r.has_mismatch);
        envelope_issue || bleed_issue || ridge_issue
    }

    /// Generate a summary string of all issues found
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref audit) = self.envelopeaudit
            && audit.isviolated
        {
            lines.push(format!("[DIAG] {}", audit));
        }

        for bleed in &self.spectral_bleed {
            if bleed.has_bleed {
                lines.push(format!("[DIAG] {}", bleed));
            }
        }

        if let Some(ref ridge) = self.dualridge
            && ridge.has_mismatch
        {
            lines.push(format!("[DIAG] {}", ridge));
        }

        if lines.is_empty() {
            "No gradient diagnostic issues detected.".to_string()
        } else {
            lines.join("\n")
        }
    }
}

// =============================================================================
// Strategy 1: Envelope Theorem (KKT) Audit
// =============================================================================

/// Compute the inner KKT residual to detect envelope theorem violations.
///
/// The analytic gradient calculation assumes that P-IRLS found an exact stationary
/// point where ∇_β L = 0. If this is not true (due to stabilization ridge, Firth
/// adjustments, or early termination), the "indirect term" of the chain rule becomes
/// significant and the gradient will be wrong.
///
/// # Arguments
/// * `kkt_residual_norm` - Norm of the full inner gradient ||∇_β L|| at the PIRLS solution
/// * `referencegradient` - Reference gradient scale (typically S_λ β) for relative normalization
/// * `ridge_used` - Ridge added by PIRLS for stabilization
/// * `beta` - Current coefficient estimate
/// * `tolerance` - Threshold for flagging violations
pub fn compute_envelopeaudit(
    kkt_residual_norm: f64,
    referencegradient: &Array1<f64>,
    ridge_used: f64,
    ridge_assumed: f64,
    beta: &Array1<f64>,
    abs_tolerance: f64,
    rel_tolerance: f64,
) -> EnvelopeAudit {
    let kkt_norm = kkt_residual_norm;
    let penalty_norm = referencegradient.dot(referencegradient).sqrt();
    let beta_norm = beta.dot(beta).sqrt();
    let scale = penalty_norm.max((ridge_assumed.abs() * beta_norm).max(1e-12));
    let rel_kkt = if scale > 0.0 { kkt_norm / scale } else { 0.0 };
    let ridge_mismatch = (ridge_used - ridge_assumed).abs() > 1e-12;
    let kktviolation = kkt_norm > abs_tolerance && rel_kkt > rel_tolerance;
    let isviolated = kktviolation || ridge_mismatch;

    let message = if ridge_mismatch && kktviolation {
        format!(
            "Envelope Violation: Inner solver ridge = {:.2e}, Outer gradient assumes ridge = {:.2e}. \
             KKT residual norm = {:.2e} (abs tol = {:.2e}, rel tol = {:.2e}). Unaccounted gradient energy: {:.2e}",
            ridge_used, ridge_assumed, kkt_norm, abs_tolerance, rel_tolerance, kkt_norm
        )
    } else if ridge_mismatch {
        format!(
            "Ridge Mismatch: PIRLS optimized for H + {:.2e}*I, but Gradient calculated for H + {:.2e}*I",
            ridge_used, ridge_assumed
        )
    } else if kktviolation {
        format!(
            "Envelope Violation: KKT residual ||∇_β L|| = {:.2e} (rel {:.2e}) exceeds tolerances (abs {:.2e}, rel {:.2e}). \
             Inner solver may not have converged to true stationary point.",
            kkt_norm, rel_kkt, abs_tolerance, rel_tolerance
        )
    } else {
        format!(
            "Envelope OK: KKT residual = {:.2e} (rel {:.2e}), ridge match = {:.2e}",
            kkt_norm, rel_kkt, ridge_used
        )
    };

    EnvelopeAudit {
        kkt_residual_norm: kkt_norm,
        innerridge: ridge_used,
        outerridge: ridge_assumed,
        isviolated,
        message,
    }
}

// =============================================================================
// Strategy 3: Spectral Bleed Trace
// =============================================================================

/// Compute the spectral bleed diagnostic for truncation consistency.
///
/// When eigenvalues are truncated in the penalty matrix (to compute log|S|_+),
/// the gradient must include a correction term for the truncated-subspace
/// H-weighted trace contribution. This diagnostic checks if the correction is adequate.
///
/// Coordinate/path requirement:
/// - `r_k`, `u_truncated`, and `h_inv_u_truncated` must all be represented in
///   the same coefficient frame.
/// - If any of these are mixed between original and transformed bases, the
///   expected-vs-applied correction comparison here will show large systematic
///   mismatch even when algebra is otherwise correct.
///
/// # Arguments
/// * `r_k` - Penalty root matrix for penalty k (R_k where S_k = R_k' R_k)
/// * `u_truncated` - Eigenvectors of the truncated (null) subspace
/// * `h_inv_u_truncated` - H⁻¹ U_⊥ (pre-solved for efficiency)
/// * `lambda_k` - Current lambda for penalty k
/// * `applied_correction` - The correction term currently applied in the gradient
/// * `rel_threshold` - Relative threshold for flagging issues
pub fn computespectral_bleed(
    penalty_k: usize,
    r_k: ArrayView2<f64>,
    u_truncated: ArrayView2<f64>,
    h_inv_u_truncated: ArrayView2<f64>,
    lambda_k: f64,
    applied_correction: f64,
    rel_threshold: f64,
) -> SpectralBleedResult {
    let truncated_count = u_truncated.ncols();
    let rank_k = r_k.nrows();

    if truncated_count == 0 || rank_k == 0 {
        return SpectralBleedResult {
            penalty_k,
            truncated_energy: 0.0,
            applied_correction,
            has_bleed: false,
            message: format!(
                "Penalty {} has no truncated modes, no bleed possible.",
                penalty_k
            ),
        };
    }

    // Compute W_k = R_k U_⊥ (rank_k × truncated_count)
    let r_k_cols = r_k.ncols().min(u_truncated.nrows());
    let mut w_k = Array2::<f64>::zeros((rank_k, truncated_count));
    for i in 0..rank_k {
        for j in 0..truncated_count {
            let mut sum = 0.0;
            for l in 0..r_k_cols {
                sum += r_k[(i, l)] * u_truncated[(l, j)];
            }
            w_k[(i, j)] = sum;
        }
    }

    // Compute M_⊥ = U_⊥' H⁻¹ U_⊥ (truncated_count × truncated_count)
    let urows = u_truncated.nrows().min(h_inv_u_truncated.nrows());
    let mut m_perp = Array2::<f64>::zeros((truncated_count, truncated_count));
    for i in 0..truncated_count {
        for j in 0..truncated_count {
            let mut sum = 0.0;
            for r in 0..urows {
                sum += u_truncated[(r, i)] * h_inv_u_truncated[(r, j)];
            }
            m_perp[(i, j)] = sum;
        }
    }

    // Error = tr(M_⊥ * W_k' W_k) = λ_k * tr(U_⊥' H⁻¹ U_⊥ * U_⊥' S_k U_⊥)
    let mut trace_error = 0.0;
    for i in 0..truncated_count {
        for j in 0..truncated_count {
            let mut wtw_ij = 0.0;
            for l in 0..rank_k {
                wtw_ij += w_k[(l, i)] * w_k[(l, j)];
            }
            trace_error += m_perp[(i, j)] * wtw_ij;
        }
    }

    let expected_correction = 0.5 * lambda_k * trace_error;
    let truncated_energy = trace_error;

    // Check if correction matches expected
    let denom = expected_correction
        .abs()
        .max(applied_correction.abs())
        .max(1e-8);
    let rel_diff = (expected_correction - applied_correction).abs() / denom;
    let has_bleed = rel_diff > rel_threshold && truncated_energy.abs() > 1e-6;

    let message = if has_bleed {
        format!(
            "Spectral Bleed at k={}: Penalty S_{} has H-weighted trace term {:.2e} in truncated subspace. \
             Expected correction {:.2e}, but applied {:.2e} (rel diff = {:.1}%)",
            penalty_k,
            penalty_k,
            truncated_energy,
            expected_correction,
            applied_correction,
            rel_diff * 100.0
        )
    } else {
        format!(
            "Spectral OK at k={}: Truncated H-weighted trace term = {:.2e}, correction matches.",
            penalty_k, truncated_energy
        )
    };

    SpectralBleedResult {
        penalty_k,
        truncated_energy,
        applied_correction,
        has_bleed,
        message,
    }
}

// =============================================================================
// Strategy 4: Dual-Ridge Consistency Check
// =============================================================================

/// Check consistency between the ridge used in different stages of computation.
///
/// When the Hessian is non-positive-definite, ensure_positive_definitewithridge
/// adds a stabilization ridge during P-IRLS. This ridge changes the objective
/// surface being optimized. If the gradient calculation uses a different ridge
/// value, it will point in the wrong direction.
///
/// # Arguments
/// * `pirlsridge` - Ridge actually used during P-IRLS iteration
/// * `costridge` - Ridge used when computing LAML cost
/// * `gradientridge` - Ridge assumed when computing analytic gradient
/// * `beta` - Current coefficient estimate
pub fn compute_dualridge_check(
    pirlsridge: f64,
    costridge: f64,
    gradientridge: f64,
    beta: &Array1<f64>,
) -> DualRidgeResult {
    let beta_norm_sq = beta.dot(beta);
    let beta_norm = beta_norm_sq.sqrt();

    let ridge_impact = pirlsridge * beta_norm;
    let phantom_penalty = 0.5 * pirlsridge * beta_norm_sq;

    let pirlscost_mismatch = (pirlsridge - costridge).abs() > 1e-12;
    let pirlsgrad_mismatch = (pirlsridge - gradientridge).abs() > 1e-12;
    let costgrad_mismatch = (costridge - gradientridge).abs() > 1e-12;
    let has_mismatch = pirlscost_mismatch || pirlsgrad_mismatch || costgrad_mismatch;

    let message = if has_mismatch {
        let mut mismatches = Vec::new();
        if pirlscost_mismatch {
            mismatches.push(format!(
                "PIRLS({:.2e}) vs Cost({:.2e})",
                pirlsridge, costridge
            ));
        }
        if pirlsgrad_mismatch {
            mismatches.push(format!(
                "PIRLS({:.2e}) vs Gradient({:.2e})",
                pirlsridge, gradientridge
            ));
        }
        if costgrad_mismatch {
            mismatches.push(format!(
                "Cost({:.2e}) vs Gradient({:.2e})",
                costridge, gradientridge
            ));
        }
        format!(
            "Ridge Mismatch detected: {}. Effective ridge impact on ||β|| = {:.2e}. \
             Phantom penalty = {:.2e}. The surface being differentiated differs from \
             the surface being optimized.",
            mismatches.join(", "),
            ridge_impact,
            phantom_penalty
        )
    } else if pirlsridge > 0.0 {
        format!(
            "Ridge Consistency OK: All stages use ridge = {:.2e}. ||β|| = {:.2e}, phantom penalty = {:.2e}",
            pirlsridge, beta_norm, phantom_penalty
        )
    } else {
        "Ridge Consistency OK: No stabilization ridge required.".to_string()
    };

    DualRidgeResult {
        pirlsridge,
        costridge,
        gradientridge,
        ridge_impact,
        phantom_penalty,
        has_mismatch,
        message,
    }
}

// =============================================================================
// Gradient-at-Perturbation Consistency Check (Bonus Strategy)
// =============================================================================

/// Check gradient internal consistency by verifying the average of gradients at
/// perturbed points matches the FD slope.
///
/// If (grad(ρ+ε) + grad(ρ))/2 ≈ FD_slope but grad(ρ) alone doesn't, there's a
/// bias term that doesn't cancel—often a sign of missing terms in the derivative.
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_envelopeaudit_noviolation() {
        let reference = arr1(&[0.0, 0.0, 0.0]);
        let beta = arr1(&[0.1, 0.2, 0.3]);
        let result = compute_envelopeaudit(0.0, &reference, 0.0, 0.0, &beta, 1e-8, 1e-6);

        assert!(!result.isviolated);
        assert!(result.kkt_residual_norm < 1e-10);
    }

    #[test]
    fn test_envelopeauditridge_mismatch() {
        let reference = arr1(&[0.9, 1.9, 2.9]);
        let beta = arr1(&[1.0, 1.0, 1.0]);
        let result = compute_envelopeaudit(1e-10, &reference, 0.1, 0.0, &beta, 1e-8, 1e-6);

        // PIRLS used ridge 0.1, but gradient assumes 0.0
        assert!(result.isviolated);
        assert!(
            result.message.contains("Ridge Mismatch")
                || result.message.contains("Envelope Violation")
        );
    }

    #[test]
    fn test_dualridge_consistency_ok() {
        let beta = arr1(&[1.0, 2.0, 3.0]);
        let result = compute_dualridge_check(0.0, 0.0, 0.0, &beta);
        assert!(!result.has_mismatch);
    }

    #[test]
    fn test_dualridge_consistency_mismatch() {
        let beta = arr1(&[1.0, 2.0, 3.0]);
        let result = compute_dualridge_check(1e-4, 0.0, 0.0, &beta);
        assert!(result.has_mismatch);
        assert!(result.phantom_penalty > 0.0);
    }
}
