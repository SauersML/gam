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

/// Print interval for rate-limited diagnostics
pub const DIAG_PRINT_INTERVAL: usize = 50;

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

/// Returns (should_print, count) - prints on first occurrence, then every DIAG_PRINT_INTERVAL
pub fn should_emit_grad_diag(counter: &AtomicUsize) -> (bool, usize) {
    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
    let should_print = count == 1 || count % DIAG_PRINT_INTERVAL == 0;
    (should_print, count)
}

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
    if bucket != last || count % MIN_EIG_DIAG_EVERY == 0 {
        H_MIN_EIG_LOG_BUCKET.store(bucket, Ordering::Relaxed);
        true
    } else {
        false
    }
}

// =============================================================================
// Formatting Utilities for Diagnostic Output
// =============================================================================

/// Approximate floating-point equality check for diagnostic deduplication.
pub fn approx_f64(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= abs + rel * a.abs().max(b.abs())
}

/// Format a condition number for display.
pub fn format_cond(cond: f64) -> String {
    if cond.is_finite() {
        format!("{:.2e}", cond)
    } else {
        "N/A".to_string()
    }
}

/// Quantize a value for deduplication (bucketing similar values together).
pub fn quantize_value(value: f64, rel: f64, abs: f64) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    let scale = abs.max(rel * value.abs());
    let quantized = (value / scale).round() * scale;
    if quantized == 0.0 { 0.0 } else { quantized }
}

/// Quantize a vector of values for deduplication.
pub fn quantize_vec(values: &[f64], rel: f64, abs: f64) -> Vec<f64> {
    values
        .iter()
        .map(|&value| quantize_value(value, rel, abs))
        .collect()
}

/// Format a range of values, collapsing to single value if min ≈ max.
pub fn format_range<F>(min: f64, max: f64, fmt: F) -> String
where
    F: Fn(f64) -> String,
{
    if approx_f64(min, max, 1e-6, 1e-9) {
        fmt(min)
    } else {
        format!("[{}, {}]", fmt(min), fmt(max))
    }
}

/// Format a series of values compactly, detecting uniform steps.
pub fn format_compact_series<F>(values: &[f64], fmt: F) -> String
where
    F: Fn(f64) -> String,
{
    if values.is_empty() {
        return "[]".to_string();
    }
    if values.len() == 1 {
        return format!("[{}]", fmt(values[0]));
    }
    let first = values[0];
    let last = values[values.len() - 1];
    let step = values[1] - values[0];
    let uniform_step = values
        .windows(2)
        .all(|pair| approx_f64(pair[1] - pair[0], step, 1e-6, 1e-9));
    if uniform_step {
        return format!("[{}..{} step {}]", fmt(first), fmt(last), fmt(step));
    }
    let (min, max) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    format!("[{}..{}] n={}", fmt(min), fmt(max), values.len())
}

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
    pub emit_warnings: bool,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            kkt_tolerance: 1e-4,
            fd_step_size: 1e-5,
            rel_error_threshold: 0.1,
            emit_warnings: true,
        }
    }
}

/// Result of envelope theorem (KKT) audit
#[derive(Clone, Debug)]
pub struct EnvelopeAudit {
    /// Norm of the inner KKT residual ∇_β L(β*, ρ)
    pub kkt_residual_norm: f64,
    /// Ridge used by the inner solver
    pub inner_ridge: f64,
    /// Ridge assumed by the outer gradient calculation
    pub outer_ridge: f64,
    /// Whether the envelope theorem is violated
    pub is_violated: bool,
    /// Human-readable diagnostic message
    pub message: String,
}

impl fmt::Display for EnvelopeAudit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Result of component-wise finite difference check
#[derive(Clone, Debug)]
pub struct ComponentFdResult {
    /// Component name (e.g., "D_p", "log|H|", "log|S|")
    pub component: String,
    /// Analytic gradient value for this component
    pub analytic: f64,
    /// Finite difference approximation
    pub numeric: f64,
    /// Relative error
    pub rel_error: f64,
    /// Whether this component has a significant mismatch
    pub has_mismatch: bool,
}

impl fmt::Display for ComponentFdResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.has_mismatch { "MISMATCH" } else { "OK" };
        write!(
            f,
            "[{}] {}: analytic={:+.4e}, numeric={:+.4e}, rel_error={:.2e}",
            status, self.component, self.analytic, self.numeric, self.rel_error
        )
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
    pub pirls_ridge: f64,
    /// Ridge used in LAML cost function
    pub cost_ridge: f64,
    /// Ridge used in gradient calculation
    pub gradient_ridge: f64,
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
    pub envelope_audit: Option<EnvelopeAudit>,
    /// Component-wise FD results for each penalty dimension
    pub component_fd: Vec<Vec<ComponentFdResult>>,
    /// Spectral bleed results for each penalty
    pub spectral_bleed: Vec<SpectralBleedResult>,
    /// Dual-ridge consistency result
    pub dual_ridge: Option<DualRidgeResult>,
    /// Total analytic gradient
    pub analytic_gradient: Option<Array1<f64>>,
    /// Total numeric gradient (FD)
    pub numeric_gradient: Option<Array1<f64>>,
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
        let envelope_issue = self.envelope_audit.as_ref().is_some_and(|a| a.is_violated);
        let component_issue = self.component_fd.iter().flatten().any(|c| c.has_mismatch);
        let bleed_issue = self.spectral_bleed.iter().any(|s| s.has_bleed);
        let ridge_issue = self.dual_ridge.as_ref().is_some_and(|r| r.has_mismatch);
        envelope_issue || component_issue || bleed_issue || ridge_issue
    }

    /// Generate a summary string of all issues found
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref audit) = self.envelope_audit {
            if audit.is_violated {
                lines.push(format!("[DIAG] {}", audit));
            }
        }

        for (k, components) in self.component_fd.iter().enumerate() {
            for comp in components {
                if comp.has_mismatch {
                    lines.push(format!("[DIAG] ρ[{}] {}", k, comp));
                }
            }
        }

        for bleed in &self.spectral_bleed {
            if bleed.has_bleed {
                lines.push(format!("[DIAG] {}", bleed));
            }
        }

        if let Some(ref ridge) = self.dual_ridge {
            if ridge.has_mismatch {
                lines.push(format!("[DIAG] {}", ridge));
            }
        }

        if lines.is_empty() {
            "No gradient diagnostic issues detected.".to_string()
        } else {
            lines.join("\n")
        }
    }

    /// Log the full diagnostic report to stderr
    pub fn log_to_stderr(&self) {
        if self.has_issues() {
            eprintln!("\n=== GRADIENT DIAGNOSTIC REPORT ===");
            eprintln!("{}", self.summary());
            eprintln!("==================================\n");
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
/// * `reference_gradient` - Reference gradient scale (typically S_λ β) for relative normalization
/// * `ridge_used` - Ridge added by PIRLS for stabilization
/// * `beta` - Current coefficient estimate
/// * `tolerance` - Threshold for flagging violations
pub fn compute_envelope_audit(
    kkt_residual_norm: f64,
    reference_gradient: &Array1<f64>,
    ridge_used: f64,
    ridge_assumed: f64,
    beta: &Array1<f64>,
    abs_tolerance: f64,
    rel_tolerance: f64,
) -> EnvelopeAudit {
    let kkt_norm = kkt_residual_norm;
    let penalty_norm = reference_gradient.dot(reference_gradient).sqrt();
    let beta_norm = beta.dot(beta).sqrt();
    let scale = penalty_norm.max((ridge_assumed.abs() * beta_norm).max(1e-12));
    let rel_kkt = if scale > 0.0 { kkt_norm / scale } else { 0.0 };
    let ridge_mismatch = (ridge_used - ridge_assumed).abs() > 1e-12;
    let kkt_violation = kkt_norm > abs_tolerance && rel_kkt > rel_tolerance;
    let is_violated = kkt_violation || ridge_mismatch;

    let message = if ridge_mismatch && kkt_violation {
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
    } else if kkt_violation {
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
        inner_ridge: ridge_used,
        outer_ridge: ridge_assumed,
        is_violated,
        message,
    }
}

// =============================================================================
// Strategy 2: Component-wise Finite Difference
// =============================================================================

/// Result of evaluating each cost component at a given rho
#[derive(Clone, Debug)]
pub struct CostComponents {
    /// Penalized deviance: D_p = -2*loglik + β'S_λ β
    pub penalized_deviance: f64,
    /// Log-determinant of Hessian: log|H|
    pub log_det_h: f64,
    /// Log-determinant of penalty (pseudo): log|S_λ|_+
    pub log_det_s: f64,
    /// Total LAML cost
    pub total: f64,
    /// Optional Firth log-det contribution
    pub firth_log_det: Option<f64>,
}

/// Compute component-wise gradient comparison between analytic and FD.
///
/// This breaks down the LAML objective into:
/// 1. Penalized deviance term: D_p/(2φ) → gradient includes dD_p/dρ_k
/// 2. Hessian log-det term: 0.5 log|H| → gradient includes 0.5 tr(H⁻¹ dH/dρ_k)
/// 3. Penalty log-det term: -0.5 log|S_λ|_+ → gradient includes -0.5 det1[k]
///
/// By checking each component separately, we can isolate which term is wrong.
pub fn compute_component_fd(
    components_plus: &CostComponents,
    components_minus: &CostComponents,
    analytic_dp_grad: f64,
    analytic_logh_grad: f64,
    analytic_logs_grad: f64,
    h: f64,
    rel_threshold: f64,
) -> Vec<ComponentFdResult> {
    let mut results = Vec::with_capacity(4);

    // D_p component
    let numeric_dp =
        (components_plus.penalized_deviance - components_minus.penalized_deviance) / (2.0 * h);
    let dp_denom = analytic_dp_grad.abs().max(numeric_dp.abs()).max(1e-8);
    let dp_rel = (analytic_dp_grad - numeric_dp).abs() / dp_denom;
    results.push(ComponentFdResult {
        component: "D_p (penalized deviance)".to_string(),
        analytic: analytic_dp_grad,
        numeric: numeric_dp,
        rel_error: dp_rel,
        has_mismatch: dp_rel > rel_threshold,
    });

    // log|H| component
    let numeric_logh = (components_plus.log_det_h - components_minus.log_det_h) / (2.0 * h);
    let logh_denom = analytic_logh_grad.abs().max(numeric_logh.abs()).max(1e-8);
    let logh_rel = (analytic_logh_grad - numeric_logh).abs() / logh_denom;
    results.push(ComponentFdResult {
        component: "log|H| (Hessian)".to_string(),
        analytic: analytic_logh_grad,
        numeric: numeric_logh,
        rel_error: logh_rel,
        has_mismatch: logh_rel > rel_threshold,
    });

    // log|S| component
    let numeric_logs = (components_plus.log_det_s - components_minus.log_det_s) / (2.0 * h);
    let logs_denom = analytic_logs_grad.abs().max(numeric_logs.abs()).max(1e-8);
    let logs_rel = (analytic_logs_grad - numeric_logs).abs() / logs_denom;
    results.push(ComponentFdResult {
        component: "log|S|_+ (penalty)".to_string(),
        analytic: analytic_logs_grad,
        numeric: numeric_logs,
        rel_error: logs_rel,
        has_mismatch: logs_rel > rel_threshold,
    });

    // Total
    let analytic_total = analytic_dp_grad + analytic_logh_grad + analytic_logs_grad;
    let numeric_total = (components_plus.total - components_minus.total) / (2.0 * h);
    let total_denom = analytic_total.abs().max(numeric_total.abs()).max(1e-8);
    let total_rel = (analytic_total - numeric_total).abs() / total_denom;
    results.push(ComponentFdResult {
        component: "Total LAML".to_string(),
        analytic: analytic_total,
        numeric: numeric_total,
        rel_error: total_rel,
        has_mismatch: total_rel > rel_threshold,
    });

    results
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
pub fn compute_spectral_bleed(
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
    let u_rows = u_truncated.nrows().min(h_inv_u_truncated.nrows());
    let mut m_perp = Array2::<f64>::zeros((truncated_count, truncated_count));
    for i in 0..truncated_count {
        for j in 0..truncated_count {
            let mut sum = 0.0;
            for r in 0..u_rows {
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
/// When the Hessian is non-positive-definite, ensure_positive_definite_with_ridge
/// adds a stabilization ridge during P-IRLS. This ridge changes the objective
/// surface being optimized. If the gradient calculation uses a different ridge
/// value, it will point in the wrong direction.
///
/// # Arguments
/// * `pirls_ridge` - Ridge actually used during P-IRLS iteration
/// * `cost_ridge` - Ridge used when computing LAML cost
/// * `gradient_ridge` - Ridge assumed when computing analytic gradient
/// * `beta` - Current coefficient estimate
pub fn compute_dual_ridge_check(
    pirls_ridge: f64,
    cost_ridge: f64,
    gradient_ridge: f64,
    beta: &Array1<f64>,
) -> DualRidgeResult {
    let beta_norm_sq = beta.dot(beta);
    let beta_norm = beta_norm_sq.sqrt();

    let ridge_impact = pirls_ridge * beta_norm;
    let phantom_penalty = 0.5 * pirls_ridge * beta_norm_sq;

    let pirls_cost_mismatch = (pirls_ridge - cost_ridge).abs() > 1e-12;
    let pirls_grad_mismatch = (pirls_ridge - gradient_ridge).abs() > 1e-12;
    let cost_grad_mismatch = (cost_ridge - gradient_ridge).abs() > 1e-12;
    let has_mismatch = pirls_cost_mismatch || pirls_grad_mismatch || cost_grad_mismatch;

    let message = if has_mismatch {
        let mut mismatches = Vec::new();
        if pirls_cost_mismatch {
            mismatches.push(format!(
                "PIRLS({:.2e}) vs Cost({:.2e})",
                pirls_ridge, cost_ridge
            ));
        }
        if pirls_grad_mismatch {
            mismatches.push(format!(
                "PIRLS({:.2e}) vs Gradient({:.2e})",
                pirls_ridge, gradient_ridge
            ));
        }
        if cost_grad_mismatch {
            mismatches.push(format!(
                "Cost({:.2e}) vs Gradient({:.2e})",
                cost_ridge, gradient_ridge
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
    } else if pirls_ridge > 0.0 {
        format!(
            "Ridge Consistency OK: All stages use ridge = {:.2e}. ||β|| = {:.2e}, phantom penalty = {:.2e}",
            pirls_ridge, beta_norm, phantom_penalty
        )
    } else {
        "Ridge Consistency OK: No stabilization ridge required.".to_string()
    };

    DualRidgeResult {
        pirls_ridge,
        cost_ridge,
        gradient_ridge,
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
pub fn gradient_perturbation_consistency(
    grad_at_rho: &Array1<f64>,
    grad_at_rho_plus: &Array1<f64>,
    fd_slope: &Array1<f64>,
    rel_threshold: f64,
) -> (bool, f64, f64) {
    // Average gradient
    let avg_grad = (grad_at_rho + grad_at_rho_plus).mapv(|v| v / 2.0);

    // Check if average matches FD better than the gradient at rho
    let diff_avg = &avg_grad - fd_slope;
    let diff_rho = grad_at_rho - fd_slope;

    let rel_error_avg = diff_avg.dot(&diff_avg).sqrt() / fd_slope.dot(fd_slope).sqrt().max(1e-8);
    let rel_error_rho = diff_rho.dot(&diff_rho).sqrt() / fd_slope.dot(fd_slope).sqrt().max(1e-8);

    let has_bias = rel_error_avg < rel_error_rho * 0.5 && rel_error_rho > rel_threshold;

    (has_bias, rel_error_avg, rel_error_rho)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_envelope_audit_no_violation() {
        let reference = arr1(&[0.0, 0.0, 0.0]);
        let beta = arr1(&[0.1, 0.2, 0.3]);
        let result = compute_envelope_audit(0.0, &reference, 0.0, 0.0, &beta, 1e-8, 1e-6);

        assert!(!result.is_violated);
        assert!(result.kkt_residual_norm < 1e-10);
    }

    #[test]
    fn test_envelope_audit_ridge_mismatch() {
        let reference = arr1(&[0.9, 1.9, 2.9]);
        let beta = arr1(&[1.0, 1.0, 1.0]);
        let result = compute_envelope_audit(1e-10, &reference, 0.1, 0.0, &beta, 1e-8, 1e-6);

        // PIRLS used ridge 0.1, but gradient assumes 0.0
        assert!(result.is_violated);
        assert!(
            result.message.contains("Ridge Mismatch")
                || result.message.contains("Envelope Violation")
        );
    }

    #[test]
    fn test_dual_ridge_consistency_ok() {
        let beta = arr1(&[1.0, 2.0, 3.0]);
        let result = compute_dual_ridge_check(0.0, 0.0, 0.0, &beta);
        assert!(!result.has_mismatch);
    }

    #[test]
    fn test_dual_ridge_consistency_mismatch() {
        let beta = arr1(&[1.0, 2.0, 3.0]);
        let result = compute_dual_ridge_check(1e-4, 0.0, 0.0, &beta);
        assert!(result.has_mismatch);
        assert!(result.phantom_penalty > 0.0);
    }
}
