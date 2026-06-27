//! Analytic diagnostic helpers for LAML/REML optimization.
//!
//! Production diagnostics inspect analytic invariants only. Runtime fitting,
//! prediction, and diagnostic APIs must consume quantities the optimizer
//! already computes. This module implements diagnostic strategies that identify
//! root causes of gradient pathologies from those analytic quantities:
//!
//! 1. KKT Audit (Envelope Theorem Check): Detects violations of the stationarity
//!    assumption used in implicit differentiation.
//!
//! 2. Spectral Bleed Trace: Detects when truncated eigenspace corrections are
//!    inconsistent with the penalty's energy in that subspace.
//!
//! 3. Dual-Ridge Consistency Check: Verifies that the ridge used by the inner
//!    solver (PIRLS) matches what the outer gradient calculation assumes.

use ndarray::Array1;
use std::fmt;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

// =============================================================================
// Rate-Limited Diagnostic Output
// =============================================================================
// These helpers prevent diagnostic spam while ensuring important messages are seen.
// Pattern: show first occurrence, then every Nth occurrence, with count indicator.

/// Rate-limited diagnostic counters for gradient calculations
pub static GRAD_DIAG_BETA_COLLAPSE_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of outer-gradient evaluations where the proposed `δρ` was exactly
/// zero (Newton step degenerated to no-op). Diagnostic only.
pub static GRAD_DIAG_DELTA_ZERO_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of outer-gradient evaluations where the `log h(ρ)` quantity was
/// clamped against the floor `log(eps)` to keep downstream divides
/// well-defined.
pub static GRAD_DIAG_LOGH_CLAMPED_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of outer-gradient evaluations that skipped the KKT envelope audit
/// because no reference scale was available (β all zero or penalty
/// reference vector all zero).
pub static GRAD_DIAG_KKT_SKIP_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Rate-limited diagnostic for Hessian minimum eigenvalue warnings
pub static H_MIN_EIG_LOG_BUCKET: AtomicI32 = AtomicI32::new(i32::MIN);
/// Count of `should_emit_h_min_eig_diag` invocations that have ever been
/// considered for emission; used together with `H_MIN_EIG_LOG_BUCKET` to
/// rate-limit one diagnostic per decade-magnitude bucket and per
/// `MIN_EIG_DIAG_EVERY` repeats within the same bucket.
pub static H_MIN_EIG_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Repeat period within a magnitude bucket for the Hessian-minimum-eigenvalue
/// diagnostic: after the first emission for a bucket, every Nth subsequent
/// invocation also emits.
pub const MIN_EIG_DIAG_EVERY: usize = 200;
/// Threshold below which a positive Hessian minimum eigenvalue is treated as
/// nearly-singular and routed through the rate-limited diagnostic.
pub const MIN_EIG_DIAG_THRESHOLD: f64 = 1e-4;

/// Diagnostic formatter shared across the outer optimizer and the custom-family
/// fitter: shows the `max_items` entries of `values` with largest absolute
/// value, formatted as `label=[i:value, ...]`.
pub fn format_top_abs(values: &Array1<f64>, label: &str, max_items: usize) -> String {
    if values.is_empty() {
        return format!("{label}=<empty>");
    }
    let mut ranked: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    ranked.sort_by(|(_, left), (_, right)| {
        right
            .abs()
            .partial_cmp(&left.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let parts: Vec<String> = ranked
        .into_iter()
        .take(max_items)
        .map(|(idx, value)| format!("{idx}:{value:.3e}"))
        .collect();
    format!("{label}=[{}]", parts.join(", "))
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
    /// Relative error threshold for flagging issues
    pub rel_error_threshold: f64,
    /// Whether to emit warnings to stderr
    pub emitwarnings: bool,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            kkt_tolerance: 1e-4,
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

/// Residual diagnostics for observed values and predicted means.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictionDiagnostics {
    pub n_obs: usize,
    pub mae: f64,
    pub rmse: f64,
    pub bias: f64,
    pub r_squared: Option<f64>,
    pub residuals: Vec<f64>,
}

/// Compute prediction residual diagnostics from observed values and predicted means.
pub fn diagnostics_from_predictions(
    observed: &[f64],
    predicted_mean: &[f64],
) -> Result<PredictionDiagnostics, String> {
    if observed.is_empty() {
        return Err("diagnostics_from_predictions requires at least one observation".to_string());
    }
    if observed.len() != predicted_mean.len() {
        return Err(format!(
            "diagnostics_from_predictions length mismatch: observed has {} values but predicted mean has {}",
            observed.len(),
            predicted_mean.len()
        ));
    }
    if observed.iter().any(|value| !value.is_finite()) {
        return Err("observed values must contain only finite numbers".to_string());
    }
    if predicted_mean.iter().any(|value| !value.is_finite()) {
        return Err("predicted mean values must contain only finite numbers".to_string());
    }

    let n_obs = observed.len();
    let n_obs_f = n_obs as f64;
    let mut residuals = Vec::with_capacity(n_obs);
    let mut abs_sum = 0.0_f64;
    let mut residual_sum = 0.0_f64;
    let mut residual_sum_squares = 0.0_f64;
    let mut observed_sum = 0.0_f64;
    for (obs, pred) in observed.iter().zip(predicted_mean.iter()) {
        let residual = obs - pred;
        residuals.push(residual);
        abs_sum += residual.abs();
        residual_sum += residual;
        residual_sum_squares += residual * residual;
        observed_sum += obs;
    }

    let observed_mean = observed_sum / n_obs_f;
    let total_sum_squares = observed
        .iter()
        .map(|value| {
            let centered = value - observed_mean;
            centered * centered
        })
        .sum::<f64>();
    let r_squared = if total_sum_squares > 0.0 {
        Some(1.0 - residual_sum_squares / total_sum_squares)
    } else {
        None
    };

    Ok(PredictionDiagnostics {
        n_obs,
        mae: abs_sum / n_obs_f,
        rmse: (residual_sum_squares / n_obs_f).sqrt(),
        bias: residual_sum / n_obs_f,
        r_squared,
        residuals,
    })
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
}

impl GradientDiagnosticReport {
    /// Create an empty report
    pub fn new() -> Self {
        Self::default()
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

/// Three-way classification of why the cert refused, computed from the
/// H_pen spectrum and the projected residual at the refusing iterate.
/// `RankDeficientHPen` is the regression canary the nullspace lead's
/// smooth-construction rework is intended to eliminate; keep this variant
/// intact when extending — it doubles as the user-facing signal for
/// "an unconstrained polynomial null space slipped past absorption."
///
/// Relocated from `gam-solve`'s `custom_family/joint_newton.rs` (issue #1521
/// crate carve): this is the neutral diagnostic carrier that `gam-solve`'s
/// REML/PIRLS core consumes when classifying a custom-family cert refusal,
/// so it must live BELOW both the core and the (extracted) custom-family
/// subsystem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KktRefusalDiagnosis {
    RankDeficientHPen,
    PhantomMultiplierWithWellConditionedH,
    ActiveSetIncomplete,
    /// Cross-block identifiability aliasing surfaced mid-inner-solve
    /// (e.g., a binding active set materialised a 2-way alias that
    /// the pre-fit audit could not see at the cold design). The fix
    /// is structural — drop or reparameterise the aliased block;
    /// rho-anneal will not recover.
    AliasingDetectedAtFit,
}

impl KktRefusalDiagnosis {
    pub fn as_str(&self) -> &'static str {
        match self {
            KktRefusalDiagnosis::RankDeficientHPen => "rank_deficient_H_pen",
            KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH => {
                "phantom_multiplier_with_well_conditioned_H"
            }
            KktRefusalDiagnosis::ActiveSetIncomplete => "active_set_incomplete",
            KktRefusalDiagnosis::AliasingDetectedAtFit => "aliasing_detected_at_fit",
        }
    }

    /// Parse the textual `diagnosis:` field embedded in the structured
    /// bubbled error string. Returns `None` when no recognised label is
    /// present (legacy / non-cert-refusal error strings).
    pub fn parse_from_error(message: &str) -> Option<Self> {
        let marker = "diagnosis: ";
        let start = message.rfind(marker)? + marker.len();
        let tail = &message[start..];
        let end = tail
            .find(|c: char| c == ';' || c == '\n' || c == ' ')
            .unwrap_or(tail.len());
        match &tail[..end] {
            "rank_deficient_H_pen" => Some(KktRefusalDiagnosis::RankDeficientHPen),
            "phantom_multiplier_with_well_conditioned_H" => {
                Some(KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH)
            }
            "active_set_incomplete" => Some(KktRefusalDiagnosis::ActiveSetIncomplete),
            "aliasing_detected_at_fit" => Some(KktRefusalDiagnosis::AliasingDetectedAtFit),
            _ => None,
        }
    }

    pub fn guidance(self) -> &'static str {
        match self {
            KktRefusalDiagnosis::RankDeficientHPen => {
                "check whether the named block has a structural or numerical null direction \
                 not identified by the likelihood/penalty combination; for Duchon-style \
                 smooths this may be a polynomial null space, while marginal-slope fits can \
                 also expose callback-owned weak directions"
            }
            KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH => {
                "check whether the named block has a near-separated or weakly identified \
                 direction despite a well-conditioned penalized Hessian; in marginal-slope \
                 fits this often indicates marginal/logslope coupling rather than a \
                 Matérn/Duchon polynomial-nullspace failure"
            }
            KktRefusalDiagnosis::ActiveSetIncomplete => {
                "check whether the named block's linear constraints need an additional \
                 active row or a tighter constrained re-solve; this is an active-set \
                 certification failure, not a polynomial-nullspace diagnosis"
            }
            KktRefusalDiagnosis::AliasingDetectedAtFit => {
                "check whether the named block aliases another block after runtime \
                 constraints or callbacks materialize; drop or reparameterize the aliased \
                 direction before fitting"
            }
        }
    }
}

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
    }

    #[test]
    fn test_envelopeaudit_detects_ridge_mismatch() {
        let reference = arr1(&[1.0, 0.0, 0.0]);
        let beta = arr1(&[0.1, 0.2, 0.3]);
        let result = compute_envelopeaudit(1e-10, &reference, 0.1, 0.0, &beta, 1e-8, 1e-6);

        assert!(result.isviolated);
        assert!(result.message.contains("Ridge Mismatch"));
    }

    #[test]
    fn test_dualridge_check_no_mismatch() {
        let beta = arr1(&[0.1, 0.2, 0.3]);
        let result = compute_dualridge_check(0.0, 0.0, 0.0, &beta);

        assert!(!result.has_mismatch);
    }

    #[test]
    fn test_dualridge_check_detects_mismatch() {
        let beta = arr1(&[0.1, 0.2, 0.3]);
        let result = compute_dualridge_check(1e-4, 0.0, 0.0, &beta);

        assert!(result.has_mismatch);
        assert!(result.message.contains("Ridge Mismatch detected"));
    }

    #[test]
    fn diagnostics_from_predictions_computes_residual_metrics() {
        let observed = [1.0, 2.0, 4.0];
        let predicted = [1.5, 1.5, 3.0];

        let result = diagnostics_from_predictions(&observed, &predicted).unwrap();

        assert_eq!(result.residuals, vec![-0.5, 0.5, 1.0]);
        assert_eq!(result.n_obs, 3);
        assert_eq!(result.mae, 2.0 / 3.0);
        assert_eq!(result.bias, 1.0 / 3.0);
        assert_eq!(result.rmse, (1.5_f64 / 3.0).sqrt());
        assert_eq!(result.r_squared, Some(1.0 - 1.5 / (14.0 / 3.0)));
    }

    #[test]
    fn diagnostics_from_predictions_omits_r_squared_for_constant_observed() {
        let observed = [2.0, 2.0];
        let predicted = [1.0, 3.0];

        let result = diagnostics_from_predictions(&observed, &predicted).unwrap();

        assert_eq!(result.r_squared, None);
    }

    #[test]
    fn diagnostics_from_predictions_rejects_invalid_inputs() {
        assert_eq!(
            diagnostics_from_predictions(&[], &[]),
            Err("diagnostics_from_predictions requires at least one observation".to_string())
        );
        assert_eq!(
            diagnostics_from_predictions(&[1.0], &[1.0, 2.0]),
            Err(
                "diagnostics_from_predictions length mismatch: observed has 1 values but predicted mean has 2"
                    .to_string()
            )
        );
        assert_eq!(
            diagnostics_from_predictions(&[f64::NAN], &[1.0]),
            Err("observed values must contain only finite numbers".to_string())
        );
        assert_eq!(
            diagnostics_from_predictions(&[1.0], &[f64::INFINITY]),
            Err("predicted mean values must contain only finite numbers".to_string())
        );
    }
}
