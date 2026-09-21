//! #4291 — the partition function of the smoothed-L¹ (smoothed-Laplace) prior.
//!
//! A smoothed-L¹ sparsifier scores one target coordinate with the energy
//!
//! ```text
//!   E(x; W, ε) = W·sqrt(x² + ε²),
//! ```
//!
//! which is a negative log DENSITY only up to its own mass
//!
//! ```text
//!   Z(W, ε) = ∫_ℝ exp(−W·sqrt(x² + ε²)) dx = 2·ε·K₁(W·ε).
//! ```
//!
//! (Gradshteyn–Ryzhik 3.365.2, `∫₀^∞ e^{−p·sqrt(x²+a²)} dx = a·K₁(a·p)`, doubled
//! by the integrand's symmetry. The `a → 0⁺` limit `a·K₁(ap) → 1/p` recovers the
//! exact Laplace normalizer `2/W`, which is the identity this file's crossover is
//! checked against.)
//!
//! **Why this module exists.** Minimizing an outer criterion over `ln W` while
//! the criterion carries `E` but not `ln Z` is not REML: `∂E/∂ln W = E ≥ 0` for
//! every target, so the strength has no interior optimum and the outer search can
//! only walk it to a box face, silently switching the penalty off. `ln Z` is the
//! term that makes the optimum interior — as `W → 0` its `−ln W` divergence
//! dominates, and as `W → ∞` its `−W·ε` slope cancels the energy's own floor
//! `E ≥ n·W·ε`. A penalty that offers `ln W` to the outer search without adding
//! this term is refused at construction rather than fitted to a face.
//!
//! Every quantity here is exact in closed form; nothing is fitted or tabulated.

use crate::basis::duchon_kernel_math::log_bessel_k1_and_k0_over_k1;

/// `ln Z(W, ε)` and its two log-coordinate derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothedLaplaceLogPartition {
    /// `ln Z(W, ε) = ln 2 + ln ε + ln K₁(W·ε)`.
    pub value: f64,
    /// `∂ ln Z / ∂ ln W = −(1 + z·K₀(z)/K₁(z))`, `z = W·ε`.
    pub log_strength_derivative: f64,
    /// `∂ ln Z / ∂ ln ε = −z·K₀(z)/K₁(z)`, `z = W·ε`.
    pub log_smoothing_derivative: f64,
}

/// `ln Z(W, ε)` for the smoothed-L¹ prior on ONE coordinate, with its `ln W` and
/// `ln ε` derivatives.
///
/// Both derivatives are the same Bessel ratio `r(z) = K₀(z)/K₁(z)` read once:
///
/// ```text
///   ∂/∂ln W [ln K₁(z)] = z·K₁′(z)/K₁(z) = −1 − z·r(z)      (K₁′ = −K₀ − K₁/z)
///   ∂/∂ln ε [ln ε + ln K₁(z)] = 1 + (−1 − z·r(z)) = −z·r(z)
/// ```
///
/// so the two channels cannot disagree about the ratio. Limits, which are the
/// controls a reader should check: as `z → 0⁺`, `r(z) → 0` and the pair tends to
/// `(−1, 0)` — the exact Laplace `Z = 2/W`, independent of `ε`; as `z → ∞`,
/// `r(z) → 1 − 1/(2z)` and the pair tends to `(−z − ½, −z + ½)`, the
/// large-argument form of `2ε·K₁`.
///
/// Refuses rather than saturating when `z = W·ε` leaves the representable
/// log-strength band: there `K₁(z)` is `0` or `∞` in `f64` and its logarithm
/// carries no digits, so a returned value would be a number that answers no
/// question the caller asked.
pub fn smoothed_laplace_log_partition(
    weight: f64,
    eps: f64,
) -> Result<SmoothedLaplaceLogPartition, String> {
    if !(weight.is_finite() && weight > 0.0) {
        return Err(format!(
            "smoothed-L1 log partition requires a finite strength W > 0; got {weight}"
        ));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(format!(
            "smoothed-L1 log partition requires a finite smoothing eps > 0; got {eps}"
        ));
    }
    let log_z = weight.ln() + eps.ln();
    if !(gam_problem::LOG_STRENGTH_MIN..=gam_problem::LOG_STRENGTH_MAX).contains(&log_z) {
        return Err(format!(
            "smoothed-L1 log partition requires ln(W·eps) in \
             [{}, {}] so K_1(W·eps) has digits; got ln(W·eps) = {log_z} \
             (W = {weight}, eps = {eps})",
            gam_problem::LOG_STRENGTH_MIN,
            gam_problem::LOG_STRENGTH_MAX
        ));
    }
    let z = weight * eps;
    let (log_k1, k0_over_k1) = log_bessel_k1_and_k0_over_k1(z);
    let z_ratio = z * k0_over_k1;
    Ok(SmoothedLaplaceLogPartition {
        value: std::f64::consts::LN_2 + eps.ln() + log_k1,
        log_strength_derivative: -(1.0 + z_ratio),
        log_smoothing_derivative: -z_ratio,
    })
}

/// The legal `ln W` band of a smoothed-L¹ strength whose partition must be
/// computable at smoothing `eps`: the ordinary log-strength band intersected with
/// the band on which `W·eps` is itself a legal log-strength.
///
/// This is the domain [`smoothed_laplace_log_partition`] refuses outside, stated
/// once so the outer search never proposes a coordinate the evaluator will
/// refuse. It introduces no constant of its own — both endpoints are
/// `LOG_STRENGTH_MIN` / `LOG_STRENGTH_MAX` shifted by a logarithm the caller
/// already owns.
pub fn smoothed_laplace_strength_log_band(eps: f64) -> Result<(f64, f64), String> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(format!(
            "smoothed-L1 strength band requires a finite smoothing eps > 0; got {eps}"
        ));
    }
    let log_eps = eps.ln();
    let lower = gam_problem::LOG_STRENGTH_MIN.max(gam_problem::LOG_STRENGTH_MIN - log_eps);
    let upper = gam_problem::LOG_STRENGTH_MAX.min(gam_problem::LOG_STRENGTH_MAX - log_eps);
    if lower >= upper {
        return Err(format!(
            "smoothed-L1 smoothing eps = {eps} leaves no strength band on which both W and \
             W·eps are legal log-strengths"
        ));
    }
    Ok((lower, upper))
}
