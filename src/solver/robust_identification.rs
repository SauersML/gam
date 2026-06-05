//! Universal under-identification robustness (WIP scaffold — see workflow wf_bcad67c6).
//!
//! Goal: make robustness to non-identification a property of the SOLVER itself,
//! inherited by every family (BMS, survival, gamlss, multinomial, ...), not a
//! per-family patch.
//!
//! A penalized GLM/GAM is ill-posed exactly when a coefficient direction is
//! identified by NEITHER the data NOR a proper prior. Two modes:
//!   1. Near-separation: improper likelihood, MLE at infinity, positive O(n)
//!      Fisher curvature (NOT a singular Hessian, so rank thresholds cannot catch
//!      it). Cure: a proper prior — Firth's Jeffreys penalty `Phi = 1/2 log|I(beta)|`,
//!      parameterization-invariant and tuning-free, with provably finite estimates
//!      (Firth 1993).
//!   2. Structural confound / rank deficiency: overlapping design blocks (e.g. a
//!      score-weighted logslope surface vs the marginal surface when the score
//!      correlates with the smooth covariates). Cure: exact orthogonal
//!      reparameterization (resolve, do not penalize).
//!
//! With both addressed the penalized objective becomes PROPER => unique finite
//! minimizer => Newton converges quadratically => no rank thresholds, no
//! pseudo-determinants, no hand-tuned ridges. Robustness becomes a theorem.
//!
//! All of this is gated behind [`RobustConfig`] (default OFF: behavior is
//! byte-identical to today's release until the gate passes and the default flips).

/// Configuration gate for the universal robustness machinery.
///
/// Default is all-off, so an unconfigured fit is identical to the pre-robustness
/// solver. The Python/FFI layer threads the user flag into these fields.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RobustConfig {
    /// Family-general Firth/Jeffreys penalty on the identifiable subspace
    /// (bounds near-separating coefficients to O(1) with automatic strength).
    pub firth_general: bool,
    /// Exact orthogonal reparameterization of overlapping design blocks
    /// (resolves structural confounds rather than penalizing them).
    pub orthogonalize_confounds: bool,
}

impl RobustConfig {
    /// True if any robustness mechanism is requested.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.firth_general || self.orthogonalize_confounds
    }
}

// TODO(wf_bcad67c6): link-general Jeffreys penalty `Phi = 1/2 log|I(beta)|`, its
//   beta-gradient `tr(I^{-1} dI/dbeta)`, and Hessian contribution `-d^2 Phi/dbeta^2`
//   for a general inverse link (probit explicit; reuse `FirthDenseOperator`; lift
//   the `(Binomial, Logit)` gates in estimate.rs / hmc.rs / pirls/loop_driver.rs).
// TODO(wf_bcad67c6): scope the Jeffreys term to the unpenalized / under-identified
//   span only (parametric + each smooth's polynomial null space); penalized smooth
//   directions keep their wiggliness prior. Reuse the HMC identifiable-subspace term.
// TODO(wf_bcad67c6): general exact orthogonalization pass for overlapping design
//   blocks via rank-revealing QR/SVD, with an exact coefficient round-trip for
//   reporting.
// TODO(wf_bcad67c6): wire into the SHARED custom-family inner Newton + outer REML so
//   every family inherits it; retire the BMS pinned ridges
//   (`marginal_penalties_with_influence_ridge`) behind this flag, kept as a fallback.
