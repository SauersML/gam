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

    /// Resolve the user-facing tri-state policy into the per-mechanism gate
    /// struct the solver consumes.
    ///
    /// `Off` ⇒ every mechanism disabled (byte-identical to the released
    /// solver). `Auto` and `Force` both request the full machinery; the
    /// distinction between "enable only where a pathology is detected" (`Auto`)
    /// and "enable unconditionally" (`Force`) is applied at the detection sites,
    /// not here, so both map to all mechanisms armed.
    #[inline]
    pub fn from_policy(policy: crate::solver::workflow::RobustIdentification) -> Self {
        match policy {
            crate::solver::workflow::RobustIdentification::Off => Self {
                firth_general: false,
                orthogonalize_confounds: false,
            },
            crate::solver::workflow::RobustIdentification::Auto
            | crate::solver::workflow::RobustIdentification::Force => Self {
                firth_general: true,
                orthogonalize_confounds: true,
            },
            // The principled, zero-downside cure: full identifiable-span Jeffreys
            // ONLY, no orthogonal-reparameterization design surgery. Jeffreys is
            // self-limiting, so it cures near-separation (in `ker(S)` OR
            // `range(S)`) by making the inner objective coercive without dropping
            // any identifiable design direction.
            crate::solver::workflow::RobustIdentification::FirthOnly => Self {
                firth_general: true,
                orthogonalize_confounds: false,
            },
        }
    }
}

impl From<crate::solver::workflow::RobustIdentification> for RobustConfig {
    #[inline]
    fn from(policy: crate::solver::workflow::RobustIdentification) -> Self {
        Self::from_policy(policy)
    }
}
