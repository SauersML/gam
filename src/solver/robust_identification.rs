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
    /// solver).
    ///
    /// `Auto` ⇒ "apply robustness where the conditioning needs it." Both
    /// mechanisms are armed, but each is *intrinsically conditioning-gated* at
    /// its application site, so arming them is a no-op on a well-conditioned
    /// fit:
    ///   * `firth_general` is the full-identifiable-span Jeffreys penalty
    ///     `Φ = ½ log|I(β)|`, which is SELF-LIMITING: its score is `O(1)`
    ///     against the data's `O(n)` Fisher information. On a data-identified
    ///     direction its only effect is the `O(1/n)` bias correction; it bites
    ///     (supplies the missing `O(1)`-bounding curvature) ONLY where the
    ///     information is near-singular, i.e. a near-separating direction. The
    ///     conditioning gate is therefore built into the math — no detector
    ///     threshold to tune.
    ///   * `orthogonalize_confounds` runs the exact W-metric overlap removal in
    ///     [`crate::solver::identifiability_canonical::canonicalize_for_identifiability_with_robust`],
    ///     which falls through byte-identically when the structural-overlap
    ///     detector finds nothing to drop (`ortho.dropped.is_empty()`). The
    ///     conditioning gate is the overlap detector itself.
    ///
    /// `Force` ⇒ request the machinery unconditionally on supported families.
    /// Identical arming to `Auto` today because both mechanisms self-gate; the
    /// variant is retained as the explicit "do not rely on detection" escape
    /// hatch for diagnostics and as a distinct knob if a future mechanism is
    /// added whose application is NOT intrinsically conditioning-gated (such a
    /// mechanism would branch on `policy == Force` here).
    ///
    /// `FirthOnly` ⇒ the principled, zero-downside cure: full identifiable-span
    /// Jeffreys ONLY, no orthogonal-reparameterization design surgery. Jeffreys
    /// is self-limiting, so it cures near-separation (in `ker(S)` OR `range(S)`)
    /// by making the inner objective coercive without dropping any identifiable
    /// design direction.
    ///
    /// Flipping the crate default to always-on robustness is a ONE-PLACE change:
    /// set the `#[default]` on [`crate::solver::workflow::RobustIdentification`]
    /// to `Auto`. Because `Auto`'s mechanisms self-gate, that flip cannot change
    /// a well-conditioned fit; it only arms the cures where the conditioning
    /// needs them. No call site reads the policy directly — every consumer goes
    /// through this resolver — so the flip needs no other edit.
    #[inline]
    pub fn from_policy(policy: crate::solver::workflow::RobustIdentification) -> Self {
        use crate::solver::workflow::RobustIdentification as P;
        match policy {
            P::Off => Self {
                firth_general: false,
                orthogonalize_confounds: false,
            },
            // Auto and Force arm the same mechanisms; the "only where a
            // pathology is detected" (Auto) vs "unconditionally" (Force)
            // distinction is realised at each mechanism's conditioning-gated
            // application site (self-limiting Jeffreys score; overlap detector),
            // not by dropping a mechanism here.
            P::Auto | P::Force => Self {
                firth_general: true,
                orthogonalize_confounds: true,
            },
            P::FirthOnly => Self {
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
