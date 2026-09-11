//! Curvature resolution: the smallest curvature a gate may honestly call
//! nonzero — and the fact that it depends on **how the curvature was
//! produced** (#2690).
//!
//! Every gate in the criterion-resolution cluster compares a **curvature**
//! against a bound. The bound is not a taste parameter and it is not one
//! formula: which bound applies depends entirely on how the curvature was
//! formed, and the candidates differ by orders of magnitude on the same
//! problem. Picking the wrong one is not a small miscalibration — it is a
//! category error, and the reason this module exists rather than a constant.
//!
//! # Law 1 — a curvature formed from criterion VALUES
//!
//! A central second difference along a unit direction `v` with step `h`,
//!
//! ```text
//!     D²f(h) = [f(x + hv) − 2f(x) + f(x − hv)] / h²
//! ```
//!
//! carries two error sources moving in opposite directions in `h`: evaluation
//! noise `4ε_f/h²` (three evaluations, each accurate only to the criterion's
//! own absolute error `ε_f`) and truncation `(h²/12)·M₄`. Balancing them gives
//! a best achievable resolution of `(2/√3)·√(ε_f·M₄)`, so a value-based
//! curvature resolves only down to `√ε_f`, never to `ε_f`.
//!
//! No production gate forms such a curvature. A second difference of
//! criterion values is a finite difference, and finite differences are
//! confined to tests. Law 1 is recorded here only as the standard a
//! value-based probe could never beat, and as the reason a gate judging an
//! ANALYTIC curvature must never borrow `ε_f`: `ε_f` is a property of a
//! criterion on one dataset at one `ρ`, measured values on this project's own
//! fixtures span three orders (`1.5e-8` and `3.79e-5`), and the error that
//! motivated #2690 was exactly a borrowed `ε_f`.
//!
//! # Law 2 — a curvature formed ANALYTICALLY
//!
//! The `1/h²` amplification of Law 1 does not exist for a Hessian that was
//! coded and evaluated directly, and `ε_f` — a statement about the accuracy of
//! the *value* — bounds the error of a separately-coded second derivative not
//! at all. Assuming otherwise is the same category error as comparing a
//! gradient against an eigenvalue.
//!
//! What bounds an analytically-formed eigenvalue is Weyl's inequality: for
//! symmetric `H` and any symmetric perturbation `δH`,
//!
//! ```text
//!     |σ_i(H + δH) − σ_i(H)|  ≤  ‖δH‖₂
//! ```
//!
//! so the resolution of `σ_i` **is** `‖δH‖₂`, the Hessian's own error. There is
//! **no propagation constant at all** — [`CurvatureResolution::analytic_weyl`]
//! returns its argument unchanged, and that is the whole content of the law.
//! The work moves entirely into obtaining `‖δH‖₂`, which is measured, never
//! assumed: an eigensolver's computed residual `max_i ‖H v_i − σ_i v_i‖₂` is a
//! certified `‖δH‖₂` for the solver's own backward error, and a Hessian's
//! reproducibility across perturbations that must leave it invariant is a
//! measured `‖δH‖₂` for the assembly's error. Those two answer different
//! questions — *"given this matrix, how wrong is σ?"* versus *"how wrong is
//! this matrix?"* — and a site that needs the second must not be handed the
//! first.
//!
//! # Why this is a type and not a function returning `f64`
//!
//! A bare `f64` resolution loses the one fact a caller most needs: which law
//! produced it. [`CurvatureResolution`] carries its [`CurvatureLaw`], so the
//! choice is made at the construction site, is visible at the comparison site,
//! and cannot be inherited by accident from a neighbouring call.
//!
//! # Every input, classified
//!
//! | input | class |
//! |---|---|
//! | `ε` (`f64::EPSILON`) | **machine** |
//! | `‖δH‖₂` | **measured, per site** — the analytic law supplies no value for it |
//!
//! # What this module deliberately does NOT do
//!
//! It does not widen anybody's floor. In particular a gradient term
//! `Σ_k |g_k| v_k²` appearing in a ρ-space curvature gate is **not** a
//! mis-derived resolution in need of this module: for `ρ = log λ` the identity
//! `H_ρ = diag(λ)·H_λ·diag(λ) + diag(g_ρ)` makes it one exact term of a chain
//! rule, and the resolution question belongs to the *other* term. Adding a
//! resolution to it would widen a bar that was never a resolution bar.

/// Which law produced a [`CurvatureResolution`].
///
/// Carried on the value so that a comparison site can see — and a reviewer can
/// grep — which law a floor came from. Production forms curvatures only
/// analytically, so the analytic Weyl law is the only one a resolution can be
/// constructed under; there is no constructor for a value-based floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurvatureLaw {
    /// The curvature was formed **analytically**. There is no step and no
    /// `1/h²` amplification; by Weyl the resolution is `‖δH‖₂`, the Hessian's
    /// own error, and nothing else.
    AnalyticWeyl,
}

impl CurvatureLaw {
    /// Human-readable name, for refusal messages that must say which law they
    /// judged by.
    pub fn label(self) -> &'static str {
        match self {
            Self::AnalyticWeyl => "analytic (Weyl, ||dH||_2)",
        }
    }
}

/// A rejected attempt to state a curvature resolution.
///
/// Every variant is a refusal to manufacture a bound from an input that cannot
/// carry one, rather than a silently substituted default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CurvatureResolutionError {
    /// An analytic resolution was requested with a Hessian error that is
    /// negative or `NaN`. Zero is admissible — an exactly-formed Hessian has
    /// `‖δH‖₂ = 0` — and `+∞` is admissible, being the honest statement that
    /// no curvature on this matrix is resolvable.
    HessianError(f64),
}

impl std::fmt::Display for CurvatureResolutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HessianError(value) => write!(
                formatter,
                "an analytic curvature resolution is Weyl's ||dH||_2 and needs a non-negative \
                 (possibly infinite) Hessian error, got {value:.6e}"
            ),
        }
    }
}

impl std::error::Error for CurvatureResolutionError {}

/// One **measured** component of `‖δH‖₂`, carrying the name of the identity
/// that measured it.
///
/// Law 2 supplies no value for `‖δH‖₂`; a site obtains it by evaluating a
/// quantity that is exactly zero in exact arithmetic and reading what came
/// back. Different identities probe different parts of the error — an
/// eigensolver residual probes the *decomposition*, an invariance residual
/// probes the *assembly* — so a component without its provenance is not
/// interpretable, and this type refuses to carry one.
///
/// See [`CurvatureResolution::analytic_weyl_from_components`] for how several
/// are combined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasuredHessianError {
    /// What was measured, in words a refusal message can print — e.g.
    /// `"eigensolver backward error"`.
    pub source: &'static str,
    /// The measured value, a certified lower bound on `‖δH‖₂`.
    pub value: f64,
}

impl MeasuredHessianError {
    /// A named measured component.
    pub fn new(source: &'static str, value: f64) -> Self {
        Self { source, value }
    }
}

impl std::fmt::Display for MeasuredHessianError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}={:.6e}", self.source, self.value)
    }
}

/// The smallest curvature a comparison may treat as nonzero, together with the
/// law that produced it.
///
/// Construct with [`CurvatureResolution::analytic_weyl`] or
/// [`CurvatureResolution::analytic_weyl_from_components`]; there is
/// deliberately no constructor that takes a bare number without naming a law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvatureResolution {
    law: CurvatureLaw,
    resolution: f64,
    /// Which measured component set the resolution, when it came from a set of
    /// them. `None` for the single-component path.
    dominant_source: Option<&'static str>,
}

impl CurvatureResolution {
    /// Law 2: the resolution of an analytically-formed eigenvalue is Weyl's
    /// `‖δH‖₂` and nothing else.
    ///
    /// The returned resolution is `hessian_error_2norm` unchanged. That is not
    /// a stub: the content of the analytic law is precisely that there is no
    /// propagation constant, so all of the work is in the caller's measurement
    /// of `‖δH‖₂` — and routing it through here is what records, at the
    /// comparison site, that the caller judged by the analytic law.
    pub fn analytic_weyl(hessian_error_2norm: f64) -> Result<Self, CurvatureResolutionError> {
        if hessian_error_2norm.is_nan() || hessian_error_2norm < 0.0 {
            return Err(CurvatureResolutionError::HessianError(hessian_error_2norm));
        }
        Ok(Self {
            law: CurvatureLaw::AnalyticWeyl,
            resolution: hessian_error_2norm,
            dominant_source: None,
        })
    }

    /// Law 2 from SEVERAL measured components of `‖δH‖₂`: the resolution is
    /// their **maximum**, and the component that set it is remembered.
    ///
    /// # Why the maximum, and not the sum
    ///
    /// Each component is obtained by evaluating an identity that is exactly
    /// zero in exact arithmetic, so each is a **certified lower bound** on the
    /// true `‖δH‖₂` — never an estimate of the whole of it, and never an
    /// independent additive contribution that could be summed. The largest
    /// lower bound is the strongest fact available, and using it is the
    /// least-conservative honest choice: it widens nothing beyond what one of
    /// the measurements has already demonstrated the assembly does.
    ///
    /// # Why several are needed
    ///
    /// The components answer different questions and are routinely orders
    /// apart. An eigensolver's residual answers *"given this matrix, how wrong
    /// is `σ`?"*; it says nothing whatsoever about how wrong the matrix is, and
    /// on an assembled criterion Hessian it under-reports the truth by many
    /// orders (measured: `7.4e-16` against an assembly inconsistency of
    /// `9.9e-8` on the same fixture, #2748). A site that has only the first has
    /// not measured `‖δH‖₂`; it has measured the eigensolver.
    ///
    /// Empty input is an error rather than a zero resolution: a caller with no
    /// measurement has no resolution, which is this module's standing rule.
    pub fn analytic_weyl_from_components(
        components: &[MeasuredHessianError],
    ) -> Result<Self, CurvatureResolutionError> {
        let mut dominant: Option<MeasuredHessianError> = None;
        for component in components {
            if component.value.is_nan() || component.value < 0.0 {
                return Err(CurvatureResolutionError::HessianError(component.value));
            }
            if dominant.is_none_or(|current| component.value > current.value) {
                dominant = Some(*component);
            }
        }
        let dominant = dominant.ok_or(CurvatureResolutionError::HessianError(f64::NAN))?;
        Ok(Self {
            law: CurvatureLaw::AnalyticWeyl,
            resolution: dominant.value,
            dominant_source: Some(dominant.source),
        })
    }

    /// The measured component that set this resolution, when it was built from
    /// a named set of them.
    pub fn dominant_source(&self) -> Option<&'static str> {
        self.dominant_source
    }

    /// Which law produced this resolution.
    pub fn law(&self) -> CurvatureLaw {
        self.law
    }

    /// The resolution itself: curvatures of smaller magnitude are not
    /// distinguishable from zero by the route that produced them.
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// Whether a measured curvature is resolved by the route that produced it,
    /// i.e. `|curvature| > resolution`.
    ///
    /// A non-finite curvature is never resolved: it carries no magnitude to
    /// compare.
    pub fn resolves(&self, curvature: f64) -> bool {
        curvature.is_finite() && curvature.abs() > self.resolution
    }
}

impl std::fmt::Display for CurvatureResolution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dominant_source {
            Some(source) => write!(
                formatter,
                "{:.6e} [{}; set by {source}]",
                self.resolution,
                self.law.label()
            ),
            None => write!(formatter, "{:.6e} [{}]", self.resolution, self.law.label()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `resolves` fires in both directions on the numbers that motivated the
    /// issue, so the predicate is a discriminator rather than a constant.
    #[test]
    fn resolves_has_a_witness_on_both_sides() {
        let resolved = CurvatureResolution::analytic_weyl(9.0e-8).expect("resolution");
        assert!(
            resolved.resolves(-3.199e-5),
            "an eigenvalue far above ||dH||_2 must be resolved"
        );
        assert!(
            !resolved.resolves(-8.0e-9),
            "an eigenvalue below ||dH||_2 must NOT be resolved"
        );
        assert!(
            !resolved.resolves(f64::NAN),
            "a non-finite curvature carries no magnitude and cannot be resolved"
        );
    }

    /// No negative or `NaN` `‖δH‖₂`: every refusal is a refusal rather than a
    /// substituted constant.
    #[test]
    fn missing_or_impossible_measurements_are_refused_not_defaulted() {
        assert_eq!(
            CurvatureResolution::analytic_weyl(-1.0),
            Err(CurvatureResolutionError::HessianError(-1.0))
        );
        assert!(matches!(
            CurvatureResolution::analytic_weyl(f64::NAN),
            Err(CurvatureResolutionError::HessianError(_))
        ));
        assert_eq!(
            CurvatureResolution::analytic_weyl(0.0).map(|resolution| resolution.resolution()),
            Ok(0.0),
            "an exactly-formed Hessian has ||dH||_2 = 0, which is admissible"
        );
        assert_eq!(
            CurvatureResolution::analytic_weyl(f64::INFINITY)
                .map(|resolution| resolution.resolves(1.0e300)),
            Ok(false),
            "an infinite ||dH||_2 is the honest statement that nothing is resolvable"
        );
    }

    /// One component is exactly the single-component constructor, so a site
    /// that has only the eigensolver's measurement does not move by an ulp.
    #[test]
    fn one_component_is_bit_identical_to_the_single_measurement_law() {
        let value = 7.414_101e-16_f64;
        let from_one =
            CurvatureResolution::analytic_weyl_from_components(&[MeasuredHessianError::new(
                "eigensolver backward error",
                value,
            )])
            .expect("one non-negative component");
        let direct = CurvatureResolution::analytic_weyl(value).expect("non-negative");
        assert_eq!(
            from_one.resolution().to_bits(),
            direct.resolution().to_bits()
        );
        assert_eq!(from_one.law(), direct.law());
    }

    /// A caller with no measurement has no resolution — the module's standing
    /// rule, enforced rather than defaulted to zero. A zero resolution would
    /// silently assert that the assembly is exact.
    #[test]
    fn no_measured_component_is_an_error_not_a_zero_resolution() {
        assert!(matches!(
            CurvatureResolution::analytic_weyl_from_components(&[]),
            Err(CurvatureResolutionError::HessianError(_))
        ));
    }

    /// A negative or `NaN` component is rejected wherever it sits in the list,
    /// not silently out-maxed by a larger sibling.
    #[test]
    fn a_negative_component_is_rejected_even_beside_a_larger_valid_one() {
        assert_eq!(
            CurvatureResolution::analytic_weyl_from_components(&[
                MeasuredHessianError::new("valid", 1.0e-6),
                MeasuredHessianError::new("invalid", -1.0e-9),
            ]),
            Err(CurvatureResolutionError::HessianError(-1.0e-9))
        );
        assert!(matches!(
            CurvatureResolution::analytic_weyl_from_components(&[
                MeasuredHessianError::new("valid", 1.0e-6),
                MeasuredHessianError::new("invalid", f64::NAN),
            ]),
            Err(CurvatureResolutionError::HessianError(_))
        ));
    }
}
