//! Curvature resolution: the smallest curvature a gate may honestly call
//! nonzero — and the fact that it obeys **two different laws** (#2690).
//!
//! Every gate in the criterion-resolution cluster compares a **curvature**
//! against a bound. The bound is not a taste parameter and it is not one
//! formula: which law applies depends entirely on **how the curvature was
//! produced**, and the two laws differ by orders of magnitude on the same
//! problem. Picking the wrong one is not a small miscalibration — it is a
//! category error, and the reason this module exists rather than a constant.
//!
//! # Law 1 — a curvature formed by a FINITE DIFFERENCE of VALUES
//!
//! Central second difference along a unit direction `v` with step `h`:
//!
//! ```text
//!     D²f(h) = [f(x + hv) − 2f(x) + f(x − hv)] / h²
//! ```
//!
//! Two error sources, moving in opposite directions in `h`:
//!
//! * **evaluation noise** — three evaluations, each accurate only to `ε_f`
//!   (the criterion's own absolute evaluation error), so the numerator carries
//!   up to `4ε_f` and the quotient `4ε_f / h²`;
//! * **truncation** — the Taylor remainder `(h²/12)·M₄`, with `M₄` a bound on
//!   the fourth directional derivative.
//!
//! ```text
//!     |D²f(h) − vᵀHv|  ≤  4·ε_f/h²  +  (h²/12)·M₄        [`finite_difference_error_bound`]
//! ```
//!
//! Minimising over `h` — `−8ε_f·h⁻³ + M₄·h/6 = 0` — gives
//!
//! ```text
//!     h*      = (48·ε_f / M₄)^{1/4}
//!     δσ_min  = (2/√3)·√(ε_f · M₄)
//! ```
//!
//! where both terms come out equal at the optimum, as they always do at the
//! minimum of an `A/h² + B·h²` balance. **The achievable curvature resolution
//! goes as `√ε_f`, not as `ε_f`.** A bound built linearly from `ε_f` is far too
//! tight; and reducing `ε_f` by 100× buys only 10× in resolvable curvature.
//!
//! ## `ε_f` is a property of a fixture, never of the codebase
//!
//! `ε_f` is the evaluation error of *a criterion, on a dataset, at a `ρ`*.
//! Measured values on this project's own fixtures span three orders
//! (`1.5e-8` and `3.79e-5`). **It must be measured per fixture and must never
//! be carried between them** — the error that motivated #2690 was exactly a
//! borrowed `ε_f`. There is therefore deliberately no default here: this
//! module will not manufacture an `ε_f`, and a caller with no measurement has
//! no resolution.
//!
//! `ε_f` and `M₄` both come free from any symmetric probe ladder that has
//! already been run. Differencing the ladder against a large-`α` fit of
//! `½·c·α²`, the residual **scales** with `α` where it is truncation and goes
//! **flat** where it is evaluation noise — a term that does not scale with the
//! step is not a step effect, and that plateau is `ε_f`. The symmetric average
//! `c̄(α) = [f(x+αv) + f(x−αv) − 2f(x)]/α² = vᵀHv + (α²/12)·M₄` then carries
//! `M₄` as 12× the slope against `α²`. Only the **symmetric** average does:
//! a one-sided column carries an odd `2(g·v)/α` term that dominates and has
//! been misread as truncation.
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
//! # Why this is a type and not two functions returning `f64`
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
//! | `2/√3`, `48`, `4`, `1/12` | **derived** — the Taylor remainder and the `A/h² + B·h²` optimum; no freedom, and gated numerically by this module's tests against [`finite_difference_error_bound`] itself |
//! | `ε_f` | **measured, per fixture** — never defaulted, never borrowed |
//! | `M₄` | **measured, per fixture** — from the symmetric average of a probe ladder |
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
/// grep — which of the two laws a floor came from. The two are not
/// interchangeable and are routinely orders apart on the same problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurvatureLaw {
    /// The curvature was formed as a central second difference of criterion
    /// **values**, so its error is `4ε_f/h² + (h²/12)·M₄` and its best
    /// achievable resolution is `(2/√3)·√(ε_f·M₄)` at `h = (48ε_f/M₄)^{1/4}`.
    FiniteDifferenceOfValues,
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
            Self::FiniteDifferenceOfValues => "finite-difference (sqrt(eps_f*M4))",
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
    /// A finite-difference resolution was requested with an evaluation error
    /// that is not a positive finite number. There is no default `ε_f`: it is
    /// fixture-specific and must be measured on the fixture in hand.
    EvaluationError(f64),
    /// A finite-difference resolution was requested with a fourth-derivative
    /// bound that is not a positive finite number.
    FourthDerivativeBound(f64),
    /// An analytic resolution was requested with a Hessian error that is
    /// negative or `NaN`. Zero is admissible — an exactly-formed Hessian has
    /// `‖δH‖₂ = 0` — and `+∞` is admissible, being the honest statement that
    /// no curvature on this matrix is resolvable.
    HessianError(f64),
}

impl std::fmt::Display for CurvatureResolutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EvaluationError(value) => write!(
                formatter,
                "a finite-difference curvature resolution needs a positive finite evaluation \
                 error eps_f measured ON THIS FIXTURE, got {value:.6e}"
            ),
            Self::FourthDerivativeBound(value) => write!(
                formatter,
                "a finite-difference curvature resolution needs a positive finite \
                 fourth-derivative bound M4, got {value:.6e}"
            ),
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
/// Construct with [`CurvatureResolution::finite_difference`],
/// [`CurvatureResolution::analytic_weyl`] or
/// [`CurvatureResolution::analytic_weyl_from_components`]; there is
/// deliberately no constructor that takes a bare number without naming a law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvatureResolution {
    law: CurvatureLaw,
    resolution: f64,
    optimal_step: Option<f64>,
    /// Which measured component set the resolution, when it came from a set of
    /// them. `None` for the single-component and finite-difference paths.
    dominant_source: Option<&'static str>,
}

impl CurvatureResolution {
    /// Law 1: the best resolution achievable by a central second difference of
    /// values, `(2/√3)·√(ε_f·M₄)`, attained at step `(48·ε_f/M₄)^{1/4}`.
    ///
    /// `evaluation_error` is `ε_f`, the criterion's absolute evaluation error
    /// **measured on the fixture in hand**; `fourth_derivative` is `M₄`, a
    /// bound on the fourth directional derivative along the probed direction.
    /// Both are measurements. Neither has a default and neither may be
    /// inherited from another fixture.
    pub fn finite_difference(
        evaluation_error: f64,
        fourth_derivative: f64,
    ) -> Result<Self, CurvatureResolutionError> {
        if !evaluation_error.is_finite() || evaluation_error <= 0.0 {
            return Err(CurvatureResolutionError::EvaluationError(evaluation_error));
        }
        if !fourth_derivative.is_finite() || fourth_derivative <= 0.0 {
            return Err(CurvatureResolutionError::FourthDerivativeBound(
                fourth_derivative,
            ));
        }
        // h* = (48 eps_f / M4)^{1/4}: the stationary point of
        // `4 eps_f/h^2 + h^2 M4/12`, where the two terms are equal.
        let optimal_step = (48.0 * evaluation_error / fourth_derivative).sqrt().sqrt();
        // Substituting h* back: each term becomes sqrt(eps_f*M4)/sqrt(3), so
        // the sum is (2/sqrt(3))*sqrt(eps_f*M4).
        let resolution = (2.0 / 3.0_f64.sqrt()) * (evaluation_error * fourth_derivative).sqrt();
        Ok(Self {
            law: CurvatureLaw::FiniteDifferenceOfValues,
            resolution,
            optimal_step: Some(optimal_step),
            dominant_source: None,
        })
    }

    /// Law 2: the resolution of an analytically-formed eigenvalue is Weyl's
    /// `‖δH‖₂` and nothing else.
    ///
    /// The returned resolution is `hessian_error_2norm` unchanged. That is not
    /// a stub: the content of the analytic law is precisely that there is no
    /// propagation constant, so all of the work is in the caller's measurement
    /// of `‖δH‖₂` — and routing it through here is what records, at the
    /// comparison site, that the caller chose this law rather than Law 1.
    pub fn analytic_weyl(hessian_error_2norm: f64) -> Result<Self, CurvatureResolutionError> {
        if hessian_error_2norm.is_nan() || hessian_error_2norm < 0.0 {
            return Err(CurvatureResolutionError::HessianError(hessian_error_2norm));
        }
        Ok(Self {
            law: CurvatureLaw::AnalyticWeyl,
            resolution: hessian_error_2norm,
            optimal_step: None,
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
            optimal_step: None,
            dominant_source: Some(dominant.source),
        })
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

/// One rung of a symmetric probe ladder: the step, and the criterion values on
/// both sides of it.
///
/// The pair is what makes the rung usable: only the SYMMETRIC average
/// `[f(x+αv) + f(x−αv) − 2f(x)]/α²` cancels the odd terms. A one-sided column
/// carries `2(g·v)/α`, which dominates everything else and has been misread as
/// truncation before (see this module's header).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymmetricProbe {
    /// The step `α`, strictly positive.
    pub step: f64,
    /// `f(x + α v)`.
    pub forward: f64,
    /// `f(x − α v)`.
    pub backward: f64,
}

impl SymmetricProbe {
    /// A rung from its two evaluations.
    pub fn new(step: f64, forward: f64, backward: f64) -> Self {
        Self {
            step,
            forward,
            backward,
        }
    }

    /// `f(x+αv) + f(x−αv) − 2f(x)`, the raw second-difference NUMERATOR.
    ///
    /// This is the quantity whose noise is `α`-independent — three evaluations,
    /// each accurate to `ε_f`, so up to `4ε_f` whatever the step. Every fit
    /// below is posed on it rather than on the quotient for exactly that
    /// reason.
    pub fn numerator(&self, baseline: f64) -> f64 {
        (self.forward - baseline) + (self.backward - baseline)
    }
}

/// What a symmetric probe ladder measured about a criterion along one
/// direction (#2748).
///
/// Every field is a MEASUREMENT taken on the fixture in hand. In particular
/// `evaluation_error` is the `ε_f` this module's header refuses to default:
/// obtained here as the ladder's own residual scatter, which is the plateau the
/// header describes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LadderCurvature {
    /// `vᵀHv` as the CRITERION reports it — the `α → 0` intercept of
    /// `c̄(α) = c + (α²/12)·M₄`.
    pub curvature: f64,
    /// The fitted intercept's own standard error, from the ladder's residuals.
    /// It covers evaluation noise and model misfit together, which is what a
    /// comparison against this curvature needs.
    pub curvature_uncertainty: f64,
    /// Twelve times the fitted slope against `α²`, i.e. the ladder's estimate
    /// of the fourth directional derivative — **signed**, as fitted.
    ///
    /// Law 1 wants a BOUND `M₄`, which is a magnitude, and a criterion whose
    /// quartic term happens to be negative is perfectly ordinary (measured:
    /// `-1.106535e-6` on `geo_disease_matern`'s iso-kappa joint arm). Keeping
    /// the sign here and taking the magnitude at the one place a bound is
    /// needed is what stops a legitimate fit from reading as an absent
    /// measurement.
    pub fourth_derivative: f64,
    /// `ε_f`, the criterion's own evaluation error, read off the residual
    /// scatter of the NUMERATOR — whose noise is step-independent — as
    /// `RMS(residual)/√2`.
    ///
    /// # Why `√2` and not `4`
    ///
    /// The error model `4ε_f/h²` counts THREE evaluations at their worst case,
    /// which is the right bound for a single second difference taken in
    /// isolation. A ladder is not that: every rung shares ONE baseline
    /// `f(x)`, so that evaluation's error is a constant offset across the whole
    /// ladder and the fit absorbs it — it contributes no SCATTER. What scatters
    /// is `f(x+αv)` and `f(x−αv)`, two independent errors per rung, so
    /// `RMS(N) = √2·ε_f` and the estimator inverts exactly that.
    ///
    /// Dividing by `4` instead understates `ε_f` by `2√2`, and it understates
    /// it in the dangerous direction: `ε_f` feeds Law 1, whose floor is what
    /// decides whether a measured curvature is a measurement at all. Measured
    /// on the `unresolvable-well #2612` fixture, planted with an evaluation
    /// error of `5e-8`: `RMS/4` reports `1.25e-8` and Law 1 then RESOLVES a
    /// `1e-4` curvature it should not (floor `8.95e-5`), while `RMS/√2` reports
    /// `3.5e-8` and the floor comes out `1.5e-4`, above the claim, which is the
    /// verdict that fixture exists to produce.
    pub evaluation_error: f64,
    /// Rungs the fit consumed.
    pub rungs: usize,
}

impl LadderCurvature {
    /// Law 1's best achievable resolution for this fixture,
    /// `(2/√3)·√(ε_f·M₄)`, from the two quantities the ladder measured.
    ///
    /// This is the finest curvature ANY central second difference of this
    /// criterion could resolve along this direction, at any step. A curvature
    /// claim below it cannot be confirmed or denied by evaluating the
    /// criterion — which is a statement about value-based measurement and, per
    /// this module's Law 2, is NOT a bound on a separately-coded analytic
    /// Hessian's error.
    pub fn finite_difference_resolution(
        &self,
    ) -> Result<CurvatureResolution, CurvatureResolutionError> {
        // `|M₄|`: the fitted quartic coefficient carries a sign, the error
        // model `(h²/12)·M₄` needs a magnitude, and `4ε_f/h² + h²|M₄|/12` is
        // the bound either way.
        CurvatureResolution::finite_difference(self.evaluation_error, self.fourth_derivative.abs())
    }

    /// The **measured** `‖δH‖₂` implied by an analytic Hessian disagreeing with
    /// this ladder about the curvature along the probed direction.
    ///
    /// # The identity
    ///
    /// For a unit direction `v`, `vᵀHv` and `d²/dα² f(x+αv)|₀` are the same
    /// number: the analytic Hessian's Rayleigh quotient IS the criterion's
    /// second directional derivative. Their difference is exactly zero in exact
    /// arithmetic, which is what makes it admissible as a
    /// [`MeasuredHessianError`] — and by Weyl `|vᵀ(δH)v| ≤ ‖δH‖₂`, so whatever
    /// the disagreement is, it is a certified LOWER BOUND on the Hessian's own
    /// error.
    ///
    /// # Why the uncertainty is subtracted
    ///
    /// The ladder's `curvature` is itself measured, so only the part of the
    /// disagreement that exceeds the ladder's own standard error has been
    /// demonstrated. Subtracting it is what keeps this a lower bound rather
    /// than an estimate; a disagreement inside the error bar returns `0.0`,
    /// i.e. *nothing measured*, never a negative or a fabricated magnitude.
    pub fn hessian_error_against(&self, analytic_curvature: f64) -> f64 {
        if !analytic_curvature.is_finite()
            || !self.curvature.is_finite()
            || !self.curvature_uncertainty.is_finite()
        {
            return 0.0;
        }
        let disagreement = (analytic_curvature - self.curvature).abs();
        (disagreement - self.curvature_uncertainty).max(0.0)
    }
}

/// Fit `c̄(α) = c + (α²/12)·M₄` to a symmetric probe ladder and read off the
/// criterion's own curvature, `M₄` and `ε_f` (#2748).
///
/// # Why this exists
///
/// This module's header states that `ε_f` and `M₄` "come free from any
/// symmetric probe ladder that has already been run". Ladders ARE run — the
/// outer certificate probes `ρ̂ ± α·v` along a disputed eigenvector before it
/// will believe a negative curvature — and their evaluations were being
/// discarded after a boolean descent test. This turns those same evaluations
/// into the two measurements the module otherwise refuses to supply, plus the
/// criterion's own verdict on the curvature in dispute.
///
/// # Why the weights are `α⁴` and not a taste
///
/// The noise on the NUMERATOR is step-independent (three evaluations, `4ε_f`),
/// so the noise on the quotient `c̄(α)` has standard deviation `∝ α⁻²` and
/// variance `∝ α⁻⁴`. Inverse-variance weighting is therefore `w = α⁴`,
/// exactly. Equivalently — and this is how it is implemented — the ordinary
/// unweighted least squares of the NUMERATOR `N(α) = c·α² + (M₄/12)·α⁴`
/// against the basis `(α², α⁴)` already IS that weighted fit, with constant
/// noise, which is the well-posed form.
///
/// # What it refuses
///
/// Fewer than three usable rungs (two parameters plus one residual degree of
/// freedom), a non-positive or non-finite step, a non-finite evaluation, or a
/// design matrix that is numerically singular because the rungs do not span
/// distinct steps. In every case the answer is `None` — an absent measurement,
/// which this module treats as no resolution at all rather than as a zero.
pub fn measure_symmetric_ladder(
    baseline: f64,
    probes: &[SymmetricProbe],
) -> Option<LadderCurvature> {
    if !baseline.is_finite() {
        return None;
    }
    // `N(α) = c·α² + (M₄/12)·α⁴` — linear in `(c, M₄/12)` with the design
    // columns `α²` and `α⁴`, and CONSTANT noise, so plain least squares here is
    // the inverse-variance-weighted fit of the quotient.
    let mut usable: Vec<(f64, f64)> = Vec::with_capacity(probes.len());
    for probe in probes {
        if !probe.step.is_finite() || probe.step <= 0.0 {
            continue;
        }
        if !probe.forward.is_finite() || !probe.backward.is_finite() {
            continue;
        }
        let numerator = probe.numerator(baseline);
        if !numerator.is_finite() {
            continue;
        }
        usable.push((probe.step, numerator));
    }
    usable.sort_by(|left, right| left.0.total_cmp(&right.0));
    usable.dedup_by(|left, right| left.0 == right.0);
    let rungs = usable.len();
    if rungs < 3 {
        return None;
    }
    // Normal equations for the 2-column design. The columns are `α²` and `α⁴`,
    // which are wildly different in scale across a decade-spanning ladder, so
    // both are scaled by the largest step first: that is an exact change of
    // variables (a diagonal rescale of the design), it is undone below, and it
    // keeps the 2x2 solve away from a spurious singularity.
    let scale = usable
        .iter()
        .fold(0.0_f64, |accumulated, (step, _)| accumulated.max(*step));
    if !(scale > 0.0) || !scale.is_finite() {
        return None;
    }
    let mut a11 = 0.0_f64;
    let mut a12 = 0.0_f64;
    let mut a22 = 0.0_f64;
    let mut b1 = 0.0_f64;
    let mut b2 = 0.0_f64;
    for (step, numerator) in &usable {
        let unit = step / scale;
        let square = unit * unit;
        let quartic = square * square;
        a11 += square * square;
        a12 += square * quartic;
        a22 += quartic * quartic;
        b1 += square * numerator;
        b2 += quartic * numerator;
    }
    let determinant = a11 * a22 - a12 * a12;
    // Singular to the normal equations' own conditioning: the rungs do not
    // separate the two basis functions, so the fit is not determined. An
    // undetermined fit is an absent measurement.
    if !(determinant.abs() > f64::EPSILON * a11.abs().max(a22.abs()) * a11.abs().max(a22.abs())) {
        return None;
    }
    let unit_curvature = (b1 * a22 - b2 * a12) / determinant;
    let unit_quartic = (b2 * a11 - b1 * a12) / determinant;
    // Undo the change of variables: `α² = scale²·unit²`, `α⁴ = scale⁴·unit⁴`.
    let curvature = unit_curvature / (scale * scale);
    let quartic_coefficient = unit_quartic / (scale * scale * scale * scale);
    let fourth_derivative = 12.0 * quartic_coefficient;
    if !curvature.is_finite() || !fourth_derivative.is_finite() {
        return None;
    }
    // Residual scatter of the NUMERATOR, which is where the noise is
    // step-independent. `rungs - 2` degrees of freedom: two fitted parameters.
    let mut residual_square_sum = 0.0_f64;
    for (step, numerator) in &usable {
        let square = step * step;
        let predicted = curvature * square + quartic_coefficient * square * square;
        let residual = numerator - predicted;
        residual_square_sum += residual * residual;
    }
    let residual_variance = residual_square_sum / ((rungs - 2) as f64);
    if !residual_variance.is_finite() || residual_variance < 0.0 {
        return None;
    }
    // Every rung shares one baseline `f(x)`, so only `f(x±αv)` scatter: two
    // independent evaluation errors per numerator, hence `RMS = √2·ε_f`.
    let evaluation_error = residual_variance.sqrt() / 2.0_f64.sqrt();
    // Standard error of the fitted intercept: `sigma^2 * (A^{-1})_11`, back in
    // the original variable.
    let inverse_11 = a22 / determinant;
    let curvature_uncertainty = (residual_variance * inverse_11).sqrt() / (scale * scale);
    if !curvature_uncertainty.is_finite() {
        return None;
    }
    Some(LadderCurvature {
        curvature,
        curvature_uncertainty,
        fourth_derivative,
        evaluation_error,
        rungs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ladder of steps spanning the interesting decades, used to check the
    /// closed forms against the error model they claim to optimise.
    fn step_ladder() -> Vec<f64> {
        let mut steps = Vec::new();
        let mut step = 1.0e-8_f64;
        while step <= 1.0e1 {
            steps.push(step);
            step *= 10.0_f64.powf(0.05);
        }
        steps
    }

    /// The closed form `h* = (48 eps_f/M4)^{1/4}` is the MINIMISER of the error
    /// model, and `(2/sqrt(3))*sqrt(eps_f*M4)` is the value there.
    ///
    /// Gated against [`finite_difference_error_bound`] by scan rather than
    /// asserted, so the `48`, the `2/sqrt(3)`, the `4` and the `1/12` are
    /// checked rather than trusted. The scan is non-vacuous by construction:
    /// the ends of the ladder are asserted to be strictly worse.
    #[test]
    fn the_closed_forms_minimise_the_error_model_they_claim_to_optimise() {
        for &(eps_f, m4) in &[
            (1.0e-16, 1.0),
            (1.5e-8, 1.47e5),
            (3.79e-5, 1.0),
            (3.79e-5, 100.0),
            (1.0e-3, 1.0e8),
        ] {
            let resolved = CurvatureResolution::finite_difference(eps_f, m4)
                .expect("positive measured inputs must yield a resolution");
            let star = resolved.optimal_step().expect("law 1 carries a step");
            let at_star = finite_difference_error_bound(eps_f, m4, star);

            assert!(
                (at_star - resolved.resolution()).abs() <= 1.0e-12 * resolved.resolution(),
                "the closed-form resolution must equal the error model at h*: \
                 model={at_star:.12e} closed={:.12e} (eps_f={eps_f:.3e}, M4={m4:.3e})",
                resolved.resolution()
            );

            let mut strictly_worse = 0usize;
            for step in step_ladder() {
                let bound = finite_difference_error_bound(eps_f, m4, step);
                assert!(
                    bound >= at_star * (1.0 - 1.0e-12),
                    "h* must minimise the error model, but h={step:.6e} gives \
                     {bound:.6e} < {at_star:.6e} (eps_f={eps_f:.3e}, M4={m4:.3e})"
                );
                if bound > at_star * 1.5 {
                    strictly_worse += 1;
                }
            }
            assert!(
                strictly_worse > 0,
                "the scan must contain steps that are materially worse than h*, \
                 or it cannot have tested minimality (eps_f={eps_f:.3e}, M4={m4:.3e})"
            );
        }
    }

    /// At `h*` the evaluation-noise term and the truncation term are equal —
    /// the property that makes the `A/h² + B·h²` optimum what it is, and the
    /// reason the resolution is `2×` one term rather than a fitted constant.
    #[test]
    fn the_two_error_terms_balance_exactly_at_the_optimal_step() {
        let (eps_f, m4) = (1.5e-8, 1.47e5);
        let resolved = CurvatureResolution::finite_difference(eps_f, m4).expect("resolution");
        let star = resolved.optimal_step().expect("law 1 carries a step");
        let noise = 4.0 * eps_f / (star * star);
        let truncation = star * star * m4 / 12.0;
        assert!(
            (noise - truncation).abs() <= 1.0e-12 * noise,
            "the two terms must balance at h*: noise={noise:.12e} truncation={truncation:.12e}"
        );
        assert!(
            (noise + truncation - resolved.resolution()).abs() <= 1.0e-12 * resolved.resolution(),
            "the resolution is the sum of the two balanced terms"
        );
    }

    /// The measured witness this law was derived on (#2690/#2665): a fixture
    /// whose evaluation floor read `eps_f = 1.5e-8` from the flat residual
    /// plateau of its alpha-ladder, and whose symmetric average fitted
    /// `M4 = 1.47e5`.
    ///
    /// Both numbers are MEASUREMENTS on one fixture and are recorded here as
    /// the law's provenance, not as defaults — nothing in this module supplies
    /// them, and transferring them to another fixture is the error #2690 was
    /// opened to stop.
    #[test]
    fn the_2665_fixture_measurements_reproduce_the_recorded_step_and_resolution() {
        let resolved = CurvatureResolution::finite_difference(1.5e-8, 1.47e5).expect("resolution");
        let star = resolved.optimal_step().expect("law 1 carries a step");
        assert!(
            (star - 1.49e-3).abs() <= 0.01 * 1.49e-3,
            "recorded h* = 1.49e-3 on the #2665 fixture, got {star:.6e}"
        );
        assert!(
            (resolved.resolution() - 5.4e-2).abs() <= 0.01 * 5.4e-2,
            "recorded delta-sigma_min = 5.4e-2 on the #2665 fixture, got {:.6e}",
            resolved.resolution()
        );
    }

    /// The analytic law has NO propagation constant: the resolution is the
    /// supplied `‖δH‖₂`, bit-identically. If this ever stops holding, the law
    /// has grown a fudge factor.
    #[test]
    fn the_analytic_law_returns_its_input_unchanged() {
        for &error in &[0.0, 7.414e-16, 1.362e-18, 9.0e-8, 1.0, f64::INFINITY] {
            let resolved = CurvatureResolution::analytic_weyl(error).expect("resolution");
            assert_eq!(
                resolved.resolution().to_bits(),
                error.to_bits(),
                "Weyl supplies no constant; the resolution IS ||dH||_2"
            );
            assert_eq!(resolved.law(), CurvatureLaw::AnalyticWeyl);
            assert!(
                resolved.optimal_step().is_none(),
                "an analytic curvature has no step"
            );
        }
    }

    /// The two laws applied to the same numeric input disagree, and the type
    /// records which was chosen. This is the confusion the module exists to
    /// prevent: `eps_f = 1e-8` read as a Weyl bound gives `1e-8`, whereas the
    /// finite-difference law with any plausible `M4` gives orders more.
    #[test]
    fn the_two_laws_are_not_interchangeable_on_the_same_number() {
        let eps_f = 1.0e-8;
        let as_weyl = CurvatureResolution::analytic_weyl(eps_f).expect("resolution");
        let as_difference = CurvatureResolution::finite_difference(eps_f, 1.0).expect("resolution");
        assert_ne!(as_weyl.law(), as_difference.law());
        assert!(
            as_difference.resolution() > 1.0e4 * as_weyl.resolution(),
            "sqrt(eps_f) is not eps_f: fd={:.6e} weyl={:.6e}",
            as_difference.resolution(),
            as_weyl.resolution()
        );
    }

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

    /// No default `eps_f`, no default `M4`, no negative `‖δH‖₂`: every refusal
    /// is a refusal rather than a substituted constant.
    #[test]
    fn missing_or_impossible_measurements_are_refused_not_defaulted() {
        assert_eq!(
            CurvatureResolution::finite_difference(0.0, 1.0),
            Err(CurvatureResolutionError::EvaluationError(0.0))
        );
        assert_eq!(
            CurvatureResolution::finite_difference(-1.0e-8, 1.0),
            Err(CurvatureResolutionError::EvaluationError(-1.0e-8))
        );
        assert!(matches!(
            CurvatureResolution::finite_difference(f64::INFINITY, 1.0),
            Err(CurvatureResolutionError::EvaluationError(_))
        ));
        assert_eq!(
            CurvatureResolution::finite_difference(1.0e-8, 0.0),
            Err(CurvatureResolutionError::FourthDerivativeBound(0.0))
        );
        assert_eq!(
            CurvatureResolution::analytic_weyl(-1.0),
            Err(CurvatureResolutionError::HessianError(-1.0))
        );
        assert!(matches!(
            CurvatureResolution::analytic_weyl(f64::NAN),
            Err(CurvatureResolutionError::HessianError(_))
        ));
    }

    /// A step that cannot be taken has an infinite error bound rather than a
    /// negative or `NaN` one.
    #[test]
    fn a_non_positive_step_has_an_infinite_error_bound() {
        assert_eq!(
            finite_difference_error_bound(1.0e-8, 1.0, 0.0),
            f64::INFINITY
        );
        assert_eq!(
            finite_difference_error_bound(1.0e-8, 1.0, -1.0e-3),
            f64::INFINITY
        );
    }

    /// Several measured components of `‖δH‖₂` combine by MAXIMUM, and the
    /// resolution remembers which one set it. This is the #2748 repair's whole
    /// arithmetic: an eigensolver residual and an assembly-inconsistency
    /// residual are both certified lower bounds on the same quantity, and the
    /// larger is the stronger fact.
    #[test]
    fn several_measured_components_resolve_to_the_largest() {
        let resolution = CurvatureResolution::analytic_weyl_from_components(&[
            MeasuredHessianError::new("eigensolver backward error", 8.342_439e-19),
            MeasuredHessianError::new("penalty-map invariance residual", 9.872_016e-8),
        ])
        .expect("two non-negative components");
        assert_eq!(resolution.resolution(), 9.872_016e-8);
        assert_eq!(resolution.law(), CurvatureLaw::AnalyticWeyl);
        assert_eq!(
            resolution.dominant_source(),
            Some("penalty-map invariance residual")
        );
        assert!(
            resolution
                .to_string()
                .contains("penalty-map invariance residual"),
            "the display must name what decided: {resolution}"
        );
        // The curvature this refused at main is inside it; a decade more is not.
        assert!(!resolution.resolves(-2.010_041e-8));
        assert!(resolution.resolves(-2.010_041e-7));
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

    /// A deterministic, reproducible "evaluation noise": the criterion is a
    /// pure function, so its error is not random — it is whatever the
    /// arithmetic did. This mimics that with a fixed hash of the step so the
    /// test is bit-reproducible and the noise does not average away across
    /// rungs the way an RNG's would.
    fn deterministic_wobble(index: usize, magnitude: f64) -> f64 {
        // Low-discrepancy sign/size pattern in [-1, 1]; no RNG, no seed.
        let phase = (index as f64) * std::f64::consts::PI * 0.6180339887498949;
        magnitude * phase.sin()
    }

    /// The ladder recovers the curvature, `M₄` and `ε_f` of a criterion whose
    /// values are constructed FROM those three numbers.
    ///
    /// The fixture is `f(α) = ½cα² + (M₄/24)α⁴ + wobble`, whose symmetric
    /// numerator is exactly `cα² + (M₄/12)α⁴ + 2·wobble` — so every quantity
    /// the fit claims has a known planted value here, including a linear term
    /// `gα` that the symmetric average must annihilate and that is deliberately
    /// made large enough to swamp the curvature if it did not.
    #[test]
    fn the_ladder_recovers_the_curvature_m4_and_eps_f_it_was_built_from() {
        let curvature = -6.4e-6_f64;
        let m4 = 3.5e-2_f64;
        let eps_f = 2.0e-12_f64;
        // A first-order term two orders above the curvature's own α=1 effect:
        // present in both evaluations, cancelled only by the symmetric average.
        let gradient = 1.0e-3_f64;
        let baseline = 2.2244462e3_f64;
        let mut probes = Vec::new();
        let mut step = 1.0_f64;
        for index in 0..12 {
            let quadratic = 0.5 * curvature * step * step;
            let quartic = m4 * step.powi(4) / 24.0;
            let forward = baseline
                + gradient * step
                + quadratic
                + quartic
                + deterministic_wobble(2 * index, eps_f);
            let backward = baseline - gradient * step
                + quadratic
                + quartic
                + deterministic_wobble(2 * index + 1, eps_f);
            probes.push(SymmetricProbe::new(step, forward, backward));
            step *= 0.5;
        }
        let measured = measure_symmetric_ladder(baseline, &probes)
            .expect("twelve well-formed rungs must yield a measurement");

        assert_eq!(measured.rungs, 12);
        assert!(
            (measured.curvature - curvature).abs() <= 1.0e-3 * curvature.abs(),
            "the intercept must recover the planted curvature: got {:.6e} want {curvature:.6e}",
            measured.curvature
        );
        assert!(
            (measured.fourth_derivative - m4).abs() <= 1.0e-3 * m4,
            "twelve times the slope must recover the planted M4: got {:.6e} want {m4:.6e}",
            measured.fourth_derivative
        );
        // `ε_f` is read from residual scatter, so it is recovered to an order,
        // not to a digit — which is all a resolution is ever used to.
        assert!(
            measured.evaluation_error > 0.1 * eps_f && measured.evaluation_error < 10.0 * eps_f,
            "the residual scatter must recover eps_f to within an order: got {:.6e} want \
             {eps_f:.6e}",
            measured.evaluation_error
        );
        // The planted first-order term is 156x the curvature's α=1 signature.
        // Recovering the curvature to 0.1% is the assertion that the symmetric
        // average annihilated it.
        assert!(
            gradient > 100.0 * curvature.abs(),
            "the negative control must actually be a hazard: g={gradient:.3e} vs \
             c={curvature:.3e}"
        );
    }

    /// The disagreement between an analytic curvature and the ladder's is a
    /// certified lower bound on `‖δH‖₂`, and an agreement measures NOTHING.
    ///
    /// This is the whole contract of [`LadderCurvature::hessian_error_against`]:
    /// it must be able to say "this matrix is wrong by at least X" and it must
    /// refuse to say anything when the two agree inside the ladder's own error
    /// bar. The second half is the one that keeps it from widening a floor.
    #[test]
    fn a_disagreement_is_measured_and_an_agreement_measures_nothing() {
        let curvature = 1.0e-8_f64;
        let m4 = 1.0e-2_f64;
        let eps_f = 1.0e-13_f64;
        let baseline = 1.0e3_f64;
        let mut probes = Vec::new();
        let mut step = 1.0_f64;
        for index in 0..10 {
            let value = baseline + 0.5 * curvature * step * step + m4 * step.powi(4) / 24.0;
            probes.push(SymmetricProbe::new(
                step,
                value + deterministic_wobble(2 * index, eps_f),
                value + deterministic_wobble(2 * index + 1, eps_f),
            ));
            step *= 0.5;
        }
        let measured = measure_symmetric_ladder(baseline, &probes).expect("measurement");

        // An analytic Hessian claiming a curvature the criterion does not have.
        let claimed = -6.4e-6_f64;
        let error = measured.hessian_error_against(claimed);
        let disagreement = (claimed - measured.curvature).abs();
        assert!(
            error > 0.0,
            "a claim {claimed:.3e} against a measured {:.3e} must be a measured error",
            measured.curvature
        );
        assert!(
            error <= disagreement,
            "the reported error must never exceed the raw disagreement: {error:.6e} > \
             {disagreement:.6e}"
        );
        assert!(
            error >= disagreement - measured.curvature_uncertainty,
            "the reported error must be the disagreement net of the ladder's own uncertainty"
        );

        // NEGATIVE CONTROL: the analytic value the ladder itself measured
        // measures no error at all. Without this the routine could report a
        // magnitude for every fit and silently widen every gate.
        assert_eq!(
            measured.hessian_error_against(measured.curvature),
            0.0,
            "agreement must measure nothing"
        );
        // And so must anything inside the error bar.
        assert_eq!(
            measured
                .hessian_error_against(measured.curvature + 0.5 * measured.curvature_uncertainty),
            0.0,
            "a disagreement inside the ladder's own error bar has demonstrated nothing"
        );
    }

    /// An absent or undetermined measurement is `None`, never a zero.
    ///
    /// A zero resolution asserts that the assembly is exact, which is the
    /// failure mode this whole module exists to prevent, so every degenerate
    /// input has to leave rather than round.
    #[test]
    fn a_ladder_that_cannot_determine_the_fit_returns_no_measurement() {
        let baseline = 1.0_f64;
        // Two rungs: two parameters, no residual degree of freedom.
        assert!(
            measure_symmetric_ladder(
                baseline,
                &[
                    SymmetricProbe::new(1.0, 1.5, 1.5),
                    SymmetricProbe::new(0.5, 1.125, 1.125),
                ]
            )
            .is_none()
        );
        // Three rungs at ONE step: the design cannot separate α² from α⁴.
        assert!(
            measure_symmetric_ladder(
                baseline,
                &[
                    SymmetricProbe::new(1.0, 1.5, 1.5),
                    SymmetricProbe::new(1.0, 1.5, 1.5),
                    SymmetricProbe::new(1.0, 1.5, 1.5),
                ]
            )
            .is_none()
        );
        // Non-finite evaluations are dropped, and dropping enough of them
        // leaves too few rungs.
        assert!(
            measure_symmetric_ladder(
                baseline,
                &[
                    SymmetricProbe::new(1.0, f64::NAN, 1.5),
                    SymmetricProbe::new(0.5, 1.125, f64::INFINITY),
                    SymmetricProbe::new(0.25, 1.03, 1.03),
                    SymmetricProbe::new(0.125, 1.008, 1.008),
                ]
            )
            .is_none()
        );
        // A non-finite baseline has no numerator at all.
        assert!(
            measure_symmetric_ladder(
                f64::NAN,
                &[
                    SymmetricProbe::new(1.0, 1.5, 1.5),
                    SymmetricProbe::new(0.5, 1.125, 1.125),
                    SymmetricProbe::new(0.25, 1.03, 1.03),
                ]
            )
            .is_none()
        );
    }

    /// The ladder's `ε_f` and `M₄` feed Law 1, and the resolution that comes
    /// back is the one the header's `(2/√3)·√(ε_f·M₄)` promises.
    #[test]
    fn the_measured_pair_reconstructs_law_one_for_this_fixture() {
        let curvature = 1.0e-4_f64;
        let m4 = 1.0_f64;
        let eps_f = 1.0e-12_f64;
        let baseline = 10.0_f64;
        let mut probes = Vec::new();
        let mut step = 1.0_f64;
        for index in 0..10 {
            let value = baseline + 0.5 * curvature * step * step + m4 * step.powi(4) / 24.0;
            probes.push(SymmetricProbe::new(
                step,
                value + deterministic_wobble(2 * index, eps_f),
                value + deterministic_wobble(2 * index + 1, eps_f),
            ));
            step *= 0.5;
        }
        let measured = measure_symmetric_ladder(baseline, &probes).expect("measurement");
        let resolution = measured
            .finite_difference_resolution()
            .expect("a positive measured pair yields Law 1");
        assert_eq!(resolution.law(), CurvatureLaw::FiniteDifferenceOfValues);
        let expected = (2.0 / 3.0_f64.sqrt())
            * (measured.evaluation_error * measured.fourth_derivative).sqrt();
        assert!(
            (resolution.resolution() - expected).abs() <= 1.0e-12 * expected,
            "Law 1 must be reconstructed from the ladder's own two measurements"
        );
        // The planted curvature is above that resolution, so this fixture is a
        // POSITIVE control for the ladder being able to see what it measured.
        assert!(
            resolution.resolves(measured.curvature),
            "a curvature {:.3e} must be resolvable at resolution {:.3e}",
            measured.curvature,
            resolution.resolution()
        );
    }
}
