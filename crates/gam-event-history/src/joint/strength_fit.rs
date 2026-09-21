//! Empirical-Bayes function strengths by the joint Laplace evidence.
//!
//! Coefficients θ carry a normalized prior π(θ | ρ) with log strengths ρ, and
//! every subject has a latent path z_i. The evidence
//! `p(y | ρ) = ∫∫ Π_i p(y_i, z_i | θ) π(θ | ρ) dz dθ` is approximated by one
//! Laplace step at the joint mode x̂ = (θ̂, ẑ):
//!
//! ```text
//! LAML(ρ) = F(x̂) + (d/2) log 2π − ½ log det 𝓗,
//! F = Σ_i f_i(z_i, θ) + log π(θ | ρ),   𝓗 = −∇²F,   d = p + Σ_i dim z_i.
//! ```
//!
//! 𝓗 is an arrow matrix: block-tridiagonal latent precisions Q_i, the
//! coefficient block A + H_π and borders B_i = −∂²f_i/∂z∂θ. Hence
//! `log det 𝓗 = Σ_i log det Q_i + log det S`, where
//! `S = A + H_π − Σ_i B_iᵀ Q_i⁻¹ B_i` is the Hessian of the profile objective
//! `max_z F`, and N(θ̂, S⁻¹) is the coefficient posterior. The exact strength
//! score is
//!
//! ```text
//! ∂LAML/∂ρ_j = ∂ρ_j log π − ½ tr(S⁻¹ ∂ρ_j H_π) − ½ D log det 𝓗[δ_j],
//! δθ_j = S⁻¹ ∂θ∂ρ_j log π,   δz_i = −Q_i⁻¹ B_i δθ_j.
//! ```
//!
//! `D log det 𝓗[δ] = tr(𝓗⁻¹ D𝓗[δ])` is linear in δ. Contracting the complete
//! density's third derivatives with the blocks of 𝓗⁻¹ once gives its gradient
//! over (θ, z). Eliminating z along δz_i = −Q_i⁻¹ B_i δθ leaves the reduced
//! coefficient vector `ḡ = g_θ − Σ_i B_iᵀ Q_i⁻¹ g_zi`, so that
//! `D log det 𝓗[δ_j] = ḡᵀ δθ_j + tr(S⁻¹ D H_π[δθ_j])`, and every strength costs
//! O(p²) beyond that one contraction.
//!
//! Only complete-density derivatives appear: a latent Laplace marginal is never
//! differentiated, and no derivative is assembled by replaying coefficient
//! pairs.
//!
//! Every acceptance is decided at the arithmetic's own resolution; no caller
//! supplies a tolerance. Bands are first-order absolute errors under the rules
//! of `law::numerical::Running`. Systems and priors evaluate their own routes at
//! Running only at convergence and classification checks, matrix routes carry
//! inline error matrices, and every solve's error enters through
//! `law::numerical::solve_forward_error`'s certificate.
//! - The coefficient mode is certified by opt's Newton-decrement verdict
//!   `½λ̂² + band_λ² ≤ band_f`, with the errors of F, its gradient and S as bands.
//! - The strength score is accepted when each component is within its own error
//!   plus `|δθ_jᵀ g|`, the first-order change of its envelope term over the
//!   certified coefficient residual.
//! Opt's trust region steers the coefficient search, and no quasi-Newton metric
//! enters the evidence or the coefficient law. The strength search reads only
//! the analytic score. A search that compares evidence values cannot see
//! progress once `½ g²/|H_ρ|` falls within the evidence's rounding floor, i.e.
//! at `|g| ≈ √(2 band_f |H_ρ|)`, far above the score's own floor. So each
//! conjugate direction ends at an exact root of the directional score.
use super::law::numerical::{
    Running, RunningSum, SolveCertificate, contraction_with_error, frobenius, log_determinant_with_error,
    product_error, solve_forward_error,
};
use super::law::{invalid, numerical};
use crate::EventHistoryError;
use crate::scalar::ln;
use faer::Side;
use gam_linalg::faer_ndarray::FaerLlt;
use gam_math::nested_dual::JetField;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

/// Σ_i max_z f_i(z, θ) and its envelope gradient in θ, on the f64 search route.
pub(super) struct ProfileValue {
    pub log_density: f64,
    pub gradient: Vec<f64>,
}

/// First-order absolute errors of the latest [`ProfileValue`]: its route
/// evaluated at `law::numerical::Running` (each entry its `rounding()`), plus the
/// envelope gradient's motion `|B_iᵀ (ẑ_i − z_i)|` over the latent modes'
/// certified residuals. The value's error over those residuals is second order.
pub(super) struct ProfileRounding {
    pub log_density: f64,
    pub gradient: Vec<f64>,
}

/// The arrow Laplace blocks at the latest profile state, without the prior,
/// with inline first-order absolute errors under Running's rules.
pub(super) struct LaplaceCurvature {
    /// S₀ = A − Σ_i B_iᵀ Q_i⁻¹ B_i.
    pub schur: Array2<f64>,
    /// Entrywise errors of `schur`, the latent solves' certificates included.
    pub schur_error: Array2<f64>,
    /// Σ_i log det Q_i.
    pub latent_log_determinant: f64,
    pub latent_log_determinant_error: f64,
    /// Σ_i dim z_i.
    pub latent_dimension: usize,
}

/// The reduced log-determinant gradient ḡ with its first-order absolute errors,
/// including those the covariance it was contracted against carries.
pub(super) struct LogDeterminantGradient {
    pub value: Vec<f64>,
    pub error: Vec<f64>,
}

/// The complete observation density of a cohort in joint Laplace form. Calls
/// follow `profile` → `curvature` → `log_determinant_gradient`: latent modes
/// are solved inside `profile`, and later calls describe that same state.
pub(super) trait LaplaceSystem {
    fn coefficients(&self) -> usize;
    fn profile(&mut self, theta: &[f64]) -> Result<ProfileValue, EventHistoryError>;
    /// The errors of the latest profile, read only at convergence and
    /// classification checks.
    fn profile_rounding(&mut self) -> Result<ProfileRounding, EventHistoryError>;
    fn curvature(&mut self) -> Result<LaplaceCurvature, EventHistoryError>;
    /// ḡ, such that `D log det 𝓗_obs[(δθ, −Q_i⁻¹ B_i δθ)] = ḡᵀ δθ` for the
    /// observation blocks of 𝓗, contracted against 𝓗⁻¹, whose coefficient block
    /// is `covariance = S⁻¹` with the prior curvature included. `covariance_error`
    /// holds the covariance's entrywise errors.
    fn log_determinant_gradient(
        &mut self,
        covariance: &Array2<f64>,
        covariance_error: &Array2<f64>,
    ) -> Result<LogDeterminantGradient, EventHistoryError>;
}

/// A normalized coefficient prior at (θ, ρ), including every normalizer and
/// chart Jacobian, so the evidence names the prior actually used.
pub(super) struct PriorCurvature {
    pub log_density: f64,
    /// ∇θ log π.
    pub gradient: Vec<f64>,
    /// ∂ρ log π.
    pub strength_gradient: Vec<f64>,
    /// ∂²ρ_j log π. It is exactly diagonal, because each strength enters one
    /// separable block.
    pub strength_second: Vec<f64>,
    /// H_π = −∇²θ log π.
    pub negative_hessian: Array2<f64>,
    /// ∂ρ_j H_π, one matrix per strength.
    pub strength_curvature: Vec<Array2<f64>>,
    /// Rows ∂θ∂ρ_j log π, strengths × coefficients.
    pub mixed: Array2<f64>,
}

/// First-order absolute errors of a [`PriorCurvature`]: its route evaluated at
/// `law::numerical::Running`, each entry its `rounding()`.
pub(super) struct PriorRounding {
    pub log_density: f64,
    pub gradient: Vec<f64>,
    pub strength_gradient: Vec<f64>,
    pub negative_hessian: Array2<f64>,
    pub strength_curvature: Vec<Array2<f64>>,
    pub mixed: Array2<f64>,
}

pub(super) trait StrengthPrior {
    fn strengths(&self) -> usize;
    fn evaluate(&self, theta: &[f64], rho: &[f64]) -> Result<PriorCurvature, EventHistoryError>;
    /// The errors of `evaluate` at (θ, ρ), read only at convergence and
    /// classification checks.
    fn rounding(&self, theta: &[f64], rho: &[f64]) -> Result<PriorRounding, EventHistoryError>;
    /// Ḣ_π = D H_π[δθ] at (θ, ρ) with its entrywise errors, including the effect
    /// of `direction_error` through |D H_π|.
    fn curvature_derivative(
        &self,
        theta: &[f64],
        rho: &[f64],
        direction: &[f64],
        direction_error: &[f64],
    ) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError>;
    /// The zero-effect face of strength j, where its function is constant, or
    /// `None` when that limit changes the observation or dynamics law instead of
    /// pinning a coefficient block.
    fn face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError>;
}

/// The certified profile mode at fixed strengths and everything evaluated
/// there.
pub(super) struct CoefficientMode {
    pub coefficients: Vec<f64>,
    pub profile: ProfileValue,
    pub profile_rounding: ProfileRounding,
    pub prior: PriorCurvature,
    pub prior_rounding: PriorRounding,
    pub curvature: LaplaceCurvature,
    /// S = S₀ + H_π and its entrywise errors.
    pub precision: Array2<f64>,
    pub precision_error: Array2<f64>,
    /// Cholesky factor of `precision`.
    factor: FaerLlt<f64>,
    /// The decrement verdict the mode was certified by.
    pub certificate: opt::DecrementEvidence,
}

impl CoefficientMode {
    /// S⁻¹, the coefficient covariance of the joint Gaussian at this mode.
    pub(super) fn covariance(&self) -> Array2<f64> {
        inverse(&self.factor)
    }

    /// S⁻¹ with its a posteriori certificate: `inverse_error` bounds it
    /// entrywise, and `solution_error` bounds every solve against S.
    fn certified_covariance(&self) -> Result<(Array2<f64>, SolveCertificate), EventHistoryError> {
        let covariance = inverse(&self.factor);
        let certificate = solve_forward_error(
            &self.precision,
            &covariance,
            &self.precision_error,
            "joint Laplace coefficient precision",
        )?;
        Ok((covariance, certificate))
    }

    /// `g = ∇θ F` at the mode, the residual the certificate accepted.
    fn gradient(&self) -> Vec<f64> {
        self.profile
            .gradient
            .iter()
            .zip(&self.prior.gradient)
            .map(|(a, b)| a + b)
            .collect()
    }
}

pub(super) struct LamlEvaluation {
    pub log_evidence: f64,
    /// Rounding floor of `log_evidence`.
    pub evidence_band: f64,
    pub gradient: Vec<f64>,
    /// Per strength, the band its score component is accepted within.
    pub gradient_band: Vec<f64>,
}

/// A certified optimum of one model on the zero-effect lattice: the full model
/// when `limits` is empty, else the reduced model with every limit's block pinned
/// together and its other strengths optimized.
pub(super) struct FaceFit {
    /// Strengths at their zero-effect limit, as parent indices, each with the
    /// status its boundary score decided at this fit.
    pub limits: Vec<ZeroEffectLimit>,
    /// Parent coordinates the model keeps, ascending.
    pub kept: Vec<usize>,
    /// Parent indices of the strengths the model optimizes, ascending.
    pub strengths: Vec<usize>,
    pub log_strengths: Vec<f64>,
    pub mode: CoefficientMode,
    pub evidence: LamlEvaluation,
    pub iterations: usize,
}

/// The error of `a + b` from its operands' errors, by Running's rule for a sum.
fn sum_error(a: f64, a_error: f64, b: f64, b_error: f64) -> f64 {
    a_error + b_error + f64::EPSILON * (a + b).abs()
}

/// A column matrix, so vector contractions share `contraction_with_error`.
fn column(vector: &[f64]) -> Array2<f64> {
    Array2::from_shape_fn((vector.len(), 1), |(i, _)| vector[i])
}

/// Model-layout values on the parent coordinates `kept`, zero elsewhere.
fn pad(values: &[f64], kept: &[usize], coefficients: usize) -> Vec<f64> {
    let mut full = vec![0.0; coefficients];
    for (&index, &value) in kept.iter().zip(values) {
        full[index] = value;
    }
    full
}

/// Model-layout log strengths on the parent strengths `free`, over `base`.
fn spread(values: &[f64], free: &[usize], base: &[f64]) -> Vec<f64> {
    let mut full = base.to_vec();
    for (&index, &value) in free.iter().zip(values) {
        full[index] = value;
    }
    full
}

/// `matrix` on the kept coordinates, embedded in the full layout with zeros.
fn embed(matrix: &Array2<f64>, kept: &[usize], coefficients: usize) -> Array2<f64> {
    let mut full = Array2::zeros((coefficients, coefficients));
    for (a, &i) in kept.iter().enumerate() {
        for (b, &k) in kept.iter().enumerate() {
            full[[i, k]] = matrix[[a, b]];
        }
    }
    full
}

/// F = profile + prior at (θ, ρ) with the errors of both, validated: the state a
/// convergence or classification check reads.
struct CheckedState {
    profile: ProfileValue,
    profile_rounding: ProfileRounding,
    prior: PriorCurvature,
    prior_rounding: PriorRounding,
}

impl CheckedState {
    fn read<Sys: LaplaceSystem + ?Sized, P: StrengthPrior + ?Sized>(
        system: &mut Sys,
        prior: &P,
        theta: &[f64],
        rho: &[f64],
    ) -> Result<Self, EventHistoryError> {
        let p = system.coefficients();
        let h = prior.strengths();
        let profile = system.profile(theta)?;
        validate_profile(&profile, p)?;
        let profile_rounding = system.profile_rounding()?;
        validate_profile_rounding(&profile_rounding, p)?;
        let prior_state = prior.evaluate(theta, rho)?;
        validate_prior(&prior_state, p, h)?;
        let prior_rounding = prior.rounding(theta, rho)?;
        validate_prior_rounding(&prior_rounding, p, h)?;
        Ok(Self {
            profile,
            profile_rounding,
            prior: prior_state,
            prior_rounding,
        })
    }

    fn objective_band(&self) -> f64 {
        sum_error(
            self.profile.log_density,
            self.profile_rounding.log_density,
            self.prior.log_density,
            self.prior_rounding.log_density,
        )
    }

    /// `g = ∇θ F`.
    fn gradient(&self) -> Vec<f64> {
        self.profile
            .gradient
            .iter()
            .zip(&self.prior.gradient)
            .map(|(a, b)| a + b)
            .collect()
    }

    fn gradient_band(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.profile.gradient.len(), |k| {
            sum_error(
                self.profile.gradient[k],
                self.profile_rounding.gradient[k],
                self.prior.gradient[k],
                self.prior_rounding.gradient[k],
            )
        })
    }
}

fn factor(matrix: &Array2<f64>) -> Result<FaerLlt<f64>, EventHistoryError> {
    let n = matrix.nrows();
    if matrix.ncols() != n || matrix.iter().any(|v| !v.is_finite()) {
        return Err(numerical(
            "joint Laplace coefficient precision is not a finite square matrix",
        ));
    }
    let view = faer::Mat::from_fn(n, n, |i, j| matrix[[i, j]]);
    FaerLlt::new(view.as_ref(), Side::Lower).map_err(|_| {
        numerical(
            "joint Laplace coefficient precision is not positive definite: the profile objective has no identified interior mode here",
        )
    })
}

fn solve(factor: &FaerLlt<f64>, rhs: &[f64]) -> Vec<f64> {
    let column = faer::Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);
    let out = factor.solve(column.as_ref());
    (0..rhs.len()).map(|i| out[(i, 0)]).collect()
}

fn inverse(factor: &FaerLlt<f64>) -> Array2<f64> {
    let n = factor.nrows();
    let identity = faer::Mat::from_fn(n, n, |i, j| f64::from(i == j));
    let out = factor.solve(identity.as_ref());
    Array2::from_shape_fn((n, n), |(i, j)| out[(i, j)])
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn validate_prior(
    prior: &PriorCurvature,
    coefficients: usize,
    strengths: usize,
) -> Result<(), EventHistoryError> {
    if prior.gradient.len() != coefficients
        || prior.strength_gradient.len() != strengths
        || prior.strength_second.len() != strengths
        || prior.negative_hessian.dim() != (coefficients, coefficients)
        || prior.strength_curvature.len() != strengths
        || prior
            .strength_curvature
            .iter()
            .any(|m| m.dim() != (coefficients, coefficients))
        || prior.mixed.dim() != (strengths, coefficients)
    {
        return Err(invalid(
            "strength prior curvature has dimensions inconsistent with its coefficients and strengths",
        ));
    }
    Ok(())
}

/// An error bound is a finite non-negative number.
fn is_bound(v: &f64) -> bool {
    v.is_finite() && *v >= 0.0
}

fn validate_prior_rounding(
    rounding: &PriorRounding,
    coefficients: usize,
    strengths: usize,
) -> Result<(), EventHistoryError> {
    if rounding.gradient.len() != coefficients
        || rounding.strength_gradient.len() != strengths
        || rounding.negative_hessian.dim() != (coefficients, coefficients)
        || rounding.strength_curvature.len() != strengths
        || rounding
            .strength_curvature
            .iter()
            .any(|m| m.dim() != (coefficients, coefficients))
        || rounding.mixed.dim() != (strengths, coefficients)
        || !std::iter::once(&rounding.log_density)
            .chain(&rounding.gradient)
            .chain(&rounding.strength_gradient)
            .chain(&rounding.negative_hessian)
            .chain(rounding.strength_curvature.iter().flatten())
            .chain(&rounding.mixed)
            .all(is_bound)
    {
        return Err(invalid(
            "strength prior errors have dimensions inconsistent with its coefficients and strengths, or are not finite non-negative bounds",
        ));
    }
    Ok(())
}

fn validate_profile(
    profile: &ProfileValue,
    coefficients: usize,
) -> Result<(), EventHistoryError> {
    if profile.gradient.len() != coefficients
        || !profile.log_density.is_finite()
        || profile.gradient.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "joint Laplace profile has dimensions inconsistent with the coefficients or non-finite values",
        ));
    }
    Ok(())
}

fn validate_profile_rounding(
    rounding: &ProfileRounding,
    coefficients: usize,
) -> Result<(), EventHistoryError> {
    if rounding.gradient.len() != coefficients
        || !std::iter::once(&rounding.log_density)
            .chain(&rounding.gradient)
            .all(is_bound)
    {
        return Err(invalid(
            "joint Laplace profile errors have a width inconsistent with the coefficients, or are not finite non-negative bounds",
        ));
    }
    Ok(())
}

fn validate_curvature(
    curvature: &LaplaceCurvature,
    coefficients: usize,
) -> Result<(), EventHistoryError> {
    if curvature.schur.dim() != (coefficients, coefficients)
        || curvature.schur_error.dim() != (coefficients, coefficients)
        || !curvature.latent_log_determinant.is_finite()
        || curvature.schur.iter().any(|v| !v.is_finite())
        || !curvature
            .schur_error
            .iter()
            .chain(std::iter::once(&curvature.latent_log_determinant_error))
            .all(is_bound)
    {
        return Err(invalid(
            "joint Laplace Schur block has dimensions inconsistent with the coefficients, or non-finite entries or errors",
        ));
    }
    Ok(())
}

/// A trial the search can back away from is recoverable. Anything else,
/// including a reference or integration resolution request, stops the search
/// and is returned with its type intact.
fn objective_error(
    error: EventHistoryError,
    refusal: &RefCell<Option<EventHistoryError>>,
) -> opt::ObjectiveEvalError {
    match error {
        EventHistoryError::NumericalFailure { .. } => opt::ObjectiveEvalError::recoverable_from(error),
        other => {
            *refusal.borrow_mut() = Some(other.clone());
            opt::ObjectiveEvalError::fatal_from(other)
        }
    }
}

/// Termination is a convergence test or a refusal, never an iteration count.
fn unbounded_iterations() -> Result<opt::MaxIterations, EventHistoryError> {
    opt::MaxIterations::new(usize::MAX).map_err(|e| invalid(e.to_string()))
}

/// −F at fixed strengths, with its exact profile Hessian S as the trust-region
/// model.
struct ProfileObjective<'a, Sys: ?Sized, P: ?Sized> {
    system: &'a mut Sys,
    prior: &'a P,
    rho: &'a [f64],
    refusal: &'a RefCell<Option<EventHistoryError>>,
}

impl<Sys: LaplaceSystem + ?Sized, P: StrengthPrior + ?Sized> ProfileObjective<'_, Sys, P> {
    fn sample(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<(opt::FirstOrderSample, PriorCurvature), opt::ObjectiveEvalError> {
        let refusal = self.refusal;
        let theta = x.to_vec();
        let profile = self
            .system
            .profile(&theta)
            .and_then(|profile| validate_profile(&profile, theta.len()).map(|()| profile))
            .map_err(|error| objective_error(error, refusal))?;
        let prior = self
            .prior
            .evaluate(&theta, self.rho)
            .and_then(|prior| validate_prior(&prior, theta.len(), self.prior.strengths()).map(|()| prior))
            .map_err(|error| objective_error(error, refusal))?;
        let sample = opt::FirstOrderSample {
            value: -(profile.log_density + prior.log_density),
            gradient: Array1::from_iter(
                profile.gradient.iter().zip(&prior.gradient).map(|(a, b)| -(a + b)),
            ),
        };
        Ok((sample, prior))
    }
}

impl<Sys: LaplaceSystem + ?Sized, P: StrengthPrior + ?Sized> opt::ZerothOrderObjective
    for ProfileObjective<'_, Sys, P>
{
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, opt::ObjectiveEvalError> {
        Ok(self.sample(x)?.0.value)
    }
}

impl<Sys: LaplaceSystem + ?Sized, P: StrengthPrior + ?Sized> opt::FirstOrderObjective
    for ProfileObjective<'_, Sys, P>
{
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<opt::FirstOrderSample, opt::ObjectiveEvalError> {
        Ok(self.sample(x)?.0)
    }
}

impl<Sys: LaplaceSystem + ?Sized, P: StrengthPrior + ?Sized> opt::OperatorObjective
    for ProfileObjective<'_, Sys, P>
{
    fn eval_value_grad_op(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<opt::OperatorSample, opt::ObjectiveEvalError> {
        let (sample, prior) = self.sample(x)?;
        let refusal = self.refusal;
        let curvature = self
            .system
            .curvature()
            .and_then(|curvature| {
                if curvature.schur.dim() == prior.negative_hessian.dim() {
                    Ok(curvature)
                } else {
                    Err(invalid(
                        "joint Laplace Schur block has dimensions inconsistent with the coefficients",
                    ))
                }
            })
            .map_err(|error| objective_error(error, refusal))?;
        Ok(opt::OperatorSample {
            value: sample.value,
            gradient: sample.gradient,
            hessian: opt::HessianValue::Dense(&curvature.schur + &prior.negative_hessian),
        })
    }
}

/// Maximize the profile objective at fixed strengths and certify the mode.
///
/// Opt's matrix-free trust region steers with the exact S. Its model-decrement
/// exit is set at the starting point's objective floor, and every other exit
/// hands its point over. The certificate then recomputes the bands at its own
/// point, and exact Newton steps on S finish the search until the verdict
/// certifies. Steps then continue while the decrement contracts, so the mode
/// rests at the gradient's own floor: the strength score and its band read the
/// coefficient residual, which value-resolution certification alone leaves at
/// √band_f. A decrement that stops contracting before certification, a saddle,
/// a weakly identified valley or an undecidable verdict is refused, never
/// reported as a mode.
pub(super) fn coefficient_mode<Sys: LaplaceSystem + ?Sized, P: StrengthPrior + ?Sized>(
    system: &mut Sys,
    prior: &P,
    initial: &[f64],
    rho: &[f64],
) -> Result<CoefficientMode, EventHistoryError> {
    let p = system.coefficients();
    if initial.len() != p
        || rho.len() != prior.strengths()
        || initial.iter().chain(rho).any(|v| !v.is_finite())
    {
        return Err(invalid(
            "coefficient mode needs finite coefficients and strengths of the model's widths",
        ));
    }
    let starting_floor = CheckedState::read(system, prior, initial, rho)?.objective_band();
    let refusal = RefCell::new(None::<EventHistoryError>);
    let start = {
        let objective = ProfileObjective {
            system: &mut *system,
            prior,
            rho,
            refusal: &refusal,
        };
        let mut solver = opt::MatrixFreeTrustRegion::new(Array1::from_vec(initial.to_vec()), objective)
            .with_tolerance(opt::Tolerance::new(f64::MIN_POSITIVE).map_err(|e| invalid(e.to_string()))?)
            .with_gradient_tolerance(opt::GradientTolerance::absolute(f64::MIN_POSITIVE))
            .with_model_decrement_tolerance(starting_floor)
            .with_max_iterations(unbounded_iterations()?);
        match solver.run() {
            Ok(solution) => solution.final_point.to_vec(),
            Err(opt::MatrixFreeTrustRegionError::TrustRegionRejectFloor { last_solution }) => {
                last_solution.final_point.to_vec()
            }
            Err(error) => {
                return Err(refusal.borrow_mut().take().unwrap_or_else(|| {
                    numerical(format!("coefficient mode search failed: {error}"))
                }));
            }
        }
    };
    let mut theta = start;
    let mut previous = f64::INFINITY;
    let mut certified: Option<Vec<f64>> = None;
    loop {
        let state = CheckedState::read(system, prior, &theta, rho)?;
        let curvature = system.curvature()?;
        validate_curvature(&curvature, p)?;
        let precision = &curvature.schur + &state.prior.negative_hessian;
        // Each entry of S is one sum of two operands with their errors; the
        // spectral band is bounded by the Frobenius norm of those errors.
        let precision_error = Array2::from_shape_fn((p, p), |index| {
            sum_error(
                curvature.schur[index],
                curvature.schur_error[index],
                state.prior.negative_hessian[index],
                state.prior_rounding.negative_hessian[index],
            )
        });
        let gradient = state.gradient();
        let bands = opt::DecrementBands {
            objective: state.objective_band(),
            gradient: state.gradient_band(),
            hessian: frobenius(precision_error.iter().copied()),
        };
        let verdict =
            opt::newton_decrement_verdict(&precision, &Array1::from_vec(gradient.clone()), None, &bands);
        if let Some(point) = certified.take() {
            if !matches!(verdict, opt::DecrementVerdict::Certified(_)) {
                // Rounding moved a step from a certified point out of
                // certification: return to that point, where the polish ends.
                theta = point;
                previous = 0.0;
                continue;
            }
        }
        match verdict {
            opt::DecrementVerdict::Certified(certificate) => {
                let factor = factor(&precision)?;
                // Certification is decided at the objective's value floor, but the
                // strength score reads the gradient, whose floor is far lower. Exact
                // Newton steps continue while λ̂² contracts, and the mode is returned
                // where rounding stops that contraction.
                if certificate.lambda_sq < previous {
                    previous = certificate.lambda_sq;
                    let step = solve(&factor, &gradient);
                    certified = Some(theta.clone());
                    for (value, delta) in theta.iter_mut().zip(&step) {
                        *value += delta;
                    }
                    continue;
                }
                return Ok(CoefficientMode {
                    coefficients: theta,
                    profile: state.profile,
                    profile_rounding: state.profile_rounding,
                    prior: state.prior,
                    prior_rounding: state.prior_rounding,
                    curvature,
                    precision,
                    precision_error,
                    factor,
                    certificate,
                });
            }
            opt::DecrementVerdict::DecrementAboveTolerance(evidence) => {
                if evidence.lambda_sq >= previous {
                    return Err(numerical(format!(
                        "coefficient mode Newton decrement stopped contracting: λ̂² {:.3e} after {previous:.3e}, objective band {:.3e}",
                        evidence.lambda_sq, evidence.band_f
                    )));
                }
                previous = evidence.lambda_sq;
                let step = solve(&factor(&precision)?, &gradient);
                for (value, delta) in theta.iter_mut().zip(&step) {
                    *value += delta;
                }
            }
            opt::DecrementVerdict::NotPositiveDefinite {
                min_curvature,
                curvature_resolution,
            } => {
                return Err(numerical(format!(
                    "coefficient stationary point is not a maximum: curvature {min_curvature:.3e} below −{curvature_resolution:.3e}"
                )));
            }
            opt::DecrementVerdict::WeaklyIdentifiedValley { directions, .. } => {
                return Err(numerical(format!(
                    "coefficient mode lies in a weakly identified valley of {directions} flat directions"
                )));
            }
            other => {
                return Err(numerical(format!(
                    "coefficient mode certificate cannot be decided: {other:?}"
                )));
            }
        }
    }
}

/// The joint Laplace evidence and its exact strength score at a mode that
/// `coefficient_mode` certified for the same system state and strengths.
pub(super) fn laml<Sys: LaplaceSystem + ?Sized, P: StrengthPrior + ?Sized>(
    system: &mut Sys,
    prior: &P,
    mode: &CoefficientMode,
    rho: &[f64],
) -> Result<LamlEvaluation, EventHistoryError> {
    let p = mode.coefficients.len();
    let h = prior.strengths();
    let (covariance, certificate) = mode.certified_covariance()?;
    let covariance_error = certificate.inverse_error();
    let lower = mode.factor.lower();
    let (schur_log_determinant, schur_log_determinant_error) = log_determinant_with_error(
        &mode.precision,
        &mode.precision_error,
        &Array2::from_shape_fn((p, p), |(i, j)| lower[(i, j)]),
        &covariance,
        &covariance_error,
    );
    // ½ d ln 2π: a rounded constant, the cited ln charge, and a halved integer
    // count as an exact scaling input.
    let normalizer = ln(&Running::exact(0.0).constant_like(2.0 * std::f64::consts::PI))
        .scale(0.5 * (p + mode.curvature.latent_dimension) as f64);
    // F + ½ d ln 2π − ½ (Σ log det Q_i + log det S) by recursive summation; halving
    // is exact.
    let mut evidence = RunningSum {
        value: mode.profile.log_density,
        error: mode.profile_rounding.log_density,
    };
    for (term, term_error) in [
        (mode.prior.log_density, mode.prior_rounding.log_density),
        (normalizer.value, normalizer.rounding()),
        (
            -0.5 * mode.curvature.latent_log_determinant,
            0.5 * mode.curvature.latent_log_determinant_error,
        ),
        (-0.5 * schur_log_determinant, 0.5 * schur_log_determinant_error),
    ] {
        evidence.add(term, term_error);
    }
    let log_evidence = evidence.value;
    let evidence_band = evidence.error;
    let reduced = system.log_determinant_gradient(&covariance, &covariance_error)?;
    if reduced.value.len() != p || reduced.error.len() != p || !reduced.error.iter().all(is_bound) {
        return Err(invalid(
            "joint Laplace log-determinant gradient has a width inconsistent with the coefficients, or errors that are not finite non-negative bounds",
        ));
    }
    let residual = mode.gradient();
    let mut gradient = Vec::with_capacity(h);
    let mut gradient_band = Vec::with_capacity(h);
    for j in 0..h {
        let mixed = mode.prior.mixed.row(j).to_vec();
        let mixed_error = mode.prior_rounding.mixed.row(j).to_vec();
        let direction = solve(&mode.factor, &mixed);
        let direction_error = certificate.solution_error(&mode.precision, &direction, &mixed, &mixed_error);
        let (prior_derivative, prior_derivative_error) =
            prior.curvature_derivative(&mode.coefficients, rho, &direction, &direction_error)?;
        if prior_derivative.dim() != (p, p) || prior_derivative_error.dim() != (p, p) {
            return Err(invalid(
                "prior curvature derivative has dimensions inconsistent with the coefficients",
            ));
        }
        // tr(C M) = Σ C_ab M_ba, each contraction with its running error.
        let (strength_trace, strength_trace_error) = contraction_with_error(
            &covariance,
            &covariance_error,
            &mode.prior.strength_curvature[j].t().to_owned(),
            &mode.prior_rounding.strength_curvature[j].t().to_owned(),
        );
        let (observation, observation_error) = contraction_with_error(
            &column(&reduced.value),
            &column(&reduced.error),
            &column(&direction),
            &column(&direction_error),
        );
        let (prior_trace, prior_trace_error) = contraction_with_error(
            &covariance,
            &covariance_error,
            &prior_derivative.t().to_owned(),
            &prior_derivative_error.t().to_owned(),
        );
        // ∂ρ log π − ½ tr − ½ ḡᵀδ − ½ tr, by recursive summation; halving is exact.
        let mut score = RunningSum {
            value: mode.prior.strength_gradient[j],
            error: mode.prior_rounding.strength_gradient[j],
        };
        for (term, term_error) in [
            (strength_trace, strength_trace_error),
            (observation, observation_error),
            (prior_trace, prior_trace_error),
        ] {
            score.add(-0.5 * term, 0.5 * term_error);
        }
        gradient.push(score.value);
        // Acceptance also allows the envelope term's first-order change over the
        // certified coefficient residual.
        gradient_band.push(score.error + dot(&direction, &residual).abs());
    }
    if !log_evidence.is_finite()
        || gradient.iter().chain(&gradient_band).any(|v| !v.is_finite())
    {
        return Err(numerical(
            "joint Laplace evidence or its strength score is not representable",
        ));
    }
    Ok(LamlEvaluation {
        log_evidence,
        evidence_band,
        gradient,
        gradient_band,
    })
}

fn resolved(evidence: &LamlEvaluation) -> bool {
    evidence
        .gradient
        .iter()
        .zip(&evidence.gradient_band)
        .all(|(g, band)| g.abs() <= *band)
}

/// `uᵀ H_env u`, with `H_env = diag(∂²ρ log π) + Cᵀ S⁻¹ C` and C the mixed rows
/// ∂θ∂ρ log π: the exact curvature of the envelope term along u, including the
/// motion of the coefficient mode. The log-determinant terms carry the rest of
/// the LAML curvature.
fn envelope_curvature(mode: &CoefficientMode, direction: &[f64]) -> f64 {
    let diagonal: f64 = direction
        .iter()
        .zip(&mode.prior.strength_second)
        .map(|(u, s)| u * u * s)
        .sum();
    let mixed: Vec<f64> = (0..mode.coefficients.len())
        .map(|q| {
            direction
                .iter()
                .enumerate()
                .map(|(j, u)| u * mode.prior.mixed[[j, q]])
                .sum()
        })
        .collect();
    diagonal + dot(&mixed, &solve(&mode.factor, &mixed))
}

/// The root in t of the directional score `g(ρ + t u)ᵀ u`, from the ascent at
/// t = 0, and the coefficients there to warm-start from. Every coefficient mode
/// is warm-started from the last one. Where the Laplace system refuses, opt backs
/// up toward the last resolved strength. `None` when opt finds no resolved root
/// toward +t, which triggers the zero-effect classification and never decides
/// it.
///
/// Opt's bracketing starts at a unit step and doubles, so the bracket count
/// reaches strengths past `ln(f64::MAX)`. Refinement halves the bracket at least
/// every two steps, so its count covers that range down to the tolerance.
fn directional_root<Sys: LaplaceSystem + ?Sized, P: StrengthPrior + ?Sized>(
    system: &mut Sys,
    prior: &P,
    rho: &[f64],
    unit: &[f64],
    start: Vec<f64>,
    tolerance: f64,
) -> Result<Option<(f64, Vec<f64>)>, EventHistoryError> {
    let range = f64::MAX.ln();
    let mut config = opt::RootConfig::new(tolerance, 0, 0);
    config.max_bracket_iters = (range + 1.0).log2().ceil() as usize;
    let halvings = (range / tolerance.max(f64::EPSILON)).log2().ceil() as usize;
    config.max_refine_iters = 2 * (halvings + 1);
    let mut warm = start;
    let oracle = |t: f64| -> Result<opt::RootProbe<EventHistoryError>, EventHistoryError> {
        let trial: Vec<f64> = rho.iter().zip(unit).map(|(r, u)| r + t * u).collect();
        let mode = match coefficient_mode(system, prior, &warm, &trial) {
            Ok(mode) => mode,
            Err(refusal @ EventHistoryError::NumericalFailure { .. }) => return Ok(Err(refusal)),
            Err(error) => return Err(error),
        };
        let evidence = match laml(system, prior, &mode, &trial) {
            Ok(evidence) => evidence,
            Err(refusal @ EventHistoryError::NumericalFailure { .. }) => return Ok(Err(refusal)),
            Err(error) => return Err(error),
        };
        let sample = opt::RootSample {
            value: dot(&evidence.gradient, unit),
            d1: -envelope_curvature(&mode, unit).abs(),
            d2: 0.0,
        };
        warm = mode.coefficients;
        Ok(Ok(sample))
    };
    match opt::find_root_monotone_resolvable(oracle, 0.0, &config, None) {
        Ok(solution) => Ok(Some((solution.root, warm))),
        Err(
            opt::ResolvableRootError::Root(opt::RootError::BracketingExhausted { .. })
            | opt::ResolvableRootError::UnresolvableBeyond { .. },
        ) => Ok(None),
        Err(opt::ResolvableRootError::Root(opt::RootError::Eval(error))) => Err(error),
        Err(error) => Err(numerical(format!("strength line search failed: {error}"))),
    }
}

/// A zero-effect face of one strength: its function is constant, so its
/// coefficient block is pinned at zero and the other blocks keep their prior.
pub(super) struct ZeroEffectFace<'p> {
    /// Full-layout coordinates of the pinned block.
    pub(super) pinned: Vec<usize>,
    /// S_j on those coordinates: the block's prior precision is λ_j S_j.
    pub(super) penalty: Array2<f64>,
    /// The normalized prior of the remaining blocks over the kept coordinates
    /// in ascending layout order, with the other strengths in their order.
    pub(super) prior: Box<dyn StrengthPrior + 'p>,
}

impl StrengthPrior for Box<dyn StrengthPrior + '_> {
    fn strengths(&self) -> usize {
        (**self).strengths()
    }

    fn evaluate(&self, theta: &[f64], rho: &[f64]) -> Result<PriorCurvature, EventHistoryError> {
        (**self).evaluate(theta, rho)
    }

    fn rounding(&self, theta: &[f64], rho: &[f64]) -> Result<PriorRounding, EventHistoryError> {
        (**self).rounding(theta, rho)
    }

    fn curvature_derivative(
        &self,
        theta: &[f64],
        rho: &[f64],
        direction: &[f64],
        direction_error: &[f64],
    ) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
        (**self).curvature_derivative(theta, rho, direction, direction_error)
    }

    fn face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError> {
        (**self).face(j)
    }
}

/// The observation system on a face: the pinned coordinates are held at zero,
/// and every quantity is read on the kept coordinates.
struct Pinned<'s, Sys: ?Sized> {
    system: &'s mut Sys,
    kept: Vec<usize>,
}

impl<Sys: LaplaceSystem + ?Sized> Pinned<'_, Sys> {
    fn padded(&self, theta: &[f64]) -> Vec<f64> {
        pad(theta, &self.kept, self.system.coefficients())
    }

    fn select(&self, full: &[f64]) -> Result<Vec<f64>, EventHistoryError> {
        if full.len() != self.system.coefficients() {
            return Err(invalid(
                "joint Laplace quantity has a width inconsistent with the coefficients",
            ));
        }
        Ok(self.kept.iter().map(|&index| full[index]).collect())
    }
}

impl<Sys: LaplaceSystem + ?Sized> LaplaceSystem for Pinned<'_, Sys> {
    fn coefficients(&self) -> usize {
        self.kept.len()
    }

    fn profile(&mut self, theta: &[f64]) -> Result<ProfileValue, EventHistoryError> {
        let full = self.padded(theta);
        let value = self.system.profile(&full)?;
        Ok(ProfileValue {
            gradient: self.select(&value.gradient)?,
            ..value
        })
    }

    fn profile_rounding(&mut self) -> Result<ProfileRounding, EventHistoryError> {
        let rounding = self.system.profile_rounding()?;
        Ok(ProfileRounding {
            gradient: self.select(&rounding.gradient)?,
            ..rounding
        })
    }

    fn curvature(&mut self) -> Result<LaplaceCurvature, EventHistoryError> {
        let curvature = self.system.curvature()?;
        validate_curvature(&curvature, self.system.coefficients())?;
        let restrict = |matrix: &Array2<f64>| {
            Array2::from_shape_fn((self.kept.len(), self.kept.len()), |(a, b)| {
                matrix[[self.kept[a], self.kept[b]]]
            })
        };
        Ok(LaplaceCurvature {
            schur: restrict(&curvature.schur),
            schur_error: restrict(&curvature.schur_error),
            ..curvature
        })
    }

    fn log_determinant_gradient(
        &mut self,
        covariance: &Array2<f64>,
        covariance_error: &Array2<f64>,
    ) -> Result<LogDeterminantGradient, EventHistoryError> {
        let p = self.system.coefficients();
        let reduced = self.system.log_determinant_gradient(
            &embed(covariance, &self.kept, p),
            &embed(covariance_error, &self.kept, p),
        )?;
        Ok(LogDeterminantGradient {
            value: self.select(&reduced.value)?,
            error: self.select(&reduced.error)?,
        })
    }
}

/// `∂LAML/∂τ_j` at `τ_j = 1/λ_j = 0` on a face, with its rounding band.
struct BoundaryScore {
    value: f64,
    /// The variance-component score `½ [g_jᵀ Σ g_j − tr(Σ H̃_jj)]`, which omits
    /// both motion terms. It decides nothing; it shows where the full score
    /// and that approximation disagree.
    variance_component: f64,
    band: f64,
}

/// The exact derivative of the joint Laplace evidence in the inverse strength
/// of a pinned block at its zero-effect limit, read at the face's certified mode
/// padded with zeros:
///
/// ```text
/// ∂LAML/∂τ_j|₀ = ½ [g_jᵀ Σ g_j − tr(Σ H̃_jj) − ḡ̃_jᵀ Σ g_j − tr(S_r⁻¹ D H_π,r[δ_r])],
/// Σ = S_j⁻¹,   H̃_jj = S₀_jj − S₀_jr S_r⁻¹ S₀_rj,   ḡ̃_j = ḡ_j − S₀_jr S_r⁻¹ ḡ_r,
/// δ_r = −S_r⁻¹ S₀_rj Σ g_j.
/// ```
///
/// g_j is the profile gradient along the block, S₀ the observation Schur block,
/// S_r the face's precision and H_π,r the face's prior curvature at the other
/// strengths `rho`. A block prior N(0, τ Σ) moves the mode by θ̂_j = τ Σ g_j and
/// θ̂_r = τ δ_r. The first term is the objective's gain and the trace the block's
/// curvature. The last two are the motion of the observation log determinant and
/// of the face prior's curvature with the mode. The block's normalizer cancels
/// its log-determinant term exactly, so the face's evidence is the limit's
/// evidence.
///
/// The band adds the rounding of the four terms and the first-order effect of
/// the face's certified coefficient residual r, which moves g_j by −S₀_jr S_r⁻¹ r.
fn boundary_score<Sys: LaplaceSystem + ?Sized>(
    system: &mut Sys,
    face: &ZeroEffectFace<'_>,
    kept: &[usize],
    rho: &[f64],
    mode: &CoefficientMode,
) -> Result<BoundaryScore, EventHistoryError> {
    let p = system.coefficients();
    let q = face.pinned.len();
    if face.penalty.dim() != (q, q) || kept.len() + q != p || mode.coefficients.len() != kept.len() {
        return Err(invalid(
            "zero-effect face has dimensions inconsistent with the coefficients",
        ));
    }
    let theta = pad(&mode.coefficients, kept, p);
    let profile = system.profile(&theta)?;
    validate_profile(&profile, p)?;
    let profile_rounding = system.profile_rounding()?;
    validate_profile_rounding(&profile_rounding, p)?;
    let curvature = system.curvature()?;
    validate_curvature(&curvature, p)?;
    let face_precision = &mode.precision;
    let (face_covariance, face_certificate) = mode.certified_covariance()?;
    let face_covariance_error = face_certificate.inverse_error();
    let log_determinant = system.log_determinant_gradient(
        &embed(&face_covariance, kept, p),
        &embed(&face_covariance_error, kept, p),
    )?;
    if log_determinant.value.len() != p
        || log_determinant.error.len() != p
        || !log_determinant.error.iter().all(is_bound)
    {
        return Err(invalid(
            "joint Laplace log-determinant gradient has a width inconsistent with the coefficients, or errors that are not finite non-negative bounds",
        ));
    }
    // Absolute errors under Running's rules, carried inline (brief 19:15;
    // `law::numerical::Running` is the reference). Each solve injects its error
    // only through its certificate.
    let schur_error = &curvature.schur_error;
    // Rows S₀_jr S_r⁻¹, one per pinned coordinate.
    let mut coupling = Vec::with_capacity(q);
    let mut coupling_error = Vec::with_capacity(q);
    for &i in &face.pinned {
        let row: Vec<f64> = kept.iter().map(|&r| curvature.schur[[i, r]]).collect();
        let row_error: Vec<f64> = kept.iter().map(|&r| schur_error[[i, r]]).collect();
        let solved = solve(&mode.factor, &row);
        coupling_error.push(face_certificate.solution_error(face_precision, &solved, &row, &row_error));
        coupling.push(solved);
    }
    let mut complement = Array2::zeros((q, q));
    let mut complement_error = Array2::zeros((q, q));
    for (a, &i) in face.pinned.iter().enumerate() {
        for (b, &k) in face.pinned.iter().enumerate() {
            let pinned_column: Vec<f64> = kept.iter().map(|&r| curvature.schur[[r, k]]).collect();
            let pinned_column_error: Vec<f64> = kept.iter().map(|&r| schur_error[[r, k]]).collect();
            let (reduction, reduction_error) = contraction_with_error(
                &column(&coupling[a]),
                &column(&coupling_error[a]),
                &column(&pinned_column),
                &column(&pinned_column_error),
            );
            complement[[a, b]] = curvature.schur[[i, k]] - reduction;
            complement_error[[a, b]] =
                sum_error(curvature.schur[[i, k]], schur_error[[i, k]], -reduction, reduction_error);
        }
    }
    let gradient: Vec<f64> = face.pinned.iter().map(|&i| profile.gradient[i]).collect();
    let gradient_error: Vec<f64> = face.pinned.iter().map(|&i| profile_rounding.gradient[i]).collect();
    let kept_log_determinant: Vec<f64> = kept.iter().map(|&r| log_determinant.value[r]).collect();
    let kept_log_determinant_error: Vec<f64> = kept.iter().map(|&r| log_determinant.error[r]).collect();
    let mut motion_gradient = Vec::with_capacity(q);
    let mut motion_gradient_error = Vec::with_capacity(q);
    for (a, &i) in face.pinned.iter().enumerate() {
        let (reduction, reduction_error) = contraction_with_error(
            &column(&coupling[a]),
            &column(&coupling_error[a]),
            &column(&kept_log_determinant),
            &column(&kept_log_determinant_error),
        );
        motion_gradient.push(log_determinant.value[i] - reduction);
        motion_gradient_error.push(sum_error(
            log_determinant.value[i],
            log_determinant.error[i],
            -reduction,
            reduction_error,
        ));
    }
    let penalty = factor(&face.penalty)?;
    let penalty_covariance = inverse(&penalty);
    let penalty_certificate = solve_forward_error(
        &face.penalty,
        &penalty_covariance,
        &Array2::zeros(face.penalty.dim()),
        "zero-effect face penalty",
    )?;
    let penalty_covariance_error = penalty_certificate.inverse_error();
    let spread = solve(&penalty, &gradient);
    let spread_error =
        penalty_certificate.solution_error(&face.penalty, &spread, &gradient, &gradient_error);
    // δ_r = −S_r⁻¹ S₀_rj Σ g_j, the kept coefficients' motion per unit τ.
    let (shift, shift_error): (Vec<f64>, Vec<f64>) = (0..kept.len())
        .map(|r| {
            let mut sum = RunningSum::default();
            for a in 0..q {
                sum.add(
                    coupling[a][r] * spread[a],
                    product_error(coupling[a][r], coupling_error[a][r], spread[a], spread_error[a]),
                );
            }
            (-sum.value, sum.error)
        })
        .unzip();
    let (prior_derivative, prior_derivative_error) =
        face.prior.curvature_derivative(&mode.coefficients, rho, &shift, &shift_error)?;
    if prior_derivative.dim() != (kept.len(), kept.len())
        || prior_derivative_error.dim() != (kept.len(), kept.len())
    {
        return Err(invalid(
            "face prior curvature derivative has dimensions inconsistent with the kept coefficients",
        ));
    }
    let (gain, gain_error) = contraction_with_error(
        &column(&gradient),
        &column(&gradient_error),
        &column(&spread),
        &column(&spread_error),
    );
    // tr(C M) = Σ C_ab M_ba.
    let (trace, trace_error) = contraction_with_error(
        &penalty_covariance,
        &penalty_covariance_error,
        &complement.t().to_owned(),
        &complement_error.t().to_owned(),
    );
    let (motion, motion_error) = contraction_with_error(
        &column(&motion_gradient),
        &column(&motion_gradient_error),
        &column(&spread),
        &column(&spread_error),
    );
    let (prior_motion, prior_motion_error) = contraction_with_error(
        &face_covariance,
        &face_covariance_error,
        &prior_derivative.t().to_owned(),
        &prior_derivative_error.t().to_owned(),
    );
    // ½ [(gain − trace) − motion − prior motion] by recursive summation; halving
    // is exact.
    let mut score = RunningSum {
        value: gain,
        error: gain_error,
    };
    score.add(-trace, trace_error);
    let variance_component = 0.5 * score.value;
    score.add(-motion, motion_error);
    score.add(-prior_motion, prior_motion_error);
    let value = 0.5 * score.value;
    let residual = mode.gradient();
    let residual_effect: f64 = coupling
        .iter()
        .zip(&spread)
        .map(|(row, s)| (s * dot(row, &residual)).abs())
        .sum();
    let band = 0.5 * score.error + residual_effect;
    if !value.is_finite() || !band.is_finite() {
        return Err(numerical(
            "zero-effect boundary score is not representable",
        ));
    }
    Ok(BoundaryScore {
        value,
        variance_component,
        band,
    })
}


/// How a strength's zero-effect limit was decided.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LimitStatus {
    /// The boundary score is resolvably negative: the evidence falls into the
    /// interior.
    Resolved { score: f64, bound: f64 },
    /// The boundary score lies within its band. The first-order score does not
    /// decide between the limit and the interior, so the limit is taken and the
    /// tie reported.
    FirstOrderTie { score: f64, bound: f64 },
}

/// A strength at its zero-effect limit, where its function is constant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZeroEffectLimit {
    pub strength: usize,
    pub status: LimitStatus,
}

/// The decision a face's boundary score makes. `None` means the evidence rises
/// into the interior.
fn limit_status(score: &BoundaryScore) -> Option<LimitStatus> {
    if score.value < -score.band {
        Some(LimitStatus::Resolved {
            score: score.value,
            bound: score.band,
        })
    } else if score.value <= score.band {
        Some(LimitStatus::FirstOrderTie {
            score: score.value,
            bound: score.band,
        })
    } else {
        None
    }
}

/// One strength's zero-effect face read at the other strengths in ρ.
struct FaceReading {
    strength: usize,
    score: BoundaryScore,
    /// The face's certified mode, padded with zeros on the pinned block.
    coefficients: Vec<f64>,
}

/// Read each candidate strength's face at the other strengths in ρ: its
/// certified mode and its boundary score. A strength whose limit changes the
/// observation or dynamics law has no face and is never read.
fn read_faces<Sys: LaplaceSystem + ?Sized, P: StrengthPrior + ?Sized>(
    system: &mut Sys,
    prior: &P,
    rho: &[f64],
    candidates: &[usize],
    coefficients: &[f64],
) -> Result<Vec<FaceReading>, EventHistoryError> {
    let p = system.coefficients();
    let mut readings = Vec::new();
    for &j in candidates {
        let Some(face) = prior.face(j)? else {
            continue;
        };
        if face.prior.strengths() + 1 != rho.len() || face.pinned.iter().any(|&i| i >= p) {
            return Err(invalid(
                "zero-effect face does not remove exactly one strength and its coordinates",
            ));
        }
        let kept: Vec<usize> = (0..p).filter(|i| !face.pinned.contains(i)).collect();
        let others: Vec<f64> = rho
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != j)
            .map(|(_, r)| *r)
            .collect();
        let start: Vec<f64> = kept.iter().map(|&i| coefficients[i]).collect();
        let mode = {
            let mut pinned = Pinned {
                system: &mut *system,
                kept: kept.clone(),
            };
            coefficient_mode(&mut pinned, &face.prior, &start, &others)?
        };
        let score = boundary_score(system, &face, &kept, &others, &mode)?;
        readings.push(FaceReading {
            strength: j,
            score,
            coefficients: pad(&mode.coefficients, &kept, p),
        });
    }
    Ok(readings)
}

/// The read strengths whose boundary score takes the zero-effect limit.
fn zero_effect_limits(readings: &[FaceReading]) -> Vec<ZeroEffectLimit> {
    readings
        .iter()
        .filter_map(|reading| {
            limit_status(&reading.score).map(|status| ZeroEffectLimit {
                strength: reading.strength,
                status,
            })
        })
        .collect()
}

/// How a strength search ends.
pub(super) enum StrengthSearch {
    /// The certified optimum of the model with the larger evidence.
    Fitted(FaceFit),
    /// A resolved interior optimum and the search at its joint zero-effect limit,
    /// whose evidence values agree within both errors. Neither is dropped.
    EvidenceTie {
        interior: FaceFit,
        limit: Box<StrengthSearch>,
    },
}

/// How one conjugate ascent on a model ends.
enum Ascent {
    /// Every score component is within its band.
    Resolved {
        log_strengths: Vec<f64>,
        mode: CoefficientMode,
        evidence: LamlEvaluation,
        iterations: usize,
    },
    /// A ray with no resolved root toward +t, from this point, growing the listed
    /// strengths.
    Unbracketed {
        log_strengths: Vec<f64>,
        coefficients: Vec<f64>,
        growing: Vec<usize>,
    },
}

/// Maximize LAML(ρ) over the strengths and their zero-effect limits, and certify
/// the optimum. EB is sup over the closed range (brief 17:50, 20:50).
///
/// The search runs on pinned sets P, where M_P pins P's blocks together and
/// optimizes the other strengths:
/// - `ascend` searches M_P's interior.
/// - At a resolved optimum O_P every free strength's face is read there. When none
///   takes its limit, O_P is the answer. Otherwise the search moves to M_{P∪C} for
///   the set C that does, and O_P is compared with that search's evidence value,
///   with both errors: the larger wins, and a resolved tie returns both.
/// - On a ray with no resolved root the growing faces are read. A limit moves the
///   search to M_{P∪C}. When none takes it, the search restarts from one face's
///   fit, at most once per strength.
/// - Before a fit with P ≠ ∅ is returned, each member of P is re-read at that fit
///   with the others still pinned, and members whose score rises into the interior
///   are released. A release into a set already entered leaves that set's answer
///   standing where it was reached, and a new limit set already entered is refused:
///   the readings cycle at the evidence's resolution. Every entered set is new, so
///   the search ends.
///
/// Models compose faces through `dyn` systems and priors, so the recursion
/// instantiates finitely.
pub(super) fn optimize_strengths<Sys: LaplaceSystem, P: StrengthPrior>(
    system: &mut Sys,
    prior: &P,
    initial_coefficients: &[f64],
    initial_log_strengths: &[f64],
) -> Result<StrengthSearch, EventHistoryError> {
    let h = prior.strengths();
    if h == 0
        || initial_log_strengths.len() != h
        || initial_log_strengths.iter().any(|v| !v.is_finite())
        || initial_coefficients.len() != system.coefficients()
    {
        return Err(invalid(
            "strength optimization needs one finite initial log strength per prior strength and coefficients of the model's width",
        ));
    }
    let mut visited = Vec::new();
    search(
        system,
        prior,
        &[],
        initial_coefficients,
        initial_log_strengths,
        initial_log_strengths,
        &mut visited,
    )?
    .ok_or_else(|| numerical("the full model's zero-effect search has no members to release"))
}

/// An interior optimum against the search at its joint zero-effect limit, by
/// evidence value with both errors.
fn compare(interior: FaceFit, limit: StrengthSearch) -> StrengthSearch {
    let other = match &limit {
        StrengthSearch::Fitted(fit) | StrengthSearch::EvidenceTie { interior: fit, .. } => &fit.evidence,
    };
    let margin = interior.evidence.log_evidence - other.log_evidence;
    let bar = interior.evidence.evidence_band + other.evidence_band;
    if margin > bar {
        StrengthSearch::Fitted(interior)
    } else if margin < -bar {
        limit
    } else {
        StrengthSearch::EvidenceTie {
            interior,
            limit: Box::new(limit),
        }
    }
}

/// Pin the blocks of `set` (strength indices of `prior`, ascending) together and
/// hand the reduced system and prior to `visit`, with the parent coordinates the
/// model keeps. Each face's prior orders the other strengths as its parent does, so
/// later indices move down by one per removal. `Ok(false)` when some strength in
/// `set` has no face.
fn with_face(
    system: &mut dyn LaplaceSystem,
    prior: &dyn StrengthPrior,
    set: &[usize],
    kept: &[usize],
    visit: &mut dyn FnMut(&mut dyn LaplaceSystem, &dyn StrengthPrior, &[usize]) -> Result<(), EventHistoryError>,
) -> Result<bool, EventHistoryError> {
    let Some((&first, rest)) = set.split_first() else {
        visit(system, prior, kept)?;
        return Ok(true);
    };
    let Some(face) = prior.face(first)? else {
        return Ok(false);
    };
    let p = system.coefficients();
    if face.prior.strengths() + 1 != prior.strengths()
        || face.pinned.iter().any(|&i| i >= p)
        || rest.iter().any(|&j| j <= first)
    {
        return Err(invalid(
            "a joint zero-effect face needs ascending strengths, each face removing exactly one strength and its coordinates",
        ));
    }
    let local: Vec<usize> = (0..p).filter(|i| !face.pinned.contains(i)).collect();
    let composed: Vec<usize> = local.iter().map(|&i| kept[i]).collect();
    let shifted: Vec<usize> = rest.iter().map(|&j| j - 1).collect();
    let mut pinned = Pinned { system, kept: local };
    with_face(&mut pinned, &*face.prior, &shifted, &composed, visit)
}

/// A model's ascent and the faces read at its end, in the model's layout.
struct ModelReading {
    /// Parent coordinates the model keeps.
    kept: Vec<usize>,
    ascent: Ascent,
    readings: Vec<FaceReading>,
}

/// Ascend the model with `pinned` at its limit from parent-layout coefficients and
/// log strengths, and read the faces of its free strengths where the ascent ends:
/// every free strength at a resolved optimum, the growing ones on a ray.
fn read_model(
    system: &mut dyn LaplaceSystem,
    prior: &dyn StrengthPrior,
    pinned: &[usize],
    coefficients: &[f64],
    log_strengths: &[f64],
) -> Result<ModelReading, EventHistoryError> {
    let p = system.coefficients();
    let free: Vec<usize> = (0..prior.strengths()).filter(|j| !pinned.contains(j)).collect();
    let identity: Vec<usize> = (0..p).collect();
    let mut outcome = None;
    let exists = with_face(
        system,
        prior,
        pinned,
        &identity,
        &mut |model: &mut dyn LaplaceSystem,
              model_prior: &dyn StrengthPrior,
              kept: &[usize]|
              -> Result<(), EventHistoryError> {
            let start: Vec<f64> = kept.iter().map(|&i| coefficients[i]).collect();
            let rho: Vec<f64> = free.iter().map(|&j| log_strengths[j]).collect();
            let ascent = ascend(model, model_prior, &start, &rho)?;
            let readings = match &ascent {
                Ascent::Resolved {
                    log_strengths: end,
                    mode,
                    ..
                } => {
                    let candidates: Vec<usize> = (0..free.len()).collect();
                    read_faces(model, model_prior, end, &candidates, &mode.coefficients)?
                }
                Ascent::Unbracketed {
                    log_strengths: end,
                    coefficients: at,
                    growing,
                } => read_faces(model, model_prior, end, growing, at)?,
            };
            outcome = Some(ModelReading {
                kept: kept.to_vec(),
                ascent,
                readings,
            });
            Ok(())
        },
    )?;
    match outcome {
        Some(reading) if exists => Ok(reading),
        _ => Err(invalid("a pinned strength has no zero-effect face")),
    }
}

/// The boundary score of pinned strength `q` at a fit of its joint limit: face `q`
/// of the model where the other members stay pinned, at parent-layout coefficients
/// and log strengths.
fn read_member(
    system: &mut dyn LaplaceSystem,
    prior: &dyn StrengthPrior,
    others: &[usize],
    q: usize,
    coefficients: &[f64],
    log_strengths: &[f64],
) -> Result<BoundaryScore, EventHistoryError> {
    let p = system.coefficients();
    let free: Vec<usize> = (0..prior.strengths()).filter(|j| !others.contains(j)).collect();
    let Some(local) = free.iter().position(|&j| j == q) else {
        return Err(invalid("a pinned strength is not free in the model that re-reads it"));
    };
    let identity: Vec<usize> = (0..p).collect();
    let mut score = None;
    let exists = with_face(
        system,
        prior,
        others,
        &identity,
        &mut |model: &mut dyn LaplaceSystem,
              model_prior: &dyn StrengthPrior,
              kept: &[usize]|
              -> Result<(), EventHistoryError> {
            let start: Vec<f64> = kept.iter().map(|&i| coefficients[i]).collect();
            let rho: Vec<f64> = free.iter().map(|&j| log_strengths[j]).collect();
            score = read_faces(model, model_prior, &rho, &[local], &start)?
                .pop()
                .map(|reading| reading.score);
            Ok(())
        },
    )?;
    match score {
        Some(score) if exists => Ok(score),
        _ => Err(invalid("a pinned strength has no zero-effect face")),
    }
}

/// The search from the model with `pinned` at its limit (parent strength indices,
/// ascending), from parent-layout coefficients and log strengths. `visited` holds
/// every pinned set this search has entered. `None` means this model's own re-read
/// released members into a set already entered.
fn search(
    system: &mut dyn LaplaceSystem,
    prior: &dyn StrengthPrior,
    pinned: &[usize],
    coefficients: &[f64],
    log_strengths: &[f64],
    initial_log_strengths: &[f64],
    visited: &mut Vec<Vec<usize>>,
) -> Result<Option<StrengthSearch>, EventHistoryError> {
    visited.push(pinned.to_vec());
    let p = system.coefficients();
    let free: Vec<usize> = (0..prior.strengths()).filter(|j| !pinned.contains(j)).collect();
    let mut start = coefficients.to_vec();
    let mut rho = log_strengths.to_vec();
    let mut restarted: Vec<usize> = Vec::new();
    loop {
        let reading = read_model(system, prior, pinned, &start, &rho)?;
        let limits: Vec<ZeroEffectLimit> = zero_effect_limits(&reading.readings)
            .into_iter()
            .map(|limit| ZeroEffectLimit {
                strength: free[limit.strength],
                ..limit
            })
            .collect();
        let mut joint = pinned.to_vec();
        joint.extend(limits.iter().map(|limit| limit.strength));
        joint.sort_unstable();
        match reading.ascent {
            Ascent::Resolved {
                log_strengths: optimum,
                mode,
                evidence,
                iterations,
            } => {
                let point = spread(&optimum, &free, &rho);
                let fitted = pad(&mode.coefficients, &reading.kept, p);
                // Each pinned strength's status, read at this fit (brief 20:50).
                let mut statuses = Vec::with_capacity(pinned.len());
                let mut released = Vec::new();
                for &q in pinned {
                    let others: Vec<usize> = pinned.iter().copied().filter(|&r| r != q).collect();
                    let score = read_member(system, prior, &others, q, &fitted, &point)?;
                    match limit_status(&score) {
                        Some(status) => statuses.push(ZeroEffectLimit { strength: q, status }),
                        None => released.push(q),
                    }
                }
                if !released.is_empty() {
                    let rest: Vec<usize> = pinned.iter().copied().filter(|q| !released.contains(q)).collect();
                    if visited.contains(&rest) {
                        return Ok(None);
                    }
                    let restart: Vec<f64> = released.iter().map(|&q| initial_log_strengths[q]).collect();
                    return search(
                        system,
                        prior,
                        &rest,
                        &fitted,
                        &spread(&restart, &released, &point),
                        initial_log_strengths,
                        visited,
                    );
                }
                let fit = FaceFit {
                    limits: statuses,
                    kept: reading.kept,
                    strengths: free,
                    log_strengths: optimum,
                    mode,
                    evidence,
                    iterations,
                };
                if limits.is_empty() {
                    return Ok(Some(StrengthSearch::Fitted(fit)));
                }
                if visited.contains(&joint) {
                    return Err(numerical(format!(
                        "the zero-effect search returned to the pinned strengths {joint:?} from {pinned:?}: its readings cycle at the evidence's resolution"
                    )));
                }
                let limit = search(system, prior, &joint, &fitted, &point, initial_log_strengths, visited)?;
                return Ok(Some(match limit {
                    Some(limit) => compare(fit, limit),
                    // The joint limit released back into an entered set: the interior
                    // optimum stands.
                    None => StrengthSearch::Fitted(fit),
                }));
            }
            Ascent::Unbracketed {
                log_strengths: end,
                coefficients: at,
                ..
            } => {
                let point = spread(&end, &free, &rho);
                let from = pad(&at, &reading.kept, p);
                if !limits.is_empty() && !visited.contains(&joint) {
                    if let Some(limit) = search(system, prior, &joint, &from, &point, initial_log_strengths, visited)? {
                        return Ok(Some(limit));
                    }
                }
                // Every growing face says the evidence rises into the interior, or
                // their joint limit released: restart from one face's fit, once per
                // strength.
                let Some(face) = reading
                    .readings
                    .iter()
                    .find(|r| !restarted.contains(&free[r.strength]))
                else {
                    return Err(numerical(format!(
                        "the evidence keeps rising with no resolved root from log strengths {point:?}, and every growing strength with a zero-effect face has already restarted from its fit"
                    )));
                };
                restarted.push(free[face.strength]);
                start = pad(&face.coefficients, &reading.kept, p);
                rho = point;
            }
        }
    }
}

/// Conjugate ascent of LAML(ρ) on one model.
///
/// Conjugate ascent directions (Polak–Ribière+, restarted every h directions and
/// whenever a direction is not an ascent) each end at the root of the directional
/// score, which opt's resolvable monotone root finder brackets and refines at the
/// finest score band. The ascent is resolved when every score component is within
/// its band; a score within its band may still be approaching a limit, which the
/// caller reads through the faces. A ray with no resolved root hands its point and
/// growing strengths back. A ray that only shrinks strengths rises toward λ → 0,
/// which a normalized prior makes improper, and is refused. A full cycle that does
/// not contract the largest score/band ratio has reached the score's resolution
/// without resolved stationarity and is refused.
fn ascend(
    system: &mut dyn LaplaceSystem,
    prior: &dyn StrengthPrior,
    initial_coefficients: &[f64],
    initial_log_strengths: &[f64],
) -> Result<Ascent, EventHistoryError> {
    let h = prior.strengths();
    let mut rho = initial_log_strengths.to_vec();
    let mut coefficients = initial_coefficients.to_vec();
    let mut direction = vec![0.0; h];
    let mut previous_gradient: Option<Vec<f64>> = None;
    let mut cycle = 0usize;
    let mut cycle_ratio = f64::INFINITY;
    let mut iterations = 0;
    loop {
        let mode = coefficient_mode(system, prior, &coefficients, &rho)?;
        let evidence = laml(system, prior, &mode, &rho)?;
        if resolved(&evidence) {
            return Ok(Ascent::Resolved {
                log_strengths: rho,
                mode,
                evidence,
                iterations,
            });
        }
        let ratio = evidence
            .gradient
            .iter()
            .zip(&evidence.gradient_band)
            .map(|(g, band)| g.abs() / band)
            .fold(0.0_f64, f64::max);
        if cycle == 0 {
            if ratio >= cycle_ratio {
                return Err(numerical(format!(
                    "strength score stopped contracting over a full cycle of conjugate directions: {:?} against bands {:?}",
                    evidence.gradient, evidence.gradient_band
                )));
            }
            cycle_ratio = ratio;
        }
        let gradient = &evidence.gradient;
        let beta = match previous_gradient.as_ref().filter(|_| cycle > 0) {
            Some(previous) => {
                ((dot(gradient, gradient) - dot(gradient, previous)) / dot(previous, previous)).max(0.0)
            }
            None => 0.0,
        };
        for (d, g) in direction.iter_mut().zip(gradient) {
            *d = g + beta * *d;
        }
        if dot(&direction, gradient) <= 0.0 {
            direction.clone_from(gradient);
        }
        let length = direction.iter().fold(0.0_f64, |a, &b| a.hypot(b));
        let unit: Vec<f64> = direction.iter().map(|d| d / length).collect();
        // The finest band: a coarser one stops the ray at the noise of its
        // largest component and leaves the finest one unresolved.
        let tolerance = evidence.gradient_band.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let Some((step, warm)) =
            directional_root(system, prior, &rho, &unit, mode.coefficients.clone(), tolerance)?
        else {
            let growing: Vec<usize> = unit
                .iter()
                .enumerate()
                .filter(|(_, u)| **u > 0.0)
                .map(|(j, _)| j)
                .collect();
            if growing.is_empty() {
                return Err(EventHistoryError::Fit {
                    reason: format!(
                        "the evidence keeps rising along {unit:?} as strengths go to zero: the prior is improper in that limit, and no fit is reported"
                    ),
                });
            }
            return Ok(Ascent::Unbracketed {
                log_strengths: rho,
                coefficients: mode.coefficients,
                growing,
            });
        };
        iterations += 1;
        for (value, u) in rho.iter_mut().zip(&unit) {
            *value += step * u;
        }
        coefficients = warm;
        previous_gradient = Some(evidence.gradient);
        cycle = (cycle + 1) % h;
    }
}

#[cfg(test)]
mod tests {
    use super::super::law::numerical::upper;
    use super::*;
    use crate::scalar::{exp, recip};

    /// Poisson counts with log rate θ₀ + θ₁ x_i + z_i, z_i ~ N(0, 1). Its
    /// complete density has nonzero third derivatives in (θ, z), so the
    /// log-determinant terms are exercised.
    struct PoissonIntercepts {
        x: Vec<f64>,
        exposure: Vec<f64>,
        count: Vec<f64>,
        theta: Vec<f64>,
        modes: Vec<f64>,
    }

    impl PoissonIntercepts {
        fn new(x: Vec<f64>) -> Self {
            Self {
                x,
                exposure: vec![1.0, 2.0, 1.5, 1.0, 2.5, 1.0],
                count: vec![0.0, 2.0, 1.0, 3.0, 4.0, 2.0],
                theta: vec![0.0; 2],
                modes: vec![0.0; 6],
            }
        }

        /// Counts near e² per unit exposure: the intercept is clearly informed.
        fn informed(x: Vec<f64>) -> Self {
            Self {
                count: vec![7.0, 15.0, 11.0, 8.0, 18.0, 7.0],
                ..Self::new(x)
            }
        }

        /// Covariates, exposures and counts mirrored about x = 0, so the profile
        /// objective is even in θ₁.
        fn symmetric() -> Self {
            Self {
                exposure: vec![1.0, 2.0, 1.5, 1.5, 2.0, 1.0],
                count: vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0],
                ..Self::new(vec![-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
            }
        }

        fn weight(&self, i: usize) -> f64 {
            self.exposure[i] * (self.theta[0] + self.theta[1] * self.x[i] + self.modes[i]).exp()
        }

        fn log_density(&self) -> f64 {
            (0..self.x.len())
                .map(|i| {
                    let eta = self.theta[0] + self.theta[1] * self.x[i] + self.modes[i];
                    self.count[i] * eta
                        - self.weight(i)
                        - 0.5 * self.modes[i].powi(2)
                        - 0.5 * (2.0 * std::f64::consts::PI).ln()
                })
                .sum()
        }
    }

    /// Subject i's route at (θ, ẑ_i) over any field: f64 where the search reads it,
    /// Running where a check reads its errors.
    struct Subject<S> {
        log_density: S,
        gradient: [S; 2],
        /// ∂f_i/∂z_i at ẑ_i, the latent mode's residual.
        latent_score: S,
        weight: S,
        /// Q_i = w_i + 1.
        precision: S,
        /// B_i = w_i r_i.
        border: [S; 2],
        /// X_i = Q_i⁻¹ B_i.
        solved: [S; 2],
        /// The subject's contribution to S₀.
        schur: [[S; 2]; 2],
    }

    impl PoissonIntercepts {
        fn subject<S: JetField>(&self, i: usize, zero: &S) -> Subject<S> {
            let input = |v: f64| zero.with_value(v);
            let slope = input(self.x[i]);
            let row = [input(1.0), slope.clone()];
            let mode = input(self.modes[i]);
            let eta = input(self.theta[0])
                .add(&input(self.theta[1]).mul(&slope))
                .add(&mode);
            let weight = input(self.exposure[i]).mul(&exp(&eta));
            let count = input(self.count[i]);
            let log_density = count
                .mul(&eta)
                .sub(&weight)
                .sub(&mode.mul(&mode).scale(0.5))
                .sub(&half_log_tau(zero));
            let pull = count.sub(&weight);
            let precision = weight.add(&input(1.0));
            let inverse = recip(&precision);
            let border = [weight.clone(), weight.mul(&slope)];
            let solved = [border[0].mul(&inverse), border[1].mul(&inverse)];
            let schur = [0, 1].map(|a| {
                [0, 1].map(|b| weight.mul(&row[a]).mul(&row[b]).sub(&border[a].mul(&solved[b])))
            });
            Subject {
                log_density,
                gradient: [pull.clone(), pull.mul(&slope)],
                latent_score: pull.sub(&mode),
                weight,
                precision,
                border,
                solved,
                schur,
            }
        }
    }

    impl LaplaceSystem for PoissonIntercepts {
        fn coefficients(&self) -> usize {
            2
        }

        fn profile(&mut self, theta: &[f64]) -> Result<ProfileValue, EventHistoryError> {
            self.theta = theta.to_vec();
            let mut log_density = 0.0;
            let mut gradient = vec![0.0; 2];
            for i in 0..self.x.len() {
                // Newton on a strictly concave scalar: the steps contract until
                // roundoff stops them contracting.
                let mut previous = f64::INFINITY;
                loop {
                    let w = self.weight(i);
                    let step = (self.count[i] - w - self.modes[i]) / (w + 1.0);
                    if step.abs() >= previous {
                        break;
                    }
                    previous = step.abs();
                    self.modes[i] += step;
                }
                let subject = self.subject(i, &0.0);
                log_density += subject.log_density;
                for k in 0..2 {
                    gradient[k] += subject.gradient[k];
                }
            }
            Ok(ProfileValue {
                log_density,
                gradient,
            })
        }

        fn profile_rounding(&mut self) -> Result<ProfileRounding, EventHistoryError> {
            // The same route at Running, plus each gradient's motion over the latent
            // modes' residuals: |B_i (ẑ_i − z_i)| ≤ |w_i r_i| (|r^z_i| + its error) / Q_i
            // to first order.
            let zero = Running::exact(0.0);
            let mut log_density = zero;
            let mut gradient = [zero; 2];
            let mut motion = [RunningSum::default(); 2];
            for i in 0..self.x.len() {
                let subject = self.subject(i, &zero);
                log_density = log_density.add(&subject.log_density);
                let residual = subject.latent_score.value.abs() + subject.latent_score.rounding();
                for k in 0..2 {
                    gradient[k] = gradient[k].add(&subject.gradient[k]);
                    motion[k].add(
                        upper(subject.border[k].value.abs() * residual / subject.precision.value),
                        0.0,
                    );
                }
            }
            Ok(ProfileRounding {
                log_density: log_density.rounding(),
                gradient: (0..2)
                    .map(|k| gradient[k].rounding() + upper(motion[k].value + motion[k].error))
                    .collect(),
            })
        }

        fn curvature(&mut self) -> Result<LaplaceCurvature, EventHistoryError> {
            // The fixture reads its one route at Running, whose values are the f64
            // route's.
            let zero = Running::exact(0.0);
            let mut schur = [[zero; 2]; 2];
            let mut latent_log_determinant = zero;
            for i in 0..self.x.len() {
                let subject = self.subject(i, &zero);
                latent_log_determinant = latent_log_determinant.add(&ln(&subject.precision));
                for a in 0..2 {
                    for b in 0..2 {
                        schur[a][b] = schur[a][b].add(&subject.schur[a][b]);
                    }
                }
            }
            Ok(LaplaceCurvature {
                schur: Array2::from_shape_fn((2, 2), |(a, b)| schur[a][b].value),
                schur_error: Array2::from_shape_fn((2, 2), |(a, b)| schur[a][b].rounding()),
                latent_log_determinant: latent_log_determinant.value,
                latent_log_determinant_error: latent_log_determinant.rounding(),
                latent_dimension: self.x.len(),
            })
        }

        fn log_determinant_gradient(
            &mut self,
            covariance: &Array2<f64>,
            covariance_error: &Array2<f64>,
        ) -> Result<LogDeterminantGradient, EventHistoryError> {
            // Every observation block of 𝓗 is linear in w_i = e_i exp(η_i + z_i):
            // ∂𝓗/∂w_i = [[r_i r_iᵀ, r_i], [r_iᵀ, 1]] on (θ, z_i). The blocks of 𝓗⁻¹ are
            // S⁻¹, −S⁻¹ X_iᵀ and Q_i⁻¹ + X_i S⁻¹ X_iᵀ, so the trace is
            // (r_i − X_i)ᵀ S⁻¹ (r_i − X_i) + 1/Q_i. Along the reduced direction,
            // dw_i = w_i (1 − w_i / Q_i) r_iᵀ δθ. Each covariance entry carries its
            // error as its running bound.
            if covariance.dim() != (2, 2) || covariance_error.dim() != (2, 2) {
                return Err(invalid("the Poisson fixture has two coefficients"));
            }
            let zero = Running::exact(0.0);
            let entry = |a: usize, b: usize| Running {
                value: covariance[[a, b]],
                mu: covariance_error[[a, b]] / f64::EPSILON,
            };
            let mut value = [zero; 2];
            for i in 0..self.x.len() {
                let subject = self.subject(i, &zero);
                let row = [zero.with_value(1.0), zero.with_value(self.x[i])];
                let lever = [row[0].sub(&subject.solved[0]), row[1].sub(&subject.solved[1])];
                let inverse = recip(&subject.precision);
                let mut trace = inverse;
                for a in 0..2 {
                    for b in 0..2 {
                        trace = trace.add(&lever[a].mul(&entry(a, b)).mul(&lever[b]));
                    }
                }
                let damping = subject
                    .weight
                    .mul(&zero.with_value(1.0).sub(&subject.weight.mul(&inverse)));
                let pull = trace.mul(&damping);
                for a in 0..2 {
                    value[a] = value[a].add(&pull.mul(&row[a]));
                }
            }
            Ok(LogDeterminantGradient {
                value: value.iter().map(|v| v.value).collect(),
                error: value.iter().map(Running::rounding).collect(),
            })
        }
    }

    fn covariates() -> Vec<f64> {
        vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    }

    /// ½ ln 2π over the field of `zero`.
    fn half_log_tau<S: JetField>(zero: &S) -> S {
        ln(&zero.constant_like(2.0 * std::f64::consts::PI)).scale(0.5)
    }

    /// A prior's route over any field, read as values at f64 and as errors at
    /// Running.
    struct PriorRoute<S> {
        log_density: S,
        gradient: Vec<S>,
        strength_gradient: Vec<S>,
        strength_second: Vec<S>,
        negative_hessian: Array2<S>,
        strength_curvature: Vec<Array2<S>>,
        mixed: Array2<S>,
    }

    impl PriorRoute<f64> {
        fn curvature(self) -> PriorCurvature {
            PriorCurvature {
                log_density: self.log_density,
                gradient: self.gradient,
                strength_gradient: self.strength_gradient,
                strength_second: self.strength_second,
                negative_hessian: self.negative_hessian,
                strength_curvature: self.strength_curvature,
                mixed: self.mixed,
            }
        }
    }

    impl PriorRoute<Running> {
        fn errors(&self) -> PriorRounding {
            PriorRounding {
                log_density: self.log_density.rounding(),
                gradient: self.gradient.iter().map(Running::rounding).collect(),
                strength_gradient: self.strength_gradient.iter().map(Running::rounding).collect(),
                negative_hessian: self.negative_hessian.map(Running::rounding),
                strength_curvature: self
                    .strength_curvature
                    .iter()
                    .map(|m| m.map(Running::rounding))
                    .collect(),
                mixed: self.mixed.map(Running::rounding),
            }
        }
    }

    /// A test prior as one route over any field: its values at f64, its errors at
    /// Running, and its curvature derivative over a direction carrying its errors.
    trait RoutedPrior {
        const STRENGTHS: usize;
        fn route<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            zero: &S,
        ) -> Result<PriorRoute<S>, EventHistoryError>;
        /// Ḣ_π[δ] over the field of the direction.
        fn derivative<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            direction: &[S],
            zero: &S,
        ) -> Result<Array2<S>, EventHistoryError>;
        fn limit_face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError>;
    }

    impl<P: RoutedPrior> StrengthPrior for P {
        fn strengths(&self) -> usize {
            P::STRENGTHS
        }

        fn evaluate(&self, theta: &[f64], rho: &[f64]) -> Result<PriorCurvature, EventHistoryError> {
            Ok(self.route(theta, rho, &0.0)?.curvature())
        }

        fn rounding(&self, theta: &[f64], rho: &[f64]) -> Result<PriorRounding, EventHistoryError> {
            Ok(self.route(theta, rho, &Running::exact(0.0))?.errors())
        }

        fn curvature_derivative(
            &self,
            theta: &[f64],
            rho: &[f64],
            direction: &[f64],
            direction_error: &[f64],
        ) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
            // Each direction entry carries its error as its running bound.
            let carried: Vec<Running> = direction
                .iter()
                .zip(direction_error)
                .map(|(&value, &error)| Running {
                    value,
                    mu: error / f64::EPSILON,
                })
                .collect();
            let derivative = self.derivative(theta, rho, &carried, &Running::exact(0.0))?;
            Ok((derivative.map(|v| v.value), derivative.map(Running::rounding)))
        }

        fn face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError> {
            self.limit_face(j)
        }
    }

    /// θ_k ~ N(0, 1/λ) independently for every k in `block`, under one strength,
    /// and nothing on the other coefficients.
    fn gaussian_block<S: JetField>(
        theta: &[f64],
        rho: &[f64],
        block: &[usize],
        zero: &S,
    ) -> Result<PriorRoute<S>, EventHistoryError> {
        let width = theta.len();
        if rho.len() != 1 || block.iter().any(|&k| k >= width) {
            return Err(invalid(
                "a Gaussian block prior has one strength and coordinates inside its width",
            ));
        }
        let input = |v: f64| zero.with_value(v);
        let gaussian = exp(&input(rho[0]));
        let z = input(0.0);
        let mut gradient = vec![z.clone(); width];
        let mut curvature = Array2::from_elem((width, width), z.clone());
        let mut mixed = Array2::from_elem((1, width), z.clone());
        let mut quadratic = z.clone();
        let mut normalizer = z;
        for &k in block {
            let coefficient = input(theta[k]);
            let pull = gaussian.mul(&coefficient);
            quadratic = quadratic.add(&pull.mul(&coefficient).scale(0.5));
            normalizer = normalizer.add(&input(rho[0]).scale(0.5).sub(&half_log_tau(zero)));
            gradient[k] = pull.neg();
            mixed[[0, k]] = pull.neg();
            curvature[[k, k]] = gaussian.clone();
        }
        Ok(PriorRoute {
            log_density: normalizer.sub(&quadratic),
            gradient,
            strength_gradient: vec![input(block.len() as f64).scale(0.5).sub(&quadratic)],
            strength_second: vec![quadratic.neg()],
            negative_hessian: curvature.clone(),
            strength_curvature: vec![curvature],
            mixed,
        })
    }

    /// One strength over the coordinates in `block` of a `width`-coefficient model,
    /// whose zero-effect face pins the whole block.
    struct SharedBlockPrior {
        width: usize,
        block: Vec<usize>,
    }

    impl RoutedPrior for SharedBlockPrior {
        const STRENGTHS: usize = 1;

        fn route<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            zero: &S,
        ) -> Result<PriorRoute<S>, EventHistoryError> {
            if theta.len() != self.width {
                return Err(invalid("the shared block prior has its own width"));
            }
            gaussian_block(theta, rho, &self.block, zero)
        }

        fn derivative<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            direction: &[S],
            zero: &S,
        ) -> Result<Array2<S>, EventHistoryError> {
            // A Gaussian block has no third derivative.
            if theta.len() != self.width || rho.len() != 1 || direction.len() != self.width {
                return Err(invalid("shared block prior direction has invalid dimensions"));
            }
            Ok(Array2::from_elem((self.width, self.width), zero.with_value(0.0)))
        }

        fn limit_face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError> {
            if j != 0 {
                return Err(invalid("the shared block prior has one strength"));
            }
            Ok(Some(ZeroEffectFace {
                pinned: self.block.clone(),
                penalty: Array2::eye(self.block.len()),
                prior: Box::new(NoPrior {
                    width: self.width - self.block.len(),
                }),
            }))
        }
    }

    /// θ₀ ~ N(0, 1/λ₀); exp(θ₁) ~ Exponential(λ₁) in the log chart. The second
    /// prior has a coefficient-dependent curvature, so Ḣ_π is exercised.
    struct MixedPrior;

    impl RoutedPrior for MixedPrior {
        const STRENGTHS: usize = 2;

        fn route<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            zero: &S,
        ) -> Result<PriorRoute<S>, EventHistoryError> {
            let input = |v: f64| zero.with_value(v);
            let gaussian = exp(&input(rho[0]));
            let level = exp(&input(rho[1]).add(&input(theta[1])));
            let pull = gaussian.mul(&input(theta[0]));
            let quadratic = pull.mul(&input(theta[0])).scale(0.5);
            let one = input(1.0);
            let z = input(0.0);
            Ok(PriorRoute {
                log_density: input(rho[0])
                    .scale(0.5)
                    .sub(&half_log_tau(zero))
                    .sub(&quadratic)
                    .add(&input(rho[1]))
                    .add(&input(theta[1]))
                    .sub(&level),
                gradient: vec![pull.neg(), one.sub(&level)],
                strength_gradient: vec![input(0.5).sub(&quadratic), one.sub(&level)],
                strength_second: vec![quadratic.neg(), level.neg()],
                negative_hessian: ndarray::array![[gaussian.clone(), z.clone()], [z.clone(), level.clone()]],
                strength_curvature: vec![
                    ndarray::array![[gaussian, z.clone()], [z.clone(), z.clone()]],
                    ndarray::array![[z.clone(), z.clone()], [z.clone(), level.clone()]],
                ],
                mixed: ndarray::array![[pull.neg(), z.clone()], [z, level.neg()]],
            })
        }

        fn derivative<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            direction: &[S],
            zero: &S,
        ) -> Result<Array2<S>, EventHistoryError> {
            let level = exp(&zero.with_value(rho[1]).add(&zero.with_value(theta[1])));
            let z = zero.with_value(0.0);
            Ok(ndarray::array![[z.clone(), z.clone()], [z, level.mul(&direction[1])]])
        }

        fn limit_face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError> {
            // θ₀'s Gaussian block pins at zero. The level chart of θ₁ has no
            // zero-effect face.
            Ok((j == 0).then(|| ZeroEffectFace {
                pinned: vec![0],
                penalty: ndarray::array![[1.0]],
                prior: Box::new(LevelPrior),
            }))
        }
    }

    /// exp(θ) ~ Exponential(λ) in the log chart, over one coordinate: the prior
    /// MixedPrior keeps on its θ₀ face.
    struct LevelPrior;

    impl RoutedPrior for LevelPrior {
        const STRENGTHS: usize = 1;

        fn route<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            zero: &S,
        ) -> Result<PriorRoute<S>, EventHistoryError> {
            let input = |v: f64| zero.with_value(v);
            let chart = input(rho[0]).add(&input(theta[0]));
            let level = exp(&chart);
            let slack = input(1.0).sub(&level);
            Ok(PriorRoute {
                log_density: chart.sub(&level),
                gradient: vec![slack.clone()],
                strength_gradient: vec![slack],
                strength_second: vec![level.neg()],
                negative_hessian: ndarray::array![[level.clone()]],
                strength_curvature: vec![ndarray::array![[level.clone()]]],
                mixed: ndarray::array![[level.neg()]],
            })
        }

        fn derivative<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            direction: &[S],
            zero: &S,
        ) -> Result<Array2<S>, EventHistoryError> {
            let level = exp(&zero.with_value(rho[0]).add(&zero.with_value(theta[0])));
            Ok(ndarray::array![[level.mul(&direction[0])]])
        }

        fn limit_face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError> {
            if j != 0 {
                return Err(invalid("the level prior has one strength"));
            }
            Ok(None)
        }
    }

    /// No prior and no strengths over `width` coordinates: the face of a prior's
    /// last strength.
    struct NoPrior {
        width: usize,
    }

    impl RoutedPrior for NoPrior {
        const STRENGTHS: usize = 0;

        fn route<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            zero: &S,
        ) -> Result<PriorRoute<S>, EventHistoryError> {
            if theta.len() != self.width || !rho.is_empty() {
                return Err(invalid("an empty prior has no strengths and its own width"));
            }
            let z = zero.with_value(0.0);
            Ok(PriorRoute {
                log_density: z.clone(),
                gradient: vec![z.clone(); self.width],
                strength_gradient: vec![],
                strength_second: vec![],
                negative_hessian: Array2::from_elem((self.width, self.width), z.clone()),
                strength_curvature: vec![],
                mixed: Array2::from_elem((0, self.width), z),
            })
        }

        fn derivative<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            direction: &[S],
            zero: &S,
        ) -> Result<Array2<S>, EventHistoryError> {
            if theta.len() != self.width || !rho.is_empty() || direction.len() != self.width {
                return Err(invalid("an empty prior has no strengths and its own width"));
            }
            Ok(Array2::from_elem((self.width, self.width), zero.with_value(0.0)))
        }

        fn limit_face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError> {
            Err(invalid(format!("an empty prior has no strength {j}")))
        }
    }

    /// θ₀ ~ N(0, 1/λ₀) and nothing on θ₁.
    struct InterceptPrior;

    impl RoutedPrior for InterceptPrior {
        const STRENGTHS: usize = 1;

        fn route<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            zero: &S,
        ) -> Result<PriorRoute<S>, EventHistoryError> {
            gaussian_block(theta, rho, &[0], zero)
        }

        fn derivative<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            direction: &[S],
            zero: &S,
        ) -> Result<Array2<S>, EventHistoryError> {
            // A Gaussian block has no third derivative.
            if rho.len() != 1 || direction.len() != theta.len() {
                return Err(invalid("intercept prior direction has invalid dimensions"));
            }
            Ok(Array2::from_elem((theta.len(), theta.len()), zero.with_value(0.0)))
        }

        fn limit_face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError> {
            if j != 0 {
                return Err(invalid("the intercept prior has one strength"));
            }
            Ok(None)
        }
    }

    /// θ₁ ~ N(0, 1/λ) and nothing on θ₀.
    struct SlopePrior;

    impl RoutedPrior for SlopePrior {
        const STRENGTHS: usize = 1;

        fn route<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            zero: &S,
        ) -> Result<PriorRoute<S>, EventHistoryError> {
            gaussian_block(theta, rho, &[1], zero)
        }

        fn derivative<S: JetField>(
            &self,
            theta: &[f64],
            rho: &[f64],
            direction: &[S],
            zero: &S,
        ) -> Result<Array2<S>, EventHistoryError> {
            // A Gaussian block has no third derivative.
            if rho.len() != 1 || direction.len() != theta.len() {
                return Err(invalid("slope prior direction has invalid dimensions"));
            }
            Ok(Array2::from_elem((theta.len(), theta.len()), zero.with_value(0.0)))
        }

        fn limit_face(&self, j: usize) -> Result<Option<ZeroEffectFace<'_>>, EventHistoryError> {
            if j != 0 {
                return Err(invalid("the slope prior has one strength"));
            }
            Ok(Some(ZeroEffectFace {
                pinned: vec![1],
                penalty: ndarray::array![[1.0]],
                prior: Box::new(NoPrior { width: 1 }),
            }))
        }
    }

    /// A quadratic cohort with one shared strength over every coordinate but the
    /// first. Coordinate 0 is flat and observed at 0.5, the `quiet` coordinates
    /// observe 0 with precision 1000, and the last observes 10 with precision 1, so
    /// ḡ = 0 and the Laplace evidence is exact. With τ = 1/λ, the evidence relative
    /// to the limit is
    /// `D(τ) = −(quiet/2) ln(1 + 1000τ) + ½ [100τ/(1 + τ) − ln(1 + τ)]`.
    /// - `D′(0) = −500·quiet + 49.5 < 0`: the limit is a local optimum.
    /// - `D′(1) = −(500/1001)·quiet + 12.25 > 0` for quiet ≤ 24, so D also has an
    ///   interior maximum.
    /// - quiet = 1: D′ changes sign once more past τ = 1 and D(4) ≈ 35 > 0, so the
    ///   interior maximum is the larger.
    /// - quiet = 12: D′(5) ≈ 0.11 > 0 > D′(6) ≈ −0.05 and D ≈ −10.3 at that maximum,
    ///   so the limit is the larger.
    fn two_modes(quiet: usize) -> (Prescribed, SharedBlockPrior) {
        let width = quiet + 2;
        let mut curvature = Array2::zeros((width, width));
        curvature[[0, 0]] = 1.0;
        for k in 1..=quiet {
            curvature[[k, k]] = 1000.0;
        }
        curvature[[width - 1, width - 1]] = 1.0;
        let mut centre = vec![0.0; width];
        centre[0] = 0.5;
        centre[width - 1] = 10.0;
        (
            Prescribed {
                curvature,
                centre,
                log_determinant_gradient: vec![0.0; width],
                theta: vec![0.0; width],
            },
            SharedBlockPrior {
                width,
                block: (1..width).collect(),
            },
        )
    }

    /// The limit's certified fit and evidence on the shared block's face, and its
    /// boundary score there.
    fn limit_reading(
        system: &mut Prescribed,
        prior: &SharedBlockPrior,
    ) -> (CoefficientMode, LamlEvaluation, BoundaryScore) {
        let face = prior.face(0).unwrap().unwrap();
        let kept = vec![0];
        let (mode, evidence) = {
            let mut pinned = Pinned {
                system: &mut *system,
                kept: kept.clone(),
            };
            let mode = coefficient_mode(&mut pinned, &face.prior, &[0.0], &[]).unwrap();
            let evidence = laml(&mut pinned, &face.prior, &mode, &[]).unwrap();
            (mode, evidence)
        };
        let score = boundary_score(system, &face, &kept, &[], &mode).unwrap();
        (mode, evidence, score)
    }

    #[test]
    fn an_interior_optimum_above_a_local_optimum_limit_is_returned() {
        let (mut system, prior) = two_modes(1);
        let start = vec![0.0; system.coefficients()];
        let search = optimize_strengths(&mut system, &prior, &start, &[0.0]).unwrap();
        let (_mode, limit, score) = limit_reading(&mut system, &prior);
        let fit = interior(search);
        println!(
            "TWO_MODES quiet=1 boundary_score={} band={} interior={} band={} limit={} band={}",
            score.value, score.band, fit.evidence.log_evidence, fit.evidence.evidence_band, limit.log_evidence, limit.evidence_band
        );
        // Measurement precondition: the limit is a local optimum, so the value
        // comparison is what decided.
        assert!(score.value < -score.band, "the limit is not a local optimum here");
        let margin = fit.evidence.log_evidence - limit.log_evidence;
        assert!(
            margin > fit.evidence.evidence_band + limit.evidence_band,
            "interior evidence margin {margin} over the limit"
        );
    }

    #[test]
    fn a_local_optimum_limit_above_the_interior_optimum_is_returned() {
        let (mut system, prior) = two_modes(12);
        let start = vec![0.0; system.coefficients()];
        let StrengthSearch::Fitted(fit) = optimize_strengths(&mut system, &prior, &start, &[0.0]).unwrap() else {
            panic!("a limit resolvably above the interior optimum was reported as an evidence tie");
        };
        let Ascent::Resolved { evidence, .. } = ascend(&mut system, &prior, &start, &[0.0]).unwrap() else {
            panic!("the full model's ascent found no interior optimum; the control does not compare two modes");
        };
        println!(
            "TWO_MODES quiet=12 limits={:?} fit={} band={} interior={} band={}",
            fit.limits, fit.evidence.log_evidence, fit.evidence.evidence_band, evidence.log_evidence, evidence.evidence_band
        );
        assert_eq!(
            fit.limits.iter().map(|limit| limit.strength).collect::<Vec<_>>(),
            vec![0],
            "limits {:?}",
            fit.limits
        );
        let margin = fit.evidence.log_evidence - evidence.log_evidence;
        assert!(
            margin > fit.evidence.evidence_band + evidence.evidence_band,
            "limit evidence margin {margin} over the interior optimum"
        );
    }

    /// The search's fit when it is the full model's interior optimum.
    fn interior(search: StrengthSearch) -> FaceFit {
        match search {
            StrengthSearch::Fitted(fit) if fit.limits.is_empty() => fit,
            StrengthSearch::Fitted(fit) => {
                panic!("an interior optimum was reported at zero-effect limits: {:?}", fit.limits)
            }
            StrengthSearch::EvidenceTie { interior, .. } => panic!(
                "an interior optimum was reported as an evidence tie with its limit, at log evidence {} with band {}",
                interior.evidence.log_evidence, interior.evidence.evidence_band
            ),
        }
    }

    fn evidence(system: &mut PoissonIntercepts, rho: &[f64]) -> (CoefficientMode, LamlEvaluation) {
        let mode = coefficient_mode(system, &MixedPrior, &[0.0, 0.0], rho).unwrap();
        let value = laml(system, &MixedPrior, &mode, rho).unwrap();
        (mode, value)
    }

    /// The central difference of LAML in one strength, with its bar: the
    /// Richardson truncation estimate plus the evidence rounding over the step.
    fn central_difference(system: &mut PoissonIntercepts, rho: [f64; 2], j: usize) -> (f64, f64) {
        let step = f64::EPSILON.cbrt();
        let mut difference = |scale: f64| {
            let mut high = rho;
            let mut low = rho;
            high[j] += scale * step;
            low[j] -= scale * step;
            let (_, up) = evidence(system, &high);
            let (_, down) = evidence(system, &low);
            (
                (up.log_evidence - down.log_evidence) / (2.0 * scale * step),
                (up.evidence_band + down.evidence_band) / (2.0 * scale * step),
            )
        };
        let (fine, fine_band) = difference(1.0);
        let (coarse, _) = difference(2.0);
        (fine, (coarse - fine).abs() / 3.0 + fine_band)
    }

    #[test]
    fn arrow_assembly_equals_the_dense_joint_laplace_evidence() {
        let mut system = PoissonIntercepts::new(covariates());
        let rho = [0.3, -0.4];
        let (mode, value) = evidence(&mut system, &rho);
        // Dense 𝓗 over (θ₀, θ₁, z₁..z₆) at the joint mode, independently of the
        // Schur assembly.
        let n = system.x.len();
        use crate::test_support::{Bound, cholesky_forward_error, cholesky_log_det};
        // The dense route is assembled over Bound from the same weights, so its
        // oracles charge the assembly's running error.
        let mut dense = Array2::from_elem((2 + n, 2 + n), Bound::exact(0.0));
        for a in 0..2 {
            for b in 0..2 {
                dense[[a, b]] = Bound::exact(mode.prior.negative_hessian[[a, b]]);
            }
        }
        for i in 0..n {
            let w = Bound::exact(system.weight(i));
            let row = [Bound::exact(1.0), Bound::exact(system.x[i])];
            for a in 0..2 {
                for b in 0..2 {
                    dense[[a, b]] = dense[[a, b]].add(&w.mul(&row[a]).mul(&row[b]));
                }
                dense[[a, 2 + i]] = w.mul(&row[a]);
                dense[[2 + i, a]] = w.mul(&row[a]);
            }
            dense[[2 + i, 2 + i]] = w.add(&Bound::exact(1.0));
        }
        // The oracle log determinant is the Cholesky recursion's running error,
        // and this bar rests on the cited runtime-libm ln charge.
        let log_determinant = cholesky_log_det(&dense).unwrap();
        let expected = Bound::exact(
            system.log_density()
                + mode.prior.log_density
                + 0.5 * (2 + n) as f64 * (2.0 * std::f64::consts::PI).ln(),
        )
        .sub(&log_determinant.scale(0.5));
        // The production evidence reports its own band.
        let bar = value.evidence_band + expected.rounding();
        assert!(
            (value.log_evidence - expected.value).abs() <= bar,
            "{} vs {}, bar {bar}",
            value.log_evidence,
            expected.value
        );
        // The certificate held: the remaining Newton decrease is inside the
        // objective's rounding band.
        assert!(0.5 * mode.certificate.lambda_sq <= mode.certificate.band_f);
        // S⁻¹ is the coefficient marginal of the joint Gaussian: the θθ block of
        // the dense 𝓗⁻¹. The production solve's band is its theorem-form forward
        // error, and the oracle's columns carry their own bounds.
        let (covariance, certificate) = mode.certified_covariance().unwrap();
        let production_band = certificate.inverse_error();
        for a in 0..2 {
            let unit: Vec<Bound> = (0..2 + n).map(|k| Bound::exact(f64::from(k == a))).collect();
            let column = cholesky_forward_error(&dense, &unit).unwrap();
            for b in 0..2 {
                let bar = production_band[[b, a]] + column[b].rounding();
                assert!(
                    (covariance[[b, a]] - column[b].value).abs() <= bar,
                    "covariance [{b},{a}]: {} vs {}, bar {bar}",
                    covariance[[b, a]],
                    column[b].value
                );
            }
        }
    }

    #[test]
    fn strength_score_is_the_derivative_of_the_evidence_at_its_moving_mode() {
        let mut system = PoissonIntercepts::new(covariates());
        let rho = [0.3, -0.4];
        let (mode, value) = evidence(&mut system, &rho);
        let covariance = mode.covariance();
        let mut envelope_distinguished = false;
        for j in 0..2 {
            let (difference, bar) = central_difference(&mut system, rho, j);
            assert!(
                (value.gradient[j] - difference).abs() <= value.gradient_band[j] + bar,
                "strength {j}: {} vs {difference}, bands {} + {bar}",
                value.gradient[j],
                value.gradient_band[j]
            );
            // Without the log-determinant terms the score is wrong beyond the
            // same bar, so the check above is not passed by the envelope alone.
            let exact = Array2::zeros((2, 2));
            let envelope = mode.prior.strength_gradient[j]
                - 0.5
                    * contraction_with_error(
                        &covariance,
                        &exact,
                        &mode.prior.strength_curvature[j].t().to_owned(),
                        &exact,
                    )
                    .0;
            envelope_distinguished |= (envelope - difference).abs() > value.gradient_band[j] + bar;
        }
        assert!(envelope_distinguished);
    }

    #[test]
    fn strength_optimum_is_certified_stationary() {
        let mut system = PoissonIntercepts::new(covariates());
        let optimum = interior(optimize_strengths(&mut system, &MixedPrior, &[0.0, 0.0], &[0.0, 0.0]).unwrap());
        assert!(resolved(&optimum.evidence));
        assert!(optimum.iterations > 0);
        assert!(0.5 * optimum.mode.certificate.lambda_sq <= optimum.mode.certificate.band_f);
        let rho = [optimum.log_strengths[0], optimum.log_strengths[1]];
        for j in 0..2 {
            let (difference, bar) = central_difference(&mut system, rho, j);
            assert!(
                difference.abs() <= optimum.evidence.gradient_band[j] + bar,
                "strength {j}: {difference}, bands {} + {bar}",
                optimum.evidence.gradient_band[j]
            );
        }
    }

    #[test]
    fn an_informed_intercept_is_an_interior_optimum_above_its_limit_evidence() {
        // Counts near e² per unit exposure inform the intercept. Independently of
        // the boundary score, the evidence at the returned optimum exceeds the
        // intercept face's evidence at the same other strength by more than both
        // evaluations' bands, and the face's boundary score is resolvably positive.
        let mut system = PoissonIntercepts::informed(covariates());
        let optimum = interior(optimize_strengths(&mut system, &MixedPrior, &[0.0, 0.0], &[0.0, 0.0]).unwrap());
        assert!(resolved(&optimum.evidence));
        let face = MixedPrior.face(0).unwrap().unwrap();
        let kept = vec![1];
        let others = [optimum.log_strengths[1]];
        let (face_mode, face_evidence) = {
            let mut pinned = Pinned {
                system: &mut system,
                kept: kept.clone(),
            };
            let mode =
                coefficient_mode(&mut pinned, &face.prior, &[optimum.mode.coefficients[1]], &others).unwrap();
            let evidence = laml(&mut pinned, &face.prior, &mode, &others).unwrap();
            (mode, evidence)
        };
        let margin = optimum.evidence.log_evidence - face_evidence.log_evidence;
        let bar = optimum.evidence.evidence_band + face_evidence.evidence_band;
        assert!(margin > bar, "evidence margin {margin} over the intercept face, bar {bar}");
        let score = boundary_score(&mut system, &face, &kept, &others, &face_mode).unwrap();
        assert!(
            score.value > score.band,
            "boundary score {} with band {}",
            score.value,
            score.band
        );
    }

    #[test]
    fn a_strength_whose_evidence_never_turns_back_is_the_zero_effect_limit() {
        // The mirrored cohort makes the profile even in θ₁, so θ̂₁ = 0, S has no
        // θ₀θ₁ coupling and the trace and log-determinant terms along δθ vanish:
        // the slope strength's score is ½ I/(I + λ) > 0 for every λ. The evidence
        // rises toward the constant-slope model without end, and no finite
        // strength is its optimum.
        let mut system = PoissonIntercepts::symmetric();
        let search = optimize_strengths(&mut system, &SlopePrior, &[0.0, 0.0], &[0.0]).unwrap();
        let StrengthSearch::Fitted(FaceFit { limits, .. }) = search else {
            panic!("a strength whose score never changes sign was reported as an evidence tie");
        };
        assert_eq!(limits.len(), 1);
        assert_eq!(limits[0].strength, 0);
        assert!(
            matches!(limits[0].status, LimitStatus::Resolved { .. }),
            "the mirrored slope's limit is not resolved: {:?}",
            limits[0].status
        );
        // At a finite strength the same evidence is resolved and its score is
        // resolvably positive: the limit is not a failure to evaluate it.
        let mode = coefficient_mode(&mut system, &SlopePrior, &[0.0, 0.0], &[0.0]).unwrap();
        let value = laml(&mut system, &SlopePrior, &mode, &[0.0]).unwrap();
        assert!(value.gradient[0] > value.gradient_band[0]);
    }

    #[test]
    fn a_strength_without_information_is_a_first_order_tie() {
        // Every covariate is zero, so the data neither pull the slope nor curve
        // it. On its face g₁ = 0, H̃₁₁ = 0 and ḡ₁ = 0 exactly, so the boundary score
        // is zero and the limit is taken as a first-order tie, reported as such.
        let mut system = PoissonIntercepts::new(vec![0.0; 6]);
        let search = optimize_strengths(&mut system, &SlopePrior, &[0.0, 0.0], &[0.0]).unwrap();
        let StrengthSearch::Fitted(FaceFit { limits, .. }) = search else {
            panic!("a strength without information was reported as an evidence tie");
        };
        assert_eq!(limits.len(), 1);
        assert_eq!(limits[0].strength, 0);
        let LimitStatus::FirstOrderTie { score, bound } = limits[0].status else {
            panic!(
                "a strength without information is not a first-order tie: {:?}",
                limits[0].status
            );
        };
        assert!(score.abs() <= bound, "score {score}, bound {bound}");
    }

    /// F(θ) = −½ (θ − c)ᵀ A (θ − c), with no latent coordinates and a prescribed
    /// log-determinant gradient. The classification reads g, S₀ and ḡ and nothing
    /// else, so they can be set to make the variance-component score and the full
    /// score disagree.
    struct Prescribed {
        curvature: Array2<f64>,
        centre: Vec<f64>,
        log_determinant_gradient: Vec<f64>,
        /// The latest profile point.
        theta: Vec<f64>,
    }

    impl Prescribed {
        fn route<S: JetField>(&self, zero: &S) -> (S, Vec<S>) {
            let input = |v: f64| zero.with_value(v);
            let offset: Vec<S> = self
                .theta
                .iter()
                .zip(&self.centre)
                .map(|(&t, &c)| input(t).sub(&input(c)))
                .collect();
            let pulled: Vec<S> = self
                .curvature
                .rows()
                .into_iter()
                .map(|row| {
                    row.iter()
                        .zip(&offset)
                        .fold(input(0.0), |total, (&a, o)| total.add(&input(a).mul(o)))
                })
                .collect();
            let log_density = offset
                .iter()
                .zip(&pulled)
                .fold(input(0.0), |total, (o, q)| total.add(&o.mul(q)))
                .scale(-0.5);
            (log_density, pulled.iter().map(JetField::neg).collect())
        }
    }

    impl LaplaceSystem for Prescribed {
        fn coefficients(&self) -> usize {
            self.centre.len()
        }

        fn profile(&mut self, theta: &[f64]) -> Result<ProfileValue, EventHistoryError> {
            if theta.len() != self.centre.len() {
                return Err(invalid("the prescribed system has its own width"));
            }
            self.theta = theta.to_vec();
            let (log_density, gradient) = self.route(&0.0);
            Ok(ProfileValue {
                log_density,
                gradient,
            })
        }

        fn profile_rounding(&mut self) -> Result<ProfileRounding, EventHistoryError> {
            let (log_density, gradient) = self.route(&Running::exact(0.0));
            Ok(ProfileRounding {
                log_density: log_density.rounding(),
                gradient: gradient.iter().map(Running::rounding).collect(),
            })
        }

        fn curvature(&mut self) -> Result<LaplaceCurvature, EventHistoryError> {
            // The prescribed blocks are exact inputs.
            Ok(LaplaceCurvature {
                schur: self.curvature.clone(),
                schur_error: Array2::zeros(self.curvature.dim()),
                latent_log_determinant: 0.0,
                latent_log_determinant_error: 0.0,
                latent_dimension: 0,
            })
        }

        fn log_determinant_gradient(
            &mut self,
            covariance: &Array2<f64>,
            covariance_error: &Array2<f64>,
        ) -> Result<LogDeterminantGradient, EventHistoryError> {
            let dim = self.curvature.dim();
            if covariance.dim() != dim || covariance_error.dim() != dim {
                return Err(invalid("the prescribed system has its own width"));
            }
            // The prescribed gradient is an exact input.
            Ok(LogDeterminantGradient {
                value: self.log_determinant_gradient.clone(),
                error: vec![0.0; self.log_determinant_gradient.len()],
            })
        }
    }

    #[test]
    fn the_decision_follows_the_full_boundary_score_where_the_variance_component_disagrees() {
        // A = [[2, 1], [1, 1]] gives H̃₁₁ = 1 − 1/2 = 1/2. With c = (0.3, 1.6) the face
        // mode is θ₀ = 1.1 and leaves g₁ = c₁ H̃₁₁ = 0.8, so the variance-component
        // score is ½(0.64 − 0.5) = 0.07 > 0. The log-determinant gradient
        // ḡ = (0, 0.5) moves ḡ̃₁ g₁ = 0.4, and the full score is ½(0.64 − 0.5 − 0.4) =
        // −0.13 < 0: the limit.
        let mut system = Prescribed {
            curvature: ndarray::array![[2.0, 1.0], [1.0, 1.0]],
            centre: vec![0.3, 1.6],
            log_determinant_gradient: vec![0.0, 0.5],
            theta: vec![0.0; 2],
        };
        let face = SlopePrior.face(0).unwrap().unwrap();
        let kept = vec![0];
        let mode = {
            let mut pinned = Pinned {
                system: &mut system,
                kept: kept.clone(),
            };
            coefficient_mode(&mut pinned, &face.prior, &[0.0], &[]).unwrap()
        };
        let score = boundary_score(&mut system, &face, &kept, &[], &mode).unwrap();
        assert!(
            score.variance_component > score.band && score.value < -score.band,
            "the variance-component score {} and the full score {} must disagree beyond the band {}",
            score.variance_component,
            score.value,
            score.band
        );
        assert!(
            matches!(limit_status(&score), Some(LimitStatus::Resolved { .. })),
            "variance-component score {}, full score {}",
            score.variance_component,
            score.value
        );
    }

    #[test]
    fn boundary_score_is_the_inverse_strength_derivative_of_the_evidence_at_the_limit() {
        // Along τ = 1/λ = e^−ρ, ∂LAML/∂τ = −e^ρ ∂LAML/∂ρ. With d(τ) = s + cτ + O(τ²),
        // 2d(τ) − d(2τ) = s + O(τ²), and |d(τ) − d(2τ)| = cτ is the truncation
        // estimate beside the two score bands.
        let mut system = PoissonIntercepts::new(covariates());
        let face = SlopePrior.face(0).unwrap().unwrap();
        let kept = vec![0];
        let mode = {
            let mut pinned = Pinned {
                system: &mut system,
                kept: kept.clone(),
            };
            coefficient_mode(&mut pinned, &face.prior, &[0.0], &[]).unwrap()
        };
        let score = boundary_score(&mut system, &face, &kept, &[], &mode).unwrap();
        let mut derivative = |tau: f64| {
            let rho = [-tau.ln()];
            let mode = coefficient_mode(&mut system, &SlopePrior, &[0.0, 0.0], &rho).unwrap();
            let evidence = laml(&mut system, &SlopePrior, &mode, &rho).unwrap();
            (-evidence.gradient[0] / tau, evidence.gradient_band[0] / tau)
        };
        let tau = f64::EPSILON.cbrt();
        let (fine, fine_band) = derivative(tau);
        let (coarse, coarse_band) = derivative(2.0 * tau);
        let extrapolated = 2.0 * fine - coarse;
        let bar = (fine - coarse).abs() + 2.0 * fine_band + coarse_band + score.band;
        assert!(
            (extrapolated - score.value).abs() <= bar,
            "boundary score {} vs extrapolated derivative {extrapolated}, bar {bar}",
            score.value
        );
        assert!(
            score.value.abs() > bar,
            "the boundary score {} is not above its bar {bar}",
            score.value
        );
    }

    #[test]
    fn an_unidentified_coefficient_is_refused_not_reported() {
        // Every covariate is zero and θ₁ has no prior: the profile objective is
        // flat along θ₁, so there is no Laplace law to report. The same system
        // with the covariates is the positive control above.
        let mut system = PoissonIntercepts::new(vec![0.0; 6]);
        let error = coefficient_mode(&mut system, &InterceptPrior, &[0.0, 0.0], &[0.2])
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("not positive definite") || error.contains("weakly identified"),
            "{error}"
        );
        assert!(optimize_strengths(&mut system, &InterceptPrior, &[0.0, 0.0], &[0.2]).is_err());
    }
}
