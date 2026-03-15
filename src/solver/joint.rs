//!
//! This module implements a unified model where the base linear predictor and
//! the flexible link correction are fitted jointly in one REML optimization.
//!
//! Architecture:
//!   η = g(Xβ) where g(u) = u + wiggle(u)
//!
//! Where:
//! - Xβ: High-dimensional predictors (high-dimensional predictors) with ridge penalty
//! - g(·): Flexible 1D link correction with a constrained spline wiggle
//!   whose intercept and linear components are projected out
//!   (g(u) = u + B(u)θ in the constrained basis)
//!
//! The algorithm:
//! - Outer: gradient-only trust-region optimization over ρ = [log(λ_base), log(λ_link)]
//! - Inner: Alternating (g|β, β|g with g'(u)*X design)
//! - LAML cost computed via logdet of joint Gauss-Newton Hessian

use crate::basis::{
    BasisOptions, Dense, KnotSource, apply_linear_extension_from_first_derivative,
    baseline_lambda_seed, compute_geometric_constraint_transform, create_basis,
};
use crate::construction::{
    EngineDims, ReparamInvariant, compute_penalty_square_roots, precompute_reparam_invariant,
    stable_reparameterization_engine, stable_reparameterizationwith_invariant_engine,
};
use crate::estimate::EstimationError;
use crate::faer_ndarray::{
    FaerEigh, fast_ata, fast_atb, fast_atv_into, fast_xt_diag_x, fast_xt_diag_y,
};
use crate::families::strategy::{FamilyStrategy, strategy_for_family};
use crate::probability::normal_cdf;
use crate::quadrature::QuadratureContext;
use crate::seeding::{SeedConfig, SeedRiskProfile};
use crate::solver::strategy::{
    ClosureObjective, Derivative, HessianResult, OuterCapability, OuterConfig, OuterEval,
};
use crate::types::{GlmLikelihoodFamily, InverseLink, LikelihoodFamily, LinkFunction, SasLinkState};
use crate::visualizer;
use faer::Side;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// NOTE on z standardization:
// We standardize u = Xβ into z_raw = (u - min_u) / (max_u - min_u).
// To avoid a C^1 discontinuity from hard clamping, we evaluate the spline basis at
// z_c = clamp(z_raw, 0, 1) and apply a first-order extension outside the knot range:
//   B_ext(z_raw) = B(z_c) + (z_raw - z_c) * B'(z_c).
// This keeps the link correction g(u) differentiable w.r.t. u (and hence β) everywhere.

/// Fixed stabilization ridge for solver numerical stability.
/// This is a SOLVER detail, not an objective function modification.
/// With spectral log-det (Wood 2011), the objective is smooth regardless of ridge.
const FIXED_STABILIZATION_RIDGE: f64 = 1e-8;
// FD audit constants removed — gradient now computed through unified evaluator.

#[inline]
fn integrated_binomial_family_from_link(
    link: LinkFunction,
    has_sas_state: bool,
) -> Option<GlmLikelihoodFamily> {
    match link {
        LinkFunction::Logit => Some(GlmLikelihoodFamily::BinomialLogit),
        LinkFunction::Probit => Some(GlmLikelihoodFamily::BinomialProbit),
        LinkFunction::CLogLog => Some(GlmLikelihoodFamily::BinomialCLogLog),
        LinkFunction::Log => None,
        LinkFunction::Sas if has_sas_state => Some(GlmLikelihoodFamily::BinomialSas),
        LinkFunction::BetaLogistic if has_sas_state => {
            Some(GlmLikelihoodFamily::BinomialBetaLogistic)
        }
        LinkFunction::Sas | LinkFunction::BetaLogistic => None,
        LinkFunction::Identity => None,
    }
}

#[inline]
fn joint_prediction_supported(link: LinkFunction) -> Result<(), EstimationError> {
    if matches!(link, LinkFunction::Sas | LinkFunction::BetaLogistic) {
        return Err(EstimationError::InvalidSpecification(
            "predict_joint does not support state-less SAS/Beta-Logistic links; explicit fitted link state is required"
                .to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn joint_point_inverse_link(link: LinkFunction, eta: f64) -> f64 {
    match link {
        LinkFunction::Identity => eta,
        LinkFunction::Log => eta.clamp(-700.0, 700.0).exp(),
        LinkFunction::Logit => {
            let e = eta.clamp(-700.0, 700.0);
            1.0 / (1.0 + (-e).exp())
        }
        LinkFunction::Probit => normal_cdf(eta),
        LinkFunction::CLogLog => {
            let e = eta.clamp(-30.0, 30.0);
            1.0 - (-e.exp()).exp()
        }
        LinkFunction::Sas | LinkFunction::BetaLogistic => unreachable!(
            "joint_point_inverse_link called for unsupported state-less SAS/Beta-Logistic link"
        ),
    }
}

#[inline]
fn seed_risk_profile_for_joint_link(link: LinkFunction) -> SeedRiskProfile {
    match link {
        LinkFunction::Identity => SeedRiskProfile::Gaussian,
        LinkFunction::Log | LinkFunction::Logit | LinkFunction::Probit | LinkFunction::CLogLog => {
            SeedRiskProfile::GeneralizedLinear
        }
        LinkFunction::Sas | LinkFunction::BetaLogistic => SeedRiskProfile::GeneralizedLinear,
    }
}

#[derive(Clone, Copy)]
struct JointPenaltyLayout {
    n_base: usize,
}

impl JointPenaltyLayout {
    fn for_state(state: &JointModelState<'_>) -> Self {
        Self {
            n_base: state.s_base.len(),
        }
    }

    fn total_penalties(self) -> usize {
        self.n_base + 1
    }

    fn validate_rho(self, rho: &Array1<f64>) -> Result<(), EstimationError> {
        let expected = self.total_penalties();
        if rho.len() != expected {
            return Err(EstimationError::LayoutError(format!(
                "rho length does not match joint penalty count: got {}, expected {}",
                rho.len(),
                expected
            )));
        }
        Ok(())
    }

    fn lambdas(self, rho: &Array1<f64>) -> (Array1<f64>, f64) {
        let lambda_base = rho.slice(s![..self.n_base]).mapv(f64::exp);
        let lambda_link = rho[self.n_base].exp();
        (lambda_base, lambda_link)
    }
}

#[derive(Clone, Copy)]
struct JointCoefLayout {
    p_base: usize,
    p_link: usize,
}

impl JointCoefLayout {
    fn new(p_base: usize, p_link: usize) -> Self {
        Self { p_base, p_link }
    }

    fn p_total(self) -> usize {
        self.p_base + self.p_link
    }
}

/// Ensure a matrix is positive definite by adding ridge if needed for solver stability.
/// This is a SOLVER detail - the ridge is only for numerical stability in linear solves,
/// not for defining the objective function. The objective uses spectral log-det which
/// handles near-singular matrices smoothly.
/// Returns the ridge value added (0.0 if matrix was already positive definite).
fn ensure_positive_definite_joint(mat: &mut Array2<f64>) -> f64 {
    use crate::faer_ndarray::FaerCholesky;
    use faer::Side;

    if mat.cholesky(Side::Lower).is_ok() {
        return 0.0; // Already positive definite, no stabilization needed
    }

    // Add diagonal stabilization for solver stability.
    //
    // IMPORTANT (math wiring): whenever the inner solver uses (A + δI) rather than A,
    // δ must be tracked and included in the outer cost as:
    //   0.5 * δ * ||β||²     and     log|S_λ + δI|_+
    // otherwise the Envelope Theorem (∂V/∂β = 0 at β̂) does not apply.
    //
    // We keep δ fixed and rho-independent to avoid introducing kinks into the
    // profiled objective when the penalty matrix changes with smoothing parameters.
    for i in 0..mat.nrows() {
        mat[[i, i]] += FIXED_STABILIZATION_RIDGE;
    }
    if mat.cholesky(Side::Lower).is_ok() {
        return FIXED_STABILIZATION_RIDGE;
    }
    eprintln!(
        "[JOINT] Warning: matrix remained non-SPD after fixed ridge (delta={:.1e}).",
        FIXED_STABILIZATION_RIDGE
    );
    FIXED_STABILIZATION_RIDGE
}

pub struct JointModelState<'a> {
    /// Response variable
    y: ArrayView1<'a, f64>,
    /// Prior weights
    weights: ArrayView1<'a, f64>,
    /// Base design matrix (X for high-dim predictors)
    x_base: ArrayView2<'a, f64>,
    /// Current base coefficients β
    beta_base: Array1<f64>,
    /// Current link wiggle coefficients θ (identity is implicit offset)
    beta_link: Array1<f64>,
    /// Penalty matrices for base block (one per λ)
    s_base: Vec<Array2<f64>>,
    /// Transformed penalty for link block (Z'SZ) - None until build_link_basis is called
    s_link_constrained: Option<Array2<f64>>,
    /// Constraint transform Z (basis → constrained basis) - None until build_link_basis is called
    link_transform: Option<Array2<f64>>,
    /// Geometric constraint transform computed from Greville abscissae (constant w.r.t. β).
    /// This is computed once when knots are initialized and reused for all subsequent
    /// basis evaluations, ensuring dZ/dβ = 0 exactly for correct analytic gradients.
    geometric_link_transform: Option<Array2<f64>>,
    /// Pre-computed projected penalty using geometric transform (Z'SZ).
    geometric_s_link_constrained: Option<Array2<f64>>,
    /// Current log-smoothing parameters (one per base penalty + one for link)
    rho: Array1<f64>,
    /// Link function (Logit or Identity)
    link: LinkFunction,
    /// Layout for base model
    layout_base: EngineDims,
    /// Number of internal knots for link spline
    n_link_knots: usize,
    /// B-spline degree (fixed at 3 = cubic)
    degree: usize,
    /// Fixed knot range from training data (min, max)
    knot_range: Option<(f64, f64)>,
    /// Knot vector for B-splines (fixed after first build)
    knot_vector: Option<Array1<f64>>,
    /// Number of constrained basis functions
    n_constrained_basis: usize,
    /// Optional per-observation SE for integrated (GHQ) likelihood.
    /// When present, uses integrated family-dispatched working updates.
    covariate_se: Option<Array1<f64>>,
    quadctx: QuadratureContext,
    /// Enable Firth bias reduction for separation protection
    firth_bias_reduction: bool,
    /// Last full linear predictor (u + wiggle) for weight-aligned constraints.
    last_eta: Option<Array1<f64>>,
    /// Ridge stabilization used in the most recent IRLS solve.
    /// This must be tracked to include 0.5*δ||β||² in the cost function,
    /// ensuring the Envelope Theorem holds for the analytic gradient.
    ridge_used: f64,
    /// Ridge stabilization used for the base block solve.
    ridge_base_used: f64,
    /// Ridge stabilization used for the link block solve.
    ridge_link_used: f64,
    /// Fitted SAS/BetaLogistic link state for integrated PIRLS.
    sas_link_state: Option<SasLinkState>,
}

struct JointFirthAdjustment {
    hat_diag: Array1<f64>,
    half_log_det: f64,
    ridge_used: f64,
}

/// Configuration for joint model fitting
#[derive(Clone)]
pub struct JointModelConfig {
    /// Maximum backfitting iterations
    pub max_backfit_iter: usize,
    /// Convergence tolerance for backfitting
    pub backfit_tol: f64,
    /// Maximum REML iterations per backfit cycle
    pub max_reml_iter: usize,
    /// REML convergence tolerance
    pub reml_tol: f64,
    /// Number of internal knots for link spline
    pub n_link_knots: usize,
    /// Enable Firth bias reduction (protects against separation in logistic regression)
    pub firth_bias_reduction: bool,
}

/// Engine-facing geometric configuration for the 1D link spline.
#[derive(Clone, Copy, Debug)]
pub struct JointLinkGeometry {
    pub n_link_knots: usize,
    pub degree: usize,
}

impl Default for JointModelConfig {
    fn default() -> Self {
        Self {
            max_backfit_iter: 20,
            backfit_tol: 1e-4,
            max_reml_iter: 50,
            reml_tol: 1e-6,
            n_link_knots: 10,
            firth_bias_reduction: false, // Off by default, enable for rare-event data
        }
    }
}

/// Result of joint model fitting - stores everything needed for prediction
pub struct JointModelResult {
    /// Fitted base coefficients β
    pub beta_base: Array1<f64>,
    /// Fitted link wiggle coefficients θ
    pub beta_link: Array1<f64>,
    /// Fitted smoothing parameters (one per penalty)
    pub lambdas: Vec<f64>,
    /// Final deviance
    pub deviance: f64,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Number of backfitting iterations
    pub backfit_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Final measured outer gradient norm.
    pub outer_gradient_norm: f64,
    /// Stored knot range for prediction (min, max)
    pub knot_range: (f64, f64),
    /// Stored knot vector for prediction
    pub knot_vector: Array1<f64>,
    /// Constraint transform for prediction
    pub link_transform: Array2<f64>,
    /// B-spline degree
    pub degree: usize,
    /// Link function used during training
    pub link: LinkFunction,
    /// Constrained link penalty matrix (Z'SZ) used in REML fit
    pub s_link_constrained: Array2<f64>,
    /// Ridge stabilization used in the final IRLS solve.
    /// Included in the cost function as 0.5*δ||β||² to satisfy the Envelope Theorem.
    pub ridge_used: f64,
    /// Posterior covariance for base coefficients: (X'W_eff X + S_λ)^{-1}
    pub beta_base_covariance: Option<Array2<f64>>,
}

impl<'a> JointModelState<'a> {
    /// Create new joint model state
    pub(crate) fn new(
        y: ArrayView1<'a, f64>,
        weights: ArrayView1<'a, f64>,
        x_base: ArrayView2<'a, f64>,
        s_base: Vec<Array2<f64>>,
        layout_base: EngineDims,
        link: LinkFunction,
        config: &JointModelConfig,
        quadctx: QuadratureContext,
    ) -> Self {
        let n_base = x_base.ncols();
        let degree = 3; // Cubic B-splines

        // Number of B-spline basis functions = n_internal_knots + degree + 1
        // After orthogonality constraint (remove 2: intercept + linear): -2
        // So: n_constrained = n_internal_knots + degree + 1 - 2 = k + degree - 1
        let n_raw_basis = config.n_link_knots + degree + 1;
        let n_constrained = n_raw_basis.saturating_sub(2);

        // Initialize β to zero, link coefficients to zero (identity is implicit offset)
        let beta_base = Array1::zeros(n_base);
        let beta_link = Array1::zeros(n_constrained);

        // Initialize rho (log-lambdas) - one per base penalty + one for link
        let n_penalties = s_base.len() + 1;
        let rho = Array1::zeros(n_penalties);

        // link_transform and s_link_constrained are None until build_link_basis is called
        // geometric_* fields are computed once when knots are initialized
        Self {
            y,
            weights,
            x_base,
            beta_base,
            beta_link,
            s_base,
            s_link_constrained: None,
            link_transform: None,
            geometric_link_transform: None,
            geometric_s_link_constrained: None,
            rho,
            link,
            layout_base,
            n_link_knots: config.n_link_knots,
            degree,
            knot_range: None,
            knot_vector: None,
            n_constrained_basis: n_constrained,
            covariate_se: None,
            quadctx,
            firth_bias_reduction: config.firth_bias_reduction,
            last_eta: None,
            ridge_used: 0.0,
            ridge_base_used: 0.0,
            ridge_link_used: 0.0,
            sas_link_state: None,
        }
    }

    /// Set per-observation SE for integrated (GHQ) likelihood.
    /// When set, the joint model uses uncertainty-aware IRLS updates.
    pub fn with_covariate_se(mut self, se: Array1<f64>) -> Self {
        self.covariate_se = Some(se);
        self
    }

    /// Set fitted SAS/BetaLogistic link state for integrated PIRLS.
    pub fn with_sas_link_state(mut self, state: SasLinkState) -> Self {
        self.sas_link_state = Some(state);
        self
    }

    /// Set rho (log-lambdas) for REML optimization
    pub fn set_rho(&mut self, rho: Array1<f64>) {
        self.rho = rho;
    }

    /// Compute the current linear predictor Xβ
    pub fn base_linear_predictor(&self) -> Array1<f64> {
        self.x_base.dot(&self.beta_base)
    }

    /// Get number of observations
    pub fn nobs(&self) -> usize {
        self.y.len()
    }

    /// Get total weight sum
    pub fn totalweight(&self) -> f64 {
        self.weights.sum()
    }

    /// Get link function
    pub fn link(&self) -> LinkFunction {
        self.link.clone()
    }

    /// Initialize the geometric constraint transform from the knot vector.
    ///
    /// This computes Z and S_c using Greville abscissae, which depend only on
    /// the knot geometry and not on β. This ensures dZ/dβ = 0 exactly, making
    /// the analytic gradient correct.
    ///
    /// Should be called once after knots are determined (first build_link_basis call).
    fn initialize_geometric_constraint(&mut self) -> Result<(), String> {
        let knot_vector = self.knot_vector.as_ref().ok_or_else(|| {
            "Cannot initialize geometric constraint: knot_vector not set".to_string()
        })?;

        let (z, s_constrained) =
            compute_geometric_constraint_transform(knot_vector, self.degree, 2)
                .map_err(|e| format!("Geometric constraint computation failed: {e}"))?;

        let n_constrained = z.ncols();

        self.geometric_link_transform = Some(z);
        self.geometric_s_link_constrained = Some(s_constrained);

        // Update dimension tracking and resize beta_link if needed
        if self.n_constrained_basis != n_constrained {
            self.n_constrained_basis = n_constrained;
            self.beta_link = Array1::zeros(n_constrained);
        }

        Ok(())
    }

    /// Build link spline basis at current Xβ values
    /// Returns ONLY the constrained wiggle basis (identity u is treated as offset)
    /// Also updates internal state with transform and projected penalty
    ///
    /// The orthogonality constraint uses Greville abscissae (geometric constraints)
    /// computed from the knot vector. This ensures Z is constant w.r.t. β,
    /// making dZ/dβ = 0 exactly and enabling correct analytic gradients.
    ///
    /// Returns Err if geometric constraint computation fails.
    pub fn build_link_basis(&mut self, eta_base: &Array1<f64>) -> Result<Array2<f64>, String> {
        let k = self.n_link_knots;
        let degree = self.degree;

        // Freeze knot range after first initialization to keep the objective stable.
        if self.knot_range.is_none() {
            let minval = eta_base.iter().cloned().fold(f64::INFINITY, f64::min);
            let maxval = eta_base.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let rangewidth = maxval - minval;
            let range = if rangewidth > 1e-6 {
                (minval, maxval)
            } else {
                let center = 0.5 * (minval + maxval);
                let pad = 1.0_f64.max(center.abs() * 1e-3);
                (center - pad, center + pad)
            };
            self.knot_range = Some(range);
        }

        // Standardize: z_raw = (u - min)/(max - min), z_c = clamp(z_raw, 0, 1).
        let (z_raw, z_c, _) = self.standardized_z(eta_base);

        // Build B-spline basis on z ∈ [0, 1]
        let data_range = (0.0, 1.0);
        let basis_result = if let Some(knots) = self.knot_vector.as_ref() {
            create_basis::<Dense>(
                z_c.view(),
                KnotSource::Provided(knots.view()),
                degree,
                BasisOptions::value(),
            )
            .map(|(basis, _)| (basis, knots.clone()))
        } else {
            create_basis::<Dense>(
                z_c.view(),
                KnotSource::Generate {
                    data_range,
                    num_internal_knots: k,
                },
                degree,
                BasisOptions::value(),
            )
        };

        match basis_result {
            Ok((bspline_basis, knots)) => {
                let mut bspline_basis = bspline_basis.as_ref().clone();
                apply_linear_extension_from_first_derivative(
                    z_raw.view(),
                    z_c.view(),
                    knots.view(),
                    degree,
                    &mut bspline_basis,
                )
                .map_err(|e| format!("B-spline extension failed: {e}"))?;

                // Store knot vector and initialize geometric constraint if first call
                let first_init = self.knot_vector.is_none();
                if first_init {
                    self.knot_vector = Some(knots);
                    // Initialize geometric constraint transform (computed ONCE from knot geometry)
                    self.initialize_geometric_constraint()?;
                }

                // Use pre-computed geometric constraint transform
                let transform = self
                    .geometric_link_transform
                    .as_ref()
                    .ok_or_else(|| "Geometric transform not initialized".to_string())?;

                // Verify dimensions match
                if transform.nrows() != bspline_basis.ncols() {
                    return Err(format!(
                        "Transform dimension mismatch: transform has {} rows but basis has {} cols",
                        transform.nrows(),
                        bspline_basis.ncols()
                    ));
                }

                // Apply transform: B_constrained = B_raw * Z
                let constrained_basis = bspline_basis.dot(transform);
                let n_constrained = constrained_basis.ncols();

                // Copy geometric transform to the standard fields for parity
                self.link_transform = self.geometric_link_transform.clone();
                self.s_link_constrained = self.geometric_s_link_constrained.clone();
                self.n_constrained_basis = n_constrained;

                if self.beta_link.len() != n_constrained {
                    self.beta_link = Array1::zeros(n_constrained);
                }

                Ok(constrained_basis)
            }
            Err(e) => {
                // B-spline basis construction failed - return error
                Err(format!("B-spline basis construction failed: {}", e))
            }
        }
    }

    /// Build constrained link basis using stored knots without mutating state.
    /// Uses the pre-computed geometric transform (constant w.r.t. β).
    pub fn build_link_basis_from_state(&self, eta_base: &Array1<f64>) -> Array2<f64> {
        let n = eta_base.len();
        let Some(knot_vector) = self.knot_vector.as_ref() else {
            return Array2::zeros((n, 0));
        };
        let (z_raw, z_c, _) = self.standardized_z(eta_base);

        let b_raw = match create_basis::<Dense>(
            z_c.view(),
            KnotSource::Provided(knot_vector.view()),
            self.degree,
            BasisOptions::value(),
        ) {
            Ok((basis, _)) => basis.as_ref().clone(),
            Err(_) => return Array2::zeros((n, 0)),
        };

        let mut b_raw = b_raw;
        if apply_linear_extension_from_first_derivative(
            z_raw.view(),
            z_c.view(),
            knot_vector.view(),
            self.degree,
            &mut b_raw,
        )
        .is_err()
        {
            return Array2::zeros((n, 0));
        }

        // Use geometric transform (preferred) or fall back to link_transform
        let transform = self
            .geometric_link_transform
            .as_ref()
            .or(self.link_transform.as_ref());

        if let Some(transform) = transform {
            if transform.ncols() > 0 && transform.nrows() == b_raw.ncols() {
                return b_raw.dot(transform);
            }
        }
        Array2::zeros((n, self.beta_link.len()))
    }

    /// Compute standardized coordinates for the link spline.
    /// Returns (z_raw, z_clamped, rangewidth) where:
    ///   z_raw = (u - min_u) / rangewidth
    ///   z_clamped = clamp(z_raw, 0, 1)
    fn standardized_z(&self, eta_base: &Array1<f64>) -> (Array1<f64>, Array1<f64>, f64) {
        let (min_u, max_u) = self.knot_range.unwrap_or((0.0, 1.0));
        let rangewidth = (max_u - min_u).max(1e-6);
        let z_raw: Array1<f64> = eta_base.mapv(|u| (u - min_u) / rangewidth);
        let z_clamped: Array1<f64> = z_raw.mapv(|z| z.clamp(0.0, 1.0));
        (z_raw, z_clamped, rangewidth)
    }

    /// Return the stored projected penalty for link block (Z'SZ)
    pub fn build_link_penalty(&self) -> Array2<f64> {
        self.s_link_constrained
            .clone()
            .unwrap_or_else(|| Array2::zeros((0, 0)))
    }

    fn build_base_penalty(&self, lambda_base: &Array1<f64>) -> Array2<f64> {
        let p = self.x_base.ncols();
        let mut penalty = Array2::<f64>::zeros((p, p));
        for (idx, s_k) in self.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if s_k.nrows() == p && s_k.ncols() == p && lambda_k > 0.0 {
                penalty.scaled_add(lambda_k, s_k);
            }
        }
        penalty
    }

    fn build_joint_penalty(
        &self,
        lambda_base: &Array1<f64>,
        lambda_link: f64,
        p_link: usize,
    ) -> Array2<f64> {
        let p_base = self.x_base.ncols();
        let coef_layout = JointCoefLayout::new(p_base, p_link);
        let p_total = coef_layout.p_total();
        let mut penalty = Array2::<f64>::zeros((p_total, p_total));
        let base_penalty = self.build_base_penalty(lambda_base);
        if p_base > 0 {
            penalty
                .slice_mut(s![..p_base, ..p_base])
                .assign(&base_penalty);
        }
        let link_penalty = self.build_link_penalty();
        if p_link > 0 && link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
            let mut scaled_link = link_penalty;
            scaled_link *= lambda_link;
            penalty
                .slice_mut(s![p_base.., p_base..])
                .assign(&scaled_link);
        }
        penalty
    }

    fn build_joint_jacobian(&self, bwiggle: &Array2<f64>, g_prime: &Array1<f64>) -> Array2<f64> {
        // Full joint Jacobian for the two-block parameter vector α = [β; θ]:
        //   η(α) = g(u), u = Xβ, g(u) = u + Bwiggle(u) θ
        // Local Gauss-Newton linearization uses:
        //   J = ∂η/∂α = [J_β | J_θ]
        //   J_β = diag(g'(u)) X
        //   J_θ = Bwiggle
        //
        // This directly encodes cross-block coupling in J'WJ. In the common
        // single-predictor split (η = Xβ + Bθ), this reduces to the familiar
        // block Hessian with cross term X'WB.
        let n = self.nobs();
        let p_base = self.x_base.ncols();
        let p_link = bwiggle.ncols();
        let coef_layout = JointCoefLayout::new(p_base, p_link);
        let mut j_mat = Array2::<f64>::zeros((n, coef_layout.p_total()));
        for i in 0..n {
            let gp = g_prime[i];
            for j in 0..p_base {
                j_mat[[i, j]] = gp * self.x_base[[i, j]];
            }
            for j in 0..p_link {
                j_mat[[i, p_base + j]] = bwiggle[[i, j]];
            }
        }
        j_mat
    }

    fn compute_jointworkingvectors(
        &mut self,
        u: &Array1<f64>,
        bwiggle: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let n = self.nobs();
        let eta = self.compute_eta_full(u, bwiggle);
        self.last_eta = Some(eta.clone());

        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut z_glm = Array1::<f64>::zeros(n);
        if let Some(se) = &self.covariate_se {
            if let Some(family) = integrated_binomial_family_from_link(self.link, self.sas_link_state.is_some()) {
                if let Err(e) = crate::pirls::update_glmvectors_integrated_by_family(
                    &self.quadctx,
                    self.y,
                    &eta,
                    se.view(),
                    family,
                    self.weights,
                    &mut mu,
                    &mut weights,
                    &mut z_glm,
                    None,
                    None,
                    self.sas_link_state.as_ref(),
                ) {
                    log::warn!(
                        "joint integrated working-vector update failed (falling back to non-integrated): {}",
                        e
                    );
                    crate::pirls::update_glmvectors(
                        self.y,
                        &eta,
                        &InverseLink::Standard(self.link),
                        self.weights,
                        &mut mu,
                        &mut weights,
                        &mut z_glm,
                        None,
                    )
                    .map_err(|e2| {
                        EstimationError::InvalidInput(format!(
                            "joint working-vector update failed for {:?}: integrated error: {}; non-integrated fallback error: {}",
                            self.link, e, e2
                        ))
                    })?;
                }
            } else {
                crate::pirls::update_glmvectors(
                    self.y,
                    &eta,
                    &InverseLink::Standard(self.link),
                    self.weights,
                    &mut mu,
                    &mut weights,
                    &mut z_glm,
                    None,
                )?;
            }
        } else {
            crate::pirls::update_glmvectors(
                self.y,
                &eta,
                &InverseLink::Standard(self.link),
                self.weights,
                &mut mu,
                &mut weights,
                &mut z_glm,
                None,
            )?;
        }
        Ok((eta, mu, weights, z_glm))
    }

    /// Solve penalized weighted least squares: (X'WX + λS + δI)β = X'Wz
    ///
    /// Returns (coefficients, ridge_used) where ridge_used is the stabilization
    /// ridge δ added to ensure positive definiteness.
    ///
    /// # Mathematical Note (Envelope Theorem)
    /// When ridge δ > 0, the solver finds β̂ that minimizes:
    ///   Lridge(β) = -ℓ(β) + 0.5*β'S_λβ + 0.5*δ||β||²
    ///
    /// For the analytic gradient to be exact, the cost function must include
    /// the same δ||β||² term. Otherwise, ∇_β V(β̂) ≠ 0 and the Envelope
    /// Theorem fails, introducing gradient error proportional to δ*||β||.
    fn solveweighted_ls<S>(
        x: &ndarray::ArrayBase<S, ndarray::Ix2>,
        z: &Array1<f64>,
        w: &Array1<f64>,
        penalty: &Array2<f64>,
        lambda: f64,
    ) -> (Array1<f64>, f64)
    where
        S: ndarray::Data<Elem = f64>,
    {
        use crate::faer_ndarray::FaerCholesky;
        use faer::Side;

        let p = x.ncols();

        if p == 0 {
            return (Array1::zeros(0), 0.0);
        }

        let mut xwx = fast_xt_diag_x(x, w);

        // Add penalty: X'WX + λS (only if penalty matches dimensions)
        if penalty.nrows() == p && penalty.ncols() == p {
            xwx = xwx + penalty * lambda;
        }

        // Conditional regularization for numerical stability.
        // The ridge δ must be tracked and included in the cost function
        // to satisfy the Envelope Theorem (see docstring).
        let ridge_used = ensure_positive_definite_joint(&mut xwx);

        let mut wz = z.clone();
        wz *= w;
        let mut xwz = Array1::<f64>::zeros(p);
        fast_atv_into(x, &wz, &mut xwz);

        // Solve (X'WX + λS + δI)β = X'Wz using Cholesky
        let coeffs = match xwx.cholesky(Side::Lower) {
            Ok(chol) => chol.solvevec(&xwz),
            Err(_) => {
                eprintln!("[JOINT] Warning: Cholesky factorization failed");
                Array1::zeros(p)
            }
        };
        (coeffs, ridge_used)
    }

    /// Compute the full-joint penalized Jeffreys/Firth adjustment from the actual
    /// local quadratic model used by the joint solver:
    ///   H = J' W J + S_lambda + δI.
    ///
    /// Mathematical target (penalized-information Jeffreys/Firth):
    ///   Φ_firth(α, λ) = 0.5 * log|H(α, λ)|
    /// with α = [β; θ]. If we write H in block form:
    ///   H = [[H_ββ, H_βθ],
    ///        [H_θβ, H_θθ]]
    /// where for the standard split predictor case:
    ///   H_ββ = X'WX + S_β
    ///   H_βθ = X'WB
    ///   H_θβ = B'WX
    ///   H_θθ = B'WB + S_θ.
    /// We compute leverage from this full H (not blockwise approximations):
    ///   h_i = (w_i^{1/2} J_i) H^{-1} (w_i^{1/2} J_i)'.
    ///
    /// This is intentionally full-joint, not blockwise. The returned leverage is
    /// the diagonal of W^(1/2) J H^{-1} J' W^(1/2), and the log-determinant is
    /// 0.5 * log|H| for the stabilized Hessian actually factorized.
    fn compute_joint_firth_adjustment(
        joint_design: &Array2<f64>,
        penalty_full: &Array2<f64>,
    ) -> Result<JointFirthAdjustment, EstimationError> {
        use crate::faer_ndarray::FaerCholesky;

        let n = joint_design.nrows();
        let p = joint_design.ncols();
        if n == 0 || p == 0 {
            return Ok(JointFirthAdjustment {
                hat_diag: Array1::zeros(n),
                half_log_det: 0.0,
                ridge_used: 0.0,
            });
        }

        let mut h_joint = fast_ata(joint_design);
        if penalty_full.nrows() == p && penalty_full.ncols() == p {
            h_joint += penalty_full;
        }
        let ridge_used = ensure_positive_definite_joint(&mut h_joint);

        let chol = h_joint.cholesky(Side::Lower).map_err(|_| {
            EstimationError::HessianNotPositiveDefinite {
                min_eigenvalue: f64::NEG_INFINITY,
            }
        })?;
        let half_log_det = chol.diag().mapv(f64::ln).sum();
        let joint_design_t = joint_design.view().reversed_axes();
        let mut solvedweighted_t = Array2::<f64>::zeros(joint_design_t.raw_dim());
        chol.solve_mat_into(&joint_design_t, &mut solvedweighted_t);

        let mut hat_diag = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut h_ii = 0.0;
            for j in 0..p {
                h_ii += joint_design[[i, j]] * solvedweighted_t[[j, i]];
            }
            hat_diag[i] = h_ii;
        }
        Ok(JointFirthAdjustment {
            hat_diag,
            half_log_det,
            ridge_used,
        })
    }

    fn build_firthworking_response(
        z_glm: &Array1<f64>,
        mu: &Array1<f64>,
        weights: &Array1<f64>,
        hat_diag: &Array1<f64>,
    ) -> Array1<f64> {
        let mut z_firth = z_glm.clone();
        for i in 0..z_firth.len() {
            let wi = weights[i];
            if wi <= 0.0 {
                continue;
            }
            let mi = mu[i];
            z_firth[i] += hat_diag[i] * (0.5 - mi) / wi;
        }
        z_firth
    }

    /// Build the shared working response for one block update from the full
    /// coupled penalized Hessian:
    ///   H = J' W J + S_lambda (+ δI for stabilization).
    /// This enforces one Jeffreys/Firth definition across base and link updates.
    ///
    /// Working-response correction for logit:
    ///   z_firth,i = z_glm,i + h_i * (0.5 - μ_i) / w_i
    /// where h_i is from the full joint leverage above.
    /// This avoids mixing objectives (e.g., solving with penalized H while
    /// correcting with an unpenalized/blockwise hat matrix).
    fn jointworking_response_for_block(
        &mut self,
        u: &Array1<f64>,
        bwiggle: &Array2<f64>,
        g_prime: &Array1<f64>,
        lambda_base: &Array1<f64>,
        lambda_link: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        let (_, mu, weights, z_glm) = self.compute_jointworkingvectors(u, bwiggle)?;
        if !(self.firth_bias_reduction && matches!(self.link, LinkFunction::Logit)) {
            return Ok((weights, z_glm));
        }

        let j_mat = self.build_joint_jacobian(bwiggle, g_prime);
        let penalty = self.build_joint_penalty(lambda_base, lambda_link, bwiggle.ncols());
        let mut jweighted = j_mat;
        let sqrtw = weights.mapv(|wi| wi.max(0.0).sqrt());
        ndarray::Zip::from(jweighted.rows_mut())
            .and(sqrtw.view())
            .for_each(|mut row, wi| row *= *wi);
        let firth = Self::compute_joint_firth_adjustment(&jweighted, &penalty)?;
        self.ridge_used = self.ridge_used.max(firth.ridge_used);
        let _ = firth.half_log_det;
        let zworking = Self::build_firthworking_response(&z_glm, &mu, &weights, &firth.hat_diag);
        Ok((weights, zworking))
    }

    /// Perform one IRLS step for the link block
    /// Uses identity (u) as OFFSET, solves only for wiggle coefficients
    /// η = u + Bwiggle(z) · θ
    pub fn irls_link_step(
        &mut self,
        bwiggle: &Array2<f64>, // Constrained wiggle basis (NOT including identity)
        u: &Array1<f64>,       // Current linear predictor u = Xβ (used as offset)
        lambda_link: f64,
        weights: &Array1<f64>,
        zworking: &Array1<f64>,
    ) -> f64 {
        // The Jeffreys/Firth adjustment is computed once from the full joint
        // penalized Hessian and passed into both block updates. The link block
        // therefore consumes the shared corrected working response directly.
        let z_adjusted: Array1<f64> = zworking - u;

        // Solve: (B'WB + λS + δI)θ = B'W(z - u)
        // Track ridge δ to include 0.5*δ||θ||² in cost (Envelope Theorem).
        self.ridge_link_used = 0.0;
        if bwiggle.ncols() > 0 {
            let penalty = self.build_link_penalty();
            let (new_theta, ridge_link) =
                Self::solveweighted_ls(bwiggle, &z_adjusted, weights, &penalty, lambda_link);

            // Update wiggle coefficients
            if new_theta.len() == self.beta_link.len() {
                self.beta_link = new_theta;
            }
            self.ridge_link_used = ridge_link;
            // Accumulate ridge (take max of base and link ridges for joint system)
            self.ridge_used = self.ridge_used.max(ridge_link);
        }

        // Recompute deviance using updated coefficients
        let eta_updated = self.compute_eta_full(u, bwiggle);
        self.recompute_deviance_from_eta(&eta_updated)
    }

    /// Perform one IRLS step for the base β block
    /// Uses Gauss-Newton with proper offset for nonlinear link:
    /// η = g(u) = u + wiggle(u), ∂η/∂β = g'(u) · x
    /// Working response for β: z_β = zworking - η + g'(u)·u
    pub fn irls_base_step(
        &mut self,
        bwiggle: &Array2<f64>, // Constrained wiggle basis
        g_prime: &Array1<f64>, // Derivative of link: g'(u) = 1 + B'(u)·θ
        lambda_base: &Array1<f64>,
        damping: f64,
        zworking: &Array1<f64>,
        weights: &Array1<f64>,
    ) -> f64 {
        let n = self.nobs();
        let p = self.x_base.ncols();

        // Current u = Xβ
        let u = self.base_linear_predictor();

        // Current η = u + Bwiggle · θ
        let wiggle: Array1<f64> = if bwiggle.ncols() > 0 && self.beta_link.len() == bwiggle.ncols()
        {
            bwiggle.dot(&self.beta_link)
        } else {
            Array1::zeros(n)
        };
        let eta: Array1<f64> = &u + &wiggle;

        // Correct working response for β update (Gauss-Newton offset):
        // z_β = z_firth - η + g'(u)·u
        let mut z_beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            z_beta[i] = zworking[i] - eta[i] + g_prime[i] * u[i];
        }

        // Weighted least squares with scaled weights to avoid explicit X_eff
        let mut w_eff = Array1::<f64>::zeros(n);
        let mut z_eff = Array1::<f64>::zeros(n);
        for i in 0..n {
            let g = g_prime[i];
            let g_safe = if g.abs() < 1e-8 {
                if g >= 0.0 { 1e-8 } else { -1e-8 }
            } else {
                g
            };
            w_eff[i] = weights[i] * g_safe * g_safe;
            z_eff[i] = z_beta[i] / g_safe;
        }

        // Build penalty for base block: S_base = Σ λ_k S_k
        let penalty = self.build_base_penalty(lambda_base);

        // Solve PWLS: (X'W_eff X + S + δI)β = X'W_eff z_eff
        // Track ridge δ to include 0.5*δ||β||² in cost (Envelope Theorem).
        let (new_beta, ridge_base) =
            Self::solveweighted_ls(&self.x_base, &z_eff, &w_eff, &penalty, 1.0);
        self.ridge_base_used = ridge_base;
        // Accumulate ridge (take max of base and link ridges for joint system)
        self.ridge_used = self.ridge_used.max(ridge_base);

        // Apply damped update
        for j in 0..p {
            if j < new_beta.len() {
                let delta = new_beta[j] - self.beta_base[j];
                self.beta_base[j] += damping * delta;
            }
        }

        // Recompute deviance using updated coefficients
        let u_updated = self.base_linear_predictor();
        let wiggle_updated: Array1<f64> =
            if bwiggle.ncols() > 0 && self.beta_link.len() == bwiggle.ncols() {
                bwiggle.dot(&self.beta_link)
            } else {
                Array1::zeros(n)
            };
        let eta_updated: Array1<f64> = &u_updated + &wiggle_updated;
        self.last_eta = Some(eta_updated.clone());
        self.recompute_deviance_from_eta(&eta_updated)
    }

    /// Compute current linear predictor: η = u + Bwiggle · θ
    pub fn compute_eta_full(&self, u: &Array1<f64>, bwiggle: &Array2<f64>) -> Array1<f64> {
        if bwiggle.ncols() > 0 && self.beta_link.len() == bwiggle.ncols() {
            let wiggle = bwiggle.dot(&self.beta_link);
            u + &wiggle
        } else {
            u.clone()
        }
    }

    /// Recompute deviance by refreshing GLM working vectors at a supplied linear predictor.
    fn recompute_deviance_from_eta(&self, eta: &Array1<f64>) -> f64 {
        let n = self.nobs();
        let mut mu_updated = Array1::<f64>::zeros(n);
        let mut weights_updated = Array1::<f64>::zeros(n);
        let mut z_updated = Array1::<f64>::zeros(n);

        if let Some(se) = &self.covariate_se {
            let inverse_link = match (self.link, &self.sas_link_state) {
                (LinkFunction::Sas, Some(state)) => InverseLink::Sas(*state),
                (LinkFunction::BetaLogistic, Some(state)) => InverseLink::BetaLogistic(*state),
                _ => InverseLink::Standard(self.link),
            };
            match self.link {
                LinkFunction::Logit | LinkFunction::Probit | LinkFunction::CLogLog
                | LinkFunction::Sas | LinkFunction::BetaLogistic => {
                    if let Err(e) = crate::pirls::update_glmvectors_integrated_for_link(
                        &self.quadctx,
                        self.y,
                        eta,
                        se.view(),
                        &inverse_link,
                        self.weights,
                        &mut mu_updated,
                        &mut weights_updated,
                        &mut z_updated,
                        None,
                    ) {
                        log::warn!(
                            "joint integrated update failed for {:?}; falling back to non-integrated update: {}",
                            self.link,
                            e
                        );
                        if let Err(e2) = crate::pirls::update_glmvectors(
                            self.y,
                            eta,
                            &inverse_link,
                            self.weights,
                            &mut mu_updated,
                            &mut weights_updated,
                            &mut z_updated,
                            None,
                        ) {
                            log::warn!("joint non-integrated fallback update failed: {}", e2);
                        }
                    }
                }
                _ => {
                    if let Err(e) = crate::pirls::update_glmvectors(
                        self.y,
                        eta,
                        &inverse_link,
                        self.weights,
                        &mut mu_updated,
                        &mut weights_updated,
                        &mut z_updated,
                        None,
                    ) {
                        log::warn!("joint working-vector update failed: {}", e);
                    }
                }
            }
        } else {
            if let Err(e) = crate::pirls::update_glmvectors(
                self.y,
                eta,
                &InverseLink::Standard(self.link),
                self.weights,
                &mut mu_updated,
                &mut weights_updated,
                &mut z_updated,
                None,
            ) {
                log::warn!("joint working-vector update failed: {}", e);
            }
        }
        self.compute_deviance(&mu_updated)
    }

    /// Compute deviance based on link function
    fn compute_deviance(&self, mu: &Array1<f64>) -> f64 {
        crate::pirls::calculate_deviance(self.y, mu, self.link.clone(), self.weights)
    }
}

///
/// Architecture:
/// - Outer loop: BFGS over smoothing params ρ (same as existing GAM fitting)
/// - Inner loop: Gauss-Newton PIRLS for coefficients (β, θ)
///
/// The model is nonlinear because g depends on u, and u depends on β:
///   η_i = g(u_i), where u_i = x_i'β + f(covariates)
///   g(u) = B(u)θ for spline basis B evaluated at u
///
/// The inner solve uses a Jacobian J instead of fixed design X:
///   J_i = [g'(u_i) * x_i | B(u_i)]
///
/// Identifiability: the wiggle basis is projected into the null space of the
/// intercept and linear functions, so the flexible correction cannot absorb the
/// global level or slope already carried by the base predictor.
pub(crate) fn fit_joint_model<'a>(
    y: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    x_base: ArrayView2<'a, f64>,
    s_base: Vec<Array2<f64>>,
    layout_base: EngineDims,
    link: LinkFunction,
    config: &JointModelConfig,
    include_base_covariance: bool,
) -> Result<JointModelResult, EstimationError> {
    fit_joint_modelwith_reml(
        y,
        weights,
        x_base,
        s_base,
        layout_base,
        link,
        config,
        None,
        include_base_covariance,
    )
}

/// Engine-facing joint model entrypoint without domain `EngineDims`.
pub fn fit_joint_model_engine<'a>(
    y: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    x_base: ArrayView2<'a, f64>,
    s_base: Vec<Array2<f64>>,
    link: LinkFunction,
    geometry: JointLinkGeometry,
    mut config: JointModelConfig,
    include_base_covariance: bool,
) -> Result<JointModelResult, EstimationError> {
    if matches!(link, LinkFunction::Sas | LinkFunction::BetaLogistic) {
        return Err(EstimationError::InvalidSpecification(
            "joint model engine does not support state-less SAS/Beta-Logistic links; use binomial stateful link fit paths"
                .to_string(),
        ));
    }
    if geometry.degree != 3 {
        return Err(EstimationError::InvalidSpecification(
            "joint engine currently supports cubic link splines only (degree=3)".to_string(),
        ));
    }
    config.n_link_knots = geometry.n_link_knots;
    let layout = EngineDims::new(x_base.ncols(), s_base.len());
    fit_joint_model(
        y,
        weights,
        x_base,
        s_base,
        layout,
        link,
        &config,
        include_base_covariance,
    )
}

/// State for REML optimization of the joint model
/// Wraps JointModelState and provides outer objective derivatives over ρ.
pub(crate) struct JointCore<'a> {
    state: JointModelState<'a>,
    config: JointModelConfig,
    base_reparam_invariant: Option<ReparamInvariant>,
    base_rs_list: Vec<Array2<f64>>,
}

pub struct JointEvalContext {
    /// Cached warm-start coefficients
    cached_beta_base: Array1<f64>,
    cached_beta_link: Array1<f64>,
    /// Cached LAML value for gradient computation
    cached_laml: Option<f64>,
    cached_rho: Array1<f64>,
    cached_edf: Option<f64>,
    cached_edf_terms: Vec<(String, f64, f64)>,
    last_backfit_iterations: usize,
    last_converged: bool,
    lasthessian_condition: Option<f64>,
    last_outer_step_norm: Option<f64>,
    last_eval_rho: Option<Array1<f64>>,
}

pub struct JointRemlState<'a> {
    core: JointCore<'a>,
    eval: JointEvalContext,
    visualizer: visualizer::VisualizerSession,
}

struct JointRemlSnapshot {
    beta_base: Array1<f64>,
    beta_link: Array1<f64>,
    rho: Array1<f64>,
    knot_range: Option<(f64, f64)>,
    knot_vector: Option<Array1<f64>>,
    link_transform: Option<Array2<f64>>,
    s_link_constrained: Option<Array2<f64>>,
    n_constrained_basis: usize,
    cached_beta_base: Array1<f64>,
    cached_beta_link: Array1<f64>,
    cached_rho: Array1<f64>,
    cached_laml: Option<f64>,
    cached_edf: Option<f64>,
    cached_edf_terms: Vec<(String, f64, f64)>,
    last_backfit_iterations: usize,
    last_converged: bool,
    lasthessian_condition: Option<f64>,
    last_outer_step_norm: Option<f64>,
    last_eval_rho: Option<Array1<f64>>,
}

impl JointRemlSnapshot {
    fn new(reml: &JointRemlState<'_>) -> Self {
        let state = &reml.core.state;
        Self {
            beta_base: state.beta_base.clone(),
            beta_link: state.beta_link.clone(),
            rho: state.rho.clone(),
            knot_range: state.knot_range,
            knot_vector: state.knot_vector.clone(),
            link_transform: state.link_transform.clone(),
            s_link_constrained: state.s_link_constrained.clone(),
            n_constrained_basis: state.n_constrained_basis,
            cached_beta_base: reml.eval.cached_beta_base.clone(),
            cached_beta_link: reml.eval.cached_beta_link.clone(),
            cached_rho: reml.eval.cached_rho.clone(),
            cached_laml: reml.eval.cached_laml,
            cached_edf: reml.eval.cached_edf,
            cached_edf_terms: reml.eval.cached_edf_terms.clone(),
            last_backfit_iterations: reml.eval.last_backfit_iterations,
            last_converged: reml.eval.last_converged,
            lasthessian_condition: reml.eval.lasthessian_condition,
            last_outer_step_norm: reml.eval.last_outer_step_norm,
            last_eval_rho: reml.eval.last_eval_rho.clone(),
        }
    }

    fn restore(&self, reml: &mut JointRemlState<'_>) {
        let state = &mut reml.core.state;
        state.beta_base = self.beta_base.clone();
        state.beta_link = self.beta_link.clone();
        state.rho = self.rho.clone();
        state.knot_range = self.knot_range;
        state.knot_vector = self.knot_vector.clone();
        state.link_transform = self.link_transform.clone();
        state.s_link_constrained = self.s_link_constrained.clone();
        state.n_constrained_basis = self.n_constrained_basis;
        reml.eval.cached_beta_base = self.cached_beta_base.clone();
        reml.eval.cached_beta_link = self.cached_beta_link.clone();
        reml.eval.cached_rho = self.cached_rho.clone();
        reml.eval.cached_laml = self.cached_laml;
        reml.eval.cached_edf = self.cached_edf;
        reml.eval.cached_edf_terms = self.cached_edf_terms.clone();
        reml.eval.last_backfit_iterations = self.last_backfit_iterations;
        reml.eval.last_converged = self.last_converged;
        reml.eval.lasthessian_condition = self.lasthessian_condition;
        reml.eval.last_outer_step_norm = self.last_outer_step_norm;
        reml.eval.last_eval_rho = self.last_eval_rho.clone();
    }
}

impl<'a> JointRemlState<'a> {
    /// Create new REML state
    pub(crate) fn new(
        y: ArrayView1<'a, f64>,
        weights: ArrayView1<'a, f64>,
        x_base: ArrayView2<'a, f64>,
        s_base: Vec<Array2<f64>>,
        layout_base: EngineDims,
        link: LinkFunction,
        config: &JointModelConfig,
        covariate_se: Option<Array1<f64>>,
        quadctx: QuadratureContext,
    ) -> Self {
        let mut state = JointModelState::new(
            y,
            weights,
            x_base,
            s_base,
            layout_base,
            link,
            config,
            quadctx,
        );
        // Set covariate_se for uncertainty-aware IRLS
        if let Some(se) = covariate_se {
            state = state.with_covariate_se(se);
        }
        let u0 = state.base_linear_predictor();
        if let Err(e) = state.build_link_basis(&u0) {
            eprintln!("[JOINT] Warning during initialization: {e}");
        }
        let n_base = state.s_base.len();
        let cached_beta_base = state.beta_base.clone();
        let cached_beta_link = state.beta_link.clone();
        let base_rs_list =
            compute_penalty_square_roots(&state.s_base).unwrap_or_else(|_| Vec::new());
        let base_reparam_invariant =
            precompute_reparam_invariant(&base_rs_list, state.layout_base.p).ok();
        Self {
            core: JointCore {
                state,
                config: config.clone(),
                base_reparam_invariant,
                base_rs_list,
            },
            eval: JointEvalContext {
                cached_beta_base,
                cached_beta_link,
                cached_laml: None,
                cached_rho: Array1::zeros(n_base + 1),
                cached_edf: None,
                cached_edf_terms: Vec::new(),
                last_backfit_iterations: 0,
                last_converged: false,
                lasthessian_condition: None,
                last_outer_step_norm: None,
                last_eval_rho: None,
            },
            visualizer: visualizer::VisualizerSession::default(),
        }
    }

    /// Compute LAML cost for a given ρ
    /// LAML = deviance + log|H_pen| - log|S_λ| (+ prior on ρ)
    pub fn compute_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let state = &mut self.core.state;
        let penalty_layout = JointPenaltyLayout::for_state(state);
        penalty_layout.validate_rho(rho)?;

        // Set ρ and warm-start from cached coefficients
        state.set_rho(rho.clone());
        state.beta_base = self.eval.cached_beta_base.clone();
        state.beta_link = self.eval.cached_beta_link.clone();

        // Run inner alternating to convergence
        let (lambda_base, lambda_link) = penalty_layout.lambdas(rho);

        let mut prev_deviance = f64::INFINITY;
        let mut iter_count = 0;
        let mut converged = false;
        for i in 0..self.core.config.max_backfit_iter {
            let progress = (i as f64) / (self.core.config.max_backfit_iter as f64);
            let damping = 0.5 + progress * 0.5;

            let u = state.base_linear_predictor();
            let bwiggle = state
                .build_link_basis(&u)
                .map_err(|e| EstimationError::InvalidSpecification(e))?;

            let g_prime_link = compute_link_derivative_from_state(&state, &u, &bwiggle);
            let (weights_link, zworking_link) = state.jointworking_response_for_block(
                &u,
                &bwiggle,
                &g_prime_link,
                &lambda_base,
                lambda_link,
            )?;
            state.irls_link_step(&bwiggle, &u, lambda_link, &weights_link, &zworking_link);

            let g_prime_base = compute_link_derivative_from_state(&state, &u, &bwiggle);
            let (weights_base, zworking_base) = state.jointworking_response_for_block(
                &u,
                &bwiggle,
                &g_prime_base,
                &lambda_base,
                lambda_link,
            )?;
            let deviance = state.irls_base_step(
                &bwiggle,
                &g_prime_base,
                &lambda_base,
                damping,
                &zworking_base,
                &weights_base,
            );

            let delta = (prev_deviance - deviance).abs() / (deviance.abs() + 1.0);
            iter_count = i + 1;
            if delta < self.core.config.backfit_tol {
                converged = true;
                break;
            }
            prev_deviance = deviance;
        }

        // Cache converged coefficients for warm-start
        self.eval.cached_beta_base = state.beta_base.clone();
        self.eval.cached_beta_link = state.beta_link.clone();

        // Unified REML/LAML evaluator: build InnerSolution and compute cost
        // through the single formula that guarantees cost/gradient coherency.
        use crate::estimate::reml::unified::{EvalMode, reml_laml_evaluate};
        let (inner_solution, edf) = Self::build_inner_solution_at_convergence(
            state,
            rho.as_slice().unwrap(),
            &lambda_base,
            lambda_link,
            self.core.base_reparam_invariant.as_ref(),
            &self.core.base_rs_list,
        )?;
        let unified_result = reml_laml_evaluate(
            &inner_solution,
            rho.as_slice().unwrap(),
            EvalMode::ValueOnly,
            None,
        )
        .map_err(|e| EstimationError::InvalidInput(e))?;
        let laml = -unified_result.cost; // unified returns cost to minimize; LAML is negated

        // Cache for gradient
        self.eval.cached_laml = Some(laml);
        self.eval.cached_rho = rho.clone();
        self.eval.cached_edf = edf;
        if self.eval.cached_edf_terms.is_empty() {
            if let Some(edf_total) = edf {
                let p_total = (state.x_base.ncols() + state.beta_link.len()) as f64;
                self.eval.cached_edf_terms = vec![("Total Model".to_string(), edf_total, p_total)];
            }
        }
        self.eval.last_backfit_iterations = iter_count;
        self.eval.last_converged = converged;

        Ok(-laml)
    }

    /// Compute cost, gradient, and outer Hessian through the unified evaluator.
    ///
    /// This method runs the inner solver (via compute_cost), builds the enriched
    /// InnerSolution with full gradient corrections and Hessian traces, and
    /// evaluates everything through `reml_laml_evaluate(ValueGradientHessian)`.
    pub fn compute_unified_eval(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<OuterEval, EstimationError> {
        use crate::estimate::reml::unified::{EvalMode, reml_laml_evaluate};

        // Step 1: Run inner solver (caches converged state).
        let cost = self.compute_cost(rho)?;

        // Step 2: Build enriched InnerSolution from converged state.
        let penalty_layout = JointPenaltyLayout::for_state(&self.core.state);
        let (lambda_base, lambda_link) = penalty_layout.lambdas(rho);
        let (inner_solution, ..) = Self::build_inner_solution_at_convergence(
            &self.core.state,
            rho.as_slice().unwrap(),
            &lambda_base,
            lambda_link,
            self.core.base_reparam_invariant.as_ref(),
            &self.core.base_rs_list,
        )?;

        // Step 3: Evaluate cost + gradient + Hessian through unified path.
        let result = reml_laml_evaluate(
            &inner_solution,
            rho.as_slice().unwrap(),
            EvalMode::ValueGradientHessian,
            None,
        )
        .map_err(|e| EstimationError::InvalidInput(e))?;

        Ok(OuterEval {
            cost,
            gradient: result.gradient.unwrap(),
            hessian: HessianResult::Analytic(result.hessian.unwrap()),
        })
    }

    /// Compute EFS (Extended Fellner-Schall) steps for the joint model by
    /// building an `InnerSolution` at the converged state and delegating to
    /// the unified `compute_efs_update`.
    pub fn compute_efs_eval(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<crate::solver::strategy::EfsEval, EstimationError> {
        use crate::estimate::reml::unified::compute_efs_update;

        let cost = self.compute_cost(rho)?;
        let penalty_layout = JointPenaltyLayout::for_state(&self.core.state);
        let (lambda_base, lambda_link) = penalty_layout.lambdas(rho);
        let (inner_solution, ..) = Self::build_inner_solution_at_convergence(
            &self.core.state,
            rho.as_slice().unwrap(),
            &lambda_base,
            lambda_link,
            self.core.base_reparam_invariant.as_ref(),
            &self.core.base_rs_list,
        )?;
        let steps = compute_efs_update(&inner_solution, rho.as_slice().unwrap());
        Ok(crate::solver::strategy::EfsEval {
            cost,
            steps,
            beta: None,
        })
    }

    fn fixed_subspace_logdet_for_penalty(
        s_lambda: &Array2<f64>,
        structural_rank: usize,
        ridge: f64,
    ) -> Result<f64, EstimationError> {
        if structural_rank == 0 || s_lambda.nrows() == 0 || s_lambda.ncols() == 0 {
            return Ok(0.0);
        }
        let rank = structural_rank.min(s_lambda.nrows()).min(s_lambda.ncols());
        let mut s_eval = s_lambda.clone();
        if ridge > 0.0 {
            let d = s_eval.nrows().min(s_eval.ncols());
            for i in 0..d {
                s_eval[[i, i]] += ridge;
            }
        }

        let (evals, _) = s_eval
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let mut order: Vec<usize> = (0..evals.len()).collect();
        order.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let max_ev: f64 = order
            .first()
            .map(|&idx| evals[idx].abs())
            .unwrap_or(1.0_f64)
            .max(1.0_f64);
        let floor = (1e-12_f64 * max_ev).max(1e-12_f64);
        Ok(order
            .iter()
            .take(rank)
            .map(|&idx| evals[idx].max(floor).ln())
            .sum())
    }

    fn structural_rank_from_penalty(s: &Array2<f64>) -> Result<usize, EstimationError> {
        if s.nrows() == 0 || s.ncols() == 0 {
            return Ok(0);
        }
        let (evals, _) = s
            .clone()
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let max_ev = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let tol = if max_ev > 0.0 { max_ev * 1e-12 } else { 1e-12 };
        Ok(evals.iter().filter(|&&ev| ev > tol).count())
    }
} // end impl JointRemlState

/// On-demand provider of Hessian derivative corrections for joint link-wiggle models.
///
/// Instead of precomputing per-k correction matrices and Ḧ_{jk} traces in the builder,
/// this struct stores the ingredients needed to compute directional derivatives of
/// H_L = J'WJ on demand. The unified REML evaluator calls `hessian_derivative_correction`
/// and `hessian_second_derivative_correction` as needed.
struct LinkWiggleDerivProvider {
    /// Joint Jacobian J (n × p_total).
    j_mat: std::sync::Arc<Array2<f64>>,
    /// √W-weighted Jacobian (n × p_total).
    jweighted: std::sync::Arc<Array2<f64>>,
    /// √W per observation.
    sqrtw: Array1<f64>,
    /// dW/dη per observation.
    w_prime: Array1<f64>,
    /// d²W/dη² per observation.
    w_double_prime: Array1<f64>,
    /// Base design matrix X (n × p_base).
    x_base: std::sync::Arc<Array2<f64>>,
    /// g''(u) per observation.
    gsecond: Array1<f64>,
    /// B'(z) · link_transform (n × p_link), derivative of link basis w.r.t. theta.
    b_prime_u: Array2<f64>,
    /// B'(z) raw (n × n_raw), first derivative of B-spline basis.
    b_prime: Array2<f64>,
    /// link_transform (n_raw × p_link).
    link_transform: Array2<f64>,
    /// 1 / range_width for z-coordinate scaling.
    invrw: f64,
    /// Number of base coefficients.
    p_base: usize,
    /// Number of link coefficients.
    p_link: usize,
    /// Total coefficients (p_base + p_link).
    p_total: usize,
    /// Number of observations.
    n: usize,
    /// Number of raw B-spline basis functions.
    n_raw: usize,
}

impl LinkWiggleDerivProvider {
    /// Compute first-order quantities for a direction delta = -v_k.
    ///
    /// Returns (dot_eta, dot_J) where:
    /// - dot_eta = J · delta (change in linear predictor)
    /// - dot_J = dJ/d(beta) in direction delta
    fn compute_first_order(&self, delta: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
        let delta_beta = delta.slice(ndarray::s![..self.p_base]);
        let delta_theta = delta.slice(ndarray::s![self.p_base..self.p_total]);

        // dot_eta = J · delta
        let dot_eta: Array1<f64> = self.j_mat.dot(delta);

        // dot_u = X · delta_beta (change in base linear predictor)
        let dot_u: Array1<f64> = self.x_base.dot(&delta_beta);

        // dot_g_prime = g'' · dot_u + b_prime_u · delta_theta
        let mut dot_g_prime = Array1::<f64>::zeros(self.n);
        for i in 0..self.n {
            dot_g_prime[i] = self.gsecond[i] * dot_u[i];
        }
        dot_g_prime += &self.b_prime_u.dot(&delta_theta);

        // dot_J = [diag(dot_g_prime) X | B_dot · link_transform · dz]
        let mut dot_j = Array2::<f64>::zeros((self.n, self.p_total));
        for i in 0..self.n {
            let scale = dot_g_prime[i];
            for j in 0..self.p_base {
                dot_j[[i, j]] = scale * self.x_base[[i, j]];
            }
            let dz = dot_u[i] * self.invrw;
            if dz.abs() > 1e-30 {
                for c in 0..self.p_link {
                    let mut val = 0.0;
                    for r in 0..self.n_raw {
                        val += self.b_prime[[i, r]] * self.link_transform[[r, c]];
                    }
                    dot_j[[i, self.p_base + c]] = val * dz;
                }
            }
        }

        (dot_eta, dot_j)
    }

    /// Build the first-order correction: dot_J'WJ + J'W·dot_J + J'diag(w_dot)J
    fn build_first_order_correction(
        &self,
        dot_eta: &Array1<f64>,
        dot_j: &Array2<f64>,
    ) -> Array2<f64> {
        // Jacobian symmetry: √W·dot_J and √W·J
        let mut w_dot_j = dot_j.clone();
        ndarray::Zip::from(w_dot_j.rows_mut())
            .and(self.sqrtw.view())
            .for_each(|mut row, &wi| row *= wi);
        let atb = fast_atb(&w_dot_j, &self.jweighted);
        let jacobian_sym = &atb + &atb.t();

        // Weight correction: J'diag(w' · dot_eta)J
        let w_dot: Array1<f64> = &self.w_prime * dot_eta;
        let mut weight_corr = Array2::<f64>::zeros((self.p_total, self.p_total));
        for i in 0..self.n {
            let wd = w_dot[i];
            if wd.abs() < 1e-30 {
                continue;
            }
            let ji = self.j_mat.row(i);
            for a in 0..self.p_total {
                let wa = wd * ji[a];
                for b in a..self.p_total {
                    let val = wa * ji[b];
                    weight_corr[[a, b]] += val;
                    if a != b {
                        weight_corr[[b, a]] += val;
                    }
                }
            }
        }

        jacobian_sym + weight_corr
    }
}

impl crate::estimate::reml::unified::HessianDerivativeProvider for LinkWiggleDerivProvider {
    fn hessian_derivative_correction(&self, v_k: &Array1<f64>) -> Option<Array2<f64>> {
        // The unified evaluator passes v_k = H⁻¹(a_k); we negate to get delta = -v_k = dβ̂/dρ_k.
        let delta = v_k.mapv(|v| -v);
        let (dot_eta, dot_j) = self.compute_first_order(&delta);
        Some(self.build_first_order_correction(&dot_eta, &dot_j))
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Option<Array2<f64>> {
        let delta_k = v_k.mapv(|v| -v);
        let delta_l = v_l.mapv(|v| -v);

        // First-order quantities for directions k and l.
        let (dot_eta_k, dot_j_k) = self.compute_first_order(&delta_k);
        let (dot_eta_l, dot_j_l) = self.compute_first_order(&delta_l);

        // Term 1: D_β H_L[u_kl] (same structure as first-order correction applied to u_kl).
        let (dot_eta_kl_base, dot_j_kl) = self.compute_first_order(u_kl);
        let term1 = self.build_first_order_correction(&dot_eta_kl_base, &dot_j_kl);

        // Term 2: cross Jacobian: dot_J_k'W·dot_J_l + dot_J_l'W·dot_J_k
        let term2 = {
            let mut w_dot_j_k = dot_j_k.clone();
            ndarray::Zip::from(w_dot_j_k.rows_mut())
                .and(self.sqrtw.view())
                .for_each(|mut row, &wi| row *= wi);
            let mut w_dot_j_l = dot_j_l.clone();
            ndarray::Zip::from(w_dot_j_l.rows_mut())
                .and(self.sqrtw.view())
                .for_each(|mut row, &wi| row *= wi);
            let atb_2 = fast_atb(&w_dot_j_k, &w_dot_j_l);
            &atb_2 + &atb_2.t()
        };

        // Term 3: dot_J_k' diag(w' · dot_eta_l) J + J' diag(w' · dot_eta_l) dot_J_k
        let term3 = {
            let w_dot_l: Array1<f64> = &self.w_prime * &dot_eta_l;
            let mut wl_dot_j_k = dot_j_k.clone();
            for i in 0..self.n {
                let scale = w_dot_l[i];
                for j in 0..self.p_total {
                    wl_dot_j_k[[i, j]] *= scale;
                }
            }
            let atb_3 = fast_atb(&wl_dot_j_k, &*self.j_mat);
            &atb_3 + &atb_3.t()
        };

        // Term 4: dot_J_l' diag(w' · dot_eta_k) J + J' diag(w' · dot_eta_k) dot_J_l
        let term4 = {
            let w_dot_k: Array1<f64> = &self.w_prime * &dot_eta_k;
            let mut wk_dot_j_l = dot_j_l.clone();
            for i in 0..self.n {
                let scale = w_dot_k[i];
                for j in 0..self.p_total {
                    wk_dot_j_l[[i, j]] *= scale;
                }
            }
            let atb_4 = fast_atb(&wk_dot_j_l, &*self.j_mat);
            &atb_4 + &atb_4.t()
        };

        // Term 5: J' diag(w'' · dot_eta_k · dot_eta_l + w' · dot_eta_kl) J
        //   where dot_eta_kl = dot_J_k · delta_l + J · u_kl
        let term5 = {
            let dot_eta_kl: Array1<f64> = dot_j_k.dot(&delta_l) + self.j_mat.dot(u_kl);
            let mut w_ddot = Array1::<f64>::zeros(self.n);
            for i in 0..self.n {
                w_ddot[i] = self.w_double_prime[i] * dot_eta_k[i] * dot_eta_l[i]
                    + self.w_prime[i] * dot_eta_kl[i];
            }
            // J' diag(w_ddot) J using sqrtw trick is not applicable here since w_ddot
            // can be negative. Use direct outer product accumulation.
            let mut result = Array2::<f64>::zeros((self.p_total, self.p_total));
            for i in 0..self.n {
                let wd = w_ddot[i];
                if wd.abs() < 1e-30 {
                    continue;
                }
                let ji = self.j_mat.row(i);
                for a in 0..self.p_total {
                    let wa = wd * ji[a];
                    for b in a..self.p_total {
                        let val = wa * ji[b];
                        result[[a, b]] += val;
                        if a != b {
                            result[[b, a]] += val;
                        }
                    }
                }
            }
            result
        };

        Some(term1 + &term2 + &term3 + &term4 + &term5)
    }

    fn has_corrections(&self) -> bool {
        true
    }
}

impl<'a> JointRemlState<'a> {
    /// Build an InnerSolution from the converged joint state for the unified
    /// REML/LAML evaluator. This replaces compute_laml_at_convergence for cost
    /// and gradient/Hessian — all are now computed by
    /// the single reml_laml_evaluate function.
    fn build_inner_solution_at_convergence(
        state: &JointModelState,
        rho: &[f64],
        lambda_base: &Array1<f64>,
        lambda_link: f64,
        base_reparam_invariant: Option<&ReparamInvariant>,
        base_rs_list: &[Array2<f64>],
    ) -> Result<
        (
            crate::estimate::reml::unified::InnerSolution<'static>,
            Option<f64>,
        ),
        EstimationError,
    > {
        use crate::estimate::reml::unified::{
            DenseSpectralOperator, DispersionHandling, InnerSolutionBuilder, PenaltyLogdetDerivs,
            embed_penalty_root, penalty_matrix_root,
        };
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;

        let n = state.nobs();
        let u = state.base_linear_predictor();
        let bwiggle = state.build_link_basis_from_state(&u);
        let eta = state.compute_eta_full(&u, &bwiggle);

        // Compute mu/weights at convergence (same as compute_laml_at_convergence).
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        // Integrated GHQ derivatives: d1 = dE[h(η+ε)]/dη, d2 = d²E[h(η+ε)]/dη².
        // Stored per observation for use in w_prime when covariate_se is active.
        let mut integrated_d1 = Array1::<f64>::zeros(n);
        let mut integrated_d2 = Array1::<f64>::zeros(n);
        let is_gaussian = matches!(state.link, LinkFunction::Identity);

        match state.link {
            LinkFunction::Identity => {
                for i in 0..n {
                    mu[i] = eta[i];
                    weights[i] = state.weights[i];
                }
            }
            _ => {
                const MIN_WEIGHT: f64 = 1e-12;
                const MIN_DMU: f64 = 1e-6;
                let family = match state.link {
                    LinkFunction::Log => LikelihoodFamily::PoissonLog,
                    LinkFunction::Logit => LikelihoodFamily::BinomialLogit,
                    LinkFunction::Probit => LikelihoodFamily::BinomialProbit,
                    LinkFunction::CLogLog => LikelihoodFamily::BinomialCLogLog,
                    LinkFunction::Sas => LikelihoodFamily::BinomialSas,
                    LinkFunction::BetaLogistic => LikelihoodFamily::BinomialBetaLogistic,
                    LinkFunction::Identity => unreachable!(),
                };
                let strategy = strategy_for_family(family, None);
                for i in 0..n {
                    let se_i = state.covariate_se.as_ref().map_or(0.0, |se| se[i]);
                    let moments = strategy
                        .integrated_moments(&state.quadctx, eta[i], se_i)
                        .expect("binomial family moments must be available");
                    let dmu = moments.d1.abs().max(MIN_DMU);
                    mu[i] = moments.mean;
                    let w = ((dmu * dmu) / moments.variance.max(1e-12)).max(MIN_WEIGHT);
                    weights[i] = state.weights[i] * w;
                    integrated_d1[i] = moments.d1;
                    integrated_d2[i] = if moments.d2.is_finite() {
                        moments.d2
                    } else {
                        0.0
                    };
                }
            }
        }
        let deviance = state.compute_deviance(&mu);

        // Build joint Hessian H = J'WJ + S_λ.
        let p_base = state.x_base.ncols();
        let p_link = bwiggle.ncols();
        let coef_layout = JointCoefLayout::new(p_base, p_link);
        let p_total = coef_layout.p_total();

        let (g_prime, gsecond, b_prime_u) = compute_link_derivative_terms_from_state(state, &u);
        let j_mat = state.build_joint_jacobian(&bwiggle, &g_prime);
        let penalty_full = state.build_joint_penalty(lambda_base, lambda_link, p_link);
        let link_penalty = state.build_link_penalty();

        let mut jweighted = j_mat.clone();
        let sqrtw = weights.mapv(|wi| wi.max(0.0).sqrt());
        ndarray::Zip::from(jweighted.rows_mut())
            .and(sqrtw.view())
            .for_each(|mut row, wi| row *= *wi);
        let mut h_full = crate::faer_ndarray::fast_ata(&jweighted);
        if penalty_full.nrows() == p_total && penalty_full.ncols() == p_total {
            h_full += &penalty_full;
        }

        // Build HessianOperator from joint Hessian (delay boxing for enrichment).
        let hop = DenseSpectralOperator::from_symmetric(&h_full).map_err(|e| {
            EstimationError::InvalidInput(format!("Joint HessianOperator failed: {e}"))
        })?;

        // Penalty logdet via reparameterization (base) + eigendecomposition (link).
        let base_reparam = if let Some(invariant) = base_reparam_invariant {
            stable_reparameterizationwith_invariant_engine(
                base_rs_list,
                &lambda_base.to_vec(),
                EngineDims::new(state.layout_base.p, base_rs_list.len()),
                invariant,
                None, // Joint fitting does not apply shrinkage floor independently
            )
        } else {
            stable_reparameterization_engine(
                base_rs_list,
                &lambda_base.to_vec(),
                EngineDims::new(state.layout_base.p, base_rs_list.len()),
            )
        }
        .map_err(|e| EstimationError::InvalidInput(format!("Reparam failed: {e}")))?;

        let base_rank = base_reparam.e_transformed.nrows();
        let base_log_det_s = Self::fixed_subspace_logdet_for_penalty(
            &base_reparam.s_transformed,
            base_rank,
            state.ridge_base_used,
        )?;

        let (link_log_det_s, link_rank) = if p_link > 0 {
            let rank = Self::structural_rank_from_penalty(&link_penalty)?;
            let s_link_lambda = link_penalty.mapv(|v| v * lambda_link);
            let log_det = Self::fixed_subspace_logdet_for_penalty(
                &s_link_lambda,
                rank,
                state.ridge_link_used,
            )?;
            (log_det, rank)
        } else {
            (0.0, 0)
        };

        let log_det_s = base_log_det_s + link_log_det_s;
        let mp = (p_base - base_rank) as f64 + (p_link - link_rank) as f64;

        // Penalty logdet first derivatives.
        // Base derivatives from reparameterization; link derivative computed directly.
        let k = rho.len();
        let n_base_penalties = lambda_base.len();
        let mut det1 = Array1::zeros(k);
        for idx in 0..n_base_penalties.min(k) {
            det1[idx] = base_reparam.det1[idx];
        }
        // Link penalty derivative: d/dρ_link log|λ_link S_link|₊ = link_rank
        // (because log|λ S|₊ = rank * log(λ) + log|S|₊, and d/dρ = rank * dλ/dρ / λ = rank)
        if k > n_base_penalties {
            det1[n_base_penalties] = link_rank as f64;
        }

        // Penalty logdet second derivatives.
        //
        // det2[k,l] = delta_{kl} det1[k] - lambda_k lambda_l tr(S⁺ S_k S⁺ S_l)
        //
        // Base penalties share a block; link penalty is a separate single-parameter
        // block with disjoint support. Cross terms between base and link are zero
        // because the penalty matrices have disjoint column support. The link
        // diagonal entry is also zero: for a single penalty S = lambda * S_link,
        // the trace term equals det1[link] exactly, cancelling the diagonal term.
        let mut det2 = Array2::<f64>::zeros((k, k));
        if n_base_penalties > 0 {
            let rs_t = &base_reparam.rs_transformed;
            let p_r = rs_t[0].ncols();

            // Build S_lambda in the transformed coordinate frame.
            let mut s_lambda = Array2::<f64>::zeros((p_r, p_r));
            let mut s_k_mats = Vec::with_capacity(n_base_penalties);
            for (idx, r_k) in rs_t.iter().enumerate() {
                let s_k = r_k.t().dot(r_k);
                s_lambda.scaled_add(lambda_base[idx], &s_k);
                s_k_mats.push(s_k);
            }
            if state.ridge_base_used > 0.0 {
                for d in 0..p_r {
                    s_lambda[[d, d]] += state.ridge_base_used;
                }
            }

            // Eigendecompose for pseudo-inverse W W^T where W = V diag(1/sqrt(eig)).
            let (eigs, vecs) = s_lambda.eigh(Side::Lower).map_err(|e| {
                EstimationError::InvalidInput(format!("det2 eigendecomposition failed: {e}"))
            })?;
            let max_ev = eigs.iter().copied().fold(0.0_f64, f64::max);
            let tol = (p_r.max(1) as f64) * f64::EPSILON * max_ev.max(1e-12);
            let n_active = eigs.iter().filter(|&&v| v > tol).count();

            let mut w_factor = Array2::zeros((p_r, n_active));
            let mut w_col = 0;
            for (idx, &ev) in eigs.iter().enumerate() {
                if ev > tol {
                    let scale = 1.0 / ev.sqrt();
                    for row in 0..p_r {
                        w_factor[[row, w_col]] = vecs[[row, idx]] * scale;
                    }
                    w_col += 1;
                }
            }

            // M_k = W^T A_k W = lambda_k * W^T S_k W (n_active x n_active, symmetric).
            // det2[k,l] = delta_{kl} det1[k] - tr(M_k M_l).
            // tr(M_k M_l) = Frobenius inner product since M_k, M_l are symmetric.
            let mut m_mats = Vec::with_capacity(n_base_penalties);
            for (idx, s_k) in s_k_mats.iter().enumerate() {
                let s_k_w = s_k.dot(&w_factor);
                let mut m_k = w_factor.t().dot(&s_k_w);
                m_k *= lambda_base[idx];
                m_mats.push(m_k);
            }

            for kk in 0..n_base_penalties {
                for ll in 0..=kk {
                    let frob: f64 = m_mats[kk]
                        .iter()
                        .zip(m_mats[ll].iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    let mut val = -frob;
                    if kk == ll {
                        val += det1[kk];
                    }
                    det2[[kk, ll]] = val;
                    det2[[ll, kk]] = val;
                }
            }
        }

        // Penalty quadratic + ridge (same as compute_laml_at_convergence).
        let mut penalty_quadratic = 0.0;
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lk = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lk > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                let sb = s_k.dot(&state.beta_base);
                penalty_quadratic += lk * state.beta_base.dot(&sb);
            }
        }
        if p_link > 0 && link_penalty.nrows() == p_link && state.beta_link.len() == p_link {
            let sb = link_penalty.dot(&state.beta_link);
            penalty_quadratic += lambda_link * state.beta_link.dot(&sb);
        }
        if state.ridge_base_used > 0.0 {
            penalty_quadratic += state.ridge_base_used * state.beta_base.dot(&state.beta_base);
        }
        if p_link > 0 && state.beta_link.len() == p_link && state.ridge_link_used > 0.0 {
            penalty_quadratic += state.ridge_link_used * state.beta_link.dot(&state.beta_link);
        }

        // Build penalty roots in the JOINT basis.
        // Each base penalty Sₖ lives in [0..p_base, 0..p_base] of the joint space.
        // The link penalty lives in [p_base..p_total, p_base..p_total].
        let mut penalty_roots: Vec<Array2<f64>> = Vec::with_capacity(k);
        for rs in base_rs_list.iter() {
            // Embed base penalty root into joint space: [R_k | 0]
            penalty_roots.push(embed_penalty_root(rs, 0, p_base, p_total));
        }
        // Link penalty root (if present).
        if p_link > 0 && k > n_base_penalties {
            let link_root = penalty_matrix_root(&link_penalty).unwrap_or_else(|_| {
                // Fallback: identity-like root
                Array2::eye(p_link)
            });
            penalty_roots.push(embed_penalty_root(&link_root, p_base, p_base + p_link, p_total));
        }

        // Concatenated beta.
        let mut beta = Array1::zeros(p_total);
        beta.slice_mut(ndarray::s![..p_base])
            .assign(&state.beta_base);
        if p_link > 0 {
            beta.slice_mut(ndarray::s![p_base..p_total])
                .assign(&state.beta_link);
        }

        // Log-likelihood: for GLM, ll = -0.5 * deviance. For Gaussian, same.
        let log_likelihood = -0.5 * deviance;

        let dispersion = if is_gaussian {
            DispersionHandling::ProfiledGaussian
        } else {
            DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            }
        };

        // ═══════════════════════════════════════════════════════════════════
        //  Enrichment: build derivative provider for non-Gaussian corrections
        // ═══════════════════════════════════════════════════════════════════
        //
        // For non-Gaussian joint models, the Hessian derivative Ḣₖ = dH/dρₖ has
        // THREE components beyond Aₖ = λₖSₖ:
        //   1. Jacobian sensitivity: Ṫₖ'WJ + J'WṪₖ  (from β-dependent g')
        //   2. Weight sensitivity:   J'diag(ẇₖ)J     (from β-dependent W)
        //   3. Basis sensitivity:    (via dot_B in Ṫₖ)
        //
        // For Gaussian (identity link), all three are zero.
        // We build a LinkWiggleDerivProvider so the unified evaluator can
        // compute corrections on demand without model-specific knowledge.

        let deriv_provider: Box<dyn crate::estimate::reml::unified::HessianDerivativeProvider> =
            if p_link > 0 {
                // w_prime = dW/dη for each observation.
                //
                // For integrated logit (covariate_se active), the IRLS weight is
                //   W = obsw · d1² / var,  var = μ(1−μ), d1 = dE[sigmoid(η+ε)]/dη.
                // Its η-derivative uses GHQ derivatives d1, d2:
                //   dW/dη = obsw · [2·d1·d2/var − d1²·var'/(var²)]
                // where var' = d1·(1−2μ).
                //
                // For standard logit (no integration): d1 = μ(1−μ), d2 = d1·(1−2μ),
                // and the formula reduces to obsw · μ(1−μ)(1−2μ).
                let mut w_prime = Array1::<f64>::zeros(n);
                let mut w_double_prime = Array1::<f64>::zeros(n);
                if matches!(state.link, LinkFunction::Logit) {
                    let use_integrated = state.covariate_se.is_some();
                    for i in 0..n {
                        let p_i = mu[i].clamp(1e-10, 1.0 - 1e-10);
                        let w_base = p_i * (1.0 - p_i);
                        if use_integrated {
                            let d1 = integrated_d1[i].abs().max(1e-8);
                            let d2 = integrated_d2[i];
                            let var = w_base.max(1e-12);
                            let var_prime = d1 * (1.0 - 2.0 * p_i);
                            let dw_deta =
                                (2.0 * d1 * d2) / var - (d1 * d1) * (var_prime / (var * var));
                            w_prime[i] = state.weights[i] * dw_deta;
                        } else {
                            w_prime[i] = state.weights[i] * w_base * (1.0 - 2.0 * p_i);
                        }
                        w_double_prime[i] =
                            state.weights[i] * w_base * ((1.0 - 2.0 * p_i).powi(2) - 2.0 * w_base);
                        if !w_prime[i].is_finite() {
                            w_prime[i] = 0.0;
                        }
                        if !w_double_prime[i].is_finite() {
                            w_double_prime[i] = 0.0;
                        }
                    }
                }

                // Basis derivatives for Jacobian sensitivity.
                let Some(knot_vector) = state.knot_vector.as_ref() else {
                    return Err(EstimationError::RemlOptimizationFailed(
                        "missing knot vector for joint builder enrichment".to_string(),
                    ));
                };
                let Some(link_transform) = state.link_transform.as_ref() else {
                    return Err(EstimationError::RemlOptimizationFailed(
                        "missing link transform for joint builder enrichment".to_string(),
                    ));
                };
                let n_raw = knot_vector.len().saturating_sub(state.degree + 1);
                let (_, _, rangewidth) = state.standardized_z(&u);
                let invrw = 1.0 / rangewidth;
                let z_c = {
                    let (_, zc, _) = state.standardized_z(&u);
                    zc
                };

                let (b_prime_arc, _) = create_basis::<Dense>(
                    z_c.view(),
                    KnotSource::Provided(knot_vector.view()),
                    state.degree,
                    BasisOptions::first_derivative(),
                )
                .map_err(|e| EstimationError::InvalidSpecification(e.to_string()))?;
                let b_prime = b_prime_arc.as_ref().clone();

                let j_mat_arc = std::sync::Arc::new(j_mat.clone());
                let jweighted_arc = std::sync::Arc::new(jweighted.clone());
                let x_base_arc = std::sync::Arc::new(state.x_base.to_owned());

                Box::new(LinkWiggleDerivProvider {
                    j_mat: j_mat_arc,
                    jweighted: jweighted_arc,
                    sqrtw: sqrtw.clone(),
                    w_prime,
                    w_double_prime,
                    x_base: x_base_arc,
                    gsecond: gsecond.clone(),
                    b_prime_u: b_prime_u.clone(),
                    b_prime,
                    link_transform: link_transform.clone(),
                    invrw,
                    p_base,
                    p_link,
                    p_total,
                    n,
                    n_raw,
                })
            } else {
                use crate::estimate::reml::unified::GaussianDerivatives;
                Box::new(GaussianDerivatives)
            };

        // Firth bias reduction in joint models with link wiggles is handled
        // through the standard (non-joint) path; this joint builder path
        // uses the deriv_provider for non-Gaussian corrections only.

        let builder = InnerSolutionBuilder::new(
            log_likelihood,
            penalty_quadratic,
            beta,
            n,
            Box::new(hop),
            penalty_roots,
            PenaltyLogdetDerivs {
                value: log_det_s,
                first: det1,
                second: Some(det2),
            },
            dispersion,
        )
        .nullspace_dim_override(mp)
        .deriv_provider(deriv_provider);
        let inner_solution = builder.build();

        // Compute EDF for diagnostics.
        let edf = Self::compute_joint_edf(
            state,
            &bwiggle,
            &g_prime,
            &weights,
            lambda_base,
            lambda_link,
        );

        Ok((inner_solution, edf))
    }

    fn compute_joint_edf(
        state: &JointModelState,
        bwiggle: &Array2<f64>,
        g_prime: &Array1<f64>,
        weights: &Array1<f64>,
        lambda_base: &Array1<f64>,
        lambda_link: f64,
    ) -> Option<f64> {
        use crate::faer_ndarray::FaerCholesky;
        use faer::Side;

        let n = state.nobs();
        let p_base = state.x_base.ncols();
        let p_link = bwiggle.ncols();
        let coef_layout = JointCoefLayout::new(p_base, p_link);
        let p_total = coef_layout.p_total();
        if p_total == 0 {
            return Some(0.0);
        }

        let mut w_eff = Array1::<f64>::zeros(n);
        for i in 0..n {
            w_eff[i] = weights[i] * g_prime[i] * g_prime[i];
        }
        let mut a_mat = fast_xt_diag_x(&state.x_base, &w_eff);
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                a_mat.scaled_add(lambda_k, s_k);
            }
        }

        let wg = weights * g_prime;
        let c_mat = fast_xt_diag_y(&state.x_base, &wg, bwiggle);

        let mut d_mat = fast_xt_diag_x(bwiggle, weights);
        let link_penalty = state.build_link_penalty();
        if link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
            d_mat.scaled_add(lambda_link, &link_penalty);
        }

        ensure_positive_definite_joint(&mut a_mat);
        ensure_positive_definite_joint(&mut d_mat);

        let mut h_mat = Array2::<f64>::zeros((p_total, p_total));
        for i in 0..p_base {
            for j in 0..p_base {
                h_mat[[i, j]] = a_mat[[i, j]];
            }
        }
        for i in 0..p_base {
            for j in 0..p_link {
                h_mat[[i, p_base + j]] = c_mat[[i, j]];
                h_mat[[p_base + j, i]] = c_mat[[i, j]];
            }
        }
        for i in 0..p_link {
            for j in 0..p_link {
                h_mat[[p_base + i, p_base + j]] = d_mat[[i, j]];
            }
        }

        let mut s_lambda = Array2::<f64>::zeros((p_total, p_total));
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                s_lambda
                    .slice_mut(s![..p_base, ..p_base])
                    .scaled_add(lambda_k, s_k);
            }
        }
        if link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
            s_lambda
                .slice_mut(s![p_base.., p_base..])
                .scaled_add(lambda_link, &link_penalty);
        }

        let chol = h_mat.cholesky(Side::Lower).ok()?;
        let mut trace = 0.0;
        for j in 0..p_total {
            let col = s_lambda.column(j).to_owned();
            let solved = chol.solvevec(&col);
            trace += solved[j];
        }

        Some(p_total as f64 - trace)
    }

    /// Extract final result after optimization
    pub fn into_result(mut self, include_base_covariance: bool) -> JointModelResult {
        let cached_edf = self.eval.cached_edf;
        let cached_iters = self.eval.last_backfit_iterations;
        let cached_converged = self.eval.last_converged;
        let rho = self.core.state.rho.clone();
        let knot_range = self.core.state.knot_range.unwrap_or((0.0, 1.0));
        let knot_vector = self
            .core
            .state
            .knot_vector
            .clone()
            .unwrap_or_else(|| Array1::zeros(0));
        let u = self.core.state.base_linear_predictor();
        let bwiggle = self.core.state.build_link_basis_from_state(&u);
        let eta = self.core.state.compute_eta_full(&u, &bwiggle);
        let deviance = self.core.state.recompute_deviance_from_eta(&eta);
        let outer_gradient_norm = self
            .compute_unified_eval(&rho)
            .ok()
            .map(|eval| eval.gradient.dot(&eval.gradient).sqrt())
            .filter(|v| v.is_finite())
            .unwrap_or(f64::NAN);

        // Re-bind state for the remainder of this method (immutable borrow is fine
        // since compute_unified_eval is done).
        let state = &self.core.state;

        // Compute base-coefficient covariance: (X' W_eff X + S_λ + δI)^{-1}
        // where W_eff = w_glm * g_prime^2, accounting for the chain rule through
        // the link wiggle.
        let beta_base_covariance = include_base_covariance
            .then(|| {
                (|| -> Option<Array2<f64>> {
                    use crate::faer_ndarray::FaerCholesky;
                    use faer::Side;

                    let n = state.nobs();
                    let p = state.x_base.ncols();
                    if p == 0 {
                        return None;
                    }

                    // Compute GLM working weights at converged eta
                    let mut mu = Array1::<f64>::zeros(n);
                    let mut w_glm = Array1::<f64>::zeros(n);
                    let mut z_glm = Array1::<f64>::zeros(n);
                    crate::pirls::update_glmvectors(
                        state.y,
                        &eta,
                        &InverseLink::Standard(state.link.clone()),
                        state.weights,
                        &mut mu,
                        &mut w_glm,
                        &mut z_glm,
                        None,
                    )
                    .ok()?;

                    // Compute link derivative g'(u) = 1 + B'(u) · θ
                    let g_prime = compute_link_derivative_from_state(state, &u, &bwiggle);

                    // Effective weights: w_eff_i = w_glm_i * g'(u_i)^2
                    let w_eff = &w_glm * &(&g_prime * &g_prime);

                    let mut h = fast_xt_diag_x(&state.x_base, &w_eff);

                    // Add penalty: S_λ = Σ λ_k S_k
                    let penalty_layout = JointPenaltyLayout::for_state(state);
                    let (lambda_base, _) = penalty_layout.lambdas(&rho);
                    for (idx, s_k) in state.s_base.iter().enumerate() {
                        let lam = lambda_base.get(idx).cloned().unwrap_or(0.0);
                        if s_k.nrows() == p && s_k.ncols() == p && lam > 0.0 {
                            h.scaled_add(lam, s_k);
                        }
                    }

                    // Add ridge stabilization matching what was used in the fit
                    if state.ridge_base_used > 0.0 {
                        for i in 0..p {
                            h[[i, i]] += state.ridge_base_used;
                        }
                    }

                    // Invert via Cholesky
                    match h.cholesky(Side::Lower) {
                        Ok(chol) => {
                            let mut cov = Array2::<f64>::eye(p);
                            chol.solve_mat_in_place(&mut cov);
                            // Verify all finite
                            if cov.iter().all(|v: &f64| v.is_finite()) {
                                Some(cov)
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    }
                })()
            })
            .flatten();

        let state = self.core.state;
        JointModelResult {
            beta_base: state.beta_base,
            beta_link: state.beta_link,
            lambdas: rho.mapv(f64::exp).to_vec(),
            deviance,
            edf: cached_edf.unwrap_or(f64::NAN),
            backfit_iterations: cached_iters,
            converged: cached_converged,
            outer_gradient_norm,
            knot_range,
            knot_vector,
            link_transform: state
                .link_transform
                .unwrap_or_else(|| Array2::eye(state.n_constrained_basis)),
            degree: state.degree,
            link: state.link.clone(),
            s_link_constrained: state.s_link_constrained.unwrap_or_else(|| {
                Array2::zeros((state.n_constrained_basis, state.n_constrained_basis))
            }),
            ridge_used: state.ridge_used,
            beta_base_covariance,
        }
    }
}

/// Fit joint model with proper REML-based lambda selection via BFGS
///
/// Uses Laplace approximate marginal likelihood (LAML) with numerical gradient.
/// For nonlinear g(u), the Hessian is Gauss-Newton (approximate).
pub(crate) fn fit_joint_modelwith_reml<'a>(
    y: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    x_base: ArrayView2<'a, f64>,
    s_base: Vec<Array2<f64>>,
    layout_base: EngineDims,
    link: LinkFunction,
    config: &JointModelConfig,
    covariate_se: Option<Array1<f64>>,
    include_base_covariance: bool,
) -> Result<JointModelResult, EstimationError> {
    if matches!(link, LinkFunction::Sas | LinkFunction::BetaLogistic) {
        return Err(EstimationError::InvalidSpecification(
            "joint REML path does not support state-less SAS/Beta-Logistic links; explicit state plumbing is required"
                .to_string(),
        ));
    }
    // Library code must not own terminal UI lifecycle implicitly.
    // Keep visualization disabled unless an explicit caller-provided session is wired in.
    let mut visualizer_session = visualizer::VisualizerSession::new(true);
    visualizer_session.set_stage("joint", "initializing");
    if config.firth_bias_reduction && matches!(link, LinkFunction::Logit) {
        visualizer_session.push_diagnostic("firth bias reduction enabled (separation protection)");
    }
    let quadctx = QuadratureContext::new();

    // Create REML state
    let mut reml_state = JointRemlState::new(
        y,
        weights,
        x_base,
        s_base,
        layout_base,
        link,
        config,
        covariate_se,
        quadctx,
    );
    reml_state.visualizer = visualizer_session;

    let n_base = reml_state.core.state.s_base.len();
    let heuristic_lambda = {
        let state = &reml_state.core.state;
        state
            .knot_vector
            .as_ref()
            .map(|knots| baseline_lambda_seed(knots, state.degree, 2))
    };
    let heuristic_lambdas = heuristic_lambda.map(|lambda| vec![lambda; n_base + 1]);
    let n_params = n_base + 1;
    let snapshot = JointRemlSnapshot::new(&reml_state);
    let outer_config = OuterConfig {
        tolerance: config.reml_tol,
        max_iter: config.max_reml_iter,
        fd_step: 1e-4,
        seed_config: SeedConfig {
            bounds: (-12.0, 12.0),
            max_seeds: if n_base < 4 { 12 } else { 16 },
            screening_budget: if n_base < 2 {
                2
            } else if n_base < 6 {
                3
            } else {
                4
            },
            screen_max_inner_iterations: 5,
            risk_profile: seed_risk_profile_for_joint_link(link),
            num_auxiliary_trailing: 0,
        },
        rho_bound: 30.0,
        heuristic_lambdas: heuristic_lambdas,
        initial_rho: None,
        fallback_sequence: Vec::new(),
    };

    let mut obj = ClosureObjective {
        state: reml_state,
        cap: OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params,
            // Joint models have both rho (penalty) and potentially psi (design-moving)
            // coordinates. Conservatively false.
            all_penalty_like: false,
            barrier_config: None,
            force_solver: None,
        },
        cost_fn: |state: &mut JointRemlState<'_>, rho: &Array1<f64>| state.compute_cost(rho),
        eval_fn: |state: &mut JointRemlState<'_>, rho: &Array1<f64>| {
            // Unified path: inner solve + enriched InnerSolution + cost/gradient/Hessian
            // all through the single reml_laml_evaluate function.
            state.compute_unified_eval(rho)
        },
        reset_fn: Some({
            let snap = snapshot;
            move |state: &mut JointRemlState<'_>| {
                snap.restore(state);
            }
        }),
        efs_fn: Some(|state: &mut JointRemlState<'_>, rho: &Array1<f64>| {
            state.compute_efs_eval(rho)
        }),
    };

    let result =
        crate::solver::strategy::run_outer(&mut obj, &outer_config, "joint flexible link")?;

    // Extract state from the objective wrapper to finalize.
    let ClosureObjective { mut state, .. } = obj;
    state.compute_cost(&result.rho)?;
    Ok(state.into_result(include_base_covariance))
}

/// Prediction result from joint model
pub struct JointModelPrediction {
    /// Calibrated linear predictor η_cal
    pub eta: Array1<f64>,
    /// Probabilities (posterior predictive mean if SE available)
    pub probabilities: Array1<f64>,
    /// Effective SE after derivative propagation (|g'(η)| × SE_base)
    pub effective_se: Option<Array1<f64>>,
}

/// Compute derivative of link function g'(u) = 1 + B'(z)·θ · dz/du
/// using finite differences on the spline basis (O(n * p_link)).
fn compute_link_derivative_from_state(
    state: &JointModelState,
    u: &Array1<f64>,
    bwiggle: &Array2<f64>,
) -> Array1<f64> {
    use crate::basis::evaluate_bspline_derivative_scalar_into;
    use crate::basis::internal::BsplineScratch;

    let n = u.len();
    let mut deriv = Array1::<f64>::ones(n);

    if bwiggle.ncols() == 0 || state.beta_link.is_empty() {
        return deriv;
    }
    let Some(knot_vector) = state.knot_vector.as_ref() else {
        return deriv;
    };

    let (_, z_c, rangewidth) = state.standardized_z(u);
    let n_raw = knot_vector.len().saturating_sub(state.degree + 1);

    // Get link_transform, return early if not set
    let Some(ref link_transform) = state.link_transform else {
        return deriv;
    };
    let n_constrained = link_transform.ncols();
    if n_raw == 0 || n_constrained == 0 || state.beta_link.len() != n_constrained {
        return deriv;
    }

    // Pre-allocate all buffers outside loop (zero-allocation, same as HMC)
    let mut deriv_raw = vec![0.0; n_raw];
    let num_basis_lower = knot_vector.len().saturating_sub(state.degree);
    let mut lower_basis = vec![0.0; num_basis_lower];
    let mut lower_scratch = BsplineScratch::new(state.degree.saturating_sub(1));

    for i in 0..n {
        let z_i = z_c[i];
        deriv_raw.fill(0.0);
        if evaluate_bspline_derivative_scalar_into(
            z_i,
            knot_vector.view(),
            state.degree,
            &mut deriv_raw,
            &mut lower_basis,
            &mut lower_scratch,
        )
        .is_err()
        {
            continue;
        }

        // d(wiggle)/dz = B'(z) @ Z @ θ
        let dwiggle_dz: f64 = if link_transform.nrows() == n_raw {
            (0..n_constrained)
                .map(|c| {
                    let b_prime_c: f64 = (0..n_raw)
                        .map(|r| deriv_raw[r] * link_transform[[r, c]])
                        .sum();
                    b_prime_c * state.beta_link[c]
                })
                .sum()
        } else {
            0.0
        };

        deriv[i] = 1.0 + dwiggle_dz / rangewidth;
    }

    deriv
}

fn compute_link_derivative_terms_from_state(
    state: &JointModelState,
    u: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    use crate::basis::evaluate_bspline_derivative_scalar_into;
    use crate::basis::evaluate_bsplinesecond_derivative_scalar_into;
    use crate::basis::internal::BsplineScratch;

    let n = u.len();
    let p_link = state.beta_link.len();
    let mut g_prime = Array1::<f64>::ones(n);
    let mut gsecond = Array1::<f64>::zeros(n);
    let mut b_prime_u = Array2::<f64>::zeros((n, p_link));

    if p_link == 0 {
        return (g_prime, gsecond, b_prime_u);
    }
    let Some(knot_vector) = state.knot_vector.as_ref() else {
        return (g_prime, gsecond, b_prime_u);
    };
    let Some(link_transform) = state.link_transform.as_ref() else {
        return (g_prime, gsecond, b_prime_u);
    };

    let n_raw = knot_vector.len().saturating_sub(state.degree + 1);
    if n_raw == 0 || link_transform.nrows() != n_raw || link_transform.ncols() != p_link {
        return (g_prime, gsecond, b_prime_u);
    }

    let (z_raw, z_c, rangewidth) = state.standardized_z(u);
    let invrw = 1.0 / rangewidth;
    let invrw2 = invrw * invrw;

    let mut deriv_raw = vec![0.0; n_raw];
    let num_basis_lower = knot_vector.len().saturating_sub(state.degree);
    let mut lower_basis = vec![0.0; num_basis_lower];
    let mut lower_scratch = BsplineScratch::new(state.degree.saturating_sub(1));

    let mut second_raw = vec![0.0; n_raw];
    let num_basis_lowersecond = knot_vector.len().saturating_sub(state.degree - 1);
    let mut deriv_lower = vec![0.0; num_basis_lowersecond.saturating_sub(1)];
    let mut lower_basissecond = vec![0.0; num_basis_lowersecond];
    let mut lower_scratchsecond = BsplineScratch::new(state.degree.saturating_sub(2));

    for i in 0..n {
        let z_i = z_c[i];

        deriv_raw.fill(0.0);
        if evaluate_bspline_derivative_scalar_into(
            z_i,
            knot_vector.view(),
            state.degree,
            &mut deriv_raw,
            &mut lower_basis,
            &mut lower_scratch,
        )
        .is_err()
        {
            continue;
        }

        let mut dwiggle_dz = 0.0;
        for c in 0..p_link {
            let mut b_prime_c = 0.0;
            for r in 0..n_raw {
                b_prime_c += deriv_raw[r] * link_transform[[r, c]];
            }
            b_prime_u[[i, c]] = b_prime_c * invrw;
            dwiggle_dz += b_prime_c * state.beta_link[c];
        }
        g_prime[i] = 1.0 + dwiggle_dz * invrw;

        // For the linearly-extended basis, the curvature is zero outside [0, 1].
        // That is, d²/dz_raw² B_ext(z_raw) = 0 when z_raw != z_c.
        if (z_raw[i] - z_c[i]).abs() <= 1e-12 {
            second_raw.fill(0.0);
            if evaluate_bsplinesecond_derivative_scalar_into(
                z_i,
                knot_vector.view(),
                state.degree,
                &mut second_raw,
                &mut deriv_lower,
                &mut lower_basissecond,
                &mut lower_scratchsecond,
            )
            .is_err()
            {
                continue;
            }
            let mut d2wiggle_dz2 = 0.0;
            for c in 0..p_link {
                let mut bsecond_c = 0.0;
                for r in 0..n_raw {
                    bsecond_c += second_raw[r] * link_transform[[r, c]];
                }
                d2wiggle_dz2 += bsecond_c * state.beta_link[c];
            }
            gsecond[i] = d2wiggle_dz2 * invrw2;
        }
    }

    (g_prime, gsecond, b_prime_u)
}

fn compute_link_derivative_from_result(
    result: &JointModelResult,
    eta_base: &Array1<f64>,
    bwiggle: &Array2<f64>,
) -> Array1<f64> {
    use crate::basis::evaluate_bspline_derivative_scalar_into;
    use crate::basis::internal::BsplineScratch;

    let n = eta_base.len();
    let mut deriv = Array1::<f64>::ones(n);
    if bwiggle.ncols() == 0 || result.beta_link.is_empty() {
        return deriv;
    }

    let (min_u, max_u) = result.knot_range;
    let rangewidth = (max_u - min_u).max(1e-6);
    let z_raw: Array1<f64> = eta_base.mapv(|u| (u - min_u) / rangewidth);
    let z_c: Array1<f64> = z_raw.mapv(|z| z.clamp(0.0, 1.0));
    let n_raw = result.knot_vector.len().saturating_sub(result.degree + 1);
    let n_constrained = result.link_transform.ncols();
    if n_raw == 0 || n_constrained == 0 || result.beta_link.len() != n_constrained {
        return deriv;
    }

    // Pre-allocate all buffers outside loop (zero-allocation, same as HMC)
    let mut deriv_raw = vec![0.0; n_raw];
    let num_basis_lower = result.knot_vector.len().saturating_sub(result.degree);
    let mut lower_basis = vec![0.0; num_basis_lower];
    let mut lower_scratch = BsplineScratch::new(result.degree.saturating_sub(1));

    for i in 0..n {
        let z_i = z_c[i];
        deriv_raw.fill(0.0);
        if evaluate_bspline_derivative_scalar_into(
            z_i,
            result.knot_vector.view(),
            result.degree,
            &mut deriv_raw,
            &mut lower_basis,
            &mut lower_scratch,
        )
        .is_err()
        {
            continue;
        }

        // d(wiggle)/dz = B'(z) @ Z @ θ
        let dwiggle_dz: f64 = if result.link_transform.nrows() == n_raw {
            (0..n_constrained)
                .map(|c| {
                    let b_prime_c: f64 = (0..n_raw)
                        .map(|r| deriv_raw[r] * result.link_transform[[r, c]])
                        .sum();
                    b_prime_c * result.beta_link[c]
                })
                .sum()
        } else {
            0.0
        };

        deriv[i] = 1.0 + dwiggle_dz / rangewidth;
    }

    deriv
}

/// Predict probabilities from a fitted joint model
///
/// Uses stored knot_range and B-spline basis for consistent prediction.
/// For SE propagation, uses derivative-propagated uncertainty.
pub fn predict_joint(
    result: &JointModelResult,
    eta_base: &Array1<f64>,
    se_base: Option<&Array1<f64>>,
) -> Result<JointModelPrediction, EstimationError> {
    joint_prediction_supported(result.link)?;
    let n = eta_base.len();

    // Use stored knot range from training for consistent standardization
    let (min_u, max_u) = result.knot_range;
    let rangewidth = (max_u - min_u).max(1e-6);

    // Standardize and apply the same linear extension used during training.
    let z_raw: Array1<f64> = eta_base.mapv(|u| (u - min_u) / rangewidth);
    let z_c: Array1<f64> = z_raw.mapv(|z| z.clamp(0.0, 1.0));

    // Build B-spline basis at prediction points using stored parameters
    let bwiggle = match create_basis::<Dense>(
        z_c.view(),
        KnotSource::Provided(result.knot_vector.view()),
        result.degree,
        BasisOptions::value(),
    ) {
        Ok((basis, _)) => {
            let mut raw = basis.as_ref().clone();
            let _ = apply_linear_extension_from_first_derivative(
                z_raw.view(),
                z_c.view(),
                result.knot_vector.view(),
                result.degree,
                &mut raw,
            );
            if result.link_transform.ncols() > 0 && result.link_transform.nrows() == raw.ncols() {
                raw.dot(&result.link_transform)
            } else {
                Array2::zeros((n, result.beta_link.len()))
            }
        }
        Err(_) => Array2::zeros((n, result.beta_link.len())),
    };

    // Compute η_cal = u + Bwiggle · θ
    let eta_cal: Array1<f64> = if bwiggle.ncols() > 0 && result.beta_link.len() == bwiggle.ncols() {
        let wiggle = bwiggle.dot(&result.beta_link);
        eta_base + &wiggle
    } else {
        eta_base.clone()
    };

    // Compute effective SE if base SE provided
    let quadctx = QuadratureContext::new();
    let (probabilities, effective_se) = if let Some(se) = se_base {
        // Compute link derivative for uncertainty propagation
        let deriv = compute_link_derivative_from_result(result, eta_base, &bwiggle);

        // Effective SE = |g'(η)| × SE_base
        let eff_se: Array1<f64> = deriv.mapv(f64::abs) * se;

        let probs = if let Some(family) = integrated_binomial_family_from_link(result.link) {
            let strategy = strategy_for_family(family.into(), None);
            (0..n)
                .map(|i| {
                    strategy
                        .integrated_moments(&quadctx, eta_cal[i], eff_se[i])
                        .map(|m| m.mean)
                        .unwrap_or_else(|_| joint_point_inverse_link(result.link, eta_cal[i]))
                })
                .collect::<Array1<f64>>()
        } else {
            eta_cal.mapv(|e| joint_point_inverse_link(result.link, e))
        };

        (probs, Some(eff_se))
    } else {
        let probs = if let Some(family) = integrated_binomial_family_from_link(result.link) {
            let strategy = strategy_for_family(family.into(), None);
            (0..n)
                .map(|i| {
                    strategy
                        .integrated_moments(&quadctx, eta_cal[i], 0.0)
                        .map(|m| m.mean)
                        .unwrap_or_else(|_| joint_point_inverse_link(result.link, eta_cal[i]))
                })
                .collect::<Array1<f64>>()
        } else {
            eta_cal.mapv(|e| joint_point_inverse_link(result.link, e))
        };
        (probs, None)
    };

    Ok(JointModelPrediction {
        eta: eta_cal,
        probabilities,
        effective_se,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn test_joint_model_state_creation() {
        let n = 100;
        let p = 10;
        let y = Array1::zeros(n);
        let weights = Array1::ones(n);
        let x = Array2::zeros((n, p));
        let s = vec![Array2::eye(p)];
        let layout = EngineDims::new(p, 1);
        let config = JointModelConfig::default();
        // Use default knot count from config.
        let quadctx = QuadratureContext::new();

        let state = JointModelState::new(
            y.view(),
            weights.view(),
            x.view(),
            s,
            layout,
            LinkFunction::Logit,
            &config,
            quadctx,
        );

        assert_eq!(state.beta_base.len(), p);
        assert_eq!(state.beta_link.len(), config.n_link_knots + 2);
    }

    #[test]
    fn test_predict_joint_basic() {
        // Create a simple result with logit link (no wiggle)
        let n_knots = 5;
        let degree = 3;
        let (basis_arc, knot_vector) = create_basis::<Dense>(
            Array1::from_vec(vec![0.0]).view(),
            KnotSource::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: n_knots - degree - 1,
            },
            degree,
            BasisOptions::value(),
        )
        .expect("basis");
        let basis = (*basis_arc).clone();
        let num_basis = knot_vector.len().saturating_sub(degree + 1);
        assert_eq!(basis.ncols(), num_basis);
        let beta_link = Array1::zeros(num_basis);

        let result = JointModelResult {
            beta_base: Array1::zeros(10),
            beta_link,
            lambdas: vec![1.0],
            deviance: 0.0,
            edf: 5.0,
            backfit_iterations: 1,
            converged: true,
            outer_gradient_norm: 0.0,
            knot_range: (0.0, 1.0),
            knot_vector,
            link_transform: Array2::eye(num_basis),
            degree,
            link: LinkFunction::Logit,
            s_link_constrained: Array2::eye(num_basis),
            ridge_used: 0.0,
            beta_base_covariance: None,
        };

        // Test with base eta values
        let eta_base = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        // Predict without SE (should give sigmoid of eta)
        let pred = predict_joint(&result, &eta_base, None).expect("joint prediction");

        assert_eq!(pred.eta.len(), 5);
        assert_eq!(pred.probabilities.len(), 5);
        assert!(pred.effective_se.is_none());

        // Check probabilities are in [0, 1]
        for p in pred.probabilities.iter() {
            assert!(*p >= 0.0 && *p <= 1.0);
        }

        // With logit link, prob at eta=0 should be ~0.5
        assert!((pred.probabilities[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_predict_jointwith_se() {
        let n_knots = 5;
        let degree = 3;
        let (basis_arc, knot_vector) = create_basis::<Dense>(
            Array1::from_vec(vec![0.0]).view(),
            KnotSource::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: n_knots - degree - 1,
            },
            degree,
            BasisOptions::value(),
        )
        .expect("basis");
        let basis = (*basis_arc).clone();
        let num_basis = knot_vector.len().saturating_sub(degree + 1);
        assert_eq!(basis.ncols(), num_basis);
        let beta_link = Array1::zeros(num_basis);

        let result = JointModelResult {
            beta_base: Array1::zeros(10),
            beta_link,
            lambdas: vec![1.0],
            deviance: 0.0,
            edf: 5.0,
            backfit_iterations: 1,
            converged: true,
            outer_gradient_norm: 0.0,
            knot_range: (0.0, 1.0),
            knot_vector,
            link_transform: Array2::eye(num_basis),
            degree,
            link: LinkFunction::Logit,
            s_link_constrained: Array2::eye(num_basis),
            ridge_used: 0.0,
            beta_base_covariance: None,
        };

        let eta_base = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let se_base = Array1::from_vec(vec![0.5, 0.5, 0.5]);

        let pred = predict_joint(&result, &eta_base, Some(&se_base)).expect("joint prediction");

        assert!(pred.effective_se.is_some());
        let eff_se = pred.effective_se.unwrap();
        assert_eq!(eff_se.len(), 3);

        // With zero wiggle, g'(u)=1 so effective SE equals base SE.
        for i in 0..3 {
            assert!((eff_se[i] - se_base[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_predict_joint_rejects_stateless_sas() {
        let degree = 3;
        let knot_vector = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let num_basis = knot_vector.len().saturating_sub(degree + 1);
        let result = JointModelResult {
            beta_base: Array1::zeros(1),
            beta_link: Array1::zeros(num_basis),
            lambdas: vec![1.0],
            deviance: 0.0,
            edf: 0.0,
            backfit_iterations: 0,
            converged: false,
            outer_gradient_norm: 1.0,
            knot_range: (0.0, 1.0),
            knot_vector,
            link_transform: Array2::eye(num_basis),
            degree,
            link: LinkFunction::Sas,
            s_link_constrained: Array2::eye(num_basis),
            ridge_used: 0.0,
            beta_base_covariance: None,
        };
        let eta_base = Array1::zeros(2);
        match predict_joint(&result, &eta_base, None) {
            Ok(_) => panic!("stateless SAS must fail"),
            Err(err) => assert!(format!("{err}").contains("state-less SAS/Beta-Logistic")),
        }
    }

    #[test]
    fn test_joint_analytic_gradient_supports_integrated_logitweights() {
        let n = 120;
        let p = 6;
        let mut rng = StdRng::seed_from_u64(4242);
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = rng.random_range(-1.5..1.5);
            }
        }
        let mut beta_true = Array1::<f64>::zeros(p);
        for j in 0..p {
            beta_true[j] = rng.random_range(-0.8..0.8);
        }
        let eta = x.dot(&beta_true);
        let y = eta.mapv(|e| 1.0 / (1.0 + (-e).exp()));
        let weights = Array1::ones(n);
        let covariate_se = Array1::from_elem(n, 0.35);
        let s = vec![Array2::eye(p)];
        let layout = EngineDims::new(p, 1);
        let config = JointModelConfig::default();

        let mut reml_state = JointRemlState::new(
            y.view(),
            weights.view(),
            x.view(),
            s,
            layout,
            LinkFunction::Logit,
            &config,
            Some(covariate_se),
            QuadratureContext::new(),
        );
        {
            let state = &mut reml_state.core.state;
            state.beta_base = beta_true.clone();
            reml_state.eval.cached_beta_base = beta_true;
            state.knot_range = None;
            state.knot_vector = None;
            state.link_transform = None;
            state.s_link_constrained = None;
            state.geometric_link_transform = None;
            state.geometric_s_link_constrained = None;
            let u = state.base_linear_predictor();
            state.build_link_basis(&u).expect("link basis");
        }

        let rho = Array1::from_vec(vec![0.0, 4.0]);
        let eval = reml_state
            .compute_unified_eval(&rho)
            .expect("integrated logit unified eval");
        assert!(eval.gradient.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_into_result_preserves_integrated_deviance_path() {
        let n = 48;
        let p = 3;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            x[[i, 2]] = ((i as f64) / 7.0).sin();
        }
        let beta = Array1::from_vec(vec![0.25, -0.9, 0.55]);
        let eta = x.dot(&beta);
        let y = eta.mapv(|e| 1.0 / (1.0 + (-e).exp()));
        let weights = Array1::ones(n);
        let s = vec![Array2::eye(p)];
        let layout = EngineDims::new(p, 1);
        let mut reml_state = JointRemlState::new(
            y.view(),
            weights.view(),
            x.view(),
            s,
            layout,
            LinkFunction::Logit,
            &JointModelConfig::default(),
            Some(Array1::from_elem(n, 0.4)),
            QuadratureContext::new(),
        );
        reml_state.core.state.beta_base = beta;
        let u = reml_state.core.state.base_linear_predictor();
        reml_state
            .core
            .state
            .build_link_basis(&u)
            .expect("link basis");
        let eta_full = reml_state
            .core
            .state
            .compute_eta_full(&u, &reml_state.core.state.build_link_basis_from_state(&u));
        let expected = reml_state.core.state.recompute_deviance_from_eta(&eta_full);
        let result = reml_state.into_result(true);
        assert!(
            (result.deviance - expected).abs() <= 1e-10,
            "integrated final deviance drifted: got {}, expected {}",
            result.deviance,
            expected
        );
    }

    #[test]
    fn test_joint_seed_profile_matches_link_family() {
        assert_eq!(
            seed_risk_profile_for_joint_link(LinkFunction::Identity),
            SeedRiskProfile::Gaussian
        );
        for link in [
            LinkFunction::Logit,
            LinkFunction::Probit,
            LinkFunction::CLogLog,
        ] {
            assert_eq!(
                seed_risk_profile_for_joint_link(link),
                SeedRiskProfile::GeneralizedLinear
            );
        }
    }

    #[test]
    fn test_joint_firth_penalized_hat_diag_smaller_than_unpenalized() {
        let weighted_joint_design = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.2, -0.1, //
                1.0, 0.8, 0.4, //
                1.0, -0.7, 0.3, //
                1.0, 1.2, -0.6, //
                1.0, -0.3, 0.9, //
            ],
        )
        .expect("shape");
        let p = weighted_joint_design.ncols();
        let penaltyzero = Array2::<f64>::zeros((p, p));
        let mut penalty_strong = Array2::<f64>::zeros((p, p));
        penalty_strong[[1, 1]] = 15.0;
        penalty_strong[[2, 2]] = 20.0;

        let unpen =
            JointModelState::compute_joint_firth_adjustment(&weighted_joint_design, &penaltyzero)
                .expect("unpenalized");
        let pen = JointModelState::compute_joint_firth_adjustment(
            &weighted_joint_design,
            &penalty_strong,
        )
        .expect("penalized");

        assert!(
            pen.hat_diag
                .iter()
                .zip(unpen.hat_diag.iter())
                .any(|(p_i, u_i)| *p_i < *u_i - 1e-10),
            "penalized hat diagonal should be smaller for at least one row"
        );
        assert!(pen.hat_diag.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(pen.half_log_det.is_finite());
    }

    #[test]
    fn test_joint_firth_fullhessian_differs_from_blockwise_approxwhen_coupled() {
        use crate::faer_ndarray::FaerCholesky;

        let n = 6;
        let p_base = 2;
        let p_link = 2;
        let mut xw = Array2::<f64>::zeros((n, p_base));
        let mut bw = Array2::<f64>::zeros((n, p_link));
        for i in 0..n {
            let t = i as f64 / (n as f64);
            xw[[i, 0]] = 1.0 + 0.1 * t;
            xw[[i, 1]] = 0.4 + t;
            // Deliberately coupled with X so X'WB is non-zero.
            bw[[i, 0]] = 0.7 * xw[[i, 0]] + 0.2 * xw[[i, 1]];
            bw[[i, 1]] = -0.2 * xw[[i, 0]] + 0.9 * xw[[i, 1]];
        }
        let coef_layout = JointCoefLayout::new(p_base, p_link);
        let mut weighted_joint_design = Array2::<f64>::zeros((n, coef_layout.p_total()));
        weighted_joint_design
            .slice_mut(s![.., ..p_base])
            .assign(&xw);
        weighted_joint_design
            .slice_mut(s![.., p_base..])
            .assign(&bw);

        let mut penalty = Array2::<f64>::zeros((coef_layout.p_total(), coef_layout.p_total()));
        penalty[[0, 0]] = 1.0;
        penalty[[1, 1]] = 2.0;
        penalty[[2, 2]] = 3.0;
        penalty[[3, 3]] = 4.0;

        let full =
            JointModelState::compute_joint_firth_adjustment(&weighted_joint_design, &penalty)
                .expect("full");

        // Blockwise approximation drops cross block terms.
        let mut h_bb = fast_ata(&xw);
        h_bb[[0, 0]] += penalty[[0, 0]];
        h_bb[[1, 1]] += penalty[[1, 1]];
        ensure_positive_definite_joint(&mut h_bb);
        let chol_bb = h_bb.cholesky(Side::Lower).expect("chol bb");
        let rhs_bb = chol_bb.solve_mat(&xw.t().to_owned());

        let mut h_ll = fast_ata(&bw);
        h_ll[[0, 0]] += penalty[[2, 2]];
        h_ll[[1, 1]] += penalty[[3, 3]];
        ensure_positive_definite_joint(&mut h_ll);
        let chol_ll = h_ll.cholesky(Side::Lower).expect("chol ll");
        let rhs_ll = chol_ll.solve_mat(&bw.t().to_owned());

        let mut blockwise_hat = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..p_base {
                acc += xw[[i, j]] * rhs_bb[[j, i]];
            }
            for j in 0..p_link {
                acc += bw[[i, j]] * rhs_ll[[j, i]];
            }
            blockwise_hat[i] = acc;
        }

        let max_diff = full
            .hat_diag
            .iter()
            .zip(blockwise_hat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-6,
            "full joint leverage should differ from blockwise approximation when X'WB != 0"
        );
    }

    #[test]
    fn joint_cloglog_inverse_link_loses_negative_tail_mass() {
        let eta = -50.0_f64;
        let stable = -(-(eta.exp())).exp_m1();
        assert!(stable > 0.0);
        let got = joint_point_inverse_link(LinkFunction::CLogLog, eta);
        assert!(
            (got - stable).abs() < 1e-30,
            "joint cloglog inverse-link should equal -expm1(-exp(eta)) in the negative tail at eta={eta}; got {} vs {}",
            got,
            stable
        );
    }
}
