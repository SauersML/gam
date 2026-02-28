//!
//! This module implements a unified model where the base linear predictor and
//! the flexible link correction are fitted jointly in one REML optimization.
//!
//! Architecture:
//!   η = g(Xβ) where g(u) = u + wiggle(u)
//!
//! Where:
//! - Xβ: High-dimensional predictors (high-dimensional predictors) with ridge penalty
//! - g(·): Flexible 1D link correction with scale anchor (g(u) = u + B(u)θ)
//!
//! The algorithm:
//! - Outer: Newton trust-region over ρ = [log(λ_base), log(λ_link)]
//! - Inner: Alternating (g|β, β|g with g'(u)*X design)
//! - LAML cost computed via logdet of joint Gauss-Newton Hessian

use crate::basis::{
    BasisOptions, Dense, KnotSource, apply_linear_extension_from_first_derivative,
    baseline_lambda_seed, compute_geometric_constraint_transform, create_basis,
    create_difference_penalty_matrix, penalty_greville_abscissae_for_knots,
};
use crate::construction::{
    EngineDims, ReparamInvariant, ReparamResult, compute_penalty_square_roots,
    precompute_reparam_invariant, stable_reparameterization_engine,
    stable_reparameterization_with_invariant_engine,
};
use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerEigh, fast_ab, fast_ata, fast_atb, fast_atv};
use crate::probability::{normal_cdf_approx, normal_pdf};
use crate::quadrature::QuadratureContext;
use crate::seeding::{SeedConfig, SeedRiskProfile, generate_rho_candidates};
use crate::types::LinkFunction;
use crate::visualizer;
use faer::Side;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use wolfe_bfgs::{
    BfgsSolution, NewtonTrustRegion, ObjectiveEvalError, ObjectiveRequest, ObjectiveSample,
};

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
// FD audit policy for analytic gradient checks.
// Keep the guardrail, but sample it periodically to avoid O(K * FD refinements)
// blow-ups in high-dimensional smoothing problems.
const JOINT_GRAD_AUDIT_CLAMP_FRAC: f64 = 0.90;
const JOINT_GRAD_AUDIT_WARMUP_EVALS: usize = 5;
const JOINT_GRAD_AUDIT_INTERVAL: usize = 20;

#[inline]
fn should_sample_joint_fd_audit(eval_num: usize) -> bool {
    eval_num <= JOINT_GRAD_AUDIT_WARMUP_EVALS
        || (JOINT_GRAD_AUDIT_INTERVAL > 0 && eval_num % JOINT_GRAD_AUDIT_INTERVAL == 0)
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
    /// When present, uses update_glm_vectors_integrated for uncertainty-aware fitting.
    covariate_se: Option<Array1<f64>>,
    quad_ctx: QuadratureContext,
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
        quad_ctx: QuadratureContext,
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
            quad_ctx,
            firth_bias_reduction: config.firth_bias_reduction,
            last_eta: None,
            ridge_used: 0.0,
            ridge_base_used: 0.0,
            ridge_link_used: 0.0,
        }
    }

    /// Set per-observation SE for integrated (GHQ) likelihood.
    /// When set, the joint model uses uncertainty-aware IRLS updates.
    pub fn with_covariate_se(mut self, se: Array1<f64>) -> Self {
        self.covariate_se = Some(se);
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
    pub fn n_obs(&self) -> usize {
        self.y.len()
    }

    /// Get total weight sum
    pub fn total_weight(&self) -> f64 {
        self.weights.sum()
    }

    /// Get link function
    pub fn link(&self) -> LinkFunction {
        self.link.clone()
    }

    /// Get number of base penalties
    pub fn n_base_penalties(&self) -> usize {
        self.s_base.len()
    }

    /// Get number of link penalties  
    pub fn n_link_penalties(&self) -> usize {
        1 // Single penalty for link wiggle
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
            let min_val = eta_base.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = eta_base.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range_width = max_val - min_val;
            let range = if range_width > 1e-6 {
                (min_val, max_val)
            } else {
                let center = 0.5 * (min_val + max_val);
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
    /// Returns (z_raw, z_clamped, range_width) where:
    ///   z_raw = (u - min_u) / range_width
    ///   z_clamped = clamp(z_raw, 0, 1)
    fn standardized_z(&self, eta_base: &Array1<f64>) -> (Array1<f64>, Array1<f64>, f64) {
        let (min_u, max_u) = self.knot_range.unwrap_or((0.0, 1.0));
        let range_width = (max_u - min_u).max(1e-6);
        let z_raw: Array1<f64> = eta_base.mapv(|u| (u - min_u) / range_width);
        let z_clamped: Array1<f64> = z_raw.mapv(|z| z.clamp(0.0, 1.0));
        (z_raw, z_clamped, range_width)
    }

    /// Return the stored projected penalty for link block (Z'SZ)
    pub fn build_link_penalty(&self) -> Array2<f64> {
        self.s_link_constrained
            .clone()
            .unwrap_or_else(|| Array2::zeros((0, 0)))
    }

    /// Solve penalized weighted least squares: (X'WX + λS + δI)β = X'Wz
    ///
    /// Returns (coefficients, ridge_used) where ridge_used is the stabilization
    /// ridge δ added to ensure positive definiteness.
    ///
    /// # Mathematical Note (Envelope Theorem)
    /// When ridge δ > 0, the solver finds β̂ that minimizes:
    ///   L_ridge(β) = -ℓ(β) + 0.5*β'S_λβ + 0.5*δ||β||²
    ///
    /// For the analytic gradient to be exact, the cost function must include
    /// the same δ||β||² term. Otherwise, ∇_β V(β̂) ≠ 0 and the Envelope
    /// Theorem fails, introducing gradient error proportional to δ*||β||.
    fn solve_weighted_ls(
        x: &Array2<f64>,
        z: &Array1<f64>,
        w: &Array1<f64>,
        penalty: &Array2<f64>,
        lambda: f64,
    ) -> (Array1<f64>, f64) {
        use crate::faer_ndarray::FaerCholesky;
        use crate::faer_ndarray::{fast_ata, fast_atv};
        use faer::Side;

        let n = x.nrows();
        let p = x.ncols();

        if p == 0 {
            return (Array1::zeros(0), 0.0);
        }

        // Compute X'WX via weighted design
        let mut x_weighted = x.clone();
        let mut z_weighted = z.clone();
        for i in 0..n {
            let wi = w[i].max(0.0).sqrt();
            for j in 0..p {
                x_weighted[[i, j]] *= wi;
            }
            z_weighted[i] *= wi;
        }
        let mut xwx = fast_ata(&x_weighted);

        // Add penalty: X'WX + λS (only if penalty matches dimensions)
        if penalty.nrows() == p && penalty.ncols() == p {
            xwx = xwx + penalty * lambda;
        }

        // Conditional regularization for numerical stability.
        // The ridge δ must be tracked and included in the cost function
        // to satisfy the Envelope Theorem (see docstring).
        let ridge_used = ensure_positive_definite_joint(&mut xwx);

        // Compute X'Wz via weighted design
        let xwz = fast_atv(&x_weighted, &z_weighted);

        // Solve (X'WX + λS + δI)β = X'Wz using Cholesky
        let coeffs = match xwx.cholesky(Side::Lower) {
            Ok(chol) => chol.solve_vec(&xwz),
            Err(_) => {
                eprintln!("[JOINT] Warning: Cholesky factorization failed");
                Array1::zeros(p)
            }
        };
        (coeffs, ridge_used)
    }

    /// Perform one IRLS step for the link block
    /// Uses identity (u) as OFFSET, solves only for wiggle coefficients
    /// η = u + B_wiggle(z) · θ
    pub fn irls_link_step(
        &mut self,
        b_wiggle: &Array2<f64>, // Constrained wiggle basis (NOT including identity)
        u: &Array1<f64>,        // Current linear predictor u = Xβ (used as offset)
        lambda_link: f64,
    ) -> f64 {
        let n = self.n_obs();

        // Current η = u + B_wiggle · θ
        let eta = if b_wiggle.ncols() > 0 && self.beta_link.len() == b_wiggle.ncols() {
            let wiggle = b_wiggle.dot(&self.beta_link);
            u + &wiggle
        } else {
            u.clone()
        };
        self.last_eta = Some(eta.clone());

        // Allocate working vectors
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut z_glm = Array1::<f64>::zeros(n);

        // Compute working response and weights.
        // Standard GLM vectors are computed first, then Firth correction is applied below
        // (see z_firth computation) using hat diagonal from the unpenalized Fisher information.
        if let (LinkFunction::Logit, Some(se)) = (&self.link, &self.covariate_se) {
            crate::pirls::update_glm_vectors_integrated(
                &self.quad_ctx,
                self.y,
                &eta,
                se.view(),
                self.weights,
                &mut mu,
                &mut weights,
                &mut z_glm,
            );
        } else {
            crate::pirls::update_glm_vectors(
                self.y,
                &eta,
                self.link.clone(),
                self.weights,
                &mut mu,
                &mut weights,
                &mut z_glm,
            );
        }

        // When Firth is enabled for logit, compute Firth correction using hat diagonal
        // Firth modifies the working response: z_firth = z + h_ii * (0.5 - μ) / w_i
        let z_firth = if self.firth_bias_reduction
            && matches!(self.link, LinkFunction::Logit)
            && b_wiggle.ncols() > 0
        {
            // Compute hat diagonal from the unpenalized Fisher information (B'WB).
            // For efficiency, use: h_ii = w_i * ||solve(H, b_i)||^2 where b_i is ith row of B√w
            let p = b_wiggle.ncols();

            // Build weighted design: B_w = sqrt(W) * B
            let mut b_weighted = b_wiggle.clone();
            for i in 0..n {
                let sqrt_w = weights[i].max(0.0).sqrt();
                for j in 0..p {
                    b_weighted[[i, j]] *= sqrt_w;
                }
            }

            // Build Fisher information: H = B'WB (no smoothing penalty for Firth adjustment)
            let btb = fast_ata(&b_weighted);
            let mut h_fisher = btb;
            // Conditional regularization for stability without changing the objective form
            ensure_positive_definite_joint(&mut h_fisher);

            // Cholesky decomposition
            use crate::faer_ndarray::FaerCholesky;
            let chol = match h_fisher.cholesky(faer::Side::Lower) {
                Ok(c) => c,
                Err(_) => {
                    // Fall back to standard IRLS if Firth fails
                    return self.compute_deviance(&self.compute_eta_full(u, b_wiggle));
                }
            };

            // Compute hat diagonal and Firth-adjusted z
            let mut z_adj = z_glm.clone();
            for i in 0..n {
                let mi = mu[i].clamp(1e-8, 1.0 - 1e-8);

                // h_ii = ||L^{-1} (b_i * sqrt(w_i))||^2
                let b_row: Array1<f64> = (0..p).map(|j| b_weighted[[i, j]]).collect();

                // Solve L * v = b_row using Cholesky factor
                let solved = chol
                    .solve_mat(&b_row.insert_axis(ndarray::Axis(1)))
                    .column(0)
                    .to_owned();
                let h_ii: f64 = solved.iter().map(|x| x * x).sum();

                let wi = weights[i];
                if wi > 0.0 {
                    // Firth correction to working response
                    // This biases coefficients toward zero when separation threatens
                    let firth_adj = h_ii * (0.5 - mi) / wi;
                    z_adj[i] += firth_adj;
                }
            }
            z_adj
        } else {
            z_glm
        };

        // Adjust working response: solve for wiggle coefficient θ where
        // η = u + B_wiggle · θ
        // So target for θ is: z_adjusted = z_firth - u
        let z_adjusted: Array1<f64> = &z_firth - u;

        // Solve: (B'WB + λS + δI)θ = B'W(z - u)
        // Track ridge δ to include 0.5*δ||θ||² in cost (Envelope Theorem).
        self.ridge_link_used = 0.0;
        if b_wiggle.ncols() > 0 {
            let penalty = self.build_link_penalty();
            let (new_theta, ridge_link) =
                Self::solve_weighted_ls(b_wiggle, &z_adjusted, &weights, &penalty, lambda_link);

            // Update wiggle coefficients
            if new_theta.len() == self.beta_link.len() {
                self.beta_link = new_theta;
            }
            self.ridge_link_used = ridge_link;
            // Accumulate ridge (take max of base and link ridges for joint system)
            self.ridge_used = self.ridge_used.max(ridge_link);
        }

        // Recompute deviance using updated coefficients
        let eta_updated = self.compute_eta_full(u, b_wiggle);
        self.recompute_deviance_from_eta(&eta_updated)
    }

    /// Perform one IRLS step for the base β block
    /// Uses Gauss-Newton with proper offset for nonlinear link:
    /// η = g(u) = u + wiggle(u), ∂η/∂β = g'(u) · x
    /// Working response for β: z_β = z_glm - η + g'(u)·u
    pub fn irls_base_step(
        &mut self,
        b_wiggle: &Array2<f64>, // Constrained wiggle basis
        g_prime: &Array1<f64>,  // Derivative of link: g'(u) = 1 + B'(u)·θ
        lambda_base: &Array1<f64>,
        damping: f64,
    ) -> f64 {
        let n = self.n_obs();
        let p = self.x_base.ncols();

        // Current u = Xβ
        let u = self.base_linear_predictor();

        // Current η = u + B_wiggle · θ
        let wiggle: Array1<f64> =
            if b_wiggle.ncols() > 0 && self.beta_link.len() == b_wiggle.ncols() {
                b_wiggle.dot(&self.beta_link)
            } else {
                Array1::zeros(n)
            };
        let eta: Array1<f64> = &u + &wiggle;

        // Compute working response and weights for current η
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut z_glm = Array1::<f64>::zeros(n);

        // Use integrated likelihood if SE available (Logit only)
        if let (LinkFunction::Logit, Some(se)) = (&self.link, &self.covariate_se) {
            crate::pirls::update_glm_vectors_integrated(
                &self.quad_ctx,
                self.y,
                &eta,
                se.view(),
                self.weights,
                &mut mu,
                &mut weights,
                &mut z_glm,
            );
        } else {
            crate::pirls::update_glm_vectors(
                self.y,
                &eta,
                self.link.clone(),
                self.weights,
                &mut mu,
                &mut weights,
                &mut z_glm,
            );
        }

        // Correct working response for β update (Gauss-Newton offset):
        // z_β = z_glm - η + g'(u)·u
        let mut z_beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            z_beta[i] = z_glm[i] - eta[i] + g_prime[i] * u[i];
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
        let mut penalty = Array2::<f64>::zeros((p, p));
        for (idx, s_k) in self.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if s_k.nrows() == p && s_k.ncols() == p && lambda_k > 0.0 {
                penalty.scaled_add(lambda_k, s_k);
            }
        }

        // When Firth is enabled for logit, compute Firth correction for base block
        // This uses the effective design X_eff = sqrt(w_eff) * X and the base penalty
        if self.firth_bias_reduction && matches!(self.link, LinkFunction::Logit) {
            // Build weighted design: X_w = sqrt(w_eff) * X
            let mut x_weighted = self.x_base.to_owned();
            for i in 0..n {
                let sqrt_w = w_eff[i].max(0.0).sqrt();
                for j in 0..p {
                    x_weighted[[i, j]] *= sqrt_w;
                }
            }

            // Build Fisher information: H = X'W_effX (no smoothing penalty for Firth adjustment)
            let xtx = fast_ata(&x_weighted);
            let mut h_fisher = xtx;
            ensure_positive_definite_joint(&mut h_fisher);

            // Cholesky decomposition
            use crate::faer_ndarray::FaerCholesky;
            if let Ok(chol) = h_fisher.cholesky(faer::Side::Lower) {
                // Compute hat diagonal and apply Firth adjustment to z_eff
                for i in 0..n {
                    let mi = mu[i].clamp(1e-8, 1.0 - 1e-8);

                    // h_ii = ||L^{-1}(x_i * sqrt(w_eff_i))||²
                    let x_row: Array1<f64> = (0..p).map(|j| x_weighted[[i, j]]).collect();
                    let solved = chol
                        .solve_mat(&x_row.insert_axis(ndarray::Axis(1)))
                        .column(0)
                        .to_owned();
                    let h_ii: f64 = solved.iter().map(|x| x * x).sum();

                    let wi = w_eff[i];
                    if wi > 0.0 {
                        // Firth correction to working response
                        let firth_adj = h_ii * (0.5 - mi) / wi;
                        z_eff[i] += firth_adj;
                    }
                }
            }
        }

        // Solve PWLS: (X'W_eff X + S + δI)β = X'W_eff z_eff
        // Track ridge δ to include 0.5*δ||β||² in cost (Envelope Theorem).
        let (new_beta, ridge_base) =
            Self::solve_weighted_ls(&self.x_base.to_owned(), &z_eff, &w_eff, &penalty, 1.0);
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
            if b_wiggle.ncols() > 0 && self.beta_link.len() == b_wiggle.ncols() {
                b_wiggle.dot(&self.beta_link)
            } else {
                Array1::zeros(n)
            };
        let eta_updated: Array1<f64> = &u_updated + &wiggle_updated;
        self.last_eta = Some(eta_updated.clone());
        self.recompute_deviance_from_eta(&eta_updated)
    }

    /// Compute current linear predictor: η = u + B_wiggle · θ
    pub fn compute_eta_full(&self, u: &Array1<f64>, b_wiggle: &Array2<f64>) -> Array1<f64> {
        if b_wiggle.ncols() > 0 && self.beta_link.len() == b_wiggle.ncols() {
            let wiggle = b_wiggle.dot(&self.beta_link);
            u + &wiggle
        } else {
            u.clone()
        }
    }

    /// Recompute deviance by refreshing GLM working vectors at a supplied linear predictor.
    fn recompute_deviance_from_eta(&self, eta: &Array1<f64>) -> f64 {
        let n = self.n_obs();
        let mut mu_updated = Array1::<f64>::zeros(n);
        let mut weights_updated = Array1::<f64>::zeros(n);
        let mut z_updated = Array1::<f64>::zeros(n);

        if let (LinkFunction::Logit, Some(se)) = (&self.link, &self.covariate_se) {
            crate::pirls::update_glm_vectors_integrated(
                &self.quad_ctx,
                self.y,
                eta,
                se.view(),
                self.weights,
                &mut mu_updated,
                &mut weights_updated,
                &mut z_updated,
            );
        } else {
            crate::pirls::update_glm_vectors(
                self.y,
                eta,
                self.link.clone(),
                self.weights,
                &mut mu_updated,
                &mut weights_updated,
                &mut z_updated,
            );
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
/// Identifiability: scale anchor enforces g(0)≈0, g'(0)≈1
pub(crate) fn fit_joint_model<'a>(
    y: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    x_base: ArrayView2<'a, f64>,
    s_base: Vec<Array2<f64>>,
    layout_base: EngineDims,
    link: LinkFunction,
    config: &JointModelConfig,
) -> Result<JointModelResult, EstimationError> {
    fit_joint_model_with_reml(y, weights, x_base, s_base, layout_base, link, config, None)
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
) -> Result<JointModelResult, EstimationError> {
    if geometry.degree != 3 {
        return Err(EstimationError::InvalidSpecification(
            "joint engine currently supports cubic link splines only (degree=3)".to_string(),
        ));
    }
    config.n_link_knots = geometry.n_link_knots;
    let layout = EngineDims::new(x_base.ncols(), s_base.len());
    fit_joint_model(y, weights, x_base, s_base, layout, link, &config)
}

/// State for REML optimization of the joint model
/// Wraps JointModelState and provides outer objective derivatives over ρ.
pub struct JointCore<'a> {
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
    eval_count: usize,
    last_hessian_condition: Option<f64>,
    last_outer_step_norm: Option<f64>,
    last_eval_rho: Option<Array1<f64>>,
}

pub struct JointRemlState<'a> {
    core: JointCore<'a>,
    eval: JointEvalContext,
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
    last_hessian_condition: Option<f64>,
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
            last_hessian_condition: reml.eval.last_hessian_condition,
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
        reml.eval.last_hessian_condition = self.last_hessian_condition;
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
        quad_ctx: QuadratureContext,
    ) -> Self {
        let mut state = JointModelState::new(
            y,
            weights,
            x_base,
            s_base,
            layout_base,
            link,
            config,
            quad_ctx,
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
                eval_count: 0,
                last_hessian_condition: None,
                last_outer_step_norm: None,
                last_eval_rho: None,
            },
        }
    }

    /// Compute LAML cost for a given ρ
    /// LAML = deviance + log|H_pen| - log|S_λ| (+ prior on ρ)
    pub fn compute_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let state = &mut self.core.state;
        let n_base = state.s_base.len();

        // Set ρ and warm-start from cached coefficients
        state.set_rho(rho.clone());
        state.beta_base = self.eval.cached_beta_base.clone();
        state.beta_link = self.eval.cached_beta_link.clone();

        // Run inner alternating to convergence
        let mut lambda_base = Array1::<f64>::zeros(n_base);
        for i in 0..n_base {
            lambda_base[i] = rho.get(i).map(|r| r.exp()).unwrap_or(1.0);
        }
        let lambda_link = rho.get(n_base).map(|r| r.exp()).unwrap_or(1.0);

        let mut prev_deviance = f64::INFINITY;
        let mut iter_count = 0;
        let mut converged = false;
        for i in 0..self.core.config.max_backfit_iter {
            let progress = (i as f64) / (self.core.config.max_backfit_iter as f64);
            let damping = 0.5 + progress * 0.5;

            let u = state.base_linear_predictor();
            let b_wiggle = state
                .build_link_basis(&u)
                .map_err(|e| EstimationError::InvalidSpecification(e))?;
            state.irls_link_step(&b_wiggle, &u, lambda_link);

            let g_prime = compute_link_derivative_from_state(&state, &u, &b_wiggle);
            let deviance = state.irls_base_step(&b_wiggle, &g_prime, &lambda_base, damping);

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

        // Compute LAML = deviance + log|H_pen| - log|S_λ|
        let (laml, edf) = Self::compute_laml_at_convergence(
            state,
            &lambda_base,
            lambda_link,
            self.core.base_reparam_invariant.as_ref(),
            &self.core.base_rs_list,
        );

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

    /// Compute LAML at the converged solution.
    /// Note: for nonlinear g(u), this uses a Gauss-Newton Hessian approximation.
    fn compute_laml_at_convergence(
        state: &JointModelState,
        lambda_base: &Array1<f64>,
        lambda_link: f64,
        base_reparam_invariant: Option<&ReparamInvariant>,
        base_rs_list: &[Array2<f64>],
    ) -> (f64, Option<f64>) {
        let n = state.n_obs();
        let u = state.base_linear_predictor();
        let b_wiggle = state.build_link_basis_from_state(&u);

        // Compute eta = u + B_wiggle * theta
        let eta = state.compute_eta_full(&u, &b_wiggle);

        // Compute mu/weights/residuals at convergence
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut residual = Array1::<f64>::zeros(n);
        match (&state.link, &state.covariate_se) {
            (LinkFunction::Logit, Some(se)) => {
                const PROB_EPS: f64 = 1e-8;
                const MIN_DMU: f64 = 1e-6;
                for i in 0..n {
                    let e = eta[i].clamp(-700.0, 700.0);
                    let se_i = se[i].max(0.0);
                    let (mu_i, dmu_deta) = crate::quadrature::logit_posterior_mean_with_deriv(
                        &state.quad_ctx,
                        e,
                        se_i,
                    );
                    let mu_c = mu_i.clamp(PROB_EPS, 1.0 - PROB_EPS);
                    mu[i] = mu_c;
                    let var = (mu_c * (1.0 - mu_c)).max(PROB_EPS);
                    let dmu_sq = dmu_deta * dmu_deta;
                    let w = dmu_sq / var;
                    weights[i] = state.weights[i] * w;
                    let denom = dmu_deta.abs().max(MIN_DMU);
                    residual[i] = weights[i] * (mu_c - state.y[i]) / denom;
                }
            }
            (LinkFunction::Logit, None) => {
                const PROB_EPS: f64 = 1e-8;
                const MIN_WEIGHT: f64 = 1e-12;
                const MIN_DMU: f64 = 1e-6;
                for i in 0..n {
                    let e = eta[i].clamp(-700.0, 700.0);
                    let mu_i = (1.0 / (1.0 + (-e).exp())).clamp(PROB_EPS, 1.0 - PROB_EPS);
                    mu[i] = mu_i;
                    let dmu = (mu_i * (1.0 - mu_i)).max(MIN_WEIGHT);
                    weights[i] = state.weights[i] * dmu;
                    let denom = dmu.max(MIN_DMU);
                    residual[i] = weights[i] * (mu_i - state.y[i]) / denom;
                }
            }
            (LinkFunction::Identity, _) => {
                for i in 0..n {
                    mu[i] = eta[i];
                    weights[i] = state.weights[i];
                    residual[i] = weights[i] * (mu[i] - state.y[i]);
                }
            }
            (LinkFunction::Probit, _) => {
                const PROB_EPS: f64 = 1e-8;
                const MIN_WEIGHT: f64 = 1e-12;
                const MIN_DMU: f64 = 1e-6;
                for i in 0..n {
                    let e = eta[i].clamp(-30.0, 30.0);
                    let mu_i = normal_cdf_approx(e).clamp(PROB_EPS, 1.0 - PROB_EPS);
                    mu[i] = mu_i;
                    let dmu = normal_pdf(e).max(MIN_DMU);
                    let var = (mu_i * (1.0 - mu_i)).max(PROB_EPS);
                    let w = ((dmu * dmu) / var).max(MIN_WEIGHT);
                    weights[i] = state.weights[i] * w;
                    residual[i] = weights[i] * (mu_i - state.y[i]) / dmu;
                }
            }
            (LinkFunction::CLogLog, _) => {
                const PROB_EPS: f64 = 1e-8;
                const MIN_WEIGHT: f64 = 1e-12;
                const MIN_DMU: f64 = 1e-6;
                for i in 0..n {
                    let e = eta[i].clamp(-30.0, 30.0);
                    let exp_eta = e.exp();
                    let surv = (-exp_eta).exp();
                    let mu_i = (1.0 - surv).clamp(PROB_EPS, 1.0 - PROB_EPS);
                    mu[i] = mu_i;
                    let dmu = (exp_eta * surv).max(MIN_DMU);
                    let var = (mu_i * (1.0 - mu_i)).max(PROB_EPS);
                    let w = ((dmu * dmu) / var).max(MIN_WEIGHT);
                    weights[i] = state.weights[i] * w;
                    residual[i] = weights[i] * (mu_i - state.y[i]) / dmu;
                }
            }
        }
        let deviance = state.compute_deviance(&mu);

        // Build joint Jacobian blocks and penalized Hessian via Schur complement
        let p_base = state.x_base.ncols();
        let p_link = b_wiggle.ncols();
        let p_total = p_base + p_link;

        let (g_prime, g_second, b_prime_u) = compute_link_derivative_terms_from_state(state, &u);

        // A = X' diag(W * g'^2 + r * g'') X + S_base
        let mut w_eff = Array1::<f64>::zeros(n);
        for i in 0..n {
            w_eff[i] = weights[i] * g_prime[i] * g_prime[i];
        }
        let mut x_weighted = state.x_base.to_owned();
        for i in 0..n {
            let scale = w_eff[i].max(0.0).sqrt();
            for j in 0..p_base {
                x_weighted[[i, j]] *= scale;
            }
        }

        let mut a_mat = crate::faer_ndarray::fast_ata(&x_weighted);
        let mut x_scaled = state.x_base.to_owned();
        for i in 0..n {
            let scale = residual[i] * g_second[i];
            for j in 0..p_base {
                x_scaled[[i, j]] *= scale;
            }
        }
        let a_resid = crate::faer_ndarray::fast_atb(&state.x_base, &x_scaled);
        a_mat += &a_resid;
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                a_mat.scaled_add(lambda_k, s_k);
            }
        }

        // Symmetrize: the Gauss-Newton construction should be symmetric, but
        // rounding and mixed products can introduce tiny asymmetry.
        for i in 0..p_base {
            for j in 0..i {
                let v = 0.5 * (a_mat[[i, j]] + a_mat[[j, i]]);
                a_mat[[i, j]] = v;
                a_mat[[j, i]] = v;
            }
        }

        // C = X' diag(W * g') B + X' diag(r) B'
        let mut wb = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i] * g_prime[i];
            for j in 0..p_link {
                wb[[i, j]] *= scale;
            }
        }
        let mut wb_resid = b_prime_u.clone();
        for i in 0..n {
            let scale = residual[i];
            for j in 0..p_link {
                wb_resid[[i, j]] *= scale;
            }
        }
        wb += &wb_resid;
        let c_mat = crate::faer_ndarray::fast_atb(&state.x_base, &wb);

        // D = B' W B + S_link
        let mut b_weighted = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i].sqrt();
            for j in 0..p_link {
                b_weighted[[i, j]] *= scale;
            }
        }
        let mut d_mat = crate::faer_ndarray::fast_ata(&b_weighted);
        let link_penalty = state.build_link_penalty();
        if link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
            d_mat.scaled_add(lambda_link, &link_penalty);
        }
        // Same symmetry rationale as for A.
        for i in 0..p_link {
            for j in 0..i {
                let v = 0.5 * (d_mat[[i, j]] + d_mat[[j, i]]);
                d_mat[[i, j]] = v;
                d_mat[[j, i]] = v;
            }
        }

        // Spectral log|H|_+ via eigendecomposition (Wood 2011)
        // This approach:
        // 1. Keeps the objective function smooth as eigenvalues cross zero
        // 2. Avoids discontinuity from conditional ridge
        // 3. Uses log|H|_+ = Σ log(λ_i) for positive eigenvalues only
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;

        // Build full joint Hessian H = [[A, C], [C^T, D]]
        let mut h_full = Array2::<f64>::zeros((p_total, p_total));
        h_full.slice_mut(s![..p_base, ..p_base]).assign(&a_mat);
        h_full.slice_mut(s![..p_base, p_base..]).assign(&c_mat);
        h_full.slice_mut(s![p_base.., ..p_base]).assign(&c_mat.t());
        h_full.slice_mut(s![p_base.., p_base..]).assign(&d_mat);

        let log_det_a = match h_full.clone().eigh(Side::Lower) {
            Ok((eigs, _)) => {
                // Spectral log-det: sum of log of positive eigenvalues
                let max_eig = eigs.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                let tol = (max_eig * 1e-12).max(1e-100);
                let log_det: f64 = eigs.iter().filter(|&&ev| ev > tol).map(|&ev| ev.ln()).sum();
                log_det
            }
            Err(_) => {
                // Eigendecomposition failed - severe numerical issues
                eprintln!("[LAML] Joint Hessian eigendecomposition failed");
                return (f64::INFINITY, None);
            }
        };

        // log|Sλ|_+ from stable reparameterization (base) and eigenvalues (link)
        let base_reparam = if let Some(invariant) = base_reparam_invariant {
            stable_reparameterization_with_invariant_engine(
                base_rs_list,
                &lambda_base.to_vec(),
                EngineDims::new(state.layout_base.p, base_rs_list.len()),
                invariant,
            )
        } else {
            stable_reparameterization_engine(
                base_rs_list,
                &lambda_base.to_vec(),
                EngineDims::new(state.layout_base.p, base_rs_list.len()),
            )
        }
        .unwrap_or_else(|_| ReparamResult {
            s_transformed: Array2::zeros((p_base, p_base)),
            log_det: 0.0,
            det1: Array1::zeros(lambda_base.len()),
            qs: Array2::eye(p_base),
            rs_transformed: vec![],
            rs_transposed: vec![],
            e_transformed: Array2::zeros((0, p_base)),
            u_truncated: Array2::zeros((p_base, p_base)), // All modes truncated in fallback
        });
        // Spectral log|S_λ|_+ for penalty matrix (Wood 2011)
        // Uses pure spectral log-det without ridge adjustment
        // Math note:
        //   The stabilized inner solves use (S_λ + δI) as the effective prior precision.
        //   For objective/gradient consistency, the LAML normalization term must match:
        //     log|S_λ + δI| = log|S_λ|_+ + M_p * log(δ)   (when S_λ is rank-deficient).
        //   Using log|S_λ|_+ while also adding 0.5*δ||β||² in the quadratic term defines
        //   an incoherent objective and breaks the envelope-theorem simplification.
        let base_rank_usize = base_reparam.e_transformed.nrows();
        let base_rank = base_rank_usize as f64;
        let base_log_det_s = match Self::fixed_subspace_logdet_for_penalty(
            &base_reparam.s_transformed,
            base_rank_usize,
            state.ridge_base_used,
        ) {
            Ok(v) => v,
            Err(_) => {
                eprintln!("[LAML] Base penalty eigendecomposition failed");
                return (f64::INFINITY, None);
            }
        };

        let (link_log_det_s, link_rank) = if p_link > 0 {
            let rank = match Self::structural_rank_from_penalty(&link_penalty) {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("[LAML] Link penalty eigendecomposition failed");
                    return (f64::INFINITY, None);
                }
            };
            let s_link_lambda = link_penalty.mapv(|v| v * lambda_link);
            let log_det = match Self::fixed_subspace_logdet_for_penalty(
                &s_link_lambda,
                rank,
                state.ridge_link_used,
            ) {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("[LAML] Link penalty eigendecomposition failed");
                    return (f64::INFINITY, None);
                }
            };
            (log_det, rank as f64)
        } else {
            (0.0, 0.0)
        };

        let log_det_s = base_log_det_s + link_log_det_s;

        // Null space dimension
        let mp = (p_base as f64 - base_rank) + (p_link as f64 - link_rank);

        // Penalized log-likelihood term: -0.5*deviance - 0.5*beta'*S_λ*beta
        // Include stabilization ridge to satisfy Envelope Theorem consistency:
        // inner solver minimizes L_inner = -ℓ + 0.5 βᵀS_λβ + 0.5 δ||β||².
        // If cost omits δ||β||², then ∇_β V = -δ β̂ ≠ 0 and dV/dρ picks up a bias
        // via (∇_β V)ᵀ dβ/dρ. Adding the ridge restores ∇_β V ≈ 0 at β̂.
        let mut penalty_term = 0.0;
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                let sb = s_k.dot(&state.beta_base);
                penalty_term += lambda_k * state.beta_base.dot(&sb);
            }
        }
        if p_link > 0 && link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
            if state.beta_link.len() == p_link {
                let sb = link_penalty.dot(&state.beta_link);
                penalty_term += lambda_link * state.beta_link.dot(&sb);
            }
        }
        if state.ridge_base_used > 0.0 {
            // 0.5 * δ ||β_base||² with δ tracked from the actual PWLS solve.
            penalty_term += state.ridge_base_used * state.beta_base.dot(&state.beta_base);
        }
        if p_link > 0 && state.beta_link.len() == p_link && state.ridge_link_used > 0.0 {
            // Same ridge term for the link block to keep cost/gradient surfaces aligned.
            penalty_term += state.ridge_link_used * state.beta_link.dot(&state.beta_link);
        }

        let laml = match state.link {
            LinkFunction::Logit | LinkFunction::Probit | LinkFunction::CLogLog => {
                let penalised_ll = -0.5 * deviance - 0.5 * penalty_term;
                let laml = penalised_ll + 0.5 * log_det_s - 0.5 * log_det_a
                    + (mp / 2.0) * (2.0 * std::f64::consts::PI).ln();
                laml
            }
            LinkFunction::Identity => {
                let dp = (deviance + penalty_term).max(1e-12);
                let denom = (n as f64 - mp).max(1.0);
                let phi = dp / denom;
                let reml_cost = dp / (2.0 * phi)
                    + 0.5 * (log_det_a - log_det_s)
                    + ((n as f64 - mp) / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();
                -reml_cost
            }
        };

        let edf = Self::compute_joint_edf(
            state,
            &b_wiggle,
            &g_prime,
            &weights,
            lambda_base,
            lambda_link,
        );

        (laml, edf)
    }

    fn compute_joint_edf(
        state: &JointModelState,
        b_wiggle: &Array2<f64>,
        g_prime: &Array1<f64>,
        weights: &Array1<f64>,
        lambda_base: &Array1<f64>,
        lambda_link: f64,
    ) -> Option<f64> {
        use crate::faer_ndarray::FaerCholesky;
        use faer::Side;

        let n = state.n_obs();
        let p_base = state.x_base.ncols();
        let p_link = b_wiggle.ncols();
        let p_total = p_base + p_link;
        if p_total == 0 {
            return Some(0.0);
        }

        let mut w_eff = Array1::<f64>::zeros(n);
        for i in 0..n {
            w_eff[i] = weights[i] * g_prime[i] * g_prime[i];
        }
        let mut x_weighted = state.x_base.to_owned();
        for i in 0..n {
            let scale = w_eff[i].sqrt();
            for j in 0..p_base {
                x_weighted[[i, j]] *= scale;
            }
        }
        let mut a_mat = crate::faer_ndarray::fast_ata(&x_weighted);
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                a_mat.scaled_add(lambda_k, s_k);
            }
        }

        let mut wb = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i] * g_prime[i];
            for j in 0..p_link {
                wb[[i, j]] *= scale;
            }
        }
        let c_mat = crate::faer_ndarray::fast_atb(&state.x_base, &wb);

        let mut b_weighted = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i].sqrt();
            for j in 0..p_link {
                b_weighted[[i, j]] *= scale;
            }
        }
        let mut d_mat = crate::faer_ndarray::fast_ata(&b_weighted);
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
                for i in 0..p_base {
                    for j in 0..p_base {
                        s_lambda[[i, j]] += lambda_k * s_k[[i, j]];
                    }
                }
            }
        }
        if link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
            for i in 0..p_link {
                for j in 0..p_link {
                    s_lambda[[p_base + i, p_base + j]] += lambda_link * link_penalty[[i, j]];
                }
            }
        }

        let chol = h_mat.cholesky(Side::Lower).ok()?;
        let mut trace = 0.0;
        for j in 0..p_total {
            let col = s_lambda.column(j).to_owned();
            let solved = chol.solve_vec(&col);
            trace += solved[j];
        }

        Some(p_total as f64 - trace)
    }

    /// Compute numerical gradient of LAML w.r.t. ρ using central differences
    fn compute_gradient_fd(&mut self, rho: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        let h = 1e-4; // Step size for numerical differentiation
        let n_rho = rho.len();
        let mut grad = Array1::<f64>::zeros(n_rho);
        let snapshot = JointRemlSnapshot::new(self);

        for k in 0..n_rho {
            snapshot.restore(self);
            // Forward step
            let mut rho_plus = rho.clone();
            rho_plus[k] += h;
            let cost_plus = self.compute_cost(&rho_plus)?;
            if !cost_plus.is_finite() {
                snapshot.restore(self);
                return Err(EstimationError::RemlOptimizationFailed(
                    "Non-finite LAML in +h finite-difference step.".to_string(),
                ));
            }

            snapshot.restore(self);
            // Backward step
            let mut rho_minus = rho.clone();
            rho_minus[k] -= h;
            let cost_minus = self.compute_cost(&rho_minus)?;
            if !cost_minus.is_finite() {
                snapshot.restore(self);
                return Err(EstimationError::RemlOptimizationFailed(
                    "Non-finite LAML in -h finite-difference step.".to_string(),
                ));
            }

            // Central difference
            grad[k] = (cost_plus - cost_minus) / (2.0 * h);
        }

        snapshot.restore(self);

        if grad.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "Non-finite gradient from finite-difference LAML.".to_string(),
            ));
        }
        Ok(grad)
    }

    /// Compute analytic gradient of LAML w.r.t. ρ using a Gauss-Newton Hessian
    /// and explicit differentiation of weights and constrained basis.
    fn compute_gradient_analytic(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<(Array1<f64>, bool), EstimationError> {
        let state = &mut self.core.state;
        let n_base = state.s_base.len();

        if rho.len() != n_base + 1 {
            return Err(EstimationError::LayoutError(
                "rho length does not match joint penalty count".to_string(),
            ));
        }

        if matches!(state.link, LinkFunction::Logit) && state.covariate_se.is_some() {
            return Err(EstimationError::RemlOptimizationFailed(
                "analytic joint gradient not implemented for integrated logit weights".to_string(),
            ));
        }
        let firth_active = state.firth_bias_reduction && matches!(state.link, LinkFunction::Logit);

        // Set ρ and warm-start from cached coefficients
        state.set_rho(rho.clone());
        state.beta_base = self.eval.cached_beta_base.clone();
        state.beta_link = self.eval.cached_beta_link.clone();

        let mut lambda_base = Array1::<f64>::zeros(n_base);
        for i in 0..n_base {
            lambda_base[i] = rho.get(i).map(|r| r.exp()).unwrap_or(1.0);
        }
        let lambda_link = rho.get(n_base).map(|r| r.exp()).unwrap_or(1.0);

        // Run inner alternating to convergence (same as compute_cost).
        let mut prev_deviance = f64::INFINITY;
        let mut converged = false;
        for i in 0..self.core.config.max_backfit_iter {
            let progress = (i as f64) / (self.core.config.max_backfit_iter as f64);
            let damping = 0.5 + progress * 0.5;

            let u = state.base_linear_predictor();
            let b_wiggle = state
                .build_link_basis(&u)
                .map_err(|e| EstimationError::InvalidSpecification(e))?;
            state.irls_link_step(&b_wiggle, &u, lambda_link);

            let g_prime = compute_link_derivative_from_state(&state, &u, &b_wiggle);
            let deviance = state.irls_base_step(&b_wiggle, &g_prime, &lambda_base, damping);

            let delta = (prev_deviance - deviance).abs() / (deviance.abs() + 1.0);
            if delta < self.core.config.backfit_tol {
                converged = true;
                break;
            }
            prev_deviance = deviance;
        }

        // Cache converged coefficients for warm-start
        self.eval.cached_beta_base = state.beta_base.clone();
        self.eval.cached_beta_link = state.beta_link.clone();
        self.eval.last_converged = converged;

        let n = state.n_obs();
        let u = state.base_linear_predictor();
        let b_wiggle = state.build_link_basis_from_state(&u);
        let eta = state.compute_eta_full(&u, &b_wiggle);

        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut residual = Array1::<f64>::zeros(n);
        match (&state.link, &state.covariate_se) {
            (LinkFunction::Logit, None) => {
                const PROB_EPS: f64 = 1e-8;
                const MIN_WEIGHT: f64 = 1e-12;
                const MIN_DMU: f64 = 1e-6;
                for i in 0..n {
                    let e = eta[i].clamp(-700.0, 700.0);
                    let mu_i = (1.0 / (1.0 + (-e).exp())).clamp(PROB_EPS, 1.0 - PROB_EPS);
                    mu[i] = mu_i;
                    let w = (mu_i * (1.0 - mu_i)).max(MIN_WEIGHT);
                    weights[i] = state.weights[i] * w;
                    let denom = w.max(MIN_DMU);
                    residual[i] = weights[i] * (mu_i - state.y[i]) / denom;
                }
            }
            (LinkFunction::Identity, _) => {
                for i in 0..n {
                    mu[i] = eta[i];
                    weights[i] = state.weights[i];
                    residual[i] = weights[i] * (mu[i] - state.y[i]);
                }
            }
            _ => {
                return Err(EstimationError::RemlOptimizationFailed(
                    "analytic joint gradient unsupported for this link".to_string(),
                ));
            }
        }

        let p_base = state.x_base.ncols();
        let p_link = b_wiggle.ncols();
        let p_total = p_base + p_link;

        let (g_prime, g_second, b_prime_u) = compute_link_derivative_terms_from_state(&state, &u);

        // Build Gauss-Newton blocks A, C, D (same as cost path).
        let mut w_eff = Array1::<f64>::zeros(n);
        for i in 0..n {
            w_eff[i] = weights[i] * g_prime[i] * g_prime[i];
        }
        let mut x_weighted = state.x_base.to_owned();
        for i in 0..n {
            let scale = w_eff[i].max(0.0).sqrt();
            for j in 0..p_base {
                x_weighted[[i, j]] *= scale;
            }
        }
        let mut a_mat = crate::faer_ndarray::fast_ata(&x_weighted);
        let mut x_scaled = state.x_base.to_owned();
        for i in 0..n {
            let scale = residual[i] * g_second[i];
            for j in 0..p_base {
                x_scaled[[i, j]] *= scale;
            }
        }
        let a_resid = crate::faer_ndarray::fast_atb(&state.x_base, &x_scaled);
        a_mat += &a_resid;
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                a_mat.scaled_add(lambda_k, s_k);
            }
        }
        ensure_positive_definite_joint(&mut a_mat);

        let mut wb = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i] * g_prime[i];
            for j in 0..p_link {
                wb[[i, j]] *= scale;
            }
        }
        let mut wb_resid = b_prime_u.clone();
        for i in 0..n {
            let scale = residual[i];
            for j in 0..p_link {
                wb_resid[[i, j]] *= scale;
            }
        }
        wb += &wb_resid;
        let c_mat = crate::faer_ndarray::fast_atb(&state.x_base, &wb);

        let mut b_weighted = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i].sqrt();
            for j in 0..p_link {
                b_weighted[[i, j]] *= scale;
            }
        }
        let mut d_mat = crate::faer_ndarray::fast_ata(&b_weighted);
        let link_penalty = state.build_link_penalty();
        if link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
            d_mat.scaled_add(lambda_link, &link_penalty);
        }
        ensure_positive_definite_joint(&mut d_mat);

        // Assemble full Hessian H from blocks.
        let mut h_mat = Array2::<f64>::zeros((p_total, p_total));
        h_mat.slice_mut(s![..p_base, ..p_base]).assign(&a_mat);
        h_mat.slice_mut(s![..p_base, p_base..]).assign(&c_mat);
        h_mat.slice_mut(s![p_base.., ..p_base]).assign(&c_mat.t());
        h_mat.slice_mut(s![p_base.., p_base..]).assign(&d_mat);

        // Eigendecomposition of Hessian for pseudo-inverse (Wood 2011)
        // The spectral approach handles non-PD matrices gracefully by zeroing
        // out small/negative eigenvalues in the pseudo-inverse.
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let (h_eigs, h_vecs): (Array1<f64>, Array2<f64>) = h_mat
            .clone()
            .eigh(Side::Lower)
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
        let h_max_eig = h_eigs.iter().cloned().fold(0.0_f64, f64::max);
        let h_tol = (h_max_eig * 1e-12).max(1e-100);
        let mut h_min_pos = f64::INFINITY;
        for &ev in &h_eigs {
            if ev > h_tol {
                h_min_pos = h_min_pos.min(ev);
            }
        }
        if h_min_pos.is_finite() && h_min_pos > 0.0 {
            self.eval.last_hessian_condition = Some((h_max_eig / h_min_pos).max(1.0));
        } else {
            self.eval.last_hessian_condition = None;
        }

        // Check for severely ill-conditioned Hessian (too many negative eigenvalues)
        let n_positive = h_eigs.iter().filter(|&&ev| ev > h_tol).count();
        if n_positive < p_total / 2 {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }

        // Build Jacobian J = [diag(g') X | B_wiggle]
        let mut j_mat = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            let gp = g_prime[i];
            for j in 0..p_base {
                j_mat[[i, j]] = gp * state.x_base[[i, j]];
            }
            for j in 0..p_link {
                j_mat[[i, p_base + j]] = b_wiggle[[i, j]];
            }
        }

        // Precompute K = H† J^T using pseudo-inverse for spectral consistency (Wood 2011)
        // H† = U diag(1/λ_i) U' where 1/λ_i = 0 for small eigenvalues
        // K = H† J' = U diag(1/λ_i) U' J' = U diag(1/λ_i) (J U)'
        // Compute J @ U (n x p_total)
        let j_u = fast_ab(&j_mat, &h_vecs);

        // Compute H† J' = U @ diag(1/λ) @ (J U)'
        // This is (p_total x n) matrix
        let mut k_mat = Array2::<f64>::zeros((p_total, n));
        for i in 0..p_total {
            let eig_i = h_eigs[i];
            if eig_i > h_tol {
                let inv_eig = 1.0 / eig_i;
                // k_mat += (1/λ_i) * u_i @ (J u_i)'
                for row in 0..p_total {
                    for col in 0..n {
                        k_mat[[row, col]] += inv_eig * h_vecs[[row, i]] * j_u[[col, i]];
                    }
                }
            }
        }

        let mut k_w = k_mat.clone();
        for i in 0..n {
            let w = weights[i];
            for j in 0..p_total {
                k_w[[j, i]] *= w;
            }
        }
        let mut diag_proj = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..p_total {
                acc += j_mat[[i, j]] * k_mat[[j, i]];
            }
            diag_proj[i] = acc;
        }

        // Block-level EDF split for live visualization: base predictor vs link wiggle.
        let mut edf_base = 0.0_f64;
        let mut edf_link = 0.0_f64;
        for i in 0..n {
            for j in 0..p_base {
                edf_base += j_mat[[i, j]] * k_mat[[j, i]];
            }
            for j in 0..p_link {
                let col = p_base + j;
                edf_link += j_mat[[i, col]] * k_mat[[col, i]];
            }
        }
        self.eval.cached_edf_terms = vec![
            ("Base Predictor".to_string(), edf_base.max(0.0), p_base as f64),
            ("Link Wiggle".to_string(), edf_link.max(0.0), p_link.max(1) as f64),
        ];

        // Precompute H†_{theta,theta} for penalty sensitivity trace using pseudo-inverse
        let mut h_inv_theta = Array2::<f64>::zeros((p_link, p_link));
        if p_link > 0 {
            // H†_{theta,theta} = (U_{theta,:} diag(1/λ) U_{theta,:}')
            // where U_{theta,:} is the (p_base:p_total, :) block of U
            for i in 0..p_total {
                let eig_i = h_eigs[i];
                if eig_i > h_tol {
                    let inv_eig = 1.0 / eig_i;
                    for row in 0..p_link {
                        for col in 0..p_link {
                            h_inv_theta[[row, col]] +=
                                inv_eig * h_vecs[[p_base + row, i]] * h_vecs[[p_base + col, i]];
                        }
                    }
                }
            }
        }

        // Firth adjoint matrix Q = H^{-1} V H^{-1}.
        //
        // Proof-style derivation (fixed-point, logit Firth):
        //   Let I = J^T W J be the Fisher information and define the Firth term:
        //     Phi = 0.5 * log|I|.
        //   The Firth-adjusted objective adds Phi to the log-likelihood, so the
        //   REML/LAML gradient gains an extra contribution from d/d rho_k Phi.
        //
        //   Using Jacobi's identity:
        //     d Phi = 0.5 * tr(I^{-1} dI).
        //   The implicit dependence of I on the parameters introduces the adjoint
        //   (a "sandwich") term in the trace of dH/d rho_k. We encode this by
        //   augmenting H^{-1} with Q so that:
        //     tr(H^{-1} dotH_std)  ->  tr((H^{-1} + Q) dotH_std),
        //   where:
        //     Q = H^{-1} V H^{-1}.
        //
        //   For logit Jeffreys prior, V is the sensitivity of the Firth penalty
        //   to hat values and reduces to:
        //     V = J^T diag(0.5 - mu) J.
        //
        //   This choice yields:
        //     tr(Q * dotH_std) = tr(H^{-1} V H^{-1} dotH_std),
        //   which is the exact adjoint correction for the Firth term, matching
        //   the fixed-point objective defined by the Gauss-Newton Hessian H.
        let mut q_firth = Array2::<f64>::zeros((p_total, p_total));
        if firth_active {
            let mut j_weighted = j_mat.clone();
            for i in 0..n {
                let nu = 0.5 - mu[i];
                for j in 0..p_total {
                    j_weighted[[i, j]] *= nu;
                }
            }
            // V = J^T diag(0.5 - mu) J (implemented as J^T * (diag(nu) J)).
            let v_mat = crate::faer_ndarray::fast_atb(&j_mat, &j_weighted);

            // Q = H† V H† using pseudo-inverse for spectral consistency
            // H† V H† = U diag(1/λ) U' V U diag(1/λ) U'
            // Let W = U' V U, then Q = U diag(1/λ) W diag(1/λ) U'
            let u_t_v = fast_atb(&h_vecs, &v_mat); // U' V
            let w_mat = fast_ab(&u_t_v, &h_vecs); // U' V U

            // Apply pseudo-inverse eigenvalue scaling and reconstruct via matmuls:
            //   Q = U (D^{-1} W D^{-1}) U^T,  D^{-1}_{ii}=1/λ_i for λ_i>tol else 0.
            // This is algebraically identical to the previous 4-deep summation but
            // reduces reconstruction from O(p^4) scalar updates to O(p^3) BLAS-style ops.
            let mut inv_diag = Array1::<f64>::zeros(p_total);
            for i in 0..p_total {
                let eig = h_eigs[i];
                if eig > h_tol {
                    inv_diag[i] = 1.0 / eig;
                }
            }
            let mut m_mat = w_mat;
            for k in 0..p_total {
                for l in 0..p_total {
                    m_mat[[k, l]] *= inv_diag[k] * inv_diag[l];
                }
            }
            let um = fast_ab(&h_vecs, &m_mat); // U M
            q_firth = fast_ab(&um, &h_vecs.t().to_owned()); // U M U^T
        }

        // Prepare basis derivatives for constraint sensitivity.
        let Some(knot_vector) = state.knot_vector.as_ref() else {
            return Err(EstimationError::RemlOptimizationFailed(
                "missing knot vector for joint analytic gradient".to_string(),
            ));
        };
        let Some(link_transform) = state.link_transform.as_ref() else {
            return Err(EstimationError::RemlOptimizationFailed(
                "missing link transform for joint analytic gradient".to_string(),
            ));
        };
        if link_transform.ncols() != p_link {
            return Err(EstimationError::RemlOptimizationFailed(
                "link transform dimension mismatch".to_string(),
            ));
        }

        let (z_raw, z_c, range_width) = state.standardized_z(&u);
        let n_raw = knot_vector.len().saturating_sub(state.degree + 1);
        if n_raw == 0 {
            return Err(EstimationError::RemlOptimizationFailed(
                "insufficient basis size for analytic gradient".to_string(),
            ));
        }

        use crate::basis::{BasisOptions, Dense, KnotSource, create_basis};
        let (b_raw_arc, _) = create_basis::<Dense>(
            z_c.view(),
            KnotSource::Provided(knot_vector.view()),
            state.degree,
            BasisOptions::value(),
        )
        .map_err(|e| EstimationError::InvalidSpecification(e.to_string()))?;
        let (b_prime_arc, _) = create_basis::<Dense>(
            z_c.view(),
            KnotSource::Provided(knot_vector.view()),
            state.degree,
            BasisOptions::first_derivative(),
        )
        .map_err(|e| EstimationError::InvalidSpecification(e.to_string()))?;

        // Apply the same C^1 linear extension used in build_link_basis:
        //   B_ext(z_raw) = B(z_c) + (z_raw - z_c) * B'(z_c).
        let mut b_raw = b_raw_arc.as_ref().clone();
        let b_prime = b_prime_arc.as_ref();
        let mut needs_ext = false;
        for i in 0..z_raw.len() {
            if (z_raw[i] - z_c[i]).abs() > 1e-12 {
                needs_ext = true;
                break;
            }
        }
        if needs_ext {
            for i in 0..z_raw.len() {
                let dz = z_raw[i] - z_c[i];
                if dz.abs() <= 1e-12 {
                    continue;
                }
                for j in 0..b_raw.ncols() {
                    b_raw[[i, j]] += dz * b_prime[[i, j]];
                }
            }
        }

        // NOTE: With geometric constraints (Greville abscissae), Z is constant w.r.t. β,
        // so dZ/dβ = 0 exactly. The constraint matrix M and its pseudoinverse are no longer
        // needed for the gradient computation.

        // Raw penalty for link block and its projection (constant for this rho).
        let s_raw = if p_link > 0 {
            let greville_for_penalty =
                penalty_greville_abscissae_for_knots(knot_vector, state.degree).unwrap_or(None);
            create_difference_penalty_matrix(
                n_raw,
                2,
                greville_for_penalty.as_ref().map(|g| g.view()),
            )
            .unwrap_or_else(|_| Array2::zeros((n_raw, n_raw)))
        } else {
            Array2::zeros((0, 0))
        };
        let v_pen = if p_link > 0 && s_raw.nrows() == n_raw {
            crate::faer_ndarray::fast_ab(&s_raw, link_transform)
        } else {
            Array2::zeros((n_raw, p_link))
        };

        let mut grad = Array1::<f64>::zeros(rho.len());
        let mut clamp_z_count = 0usize;
        let mut clamp_mu_count = 0usize;
        for i in 0..n {
            if (z_raw[i] - z_c[i]).abs() > 1e-12 {
                clamp_z_count += 1;
            }
            if matches!(state.link, LinkFunction::Logit) && (mu[i] <= 1e-8 || mu[i] >= 1.0 - 1e-8) {
                clamp_mu_count += 1;
            }
        }
        let clamp_z_frac = clamp_z_count as f64 / n.max(1) as f64;
        let clamp_mu_frac = clamp_mu_count as f64 / n.max(1) as f64;
        // Trigger the expensive FD audit only when clamping is substantial.
        // Mild boundary contact is common in expressive link/spline fits and
        // does not warrant a full gradient replacement on every outer step.
        let audit_needed = !converged
            || clamp_z_frac > JOINT_GRAD_AUDIT_CLAMP_FRAC
            || clamp_mu_frac > JOINT_GRAD_AUDIT_CLAMP_FRAC;
        // Penalty det derivative for base penalties.
        let base_reparam = if let Some(invariant) = self.core.base_reparam_invariant.as_ref() {
            stable_reparameterization_with_invariant_engine(
                &self.core.base_rs_list,
                &lambda_base.to_vec(),
                EngineDims::new(state.layout_base.p, self.core.base_rs_list.len()),
                invariant,
            )
        } else {
            stable_reparameterization_engine(
                &self.core.base_rs_list,
                &lambda_base.to_vec(),
                EngineDims::new(state.layout_base.p, self.core.base_rs_list.len()),
            )
        }
        .unwrap_or_else(|_| ReparamResult {
            s_transformed: Array2::zeros((p_base, p_base)),
            log_det: 0.0,
            det1: Array1::zeros(lambda_base.len()),
            qs: Array2::eye(p_base),
            rs_transformed: vec![],
            rs_transposed: vec![],
            e_transformed: Array2::zeros((0, p_base)),
            u_truncated: Array2::zeros((p_base, p_base)),
        });

        // Rank of link penalty for det_term (spectral rank)
        let link_det1_no_ridge = if p_link > 0 && link_penalty.nrows() == p_link {
            use crate::faer_ndarray::FaerEigh;
            let (eigs, _): (Array1<f64>, Array2<f64>) = link_penalty
                .clone()
                .eigh(Side::Lower)
                .unwrap_or_else(|_| (Array1::zeros(p_link), Array2::eye(p_link)));
            let max_eig = eigs.iter().cloned().fold(0.0_f64, f64::max);
            let tol = if max_eig > 0.0 {
                max_eig * 1e-12
            } else {
                1e-12
            };
            eigs.iter().filter(|&&ev| ev > tol).count() as f64
        } else {
            0.0
        };

        let mut dot_u = Array1::<f64>::zeros(n);
        let mut dot_z = Array1::<f64>::zeros(n);
        let mut dot_eta = Array1::<f64>::zeros(n);
        let mut w_prime = Array1::<f64>::zeros(n);
        let mut w_dot = Array1::<f64>::zeros(n);
        let mut b_dot = Array2::<f64>::zeros((n, n_raw));
        // c_dot, weighted_c_dot, weighted_b_dot removed: no longer needed since
        // Z is now computed from Greville abscissae (geometric constraint) and dZ/dβ = 0
        let mut dot_j_theta = Array2::<f64>::zeros((n, p_link));
        let mut dot_j_beta = Array2::<f64>::zeros((n, p_base));
        let mut dot_j = Array2::<f64>::zeros((n, p_total));

        for k in 0..rho.len() {
            let is_link = k == n_base;
            let lambda_k = if is_link { lambda_link } else { lambda_base[k] };

            // Compute rhs for the implicit function theorem (IFT).
            // For inner minimization of -L = -log p(y|β) + 0.5*β'S_λβ:
            //   Stationarity: ∇(-L) = -∇log p + S_λ β = 0
            //   Hessian: H = X'WX + S_λ
            // Differentiating stationarity w.r.t. ρ_k:
            //   H · ∂β/∂ρ_k + λ_k S_k β = 0
            //   ∂β/∂ρ_k = -H^{-1}(λ_k S_k β)
            // So rhs = -λ_k S_k β gives delta = H^{-1}(rhs) = -H^{-1}(λ_k S_k β) = ∂β/∂ρ_k
            let mut rhs = Array1::<f64>::zeros(p_total);
            if is_link {
                if p_link > 0 && link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
                    let sb = link_penalty.dot(&state.beta_link);
                    for i in 0..p_link {
                        rhs[p_base + i] = -lambda_k * sb[i];
                    }
                }
            } else if let Some(s_k) = state.s_base.get(k) {
                if s_k.nrows() == p_base && s_k.ncols() == p_base {
                    let sb = s_k.dot(&state.beta_base);
                    for i in 0..p_base {
                        rhs[i] = -lambda_k * sb[i];
                    }
                }
            }

            // Compute delta = H† rhs using pseudo-inverse for spectral consistency
            // delta = ∂β/∂ρ_k represents how the coefficients move when smoothing changes
            // H† rhs = U diag(1/λ) U' rhs = U diag(1/λ) c where c = U' rhs
            let c = fast_atv(&h_vecs, &rhs);
            let mut delta = Array1::<f64>::zeros(p_total);
            for i in 0..p_total {
                let eig_i = h_eigs[i];
                if eig_i > h_tol {
                    let scaled = c[i] / eig_i;
                    for j in 0..p_total {
                        delta[j] += scaled * h_vecs[[j, i]];
                    }
                }
            }
            let delta_beta = delta.slice(s![..p_base]).to_owned();
            let delta_theta = delta.slice(s![p_base..]).to_owned();

            dot_u.assign(
                &fast_ab(
                    &state.x_base,
                    &delta_beta.clone().insert_axis(ndarray::Axis(1)),
                )
                .column(0)
                .to_owned(),
            );

            // dot_z_raw = d/ dρ_k [ (u - min_u) / range_width ]
            //          = (1 / range_width) * dot_u.
            // For the linearly-extended basis, dB_ext/dz_raw = B'(z_c) everywhere.
            let inv_rw = 1.0 / range_width;
            for i in 0..n {
                dot_z[i] = dot_u[i] * inv_rw;
            }

            dot_eta.assign(
                &fast_ab(&j_mat, &delta.clone().insert_axis(ndarray::Axis(1)))
                    .column(0)
                    .to_owned(),
            );

            // w' for logit (clamped)
            if matches!(state.link, LinkFunction::Logit) {
                const PROB_EPS: f64 = 1e-8;
                const MIN_WEIGHT: f64 = 1e-12;
                for i in 0..n {
                    let mu_i = mu[i];
                    let w_base = mu_i * (1.0 - mu_i);
                    if mu_i <= PROB_EPS || mu_i >= 1.0 - PROB_EPS || w_base < MIN_WEIGHT {
                        w_prime[i] = 0.0;
                    } else {
                        w_prime[i] = state.weights[i] * w_base * (1.0 - 2.0 * mu_i);
                    }
                }
            } else {
                w_prime.fill(0.0);
            }

            for i in 0..n {
                w_dot[i] = w_prime[i] * dot_eta[i];
            }

            // dB_ext/dρ = (dB_ext/dz_raw) * dz_raw/dρ with dB_ext/dz_raw = B'(z_c).
            b_dot.fill(0.0);
            for i in 0..n {
                let dz = dot_z[i];
                if dz == 0.0 {
                    continue;
                }
                for j in 0..n_raw {
                    b_dot[[i, j]] = b_prime[[i, j]] * dz;
                }
            }

            // Z is now computed from Greville abscissae (geometric constraint) and is
            // constant w.r.t. β. Therefore dZ/dβ = 0 exactly, and z_dot = 0.
            // This eliminates the entire M_dot computation and constraint sensitivity terms.
            let z_dot = Array2::<f64>::zeros((n_raw, p_link));

            dot_j_theta.assign(&crate::faer_ndarray::fast_ab(&b_dot, link_transform));
            dot_j_theta += &crate::faer_ndarray::fast_ab(&b_raw, &z_dot);

            // dot_g_prime
            let mut dot_g_prime = Array1::<f64>::zeros(n);
            for i in 0..n {
                dot_g_prime[i] = g_second[i] * dot_u[i];
            }
            let b_prime_delta = b_prime_u.dot(&delta_theta);
            dot_g_prime += &b_prime_delta;

            let z_dot_theta = z_dot.dot(&state.beta_link);
            if z_dot_theta.len() == n_raw {
                let mut z_term = b_prime.dot(&z_dot_theta);
                for i in 0..n {
                    z_term[i] *= inv_rw;
                }
                dot_g_prime += &z_term;
            }

            // dot_J_beta = diag(dot_g_prime) X
            dot_j_beta.fill(0.0);
            for i in 0..n {
                let scale = dot_g_prime[i];
                for j in 0..p_base {
                    dot_j_beta[[i, j]] = scale * state.x_base[[i, j]];
                }
            }

            // dot_J = [dot_J_beta | dot_J_theta]
            dot_j.fill(0.0);
            dot_j.slice_mut(s![.., ..p_base]).assign(&dot_j_beta);
            dot_j.slice_mut(s![.., p_base..]).assign(&dot_j_theta);
            // Trace for likelihood curvature: 2 tr(K_w * dot_J) + tr(diag(J H^{-1} J^T) * W_dot)
            let mut trace = 0.0;
            let mut trace_k = 0.0;
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..p_total {
                    acc += k_w[[j, i]] * dot_j[[i, j]];
                }
                trace_k += acc;
                trace += diag_proj[i] * w_dot[i];
            }
            trace += 2.0 * trace_k;

            // Trace for lambda_k * S_k using pseudo-inverse (Wood 2011)
            // tr(H† S_k) = Σ (u_i^T S_k u_i) / λ_i for positive eigenvalues
            let mut s_k_full = Array2::<f64>::zeros((p_total, p_total));
            if is_link {
                s_k_full
                    .slice_mut(s![p_base.., p_base..])
                    .assign(&link_penalty);
            } else if let Some(s_k) = state.s_base.get(k) {
                if s_k.nrows() == p_base && s_k.ncols() == p_base {
                    s_k_full.slice_mut(s![..p_base, ..p_base]).assign(s_k);
                }
            }
            if p_total > 0 {
                // Pseudo-inverse trace: Σ (u_i^T S_k u_i) / λ_i for positive eigenvalues
                let mut trace_lambda = 0.0;
                for i in 0..p_total {
                    let eig_i = h_eigs[i];
                    if eig_i > h_tol {
                        // Extract eigenvector u_i as owned array
                        let u_i: Array1<f64> = h_vecs.column(i).to_owned();
                        // Compute u_i^T S_k u_i
                        let s_k_u = s_k_full.dot(&u_i);
                        let quadratic = u_i.dot(&s_k_u);
                        trace_lambda += quadratic / eig_i;
                    }
                    // For small/negative eigenvalues, contribution is 0 (pseudo-inverse)
                }
                trace += lambda_k * trace_lambda;
            }

            // Penalty manifold sensitivity for the link block: dot(S_link) = Z_dot^T S_raw Z + Z^T S_raw Z_dot.
            if p_link > 0 && v_pen.nrows() == n_raw {
                let left = crate::faer_ndarray::fast_atb(&z_dot, &v_pen);
                let right = crate::faer_ndarray::fast_atb(&v_pen, &z_dot);
                let dot_s_link = left + right;
                let mut trace_penalty = 0.0;
                for i in 0..p_link {
                    for j in 0..p_link {
                        trace_penalty += h_inv_theta[[i, j]] * dot_s_link[[j, i]];
                    }
                }
                trace += lambda_link * trace_penalty;
            }

            // Firth adjoint correction: add tr(Q * dotH_std) for logit Firth.
            //
            // dotH_std is the directional derivative of the standard Hessian:
            //   dotH_std = dotJ^T W J + J^T W dotJ + J^T W_dot J + dotS_lambda
            // where dotS_lambda includes the penalty manifold sensitivity of the link block.
            if firth_active {
                // dotH_std = dotJ^T W J + J^T W dotJ + J^T W_dot J, with penalty sensitivity added to theta block.
                let mut dot_h_std = Array2::<f64>::zeros((p_total, p_total));
                {
                    use crate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
                    use faer::linalg::matmul::matmul;
                    use faer::{Accum, Par, get_global_parallelism};

                    let par = if n < 128 || p_total < 128 {
                        Par::Seq
                    } else {
                        get_global_parallelism()
                    };
                    let mut dot_h_view = array2_to_mat_mut(&mut dot_h_std);

                    let mut wj = j_mat.clone();
                    for i in 0..n {
                        let w = weights[i];
                        for j in 0..p_total {
                            wj[[i, j]] *= w;
                        }
                    }
                    let mut wdotj = j_mat.clone();
                    for i in 0..n {
                        let w = w_dot[i];
                        for j in 0..p_total {
                            wdotj[[i, j]] *= w;
                        }
                    }
                    let mut wdotj2 = dot_j.clone();
                    for i in 0..n {
                        let w = weights[i];
                        for j in 0..p_total {
                            wdotj2[[i, j]] *= w;
                        }
                    }

                    // Bind FaerArrayView temporaries to local variables to extend their lifetime
                    let dj_wrapper = FaerArrayView::new(&dot_j);
                    let dj_view = dj_wrapper.as_ref();
                    let wj_wrapper = FaerArrayView::new(&wj);
                    let wj_view = wj_wrapper.as_ref();
                    let j_wrapper = FaerArrayView::new(&j_mat);
                    let j_view = j_wrapper.as_ref();
                    let wdotj_wrapper = FaerArrayView::new(&wdotj);
                    let wdotj_view = wdotj_wrapper.as_ref();
                    let wdotj2_wrapper = FaerArrayView::new(&wdotj2);
                    let wdotj2_view = wdotj2_wrapper.as_ref();

                    matmul(
                        dot_h_view.as_mut(),
                        Accum::Replace,
                        dj_view.transpose(),
                        wj_view,
                        1.0,
                        par,
                    );
                    matmul(
                        dot_h_view.as_mut(),
                        Accum::Add,
                        j_view.transpose(),
                        wdotj2_view,
                        1.0,
                        par,
                    );
                    matmul(
                        dot_h_view.as_mut(),
                        Accum::Add,
                        j_view.transpose(),
                        wdotj_view,
                        1.0,
                        par,
                    );
                }

                // Add penalty manifold sensitivity into dotH_std theta block.
                if p_link > 0 && v_pen.nrows() == n_raw {
                    let left = crate::faer_ndarray::fast_atb(&z_dot, &v_pen);
                    let right = crate::faer_ndarray::fast_atb(&v_pen, &z_dot);
                    let dot_s_link = left + right;
                    let mut link_block = dot_h_std.slice_mut(s![p_base.., p_base..]);
                    link_block += &dot_s_link.mapv(|v| v * lambda_link);
                }

                // Accumulate tr(Q * dotH_std) = sum_ij Q_ij * dotH_std_ji.
                let mut trace_q = 0.0;
                for i in 0..p_total {
                    for j in 0..p_total {
                        trace_q += q_firth[[i, j]] * dot_h_std[[j, i]];
                    }
                }
                trace += trace_q;
            }

            let penalty_term = if is_link {
                if p_link > 0 && link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
                    let sb = link_penalty.dot(&state.beta_link);
                    0.5 * lambda_k * state.beta_link.dot(&sb)
                } else {
                    0.0
                }
            } else if let Some(s_k) = state.s_base.get(k) {
                if s_k.nrows() == p_base && s_k.ncols() == p_base {
                    let sb = s_k.dot(&state.beta_base);
                    0.5 * lambda_k * state.beta_base.dot(&sb)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Spectral det_term (Wood 2011): d/dλ_k[log|S_λ|_+] = rank(S_k)
            // With spectral log-det, no ridge adjustment is needed.
            let det_term = if is_link {
                0.5 * link_det1_no_ridge
            } else if k < base_reparam.det1.len() {
                0.5 * base_reparam.det1[k]
            } else {
                0.0
            };

            let grad_laml = penalty_term - det_term + 0.5 * trace;
            grad[k] = grad_laml;
        }

        Ok((grad, audit_needed))
    }

    /// Compute gradient of LAML w.r.t. ρ using analytic path, with FD fallback.
    pub fn compute_gradient(&mut self, rho: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        match self.compute_gradient_analytic(rho) {
            Ok((grad, audit_needed)) => {
                const GRAD_FD_REL_TOL: f64 = 1e-2;
                let eval_num = self.eval.eval_count;
                let audit_slot = should_sample_joint_fd_audit(eval_num);
                if audit_needed && audit_slot {
                    match self.compute_gradient_fd(rho) {
                        Ok(fd_grad) => {
                            let mut diff_norm = 0.0;
                            let mut fd_norm = 0.0;
                            for i in 0..fd_grad.len() {
                                let d = grad[i] - fd_grad[i];
                                diff_norm += d * d;
                                fd_norm += fd_grad[i] * fd_grad[i];
                            }
                            let rel = diff_norm.sqrt() / (fd_norm.sqrt() + 1.0);
                            if rel > GRAD_FD_REL_TOL {
                                return Err(EstimationError::RemlOptimizationFailed(format!(
                                    "Analytic/FD gradient mismatch (rel {:.3e}) in joint REML.",
                                    rel
                                )));
                            }
                        }
                        Err(err) => {
                            eprintln!("[JOINT][REML] FD audit failed: {err}");
                        }
                    }
                } else if audit_needed && eval_num > JOINT_GRAD_AUDIT_WARMUP_EVALS {
                    log::debug!(
                        "[JOINT][REML] Skipping FD audit at eval {} (periodic audit every {} evals).",
                        eval_num,
                        JOINT_GRAD_AUDIT_INTERVAL
                    );
                }
                Ok(grad)
            }
            Err(err) => {
                eprintln!(
                    "[JOINT][REML] Analytic gradient unavailable: {err}. Falling back to FD."
                );
                self.compute_gradient_fd(rho)
            }
        }
    }

    #[cfg(test)]
    pub fn compute_gradient_analytic_for_test(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<(Array1<f64>, bool), EstimationError> {
        self.compute_gradient_analytic(rho)
    }

    #[cfg(test)]
    pub fn compute_gradient_fd_for_test(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        self.compute_gradient_fd(rho)
    }

    /// Combined cost and gradient for outer optimization
    pub fn cost_and_grad(&mut self, rho: &Array1<f64>) -> (f64, Array1<f64>) {
        self.eval.last_outer_step_norm = self.eval.last_eval_rho.as_ref().map(|prev| {
            let mut sum = 0.0_f64;
            for i in 0..rho.len().min(prev.len()) {
                let d = rho[i] - prev[i];
                sum += d * d;
            }
            sum.sqrt()
        });
        self.eval.last_eval_rho = Some(rho.clone());
        self.eval.eval_count += 1;
        let eval_num = self.eval.eval_count;
        let cost = match self.compute_cost(rho) {
            Ok(val) if val.is_finite() => val,
            Ok(_) => {
                eprintln!("[JOINT][REML] Non-finite cost; returning large penalty.");
                visualizer::push_diagnostic("warning: non-finite joint REML cost encountered");
                return (f64::INFINITY, Array1::from_elem(rho.len(), f64::NAN));
            }
            Err(err) => {
                eprintln!("[JOINT][REML] Cost evaluation failed: {err}");
                visualizer::push_diagnostic(&format!("warning: cost evaluation failed: {err}"));
                return (f64::INFINITY, Array1::from_elem(rho.len(), f64::NAN));
            }
        };
        let grad = match self.compute_gradient(rho) {
            Ok(grad) => grad,
            Err(err) => {
                eprintln!("[JOINT][REML] Gradient evaluation failed: {err}");
                visualizer::push_diagnostic(&format!("warning: gradient evaluation failed: {err}"));
                Array1::from_elem(rho.len(), f64::NAN)
            }
        };
        let grad_norm = grad.dot(&grad).sqrt();
        // Push side-panel state before the chart update so the next rendered frame
        // includes objective + EDF + diagnostics together.
        visualizer::set_edf_terms(&self.eval.cached_edf_terms);
        visualizer::set_diagnostics(
            self.eval.last_hessian_condition,
            self.eval.last_outer_step_norm,
            Some(self.core.state.ridge_used),
        );
        visualizer::update(cost, grad_norm, "optimizing", eval_num as f64, "eval");
        if !self.eval.last_converged {
            visualizer::push_diagnostic(&format!(
                "inner backfit not converged (iterations={})",
                self.eval.last_backfit_iterations
            ));
        } else {
            visualizer::push_diagnostic(&format!(
                "inner backfit converged in {} iterations",
                self.eval.last_backfit_iterations
            ));
        }
        if self.core.state.ridge_used > 0.0 {
            visualizer::push_diagnostic(&format!(
                "stabilization ridge active: {:.3e}",
                self.core.state.ridge_used
            ));
        }
        (cost, grad)
    }

    fn compute_hessian_fd(&mut self, rho: &Array1<f64>) -> Result<Array2<f64>, EstimationError> {
        let n_rho = rho.len();
        let mut hess = Array2::<f64>::zeros((n_rho, n_rho));
        if n_rho == 0 {
            return Ok(hess);
        }
        for j in 0..n_rho {
            let h = (1e-4 * (1.0 + rho[j].abs())).max(1e-6);
            let mut rho_plus = rho.clone();
            rho_plus[j] += h;
            let mut rho_minus = rho.clone();
            rho_minus[j] -= h;
            let g_plus = self.compute_gradient(&rho_plus)?;
            let g_minus = self.compute_gradient(&rho_minus)?;
            if g_plus.len() != n_rho || g_minus.len() != n_rho {
                return Err(EstimationError::RemlOptimizationFailed(
                    "joint FD Hessian gradient length mismatch".to_string(),
                ));
            }
            for i in 0..n_rho {
                hess[[i, j]] = (g_plus[i] - g_minus[i]) / (2.0 * h);
            }
        }
        for i in 0..n_rho {
            for j in 0..i {
                let v = 0.5 * (hess[[i, j]] + hess[[j, i]]);
                hess[[i, j]] = v;
                hess[[j, i]] = v;
            }
        }
        if hess.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "non-finite values in joint FD Hessian".to_string(),
            ));
        }
        Ok(hess)
    }

    /// Extract final result after optimization
    pub fn into_result(self) -> JointModelResult {
        let cached_edf = self.eval.cached_edf;
        let cached_iters = self.eval.last_backfit_iterations;
        let cached_converged = self.eval.last_converged;
        let state = self.core.state;
        let rho = state.rho.clone();
        let knot_range = state.knot_range.unwrap_or((0.0, 1.0));
        let knot_vector = state
            .knot_vector
            .clone()
            .unwrap_or_else(|| Array1::zeros(0));
        let b_wiggle = state.build_link_basis_from_state(&state.base_linear_predictor());
        let eta = state.compute_eta_full(&state.base_linear_predictor(), &b_wiggle);
        let mut mu = Array1::<f64>::zeros(state.n_obs());
        let mut weights = Array1::<f64>::zeros(state.n_obs());
        let mut z = Array1::<f64>::zeros(state.n_obs());
        crate::pirls::update_glm_vectors(
            state.y,
            &eta,
            state.link.clone(),
            state.weights,
            &mut mu,
            &mut weights,
            &mut z,
        );
        let deviance = state.compute_deviance(&mu);
        JointModelResult {
            beta_base: state.beta_base,
            beta_link: state.beta_link,
            lambdas: rho.mapv(f64::exp).to_vec(),
            deviance,
            edf: cached_edf.unwrap_or(f64::NAN),
            backfit_iterations: cached_iters,
            converged: cached_converged,
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
        }
    }
}

/// Fit joint model with proper REML-based lambda selection via BFGS
///
/// Uses Laplace approximate marginal likelihood (LAML) with numerical gradient.
/// For nonlinear g(u), the Hessian is Gauss-Newton (approximate).
pub(crate) fn fit_joint_model_with_reml<'a>(
    y: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    x_base: ArrayView2<'a, f64>,
    s_base: Vec<Array2<f64>>,
    layout_base: EngineDims,
    link: LinkFunction,
    config: &JointModelConfig,
    covariate_se: Option<Array1<f64>>,
) -> Result<JointModelResult, EstimationError> {
    let _viz_guard = visualizer::init_guard(true);
    visualizer::set_stage("joint", "initializing");
    if config.firth_bias_reduction && matches!(link, LinkFunction::Logit) {
        visualizer::push_diagnostic("firth bias reduction enabled (separation protection)");
    }
    let quad_ctx = QuadratureContext::new();

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
        quad_ctx,
    );

    let n_base = reml_state.core.state.s_base.len();
    let heuristic_lambda = {
        let state = &reml_state.core.state;
        state
            .knot_vector
            .as_ref()
            .map(|knots| baseline_lambda_seed(knots, state.degree, 2))
    };
    let heuristic_lambdas = heuristic_lambda.map(|lambda| vec![lambda; n_base + 1]);
    let seed_config = SeedConfig {
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
        risk_profile: SeedRiskProfile::Survival,
    };
    let seed_candidates =
        generate_rho_candidates(n_base + 1, heuristic_lambdas.as_deref(), &seed_config);
    // Bounded ρ optimization.
    const RHO_BOUND: f64 = 30.0;

    let mut candidate_plans: Vec<(String, Array1<f64>)> = if seed_candidates.is_empty() {
        vec![("fallback-symmetric".to_string(), Array1::zeros(n_base + 1))]
    } else {
        seed_candidates
            .into_iter()
            .enumerate()
            .map(|(i, rho)| (format!("seed_{i}"), rho))
            .collect()
    };

    let mut successful_runs: Vec<(String, BfgsSolution, f64)> = Vec::new();
    let mut last_error: Option<EstimationError> = None;
    let total_candidates = candidate_plans.len();
    let mut candidate_idx = 0usize;
    let snapshot = JointRemlSnapshot::new(&reml_state);

    for (label, rho) in candidate_plans.drain(..) {
        candidate_idx += 1;
        visualizer::set_stage("joint", &format!("candidate {label}"));
        visualizer::set_progress("Candidates", candidate_idx, Some(total_candidates));

        snapshot.restore(&mut reml_state);
        let lower = Array1::<f64>::from_elem(rho.len(), -RHO_BOUND);
        let upper = Array1::<f64>::from_elem(rho.len(), RHO_BOUND);
        let mut full_cache: Option<(Array1<f64>, f64, Array1<f64>, Array2<f64>)> = None;
        let mut solver = NewtonTrustRegion::new(rho, |rho, request| {
            if let Some((rho_c, cost_c, grad_c, hess_c)) = &full_cache
                && rho.len() == rho_c.len()
                && rho
                    .iter()
                    .zip(rho_c.iter())
                    .all(|(&a, &b)| (a - b).abs() <= 1e-12)
            {
                return Ok(match request {
                    ObjectiveRequest::CostOnly => ObjectiveSample::cost_only(*cost_c),
                    ObjectiveRequest::CostAndGradient => {
                        ObjectiveSample::cost_and_gradient(*cost_c, grad_c.clone())
                    }
                    ObjectiveRequest::GradientAndHessian
                    | ObjectiveRequest::CostGradientHessian => {
                        ObjectiveSample::cost_gradient_hessian(
                            *cost_c,
                            grad_c.clone(),
                            hess_c.clone(),
                        )
                    }
                });
            }

            let cost = match reml_state.compute_cost(rho) {
                Ok(v) => v,
                Err(_) => f64::INFINITY,
            };
            if !cost.is_finite() {
                return Err(ObjectiveEvalError::recoverable(
                    "non-finite joint REML cost",
                ));
            }
            match request {
                ObjectiveRequest::CostOnly => Ok(ObjectiveSample::cost_only(cost)),
                ObjectiveRequest::CostAndGradient => {
                    let grad = match reml_state.compute_gradient(rho) {
                        Ok(g) if g.iter().all(|v| v.is_finite()) => g,
                        _ => {
                            return Err(ObjectiveEvalError::recoverable(
                                "non-finite joint REML gradient",
                            ));
                        }
                    };
                    Ok(ObjectiveSample::cost_and_gradient(cost, grad))
                }
                ObjectiveRequest::GradientAndHessian | ObjectiveRequest::CostGradientHessian => {
                    let grad = match reml_state.compute_gradient(rho) {
                        Ok(g) if g.iter().all(|v| v.is_finite()) => g,
                        _ => {
                            return Err(ObjectiveEvalError::recoverable(
                                "non-finite joint REML gradient",
                            ));
                        }
                    };
                    let hess = match reml_state.compute_hessian_fd(rho) {
                        Ok(h) if h.iter().all(|v| v.is_finite()) => h,
                        _ => {
                            return Err(ObjectiveEvalError::recoverable(
                                "non-finite joint REML Hessian",
                            ));
                        }
                    };
                    full_cache = Some((rho.clone(), cost, grad.clone(), hess.clone()));
                    Ok(ObjectiveSample::cost_gradient_hessian(cost, grad, hess))
                }
            }
        })
        .with_bounds(lower, upper, 1e-6)
        .with_tolerance(config.reml_tol)
        .with_max_iterations(config.max_reml_iter)
        .with_initial_trust_radius(1.0)
        .with_max_trust_radius(1e6)
        .with_acceptance_threshold(0.1);

        let solution = match solver.run() {
            Ok(solution) => solution,
            Err(wolfe_bfgs::NewtonTrustRegionError::MaxIterationsReached { last_solution }) => {
                *last_solution
            }
            Err(e) => {
                last_error = Some(EstimationError::RemlOptimizationFailed(format!(
                    "Newton trust-region failed for joint model: {e:?}"
                )));
                continue;
            }
        };
        let final_value = solution.final_value;
        successful_runs.push((label, solution, final_value));
    }

    let (_, best_solution, _) =
        match successful_runs
            .into_iter()
            .min_by(|a, b| match a.2.partial_cmp(&b.2) {
                Some(order) => order,
                None => std::cmp::Ordering::Equal,
            }) {
            Some(best) => best,
            None => {
                return Err(last_error.unwrap_or_else(|| {
                    EstimationError::RemlOptimizationFailed(
                        "All joint REML candidate runs failed.".to_string(),
                    )
                }));
            }
        };

    let best_rho = best_solution.final_point;
    let _ = reml_state.compute_cost(&best_rho)?;
    Ok(reml_state.into_result())
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
    b_wiggle: &Array2<f64>,
) -> Array1<f64> {
    use crate::basis::evaluate_bspline_derivative_scalar_into;
    use crate::basis::internal::BsplineScratch;

    let n = u.len();
    let mut deriv = Array1::<f64>::ones(n);

    if b_wiggle.ncols() == 0 || state.beta_link.is_empty() {
        return deriv;
    }
    let Some(knot_vector) = state.knot_vector.as_ref() else {
        return deriv;
    };

    let (_z_raw, z_c, range_width) = state.standardized_z(u);
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
        let d_wiggle_dz: f64 = if link_transform.nrows() == n_raw {
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

        deriv[i] = 1.0 + d_wiggle_dz / range_width;
    }

    deriv
}

fn compute_link_derivative_terms_from_state(
    state: &JointModelState,
    u: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    use crate::basis::evaluate_bspline_derivative_scalar_into;
    use crate::basis::evaluate_bspline_second_derivative_scalar_into;
    use crate::basis::internal::BsplineScratch;

    let n = u.len();
    let p_link = state.beta_link.len();
    let mut g_prime = Array1::<f64>::ones(n);
    let mut g_second = Array1::<f64>::zeros(n);
    let mut b_prime_u = Array2::<f64>::zeros((n, p_link));

    if p_link == 0 {
        return (g_prime, g_second, b_prime_u);
    }
    let Some(knot_vector) = state.knot_vector.as_ref() else {
        return (g_prime, g_second, b_prime_u);
    };
    let Some(link_transform) = state.link_transform.as_ref() else {
        return (g_prime, g_second, b_prime_u);
    };

    let n_raw = knot_vector.len().saturating_sub(state.degree + 1);
    if n_raw == 0 || link_transform.nrows() != n_raw || link_transform.ncols() != p_link {
        return (g_prime, g_second, b_prime_u);
    }

    let (z_raw, z_c, range_width) = state.standardized_z(u);
    let inv_rw = 1.0 / range_width;
    let inv_rw2 = inv_rw * inv_rw;

    let mut deriv_raw = vec![0.0; n_raw];
    let num_basis_lower = knot_vector.len().saturating_sub(state.degree);
    let mut lower_basis = vec![0.0; num_basis_lower];
    let mut lower_scratch = BsplineScratch::new(state.degree.saturating_sub(1));

    let mut second_raw = vec![0.0; n_raw];
    let num_basis_lower_second = knot_vector.len().saturating_sub(state.degree - 1);
    let mut deriv_lower = vec![0.0; num_basis_lower_second.saturating_sub(1)];
    let mut lower_basis_second = vec![0.0; num_basis_lower_second];
    let mut lower_scratch_second = BsplineScratch::new(state.degree.saturating_sub(2));

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

        let mut d_wiggle_dz = 0.0;
        for c in 0..p_link {
            let mut b_prime_c = 0.0;
            for r in 0..n_raw {
                b_prime_c += deriv_raw[r] * link_transform[[r, c]];
            }
            b_prime_u[[i, c]] = b_prime_c * inv_rw;
            d_wiggle_dz += b_prime_c * state.beta_link[c];
        }
        g_prime[i] = 1.0 + d_wiggle_dz * inv_rw;

        // For the linearly-extended basis, the curvature is zero outside [0, 1].
        // That is, d²/dz_raw² B_ext(z_raw) = 0 when z_raw != z_c.
        if (z_raw[i] - z_c[i]).abs() <= 1e-12 {
            second_raw.fill(0.0);
            if evaluate_bspline_second_derivative_scalar_into(
                z_i,
                knot_vector.view(),
                state.degree,
                &mut second_raw,
                &mut deriv_lower,
                &mut lower_basis_second,
                &mut lower_scratch_second,
            )
            .is_err()
            {
                continue;
            }
            let mut d2_wiggle_dz2 = 0.0;
            for c in 0..p_link {
                let mut b_second_c = 0.0;
                for r in 0..n_raw {
                    b_second_c += second_raw[r] * link_transform[[r, c]];
                }
                d2_wiggle_dz2 += b_second_c * state.beta_link[c];
            }
            g_second[i] = d2_wiggle_dz2 * inv_rw2;
        }
    }

    (g_prime, g_second, b_prime_u)
}

/// Public version for use in HMC Hessian computation
pub fn compute_link_derivative_from_result_public(
    result: &JointModelResult,
    eta_base: &Array1<f64>,
    b_wiggle: &Array2<f64>,
) -> Array1<f64> {
    compute_link_derivative_from_result(result, eta_base, b_wiggle)
}

fn compute_link_derivative_from_result(
    result: &JointModelResult,
    eta_base: &Array1<f64>,
    b_wiggle: &Array2<f64>,
) -> Array1<f64> {
    use crate::basis::evaluate_bspline_derivative_scalar_into;
    use crate::basis::internal::BsplineScratch;

    let n = eta_base.len();
    let mut deriv = Array1::<f64>::ones(n);
    if b_wiggle.ncols() == 0 || result.beta_link.is_empty() {
        return deriv;
    }

    let (min_u, max_u) = result.knot_range;
    let range_width = (max_u - min_u).max(1e-6);
    let z_raw: Array1<f64> = eta_base.mapv(|u| (u - min_u) / range_width);
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
        let d_wiggle_dz: f64 = if result.link_transform.nrows() == n_raw {
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

        deriv[i] = 1.0 + d_wiggle_dz / range_width;
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
) -> JointModelPrediction {
    let n = eta_base.len();

    // Use stored knot range from training for consistent standardization
    let (min_u, max_u) = result.knot_range;
    let range_width = (max_u - min_u).max(1e-6);

    // Standardize and apply the same linear extension used during training.
    let z_raw: Array1<f64> = eta_base.mapv(|u| (u - min_u) / range_width);
    let z_c: Array1<f64> = z_raw.mapv(|z| z.clamp(0.0, 1.0));

    // Build B-spline basis at prediction points using stored parameters
    let b_wiggle = match create_basis::<Dense>(
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

    // Compute η_cal = u + B_wiggle · θ
    let eta_cal: Array1<f64> = if b_wiggle.ncols() > 0 && result.beta_link.len() == b_wiggle.ncols()
    {
        let wiggle = b_wiggle.dot(&result.beta_link);
        eta_base + &wiggle
    } else {
        eta_base.clone()
    };

    // Compute effective SE if base SE provided
    let quad_ctx = QuadratureContext::new();
    let (probabilities, effective_se) = if let Some(se) = se_base {
        // Compute link derivative for uncertainty propagation
        let deriv = compute_link_derivative_from_result(result, eta_base, &b_wiggle);

        // Effective SE = |g'(η)| × SE_base
        let eff_se: Array1<f64> = deriv.mapv(f64::abs) * se;

        let probs = match result.link {
            LinkFunction::Logit => (0..n)
                .map(|i| crate::quadrature::logit_posterior_mean(&quad_ctx, eta_cal[i], eff_se[i]))
                .collect::<Array1<f64>>(),
            LinkFunction::Probit => (0..n)
                .map(|i| crate::quadrature::probit_posterior_mean(eta_cal[i], eff_se[i]))
                .collect::<Array1<f64>>(),
            LinkFunction::CLogLog => (0..n)
                .map(|i| {
                    crate::quadrature::cloglog_posterior_mean(&quad_ctx, eta_cal[i], eff_se[i])
                })
                .collect::<Array1<f64>>(),
            LinkFunction::Identity => eta_cal.clone(),
        };

        (probs, Some(eff_se))
    } else {
        let probs = match result.link {
            LinkFunction::Logit => eta_cal.mapv(|e| 1.0 / (1.0 + (-e).exp())),
            LinkFunction::Probit => eta_cal.mapv(normal_cdf_approx),
            LinkFunction::CLogLog => eta_cal.mapv(|e| {
                let e = e.clamp(-30.0, 30.0);
                1.0 - (-(e.exp())).exp()
            }),
            LinkFunction::Identity => eta_cal.clone(),
        };
        (probs, None)
    };

    JointModelPrediction {
        eta: eta_cal,
        probabilities,
        effective_se,
    }
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
        let quad_ctx = QuadratureContext::new();

        let state = JointModelState::new(
            y.view(),
            weights.view(),
            x.view(),
            s,
            layout,
            LinkFunction::Logit,
            &config,
            quad_ctx,
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
            knot_range: (0.0, 1.0),
            knot_vector,
            link_transform: Array2::eye(num_basis),
            degree,
            link: LinkFunction::Logit,
            s_link_constrained: Array2::eye(num_basis),
            ridge_used: 0.0,
        };

        // Test with base eta values
        let eta_base = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        // Predict without SE (should give sigmoid of eta)
        let pred = predict_joint(&result, &eta_base, None);

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
    fn test_predict_joint_with_se() {
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
            knot_range: (0.0, 1.0),
            knot_vector,
            link_transform: Array2::eye(num_basis),
            degree,
            link: LinkFunction::Logit,
            s_link_constrained: Array2::eye(num_basis),
            ridge_used: 0.0,
        };

        let eta_base = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let se_base = Array1::from_vec(vec![0.5, 0.5, 0.5]);

        let pred = predict_joint(&result, &eta_base, Some(&se_base));

        assert!(pred.effective_se.is_some());
        let eff_se = pred.effective_se.unwrap();
        assert_eq!(eff_se.len(), 3);

        // With zero wiggle, g'(u)=1 so effective SE equals base SE.
        for i in 0..3 {
            assert!((eff_se[i] - se_base[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_joint_analytic_gradient_matches_fd() {
        fn run_case(
            n: usize,
            p: usize,
            ill_conditioned: bool,
            seed: u64,
        ) -> Result<Option<(Array1<f64>, Array1<f64>, bool)>, String> {
            let weights = Array1::ones(n);
            let s = vec![Array2::eye(p)];
            let layout = EngineDims::new(p, 1);
            let config = JointModelConfig::default();

            let mut last_err: Option<String> = None;
            for attempt in 0..3 {
                let mut rng = StdRng::seed_from_u64(seed + attempt as u64);
                let mut x = Array2::<f64>::zeros((n, p));
                for i in 0..n {
                    for j in 0..p {
                        x[[i, j]] = rng.random_range(-2.0..2.0);
                    }
                }
                if ill_conditioned && p > 1 {
                    for i in 0..n {
                        x[[i, 1]] = x[[i, 0]] * (1.0 + 1e-6) + rng.random_range(-1e-8..1e-8);
                    }
                }

                let mut beta_true = Array1::<f64>::zeros(p);
                for j in 0..p {
                    beta_true[j] = rng.random_range(-1.0..1.0);
                }
                let eta = x.dot(&beta_true);
                let mut y = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let mu = 1.0 / (1.0 + (-eta[i]).exp());
                    y[i] = if ill_conditioned {
                        if rng.random::<f64>() < mu { 1.0 } else { 0.0 }
                    } else {
                        mu
                    };
                }

                let mut reml_state = JointRemlState::new(
                    y.view(),
                    weights.view(),
                    x.view(),
                    s.clone(),
                    layout.clone(),
                    LinkFunction::Logit,
                    &config,
                    None,
                    QuadratureContext::new(),
                );
                {
                    let state = &mut reml_state.core.state;
                    state.beta_base = beta_true.clone();
                    reml_state.eval.cached_beta_base = beta_true.clone();
                    // Reset knot range and constraints so basis is built from the current u,
                    // not the zero-initialized predictor from JointRemlState::new.
                    state.knot_range = None;
                    state.knot_vector = None;
                    state.link_transform = None;
                    state.s_link_constrained = None;
                    state.geometric_link_transform = None;
                    state.geometric_s_link_constrained = None;
                    let u = state.base_linear_predictor();
                    if state.build_link_basis(&u).is_err() {
                        last_err = Some("link basis failed".to_string());
                        continue;
                    }
                }

                let rho = Array1::from_vec(vec![0.0, 5.0]);
                match reml_state.compute_gradient_analytic_for_test(&rho) {
                    Ok((ga, audit_needed)) => match reml_state.compute_gradient_fd_for_test(&rho) {
                        Ok(gf) => return Ok(Some((ga, gf, audit_needed))),
                        Err(err) => {
                            last_err = Some(err.to_string());
                            continue;
                        }
                    },
                    Err(err) => {
                        last_err = Some(err.to_string());
                        continue;
                    }
                }
            }
            Err(last_err.unwrap_or_else(|| "unknown".to_string()))
        }

        for &(n, ill) in &[(120, false), (120, true), (800, false), (800, true)] {
            match run_case(n, 6, ill, 123) {
                Ok(Some((grad_analytic, grad_fd, audit_needed))) => {
                    let mut diff_norm = 0.0;
                    let mut fd_norm = 0.0;
                    for i in 0..grad_fd.len() {
                        let d = grad_analytic[i] - grad_fd[i];
                        diff_norm += d * d;
                        fd_norm += grad_fd[i] * grad_fd[i];
                    }
                    let rel = diff_norm.sqrt() / (fd_norm.sqrt().max(1.0));
                    assert!(
                        rel < 2e-2,
                        "analytic/FD gradient mismatch: n={n} ill={ill} audit={audit_needed} rel={rel:.3e}"
                    );
                }
                Ok(None) => {
                    if !ill {
                        panic!("analytic gradient: no stable case for n={n}");
                    }
                }
                Err(err) => {
                    if !ill {
                        panic!("analytic gradient: {err}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_joint_fd_audit_sampling_schedule() {
        // Warmup phase samples every eval.
        for eval in 1..=JOINT_GRAD_AUDIT_WARMUP_EVALS {
            assert!(should_sample_joint_fd_audit(eval));
        }
        // After warmup, only periodic checkpoints are sampled.
        assert!(!should_sample_joint_fd_audit(
            JOINT_GRAD_AUDIT_WARMUP_EVALS + 1
        ));
        assert!(!should_sample_joint_fd_audit(JOINT_GRAD_AUDIT_INTERVAL - 1));
        assert!(should_sample_joint_fd_audit(JOINT_GRAD_AUDIT_INTERVAL));
        assert!(should_sample_joint_fd_audit(JOINT_GRAD_AUDIT_INTERVAL * 2));
    }
}
