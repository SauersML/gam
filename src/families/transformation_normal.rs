//! Conditional transformation model: estimate h(y|x) such that h(Y|x) ~ N(0,1).
//!
//! Given a response variable y and covariates x with a pre-built covariate design
//! operator, this family estimates a smooth monotone transformation h(y | x) mapping
//! the conditional distribution of Y|x onto a standard normal.
//!
//! The response-direction basis is `[1, y, anchored B-spline deviations]`, tensored
//! with an arbitrary covariate design operator. The deviations are B-splines with
//! value and first derivative projected to zero at the response median, so setting
//! deviation coefficients to zero recovers an affine (location-scale) transformation
//! exactly. Monotonicity is enforced by the natural `log(h')` barrier in the
//! likelihood combined with a fraction-to-boundary line search.
//!
//! The log-likelihood per observation is the change-of-variables density for a
//! standard normal target:
//!
//!   ℓ_i = -½ h_i² + log(h'_i)
//!
//! where h_i = x_val[i,:] · β and h'_i = x_deriv[i,:] · β.

use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
};
use crate::faer_ndarray::{default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_nullspace_basis};
use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyPsiDerivativeOperator, CustomFamilyWarmStart, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, ExactOuterDerivativeOrder, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix, build_block_spatial_psi_derivatives,
    custom_family_outer_derivatives, evaluate_custom_family_joint_hyper,
    evaluate_custom_family_joint_hyper_efs, fit_custom_family,
};
use crate::families::gamlss::{
    initializewiggle_knots_from_seed, solve_penalizedweighted_projection,
};
use crate::matrix::{
    DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator, SymmetricMatrix,
};
use crate::pirls::LinearInequalityConstraints;
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use crate::solver::estimate::UnifiedFitResult;
use ndarray::{Array1, Array2, ArrayView2, s};
use std::cell::RefCell;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the response-direction basis in the transformation model.
#[derive(Clone, Debug)]
pub struct TransformationNormalConfig {
    /// B-spline degree for the response-direction deviation basis (default 3).
    pub response_degree: usize,
    /// Number of interior knots for the response-direction deviation basis (default 10).
    pub response_num_internal_knots: usize,
    /// Difference penalty order for the response-direction roughness penalty (default 2).
    pub response_penalty_order: usize,
    /// Additional penalty orders for the response-direction (default [1]).
    pub response_extra_penalty_orders: Vec<usize>,
    /// Whether to add a global identity (ridge) penalty (default true).
    pub double_penalty: bool,
}

impl Default for TransformationNormalConfig {
    fn default() -> Self {
        Self {
            response_degree: 3,
            response_num_internal_knots: 10,
            response_penalty_order: 2,
            response_extra_penalty_orders: vec![1],
            double_penalty: true,
        }
    }
}

/// Hard cap for the tensor-product width used by the transformation-normal
/// response basis. The fit repeatedly factorizes dense penalized Hessians, so
/// letting the response basis stay at its default size when the covariate side
/// is already wide creates cubic blowups on small datasets.
const MAX_TRANSFORMATION_TENSOR_WIDTH: usize = 160;
const STANDARD_NORMAL_LOG_ABS_MEAN: f64 = -0.635_181_422_730_739_1;

/// Optional warm-start for the transformation model: per-observation location and
/// scale values from a prior mean/SD normalizer.
#[derive(Clone, Debug)]
pub struct TransformationWarmStart {
    /// μ(x_i): conditional mean of the response at each observation's covariates.
    pub location: Array1<f64>,
    /// τ(x_i): conditional standard deviation at each observation's covariates.
    pub scale: Array1<f64>,
}

// ---------------------------------------------------------------------------
// The family
// ---------------------------------------------------------------------------

/// Conditional transformation model mapping Y|x to N(0,1).
///
/// Single-block `CustomFamily`. The block design is `x_val` (tensor product of
/// response value basis × covariate design). The family internally holds `x_deriv`
/// (tensor product of response derivative basis × covariate design) for the
/// Jacobian term in the likelihood.
#[derive(Clone)]
pub struct TransformationNormalFamily {
    // --- Tensor product design matrices ---
    /// Value design operator: keeps the tensor factors separate and materializes
    /// only row chunks or explicitly requested dense diagnostics.
    x_val_kron: KroneckerDesign,
    /// Derivative design operator: keeps the tensor factors separate.
    x_deriv_kron: KroneckerDesign,

    // --- Response-direction basis (fixed, does not depend on κ) ---
    /// Response value basis: n × p_resp. Columns: [1, y, dev_1(y), ..., dev_k(y)].
    response_val_basis: Array2<f64>,
    /// Response derivative basis: n × p_resp. Columns: [0, 1, dev'_1(y), ..., dev'_k(y)].
    response_deriv_basis: Array2<f64>,

    // --- Covariate side (rebuilt on κ change) ---
    /// Original covariate design used on the right side of the tensor product.
    covariate_design: DesignMatrix,
    /// Optional non-negative row weights folded directly into the likelihood.
    weights: Arc<Array1<f64>>,
    /// Additive offset for the transformation linear predictor.
    offset: Arc<Array1<f64>>,
    // --- Tensor penalties ---
    tensor_penalties: Vec<PenaltyMatrix>,

    // --- Initial values ---
    initial_beta: Array1<f64>,
    initial_log_lambdas: Array1<f64>,

    // --- Config ---
    block_name: String,

    // --- Response basis metadata (for reconstruction at predict time) ---
    response_knots: Array1<f64>,
    response_transform: Array2<f64>,
    response_degree: usize,
    response_median: f64,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl TransformationNormalFamily {
    /// Build a transformation model from response values and a pre-built covariate
    /// design operator with associated penalties.
    ///
    /// # Arguments
    ///
    /// * `response` - The response variable y (n observations).
    /// * `covariate_design` - Pre-built covariate-side design operator (n × p_cov).
    /// * `covariate_penalties` - Penalty matrices for the covariate basis.
    /// * `config` - Response-direction basis configuration.
    /// * `warm_start` - Optional location/scale from a prior normalizer.
    pub fn new(
        response: &Array1<f64>,
        weights: &Array1<f64>,
        offset: &Array1<f64>,
        covariate_design: DesignMatrix,
        covariate_penalties: Vec<PenaltyMatrix>,
        config: &TransformationNormalConfig,
        warm_start: Option<&TransformationWarmStart>,
    ) -> Result<Self, String> {
        let n = response.len();
        if covariate_design.nrows() != n {
            return Err(format!(
                "response length {} != covariate design rows {}",
                n,
                covariate_design.nrows()
            ));
        }
        let p_cov = covariate_design.ncols();
        if p_cov == 0 {
            return Err("covariate design has zero columns".to_string());
        }
        if weights.len() != n {
            return Err(format!(
                "response length {} != weights length {}",
                n,
                weights.len()
            ));
        }
        if offset.len() != n {
            return Err(format!(
                "response length {} != offset length {}",
                n,
                offset.len()
            ));
        }
        for (i, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() {
                return Err(format!("weights[{i}] is not finite: {weight}"));
            }
            if weight < 0.0 {
                return Err(format!("weights[{i}] must be non-negative: {weight}"));
            }
        }
        for (i, &value) in offset.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("offset[{i}] is not finite: {value}"));
            }
        }
        for (i, sp) in covariate_penalties.iter().enumerate() {
            let (r, c) = sp.shape();
            if r != p_cov || c != p_cov {
                return Err(format!(
                    "covariate penalty {} has shape ({r}, {c}), expected ({p_cov}, {p_cov})",
                    i,
                ));
            }
        }

        // ----- 1. Build response-direction basis -----
        let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
            build_response_basis(response, config)?;
        let p_resp = resp_val.ncols();

        // ----- 2. Row-wise Kronecker product (operator form) -----
        let x_val_kron = KroneckerDesign::new(&resp_val, covariate_design.clone())?;
        let x_deriv_kron = KroneckerDesign::new(&resp_deriv, covariate_design.clone())?;
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_kron.ncols(), p_total);

        // ----- 3. Warm start -----
        let initial_beta = compute_warm_start(
            response,
            weights,
            &covariate_design,
            &covariate_penalties,
            p_resp,
            p_cov,
            warm_start,
        )?;

        // ----- 4. Tensor penalties (Kronecker-separable) -----
        let tensor_penalties = build_tensor_penalties_kronecker(
            &resp_penalties,
            covariate_penalties,
            p_resp,
            p_cov,
            config,
        )?;

        // ----- 5. Initial log-lambdas (one per penalty, start at 0.0) -----
        let initial_log_lambdas = Array1::zeros(tensor_penalties.len());

        // Compute response median for anchoring
        let mut sorted_resp = response.to_vec();
        sorted_resp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let resp_median = if sorted_resp.len() % 2 == 1 {
            sorted_resp[sorted_resp.len() / 2]
        } else {
            0.5 * (sorted_resp[sorted_resp.len() / 2 - 1] + sorted_resp[sorted_resp.len() / 2])
        };

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            response_val_basis: resp_val,
            response_deriv_basis: resp_deriv,
            covariate_design,
            weights: Arc::new(weights.clone()),
            offset: Arc::new(offset.clone()),
            tensor_penalties,
            initial_beta,
            initial_log_lambdas,
            block_name: "transformation".to_string(),
            response_knots: resp_knots,
            response_transform: resp_transform,
            response_degree: config.response_degree,
            response_median: resp_median,
        })
    }

    /// Build from a prebuilt response basis, skipping response basis construction.
    ///
    /// For the outer loop where the response basis is precomputed once and reused
    /// across κ iterations.
    pub fn from_prebuilt_response_basis(
        response_val_basis: Array2<f64>,
        response_deriv_basis: Array2<f64>,
        response_penalties: Vec<Array2<f64>>,
        response_knots: Array1<f64>,
        response_degree: usize,
        response_transform: Array2<f64>,
        weights: &Array1<f64>,
        offset: &Array1<f64>,
        covariate_design: DesignMatrix,
        covariate_penalties: Vec<PenaltyMatrix>,
        config: &TransformationNormalConfig,
        warm_start: Option<&TransformationWarmStart>,
    ) -> Result<Self, String> {
        let n = response_val_basis.nrows();
        if covariate_design.nrows() != n {
            return Err(format!(
                "response basis rows {} != covariate design rows {}",
                n,
                covariate_design.nrows()
            ));
        }
        let p_cov = covariate_design.ncols();
        if p_cov == 0 {
            return Err("covariate design has zero columns".to_string());
        }
        if weights.len() != n {
            return Err(format!(
                "response basis rows {} != weights length {}",
                n,
                weights.len()
            ));
        }
        if offset.len() != n {
            return Err(format!(
                "response basis rows {} != offset length {}",
                n,
                offset.len()
            ));
        }
        for (i, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() {
                return Err(format!("weights[{i}] is not finite: {weight}"));
            }
            if weight < 0.0 {
                return Err(format!("weights[{i}] must be non-negative: {weight}"));
            }
        }
        for (i, &value) in offset.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("offset[{i}] is not finite: {value}"));
            }
        }
        for (i, sp) in covariate_penalties.iter().enumerate() {
            let (r, c) = sp.shape();
            if r != p_cov || c != p_cov {
                return Err(format!(
                    "covariate penalty {} has shape ({r}, {c}), expected ({p_cov}, {p_cov})",
                    i,
                ));
            }
        }

        let p_resp = response_val_basis.ncols();

        // Row-wise Kronecker product (operator form).
        let x_val_kron = KroneckerDesign::new(&response_val_basis, covariate_design.clone())?;
        let x_deriv_kron = KroneckerDesign::new(&response_deriv_basis, covariate_design.clone())?;
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_kron.ncols(), p_total);

        // Warm start: need response values for location-scale init.
        // Extract response from column 1 of response_val_basis (which stores y).
        let response_approx = response_val_basis.column(1).to_owned();
        let initial_beta = compute_warm_start(
            &response_approx,
            weights,
            &covariate_design,
            &covariate_penalties,
            p_resp,
            p_cov,
            warm_start,
        )?;

        // Tensor penalties (Kronecker-separable).
        let tensor_penalties = build_tensor_penalties_kronecker(
            &response_penalties,
            covariate_penalties,
            p_resp,
            p_cov,
            config,
        )?;

        let initial_log_lambdas = Array1::zeros(tensor_penalties.len());

        // Compute response median from column 1 (y values)
        let mut sorted_resp = response_approx.to_vec();
        sorted_resp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let resp_median = if sorted_resp.len() % 2 == 1 {
            sorted_resp[sorted_resp.len() / 2]
        } else {
            0.5 * (sorted_resp[sorted_resp.len() / 2 - 1] + sorted_resp[sorted_resp.len() / 2])
        };

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            response_val_basis,
            response_deriv_basis,
            covariate_design,
            weights: Arc::new(weights.clone()),
            offset: Arc::new(offset.clone()),
            tensor_penalties,
            initial_beta,
            initial_log_lambdas,
            block_name: "transformation".to_string(),
            response_knots: response_knots.clone(),
            response_transform: response_transform.clone(),
            response_degree,
            response_median: resp_median,
        })
    }

    /// Response basis metadata for serialization/prediction.
    pub fn response_knots(&self) -> &Array1<f64> {
        &self.response_knots
    }
    pub fn response_transform(&self) -> &Array2<f64> {
        &self.response_transform
    }
    pub fn response_degree(&self) -> usize {
        self.response_degree
    }
    pub fn response_median(&self) -> f64 {
        self.response_median
    }

    /// Return the `ParameterBlockSpec` for this family (single block).
    pub fn block_spec(&self) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: self.block_name.clone(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(self.x_val_kron.clone()))),
            offset: self.offset.as_ref().clone(),
            penalties: self.tensor_penalties.clone(),
            nullspace_dims: vec![],
            initial_log_lambdas: self.initial_log_lambdas.clone(),
            initial_beta: Some(self.initial_beta.clone()),
        }
    }

    /// Total number of coefficients.
    pub fn p_total(&self) -> usize {
        self.x_val_kron.ncols()
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.x_val_kron.nrows()
    }

    // --- Internal helpers ---

    /// Compute h and h' from the current coefficients.
    ///
    /// Uses the Kronecker-aware operators directly, avoiding full
    /// n × p_total matrix-vector products through a materialized tensor product.
    fn compute_h_and_h_prime(&self, beta: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h = self.x_val_kron.forward_mul(beta) + self.offset.as_ref();
        let h_prime = self.x_deriv_kron.forward_mul(beta);
        (h, h_prime)
    }
}

fn weighted_crossprod_dense(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
) -> Array2<f64> {
    debug_assert_eq!(left.nrows(), weights.len());
    debug_assert_eq!(right.nrows(), weights.len());
    let mut weighted_right = right.clone();
    for i in 0..weighted_right.nrows() {
        let wi = weights[i];
        for j in 0..weighted_right.ncols() {
            weighted_right[[i, j]] *= wi;
        }
    }
    fast_atb(left, &weighted_right)
}

// ---------------------------------------------------------------------------
// CustomFamily implementation
// ---------------------------------------------------------------------------

impl CustomFamily for TransformationNormalFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 1 {
            return Err(format!(
                "TransformationNormalFamily expects 1 block, got {}",
                block_states.len()
            ));
        }
        let beta = &block_states[0].beta;
        let (h, h_prime) = self.compute_h_and_h_prime(beta);
        let n = h.len();

        // Check monotonicity (soft: warn but don't fail; constraints should enforce).
        let min_h_prime = h_prime.iter().copied().fold(f64::INFINITY, f64::min);
        if min_h_prime <= 0.0 {
            return Err(format!(
                "TransformationNormalFamily: h' has non-positive values (min = {min_h_prime:.6e}). \
                 Monotonicity constraint may be violated."
            ));
        }

        // Log-likelihood: Σ [-½ h² + log(h')]
        let mut log_likelihood = 0.0;
        for i in 0..n {
            log_likelihood += self.weights[i] * (-0.5 * h[i] * h[i] + h_prime[i].ln());
        }

        // Gradient of log-likelihood: ∇ℓ = -X_val^T h + X_deriv^T (1/h')
        let inv_h_prime = h_prime.mapv(|v| 1.0 / v);
        let weighted_h = &h * self.weights.as_ref();
        let weighted_inv_h_prime = &inv_h_prime * self.weights.as_ref();
        // gradient = -X_val^T h + X_deriv^T inv_h_prime
        let grad = {
            let neg_xvt_h = self.x_val_kron.transpose_mul(&weighted_h).mapv(|v| -v);
            let xdt_inv = self.x_deriv_kron.transpose_mul(&weighted_inv_h_prime);
            neg_xvt_h + &xdt_inv
        };

        // Hessian of negative log-likelihood: -∇²ℓ = X_val^T X_val + X_deriv^T diag(1/h'²) X_deriv
        let inv_h_prime_sq = h_prime.mapv(|v| 1.0 / (v * v));
        let weighted_inv_h_prime_sq = &inv_h_prime_sq * self.weights.as_ref();
        let hessian = {
            let xtx_val = self.x_val_kron.weighted_gram(self.weights.as_ref());
            // X_deriv^T diag(w) X_deriv where w = 1/h'^2
            let xtx_deriv = self.x_deriv_kron.weighted_gram(&weighted_inv_h_prime_sq);
            xtx_val + &xtx_deriv
        };

        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: grad,
                hessian: SymmetricMatrix::Dense(hessian),
            }],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 1 {
            return Err("expected 1 block".to_string());
        }
        let (h, h_prime) = self.compute_h_and_h_prime(&block_states[0].beta);
        let mut ll = 0.0;
        for i in 0..h.len() {
            if h_prime[i] <= 0.0 {
                // The barrier line search should prevent this, but if reached
                // return -inf so the backtracking loop rejects the step.
                return Ok(f64::NEG_INFINITY);
            }
            ll += self.weights[i] * (-0.5 * h[i] * h[i] + h_prime[i].ln());
        }
        Ok(ll)
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        // The Hessian depends on β through 1/h'² where h' = X_deriv · β.
        true
    }

    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        crate::custom_family::cost_gated_outer_order(specs)
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if block_index != 0 {
            return Ok(None);
        }
        let beta = &block_states[0].beta;
        let h_prime = self.x_deriv_kron.forward_mul(beta);
        let d_h_prime = self.x_deriv_kron.forward_mul(delta);

        // Fraction-to-boundary rule: find the largest alpha in (0, 1] such that
        // h'(beta + alpha * delta) > 0 at every observation.
        //
        // For each i where d_h'[i] < 0, the step that drives h'[i] to zero is
        // alpha_i = h'[i] / (-d_h'[i]).  We take the minimum and apply a 0.995
        // safety factor (standard in interior-point methods).
        let mut alpha_max = 1.0_f64;
        for i in 0..h_prime.len() {
            let dh = d_h_prime[i];
            if dh < -1e-14 {
                let hit = h_prime[i] / (-dh);
                if hit < alpha_max {
                    alpha_max = hit;
                }
            }
        }
        let tau = 0.995;
        let alpha_safe = tau * alpha_max;
        if alpha_safe < 1.0 {
            Ok(Some(alpha_safe))
        } else {
            Ok(None)
        }
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        // Monotonicity is enforced by the natural log(h') barrier in the
        // likelihood combined with the fraction-to-boundary line search
        // (max_feasible_step_size).  No explicit constraints needed.
        Ok(None)
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_index != 0 {
            return Ok(None);
        }
        let beta = &block_states[0].beta;
        let h_prime = self.x_deriv_kron.forward_mul(beta);
        let d_h_prime = self.x_deriv_kron.forward_mul(d_beta);

        // ∂H/∂β · d_beta = -2 X_deriv^T diag((X_deriv · d_beta) / h'^3) X_deriv
        let n = h_prime.len();
        let mut weight = Array1::zeros(n);
        for i in 0..n {
            weight[i] =
                -2.0 * self.weights[i] * d_h_prime[i] / (h_prime[i] * h_prime[i] * h_prime[i]);
        }
        let dd = self.x_deriv_kron.weighted_gram(&weight);
        Ok(Some(dd))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Single block: joint Hessian = block Hessian.
        let beta = &block_states[0].beta;
        let h_prime = self.x_deriv_kron.forward_mul(beta);
        let inv_h_prime_sq = h_prime.mapv(|v| 1.0 / (v * v));
        let weighted_inv_h_prime_sq = &inv_h_prime_sq * self.weights.as_ref();
        let xtx_val = self.x_val_kron.weighted_gram(self.weights.as_ref());
        let xtx_deriv = self.x_deriv_kron.weighted_gram(&weighted_inv_h_prime_sq);
        Ok(Some(xtx_val + &xtx_deriv))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_hessian_directional_derivative(block_states, 0, d_beta_flat)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = &block_states[0].beta;
        let h_prime = self.x_deriv_kron.forward_mul(beta);
        let d_h_prime_u = self.x_deriv_kron.forward_mul(d_beta_u_flat);
        let d_h_prime_v = self.x_deriv_kron.forward_mul(d_beta_v_flat);

        // H(β) = X_valᵀ W X_val + X_derivᵀ diag(w / h'²) X_deriv
        // so D²H[u,v] = 6 X_derivᵀ diag(w (X_deriv u)(X_deriv v) / h'⁴) X_deriv.
        let n = h_prime.len();
        let mut weight = Array1::zeros(n);
        for i in 0..n {
            let hp = h_prime[i];
            weight[i] =
                6.0 * self.weights[i] * d_h_prime_u[i] * d_h_prime_v[i] / (hp * hp * hp * hp);
        }
        Ok(Some(self.x_deriv_kron.weighted_gram(&weight)))
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        psi_derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if psi_derivs.is_empty() || psi_index >= psi_derivs[0].len() {
            return Ok(None);
        }
        let deriv = &psi_derivs[0][psi_index];
        let beta = &block_states[0].beta;
        let (h, h_prime) = self.compute_h_and_h_prime(beta);
        let n = h.len();
        let inv_h_prime: Array1<f64> = h_prime.mapv(|v| 1.0 / v);
        let inv_h_prime_sq: Array1<f64> = h_prime.mapv(|v| 1.0 / (v * v));
        let inv_h_prime_cu: Array1<f64> = h_prime.mapv(|v| 1.0 / (v * v * v));
        let weighted_h = &h * self.weights.as_ref();
        let weighted_inv_h_prime = &inv_h_prime * self.weights.as_ref();
        let weighted_inv_h_prime_sq = &inv_h_prime_sq * self.weights.as_ref();

        let op = deriv
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis = deriv.implicit_axis;
        let v_val = op
            .forward_mul(axis, &beta.view())
            .map_err(|e| format!("tensor psi forward_mul failed: {e}"))?;
        let v_deriv = op
            .forward_mul_deriv(axis, beta)
            .map_err(|e| format!("tensor psi derivative forward_mul failed: {e}"))?;

        let mut obj_psi = 0.0;
        for i in 0..n {
            obj_psi += self.weights[i] * (h[i] * v_val[i] - inv_h_prime[i] * v_deriv[i]);
        }

        let score_psi = {
            let term1 = op
                .transpose_mul(axis, &weighted_h.view())
                .map_err(|e| format!("tensor psi transpose_mul failed: {e}"))?;
            let weighted_v_val = &v_val * self.weights.as_ref();
            let term2 = self.x_val_kron.transpose_mul(&weighted_v_val);
            let term3 = op
                .transpose_mul_deriv(axis, &weighted_inv_h_prime)
                .map_err(|e| format!("tensor psi derivative transpose_mul failed: {e}"))?
                .mapv(|v| -v);
            let w_deriv = &v_deriv * &weighted_inv_h_prime_sq;
            let term4 = self.x_deriv_kron.transpose_mul(&w_deriv);
            term1 + &term2 + &term3 + &term4
        };

        let hessian_psi = {
            let xvt_xvp = op
                .weighted_cross_with_cov_first(
                    &self.response_val_basis,
                    &self.response_val_basis,
                    axis,
                    self.weights.as_ref(),
                )
                .map_err(|e| format!("tensor psi weighted_cross(value) failed: {e}"))?;
            let sym_val = &xvt_xvp + &xvt_xvp.t();

            let xdt_xdp = op
                .weighted_cross_with_cov_first(
                    &self.response_deriv_basis,
                    &self.response_deriv_basis,
                    axis,
                    &weighted_inv_h_prime_sq,
                )
                .map_err(|e| format!("tensor psi weighted_cross(derivative) failed: {e}"))?;
            let sym_deriv = &xdt_xdp + &xdt_xdp.t();

            let w_cubic = ((&v_deriv * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
            let cubic_term = self.x_deriv_kron.weighted_gram(&w_cubic);

            sym_val + &sym_deriv + &cubic_term
        };

        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi: obj_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        psi_derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if psi_derivs.is_empty() || psi_i >= psi_derivs[0].len() || psi_j >= psi_derivs[0].len() {
            return Ok(None);
        }
        let deriv_i = &psi_derivs[0][psi_i];
        let deriv_j = &psi_derivs[0][psi_j];
        let beta = &block_states[0].beta;
        let (h, h_prime) = self.compute_h_and_h_prime(beta);
        let n = h.len();
        let inv_h_prime = h_prime.mapv(|v| 1.0 / v);
        let inv_h_prime_sq = h_prime.mapv(|v| 1.0 / (v * v));
        let inv_h_prime_cu = h_prime.mapv(|v| 1.0 / (v * v * v));
        let inv_h_prime_qu = h_prime.mapv(|v| 1.0 / (v * v * v * v));
        let weighted_h = &h * self.weights.as_ref();
        let weighted_inv_h_prime = &inv_h_prime * self.weights.as_ref();
        let weighted_inv_h_prime_sq = &inv_h_prime_sq * self.weights.as_ref();

        let op = deriv_i
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis_i = deriv_i.implicit_axis;
        let axis_j = deriv_j.implicit_axis;

        let v_i_val = op
            .forward_mul(axis_i, &beta.view())
            .map_err(|e| format!("tensor psi second-order forward_mul(i) failed: {e}"))?;
        let v_j_val = op
            .forward_mul(axis_j, &beta.view())
            .map_err(|e| format!("tensor psi second-order forward_mul(j) failed: {e}"))?;
        let v_ij_val = if axis_i == axis_j {
            op.forward_mul_second_diag(axis_i, &beta.view())
        } else {
            op.forward_mul_second_cross(axis_i, axis_j, &beta.view())
        }
        .map_err(|e| format!("tensor psi second-order forward_mul(value second) failed: {e}"))?;

        let v_i_deriv = op
            .forward_mul_deriv(axis_i, beta)
            .map_err(|e| format!("tensor psi second-order forward_mul_deriv(i) failed: {e}"))?;
        let v_j_deriv = op
            .forward_mul_deriv(axis_j, beta)
            .map_err(|e| format!("tensor psi second-order forward_mul_deriv(j) failed: {e}"))?;
        let v_ij_deriv = op
            .forward_mul_second_deriv(axis_i, axis_j, &beta.view())
            .map_err(|e| {
                format!("tensor psi second-order forward_mul(derivative second) failed: {e}")
            })?;

        let mut objective_psi_psi = 0.0;
        for row in 0..n {
            objective_psi_psi += self.weights[row]
                * (v_i_val[row] * v_j_val[row] + h[row] * v_ij_val[row]
                    - inv_h_prime[row] * v_ij_deriv[row]
                    + inv_h_prime_sq[row] * v_i_deriv[row] * v_j_deriv[row]);
        }

        let score_psi_psi = {
            let term1 = op
                .transpose_mul(axis_i, &(&v_j_val * self.weights.as_ref()).view())
                .map_err(|e| format!("tensor psi second-order transpose_mul(i) failed: {e}"))?;
            let term2 = op
                .transpose_mul(axis_j, &(&v_i_val * self.weights.as_ref()).view())
                .map_err(|e| format!("tensor psi second-order transpose_mul(j) failed: {e}"))?;
            let term3 = if axis_i == axis_j {
                op.transpose_mul_second_diag(axis_i, &weighted_h.view())
            } else {
                op.transpose_mul_second_cross(axis_i, axis_j, &weighted_h.view())
            }
            .map_err(|e| {
                format!("tensor psi second-order transpose_mul(value second) failed: {e}")
            })?;
            let term4 = self
                .x_val_kron
                .transpose_mul(&(&v_ij_val * self.weights.as_ref()));
            let term5 = op
                .transpose_mul_deriv(axis_i, &(&v_j_deriv * &weighted_inv_h_prime_sq))
                .map_err(|e| {
                    format!("tensor psi second-order transpose_mul_deriv(i) failed: {e}")
                })?;
            let term6 = op
                .transpose_mul_deriv(axis_j, &(&v_i_deriv * &weighted_inv_h_prime_sq))
                .map_err(|e| {
                    format!("tensor psi second-order transpose_mul_deriv(j) failed: {e}")
                })?;
            let term7 = op
                .transpose_mul_second_deriv(axis_i, axis_j, &weighted_inv_h_prime.view())
                .map_err(|e| {
                    format!("tensor psi second-order transpose_mul(derivative second) failed: {e}")
                })?
                .mapv(|v| -v);
            let term8 = self
                .x_deriv_kron
                .transpose_mul(&(&v_ij_deriv * &weighted_inv_h_prime_sq));
            let cubic = ((&v_i_deriv * &v_j_deriv) * &inv_h_prime_cu * self.weights.as_ref())
                .mapv(|v| -2.0 * v);
            let term9 = self.x_deriv_kron.transpose_mul(&cubic);
            term1 + &term2 + &term3 + &term4 + &term5 + &term6 + &term7 + &term8 + &term9
        };

        let hessian_psi_psi = {
            let x_val = self.x_val_kron.to_dense();
            let x_deriv = self.x_deriv_kron.to_dense();
            let x_i_val = op
                .materialize_first(axis_i)
                .map_err(|e| format!("tensor psi second-order materialize_first(i) failed: {e}"))?;
            let x_j_val = op
                .materialize_first(axis_j)
                .map_err(|e| format!("tensor psi second-order materialize_first(j) failed: {e}"))?;
            let x_ij_val = if axis_i == axis_j {
                op.materialize_second_diag(axis_i)
            } else {
                op.materialize_second_cross(axis_i, axis_j)
            }
            .map_err(|e| {
                format!("tensor psi second-order materialize_second(value) failed: {e}")
            })?;
            let x_i_deriv = op.materialize_first_deriv(axis_i).map_err(|e| {
                format!("tensor psi second-order materialize_first_deriv(i) failed: {e}")
            })?;
            let x_j_deriv = op.materialize_first_deriv(axis_j).map_err(|e| {
                format!("tensor psi second-order materialize_first_deriv(j) failed: {e}")
            })?;
            let x_ij_deriv = op.materialize_second_deriv(axis_i, axis_j).map_err(|e| {
                format!("tensor psi second-order materialize_second_deriv failed: {e}")
            })?;

            let mut hess = weighted_crossprod_dense(&x_i_val, self.weights.as_ref(), &x_j_val);
            hess += &weighted_crossprod_dense(&x_j_val, self.weights.as_ref(), &x_i_val);
            hess += &weighted_crossprod_dense(&x_ij_val, self.weights.as_ref(), &x_val);
            hess += &weighted_crossprod_dense(&x_val, self.weights.as_ref(), &x_ij_val);
            hess += &weighted_crossprod_dense(&x_i_deriv, &weighted_inv_h_prime_sq, &x_j_deriv);
            hess += &weighted_crossprod_dense(&x_j_deriv, &weighted_inv_h_prime_sq, &x_i_deriv);
            hess += &weighted_crossprod_dense(&x_ij_deriv, &weighted_inv_h_prime_sq, &x_deriv);
            hess += &weighted_crossprod_dense(&x_deriv, &weighted_inv_h_prime_sq, &x_ij_deriv);

            let cubic_i =
                ((&v_j_deriv * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
            hess += &weighted_crossprod_dense(&x_i_deriv, &cubic_i, &x_deriv);
            hess += &weighted_crossprod_dense(&x_deriv, &cubic_i, &x_i_deriv);

            let cubic_j =
                ((&v_i_deriv * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
            hess += &weighted_crossprod_dense(&x_j_deriv, &cubic_j, &x_deriv);
            hess += &weighted_crossprod_dense(&x_deriv, &cubic_j, &x_j_deriv);

            let cubic_second =
                ((&v_ij_deriv * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
            hess += &self.x_deriv_kron.weighted_gram(&cubic_second);

            let quartic = ((&v_i_deriv * &v_j_deriv) * &inv_h_prime_qu * self.weights.as_ref())
                .mapv(|v| 6.0 * v);
            hess += &self.x_deriv_kron.weighted_gram(&quartic);
            0.5 * (&hess + &hess.t())
        };

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        psi_derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if psi_derivs.is_empty() || psi_index >= psi_derivs[0].len() {
            return Ok(None);
        }
        let deriv = &psi_derivs[0][psi_index];
        let beta = &block_states[0].beta;
        let h_prime = self.x_deriv_kron.forward_mul(beta);
        let d_h_prime = self.x_deriv_kron.forward_mul(d_beta_flat);
        let inv_h_prime_cu = h_prime.mapv(|v| 1.0 / (v * v * v));
        let inv_h_prime_qu = h_prime.mapv(|v| 1.0 / (v * v * v * v));

        let op = deriv
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis = deriv.implicit_axis;

        let v_deriv = op
            .forward_mul_deriv(axis, beta)
            .map_err(|e| format!("tensor psi hessian drift forward_mul_deriv failed: {e}"))?;
        let d_v_deriv = op.forward_mul_deriv(axis, d_beta_flat).map_err(|e| {
            format!("tensor psi hessian drift directional forward_mul_deriv failed: {e}")
        })?;

        let x_deriv = self.x_deriv_kron.to_dense();
        let x_psi_deriv = op
            .materialize_first_deriv(axis)
            .map_err(|e| format!("tensor psi hessian drift materialize_first_deriv failed: {e}"))?;

        let cubic_h = ((&d_h_prime * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
        let mut hess = weighted_crossprod_dense(&x_psi_deriv, &cubic_h, &x_deriv)
            + &weighted_crossprod_dense(&x_deriv, &cubic_h, &x_psi_deriv);

        let cubic_v = ((&d_v_deriv * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
        hess += &self.x_deriv_kron.weighted_gram(&cubic_v);

        let quartic =
            ((&v_deriv * &d_h_prime) * &inv_h_prime_qu * self.weights.as_ref()).mapv(|v| 6.0 * v);
        hess += &self.x_deriv_kron.weighted_gram(&quartic);

        Ok(Some(0.5 * (&hess + &hess.t())))
    }
}

// ---------------------------------------------------------------------------
// Response-direction basis construction
// ---------------------------------------------------------------------------

/// Build the response-direction basis: `[1, y, anchored deviations]`.
///
/// Returns (value_basis, derivative_basis, penalties, knots, transform).
fn build_response_basis(
    response: &Array1<f64>,
    config: &TransformationNormalConfig,
) -> Result<
    (
        Array2<f64>,
        Array2<f64>,
        Vec<Array2<f64>>,
        Array1<f64>,
        Array2<f64>,
    ),
    String,
> {
    let n = response.len();
    if n < 4 {
        return Err(format!("need at least 4 observations, got {n}"));
    }
    for (i, &v) in response.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("response[{i}] is not finite: {v}"));
        }
    }

    // --- Build B-spline knots for the deviation part ---
    let knots = initializewiggle_knots_from_seed(
        response.view(),
        config.response_degree,
        config.response_num_internal_knots,
    )?;

    // --- Deviation transform: project out value and first derivative at the median ---
    let transform = response_deviation_transform(&knots, config.response_degree, response)?;
    let raw_dim = transform.nrows();
    let dev_dim = transform.ncols();

    // --- Evaluate full B-spline basis at response values ---
    let (raw_val_basis, _) = create_basis::<Dense>(
        response.view(),
        KnotSource::Provided(knots.view()),
        config.response_degree,
        BasisOptions::value(),
    )
    .map_err(|e| e.to_string())?;
    let raw_val = raw_val_basis.as_ref().clone();

    let (raw_deriv_basis, _) = create_basis::<Dense>(
        response.view(),
        KnotSource::Provided(knots.view()),
        config.response_degree,
        BasisOptions::first_derivative(),
    )
    .map_err(|e| e.to_string())?;
    let raw_deriv = raw_deriv_basis.as_ref().clone();

    // --- Apply deviation transform: dev = raw · Z ---
    let dev_val = raw_val.dot(&transform); // n × dev_dim
    let dev_deriv = raw_deriv.dot(&transform); // n × dev_dim

    // --- Assemble full response basis: [1, y, dev] ---
    let p_resp = 2 + dev_dim;
    let mut resp_val = Array2::<f64>::zeros((n, p_resp));
    let mut resp_deriv = Array2::<f64>::zeros((n, p_resp));

    // Column 0: intercept (value = 1, derivative = 0)
    resp_val.column_mut(0).fill(1.0);
    // resp_deriv column 0 stays 0

    // Column 1: y (value = y, derivative = 1)
    resp_val.column_mut(1).assign(&response.view());
    resp_deriv.column_mut(1).fill(1.0);

    // Columns 2..: deviations
    resp_val.slice_mut(s![.., 2..]).assign(&dev_val);
    resp_deriv.slice_mut(s![.., 2..]).assign(&dev_deriv);

    // --- Response-direction penalties ---
    // Penalty acts on the deviation part only; affine columns [1, y] are in the nullspace.
    let mut resp_penalties = Vec::new();

    let add_penalty = |order: usize, penalties: &mut Vec<Array2<f64>>| -> Result<(), String> {
        if order == 0 || order >= raw_dim {
            return Ok(());
        }
        let raw_pen =
            create_difference_penalty_matrix(raw_dim, order, None).map_err(|e| e.to_string())?;
        let dev_pen = fast_ab(&fast_atb(&transform, &raw_pen), &transform); // dev_dim × dev_dim
        // Embed in full response basis: zeros for [1, y], dev_pen for deviation part.
        let mut full_pen = Array2::<f64>::zeros((p_resp, p_resp));
        full_pen.slice_mut(s![2.., 2..]).assign(&dev_pen);
        penalties.push(full_pen);
        Ok(())
    };

    add_penalty(config.response_penalty_order, &mut resp_penalties)?;
    for &order in &config.response_extra_penalty_orders {
        if order == config.response_penalty_order {
            continue;
        }
        add_penalty(order, &mut resp_penalties)?;
    }

    Ok((resp_val, resp_deriv, resp_penalties, knots, transform))
}

fn effective_response_num_internal_knots(
    config: &TransformationNormalConfig,
    n_obs: usize,
    p_cov: usize,
) -> usize {
    let sample_cap = (n_obs / 10).max(1);
    let min_internal = 1usize;
    let max_resp_cols_from_tensor =
        (MAX_TRANSFORMATION_TENSOR_WIDTH / p_cov.max(1)).max(config.response_degree + 2);
    let tensor_cap = max_resp_cols_from_tensor
        .saturating_sub(config.response_degree + 1)
        .max(min_internal);
    config
        .response_num_internal_knots
        .min(sample_cap)
        .min(tensor_cap)
        .max(min_internal)
}

/// Build the nullspace projection that anchors B-spline deviations at the response
/// median (value = 0 and first derivative = 0 at the median).
fn response_deviation_transform(
    knots: &Array1<f64>,
    degree: usize,
    response: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    // Compute median.
    let mut sorted = response.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
    };
    let anchor = Array1::from_vec(vec![median]);

    // Evaluate basis value and first derivative at anchor.
    let (val_basis, _) = create_basis::<Dense>(
        anchor.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .map_err(|e| e.to_string())?;
    let (d1_basis, _) = create_basis::<Dense>(
        anchor.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .map_err(|e| e.to_string())?;

    let k = val_basis.ncols();
    let mut c = Array2::<f64>::zeros((2, k));
    c.row_mut(0).assign(&val_basis.row(0));
    c.row_mut(1).assign(&d1_basis.row(0));

    let (z, rank) = rrqr_nullspace_basis(&c.t(), default_rrqr_rank_alpha())
        .map_err(|e| format!("response deviation RRQR failed: {e}"))?;
    if rank >= k || z.ncols() == 0 {
        return Err(
            "response deviation anchor constraints removed all columns; increase basis richness"
                .to_string(),
        );
    }
    Ok(z)
}

// ---------------------------------------------------------------------------
// Tensor product construction
// ---------------------------------------------------------------------------

/// Row-wise Kronecker product of two matrices (same number of rows).
///
/// output\[i, j * p_b + k\] = a\[i, j\] * b\[i, k\]
fn rowwise_kronecker(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    assert_eq!(a.nrows(), b.nrows());
    let n = a.nrows();
    let pa = a.ncols();
    let pb = b.ncols();
    let mut out = Array2::<f64>::zeros((n, pa * pb));
    for i in 0..n {
        for j in 0..pa {
            let a_ij = a[[i, j]];
            if a_ij == 0.0 {
                continue;
            }
            for k in 0..pb {
                out[[i, j * pb + k]] = a_ij * b[[i, k]];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Kronecker-aware operator for biobank-scale tensor products
// ---------------------------------------------------------------------------

/// A lazy representation of the row-wise Kronecker product `A ⊙ B`
/// (face-splitting product / Khatri–Rao row-wise product).
///
/// The response basis is kept as a dense left factor and the covariate side is
/// kept in its native `DesignMatrix` representation. Products are evaluated in
/// factored form:
///
///   forward_mul(β):  reshape β → (p_a, p_b), then result[i] = Σ_j A[i,j] * (B[i,:] · β[j,:])
///   transpose_mul(v): result[j, k] = Σ_i v[i] * A[i,j] * B[i,k]
///
/// Storage: O(n·p_a + storage(B)) vs O(n·p_a·p_b) for the materialized form.
#[derive(Clone)]
enum KroneckerDesign {
    Factored {
        left: Array2<f64>,   // n × p_a
        right: DesignMatrix, // n × p_b
    },
}

impl KroneckerDesign {
    fn new(left: &Array2<f64>, right: DesignMatrix) -> Result<Self, String> {
        if left.nrows() != right.nrows() {
            return Err(format!(
                "KroneckerDesign row mismatch: left={}, right={}",
                left.nrows(),
                right.nrows()
            ));
        }
        Ok(KroneckerDesign::Factored {
            left: left.clone(),
            right,
        })
    }

    fn nrows(&self) -> usize {
        match self {
            KroneckerDesign::Factored { left, .. } => left.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            KroneckerDesign::Factored { left, right } => left.ncols() * right.ncols(),
        }
    }

    /// Compute `self · beta` where beta has length p_a * p_b.
    /// Returns an n-vector.
    fn forward_mul(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::Factored { left, right } => {
                let pa = left.ncols();
                let pb = right.ncols();
                let n = left.nrows();
                debug_assert_eq!(beta.len(), pa * pb);
                let beta_mat = beta.view().into_shape_with_order((pa, pb)).unwrap();
                let mut result = Array1::zeros(n);
                if let Some(right_dense) = right.as_dense_ref() {
                    let right_beta = fast_ab(right_dense, &beta_mat.t().to_owned());
                    for i in 0..n {
                        let mut acc = 0.0;
                        for j in 0..pa {
                            acc += left[[i, j]] * right_beta[[i, j]];
                        }
                        result[i] = acc;
                    }
                    return result;
                }
                for j in 0..pa {
                    let cov_part = right.apply(&beta_mat.row(j).to_owned());
                    for i in 0..n {
                        result[i] += left[[i, j]] * cov_part[i];
                    }
                }
                result
            }
        }
    }

    /// Compute `self^T · v` where v is an n-vector.
    /// Returns a (p_a * p_b)-vector.
    fn transpose_mul(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::Factored { left, right } => {
                let n = left.nrows();
                let pa = left.ncols();
                let pb = right.ncols();
                debug_assert_eq!(v.len(), n);
                let mut out = Array1::<f64>::zeros(pa * pb);
                for j in 0..pa {
                    let mut weighted_v = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        weighted_v[i] = v[i] * left[[i, j]];
                    }
                    let cov_block = right.apply_transpose(&weighted_v);
                    out.slice_mut(s![j * pb..(j + 1) * pb]).assign(&cov_block);
                }
                out
            }
        }
    }

    /// Compute `self^T · diag(w) · self` (weighted Gram).
    fn weighted_gram(&self, w: &Array1<f64>) -> Array2<f64> {
        match self {
            KroneckerDesign::Factored { .. } => {
                let dense = self.to_dense();
                let wm = weight_rows(&dense, w);
                fast_atb(&wm, &dense)
            }
        }
    }
}

impl LinearOperator for KroneckerDesign {
    fn nrows(&self) -> usize {
        KroneckerDesign::nrows(self)
    }

    fn ncols(&self) -> usize {
        KroneckerDesign::ncols(self)
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.forward_mul(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.transpose_mul(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "KroneckerDesign::diag_xtw_x dimension mismatch: weights={}, nrows={}",
                weights.len(),
                self.nrows()
            ));
        }
        Ok(self.weighted_gram(weights))
    }
}

// KroneckerDesign contains owned data + DesignMatrix (which is Send+Sync),
// so it is safe to send/share across threads.
unsafe impl Send for KroneckerDesign {}
unsafe impl Sync for KroneckerDesign {}

impl DenseDesignOperator for KroneckerDesign {
    fn row_chunk(&self, rows: std::ops::Range<usize>) -> Array2<f64> {
        match self {
            KroneckerDesign::Factored { left, right } => {
                let left_chunk = left.slice(s![rows.clone(), ..]).to_owned();
                let right_chunk = right.row_chunk(rows);
                rowwise_kronecker(&left_chunk, &right_chunk)
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        match self {
            KroneckerDesign::Factored { left, right } => {
                rowwise_kronecker(left, right.as_dense_cow().as_ref())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kronecker-form penalties
// ---------------------------------------------------------------------------

/// A penalty matrix in separable Kronecker form: `S_left ⊗ S_right`.
///
/// Build tensor product penalties in Kronecker-separable form.
fn build_tensor_penalties_kronecker(
    response_penalties: &[Array2<f64>],
    covariate_penalties: Vec<PenaltyMatrix>,
    p_resp: usize,
    p_cov: usize,
    config: &TransformationNormalConfig,
) -> Result<Vec<PenaltyMatrix>, String> {
    let eye_resp = Array2::<f64>::eye(p_resp);
    let eye_cov = Array2::<f64>::eye(p_cov);
    let mut penalties = Vec::new();

    // Covariate penalties: I_resp ⊗ S_cov_m
    for s_cov in covariate_penalties {
        match s_cov {
            PenaltyMatrix::Dense(right) => penalties.push(PenaltyMatrix::KroneckerFactored {
                left: eye_resp.clone(),
                right,
            }),
            penalty @ PenaltyMatrix::Blockwise { .. } => {
                penalties.push(PenaltyMatrix::KroneckerFactored {
                    left: eye_resp.clone(),
                    right: penalty.to_dense(),
                })
            }
            PenaltyMatrix::KroneckerFactored { .. } => {
                return Err(
                    "transformation covariate penalties must be single-block, not already Kronecker-factored"
                        .to_string(),
                )
            }
        }
    }

    // Response penalties: S_resp_m ⊗ I_cov
    for s_resp in response_penalties {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: s_resp.clone(),
            right: eye_cov.clone(),
        });
    }

    // Double penalty: global ridge (I_resp ⊗ I_cov = I_total)
    if config.double_penalty {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: eye_resp,
            right: eye_cov,
        });
    }

    Ok(penalties)
}

// ---------------------------------------------------------------------------
// Warm start
// ---------------------------------------------------------------------------

/// Compute initial β so that h(y|x) ≈ (y - μ(x)) / τ(x).
///
/// If no warm start is provided, estimate a penalized conditional location-scale
/// surrogate on the covariate design and project that affine normalizer back
/// into the transformation basis.
fn compute_warm_start(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    covariate_design: &DesignMatrix,
    covariate_penalties: &[PenaltyMatrix],
    p_resp: usize,
    p_cov: usize,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<Array1<f64>, String> {
    let n = response.len();
    let p_total = p_resp * p_cov;
    let mut beta = Array1::zeros(p_total);
    if p_resp < 2 {
        return Err(format!(
            "transformation warm start requires at least 2 response basis rows, got {p_resp}"
        ));
    }

    // Target: for the intercept row (j=0), Θ[0,:] · cov[i,:] = -μ(x_i)/τ(x_i)
    //         for the linear row (j=1),  Θ[1,:] · cov[i,:] = 1/τ(x_i)
    //         for deviation rows (j≥2),  Θ[j,:] = 0

    let default_ws;
    let ws = match warm_start {
        Some(ws) => ws,
        None => {
            default_ws = estimate_default_warm_start(
                response,
                weights,
                covariate_design,
                covariate_penalties,
            )?;
            &default_ws
        }
    };
    if ws.location.len() != n || ws.scale.len() != n {
        return Err("warm start location/scale length mismatch".to_string());
    }
    let mut target_intercept = Array1::zeros(n);
    let mut target_slope = Array1::zeros(n);
    for i in 0..n {
        let tau = ws.scale[i].max(1e-12);
        target_intercept[i] = -ws.location[i] / tau;
        target_slope[i] = 1.0 / tau;
    }

    let projection_log_lambdas = Array1::zeros(covariate_penalties.len());
    let zero_offset = Array1::zeros(n);
    let coeff_int = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &target_intercept,
        weights,
        covariate_penalties,
        &projection_log_lambdas,
        1e-8,
    )?;
    let coeff_slope = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &target_slope,
        weights,
        covariate_penalties,
        &projection_log_lambdas,
        1e-8,
    )?;

    beta.slice_mut(s![0..p_cov]).assign(&coeff_int);
    beta.slice_mut(s![p_cov..2 * p_cov]).assign(&coeff_slope);

    Ok(beta)
}

fn estimate_default_warm_start(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    covariate_design: &DesignMatrix,
    covariate_penalties: &[PenaltyMatrix],
) -> Result<TransformationWarmStart, String> {
    let n = response.len();
    if weights.len() != n {
        return Err(format!(
            "transformation warm start weights length mismatch: response={}, weights={}",
            n,
            weights.len()
        ));
    }
    let zero_offset = Array1::zeros(n);
    let log_lambdas = Array1::zeros(covariate_penalties.len());
    let beta_location = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        response,
        weights,
        covariate_penalties,
        &log_lambdas,
        1e-8,
    )?;
    let location = covariate_design.matrixvectormultiply(&beta_location);
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err("transformation warm start requires positive finite total weight".to_string());
    }
    let weighted_ss = response
        .iter()
        .zip(location.iter())
        .zip(weights.iter())
        .map(|((&y, &mu), &w)| {
            let resid = y - mu;
            w * resid * resid
        })
        .sum::<f64>();
    if !weighted_ss.is_finite() {
        return Err("transformation warm start residual variance is not finite".to_string());
    }
    let global_scale = (weighted_ss / weight_sum).sqrt().max(1e-6);
    let residual_floor = global_scale * 1e-3 + 1e-12;
    let log_scale_target =
        Array1::from_iter(response.iter().zip(location.iter()).map(|(&y, &mu)| {
            (y - mu).abs().max(residual_floor).ln() - STANDARD_NORMAL_LOG_ABS_MEAN
        }));
    let beta_log_scale = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &log_scale_target,
        weights,
        covariate_penalties,
        &log_lambdas,
        1e-8,
    )?;
    let scale = covariate_design
        .matrixvectormultiply(&beta_log_scale)
        .mapv(|eta| eta.exp().max(residual_floor));

    Ok(TransformationWarmStart { location, scale })
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Multiply each row of a matrix by the corresponding weight.
fn weight_rows(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    debug_assert_eq!(n, w.len());
    let mut out = Array2::zeros((n, p));
    for i in 0..n {
        let wi = w[i];
        for j in 0..p {
            out[[i, j]] = x[[i, j]] * wi;
        }
    }
    out
}

#[derive(Clone)]
struct TensorKroneckerPsiOperator {
    response_val_basis: Arc<Array2<f64>>,
    response_deriv_basis: Arc<Array2<f64>>,
    covariate_design: DesignMatrix,
    covariate_derivs: Vec<CustomFamilyBlockPsiDerivative>,
}

impl TensorKroneckerPsiOperator {
    fn n_data(&self) -> usize {
        self.response_val_basis.nrows()
    }

    fn p_resp(&self) -> usize {
        self.response_val_basis.ncols()
    }

    fn p_cov(&self) -> usize {
        self.covariate_design.ncols()
    }

    fn p_out(&self) -> usize {
        self.p_resp() * self.p_cov()
    }

    fn cov_deriv(
        &self,
        axis: usize,
    ) -> Result<&CustomFamilyBlockPsiDerivative, crate::terms::basis::BasisError> {
        self.covariate_derivs.get(axis).ok_or_else(|| {
            crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker psi axis {axis} out of bounds for {} axes",
                self.covariate_derivs.len()
            ))
        })
    }

    fn cov_forward_first(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.dot(u));
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi operator for axis {axis}"
            )));
        };
        op.forward_mul(deriv.implicit_axis, u)
    }

    fn cov_transpose_first(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.t().dot(v));
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi transpose operator for axis {axis}"
            )));
        };
        op.transpose_mul(deriv.implicit_axis, v)
    }

    fn cov_forward_second(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == self.cov_deriv(axis_e)?.implicit_group_id
        {
            if deriv_d.implicit_axis == self.cov_deriv(axis_e)?.implicit_axis {
                return op.forward_mul_second_diag(deriv_d.implicit_axis, u);
            }
            return op.forward_mul_second_cross(
                deriv_d.implicit_axis,
                self.cov_deriv(axis_e)?.implicit_axis,
                u,
            );
        }
        if let Some(rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = rows.get(axis_e)
        {
            if mat.nrows() == self.n_data() && mat.ncols() == self.p_cov() {
                return Ok(mat.dot(u));
            }
        }
        Ok(Array1::<f64>::zeros(self.n_data()))
    }

    fn cov_transpose_second(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == self.cov_deriv(axis_e)?.implicit_group_id
        {
            if deriv_d.implicit_axis == self.cov_deriv(axis_e)?.implicit_axis {
                return op.transpose_mul_second_diag(deriv_d.implicit_axis, v);
            }
            return op.transpose_mul_second_cross(
                deriv_d.implicit_axis,
                self.cov_deriv(axis_e)?.implicit_axis,
                v,
            );
        }
        if let Some(rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = rows.get(axis_e)
        {
            if mat.nrows() == self.n_data() && mat.ncols() == self.p_cov() {
                return Ok(mat.t().dot(v));
            }
        }
        Ok(Array1::<f64>::zeros(self.p_cov()))
    }

    fn materialize_cov_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.clone());
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi materialization for axis {axis}"
            )));
        };
        op.materialize_first(deriv.implicit_axis)
    }

    fn materialize_cov_second(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == self.cov_deriv(axis_e)?.implicit_group_id
        {
            if deriv_d.implicit_axis == self.cov_deriv(axis_e)?.implicit_axis {
                return op.materialize_second_diag(deriv_d.implicit_axis);
            }
            return op.materialize_second_cross(
                deriv_d.implicit_axis,
                self.cov_deriv(axis_e)?.implicit_axis,
            );
        }
        if let Some(rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = rows.get(axis_e)
        {
            if mat.nrows() == self.n_data() && mat.ncols() == self.p_cov() {
                return Ok(mat.clone());
            }
        }
        Ok(Array2::<f64>::zeros((self.n_data(), self.p_cov())))
    }

    fn lifted_forward(
        &self,
        resp_basis: &Array2<f64>,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let beta = u
            .to_owned()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|_| {
                crate::terms::basis::BasisError::InvalidInput(
                    "tensor psi coefficient reshape failed".to_string(),
                )
            })?;
        let mut out = Array1::<f64>::zeros(n);
        for j in 0..p_resp {
            let cov_part = self.cov_forward_first(axis, &beta.row(j))?;
            for i in 0..n {
                out[i] += resp_basis[[i, j]] * cov_part[i];
            }
        }
        Ok(out)
    }

    fn lifted_transpose(
        &self,
        resp_basis: &Array2<f64>,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for j in 0..p_resp {
            let mut weighted_v = Array1::<f64>::zeros(n);
            for i in 0..n {
                weighted_v[i] = resp_basis[[i, j]] * v[i];
            }
            let cov_block = self.cov_transpose_first(axis, &weighted_v.view())?;
            out.slice_mut(s![j * p_cov..(j + 1) * p_cov])
                .assign(&cov_block);
        }
        Ok(out)
    }

    fn lifted_forward_second(
        &self,
        resp_basis: &Array2<f64>,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let beta = u
            .to_owned()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|_| {
                crate::terms::basis::BasisError::InvalidInput(
                    "tensor psi second coefficient reshape failed".to_string(),
                )
            })?;
        let mut out = Array1::<f64>::zeros(n);
        for j in 0..p_resp {
            let cov_part = self.cov_forward_second(axis_d, axis_e, &beta.row(j))?;
            for i in 0..n {
                out[i] += resp_basis[[i, j]] * cov_part[i];
            }
        }
        Ok(out)
    }

    fn lifted_transpose_second(
        &self,
        resp_basis: &Array2<f64>,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for j in 0..p_resp {
            let mut weighted_v = Array1::<f64>::zeros(n);
            for i in 0..n {
                weighted_v[i] = resp_basis[[i, j]] * v[i];
            }
            let cov_block = self.cov_transpose_second(axis_d, axis_e, &weighted_v.view())?;
            out.slice_mut(s![j * p_cov..(j + 1) * p_cov])
                .assign(&cov_block);
        }
        Ok(out)
    }

    fn materialize_lifted(&self, resp_basis: &Array2<f64>, cov: &Array2<f64>) -> Array2<f64> {
        rowwise_kronecker(resp_basis, cov)
    }

    fn forward_mul_deriv(
        &self,
        axis: usize,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_forward(&self.response_deriv_basis, axis, &u.view())
    }

    fn transpose_mul_deriv(
        &self,
        axis: usize,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_transpose(&self.response_deriv_basis, axis, &v.view())
    }

    fn weighted_cross_with_cov_first(
        &self,
        left_resp_basis: &Array2<f64>,
        right_resp_basis: &Array2<f64>,
        axis: usize,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_left_resp = left_resp_basis.ncols();
        let p_right_resp = right_resp_basis.ncols();
        let p_cov = self.p_cov();
        let deriv = self.cov_deriv(axis)?;
        let cov_psi_dense =
            (deriv.x_psi.nrows() == n && deriv.x_psi.ncols() == p_cov).then_some(&deriv.x_psi);
        let mut out = Array2::<f64>::zeros((p_left_resp * p_cov, p_right_resp * p_cov));
        for a in 0..p_left_resp {
            for b in 0..p_right_resp {
                let mut pair_weights = Array1::<f64>::zeros(n);
                for i in 0..n {
                    pair_weights[i] =
                        weights[i] * left_resp_basis[[i, a]] * right_resp_basis[[i, b]];
                }
                let block = if let Some(cov_psi) = cov_psi_dense {
                    let mut block = Array2::<f64>::zeros((p_cov, p_cov));
                    for j in 0..p_cov {
                        let weighted_col = &pair_weights * &cov_psi.column(j).to_owned();
                        let col = self.covariate_design.apply_transpose(&weighted_col);
                        block.column_mut(j).assign(&col);
                    }
                    block
                } else {
                    let mut block = Array2::<f64>::zeros((p_cov, p_cov));
                    let mut e = Array1::<f64>::zeros(p_cov);
                    for j in 0..p_cov {
                        e[j] = 1.0;
                        let col = self.cov_forward_first(axis, &e.view())?;
                        e[j] = 0.0;
                        let weighted_col = &pair_weights * &col;
                        let cov_t_weighted = self.covariate_design.apply_transpose(&weighted_col);
                        block.column_mut(j).assign(&cov_t_weighted);
                    }
                    block
                };
                out.slice_mut(s![a * p_cov..(a + 1) * p_cov, b * p_cov..(b + 1) * p_cov])
                    .assign(&block);
            }
        }
        Ok(out)
    }

    fn forward_mul_second_deriv(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_forward_second(&self.response_deriv_basis, axis_d, axis_e, u)
    }

    fn transpose_mul_second_deriv(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_transpose_second(&self.response_deriv_basis, axis_d, axis_e, v)
    }

    fn materialize_first_deriv(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self.materialize_lifted(
            &self.response_deriv_basis,
            &self.materialize_cov_first(axis)?,
        ))
    }

    fn materialize_second_deriv(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self.materialize_lifted(
            &self.response_deriv_basis,
            &self.materialize_cov_second(axis_d, axis_e)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_matrix_derivativefd;
    use ndarray::array;

    fn toy_covariate_design_and_derivs(
        psi: &Array1<f64>,
    ) -> (Array2<f64>, Vec<CustomFamilyBlockPsiDerivative>) {
        let x0 = array![[1.00, 0.40], [1.10, 0.35], [1.20, 0.45], [0.95, 0.50],];
        let x_a = array![[0.10, -0.02], [0.08, 0.01], [0.12, -0.01], [0.09, 0.03],];
        let x_b = array![[-0.04, 0.06], [-0.02, 0.05], [-0.03, 0.04], [-0.01, 0.07],];
        let x_aa = array![[0.02, 0.00], [0.01, 0.01], [0.02, -0.01], [0.01, 0.02],];
        let x_ab = array![[0.01, -0.01], [0.00, 0.02], [0.01, 0.01], [0.00, -0.01],];
        let x_bb = array![[-0.01, 0.02], [-0.02, 0.01], [-0.01, 0.00], [-0.02, 0.02],];
        let design = &x0
            + &(x_a.clone() * psi[0])
            + &(x_b.clone() * psi[1])
            + &(x_aa.clone() * (0.5 * psi[0] * psi[0]))
            + &(x_ab.clone() * (psi[0] * psi[1]))
            + &(x_bb.clone() * (0.5 * psi[1] * psi[1]));
        let d_a = &x_a + &(x_aa.clone() * psi[0]) + &(x_ab.clone() * psi[1]);
        let d_b = &x_b + &(x_ab.clone() * psi[0]) + &(x_bb.clone() * psi[1]);
        let deriv_a = CustomFamilyBlockPsiDerivative::new(
            None,
            d_a,
            Array2::zeros((0, 0)),
            None,
            Some(vec![x_aa.clone(), x_ab.clone()]),
            None,
            None,
        );
        let deriv_b = CustomFamilyBlockPsiDerivative::new(
            None,
            d_b,
            Array2::zeros((0, 0)),
            None,
            Some(vec![x_ab, x_bb]),
            None,
            None,
        );
        (design, vec![deriv_a, deriv_b])
    }

    fn toy_family_and_derivatives(
        psi: &Array1<f64>,
    ) -> (
        TransformationNormalFamily,
        Vec<Vec<CustomFamilyBlockPsiDerivative>>,
        ParameterBlockState,
        ParameterBlockSpec,
    ) {
        let response_val_basis = array![[1.0, -1.0], [1.0, -0.2], [1.0, 0.6], [1.0, 1.3],];
        let response_deriv_basis = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],];
        let weights = Array1::from_elem(response_val_basis.nrows(), 1.0);
        let offset = Array1::zeros(response_val_basis.nrows());
        let (cov_design, cov_derivs) = toy_covariate_design_and_derivs(psi);
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            response_val_basis,
            response_deriv_basis,
            vec![],
            Array1::zeros(0),
            1,
            Array2::zeros((0, 0)),
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
            vec![],
            &TransformationNormalConfig {
                double_penalty: false,
                ..TransformationNormalConfig::default()
            },
            None,
        )
        .expect("toy transformation family");
        let derivative_blocks =
            vec![build_tensor_psi_derivatives(&family, &cov_derivs).expect("tensor psi derivs")];
        let beta = array![0.15, -0.05, 0.80, 0.30];
        let h_prime = family.x_deriv_kron.forward_mul(&beta);
        assert!(
            h_prime.iter().all(|v| *v > 0.25),
            "toy beta must keep h' positive, got {h_prime:?}"
        );
        let state = ParameterBlockState {
            beta,
            eta: Array1::zeros(h_prime.len()),
        };
        let spec = family.block_spec();
        (family, derivative_blocks, state, spec)
    }

    #[test]
    fn transformation_normal_joint_psi_second_order_terms_match_fd() {
        let psi = array![0.15, -0.10];
        let h = 1e-6;
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let states = vec![state.clone()];
        let specs = vec![spec];

        let analytic = family
            .exact_newton_joint_psisecond_order_terms(&states, &specs, &derivative_blocks, 0, 1)
            .expect("analytic psi second-order terms")
            .expect("psi second-order terms should be present");

        let eval_first = |psi_eval: &Array1<f64>| {
            let (f_eval, deriv_eval, state_eval, spec_eval) = toy_family_and_derivatives(psi_eval);
            let states_eval = vec![state_eval];
            let specs_eval = vec![spec_eval];
            f_eval
                .exact_newton_joint_psi_terms(&states_eval, &specs_eval, &deriv_eval, 0)
                .expect("first-order psi terms")
                .expect("first-order terms should be present")
        };

        let mut psi_plus = psi.clone();
        psi_plus[1] += h;
        let plus = eval_first(&psi_plus);
        let mut psi_minus = psi.clone();
        psi_minus[1] -= h;
        let minus = eval_first(&psi_minus);

        let objective_fd = (plus.objective_psi - minus.objective_psi) / (2.0 * h);
        assert!(
            (analytic.objective_psi_psi - objective_fd).abs() < 1e-5,
            "objective psi second-order mismatch: analytic={}, fd={objective_fd}",
            analytic.objective_psi_psi
        );

        let score_fd = (&plus.score_psi - &minus.score_psi) / (2.0 * h);
        for idx in 0..score_fd.len() {
            assert!(
                (analytic.score_psi_psi[idx] - score_fd[idx]).abs() < 1e-5,
                "score psi second-order mismatch at {idx}: analytic={}, fd={}",
                analytic.score_psi_psi[idx],
                score_fd[idx]
            );
        }

        let hess_fd = (&plus.hessian_psi - &minus.hessian_psi) / (2.0 * h);
        assert_matrix_derivativefd(
            &hess_fd,
            &analytic.hessian_psi_psi,
            2e-4,
            "transformation normal psi second-order Hessian",
        );
    }

    #[test]
    fn transformation_normal_joint_psihessian_directional_derivative_matches_fd() {
        let psi = array![0.15, -0.10];
        let h = 1e-6;
        let direction = array![0.02, -0.01, 0.03, 0.015];
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let specs = vec![spec];

        let analytic = family
            .exact_newton_joint_psihessian_directional_derivative(
                std::slice::from_ref(&state),
                &specs,
                &derivative_blocks,
                0,
                &direction,
            )
            .expect("analytic psi hessian directional derivative")
            .expect("psi hessian directional derivative should be present");

        let eval_hess = |beta: &Array1<f64>| {
            let mut shifted_state = state.clone();
            shifted_state.beta = beta.clone();
            family
                .exact_newton_joint_psi_terms(
                    std::slice::from_ref(&shifted_state),
                    &specs,
                    &derivative_blocks,
                    0,
                )
                .expect("first-order psi terms at shifted beta")
                .expect("shifted first-order terms should be present")
                .hessian_psi
        };

        let beta_plus = &state.beta + &(direction.clone() * h);
        let beta_minus = &state.beta - &(direction * h);
        let fd = (eval_hess(&beta_plus) - eval_hess(&beta_minus)) / (2.0 * h);
        assert_matrix_derivativefd(
            &fd,
            &analytic,
            2e-4,
            "transformation normal psi hessian directional derivative",
        );
    }

    #[test]
    fn transformation_normal_joint_hessian_second_directional_derivative_matches_fd() {
        let psi = array![0.15, -0.10];
        let h = 1e-6;
        let dir_u = array![0.02, -0.01, 0.03, 0.015];
        let dir_v = array![-0.01, 0.02, 0.01, -0.025];
        let (family, _, state, _) = toy_family_and_derivatives(&psi);

        let analytic = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                std::slice::from_ref(&state),
                &dir_u,
                &dir_v,
            )
            .expect("analytic second directional derivative")
            .expect("second directional derivative should be present");

        let eval_dh = |beta: &Array1<f64>| {
            let shifted_state = ParameterBlockState {
                beta: beta.clone(),
                eta: state.eta.clone(),
            };
            family
                .exact_newton_joint_hessian_directional_derivative(
                    std::slice::from_ref(&shifted_state),
                    &dir_u,
                )
                .expect("first directional derivative at shifted beta")
                .expect("shifted first directional derivative should be present")
        };

        let beta_plus = &state.beta + &(dir_v.clone() * h);
        let beta_minus = &state.beta - &(dir_v * h);
        let fd = (eval_dh(&beta_plus) - eval_dh(&beta_minus)) / (2.0 * h);
        assert_matrix_derivativefd(&fd, &analytic, 2e-4, "transformation normal joint d2H");
    }
}

impl CustomFamilyPsiDerivativeOperator for TensorKroneckerPsiOperator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn n_data(&self) -> usize {
        TensorKroneckerPsiOperator::n_data(self)
    }

    fn p_out(&self) -> usize {
        TensorKroneckerPsiOperator::p_out(self)
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_transpose(&self.response_val_basis, axis, v)
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_forward(&self.response_val_basis, axis, u)
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_transpose_second(&self.response_val_basis, axis, axis, v)
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_transpose_second(&self.response_val_basis, axis_d, axis_e, v)
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_forward_second(&self.response_val_basis, axis, axis, u)
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_forward_second(&self.response_val_basis, axis_d, axis_e, u)
    }

    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self.materialize_lifted(&self.response_val_basis, &self.materialize_cov_first(axis)?))
    }

    fn materialize_second_diag(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self.materialize_lifted(
            &self.response_val_basis,
            &self.materialize_cov_second(axis, axis)?,
        ))
    }

    fn materialize_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self.materialize_lifted(
            &self.response_val_basis,
            &self.materialize_cov_second(axis_d, axis_e)?,
        ))
    }
}

fn extract_covariate_penalty_factor(penalty: &PenaltyMatrix) -> Result<Array2<f64>, String> {
    match penalty {
        PenaltyMatrix::Dense(matrix) => Ok(matrix.clone()),
        PenaltyMatrix::Blockwise { .. } => Ok(penalty.to_dense()),
        PenaltyMatrix::KroneckerFactored { .. } => Err(
            "transformation covariate psi penalties must be single-block, not already Kronecker-factored"
                .to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Psi derivative builder (for κ optimization of the covariate basis)
// ---------------------------------------------------------------------------

/// Build `CustomFamilyBlockPsiDerivative` objects for the tensor product.
///
/// Given covariate-side psi derivatives (∂cov_design/∂ψ_a and ∂S_cov/∂ψ_a),
/// this constructs the corresponding tensor-product-space psi derivatives that
/// the REML evaluator needs.
///
/// Each output entry contains:
/// - implicit `x_psi` / `x_psi_psi` operators that preserve Kronecker structure
/// - factored tensor penalty derivatives `I_resp ⊗ ∂S_cov/∂ψ`
pub fn build_tensor_psi_derivatives(
    family: &TransformationNormalFamily,
    covariate_psi_derivs: &[CustomFamilyBlockPsiDerivative],
) -> Result<Vec<CustomFamilyBlockPsiDerivative>, String> {
    let p_resp = family.response_val_basis.ncols();
    let n_axes = covariate_psi_derivs.len();
    let eye_resp = Array2::<f64>::eye(p_resp);
    let shared_operator: Arc<dyn CustomFamilyPsiDerivativeOperator> =
        Arc::new(TensorKroneckerPsiOperator {
            response_val_basis: Arc::new(family.response_val_basis.clone()),
            response_deriv_basis: Arc::new(family.response_deriv_basis.clone()),
            covariate_design: family.covariate_design.clone(),
            covariate_derivs: covariate_psi_derivs.to_vec(),
        });

    let mut derivs = Vec::with_capacity(n_axes);
    for a in 0..n_axes {
        let cov_deriv = &covariate_psi_derivs[a];
        let s_psi_penalty_components = cov_deriv
            .s_psi_penalty_components
            .as_ref()
            .map(|components| {
                components
                    .iter()
                    .map(|(idx, ds_cov)| -> Result<_, String> {
                        Ok((
                            *idx,
                            PenaltyMatrix::KroneckerFactored {
                                left: eye_resp.clone(),
                                right: extract_covariate_penalty_factor(ds_cov)?,
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .or_else(|| {
                cov_deriv.s_psi_components.as_ref().map(|components| {
                    components
                        .iter()
                        .map(|(idx, ds_cov)| {
                            (
                                *idx,
                                PenaltyMatrix::KroneckerFactored {
                                    left: eye_resp.clone(),
                                    right: ds_cov.clone(),
                                },
                            )
                        })
                        .collect::<Vec<_>>()
                })
            });
        let s_psi_psi_penalty_components = cov_deriv
            .s_psi_psi_penalty_components
            .as_ref()
            .map(|rows| {
                rows.iter()
                    .map(|cov_pen_pairs| -> Result<_, String> {
                        cov_pen_pairs
                            .iter()
                            .map(|(idx, ds2)| -> Result<_, String> {
                                Ok((
                                    *idx,
                                    PenaltyMatrix::KroneckerFactored {
                                        left: eye_resp.clone(),
                                        right: extract_covariate_penalty_factor(ds2)?,
                                    },
                                ))
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .or_else(|| {
                cov_deriv.s_psi_psi_components.as_ref().map(|rows| {
                    rows.iter()
                        .map(|cov_pen_pairs| {
                            cov_pen_pairs
                                .iter()
                                .map(|(idx, ds2)| {
                                    (
                                        *idx,
                                        PenaltyMatrix::KroneckerFactored {
                                            left: eye_resp.clone(),
                                            right: ds2.clone(),
                                        },
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            });

        let mut deriv = CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::<f64>::zeros((0, 0)),
            Array2::<f64>::zeros((0, 0)),
            None,
            None,
            None,
            None,
        );
        deriv.s_psi_penalty_components = s_psi_penalty_components;
        deriv.s_psi_psi_penalty_components = s_psi_psi_penalty_components;
        deriv.implicit_operator = Some(Arc::clone(&shared_operator));
        deriv.implicit_axis = a;
        deriv.implicit_group_id = Some(0);
        derivs.push(deriv);
    }

    Ok(derivs)
}

// ---------------------------------------------------------------------------
// Top-level fit function
// ---------------------------------------------------------------------------

/// Result of `fit_transformation_normal`.
pub struct TransformationNormalFitResult {
    pub family: TransformationNormalFamily,
    pub fit: UnifiedFitResult,
    pub covariate_spec_resolved: TermCollectionSpec,
    pub covariate_design: TermCollectionDesign,
}

/// Fit a conditional transformation model with N-block spatial length-scale
/// optimization over the covariate side.
///
/// The response-direction basis is built once (it does not depend on κ).
/// If no spatial length-scale terms are present in the covariate spec, the
/// model is fit directly. Otherwise, the N-block joint hyper-parameter
/// optimizer is used with a single block (the covariate spec).
pub fn fit_transformation_normal(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    offset: &Array1<f64>,
    covariate_data: ArrayView2<'_, f64>,
    covariate_spec: &TermCollectionSpec,
    config: &TransformationNormalConfig,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<TransformationNormalFitResult, String> {
    let options = options.clone();

    // 1. Build a bootstrap covariate design first so the response basis can
    // adapt to the tensor width instead of always using the global default.
    let boot_design = build_term_collection_design(covariate_data, covariate_spec)
        .map_err(|e| format!("failed to build bootstrap covariate design: {e}"))?;
    let boot_spec = freeze_term_collection_from_design(covariate_spec, &boot_design)
        .map_err(|e| format!("failed to freeze bootstrap covariate spatial basis centers: {e}"))?;
    let mut effective_config = config.clone();
    effective_config.response_num_internal_knots =
        effective_response_num_internal_knots(config, response.len(), boot_design.design.ncols());

    // 2. Build response basis ONCE — it is independent of κ once the effective
    // response complexity has been chosen.
    let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
        build_response_basis(response, &effective_config)?;

    // 3. Check whether spatial κ optimization is needed.
    let spatial_terms = spatial_length_scale_term_indices(covariate_spec);

    if spatial_terms.is_empty() || !kappa_options.enabled {
        // ------------------------------------------------------------------
        // NO κ: build family directly, fit, return.
        // ------------------------------------------------------------------
        let cov_design = boot_design;
        let cov_spec_resolved = boot_spec;

        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            resp_val,
            resp_deriv,
            resp_penalties,
            resp_knots.clone(),
            effective_config.response_degree,
            resp_transform,
            weights,
            offset,
            cov_design.design.clone(),
            cov_design
                .penalties
                .iter()
                .map(|bp| PenaltyMatrix::from_blockwise(bp.clone(), cov_design.design.ncols()))
                .collect(),
            &effective_config,
            warm_start,
        )?;
        let blocks = vec![family.block_spec()];
        let fit = fit_custom_family(&family, &blocks, &options)
            .map_err(|e| format!("transformation fit failed: {e}"))?;

        return Ok(TransformationNormalFitResult {
            family,
            fit,
            covariate_spec_resolved: cov_spec_resolved,
            covariate_design: cov_design,
        });
    }

    // ------------------------------------------------------------------
    // YES κ: use the N-block spatial length-scale optimizer (1 block).
    // ------------------------------------------------------------------

    // Build ExactJointHyperSetup for 1 block.
    let n_penalties = boot_design.penalties.len();
    let rho0 = Array1::<f64>::zeros(n_penalties);
    let rho_lower = Array1::<f64>::from_elem(n_penalties, -12.0);
    let rho_upper = Array1::<f64>::from_elem(n_penalties, 12.0);
    let kappa0 = SpatialLogKappaCoords::from_length_scales_aniso(
        covariate_spec,
        &spatial_terms,
        kappa_options,
    );
    let kappa_dims = kappa0.dims_per_term().to_vec();
    let kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso(&kappa_dims, kappa_options);
    let kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso(&kappa_dims, kappa_options);
    let joint_setup =
        ExactJointHyperSetup::new(rho0, rho_lower, rho_upper, kappa0, kappa_lower, kappa_upper);

    // Check analytic derivative capability.
    let analytic_psi_available =
        build_block_spatial_psi_derivatives(covariate_data, &boot_spec, &boot_design)?.is_some();

    // Build an initial family + blocks for capability probing.
    let probe_family = TransformationNormalFamily::from_prebuilt_response_basis(
        resp_val.clone(),
        resp_deriv.clone(),
        resp_penalties.clone(),
        resp_knots.clone(),
        effective_config.response_degree,
        resp_transform.clone(),
        weights,
        offset,
        boot_design.design.clone(),
        boot_design
            .penalties
            .iter()
            .map(|bp| PenaltyMatrix::from_blockwise(bp.clone(), boot_design.design.ncols()))
            .collect(),
        &effective_config,
        warm_start,
    )?;
    let probe_blocks = vec![probe_family.block_spec()];
    let (cap_gradient, cap_hessian) =
        custom_family_outer_derivatives(&probe_family, &probe_blocks, &options);
    let analytic_gradient = analytic_psi_available
        && matches!(
            cap_gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    let analytic_hessian = analytic_psi_available
        && matches!(
            cap_hessian,
            crate::solver::outer_strategy::Derivative::Analytic
        );

    // Shared mutable state for warm-starting across optimizer iterations.
    let beta_hint: RefCell<Option<Array1<f64>>> = RefCell::new(None);
    let exact_warm_start: RefCell<Option<CustomFamilyWarmStart>> = RefCell::new(None);

    // Clone response basis parts for use inside closures.
    let rv = resp_val.clone();
    let rd = resp_deriv.clone();
    let rp = resp_penalties.clone();
    let rk = resp_knots.clone();
    let rt = resp_transform.clone();
    let rdeg = effective_config.response_degree;
    let cfg = effective_config.clone();
    let ws = warm_start.cloned();

    // Helper: build family from prebuilt response basis + covariate design.
    let make_family =
        |cov_design: &TermCollectionDesign| -> Result<TransformationNormalFamily, String> {
            TransformationNormalFamily::from_prebuilt_response_basis(
                rv.clone(),
                rd.clone(),
                rp.clone(),
                rk.clone(),
                rdeg,
                rt.clone(),
                weights,
                offset,
                cov_design.design.clone(),
                cov_design
                    .penalties
                    .iter()
                    .map(|bp| PenaltyMatrix::from_blockwise(bp.clone(), cov_design.design.ncols()))
                    .collect(),
                &cfg,
                ws.as_ref(),
            )
        };

    // Helper: build blocks from family + beta hint.
    let make_blocks = |family: &TransformationNormalFamily| -> Vec<ParameterBlockSpec> {
        let mut spec = family.block_spec();
        if let Some(hint) = beta_hint.borrow().as_ref() {
            if hint.len() == spec.design.ncols() {
                spec.initial_beta = Some(hint.clone());
            }
        }
        vec![spec]
    };

    let block_specs_slice = [boot_spec.clone()];
    let block_term_indices_slice = [spatial_terms.clone()];

    let solved = optimize_spatial_length_scale_exact_joint(
        covariate_data,
        &block_specs_slice,
        &block_term_indices_slice,
        kappa_options,
        &joint_setup,
        crate::seeding::SeedRiskProfile::Gaussian,
        analytic_gradient,
        analytic_hessian,
        // fit_fn
        |_, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let family = make_family(&designs[0])?;
            let blocks = make_blocks(&family);
            let fit = fit_custom_family(&family, &blocks, &options)
                .map_err(|e| format!("transformation fit_fn: {e}"))?;
            // Update warm start hints.
            if let Some(block) = fit.block_states.first() {
                *beta_hint.borrow_mut() = Some(block.beta.clone());
            }
            Ok((family, fit))
        },
        // exact_fn
        |rho, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], need_hessian| {
            let family = make_family(&designs[0])?;
            let blocks = make_blocks(&family);

            // Build covariate-side psi derivatives, then lift to tensor product space.
            let cov_psi_derivs =
                build_block_spatial_psi_derivatives(covariate_data, &specs[0], &designs[0])?
                    .ok_or_else(|| {
                        "missing covariate spatial psi derivatives for transformation model"
                            .to_string()
                    })?;

            let tensor_derivs = build_tensor_psi_derivatives(&family, &cov_psi_derivs)?;

            // Single block: derivative_blocks[0] = tensor_derivs.
            let derivative_blocks = vec![tensor_derivs];

            let eval = evaluate_custom_family_joint_hyper(
                &family,
                &blocks,
                &options,
                rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                if need_hessian {
                    crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian
                } else {
                    crate::solver::estimate::reml::unified::EvalMode::ValueAndGradient
                },
            )
            .map_err(|e| format!("transformation exact_fn: {e}"))?;

            exact_warm_start.replace(Some(eval.warm_start));

            if need_hessian && !eval.outer_hessian.is_analytic() {
                return Err(
                    "transformation exact joint objective did not return an outer Hessian"
                        .to_string(),
                );
            }

            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |rho, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let family = make_family(&designs[0])?;
            let blocks = make_blocks(&family);
            let cov_psi_derivs =
                build_block_spatial_psi_derivatives(covariate_data, &specs[0], &designs[0])?
                    .ok_or_else(|| {
                        "missing covariate spatial psi derivatives for transformation model"
                            .to_string()
                    })?;
            let tensor_derivs = build_tensor_psi_derivatives(&family, &cov_psi_derivs)?;
            let derivative_blocks = vec![tensor_derivs];
            let eval = evaluate_custom_family_joint_hyper_efs(
                &family,
                &blocks,
                &options,
                rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
            )
            .map_err(|e| format!("transformation exact_efs_fn: {e}"))?;
            exact_warm_start.replace(Some(eval.warm_start));
            Ok(eval.efs_eval)
        },
    )?;

    // Extract the family and fit from the optimizer result.
    let (family, fit) = solved.fit;

    Ok(TransformationNormalFitResult {
        family,
        fit,
        covariate_spec_resolved: solved.resolved_specs.into_iter().next().unwrap(),
        covariate_design: solved.designs.into_iter().next().unwrap(),
    })
}
