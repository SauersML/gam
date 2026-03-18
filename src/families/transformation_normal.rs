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
//! exactly. Monotonicity is enforced via `LinearInequalityConstraints` on the
//! derivative design matrix.
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
    CustomFamilyPsiDerivativeOperator, CustomFamilyWarmStart, ExactNewtonJointPsiTerms,
    ExactOuterDerivativeOrder, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix, build_block_spatial_psi_derivatives, custom_family_outer_capability,
    evaluate_custom_family_joint_hyper, fit_custom_family,
};
use crate::families::gamlss::initializewiggle_knots_from_seed;
use crate::matrix::{DesignMatrix, DesignOperator, LinearOperator, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_spatial_length_scale_terms_from_design, optimize_spatial_length_scale_exact_joint,
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
    /// Number of grid points for monotonicity constraint evaluation (default 64).
    pub derivative_grid_size: usize,
    /// Lower bound for the Jacobian h' (default 1e-6).
    pub monotonicity_eps: f64,
}

impl Default for TransformationNormalConfig {
    fn default() -> Self {
        Self {
            response_degree: 3,
            response_num_internal_knots: 10,
            response_penalty_order: 2,
            response_extra_penalty_orders: vec![1],
            double_penalty: true,
            derivative_grid_size: 64,
            monotonicity_eps: 1e-6,
        }
    }
}

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
    // --- Tensor penalties ---
    tensor_penalties: Vec<PenaltyMatrix>,

    // --- Monotonicity constraints ---
    monotonicity_constraints: LinearInequalityConstraints,

    // --- Initial values ---
    initial_beta: Array1<f64>,
    initial_log_lambdas: Array1<f64>,

    // --- Config ---
    block_name: String,
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

        // ----- 3. Tensor penalties (Kronecker-separable) -----
        let tensor_penalties = build_tensor_penalties_kronecker(
            &resp_penalties,
            covariate_penalties,
            p_resp,
            p_cov,
            config,
        )?;

        // ----- 4. Monotonicity constraints -----
        let monotonicity_constraints = build_subsampled_monotonicity_constraints(
            &x_deriv_kron,
            response,
            &resp_knots,
            config.response_degree,
            &resp_transform,
            &covariate_design,
            config,
        )?;

        // ----- 5. Warm start -----
        let initial_beta =
            compute_warm_start(response, &covariate_design, p_resp, p_cov, warm_start)?;

        // ----- 6. Initial log-lambdas (one per penalty, start at 0.0) -----
        let initial_log_lambdas = Array1::zeros(tensor_penalties.len());

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            response_val_basis: resp_val,
            response_deriv_basis: resp_deriv,
            covariate_design,
            tensor_penalties,
            monotonicity_constraints,
            initial_beta,
            initial_log_lambdas,
            block_name: "transformation".to_string(),
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
        let x_deriv_kron =
            KroneckerDesign::new(&response_deriv_basis, covariate_design.clone())?;
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_kron.ncols(), p_total);

        // Tensor penalties (Kronecker-separable).
        let tensor_penalties = build_tensor_penalties_kronecker(
            &response_penalties,
            covariate_penalties,
            p_resp,
            p_cov,
            config,
        )?;

        // Monotonicity constraints.
        // Extract response values from column 1 of the value basis (which stores y)
        // to build the boundary grid.  For biobank scale, subsample training rows.
        let response_approx_for_grid = response_val_basis.column(1).to_owned();
        let monotonicity_constraints = build_subsampled_monotonicity_constraints(
            &x_deriv_kron,
            &response_approx_for_grid,
            &response_knots,
            response_degree,
            &response_transform,
            &covariate_design,
            config,
        )?;

        // Warm start: need response values for location-scale init.
        // Extract response from column 1 of response_val_basis (which stores y).
        let response_approx = response_val_basis.column(1).to_owned();
        let initial_beta = compute_warm_start(
            &response_approx,
            &covariate_design,
            p_resp,
            p_cov,
            warm_start,
        )?;

        let initial_log_lambdas = Array1::zeros(tensor_penalties.len());

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            response_val_basis,
            response_deriv_basis,
            covariate_design,
            tensor_penalties,
            monotonicity_constraints,
            initial_beta,
            initial_log_lambdas,
            block_name: "transformation".to_string(),
        })
    }

    /// Return the `ParameterBlockSpec` for this family (single block).
    pub fn block_spec(&self) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: self.block_name.clone(),
            design: DesignMatrix::Operator(Arc::new(self.x_val_kron.clone())),
            offset: Array1::zeros(self.x_val_kron.nrows()),
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
        let h = self.x_val_kron.forward_mul(beta);
        let h_prime = self.x_deriv_kron.forward_mul(beta);
        (h, h_prime)
    }

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
            log_likelihood += -0.5 * h[i] * h[i] + h_prime[i].ln();
        }

        // Gradient of log-likelihood: ∇ℓ = -X_val^T h + X_deriv^T (1/h')
        let inv_h_prime = h_prime.mapv(|v| 1.0 / v);
        // gradient = -X_val^T h + X_deriv^T inv_h_prime
        let grad = {
            let neg_xvt_h = self.x_val_kron.transpose_mul(&h).mapv(|v| -v);
            let xdt_inv = self.x_deriv_kron.transpose_mul(&inv_h_prime);
            neg_xvt_h + &xdt_inv
        };

        // Hessian of negative log-likelihood: -∇²ℓ = X_val^T X_val + X_deriv^T diag(1/h'²) X_deriv
        let inv_h_prime_sq = h_prime.mapv(|v| 1.0 / (v * v));
        let hessian = {
            let xtx_val = self.x_val_kron.gram();
            // X_deriv^T diag(w) X_deriv where w = 1/h'^2
            let xtx_deriv = self.x_deriv_kron.weighted_gram(&inv_h_prime_sq);
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
                return Err("h' non-positive in log_likelihood_only".to_string());
            }
            ll += -0.5 * h[i] * h[i] + h_prime[i].ln();
        }
        Ok(ll)
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        // The Hessian depends on β through 1/h'² where h' = X_deriv · β.
        true
    }

    fn exact_outer_derivative_order(
        &self,
        _: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        let n = self.n_obs();
        let p = self.p_total();
        // Downgrade to first-order if the Hessian directional derivative is too
        // expensive (O(n·p²)).
        if (n as u64) * (p as u64) * (p as u64) > 2_000_000 {
            ExactOuterDerivativeOrder::First
        } else {
            ExactOuterDerivativeOrder::Second
        }
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_index: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_index != 0 {
            return Ok(None);
        }
        Ok(Some(self.monotonicity_constraints.clone()))
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
            weight[i] = -2.0 * d_h_prime[i] / (h_prime[i] * h_prime[i] * h_prime[i]);
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
        let xtx_val = self.x_val_kron.gram();
        let xtx_deriv = self.x_deriv_kron.weighted_gram(&inv_h_prime_sq);
        Ok(Some(xtx_val + &xtx_deriv))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_hessian_directional_derivative(block_states, 0, d_beta_flat)
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
            obj_psi += h[i] * v_val[i] - inv_h_prime[i] * v_deriv[i];
        }

        let score_psi = {
            let term1 = op
                .transpose_mul(axis, &h.view())
                .map_err(|e| format!("tensor psi transpose_mul failed: {e}"))?;
            let term2 = self.x_val_kron.transpose_mul(&v_val);
            let term3 = op
                .transpose_mul_deriv(axis, &inv_h_prime)
                .map_err(|e| format!("tensor psi derivative transpose_mul failed: {e}"))?
                .mapv(|v| -v);
            let w_deriv = &v_deriv * &inv_h_prime_sq;
            let term4 = self.x_deriv_kron.transpose_mul(&w_deriv);
            term1 + &term2 + &term3 + &term4
        };

        let hessian_psi = {
            let xvt_xvp = op
                .weighted_cross_with_cov_first(
                    &self.response_val_basis,
                    &self.response_val_basis,
                    axis,
                    &Array1::ones(n),
                )
                .map_err(|e| format!("tensor psi weighted_cross(value) failed: {e}"))?;
            let sym_val = &xvt_xvp + &xvt_xvp.t();

            let xdt_xdp = op
                .weighted_cross_with_cov_first(
                    &self.response_deriv_basis,
                    &self.response_deriv_basis,
                    axis,
                    &inv_h_prime_sq,
                )
                .map_err(|e| format!("tensor psi weighted_cross(derivative) failed: {e}"))?;
            let sym_deriv = &xdt_xdp + &xdt_xdp.t();

            let w_cubic = (&v_deriv * &inv_h_prime_cu).mapv(|v| -2.0 * v);
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
        left: Array2<f64>,  // n × p_a
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

    /// Compute `self^T · self`, the Gram matrix.
    /// Returns a (p_a·p_b) × (p_a·p_b) matrix.
    fn gram(&self) -> Array2<f64> {
        match self {
            KroneckerDesign::Factored { .. } => {
                let dense = self.to_dense();
                fast_atb(&dense, &dense)
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

impl DesignOperator for KroneckerDesign {
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
// Monotonicity constraints
// ---------------------------------------------------------------------------

/// Evenly spaced grid over the response range for boundary constraint evaluation.
fn response_derivative_grid(
    response: &Array1<f64>,
    config: &TransformationNormalConfig,
) -> Result<Array1<f64>, String> {
    let grid_size = config.derivative_grid_size;
    if grid_size < 2 {
        return Ok(Array1::zeros(0));
    }
    let min_val = response.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = response.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !min_val.is_finite() || !max_val.is_finite() {
        return Err("non-finite response values for derivative grid".to_string());
    }
    let range = max_val - min_val;
    if range < 1e-12 {
        return Ok(Array1::from_vec(vec![min_val; grid_size]));
    }
    // Extend slightly beyond data range.
    let margin = 0.05 * range;
    let left = min_val - margin;
    let right = max_val + margin;
    Ok(Array1::from_iter((0..grid_size).map(|i| {
        let t = i as f64 / ((grid_size - 1) as f64);
        left + (right - left) * t
    })))
}

/// Subsample covariate rows uniformly for boundary constraints.
fn subsample_covariate_rows(design: &DesignMatrix, max_rows: usize) -> Array2<f64> {
    let n = design.nrows();
    if n <= max_rows {
        return design.row_chunk(0..n);
    }
    let step = n as f64 / max_rows as f64;
    let indices: Vec<usize> = (0..max_rows)
        .map(|i| ((i as f64 * step) as usize).min(n - 1))
        .collect();
    let mut out = Array2::zeros((indices.len(), design.ncols()));
    for (r, &idx) in indices.iter().enumerate() {
        let row = design.row_chunk(idx..idx + 1);
        out.row_mut(r).assign(&row.row(0));
    }
    out
}

/// Maximum number of training-point constraint rows for biobank-scale monotonicity.
///
/// When the training set is large (e.g. 400K), using all n derivative rows as
/// hard constraints is prohibitively expensive.  This cap limits the training-point
/// portion of the constraint matrix.  Boundary grid rows are added on top of this.
const MAX_MONOTONICITY_TRAINING_ROWS: usize = 8192;

/// Build monotonicity constraints from a KroneckerDesign, subsampling for
/// large problems.
///
/// This mirrors what `build_monotonicity_constraints` does for the non-prebuilt
/// path: it includes BOTH training-point derivative rows (subsampled if n is
/// large) AND boundary grid rows that evaluate h'(y|x) on a fine grid of
/// response values crossed with subsampled covariate rows.  The boundary grid
/// is critical for preventing B-spline derivatives from going negative between
/// observed response values, especially near domain boundaries.
fn build_subsampled_monotonicity_constraints(
    x_deriv_kron: &KroneckerDesign,
    response_values: &Array1<f64>,
    response_knots: &Array1<f64>,
    response_degree: usize,
    response_transform: &Array2<f64>,
    covariate_design: &DesignMatrix,
    config: &TransformationNormalConfig,
) -> Result<LinearInequalityConstraints, String> {
    let n = x_deriv_kron.nrows();
    let p = x_deriv_kron.ncols();
    let eps = config.monotonicity_eps;

    // --- Part 1: training-point derivative rows (subsampled if large) ---
    let mut constraint_rows: Vec<Array1<f64>> = Vec::new();

    if n <= MAX_MONOTONICITY_TRAINING_ROWS {
        // Use all training-point rows.
        for i in 0..n {
            let row = x_deriv_kron.row_chunk(i..i + 1).row(0).to_owned();
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-12 {
                constraint_rows.push(row);
            }
        }
    } else {
        // Subsample uniformly.
        let step = n as f64 / MAX_MONOTONICITY_TRAINING_ROWS as f64;
        let indices: Vec<usize> = (0..MAX_MONOTONICITY_TRAINING_ROWS)
            .map(|i| ((i as f64 * step) as usize).min(n - 1))
            .collect();
        for &idx in &indices {
            let row = x_deriv_kron.row_chunk(idx..idx + 1).row(0).to_owned();
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-12 {
                constraint_rows.push(row);
            }
        }
    }

    // --- Part 2: boundary grid rows ---
    // Evaluate the response derivative basis on a fine grid of response values,
    // crossed with a subsample of covariate rows.  This catches negativity
    // between observed response values and near domain boundaries.
    let grid = response_derivative_grid(response_values, config)?;
    let p_cov = covariate_design.ncols();
    if !grid.is_empty() && p_cov > 0 {
        let (raw_d1, _) = create_basis::<Dense>(
            grid.view(),
            KnotSource::Provided(response_knots.view()),
            response_degree,
            BasisOptions::first_derivative(),
        )
        .map_err(|e| e.to_string())?;
        let dev_d1 = raw_d1.as_ref().dot(response_transform);
        let p_resp = 2 + dev_d1.ncols();
        let g = grid.len();
        let mut grid_resp_deriv = Array2::<f64>::zeros((g, p_resp));
        grid_resp_deriv.column_mut(1).fill(1.0); // derivative of y is 1
        grid_resp_deriv.slice_mut(s![.., 2..]).assign(&dev_d1);

        // Use a subsample of covariate rows for the boundary constraint.
        let cov_subsample = subsample_covariate_rows(covariate_design, 50);
        for gi in 0..g {
            for ci in 0..cov_subsample.nrows() {
                let mut row = Array1::<f64>::zeros(p);
                for j in 0..p_resp {
                    let r = grid_resp_deriv[[gi, j]];
                    if r == 0.0 {
                        continue;
                    }
                    for k in 0..p_cov {
                        row[j * p_cov + k] = r * cov_subsample[[ci, k]];
                    }
                }
                let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    constraint_rows.push(row);
                }
            }
        }
    }

    let n_constraints = constraint_rows.len();
    if n_constraints == 0 {
        return Err("no valid constraint rows for monotonicity".to_string());
    }
    let mut a = Array2::<f64>::zeros((n_constraints, p));
    for (i, row) in constraint_rows.into_iter().enumerate() {
        a.row_mut(i).assign(&row);
    }
    let b = Array1::from_elem(n_constraints, eps);

    Ok(LinearInequalityConstraints { a, b })
}

// ---------------------------------------------------------------------------
// Warm start
// ---------------------------------------------------------------------------

/// Compute initial β so that h(y|x) ≈ (y - μ(x)) / τ(x).
///
/// If no warm start is provided, uses global mean and SD.
fn compute_warm_start(
    response: &Array1<f64>,
    covariate_design: &DesignMatrix,
    p_resp: usize,
    p_cov: usize,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<Array1<f64>, String> {
    let n = response.len();
    let p_total = p_resp * p_cov;
    let mut beta = Array1::zeros(p_total);

    // Target: for the intercept row (j=0), Θ[0,:] · cov[i,:] = -μ(x_i)/τ(x_i)
    //         for the linear row (j=1),  Θ[1,:] · cov[i,:] = 1/τ(x_i)
    //         for deviation rows (j≥2),  Θ[j,:] = 0

    let (target_intercept, target_slope) = match warm_start {
        Some(ws) => {
            if ws.location.len() != n || ws.scale.len() != n {
                return Err("warm start location/scale length mismatch".to_string());
            }
            let mut intercept = Array1::zeros(n);
            let mut slope = Array1::zeros(n);
            for i in 0..n {
                let tau = ws.scale[i].max(1e-12);
                intercept[i] = -ws.location[i] / tau;
                slope[i] = 1.0 / tau;
            }
            (intercept, slope)
        }
        None => {
            let mean = response.mean().unwrap_or(0.0);
            let var = response
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f64>()
                / (n.max(2) - 1) as f64;
            let sd = var.sqrt().max(1e-12);
            let intercept = Array1::from_elem(n, -mean / sd);
            let slope = Array1::from_elem(n, 1.0 / sd);
            (intercept, slope)
        }
    };

    // Least-squares fit: Θ[j,:] = (cov^T cov)^{-1} cov^T target_j
    // Use pseudoinverse via normal equations.
    let ctc = covariate_design
        .diag_xtw_x(&Array1::ones(covariate_design.nrows()))
        .map_err(|e| format!("warm start X'X failed: {e}"))?;
    // Add small ridge for numerical stability.
    let mut ctc_ridge = ctc;
    for i in 0..p_cov {
        ctc_ridge[[i, i]] += 1e-10;
    }

    // Solve via Cholesky (ctc_ridge is SPD).
    use crate::faer_ndarray::FaerCholesky;
    match ctc_ridge.cholesky(faer::Side::Lower) {
        Ok(chol) => {
            let ct_int = covariate_design.apply_transpose(&target_intercept);
            let coeff_int = chol.solvevec(&ct_int);
            let ct_slope = covariate_design.apply_transpose(&target_slope);
            let coeff_slope = chol.solvevec(&ct_slope);

            // Place into β: Θ[0,:] = coeff_int, Θ[1,:] = coeff_slope
            beta.slice_mut(s![0..p_cov]).assign(&coeff_int);
            beta.slice_mut(s![p_cov..2 * p_cov]).assign(&coeff_slope);
        }
        Err(_) => {
            // Fallback: use global mean/SD.
            let mean = response.mean().unwrap_or(0.0);
            let var = response
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f64>()
                / (n.max(2) - 1) as f64;
            let sd = var.sqrt().max(1e-12);
            // Only set the first covariate column (assumes intercept is first column).
            if p_cov > 0 {
                beta[0] = -mean / sd; // intercept row, first cov column
                beta[p_cov] = 1.0 / sd; // slope row, first cov column
            }
        }
    }

    Ok(beta)
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

    fn cov_deriv(&self, axis: usize) -> Result<&CustomFamilyBlockPsiDerivative, crate::terms::basis::BasisError> {
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
            out.slice_mut(s![j * p_cov..(j + 1) * p_cov]).assign(&cov_block);
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
            out.slice_mut(s![j * p_cov..(j + 1) * p_cov]).assign(&cov_block);
        }
        Ok(out)
    }

    fn materialize_lifted(
        &self,
        resp_basis: &Array2<f64>,
        cov: &Array2<f64>,
    ) -> Array2<f64> {
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
                    pair_weights[i] = weights[i] * left_resp_basis[[i, a]] * right_resp_basis[[i, b]];
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
                out.slice_mut(s![
                    a * p_cov..(a + 1) * p_cov,
                    b * p_cov..(b + 1) * p_cov
                ])
                .assign(&block);
            }
        }
        Ok(out)
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
    let shared_operator: Arc<dyn CustomFamilyPsiDerivativeOperator> = Arc::new(
        TensorKroneckerPsiOperator {
            response_val_basis: Arc::new(family.response_val_basis.clone()),
            response_deriv_basis: Arc::new(family.response_deriv_basis.clone()),
            covariate_design: family.covariate_design.clone(),
            covariate_derivs: covariate_psi_derivs.to_vec(),
        },
    );

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
    covariate_data: ArrayView2<'_, f64>,
    covariate_spec: &TermCollectionSpec,
    config: &TransformationNormalConfig,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<TransformationNormalFitResult, String> {
    // 1. Build response basis ONCE — it is independent of κ.
    let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
        build_response_basis(response, config)?;

    // 2. Check whether spatial κ optimization is needed.
    let spatial_terms = spatial_length_scale_term_indices(covariate_spec);

    if spatial_terms.is_empty() || !kappa_options.enabled {
        // ------------------------------------------------------------------
        // NO κ: build family directly, fit, return.
        // ------------------------------------------------------------------
        let cov_design = build_term_collection_design(covariate_data, covariate_spec)
            .map_err(|e| format!("failed to build covariate design: {e}"))?;
        let cov_spec_resolved =
            freeze_spatial_length_scale_terms_from_design(covariate_spec, &cov_design)
                .map_err(|e| format!("failed to freeze covariate spatial basis centers: {e}"))?;

        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            resp_val,
            resp_deriv,
            resp_penalties,
            resp_knots.clone(),
            config.response_degree,
            resp_transform,
            cov_design.design.clone(),
            cov_design
                .penalties
                .iter()
                .map(|bp| PenaltyMatrix::from_blockwise(bp.clone(), cov_design.design.ncols()))
                .collect(),
            config,
            warm_start,
        )?;
        let blocks = vec![family.block_spec()];
        let fit = fit_custom_family(&family, &blocks, options)
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

    // Build bootstrap covariate design and frozen spec.
    let boot_design = build_term_collection_design(covariate_data, covariate_spec)
        .map_err(|e| format!("failed to build bootstrap covariate design: {e}"))?;
    let boot_spec = freeze_spatial_length_scale_terms_from_design(covariate_spec, &boot_design)
        .map_err(|e| format!("failed to freeze bootstrap covariate spatial basis centers: {e}"))?;

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
        config.response_degree,
        resp_transform.clone(),
        boot_design.design.clone(),
        boot_design
            .penalties
            .iter()
            .map(|bp| PenaltyMatrix::from_blockwise(bp.clone(), boot_design.design.ncols()))
            .collect(),
        config,
        warm_start,
    )?;
    let probe_blocks = vec![probe_family.block_spec()];
    let joint_cap = custom_family_outer_capability(
        &probe_family,
        &probe_blocks,
        options,
        joint_setup.rho_dim() + joint_setup.log_kappa_dim(),
        joint_setup.log_kappa_dim() > 0,
    );
    let analytic_gradient = analytic_psi_available
        && matches!(
            joint_cap.gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    let analytic_hessian = analytic_psi_available
        && matches!(
            joint_cap.hessian,
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
    let rdeg = config.response_degree;
    let cfg = config.clone();
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
        analytic_gradient,
        analytic_hessian,
        // fit_fn
        |_, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let family = make_family(&designs[0])?;
            let blocks = make_blocks(&family);
            let fit = fit_custom_family(&family, &blocks, options)
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
                options,
                rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                need_hessian,
            )
            .map_err(|e| format!("transformation exact_fn: {e}"))?;

            exact_warm_start.replace(Some(eval.warm_start));

            if need_hessian && eval.outer_hessian.is_none() {
                return Err(
                    "transformation exact joint objective did not return an outer Hessian"
                        .to_string(),
                );
            }

            Ok((eval.objective, eval.gradient, eval.outer_hessian))
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
