//! Conditional transformation model: estimate h(y|x) such that h(Y|x) ~ N(0,1).
//!
//! Given a response variable y and covariates x with a pre-built covariate design
//! matrix, this family estimates a smooth monotone transformation h(y | x) mapping
//! the conditional distribution of Y|x onto a standard normal.
//!
//! The response-direction basis is `[1, y, anchored B-spline deviations]`, tensored
//! with an arbitrary covariate design matrix. The deviations are B-splines with
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
use crate::construction::kronecker_product;
use crate::faer_ndarray::{default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_nullspace_basis};
use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyWarmStart, ExactNewtonJointPsiTerms, ExactOuterDerivativeOrder, FamilyEvaluation,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, build_block_spatial_psi_derivatives,
    custom_family_outer_capability, evaluate_custom_family_joint_hyper, fit_custom_family,
};
use crate::families::gamlss::initializewiggle_knots_from_seed;
use crate::matrix::{DesignMatrix, DesignOperator, SymmetricMatrix};
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
    /// Value design operator: stores factors separately for large problems and
    /// only materializes row-wise chunks on explicit dense fallback paths.
    x_val_kron: KroneckerDesign,
    /// Derivative design operator: stores factors separately for large problems.
    x_deriv_kron: KroneckerDesign,

    // --- Response-direction basis (fixed, does not depend on κ) ---
    /// Response value basis: n × p_resp. Columns: [1, y, dev_1(y), ..., dev_k(y)].
    response_val_basis: Array2<f64>,
    /// Response derivative basis: n × p_resp. Columns: [0, 1, dev'_1(y), ..., dev'_k(y)].
    response_deriv_basis: Array2<f64>,

    // --- Covariate side (rebuilt on κ change) ---
    /// Number of covariate basis columns.
    p_cov: usize,

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
    /// design matrix with associated penalties.
    ///
    /// # Arguments
    ///
    /// * `response` - The response variable y (n observations).
    /// * `covariate_design` - Pre-built covariate-side design matrix (n × p_cov).
    /// * `covariate_penalties` - Penalty matrices for the covariate basis (each p_cov × p_cov).
    /// * `config` - Response-direction basis configuration.
    /// * `warm_start` - Optional location/scale from a prior normalizer.
    pub fn new(
        response: &Array1<f64>,
        covariate_design: &Array2<f64>,
        covariate_penalties: &[Array2<f64>],
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
            if sp.nrows() != p_cov || sp.ncols() != p_cov {
                return Err(format!(
                    "covariate penalty {} has shape {:?}, expected ({p_cov}, {p_cov})",
                    i,
                    sp.dim()
                ));
            }
        }

        // ----- 1. Build response-direction basis -----
        let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
            build_response_basis(response, config)?;
        let p_resp = resp_val.ncols();

        // ----- 2. Row-wise Kronecker product (operator form) -----
        let x_val_kron = KroneckerDesign::new(&resp_val, covariate_design);
        let x_deriv_kron = KroneckerDesign::new(&resp_deriv, covariate_design);
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
            covariate_design,
            config,
        )?;

        // ----- 5. Warm start -----
        let initial_beta =
            compute_warm_start(response, covariate_design, p_resp, p_cov, warm_start)?;

        // ----- 6. Initial log-lambdas (one per penalty, start at 0.0) -----
        let initial_log_lambdas = Array1::zeros(tensor_penalties.len());

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            response_val_basis: resp_val,
            response_deriv_basis: resp_deriv,
            p_cov,
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
        covariate_design: &Array2<f64>,
        covariate_penalties: &[Array2<f64>],
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
            if sp.nrows() != p_cov || sp.ncols() != p_cov {
                return Err(format!(
                    "covariate penalty {} has shape {:?}, expected ({p_cov}, {p_cov})",
                    i,
                    sp.dim()
                ));
            }
        }

        let p_resp = response_val_basis.ncols();

        // Row-wise Kronecker product (operator form).
        let x_val_kron = KroneckerDesign::new(&response_val_basis, covariate_design);
        let x_deriv_kron = KroneckerDesign::new(&response_deriv_basis, covariate_design);
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_kron.ncols(), p_total);

        // Tensor penalties (dense + Kronecker-separable).
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
            covariate_design,
            config,
        )?;

        // Warm start: need response values for location-scale init.
        // Extract response from column 1 of response_val_basis (which stores y).
        let response_approx = response_val_basis.column(1).to_owned();
        let initial_beta = compute_warm_start(
            &response_approx,
            covariate_design,
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
            p_cov,
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
    /// Uses the Kronecker-aware operators when in factored mode, avoiding
    /// full n × p_total matrix-vector products through the materialized
    /// tensor product.
    fn compute_h_and_h_prime(&self, beta: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h = self.x_val_kron.forward_mul(beta);
        let h_prime = self.x_deriv_kron.forward_mul(beta);
        (h, h_prime)
    }

    /// Extract the covariate psi derivative from a tensor-product x_psi matrix.
    ///
    /// Since response_val_basis[:,0] = 1 (intercept column), the first p_cov columns
    /// of x_val_psi are exactly ∂(covariate_design)/∂ψ.
    fn extract_covariate_psi(&self, x_val_psi: &Array2<f64>) -> Array2<f64> {
        x_val_psi.slice(s![.., 0..self.p_cov]).to_owned()
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
        let x_val_psi = &psi_derivs[0][psi_index].x_psi; // ∂x_val/∂ψ, n × p_total

        let beta = &block_states[0].beta;
        let (h, h_prime) = self.compute_h_and_h_prime(beta);
        let n = h.len();

        // Build ∂x_deriv/∂ψ from x_val_psi.
        // x_val_psi[:, 0:p_cov] = ∂cov_design/∂ψ (because resp_val[:,0] = 1).
        let cov_psi = self.extract_covariate_psi(x_val_psi);
        let x_deriv_psi_kron = KroneckerDesign::new(&self.response_deriv_basis, &cov_psi);
        let x_deriv_psi = x_deriv_psi_kron.to_dense();

        // v_val = x_val_psi · β (n-vector): ∂h/∂ψ
        let v_val = x_val_psi.dot(beta);
        // v_deriv = x_deriv_psi · β (n-vector): ∂h'/∂ψ
        let v_deriv = x_deriv_psi_kron.forward_mul(beta);

        let inv_h_prime: Array1<f64> = h_prime.mapv(|v| 1.0 / v);
        let inv_h_prime_sq: Array1<f64> = h_prime.mapv(|v| 1.0 / (v * v));

        // --- objective_psi: ∂(-ℓ)/∂ψ at fixed β ---
        // = Σ [h_i · v_val_i - (1/h'_i) · v_deriv_i]
        let mut obj_psi = 0.0;
        for i in 0..n {
            obj_psi += h[i] * v_val[i] - inv_h_prime[i] * v_deriv[i];
        }

        // --- score_psi: ∂²(-ℓ)/(∂β∂ψ) at fixed β (p-vector) ---
        // = x_val_psi^T h + x_val^T v_val - x_deriv_psi^T (1/h') + x_deriv^T (v_deriv/h'^2)
        let score_psi = {
            let term1 = x_val_psi.t().dot(&h);
            let term2 = self.x_val_kron.transpose_mul(&v_val);
            let term3 = x_deriv_psi_kron.transpose_mul(&inv_h_prime).mapv(|v| -v);
            let w_deriv = &v_deriv * &inv_h_prime_sq;
            let term4 = self.x_deriv_kron.transpose_mul(&w_deriv);
            term1 + &term2 + &term3 + &term4
        };

        // --- hessian_psi: ∂H_L/∂ψ at fixed β (p × p) ---
        // = 2·sym(x_val^T x_val_psi)
        //   + 2·sym(x_deriv^T diag(1/h'^2) x_deriv_psi)
        //   - 2·x_deriv^T diag(v_deriv / h'^3) x_deriv
        let hessian_psi = {
            // sym(A^T B) = (A^T B + B^T A) / 2; we compute A^T B and symmetrize.
            let xvt_xvp = self.x_val_kron.weighted_cross(&Array1::ones(n), x_val_psi);
            let sym_val = &xvt_xvp + &xvt_xvp.t();

            let xdt_xdp = self
                .x_deriv_kron
                .weighted_cross(&inv_h_prime_sq, &*x_deriv_psi);
            let sym_deriv = &xdt_xdp + &xdt_xdp.t();

            let mut w_cubic = Array1::zeros(n);
            for i in 0..n {
                w_cubic[i] = -2.0 * v_deriv[i] / (h_prime[i] * h_prime[i] * h_prime[i]);
            }
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

/// Materialization threshold: if n * p_resp * p_cov exceeds this, keep
/// factors separate and use operator-form mul.  Below the threshold the
/// full n × (p_resp · p_cov) dense matrix is materialized for BLAS speed.
const KRONECKER_MATERIALIZE_THRESHOLD: u64 = 500_000;

/// A lazy representation of the row-wise Kronecker product `A ⊙ B`
/// (face-splitting product / Khatri–Rao row-wise product).
///
/// For small problems the full dense matrix is materialized at construction
/// time and all operations delegate to standard BLAS.  For large problems
/// only the two factor matrices are stored and products are evaluated in
/// factored form:
///
///   forward_mul(β):  reshape β → (p_a, p_b), then result[i] = Σ_j A[i,j] * (B[i,:] · β[j,:])
///   transpose_mul(v): result[j, k] = Σ_i v[i] * A[i,j] * B[i,k]
///
/// Storage: O(n·(p_a + p_b)) vs O(n·p_a·p_b) for the materialized form.
#[derive(Clone)]
enum KroneckerDesign {
    /// Full dense materialization (small problems).
    Dense(Array2<f64>),
    /// Factored form: (left_factor, right_factor) where the tensor product
    /// row i is `left[i,:] ⊗ right[i,:]`.
    Factored {
        left: Array2<f64>,  // n × p_a
        right: Array2<f64>, // n × p_b
    },
}

impl KroneckerDesign {
    /// Build from two factor matrices, choosing materialization or factored
    /// form based on the size threshold.
    fn new(left: &Array2<f64>, right: &Array2<f64>) -> Self {
        let n = left.nrows() as u64;
        let pa = left.ncols() as u64;
        let pb = right.ncols() as u64;
        if n * pa * pb <= KRONECKER_MATERIALIZE_THRESHOLD {
            KroneckerDesign::Dense(rowwise_kronecker(left, right))
        } else {
            KroneckerDesign::Factored {
                left: left.clone(),
                right: right.clone(),
            }
        }
    }

    /// Force-materialize (used when we truly need the full matrix, e.g. for
    /// legacy code paths that have not been converted to operator form).
    fn to_dense(&self) -> std::borrow::Cow<'_, Array2<f64>> {
        match self {
            KroneckerDesign::Dense(m) => std::borrow::Cow::Borrowed(m),
            KroneckerDesign::Factored { left, right } => {
                std::borrow::Cow::Owned(rowwise_kronecker(left, right))
            }
        }
    }

    fn nrows(&self) -> usize {
        match self {
            KroneckerDesign::Dense(m) => m.nrows(),
            KroneckerDesign::Factored { left, .. } => left.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            KroneckerDesign::Dense(m) => m.ncols(),
            KroneckerDesign::Factored { left, right } => left.ncols() * right.ncols(),
        }
    }

    /// Compute `self · beta` where beta has length p_a * p_b.
    /// Returns an n-vector.
    fn forward_mul(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::Dense(m) => m.dot(beta),
            KroneckerDesign::Factored { left, right } => {
                let pa = left.ncols();
                let pb = right.ncols();
                let n = left.nrows();
                debug_assert_eq!(beta.len(), pa * pb);
                // Reshape beta into (pa, pb) and compute:
                //   result[i] = Σ_j left[i,j] * (right[i,:] · beta_mat[j,:])
                let beta_mat = beta.view().into_shape_with_order((pa, pb)).unwrap();
                // First compute X_cov @ beta_mat^T  → n × p_a
                // (right · beta_mat^T)[i, j] = Σ_k right[i,k] * beta_mat[j,k]
                let right_beta = fast_ab(right, &beta_mat.t().to_owned());
                // Then element-wise multiply with left and sum across columns.
                let mut result = Array1::zeros(n);
                for i in 0..n {
                    let mut acc = 0.0;
                    for j in 0..pa {
                        acc += left[[i, j]] * right_beta[[i, j]];
                    }
                    result[i] = acc;
                }
                result
            }
        }
    }

    /// Compute `self^T · v` where v is an n-vector.
    /// Returns a (p_a * p_b)-vector.
    fn transpose_mul(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::Dense(m) => m.t().dot(v),
            KroneckerDesign::Factored { left, right } => {
                let n = left.nrows();
                debug_assert_eq!(v.len(), n);
                // result[j, k] = Σ_i v[i] * left[i,j] * right[i,k]
                // = (left^T diag(v) right)[j, k]
                // Compute diag(v) * right first (weight rows of right by v).
                let weighted_right = weight_rows(right, v);
                let result_mat = fast_atb(left, &weighted_right); // pa × pb
                // Flatten to (pa * pb) vector in row-major order.
                Array1::from_iter(result_mat.iter().copied())
            }
        }
    }

    /// Compute `self^T · self`, the Gram matrix.
    /// Returns a (p_a·p_b) × (p_a·p_b) matrix.
    fn gram(&self) -> Array2<f64> {
        match self {
            KroneckerDesign::Dense(m) => fast_atb(m, m),
            KroneckerDesign::Factored { .. } => {
                // For row-wise Kronecker X where row i = l_i ⊗ r_i, the Gram
                // matrix X'X = Σ_i (l_i ⊗ r_i)(l_i ⊗ r_i)' = Σ_i (l_i l_i') ⊗ (r_i r_i').
                // This is a sum of Kronecker products, NOT a single Kronecker product,
                // so it does not factor as (L'L) ⊗ (R'R).  Materialization is therefore
                // the correct approach (short of PCG with operator-form matvec).
                let dense = self.to_dense();
                fast_atb(&dense, &dense)
            }
        }
    }

    /// Compute `self^T · diag(w) · self` (weighted Gram).
    fn weighted_gram(&self, w: &Array1<f64>) -> Array2<f64> {
        match self {
            KroneckerDesign::Dense(m) => {
                let wm = weight_rows(m, w);
                fast_atb(&wm, m)
            }
            KroneckerDesign::Factored { .. } => {
                // Same reasoning as gram(): X'diag(w)X for row-wise Kronecker is
                // Σ_i w_i (l_i ⊗ r_i)(l_i ⊗ r_i)', which is not separable.
                let dense = self.to_dense();
                let wm = weight_rows(&dense, w);
                fast_atb(&wm, &dense)
            }
        }
    }

    /// Compute `self^T · diag(w) · other` where other is a dense matrix with
    /// the same number of rows.
    fn weighted_cross(&self, w: &Array1<f64>, other: &Array2<f64>) -> Array2<f64> {
        match self {
            KroneckerDesign::Dense(m) => {
                let wm = weight_rows(m, w);
                fast_atb(&wm, other)
            }
            KroneckerDesign::Factored { .. } => {
                let dense = self.to_dense();
                let wm = weight_rows(&dense, w);
                fast_atb(&wm, other)
            }
        }
    }
}

impl DesignOperator for KroneckerDesign {
    fn nrows(&self) -> usize {
        Self::nrows(self)
    }

    fn ncols(&self) -> usize {
        Self::ncols(self)
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

    fn row_chunk(&self, rows: std::ops::Range<usize>) -> Array2<f64> {
        match self {
            KroneckerDesign::Dense(m) => m.slice(s![rows, ..]).to_owned(),
            KroneckerDesign::Factored { left, right } => {
                let left_chunk = left.slice(s![rows.clone(), ..]).to_owned();
                let right_chunk = right.slice(s![rows, ..]).to_owned();
                rowwise_kronecker(&left_chunk, &right_chunk)
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        match self {
            KroneckerDesign::Dense(m) => m.clone(),
            KroneckerDesign::Factored { left, right } => rowwise_kronecker(left, right),
        }
    }
}

// ---------------------------------------------------------------------------
// Kronecker-form penalties
// ---------------------------------------------------------------------------

/// A penalty matrix in separable Kronecker form: `S_left ⊗ S_right`.
///
/// Build tensor product penalties in Kronecker-separable form.
///
/// Returns both the materialized dense penalties (for the solver) and the
/// separable Kronecker form (for operator-form penalty application).
fn build_tensor_penalties_kronecker(
    response_penalties: &[Array2<f64>],
    covariate_penalties: &[Array2<f64>],
    p_resp: usize,
    p_cov: usize,
    config: &TransformationNormalConfig,
) -> Result<Vec<PenaltyMatrix>, String> {
    let eye_resp = Array2::<f64>::eye(p_resp);
    let eye_cov = Array2::<f64>::eye(p_cov);
    let mut penalties = Vec::new();

    // Covariate penalties: I_resp ⊗ S_cov_m
    for s_cov in covariate_penalties {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: eye_resp.clone(),
            right: s_cov.clone(),
        });
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

/// Build monotonicity constraints: x_deriv · β ≥ ε at training points plus a
/// boundary grid. Compress collinear constraint rows for efficiency.
fn build_monotonicity_constraints(
    response: &Array1<f64>,
    covariate_design: &Array2<f64>,
    x_deriv: &Array2<f64>,
    response_knots: &Array1<f64>,
    response_degree: usize,
    response_transform: &Array2<f64>,
    config: &TransformationNormalConfig,
) -> Result<LinearInequalityConstraints, String> {
    let n = x_deriv.nrows();
    let p = x_deriv.ncols();
    let eps = config.monotonicity_eps;

    // Start with all training-point derivative rows.
    let mut constraint_rows: Vec<Array1<f64>> = Vec::with_capacity(n + config.derivative_grid_size);
    for i in 0..n {
        let row = x_deriv.row(i).to_owned();
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-12 {
            constraint_rows.push(row);
        }
    }

    // Add boundary grid rows: a fine grid of response values evaluated against
    // a representative subset of covariate basis rows.
    let grid = response_derivative_grid(response, config)?;
    if !grid.is_empty() && covariate_design.nrows() > 0 {
        // Evaluate response derivative basis at grid points using provided knots/transform.
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
                let p_cov = covariate_design.ncols();
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
fn subsample_covariate_rows(design: &Array2<f64>, max_rows: usize) -> Array2<f64> {
    let n = design.nrows();
    if n <= max_rows {
        return design.to_owned();
    }
    let step = n as f64 / max_rows as f64;
    let indices: Vec<usize> = (0..max_rows)
        .map(|i| ((i as f64 * step) as usize).min(n - 1))
        .collect();
    let mut out = Array2::zeros((indices.len(), design.ncols()));
    for (r, &idx) in indices.iter().enumerate() {
        out.row_mut(r).assign(&design.row(idx));
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
    covariate_design: &Array2<f64>,
    config: &TransformationNormalConfig,
) -> Result<LinearInequalityConstraints, String> {
    let n = x_deriv_kron.nrows();
    let p = x_deriv_kron.ncols();
    let eps = config.monotonicity_eps;

    let dense = x_deriv_kron.to_dense();

    // --- Part 1: training-point derivative rows (subsampled if large) ---
    let mut constraint_rows: Vec<Array1<f64>> = Vec::new();

    if n <= MAX_MONOTONICITY_TRAINING_ROWS {
        // Use all training-point rows.
        for i in 0..n {
            let row = dense.row(i).to_owned();
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
            let row = dense.row(idx).to_owned();
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
    covariate_design: &Array2<f64>,
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
    let ctc = fast_atb(covariate_design, covariate_design);
    // Add small ridge for numerical stability.
    let mut ctc_ridge = ctc;
    for i in 0..p_cov {
        ctc_ridge[[i, i]] += 1e-10;
    }

    // Solve via Cholesky (ctc_ridge is SPD).
    use crate::faer_ndarray::FaerCholesky;
    match ctc_ridge.cholesky(faer::Side::Lower) {
        Ok(chol) => {
            let ct_int = covariate_design.t().dot(&target_intercept);
            let coeff_int = chol.solvevec(&ct_int);
            let ct_slope = covariate_design.t().dot(&target_slope);
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
/// - `x_psi`: rowwise_kronecker(resp_val, cov_psi_a) — the design derivative
/// - `s_psi`: the tensor penalty derivative (for the S_cov penalties that move with ψ)
pub fn build_tensor_psi_derivatives(
    family: &TransformationNormalFamily,
    covariate_psi_designs: &[Array2<f64>], // per-axis ∂cov_design/∂ψ_a
    covariate_psi_penalties: &[Vec<(usize, Array2<f64>)>], // per-axis [(cov_penalty_idx, ∂S_cov/∂ψ_a)]
    covariate_psi_second_designs: Option<&[Vec<Array2<f64>>]>, // per-axis-pair ∂²cov_design/∂ψ_a∂ψ_b
    covariate_psi_second_penalties: Option<&[Vec<Vec<(usize, Array2<f64>)>>]>,
) -> Result<Vec<CustomFamilyBlockPsiDerivative>, String> {
    let p_resp = family.response_val_basis.ncols();
    let p_cov = family.p_cov;
    let n_axes = covariate_psi_designs.len();
    let eye_resp = Array2::<f64>::eye(p_resp);

    let mut derivs = Vec::with_capacity(n_axes);
    for a in 0..n_axes {
        // ∂x_val/∂ψ_a = rowwise_kronecker(resp_val, cov_psi_a)
        let x_psi = rowwise_kronecker(&family.response_val_basis, &covariate_psi_designs[a]);

        // ∂S_tensor/∂ψ_a: for each covariate penalty that moves with ψ_a,
        // the tensor penalty I_resp ⊗ S_cov_m has derivative I_resp ⊗ ∂S_cov_m/∂ψ_a.
        let mut s_psi = Array2::<f64>::zeros((p_resp * p_cov, p_resp * p_cov));
        let mut s_psi_components = Vec::new();
        for &(cov_pen_idx, ref ds_cov) in &covariate_psi_penalties[a] {
            let ds_tensor = kronecker_product(&eye_resp, ds_cov);
            s_psi = s_psi + &ds_tensor;
            // The tensor penalty index for covariate penalty m is just m
            // (covariate penalties come first in the tensor penalty list).
            s_psi_components.push((cov_pen_idx, ds_tensor));
        }

        // Second-order design derivatives (optional).
        let x_psi_psi = covariate_psi_second_designs.map(|second| {
            second[a]
                .iter()
                .map(|cov_psi2| rowwise_kronecker(&family.response_val_basis, cov_psi2))
                .collect::<Vec<_>>()
        });

        // Second-order penalty derivatives (optional).
        let s_psi_psi = covariate_psi_second_penalties.map(|second| {
            second[a]
                .iter()
                .map(|cov_pen_pairs| {
                    let mut mat = Array2::<f64>::zeros((p_resp * p_cov, p_resp * p_cov));
                    for &(_, ref ds2) in cov_pen_pairs {
                        mat = mat + &kronecker_product(&eye_resp, ds2);
                    }
                    mat
                })
                .collect::<Vec<_>>()
        });

        let s_psi_psi_components = covariate_psi_second_penalties.map(|second| {
            second[a]
                .iter()
                .map(|cov_pen_pairs| {
                    cov_pen_pairs
                        .iter()
                        .map(|(idx, ds2)| (*idx, kronecker_product(&eye_resp, ds2)))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });

        let mut deriv = CustomFamilyBlockPsiDerivative::new(
            Some(a), // penalty_index: the a-th covariate penalty
            x_psi,
            s_psi,
            Some(s_psi_components),
            x_psi_psi,
            s_psi_psi,
            s_psi_psi_components,
        );
        deriv.implicit_axis = a;
        deriv.implicit_group_id = if n_axes > 1 { Some(0) } else { None };
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

        let cov_dense_for_tensor: Vec<Array2<f64>> = {
            let p_cov = cov_design.design.ncols();
            cov_design
                .penalties
                .iter()
                .map(|bp| bp.to_global(p_cov))
                .collect()
        };
        let cov_dense = cov_design.design.as_dense_cow();
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            resp_val,
            resp_deriv,
            resp_penalties,
            resp_knots.clone(),
            config.response_degree,
            resp_transform,
            &cov_dense,
            &cov_dense_for_tensor,
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
    let boot_dense_for_tensor: Vec<Array2<f64>> = {
        let p_cov = boot_design.design.ncols();
        boot_design
            .penalties
            .iter()
            .map(|bp| bp.to_global(p_cov))
            .collect()
    };
    let boot_dense = boot_design.design.as_dense_cow();
    let probe_family = TransformationNormalFamily::from_prebuilt_response_basis(
        resp_val.clone(),
        resp_deriv.clone(),
        resp_penalties.clone(),
        resp_knots.clone(),
        config.response_degree,
        resp_transform.clone(),
        &boot_dense,
        &boot_dense_for_tensor,
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
            let gp: Vec<Array2<f64>> = {
                let p_cov = cov_design.design.ncols();
                cov_design
                    .penalties
                    .iter()
                    .map(|bp| bp.to_global(p_cov))
                    .collect()
            };
            let cov_dense = cov_design.design.as_dense_cow();
            TransformationNormalFamily::from_prebuilt_response_basis(
                rv.clone(),
                rd.clone(),
                rp.clone(),
                rk.clone(),
                rdeg,
                rt.clone(),
                &cov_dense,
                &gp,
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

            // Extract covariate-space arrays for build_tensor_psi_derivatives.
            let cov_psi_designs: Vec<Array2<f64>> =
                cov_psi_derivs.iter().map(|d| d.x_psi.clone()).collect();
            let cov_psi_penalties: Vec<Vec<(usize, Array2<f64>)>> = cov_psi_derivs
                .iter()
                .map(|d| d.s_psi_components.clone().unwrap_or_default())
                .collect();

            // Second-order design derivatives.
            let has_second_designs = cov_psi_derivs.iter().all(|d| d.x_psi_psi.is_some());
            let cov_psi_second_designs: Option<Vec<Vec<Array2<f64>>>> = if has_second_designs {
                Some(
                    cov_psi_derivs
                        .iter()
                        .map(|d| d.x_psi_psi.clone().unwrap_or_default())
                        .collect(),
                )
            } else {
                None
            };

            // Second-order penalty derivatives.
            let has_second_penalties = cov_psi_derivs
                .iter()
                .all(|d| d.s_psi_psi_components.is_some());
            let cov_psi_second_penalties: Option<Vec<Vec<Vec<(usize, Array2<f64>)>>>> =
                if has_second_penalties {
                    Some(
                        cov_psi_derivs
                            .iter()
                            .map(|d| d.s_psi_psi_components.clone().unwrap_or_default())
                            .collect(),
                    )
                } else {
                    None
                };

            let tensor_derivs = build_tensor_psi_derivatives(
                &family,
                &cov_psi_designs,
                &cov_psi_penalties,
                cov_psi_second_designs.as_deref(),
                cov_psi_second_penalties.as_deref(),
            )?;

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
