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
    CustomFamilyJointHyperResult, CustomFamilyWarmStart, ExactNewtonJointPsiTerms,
    ExactOuterDerivativeOrder, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    evaluate_custom_family_joint_hyper, fit_custom_family,
};
use crate::families::gamlss::initializewiggle_knots_from_seed;
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::smooth::{
    SpatialLengthScaleOptimizationOptions, TermCollectionDesign, TermCollectionSpec,
    build_term_collection_design, freeze_spatial_length_scale_terms_from_design,
    spatial_length_scale_term_indices,
};
use crate::solver::estimate::UnifiedFitResult;
use ndarray::{Array1, Array2, s};
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
pub struct TransformationNormalFamily {
    // --- Tensor product design matrices ---
    /// Value design: n × p_total. The block's official design matrix.
    x_val: Arc<Array2<f64>>,
    /// Derivative design: n × p_total. Family-owned, not exposed to the solver.
    x_deriv: Array2<f64>,

    // --- Response-direction basis (fixed, does not depend on κ) ---
    /// Response value basis: n × p_resp. Columns: [1, y, dev_1(y), ..., dev_k(y)].
    response_val_basis: Array2<f64>,
    /// Response derivative basis: n × p_resp. Columns: [0, 1, dev'_1(y), ..., dev'_k(y)].
    response_deriv_basis: Array2<f64>,
    /// Response roughness penalty: p_resp × p_resp. Zeros for [1,y], difference
    /// penalty on the deviation part.
    response_penalties: Vec<Array2<f64>>,

    // --- Covariate side (rebuilt on κ change) ---
    /// Number of covariate basis columns.
    p_cov: usize,

    // --- Tensor penalties ---
    tensor_penalties: Vec<Array2<f64>>,

    // --- Monotonicity constraints ---
    monotonicity_constraints: LinearInequalityConstraints,

    // --- Initial values ---
    initial_beta: Array1<f64>,
    initial_log_lambdas: Array1<f64>,

    // --- Config ---
    epsilon: f64,
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
        let (resp_val, resp_deriv, resp_penalties) =
            build_response_basis(response, config)?;
        let p_resp = resp_val.ncols();

        // ----- 2. Row-wise Kronecker product -----
        let x_val = rowwise_kronecker(&resp_val, covariate_design);
        let x_deriv = rowwise_kronecker(&resp_deriv, covariate_design);
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val.ncols(), p_total);
        debug_assert_eq!(x_deriv.ncols(), p_total);

        // ----- 3. Tensor penalties -----
        let tensor_penalties =
            build_tensor_penalties(&resp_penalties, covariate_penalties, p_resp, p_cov, config)?;

        // ----- 4. Monotonicity constraints -----
        let monotonicity_constraints = build_monotonicity_constraints(
            response,
            covariate_design,
            &x_deriv,
            config,
        )?;

        // ----- 5. Warm start -----
        let initial_beta = compute_warm_start(
            response,
            covariate_design,
            p_resp,
            p_cov,
            warm_start,
        )?;

        // ----- 6. Initial log-lambdas (one per penalty, start at 0.0) -----
        let initial_log_lambdas = Array1::zeros(tensor_penalties.len());

        Ok(Self {
            x_val: Arc::new(x_val),
            x_deriv,
            response_val_basis: resp_val,
            response_deriv_basis: resp_deriv,
            response_penalties: resp_penalties,
            p_cov,
            tensor_penalties,
            monotonicity_constraints,
            initial_beta,
            initial_log_lambdas,
            epsilon: config.monotonicity_eps,
            block_name: "transformation".to_string(),
        })
    }

    /// Return the `ParameterBlockSpec` for this family (single block).
    pub fn block_spec(&self) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: self.block_name.clone(),
            design: DesignMatrix::Dense(self.x_val.clone()),
            offset: Array1::zeros(self.x_val.nrows()),
            penalties: self.tensor_penalties.clone(),
            initial_log_lambdas: self.initial_log_lambdas.clone(),
            initial_beta: Some(self.initial_beta.clone()),
        }
    }

    /// Total number of coefficients.
    pub fn p_total(&self) -> usize {
        self.x_val.ncols()
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.x_val.nrows()
    }

    /// Evaluate the fitted transformation h(y|x) at new data points.
    ///
    /// Returns z = h(y|x) for each observation.
    pub fn predict_transform(
        &self,
        response: &Array1<f64>,
        covariate_design: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = response.len();
        if covariate_design.nrows() != n {
            return Err("response and covariate design row counts differ".to_string());
        }
        if beta.len() != self.p_total() {
            return Err(format!(
                "beta length {} != p_total {}",
                beta.len(),
                self.p_total()
            ));
        }
        // Evaluate response basis at new response values.
        let resp_val = evaluate_response_value_basis_at(
            response,
            &self.response_val_basis,
            self.response_deriv_basis.ncols(), // p_resp
        )?;
        let x_val_new = rowwise_kronecker(&resp_val, covariate_design);
        Ok(x_val_new.dot(beta))
    }

    /// Rebuild the family at a new covariate design (e.g., after κ change).
    ///
    /// The response-direction basis is fixed; only the covariate side and the
    /// resulting tensor products, penalties, and constraints are rebuilt.
    pub fn rebuild_at_covariate_design(
        &mut self,
        covariate_design: &Array2<f64>,
        covariate_penalties: &[Array2<f64>],
        config: &TransformationNormalConfig,
    ) -> Result<(), String> {
        let n = self.response_val_basis.nrows();
        if covariate_design.nrows() != n {
            return Err(format!(
                "new covariate design has {} rows, expected {n}",
                covariate_design.nrows()
            ));
        }
        let p_cov = covariate_design.ncols();
        let p_resp = self.response_val_basis.ncols();

        self.x_val = Arc::new(rowwise_kronecker(&self.response_val_basis, covariate_design));
        self.x_deriv = rowwise_kronecker(&self.response_deriv_basis, covariate_design);
        self.p_cov = p_cov;
        self.tensor_penalties = build_tensor_penalties(
            &self.response_penalties,
            covariate_penalties,
            p_resp,
            p_cov,
            config,
        )?;

        // Rebuild constraints with new x_deriv.
        // We need the original response values to rebuild the boundary grid,
        // but we can reuse the current x_deriv rows plus recompute the grid.
        // For simplicity, rebuild from the current x_deriv rows only.
        self.monotonicity_constraints = LinearInequalityConstraints {
            a: self.x_deriv.clone(),
            b: Array1::from_elem(self.x_deriv.nrows(), self.epsilon),
        };

        self.initial_log_lambdas = Array1::zeros(self.tensor_penalties.len());
        Ok(())
    }

    // --- Internal helpers for evaluate ---

    /// Compute h and h' from the current coefficients.
    fn compute_h_and_h_prime(&self, beta: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h = self.x_val.dot(beta);
        let h_prime = self.x_deriv.dot(beta);
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
    fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
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
            let neg_xvt_h = self.x_val.t().dot(&h).mapv(|v| -v);
            let xdt_inv = self.x_deriv.t().dot(&inv_h_prime);
            neg_xvt_h + &xdt_inv
        };

        // Hessian of negative log-likelihood: -∇²ℓ = X_val^T X_val + X_deriv^T diag(1/h'²) X_deriv
        let inv_h_prime_sq = h_prime.mapv(|v| 1.0 / (v * v));
        let hessian = {
            let xtx_val = fast_atb(self.x_val.as_ref(), self.x_val.as_ref());
            // X_deriv^T diag(w) X_deriv where w = 1/h'^2
            let weighted_xd = weight_rows(&self.x_deriv, &inv_h_prime_sq);
            let xtx_deriv = fast_atb(&weighted_xd, &self.x_deriv);
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

    fn log_likelihood_only(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
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
        _specs: &[ParameterBlockSpec],
        _options: &BlockwiseFitOptions,
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
        _block_states: &[ParameterBlockState],
        block_index: usize,
        _spec: &ParameterBlockSpec,
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
        let h_prime = self.x_deriv.dot(beta);
        let d_h_prime = self.x_deriv.dot(d_beta);

        // ∂H/∂β · d_beta = -2 X_deriv^T diag((X_deriv · d_beta) / h'^3) X_deriv
        let n = h_prime.len();
        let mut weight = Array1::zeros(n);
        for i in 0..n {
            weight[i] = -2.0 * d_h_prime[i] / (h_prime[i] * h_prime[i] * h_prime[i]);
        }
        let weighted_xd = weight_rows(&self.x_deriv, &weight);
        let dd = fast_atb(&weighted_xd, &self.x_deriv);
        Ok(Some(dd))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Single block: joint Hessian = block Hessian.
        let beta = &block_states[0].beta;
        let h_prime = self.x_deriv.dot(beta);
        let inv_h_prime_sq = h_prime.mapv(|v| 1.0 / (v * v));
        let xtx_val = fast_atb(self.x_val.as_ref(), self.x_val.as_ref());
        let weighted_xd = weight_rows(&self.x_deriv, &inv_h_prime_sq);
        let xtx_deriv = fast_atb(&weighted_xd, &self.x_deriv);
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
        _specs: &[ParameterBlockSpec],
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
        let x_deriv_psi = rowwise_kronecker(&self.response_deriv_basis, &cov_psi);

        // v_val = x_val_psi · β (n-vector): ∂h/∂ψ
        let v_val = x_val_psi.dot(beta);
        // v_deriv = x_deriv_psi · β (n-vector): ∂h'/∂ψ
        let v_deriv = x_deriv_psi.dot(beta);

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
            let term2 = self.x_val.t().dot(&v_val);
            let term3 = x_deriv_psi.t().dot(&inv_h_prime).mapv(|v| -v);
            let w_deriv = &v_deriv * &inv_h_prime_sq;
            let term4 = self.x_deriv.t().dot(&w_deriv);
            term1 + &term2 + &term3 + &term4
        };

        // --- hessian_psi: ∂H_L/∂ψ at fixed β (p × p) ---
        // = 2·sym(x_val^T x_val_psi)
        //   + 2·sym(x_deriv^T diag(1/h'^2) x_deriv_psi)
        //   - 2·x_deriv^T diag(v_deriv / h'^3) x_deriv
        let hessian_psi = {
            // sym(A^T B) = (A^T B + B^T A) / 2; we compute A^T B and symmetrize.
            let xvt_xvp = fast_atb(self.x_val.as_ref(), x_val_psi);
            let sym_val = &xvt_xvp + &xvt_xvp.t();

            let weighted_xd = weight_rows(&self.x_deriv, &inv_h_prime_sq);
            let xdt_xdp = fast_atb(&weighted_xd, &x_deriv_psi);
            let sym_deriv = &xdt_xdp + &xdt_xdp.t();

            let mut w_cubic = Array1::zeros(n);
            for i in 0..n {
                w_cubic[i] = -2.0 * v_deriv[i] / (h_prime[i] * h_prime[i] * h_prime[i]);
            }
            let weighted_xd2 = weight_rows(&self.x_deriv, &w_cubic);
            let cubic_term = fast_atb(&weighted_xd2, &self.x_deriv);

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
/// Returns (value_basis, derivative_basis, penalties).
fn build_response_basis(
    response: &Array1<f64>,
    config: &TransformationNormalConfig,
) -> Result<(Array2<f64>, Array2<f64>, Vec<Array2<f64>>), String> {
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
    let dev_val = raw_val.dot(&transform);   // n × dev_dim
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
    resp_val
        .slice_mut(s![.., 2..])
        .assign(&dev_val);
    resp_deriv
        .slice_mut(s![.., 2..])
        .assign(&dev_deriv);

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
        full_pen
            .slice_mut(s![2.., 2..])
            .assign(&dev_pen);
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

    Ok((resp_val, resp_deriv, resp_penalties))
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

/// Build tensor product penalties: one per covariate penalty (I_resp ⊗ S_cov_m),
/// one per response penalty (S_resp_m ⊗ I_cov), plus optional double penalty.
fn build_tensor_penalties(
    response_penalties: &[Array2<f64>],
    covariate_penalties: &[Array2<f64>],
    p_resp: usize,
    p_cov: usize,
    config: &TransformationNormalConfig,
) -> Result<Vec<Array2<f64>>, String> {
    let eye_resp = Array2::<f64>::eye(p_resp);
    let eye_cov = Array2::<f64>::eye(p_cov);
    let mut penalties = Vec::new();

    // Covariate penalties: I_resp ⊗ S_cov_m
    for s_cov in covariate_penalties {
        penalties.push(kronecker_product(&eye_resp, s_cov));
    }

    // Response penalties: S_resp_m ⊗ I_cov
    for s_resp in response_penalties {
        penalties.push(kronecker_product(s_resp, &eye_cov));
    }

    // Double penalty: global ridge
    if config.double_penalty {
        penalties.push(Array2::<f64>::eye(p_resp * p_cov));
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
        // Evaluate response derivative basis at grid points.
        let knots = initializewiggle_knots_from_seed(
            response.view(),
            config.response_degree,
            config.response_num_internal_knots,
        )?;
        let transform =
            response_deviation_transform(&knots, config.response_degree, response)?;
        let (raw_d1, _) = create_basis::<Dense>(
            grid.view(),
            KnotSource::Provided(knots.view()),
            config.response_degree,
            BasisOptions::first_derivative(),
        )
        .map_err(|e| e.to_string())?;
        let dev_d1 = raw_d1.as_ref().dot(&transform);
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
    let indices: Vec<usize> = (0..max_rows).map(|i| ((i as f64 * step) as usize).min(n - 1)).collect();
    let mut out = Array2::zeros((indices.len(), design.ncols()));
    for (r, &idx) in indices.iter().enumerate() {
        out.row_mut(r).assign(&design.row(idx));
    }
    out
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
            let var = response.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>()
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
            let var = response.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>()
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

/// Evaluate the response value basis at new response values.
///
/// This is a simplified version for prediction: it rebuilds [1, y, dev(y)] at new y.
/// Note: the knots and transform are derived from the training data and stored in the
/// family's response_val_basis dimensions; for full prediction support, the family
/// should store knots and transform explicitly. This placeholder returns an error
/// indicating that full prediction requires stored knots.
fn evaluate_response_value_basis_at(
    _response: &Array1<f64>,
    _training_basis: &Array2<f64>,
    _p_resp: usize,
) -> Result<Array2<f64>, String> {
    Err(
        "predict_transform requires stored knots and transform; \
         use TransformationNormalPredictable for out-of-sample prediction"
            .to_string(),
    )
}

// ---------------------------------------------------------------------------
// Prediction support
// ---------------------------------------------------------------------------

/// Extended family that stores knots and transform for out-of-sample prediction.
pub struct TransformationNormalPredictable {
    /// The fitted family.
    pub family: TransformationNormalFamily,
    /// B-spline knots for the response direction.
    pub response_knots: Array1<f64>,
    /// B-spline degree for the response direction.
    pub response_degree: usize,
    /// Nullspace transform for the deviation basis.
    pub response_transform: Array2<f64>,
}

impl TransformationNormalPredictable {
    /// Build a predictable family (stores knots and transform alongside the family).
    pub fn new(
        response: &Array1<f64>,
        covariate_design: &Array2<f64>,
        covariate_penalties: &[Array2<f64>],
        config: &TransformationNormalConfig,
        warm_start: Option<&TransformationWarmStart>,
    ) -> Result<Self, String> {
        let knots = initializewiggle_knots_from_seed(
            response.view(),
            config.response_degree,
            config.response_num_internal_knots,
        )?;
        let transform =
            response_deviation_transform(&knots, config.response_degree, response)?;
        let family = TransformationNormalFamily::new(
            response,
            covariate_design,
            covariate_penalties,
            config,
            warm_start,
        )?;
        Ok(Self {
            family,
            response_knots: knots,
            response_degree: config.response_degree,
            response_transform: transform,
        })
    }

    /// Evaluate h(y|x) = z at new data points.
    pub fn predict_transform(
        &self,
        response: &Array1<f64>,
        covariate_design: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = response.len();
        if covariate_design.nrows() != n {
            return Err("response and covariate design row counts differ".to_string());
        }
        let p_total = beta.len();

        // Evaluate response basis at new values.
        let resp_val = self.evaluate_response_value_basis(response)?;
        let x_val_new = rowwise_kronecker(&resp_val, covariate_design);
        if x_val_new.ncols() != p_total {
            return Err(format!(
                "prediction design has {} columns but beta has {} entries",
                x_val_new.ncols(),
                p_total
            ));
        }
        Ok(x_val_new.dot(beta))
    }

    /// Evaluate the derivative h'(y|x) at new data points (for diagnostics).
    pub fn predict_jacobian(
        &self,
        response: &Array1<f64>,
        covariate_design: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = response.len();
        if covariate_design.nrows() != n {
            return Err("response and covariate design row counts differ".to_string());
        }
        let resp_deriv = self.evaluate_response_deriv_basis(response)?;
        let x_deriv_new = rowwise_kronecker(&resp_deriv, covariate_design);
        Ok(x_deriv_new.dot(beta))
    }

    fn evaluate_response_value_basis(
        &self,
        response: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let n = response.len();
        let (raw_val, _) = create_basis::<Dense>(
            response.view(),
            KnotSource::Provided(self.response_knots.view()),
            self.response_degree,
            BasisOptions::value(),
        )
        .map_err(|e| e.to_string())?;
        let dev_val = raw_val.as_ref().dot(&self.response_transform);
        let dev_dim = dev_val.ncols();
        let p_resp = 2 + dev_dim;
        let mut basis = Array2::<f64>::zeros((n, p_resp));
        basis.column_mut(0).fill(1.0);
        basis.column_mut(1).assign(&response.view());
        basis.slice_mut(s![.., 2..]).assign(&dev_val);
        Ok(basis)
    }

    fn evaluate_response_deriv_basis(
        &self,
        response: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let n = response.len();
        let (raw_d1, _) = create_basis::<Dense>(
            response.view(),
            KnotSource::Provided(self.response_knots.view()),
            self.response_degree,
            BasisOptions::first_derivative(),
        )
        .map_err(|e| e.to_string())?;
        let dev_d1 = raw_d1.as_ref().dot(&self.response_transform);
        let dev_dim = dev_d1.ncols();
        let p_resp = 2 + dev_dim;
        let mut basis = Array2::<f64>::zeros((n, p_resp));
        // Column 0: d(1)/dy = 0
        // Column 1: d(y)/dy = 1
        basis.column_mut(1).fill(1.0);
        basis.slice_mut(s![.., 2..]).assign(&dev_d1);
        Ok(basis)
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
/// - `x_psi`: rowwise_kronecker(resp_val, cov_psi_a) — the design derivative
/// - `s_psi`: the tensor penalty derivative (for the S_cov penalties that move with ψ)
pub fn build_tensor_psi_derivatives(
    family: &TransformationNormalFamily,
    covariate_psi_designs: &[Array2<f64>],    // per-axis ∂cov_design/∂ψ_a
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
// Fitting orchestrator
// ---------------------------------------------------------------------------

/// Result of fitting a transformation model.
pub struct TransformationNormalFitResult {
    /// The fitted family (with final covariate design).
    pub family: TransformationNormalFamily,
    /// The inner fit result (coefficients, smoothing parameters, covariance).
    pub fit: UnifiedFitResult,
    /// Final resolved covariate spec (with optimized κ if applicable).
    pub covariate_spec_resolved: TermCollectionSpec,
    /// Final covariate design.
    pub covariate_design: TermCollectionDesign,
    /// Response-direction knots (for out-of-sample prediction).
    pub response_knots: Array1<f64>,
    /// Response-direction degree.
    pub response_degree: usize,
    /// Response-direction deviation transform.
    pub response_transform: Array2<f64>,
}

/// Fit a conditional transformation model with full REML, including joint
/// (λ, κ) optimization when the covariate basis has spatial length scales.
///
/// This is the unified entry point. It:
/// 1. Builds the covariate design from the spec
/// 2. Builds the TransformationNormalFamily (response basis + tensor product)
/// 3. If κ optimization is enabled and the covariate spec has spatial terms:
///    runs the joint [ρ, ψ] outer loop via `evaluate_custom_family_joint_hyper`
/// 4. Otherwise: calls `fit_custom_family` for λ-only optimization
///
/// # Arguments
///
/// * `response` - The response variable y.
/// * `covariate_data` - Raw covariate data (n × d) for building the covariate basis.
/// * `covariate_spec` - Covariate-side term collection specification.
/// * `config` - Response-direction basis configuration.
/// * `options` - Blockwise fitting options (inner tolerance, outer iterations, etc.).
/// * `kappa_options` - Spatial length-scale optimization options.
/// * `warm_start` - Optional location/scale from a prior normalizer.
pub fn fit_transformation_normal(
    response: &Array1<f64>,
    covariate_data: ndarray::ArrayView2<'_, f64>,
    covariate_spec: &TermCollectionSpec,
    config: &TransformationNormalConfig,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<TransformationNormalFitResult, String> {
    // Build initial covariate design.
    let covariate_design = build_term_collection_design(covariate_data, covariate_spec)
        .map_err(|e| format!("failed to build covariate design: {e}"))?;
    let resolved_spec =
        freeze_spatial_length_scale_terms_from_design(covariate_spec, &covariate_design)
            .map_err(|e| format!("failed to freeze covariate spatial basis: {e}"))?;

    // Build response-direction knots and transform (stored for prediction).
    let response_knots = initializewiggle_knots_from_seed(
        response.view(),
        config.response_degree,
        config.response_num_internal_knots,
    )?;
    let response_transform =
        response_deviation_transform(&response_knots, config.response_degree, response)?;

    // Extract covariate design matrix and penalties.
    let cov_mat = covariate_design.design.to_dense();
    let cov_penalties: Vec<Array2<f64>> = covariate_design.penalties.clone();

    // Build the family.
    let family = TransformationNormalFamily::new(
        response, &cov_mat, &cov_penalties, config, warm_start,
    )?;

    // Check whether κ optimization is needed.
    let spatial_terms = spatial_length_scale_term_indices(&resolved_spec);
    let has_kappa = kappa_options.enabled && !spatial_terms.is_empty();

    if !has_kappa {
        // --- No κ optimization: use fit_custom_family directly ---
        let spec = family.block_spec();
        let fit = fit_custom_family(&family, &[spec], options)
            .map_err(|e| format!("transformation normal fit failed: {e}"))?;
        return Ok(TransformationNormalFitResult {
            family,
            fit,
            covariate_spec_resolved: resolved_spec,
            covariate_design,
            response_knots,
            response_degree: config.response_degree,
            response_transform,
        });
    }

    // --- Joint (λ, κ) optimization ---
    // Use the same ClosureObjective + run_outer pattern as the two-block spatial
    // optimizer, but adapted for our single-block tensor-product family.

    use crate::families::custom_family::build_block_spatial_psi_derivatives;
    use crate::solver::outer_strategy::{
        ClosureObjective, Derivative, EfsEval, FallbackPolicy, HessianResult, OuterCapability,
        OuterConfig, OuterEval,
    };

    // Count psi dimensions (one per aniso axis per spatial term).
    let psi_dim: usize = spatial_terms.iter().map(|&ti| {
        resolved_spec.smooth_terms.get(ti)
            .and_then(|t| t.spatial_aniso_log_scales.as_ref())
            .map(|s| s.len())
            .unwrap_or(1)
    }).sum();

    // Build initial theta = [rho, psi].
    let spec0 = family.block_spec();
    let rho0 = spec0.initial_log_lambdas.clone();
    let rho_dim = rho0.len();

    let mut psi0 = Vec::with_capacity(psi_dim);
    for &ti in &spatial_terms {
        if let Some(term) = resolved_spec.smooth_terms.get(ti) {
            if let Some(scales) = &term.spatial_aniso_log_scales {
                psi0.extend(scales.iter().copied());
            } else if let Some(ls) = term.length_scale {
                psi0.push((1.0 / ls).ln());
            } else {
                psi0.push(0.0);
            }
        }
    }

    let theta_dim = rho_dim + psi_dim;
    let mut theta0 = Array1::zeros(theta_dim);
    theta0.slice_mut(s![..rho_dim]).assign(&rho0);
    for (i, &v) in psi0.iter().enumerate() {
        theta0[rho_dim + i] = v;
    }

    // Bounds: rho is unbounded (large range), psi is bounded by kappa_options.
    let log_kappa_lower = (1.0 / kappa_options.max_length_scale).ln();
    let log_kappa_upper = (1.0 / kappa_options.min_length_scale).ln();
    let mut lower = Array1::from_elem(theta_dim, -12.0);
    let mut upper = Array1::from_elem(theta_dim, 12.0);
    for i in rho_dim..theta_dim {
        lower[i] = log_kappa_lower;
        upper[i] = log_kappa_upper;
    }

    let outer_config = OuterConfig {
        tolerance: kappa_options.rel_tol.max(1e-6),
        max_iter: kappa_options.max_outer_iter.max(1),
        fd_step: 1e-4,
        bounds: Some((lower, upper)),
        seed_config: crate::seeding::SeedConfig {
            max_seeds: 1,
            screening_budget: 1,
            num_auxiliary_trailing: psi_dim,
            ..Default::default()
        },
        rho_bound: 12.0,
        heuristic_lambdas: Some(theta0.as_slice().unwrap().to_vec()),
        initial_rho: Some(theta0.clone()),
        fallback_policy: FallbackPolicy::Automatic,
    };

    // Mutable state for the outer loop.
    struct OuterState {
        current_spec: TermCollectionSpec,
        current_design: TermCollectionDesign,
        current_psi: Vec<f64>,
        warm: Option<CustomFamilyWarmStart>,
    }

    let mut state = OuterState {
        current_spec: resolved_spec.clone(),
        current_design: covariate_design.clone(),
        current_psi: psi0.clone(),
        warm: None,
    };

    // Helper: rebuild covariate design at new psi values.
    let rebuild_at_psi = |psi_vals: &[f64],
                          base_spec: &TermCollectionSpec,
                          spatial_terms: &[usize],
                          data: ndarray::ArrayView2<'_, f64>|
     -> Result<(TermCollectionSpec, TermCollectionDesign), String> {
        let mut spec = base_spec.clone();
        let mut offset = 0;
        for &ti in spatial_terms {
            if let Some(term) = spec.smooth_terms.get_mut(ti) {
                if let Some(scales) = term.spatial_aniso_log_scales.as_mut() {
                    let d = scales.len();
                    for j in 0..d {
                        scales[j] = psi_vals[offset + j];
                    }
                    offset += d;
                } else {
                    let kappa = psi_vals[offset].exp();
                    term.length_scale = Some(1.0 / kappa);
                    offset += 1;
                }
            }
        }
        let design = build_term_collection_design(data, &spec)
            .map_err(|e| format!("failed to rebuild covariate design at new κ: {e}"))?;
        let frozen = freeze_spatial_length_scale_terms_from_design(&spec, &design)
            .map_err(|e| format!("failed to freeze rebuilt spec: {e}"))?;
        Ok((frozen, design))
    };

    // References for closures.
    let spatial_terms_ref = &spatial_terms;
    let covariate_spec_ref = &resolved_spec;
    let config_ref = config;
    let response_ref = response;
    let warm_start_ref = warm_start;

    let mut obj = ClosureObjective {
        state: &mut state,
        cap: OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable, // Start conservative; upgrade if small enough.
            n_params: theta_dim,
            all_penalty_like: false,
            has_psi_coords: true,
            fixed_point_available: false,
            barrier_config: None,
        },
        cost_fn: |ctx: &mut &mut OuterState, theta: &Array1<f64>| {
            let psi_vals: Vec<f64> = theta.slice(s![rho_dim..]).to_vec();
            if psi_vals != ctx.current_psi {
                match rebuild_at_psi(
                    &psi_vals, covariate_spec_ref, spatial_terms_ref, covariate_data,
                ) {
                    Ok((spec, design)) => {
                        ctx.current_spec = spec;
                        ctx.current_design = design;
                        ctx.current_psi = psi_vals;
                    }
                    Err(_) => return Ok(f64::INFINITY),
                }
            }
            let cov_mat = ctx.current_design.design.to_dense();
            let cov_pens: Vec<Array2<f64>> = ctx.current_design.penalties.clone();
            match TransformationNormalFamily::new(
                response_ref, &cov_mat, &cov_pens, config_ref, warm_start_ref,
            ) {
                Ok(fam) => {
                    let sp = fam.block_spec();
                    match fit_custom_family(&fam, &[sp], options) {
                        Ok(fit) => Ok(fit.log_likelihood),
                        Err(_) => Ok(f64::INFINITY),
                    }
                }
                Err(_) => Ok(f64::INFINITY),
            }
        },
        eval_fn: |ctx: &mut &mut OuterState, theta: &Array1<f64>| {
            use crate::solver::estimate::EstimationError;
            let map_err = |msg: String| EstimationError::RemlOptimizationFailed(msg);

            let rho = theta.slice(s![..rho_dim]).to_owned();
            let psi_vals: Vec<f64> = theta.slice(s![rho_dim..]).to_vec();
            if psi_vals != ctx.current_psi {
                let (spec, design) = rebuild_at_psi(
                    &psi_vals, covariate_spec_ref, spatial_terms_ref, covariate_data,
                ).map_err(map_err)?;
                ctx.current_spec = spec;
                ctx.current_design = design;
                ctx.current_psi = psi_vals;
            }
            let cov_mat = ctx.current_design.design.to_dense();
            let cov_pens: Vec<Array2<f64>> = ctx.current_design.penalties.clone();
            let family = TransformationNormalFamily::new(
                response_ref, &cov_mat, &cov_pens, config_ref, warm_start_ref,
            ).map_err(|e| map_err(e))?;
            let spec = family.block_spec();

            // Build psi derivatives for the covariate basis.
            let psi_derivs = build_block_spatial_psi_derivatives(
                covariate_data, &ctx.current_spec, &ctx.current_design,
            ).map_err(|e| map_err(e))?;

            // Lift to tensor product space.
            let derivative_blocks = match psi_derivs {
                Some(cov_psi) => {
                    let cov_psi_designs: Vec<Array2<f64>> =
                        cov_psi.iter().map(|d| d.x_psi.clone()).collect();
                    let cov_psi_penalties: Vec<Vec<(usize, Array2<f64>)>> =
                        cov_psi.iter().map(|d| {
                            d.s_psi_components.clone().unwrap_or_default()
                        }).collect();
                    let tensor_psi = build_tensor_psi_derivatives(
                        &family, &cov_psi_designs, &cov_psi_penalties, None, None,
                    ).map_err(|e| map_err(e))?;
                    vec![tensor_psi]
                }
                None => vec![Vec::new()],
            };

            let eval = evaluate_custom_family_joint_hyper(
                &family,
                &[spec],
                options,
                &rho,
                &derivative_blocks,
                ctx.warm.as_ref(),
                false, // gradient-only for now
            ).map_err(|e| map_err(format!("{e}")))?;

            ctx.warm = Some(eval.warm_start);

            if !eval.objective.is_finite() {
                return Ok(OuterEval::infeasible(theta_dim));
            }
            if eval.gradient.iter().any(|v| !v.is_finite()) {
                return Err(map_err(
                    "transformation normal joint gradient non-finite".to_string(),
                ));
            }
            Ok(OuterEval {
                cost: eval.objective,
                gradient: eval.gradient,
                hessian: match eval.outer_hessian {
                    Some(h) if h.iter().all(|v| v.is_finite()) => HessianResult::Analytic(h),
                    _ => HessianResult::Unavailable,
                },
            })
        },
        reset_fn: None::<fn(&mut &mut OuterState)>,
        efs_fn: None::<
            fn(&mut &mut OuterState, &Array1<f64>) -> Result<EfsEval, crate::solver::estimate::EstimationError>,
        >,
    };

    let result = crate::solver::outer_strategy::run_outer(
        &mut obj, &outer_config, "transformation-normal joint spatial",
    ).map_err(|e| format!("outer optimization failed: {e}"))?;

    // Final fit at the optimized theta.
    let theta_star = result.rho;
    let psi_star: Vec<f64> = theta_star.slice(s![rho_dim..]).to_vec();
    let (final_spec, final_design) =
        rebuild_at_psi(&psi_star, covariate_spec_ref, spatial_terms_ref, covariate_data)?;
    let cov_mat = final_design.design.to_dense();
    let cov_pens: Vec<Array2<f64>> = final_design.penalties.clone();
    let final_family = TransformationNormalFamily::new(
        response, &cov_mat, &cov_pens, config, warm_start,
    )?;
    let final_spec_block = final_family.block_spec();
    let final_fit = fit_custom_family(&final_family, &[final_spec_block], options)
        .map_err(|e| format!("final fit failed: {e}"))?;

    Ok(TransformationNormalFitResult {
        family: final_family,
        fit: final_fit,
        covariate_spec_resolved: final_spec,
        covariate_design: final_design,
        response_knots,
        response_degree: config.response_degree,
        response_transform,
    })
}
