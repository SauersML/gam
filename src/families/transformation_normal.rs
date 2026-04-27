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
//! exactly. Monotonicity is enforced by explicit derivative lower-bound
//! constraints on a response grid, plus the natural `log(h')` barrier in the
//! likelihood and a fraction-to-boundary line search.
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
use crate::faer_ndarray::{
    default_rrqr_rank_alpha, fast_ab, fast_ab_into, fast_atb, rrqr_nullspace_basis,
};
use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyPsiDerivativeOperator, CustomFamilyWarmStart,
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactOuterDerivativeOrder,
    FamilyEvaluation, MaterializablePsiDerivativeOperator, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix, build_block_spatial_psi_derivatives, custom_family_outer_derivatives,
    evaluate_custom_family_joint_hyper, evaluate_custom_family_joint_hyper_efs, fit_custom_family,
};
use crate::families::gamlss::{
    initializewiggle_knots_from_seed, solve_penalizedweighted_projection,
};
use crate::matrix::{
    DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator, SymmetricMatrix,
};
use crate::pirls::LinearInequalityConstraints;
use crate::resource::{MatrixMaterializationError, ResourcePolicy};
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, get_spatial_aniso_log_scales, get_spatial_length_scale,
    optimize_spatial_length_scale_exact_joint, spatial_length_scale_term_indices,
    sync_aniso_contrasts_from_metadata,
};
use crate::solver::estimate::UnifiedFitResult;
use crate::solver::estimate::reml::unified::HyperOperator;
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, s};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

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

/// Baseline cap for the tensor-product width used by the transformation-normal
/// response basis. Small datasets should stay compact because the fit
/// repeatedly factorizes dense penalized Hessians.
const BASE_TRANSFORMATION_TENSOR_WIDTH: usize = 160;
/// Large samples can support a richer response basis without the aggressive
/// underfitting forced by the small-sample cap above. This upper cap keeps the
/// tensor width bounded even when the covariate side is narrow.
const LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH: usize = 320;
const STANDARD_NORMAL_LOG_ABS_MEAN: f64 = -0.635_181_422_730_739_1;
const TRANSFORMATION_MONOTONICITY_EPS: f64 = 1.0e-8;
const TRANSFORMATION_TAIL_GUARD_FRACTION: f64 = 0.25;
const TRANSFORMATION_RESPONSE_GRID_MAX_QUANTILES: usize = 129;
const TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS: usize = 4;

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
    /// Derivative design operator on the monotonicity grid: the virtual row
    /// space is the Cartesian product of all training covariate rows with a
    /// bounded response grid (knots, support quantiles, interval guard points,
    /// tail guards). Stored as the `KroneckerDesign::Kronecker` variant —
    /// the (n_grid × p_resp) response-direction factor and the unmodified
    /// covariate design are kept separate, so the n_cov*n_grid × p_total
    /// row-replicated form is never materialized. Only `forward_mul` is
    /// invoked downstream (interior-point fraction-to-boundary line search).
    x_deriv_grid_kron: KroneckerDesign,

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
    /// Constant `X_val^T W X_val` term reused across all Newton steps.
    x_val_weighted_gram: Array2<f64>,

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

    // --- Resource policy for chunked dense materialization ---
    /// Resource policy threaded through factored weighted-cross helpers and the
    /// `KroneckerDesign::weighted_gram` callers inside this family. Stored on
    /// the struct because the `CustomFamily` trait surface cannot grow new
    /// parameters per call.
    policy: ResourcePolicy,

    // --- Active-set certificate for the monotonicity-grid line search ---
    /// Cached `(active_pairs, m_inactive, ||r_g||, ||X_cov_i||)` summary for
    /// `KroneckerDesign::min_step_to_boundary_with_active_set`. `Mutex` so
    /// the `&self` `max_feasible_step_size` callsite can mutate it while the
    /// surrounding family stays `Send + Sync` (required by
    /// `ExactNewtonJointHessianWorkspace: Send + Sync`).
    active_set_cache: Mutex<KroneckerActiveSetCache>,
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
        let x_val_kron = KroneckerDesign::new_khatri_rao(&resp_val, covariate_design.clone())?;
        let x_deriv_kron = KroneckerDesign::new_khatri_rao(&resp_deriv, covariate_design.clone())?;
        let x_deriv_grid_kron = build_monotonicity_derivative_grid_kron(
            response,
            &resp_knots,
            config.response_degree,
            &resp_transform,
            &covariate_design,
        )?;
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_grid_kron.ncols(), p_total);
        // Hard invariant (release+debug): the monotonicity grid design must
        // remain the Kronecker variant. Switching it to the row-wise KhatriRao
        // variant re-introduces the n_cov*n_grid row replication and OOMs at
        // biobank scale. Cost is one match per family construction.
        assert!(
            matches!(x_deriv_grid_kron, KroneckerDesign::Kronecker { .. }),
            "x_deriv_grid_kron must be the Kronecker variant — KhatriRao re-introduces O(n_cov*n_grid*p_total) row-replicated materialization",
        );

        // ----- 3. Warm start -----
        let initial_beta = compute_warm_start(
            response,
            weights,
            offset,
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
        let policy = ResourcePolicy::default_library();
        let x_val_weighted_gram = x_val_kron.weighted_gram(weights, &policy);

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
            x_deriv_grid_kron,
            response_val_basis: resp_val,
            response_deriv_basis: resp_deriv,
            covariate_design,
            weights: Arc::new(weights.clone()),
            offset: Arc::new(offset.clone()),
            tensor_penalties,
            x_val_weighted_gram,
            initial_beta,
            initial_log_lambdas,
            block_name: "transformation".to_string(),
            response_knots: resp_knots,
            response_transform: resp_transform,
            response_degree: config.response_degree,
            response_median: resp_median,
            policy,
            active_set_cache: Mutex::new(KroneckerActiveSetCache::new()),
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
        let x_val_kron =
            KroneckerDesign::new_khatri_rao(&response_val_basis, covariate_design.clone())?;
        let x_deriv_kron =
            KroneckerDesign::new_khatri_rao(&response_deriv_basis, covariate_design.clone())?;
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_kron.ncols(), p_total);

        // Warm start: need response values for location-scale init.
        // Extract response from column 1 of response_val_basis (which stores y).
        let response_approx = response_val_basis.column(1).to_owned();
        let x_deriv_grid_kron = build_monotonicity_derivative_grid_kron(
            &response_approx,
            &response_knots,
            response_degree,
            &response_transform,
            &covariate_design,
        )?;
        debug_assert_eq!(x_deriv_grid_kron.ncols(), p_total);
        // Hard invariant — see TransformationNormalFamily::new for rationale.
        assert!(
            matches!(x_deriv_grid_kron, KroneckerDesign::Kronecker { .. }),
            "x_deriv_grid_kron must be the Kronecker variant — KhatriRao re-introduces O(n_cov*n_grid*p_total) row-replicated materialization",
        );
        let initial_beta = compute_warm_start(
            &response_approx,
            weights,
            offset,
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
        let policy = ResourcePolicy::default_library();
        let x_val_weighted_gram = x_val_kron.weighted_gram(weights, &policy);

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
            x_deriv_grid_kron,
            response_val_basis,
            response_deriv_basis,
            covariate_design,
            weights: Arc::new(weights.clone()),
            offset: Arc::new(offset.clone()),
            tensor_penalties,
            x_val_weighted_gram,
            initial_beta,
            initial_log_lambdas,
            block_name: "transformation".to_string(),
            response_knots: response_knots.clone(),
            response_transform: response_transform.clone(),
            response_degree,
            response_median: resp_median,
            policy,
            active_set_cache: Mutex::new(KroneckerActiveSetCache::new()),
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

/// Weighted cross-product of two rowwise-Kronecker designs, kept strictly
/// factored: output block (a, c) equals `B^T diag(w_i A_{ia} C_{ic}) D`.
fn factored_weighted_cross(
    a: &Array2<f64>,
    b: &Array2<f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    c: &Array2<f64>,
    d: &Array2<f64>,
    policy: &ResourcePolicy,
) -> Result<Array2<f64>, String> {
    let n = weights.len();
    if a.nrows() != n || b.nrows() != n || c.nrows() != n || d.nrows() != n {
        return Err(format!(
            "factored_weighted_cross row mismatch: weights={n}, a={}, b={}, c={}, d={}",
            a.nrows(),
            b.nrows(),
            c.nrows(),
            d.nrows()
        ));
    }
    let pa = a.ncols();
    let pc = c.ncols();
    let pb = b.ncols();
    let pd = d.ncols();

    let mut out = Array2::<f64>::zeros((pa * pb, pc * pd));
    let mut pair_weights = Array1::<f64>::zeros(n);

    for ia in 0..pa {
        let a_col = a.column(ia);
        for ic in 0..pc {
            let c_col = c.column(ic);
            for r in 0..n {
                pair_weights[r] = weights[r] * a_col[r] * c_col[r];
            }
            let block = chunked_weighted_bt_d(b, pair_weights.view(), d, policy);
            let mut slice = out.slice_mut(s![ia * pb..(ia + 1) * pb, ic * pd..(ic + 1) * pd]);
            slice.assign(&block);
        }
    }

    Ok(out)
}

/// Chunked weighted B^T diag(w) D product without materializing any
/// full rowwise-Kronecker intermediate.
fn chunked_weighted_bt_d(
    b: &Array2<f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    d: &Array2<f64>,
    policy: &ResourcePolicy,
) -> Array2<f64> {
    let n = weights.len();
    let pb = b.ncols();
    let pd = d.ncols();
    let rows_per_chunk =
        crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, pb + pd);
    let mut out = Array2::<f64>::zeros((pb, pd));
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let bl = b.slice(s![start..end, ..]);
        let dl = d.slice(s![start..end, ..]);
        let mut dw = dl.to_owned();
        for local in 0..(end - start) {
            let w = weights[start + local];
            for j in 0..pd {
                dw[[local, j]] *= w;
            }
        }
        out += &bl.t().dot(&dw);
    }
    out
}

/// Chunked weighted `B^T diag(w) D` product where `B` and `D` are
/// operator-backed `DesignMatrix` instances. Materializes only one row chunk
/// at a time using the operator's `row_chunk` primitive, so neither factor's
/// full dense form ever lives in memory.
fn chunked_weighted_bt_d_designmatrix(
    b: &DesignMatrix,
    weights: ndarray::ArrayView1<'_, f64>,
    d: &DesignMatrix,
    policy: &ResourcePolicy,
) -> Result<Array2<f64>, String> {
    let n = weights.len();
    let pb = b.ncols();
    let pd = d.ncols();
    let rows_per_chunk =
        crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, pb + pd);
    let mut out = Array2::<f64>::zeros((pb, pd));
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let bl = b.try_row_chunk(start..end).map_err(|e| e.to_string())?;
        let dl = d.try_row_chunk(start..end).map_err(|e| e.to_string())?;
        let mut dw = dl;
        for local in 0..(end - start) {
            let w = weights[start + local];
            if w != 1.0 {
                for j in 0..pd {
                    dw[[local, j]] *= w;
                }
            }
        }
        out += &bl.t().dot(&dw);
    }
    Ok(out)
}

/// Apply a Khatri-Rao weighted Gram to a vector, fusing the forward and
/// transpose halves through a single `(n × p_resp)` buffer.
///
/// Computes `out = (A ⊙ B)^T diag(weights) (A ⊙ B) v`, where
///   - `left = A` is the dense response factor `(n × p_resp)`,
///   - `right = B` is the dense covariate factor `(n × p_cov)`,
///   - `weights` is the per-row diagonal,
///   - `v` flattens the response-major coefficient block `(p_resp × p_cov)`.
///
/// The standard factored apply
/// `transpose_mul(weights · forward_mul(v))` allocates the `(n × p_resp)`
/// intermediate twice — once inside `forward_mul` (`B · V^T`) and once
/// inside `transpose_mul` (`weighted_left = scaled · A` row-wise). This
/// fused helper allocates that buffer exactly once: it is filled by the
/// forward gemm `R = B · V^T`, then overwritten in place row by row to
/// `U[i, a] = w_i · (A[i, :] · R[i, :]) · A[i, a]`, then consumed by the
/// transpose gemm `U^T · B`. FLOP count is unchanged (two BLAS gemms plus
/// one `O(n · p_resp)` row pass); peak transient memory drops by
/// `n · p_resp · 8` bytes — at biobank scale that is ~33 MiB saved per
/// matvec, multiplied across the inner-PCG and outer-trace matvec budgets.
fn fused_khatri_rao_weighted_gram_apply(
    left: &Array2<f64>,
    right: &Array2<f64>,
    weights: &Array1<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    let n = left.nrows();
    let p_resp = left.ncols();
    let p_cov = right.ncols();
    debug_assert_eq!(right.nrows(), n);
    debug_assert_eq!(weights.len(), n);
    debug_assert_eq!(v.len(), p_resp * p_cov);
    let v_mat = v
        .view()
        .into_shape_with_order((p_resp, p_cov))
        .expect("v reshape to (p_resp, p_cov) — caller validates length");
    // Forward phase: buf = B · V^T.
    let mut buf = fast_ab(right, &v_mat.t().to_owned());
    // Inner pass: overwrite buf row-by-row to U[i, :] = w_i · xv_i · A[i, :].
    ndarray::Zip::from(buf.rows_mut())
        .and(left.rows())
        .and(weights.view())
        .for_each(|mut buf_row, left_row, &w| {
            let xv_i = left_row.dot(&buf_row);
            let factor = w * xv_i;
            buf_row.assign(&left_row);
            buf_row *= factor;
        });
    // Transpose phase: result_mat = U^T · B → (p_resp × p_cov).
    let result_mat = fast_atb(&buf, right);
    result_mat
        .into_shape_with_order((p_resp * p_cov,))
        .expect("p_resp · p_cov flatten")
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

        // Hard monotonicity gate: ℓ = -½h² + log h' is only defined for h' > 0,
        // and the barrier line search in `log_likelihood_only` retreats via
        // NEG_INFINITY before a step that would push any h' to zero. Reaching
        // this branch in `evaluate` means the constraint solver handed us an
        // invalid iterate, so fail loudly instead of silently producing NaNs.
        let min_h_prime = h_prime.iter().copied().fold(f64::INFINITY, f64::min);
        if min_h_prime <= 0.0 {
            return Err(format!(
                "TransformationNormalFamily: h' has non-positive values (min = {min_h_prime:.6e}). \
                 Monotonicity constraint may be violated."
            ));
        }

        // Single fused pass over rows: accumulate ℓ = Σ w·(-½h² + log h') and
        // build the weighted vectors the gradient and Hessian consume. The
        // unfused version allocated five full-length O(n) temporaries
        // (inv_h_prime, weighted_h, weighted_inv_h_prime, inv_h_prime_sq,
        // weighted_inv_h_prime_sq) with two reciprocals per row; at biobank n
        // that is ~40·n bytes of allocator churn and 2·n divisions per call.
        // Here we allocate three vectors, divide once, and derive the squared
        // quantity by multiplication.
        let mut log_likelihood = 0.0;
        let mut weighted_h = Array1::<f64>::zeros(n);
        let mut weighted_inv_hp = Array1::<f64>::zeros(n);
        let mut weighted_inv_hp_sq = Array1::<f64>::zeros(n);
        for i in 0..n {
            let hi = h[i];
            let hpi = h_prime[i];
            let wi = self.weights[i];
            let inv = 1.0 / hpi;
            let w_inv = wi * inv;
            log_likelihood += wi * (-0.5 * hi * hi + hpi.ln());
            weighted_h[i] = wi * hi;
            weighted_inv_hp[i] = w_inv;
            weighted_inv_hp_sq[i] = w_inv * inv;
        }

        // Gradient of log-likelihood: ∇ℓ = -X_val^T (w·h) + X_deriv^T (w/h')
        let grad = {
            let xdt_inv = self.x_deriv_kron.transpose_mul(&weighted_inv_hp);
            let xvt_h = self.x_val_kron.transpose_mul(&weighted_h);
            xdt_inv - xvt_h
        };

        // Negative Hessian of log-likelihood:
        //   -∇²ℓ = X_val^T diag(w) X_val + X_deriv^T diag(w/h'²) X_deriv
        // The first term is precomputed once as `x_val_weighted_gram`.
        let hessian = {
            let mut xtx_deriv = self
                .x_deriv_kron
                .weighted_gram(&weighted_inv_hp_sq, &self.policy);
            xtx_deriv += &self.x_val_weighted_gram;
            xtx_deriv
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

    /// Log-likelihood + flat joint gradient without building the dense Hessian.
    ///
    /// The default trait implementation returns `None`, so the joint-Newton
    /// inner solver falls back to `evaluate()` to obtain the gradient — and
    /// that side-effects a full `Θ(n p²)` `weighted_gram` Hessian build at
    /// every inner iteration. CTN's gradient is structurally
    ///
    ///   `∇ℓ = -X_val^T (w·h) + X_deriv^T (w/h')`,
    ///
    /// which is two `transpose_mul`s through the existing Khatri-Rao operators
    /// and one `Θ(n)` row reduction — `Θ(n p)` total. At biobank scale that is
    /// ~10⁷ FLOPs per call versus ~3·10¹⁰ for the full `evaluate`, so wiring
    /// this override is the gating condition for routing CTN's inner solve
    /// through the matrix-free joint-Newton path without paying the dense H
    /// tax on every gradient refresh.
    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        if block_states.len() != 1 {
            return Err(format!(
                "TransformationNormalFamily expects 1 block, got {}",
                block_states.len()
            ));
        }
        let beta = &block_states[0].beta;
        let (h, h_prime) = self.compute_h_and_h_prime(beta);
        let n = h.len();
        let min_h_prime = h_prime.iter().copied().fold(f64::INFINITY, f64::min);
        if min_h_prime <= 0.0 {
            return Err(format!(
                "TransformationNormalFamily: h' has non-positive values (min = {min_h_prime:.6e}). \
                 Monotonicity constraint may be violated."
            ));
        }
        // Fused single-pass O(n) reduction: log-likelihood + per-row weighted
        // vectors needed for the two transpose_muls below.
        let mut log_likelihood = 0.0;
        let mut weighted_h = Array1::<f64>::zeros(n);
        let mut weighted_inv_hp = Array1::<f64>::zeros(n);
        let weights = self.weights.as_ref();
        for i in 0..n {
            let wi = weights[i];
            let hi = h[i];
            let hpi = h_prime[i];
            log_likelihood += wi * (-0.5 * hi * hi + hpi.ln());
            weighted_h[i] = wi * hi;
            weighted_inv_hp[i] = wi / hpi;
        }
        let mut gradient = self.x_deriv_kron.transpose_mul(&weighted_inv_hp);
        gradient -= &self.x_val_kron.transpose_mul(&weighted_h);
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood,
            gradient,
        }))
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        // The Hessian depends on β through 1/h'² where h' = X_deriv · β.
        true
    }

    fn coefficient_hessian_cost(&self, _specs: &[ParameterBlockSpec]) -> u64 {
        // Khatri–Rao tensor design: the coefficient block is X = R ⊙ C with
        // rows length p_resp · p_cov. Two regimes:
        //
        // * **Dense regime** (small enough that the unified evaluator builds
        //   `weighted_gram` directly): per-evaluation cost is the dense
        //   `n · (p_resp · p_cov)²` Khatri–Rao gram build.
        //
        // * **Matrix-free regime** (large enough that
        //   `use_joint_matrix_free_path` returns true and the evaluator
        //   factors `H v` through `forward_mul` / `transpose_mul` on the
        //   Khatri–Rao operands): per-`Hv` matvec cost is just
        //   `n · (p_resp + p_cov)` flops — see `ctn_matrix_free_workspace`.
        //   The trait doc specifies that matrix-free families report the
        //   per-`Hv` cost so the gate reflects the operator path actually
        //   used at fit time, not the dense build the evaluator skips.
        let n_usize = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols() as u64;
        let p_cov = self.covariate_design.ncols() as u64;
        let p_total = p_resp.saturating_mul(p_cov);
        let n = n_usize as u64;
        if crate::custom_family::use_joint_matrix_free_path(p_total as usize, n_usize) {
            n.saturating_mul(p_resp.saturating_add(p_cov))
        } else {
            n.saturating_mul(p_total.saturating_mul(p_total))
        }
    }

    fn coefficient_gradient_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Honest CTN per-gradient cost. The default
        // (`coefficient_hessian_cost / 2`) only counts the inner Newton solve
        // and undercounts each outer evaluation, because CTN also runs a
        // monotonicity fraction-to-boundary scan via
        // `KronDesign::min_step_to_boundary` on every gradient/cost step.
        // That pass walks the n_cov × n_grid virtual row space, forming
        // chunk-local `c · Rᵀ` and `d · Rᵀ` factors, with cost
        // `2 · n · n_grid · p_resp + 2 · n · p_resp · p_cov`. At biobank
        // scale (n=320 000, n_grid=293, p_resp=32, p_cov=23) that is
        // ≈ 6.5·10⁹ multiply-adds per pass — orders of magnitude above
        // the matrix-free `Hv` cost — so leaving it out of the gradient
        // cost reports a per-eval budget tens of times smaller than
        // reality, and `cost_gated_first_order_max_iter` lets the BFGS
        // loop request far more outer iterations than the global FLOP
        // budget can pay for. That is the proximate cause of the
        // `rust_margslope_aniso_duchon16d_linkwiggle_scorewarp_fast`
        // and survival-marginal-slope timeouts: the CTN baseline
        // custom-family fit runs to its full requested `outer_max_iter`
        // because the gate never fires.
        let inner = self.coefficient_hessian_cost(specs) / 2;
        let n = self.response_val_basis.nrows() as u64;
        let p_resp = self.response_val_basis.ncols() as u64;
        let p_cov = self.covariate_design.ncols() as u64;
        let n_grid = match &self.x_deriv_grid_kron {
            KroneckerDesign::Kronecker { response_grid, .. } => response_grid.nrows() as u64,
            KroneckerDesign::KhatriRao { .. } => 0,
        };
        let monotonicity_pass = n
            .saturating_mul(2u64.saturating_mul(n_grid).saturating_mul(p_resp))
            .saturating_add(n.saturating_mul(2u64.saturating_mul(p_resp).saturating_mul(p_cov)));
        inner.saturating_add(monotonicity_pass)
    }

    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        crate::custom_family::cost_gated_outer_order(specs, self.coefficient_hessian_cost(specs))
    }

    fn outer_seed_config(&self, n_params: usize) -> crate::seeding::SeedConfig {
        crate::seeding::SeedConfig {
            bounds: (-12.0, 12.0),
            max_seeds: if n_params <= 8 { 1 } else { 2 },
            seed_budget: 1,
            screen_max_inner_iterations: 2,
            risk_profile: crate::seeding::SeedRiskProfile::Gaussian,
            num_auxiliary_trailing: 0,
        }
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

        // Fraction-to-boundary: find the largest α ∈ (0, 1] such that
        // h'(β + α · δ) > 0 at every observed row and h'(y_g; x_i) > ε at
        // every (covariate, response-grid) pair on the monotonicity grid.
        //
        // Each design owns its own streaming reduction over its virtual rows
        // (KhatriRao: n observations; Kronecker: n_cov × n_grid pairs without
        // materializing the dense forward image). Both return the smallest
        // binding step ratio, or +∞ if no row binds. The grid uses the
        // strict-feasibility margin TRANSFORMATION_MONOTONICITY_EPS; observed
        // rows use 0.0 (the log-h' barrier in the likelihood already keeps
        // h' away from zero on observed rows). Composing via `f64::min` is
        // associative for non-NaN inputs, so the final α_max is bit-equivalent
        // to a single scan over the union of binding rows.
        //
        // Observation-row reduction stays on the un-cached path because the
        // KhatriRao variant streams over `n` observation rows directly via
        // `forward_mul`; there is no factored projection to reuse there.
        let alpha_obs = self
            .x_deriv_kron
            .min_step_to_boundary(beta, delta, 0.0, 1e-14);
        // Monotonicity-grid reduction: the `Kronecker` variant lets us project
        // β and δ through the covariate factor once per call and route the
        // chunked (i, g) reduction through the cached entry point. The
        // `KhatriRao` branch of `project_kronecker_factor` returns `None`, so
        // we transparently fall back to the un-cached path for any future
        // configuration that swaps the grid design out of the Kronecker form.
        let alpha_grid = match (
            self.x_deriv_grid_kron.project_kronecker_factor(beta),
            self.x_deriv_grid_kron.project_kronecker_factor(delta),
        ) {
            (Some(c_beta), Some(d_delta)) => {
                // Active-set fast path with full-grid certificate fallback.
                // Bit-equivalent to `min_step_to_boundary_with_projections`
                // when the certificate accepts; otherwise refreshes via a
                // full grid scan and returns the same `α`.
                let mut cache = self
                    .active_set_cache
                    .lock()
                    .expect("active-set cache mutex poisoned");
                self.x_deriv_grid_kron.min_step_to_boundary_with_active_set(
                    c_beta.view(),
                    d_delta.view(),
                    delta,
                    TRANSFORMATION_MONOTONICITY_EPS,
                    1e-14,
                    &mut cache,
                )
            }
            _ => self.x_deriv_grid_kron.min_step_to_boundary(
                beta,
                delta,
                TRANSFORMATION_MONOTONICITY_EPS,
                1e-14,
            ),
        };
        let alpha_max = 1.0_f64.min(alpha_obs).min(alpha_grid);

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
        block_index: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_index != 0 {
            return Ok(None);
        }
        // The CTN tensor design is intentionally factored. The dense active-set
        // constraint API cannot represent that without persisting the full
        // n_grid x p_response x p_covariate matrix, so monotonicity is enforced
        // by the likelihood barrier plus the fraction-to-boundary step rule.
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
        let dd = self.x_deriv_kron.weighted_gram(&weight, &self.policy);
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
        let mut xtx_deriv = self
            .x_deriv_kron
            .weighted_gram(&weighted_inv_h_prime_sq, &self.policy);
        xtx_deriv += &self.x_val_weighted_gram;
        Ok(Some(xtx_deriv))
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
        Ok(Some(self.x_deriv_kron.weighted_gram(&weight, &self.policy)))
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
            let cubic_term = self.x_deriv_kron.weighted_gram(&w_cubic, &self.policy);

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
            // Stay factored: keep (response_basis, covariate_factor) pairs and
            // never materialize n x (p_resp * p_cov) rowwise-Kronecker matrices.
            // The resource policy is the one stored on the family so chunk
            // sizing matches the surrounding workload.
            let policy: &ResourcePolicy = &self.policy;
            let cov_design_dense = op.covariate_design.as_dense_ref().ok_or_else(|| {
                "TransformationNormalFamily exact Hessian requires dense covariate design"
                    .to_string()
            })?;
            let cov_i = op.materialize_cov_first(axis_i).map_err(|e| {
                format!("tensor psi second-order materialize_cov_first(i) failed: {e}")
            })?;
            let cov_j = op.materialize_cov_first(axis_j).map_err(|e| {
                format!("tensor psi second-order materialize_cov_first(j) failed: {e}")
            })?;
            let cov_ij = op.materialize_cov_second(axis_i, axis_j).map_err(|e| {
                format!("tensor psi second-order materialize_cov_second failed: {e}")
            })?;

            let resp_val = op.response_val_basis.as_ref();
            let resp_deriv = op.response_deriv_basis.as_ref();
            let w_view = self.weights.view();
            let w_inv_h2_view = weighted_inv_h_prime_sq.view();

            let mut hess =
                factored_weighted_cross(resp_val, &cov_i, w_view, resp_val, &cov_j, policy)?;
            hess += &factored_weighted_cross(resp_val, &cov_j, w_view, resp_val, &cov_i, policy)?;
            hess += &factored_weighted_cross(
                resp_val,
                &cov_ij,
                w_view,
                resp_val,
                cov_design_dense,
                policy,
            )?;
            hess += &factored_weighted_cross(
                resp_val,
                cov_design_dense,
                w_view,
                resp_val,
                &cov_ij,
                policy,
            )?;
            hess += &factored_weighted_cross(
                resp_deriv,
                &cov_i,
                w_inv_h2_view,
                resp_deriv,
                &cov_j,
                policy,
            )?;
            hess += &factored_weighted_cross(
                resp_deriv,
                &cov_j,
                w_inv_h2_view,
                resp_deriv,
                &cov_i,
                policy,
            )?;
            hess += &factored_weighted_cross(
                resp_deriv,
                &cov_ij,
                w_inv_h2_view,
                resp_deriv,
                cov_design_dense,
                policy,
            )?;
            hess += &factored_weighted_cross(
                resp_deriv,
                cov_design_dense,
                w_inv_h2_view,
                resp_deriv,
                &cov_ij,
                policy,
            )?;

            let cubic_i =
                ((&v_j_deriv * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
            hess += &factored_weighted_cross(
                resp_deriv,
                &cov_i,
                cubic_i.view(),
                resp_deriv,
                cov_design_dense,
                policy,
            )?;
            hess += &factored_weighted_cross(
                resp_deriv,
                cov_design_dense,
                cubic_i.view(),
                resp_deriv,
                &cov_i,
                policy,
            )?;

            let cubic_j =
                ((&v_i_deriv * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
            hess += &factored_weighted_cross(
                resp_deriv,
                &cov_j,
                cubic_j.view(),
                resp_deriv,
                cov_design_dense,
                policy,
            )?;
            hess += &factored_weighted_cross(
                resp_deriv,
                cov_design_dense,
                cubic_j.view(),
                resp_deriv,
                &cov_j,
                policy,
            )?;

            let cubic_second =
                ((&v_ij_deriv * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
            hess += &self.x_deriv_kron.weighted_gram(&cubic_second, policy);

            let quartic = ((&v_i_deriv * &v_j_deriv) * &inv_h_prime_qu * self.weights.as_ref())
                .mapv(|v| 6.0 * v);
            hess += &self.x_deriv_kron.weighted_gram(&quartic, policy);
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

        let policy = &self.policy;
        let cov_design_dense = op.covariate_design.as_dense_ref().ok_or_else(|| {
            "TransformationNormalFamily hessian drift requires dense covariate design".to_string()
        })?;
        let cov_psi = op
            .materialize_cov_first(axis)
            .map_err(|e| format!("tensor psi hessian drift materialize_cov_first failed: {e}"))?;
        let resp_deriv = op.response_deriv_basis.as_ref();

        let cubic_h = ((&d_h_prime * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
        let mut hess = factored_weighted_cross(
            resp_deriv,
            &cov_psi,
            cubic_h.view(),
            resp_deriv,
            cov_design_dense,
            policy,
        )?;
        hess += &factored_weighted_cross(
            resp_deriv,
            cov_design_dense,
            cubic_h.view(),
            resp_deriv,
            &cov_psi,
            policy,
        )?;

        let cubic_v = ((&d_v_deriv * &inv_h_prime_cu) * self.weights.as_ref()).mapv(|v| -2.0 * v);
        hess += &self.x_deriv_kron.weighted_gram(&cubic_v, &self.policy);

        let quartic =
            ((&v_deriv * &d_h_prime) * &inv_h_prime_qu * self.weights.as_ref()).mapv(|v| 6.0 * v);
        hess += &self.x_deriv_kron.weighted_gram(&quartic, &self.policy);

        Ok(Some(0.5 * (&hess + &hess.t())))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        if block_states.len() != 1 {
            return Err(format!(
                "TransformationNormalFamily expects 1 block, got {}",
                block_states.len()
            ));
        }
        let beta = &block_states[0].beta;
        let h_prime = self.x_deriv_kron.forward_mul(beta);
        // Build weighted_inv_hp_sq[i] = w_i / h'_i² and validate h' > 0 in one
        // pass — the inner solver's barrier line search keeps h' strictly
        // positive on observed rows, so the only way to land here with a
        // non-positive value is an upstream invariant break worth surfacing
        // loudly.
        let mut weighted_inv_hp_sq = Array1::<f64>::zeros(h_prime.len());
        for i in 0..h_prime.len() {
            let hp = h_prime[i];
            if !hp.is_finite() || hp <= 0.0 {
                return Err(format!(
                    "TransformationNormalFamily Hessian workspace: h'[{i}] = {hp} is not strictly positive"
                ));
            }
            weighted_inv_hp_sq[i] = self.weights[i] / (hp * hp);
        }
        let workspace = TransformationNormalJointHessianWorkspace::new(
            Arc::new(self.clone()),
            h_prime,
            weighted_inv_hp_sq,
        )?;
        Ok(Some(
            Arc::new(workspace) as Arc<dyn ExactNewtonJointHessianWorkspace>
        ))
    }

    fn supports_matrix_free_joint_hessian(
        &self,
        _specs: &[ParameterBlockSpec],
    ) -> bool {
        // CTN's joint Hessian is always supplied as a matrix-free Hv
        // operator: `exact_newton_joint_hessian_workspace` above unconditionally
        // returns a `TransformationNormalJointHessianWorkspace` for any single
        // block (the `block_states.len() != 1` guard surfaces a panic, never a
        // `None`), so the unified outer evaluator will see
        // `JointHessianSource::Operator` rather than a dense matrix as soon as
        // `use_joint_matrix_free_path` fires on `(p, n)`. Returning `true` here
        // tells the outer planner to suppress the cost-driven
        // `prefer_gradient_only` downgrade — ARC's `run_operator_trust_region`
        // path will absorb the per-iteration cost via O(n·p) HVPs.
        true
    }
}

// ---------------------------------------------------------------------------
// Matrix-free joint Hessian workspace (Khatri-Rao operator-only)
// ---------------------------------------------------------------------------

/// Per-evaluation workspace for the CTN joint Hessian.
///
/// Caches the row-space quantities needed to apply
///   H = X_val^T diag(w) X_val + X_deriv^T diag(w / h'^2) X_deriv
/// without materializing X_val, X_deriv, or H itself. All matvecs go through
/// the Khatri-Rao operator primitives on `KroneckerDesign`, which run in
/// `O(n * p_resp * p_cov) = O(n * p)` per call. This avoids both the
/// `O(n * p^2)` cost of the dense weighted Gram and the `O(p^2)` storage of
/// the materialized joint Hessian.
struct TransformationNormalJointHessianWorkspace {
    /// Shared family handle. Cloning the workspace's family for each downstream
    /// matrix-free operator (dH, d²H per psi coord and per pair) would copy
    /// the full row-space Kronecker designs (~hundreds of MiB at biobank
    /// scale) per call. Arc-sharing makes operator construction O(1).
    family: Arc<TransformationNormalFamily>,
    /// h'_i = X_deriv · β at the current iterate. Cached so the matrix-free
    /// directional-derivative operators can reuse it without rerunning
    /// `forward_mul(beta)`.
    h_prime: Arc<Array1<f64>>,
    /// Row weights w / h'^2 for the X_deriv^T diag(·) X_deriv summand.
    weighted_inv_hp_sq: Array1<f64>,
}

impl TransformationNormalJointHessianWorkspace {
    fn new(
        family: Arc<TransformationNormalFamily>,
        h_prime: Array1<f64>,
        weighted_inv_hp_sq: Array1<f64>,
    ) -> Result<Self, String> {
        Ok(Self {
            family,
            h_prime: Arc::new(h_prime),
            weighted_inv_hp_sq,
        })
    }

    fn p_total(&self) -> usize {
        self.family.x_val_kron.ncols()
    }

    fn apply_hessian(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.p_total() {
            return Err(format!(
                "CTN joint Hessian matvec: input length {} != p_total {}",
                v.len(),
                self.p_total()
            ));
        }
        // Term 1: (X_val^T W X_val) · v.
        //
        // The val gram is constant in β (the weights are the family's row
        // weights, not the β-dependent 1/h'² kernel), so the family caches it
        // once at construction. A dense gemv on the cached p×p matrix is
        // O(p²) ≈ 9·10⁴ FLOPs at biobank scale — strictly cheaper than the
        // factored apply (~10⁷ FLOPs) and the only term whose constant
        // structure the workspace can exploit.
        let mut out = self.family.x_val_weighted_gram.dot(v);

        // Term 2: factored (X_deriv^T diag(w/h'²) X_deriv) · v.
        //
        // The deriv kernel changes with β, so we apply it through the
        // Khatri-Rao factor pair `(response_deriv_basis, covariate_design)`.
        // Dense covariates take the fused path that allocates one
        // `(n × p_resp)` buffer; sparse / operator-backed covariates fall
        // through to the existing factored apply on `x_deriv_kron`, which
        // uses the operator's `apply` / `apply_transpose` chunk primitives.
        out += &self.factored_deriv_apply(v);
        Ok(out)
    }

    /// Factored apply of `X_deriv^T diag(weighted_inv_hp_sq) X_deriv` to `v`.
    ///
    /// Selects the fused dense-covariate path when available; otherwise calls
    /// through `KroneckerDesign::forward_mul` / `transpose_mul`.
    fn factored_deriv_apply(&self, v: &Array1<f64>) -> Array1<f64> {
        if let Some(cov_dense) = self.family.covariate_design.as_dense_ref() {
            fused_khatri_rao_weighted_gram_apply(
                &self.family.response_deriv_basis,
                cov_dense,
                &self.weighted_inv_hp_sq,
                v,
            )
        } else {
            let mut deriv_image = self.family.x_deriv_kron.forward_mul(v);
            deriv_image *= &self.weighted_inv_hp_sq;
            self.family.x_deriv_kron.transpose_mul(&deriv_image)
        }
    }

    /// Exact diagonal of the unpenalized joint Hessian.
    ///
    /// `H = (X_val^T W X_val) + (X_deriv^T diag(w/h'²) X_deriv)`. The first
    /// term's diagonal is a single `diag()` extraction off the cached
    /// `x_val_weighted_gram`; the second term is the Khatri–Rao identity
    ///
    ///   `H_deriv_(a,b),(a,b) = Σ_i (w/h'²)_i · A_deriv[i,a]² · B[i,b]²`,
    ///
    /// which we compute as `(M_w)^T · B²` block-by-block over row chunks,
    /// where `M_w[i,a] = (w/h'²)_i · A_deriv[i,a]²`. Each chunk is one
    /// `fast_atb` BLAS call, the chunks are reduced in parallel, and the
    /// covariate design is consumed via `try_row_chunk` so sparse covariates
    /// stream without densifying.
    fn compute_diagonal(&self) -> Result<Array1<f64>, String> {
        let p_resp = self.family.response_deriv_basis.ncols();
        let p_cov = self.family.covariate_design.ncols();
        let total = p_resp * p_cov;
        let cached_diag = self.family.x_val_weighted_gram.diag();
        if cached_diag.len() != total {
            return Err(format!(
                "CTN diagonal: cached val gram diag length {} != p_total {}",
                cached_diag.len(),
                total
            ));
        }
        // Term 1: precomputed val diagonal — one Array1 clone, no recomputation.
        let mut diag = cached_diag.to_owned();

        // Term 2: factored deriv diagonal, streamed in row chunks.
        let n = self.family.weights.len();
        if n == 0 {
            return Ok(diag);
        }
        let policy = &self.family.policy;
        let rows_per_chunk = crate::resource::rows_for_target_bytes(
            policy.row_chunk_target_bytes,
            (p_cov + p_resp).max(1),
        )
        .max(1);
        let chunks: Vec<(usize, usize)> = (0..n)
            .step_by(rows_per_chunk)
            .map(|s| (s, (s + rows_per_chunk).min(n)))
            .collect();
        let resp_deriv = &self.family.response_deriv_basis;
        let cov_design = &self.family.covariate_design;
        let weighted_inv_hp_sq = &self.weighted_inv_hp_sq;

        let partial: Array2<f64> = chunks
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_resp, p_cov)),
                |mut local, (start, end)| -> Result<Array2<f64>, String> {
                    let m = end - start;
                    // Pre-square the covariate row chunk in place — the row_chunk
                    // helper already returns an owned Array2 we are free to mutate.
                    let mut cov_sq_chunk = cov_design
                        .try_row_chunk(start..end)
                        .map_err(|e| format!("CTN diagonal covariate row_chunk: {e}"))?;
                    cov_sq_chunk.mapv_inplace(|v| v * v);
                    // M_w[i, a] = (w/h'²)_{start+i} · A_deriv[start+i, a]².
                    let mut m_w = Array2::<f64>::zeros((m, p_resp));
                    for i_local in 0..m {
                        let i = start + i_local;
                        let c = weighted_inv_hp_sq[i];
                        for a in 0..p_resp {
                            let d = resp_deriv[[i, a]];
                            m_w[[i_local, a]] = c * d * d;
                        }
                    }
                    // (p_resp × p_cov) += M_w^T · B² — one BLAS gemm per chunk.
                    local += &fast_atb(&m_w, &cov_sq_chunk);
                    Ok(local)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_resp, p_cov)),
                |mut a, b| {
                    a += &b;
                    Ok(a)
                },
            )?;
        // Flatten (p_resp × p_cov) into the row-major p_total layout.
        for a in 0..p_resp {
            let mut slice = diag.slice_mut(s![a * p_cov..(a + 1) * p_cov]);
            slice += &partial.row(a);
        }
        Ok(diag)
    }
}

impl ExactNewtonJointHessianWorkspace for TransformationNormalJointHessianWorkspace {
    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.apply_hessian(v)?))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.compute_diagonal()?))
    }

    fn directional_derivative(
        &self,
        _d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Dense form is reached only when a caller forces materialization via
        // `into_operator().to_dense()`. Returning `None` keeps consumers on the
        // operator path supplied below; legacy callers that explicitly need a
        // p×p matrix fall through to `family.exact_newton_joint_hessian_directional_derivative_with_specs`
        // (the dense `weighted_gram` route) at the call site, so behavior is
        // preserved without forcing a dense build here.
        Ok(None)
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let p_total = self.p_total();
        if d_beta_flat.len() != p_total {
            return Err(format!(
                "CTN directional_derivative_operator length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                p_total
            ));
        }
        let d_h_prime = self.family.x_deriv_kron.forward_mul(d_beta_flat);
        let op = TransformationNormalDhMatrixFreeOperator::new(
            Arc::clone(&self.family),
            Arc::clone(&self.h_prime),
            d_h_prime,
        );
        Ok(Some(Arc::new(op) as Arc<dyn HyperOperator>))
    }

    fn second_directional_derivative(
        &self,
        _u: &Array1<f64>,
        _v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let p_total = self.p_total();
        if d_beta_u.len() != p_total || d_beta_v.len() != p_total {
            return Err(format!(
                "CTN second_directional_derivative_operator length mismatch: u={}, v={}, expected {}",
                d_beta_u.len(),
                d_beta_v.len(),
                p_total
            ));
        }
        let d_h_prime_u = self.family.x_deriv_kron.forward_mul(d_beta_u);
        let d_h_prime_v = self.family.x_deriv_kron.forward_mul(d_beta_v);
        let op = TransformationNormalD2hMatrixFreeOperator::new(
            Arc::clone(&self.family),
            Arc::clone(&self.h_prime),
            d_h_prime_u,
            d_h_prime_v,
        );
        Ok(Some(Arc::new(op) as Arc<dyn HyperOperator>))
    }
}

/// Matrix-free directional derivative of the CTN joint Hessian.
///
/// Encodes
///   `dH[v_dir] = -2 X_derivᵀ diag(w · (X_deriv v_dir) / h'³) X_deriv`
/// by caching the per-row weight kernel
///   `c_i = -2 w_i (X_deriv v_dir)_i / h'_i³`
/// at construction time. Each `mul_vec(w)` call costs `Θ(n (p_resp + p_cov))`
/// via the same Khatri-Rao primitives the joint Hessian matvec uses, so the
/// O(p²) dense `weighted_gram` build is never performed and stochastic
/// trace estimators (`MatrixFreeSpdOperator::trace_logdet_operator`) consume
/// the operator directly without materialization.
struct TransformationNormalDhMatrixFreeOperator {
    family: Arc<TransformationNormalFamily>,
    weight_kernel: Array1<f64>,
}

impl TransformationNormalDhMatrixFreeOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        h_prime: Arc<Array1<f64>>,
        d_h_prime: Array1<f64>,
    ) -> Self {
        let n = h_prime.len();
        debug_assert_eq!(d_h_prime.len(), n);
        let weights = family.weights.as_ref();
        // weight_kernel[i] = -2 · w_i · d_h_prime[i] / h_prime[i]³
        let weight_kernel = ndarray::Zip::from(weights.view())
            .and(&*h_prime)
            .and(&d_h_prime)
            .map_collect(|&w, &hp, &dhp| -2.0 * w * dhp / (hp * hp * hp));
        Self {
            family,
            weight_kernel,
        }
    }

    fn p_total(&self) -> usize {
        self.family.x_deriv_kron.ncols()
    }

    /// `(dH) · v = X_deriv^T diag(weight_kernel) X_deriv v`.
    ///
    /// Dense covariate: single fused (n × p_resp) buffer (see
    /// `fused_khatri_rao_weighted_gram_apply`). Sparse/operator covariate:
    /// fall through to `forward_mul → in-place scale → transpose_mul` on
    /// the cached `KroneckerDesign`. Two BLAS gemms either way.
    fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        if let Some(cov_dense) = self.family.covariate_design.as_dense_ref() {
            fused_khatri_rao_weighted_gram_apply(
                &self.family.response_deriv_basis,
                cov_dense,
                &self.weight_kernel,
                v,
            )
        } else {
            let mut image = self.family.x_deriv_kron.forward_mul(v);
            image *= &self.weight_kernel;
            self.family.x_deriv_kron.transpose_mul(&image)
        }
    }
}

impl HyperOperator for TransformationNormalDhMatrixFreeOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    /// Materialize `dH` as a dense p×p matrix using the analytic identity
    /// `dH = X_deriv^T diag(weight_kernel) X_deriv`.
    ///
    /// Reuses `KroneckerDesign::weighted_gram`, which performs the symmetric
    /// Khatri–Rao gram in `Θ(p_resp² · n · p_cov²)` BLAS work — strictly fewer
    /// FLOPs than `p` separate matvecs (which would each repeat both gemms
    /// without amortizing across the symmetric structure). Reached only when
    /// a legacy consumer goes through `DriftDerivResult::into_operator().to_dense()`.
    fn to_dense(&self) -> Array2<f64> {
        self.family
            .x_deriv_kron
            .weighted_gram(&self.weight_kernel, &self.family.policy)
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

/// Matrix-free second directional derivative of the CTN joint Hessian.
///
/// Encodes
///   `d²H[u, v] = 6 X_derivᵀ diag(w (X_deriv u)(X_deriv v) / h'⁴) X_deriv`
/// with the same factored apply pattern as `TransformationNormalDhMatrixFreeOperator`.
/// Used by the unified evaluator's second-order trace identities and second
/// directional drift evaluations on the outer Hessian.
struct TransformationNormalD2hMatrixFreeOperator {
    family: Arc<TransformationNormalFamily>,
    weight_kernel: Array1<f64>,
}

impl TransformationNormalD2hMatrixFreeOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        h_prime: Arc<Array1<f64>>,
        d_h_prime_u: Array1<f64>,
        d_h_prime_v: Array1<f64>,
    ) -> Self {
        let n = h_prime.len();
        debug_assert_eq!(d_h_prime_u.len(), n);
        debug_assert_eq!(d_h_prime_v.len(), n);
        let weights = family.weights.as_ref();
        // weight_kernel[i] = 6 · w_i · d_h_prime_u[i] · d_h_prime_v[i] / h_prime[i]⁴
        let weight_kernel = ndarray::Zip::from(weights.view())
            .and(&*h_prime)
            .and(&d_h_prime_u)
            .and(&d_h_prime_v)
            .map_collect(|&w, &hp, &dhp_u, &dhp_v| {
                let hp2 = hp * hp;
                6.0 * w * dhp_u * dhp_v / (hp2 * hp2)
            });
        Self {
            family,
            weight_kernel,
        }
    }

    fn p_total(&self) -> usize {
        self.family.x_deriv_kron.ncols()
    }

    /// `(d²H) · w = X_deriv^T diag(weight_kernel) X_deriv w`.
    ///
    /// Same fused dense-covariate path / KroneckerDesign fallback as the dH
    /// apply.
    fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        if let Some(cov_dense) = self.family.covariate_design.as_dense_ref() {
            fused_khatri_rao_weighted_gram_apply(
                &self.family.response_deriv_basis,
                cov_dense,
                &self.weight_kernel,
                v,
            )
        } else {
            let mut image = self.family.x_deriv_kron.forward_mul(v);
            image *= &self.weight_kernel;
            self.family.x_deriv_kron.transpose_mul(&image)
        }
    }
}

impl HyperOperator for TransformationNormalD2hMatrixFreeOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    /// Dense form via `KroneckerDesign::weighted_gram` — see
    /// `TransformationNormalDhMatrixFreeOperator::to_dense` for the
    /// FLOP-count rationale.
    fn to_dense(&self) -> Array2<f64> {
        self.family
            .x_deriv_kron
            .weighted_gram(&self.weight_kernel, &self.family.policy)
    }

    fn is_implicit(&self) -> bool {
        true
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

fn evaluate_response_derivative_basis(
    values: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    transform: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    for (i, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "response monotonicity grid value {i} is not finite: {value}"
            ));
        }
    }

    let dev_dim = transform.ncols();
    let p_resp = 2 + dev_dim;
    let mut resp_deriv = Array2::<f64>::zeros((values.len(), p_resp));
    resp_deriv.column_mut(1).fill(1.0);

    if dev_dim == 0 {
        return Ok(resp_deriv);
    }
    if knots.is_empty() {
        return Err(
            "response derivative grid needs knots when deviation columns are present".to_string(),
        );
    }
    let (raw_deriv_basis, _) = create_basis::<Dense>(
        values.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .map_err(|e| e.to_string())?;
    let raw_deriv = raw_deriv_basis.as_ref().clone();
    if raw_deriv.ncols() != transform.nrows() {
        return Err(format!(
            "response derivative transform shape mismatch: raw cols={} transform rows={}",
            raw_deriv.ncols(),
            transform.nrows()
        ));
    }
    let dev_deriv = raw_deriv.dot(transform);
    resp_deriv.slice_mut(s![.., 2..]).assign(&dev_deriv);
    Ok(resp_deriv)
}

fn transformation_monotonicity_response_grid(
    response: &Array1<f64>,
    knots: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    if response.is_empty() {
        return Err(
            "cannot build transformation monotonicity grid with no response values".to_string(),
        );
    }
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut values = Vec::with_capacity(
        TRANSFORMATION_RESPONSE_GRID_MAX_QUANTILES
            + knots.len() * (TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS + 1)
            + 4,
    );
    for (i, &value) in response.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("response[{i}] is not finite: {value}"));
        }
        min_y = min_y.min(value);
        max_y = max_y.max(value);
    }
    let mut sorted_response = response.to_vec();
    sorted_response.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted_response.len() == 1 {
        values.push(sorted_response[0]);
    } else {
        let quantiles = sorted_response
            .len()
            .min(TRANSFORMATION_RESPONSE_GRID_MAX_QUANTILES);
        for q in 0..quantiles {
            let idx = if quantiles == 1 {
                0
            } else {
                q * (sorted_response.len() - 1) / (quantiles - 1)
            };
            values.push(sorted_response[idx]);
        }
    }
    for &knot in knots.iter() {
        if knot.is_finite() {
            values.push(knot);
            min_y = min_y.min(knot);
            max_y = max_y.max(knot);
        }
    }
    let span = (max_y - min_y).abs().max(1.0);
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let base_values = values.clone();
    for window in base_values.windows(2) {
        let left = window[0];
        let right = window[1];
        let width = right - left;
        if width <= 1.0e-12 * span {
            continue;
        }
        for sidx in 1..TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS {
            let frac = sidx as f64 / TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS as f64;
            values.push(left + frac * width);
        }
    }
    let guard = TRANSFORMATION_TAIL_GUARD_FRACTION * span;
    values.push(min_y - guard);
    values.push(max_y + guard);

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12 * span);
    Ok(Array1::from_vec(values))
}

fn build_monotonicity_derivative_grid_kron(
    response: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    transform: &Array2<f64>,
    covariate_design: &DesignMatrix,
) -> Result<KroneckerDesign, String> {
    let response_grid = transformation_monotonicity_response_grid(response, knots)?;
    let response_deriv_grid =
        evaluate_response_derivative_basis(&response_grid, knots, degree, transform)?;
    // Build the operator factored: the small (n_grid × p_resp) response-side
    // factor and the unmodified covariate design. The implied virtual row
    // space is n_cov × n_grid — never materialized.
    KroneckerDesign::new_kronecker(response_deriv_grid, covariate_design.clone())
}

fn effective_response_num_internal_knots(
    config: &TransformationNormalConfig,
    n_obs: usize,
    p_cov: usize,
) -> usize {
    let sample_cap = (n_obs / 10).max(1);
    let min_internal = 1usize;
    let tensor_width_cap = (BASE_TRANSFORMATION_TENSOR_WIDTH + n_obs / 25)
        .min(LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH);
    let max_resp_cols_from_tensor =
        (tensor_width_cap / p_cov.max(1)).max(config.response_degree + 2);
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

fn assert_rowwise_kronecker_dimensions(n: usize, p_resp: usize, p_cov: usize, context: &str) {
    assert!(
        p_resp > 0 && p_cov > 0,
        "{context} rowwise Kronecker dimensions must be non-empty: n={n}, p_resp={p_resp}, p_cov={p_cov}"
    );
}

fn assert_no_rowwise_kronecker_materialization(n: usize, p_resp: usize, p_cov: usize) -> ! {
    let bytes = n
        .saturating_mul(p_resp)
        .saturating_mul(p_cov)
        .saturating_mul(std::mem::size_of::<f64>());
    panic!(
        "CTN KroneckerDesign must remain factored; refused persistent n x p_response x p_covariate materialization (n={n}, p_response={p_resp}, p_covariate={p_cov}, dense={:.1} MiB)",
        bytes as f64 / (1024.0 * 1024.0),
    );
}

// ---------------------------------------------------------------------------
// Kronecker-aware operator for biobank-scale tensor products
// ---------------------------------------------------------------------------

/// Active-set certificate cache for the cached `min_step_to_boundary` path on
/// the `Kronecker` variant. See `min_step_to_boundary_with_active_set` for the
/// math and validity proof. Caller is responsible for bumping `c_version`
/// whenever the underlying β projection changes; the certificate is otherwise
/// invariant in δ (the refresh recomputes the inactive bound from `c` only).
#[derive(Clone, Debug)]
struct KroneckerActiveSetCache {
    /// Pairs `(i, g)` whose `h_{i,g} - slack ≤ τ`. Empty before first refresh.
    active_pairs: Vec<(usize, usize)>,
    /// `min h_{i,g}` over inactive `(i,g)` (the smallest non-active feasibility
    /// margin); `+∞` when every pair is active or the cache is unrefreshed.
    min_inactive_margin: f64,
    /// `max_g ||r_g||₂`, response_grid invariant.
    max_response_norm: Option<f64>,
    /// `max_i ||X_cov[i,:]||₂`, covariate-design invariant (per family).
    max_covariate_norm: Option<f64>,
    /// Caller-managed monotone version stamp on the current `c` projection.
    c_version: u64,
    /// `c_version` value at the most recent refresh; equals `c_version` ⇒ fresh.
    cached_for_version: u64,
    /// Triplet fingerprint of the `c` projection at the most recent refresh.
    /// Used to detect β changes implicitly without a hash over the full
    /// `(n × p_resp)` matrix: corner samples `c[0,0]`, `c[mid,mid]`,
    /// `c[n-1,p_resp-1]`. If any differ, the cache is treated as stale.
    c_fingerprint: [f64; 3],
    /// Sentinel `(n, n_grid, p_resp, slack, dh_eps)` at refresh time, used to
    /// invalidate the cache if any of these change between calls.
    n: usize,
    n_grid: usize,
    p_resp: usize,
    slack: f64,
    dh_eps: f64,
}

impl KroneckerActiveSetCache {
    fn new() -> Self {
        Self {
            active_pairs: Vec::new(),
            min_inactive_margin: f64::INFINITY,
            max_response_norm: None,
            max_covariate_norm: None,
            c_version: 0,
            cached_for_version: u64::MAX, // distinct from c_version=0 ⇒ stale
            c_fingerprint: [f64::NAN; 3],
            n: 0,
            n_grid: 0,
            p_resp: 0,
            slack: f64::NAN,
            dh_eps: f64::NAN,
        }
    }

    fn fingerprint_of(c: ndarray::ArrayView2<'_, f64>) -> [f64; 3] {
        let n = c.nrows();
        let p = c.ncols();
        if n == 0 || p == 0 {
            return [0.0; 3];
        }
        let mid_i = n / 2;
        let mid_j = p / 2;
        [
            c[[0, 0]],
            c[[mid_i, mid_j]],
            c[[n - 1, p - 1]],
        ]
    }

    /// Mark the active-set certificate as stale because `c` (β projection)
    /// has changed. Bumps the monotone version stamp; the next call to
    /// `min_step_to_boundary_with_active_set` will full-scan to refresh.
    fn invalidate_for_new_beta(&mut self) {
        self.c_version = self.c_version.wrapping_add(1);
    }
}

/// Discriminated union over two factored representations of a Kronecker-shaped
/// design. Both variants compute `forward_mul` and `transpose_mul` from the
/// natural factor pair without ever materializing the full matrix.
///
/// `KhatriRao` is the row-wise Kronecker (face-splitting / Khatri–Rao) product
/// `A ⊙ B`: both factors share `n` rows, and each output row is the elementwise
/// Kronecker of the corresponding factor rows. Used for designs whose virtual
/// rows already correspond one-to-one with covariate rows (value and
/// derivative designs at observation points).
///
/// `Kronecker` is the full tensor product. The virtual row space is the
/// Cartesian product `n_cov × n_grid`; the two factors do not share a row
/// count and are never replicated. Used for the monotonicity grid, where each
/// covariate row is paired with every response grid point.
#[derive(Clone)]
enum KroneckerDesign {
    /// Row-wise Khatri–Rao product `A ⊙ B`.
    ///
    /// Element-wise definition (with `n` shared rows, `p_a` and `p_b` columns):
    /// ```text
    ///     (A ⊙ B)[i, a*p_b + b]  =  A[i, a] · B[i, b]
    /// ```
    /// Forward identity (used by `forward_mul`):
    /// ```text
    ///     ((A ⊙ B) β)[i] = Σ_{a,b} A[i,a] · B[i,b] · β_mat[a,b]
    ///                    = Σ_a A[i,a] · (B · β_mat[a, :])[i]
    /// ```
    /// where `β_mat[a, b] = β[a*p_b + b]` (row-major reshape into `p_a × p_b`).
    ///
    /// Storage: `O(n·p_a + storage(B))`. The dense `n × (p_a · p_b)`
    /// materialization is never built.
    KhatriRao {
        left: Array2<f64>,   // n × p_a
        right: DesignMatrix, // n × p_b
    },
    /// Full Kronecker product `R ⊗ C` with covariate-major row flattening.
    ///
    /// The virtual row space is the Cartesian product `n_cov × n_grid` with row
    /// index `i*n_grid + g` (covariate-major); columns use the same response-
    /// major ordering `a*p_cov + b` as `KhatriRao`. Defined elementwise:
    /// ```text
    ///     X[(i, g), (a, b)]  =  R[g, a] · C[i, b]
    /// ```
    /// where `R = response_grid` (n_grid × p_resp) and `C = covariate`
    /// (n_cov × p_cov).
    ///
    /// Forward identity (used by `forward_mul`):
    /// ```text
    ///     (X β)[i*n_grid + g]
    ///       = Σ_{a,b} R[g,a] · C[i,b] · β_mat[a,b]
    ///       = Σ_a R[g,a] · (C · β_matᵀ)[i, a]
    ///       = ((C · β_matᵀ) · Rᵀ)[i, g]
    /// ```
    /// with `β_mat[a, b] = β[a*p_cov + b]` (row-major reshape into
    /// `p_resp × p_cov`).
    ///
    /// Transpose identity (used by `transpose_mul`):
    /// ```text
    ///     (Xᵀ v)[(a, b)]
    ///       = Σ_{i,g} v[i*n_grid+g] · R[g,a] · C[i,b]
    ///       = Σ_i C[i,b] · (V · R)[i, a]            with V[i,g] = v[i*n_grid+g]
    ///       = (Cᵀ · (V · R)[:, a])[b]
    /// ```
    ///
    /// In exact arithmetic these identities are a strict reassociation of the
    /// materialized `Σ_{a,b}` sum; in IEEE-754 the factored and replicated
    /// forms agree to within the BLAS-accumulation rounding floor (~1e-15).
    /// Storage: `O(n_grid · p_resp + storage(C))` instead of the row-replicated
    /// `O(n_cov · n_grid · (p_resp + p_cov))`.
    Kronecker {
        /// (n_grid × p_resp) — small response-direction derivative basis on
        /// the monotonicity grid. Independent of covariate rows.
        response_grid: Array2<f64>,
        /// (n_cov × p_cov) — original covariate design, never replicated.
        covariate: DesignMatrix,
    },
}

impl KroneckerDesign {
    fn new_khatri_rao(left: &Array2<f64>, right: DesignMatrix) -> Result<Self, String> {
        if left.nrows() != right.nrows() {
            return Err(format!(
                "KroneckerDesign row mismatch: left={}, right={}",
                left.nrows(),
                right.nrows()
            ));
        }
        assert_rowwise_kronecker_dimensions(left.nrows(), left.ncols(), right.ncols(), "CTN");
        Ok(KroneckerDesign::KhatriRao {
            left: left.clone(),
            right,
        })
    }

    /// Construct the outer-factored variant used by the monotonicity grid: the
    /// virtual row space is the Cartesian product `n_cov × n_grid`, and the
    /// two factors never need to share a row count.
    fn new_kronecker(response_grid: Array2<f64>, covariate: DesignMatrix) -> Result<Self, String> {
        if response_grid.ncols() == 0 {
            return Err("Kronecker response_grid has zero columns".to_string());
        }
        if covariate.ncols() == 0 {
            return Err("Kronecker covariate has zero columns".to_string());
        }
        Ok(KroneckerDesign::Kronecker {
            response_grid,
            covariate,
        })
    }

    fn nrows(&self) -> usize {
        match self {
            KroneckerDesign::KhatriRao { left, .. } => left.nrows(),
            KroneckerDesign::Kronecker {
                response_grid,
                covariate,
            } => covariate.nrows() * response_grid.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            KroneckerDesign::KhatriRao { left, right } => left.ncols() * right.ncols(),
            KroneckerDesign::Kronecker {
                response_grid,
                covariate,
            } => response_grid.ncols() * covariate.ncols(),
        }
    }

    /// Compute `self · beta` where beta has length p_a * p_b.
    /// Returns an n-vector.
    fn forward_mul(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
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
            KroneckerDesign::Kronecker {
                response_grid,
                covariate,
            } => {
                let n_cov = covariate.nrows();
                let n_grid = response_grid.nrows();
                let p_resp = response_grid.ncols();
                let p_cov = covariate.ncols();
                debug_assert_eq!(beta.len(), p_resp * p_cov);
                let beta_mat = beta.view().into_shape_with_order((p_resp, p_cov)).unwrap();
                // inner[i, a] = covariate.apply(beta_mat[a, :])[i] — shape (n_cov × p_resp).
                let mut inner = Array2::<f64>::zeros((n_cov, p_resp));
                for a in 0..p_resp {
                    let cov_part = covariate.apply(&beta_mat.row(a).to_owned());
                    inner.column_mut(a).assign(&cov_part);
                }
                // result_2d = inner · response_grid^T  →  (n_cov × n_grid),
                // written directly into the row-major buffer that the returned
                // Array1 will own. Default Array2 layout is row-major C-order,
                // so flattening yields result[i*n_grid + g] = result_2d[i, g] —
                // exactly the covariate-major virtual-row layout.
                // response_grid.t() is a zero-copy transposed view; faer accepts
                // any positive-strided layout, so no temporary copy is made.
                let mut result_2d = Array2::<f64>::zeros((n_cov, n_grid));
                fast_ab_into(&inner, &response_grid.t(), &mut result_2d);
                result_2d
                    .into_shape_with_order((n_cov * n_grid,))
                    .expect("row-major Array2 flattens to Array1 of length n_cov*n_grid")
            }
        }
    }

    /// Streaming fraction-to-boundary reduction.
    ///
    /// Returns the smallest positive α for which there exists a virtual row
    /// `r` with `(self · δ)[r] < -dh_eps` and
    /// `(self · β)[r] + α · (self · δ)[r] = slack`, i.e.
    ///
    /// ```text
    ///     α = ((self · β)[r] - slack) / (-(self · δ)[r])
    /// ```
    ///
    /// minimized over all such `r`. Returns `f64::INFINITY` if no row violates
    /// the descent threshold.
    ///
    /// For the `Kronecker` variant the reduction streams over the
    /// `n_cov × n_grid` virtual rows from the factored representation: only an
    /// `n_cov × p_resp` projection (`C = X · β_matᵀ`, `D = X · δ_matᵀ`) and a
    /// per-thread `chunk_rows × n_grid` scratch buffer are held in memory, so
    /// the dense `n_cov · n_grid · 8 B` forward image is never materialized.
    ///
    /// The reduction is `f64::min`, which is associative for non-NaN inputs,
    /// so the parallel and serial reductions are bit-equivalent.
    fn min_step_to_boundary(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
        slack: f64,
        dh_eps: f64,
    ) -> f64 {
        match self {
            KroneckerDesign::KhatriRao { .. } => {
                // n observation rows: a single forward_mul pair fits in cache,
                // and the tight serial scan dominates.
                let h = self.forward_mul(beta);
                let dh = self.forward_mul(delta);
                let mut alpha = f64::INFINITY;
                for i in 0..h.len() {
                    let dval = dh[i];
                    if dval < -dh_eps {
                        let hit = (h[i] - slack) / (-dval);
                        if hit < alpha {
                            alpha = hit;
                        }
                    }
                }
                alpha
            }
            KroneckerDesign::Kronecker {
                response_grid,
                covariate,
            } => {
                let n = covariate.nrows();
                let p_resp = response_grid.ncols();
                let p_cov = covariate.ncols();
                debug_assert_eq!(beta.len(), p_resp * p_cov);
                debug_assert_eq!(delta.len(), p_resp * p_cov);
                let beta_mat = beta.view().into_shape_with_order((p_resp, p_cov)).unwrap();
                let delta_mat = delta.view().into_shape_with_order((p_resp, p_cov)).unwrap();
                // Project β and δ through the covariate operator once. Each
                // takes p_resp `apply` calls — identical cost to a single
                // `forward_mul`. Together: `n × p_resp` doubles ≈ 60 MiB at
                // biobank scale, vs the `2 · n × n_grid ≈ 3 GiB` the prior
                // pair of `forward_mul` outputs held simultaneously.
                let mut c = Array2::<f64>::zeros((n, p_resp));
                let mut d = Array2::<f64>::zeros((n, p_resp));
                for a in 0..p_resp {
                    let beta_row = beta_mat.row(a).to_owned();
                    let delta_row = delta_mat.row(a).to_owned();
                    c.column_mut(a).assign(&covariate.apply(&beta_row));
                    d.column_mut(a).assign(&covariate.apply(&delta_row));
                }
                Self::min_step_kronecker_reduce(response_grid, c.view(), d.view(), slack, dh_eps)
            }
        }
    }

    /// Streaming reduction over the `n_cov × n_grid` virtual rows from
    /// pre-computed `C = X · β_matᵀ` and `D = X · δ_matᵀ` projections.
    ///
    /// Splitting this out lets the inner-Newton outer loop cache `C` across
    /// successive `min_step_to_boundary` calls when only `β` (or only `δ`)
    /// has changed: a single `apply` pass per fresh column is enough to
    /// refresh the projection, instead of re-projecting both factors from
    /// scratch each time. The reduction itself is identical to the inline
    /// path in `min_step_to_boundary`, so the cached and un-cached paths
    /// agree to BLAS-rounding.
    ///
    /// `c` and `d` must each be `(n_cov × p_resp)` and share the column
    /// ordering with `response_grid` (`p_resp` columns). The active
    /// monotonicity certification at accepted iterates can call this
    /// directly with the freshly-projected pair so the full grid is scanned
    /// exactly once per accepted step.
    fn min_step_kronecker_reduce(
        response_grid: &Array2<f64>,
        c: ndarray::ArrayView2<'_, f64>,
        d: ndarray::ArrayView2<'_, f64>,
        slack: f64,
        dh_eps: f64,
    ) -> f64 {
        let n = c.nrows();
        let n_grid = response_grid.nrows();
        let p_resp = response_grid.ncols();
        debug_assert_eq!(c.ncols(), p_resp);
        debug_assert_eq!(d.nrows(), n);
        debug_assert_eq!(d.ncols(), p_resp);
        // Stream the (i, g) reduction in row chunks. For each chunk:
        //   H_chunk  = C[chunk] · Rᵀ      (m × n_grid)
        //   dH_chunk = D[chunk] · Rᵀ      (m × n_grid)
        // and reduce the fraction-to-boundary minimum locally before
        // dropping the buffers. Per chunk: 2 · m · n_grid · 8 B
        // ≈ 9 MiB at m = 1024 and biobank-scale n_grid.
        const CHUNK_ROWS: usize = 1024;
        let n_chunks = n.div_ceil(CHUNK_ROWS);
        let r_t = response_grid.t();
        (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * CHUNK_ROWS;
                let end = (start + CHUNK_ROWS).min(n);
                let m = end - start;
                let c_chunk = c.slice(s![start..end, ..]);
                let d_chunk = d.slice(s![start..end, ..]);
                let mut h_chunk = Array2::<f64>::zeros((m, n_grid));
                let mut dh_chunk = Array2::<f64>::zeros((m, n_grid));
                fast_ab_into(&c_chunk, &r_t, &mut h_chunk);
                fast_ab_into(&d_chunk, &r_t, &mut dh_chunk);
                let mut local = f64::INFINITY;
                for i_local in 0..m {
                    for g in 0..n_grid {
                        let dval = dh_chunk[[i_local, g]];
                        if dval < -dh_eps {
                            let hit = (h_chunk[[i_local, g]] - slack) / (-dval);
                            if hit < local {
                                local = hit;
                            }
                        }
                    }
                }
                local
            })
            .reduce(|| f64::INFINITY, f64::min)
    }

    /// Project a `(p_resp × p_cov)` row-major coefficient vector through the
    /// covariate factor of the `Kronecker` variant. Returns the
    /// `(n_cov × p_resp)` matrix `X · β_matᵀ`.
    ///
    /// Used by the cached `min_step_to_boundary` path: callers pass the
    /// freshly-projected `C` and `D` so the projection cost is paid once per
    /// fresh `β` / `δ` rather than once per outer iteration. Returns `None`
    /// for the `KhatriRao` variant since that branch streams over `n`
    /// observation rows directly and benefits no caching.
    fn project_kronecker_factor(&self, beta: &Array1<f64>) -> Option<Array2<f64>> {
        match self {
            KroneckerDesign::KhatriRao { .. } => None,
            KroneckerDesign::Kronecker {
                response_grid,
                covariate,
            } => {
                let n = covariate.nrows();
                let p_resp = response_grid.ncols();
                let p_cov = covariate.ncols();
                debug_assert_eq!(beta.len(), p_resp * p_cov);
                let beta_mat = beta.view().into_shape_with_order((p_resp, p_cov)).unwrap();
                let mut out = Array2::<f64>::zeros((n, p_resp));
                for a in 0..p_resp {
                    let row_a = beta_mat.row(a).to_owned();
                    out.column_mut(a).assign(&covariate.apply(&row_a));
                }
                Some(out)
            }
        }
    }

    /// Cached `min_step_to_boundary` for the `Kronecker` variant.
    ///
    /// Bit-equivalent to `min_step_to_boundary(β, δ, slack, dh_eps)` when
    /// `c = X · β_matᵀ` and `d = X · δ_matᵀ`, but reuses the projections
    /// supplied by the caller so a line-search probe that only changes one
    /// of `(β, δ)` need refresh just one factor with a single `apply` pass.
    /// Panics if invoked on a `KhatriRao` design — those callers should keep
    /// using the un-cached entry point since their `forward_mul` already
    /// streams over the `n` observation rows.
    fn min_step_to_boundary_with_projections(
        &self,
        c: ndarray::ArrayView2<'_, f64>,
        d: ndarray::ArrayView2<'_, f64>,
        slack: f64,
        dh_eps: f64,
    ) -> f64 {
        match self {
            KroneckerDesign::KhatriRao { .. } => panic!(
                "min_step_to_boundary_with_projections is only defined for the \
                 Kronecker variant; KhatriRao callers should use the un-cached \
                 streaming path"
            ),
            KroneckerDesign::Kronecker { response_grid, .. } => {
                Self::min_step_kronecker_reduce(response_grid, c, d, slack, dh_eps)
            }
        }
    }

    /// Maximum row L2-norm of the response-grid factor (`max_g ||r_g||₂`).
    ///
    /// Returns `None` for the `KhatriRao` variant — it has no separate response
    /// factor to summarize.
    fn max_response_norm(&self) -> Option<f64> {
        match self {
            KroneckerDesign::KhatriRao { .. } => None,
            KroneckerDesign::Kronecker { response_grid, .. } => {
                let mut m = 0.0_f64;
                for g in 0..response_grid.nrows() {
                    let row = response_grid.row(g);
                    let s2: f64 = row.iter().map(|v| v * v).sum();
                    m = m.max(s2.sqrt());
                }
                Some(m)
            }
        }
    }

    /// Maximum row L2-norm of the covariate factor (`max_i ||X_cov[i,:]||₂`).
    ///
    /// Streamed via `try_row_chunk` so sparse / operator-backed covariates
    /// stay chunkable. Returns `None` for the `KhatriRao` variant.
    fn max_covariate_norm(&self) -> Option<f64> {
        match self {
            KroneckerDesign::KhatriRao { .. } => None,
            KroneckerDesign::Kronecker { covariate, .. } => {
                let n = covariate.nrows();
                if n == 0 {
                    return Some(0.0);
                }
                let p = covariate.ncols();
                if let Some(dense) = covariate.as_dense_ref() {
                    let mut m = 0.0_f64;
                    for i in 0..n {
                        let row = dense.row(i);
                        let s2: f64 = row.iter().map(|v| v * v).sum();
                        m = m.max(s2.sqrt());
                    }
                    return Some(m);
                }
                let chunk_rows = crate::resource::rows_for_target_bytes(
                    8 * 1024 * 1024,
                    p.max(1),
                )
                .max(1);
                let mut m = 0.0_f64;
                let mut start = 0usize;
                while start < n {
                    let end = (start + chunk_rows).min(n);
                    let chunk = covariate
                        .try_row_chunk(start..end)
                        .expect("covariate.try_row_chunk for max-row-norm precompute");
                    for i in 0..(end - start) {
                        let row = chunk.row(i);
                        let s2: f64 = row.iter().map(|v| v * v).sum();
                        m = m.max(s2.sqrt());
                    }
                    start = end;
                }
                Some(m)
            }
        }
    }

    /// Cached `min_step_to_boundary` for the `Kronecker` variant with an
    /// active-set certificate fast path.
    ///
    /// Bit-equivalent to `min_step_to_boundary(β, δ, slack, dh_eps)` whenever
    /// the certificate accepts; otherwise refreshes the certificate by running
    /// a full grid scan and returns the same `α` the un-cached path would.
    /// Math (proof of correctness):
    ///   For each `(i, g)`: `h_{i,g} = r_g · c_i^T`, `d_{i,g} = r_g · d_i^T`.
    ///   Lipschitz bound `|d_{i,g}| ≤ ||r_g||₂ · ||ΔB||_F · ||X_cov[i,:]||₂`.
    ///   Let `L = max_g ||r_g||₂ · max_i ||X_cov[i,:]||₂ · ||ΔB||_F`.
    ///   `α_A` = min over active `(i,g)` with `d_{i,g}<-dh_eps` of
    ///     `(h_{i,g} - slack)/(-d_{i,g})`. If `m_inactive - α_A · L > slack`
    ///   then no inactive constraint can bind before `α_A`, so `α_A` is exact.
    fn min_step_to_boundary_with_active_set(
        &self,
        c: ndarray::ArrayView2<'_, f64>,
        d: ndarray::ArrayView2<'_, f64>,
        delta: &Array1<f64>,
        slack: f64,
        dh_eps: f64,
        cache: &mut KroneckerActiveSetCache,
    ) -> f64 {
        let response_grid = match self {
            KroneckerDesign::KhatriRao { .. } => panic!(
                "min_step_to_boundary_with_active_set is only defined for the \
                 Kronecker variant; KhatriRao callers should use the un-cached \
                 streaming path"
            ),
            KroneckerDesign::Kronecker { response_grid, .. } => response_grid,
        };
        let n = c.nrows();
        let n_grid = response_grid.nrows();
        let p_resp = response_grid.ncols();
        debug_assert_eq!(c.ncols(), p_resp);
        debug_assert_eq!(d.nrows(), n);
        debug_assert_eq!(d.ncols(), p_resp);

        // ||ΔB||_F: from the row-major (p_resp × p_cov) reshape of `delta`.
        let p_cov = delta.len() / p_resp.max(1);
        debug_assert_eq!(delta.len(), p_resp * p_cov);
        let delta_fro: f64 = delta.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Initialize / refresh the row-norm caches if missing.
        if cache.max_response_norm.is_none() {
            cache.max_response_norm = self.max_response_norm();
        }
        if cache.max_covariate_norm.is_none() {
            cache.max_covariate_norm = self.max_covariate_norm();
        }
        let max_response = cache.max_response_norm.unwrap_or(f64::INFINITY);
        let max_covariate = cache.max_covariate_norm.unwrap_or(f64::INFINITY);
        let lipschitz = max_response * max_covariate * delta_fro;

        // If the cache is stale (β changed), or its dimensions disagree with
        // the current projection shape, refresh by a full grid scan. We
        // fingerprint `c` by three corner samples — when the caller has
        // accepted a Newton step, β (and therefore `c`) changes and at least
        // one of those samples will differ.
        let fp = KroneckerActiveSetCache::fingerprint_of(c);
        let cache_fresh = cache.cached_for_version == cache.c_version
            && cache.n == n
            && cache.n_grid == n_grid
            && cache.p_resp == p_resp
            && (cache.slack - slack).abs() <= f64::EPSILON
            && (cache.dh_eps - dh_eps).abs() <= f64::EPSILON
            && cache.c_fingerprint == fp;

        if cache_fresh {
            // Active-set fast path: scan only the cached active set with the
            // fresh `d` projection. Compute α_A.
            let mut alpha_active = f64::INFINITY;
            for &(i, g) in &cache.active_pairs {
                let r_g = response_grid.row(g);
                let mut hv = 0.0_f64;
                let mut dv = 0.0_f64;
                for a in 0..p_resp {
                    hv += r_g[a] * c[[i, a]];
                    dv += r_g[a] * d[[i, a]];
                }
                if dv < -dh_eps {
                    let hit = (hv - slack) / (-dv);
                    if hit < alpha_active {
                        alpha_active = hit;
                    }
                }
            }
            // Certificate: m_inactive - α_A · L > slack ⇒ no inactive can bind
            // at any α ≤ α_A. The cached `min_inactive_margin = min (h_{i,g}
            // - slack)` over (i,g) ∉ A was computed in `refresh_active_set_cache`
            // already net of `slack`, so the strict-feasibility comparison
            // collapses to `m_inactive > α_A · L` (the comment-form expansion
            // would double-count slack). When α_A = +∞ (no active pair binds),
            // we still need to certify no inactive pair binds within the
            // caller's eventual `α_max ≤ 1.0` step (downstream callsite caps
            // at min(1.0, ...)), so the bound becomes `m_inactive > 1.0 · L`.
            let alpha_for_bound = if alpha_active.is_finite() {
                alpha_active
            } else {
                1.0
            };
            let bound_ok = cache.min_inactive_margin > alpha_for_bound * lipschitz;
            if bound_ok {
                return alpha_active;
            }
        }

        // Cache stale or certificate failed: full grid scan refreshes both
        // the answer AND the active set.
        let alpha_full = Self::min_step_kronecker_reduce(response_grid, c, d, slack, dh_eps);
        Self::refresh_active_set_cache(
            response_grid,
            c,
            slack,
            dh_eps,
            n,
            n_grid,
            p_resp,
            cache,
        );
        alpha_full
    }

    /// Recompute the active-set cache from the current `c` projection.
    /// `active_pairs` collects (i, g) with `h_{i,g} - slack ≤ τ`; the
    /// `min_inactive_margin` is the smallest `h_{i,g} - slack` over inactive
    /// pairs (= the tightest non-active constraint).
    fn refresh_active_set_cache(
        response_grid: &Array2<f64>,
        c: ndarray::ArrayView2<'_, f64>,
        slack: f64,
        dh_eps: f64,
        n: usize,
        n_grid: usize,
        p_resp: usize,
        cache: &mut KroneckerActiveSetCache,
    ) {
        // Compute H = c · response_gridᵀ in chunked passes; classify rows.
        const CHUNK_ROWS: usize = 1024;
        // Active-set tolerance: small relative-to-slack with an additive
        // floor. The certificate is valid for any τ ≥ 0; this scale keeps
        // the active set typically ≤ 0.1% of pairs at biobank scale.
        let scale = c.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let tau = (slack.abs() * 1e-3 + 1e-12).max(dh_eps * scale);

        let r_t = response_grid.t();
        let mut active_pairs: Vec<(usize, usize)> = Vec::new();
        let mut min_inactive = f64::INFINITY;
        let mut start = 0usize;
        while start < n {
            let end = (start + CHUNK_ROWS).min(n);
            let m = end - start;
            let c_chunk = c.slice(s![start..end, ..]);
            let mut h_chunk = Array2::<f64>::zeros((m, n_grid));
            fast_ab_into(&c_chunk, &r_t, &mut h_chunk);
            for i_local in 0..m {
                let i = start + i_local;
                for g in 0..n_grid {
                    let h = h_chunk[[i_local, g]];
                    let margin = h - slack;
                    if margin <= tau {
                        active_pairs.push((i, g));
                    } else if margin < min_inactive {
                        min_inactive = margin;
                    }
                }
            }
            start = end;
        }

        cache.active_pairs = active_pairs;
        cache.min_inactive_margin = min_inactive;
        cache.n = n;
        cache.n_grid = n_grid;
        cache.p_resp = p_resp;
        cache.slack = slack;
        cache.dh_eps = dh_eps;
        cache.c_fingerprint = KroneckerActiveSetCache::fingerprint_of(c);
        cache.cached_for_version = cache.c_version;
    }

    /// Compute `self^T · v` where v is an n-vector.
    /// Returns a (p_a * p_b)-vector.
    fn transpose_mul(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let n = left.nrows();
                let pa = left.ncols();
                let pb = right.ncols();
                debug_assert_eq!(v.len(), n);
                if let Some(right_dense) = right.as_dense_ref() {
                    let weighted_left = weight_rows(left, v);
                    let blocks = fast_atb(right_dense, &weighted_left).reversed_axes();
                    let mut out = Array1::<f64>::zeros(pa * pb);
                    for j in 0..pa {
                        out.slice_mut(s![j * pb..(j + 1) * pb])
                            .assign(&blocks.row(j));
                    }
                    return out;
                }
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
            KroneckerDesign::Kronecker {
                response_grid,
                covariate,
            } => {
                let n_cov = covariate.nrows();
                let n_grid = response_grid.nrows();
                let p_resp = response_grid.ncols();
                let p_cov = covariate.ncols();
                debug_assert_eq!(v.len(), n_cov * n_grid);
                // v reshaped as V (n_cov × n_grid) row-major view — matches the
                // covariate-major flattening v[i*n_grid + g] = V[i, g]. Faer
                // accepts the zero-copy view, so V is never duplicated.
                let v_mat = v.view().into_shape_with_order((n_cov, n_grid)).unwrap();
                // tmp = V · response_grid  →  (n_cov × p_resp).
                let mut tmp = Array2::<f64>::zeros((n_cov, p_resp));
                fast_ab_into(&v_mat, response_grid, &mut tmp);
                // For each a, out[a*p_cov..(a+1)*p_cov] = covariate^T · tmp[:, a].
                let mut out = Array1::<f64>::zeros(p_resp * p_cov);
                for a in 0..p_resp {
                    let col_a = tmp.column(a).to_owned();
                    let block = covariate.apply_transpose(&col_a);
                    out.slice_mut(s![a * p_cov..(a + 1) * p_cov]).assign(&block);
                }
                out
            }
        }
    }

    /// Compute `self^T · diag(w) · self` (weighted Gram).
    ///
    /// Thin wrapper over `weighted_cross_with(self, self, ...)`. Callers thread
    /// a real `ResourcePolicy` so chunk sizing matches the surrounding workload.
    fn weighted_gram(&self, w: &Array1<f64>, policy: &ResourcePolicy) -> Array2<f64> {
        self.weighted_cross_with(w.view(), self, policy)
            .expect("validated KroneckerDesign weighted Gram dimensions")
    }

    /// Compute `self^T · diag(w) · other` while keeping rowwise-Kronecker
    /// designs in factored form. Returns a dense (pa*pb) x (pc*pd) block matrix.
    pub(crate) fn weighted_cross_with(
        &self,
        weights: ndarray::ArrayView1<'_, f64>,
        other: &KroneckerDesign,
        policy: &ResourcePolicy,
    ) -> Result<Array2<f64>, String> {
        if matches!(self, KroneckerDesign::Kronecker { .. })
            || matches!(other, KroneckerDesign::Kronecker { .. })
        {
            return Err(
                "KroneckerDesign::weighted_cross_with is not supported for the Kronecker \
                 monotonicity-grid variant: the virtual row space (n_cov × n_grid) does not \
                 align with weights of the underlying observations, and no caller in the \
                 transformation-normal family computes a weighted Gram on this design."
                    .to_string(),
            );
        }
        match (self, other) {
            (
                KroneckerDesign::KhatriRao { left: a, right: b },
                KroneckerDesign::KhatriRao { left: c, right: d },
            ) => {
                // If both covariate sides are dense, stay fully factored.
                if let (Some(b_dense), Some(d_dense)) = (b.as_dense_ref(), d.as_dense_ref()) {
                    return factored_weighted_cross(a, b_dense, weights, c, d_dense, policy);
                }
                // Fallback: operator-backed covariate side — iterate (a, c)
                // pairs and let the operator handle the B^T diag(w) D block.
                let n = weights.len();
                let pa = a.ncols();
                let pc = c.ncols();
                let pb = b.ncols();
                let pd = d.ncols();
                if a.nrows() != n || b.nrows() != n || c.nrows() != n || d.nrows() != n {
                    return Err(format!(
                        "KroneckerDesign::weighted_cross_with row mismatch: weights={n}, \
                         a={}, b={}, c={}, d={}",
                        a.nrows(),
                        b.nrows(),
                        c.nrows(),
                        d.nrows()
                    ));
                }
                let mut out = Array2::<f64>::zeros((pa * pb, pc * pd));
                let mut pair_weights = Array1::<f64>::zeros(n);
                for ia in 0..pa {
                    let a_col = a.column(ia);
                    for ic in 0..pc {
                        let c_col = c.column(ic);
                        for r in 0..n {
                            pair_weights[r] = weights[r] * a_col[r] * c_col[r];
                        }
                        // Route through the chunked DesignMatrix helper so the
                        // operator-backed covariate factors stay row-chunkable
                        // and never materialize n × p_cov in one shot.
                        let block =
                            chunked_weighted_bt_d_designmatrix(b, pair_weights.view(), d, policy)?;
                        out.slice_mut(s![ia * pb..(ia + 1) * pb, ic * pd..(ic + 1) * pd])
                            .assign(&block);
                    }
                }
                Ok(out)
            }
            _ => unreachable!("Kronecker cross-with case is rejected by the early return above"),
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
        // The `LinearOperator` trait fixes the signature, so this entry point
        // defaults the resource policy. Internal callers in this file go
        // through `weighted_gram` directly with their own policy.
        let policy = ResourcePolicy::default_library();
        Ok(self.weighted_gram(weights, &policy))
    }
}

// KroneckerDesign contains owned data + DesignMatrix (which is Send+Sync),
// so it is safe to send/share across threads.
unsafe impl Send for KroneckerDesign {}
unsafe impl Sync for KroneckerDesign {}

impl DenseDesignOperator for KroneckerDesign {
    fn row_chunk_into(
        &self,
        rows: std::ops::Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "KroneckerDesign::row_chunk_into shape mismatch",
            });
        }
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                assert_rowwise_kronecker_dimensions(
                    rows.end.saturating_sub(rows.start),
                    left.ncols(),
                    right.ncols(),
                    "CTN row chunk",
                );
                let left_chunk = left.slice(s![rows.clone(), ..]).to_owned();
                let right_chunk = right.try_row_chunk(rows)?;
                out.assign(&rowwise_kronecker(&left_chunk, &right_chunk));
            }
            KroneckerDesign::Kronecker { .. } => {
                panic!(
                    "KroneckerDesign::row_chunk_into is not supported on the Kronecker \
                     monotonicity-grid design: it would materialize the n_cov*n_grid × p_total \
                     row-replicated form (the very allocation this variant exists to avoid). \
                     Use forward_mul / transpose_mul instead."
                );
            }
        }
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                assert_no_rowwise_kronecker_materialization(
                    left.nrows(),
                    left.ncols(),
                    right.ncols(),
                );
            }
            KroneckerDesign::Kronecker {
                response_grid,
                covariate,
            } => {
                assert_no_rowwise_kronecker_materialization(
                    covariate.nrows() * response_grid.nrows(),
                    response_grid.ncols(),
                    covariate.ncols(),
                );
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
    offset: &Array1<f64>,
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

    // Target: for the intercept row (j=0),
    //         Θ[0,:] · cov[i,:] = -μ(x_i)/τ(x_i) - offset[i]
    //         so h_i = offset[i] + Θ[0,:]·cov[i,:] + y_i Θ[1,:]·cov[i,:]
    //         starts at (y_i - μ(x_i))/τ(x_i).
    //         For the linear row (j=1), Θ[1,:] · cov[i,:] = 1/τ(x_i).
    //         For deviation rows (j≥2), Θ[j,:] = 0.

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
        let inv_tau = 1.0 / tau;
        target_intercept[i] = -ws.location[i] * inv_tau - offset[i];
        target_slope[i] = inv_tau;
    }

    let projection_log_lambdas = Array1::zeros(covariate_penalties.len());
    let zero_offset = Array1::zeros(n);
    // Use the minimum ridge that `solve_penalizedweighted_projection` keeps as
    // its own numerical floor (1e-12). A larger ridge here biases the
    // projection and prevents the warm start from exactly absorbing the
    // offset into the affine seed even when X'WX is perfectly conditioned.
    let coeff_int = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &target_intercept,
        weights,
        covariate_penalties,
        &projection_log_lambdas,
        1e-12,
    )?;
    let coeff_slope = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &target_slope,
        weights,
        covariate_penalties,
        &projection_log_lambdas,
        1e-12,
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
        let mat_op = op.as_materializable().ok_or_else(|| {
            crate::terms::basis::BasisError::InvalidInput(format!(
                "covariate psi operator for axis {axis} does not support dense materialization"
            ))
        })?;
        mat_op.materialize_first(deriv.implicit_axis)
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
            let mat_op = op.as_materializable().ok_or_else(|| {
                crate::terms::basis::BasisError::InvalidInput(format!(
                    "covariate psi operator for axes {axis_d},{axis_e} does not support dense materialization"
                ))
            })?;
            if deriv_d.implicit_axis == self.cov_deriv(axis_e)?.implicit_axis {
                return mat_op.materialize_second_diag(deriv_d.implicit_axis);
            }
            return mat_op.materialize_second_cross(
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
        let cov_design_dense = self.covariate_design.as_dense_ref();
        let mut out = Array2::<f64>::zeros((p_left_resp * p_cov, p_right_resp * p_cov));
        for a in 0..p_left_resp {
            for b in 0..p_right_resp {
                let mut pair_weights = Array1::<f64>::zeros(n);
                for i in 0..n {
                    pair_weights[i] =
                        weights[i] * left_resp_basis[[i, a]] * right_resp_basis[[i, b]];
                }
                let block = if let (Some(cov_design), Some(cov_psi)) =
                    (cov_design_dense, cov_psi_dense)
                {
                    weighted_crossprod_dense(cov_design, &pair_weights, cov_psi)
                } else if let Some(cov_psi) = cov_psi_dense {
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
    fn transformation_normal_uses_compact_gaussian_outer_seeding() {
        let psi = array![0.15, -0.10];
        let (family, _, _, _) = toy_family_and_derivatives(&psi);
        let seed_config = family.outer_seed_config(6);
        assert_eq!(seed_config.bounds, (-12.0, 12.0));
        assert_eq!(seed_config.max_seeds, 1);
        assert_eq!(seed_config.seed_budget, 1);
        assert_eq!(seed_config.screen_max_inner_iterations, 2);
        assert_eq!(
            seed_config.risk_profile,
            crate::seeding::SeedRiskProfile::Gaussian
        );
        assert_eq!(seed_config.num_auxiliary_trailing, 0);
    }

    #[test]
    fn max_feasible_step_size_uses_cached_kronecker_reduction() {
        // End-to-end equivalence check for the production line-search caller.
        // `max_feasible_step_size` projects β and δ through the covariate factor
        // once and routes the monotonicity-grid reduction through
        // `min_step_to_boundary_with_projections`; this test confirms the
        // wired path matches the un-cached baseline exactly on a small CTN
        // configuration (toy family with a `Kronecker` grid design).
        let psi = array![0.15, -0.10];
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        // Sanity: the toy fixture really uses the Kronecker grid variant the
        // cached path was built for; if anything ever rewires this design to
        // KhatriRao the production caller must keep working via fallback, so
        // the assertion documents the precondition rather than gating it.
        assert!(
            matches!(
                family.x_deriv_grid_kron,
                KroneckerDesign::Kronecker { .. }
            ),
            "toy family must keep the Kronecker grid variant for cached-path coverage"
        );
        // δ direction with a negative leading h' contribution so the grid
        // reduction binds above ε rather than returning +∞ (which would still
        // be bit-equivalent but would not exercise the streaming reduction).
        let delta = array![-0.20, 0.05, -0.10, 0.05];

        // Production path (cached internally).
        let block_states = vec![state.clone()];
        let alpha_prod = family
            .max_feasible_step_size(&block_states, 0, &delta)
            .expect("toy max_feasible_step_size returns Ok");

        // Hand-rolled un-cached baseline that mirrors the pre-wiring
        // implementation: both designs use `min_step_to_boundary` directly,
        // with no projection caching. If the cached and un-cached reductions
        // ever diverge (e.g. a chunk-size change), this test fails before the
        // production line search silently picks up the regression.
        let beta = &block_states[0].beta;
        let alpha_obs_uncached = family
            .x_deriv_kron
            .min_step_to_boundary(beta, &delta, 0.0, 1e-14);
        let alpha_grid_uncached = family.x_deriv_grid_kron.min_step_to_boundary(
            beta,
            &delta,
            TRANSFORMATION_MONOTONICITY_EPS,
            1e-14,
        );
        let alpha_max_uncached = 1.0_f64.min(alpha_obs_uncached).min(alpha_grid_uncached);
        let tau = 0.995;
        let alpha_safe_uncached = tau * alpha_max_uncached;
        let expected = if alpha_safe_uncached < 1.0 {
            Some(alpha_safe_uncached)
        } else {
            None
        };

        assert_eq!(
            alpha_prod, expected,
            "wired max_feasible_step_size must match the un-cached baseline bit-for-bit \
             (cached α = {alpha_prod:?}, un-cached α = {expected:?})"
        );
    }

    #[test]
    fn kronecker_min_step_to_boundary_cached_matches_uncached() {
        // Small CTN-like Kronecker design: covariate factor with n_cov=6 rows,
        // p_cov=3 columns and a response monotonicity grid with n_grid=5 rows,
        // p_resp=4 columns. The values are chosen so several virtual rows hit
        // the descent threshold for a non-trivial reduction.
        let covariate = DesignMatrix::Dense(DenseDesignMatrix::from(array![
            [1.0, 0.40, -0.10],
            [1.0, 0.55, 0.05],
            [1.0, -0.20, 0.30],
            [1.0, 0.10, -0.40],
            [1.0, 0.70, 0.15],
            [1.0, -0.35, 0.25],
        ]));
        let response_grid = array![
            [1.0, 0.10, -0.05, 0.20],
            [1.0, 0.30, 0.20, -0.10],
            [1.0, -0.15, 0.40, 0.05],
            [1.0, 0.45, -0.20, 0.30],
            [1.0, -0.30, 0.05, 0.40],
        ];
        let design = KroneckerDesign::new_kronecker(response_grid.clone(), covariate.clone())
            .expect("toy Kronecker design");
        // β positive on the leading response coordinate so h'(grid) starts
        // safely above the slack threshold; δ has a negative leading entry to
        // drive at least one virtual row toward the boundary. Layout follows
        // the row-major `(p_resp × p_cov)` reshape used by `forward_mul`.
        let beta = array![
            0.80, 0.10, -0.05, // a = 0
            0.20, 0.05, 0.20, // a = 1
            0.00, -0.10, 0.10, // a = 2
            -0.05, 0.15, 0.00, // a = 3
        ];
        let delta = array![
            -0.40, 0.05, 0.10, // a = 0
            -0.15, -0.10, -0.05, // a = 1
            0.05, 0.00, 0.05, // a = 2
            0.00, -0.10, 0.05, // a = 3
        ];
        let slack = 1e-3;
        let dh_eps = 1e-14;

        let alpha_uncached = design.min_step_to_boundary(&beta, &delta, slack, dh_eps);
        assert!(
            alpha_uncached.is_finite() && alpha_uncached > 0.0,
            "un-cached α must be a finite positive bound for this fixture: {alpha_uncached}"
        );

        // Cached path: project β and δ once via the helper, hand the (n × p_resp)
        // factors to the cached entry point.
        let c = design
            .project_kronecker_factor(&beta)
            .expect("Kronecker variant projects β");
        let d = design
            .project_kronecker_factor(&delta)
            .expect("Kronecker variant projects δ");
        let alpha_cached =
            design.min_step_to_boundary_with_projections(c.view(), d.view(), slack, dh_eps);
        assert_eq!(
            alpha_cached, alpha_uncached,
            "cached min_step_to_boundary_with_projections must be bit-equivalent to the un-cached path"
        );

        // Reusing C across a fresh δ′ (the line-search reuse pattern) must
        // still match the un-cached call with that δ′.
        let delta_prime = &delta * 0.5;
        let d_prime = design
            .project_kronecker_factor(&delta_prime)
            .expect("Kronecker variant projects δ′");
        let alpha_cached_prime = design.min_step_to_boundary_with_projections(
            c.view(),
            d_prime.view(),
            slack,
            dh_eps,
        );
        let alpha_uncached_prime = design.min_step_to_boundary(&beta, &delta_prime, slack, dh_eps);
        assert_eq!(
            alpha_cached_prime, alpha_uncached_prime,
            "cached path with reused C and refreshed D must equal the un-cached path"
        );
    }

    #[test]
    fn warm_start_absorbs_offset_into_affine_seed() {
        let response = array![2.0, 5.0];
        let response_val_basis = array![[1.0, 2.0], [1.0, 5.0]];
        let response_deriv_basis = array![[0.0, 1.0], [0.0, 1.0]];
        let weights = array![1.0, 1.0];
        let offset = array![0.7, 0.7];
        let covariate_design = DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [1.0]]));
        let warm_start = TransformationWarmStart {
            location: array![1.0, 1.0],
            scale: array![2.0, 2.0],
        };
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            response_val_basis,
            response_deriv_basis,
            vec![],
            Array1::zeros(0),
            1,
            Array2::zeros((0, 0)),
            &weights,
            &offset,
            covariate_design,
            vec![],
            &TransformationNormalConfig {
                double_penalty: false,
                ..TransformationNormalConfig::default()
            },
            Some(&warm_start),
        )
        .expect("transformation family");

        let (h, h_prime) = family.compute_h_and_h_prime(&family.initial_beta);
        let expected_h = array![0.5, 2.0];
        let expected_h_prime = array![0.5, 0.5];

        for i in 0..expected_h.len() {
            assert!(
                (h[i] - expected_h[i]).abs() < 1e-12,
                "h[{i}] mismatch: got {}, expected {}",
                h[i],
                expected_h[i]
            );
            assert!(
                (h_prime[i] - expected_h_prime[i]).abs() < 1e-12,
                "h_prime[{i}] mismatch: got {}, expected {}",
                h_prime[i],
                expected_h_prime[i]
            );
        }

        assert_eq!(response.len(), family.n_obs());
    }

    #[test]
    fn kronecker_dense_fast_paths_match_dense_materialization() {
        let left = array![[1.0, -0.4], [0.5, 0.3], [-0.2, 0.9], [1.1, -0.7],];
        let right = array![
            [0.2, 1.0, -0.3],
            [0.4, -0.5, 0.8],
            [0.7, 0.1, 0.6],
            [-0.2, 0.9, 0.5],
        ];
        let weights = array![0.7, 1.4, 0.9, 1.2];
        let v = array![0.6, -0.3, 0.5, 0.8];
        let kron = KroneckerDesign::new_khatri_rao(
            &left,
            DesignMatrix::Dense(DenseDesignMatrix::from(right.clone())),
        )
        .expect("kronecker design");

        let dense = rowwise_kronecker(&left, &right);
        let expected_transpose = dense.t().dot(&v);
        let expected_gram = fast_atb(&weight_rows(&dense, &weights), &dense);

        let got_transpose = kron.transpose_mul(&v);
        let got_gram = kron.weighted_gram(&weights, &ResourcePolicy::default_library());

        let transpose_err = (&got_transpose - &expected_transpose)
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        let gram_err = (&got_gram - &expected_gram)
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        assert!(
            transpose_err < 1e-10,
            "Kronecker transpose fast path mismatch: max_abs={transpose_err}"
        );
        assert!(
            gram_err < 1e-10,
            "Kronecker weighted Gram fast path mismatch: max_abs={gram_err}"
        );
    }

    #[test]
    fn large_samples_allow_richer_response_basis_than_small_samples() {
        let config = TransformationNormalConfig::default();
        let small = effective_response_num_internal_knots(&config, 40, 20);
        let large = effective_response_num_internal_knots(&config, 4000, 20);
        assert!(large >= small);
        assert!(
            large > small,
            "large-sample tensor cap should relax the small-sample response bottleneck"
        );
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

    #[test]
    fn ctn_joint_hessian_workspace_matvec_matches_dense() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();

        let dense = family
            .exact_newton_joint_hessian(std::slice::from_ref(&state))
            .expect("dense joint Hessian build")
            .expect("dense joint Hessian present");
        assert_eq!(dense.nrows(), p);
        assert_eq!(dense.ncols(), p);

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");

        // Diagonal must agree element-wise (matrix-free pre-square path vs. dense gram).
        let diag_op = workspace
            .hessian_diagonal()
            .expect("diagonal call")
            .expect("diagonal present");
        assert_eq!(diag_op.len(), p);
        for i in 0..p {
            let want = dense[[i, i]];
            let got = diag_op[i];
            assert!(
                (want - got).abs() <= 1e-12 * want.abs().max(1.0) + 1e-12,
                "diagonal mismatch at {i}: dense={want:.6e}, workspace={got:.6e}"
            );
        }

        // Hessian-vector product must agree with dense H · v across a few
        // randomly chosen directions (deterministic seed for stability).
        let directions = vec![
            Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.3, -0.7, 0.5, -0.2]),
            Array1::from_vec(vec![-0.42, 0.11, 0.93, 0.05]),
        ];
        for (k, v) in directions.iter().enumerate() {
            assert_eq!(v.len(), p);
            let want = dense.dot(v);
            let got = workspace
                .hessian_matvec(v)
                .expect("matvec call")
                .expect("matvec present");
            assert_eq!(got.len(), p);
            for i in 0..p {
                let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "matvec[{k}, {i}] mismatch: dense={:.6e}, workspace={:.6e}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn ctn_coefficient_hessian_cost_uses_dense_for_small_problems() {
        // Toy family: n=4, p_resp=2, p_cov=2 → p_total=4. The matrix-free
        // gate `use_joint_matrix_free_path(4, 4)` returns false (well below
        // every threshold), so the override must report the dense Khatri–Rao
        // gram cost n·(p_resp·p_cov)² = 4·16 = 64.
        let psi = array![0.15, -0.10];
        let (family, _, _, _) = toy_family_and_derivatives(&psi);
        let n = family.response_val_basis.nrows() as u64;
        let p_resp = family.response_val_basis.ncols() as u64;
        let p_cov = family.covariate_design.ncols() as u64;
        assert!(!crate::custom_family::use_joint_matrix_free_path(
            (p_resp * p_cov) as usize,
            n as usize,
        ));
        let p_total = p_resp * p_cov;
        let expected_dense = n * p_total * p_total;
        assert_eq!(family.coefficient_hessian_cost(&[]), expected_dense);
    }

    #[test]
    fn ctn_coefficient_hessian_cost_switches_to_matvec_when_matrix_free_active() {
        // p_resp=2, p_cov=256 → p_total=512 ≥ JOINT_MATRIX_FREE_MIN_DIM, so
        // matrix-free is ALWAYS active for any n. The override must report the
        // per-Hv matvec cost n·(p_resp + p_cov), not the dense p² gram.
        // n=8 keeps the test allocation small (~16 KB for covariate_design).
        let n = 8usize;
        let p_cov = 256usize;
        let mut response_val_basis = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            response_val_basis[[i, 0]] = 1.0;
            response_val_basis[[i, 1]] = i as f64;
        }
        let mut response_deriv_basis = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            response_deriv_basis[[i, 1]] = 1.0;
        }
        let weights = Array1::from_elem(n, 1.0);
        let offset = Array1::zeros(n);
        let cov_design = Array2::<f64>::zeros((n, p_cov));
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
        .expect("matrix-free-eligible CTN family");
        let p_resp = family.response_val_basis.ncols() as u64;
        let actual_p_cov = family.covariate_design.ncols() as u64;
        let p_total = p_resp * actual_p_cov;
        assert!(crate::custom_family::use_joint_matrix_free_path(
            p_total as usize,
            n,
        ));
        let expected_matvec = (n as u64) * (p_resp + actual_p_cov);
        assert_eq!(family.coefficient_hessian_cost(&[]), expected_matvec);
        // Sanity: the matrix-free cost is dramatically smaller than the dense
        // would have been (the whole point of branching).
        let dense_cost = (n as u64) * p_total * p_total;
        assert!(expected_matvec < dense_cost / 100);
    }

    #[test]
    fn ctn_joint_hessian_workspace_dh_operator_matches_dense() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let d_beta = array![0.07, -0.04, 0.21, 0.08];
        assert_eq!(d_beta.len(), p);

        let dense_dh = family
            .exact_newton_joint_hessian_directional_derivative(
                std::slice::from_ref(&state),
                &d_beta,
            )
            .expect("dense dH build")
            .expect("dense dH present");

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let dh_op = workspace
            .directional_derivative_operator(&d_beta)
            .expect("dH operator call")
            .expect("dH operator present");

        let probes = vec![
            Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.3, -0.7, 0.5, -0.2]),
            Array1::from_vec(vec![-0.42, 0.11, 0.93, 0.05]),
        ];
        for (k, w) in probes.iter().enumerate() {
            assert_eq!(w.len(), p);
            let want = dense_dh.dot(w);
            let got = dh_op.mul_vec(w);
            assert_eq!(got.len(), p);
            for i in 0..p {
                let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "dH op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn ctn_joint_hessian_workspace_d2h_operator_matches_dense() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let dir_u = array![0.02, -0.01, 0.03, 0.015];
        let dir_v = array![-0.01, 0.02, 0.01, -0.025];

        let dense_d2h = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                std::slice::from_ref(&state),
                &dir_u,
                &dir_v,
            )
            .expect("dense d2H build")
            .expect("dense d2H present");

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let d2h_op = workspace
            .second_directional_derivative_operator(&dir_u, &dir_v)
            .expect("d2H operator call")
            .expect("d2H operator present");

        let probes = vec![
            Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.3, -0.7, 0.5, -0.2]),
            Array1::from_vec(vec![-0.42, 0.11, 0.93, 0.05]),
        ];
        for (k, w) in probes.iter().enumerate() {
            assert_eq!(w.len(), p);
            let want = dense_d2h.dot(w);
            let got = d2h_op.mul_vec(w);
            assert_eq!(got.len(), p);
            for i in 0..p {
                let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "d2H op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn ctn_exact_newton_joint_gradient_evaluation_matches_evaluate() {
        // The joint-Newton inner solver prefers
        // `exact_newton_joint_gradient_evaluation` over `evaluate()` to refresh
        // the gradient between cycles. Lock in that the override returns
        // exactly the same log-likelihood and flat gradient that the dense
        // path produces (up to floating-point summation order).
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();

        let eval = family
            .evaluate(std::slice::from_ref(&state))
            .expect("evaluate must succeed on the toy fixture");
        let want_ll = eval.log_likelihood;
        let want_grad = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient.clone(),
            _ => panic!("CTN must report an ExactNewton block working set"),
        };
        assert_eq!(want_grad.len(), p);

        let gradient_eval = family
            .exact_newton_joint_gradient_evaluation(
                std::slice::from_ref(&state),
                &[spec.clone()],
            )
            .expect("gradient-only call")
            .expect("gradient-only result must be present");
        assert!(
            (want_ll - gradient_eval.log_likelihood).abs() <= 1e-12 * want_ll.abs().max(1.0) + 1e-12,
            "log-likelihood mismatch: evaluate={:.6e}, gradient-only={:.6e}",
            want_ll,
            gradient_eval.log_likelihood,
        );
        assert_eq!(gradient_eval.gradient.len(), p);
        for i in 0..p {
            let tol = 1e-12 * want_grad[i].abs().max(1.0) + 1e-12;
            assert!(
                (want_grad[i] - gradient_eval.gradient[i]).abs() <= tol,
                "gradient mismatch at {i}: evaluate={:.6e}, gradient-only={:.6e}",
                want_grad[i],
                gradient_eval.gradient[i],
            );
        }
    }

    #[test]
    fn fused_khatri_rao_weighted_gram_apply_matches_explicit_dense() {
        // Hand-constructed oracle: build (A ⊙ B) explicitly, compute
        // X^T diag(w) X · v, and compare to the fused helper. Locks in the
        // math (factored vs replicated) without going through the workspace.
        let left = array![
            [1.0, 0.5, -0.2],
            [-0.3, 1.1, 0.4],
            [0.2, -0.7, 0.9],
            [0.6, 0.3, -0.5],
            [1.4, -0.1, 0.7],
        ];
        let right = array![
            [2.0, -1.0],
            [-0.5, 0.8],
            [1.2, 0.3],
            [-0.7, 1.1],
            [0.4, -0.6]
        ];
        let weights = array![0.9, 1.3, 0.7, 1.1, 0.5];
        let v = array![0.1, -0.2, 0.3, -0.4, 0.5, -0.6];
        let n = left.nrows();
        let p_resp = left.ncols();
        let p_cov = right.ncols();
        assert_eq!(weights.len(), n);
        assert_eq!(v.len(), p_resp * p_cov);
        // Replicated reference X[i, a*p_cov + b] = A[i, a] · B[i, b].
        let mut x_rep = Array2::<f64>::zeros((n, p_resp * p_cov));
        for i in 0..n {
            for a in 0..p_resp {
                for b in 0..p_cov {
                    x_rep[[i, a * p_cov + b]] = left[[i, a]] * right[[i, b]];
                }
            }
        }
        let mut weighted_x = x_rep.clone();
        for i in 0..n {
            let w = weights[i];
            for j in 0..(p_resp * p_cov) {
                weighted_x[[i, j]] *= w;
            }
        }
        let dense_h = x_rep.t().dot(&weighted_x);
        let want = dense_h.dot(&v);
        let got = fused_khatri_rao_weighted_gram_apply(&left, &right, &weights, &v);
        assert_eq!(got.len(), want.len());
        for i in 0..want.len() {
            let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "fused mismatch at {i}: dense={:.6e}, fused={:.6e}",
                want[i],
                got[i],
            );
        }
    }

    #[test]
    fn kronecker_variant_matches_materialized_form() {
        // Tiny inputs: n_cov=4 covariate rows, n_grid=3 monotonicity-grid points,
        // p_resp=2 response-direction columns, p_cov=2 covariate columns.
        let response_grid = array![[1.0, 0.5], [0.7, -0.1], [-0.3, 0.9]];
        let cov = array![[0.20, 1.10], [-0.40, 0.80], [0.55, -0.25], [0.10, 0.65]];
        let n_cov = cov.nrows();
        let n_grid = response_grid.nrows();
        let p_resp = response_grid.ncols();
        let p_cov = cov.ncols();
        let p_total = p_resp * p_cov;
        let n_virtual = n_cov * n_grid;

        let design = KroneckerDesign::new_kronecker(
            response_grid.clone(),
            DesignMatrix::Dense(DenseDesignMatrix::from(cov.clone())),
        )
        .expect("Kronecker variant construction");
        assert_eq!(design.nrows(), n_virtual);
        assert_eq!(design.ncols(), p_total);

        // Materialize the reference: virtual row (i, g) has design row equal
        // to response_grid[g, :] ⊗ cov[i, :] (response-major flattening).
        let mut reference = Array2::<f64>::zeros((n_virtual, p_total));
        for i in 0..n_cov {
            for g in 0..n_grid {
                let row = i * n_grid + g;
                for a in 0..p_resp {
                    for b in 0..p_cov {
                        reference[[row, a * p_cov + b]] = response_grid[[g, a]] * cov[[i, b]];
                    }
                }
            }
        }

        // forward_mul: choose a deterministic non-trivial beta and compare.
        let beta = Array1::from_vec((0..p_total).map(|k| 0.3 + 0.17 * k as f64).collect());
        let got_forward = design.forward_mul(&beta);
        let expected_forward = reference.dot(&beta);
        assert_eq!(got_forward.len(), n_virtual);
        for i in 0..n_virtual {
            assert!(
                (got_forward[i] - expected_forward[i]).abs() < 1e-12,
                "forward_mul mismatch at row {i}: got={}, expected={}",
                got_forward[i],
                expected_forward[i]
            );
        }

        // transpose_mul: virtual-row vector v, compare against reference^T · v.
        let v = Array1::from_vec((0..n_virtual).map(|k| -0.25 + 0.31 * k as f64).collect());
        let got_transpose = design.transpose_mul(&v);
        let expected_transpose = reference.t().dot(&v);
        assert_eq!(got_transpose.len(), p_total);
        for k in 0..p_total {
            assert!(
                (got_transpose[k] - expected_transpose[k]).abs() < 1e-12,
                "transpose_mul mismatch at col {k}: got={}, expected={}",
                got_transpose[k],
                expected_transpose[k]
            );
        }
    }

    /// Sweep over a representative cross-section of (n_cov, n_grid, p_resp,
    /// p_cov) shapes — including degenerate p_resp=1 / p_cov=1 cases and
    /// asymmetric n_cov / n_grid ratios — to confirm the factored Kronecker
    /// identities hold across the parameter space, not just at one point. All
    /// inputs are deterministic (a Numerical-Recipes LCG over a fixed seed)
    /// so the test is reproducible without an RNG dependency.
    #[test]
    fn kronecker_variant_matches_materialized_form_across_shapes() {
        fn lcg_sequence(seed: u64, len: usize) -> Vec<f64> {
            let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                out.push(2.0 * u - 1.0);
            }
            out
        }
        fn make_array2(seed: u64, rows: usize, cols: usize) -> Array2<f64> {
            Array2::from_shape_vec((rows, cols), lcg_sequence(seed, rows * cols))
                .expect("dimensions match")
        }
        fn make_array1(seed: u64, len: usize) -> Array1<f64> {
            Array1::from_vec(lcg_sequence(seed, len))
        }

        // Shapes: (n_cov, n_grid, p_resp, p_cov). Includes the structural edge
        // cases p_resp=1 and p_cov=1 (the elementwise definition collapses to
        // an outer product), n_cov=1 and n_grid=1, an asymmetric grid-heavy
        // case, and a moderate generic shape.
        let shapes: &[(usize, usize, usize, usize)] = &[
            (1, 5, 3, 4),
            (5, 1, 3, 4),
            (5, 7, 1, 4),
            (5, 7, 3, 1),
            (3, 11, 2, 5),
            (10, 4, 4, 3),
            (8, 13, 2, 4),
            (4, 4, 4, 4),
        ];

        for (idx, &(n_cov, n_grid, p_resp, p_cov)) in shapes.iter().enumerate() {
            let seed = 0xA17B_C0DE_u64.wrapping_add(idx as u64);
            let response_grid = make_array2(seed ^ 0x1, n_grid, p_resp);
            let cov = make_array2(seed ^ 0x2, n_cov, p_cov);
            let beta = make_array1(seed ^ 0x3, p_resp * p_cov);
            let v = make_array1(seed ^ 0x4, n_cov * n_grid);

            let design = KroneckerDesign::new_kronecker(
                response_grid.clone(),
                DesignMatrix::Dense(DenseDesignMatrix::from(cov.clone())),
            )
            .expect("Kronecker variant construction");

            // Materialize the reference matrix from the elementwise definition
            // X[(i, g), (a, b)] = R[g, a] · C[i, b] with row-flatten covariate-
            // major and column-flatten response-major.
            let p_total = p_resp * p_cov;
            let n_virtual = n_cov * n_grid;
            let mut reference = Array2::<f64>::zeros((n_virtual, p_total));
            for i in 0..n_cov {
                for g in 0..n_grid {
                    let row = i * n_grid + g;
                    for a in 0..p_resp {
                        for b in 0..p_cov {
                            reference[[row, a * p_cov + b]] = response_grid[[g, a]] * cov[[i, b]];
                        }
                    }
                }
            }

            // forward_mul: factored vs. materialized.
            let got_forward = design.forward_mul(&beta);
            let expected_forward = reference.dot(&beta);
            for k in 0..n_virtual {
                let diff = (got_forward[k] - expected_forward[k]).abs();
                assert!(
                    diff < 1e-12,
                    "shape {:?} #{idx}: forward_mul mismatch at virtual row {k} (diff={diff:e})",
                    (n_cov, n_grid, p_resp, p_cov),
                );
            }

            // transpose_mul: factored vs. materialized.
            let got_transpose = design.transpose_mul(&v);
            let expected_transpose = reference.t().dot(&v);
            for k in 0..p_total {
                let diff = (got_transpose[k] - expected_transpose[k]).abs();
                assert!(
                    diff < 1e-12,
                    "shape {:?} #{idx}: transpose_mul mismatch at col {k} (diff={diff:e})",
                    (n_cov, n_grid, p_resp, p_cov),
                );
            }

            // Adjoint identity: ⟨v, X β⟩ = ⟨Xᵀ v, β⟩. If both forms match the
            // same reference this is implied, but checking it directly catches
            // a class of off-by-one transpose bugs that would happen to flip
            // both forms consistently.
            let lhs: f64 = v.iter().zip(got_forward.iter()).map(|(a, b)| a * b).sum();
            let rhs: f64 = got_transpose
                .iter()
                .zip(beta.iter())
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                (lhs - rhs).abs() < 1e-10,
                "shape {:?} #{idx}: adjoint identity ⟨v, Xβ⟩ ≠ ⟨Xᵀv, β⟩ (lhs={lhs}, rhs={rhs})",
                (n_cov, n_grid, p_resp, p_cov),
            );
        }
    }

    /// When most virtual rows are far from the slack boundary the active-set
    /// certificate accepts and the cached fast path must produce the same
    /// `α` the full-grid scan would. Bit-equivalence is the contract — any
    /// drift is a correctness bug, not a numerical-precision issue, since
    /// both paths reduce identical sums by the associative `f64::min`.
    #[test]
    fn ctn_active_set_certificate_matches_full_grid_when_bound_passes() {
        // n=64, p_resp=8, p_cov=4, n_grid=16. Build smooth, well-separated
        // factors so most (i, g) pairs have generous feasibility margin.
        let n = 64usize;
        let p_resp = 8usize;
        let p_cov = 4usize;
        let n_grid = 16usize;
        let mut covariate = Array2::<f64>::zeros((n, p_cov));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            covariate[[i, 0]] = 1.0;
            covariate[[i, 1]] = 0.20 + 0.05 * t;
            covariate[[i, 2]] = -0.10 + 0.04 * (t * t - 0.5);
            covariate[[i, 3]] = 0.08 * (t - 0.5);
        }
        let mut response_grid = Array2::<f64>::zeros((n_grid, p_resp));
        for g in 0..n_grid {
            let s = g as f64 / (n_grid as f64 - 1.0);
            response_grid[[g, 0]] = 1.0;
            for a in 1..p_resp {
                response_grid[[g, a]] = 0.05 * (a as f64) * (s - 0.5);
            }
        }
        let design = KroneckerDesign::new_kronecker(
            response_grid.clone(),
            DesignMatrix::Dense(DenseDesignMatrix::from(covariate)),
        )
        .expect("Kronecker design");

        // β has dominant value on the (a=0, b=0) entry so h ≈ 1.0 on every
        // virtual row, well above slack. δ has small negative leading entry
        // to push some virtual rows toward the boundary, but |Δ| stays small
        // enough that L · α_A is comfortably below m_inactive.
        let mut beta = Array1::<f64>::zeros(p_resp * p_cov);
        beta[0] = 1.0;
        let mut delta = Array1::<f64>::zeros(p_resp * p_cov);
        delta[0] = -1.0e-2;

        let slack = 1e-3;
        let dh_eps = 1e-14;

        // Reference: full-grid reduction via the un-cached entry point.
        let alpha_full = design.min_step_to_boundary(&beta, &delta, slack, dh_eps);

        let c = design
            .project_kronecker_factor(&beta)
            .expect("Kronecker variant projects β");
        let d = design
            .project_kronecker_factor(&delta)
            .expect("Kronecker variant projects δ");
        let mut cache = KroneckerActiveSetCache::new();
        // First call refreshes the cache (cache.cached_for_version != c_version
        // ⇒ cache miss); the answer matches full-grid by construction.
        let alpha_first = design.min_step_to_boundary_with_active_set(
            c.view(),
            d.view(),
            &delta,
            slack,
            dh_eps,
            &mut cache,
        );
        assert_eq!(
            alpha_first, alpha_full,
            "first active-set call must full-scan and match the un-cached path"
        );
        // Cache must now be marked fresh and hold a finite inactive bound.
        assert_eq!(cache.cached_for_version, cache.c_version);
        assert!(
            cache.min_inactive_margin.is_finite(),
            "fixture must produce at least one inactive pair so m_inactive is finite"
        );
        // Second call: same β-projection ⇒ certificate path. Must be exactly
        // bit-equal to the full-grid reduction.
        let alpha_certified = design.min_step_to_boundary_with_active_set(
            c.view(),
            d.view(),
            &delta,
            slack,
            dh_eps,
            &mut cache,
        );
        assert_eq!(
            alpha_certified, alpha_full,
            "active-set certificate fast path must be bit-equivalent to the full-grid scan"
        );
    }

    /// When the certificate's Lipschitz bound is too loose to certify (an
    /// inactive pair is near-binding), the implementation must fall back to a
    /// full grid scan and return the correct `α`. We refresh the cache, then
    /// shrink `m_inactive` to drive the certificate into its fail branch and
    /// confirm the fallback recovers the exact full-grid answer.
    #[test]
    fn ctn_active_set_falls_back_to_full_grid_when_bound_fails() {
        // p_resp=2, p_cov=1, n=3, n_grid=2 ⇒ 6 virtual pairs. Response grid
        // r_g = e_a (axis-aligned) so h_{i,g} = c[i, g] and d_{i,g} = d[i, g];
        // diagonal indexing makes the fixture readable.
        let n = 3usize;
        let p_resp = 2usize;
        let response_grid = array![[1.0, 0.0], [0.0, 1.0]];
        let mut cov_arr = Array2::<f64>::zeros((n, 1));
        cov_arr[[0, 0]] = 1.0;
        cov_arr[[1, 0]] = 1.0;
        cov_arr[[2, 0]] = 1.0;
        let design = KroneckerDesign::new_kronecker(
            response_grid.clone(),
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_arr)),
        )
        .expect("Kronecker design");

        // c shape (n × p_resp). Pair (0, 0) is the binder (h = slack + 1e-12)
        // with strongly negative d so α_A is tiny and finite. Pair (1, 1) is
        // the near-inactive: h_{1,1} = slack + 1e-3 — sits just outside τ but
        // small enough that the conservative L bound can exceed m_inactive.
        // Remaining pairs sit far above slack.
        let slack = 1e-3;
        let dh_eps = 1e-14;
        let mut c = Array2::<f64>::zeros((n, p_resp));
        c[[0, 0]] = slack + 1e-12;
        c[[0, 1]] = 5.0;
        c[[1, 0]] = 5.0;
        c[[1, 1]] = slack + 1e-3;
        c[[2, 0]] = 5.0;
        c[[2, 1]] = 5.0;
        // Only the binder has nonzero d, so the full-grid α equals (h-slack)/(-d).
        let mut d = Array2::<f64>::zeros((n, p_resp));
        d[[0, 0]] = -1.0;
        let delta = Array1::<f64>::from(vec![1.0, 0.0]);
        let alpha_ref = (c[[0, 0]] - slack) / (-d[[0, 0]]);

        let mut cache = KroneckerActiveSetCache::new();
        // First call: cache miss ⇒ full-grid scan, populates cache.
        let alpha_first = design.min_step_to_boundary_with_active_set(
            c.view(),
            d.view(),
            &delta,
            slack,
            dh_eps,
            &mut cache,
        );
        assert_eq!(alpha_first, alpha_ref);
        // The near-inactive pair (1,1) has h - slack = 1e-3, well outside τ
        // (≈1e-6) but the smallest among inactive pairs, so it drives
        // cache.min_inactive_margin to ~1e-3.
        assert!(
            cache.min_inactive_margin <= 1e-3 + 1e-12,
            "near-inactive pair must drive m_inactive ≤ 1e-3, got {}",
            cache.min_inactive_margin
        );
        // For this fixture max ||r_g|| = 1, max ||c_i|| = 1, ‖ΔB‖_F = 1 ⇒
        // L = 1. With α_A = 1e-12 the certificate margin (1e-3) trivially
        // exceeds α_A · L, so it would actually accept here. Drive the
        // fail-bound branch by mutating m_inactive directly — this mirrors
        // the production case where ‖ΔB‖_F is large enough that L overshoots
        // the truth and the conservative bound is too tight.
        cache.min_inactive_margin = 0.0;
        let alpha_after = design.min_step_to_boundary_with_active_set(
            c.view(),
            d.view(),
            &delta,
            slack,
            dh_eps,
            &mut cache,
        );
        // Bound-failure branch must fall back to a full grid scan and return
        // the correct α. The full scan also refreshes the cache, restoring
        // `min_inactive_margin` to its true value.
        assert_eq!(
            alpha_after, alpha_ref,
            "fallback path must return the full-grid α when the certificate bound fails"
        );
        assert!(
            cache.min_inactive_margin > 0.0,
            "fallback must rerun the refresh and restore m_inactive"
        );
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

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let mat = MaterializablePsiDerivativeOperator::materialize_first(self, axis)?;
        Ok(mat.slice(ndarray::s![rows, ..]).to_owned())
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let mat = MaterializablePsiDerivativeOperator::materialize_second_diag(self, axis)?;
        Ok(mat.slice(ndarray::s![rows, ..]).to_owned())
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let mat =
            MaterializablePsiDerivativeOperator::materialize_second_cross(self, axis_d, axis_e)?;
        Ok(mat.slice(ndarray::s![rows, ..]).to_owned())
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for TensorKroneckerPsiOperator {
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

#[derive(Clone)]
struct TransformationExactGeometryCache {
    key: Vec<u64>,
    family: TransformationNormalFamily,
    base_block_spec: ParameterBlockSpec,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
}

fn transformation_spatial_geometry_key(
    spec: &TermCollectionSpec,
    spatial_terms: &[usize],
) -> Vec<u64> {
    let mut key = Vec::with_capacity(1 + spatial_terms.len() * 8);
    key.push(spatial_terms.len() as u64);
    for &term_idx in spatial_terms {
        key.push(term_idx as u64);
        key.push(
            get_spatial_length_scale(spec, term_idx)
                .map(f64::to_bits)
                .unwrap_or(u64::MAX),
        );
        match get_spatial_aniso_log_scales(spec, term_idx) {
            Some(eta) => {
                key.push(eta.len() as u64);
                key.extend(eta.into_iter().map(f64::to_bits));
            }
            None => key.push(u64::MAX - 1),
        }
    }
    key
}

fn build_blocks_from_base_spec(
    base_block_spec: &ParameterBlockSpec,
    beta_hint: Option<&Array1<f64>>,
) -> Vec<ParameterBlockSpec> {
    let mut spec = base_block_spec.clone();
    if let Some(hint) = beta_hint
        && hint.len() == spec.design.ncols()
    {
        spec.initial_beta = Some(hint.clone());
    }
    vec![spec]
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

    let kappa0 = SpatialLogKappaCoords::from_length_scales_aniso(
        covariate_spec,
        &spatial_terms,
        kappa_options,
    )
    .reseed_from_data(
        covariate_data,
        covariate_spec,
        &spatial_terms,
        kappa_options,
    );
    let kappa_dims = kappa0.dims_per_term().to_vec();
    let kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        covariate_data,
        covariate_spec,
        &spatial_terms,
        &kappa_dims,
        kappa_options,
    );
    let kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        covariate_data,
        covariate_spec,
        &spatial_terms,
        &kappa_dims,
        kappa_options,
    );
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let kappa0 = kappa0.clamp_to_bounds(&kappa_lower, &kappa_upper);

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
    let probe_block = probe_family.block_spec();
    let n_penalties = probe_block.initial_log_lambdas.len();
    log::info!(
        "[transformation-normal] exact joint setup: rho_dim={} log_kappa_dim={} dims_per_term={:?}",
        n_penalties,
        kappa0.len(),
        kappa_dims,
    );
    let mut rho0 = Array1::<f64>::zeros(n_penalties);
    let rho_lower = Array1::<f64>::from_elem(n_penalties, -12.0);
    let rho_upper = Array1::<f64>::from_elem(n_penalties, 12.0);
    let probe_blocks = vec![probe_block];
    let (_, cap_hessian) = custom_family_outer_derivatives(&probe_family, &probe_blocks, &options);
    let analytic_gradient = analytic_psi_available;
    let analytic_hessian = analytic_psi_available
        && matches!(
            cap_hessian,
            crate::solver::outer_strategy::Derivative::Analytic
        );

    log::info!(
        "[transformation-normal] starting baseline custom-family fit before exact joint optimization \
         (rho_dim={}, log_kappa_dim={})",
        n_penalties,
        kappa0.len(),
    );
    let baseline_start = std::time::Instant::now();
    let baseline_fit = fit_custom_family(&probe_family, &probe_blocks, &options).ok();
    log::info!(
        "[transformation-normal] baseline fit {} in {:.2}s",
        if baseline_fit.is_some() {
            "succeeded"
        } else {
            "failed (continuing with default rho seed)"
        },
        baseline_start.elapsed().as_secs_f64(),
    );
    if let Some(fit) = baseline_fit.as_ref() {
        if fit.log_lambdas.len() == n_penalties {
            rho0 = fit.log_lambdas.clone();
        }
    }

    if !analytic_psi_available {
        let fit = baseline_fit.ok_or_else(|| {
            "transformation fit failed before scale-dimensions optimization and analytic spatial psi derivatives are unavailable"
                .to_string()
        })?;
        let mut cov_spec_resolved = boot_spec.clone();
        sync_aniso_contrasts_from_metadata(&mut cov_spec_resolved, &boot_design.smooth);
        return Ok(TransformationNormalFitResult {
            family: probe_family,
            fit,
            covariate_spec_resolved: cov_spec_resolved,
            covariate_design: boot_design,
        });
    }

    // Shared mutable state for warm-starting across optimizer iterations.
    let beta_hint: RefCell<Option<Array1<f64>>> = RefCell::new(
        baseline_fit
            .as_ref()
            .and_then(|fit| fit.block_states.first().map(|block| block.beta.clone())),
    );
    let exact_warm_start: RefCell<Option<CustomFamilyWarmStart>> = RefCell::new(None);

    let joint_setup =
        ExactJointHyperSetup::new(rho0, rho_lower, rho_upper, kappa0, kappa_lower, kappa_upper);

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

    let block_specs_slice = [boot_spec.clone()];
    let block_term_indices_slice = [spatial_terms.clone()];
    let exact_geometry_cache: RefCell<Option<TransformationExactGeometryCache>> =
        RefCell::new(None);
    let spatial_terms_for_cache = spatial_terms.clone();

    let ensure_exact_geometry = |spec: &TermCollectionSpec,
                                 design: &TermCollectionDesign|
     -> Result<(), String> {
        let key = transformation_spatial_geometry_key(spec, &spatial_terms_for_cache);
        let needs_rebuild = exact_geometry_cache
            .borrow()
            .as_ref()
            .map(|cached| cached.key != key)
            .unwrap_or(true);
        if !needs_rebuild {
            return Ok(());
        }

        let geom_start = std::time::Instant::now();
        let family = make_family(design)?;
        let cov_psi_derivs = build_block_spatial_psi_derivatives(covariate_data, spec, design)?
            .ok_or_else(|| {
                "missing covariate spatial psi derivatives for transformation model".to_string()
            })?;
        let tensor_derivs = build_tensor_psi_derivatives(&family, &cov_psi_derivs)?;

        log::debug!(
            "[transformation-normal] rebuilt exact geometry cache for {} spatial terms in {:.3}s",
            spatial_terms_for_cache.len(),
            geom_start.elapsed().as_secs_f64(),
        );

        // The exact-inner warm start embeds geometry-dependent state.
        // Reuse it across rho updates, but drop it when the spatial basis changes.
        exact_warm_start.replace(None);
        exact_geometry_cache.replace(Some(TransformationExactGeometryCache {
            key,
            base_block_spec: family.block_spec(),
            family,
            derivative_blocks: vec![tensor_derivs],
        }));
        Ok(())
    };

    log::info!(
        "[transformation-normal] entering exact joint outer optimization \
         (analytic_gradient={}, analytic_hessian={})",
        analytic_gradient,
        analytic_hessian,
    );
    let solved = optimize_spatial_length_scale_exact_joint(
        covariate_data,
        &block_specs_slice,
        &block_term_indices_slice,
        kappa_options,
        &joint_setup,
        crate::seeding::SeedRiskProfile::Gaussian,
        analytic_gradient,
        analytic_hessian,
        // Transformation-normal has β-dependent H (through 1/h'²), so the
        // EFS Wood-Fasiolo PSD invariant fails — disable fixed-point so the
        // planner cannot pick EFS / Hybrid-EFS. With fixed-point ruled out
        // and analytic gradient + Hessian declared, the planner then chooses
        // ARC by default and BFGS only when TauTauHessianPolicy reports
        // prefer_gradient_only (e.g. multi-dimensional Duchon, dense tau
        // cache exceeding budget) — which IS the typical biobank-scale CTN
        // configuration with --scale-dimensions and ≥4 PCs.
        true,
        // fit_fn
        |_, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let cache_ref = exact_geometry_cache.borrow();
            let geometry = cache_ref
                .as_ref()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let blocks =
                build_blocks_from_base_spec(&geometry.base_block_spec, beta_hint.borrow().as_ref());
            let fit = fit_custom_family(&geometry.family, &blocks, &options)
                .map_err(|e| format!("transformation fit_fn: {e}"))?;
            // Update warm start hints.
            if let Some(block) = fit.block_states.first() {
                *beta_hint.borrow_mut() = Some(block.beta.clone());
            }
            Ok((geometry.family.clone(), fit))
        },
        // exact_fn
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], eval_mode| {
            use crate::solver::estimate::reml::unified::EvalMode;
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let cache_ref = exact_geometry_cache.borrow();
            let geometry = cache_ref
                .as_ref()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let blocks =
                build_blocks_from_base_spec(&geometry.base_block_spec, beta_hint.borrow().as_ref());
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();

            let eval = evaluate_custom_family_joint_hyper(
                &geometry.family,
                &blocks,
                &options,
                &rho,
                &geometry.derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                eval_mode,
            )
            .map_err(|e| format!("transformation exact_fn: {e}"))?;

            if !eval.objective.is_finite() {
                log::warn!(
                    "transformation exact joint returned non-finite objective: eval_mode={:?} rho={:?} gradient_len={}",
                    eval_mode,
                    rho,
                    eval.gradient.len(),
                );
            }

            exact_warm_start.replace(Some(eval.warm_start));

            if matches!(eval_mode, EvalMode::ValueGradientHessian) && !eval.outer_hessian.is_analytic() {
                return Err(
                    "transformation exact joint objective did not return an outer Hessian"
                        .to_string(),
                );
            }

            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let cache_ref = exact_geometry_cache.borrow();
            let geometry = cache_ref
                .as_ref()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let blocks =
                build_blocks_from_base_spec(&geometry.base_block_spec, beta_hint.borrow().as_ref());
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let eval = evaluate_custom_family_joint_hyper_efs(
                &geometry.family,
                &blocks,
                &options,
                &rho,
                &geometry.derivative_blocks,
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
