//! Conditional transformation model: estimate h(y|x) such that h(Y|x) ~ N(0,1).
//!
//! Given a response variable y and covariates x with a pre-built covariate design
//! operator, this family estimates a smooth monotone transformation h(y | x) mapping
//! the conditional distribution of Y|x onto a standard normal.
//!
//! The response-direction basis is `[1, I_1(y), ..., I_K(y)]`, tensored with an
//! arbitrary covariate design operator. Column 0 is an unconstrained location
//! component `b(x)`. The I-spline columns are shape components with squared
//! covariate-side coefficients, giving the SCOP representation
//! `h(y, x) = b(x) + ε·(y−median_y) + Σ_k I_k(y) γ_k(x)^2` and
//! `h'(y, x) = ε + Σ_k M_k(y) γ_k(x)^2`. Monotonicity is structural:
//! the fixed derivative floor `ε` keeps the change-of-variables log-density
//! away from the `log(0)` singularity, while the non-negative M-spline basis
//! and squared covariate-side coefficients supply the learned shape.
//!
//! The log-likelihood per observation is the finite-support normalized
//! change-of-variables density for a standard normal target:
//!
//!   ℓ_i = -½ h_i² + log(h'_i) - log(Φ(h_U(x_i)) - Φ(h_L(x_i)))
//!
//! where `h_i = b(x_i) + ε·(y_i−median_y) + Σ_k I_k(y_i) γ_k(x_i)^2`
//! and `h'_i = ε + Σ_k M_k(y_i) γ_k(x_i)^2`. The endpoint normalizer is
//! required because the I-spline response basis saturates at finite support
//! values rather than mapping onto the full real line.

use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
    create_ispline_derivative_dense,
};
use crate::faer_ndarray::{fast_ab, fast_ab_into, fast_atb};
use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyPsiDerivativeOperator, ExactNewtonJointGradientEvaluation,
    ExactNewtonJointHessianWorkspace, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, FamilyEvaluation, MaterializablePsiDerivativeOperator,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, build_block_spatial_psi_derivatives,
    evaluate_custom_family_joint_hyper, evaluate_custom_family_joint_hyper_efs, fit_custom_family,
    fit_custom_family_fixed_log_lambdas,
};
use crate::families::gamlss::{
    initializewiggle_knots_from_seed, solve_penalizedweighted_projection,
};
use crate::inference::model::TransformationScoreCalibration;
use crate::matrix::{
    DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator, SymmetricMatrix,
};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{log1mexp_positive, normal_logcdf};
use crate::resource::{MatrixMaterializationError, ResourcePolicy};
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use crate::solver::estimate::UnifiedFitResult;
use crate::solver::estimate::reml::unified::HyperOperator;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, s};
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
/// E[log |Z|] for Z ~ N(0, 1), used to put local log-absolute residual
/// projections on the standard-normal scale.
const STANDARD_NORMAL_MEAN_LOG_ABS: f64 = -0.635_181_422_730_739_1;
/// Relative tolerance for response-grid deduplication and zero-width-gap
/// skipping. Used by both the fit-time grid builder
/// (`transformation_monotonicity_response_grid`) and the predict-time
/// derivative-grid builder. Kept in sync so the two grids generate the
/// same point set on identical inputs (otherwise predict could hit a
/// distinct grid point that the fit-time barrier never bounded).
pub const TRANSFORMATION_GRID_RELATIVE_TOL: f64 = 1.0e-12;

/// Strict-feasibility margin for `h' > 0` on the monotonicity grid. Used
/// both by the fit-time fraction-to-boundary line search (so accepted β
/// keeps `h'(grid) ≥ EPS`) and by the predict-time monotonicity check
/// in `inference::predict_input` (which rejects predictions whose minimum
/// `h'` on the response grid drops below this threshold). Keeping these
/// in sync prevents the predict path from rejecting fits that the
/// optimizer accepted as feasible — and vice versa.
pub const TRANSFORMATION_MONOTONICITY_EPS: f64 = 1.0e-8;
/// SCOP-CTN supplies ψ-axis second-order curvature as matrix-free HVPs, so the
/// shared dense tau-tau memory policy must not downgrade analytic Hessian
/// planning to gradient-only BFGS for this family.
const CTN_ALLOW_TAU_TAU_GRADIENT_ONLY_PREFERENCE: bool = false;
/// Absolute bound for feasible transformation scores on the standard-normal
/// scale. The CTN likelihood targets `h(Y|x) ~ N(0,1)`; accepting exact-Newton
/// iterates with finite positive `h'` but astronomical `|h|` lets curvature
/// diagnostics overflow into meaningless values. This is a numerical runaway
/// guard, not a statistical plausibility filter: startup seeds can temporarily
/// land outside practically observable normal quantiles before the line search
/// moves them back into the likelihood's high-density region.
pub const TRANSFORMATION_NORMAL_H_ABS_MAX: f64 = 1.0e6;
/// Maximum number of response quantiles drawn into the monotonicity grid.
/// Shared between fit and predict so the grid construction is identical.
pub const TRANSFORMATION_RESPONSE_GRID_MAX_QUANTILES: usize = 129;
/// Number of equal subdivisions inserted between consecutive grid points
/// (knots + quantiles). Shared between fit and predict.
pub const TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS: usize = 4;

fn beta_bits_match(cached: &Array1<f64>, candidate: &Array1<f64>) -> bool {
    cached.len() == candidate.len()
        && cached
            .iter()
            .zip(candidate.iter())
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
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
    /// Response value basis: n × p_resp. Columns: [1, I_1(y), ..., I_k(y)].
    response_val_basis: Array2<f64>,
    /// Response value basis at the finite lower support endpoint.
    response_lower_basis: Array1<f64>,
    /// Response value basis at the finite upper support endpoint.
    response_upper_basis: Array1<f64>,
    /// Response derivative basis: n × p_resp. Columns: [0, M_1(y), ..., M_k(y)].
    response_deriv_basis: Array2<f64>,

    // --- Covariate side (rebuilt on κ change) ---
    /// Original covariate design used on the right side of the tensor product.
    covariate_design: DesignMatrix,
    /// Dense covariate block shared by row-quantity and endpoint evaluations.
    ///
    /// CTN row quantities are rebuilt at every accepted/probed β, but the
    /// covariate design is fixed for the family. Caching this immutable
    /// `n × p_cov` block avoids repeated chunk materialization and keeps
    /// biobank-scale runs from churning large transient allocations.
    covariate_dense_cache: Arc<Mutex<Option<Arc<Array2<f64>>>>>,
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
    response_floor_offset: Arc<Array1<f64>>,
    response_lower_floor_offset: f64,
    response_upper_floor_offset: f64,

    /// Last row-space transformation quantities for an exact beta vector.
    ///
    /// CTN line searches and exact-Newton workspace construction frequently ask
    /// for likelihood, gradient, and Hessian row factors at the same candidate
    /// coefficients. This cache keeps the expensive Khatri-Rao forward products
    /// and reciprocal powers behind a single exact-keyed entry instead of
    /// recomputing `h`, `h'`, `1/h'`, and derivative powers per call.
    row_quantity_cache: Arc<Mutex<Option<TransformationNormalRowQuantityCache>>>,
}

#[derive(Clone)]
struct TransformationNormalRowQuantityCache {
    beta: Arc<Array1<f64>>,
    h: Arc<Array1<f64>>,
    h_prime: Arc<Array1<f64>>,
    endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    log_likelihood: f64,
}

#[derive(Debug)]
struct TransformationNormalRowDerived {
    log_likelihood: f64,
    endpoint_q: Vec<LogNormalCdfDiffDerivatives>,
}

impl TransformationNormalRowQuantityCache {
    fn matches_beta(&self, beta: &Array1<f64>) -> bool {
        beta_bits_match(&self.beta, beta)
    }
}

fn build_transformation_row_derived(
    h: &Array1<f64>,
    h_prime: &Array1<f64>,
    h_lower: &Array1<f64>,
    h_upper: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<TransformationNormalRowDerived, String> {
    let n = h_prime.len();
    debug_assert_eq!(h.len(), n);
    debug_assert_eq!(h_lower.len(), n);
    debug_assert_eq!(h_upper.len(), n);
    debug_assert_eq!(weights.len(), n);

    let mut log_likelihood = 0.0;
    let mut endpoint_q = Vec::with_capacity(n);

    if let Some((i, value)) = h
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "TransformationNormalFamily row_quantities: h[{i}] = {value} is not finite"
        ));
    }
    if let Some((i, value)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "TransformationNormalFamily row_quantities: weight[{i}] = {value} is not finite"
        ));
    }
    for i in 0..n {
        let hp = h_prime[i];
        let inv_h_prime = 1.0 / hp;
        let inv_h_prime_sq = inv_h_prime * inv_h_prime;
        let inv_h_prime_cu = inv_h_prime_sq * inv_h_prime;
        let inv_h_prime_qu = inv_h_prime_sq * inv_h_prime_sq;
        let weighted_h = weights[i] * h[i];
        let weighted_inv_h_prime = weights[i] * inv_h_prime;
        let weighted_inv_h_prime_sq = weights[i] * inv_h_prime_sq;
        let q = log_normal_cdf_diff_derivatives(h_upper[i], h_lower[i]).map_err(|e| {
            format!("TransformationNormalFamily row_quantities: row {i} invalid endpoint normalizer: {e}")
        })?;
        let log_z = q.log_z;
        log_likelihood += weights[i] * (-0.5 * h[i] * h[i] + hp.ln() - log_z);
        let derived_values = [
            ("1/h'", inv_h_prime),
            ("1/h'^2", inv_h_prime_sq),
            ("1/h'^3", inv_h_prime_cu),
            ("1/h'^4", inv_h_prime_qu),
            ("w*h", weighted_h),
            ("w/h'", weighted_inv_h_prime),
            ("w/h'^2", weighted_inv_h_prime_sq),
            ("log normalizer", log_z),
        ];
        for (name, value) in derived_values {
            if !value.is_finite() {
                return Err(format!(
                    "TransformationNormalFamily row_quantities: {name} at row {i} is not finite ({value}); h'={} is outside the finite exact-derivative range",
                    h_prime[i],
                ));
            }
        }
        endpoint_q.push(q);
    }
    if !log_likelihood.is_finite() {
        return Err(format!(
            "TransformationNormalFamily row_quantities: log-likelihood is not finite ({log_likelihood})"
        ));
    }

    Ok(TransformationNormalRowDerived {
        log_likelihood,
        endpoint_q,
    })
}

fn log_normal_cdf_diff(upper: f64, lower: f64) -> Result<f64, String> {
    if !(upper.is_finite() && lower.is_finite()) {
        return Err(format!(
            "finite support endpoints required, got lower={lower}, upper={upper}"
        ));
    }
    if upper <= lower {
        return Err(format!(
            "upper endpoint score must exceed lower endpoint score, got lower={lower:.6e}, upper={upper:.6e}"
        ));
    }
    if lower > 0.0 {
        return log_normal_cdf_diff(-lower, -upper);
    }
    let log_upper = normal_logcdf(upper);
    let log_lower = normal_logcdf(lower);
    let gap = log_upper - log_lower;
    if !(gap.is_finite() && gap > 0.0) {
        return Err(format!(
            "normal CDF endpoint mass is not representable, lower={lower:.6e}, upper={upper:.6e}"
        ));
    }
    let log_z = log_upper + log1mexp_positive(gap);
    if !log_z.is_finite() {
        return Err(format!(
            "normal CDF endpoint mass underflowed, lower={lower:.6e}, upper={upper:.6e}"
        ));
    }
    Ok(log_z)
}

fn signed_normal_pdf_ratio(
    x: f64,
    polynomial_factor: f64,
    log_z: f64,
    factorial_scale: f64,
) -> f64 {
    if polynomial_factor == 0.0 {
        return 0.0;
    }
    const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
    let log_abs =
        polynomial_factor.abs().ln() - 0.5 * x * x - LOG_SQRT_2PI - factorial_scale.ln() - log_z;
    polynomial_factor.signum() * log_abs.exp()
}

#[derive(Clone, Copy, Debug)]
struct LogNormalCdfDiffDerivatives {
    log_z: f64,
    first: [f64; 2],
    second: [[f64; 2]; 2],
    third: [[[f64; 2]; 2]; 2],
    fourth: [[[[f64; 2]; 2]; 2]; 2],
}

fn endpoint_chain_first(q: &LogNormalCdfDiffDerivatives, a: [f64; 2]) -> f64 {
    q.first[0] * a[0] + q.first[1] * a[1]
}

fn endpoint_chain_second(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    ab: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, ab);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j] * a[i] * b[j];
        }
    }
    out
}

fn endpoint_chain_third(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    ab: [f64; 2],
    ac: [f64; 2],
    bc: [f64; 2],
    abc: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, abc);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j] * (ab[i] * c[j] + ac[i] * b[j] + bc[i] * a[j]);
            for k in 0..2 {
                out += q.third[i][j][k] * a[i] * b[j] * c[k];
            }
        }
    }
    out
}

fn endpoint_chain_fourth(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
    ab: [f64; 2],
    ac: [f64; 2],
    ad: [f64; 2],
    bc: [f64; 2],
    bd: [f64; 2],
    cd: [f64; 2],
    abc: [f64; 2],
    abd: [f64; 2],
    acd: [f64; 2],
    bcd: [f64; 2],
    abcd: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, abcd);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j]
                * (abc[i] * d[j]
                    + abd[i] * c[j]
                    + acd[i] * b[j]
                    + bcd[i] * a[j]
                    + ab[i] * cd[j]
                    + ac[i] * bd[j]
                    + ad[i] * bc[j]);
            for k in 0..2 {
                out += q.third[i][j][k]
                    * (ab[i] * c[j] * d[k]
                        + ac[i] * b[j] * d[k]
                        + ad[i] * b[j] * c[k]
                        + bc[i] * a[j] * d[k]
                        + bd[i] * a[j] * c[k]
                        + cd[i] * a[j] * b[k]);
                for l in 0..2 {
                    out += q.fourth[i][j][k][l] * a[i] * b[j] * c[k] * d[l];
                }
            }
        }
    }
    out
}

fn factorial(n: usize) -> f64 {
    match n {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        _ => unreachable!("CTN normalizer derivatives only need order <= 4"),
    }
}

fn poly_mul_truncated(a: &[[f64; 5]; 5], b: &[[f64; 5]; 5]) -> [[f64; 5]; 5] {
    let mut out = [[0.0; 5]; 5];
    for ia in 0..=4 {
        for ib in 0..=(4 - ia) {
            let av = a[ia][ib];
            if av == 0.0 {
                continue;
            }
            for ja in 0..=(4 - ia) {
                for jb in 0..=(4 - ia - ja).min(4 - ib) {
                    let bv = b[ja][jb];
                    if bv != 0.0 && ia + ib + ja + jb <= 4 {
                        out[ia + ja][ib + jb] += av * bv;
                    }
                }
            }
        }
    }
    out
}

fn log_normal_cdf_diff_derivatives(
    upper: f64,
    lower: f64,
) -> Result<LogNormalCdfDiffDerivatives, String> {
    let log_z = log_normal_cdf_diff(upper, lower)?;
    if !log_z.is_finite() {
        return Err(format!(
            "normal CDF endpoint log-mass is not finite, lower={lower:.6e}, upper={upper:.6e}"
        ));
    }

    let s_u = [
        0.0,
        1.0,
        -upper,
        upper * upper - 1.0,
        -(upper * upper * upper - 3.0 * upper),
    ];
    let s_l = [
        0.0,
        -1.0,
        lower,
        -(lower * lower - 1.0),
        lower * lower * lower - 3.0 * lower,
    ];

    let mut r = [[0.0; 5]; 5];
    for order in 1..=4 {
        let factor = factorial(order);
        r[order][0] = signed_normal_pdf_ratio(upper, s_u[order], log_z, factor);
        r[0][order] = signed_normal_pdf_ratio(lower, s_l[order], log_z, factor);
        if !(r[order][0].is_finite() && r[0][order].is_finite()) {
            return Err(format!(
                "normal CDF endpoint derivative ratio is not representable at order {order}, \
                 lower={lower:.6e}, upper={upper:.6e}, log_z={log_z:.6e}"
            ));
        }
    }

    let r2 = poly_mul_truncated(&r, &r);
    let r3 = poly_mul_truncated(&r2, &r);
    let r4 = poly_mul_truncated(&r3, &r);
    let mut q = [[0.0; 5]; 5];
    for i in 0..=4 {
        for j in 0..=(4 - i) {
            q[i][j] = r[i][j] - 0.5 * r2[i][j] + r3[i][j] / 3.0 - 0.25 * r4[i][j];
        }
    }

    let mut first = [0.0; 2];
    first[0] = q[1][0];
    first[1] = q[0][1];

    let mut second = [[0.0; 2]; 2];
    let mut third = [[[0.0; 2]; 2]; 2];
    let mut fourth = [[[[0.0; 2]; 2]; 2]; 2];
    for a in 0..2 {
        for b in 0..2 {
            let nu = (a == 0) as usize + (b == 0) as usize;
            let nl = 2 - nu;
            second[a][b] = q[nu][nl] * factorial(nu) * factorial(nl);
            for c in 0..2 {
                let nu = (a == 0) as usize + (b == 0) as usize + (c == 0) as usize;
                let nl = 3 - nu;
                third[a][b][c] = q[nu][nl] * factorial(nu) * factorial(nl);
                for d in 0..2 {
                    let nu = (a == 0) as usize
                        + (b == 0) as usize
                        + (c == 0) as usize
                        + (d == 0) as usize;
                    let nl = 4 - nu;
                    fourth[a][b][c][d] = q[nu][nl] * factorial(nu) * factorial(nl);
                }
            }
        }
    }

    Ok(LogNormalCdfDiffDerivatives {
        log_z,
        first,
        second,
        third,
        fourth,
    })
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
        let (response_lower_basis, response_upper_basis) =
            response_endpoint_value_bases(&resp_transform);

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
            &x_val_kron,
            &x_deriv_kron,
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

        // ----- 5. CTN-specific smoothing seed from likelihood/penalty scales -----
        let initial_log_lambdas =
            ctn_penalty_scale_log_lambdas(&tensor_penalties, &x_val_weighted_gram);

        // Compute response median for anchoring
        let mut sorted_resp = response.to_vec();
        sorted_resp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let resp_median = if sorted_resp.len() % 2 == 1 {
            sorted_resp[sorted_resp.len() / 2]
        } else {
            0.5 * (sorted_resp[sorted_resp.len() / 2 - 1] + sorted_resp[sorted_resp.len() / 2])
        };
        let (response_floor_offset, response_lower_floor_offset, response_upper_floor_offset) =
            response_floor_offsets(response, &resp_knots, resp_median);

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            x_deriv_grid_kron,
            response_val_basis: resp_val,
            response_lower_basis,
            response_upper_basis,
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
            response_floor_offset: Arc::new(response_floor_offset),
            response_lower_floor_offset,
            response_upper_floor_offset,
            covariate_dense_cache: Arc::new(Mutex::new(None)),
            row_quantity_cache: Arc::new(Mutex::new(None)),
        })
    }

    /// Build from a prebuilt response basis, skipping response basis construction.
    ///
    /// For the outer loop where the response basis is precomputed once and reused
    /// across κ iterations.
    pub fn from_prebuilt_response_basis(
        response: &Array1<f64>,
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
        if response.len() != n {
            return Err(format!(
                "response length {} != response basis rows {}",
                response.len(),
                n
            ));
        }
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
        if response_transform.ncols() + 1 != p_resp {
            return Err(format!(
                "response transform columns {} imply p_resp {}, but response value basis has {} columns",
                response_transform.ncols(),
                response_transform.ncols() + 1,
                p_resp
            ));
        }
        let (response_lower_basis, response_upper_basis) =
            response_endpoint_value_bases(&response_transform);

        // Row-wise Kronecker product (operator form).
        let x_val_kron =
            KroneckerDesign::new_khatri_rao(&response_val_basis, covariate_design.clone())?;
        let x_deriv_kron =
            KroneckerDesign::new_khatri_rao(&response_deriv_basis, covariate_design.clone())?;
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_kron.ncols(), p_total);

        let x_deriv_grid_kron = build_monotonicity_derivative_grid_kron(
            response,
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
            response,
            weights,
            offset,
            &x_val_kron,
            &x_deriv_kron,
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

        let initial_log_lambdas =
            ctn_penalty_scale_log_lambdas(&tensor_penalties, &x_val_weighted_gram);

        // Compute response median.
        let mut sorted_resp = response.to_vec();
        sorted_resp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let resp_median = if sorted_resp.len() % 2 == 1 {
            sorted_resp[sorted_resp.len() / 2]
        } else {
            0.5 * (sorted_resp[sorted_resp.len() / 2 - 1] + sorted_resp[sorted_resp.len() / 2])
        };
        let (response_floor_offset, response_lower_floor_offset, response_upper_floor_offset) =
            response_floor_offsets(response, &response_knots, resp_median);

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            x_deriv_grid_kron,
            response_val_basis,
            response_lower_basis,
            response_upper_basis,
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
            response_floor_offset: Arc::new(response_floor_offset),
            response_lower_floor_offset,
            response_upper_floor_offset,
            covariate_dense_cache: Arc::new(Mutex::new(None)),
            row_quantity_cache: Arc::new(Mutex::new(None)),
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
        let offset = self.offset.as_ref() + self.response_floor_offset.as_ref();
        ParameterBlockSpec {
            name: self.block_name.clone(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(self.x_val_kron.clone()))),
            offset,
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

    fn covariate_dense_arc(&self) -> Result<Arc<Array2<f64>>, String> {
        let mut cache = self
            .covariate_dense_cache
            .lock()
            .expect("CTN covariate dense cache mutex poisoned");
        if let Some(cached) = cache.as_ref() {
            return Ok(cached.clone());
        }
        let dense = Arc::new(
            self.covariate_design
                .try_row_chunk(0..self.response_val_basis.nrows())
                .map_err(|e| format!("SCOP covariate dense materialization failed: {e}"))?,
        );
        *cache = Some(dense.clone());
        Ok(dense)
    }

    #[cfg(test)]
    fn scop_endpoint_values(
        &self,
        beta: &Array1<f64>,
        beta_mat: ArrayView2<'_, f64>,
        cov: ArrayView2<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        if beta.len() != p_resp * self.covariate_design.ncols() {
            return Err(format!(
                "SCOP endpoint beta length {} != p_resp({p_resp}) * p_cov({})",
                beta.len(),
                self.covariate_design.ncols()
            ));
        }
        let mut lower = Array1::<f64>::zeros(n);
        let mut upper = Array1::<f64>::zeros(n);
        let mut gamma = vec![0.0; p_resp];
        for i in 0..n {
            let cov_row = cov.row(i);
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
            }
            let mut h_l = self.response_lower_basis[0] * gamma[0]
                + self.offset[i]
                + self.response_lower_floor_offset;
            let mut h_u = self.response_upper_basis[0] * gamma[0]
                + self.offset[i]
                + self.response_upper_floor_offset;
            for k in 1..p_resp {
                h_l += self.response_lower_basis[k] * gamma[k] * gamma[k];
                h_u += self.response_upper_basis[k] * gamma[k] * gamma[k];
            }
            lower[i] = h_l;
            upper[i] = h_u;
        }
        Ok((lower, upper))
    }

    fn row_quantities(
        &self,
        beta: &Array1<f64>,
    ) -> Result<TransformationNormalRowQuantityCache, String> {
        {
            let cache = self
                .row_quantity_cache
                .lock()
                .expect("CTN row quantity cache mutex poisoned");
            if let Some(cached) = cache.as_ref().filter(|cached| cached.matches_beta(beta)) {
                return Ok(cached.clone());
            }
        }

        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP endpoint beta reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc()?;

        // SCOP-CTN: h(y, x) = b(x) + Σ_k γ_k(x)² · I_k(y), with
        // γ_k(x) = ψ(x)ᵀ Γ_{k,:} and h'(y, x) = Σ_k γ_k(x)² · M_k(y).
        // Response column 0 is the unconstrained affine/location component
        // b(x); all remaining response columns are squared shape components.
        //
        // The observed value, derivative value, and finite-support endpoints
        // all depend on the same covariate-side γ_k(x_i).  Compute γ once and
        // fan it out exactly; the previous path projected β through the same
        // covariate design three times per row-quantity build.
        let gamma = fast_ab(cov.as_ref(), &beta_mat.t().to_owned());
        let n = gamma.nrows();
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let rows: Vec<(f64, f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let gamma_row = gamma.row(i);
                let val_row = self.response_val_basis.row(i);
                let deriv_row = self.response_deriv_basis.row(i);
                let g0 = gamma_row[0];
                let offset_i = self.offset[i];
                let mut h_acc = val_row[0] * g0 + offset_i + self.response_floor_offset[i];
                let mut hp_acc = deriv_row[0] * g0 + TRANSFORMATION_MONOTONICITY_EPS;
                let mut lower_acc =
                    self.response_lower_basis[0] * g0 + offset_i + self.response_lower_floor_offset;
                let mut upper_acc =
                    self.response_upper_basis[0] * g0 + offset_i + self.response_upper_floor_offset;
                for k in 1..p_resp {
                    let g_sq = gamma_row[k] * gamma_row[k];
                    h_acc += val_row[k] * g_sq;
                    hp_acc += deriv_row[k] * g_sq;
                    lower_acc += self.response_lower_basis[k] * g_sq;
                    upper_acc += self.response_upper_basis[k] * g_sq;
                }
                (h_acc, hp_acc, lower_acc, upper_acc)
            })
            .collect();
        let mut h = Array1::<f64>::zeros(n);
        let mut h_prime = Array1::<f64>::zeros(n);
        let mut h_lower = Array1::<f64>::zeros(n);
        let mut h_upper = Array1::<f64>::zeros(n);
        for (i, (h_i, hp_i, lower_i, upper_i)) in rows.into_iter().enumerate() {
            h[i] = h_i;
            h_prime[i] = hp_i;
            h_lower[i] = lower_i;
            h_upper[i] = upper_i;
        }
        for (i, &value) in h.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "TransformationNormalFamily row_quantities: h[{i}] = {value} is not finite"
                ));
            }
            if value.abs() > TRANSFORMATION_NORMAL_H_ABS_MAX {
                return Err(format!(
                    "TransformationNormalFamily row_quantities: h[{i}] = {value:.6e} exceeds the standard-normal domain bound ±{TRANSFORMATION_NORMAL_H_ABS_MAX}"
                ));
            }
        }
        // Hard monotonicity / finiteness gate: the reciprocal powers `1/h'^k`
        // for k ∈ {1,2,3,4} feed the gradient, Hessian, and psi-psi outer
        // Hessian formulas. A non-finite or non-positive h' produces +∞ /
        // signed-∞ reciprocals which then collide with zero-valued probe
        // vectors (`v_*_deriv * weights`) to yield NaN entries throughout the
        // dense psi-psi block (`hessian_psi_psi`). The likelihood gate in
        // `evaluate` already rejects such β; surface the same error here so
        // outer-Hessian probe callsites that call `row_quantities` directly
        // (psi/psi second-order terms, etc.) produce a clean Err for the
        // outer evaluator to retreat on, rather than a NaN dense block that
        // routes a flagrant non-finite Hessian back into the planner.
        let mut min_hp = f64::INFINITY;
        let mut nonfinite_idx: Option<usize> = None;
        for (i, &hp) in h_prime.iter().enumerate() {
            if !hp.is_finite() {
                nonfinite_idx = Some(i);
                break;
            }
            if hp < min_hp {
                min_hp = hp;
            }
        }
        if let Some(i) = nonfinite_idx {
            return Err(format!(
                "TransformationNormalFamily row_quantities: h'[{i}] = {} is not finite",
                h_prime[i]
            ));
        }
        if min_hp <= 0.0 {
            return Err(format!(
                "TransformationNormalFamily row_quantities: h' has non-positive values (min = {min_hp:.6e}). \
                 Monotonicity constraint may be violated."
            ));
        }
        // Compute exact f64 row derivatives. If any required reciprocal power
        // is outside the finite representable range, surface an evaluation
        // error so the outer solver can retreat; do not clamp or approximate
        // the analytic Hessian terms.
        let derived = build_transformation_row_derived(
            &h,
            &h_prime,
            &h_lower,
            &h_upper,
            self.weights.as_ref(),
        )?;
        let row_quantities = TransformationNormalRowQuantityCache {
            beta: Arc::new(beta.clone()),
            h: Arc::new(h),
            h_prime: Arc::new(h_prime),
            endpoint_q: Arc::new(derived.endpoint_q),
            log_likelihood: derived.log_likelihood,
        };

        let mut cache = self
            .row_quantity_cache
            .lock()
            .expect("CTN row quantity cache mutex poisoned");
        *cache = Some(row_quantities.clone());
        Ok(row_quantities)
    }

    fn scop_gradient_and_negative_hessian(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(format!(
                "SCOP gradient beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                beta.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let cov = self
            .covariate_design
            .try_row_chunk(0..n)
            .map_err(|e| format!("SCOP gradient requires row chunk: {e}"))?;
        let weights = self.weights.as_ref();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut gradient = Array1::<f64>::zeros(p_total);
        let mut hessian = Array2::<f64>::zeros((p_total, p_total));
        let mut gamma = vec![0.0; p_resp];
        let mut dh_factor = vec![0.0; p_resp];
        let mut dhp_factor = vec![0.0; p_resp];
        let mut second_diag = vec![0.0; p_resp];
        let mut lower_factor = vec![0.0; p_resp];
        let mut upper_factor = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;

            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
            }

            let q = row_quantities.endpoint_q[i];
            lower_factor[0] = self.response_lower_basis[0];
            upper_factor[0] = self.response_upper_basis[0];
            for k in 1..p_resp {
                lower_factor[k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                upper_factor[k] = 2.0 * self.response_upper_basis[k] * gamma[k];
            }

            second_diag.fill(0.0);
            dh_factor[0] = rv[0];
            dhp_factor[0] = rd[0];
            for k in 1..p_resp {
                dh_factor[k] = 2.0 * rv[k] * gamma[k];
                dhp_factor[k] = 2.0 * rd[k] * gamma[k];
                second_diag[k] = 2.0 * (hi * rv[k] - rd[k] * inv_hp);
            }

            for k in 0..p_resp {
                let normalizer_score_factor =
                    q.first[0] * upper_factor[k] + q.first[1] * lower_factor[k];
                let score_factor =
                    wi * (-hi * dh_factor[k] + dhp_factor[k] * inv_hp - normalizer_score_factor);
                for c in 0..p_cov {
                    gradient[k * p_cov + c] += score_factor * cov_row[c];
                }
            }

            for k in 0..p_resp {
                for l in 0..p_resp {
                    let mut block_factor =
                        dh_factor[k] * dh_factor[l] + dhp_factor[k] * dhp_factor[l] * inv_hp_sq;
                    if k == l {
                        block_factor += second_diag[k];
                    }
                    let upper_ab = if k == l && k > 0 {
                        2.0 * self.response_upper_basis[k]
                    } else {
                        0.0
                    };
                    let lower_ab = if k == l && k > 0 {
                        2.0 * self.response_lower_basis[k]
                    } else {
                        0.0
                    };
                    block_factor += q.first[0] * upper_ab
                        + q.first[1] * lower_ab
                        + q.second[0][0] * upper_factor[k] * upper_factor[l]
                        + q.second[0][1] * upper_factor[k] * lower_factor[l]
                        + q.second[1][0] * lower_factor[k] * upper_factor[l]
                        + q.second[1][1] * lower_factor[k] * lower_factor[l];
                    block_factor *= wi;
                    if block_factor == 0.0 {
                        continue;
                    }
                    for c in 0..p_cov {
                        let row_idx = k * p_cov + c;
                        let left = block_factor * cov_row[c];
                        for d in 0..p_cov {
                            hessian[[row_idx, l * p_cov + d]] += left * cov_row[d];
                        }
                    }
                }
            }
        }

        Ok((gradient, hessian))
    }

    fn scop_gradient(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(format!(
                "SCOP gradient beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                beta.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let cov = self
            .covariate_design
            .try_row_chunk(0..n)
            .map_err(|e| format!("SCOP gradient requires row chunk: {e}"))?;
        let weights = self.weights.as_ref();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut gradient = Array1::<f64>::zeros(p_total);
        let mut gamma = vec![0.0; p_resp];
        let mut lower_factor = vec![0.0; p_resp];
        let mut upper_factor = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hi = h[i];
            let inv_hp = 1.0 / h_prime[i];

            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
            }

            let q = row_quantities.endpoint_q[i];
            lower_factor[0] = self.response_lower_basis[0];
            upper_factor[0] = self.response_upper_basis[0];
            for k in 1..p_resp {
                lower_factor[k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                upper_factor[k] = 2.0 * self.response_upper_basis[k] * gamma[k];
            }

            let normalizer_score0 = q.first[0] * upper_factor[0] + q.first[1] * lower_factor[0];
            let score0 = wi * (-hi * rv[0] + rd[0] * inv_hp - normalizer_score0);
            for c in 0..p_cov {
                gradient[c] += score0 * cov_row[c];
            }
            for k in 1..p_resp {
                let normalizer_score = q.first[0] * upper_factor[k] + q.first[1] * lower_factor[k];
                let score_factor =
                    wi * (2.0 * gamma[k] * (-hi * rv[k] + rd[k] * inv_hp) - normalizer_score);
                let offset = k * p_cov;
                for c in 0..p_cov {
                    gradient[offset + c] += score_factor * cov_row[c];
                }
            }
        }

        Ok(gradient)
    }

    /// Directional derivative of the SCOP negative Hessian.
    ///
    /// For one observation with covariate row `c`, response value row `r`, and
    /// derivative row `m`, define `g_k = beta_k·c`, `u_k = direction_k·c`,
    ///
    /// `h  = r_0 g_0 + Σ_{k>0} r_k g_k^2`
    /// `hp = m_0 g_0 + Σ_{k>0} m_k g_k^2`
    /// `Hu  = r_0 u_0 + Σ_{k>0} 2 r_k g_k u_k`
    /// `HPu = m_0 u_0 + Σ_{k>0} 2 m_k g_k u_k`.
    ///
    /// For coefficient `a=(k,p)`:
    ///
    /// `h_a  = r_k c_p` for `k=0`, else `2 r_k g_k c_p`
    /// `hp_a = m_k c_p` for `k=0`, else `2 m_k g_k c_p`
    /// `D h_a[u]  = 0` for `k=0`, else `2 r_k u_k c_p`
    /// `D hp_a[u] = 0` for `k=0`, else `2 m_k u_k c_p`.
    ///
    /// The per-row negative Hessian block is
    ///
    /// `H_ab = w [ h_a h_b + h h_ab + hp_a hp_b / hp^2 - hp_ab / hp ]`,
    ///
    /// where `h_ab = 2 r_k c_p c_q` and `hp_ab = 2 m_k c_p c_q` only when
    /// `a` and `b` are in the same squared shape component `k>0`; otherwise
    /// both second derivatives are zero. Differentiating this expression in
    /// direction `u` gives the exact formula implemented below:
    ///
    /// `D H_ab[u] = w [ (D h_a)h_b + h_a(D h_b) + Hu h_ab
    ///              + ((D hp_a)hp_b + hp_a(D hp_b))/hp^2
    ///              - 2 hp_a hp_b HPu/hp^3 + hp_ab HPu/hp^2 ]`.
    fn scop_hessian_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction.len() != p_total {
            return Err(format!(
                "SCOP Hessian directional derivative length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP direction reshape failed: {e}"))?;
        let cov = self
            .covariate_design
            .try_row_chunk(0..n)
            .map_err(|e| format!("SCOP Hessian directional derivative requires row chunk: {e}"))?;
        let weights = self.weights.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut out = Array2::<f64>::zeros((p_total, p_total));

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;

            let mut gamma = vec![0.0; p_resp];
            let mut gamma_dir = vec![0.0; p_resp];
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
            }

            let mut h_dir = rv[0] * gamma_dir[0];
            let mut hp_dir = rd[0] * gamma_dir[0];
            let mut endpoint_dir = [
                self.response_upper_basis[0] * gamma_dir[0],
                self.response_lower_basis[0] * gamma_dir[0],
            ];
            for k in 1..p_resp {
                h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                endpoint_dir[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_dir[k];
                endpoint_dir[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_dir[k];
            }
            let q = row_quantities.endpoint_q[i];

            let mut h_factor = vec![0.0; p_resp];
            let mut hp_factor = vec![0.0; p_resp];
            let mut h_factor_dir = vec![0.0; p_resp];
            let mut hp_factor_dir = vec![0.0; p_resp];
            let mut endpoint_factor = [vec![0.0; p_resp], vec![0.0; p_resp]];
            let mut endpoint_factor_dir = [vec![0.0; p_resp], vec![0.0; p_resp]];
            h_factor[0] = rv[0];
            hp_factor[0] = rd[0];
            endpoint_factor[0][0] = self.response_upper_basis[0];
            endpoint_factor[1][0] = self.response_lower_basis[0];
            for k in 1..p_resp {
                h_factor[k] = 2.0 * rv[k] * gamma[k];
                hp_factor[k] = 2.0 * rd[k] * gamma[k];
                h_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                hp_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
                endpoint_factor[0][k] = 2.0 * self.response_upper_basis[k] * gamma[k];
                endpoint_factor[1][k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                endpoint_factor_dir[0][k] = 2.0 * self.response_upper_basis[k] * gamma_dir[k];
                endpoint_factor_dir[1][k] = 2.0 * self.response_lower_basis[k] * gamma_dir[k];
            }

            for k in 0..p_resp {
                for l in 0..p_resp {
                    let same_shape = k == l && k > 0;
                    let mut normalizer_block = 0.0;
                    for a in 0..2 {
                        let h_a_ab = if same_shape {
                            2.0 * if a == 0 {
                                self.response_upper_basis[k]
                            } else {
                                self.response_lower_basis[k]
                            }
                        } else {
                            0.0
                        };
                        for b in 0..2 {
                            normalizer_block += q.second[a][b] * endpoint_dir[b] * h_a_ab;
                            normalizer_block += q.second[a][b]
                                * (endpoint_factor_dir[a][k] * endpoint_factor[b][l]
                                    + endpoint_factor[a][k] * endpoint_factor_dir[b][l]);
                            for c_ep in 0..2 {
                                normalizer_block += q.third[a][b][c_ep]
                                    * endpoint_dir[c_ep]
                                    * endpoint_factor[a][k]
                                    * endpoint_factor[b][l];
                            }
                        }
                    }
                    for c in 0..p_cov {
                        let row_idx = k * p_cov + c;
                        let h_a = h_factor[k] * cov_row[c];
                        let hp_a = hp_factor[k] * cov_row[c];
                        let dh_a = h_factor_dir[k] * cov_row[c];
                        let dhp_a = hp_factor_dir[k] * cov_row[c];
                        for d in 0..p_cov {
                            let col_idx = l * p_cov + d;
                            let h_b = h_factor[l] * cov_row[d];
                            let hp_b = hp_factor[l] * cov_row[d];
                            let dh_b = h_factor_dir[l] * cov_row[d];
                            let dhp_b = hp_factor_dir[l] * cov_row[d];
                            let (h_ab, hp_ab) = if same_shape {
                                (
                                    2.0 * rv[k] * cov_row[c] * cov_row[d],
                                    2.0 * rd[k] * cov_row[c] * cov_row[d],
                                )
                            } else {
                                (0.0, 0.0)
                            };
                            let value = dh_a * h_b
                                + h_a * dh_b
                                + h_dir * h_ab
                                + (dhp_a * hp_b + hp_a * dhp_b) * inv_hp_sq
                                - 2.0 * hp_a * hp_b * hp_dir * inv_hp_cu
                                + hp_ab * hp_dir * inv_hp_sq
                                + normalizer_block * cov_row[c] * cov_row[d];
                            out[[row_idx, col_idx]] += wi * value;
                        }
                    }
                }
            }
        }

        Ok(0.5 * (&out + &out.t()))
    }

    /// Second directional derivative of the SCOP negative Hessian.
    ///
    /// Reuse the notation from `scop_hessian_directional_derivative`, with
    /// two beta-space directions `u` and `v`. Since each shape component is
    /// quadratic in `g_k`, all third beta derivatives of `h` and `hp` are zero,
    /// while the mixed second directional derivatives are
    ///
    /// `Huv  = Σ_{k>0} 2 r_k u_k v_k`
    /// `HPuv = Σ_{k>0} 2 m_k u_k v_k`.
    ///
    /// Differentiating
    /// `D H_ab[u] = w[(D h_a)h_b + h_a(D h_b) + Hu h_ab
    ///              + ((D hp_a)hp_b + hp_a(D hp_b))/hp^2
    ///              - 2 hp_a hp_b HPu/hp^3 + hp_ab HPu/hp^2]`
    /// once more in direction `v` gives the expression implemented below:
    ///
    /// `D²H_ab[u,v] = w[
    ///      h_{a,u} h_{b,v} + h_{a,v} h_{b,u} + Huv h_ab
    ///    + (hp_{a,u} hp_{b,v} + hp_{a,v} hp_{b,u})/hp²
    ///    - 2(hp_{a,u} hp_b + hp_a hp_{b,u}) HPv/hp³
    ///    - 2(hp_{a,v} hp_b + hp_a hp_{b,v}) HPu/hp³
    ///    - 2 hp_a hp_b HPuv/hp³
    ///    + 6 hp_a hp_b HPu HPv/hp⁴
    ///    + hp_ab HPuv/hp²
    ///    - 2 hp_ab HPu HPv/hp³ ]`.
    fn scop_hessian_second_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction_u.len() != p_total || direction_v.len() != p_total {
            return Err(format!(
                "SCOP Hessian second directional derivative length mismatch: beta={}, u={}, v={}, expected={p_total}",
                beta.len(),
                direction_u.len(),
                direction_v.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let dir_u_mat = direction_u
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP u direction reshape failed: {e}"))?;
        let dir_v_mat = direction_v
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP v direction reshape failed: {e}"))?;
        let cov = self.covariate_design.try_row_chunk(0..n).map_err(|e| {
            format!("SCOP Hessian second directional derivative requires row chunk: {e}")
        })?;
        let weights = self.weights.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut out = Array2::<f64>::zeros((p_total, p_total));

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let inv_hp_qu = inv_hp_sq * inv_hp_sq;

            let mut gamma = vec![0.0; p_resp];
            let mut gamma_u = vec![0.0; p_resp];
            let mut gamma_v = vec![0.0; p_resp];
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_u[k] = dir_u_mat.row(k).dot(&cov_row);
                gamma_v[k] = dir_v_mat.row(k).dot(&cov_row);
            }

            let mut hp_u = rd[0] * gamma_u[0];
            let mut hp_v = rd[0] * gamma_v[0];
            let mut h_uv = 0.0;
            let mut hp_uv = 0.0;
            let mut endpoint_u = [
                self.response_upper_basis[0] * gamma_u[0],
                self.response_lower_basis[0] * gamma_u[0],
            ];
            let mut endpoint_v = [
                self.response_upper_basis[0] * gamma_v[0],
                self.response_lower_basis[0] * gamma_v[0],
            ];
            let mut endpoint_uv = [0.0, 0.0];
            for k in 1..p_resp {
                hp_u += 2.0 * rd[k] * gamma[k] * gamma_u[k];
                hp_v += 2.0 * rd[k] * gamma[k] * gamma_v[k];
                h_uv += 2.0 * rv[k] * gamma_u[k] * gamma_v[k];
                hp_uv += 2.0 * rd[k] * gamma_u[k] * gamma_v[k];
                endpoint_u[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_u[k];
                endpoint_u[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_u[k];
                endpoint_v[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_v[k];
                endpoint_v[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_v[k];
                endpoint_uv[0] += 2.0 * self.response_upper_basis[k] * gamma_u[k] * gamma_v[k];
                endpoint_uv[1] += 2.0 * self.response_lower_basis[k] * gamma_u[k] * gamma_v[k];
            }
            let q = row_quantities.endpoint_q[i];

            let mut h_factor = vec![0.0; p_resp];
            let mut hp_factor = vec![0.0; p_resp];
            let mut h_factor_u = vec![0.0; p_resp];
            let mut hp_factor_u = vec![0.0; p_resp];
            let mut h_factor_v = vec![0.0; p_resp];
            let mut hp_factor_v = vec![0.0; p_resp];
            let mut endpoint_factor = [vec![0.0; p_resp], vec![0.0; p_resp]];
            let mut endpoint_factor_u = [vec![0.0; p_resp], vec![0.0; p_resp]];
            let mut endpoint_factor_v = [vec![0.0; p_resp], vec![0.0; p_resp]];
            h_factor[0] = rv[0];
            hp_factor[0] = rd[0];
            endpoint_factor[0][0] = self.response_upper_basis[0];
            endpoint_factor[1][0] = self.response_lower_basis[0];
            for k in 1..p_resp {
                h_factor[k] = 2.0 * rv[k] * gamma[k];
                hp_factor[k] = 2.0 * rd[k] * gamma[k];
                h_factor_u[k] = 2.0 * rv[k] * gamma_u[k];
                hp_factor_u[k] = 2.0 * rd[k] * gamma_u[k];
                h_factor_v[k] = 2.0 * rv[k] * gamma_v[k];
                hp_factor_v[k] = 2.0 * rd[k] * gamma_v[k];
                endpoint_factor[0][k] = 2.0 * self.response_upper_basis[k] * gamma[k];
                endpoint_factor[1][k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                endpoint_factor_u[0][k] = 2.0 * self.response_upper_basis[k] * gamma_u[k];
                endpoint_factor_u[1][k] = 2.0 * self.response_lower_basis[k] * gamma_u[k];
                endpoint_factor_v[0][k] = 2.0 * self.response_upper_basis[k] * gamma_v[k];
                endpoint_factor_v[1][k] = 2.0 * self.response_lower_basis[k] * gamma_v[k];
            }

            for k in 0..p_resp {
                for l in 0..p_resp {
                    let same_shape = k == l && k > 0;
                    let mut normalizer_block = 0.0;
                    for a in 0..2 {
                        let h_a_ab = if same_shape {
                            2.0 * if a == 0 {
                                self.response_upper_basis[k]
                            } else {
                                self.response_lower_basis[k]
                            }
                        } else {
                            0.0
                        };
                        for b in 0..2 {
                            normalizer_block += q.second[a][b] * endpoint_uv[b] * h_a_ab;
                            for c_ep in 0..2 {
                                normalizer_block +=
                                    q.third[a][b][c_ep] * endpoint_v[c_ep] * endpoint_u[b] * h_a_ab;
                                normalizer_block += q.third[a][b][c_ep]
                                    * endpoint_uv[c_ep]
                                    * endpoint_factor[a][k]
                                    * endpoint_factor[b][l];
                                normalizer_block += q.third[a][b][c_ep]
                                    * endpoint_u[c_ep]
                                    * (endpoint_factor_v[a][k] * endpoint_factor[b][l]
                                        + endpoint_factor[a][k] * endpoint_factor_v[b][l]);
                                normalizer_block += q.third[a][b][c_ep]
                                    * endpoint_v[c_ep]
                                    * endpoint_factor_u[a][k]
                                    * endpoint_factor[b][l];
                                normalizer_block += q.third[a][b][c_ep]
                                    * endpoint_v[c_ep]
                                    * endpoint_factor[a][k]
                                    * endpoint_factor_u[b][l];
                                for d_ep in 0..2 {
                                    normalizer_block += q.fourth[a][b][c_ep][d_ep]
                                        * endpoint_v[d_ep]
                                        * endpoint_u[c_ep]
                                        * endpoint_factor[a][k]
                                        * endpoint_factor[b][l];
                                }
                            }
                            normalizer_block += q.second[a][b]
                                * (endpoint_factor_u[a][k] * endpoint_factor_v[b][l]
                                    + endpoint_factor_v[a][k] * endpoint_factor_u[b][l]);
                        }
                    }
                    for c in 0..p_cov {
                        let row_idx = k * p_cov + c;
                        let hp_a = hp_factor[k] * cov_row[c];
                        let dh_a_u = h_factor_u[k] * cov_row[c];
                        let dhp_a_u = hp_factor_u[k] * cov_row[c];
                        let dh_a_v = h_factor_v[k] * cov_row[c];
                        let dhp_a_v = hp_factor_v[k] * cov_row[c];
                        for d in 0..p_cov {
                            let col_idx = l * p_cov + d;
                            let hp_b = hp_factor[l] * cov_row[d];
                            let dh_b_u = h_factor_u[l] * cov_row[d];
                            let dhp_b_u = hp_factor_u[l] * cov_row[d];
                            let dh_b_v = h_factor_v[l] * cov_row[d];
                            let dhp_b_v = hp_factor_v[l] * cov_row[d];
                            let (h_ab, hp_ab) = if same_shape {
                                (
                                    2.0 * rv[k] * cov_row[c] * cov_row[d],
                                    2.0 * rd[k] * cov_row[c] * cov_row[d],
                                )
                            } else {
                                (0.0, 0.0)
                            };
                            let value = dh_a_u * dh_b_v
                                + dh_a_v * dh_b_u
                                + h_uv * h_ab
                                + (dhp_a_u * dhp_b_v + dhp_a_v * dhp_b_u) * inv_hp_sq
                                - 2.0 * (dhp_a_u * hp_b + hp_a * dhp_b_u) * hp_v * inv_hp_cu
                                - 2.0 * (dhp_a_v * hp_b + hp_a * dhp_b_v) * hp_u * inv_hp_cu
                                - 2.0 * hp_a * hp_b * hp_uv * inv_hp_cu
                                + 6.0 * hp_a * hp_b * hp_u * hp_v * inv_hp_qu
                                + hp_ab * hp_uv * inv_hp_sq
                                - 2.0 * hp_ab * hp_u * hp_v * inv_hp_cu
                                + normalizer_block * cov_row[c] * cov_row[d];
                            out[[row_idx, col_idx]] += wi * value;
                        }
                    }
                }
            }
        }

        Ok(0.5 * (&out + &out.t()))
    }

    fn scop_hessian_matvec(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || probe.len() != p_total {
            return Err(format!(
                "SCOP Hessian matvec length mismatch: beta={}, probe={}, expected={p_total}",
                beta.len(),
                probe.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let probe_mat = probe
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP probe reshape failed: {e}"))?;
        let cov = self
            .covariate_design
            .try_row_chunk(0..n)
            .map_err(|e| format!("SCOP Hessian matvec requires row chunk: {e}"))?;
        let weights = self.weights.as_ref();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut out = Array1::<f64>::zeros(p_total);

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;

            let mut gamma = vec![0.0; p_resp];
            let mut probe_gamma = vec![0.0; p_resp];
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                probe_gamma[k] = probe_mat.row(k).dot(&cov_row);
            }

            let mut h_probe = rv[0] * probe_gamma[0];
            let mut hp_probe = rd[0] * probe_gamma[0];
            let mut lower_probe = self.response_lower_basis[0] * probe_gamma[0];
            let mut upper_probe = self.response_upper_basis[0] * probe_gamma[0];
            for k in 1..p_resp {
                h_probe += 2.0 * rv[k] * gamma[k] * probe_gamma[k];
                hp_probe += 2.0 * rd[k] * gamma[k] * probe_gamma[k];
                lower_probe += 2.0 * self.response_lower_basis[k] * gamma[k] * probe_gamma[k];
                upper_probe += 2.0 * self.response_upper_basis[k] * gamma[k] * probe_gamma[k];
            }
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let h_factor = if k == 0 {
                    rv[0]
                } else {
                    2.0 * rv[k] * gamma[k]
                };
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let second_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * (hi * rv[k] - rd[k] * inv_hp) * probe_gamma[k]
                };
                let lower_factor = if k == 0 {
                    self.response_lower_basis[0]
                } else {
                    2.0 * self.response_lower_basis[k] * gamma[k]
                };
                let upper_factor = if k == 0 {
                    self.response_upper_basis[0]
                } else {
                    2.0 * self.response_upper_basis[k] * gamma[k]
                };
                let lower_factor_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * self.response_lower_basis[k] * probe_gamma[k]
                };
                let upper_factor_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * self.response_upper_basis[k] * probe_gamma[k]
                };
                let normalizer_probe = q.first[0] * upper_factor_probe
                    + q.first[1] * lower_factor_probe
                    + (q.second[0][0] * upper_factor + q.second[1][0] * lower_factor) * upper_probe
                    + (q.second[0][1] * upper_factor + q.second[1][1] * lower_factor) * lower_probe;
                let scalar = wi
                    * (h_factor * h_probe
                        + hp_factor * hp_probe * inv_hp_sq
                        + second_probe
                        + normalizer_probe);
                for c in 0..p_cov {
                    out[k * p_cov + c] += scalar * cov_row[c];
                }
            }
        }

        Ok(out)
    }

    fn scop_hessian_directional_matvec(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction.len() != p_total || probe.len() != p_total {
            return Err(format!(
                "SCOP dH matvec length mismatch: beta={}, direction={}, probe={}, expected={p_total}",
                beta.len(),
                direction.len(),
                probe.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP direction reshape failed: {e}"))?;
        let probe_mat = probe
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP probe reshape failed: {e}"))?;
        let cov = self
            .covariate_design
            .try_row_chunk(0..n)
            .map_err(|e| format!("SCOP dH matvec requires row chunk: {e}"))?;
        let weights = self.weights.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut out = Array1::<f64>::zeros(p_total);

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;

            let mut gamma = vec![0.0; p_resp];
            let mut gamma_dir = vec![0.0; p_resp];
            let mut gamma_probe = vec![0.0; p_resp];
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                gamma_probe[k] = probe_mat.row(k).dot(&cov_row);
            }

            let mut h_dir = rv[0] * gamma_dir[0];
            let mut hp_dir = rd[0] * gamma_dir[0];
            let mut h_probe = rv[0] * gamma_probe[0];
            let mut hp_probe = rd[0] * gamma_probe[0];
            let mut h_dir_probe = 0.0;
            let mut hp_dir_probe = 0.0;
            let mut endpoint_dir = [
                self.response_upper_basis[0] * gamma_dir[0],
                self.response_lower_basis[0] * gamma_dir[0],
            ];
            let mut endpoint_probe = [
                self.response_upper_basis[0] * gamma_probe[0],
                self.response_lower_basis[0] * gamma_probe[0],
            ];
            let mut endpoint_dir_probe = [0.0, 0.0];
            for k in 1..p_resp {
                h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                h_probe += 2.0 * rv[k] * gamma[k] * gamma_probe[k];
                hp_probe += 2.0 * rd[k] * gamma[k] * gamma_probe[k];
                h_dir_probe += 2.0 * rv[k] * gamma_dir[k] * gamma_probe[k];
                hp_dir_probe += 2.0 * rd[k] * gamma_dir[k] * gamma_probe[k];
                endpoint_dir[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_dir[k];
                endpoint_dir[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_dir[k];
                endpoint_probe[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_probe[k];
                endpoint_probe[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_probe[k];
                endpoint_dir_probe[0] +=
                    2.0 * self.response_upper_basis[k] * gamma_dir[k] * gamma_probe[k];
                endpoint_dir_probe[1] +=
                    2.0 * self.response_lower_basis[k] * gamma_dir[k] * gamma_probe[k];
            }
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let h_factor = if k == 0 {
                    rv[0]
                } else {
                    2.0 * rv[k] * gamma[k]
                };
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let h_factor_dir = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_dir[k]
                };
                let hp_factor_dir = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_dir[k]
                };
                let h_second_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_probe[k]
                };
                let hp_second_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_probe[k]
                };
                let endpoint_factor = [
                    if k == 0 {
                        self.response_upper_basis[0]
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma[k]
                    },
                    if k == 0 {
                        self.response_lower_basis[0]
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma[k]
                    },
                ];
                let endpoint_factor_dir = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_dir[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_dir[k]
                    },
                ];
                let endpoint_factor_probe = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_probe[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_probe[k]
                    },
                ];
                let mut normalizer_scalar = 0.0;
                for a in 0..2 {
                    for b in 0..2 {
                        normalizer_scalar +=
                            q.second[a][b] * endpoint_dir[b] * endpoint_factor_probe[a];
                        normalizer_scalar += q.second[a][b]
                            * (endpoint_factor_dir[a] * endpoint_probe[b]
                                + endpoint_factor[a] * endpoint_dir_probe[b]);
                        for c_ep in 0..2 {
                            normalizer_scalar += q.third[a][b][c_ep]
                                * endpoint_dir[c_ep]
                                * endpoint_factor[a]
                                * endpoint_probe[b];
                        }
                    }
                }
                let scalar = wi
                    * (h_factor_dir * h_probe
                        + h_factor * h_dir_probe
                        + h_dir * h_second_probe
                        + (hp_factor_dir * hp_probe + hp_factor * hp_dir_probe) * inv_hp_sq
                        - 2.0 * hp_factor * hp_probe * hp_dir * inv_hp_cu
                        + hp_second_probe * hp_dir * inv_hp_sq
                        + normalizer_scalar);
                for c in 0..p_cov {
                    out[k * p_cov + c] += scalar * cov_row[c];
                }
            }
        }

        Ok(out)
    }

    fn scop_hessian_second_directional_matvec(
        &self,
        beta: &Array1<f64>,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total
            || direction_u.len() != p_total
            || direction_v.len() != p_total
            || probe.len() != p_total
        {
            return Err(format!(
                "SCOP d2H matvec length mismatch: beta={}, u={}, v={}, probe={}, expected={p_total}",
                beta.len(),
                direction_u.len(),
                direction_v.len(),
                probe.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let dir_u_mat = direction_u
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP u direction reshape failed: {e}"))?;
        let dir_v_mat = direction_v
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP v direction reshape failed: {e}"))?;
        let probe_mat = probe
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP probe reshape failed: {e}"))?;
        let cov = self
            .covariate_design
            .try_row_chunk(0..n)
            .map_err(|e| format!("SCOP d2H matvec requires row chunk: {e}"))?;
        let weights = self.weights.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut out = Array1::<f64>::zeros(p_total);

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let inv_hp_qu = inv_hp_sq * inv_hp_sq;

            let mut gamma = vec![0.0; p_resp];
            let mut gamma_u = vec![0.0; p_resp];
            let mut gamma_v = vec![0.0; p_resp];
            let mut gamma_probe = vec![0.0; p_resp];
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_u[k] = dir_u_mat.row(k).dot(&cov_row);
                gamma_v[k] = dir_v_mat.row(k).dot(&cov_row);
                gamma_probe[k] = probe_mat.row(k).dot(&cov_row);
            }

            let mut hp_u = rd[0] * gamma_u[0];
            let mut hp_v = rd[0] * gamma_v[0];
            let mut hp_probe = rd[0] * gamma_probe[0];
            let mut h_uv = 0.0;
            let mut hp_uv = 0.0;
            let mut h_u_probe = 0.0;
            let mut hp_u_probe = 0.0;
            let mut h_v_probe = 0.0;
            let mut hp_v_probe = 0.0;
            let mut endpoint_u = [
                self.response_upper_basis[0] * gamma_u[0],
                self.response_lower_basis[0] * gamma_u[0],
            ];
            let mut endpoint_v = [
                self.response_upper_basis[0] * gamma_v[0],
                self.response_lower_basis[0] * gamma_v[0],
            ];
            let mut endpoint_probe = [
                self.response_upper_basis[0] * gamma_probe[0],
                self.response_lower_basis[0] * gamma_probe[0],
            ];
            let mut endpoint_uv = [0.0, 0.0];
            let mut endpoint_u_probe = [0.0, 0.0];
            let mut endpoint_v_probe = [0.0, 0.0];
            for k in 1..p_resp {
                hp_u += 2.0 * rd[k] * gamma[k] * gamma_u[k];
                hp_v += 2.0 * rd[k] * gamma[k] * gamma_v[k];
                hp_probe += 2.0 * rd[k] * gamma[k] * gamma_probe[k];
                h_uv += 2.0 * rv[k] * gamma_u[k] * gamma_v[k];
                hp_uv += 2.0 * rd[k] * gamma_u[k] * gamma_v[k];
                h_u_probe += 2.0 * rv[k] * gamma_u[k] * gamma_probe[k];
                hp_u_probe += 2.0 * rd[k] * gamma_u[k] * gamma_probe[k];
                h_v_probe += 2.0 * rv[k] * gamma_v[k] * gamma_probe[k];
                hp_v_probe += 2.0 * rd[k] * gamma_v[k] * gamma_probe[k];
                endpoint_u[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_u[k];
                endpoint_u[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_u[k];
                endpoint_v[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_v[k];
                endpoint_v[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_v[k];
                endpoint_probe[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_probe[k];
                endpoint_probe[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_probe[k];
                endpoint_uv[0] += 2.0 * self.response_upper_basis[k] * gamma_u[k] * gamma_v[k];
                endpoint_uv[1] += 2.0 * self.response_lower_basis[k] * gamma_u[k] * gamma_v[k];
                endpoint_u_probe[0] +=
                    2.0 * self.response_upper_basis[k] * gamma_u[k] * gamma_probe[k];
                endpoint_u_probe[1] +=
                    2.0 * self.response_lower_basis[k] * gamma_u[k] * gamma_probe[k];
                endpoint_v_probe[0] +=
                    2.0 * self.response_upper_basis[k] * gamma_v[k] * gamma_probe[k];
                endpoint_v_probe[1] +=
                    2.0 * self.response_lower_basis[k] * gamma_v[k] * gamma_probe[k];
            }
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let h_factor_u = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_u[k]
                };
                let hp_factor_u = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_u[k]
                };
                let h_factor_v = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_v[k]
                };
                let hp_factor_v = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_v[k]
                };
                let h_second_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_probe[k]
                };
                let hp_second_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_probe[k]
                };
                let endpoint_factor = [
                    if k == 0 {
                        self.response_upper_basis[0]
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma[k]
                    },
                    if k == 0 {
                        self.response_lower_basis[0]
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma[k]
                    },
                ];
                let endpoint_factor_u = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_u[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_u[k]
                    },
                ];
                let endpoint_factor_v = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_v[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_v[k]
                    },
                ];
                let endpoint_factor_probe = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_probe[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_probe[k]
                    },
                ];
                let mut normalizer_scalar = 0.0;
                for a in 0..2 {
                    for b in 0..2 {
                        normalizer_scalar +=
                            q.second[a][b] * endpoint_uv[b] * endpoint_factor_probe[a];
                        for c_ep in 0..2 {
                            normalizer_scalar += q.third[a][b][c_ep]
                                * endpoint_v[c_ep]
                                * endpoint_u[b]
                                * endpoint_factor_probe[a];
                            normalizer_scalar += q.third[a][b][c_ep]
                                * endpoint_uv[c_ep]
                                * endpoint_factor[a]
                                * endpoint_probe[b];
                            normalizer_scalar += q.third[a][b][c_ep]
                                * endpoint_u[c_ep]
                                * (endpoint_factor_v[a] * endpoint_probe[b]
                                    + endpoint_factor[a] * endpoint_v_probe[b]);
                            normalizer_scalar += q.third[a][b][c_ep]
                                * endpoint_v[c_ep]
                                * endpoint_factor_u[a]
                                * endpoint_probe[b];
                            normalizer_scalar += q.third[a][b][c_ep]
                                * endpoint_v[c_ep]
                                * endpoint_factor[a]
                                * endpoint_u_probe[b];
                            for d_ep in 0..2 {
                                normalizer_scalar += q.fourth[a][b][c_ep][d_ep]
                                    * endpoint_v[d_ep]
                                    * endpoint_u[c_ep]
                                    * endpoint_factor[a]
                                    * endpoint_probe[b];
                            }
                        }
                        normalizer_scalar += q.second[a][b]
                            * (endpoint_factor_u[a] * endpoint_v_probe[b]
                                + endpoint_factor_v[a] * endpoint_u_probe[b]);
                    }
                }
                let scalar = wi
                    * (h_factor_u * h_v_probe
                        + h_factor_v * h_u_probe
                        + h_uv * h_second_probe
                        + (hp_factor_u * hp_v_probe + hp_factor_v * hp_u_probe) * inv_hp_sq
                        - 2.0
                            * (hp_factor_u * hp_probe + hp_factor * hp_u_probe)
                            * hp_v
                            * inv_hp_cu
                        - 2.0
                            * (hp_factor_v * hp_probe + hp_factor * hp_v_probe)
                            * hp_u
                            * inv_hp_cu
                        - 2.0 * hp_factor * hp_probe * hp_uv * inv_hp_cu
                        + 6.0 * hp_factor * hp_probe * hp_u * hp_v * inv_hp_qu
                        + hp_second_probe * hp_uv * inv_hp_sq
                        - 2.0 * hp_second_probe * hp_u * hp_v * inv_hp_cu
                        + normalizer_scalar);
                for c in 0..p_cov {
                    out[k * p_cov + c] += scalar * cov_row[c];
                }
            }
        }

        Ok(out)
    }

    fn scop_hessian_diagonal(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(format!(
                "SCOP Hessian diagonal beta length {} != expected {p_total}",
                beta.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let cov = self
            .covariate_design
            .try_row_chunk(0..n)
            .map_err(|e| format!("SCOP Hessian diagonal requires row chunk: {e}"))?;
        let weights = self.weights.as_ref();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut diag = Array1::<f64>::zeros(p_total);
        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let mut gamma = vec![0.0; p_resp];
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
            }
            let q = row_quantities.endpoint_q[i];
            for k in 0..p_resp {
                let h_factor = if k == 0 {
                    rv[0]
                } else {
                    2.0 * rv[k] * gamma[k]
                };
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let second = if k == 0 {
                    0.0
                } else {
                    2.0 * (hi * rv[k] - rd[k] * inv_hp)
                };
                let lower_factor = if k == 0 {
                    self.response_lower_basis[0]
                } else {
                    2.0 * self.response_lower_basis[k] * gamma[k]
                };
                let upper_factor = if k == 0 {
                    self.response_upper_basis[0]
                } else {
                    2.0 * self.response_upper_basis[k] * gamma[k]
                };
                let lower_second = if k == 0 {
                    0.0
                } else {
                    2.0 * self.response_lower_basis[k]
                };
                let upper_second = if k == 0 {
                    0.0
                } else {
                    2.0 * self.response_upper_basis[k]
                };
                let normalizer_second = q.first[0] * upper_second
                    + q.first[1] * lower_second
                    + q.second[0][0] * upper_factor * upper_factor
                    + (q.second[0][1] + q.second[1][0]) * upper_factor * lower_factor
                    + q.second[1][1] * lower_factor * lower_factor;
                for c in 0..p_cov {
                    let cc = cov_row[c] * cov_row[c];
                    diag[k * p_cov + c] += wi
                        * ((h_factor * h_factor + hp_factor * hp_factor * inv_hp_sq) * cc
                            + second * cc
                            + normalizer_second * cc);
                }
            }
        }
        Ok(diag)
    }

    /// First derivative of the profiled objective pieces with respect to one
    /// covariate-basis deformation axis `psi`.
    ///
    /// The SCOP map is the same as above, but `psi` differentiates the
    /// covariate row, not `beta`: `g_k,psi = beta_k·c_psi`. Thus
    /// `h_psi = r_0 g_0,psi + Σ_{k>0} 2 r_k g_k g_k,psi` and likewise for
    /// `hp_psi`. The objective contribution is
    ///
    /// `L_psi = w [ h h_psi - hp_psi / hp ]`
    ///
    /// because the minimized criterion is `0.5 h^2 - log(hp)`. Differentiating
    /// once with respect to `beta_a` gives `score_psi`; differentiating again
    /// gives `hessian_psi`. The code below expands those derivatives directly,
    /// including both paths through `c` and `c_psi` for same-component squared
    /// shape terms.
    fn scop_psi_terms(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        op: &TensorKroneckerPsiOperator,
        axis: usize,
    ) -> Result<ExactNewtonJointPsiTerms, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(format!(
                "SCOP psi terms beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                beta.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi beta reshape failed: {e}"))?;
        let cov = self
            .covariate_design
            .try_row_chunk(0..n)
            .map_err(|e| format!("SCOP psi terms require covariate row chunk: {e}"))?;
        let cov_psi = op
            .materialize_cov_first_axis(axis)
            .map_err(|e| format!("SCOP psi materialize_cov_first failed: {e}"))?;
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(format!(
                "SCOP psi covariate derivative shape {}x{} != expected {}x{}",
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ));
        }

        let weights = self.weights.as_ref();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut objective_psi = 0.0;
        let mut score_psi = Array1::<f64>::zeros(p_total);
        let mut hessian_psi = Array2::<f64>::zeros((p_total, p_total));

        for i in 0..n {
            let cov_row = cov.row(i);
            let psi_row = cov_psi.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let q = row_quantities.endpoint_q[i];

            let mut gamma = vec![0.0; p_resp];
            let mut gamma_psi = vec![0.0; p_resp];
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
            }

            let mut h_psi = rv[0] * gamma_psi[0];
            let mut hp_psi = rd[0] * gamma_psi[0];
            for k in 1..p_resp {
                h_psi += 2.0 * rv[k] * gamma[k] * gamma_psi[k];
                hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
            }

            let endpoint_basis = [
                self.response_upper_basis
                    .as_slice()
                    .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
                self.response_lower_basis
                    .as_slice()
                    .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
            ];
            let mut endpoint_psi = [0.0; 2];
            let mut endpoint_factor = vec![[0.0; 2]; p_resp];
            let mut endpoint_psi_cov_factor = vec![[0.0; 2]; p_resp];
            let mut endpoint_psi_psi_factor = vec![[0.0; 2]; p_resp];
            for e in 0..2 {
                let basis = endpoint_basis[e];
                endpoint_psi[e] = basis[0] * gamma_psi[0];
                endpoint_factor[0][e] = basis[0];
                endpoint_psi_psi_factor[0][e] = basis[0];
                for k in 1..p_resp {
                    endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
                    endpoint_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_psi_cov_factor[k][e] = 2.0 * basis[k] * gamma_psi[k];
                    endpoint_psi_psi_factor[k][e] = 2.0 * basis[k] * gamma[k];
                }
            }

            objective_psi +=
                wi * (hi * h_psi - hp_psi * inv_hp + endpoint_chain_first(&q, endpoint_psi));

            let mut h_factor = vec![0.0; p_resp];
            let mut hp_factor = vec![0.0; p_resp];
            let mut hpsi_cov_factor = vec![0.0; p_resp];
            let mut hppsi_cov_factor = vec![0.0; p_resp];
            let mut hpsi_psi_factor = vec![0.0; p_resp];
            let mut hppsi_psi_factor = vec![0.0; p_resp];
            h_factor[0] = rv[0];
            hp_factor[0] = rd[0];
            hpsi_psi_factor[0] = rv[0];
            hppsi_psi_factor[0] = rd[0];
            for k in 1..p_resp {
                h_factor[k] = 2.0 * rv[k] * gamma[k];
                hp_factor[k] = 2.0 * rd[k] * gamma[k];
                hpsi_cov_factor[k] = 2.0 * rv[k] * gamma_psi[k];
                hppsi_cov_factor[k] = 2.0 * rd[k] * gamma_psi[k];
                hpsi_psi_factor[k] = 2.0 * rv[k] * gamma[k];
                hppsi_psi_factor[k] = 2.0 * rd[k] * gamma[k];
            }

            for k in 0..p_resp {
                for c in 0..p_cov {
                    let idx = k * p_cov + c;
                    let h_a = h_factor[k] * cov_row[c];
                    let hp_a = hp_factor[k] * cov_row[c];
                    let hpsi_a = hpsi_cov_factor[k] * cov_row[c] + hpsi_psi_factor[k] * psi_row[c];
                    let hppsi_a =
                        hppsi_cov_factor[k] * cov_row[c] + hppsi_psi_factor[k] * psi_row[c];
                    let endpoint_a = [
                        endpoint_factor[k][0] * cov_row[c],
                        endpoint_factor[k][1] * cov_row[c],
                    ];
                    let endpoint_psi_a = [
                        endpoint_psi_cov_factor[k][0] * cov_row[c]
                            + endpoint_psi_psi_factor[k][0] * psi_row[c],
                        endpoint_psi_cov_factor[k][1] * cov_row[c]
                            + endpoint_psi_psi_factor[k][1] * psi_row[c],
                    ];
                    score_psi[idx] += wi
                        * (h_a * h_psi + hi * hpsi_a - hppsi_a * inv_hp
                            + hp_psi * hp_a * inv_hp_sq
                            + endpoint_chain_second(&q, endpoint_psi, endpoint_a, endpoint_psi_a));
                }
            }

            for k in 0..p_resp {
                for l in 0..p_resp {
                    let same_shape = k == l && k > 0;
                    for c in 0..p_cov {
                        let row_idx = k * p_cov + c;
                        let h_a = h_factor[k] * cov_row[c];
                        let hp_a = hp_factor[k] * cov_row[c];
                        let hpsi_a =
                            hpsi_cov_factor[k] * cov_row[c] + hpsi_psi_factor[k] * psi_row[c];
                        let hppsi_a =
                            hppsi_cov_factor[k] * cov_row[c] + hppsi_psi_factor[k] * psi_row[c];
                        for d in 0..p_cov {
                            let col_idx = l * p_cov + d;
                            let h_b = h_factor[l] * cov_row[d];
                            let hp_b = hp_factor[l] * cov_row[d];
                            let hpsi_b =
                                hpsi_cov_factor[l] * cov_row[d] + hpsi_psi_factor[l] * psi_row[d];
                            let hppsi_b =
                                hppsi_cov_factor[l] * cov_row[d] + hppsi_psi_factor[l] * psi_row[d];
                            let endpoint_a = [
                                endpoint_factor[k][0] * cov_row[c],
                                endpoint_factor[k][1] * cov_row[c],
                            ];
                            let endpoint_b = [
                                endpoint_factor[l][0] * cov_row[d],
                                endpoint_factor[l][1] * cov_row[d],
                            ];
                            let endpoint_psi_a = [
                                endpoint_psi_cov_factor[k][0] * cov_row[c]
                                    + endpoint_psi_psi_factor[k][0] * psi_row[c],
                                endpoint_psi_cov_factor[k][1] * cov_row[c]
                                    + endpoint_psi_psi_factor[k][1] * psi_row[c],
                            ];
                            let endpoint_psi_b = [
                                endpoint_psi_cov_factor[l][0] * cov_row[d]
                                    + endpoint_psi_psi_factor[l][0] * psi_row[d],
                                endpoint_psi_cov_factor[l][1] * cov_row[d]
                                    + endpoint_psi_psi_factor[l][1] * psi_row[d],
                            ];
                            let (h_ab, hp_ab, hpsi_ab, hppsi_ab) = if same_shape {
                                (
                                    2.0 * rv[k] * cov_row[c] * cov_row[d],
                                    2.0 * rd[k] * cov_row[c] * cov_row[d],
                                    2.0 * rv[k]
                                        * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                    2.0 * rd[k]
                                        * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                )
                            } else {
                                (0.0, 0.0, 0.0, 0.0)
                            };
                            let (endpoint_ab, endpoint_psi_ab) = if same_shape {
                                (
                                    [
                                        2.0 * endpoint_basis[0][k] * cov_row[c] * cov_row[d],
                                        2.0 * endpoint_basis[1][k] * cov_row[c] * cov_row[d],
                                    ],
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                        2.0 * endpoint_basis[1][k]
                                            * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                    ],
                                )
                            } else {
                                ([0.0; 2], [0.0; 2])
                            };
                            let value = hpsi_a * h_b
                                + h_a * hpsi_b
                                + h_psi * h_ab
                                + hi * hpsi_ab
                                + (hppsi_a * hp_b + hp_a * hppsi_b) * inv_hp_sq
                                - 2.0 * hp_a * hp_b * hp_psi * inv_hp_cu
                                - hppsi_ab * inv_hp
                                + hp_ab * hp_psi * inv_hp_sq
                                + endpoint_chain_third(
                                    &q,
                                    endpoint_psi,
                                    endpoint_a,
                                    endpoint_b,
                                    endpoint_psi_a,
                                    endpoint_psi_b,
                                    endpoint_ab,
                                    endpoint_psi_ab,
                                );
                            hessian_psi[[row_idx, col_idx]] += wi * value;
                        }
                    }
                }
            }
        }

        Ok(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: 0.5 * (&hessian_psi + &hessian_psi.t()),
            hessian_psi_operator: None,
        })
    }

    fn scop_psi_psi_value_score_hvp_from_cov(
        &self,
        beta: &Array1<f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        direction: Option<&Array1<f64>>,
    ) -> Result<(f64, Array1<f64>, Option<Array1<f64>>), String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if row_start > total_n || row_start + n > total_n {
            return Err(format!(
                "SCOP psi-psi row window [{row_start}, {}) exceeds n={total_n}",
                row_start + n
            ));
        }
        if beta.len() != p_total {
            return Err(format!(
                "SCOP psi-psi beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                beta.len()
            ));
        }
        if endpoint_q.len() != n {
            return Err(format!(
                "SCOP psi-psi endpoint normalizer cache length {} != n={n}",
                endpoint_q.len()
            ));
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(format!(
                    "SCOP psi-psi {name} shape {}x{} != expected {}x{}",
                    mat.nrows(),
                    mat.ncols(),
                    n,
                    p_cov
                ));
            }
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi beta reshape failed: {e}"))?;
        let direction_mat = match direction {
            Some(v) => {
                if v.len() != p_total {
                    return Err(format!(
                        "SCOP psi-psi HVP direction length {} != p_total {p_total}",
                        v.len()
                    ));
                }
                Some(
                    v.view()
                        .into_shape_with_order((p_resp, p_cov))
                        .map_err(|e| format!("SCOP psi-psi direction reshape failed: {e}"))?,
                )
            }
            None => None,
        };
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        if direction_mat.is_none() {
            let weights = self.weights.as_ref();

            struct PsiPairScoreAccum {
                objective: f64,
                score: Array1<f64>,
                gamma: Vec<f64>,
                gamma_i: Vec<f64>,
                gamma_j: Vec<f64>,
                gamma_ij: Vec<f64>,
            }

            impl PsiPairScoreAccum {
                fn new(p_total: usize, p_resp: usize) -> Self {
                    Self {
                        objective: 0.0,
                        score: Array1::<f64>::zeros(p_total),
                        gamma: vec![0.0; p_resp],
                        gamma_i: vec![0.0; p_resp],
                        gamma_j: vec![0.0; p_resp],
                        gamma_ij: vec![0.0; p_resp],
                    }
                }

                fn merge(mut self, rhs: Self) -> Self {
                    self.objective += rhs.objective;
                    self.score.scaled_add(1.0, &rhs.score);
                    self
                }
            }

            let accum = (0..n)
                .into_par_iter()
                .fold(
                    || PsiPairScoreAccum::new(p_total, p_resp),
                    |mut acc, row_idx| {
                        let cov_row = cov.row(row_idx);
                        let cov_i_row = cov_i.row(row_idx);
                        let cov_j_row = cov_j.row(row_idx);
                        let cov_ij_row = cov_ij.row(row_idx);
                        let global_row = row_start + row_idx;
                        let rv = self.response_val_basis.row(global_row);
                        let rd = self.response_deriv_basis.row(global_row);

                        for k in 0..p_resp {
                            let beta_k = beta_mat.row(k);
                            acc.gamma[k] = beta_k.dot(&cov_row);
                            acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                            acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                            acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                        }

                        let mut h = rv[0] * acc.gamma[0];
                        let mut hp = rd[0] * acc.gamma[0];
                        let mut h_i = rv[0] * acc.gamma_i[0];
                        let mut h_j = rv[0] * acc.gamma_j[0];
                        let mut h_ij = rv[0] * acc.gamma_ij[0];
                        let mut hp_i = rd[0] * acc.gamma_i[0];
                        let mut hp_j = rd[0] * acc.gamma_j[0];
                        let mut hp_ij = rd[0] * acc.gamma_ij[0];
                        for k in 1..p_resp {
                            let g = acc.gamma[k];
                            let gi = acc.gamma_i[k];
                            let gj = acc.gamma_j[k];
                            let gij = acc.gamma_ij[k];
                            h += rv[k] * g * g;
                            hp += rd[k] * g * g;
                            h_i += 2.0 * rv[k] * g * gi;
                            h_j += 2.0 * rv[k] * g * gj;
                            h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
                            hp_i += 2.0 * rd[k] * g * gi;
                            hp_j += 2.0 * rd[k] * g * gj;
                            hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);
                        }

                        let inv_hp = 1.0 / hp;
                        let inv_hp_sq = inv_hp * inv_hp;
                        let inv_hp_cu = inv_hp_sq * inv_hp;
                        let q = endpoint_q[row_idx];
                        let mut endpoint_i = [0.0; 2];
                        let mut endpoint_j = [0.0; 2];
                        let mut endpoint_ij = [0.0; 2];
                        for e in 0..2 {
                            let basis = endpoint_basis[e];
                            endpoint_i[e] = basis[0] * acc.gamma_i[0];
                            endpoint_j[e] = basis[0] * acc.gamma_j[0];
                            endpoint_ij[e] = basis[0] * acc.gamma_ij[0];
                            for k in 1..p_resp {
                                endpoint_i[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_i[k];
                                endpoint_j[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_j[k];
                                endpoint_ij[e] += 2.0
                                    * basis[k]
                                    * (acc.gamma_j[k] * acc.gamma_i[k]
                                        + acc.gamma[k] * acc.gamma_ij[k]);
                            }
                        }
                        let value = h_i * h_j + h * h_ij - hp_ij * inv_hp
                            + hp_i * hp_j * inv_hp_sq
                            + endpoint_chain_second(&q, endpoint_i, endpoint_j, endpoint_ij);
                        let wi = weights[global_row];
                        acc.objective += wi * value;

                        for k in 0..p_resp {
                            let offset = k * p_cov;
                            let (rvk, rdk) = (rv[k], rd[k]);
                            let (g, gi, gj, gij) = (
                                acc.gamma[k],
                                acc.gamma_i[k],
                                acc.gamma_j[k],
                                acc.gamma_ij[k],
                            );
                            for cidx in 0..p_cov {
                                let c = cov_row[cidx];
                                let ci = cov_i_row[cidx];
                                let cj = cov_j_row[cidx];
                                let cij = cov_ij_row[cidx];
                                let (dh, dhp, dh_i, dh_j, dh_ij, dhp_i, dhp_j, dhp_ij) = if k == 0 {
                                    (
                                        rvk * c,
                                        rdk * c,
                                        rvk * ci,
                                        rvk * cj,
                                        rvk * cij,
                                        rdk * ci,
                                        rdk * cj,
                                        rdk * cij,
                                    )
                                } else {
                                    (
                                        2.0 * rvk * g * c,
                                        2.0 * rdk * g * c,
                                        2.0 * rvk * (gi * c + g * ci),
                                        2.0 * rvk * (gj * c + g * cj),
                                        2.0 * rvk * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * rdk * (gi * c + g * ci),
                                        2.0 * rdk * (gj * c + g * cj),
                                        2.0 * rdk * (gj * ci + gi * cj + gij * c + g * cij),
                                    )
                                };
                                let endpoint_a = if k == 0 {
                                    [endpoint_basis[0][k] * c, endpoint_basis[1][k] * c]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * g * c,
                                        2.0 * endpoint_basis[1][k] * g * c,
                                    ]
                                };
                                let endpoint_i_a = if k == 0 {
                                    [endpoint_basis[0][k] * ci, endpoint_basis[1][k] * ci]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (gi * c + g * ci),
                                        2.0 * endpoint_basis[1][k] * (gi * c + g * ci),
                                    ]
                                };
                                let endpoint_j_a = if k == 0 {
                                    [endpoint_basis[0][k] * cj, endpoint_basis[1][k] * cj]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (gj * c + g * cj),
                                        2.0 * endpoint_basis[1][k] * (gj * c + g * cj),
                                    ]
                                };
                                let endpoint_ij_a = if k == 0 {
                                    [endpoint_basis[0][k] * cij, endpoint_basis[1][k] * cij]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * endpoint_basis[1][k]
                                            * (gj * ci + gi * cj + gij * c + g * cij),
                                    ]
                                };
                                let grad = dh_i * h_j + h_i * dh_j + dh * h_ij + h * dh_ij
                                    - dhp_ij * inv_hp
                                    + hp_ij * dhp * inv_hp_sq
                                    + (dhp_i * hp_j + hp_i * dhp_j) * inv_hp_sq
                                    - 2.0 * hp_i * hp_j * dhp * inv_hp_cu
                                    + endpoint_chain_third(
                                        &q,
                                        endpoint_i,
                                        endpoint_j,
                                        endpoint_a,
                                        endpoint_ij,
                                        endpoint_i_a,
                                        endpoint_j_a,
                                        endpoint_ij_a,
                                    );
                                acc.score[offset + cidx] += wi * grad;
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || PsiPairScoreAccum::new(p_total, p_resp),
                    |left, right| left.merge(right),
                );

            return Ok((accum.objective, accum.score, None));
        }

        let weights = self.weights.as_ref();
        let direction_mat = direction_mat.expect("directional CTN psi-psi path requires direction");

        struct PsiPairDirectionalAccum {
            hvp: Array1<f64>,
            gamma: Vec<f64>,
            gamma_i: Vec<f64>,
            gamma_j: Vec<f64>,
            gamma_ij: Vec<f64>,
            gamma_dot: Vec<f64>,
            gamma_i_dot: Vec<f64>,
            gamma_j_dot: Vec<f64>,
            gamma_ij_dot: Vec<f64>,
        }

        impl PsiPairDirectionalAccum {
            fn new(p_total: usize, p_resp: usize) -> Self {
                Self {
                    hvp: Array1::<f64>::zeros(p_total),
                    gamma: vec![0.0; p_resp],
                    gamma_i: vec![0.0; p_resp],
                    gamma_j: vec![0.0; p_resp],
                    gamma_ij: vec![0.0; p_resp],
                    gamma_dot: vec![0.0; p_resp],
                    gamma_i_dot: vec![0.0; p_resp],
                    gamma_j_dot: vec![0.0; p_resp],
                    gamma_ij_dot: vec![0.0; p_resp],
                }
            }

            fn merge(mut self, rhs: Self) -> Self {
                self.hvp.scaled_add(1.0, &rhs.hvp);
                self
            }
        }

        let accum = (0..n)
            .into_par_iter()
            .fold(
                || PsiPairDirectionalAccum::new(p_total, p_resp),
                |mut acc, row_idx| {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        let dir_k = direction_mat.row(k);
                        acc.gamma[k] = beta_k.dot(&cov_row);
                        acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                        acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                        acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                        acc.gamma_dot[k] = dir_k.dot(&cov_row);
                        acc.gamma_i_dot[k] = dir_k.dot(&cov_i_row);
                        acc.gamma_j_dot[k] = dir_k.dot(&cov_j_row);
                        acc.gamma_ij_dot[k] = dir_k.dot(&cov_ij_row);
                    }

                    let mut h = rv[0] * acc.gamma[0];
                    let mut hp = rd[0] * acc.gamma[0];
                    let mut h_i = rv[0] * acc.gamma_i[0];
                    let mut h_j = rv[0] * acc.gamma_j[0];
                    let mut h_ij = rv[0] * acc.gamma_ij[0];
                    let mut hp_i = rd[0] * acc.gamma_i[0];
                    let mut hp_j = rd[0] * acc.gamma_j[0];
                    let mut hp_ij = rd[0] * acc.gamma_ij[0];
                    let mut h_dot = rv[0] * acc.gamma_dot[0];
                    let mut hp_dot = rd[0] * acc.gamma_dot[0];
                    let mut h_i_dot = rv[0] * acc.gamma_i_dot[0];
                    let mut h_j_dot = rv[0] * acc.gamma_j_dot[0];
                    let mut h_ij_dot = rv[0] * acc.gamma_ij_dot[0];
                    let mut hp_i_dot = rd[0] * acc.gamma_i_dot[0];
                    let mut hp_j_dot = rd[0] * acc.gamma_j_dot[0];
                    let mut hp_ij_dot = rd[0] * acc.gamma_ij_dot[0];

                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let gi = acc.gamma_i[k];
                        let gj = acc.gamma_j[k];
                        let gij = acc.gamma_ij[k];
                        let u = acc.gamma_dot[k];
                        let ui = acc.gamma_i_dot[k];
                        let uj = acc.gamma_j_dot[k];
                        let uij = acc.gamma_ij_dot[k];
                        h += rv[k] * g * g;
                        hp += rd[k] * g * g;
                        h_i += 2.0 * rv[k] * g * gi;
                        h_j += 2.0 * rv[k] * g * gj;
                        h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
                        hp_i += 2.0 * rd[k] * g * gi;
                        hp_j += 2.0 * rd[k] * g * gj;
                        hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);
                        h_dot += 2.0 * rv[k] * g * u;
                        hp_dot += 2.0 * rd[k] * g * u;
                        h_i_dot += 2.0 * rv[k] * (u * gi + g * ui);
                        h_j_dot += 2.0 * rv[k] * (u * gj + g * uj);
                        h_ij_dot += 2.0 * rv[k] * (uj * gi + gj * ui + u * gij + g * uij);
                        hp_i_dot += 2.0 * rd[k] * (u * gi + g * ui);
                        hp_j_dot += 2.0 * rd[k] * (u * gj + g * uj);
                        hp_ij_dot += 2.0 * rd[k] * (uj * gi + gj * ui + u * gij + g * uij);
                    }

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let wi = weights[global_row];
                    let q = endpoint_q[row_idx];
                    let mut endpoint_i = [0.0; 2];
                    let mut endpoint_j = [0.0; 2];
                    let mut endpoint_ij = [0.0; 2];
                    let mut endpoint_d = [0.0; 2];
                    let mut endpoint_i_d = [0.0; 2];
                    let mut endpoint_j_d = [0.0; 2];
                    let mut endpoint_ij_d = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_i[e] = basis[0] * acc.gamma_i[0];
                        endpoint_j[e] = basis[0] * acc.gamma_j[0];
                        endpoint_ij[e] = basis[0] * acc.gamma_ij[0];
                        endpoint_d[e] = basis[0] * acc.gamma_dot[0];
                        endpoint_i_d[e] = basis[0] * acc.gamma_i_dot[0];
                        endpoint_j_d[e] = basis[0] * acc.gamma_j_dot[0];
                        endpoint_ij_d[e] = basis[0] * acc.gamma_ij_dot[0];
                        for k in 1..p_resp {
                            endpoint_i[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_i[k];
                            endpoint_j[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_j[k];
                            endpoint_ij[e] += 2.0
                                * basis[k]
                                * (acc.gamma_j[k] * acc.gamma_i[k]
                                    + acc.gamma[k] * acc.gamma_ij[k]);
                            endpoint_d[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_dot[k];
                            endpoint_i_d[e] += 2.0
                                * basis[k]
                                * (acc.gamma_dot[k] * acc.gamma_i[k]
                                    + acc.gamma[k] * acc.gamma_i_dot[k]);
                            endpoint_j_d[e] += 2.0
                                * basis[k]
                                * (acc.gamma_dot[k] * acc.gamma_j[k]
                                    + acc.gamma[k] * acc.gamma_j_dot[k]);
                            endpoint_ij_d[e] += 2.0
                                * basis[k]
                                * (acc.gamma_j_dot[k] * acc.gamma_i[k]
                                    + acc.gamma_j[k] * acc.gamma_i_dot[k]
                                    + acc.gamma_dot[k] * acc.gamma_ij[k]
                                    + acc.gamma[k] * acc.gamma_ij_dot[k]);
                        }
                    }

                    for k in 0..p_resp {
                        let offset = k * p_cov;
                        let (rvk, rdk) = (rv[k], rd[k]);
                        let (g, gi, gj, gij) = (
                            acc.gamma[k],
                            acc.gamma_i[k],
                            acc.gamma_j[k],
                            acc.gamma_ij[k],
                        );
                        let (u, ui, uj, uij) = (
                            acc.gamma_dot[k],
                            acc.gamma_i_dot[k],
                            acc.gamma_j_dot[k],
                            acc.gamma_ij_dot[k],
                        );
                        for cidx in 0..p_cov {
                            let c = cov_row[cidx];
                            let ci = cov_i_row[cidx];
                            let cj = cov_j_row[cidx];
                            let cij = cov_ij_row[cidx];
                            let (
                                dh,
                                dhp,
                                dh_i,
                                dh_j,
                                dh_ij,
                                dhp_i,
                                dhp_j,
                                dhp_ij,
                                ddh,
                                ddhp,
                                ddh_i,
                                ddh_j,
                                ddh_ij,
                                ddhp_i,
                                ddhp_j,
                                ddhp_ij,
                            ) = if k == 0 {
                                (
                                    rvk * c,
                                    rdk * c,
                                    rvk * ci,
                                    rvk * cj,
                                    rvk * cij,
                                    rdk * ci,
                                    rdk * cj,
                                    rdk * cij,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                )
                            } else {
                                (
                                    2.0 * rvk * g * c,
                                    2.0 * rdk * g * c,
                                    2.0 * rvk * (gi * c + g * ci),
                                    2.0 * rvk * (gj * c + g * cj),
                                    2.0 * rvk * (gj * ci + gi * cj + gij * c + g * cij),
                                    2.0 * rdk * (gi * c + g * ci),
                                    2.0 * rdk * (gj * c + g * cj),
                                    2.0 * rdk * (gj * ci + gi * cj + gij * c + g * cij),
                                    2.0 * rvk * u * c,
                                    2.0 * rdk * u * c,
                                    2.0 * rvk * (ui * c + u * ci),
                                    2.0 * rvk * (uj * c + u * cj),
                                    2.0 * rvk * (uj * ci + ui * cj + uij * c + u * cij),
                                    2.0 * rdk * (ui * c + u * ci),
                                    2.0 * rdk * (uj * c + u * cj),
                                    2.0 * rdk * (uj * ci + ui * cj + uij * c + u * cij),
                                )
                            };

                            let endpoint_a = if k == 0 {
                                [endpoint_basis[0][k] * c, endpoint_basis[1][k] * c]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * g * c,
                                    2.0 * endpoint_basis[1][k] * g * c,
                                ]
                            };
                            let endpoint_i_a = if k == 0 {
                                [endpoint_basis[0][k] * ci, endpoint_basis[1][k] * ci]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * (gi * c + g * ci),
                                    2.0 * endpoint_basis[1][k] * (gi * c + g * ci),
                                ]
                            };
                            let endpoint_j_a = if k == 0 {
                                [endpoint_basis[0][k] * cj, endpoint_basis[1][k] * cj]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * (gj * c + g * cj),
                                    2.0 * endpoint_basis[1][k] * (gj * c + g * cj),
                                ]
                            };
                            let endpoint_ij_a = if k == 0 {
                                [endpoint_basis[0][k] * cij, endpoint_basis[1][k] * cij]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k]
                                        * (gj * ci + gi * cj + gij * c + g * cij),
                                    2.0 * endpoint_basis[1][k]
                                        * (gj * ci + gi * cj + gij * c + g * cij),
                                ]
                            };
                            let endpoint_a_d = if k == 0 {
                                [0.0; 2]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * u * c,
                                    2.0 * endpoint_basis[1][k] * u * c,
                                ]
                            };
                            let endpoint_i_a_d = if k == 0 {
                                [0.0; 2]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * (ui * c + u * ci),
                                    2.0 * endpoint_basis[1][k] * (ui * c + u * ci),
                                ]
                            };
                            let endpoint_j_a_d = if k == 0 {
                                [0.0; 2]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * (uj * c + u * cj),
                                    2.0 * endpoint_basis[1][k] * (uj * c + u * cj),
                                ]
                            };
                            let endpoint_ij_a_d = if k == 0 {
                                [0.0; 2]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k]
                                        * (uj * ci + ui * cj + uij * c + u * cij),
                                    2.0 * endpoint_basis[1][k]
                                        * (uj * ci + ui * cj + uij * c + u * cij),
                                ]
                            };
                            let n1 = dhp_i * hp_j + hp_i * dhp_j;
                            let n1_dot =
                                ddhp_i * hp_j + dhp_i * hp_j_dot + hp_i_dot * dhp_j + hp_i * ddhp_j;
                            let n2_dot =
                                hp_i_dot * hp_j * dhp + hp_i * hp_j_dot * dhp + hp_i * hp_j * ddhp;
                            let hv = ddh_i * h_j
                                + dh_i * h_j_dot
                                + h_i_dot * dh_j
                                + h_i * ddh_j
                                + ddh * h_ij
                                + dh * h_ij_dot
                                + h_dot * dh_ij
                                + h * ddh_ij
                                - ddhp_ij * inv_hp
                                + dhp_ij * hp_dot * inv_hp_sq
                                + hp_ij_dot * dhp * inv_hp_sq
                                + hp_ij * ddhp * inv_hp_sq
                                - 2.0 * hp_ij * dhp * hp_dot * inv_hp_cu
                                + n1_dot * inv_hp_sq
                                - 2.0 * n1 * hp_dot * inv_hp_cu
                                - 2.0 * n2_dot * inv_hp_cu
                                + 6.0 * hp_i * hp_j * dhp * hp_dot * inv_hp_qu
                                + endpoint_chain_fourth(
                                    &q,
                                    endpoint_i,
                                    endpoint_j,
                                    endpoint_a,
                                    endpoint_d,
                                    endpoint_ij,
                                    endpoint_i_a,
                                    endpoint_i_d,
                                    endpoint_j_a,
                                    endpoint_j_d,
                                    endpoint_a_d,
                                    endpoint_ij_a,
                                    endpoint_ij_d,
                                    endpoint_i_a_d,
                                    endpoint_j_a_d,
                                    endpoint_ij_a_d,
                                );
                            acc.hvp[offset + cidx] += wi * hv;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || PsiPairDirectionalAccum::new(p_total, p_resp),
                |left, right| left.merge(right),
            );

        Ok((0.0, Array1::<f64>::zeros(p_total), Some(accum.hvp)))
    }

    fn scop_psi_psi_bilinear_from_cov(
        &self,
        beta: &Array1<f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<f64, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if row_start > total_n || row_start + n > total_n {
            return Err(format!(
                "SCOP psi-psi bilinear row window [{row_start}, {}) exceeds n={total_n}",
                row_start + n
            ));
        }
        if beta.len() != p_total || left.len() != p_total || right.len() != p_total {
            return Err(format!(
                "SCOP psi-psi bilinear length mismatch: beta={}, left={}, right={}, expected={p_total}",
                beta.len(),
                left.len(),
                right.len()
            ));
        }
        if endpoint_q.len() != n {
            return Err(format!(
                "SCOP psi-psi bilinear endpoint normalizer cache length {} != n={n}",
                endpoint_q.len()
            ));
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(format!(
                    "SCOP psi-psi bilinear {name} shape {}x{} != expected {}x{}",
                    mat.nrows(),
                    mat.ncols(),
                    n,
                    p_cov
                ));
            }
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi bilinear beta reshape failed: {e}"))?;
        let left_mat = left
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi bilinear left reshape failed: {e}"))?;
        let right_mat = right
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi bilinear right reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiPairBilinearAccum {
            value: f64,
            gamma: Vec<f64>,
            gamma_i: Vec<f64>,
            gamma_j: Vec<f64>,
            gamma_ij: Vec<f64>,
            left: Vec<f64>,
            left_i: Vec<f64>,
            left_j: Vec<f64>,
            left_ij: Vec<f64>,
            right: Vec<f64>,
            right_i: Vec<f64>,
            right_j: Vec<f64>,
            right_ij: Vec<f64>,
        }

        impl PsiPairBilinearAccum {
            fn new(p_resp: usize) -> Self {
                Self {
                    value: 0.0,
                    gamma: vec![0.0; p_resp],
                    gamma_i: vec![0.0; p_resp],
                    gamma_j: vec![0.0; p_resp],
                    gamma_ij: vec![0.0; p_resp],
                    left: vec![0.0; p_resp],
                    left_i: vec![0.0; p_resp],
                    left_j: vec![0.0; p_resp],
                    left_ij: vec![0.0; p_resp],
                    right: vec![0.0; p_resp],
                    right_i: vec![0.0; p_resp],
                    right_j: vec![0.0; p_resp],
                    right_ij: vec![0.0; p_resp],
                }
            }
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let weights = self.weights.as_ref();
        let total = (0..n)
            .into_par_iter()
            .fold(
                || PsiPairBilinearAccum::new(p_resp),
                |mut acc, row_idx| {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        let left_k = left_mat.row(k);
                        let right_k = right_mat.row(k);
                        acc.gamma[k] = beta_k.dot(&cov_row);
                        acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                        acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                        acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                        acc.left[k] = left_k.dot(&cov_row);
                        acc.left_i[k] = left_k.dot(&cov_i_row);
                        acc.left_j[k] = left_k.dot(&cov_j_row);
                        acc.left_ij[k] = left_k.dot(&cov_ij_row);
                        acc.right[k] = right_k.dot(&cov_row);
                        acc.right_i[k] = right_k.dot(&cov_i_row);
                        acc.right_j[k] = right_k.dot(&cov_j_row);
                        acc.right_ij[k] = right_k.dot(&cov_ij_row);
                    }

                    let mut h = rv[0] * acc.gamma[0];
                    let mut hp = rd[0] * acc.gamma[0];
                    let mut h_i = rv[0] * acc.gamma_i[0];
                    let mut h_j = rv[0] * acc.gamma_j[0];
                    let mut h_ij = rv[0] * acc.gamma_ij[0];
                    let mut hp_i = rd[0] * acc.gamma_i[0];
                    let mut hp_j = rd[0] * acc.gamma_j[0];
                    let mut hp_ij = rd[0] * acc.gamma_ij[0];

                    let mut h_l = rv[0] * acc.left[0];
                    let mut hp_l = rd[0] * acc.left[0];
                    let mut h_i_l = rv[0] * acc.left_i[0];
                    let mut h_j_l = rv[0] * acc.left_j[0];
                    let mut h_ij_l = rv[0] * acc.left_ij[0];
                    let mut hp_i_l = rd[0] * acc.left_i[0];
                    let mut hp_j_l = rd[0] * acc.left_j[0];
                    let mut hp_ij_l = rd[0] * acc.left_ij[0];

                    let mut h_r = rv[0] * acc.right[0];
                    let mut hp_r = rd[0] * acc.right[0];
                    let mut h_i_r = rv[0] * acc.right_i[0];
                    let mut h_j_r = rv[0] * acc.right_j[0];
                    let mut h_ij_r = rv[0] * acc.right_ij[0];
                    let mut hp_i_r = rd[0] * acc.right_i[0];
                    let mut hp_j_r = rd[0] * acc.right_j[0];
                    let mut hp_ij_r = rd[0] * acc.right_ij[0];

                    let mut h_lr = 0.0;
                    let mut hp_lr = 0.0;
                    let mut h_i_lr = 0.0;
                    let mut h_j_lr = 0.0;
                    let mut h_ij_lr = 0.0;
                    let mut hp_i_lr = 0.0;
                    let mut hp_j_lr = 0.0;
                    let mut hp_ij_lr = 0.0;

                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let gi = acc.gamma_i[k];
                        let gj = acc.gamma_j[k];
                        let gij = acc.gamma_ij[k];
                        let l = acc.left[k];
                        let li = acc.left_i[k];
                        let lj = acc.left_j[k];
                        let lij = acc.left_ij[k];
                        let r = acc.right[k];
                        let ri = acc.right_i[k];
                        let rj = acc.right_j[k];
                        let rij = acc.right_ij[k];

                        h += rv[k] * g * g;
                        hp += rd[k] * g * g;
                        h_i += 2.0 * rv[k] * g * gi;
                        h_j += 2.0 * rv[k] * g * gj;
                        h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
                        hp_i += 2.0 * rd[k] * g * gi;
                        hp_j += 2.0 * rd[k] * g * gj;
                        hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);

                        h_l += 2.0 * rv[k] * g * l;
                        hp_l += 2.0 * rd[k] * g * l;
                        h_i_l += 2.0 * rv[k] * (l * gi + g * li);
                        h_j_l += 2.0 * rv[k] * (l * gj + g * lj);
                        h_ij_l += 2.0 * rv[k] * (lj * gi + gj * li + l * gij + g * lij);
                        hp_i_l += 2.0 * rd[k] * (l * gi + g * li);
                        hp_j_l += 2.0 * rd[k] * (l * gj + g * lj);
                        hp_ij_l += 2.0 * rd[k] * (lj * gi + gj * li + l * gij + g * lij);

                        h_r += 2.0 * rv[k] * g * r;
                        hp_r += 2.0 * rd[k] * g * r;
                        h_i_r += 2.0 * rv[k] * (r * gi + g * ri);
                        h_j_r += 2.0 * rv[k] * (r * gj + g * rj);
                        h_ij_r += 2.0 * rv[k] * (rj * gi + gj * ri + r * gij + g * rij);
                        hp_i_r += 2.0 * rd[k] * (r * gi + g * ri);
                        hp_j_r += 2.0 * rd[k] * (r * gj + g * rj);
                        hp_ij_r += 2.0 * rd[k] * (rj * gi + gj * ri + r * gij + g * rij);

                        h_lr += 2.0 * rv[k] * l * r;
                        hp_lr += 2.0 * rd[k] * l * r;
                        h_i_lr += 2.0 * rv[k] * (l * ri + r * li);
                        h_j_lr += 2.0 * rv[k] * (l * rj + r * lj);
                        h_ij_lr += 2.0 * rv[k] * (lj * ri + rj * li + l * rij + r * lij);
                        hp_i_lr += 2.0 * rd[k] * (l * ri + r * li);
                        hp_j_lr += 2.0 * rd[k] * (l * rj + r * lj);
                        hp_ij_lr += 2.0 * rd[k] * (lj * ri + rj * li + l * rij + r * lij);
                    }

                    let q = endpoint_q[row_idx];
                    let mut endpoint_i = [0.0; 2];
                    let mut endpoint_j = [0.0; 2];
                    let mut endpoint_ij = [0.0; 2];
                    let mut endpoint_l = [0.0; 2];
                    let mut endpoint_r = [0.0; 2];
                    let mut endpoint_i_l = [0.0; 2];
                    let mut endpoint_j_l = [0.0; 2];
                    let mut endpoint_ij_l = [0.0; 2];
                    let mut endpoint_i_r = [0.0; 2];
                    let mut endpoint_j_r = [0.0; 2];
                    let mut endpoint_ij_r = [0.0; 2];
                    let mut endpoint_l_r = [0.0; 2];
                    let mut endpoint_i_l_r = [0.0; 2];
                    let mut endpoint_j_l_r = [0.0; 2];
                    let mut endpoint_ij_l_r = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_i[e] = basis[0] * acc.gamma_i[0];
                        endpoint_j[e] = basis[0] * acc.gamma_j[0];
                        endpoint_ij[e] = basis[0] * acc.gamma_ij[0];
                        endpoint_l[e] = basis[0] * acc.left[0];
                        endpoint_r[e] = basis[0] * acc.right[0];
                        endpoint_i_l[e] = basis[0] * acc.left_i[0];
                        endpoint_j_l[e] = basis[0] * acc.left_j[0];
                        endpoint_ij_l[e] = basis[0] * acc.left_ij[0];
                        endpoint_i_r[e] = basis[0] * acc.right_i[0];
                        endpoint_j_r[e] = basis[0] * acc.right_j[0];
                        endpoint_ij_r[e] = basis[0] * acc.right_ij[0];
                        for k in 1..p_resp {
                            let basis_k = basis[k];
                            let g = acc.gamma[k];
                            let gi = acc.gamma_i[k];
                            let gj = acc.gamma_j[k];
                            let gij = acc.gamma_ij[k];
                            let l = acc.left[k];
                            let li = acc.left_i[k];
                            let lj = acc.left_j[k];
                            let lij = acc.left_ij[k];
                            let r = acc.right[k];
                            let ri = acc.right_i[k];
                            let rj = acc.right_j[k];
                            let rij = acc.right_ij[k];
                            endpoint_i[e] += 2.0 * basis_k * g * gi;
                            endpoint_j[e] += 2.0 * basis_k * g * gj;
                            endpoint_ij[e] += 2.0 * basis_k * (gj * gi + g * gij);
                            endpoint_l[e] += 2.0 * basis_k * g * l;
                            endpoint_r[e] += 2.0 * basis_k * g * r;
                            endpoint_i_l[e] += 2.0 * basis_k * (l * gi + g * li);
                            endpoint_j_l[e] += 2.0 * basis_k * (l * gj + g * lj);
                            endpoint_ij_l[e] +=
                                2.0 * basis_k * (lj * gi + gj * li + l * gij + g * lij);
                            endpoint_i_r[e] += 2.0 * basis_k * (r * gi + g * ri);
                            endpoint_j_r[e] += 2.0 * basis_k * (r * gj + g * rj);
                            endpoint_ij_r[e] +=
                                2.0 * basis_k * (rj * gi + gj * ri + r * gij + g * rij);
                            endpoint_l_r[e] += 2.0 * basis_k * l * r;
                            endpoint_i_l_r[e] += 2.0 * basis_k * (l * ri + r * li);
                            endpoint_j_l_r[e] += 2.0 * basis_k * (l * rj + r * lj);
                            endpoint_ij_l_r[e] +=
                                2.0 * basis_k * (lj * ri + rj * li + l * rij + r * lij);
                        }
                    }

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let numerator_l = hp_i_l * hp_j + hp_i * hp_j_l;
                    let numerator_r = hp_i_r * hp_j + hp_i * hp_j_r;
                    let numerator_lr =
                        hp_i_lr * hp_j + hp_i_l * hp_j_r + hp_i_r * hp_j_l + hp_i * hp_j_lr;
                    let value_lr = h_i_lr * h_j
                        + h_i_l * h_j_r
                        + h_i_r * h_j_l
                        + h_i * h_j_lr
                        + h_lr * h_ij
                        + h_l * h_ij_r
                        + h_r * h_ij_l
                        + h * h_ij_lr
                        - hp_ij_lr * inv_hp
                        + hp_ij_l * hp_r * inv_hp_sq
                        + hp_ij_r * hp_l * inv_hp_sq
                        + hp_ij * hp_lr * inv_hp_sq
                        - 2.0 * hp_ij * hp_l * hp_r * inv_hp_cu
                        + numerator_lr * inv_hp_sq
                        - 2.0 * numerator_l * hp_r * inv_hp_cu
                        - 2.0 * numerator_r * hp_l * inv_hp_cu
                        - 2.0 * hp_i * hp_j * hp_lr * inv_hp_cu
                        + 6.0 * hp_i * hp_j * hp_l * hp_r * inv_hp_qu
                        + endpoint_chain_fourth(
                            &q,
                            endpoint_i,
                            endpoint_j,
                            endpoint_l,
                            endpoint_r,
                            endpoint_ij,
                            endpoint_i_l,
                            endpoint_i_r,
                            endpoint_j_l,
                            endpoint_j_r,
                            endpoint_l_r,
                            endpoint_ij_l,
                            endpoint_ij_r,
                            endpoint_i_l_r,
                            endpoint_j_l_r,
                            endpoint_ij_l_r,
                        );
                    acc.value += weights[global_row] * value_lr;
                    acc
                },
            )
            .reduce(
                || PsiPairBilinearAccum::new(p_resp),
                |mut left, right| {
                    left.value += right.value;
                    left
                },
            )
            .value;
        Ok(total)
    }

    fn scop_psi_pair_rows_per_chunk(&self, p_cov: usize) -> usize {
        let policy = ResourcePolicy::default_library();
        crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, 4 * p_cov.max(1))
            .max(1)
    }

    fn scop_psi_pair_cov_chunks(
        &self,
        op: &TensorKroneckerPsiOperator,
        axis_i: usize,
        axis_j: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
        let cov = self
            .covariate_design
            .try_row_chunk(rows.clone())
            .map_err(|e| format!("SCOP psi-psi covariate row chunk failed: {e}"))?;
        let cov_i = op
            .cov_first_axis_row_chunk(axis_i, rows.clone())
            .map_err(|e| format!("SCOP psi-psi covariate first-axis row chunk(i) failed: {e}"))?;
        let cov_j = op
            .cov_first_axis_row_chunk(axis_j, rows.clone())
            .map_err(|e| format!("SCOP psi-psi covariate first-axis row chunk(j) failed: {e}"))?;
        let cov_ij = op
            .cov_second_axis_row_chunk(axis_i, axis_j, rows)
            .map_err(|e| format!("SCOP psi-psi covariate second-axis row chunk failed: {e}"))?;
        Ok((cov, cov_i, cov_j, cov_ij))
    }

    fn scop_psi_psi_value_score_hvp_from_operator(
        &self,
        beta: &Array1<f64>,
        op: &TensorKroneckerPsiOperator,
        axis_i: usize,
        axis_j: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        direction: Option<&Array1<f64>>,
    ) -> Result<(f64, Array1<f64>, Option<Array1<f64>>), String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if endpoint_q.len() != n {
            return Err(format!(
                "SCOP psi-psi operator endpoint normalizer cache length {} != n={n}",
                endpoint_q.len()
            ));
        }
        let rows_per_chunk = self.scop_psi_pair_rows_per_chunk(p_cov).min(n.max(1));
        let mut objective = 0.0;
        let mut score = Array1::<f64>::zeros(p_total);
        let mut hvp = direction.map(|_| Array1::<f64>::zeros(p_total));

        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let (cov, cov_i, cov_j, cov_ij) =
                self.scop_psi_pair_cov_chunks(op, axis_i, axis_j, rows.clone())?;
            let (obj_chunk, score_chunk, hvp_chunk) = self.scop_psi_psi_value_score_hvp_from_cov(
                beta,
                cov.view(),
                cov_i.view(),
                cov_j.view(),
                cov_ij.view(),
                start,
                &endpoint_q[start..end],
                direction,
            )?;
            objective += obj_chunk;
            score.scaled_add(1.0, &score_chunk);
            if let (Some(total), Some(chunk)) = (hvp.as_mut(), hvp_chunk.as_ref()) {
                total.scaled_add(1.0, chunk);
            }
        }

        Ok((objective, score, hvp))
    }

    fn scop_psi_psi_bilinear_from_operator(
        &self,
        beta: &Array1<f64>,
        op: &TensorKroneckerPsiOperator,
        axis_i: usize,
        axis_j: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<f64, String> {
        let n = self.response_val_basis.nrows();
        let p_cov = self.covariate_design.ncols();
        if endpoint_q.len() != n {
            return Err(format!(
                "SCOP psi-psi bilinear operator endpoint normalizer cache length {} != n={n}",
                endpoint_q.len()
            ));
        }
        let rows_per_chunk = self.scop_psi_pair_rows_per_chunk(p_cov).min(n.max(1));
        let mut total = 0.0;
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let (cov, cov_i, cov_j, cov_ij) =
                self.scop_psi_pair_cov_chunks(op, axis_i, axis_j, rows.clone())?;
            total += self.scop_psi_psi_bilinear_from_cov(
                beta,
                cov.view(),
                cov_i.view(),
                cov_j.view(),
                cov_ij.view(),
                start,
                &endpoint_q[start..end],
                left,
                right,
            )?;
        }
        Ok(total)
    }

    fn scop_psi_hessian_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        op: &TensorKroneckerPsiOperator,
        axis: usize,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction.len() != p_total {
            return Err(format!(
                "SCOP psi Hessian directional derivative length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ));
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi hessian beta reshape failed: {e}"))?;
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi hessian direction reshape failed: {e}"))?;
        let cov = self
            .covariate_design
            .try_row_chunk(0..n)
            .map_err(|e| format!("SCOP psi hessian direction requires covariate row chunk: {e}"))?;
        let cov_psi = op
            .materialize_cov_first_axis(axis)
            .map_err(|e| format!("SCOP psi hessian materialize_cov_first failed: {e}"))?;
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(format!(
                "SCOP psi hessian covariate derivative shape {}x{} != expected {}x{}",
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ));
        }

        let weights = self.weights.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];
        let mut out = Array2::<f64>::zeros((p_total, p_total));

        for i in 0..n {
            let cov_row = cov.row(i);
            let psi_row = cov_psi.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let inv_hp_qu = inv_hp_sq * inv_hp_sq;

            let mut gamma = vec![0.0; p_resp];
            let mut gamma_dir = vec![0.0; p_resp];
            let mut gamma_psi = vec![0.0; p_resp];
            let mut gamma_psi_dir = vec![0.0; p_resp];
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                gamma_psi_dir[k] = dir_mat.row(k).dot(&psi_row);
            }

            let mut h_dir = rv[0] * gamma_dir[0];
            let mut hp_dir = rd[0] * gamma_dir[0];
            let mut hp_psi = rd[0] * gamma_psi[0];
            let mut h_psi_dir = rv[0] * gamma_psi_dir[0];
            let mut hp_psi_dir = rd[0] * gamma_psi_dir[0];
            for k in 1..p_resp {
                h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
                h_psi_dir +=
                    2.0 * rv[k] * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
                hp_psi_dir +=
                    2.0 * rd[k] * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
            }
            let q = row_quantities.endpoint_q[i];
            let mut endpoint_psi = [0.0; 2];
            let mut endpoint_dir = [0.0; 2];
            let mut endpoint_psi_dir = [0.0; 2];
            let mut endpoint_factor = vec![[0.0; 2]; p_resp];
            let mut endpoint_factor_dir = vec![[0.0; 2]; p_resp];
            let mut endpoint_psi_cov_factor = vec![[0.0; 2]; p_resp];
            let mut endpoint_psi_psi_factor = vec![[0.0; 2]; p_resp];
            let mut endpoint_psi_cov_factor_dir = vec![[0.0; 2]; p_resp];
            let mut endpoint_psi_psi_factor_dir = vec![[0.0; 2]; p_resp];
            for e in 0..2 {
                let basis = endpoint_basis[e];
                endpoint_psi[e] = basis[0] * gamma_psi[0];
                endpoint_dir[e] = basis[0] * gamma_dir[0];
                endpoint_psi_dir[e] = basis[0] * gamma_psi_dir[0];
                endpoint_factor[0][e] = basis[0];
                endpoint_psi_psi_factor[0][e] = basis[0];
                for k in 1..p_resp {
                    endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
                    endpoint_dir[e] += 2.0 * basis[k] * gamma[k] * gamma_dir[k];
                    endpoint_psi_dir[e] += 2.0
                        * basis[k]
                        * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
                    endpoint_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_factor_dir[k][e] = 2.0 * basis[k] * gamma_dir[k];
                    endpoint_psi_cov_factor[k][e] = 2.0 * basis[k] * gamma_psi[k];
                    endpoint_psi_psi_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_psi_cov_factor_dir[k][e] = 2.0 * basis[k] * gamma_psi_dir[k];
                    endpoint_psi_psi_factor_dir[k][e] = 2.0 * basis[k] * gamma_dir[k];
                }
            }
            let d_inv_hp = -hp_dir * inv_hp_sq;
            let d_inv_hp_sq = -2.0 * hp_dir * inv_hp_cu;
            let d_inv_hp_cu = -3.0 * hp_dir * inv_hp_qu;

            let mut h_factor = vec![0.0; p_resp];
            let mut hp_factor = vec![0.0; p_resp];
            let mut h_factor_dir = vec![0.0; p_resp];
            let mut hp_factor_dir = vec![0.0; p_resp];
            let mut hpsi_cov_factor = vec![0.0; p_resp];
            let mut hppsi_cov_factor = vec![0.0; p_resp];
            let mut hpsi_psi_factor = vec![0.0; p_resp];
            let mut hppsi_psi_factor = vec![0.0; p_resp];
            let mut hpsi_cov_factor_dir = vec![0.0; p_resp];
            let mut hppsi_cov_factor_dir = vec![0.0; p_resp];
            let mut hpsi_psi_factor_dir = vec![0.0; p_resp];
            let mut hppsi_psi_factor_dir = vec![0.0; p_resp];

            h_factor[0] = rv[0];
            hp_factor[0] = rd[0];
            hpsi_psi_factor[0] = rv[0];
            hppsi_psi_factor[0] = rd[0];
            for k in 1..p_resp {
                h_factor[k] = 2.0 * rv[k] * gamma[k];
                hp_factor[k] = 2.0 * rd[k] * gamma[k];
                h_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                hp_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
                hpsi_cov_factor[k] = 2.0 * rv[k] * gamma_psi[k];
                hppsi_cov_factor[k] = 2.0 * rd[k] * gamma_psi[k];
                hpsi_psi_factor[k] = 2.0 * rv[k] * gamma[k];
                hppsi_psi_factor[k] = 2.0 * rd[k] * gamma[k];
                hpsi_cov_factor_dir[k] = 2.0 * rv[k] * gamma_psi_dir[k];
                hppsi_cov_factor_dir[k] = 2.0 * rd[k] * gamma_psi_dir[k];
                hpsi_psi_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                hppsi_psi_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
            }

            for k in 0..p_resp {
                for l in 0..p_resp {
                    let same_shape = k == l && k > 0;
                    for c in 0..p_cov {
                        let row_idx = k * p_cov + c;
                        let h_a = h_factor[k] * cov_row[c];
                        let hp_a = hp_factor[k] * cov_row[c];
                        let h_a_dir = h_factor_dir[k] * cov_row[c];
                        let hp_a_dir = hp_factor_dir[k] * cov_row[c];
                        let hpsi_a =
                            hpsi_cov_factor[k] * cov_row[c] + hpsi_psi_factor[k] * psi_row[c];
                        let hppsi_a =
                            hppsi_cov_factor[k] * cov_row[c] + hppsi_psi_factor[k] * psi_row[c];
                        let hpsi_a_dir = hpsi_cov_factor_dir[k] * cov_row[c]
                            + hpsi_psi_factor_dir[k] * psi_row[c];
                        let hppsi_a_dir = hppsi_cov_factor_dir[k] * cov_row[c]
                            + hppsi_psi_factor_dir[k] * psi_row[c];
                        for d in 0..p_cov {
                            let col_idx = l * p_cov + d;
                            let h_b = h_factor[l] * cov_row[d];
                            let hp_b = hp_factor[l] * cov_row[d];
                            let h_b_dir = h_factor_dir[l] * cov_row[d];
                            let hp_b_dir = hp_factor_dir[l] * cov_row[d];
                            let hpsi_b =
                                hpsi_cov_factor[l] * cov_row[d] + hpsi_psi_factor[l] * psi_row[d];
                            let hppsi_b =
                                hppsi_cov_factor[l] * cov_row[d] + hppsi_psi_factor[l] * psi_row[d];
                            let hpsi_b_dir = hpsi_cov_factor_dir[l] * cov_row[d]
                                + hpsi_psi_factor_dir[l] * psi_row[d];
                            let hppsi_b_dir = hppsi_cov_factor_dir[l] * cov_row[d]
                                + hppsi_psi_factor_dir[l] * psi_row[d];
                            let (h_ab, hp_ab, hpsi_ab, hppsi_ab) = if same_shape {
                                (
                                    2.0 * rv[k] * cov_row[c] * cov_row[d],
                                    2.0 * rd[k] * cov_row[c] * cov_row[d],
                                    2.0 * rv[k]
                                        * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                    2.0 * rd[k]
                                        * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                )
                            } else {
                                (0.0, 0.0, 0.0, 0.0)
                            };
                            let endpoint_a = [
                                endpoint_factor[k][0] * cov_row[c],
                                endpoint_factor[k][1] * cov_row[c],
                            ];
                            let endpoint_b = [
                                endpoint_factor[l][0] * cov_row[d],
                                endpoint_factor[l][1] * cov_row[d],
                            ];
                            let endpoint_psi_a = [
                                endpoint_psi_cov_factor[k][0] * cov_row[c]
                                    + endpoint_psi_psi_factor[k][0] * psi_row[c],
                                endpoint_psi_cov_factor[k][1] * cov_row[c]
                                    + endpoint_psi_psi_factor[k][1] * psi_row[c],
                            ];
                            let endpoint_psi_b = [
                                endpoint_psi_cov_factor[l][0] * cov_row[d]
                                    + endpoint_psi_psi_factor[l][0] * psi_row[d],
                                endpoint_psi_cov_factor[l][1] * cov_row[d]
                                    + endpoint_psi_psi_factor[l][1] * psi_row[d],
                            ];
                            let endpoint_a_dir = [
                                endpoint_factor_dir[k][0] * cov_row[c],
                                endpoint_factor_dir[k][1] * cov_row[c],
                            ];
                            let endpoint_b_dir = [
                                endpoint_factor_dir[l][0] * cov_row[d],
                                endpoint_factor_dir[l][1] * cov_row[d],
                            ];
                            let endpoint_psi_a_dir = [
                                endpoint_psi_cov_factor_dir[k][0] * cov_row[c]
                                    + endpoint_psi_psi_factor_dir[k][0] * psi_row[c],
                                endpoint_psi_cov_factor_dir[k][1] * cov_row[c]
                                    + endpoint_psi_psi_factor_dir[k][1] * psi_row[c],
                            ];
                            let endpoint_psi_b_dir = [
                                endpoint_psi_cov_factor_dir[l][0] * cov_row[d]
                                    + endpoint_psi_psi_factor_dir[l][0] * psi_row[d],
                                endpoint_psi_cov_factor_dir[l][1] * cov_row[d]
                                    + endpoint_psi_psi_factor_dir[l][1] * psi_row[d],
                            ];
                            let (endpoint_ab, endpoint_psi_ab) = if same_shape {
                                (
                                    [
                                        2.0 * endpoint_basis[0][k] * cov_row[c] * cov_row[d],
                                        2.0 * endpoint_basis[1][k] * cov_row[c] * cov_row[d],
                                    ],
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                        2.0 * endpoint_basis[1][k]
                                            * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                    ],
                                )
                            } else {
                                ([0.0; 2], [0.0; 2])
                            };
                            let numerator = hppsi_a * hp_b + hp_a * hppsi_b;
                            let numerator_dir = hppsi_a_dir * hp_b
                                + hppsi_a * hp_b_dir
                                + hp_a_dir * hppsi_b
                                + hp_a * hppsi_b_dir;
                            let barrier_product = hp_a * hp_b * hp_psi;
                            let barrier_product_dir = hp_a_dir * hp_b * hp_psi
                                + hp_a * hp_b_dir * hp_psi
                                + hp_a * hp_b * hp_psi_dir;
                            let value = hpsi_a_dir * h_b
                                + hpsi_a * h_b_dir
                                + h_a_dir * hpsi_b
                                + h_a * hpsi_b_dir
                                + h_psi_dir * h_ab
                                + h_dir * hpsi_ab
                                + numerator_dir * inv_hp_sq
                                + numerator * d_inv_hp_sq
                                - 2.0
                                    * (barrier_product_dir * inv_hp_cu
                                        + barrier_product * d_inv_hp_cu)
                                - hppsi_ab * d_inv_hp
                                + hp_ab * hp_psi_dir * inv_hp_sq
                                + hp_ab * hp_psi * d_inv_hp_sq
                                + endpoint_chain_fourth(
                                    &q,
                                    endpoint_psi,
                                    endpoint_a,
                                    endpoint_b,
                                    endpoint_dir,
                                    endpoint_psi_a,
                                    endpoint_psi_b,
                                    endpoint_psi_dir,
                                    endpoint_ab,
                                    endpoint_a_dir,
                                    endpoint_b_dir,
                                    endpoint_psi_ab,
                                    endpoint_psi_a_dir,
                                    endpoint_psi_b_dir,
                                    [0.0; 2],
                                    [0.0; 2],
                                );
                            out[[row_idx, col_idx]] += wi * value;
                        }
                    }
                }
            }
        }

        Ok(0.5 * (&out + &out.t()))
    }
}

fn ctn_penalty_scale_log_lambdas(
    penalties: &[PenaltyMatrix],
    likelihood_gram: &Array2<f64>,
) -> Array1<f64> {
    if penalties.is_empty() {
        return Array1::zeros(0);
    }

    let likelihood_scale = matrix_diag_mean_abs(likelihood_gram).max(1.0e-8);
    Array1::from_iter(penalties.iter().map(|penalty| {
        let penalty_scale = penalty_diag_scale(penalty).max(1.0e-8);
        // Lower-bound the SEED log-lambda at 0 (i.e., λ ≥ 1) so we never
        // start the outer optimizer in the under-regularized regime where
        // the CTN inner is structurally rank-deficient (small-n / p > n).
        // The outer BFGS is free to step below 0 when there's enough data
        // to support it; only the cold-start is constrained. Without this
        // floor, ratios like n=64, p_resp×p_cov ≈ 200 produce a seed of
        // log_lambda ≈ -12 (λ ≈ 6e-6), which leaves the inner solve to
        // pick wild β coefficients and cascade into predict-time monotonicity
        // violations (h' < 0 on the response grid, observed as -1e15 spikes
        // in CI synthetic-biobank tests).
        (likelihood_scale / penalty_scale).ln().clamp(0.0, 12.0)
    }))
}

fn penalty_diag_scale(penalty: &PenaltyMatrix) -> f64 {
    match penalty {
        PenaltyMatrix::Dense(matrix) => {
            matrix_diag_mean_abs(matrix).max(matrix_frobenius_rms(matrix))
        }
        PenaltyMatrix::KroneckerFactored { left, right } => {
            let diag_scale = matrix_diag_mean_abs(left) * matrix_diag_mean_abs(right);
            let frob_scale = matrix_frobenius_rms(left) * matrix_frobenius_rms(right);
            diag_scale.max(frob_scale)
        }
        PenaltyMatrix::Blockwise { local, .. } => {
            matrix_diag_mean_abs(local).max(matrix_frobenius_rms(local))
        }
    }
}

fn matrix_diag_mean_abs(matrix: &Array2<f64>) -> f64 {
    let d = matrix.nrows().min(matrix.ncols());
    if d == 0 {
        return 0.0;
    }
    matrix.diag().iter().map(|v| v.abs()).sum::<f64>() / d as f64
}

fn matrix_frobenius_rms(matrix: &Array2<f64>) -> f64 {
    let d = matrix.nrows().max(1).min(matrix.ncols().max(1));
    (matrix.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt()
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
///
/// Each chunk weights `D` rows in-place, then accumulates `B[chunk]^T · DW`
/// directly into `out` via a faer SIMD matmul with `Accum::Add`. This
/// eliminates the per-chunk `Array2` from the previous `out += &bl.t().dot(&dw)`
/// pattern (one allocation + one element-wise `+=` pass per chunk) and uses
/// the multi-threaded faer kernel instead of ndarray's serial dot.
fn chunked_weighted_bt_d(
    b: &Array2<f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    d: &Array2<f64>,
    policy: &ResourcePolicy,
) -> Array2<f64> {
    use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, matmul_parallelism};
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let n = weights.len();
    let pb = b.ncols();
    let pd = d.ncols();
    let rows_per_chunk =
        crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, pb + pd);
    let mut out = Array2::<f64>::zeros((pb, pd));
    if n == 0 || pb == 0 || pd == 0 {
        return out;
    }
    let mut out_view = array2_to_matmut(&mut out);
    let mut dw_buf = Array2::<f64>::zeros((rows_per_chunk.min(n), pd));
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let rows = end - start;
        let bl = b.slice(s![start..end, ..]);
        let dl = d.slice(s![start..end, ..]);
        {
            let mut dw_slice = dw_buf.slice_mut(s![..rows, ..]);
            for local in 0..rows {
                let w = weights[start + local];
                let drow = dl.row(local);
                let mut wrow = dw_slice.row_mut(local);
                ndarray::Zip::from(&mut wrow)
                    .and(&drow)
                    .for_each(|dst, &src| *dst = w * src);
            }
        }
        let bl_view = FaerArrayView::new(&bl);
        let dw_slice = dw_buf.slice(s![..rows, ..]);
        let dw_view = FaerArrayView::new(&dw_slice);
        let par = matmul_parallelism(pb, pd, rows);
        matmul(
            out_view.as_mut(),
            Accum::Add,
            bl_view.as_ref().transpose(),
            dw_view.as_ref(),
            1.0,
            par,
        );
    }
    out
}

/// Chunked weighted `B^T diag(w) D` product where `B` and `D` are
/// operator-backed `DesignMatrix` instances. Materializes only one row chunk
/// at a time using the operator's `row_chunk` primitive, so neither factor's
/// full dense form ever lives in memory.
///
/// Each chunk's contribution accumulates into `out` via a faer SIMD matmul
/// with `Accum::Add` rather than `out += &bl.t().dot(&dw)`. This drops one
/// `Array2` allocation per chunk and routes the inner GEMM through faer's
/// multi-threaded kernel with a work-aware parallelism choice.
fn chunked_weighted_bt_d_designmatrix(
    b: &DesignMatrix,
    weights: ndarray::ArrayView1<'_, f64>,
    d: &DesignMatrix,
    policy: &ResourcePolicy,
) -> Result<Array2<f64>, String> {
    use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, matmul_parallelism};
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let n = weights.len();
    let pb = b.ncols();
    let pd = d.ncols();
    let rows_per_chunk =
        crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, pb + pd);
    let mut out = Array2::<f64>::zeros((pb, pd));
    if n == 0 || pb == 0 || pd == 0 {
        return Ok(out);
    }
    let mut out_view = array2_to_matmut(&mut out);
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let rows = end - start;
        let bl = b.try_row_chunk(start..end).map_err(|e| e.to_string())?;
        let mut dw = d.try_row_chunk(start..end).map_err(|e| e.to_string())?;
        for local in 0..rows {
            let w = weights[start + local];
            if w != 1.0 {
                let mut wrow = dw.row_mut(local);
                wrow.mapv_inplace(|v| w * v);
            }
        }
        let bl_view = FaerArrayView::new(&bl);
        let dw_view = FaerArrayView::new(&dw);
        let par = matmul_parallelism(pb, pd, rows);
        matmul(
            out_view.as_mut(),
            Accum::Add,
            bl_view.as_ref().transpose(),
            dw_view.as_ref(),
            1.0,
            par,
        );
    }
    Ok(out)
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
        let evaluate_start = std::time::Instant::now();
        let beta = &block_states[0].beta;
        let row_q_start = std::time::Instant::now();
        let row_quantities = self.row_quantities(beta)?;
        log::info!(
            "[STAGE] CTN row_quantities (h, h', 1/h', powers) n={} elapsed={:.3}s",
            row_quantities.h.len(),
            row_q_start.elapsed().as_secs_f64(),
        );
        let h = row_quantities.h.as_ref();
        let n = h.len();

        let log_likelihood = row_quantities.log_likelihood;
        // SCOP gradient and exact negative Hessian. Response column 0 is the
        // linear location component b(x); response columns >=1 are squared
        // γ_k(x)^2 shape components.
        let grad_start = std::time::Instant::now();
        let (grad, hessian) = self.scop_gradient_and_negative_hessian(beta, &row_quantities)?;
        log::info!(
            "[STAGE] CTN gradient terms n={} p={} elapsed={:.3}s",
            n,
            grad.len(),
            grad_start.elapsed().as_secs_f64(),
        );

        let hess_start = std::time::Instant::now();
        let p_dim = hessian.nrows() as u64;
        let n_u64 = n as u64;
        log::info!(
            "[STAGE] CTN hessian terms (SCOP exact dense) n={} p={} flops~{} elapsed={:.3}s",
            n,
            p_dim,
            n_u64.saturating_mul(p_dim).saturating_mul(p_dim),
            hess_start.elapsed().as_secs_f64(),
        );
        log::info!(
            "[STAGE] CTN evaluate end n={} p={} elapsed={:.3}s",
            n,
            p_dim,
            evaluate_start.elapsed().as_secs_f64(),
        );

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
        // The line search uses NEG_INFINITY as the barrier-violation signal,
        // so we can't propagate the row_quantities Err here. Translate any
        // h' validation failure back into the NEG_INFINITY rejection contract.
        let row_quantities = match self.row_quantities(&block_states[0].beta) {
            Ok(rq) => rq,
            Err(_) => return Ok(f64::NEG_INFINITY),
        };
        Ok(row_quantities.log_likelihood)
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
        let row_quantities = self.row_quantities(beta)?;
        let log_likelihood = row_quantities.log_likelihood;
        let gradient = self.scop_gradient(beta, &row_quantities)?;
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
        //   This is only an inner coefficient-space cost estimate; outer
        //   θθ Hessian availability is declared separately.
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
        // numerical-underflow guard over the derivative grid via
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
        let scan_start = std::time::Instant::now();

        // Numerical-underflow guard for the learned, beta-dependent part of
        // the SCOP derivative. The actual derivative is
        //
        //   h'(y, x) = ε + q(y, x; β),    q = Σ M_r(y) γ_r(x)^2.
        //
        // Since ε is integrated directly into h and h', q=0 is a perfectly
        // smooth interior point of the likelihood; treating q=ε as a hard
        // line-search boundary makes the inner mode a constrained/KKT point
        // while the exact outer REML Hessian assumes an unconstrained smooth
        // mode. That mismatch is what produced enormous ARC model decreases
        // and dozens of trust-region rejections on small CTN demos. The scan
        // therefore only rejects catastrophic numerical drift below −ε, where
        // even the structural floor could be cancelled.
        //
        // Each design owns its own streaming reduction over its virtual rows
        // (KhatriRao: n observations; Kronecker: n_cov × n_grid pairs without
        // materializing the dense forward image). Both return the smallest
        // binding step ratio, or +∞ if no row binds. Composing via `f64::min`
        // is associative for non-NaN inputs, so the final α_max is
        // bit-equivalent to a single scan over the union of binding rows.
        //
        // Observation-row reduction stays on the un-cached path because the
        // KhatriRao variant streams over `n` observation rows directly via
        // `forward_mul`; there is no factored projection to reuse there.
        let alpha_obs = self.x_deriv_kron.scop_affine_squared_min_step_to_boundary(
            beta,
            delta,
            -TRANSFORMATION_MONOTONICITY_EPS,
            1e-14,
        );
        let alpha_grid = self
            .x_deriv_grid_kron
            .scop_affine_squared_min_step_to_boundary(
                beta,
                delta,
                -TRANSFORMATION_MONOTONICITY_EPS,
                1e-14,
            );
        let alpha_max = 1.0_f64.min(alpha_obs).min(alpha_grid);
        log::info!(
            "[STAGE] CTN monotonicity grid scan alpha_obs={:.3e} alpha_grid={:.3e} elapsed={:.3}s",
            alpha_obs,
            alpha_grid,
            scan_start.elapsed().as_secs_f64(),
        );

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
        // The CTN tensor design is intentionally factored. Strict monotonicity
        // is encoded structurally as `h' = ε + Σ M_r γ_r²`, so there are no
        // dense active-set constraints to expose here.
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
        let row_quantities = self.row_quantities(beta)?;
        let dd = self.scop_hessian_directional_derivative(beta, d_beta, &row_quantities)?;
        Ok(Some(dd))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Single block: joint Hessian = block Hessian.
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        let (_, hessian) = self.scop_gradient_and_negative_hessian(beta, &row_quantities)?;
        Ok(Some(hessian))
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
        let row_quantities = self.row_quantities(beta)?;
        let d2 = self.scop_hessian_second_directional_derivative(
            beta,
            d_beta_u_flat,
            d_beta_v_flat,
            &row_quantities,
        )?;
        Ok(Some(d2))
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
        let psi_first_start = std::time::Instant::now();
        let deriv = &psi_derivs[0][psi_index];
        let beta = &block_states[0].beta;
        let row = self.row_quantities(beta)?;
        let op = deriv
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis = deriv.implicit_axis;
        let terms = self.scop_psi_terms(beta, &row, op, axis)?;

        log::info!(
            "[STAGE] CTN psi first-order terms axis={} psi_index={} elapsed={:.3}s",
            deriv.implicit_axis,
            psi_index,
            psi_first_start.elapsed().as_secs_f64(),
        );

        Ok(Some(terms))
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
        let psi_pair_start = std::time::Instant::now();
        let deriv_i = &psi_derivs[0][psi_i];
        let deriv_j = &psi_derivs[0][psi_j];
        let beta = &block_states[0].beta;
        let row = self.row_quantities(beta)?;
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(format!(
                "SCOP psi-psi terms beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                beta.len()
            ));
        }

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

        let (objective_psi_psi, score_psi_psi, _) = self
            .scop_psi_psi_value_score_hvp_from_operator(
                beta,
                op,
                axis_i,
                axis_j,
                row.endpoint_q.as_slice(),
                None,
            )?;
        let hessian_psi_psi_operator: Box<dyn HyperOperator> =
            Box::new(TransformationNormalPsiPsiHessianOperator::new(
                Arc::new(self.clone()),
                beta.clone(),
                Arc::clone(
                    deriv_i
                        .implicit_operator
                        .as_ref()
                        .expect("validated CTN psi derivative has an implicit operator"),
                ),
                axis_i,
                axis_j,
                Arc::clone(&row.endpoint_q),
            ));

        // Result-validation gate. A trial point can still make the SCOP row
        // terms non-finite through an invalid h' or an exploding ψ second
        // derivative in the covariate basis. Surface that as an infeasible
        // exact-Newton evaluation instead of passing NaNs into the unified
        // outer evaluator.
        if !objective_psi_psi.is_finite() || !score_psi_psi.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "TransformationNormalFamily exact ψ-ψ second-order terms produced \
                 non-finite values at psi_i={psi_i}, psi_j={psi_j}: \
                 obj_finite={}, score_all_finite={}. \
                 The outer evaluator should retreat from this trial point.",
                objective_psi_psi.is_finite(),
                score_psi_psi.iter().all(|v| v.is_finite()),
            ));
        }

        log::info!(
            "[STAGE] CTN psi-psi pair (psi_i={}, psi_j={}, axes={},{}) elapsed={:.3}s",
            psi_i,
            psi_j,
            deriv_i.implicit_axis,
            deriv_j.implicit_axis,
            psi_pair_start.elapsed().as_secs_f64(),
        );

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(hessian_psi_psi_operator),
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
        let op = deriv
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis = deriv.implicit_axis;
        let row = self.row_quantities(beta)?;
        let hess =
            self.scop_psi_hessian_directional_derivative(beta, d_beta_flat, &row, op, axis)?;
        Ok(Some(hess))
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
        let row_quantities = self.row_quantities(beta)?;
        let workspace = TransformationNormalJointHessianWorkspace::new(
            Arc::new(self.clone()),
            beta.clone(),
            row_quantities.clone(),
        )?;
        Ok(Some(
            Arc::new(workspace) as Arc<dyn ExactNewtonJointHessianWorkspace>
        ))
    }

    fn inner_coefficient_hessian_hvp_available(&self, _specs: &[ParameterBlockSpec]) -> bool {
        // CTN's SCOP coefficient-space joint Hessian is supplied as a
        // row-streaming matrix-free Hv operator.
        true
    }

    fn outer_hyper_hessian_hvp_available(&self, _specs: &[ParameterBlockSpec]) -> bool {
        true
    }

    fn outer_hyper_hessian_dense_available(&self, _specs: &[ParameterBlockSpec]) -> bool {
        // Dense materialization remains mathematically available through the
        // outer-HVP operator, but SCOP's primary production path is the
        // matrix-free θθ operator above.
        true
    }
}

// ---------------------------------------------------------------------------
// Matrix-free joint Hessian workspace (Khatri-Rao operator-only)
// ---------------------------------------------------------------------------

/// Per-evaluation workspace for the SCOP-CTN joint Hessian.
///
/// The old linear-`h` CTN Hessian had the form
/// `X_val' W X_val + X_deriv' diag(w / h'^2) X_deriv`. SCOP is nonlinear in
/// the shape rows, so `H v` must be evaluated through the rowwise chain rule.
/// This workspace keeps the accepted `β` and row quantities and applies
/// `H`, `D H[u]`, and `D²H[u,v]` without materializing a dense p×p matrix.
struct TransformationNormalJointHessianWorkspace {
    /// Shared family handle. Cloning the workspace's family for each downstream
    /// matrix-free operator (dH, d²H per psi coord and per pair) would copy
    /// the full row-space Kronecker designs (~hundreds of MiB at biobank
    /// scale) per call. Arc-sharing makes operator construction O(1).
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    row_quantities: TransformationNormalRowQuantityCache,
}

impl TransformationNormalJointHessianWorkspace {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        row_quantities: TransformationNormalRowQuantityCache,
    ) -> Result<Self, String> {
        Ok(Self {
            family,
            beta,
            row_quantities,
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
        self.family
            .scop_hessian_matvec(&self.beta, &self.row_quantities, v)
    }

    /// Exact diagonal of the unpenalized joint Hessian.
    fn compute_diagonal(&self) -> Result<Array1<f64>, String> {
        self.family
            .scop_hessian_diagonal(&self.beta, &self.row_quantities)
    }
}

impl ExactNewtonJointHessianWorkspace for TransformationNormalJointHessianWorkspace {
    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.apply_hessian(v)?))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.compute_diagonal()?))
    }

    fn directional_derivative(&self, _: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
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
        let op = TransformationNormalDhMatrixFreeOperator::new(
            Arc::clone(&self.family),
            self.beta.clone(),
            self.row_quantities.clone(),
            d_beta_flat.clone(),
        );
        Ok(Some(Arc::new(op) as Arc<dyn HyperOperator>))
    }

    fn second_directional_derivative(
        &self,
        _: &Array1<f64>,
        _: &Array1<f64>,
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
        let op = TransformationNormalD2hMatrixFreeOperator::new(
            Arc::clone(&self.family),
            self.beta.clone(),
            self.row_quantities.clone(),
            d_beta_u.clone(),
            d_beta_v.clone(),
        );
        Ok(Some(Arc::new(op) as Arc<dyn HyperOperator>))
    }
}

/// Matrix-free directional derivative of the CTN joint Hessian.
///
/// SCOP makes the derivative row-dependent through `γ_k(x)`, so this operator
/// evaluates `D H[direction] · v` by streaming rows through the exact chain
/// rule instead of using the old scalar-weighted `X_deriv' diag(.) X_deriv`
/// identity.
struct TransformationNormalDhMatrixFreeOperator {
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    row_quantities: TransformationNormalRowQuantityCache,
    direction: Array1<f64>,
}

impl TransformationNormalDhMatrixFreeOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        row_quantities: TransformationNormalRowQuantityCache,
        direction: Array1<f64>,
    ) -> Self {
        Self {
            family,
            beta,
            row_quantities,
            direction,
        }
    }

    fn p_total(&self) -> usize {
        self.family.x_deriv_kron.ncols()
    }

    fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_hessian_directional_matvec(&self.beta, &self.direction, &self.row_quantities, v)
            .expect("validated CTN dH operator inputs should not fail")
    }
}

impl HyperOperator for TransformationNormalDhMatrixFreeOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let p = factor.nrows();
        let k = factor.ncols();
        let cols: Vec<Array1<f64>> = (0..k)
            .into_par_iter()
            .map(|c| self.apply(&factor.column(c).to_owned()))
            .collect();
        let mut out = Array2::<f64>::zeros((p, k));
        for (c, bv) in cols.into_iter().enumerate() {
            out.column_mut(c).assign(&bv);
        }
        out
    }

    fn to_dense(&self) -> Array2<f64> {
        self.family
            .scop_hessian_directional_derivative(&self.beta, &self.direction, &self.row_quantities)
            .expect("validated CTN dH operator inputs should not fail")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

/// Matrix-free second directional derivative of the CTN joint Hessian.
///
/// This is the SCOP rowwise chain-rule operator for `D²H[u, v] · w`; it keeps
/// the memory profile of matrix-free REML while matching the dense exact
/// second derivative.
struct TransformationNormalD2hMatrixFreeOperator {
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    row_quantities: TransformationNormalRowQuantityCache,
    direction_u: Array1<f64>,
    direction_v: Array1<f64>,
}

impl TransformationNormalD2hMatrixFreeOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        row_quantities: TransformationNormalRowQuantityCache,
        direction_u: Array1<f64>,
        direction_v: Array1<f64>,
    ) -> Self {
        Self {
            family,
            beta,
            row_quantities,
            direction_u,
            direction_v,
        }
    }

    fn p_total(&self) -> usize {
        self.family.x_deriv_kron.ncols()
    }

    fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_hessian_second_directional_matvec(
                &self.beta,
                &self.direction_u,
                &self.direction_v,
                &self.row_quantities,
                v,
            )
            .expect("validated CTN d2H operator inputs should not fail")
    }
}

impl HyperOperator for TransformationNormalD2hMatrixFreeOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let p = factor.nrows();
        let k = factor.ncols();
        let cols: Vec<Array1<f64>> = (0..k)
            .into_par_iter()
            .map(|c| self.apply(&factor.column(c).to_owned()))
            .collect();
        let mut out = Array2::<f64>::zeros((p, k));
        for (c, bv) in cols.into_iter().enumerate() {
            out.column_mut(c).assign(&bv);
        }
        out
    }

    fn to_dense(&self) -> Array2<f64> {
        self.family
            .scop_hessian_second_directional_derivative(
                &self.beta,
                &self.direction_u,
                &self.direction_v,
                &self.row_quantities,
            )
            .expect("validated CTN d2H operator inputs should not fail")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

struct TransformationNormalPsiPsiHessianOperator {
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    axis_i: usize,
    axis_j: usize,
    endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
}

impl TransformationNormalPsiPsiHessianOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis_i: usize,
        axis_j: usize,
        endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    ) -> Self {
        Self {
            family,
            beta,
            op,
            axis_i,
            axis_j,
            endpoint_q,
        }
    }

    fn p_total(&self) -> usize {
        self.beta.len()
    }

    fn tensor_op(&self) -> &TensorKroneckerPsiOperator {
        self.op
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .expect("validated CTN psi-psi operator must remain tensor-backed")
    }

    fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_psi_psi_value_score_hvp_from_operator(
                &self.beta,
                self.tensor_op(),
                self.axis_i,
                self.axis_j,
                self.endpoint_q.as_slice(),
                Some(v),
            )
            .expect("validated CTN psi-psi operator inputs should not fail")
            .2
            .expect("CTN psi-psi operator called without HVP output")
    }
}

impl HyperOperator for TransformationNormalPsiPsiHessianOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        debug_assert_eq!(v.len(), self.p_total());
        debug_assert_eq!(u.len(), self.p_total());
        self.family
            .scop_psi_psi_bilinear_from_operator(
                &self.beta,
                self.tensor_op(),
                self.axis_i,
                self.axis_j,
                self.endpoint_q.as_slice(),
                v,
                u,
            )
            .expect("validated CTN psi-psi bilinear inputs should not fail")
    }

    fn has_fast_bilinear_view(&self) -> bool {
        true
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let p = factor.nrows();
        let k = factor.ncols();
        let cols: Vec<Array1<f64>> = (0..k)
            .into_par_iter()
            .map(|c| self.apply(&factor.column(c).to_owned()))
            .collect();
        let mut out = Array2::<f64>::zeros((p, k));
        for (c, bv) in cols.into_iter().enumerate() {
            out.column_mut(c).assign(&bv);
        }
        out
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.p_total();
        let identity = Array2::<f64>::eye(p);
        self.mul_mat(&identity)
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Response-direction basis construction
// ---------------------------------------------------------------------------

/// Build the response-direction basis: an unconstrained location column plus
/// I-spline values `I_k(y)` with derivatives `M_k(y) = I'_k(y)`.
///
/// Returns (value_basis = `[1, I_k]`, derivative_basis = `[0, M_k]`,
/// penalties embedded with an unpenalized location row/column, regenerated
/// I-spline knots, identity coef_transform for the I-spline shape block).
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

    let response_degree = config.response_degree;
    if response_degree < 1 {
        return Err(format!(
            "response_degree must be >= 1 for the I-spline basis, got {response_degree}"
        ));
    }
    let k_internal = config.response_num_internal_knots;
    let k_prime = k_internal.checked_sub(2).ok_or_else(|| {
        format!(
            "response_num_internal_knots = {k_internal}; I-spline contract \
             requires K' = K − 2 ≥ 0, so need K ≥ 2"
        )
    })?;

    // Regenerate clamped knots. The I-spline builder integrates a degree
    // `(response_degree + 1)` B-spline basis into a degree-`response_degree`
    // value basis, so the seed-time degree passed here is `response_degree + 1`.
    let mut knots =
        initializewiggle_knots_from_seed(response.view(), response_degree + 1, k_prime)?;
    let response_min = response.iter().copied().fold(f64::INFINITY, f64::min);
    let response_max = response.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let response_span = (response_max - response_min).abs().max(1.0);
    let support_guard = response_span * 1.0e-3;
    let boundary_repeats = response_degree + 2;
    if knots.len() >= 2 * boundary_repeats {
        for idx in 0..boundary_repeats {
            knots[idx] = response_min - support_guard;
            let right_idx = knots.len() - 1 - idx;
            knots[right_idx] = response_max + support_guard;
        }
    }

    // I-spline value basis I_k(y).
    let (i_val_basis, _) = create_basis::<Dense>(
        response.view(),
        KnotSource::Provided(knots.view()),
        response_degree,
        BasisOptions::i_spline(),
    )
    .map_err(|e| e.to_string())?;
    let shape_val = i_val_basis.as_ref().clone();
    let p_shape = shape_val.ncols();

    // M-spline derivative basis M_k(y) = I'_k(y).
    let shape_deriv = create_ispline_derivative_dense(response.view(), &knots, response_degree, 1)
        .map_err(|e| e.to_string())?;
    if shape_deriv.ncols() != p_shape {
        return Err(format!(
            "I-spline derivative column count {} does not match value basis {p_shape}",
            shape_deriv.ncols()
        ));
    }
    if shape_deriv.nrows() != n {
        return Err(format!(
            "I-spline derivative row count {} does not match n = {n}",
            shape_deriv.nrows()
        ));
    }

    let p_resp = p_shape + 1;
    let mut resp_val = Array2::<f64>::zeros((n, p_resp));
    resp_val.column_mut(0).fill(1.0);
    resp_val.slice_mut(s![.., 1..]).assign(&shape_val);
    let mut resp_deriv = Array2::<f64>::zeros((n, p_resp));
    resp_deriv.slice_mut(s![.., 1..]).assign(&shape_deriv);

    // SCOP-CTN coef-transform is identity: I-splines have no constant in
    // their span, and squaring γ removes the per-component sign null direction.
    let transform = Array2::<f64>::eye(p_shape);

    // SCOP keeps a Gaussian quadratic prior on the latent γ shape factors,
    // not on the final non-negative I-spline coefficient α = γ². That is a
    // deliberate latent-prior penalty: replacing it with a final-function
    // roughness penalty would make the penalty nonlinear in β and would need
    // a different outer objective normalizer than the quadratic REML term.
    // Embed the latent response penalty into the full response block with an
    // unpenalized location row/column.
    let mut resp_penalties = Vec::new();
    let add_penalty = |order: usize, penalties: &mut Vec<Array2<f64>>| -> Result<(), String> {
        if order == 0 || order >= p_shape {
            return Ok(());
        }
        let shape_pen =
            create_difference_penalty_matrix(p_shape, order, None).map_err(|e| e.to_string())?;
        let mut full_pen = Array2::<f64>::zeros((p_resp, p_resp));
        full_pen.slice_mut(s![1.., 1..]).assign(&shape_pen);
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

/// Evaluate the SCOP derivative response basis `[0, M_k(y)]` at `values`.
///
/// `transform` is the SCOP-CTN coef-transform; under the I-spline response
/// basis it is the identity (size p_shape × p_shape), but it is still applied
/// to handle the (atypical) case of a non-identity caller-provided transform
/// without requiring the call sites to be aware of it. The output has shape
/// `values.len() × (1 + transform.ncols())`.
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
    if knots.is_empty() {
        return Err("response derivative grid needs knots".to_string());
    }
    if degree < 1 {
        return Err(format!(
            "response degree must be >= 1 for I-spline derivative, got {degree}"
        ));
    }

    // M_k(y) = I'_k(y) at the grid points.
    let raw_m = create_ispline_derivative_dense(values.view(), knots, degree, 1)
        .map_err(|e| e.to_string())?;

    if raw_m.ncols() != transform.nrows() {
        return Err(format!(
            "response derivative transform shape mismatch: M_k cols={} transform rows={}",
            raw_m.ncols(),
            transform.nrows()
        ));
    }
    let shape_deriv = fast_ab(&raw_m, transform);
    let mut out = Array2::<f64>::zeros((values.len(), shape_deriv.ncols() + 1));
    out.slice_mut(s![.., 1..]).assign(&shape_deriv);
    Ok(out)
}

fn response_endpoint_value_bases(transform: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let mut lower = Array1::<f64>::zeros(transform.ncols() + 1);
    let mut upper = Array1::<f64>::zeros(transform.ncols() + 1);
    lower[0] = 1.0;
    upper[0] = 1.0;
    for col in 0..transform.ncols() {
        upper[col + 1] = transform.column(col).sum();
    }
    (lower, upper)
}

fn response_floor_offsets(
    response: &Array1<f64>,
    knots: &Array1<f64>,
    response_median: f64,
) -> (Array1<f64>, f64, f64) {
    let row_offsets = response.mapv(|y| TRANSFORMATION_MONOTONICITY_EPS * (y - response_median));
    let lower_y = knots
        .first()
        .copied()
        .unwrap_or_else(|| response.iter().copied().fold(f64::INFINITY, f64::min));
    let upper_y = knots
        .last()
        .copied()
        .unwrap_or_else(|| response.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    (
        row_offsets,
        TRANSFORMATION_MONOTONICITY_EPS * (lower_y - response_median),
        TRANSFORMATION_MONOTONICITY_EPS * (upper_y - response_median),
    )
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
        if width <= TRANSFORMATION_GRID_RELATIVE_TOL * span {
            continue;
        }
        for sidx in 1..TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS {
            let frac = sidx as f64 / TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS as f64;
            values.push(left + frac * width);
        }
    }
    // No tail guard. The B-spline derivative basis clamps `x` to the basis
    // support `[t_d, t_{p_basis}]`, so any check at `min_y - guard`
    // evaluates the same `B_k'` as the boundary point — already covered by
    // the existing knot/quantile entries. The historical 25% extension was
    // a no-op redundant scan that nonetheless drove the perception that
    // CTN was extrapolating outside the basis. Removing it makes the fit
    // and predict checks operate on the same support.
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() <= TRANSFORMATION_GRID_RELATIVE_TOL * span);
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
    let response_deriv_grid_raw =
        evaluate_response_derivative_basis(&response_grid, knots, degree, transform)?;
    let keep_rows: Vec<usize> = (0..response_deriv_grid_raw.nrows())
        .filter(|&i| {
            response_deriv_grid_raw
                .row(i)
                .slice(s![1..])
                .iter()
                .any(|&v| v > 0.0)
        })
        .collect();
    if keep_rows.is_empty() {
        return Err("SCOP monotonicity grid has no positive derivative rows".to_string());
    }
    let mut response_deriv_grid =
        Array2::<f64>::zeros((keep_rows.len(), response_deriv_grid_raw.ncols()));
    for (out_i, &src_i) in keep_rows.iter().enumerate() {
        response_deriv_grid
            .row_mut(out_i)
            .assign(&response_deriv_grid_raw.row(src_i));
    }
    // SCOP monotonicity is coefficient-side: `h'(y,x)=Σ_k M_k(y)γ_k(x)^2`.
    // The old affine-deviation loose-bound rows encoded linear B-spline
    // derivative identities and are invalid for squared I-spline shape
    // components. The M-spline grid rows are therefore the entire response-side
    // constraint set here; covariate axis-extreme rows below keep the same
    // train/predict covariate support contract.
    // Augment the covariate side with `2 · p_cov` axis-extreme rows so the
    // loose-bound certificate `β₁(x) + δ_k(x) ≥ ε` holds at every corner of
    // the axis-aligned box `[col_min_j, col_max_j]^{p_cov}` reached after
    // `axis_clip_to_training_ranges` on the predict path. Each axis-extreme
    // row is the training-mean covariate vector with column j replaced by
    // the training column min (then max) of that column. Rows of the
    // training design are unchanged; the augmented matrix simply has
    // `2 · p_cov` extra rows appended after the original n. The resulting
    // covariate side lives in the same (already-standardized) space the
    // training design was built in — we read training stats directly out
    // of `covariate_design` and emit values in that same coordinate
    // system, so no extra transform is applied.
    let augmented_covariate_design = augment_covariate_design_with_axis_extremes(covariate_design)?;
    // Build the operator factored: the small (n_grid × p_resp) response-side
    // factor and the augmented covariate design. The implied virtual row
    // space is (n_cov + 2·p_cov) × n_grid — never materialized.
    KroneckerDesign::new_kronecker(response_deriv_grid, augmented_covariate_design)
}

/// Append `2 · p_cov` axis-extreme rows to a covariate design.
///
/// Each axis-extreme row is the training-mean covariate vector with column
/// `j` replaced by the training column min (rows 0..p_cov) or training
/// column max (rows p_cov..2·p_cov). These rows act as additional
/// monotonicity feasibility constraints in the loose-bound row set so the
/// certificate `β₁(x) + δ_k(x) ≥ ε` holds at the corners of the
/// axis-aligned predict-time box, not only at training rows.
///
/// Sparse designs are materialized through `try_to_dense_by_chunks` so the
/// per-column min / mean / max can be read; for biobank-scale dense CTN
/// (the common case) this is a single dense pass.
fn augment_covariate_design_with_axis_extremes(
    covariate_design: &DesignMatrix,
) -> Result<DesignMatrix, String> {
    let n_train = covariate_design.nrows();
    let p_cov = covariate_design.ncols();
    if n_train == 0 || p_cov == 0 {
        return Ok(covariate_design.clone());
    }
    // Per-column training mean / min / max via chunked row scan, so giant
    // sparse designs never densify in one shot.
    let mut col_sum = Array1::<f64>::zeros(p_cov);
    let mut col_min = Array1::<f64>::from_elem(p_cov, f64::INFINITY);
    let mut col_max = Array1::<f64>::from_elem(p_cov, f64::NEG_INFINITY);
    let chunk_rows = 4096usize.min(n_train.max(1));
    for start in (0..n_train).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n_train);
        let chunk = covariate_design.try_row_chunk(start..end).map_err(|e| {
            format!(
                "monotonicity grid axis-extreme augmentation: failed to read row chunk \
                 [{start},{end}): {e}"
            )
        })?;
        for j in 0..p_cov {
            let col = chunk.column(j);
            let mut sum = 0.0;
            let mut lo = col_min[j];
            let mut hi = col_max[j];
            for &v in col.iter() {
                sum += v;
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            col_sum[j] += sum;
            col_min[j] = lo;
            col_max[j] = hi;
        }
    }
    let inv_n = 1.0 / n_train as f64;
    let col_mean = col_sum.mapv(|s| s * inv_n);

    // Build the augmented dense matrix: original rows on top, axis-extreme
    // rows on the bottom. The original rows are streamed back in chunks to
    // avoid a redundant n × p_cov peak.
    let extra_rows = 2 * p_cov;
    let mut augmented = Array2::<f64>::zeros((n_train + extra_rows, p_cov));
    for start in (0..n_train).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n_train);
        let chunk = covariate_design.try_row_chunk(start..end).map_err(|e| {
            format!(
                "monotonicity grid axis-extreme augmentation: failed to copy row chunk \
                 [{start},{end}): {e}"
            )
        })?;
        augmented.slice_mut(s![start..end, ..]).assign(&chunk);
    }
    for j in 0..p_cov {
        let mut row_min = augmented.row_mut(n_train + j);
        row_min.assign(&col_mean);
        row_min[j] = col_min[j];
        let mut row_max = augmented.row_mut(n_train + p_cov + j);
        row_max.assign(&col_mean);
        row_max[j] = col_max[j];
    }
    Ok(DesignMatrix::from(augmented))
}

fn effective_response_num_internal_knots(
    config: &TransformationNormalConfig,
    n_obs: usize,
    p_cov: usize,
) -> usize {
    // I-spline contract requires K' = K − 2 ≥ 0, i.e. K ≥ 2 internal knots.
    let min_internal = 2usize;
    let sample_cap = (n_obs / 10).max(min_internal);
    let tensor_width_cap = (BASE_TRANSFORMATION_TENSOR_WIDTH + n_obs / 25)
        .min(LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH);
    let max_resp_cols_from_tensor =
        (tensor_width_cap / p_cov.max(1)).max(config.response_degree + 2);
    // One response column is the unconstrained location; the remaining columns
    // are the I-spline shape block controlled by response_num_internal_knots.
    let max_shape_cols_from_tensor = max_resp_cols_from_tensor.saturating_sub(1);
    let tensor_cap = max_shape_cols_from_tensor
        .saturating_sub(config.response_degree + 1)
        .max(min_internal);
    config
        .response_num_internal_knots
        .min(sample_cap)
        .min(tensor_cap)
        .max(min_internal)
}

// ---------------------------------------------------------------------------
// Tensor product construction
// ---------------------------------------------------------------------------

/// Row-wise Kronecker product of two matrices (same number of rows).
///
/// output\[i, j * p_b + k\] = a\[i, j\] * b\[i, k\]
fn rowwise_kronecker(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    rowwise_kronecker_views(a.view(), b.view())
}

fn rowwise_kronecker_views(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    assert_eq!(a.nrows(), b.nrows());
    let n = a.nrows();
    let pa = a.ncols();
    let pb = b.ncols();
    let mut out = Array2::<f64>::zeros((n, pa * pb));
    {
        use rayon::prelude::*;
        out.axis_chunks_iter_mut(ndarray::Axis(0), 1024)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut out_chunk)| {
                let start = chunk_idx * 1024;
                let rows = out_chunk.nrows();
                for local in 0..rows {
                    let i = start + local;
                    for j in 0..pa {
                        let a_ij = a[[i, j]];
                        if a_ij == 0.0 {
                            continue;
                        }
                        for k in 0..pb {
                            out_chunk[[local, j * pb + k]] = a_ij * b[[i, k]];
                        }
                    }
                }
            });
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

#[cfg(test)]
#[derive(Clone, Debug)]
struct KroneckerActiveSetCache {
    active_pairs: Vec<(usize, usize)>,
    min_inactive_margin: f64,
    max_response_norm: Option<f64>,
    max_covariate_norm: Option<f64>,
    c_version: u64,
    cached_for_version: u64,
    c_fingerprint: [f64; 3],
    n: usize,
    n_grid: usize,
    p_resp: usize,
    slack: f64,
    dh_eps: f64,
}

#[cfg(test)]
impl KroneckerActiveSetCache {
    fn new() -> Self {
        Self {
            active_pairs: Vec::new(),
            min_inactive_margin: f64::INFINITY,
            max_response_norm: None,
            max_covariate_norm: None,
            c_version: 0,
            cached_for_version: u64::MAX,
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
        [c[[0, 0]], c[[mid_i, mid_j]], c[[n - 1, p - 1]]]
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

fn scop_quadratic_boundary_root(
    response_row: ArrayView1<'_, f64>,
    gamma_row: ArrayView1<'_, f64>,
    dgamma_row: ArrayView1<'_, f64>,
    slack: f64,
    dh_eps: f64,
) -> f64 {
    let p_resp = response_row.len();
    debug_assert_eq!(gamma_row.len(), p_resp);
    debug_assert_eq!(dgamma_row.len(), p_resp);

    let mut a = 0.0;
    let mut b = response_row[0] * dgamma_row[0];
    let mut c = response_row[0] * gamma_row[0] - slack;
    for k in 1..p_resp {
        let r = response_row[k];
        let g = gamma_row[k];
        let d = dgamma_row[k];
        a += r * d * d;
        b += 2.0 * r * g * d;
        c += r * g * g;
    }
    if !(c.is_finite() && a.is_finite() && b.is_finite()) {
        return 0.0;
    }
    if c <= 0.0 {
        return 0.0;
    }
    if a.abs() <= dh_eps {
        if b < -dh_eps {
            let root = -c / b;
            return if root.is_finite() && root > 0.0 {
                root
            } else {
                f64::INFINITY
            };
        }
        return f64::INFINITY;
    }
    let disc = b.mul_add(b, -4.0 * a * c);
    if disc < 0.0 || !disc.is_finite() {
        return f64::INFINITY;
    }
    let sqrt_disc = disc.sqrt();
    let r1 = (-b - sqrt_disc) / (2.0 * a);
    let r2 = (-b + sqrt_disc) / (2.0 * a);
    let mut best = f64::INFINITY;
    if r1.is_finite() && r1 > 0.0 {
        best = best.min(r1);
    }
    if r2.is_finite() && r2 > 0.0 {
        best = best.min(r2);
    }
    best
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
                    ndarray::Zip::from(&mut result)
                        .and(left.rows())
                        .and(right_beta.rows())
                        .par_for_each(|r, l_row, rb_row| {
                            let mut acc = 0.0;
                            for j in 0..pa {
                                acc += l_row[j] * rb_row[j];
                            }
                            *r = acc;
                        });
                    return result;
                }
                for j in 0..pa {
                    let cov_part = right.apply(&beta_mat.row(j).to_owned());
                    ndarray::Zip::from(&mut result)
                        .and(&cov_part)
                        .and(left.column(j))
                        .par_for_each(|r, &c, &l| *r += l * c);
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

    /// SCOP-CTN forward: compute
    /// `left[i,0] · γ_0(x_i) + Σ_{k>=1} left[i,k] · γ_k(x_i)²`
    /// where `γ_k(x_i) = (right · β_mat[k, :])[i]` and
    /// `β_mat[k, j] = beta[k * p_cov + j]` (row-major reshape into
    /// `p_resp × p_cov`).
    ///
    /// Equivalent to forming `γ_mat = right · β_matᵀ` (shape `n × p_resp`),
    /// pointwise squaring columns `k>=1`, and contracting against the
    /// corresponding response basis row. The squaring is post-operator — the
    /// underlying `right` operator and the row-replicated Khatri-Rao image
    /// are never materialized. Storage cost matches `forward_mul`: an
    /// intermediate `n × p_resp` `right_beta` plus the `n` output.
    fn scop_affine_squared_forward(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let pa = left.ncols();
                let pb = right.ncols();
                let n = left.nrows();
                debug_assert_eq!(beta.len(), pa * pb);
                let beta_mat = beta.view().into_shape_with_order((pa, pb)).unwrap();
                let mut result = Array1::zeros(n);
                if let Some(right_dense) = right.as_dense_ref() {
                    // right_beta[i, k] = γ_k(x_i)
                    let right_beta = fast_ab(right_dense, &beta_mat.t().to_owned());
                    ndarray::Zip::from(&mut result)
                        .and(left.rows())
                        .and(right_beta.rows())
                        .par_for_each(|r, l_row, gamma_row| {
                            let mut acc = l_row[0] * gamma_row[0];
                            for k in 1..pa {
                                let g = gamma_row[k];
                                acc += l_row[k] * g * g;
                            }
                            *r = acc;
                        });
                    return result;
                }
                // Sparse-right fallback: materialize γ_k column-by-column.
                let mut gamma_cols = Array2::<f64>::zeros((n, pa));
                for k in 0..pa {
                    let cov_part = right.apply(&beta_mat.row(k).to_owned());
                    gamma_cols.column_mut(k).assign(&cov_part);
                }
                ndarray::Zip::from(&mut result)
                    .and(left.rows())
                    .and(gamma_cols.rows())
                    .par_for_each(|r, l_row, gamma_row| {
                        let mut acc = l_row[0] * gamma_row[0];
                        for k in 1..pa {
                            let g = gamma_row[k];
                            acc += l_row[k] * g * g;
                        }
                        *r = acc;
                    });
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
                // gamma[i, k] = γ_k(x_i) — covariate-side per response component.
                let mut gamma = Array2::<f64>::zeros((n_cov, p_resp));
                for k in 0..p_resp {
                    let cov_part = covariate.apply(&beta_mat.row(k).to_owned());
                    gamma.column_mut(k).assign(&cov_part);
                }
                // features[i,0] = γ_0(x_i); features[i,k>=1] = γ_k(x_i)².
                let mut features = gamma;
                for mut row in features.rows_mut() {
                    for k in 1..p_resp {
                        row[k] *= row[k];
                    }
                }
                // result_2d[i, g] = response_grid[g,0]γ_0(x_i)
                //               + Σ_{k>=1} response_grid[g,k]γ_k(x_i)².
                let mut result_2d = Array2::<f64>::zeros((n_cov, n_grid));
                fast_ab_into(&features, &response_grid.t(), &mut result_2d);
                result_2d
                    .into_shape_with_order((n_cov * n_grid,))
                    .expect("row-major Array2 flattens to Array1 of length n_cov*n_grid")
            }
        }
    }

    fn scop_affine_squared_min_step_to_boundary(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
        slack: f64,
        dh_eps: f64,
    ) -> f64 {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let p_resp = left.ncols();
                let p_cov = right.ncols();
                let n = left.nrows();
                debug_assert_eq!(beta.len(), p_resp * p_cov);
                debug_assert_eq!(delta.len(), p_resp * p_cov);
                let beta_mat = beta.view().into_shape_with_order((p_resp, p_cov)).unwrap();
                let delta_mat = delta.view().into_shape_with_order((p_resp, p_cov)).unwrap();
                let mut gamma = Array2::<f64>::zeros((n, p_resp));
                let mut dgamma = Array2::<f64>::zeros((n, p_resp));
                for k in 0..p_resp {
                    gamma
                        .column_mut(k)
                        .assign(&right.apply(&beta_mat.row(k).to_owned()));
                    dgamma
                        .column_mut(k)
                        .assign(&right.apply(&delta_mat.row(k).to_owned()));
                }
                use rayon::prelude::*;
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        scop_quadratic_boundary_root(
                            left.row(i),
                            gamma.row(i),
                            dgamma.row(i),
                            slack,
                            dh_eps,
                        )
                    })
                    .reduce(|| f64::INFINITY, f64::min)
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
                debug_assert_eq!(delta.len(), p_resp * p_cov);
                let beta_mat = beta.view().into_shape_with_order((p_resp, p_cov)).unwrap();
                let delta_mat = delta.view().into_shape_with_order((p_resp, p_cov)).unwrap();
                let mut gamma = Array2::<f64>::zeros((n_cov, p_resp));
                let mut dgamma = Array2::<f64>::zeros((n_cov, p_resp));
                for k in 0..p_resp {
                    gamma
                        .column_mut(k)
                        .assign(&covariate.apply(&beta_mat.row(k).to_owned()));
                    dgamma
                        .column_mut(k)
                        .assign(&covariate.apply(&delta_mat.row(k).to_owned()));
                }
                use rayon::prelude::*;
                (0..n_cov)
                    .into_par_iter()
                    .map(|i| {
                        let g_row = gamma.row(i);
                        let dg_row = dgamma.row(i);
                        let mut best = f64::INFINITY;
                        for grid_idx in 0..n_grid {
                            let root = scop_quadratic_boundary_root(
                                response_grid.row(grid_idx),
                                g_row,
                                dg_row,
                                slack,
                                dh_eps,
                            );
                            best = best.min(root);
                        }
                        best
                    })
                    .reduce(|| f64::INFINITY, f64::min)
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
    #[cfg(test)]
    fn min_step_to_boundary(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
        slack: f64,
        dh_eps: f64,
    ) -> f64 {
        match self {
            KroneckerDesign::KhatriRao { .. } => {
                let h = self.forward_mul(beta);
                let dh = self.forward_mul(delta);
                use rayon::prelude::*;
                h.as_slice()
                    .unwrap()
                    .par_iter()
                    .zip(dh.as_slice().unwrap().par_iter())
                    .map(|(&hi, &dval)| {
                        if dval < -dh_eps {
                            (hi - slack) / (-dval)
                        } else {
                            f64::INFINITY
                        }
                    })
                    .reduce(|| f64::INFINITY, f64::min)
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
                let mut cache = KroneckerActiveSetCache::new();
                Self::scan_kronecker_boundary_and_refresh_cache(
                    response_grid,
                    c.view(),
                    d.view(),
                    slack,
                    dh_eps,
                    n,
                    response_grid.nrows(),
                    p_resp,
                    &mut cache,
                )
            }
        }
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
    #[cfg(test)]
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

    /// Maximum row L2-norm of the response-grid factor (`max_g ||r_g||₂`).
    ///
    /// Returns `None` for the `KhatriRao` variant — it has no separate response
    /// factor to summarize.
    #[cfg(test)]
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
    #[cfg(test)]
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
                    use rayon::prelude::*;
                    let m = dense
                        .axis_iter(ndarray::Axis(0))
                        .into_par_iter()
                        .map(|row| {
                            let s2: f64 = row.iter().map(|v| v * v).sum();
                            s2.sqrt()
                        })
                        .reduce(|| 0.0_f64, f64::max);
                    return Some(m);
                }
                let chunk_rows =
                    crate::resource::rows_for_target_bytes(8 * 1024 * 1024, p.max(1)).max(1);
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
    #[cfg(test)]
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
            // Certificate: m_inactive - α_A · L > slack ⇒ no inactive pair can
            // bind at any α ≤ α_A.  The cached `min_inactive_margin = min
            // (h_{i,g} - slack)` over (i,g) ∉ A was already netted of `slack`
            // in `refresh_active_set_cache`, so the strict-feasibility test
            // collapses to `m_inactive > α_A · L`.
            //
            // When `α_active = +∞` (no active pair binds along `d`), the
            // Lipschitz bound `|d_{i,g}| ≤ L` gives a LOWER bound on inactive
            // hit points (they bind at α ≥ m_inactive / L) but NOT an upper
            // bound on the minimum.  So the certificate cannot prove
            // bit-equivalence with the full-grid scan in that regime — we
            // must fall through to the full scan to find the true minimum.
            // Substituting `α_for_bound = 1.0` to certify within a downstream
            // `min(1.0, ·)` cap is unprincipled because (a) it changes this
            // function's stated contract from "exact min α" to "exact min α
            // within α ≤ 1.0", and (b) the assertion in
            // `ctn_active_set_certificate_matches_full_grid_when_bound_passes`
            // will catch any drift.
            if alpha_active.is_finite() && cache.min_inactive_margin > alpha_active * lipschitz {
                log::info!(
                    "[STAGE] CTN monotonicity certificate hit |A|={} m_inactive={:.3e} \
                     α_A·L={:.3e} → α={:.3e} (skipped n·n_grid={} pair scan)",
                    cache.active_pairs.len(),
                    cache.min_inactive_margin,
                    alpha_active * lipschitz,
                    alpha_active,
                    n * n_grid,
                );
                return alpha_active;
            }
        }

        // Cache stale or certificate failed: refresh the active set in the
        // same full-grid scan that computes the exact boundary step.
        log::info!(
            "[STAGE] CTN monotonicity certificate miss (cache_fresh={}) → full grid scan \
             over {} (i,g) pairs",
            cache_fresh,
            n * n_grid,
        );
        Self::scan_kronecker_boundary_and_refresh_cache(
            response_grid,
            c,
            d,
            slack,
            dh_eps,
            n,
            n_grid,
            p_resp,
            cache,
        )
    }

    /// Scan the full `n_cov × n_grid` virtual rows, recompute the active-set
    /// cache from the current `c` projection, and return the exact boundary
    /// step for the current `d` projection.
    ///
    /// Both cached and uncached Kronecker line searches use this one streaming
    /// reducer, so the full-grid scan has a single implementation.
    ///
    /// `active_pairs` collects (i, g) with `h_{i,g} - slack ≤ τ`; the
    /// `min_inactive_margin` is the smallest `h_{i,g} - slack` over inactive
    /// pairs (= the tightest non-active constraint).
    #[cfg(test)]
    fn scan_kronecker_boundary_and_refresh_cache(
        response_grid: &Array2<f64>,
        c: ndarray::ArrayView2<'_, f64>,
        d: ndarray::ArrayView2<'_, f64>,
        slack: f64,
        dh_eps: f64,
        n: usize,
        n_grid: usize,
        p_resp: usize,
        cache: &mut KroneckerActiveSetCache,
    ) -> f64 {
        debug_assert_eq!(d.nrows(), n);
        debug_assert_eq!(d.ncols(), p_resp);
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        // Compute H = c · response_gridᵀ and dH = d · response_gridᵀ in
        // chunked passes; classify rows and reduce the boundary step.
        const CHUNK_ROWS: usize = 1024;
        // Active-set tolerance: small relative-to-slack with an additive
        // floor. The certificate is valid for any τ ≥ 0; this scale keeps
        // the active set typically ≤ 0.1% of pairs at biobank scale.
        let scale = c.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let tau = (slack.abs() * 1e-3 + 1e-12).max(dh_eps * scale);

        let r_t = response_grid.t();
        let n_chunks = n.div_ceil(CHUNK_ROWS);
        let (active_pairs, min_inactive, alpha_full): (Vec<(usize, usize)>, f64, f64) = (0
            ..n_chunks)
            .into_par_iter()
            .map(|chunk_idx: usize| {
                let start = chunk_idx * CHUNK_ROWS;
                let end = (start + CHUNK_ROWS).min(n);
                let m = end - start;
                let c_chunk = c.slice(s![start..end, ..]);
                let d_chunk = d.slice(s![start..end, ..]);
                let mut h_chunk = Array2::<f64>::zeros((m, n_grid));
                let mut dh_chunk = Array2::<f64>::zeros((m, n_grid));
                fast_ab_into(&c_chunk, &r_t, &mut h_chunk);
                fast_ab_into(&d_chunk, &r_t, &mut dh_chunk);
                let mut local_active: Vec<(usize, usize)> = Vec::new();
                let mut local_min = f64::INFINITY;
                let mut local_alpha = f64::INFINITY;
                for i_local in 0..m {
                    let i = start + i_local;
                    for g in 0..n_grid {
                        let h = h_chunk[[i_local, g]];
                        let dval = dh_chunk[[i_local, g]];
                        if dval < -dh_eps {
                            let hit = (h - slack) / (-dval);
                            if hit < local_alpha {
                                local_alpha = hit;
                            }
                        }
                        let margin = h - slack;
                        if margin <= tau {
                            local_active.push((i, g));
                        } else if margin < local_min {
                            local_min = margin;
                        }
                    }
                }
                (local_active, local_min, local_alpha)
            })
            .reduce(
                || (Vec::new(), f64::INFINITY, f64::INFINITY),
                |(mut a, am, aa), (b, bm, ba)| {
                    a.extend(b);
                    (a, am.min(bm), aa.min(ba))
                },
            );

        cache.active_pairs = active_pairs;
        cache.min_inactive_margin = min_inactive;
        cache.n = n;
        cache.n_grid = n_grid;
        cache.p_resp = p_resp;
        cache.slack = slack;
        cache.dh_eps = dh_eps;
        cache.c_fingerprint = KroneckerActiveSetCache::fingerprint_of(c);
        cache.cached_for_version = cache.c_version;
        alpha_full
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
                    ndarray::Zip::from(&mut weighted_v)
                        .and(v)
                        .and(left.column(j))
                        .par_for_each(|w, &vi, &li| *w = vi * li);
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
    let eye_cov = Array2::<f64>::eye(p_cov);
    let mut penalties = Vec::new();

    let mut shape_resp = Array2::<f64>::eye(p_resp);
    shape_resp[[0, 0]] = 0.0;

    // Covariate roughness is a latent γ prior on the squared monotone shape
    // rows. The derivative-free location row is the conditional centering
    // field itself; penalizing it by REML under-corrects broad population
    // shifts and leaves h(Y|x) calibrated only marginally instead of
    // conditionally.
    for s_cov in covariate_penalties {
        let right = match s_cov {
            PenaltyMatrix::Dense(right) => right,
            penalty @ PenaltyMatrix::Blockwise { .. } => penalty.to_dense(),
            PenaltyMatrix::KroneckerFactored { .. } => {
                return Err(
                    "transformation covariate penalties must be single-block, not already Kronecker-factored"
                        .to_string(),
                )
            }
        };
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: shape_resp.clone(),
            right,
        });
    }

    // Response penalties: S_resp_m ⊗ I_cov
    for s_resp in response_penalties {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: s_resp.clone(),
            right: eye_cov.clone(),
        });
    }

    // Double penalty: shape-row ridge only. The location row is identified by
    // the likelihood as the conditional centering field; keep it outside every
    // SCOP roughness/shrinkage penalty so population shifts can be calibrated
    // in the selected covariate span.
    if config.double_penalty {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: shape_resp,
            right: eye_cov,
        });
    }

    Ok(penalties)
}

// ---------------------------------------------------------------------------
// Warm start
// ---------------------------------------------------------------------------

/// Compute initial β so that the SCOP-CTN model starts with a positive
/// derivative and approximately centered transformed response.
///
/// SCOP is nonlinear in the shape rows: `h'=Σ M_k(y)γ_k(x)^2`. A linear joint
/// least-squares solve fits the wrong parameterization, so the warm start
/// initializes shape rows to a positive constant derivative scale and then
/// solves only the location row `b(x)` against the remaining affine target.
fn compute_warm_start(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    offset: &Array1<f64>,
    x_val_kron: &KroneckerDesign,
    x_deriv_kron: &KroneckerDesign,
    covariate_design: &DesignMatrix,
    covariate_penalties: &[PenaltyMatrix],
    p_resp: usize,
    p_cov: usize,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<Array1<f64>, String> {
    let n = response.len();
    let p_total = p_resp * p_cov;
    if p_resp < 2 {
        return Err(format!(
            "transformation warm start requires at least 2 response basis columns, got {p_resp}"
        ));
    }

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

    // Per-row affine targets for the transformation scale.
    let mut target_h = Array1::<f64>::zeros(n);
    let mut target_hp = Array1::<f64>::zeros(n);
    for i in 0..n {
        let tau = ws.scale[i].max(1e-12);
        let inv_tau = 1.0 / tau;
        target_h[i] = (response[i] - ws.location[i]) * inv_tau - offset[i];
        target_hp[i] = inv_tau;
    }

    // β-native SCOP seed. A tempting alternative is to solve a linear
    // least-squares problem for α_k(x)=γ_k(x)^2 and then project sqrt(α_k)
    // back into the covariate basis. That is not an invariant transformation:
    // squaring the projected sqrt field no longer equals the solved α field,
    // and small projection errors can explode into huge positive I-spline
    // shape terms. Instead seed the monotone shape rows directly with a
    // constant positive γ that matches the weighted average derivative target,
    // then solve only the unconstrained location row in β-space.
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err("SCOP warm start requires positive finite total weight".to_string());
    }
    let mean_target_hp = weights
        .iter()
        .zip(target_hp.iter())
        .map(|(&w, &hp)| w * hp)
        .sum::<f64>()
        / weight_sum;
    if !(mean_target_hp.is_finite() && mean_target_hp > 0.0) {
        return Err(format!(
            "SCOP warm start derivative target is not positive finite: {mean_target_hp}"
        ));
    }

    let mut beta = Array1::<f64>::zeros(p_total);
    for k in 1..p_resp {
        beta[k * p_cov] = 1.0;
    }
    let unit_shape_hp = x_deriv_kron.scop_affine_squared_forward(&beta);
    let mean_unit_shape_hp = weights
        .iter()
        .zip(unit_shape_hp.iter())
        .map(|(&w, &hp)| w * hp)
        .sum::<f64>()
        / weight_sum;
    if !(mean_unit_shape_hp.is_finite() && mean_unit_shape_hp > 0.0) {
        return Err(format!(
            "SCOP warm start unit shape derivative is not positive finite: {mean_unit_shape_hp}"
        ));
    }
    let gamma_const = (mean_target_hp / mean_unit_shape_hp).sqrt();
    if !(gamma_const.is_finite() && gamma_const > 0.0) {
        return Err(format!(
            "SCOP warm start shape scale is not positive finite: {gamma_const}"
        ));
    }
    beta.fill(0.0);
    for k in 1..p_resp {
        beta[k * p_cov] = gamma_const;
    }

    let shape_h = x_val_kron.scop_affine_squared_forward(&beta);
    let location_target = &target_h - &shape_h;
    let zero_offset = Array1::<f64>::zeros(n);
    let log_lambdas = Array1::<f64>::zeros(covariate_penalties.len());
    let location_beta = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &location_target,
        weights,
        covariate_penalties,
        &log_lambdas,
        1e-12,
    )?;
    for c in 0..p_cov {
        beta[c] = location_beta[c];
    }

    if beta.iter().any(|v| !v.is_finite()) {
        return Err("SCOP warm start produced non-finite coefficients".to_string());
    }
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
            (y - mu).abs().max(residual_floor).ln() - STANDARD_NORMAL_MEAN_LOG_ABS
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

fn transformation_calibration_feature_cols(spec: &TermCollectionSpec) -> Vec<usize> {
    let mut cols = Vec::new();
    for linear in &spec.linear_terms {
        cols.push(linear.feature_col);
    }
    for smooth in &spec.smooth_terms {
        match &smooth.basis {
            crate::smooth::SmoothBasisSpec::BSpline1D { feature_col, .. } => {
                cols.push(*feature_col);
            }
            crate::smooth::SmoothBasisSpec::ThinPlate { feature_cols, .. }
            | crate::smooth::SmoothBasisSpec::Matern { feature_cols, .. }
            | crate::smooth::SmoothBasisSpec::Duchon { feature_cols, .. }
            | crate::smooth::SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
                cols.extend(feature_cols.iter().copied());
            }
        }
    }
    cols.sort_unstable();
    cols.dedup();
    cols
}

fn build_transformation_score_calibration_design(
    data: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    feature_cols: &[usize],
    feature_center: Option<&[f64]>,
    feature_scale: Option<&[f64]>,
    rbf_centers: Option<&[Vec<f64>]>,
    rbf_bandwidth: Option<f64>,
) -> Result<(Array2<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, f64), String> {
    let n = data.nrows();
    let d = feature_cols.len();
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(
            "transformation score calibration requires positive finite weights".to_string(),
        );
    }
    let mut center = vec![0.0; d];
    let mut scale = vec![1.0; d];
    if let (Some(saved_center), Some(saved_scale)) = (feature_center, feature_scale) {
        if saved_center.len() != d || saved_scale.len() != d {
            return Err(format!(
                "transformation score calibration feature normalization mismatch: center={}, scale={}, d={d}",
                saved_center.len(),
                saved_scale.len()
            ));
        }
        center.copy_from_slice(saved_center);
        scale.copy_from_slice(saved_scale);
    } else {
        for (j, &col) in feature_cols.iter().enumerate() {
            if col >= data.ncols() {
                return Err(format!(
                    "transformation score calibration feature column {col} out of range for {} columns",
                    data.ncols()
                ));
            }
            center[j] = (0..n).map(|i| weights[i] * data[[i, col]]).sum::<f64>() / weight_sum;
            let var = (0..n)
                .map(|i| {
                    let r = data[[i, col]] - center[j];
                    weights[i] * r * r
                })
                .sum::<f64>()
                / weight_sum;
            scale[j] = var.sqrt().max(1.0e-12);
        }
    }

    let mut z = Array2::<f64>::zeros((n, d));
    for (j, &col) in feature_cols.iter().enumerate() {
        if col >= data.ncols() {
            return Err(format!(
                "transformation score calibration feature column {col} out of range for {} columns",
                data.ncols()
            ));
        }
        for i in 0..n {
            z[[i, j]] = (data[[i, col]] - center[j]) / scale[j];
        }
    }

    let centers = if let Some(saved) = rbf_centers {
        saved.to_vec()
    } else if d == 0 {
        Vec::new()
    } else {
        let k = n.min(32).max(1);
        (0..k)
            .map(|m| {
                let row = if k == 1 { 0 } else { m * (n - 1) / (k - 1) };
                z.row(row).to_vec()
            })
            .collect::<Vec<_>>()
    };
    for center_row in &centers {
        if center_row.len() != d {
            return Err(format!(
                "transformation score calibration RBF center width {} != feature width {d}",
                center_row.len()
            ));
        }
    }
    let bandwidth = rbf_bandwidth.unwrap_or_else(|| (d as f64).sqrt().max(1.0));
    if !(bandwidth.is_finite() && bandwidth > 0.0) {
        return Err(format!(
            "transformation score calibration requires positive finite RBF bandwidth, got {bandwidth}"
        ));
    }

    let p_poly = 1 + d + d * (d + 1) / 2;
    let p = p_poly + centers.len();
    let mut design = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        design[[i, 0]] = 1.0;
        for j in 0..d {
            design[[i, 1 + j]] = z[[i, j]];
        }
        let mut col = 1 + d;
        for a in 0..d {
            for b in a..d {
                design[[i, col]] = z[[i, a]] * z[[i, b]];
                col += 1;
            }
        }
        for (m, center_row) in centers.iter().enumerate() {
            let dist2 = (0..d)
                .map(|j| {
                    let r = z[[i, j]] - center_row[j];
                    r * r
                })
                .sum::<f64>();
            design[[i, p_poly + m]] = (-0.5 * dist2 / (bandwidth * bandwidth)).exp();
        }
    }
    Ok((design, center, scale, centers, bandwidth))
}

fn calibrate_transformation_scores(
    family: &TransformationNormalFamily,
    mut fit: UnifiedFitResult,
    calibration_data: ArrayView2<'_, f64>,
    covariate_spec: &TermCollectionSpec,
) -> Result<(UnifiedFitResult, TransformationScoreCalibration), String> {
    let Some(block_state) = fit.block_states.first() else {
        return Err("transformation score calibration requires one fitted block".to_string());
    };
    let p_resp = family.response_val_basis.ncols();
    let p_cov = family.covariate_design.ncols();
    let p_total = p_resp * p_cov;
    if block_state.beta.len() != p_total {
        return Err(format!(
            "transformation calibration beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
            block_state.beta.len()
        ));
    }

    let row_quantities = family.row_quantities(&block_state.beta)?;
    let h = row_quantities.h.as_ref();
    let zero_offset = Array1::<f64>::zeros(family.n_obs());
    let empty_penalties: Vec<PenaltyMatrix> = Vec::new();
    let empty_log_lambdas = Array1::<f64>::zeros(0);
    let feature_cols = transformation_calibration_feature_cols(covariate_spec);
    let (calibration_matrix, feature_center, feature_scale, rbf_centers, rbf_bandwidth) =
        build_transformation_score_calibration_design(
            calibration_data,
            family.weights.as_ref(),
            &feature_cols,
            None,
            None,
            None,
            None,
        )?;
    let calibration_design = DesignMatrix::from(calibration_matrix);
    let location_beta = solve_penalizedweighted_projection(
        &calibration_design,
        &zero_offset,
        h,
        family.weights.as_ref(),
        &empty_penalties,
        &empty_log_lambdas,
        1.0e-12,
    )?;

    let location = calibration_design.matrixvectormultiply(&location_beta);
    let centered_h = h - &location;
    let weight_sum = family.weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err("transformation calibration requires positive finite total weight".to_string());
    }
    let centered_weighted_sum = centered_h
        .iter()
        .zip(family.weights.iter())
        .map(|(&value, &weight)| weight * value)
        .sum::<f64>();
    let centered_mean = centered_weighted_sum / weight_sum;
    let centered_var = centered_h
        .iter()
        .zip(family.weights.iter())
        .map(|(&value, &weight)| weight * (value - centered_mean).powi(2))
        .sum::<f64>()
        / weight_sum;
    if !(centered_var.is_finite() && centered_var > 1.0e-24) {
        return Err(format!(
            "transformation calibration produced non-positive centered score variance {centered_var}"
        ));
    }

    let residual_floor = centered_var.sqrt() * 1.0e-6 + 1.0e-12;
    let log_scale_target = Array1::from_iter(
        centered_h
            .iter()
            .map(|&value| value.abs().max(residual_floor).ln() - STANDARD_NORMAL_MEAN_LOG_ABS),
    );
    let log_scale_beta = solve_penalizedweighted_projection(
        &calibration_design,
        &zero_offset,
        &log_scale_target,
        family.weights.as_ref(),
        &empty_penalties,
        &empty_log_lambdas,
        1.0e-12,
    )?;
    let log_scale_eta = calibration_design.matrixvectormultiply(&log_scale_beta);
    let scaled = Array1::from_iter(
        centered_h
            .iter()
            .zip(log_scale_eta.iter())
            .map(|(&value, &log_scale)| value / log_scale.exp()),
    );
    if scaled.iter().any(|value| !value.is_finite()) {
        return Err(
            "transformation calibration produced non-finite conditionally scaled scores"
                .to_string(),
        );
    }

    let global_mean = scaled
        .iter()
        .zip(family.weights.iter())
        .map(|(&value, &weight)| weight * value)
        .sum::<f64>()
        / weight_sum;
    let global_var = scaled
        .iter()
        .zip(family.weights.iter())
        .map(|(&value, &weight)| weight * (value - global_mean).powi(2))
        .sum::<f64>()
        / weight_sum;
    if !(global_var.is_finite() && global_var > 1.0e-24) {
        return Err(format!(
            "transformation calibration produced non-positive global score variance {global_var}"
        ));
    }
    let global_sd = global_var.sqrt();
    let calibrated_h = scaled.mapv(|value| (value - global_mean) / global_sd);
    if calibrated_h
        .iter()
        .any(|value| !value.is_finite() || value.abs() > TRANSFORMATION_NORMAL_H_ABS_MAX)
    {
        return Err(
            "transformation calibration produced non-finite or out-of-range scores".to_string(),
        );
    }
    let log_likelihood = row_quantities
        .h_prime
        .iter()
        .zip(log_scale_eta.iter())
        .zip(calibrated_h.iter())
        .zip(family.weights.iter())
        .map(|(((&hp, &log_scale), &z), &weight)| {
            weight * (-0.5 * z * z + hp.ln() - log_scale - global_sd.ln())
        })
        .sum::<f64>();
    if !log_likelihood.is_finite() {
        return Err("transformation calibration produced non-finite log-likelihood".to_string());
    }

    if let Some(state) = fit.block_states.first_mut() {
        state.eta = calibrated_h;
    }
    fit.log_likelihood = log_likelihood;
    fit.deviance = -2.0 * log_likelihood;
    Ok((
        fit,
        TransformationScoreCalibration {
            feature_cols,
            feature_center,
            feature_scale,
            rbf_centers,
            rbf_bandwidth,
            location_beta: location_beta.to_vec(),
            log_scale_beta: log_scale_beta.to_vec(),
            global_mean,
            global_sd,
        },
    ))
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
    ndarray::Zip::from(out.rows_mut())
        .and(x.rows())
        .and(w)
        .par_for_each(|mut o_row, x_row, &wi| {
            for j in 0..p {
                o_row[j] = x_row[j] * wi;
            }
        });
    out
}

struct TensorKroneckerPsiOperator {
    response_val_basis: Arc<Array2<f64>>,
    covariate_design: DesignMatrix,
    covariate_derivs: Vec<CustomFamilyBlockPsiDerivative>,
    covariate_first_cache: Arc<Vec<Mutex<Option<Arc<Array2<f64>>>>>>,
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
            return Ok(crate::faer_ndarray::fast_av(&deriv.x_psi, u));
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
            return Ok(crate::faer_ndarray::fast_atv(&deriv.x_psi, v));
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
                return Ok(crate::faer_ndarray::fast_av(mat, u));
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
                return Ok(crate::faer_ndarray::fast_atv(mat, v));
            }
        }
        Ok(Array1::<f64>::zeros(self.p_cov()))
    }

    fn cov_first_axis_row_chunk(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.slice(s![rows, ..]).to_owned());
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi row chunk operator for axis {axis}"
            )));
        };
        op.row_chunk_first(deriv.implicit_axis, rows)
    }

    fn cov_second_axis_row_chunk(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        let deriv_e = self.cov_deriv(axis_e)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == deriv_e.implicit_group_id
        {
            if deriv_d.implicit_axis == deriv_e.implicit_axis {
                return op.row_chunk_second_diag(deriv_d.implicit_axis, rows);
            }
            return op.row_chunk_second_cross(deriv_d.implicit_axis, deriv_e.implicit_axis, rows);
        }
        if let Some(second_rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = second_rows.get(axis_e)
        {
            if mat.nrows() == self.n_data() && mat.ncols() == self.p_cov() {
                return Ok(mat.slice(s![rows, ..]).to_owned());
            }
        }
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p_cov())))
    }

    fn lifted_row_chunk_from_cov(
        &self,
        rows: std::ops::Range<usize>,
        cov: &Array2<f64>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let n_rows = rows.end - rows.start;
        if cov.nrows() != n_rows || cov.ncols() != self.p_cov() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker covariate row chunk shape {}x{} != expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                n_rows,
                self.p_cov()
            )));
        }
        let resp = self.response_val_basis.slice(s![rows, ..]);
        Ok(rowwise_kronecker_views(resp, cov.view()))
    }

    fn materialize_cov_first_axis_uncached(
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

    fn materialize_cov_first_axis_arc(
        &self,
        axis: usize,
    ) -> Result<Arc<Array2<f64>>, crate::terms::basis::BasisError> {
        if axis >= self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker psi axis {axis} out of bounds for {} axes",
                self.covariate_derivs.len()
            )));
        }
        let axis_cache = &self.covariate_first_cache[axis];
        let mut cache = axis_cache.lock().map_err(|_| {
            crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker covariate first-derivative cache mutex poisoned for axis {axis}"
            ))
        })?;
        if let Some(cached) = cache.as_ref() {
            return Ok(cached.clone());
        }

        let materialized = Arc::new(self.materialize_cov_first_axis_uncached(axis)?);
        *cache = Some(materialized.clone());
        Ok(materialized)
    }

    fn materialize_cov_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let mut unit = Array1::<f64>::zeros(self.covariate_derivs.len());
        unit[axis] = 1.0;
        self.materialize_cov_directional(&unit.view())
    }

    /// Per-axis covariate first-derivative materialization for axis `axis`,
    /// equivalent to the unit-vector dispatch through
    /// [`materialize_cov_directional`].
    fn materialize_cov_first_axis(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok((*self.materialize_cov_first_axis_arc(axis)?).clone())
    }

    /// Directional `Σ_j v_psi[j] · ∂C/∂ψ_j` returning an `n × p_cov` matrix.
    /// Calling with `v_psi = e_axis` matches [`materialize_cov_first_axis`] at axis.
    /// Used by the directional outer-HVP path to compute `cov_v` once per HVP
    /// instead of materializing each per-axis cov_j.
    fn materialize_cov_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        if v_psi.len() != self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vector length {} does not match {} ψ axes",
                v_psi.len(),
                self.covariate_derivs.len()
            )));
        }
        let mut out = Array2::<f64>::zeros((self.n_data(), self.p_cov()));
        for (axis, &coef) in v_psi.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            let contrib = self.materialize_cov_first_axis(axis)?;
            out.scaled_add(coef, &contrib);
        }
        Ok(out)
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
            ndarray::Zip::from(&mut out)
                .and(&cov_part)
                .and(resp_basis.column(j))
                .par_for_each(|o, &c, &r| *o += r * c);
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
            ndarray::Zip::from(&mut weighted_v)
                .and(resp_basis.column(j))
                .and(v)
                .par_for_each(|w, &r, &vi| *w = r * vi);
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
            ndarray::Zip::from(&mut out)
                .and(&cov_part)
                .and(resp_basis.column(j))
                .par_for_each(|o, &c, &r| *o += r * c);
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
            ndarray::Zip::from(&mut weighted_v)
                .and(resp_basis.column(j))
                .and(v)
                .par_for_each(|w, &r, &vi| *w = r * vi);
            let cov_block = self.cov_transpose_second(axis_d, axis_e, &weighted_v.view())?;
            out.slice_mut(s![j * p_cov..(j + 1) * p_cov])
                .assign(&cov_block);
        }
        Ok(out)
    }

    fn materialize_lifted(&self, resp_basis: &Array2<f64>, cov: &Array2<f64>) -> Array2<f64> {
        rowwise_kronecker(resp_basis, cov)
    }

    /// Internal directional accumulator on a chosen response basis:
    /// returns `Σ_j v_psi[j] · lifted_forward(resp_basis, j, β)`.
    ///
    /// Skips axes with `v_psi[j] == 0`, so calls with `v_psi = e_k` are
    /// numerically equivalent to a single direct `lifted_forward(_, k, β)` call
    /// (no extra n-vector accumulation, no rounding).
    fn lifted_forward_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if v_psi.len() != self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vector length {} does not match {} ψ axes",
                v_psi.len(),
                self.covariate_derivs.len()
            )));
        }
        let mut out = Array1::<f64>::zeros(self.n_data());
        for (axis, &coef) in v_psi.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            let contrib = self.lifted_forward(resp_basis, axis, beta)?;
            out.scaled_add(coef, &contrib);
        }
        Ok(out)
    }

    /// Internal directional accumulator on a chosen response basis:
    /// returns `Σ_j v_psi[j] · lifted_transpose(resp_basis, j, residual)`.
    ///
    /// Returns a single `(p_resp · p_cov)`-vector, NOT a stack indexed by axis.
    fn lifted_transpose_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if v_psi.len() != self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vector length {} does not match {} ψ axes",
                v_psi.len(),
                self.covariate_derivs.len()
            )));
        }
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for (axis, &coef) in v_psi.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            let contrib = self.lifted_transpose(resp_basis, axis, residual)?;
            out.scaled_add(coef, &contrib);
        }
        Ok(out)
    }

    /// Internal bilinear directional accumulator on a chosen response basis:
    /// returns `Σ_{j,k} v_psi[j] · w_psi[k] · lifted_transpose_second(resp_basis, j, k, residual)`.
    /// Mirror of [`lifted_forward_second_directional`] for the transpose direction.
    fn lifted_transpose_second_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        if v_psi.len() != q || w_psi.len() != q {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vectors length ({}, {}) do not match {} ψ axes",
                v_psi.len(),
                w_psi.len(),
                q
            )));
        }
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for j in 0..q {
            for k in j..q {
                let coef = if j == k {
                    v_psi[j] * w_psi[j]
                } else {
                    v_psi[j] * w_psi[k] + v_psi[k] * w_psi[j]
                };
                if coef == 0.0 {
                    continue;
                }
                let contrib = self.lifted_transpose_second(resp_basis, j, k, residual)?;
                out.scaled_add(coef, &contrib);
            }
        }
        Ok(out)
    }

    /// Internal bilinear directional accumulator on a chosen response basis:
    /// returns `Σ_{j,k} v_psi[j] · w_psi[k] · lifted_forward_second(resp_basis, j, k, β)`.
    fn lifted_forward_second_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        if v_psi.len() != q || w_psi.len() != q {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vectors length ({}, {}) do not match {} ψ axes",
                v_psi.len(),
                w_psi.len(),
                q
            )));
        }
        let mut out = Array1::<f64>::zeros(self.n_data());
        for j in 0..q {
            for k in j..q {
                let coef = if j == k {
                    v_psi[j] * w_psi[j]
                } else {
                    v_psi[j] * w_psi[k] + v_psi[k] * w_psi[j]
                };
                if coef == 0.0 {
                    continue;
                }
                let contrib = self.lifted_forward_second(resp_basis, j, k, beta)?;
                out.scaled_add(coef, &contrib);
            }
        }
        Ok(out)
    }

    /// Directional `Σ_j v_psi[j] · ∂(X · β)/∂ψ_j` on the value response basis.
    /// Calling with `v_psi = e_k` returns the same n-vector as
    /// [`forward_mul`](Self::forward_mul) at axis `k` (zero entries skipped).
    fn forward_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_forward_directional(&resp_basis, v_psi, beta)
    }

    /// Directional transpose against the value response basis.
    /// Calling with `v_psi = e_k` matches the per-axis `transpose_mul(k, residual)`
    /// surface on the trait.
    fn transpose_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_transpose_directional(&resp_basis, v_psi, residual)
    }

    /// Bilinear directional `Σ_{j,k} v_psi[j] · w_psi[k] · ∂²(X · β)/∂ψ_j∂ψ_k`
    /// on the value response basis. With `v_psi = e_a, w_psi = e_b` matches
    /// `forward_mul_second_diag(a)` (when a==b) or `forward_mul_second_cross(a,b)`.
    fn forward_second_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_forward_second_directional(&resp_basis, v_psi, w_psi, beta)
    }

    /// Bilinear directional second-order transpose on the value response basis.
    /// With `v_psi = e_a, w_psi = e_b` matches the per-axis-pair
    /// `transpose_mul_second_diag(a)` (when a==b) or `transpose_mul_second_cross(a,b)`.
    fn transpose_second_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_transpose_second_directional(&resp_basis, v_psi, w_psi, residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_matrix_derivativefd;
    use ndarray::array;

    #[test]
    fn ctn_penalty_scale_seed_uses_likelihood_to_penalty_ratio() {
        let likelihood_gram = array![[8.0, 0.0], [0.0, 8.0]];
        let penalties = vec![
            PenaltyMatrix::Dense(array![[2.0, 0.0], [0.0, 2.0]]),
            PenaltyMatrix::Dense(array![[4.0, 0.0], [0.0, 4.0]]),
        ];
        let rho = ctn_penalty_scale_log_lambdas(&penalties, &likelihood_gram);
        assert!((rho[0] - 4.0_f64.ln()).abs() < 1.0e-12);
        assert!((rho[1] - 2.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn transformation_score_calibration_allows_intercept_only_design() {
        let data = array![[0.0], [1.0], [2.0]];
        let weights = Array1::ones(data.nrows());
        let (design, center, scale, rbf_centers, bandwidth) =
            build_transformation_score_calibration_design(
                data.view(),
                &weights,
                &[],
                None,
                None,
                None,
                None,
            )
            .expect("intercept-only calibration design");

        assert_eq!(design.dim(), (data.nrows(), 1));
        assert!(design.column(0).iter().all(|&value| value == 1.0));
        assert!(center.is_empty());
        assert!(scale.is_empty());
        assert!(rbf_centers.is_empty());
        assert_eq!(bandwidth, 1.0);
    }

    #[test]
    fn tensor_psi_penalty_derivatives_follow_shape_only_scop_layout() {
        let response = array![-1.0, -0.2, 0.6, 1.3];
        let (val_basis, deriv_basis, knots, transform, p_resp) = toy_response_basis(&response);
        let weights = Array1::from_elem(response.len(), 1.0);
        let offset = Array1::zeros(response.len());
        let cov_design = array![[1.0, 0.2], [1.0, -0.1], [1.0, 0.4], [1.0, -0.3]];
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_design.clone())),
            vec![],
            &toy_scop_ctn_config(),
            None,
        )
        .expect("toy transformation family");

        let ds0 = array![[1.0, 0.25], [0.25, 2.0]];
        let ds1 = array![[3.0, -0.5], [-0.5, 4.0]];
        let ds1_second = array![[5.0, 0.75], [0.75, 6.0]];
        let mut cov_deriv = CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::zeros((response.len(), cov_design.ncols())),
            Array2::zeros((0, 0)),
            Some(vec![(0, ds0.clone()), (1, ds1.clone())]),
            None,
            None,
            Some(vec![vec![(1, ds1_second.clone())]]),
        );
        cov_deriv.s_psi_penalty_components = Some(vec![
            (0, PenaltyMatrix::Dense(ds0.clone())),
            (1, PenaltyMatrix::Dense(ds1.clone())),
        ]);
        cov_deriv.s_psi_psi_penalty_components =
            Some(vec![vec![(1, PenaltyMatrix::Dense(ds1_second.clone()))]]);

        let tensor_derivs =
            build_tensor_psi_derivatives(&family, &[cov_deriv]).expect("tensor derivatives");
        let first = tensor_derivs[0]
            .s_psi_penalty_components
            .as_ref()
            .expect("first derivatives");
        let got_indices: Vec<usize> = first.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(got_indices, vec![0, 1]);
        assert_shape_penalty_component(&first[0].1, p_resp, &ds0);
        assert_shape_penalty_component(&first[1].1, p_resp, &ds1);

        let second = tensor_derivs[0]
            .s_psi_psi_penalty_components
            .as_ref()
            .expect("second derivatives");
        assert_eq!(second.len(), 1);
        let got_second_indices: Vec<usize> = second[0].iter().map(|(idx, _)| *idx).collect();
        assert_eq!(got_second_indices, vec![1]);
        assert_shape_penalty_component(&second[0][0].1, p_resp, &ds1_second);
    }

    #[test]
    fn tensor_psi_row_chunks_are_window_consistent() {
        let response = array![-1.0, -0.2, 0.6, 1.3];
        let (val_basis, deriv_basis, knots, transform, _) = toy_response_basis(&response);
        let psi = array![0.15, -0.10];
        let (cov_design, cov_derivs) = toy_covariate_design_and_derivs(&psi);
        let weights = Array1::from_elem(response.len(), 1.0);
        let offset = Array1::zeros(response.len());
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
            vec![],
            &toy_scop_ctn_config(),
            None,
        )
        .expect("toy transformation family");

        let tensor_derivs =
            build_tensor_psi_derivatives(&family, &cov_derivs).expect("tensor derivatives");
        let op = tensor_derivs[0]
            .implicit_operator
            .as_ref()
            .expect("tensor psi operator should be implicit");
        let mat_op = op
            .as_materializable()
            .expect("toy tensor psi operator should remain materializable for reference");
        let rows = 1..3;

        let first_dense = mat_op
            .materialize_first(0)
            .expect("dense first derivative reference");
        let first_chunk = op
            .row_chunk_first(0, rows.clone())
            .expect("chunked first derivative");
        assert_eq!(
            first_chunk,
            first_dense.slice(s![rows.clone(), ..]).to_owned()
        );

        let second_diag_full = op
            .row_chunk_second_diag(0, 0..op.n_data())
            .expect("full row-chunk second diagonal reference");
        let second_diag_chunk = op
            .row_chunk_second_diag(0, rows.clone())
            .expect("chunked second diagonal derivative");
        assert_eq!(
            second_diag_chunk,
            second_diag_full.slice(s![rows.clone(), ..]).to_owned()
        );

        let second_cross_full = op
            .row_chunk_second_cross(0, 1, 0..op.n_data())
            .expect("full row-chunk second cross reference");
        let second_cross_chunk = op
            .row_chunk_second_cross(0, 1, rows.clone())
            .expect("chunked second cross derivative");
        assert_eq!(
            second_cross_chunk,
            second_cross_full.slice(s![rows, ..]).to_owned()
        );
    }

    fn assert_shape_penalty_component(
        penalty: &PenaltyMatrix,
        p_resp: usize,
        expected_right: &Array2<f64>,
    ) {
        let PenaltyMatrix::KroneckerFactored { left, right } = penalty else {
            panic!("expected KroneckerFactored penalty component");
        };
        assert_eq!(right, expected_right);
        assert_eq!(left.nrows(), p_resp);
        assert_eq!(left.ncols(), p_resp);
        for r in 0..p_resp {
            for c in 0..p_resp {
                let expected = if r == c && r > 0 { 1.0 } else { 0.0 };
                assert_eq!(left[[r, c]], expected);
            }
        }
    }

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

    /// Minimal SCOP-CTN config used by every toy fixture in this test module:
    /// degree-1 I-splines on 2 internal knots produce the smallest valid
    /// SCOP-CTN configuration (p_resp = 4 monotone basis columns).
    fn toy_scop_ctn_config() -> TransformationNormalConfig {
        TransformationNormalConfig {
            double_penalty: false,
            response_degree: 1,
            response_num_internal_knots: 2,
            ..TransformationNormalConfig::default()
        }
    }

    /// Build (val, deriv, knots, transform, p_resp) from a real
    /// `build_response_basis` call so test fixtures match the production
    /// I-spline contract exactly.
    fn toy_response_basis(
        response: &Array1<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array2<f64>, usize) {
        let config = toy_scop_ctn_config();
        let (val, deriv, _penalties, knots, transform) =
            build_response_basis(response, &config).expect("toy response basis builds");
        let p_resp = val.ncols();
        (val, deriv, knots, transform, p_resp)
    }

    /// Deterministic probe vector of length `p_total` used by tests that
    /// previously hand-rolled p_total=4 arrays. Generated from a tiny PRNG so
    /// each call with a different seed yields linearly-independent probes.
    fn toy_probe_vector(p_total: usize, seed: u64) -> Array1<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        Array1::from_iter((0..p_total).map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 11) as f64 / (1u64 << 53) as f64;
            (bits - 0.5) * 0.8
        }))
    }

    fn toy_family_and_derivatives(
        psi: &Array1<f64>,
    ) -> (
        TransformationNormalFamily,
        Vec<Vec<CustomFamilyBlockPsiDerivative>>,
        ParameterBlockState,
        ParameterBlockSpec,
    ) {
        let response = array![-1.0, -0.2, 0.6, 1.3];
        let (val_basis, deriv_basis, knots, transform, p_resp) = toy_response_basis(&response);
        let weights = Array1::from_elem(response.len(), 1.0);
        let offset = Array1::zeros(response.len());
        let (cov_design, cov_derivs) = toy_covariate_design_and_derivs(psi);
        let p_cov = cov_design.ncols();
        let p_total = p_resp * p_cov;
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
            vec![],
            &toy_scop_ctn_config(),
            None,
        )
        .expect("toy transformation family");
        let derivative_blocks =
            vec![build_tensor_psi_derivatives(&family, &cov_derivs).expect("tensor psi derivs")];
        // Positive γ across the response axis with mild covariate variation so
        // h' = (M ⊗_row B_cov)·β stays strictly positive on every row (M-splines
        // are non-negative; the toy covariate design is positive-valued).
        let mut beta_vec = Vec::with_capacity(p_total);
        for k in 0..p_resp {
            let base = 0.6 + 0.05 * k as f64;
            for j in 0..p_cov {
                if j == 0 {
                    beta_vec.push(base);
                } else {
                    beta_vec.push(0.05 + 0.02 * k as f64 * (j as f64));
                }
            }
        }
        let beta = Array1::from(beta_vec);
        assert_eq!(beta.len(), p_total);
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
    fn ctn_row_quantity_cache_matches_direct_formulas() {
        let psi = array![0.15, -0.10];
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        let row = family
            .row_quantities(&state.beta)
            .expect("toy row quantities");
        // SCOP-CTN forward: h = X_val · γ²-affine + offset + ε(y−median),
        // h' = X_deriv · γ²-affine + ε.
        let direct_h = family.x_val_kron.scop_affine_squared_forward(&state.beta)
            + family.offset.as_ref()
            + family.response_floor_offset.as_ref();
        let direct_h_prime = family
            .x_deriv_kron
            .scop_affine_squared_forward(&state.beta)
            .mapv(|hp| hp + TRANSFORMATION_MONOTONICITY_EPS);
        let weights = family.weights.as_ref();

        for i in 0..direct_h.len() {
            assert!(
                (row.h[i] - direct_h[i]).abs() <= 1.0e-14,
                "h[{i}] mismatch: cached={} direct={}",
                row.h[i],
                direct_h[i]
            );
            assert!(
                (row.h_prime[i] - direct_h_prime[i]).abs() <= 1.0e-14,
                "h_prime[{i}] mismatch: cached={} direct={}",
                row.h_prime[i],
                direct_h_prime[i]
            );
        }

        let p_resp = family.response_val_basis.ncols();
        let p_cov = family.covariate_design.ncols();
        let beta_mat = state
            .beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .expect("toy beta reshape");
        let cov = family
            .covariate_design
            .try_row_chunk(0..family.n_obs())
            .expect("toy covariate rows");
        let (h_lower, h_upper) = family
            .scop_endpoint_values(&state.beta, beta_mat, cov.view())
            .expect("toy endpoint values");

        let mut expected_ll = 0.0;
        for i in 0..direct_h.len() {
            let hp = direct_h_prime[i];
            let log_z = log_normal_cdf_diff(h_upper[i], h_lower[i]).expect("endpoint mass");
            expected_ll += weights[i] * (-0.5 * direct_h[i] * direct_h[i] + hp.ln() - log_z);
        }

        assert!(
            (row.log_likelihood - expected_ll).abs() <= 1.0e-14,
            "cached log-likelihood={} expected={expected_ll}",
            row.log_likelihood
        );
    }

    #[test]
    fn ctn_endpoint_normalizer_derivatives_are_finite_in_positive_tail() {
        let q =
            log_normal_cdf_diff_derivatives(38.0, 37.0).expect("positive-tail endpoint normalizer");
        assert!(q.first[0].is_finite());
        assert!(q.first[1].is_finite());
        assert!(q.second[0][0].is_finite());
        assert!(q.third[0][0][0].is_finite());
        assert!(q.fourth[0][0][0][0].is_finite());
        assert!(q.first[0] > 0.0);
        assert!(q.first[1] < 0.0);
    }

    #[test]
    fn ctn_row_quantity_cache_is_exact_beta_keyed() {
        let psi = array![0.15, -0.10];
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        let row_a = family
            .row_quantities(&state.beta)
            .expect("first row quantity build");
        let row_a_again = family
            .row_quantities(&state.beta)
            .expect("same beta row quantity lookup");
        assert!(Arc::ptr_eq(&row_a.h, &row_a_again.h));
        assert!(Arc::ptr_eq(&row_a.h_prime, &row_a_again.h_prime));

        let mut beta_b = state.beta.clone();
        beta_b[0] += 0.125;
        let row_b = family
            .row_quantities(&beta_b)
            .expect("updated beta row quantity build");
        assert!(!Arc::ptr_eq(&row_a.h, &row_b.h));
        assert!(row_b.matches_beta(&beta_b));
        assert!(!row_b.matches_beta(&state.beta));
        assert!(
            row_a
                .h
                .iter()
                .zip(row_b.h.iter())
                .any(|(&left, &right)| left.to_bits() != right.to_bits())
        );

        let row_b_again = family
            .row_quantities(&beta_b)
            .expect("updated beta row quantity lookup");
        assert!(Arc::ptr_eq(&row_b.h, &row_b_again.h));
    }

    #[test]
    fn ctn_row_quantities_reject_nonrepresentable_exact_derivatives() {
        let h = array![0.0];
        let h_prime = array![1.0e-100];
        let h_lower = array![-8.0];
        let h_upper = array![8.0];
        let weights = array![1.0];
        let err = build_transformation_row_derived(&h, &h_prime, &h_lower, &h_upper, &weights)
            .expect_err("1/h'^4 overflows f64 and must not be clamped");
        assert!(
            err.contains("1/h'^4") && err.contains("outside the finite exact-derivative range"),
            "unexpected error: {err}"
        );
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
        // `max_feasible_step_size` projects β and δ through the covariate
        // factor once and routes the monotonicity-grid reduction through
        // `min_step_to_boundary_with_active_set`; this test confirms the
        // wired path matches the un-cached baseline exactly on a small CTN
        // configuration (toy family with a `Kronecker` grid design).
        let psi = array![0.15, -0.10];
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        // Sanity: the toy fixture really uses the Kronecker grid variant the
        // cached path was built for; if anything ever rewires this design to
        // KhatriRao the production caller must keep working via fallback, so
        // the assertion documents the precondition rather than gating it.
        assert!(
            matches!(family.x_deriv_grid_kron, KroneckerDesign::Kronecker { .. }),
            "toy family must keep the Kronecker grid variant for cached-path coverage"
        );
        // δ direction with a negative leading derivative contribution. The
        // structural ε floor means the production guard binds only if the
        // learned derivative part would drift below −ε.
        let p_total = state.beta.len();
        let mut delta = toy_probe_vector(p_total, 0xDE17A);
        delta[0] = -0.30;

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
        let alpha_obs_uncached = family.x_deriv_kron.min_step_to_boundary(
            beta,
            &delta,
            -TRANSFORMATION_MONOTONICITY_EPS,
            1e-14,
        );
        let alpha_grid_uncached = family.x_deriv_grid_kron.min_step_to_boundary(
            beta,
            &delta,
            -TRANSFORMATION_MONOTONICITY_EPS,
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
    fn warm_start_absorbs_offset_into_affine_seed() {
        // The SCOP squared-γ warm start is built directly in β-space: choose a
        // positive constant shape seed for h', subtract its induced value
        // contribution, then solve the unconstrained location row. The fixed
        // monotonicity floor is part of h, so the value target includes
        // ε(y-median) and the derivative target includes ε.
        let response = array![2.0, 3.0, 4.0, 5.0];
        let (val_basis, deriv_basis, knots, transform, _p_resp) = toy_response_basis(&response);
        let weights = Array1::from_elem(response.len(), 1.0);
        let offset = Array1::from_elem(response.len(), 0.7);
        let cov_rows = response.len();
        let covariate_design = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem(
            (cov_rows, 1),
            1.0,
        )));
        let warm_start = TransformationWarmStart {
            location: Array1::from_elem(response.len(), 1.0),
            scale: Array1::from_elem(response.len(), 2.0),
        };
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            covariate_design,
            vec![],
            &toy_scop_ctn_config(),
            Some(&warm_start),
        )
        .expect("transformation family");

        let row = family
            .row_quantities(&family.initial_beta)
            .expect("row quantities at initial beta");
        let h = row.h.as_ref();
        let h_prime = row.h_prime.as_ref();
        // expected_h[i] = (response[i] - location)/scale = (y - 1)/2.
        let expected_h: Array1<f64> = response.mapv(|y| {
            (y - 1.0) / 2.0 + TRANSFORMATION_MONOTONICITY_EPS * (y - family.response_median())
        });
        let expected_h_prime =
            Array1::from_elem(response.len(), 0.5 + TRANSFORMATION_MONOTONICITY_EPS);

        for i in 0..expected_h.len() {
            assert!(
                (h[i] - expected_h[i]).abs() < 1e-9,
                "h[{i}] mismatch: got {}, expected {}",
                h[i],
                expected_h[i]
            );
            assert!(
                (h_prime[i] - expected_h_prime[i]).abs() < 1e-9,
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
        // The CTN psi-psi second-order kernel exposes its dense p_total×p_total
        // block through `hessian_psi_psi` when the family materializes it
        // eagerly, or through an operator-backed `hessian_psi_psi_operator`
        // when the family stages the Hessian as HVPs. The FD comparison needs
        // the dense matrix either way, so materialize the operator on demand.
        let analytic_hessian = if analytic.hessian_psi_psi.nrows() > 0 {
            analytic.hessian_psi_psi.clone()
        } else {
            analytic
                .hessian_psi_psi_operator
                .as_ref()
                .expect("CTN psi-psi must expose either dense Hessian or operator")
                .to_dense()
        };
        assert_matrix_derivativefd(
            &hess_fd,
            &analytic_hessian,
            2e-4,
            "transformation normal psi second-order Hessian",
        );
    }

    #[test]
    fn transformation_normal_joint_psi_first_order_matches_normalized_loglik_fd() {
        let psi = array![0.15, -0.10];
        let h = 1e-6;
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let beta = state.beta.clone();
        let states = vec![state.clone()];
        let specs = vec![spec];

        let analytic = family
            .exact_newton_joint_psi_terms(&states, &specs, &derivative_blocks, 0)
            .expect("analytic psi first-order terms")
            .expect("first-order terms should be present");

        let eval_negative_loglik = |psi_eval: &Array1<f64>| {
            let (f_eval, _, mut state_eval, _) = toy_family_and_derivatives(psi_eval);
            state_eval.beta = beta.clone();
            -f_eval
                .log_likelihood_only(std::slice::from_ref(&state_eval))
                .expect("log-likelihood at perturbed psi")
        };

        let mut psi_plus = psi.clone();
        psi_plus[0] += h;
        let mut psi_minus = psi.clone();
        psi_minus[0] -= h;
        let fd = (eval_negative_loglik(&psi_plus) - eval_negative_loglik(&psi_minus)) / (2.0 * h);

        assert!(
            (analytic.objective_psi - fd).abs() < 1e-6,
            "normalized CTN psi objective mismatch: analytic={}, fd={fd}",
            analytic.objective_psi
        );
    }

    #[test]
    fn transformation_normal_joint_psi_second_order_terms_are_operator_backed() {
        let psi = array![0.15, -0.10];
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let states = vec![state.clone()];
        let specs = vec![spec];

        let terms = family
            .exact_newton_joint_psisecond_order_terms(&states, &specs, &derivative_blocks, 0, 1)
            .expect("analytic psi second-order terms")
            .expect("psi second-order terms should be present");

        assert_eq!(terms.hessian_psi_psi.nrows(), 0);
        assert_eq!(terms.hessian_psi_psi.ncols(), 0);
        let op = terms
            .hessian_psi_psi_operator
            .as_ref()
            .expect("CTN psi-psi Hessian must be operator-backed");
        assert!(op.is_implicit());
        let p = state.beta.len();
        assert_eq!(op.dim(), p);
        assert!(op.has_fast_bilinear_view());

        let dense = op.to_dense();
        assert_eq!(dense.nrows(), p);
        assert_eq!(dense.ncols(), p);

        let v = toy_probe_vector(p, 901);
        let got_vec = op.mul_vec(&v);
        let want_vec = dense.dot(&v);
        for i in 0..p {
            let tol = 1e-10 * want_vec[i].abs().max(1.0) + 1e-10;
            assert!(
                (got_vec[i] - want_vec[i]).abs() <= tol,
                "psi-psi operator matvec mismatch at {i}: got={:.6e}, want={:.6e}",
                got_vec[i],
                want_vec[i]
            );
        }

        let mut factor = Array2::<f64>::zeros((p, 3));
        for (col, seed) in [902_u64, 903, 904].into_iter().enumerate() {
            factor.column_mut(col).assign(&toy_probe_vector(p, seed));
        }
        let got_mat = op.mul_mat(&factor);
        let want_mat = dense.dot(&factor);
        for row in 0..p {
            for col in 0..factor.ncols() {
                let tol = 1e-10 * want_mat[[row, col]].abs().max(1.0) + 1e-10;
                assert!(
                    (got_mat[[row, col]] - want_mat[[row, col]]).abs() <= tol,
                    "psi-psi operator mul_mat mismatch at ({row}, {col}): got={:.6e}, want={:.6e}",
                    got_mat[[row, col]],
                    want_mat[[row, col]]
                );
            }
        }

        let left = toy_probe_vector(p, 905);
        let right = toy_probe_vector(p, 906);
        let got_bilinear = op.bilinear_view(left.view(), right.view());
        let want_bilinear = right.dot(&dense.dot(&left));
        let tol = 1e-10 * want_bilinear.abs().max(1.0) + 1e-10;
        assert!(
            (got_bilinear - want_bilinear).abs() <= tol,
            "psi-psi operator bilinear mismatch: got={:.6e}, want={:.6e}",
            got_bilinear,
            want_bilinear
        );
    }

    #[test]
    fn transformation_normal_joint_psihessian_directional_derivative_matches_fd() {
        let psi = array![0.15, -0.10];
        let h = 1e-6;
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let direction = toy_probe_vector(spec.design.ncols(), 701);
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
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        let p = state.beta.len();
        let dir_u = toy_probe_vector(p, 801);
        let dir_v = toy_probe_vector(p, 802);

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
            toy_probe_vector(p, 101),
            toy_probe_vector(p, 102),
            toy_probe_vector(p, 103),
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
        // p_cov=256 keeps p_total = p_resp · p_cov ≥ JOINT_MATRIX_FREE_MIN_DIM
        // so matrix-free is ALWAYS active for any n. The override must report
        // the per-Hv matvec cost n·(p_resp + p_cov), not the dense p² gram.
        // n=8 keeps the test allocation small (~16 KB for covariate_design).
        let n = 8usize;
        let p_cov = 256usize;
        let response = Array1::from_iter((0..n).map(|i| (i as f64) / (n - 1) as f64));
        let (val_basis, deriv_basis, knots, transform, _p_resp) = toy_response_basis(&response);
        let weights = Array1::from_elem(n, 1.0);
        let offset = Array1::zeros(n);
        // Non-degenerate covariate design: small column-wise variation makes
        // the joint warm-start solve well-posed without changing the
        // matrix-free gating behavior tested below.
        let mut cov_design = Array2::<f64>::zeros((n, p_cov));
        for i in 0..n {
            for j in 0..p_cov {
                cov_design[[i, j]] = 0.1 + 0.01 * (i as f64) + 0.001 * (j as f64);
            }
        }
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
            vec![],
            &toy_scop_ctn_config(),
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
    fn ctn_inner_and_outer_hvp_capabilities_are_advertised() {
        let psi = array![0.15, -0.10];
        let (family, derivative_blocks, _, spec) = toy_family_and_derivatives(&psi);
        let specs = std::slice::from_ref(&spec);

        assert!(
            !CTN_ALLOW_TAU_TAU_GRADIENT_ONLY_PREFERENCE,
            "SCOP-CTN must keep analytic Hessian planning available to ARC; \
             dense tau-tau policy must not downgrade it to gradient-only BFGS"
        );
        assert!(family.inner_coefficient_hessian_hvp_available(specs));
        assert!(family.outer_hyper_hessian_hvp_available(specs));
        assert!(family.outer_hyper_hessian_dense_available(specs));
        assert_eq!(
            family.exact_outer_derivative_order(specs, &BlockwiseFitOptions::default()),
            crate::custom_family::ExactOuterDerivativeOrder::Second
        );

        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            use_outer_hessian: true,
            ..BlockwiseFitOptions::default()
        };
        let (gradient, hessian) = custom_family_outer_derivatives(&family, specs, &options);
        assert_eq!(
            gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
        assert_eq!(hessian, crate::solver::outer_strategy::Derivative::Analytic);

        let rho_dim = spec.initial_log_lambdas.len();
        let psi_dim = derivative_blocks[0].len();
        let outer_plan =
            crate::solver::outer_strategy::plan(&crate::solver::outer_strategy::OuterCapability {
                gradient,
                hessian,
                n_params: rho_dim + psi_dim,
                psi_dim,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: true,
            });
        assert_eq!(
            outer_plan.solver,
            crate::solver::outer_strategy::Solver::Arc
        );
        assert_eq!(
            outer_plan.hessian_source,
            crate::solver::outer_strategy::HessianSource::Analytic
        );
    }

    #[test]
    fn ctn_joint_hessian_workspace_dh_operator_matches_dense() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let d_beta = toy_probe_vector(p, 201);
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
            toy_probe_vector(p, 202),
            toy_probe_vector(p, 203),
            toy_probe_vector(p, 204),
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
        let dir_u = toy_probe_vector(p, 301);
        let dir_v = toy_probe_vector(p, 302);

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
            toy_probe_vector(p, 303),
            toy_probe_vector(p, 304),
            toy_probe_vector(p, 305),
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

    /// FD check for the cached CTN barrier dH operator (third-derivative formula
    /// `D(∇²B)[u]v = -2 μ Dᵀ((Du)(Dv)/c³)`).
    ///
    /// At fixed direction `d_beta`, builds `H(β ± ε d_beta) v` matrix-free via
    /// `apply_hessian` and checks that the centered FD quotient converges to the
    /// operator's `mul_vec(v)`. This locks in both the analytic formula and the
    /// `inv_hp_cu` cache (a stale cache would only show up under ε perturbation,
    /// not in the dense-equivalence test that probes a single iterate).
    #[test]
    fn ctn_dh_operator_matches_fd_under_beta_perturbation() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let d_beta = toy_probe_vector(p, 401);
        let v = toy_probe_vector(p, 402);
        assert_eq!(d_beta.len(), p);
        assert_eq!(v.len(), p);

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let want = workspace
            .directional_derivative_operator(&d_beta)
            .expect("dH op call")
            .expect("dH op present")
            .mul_vec(&v);

        let eps = 1e-5;
        let make_state = |scale: f64| ParameterBlockState {
            beta: &state.beta + &(d_beta.mapv(|b| scale * b)),
            eta: state.eta.clone(),
        };
        let plus = family
            .exact_newton_joint_hessian_workspace(
                std::slice::from_ref(&make_state(eps)),
                &[spec.clone()],
            )
            .expect("plus workspace")
            .expect("plus workspace present");
        let minus = family
            .exact_newton_joint_hessian_workspace(
                std::slice::from_ref(&make_state(-eps)),
                &[spec.clone()],
            )
            .expect("minus workspace")
            .expect("minus workspace present");
        let hv_plus = plus
            .hessian_matvec(&v)
            .expect("plus matvec")
            .expect("plus matvec");
        let hv_minus = minus
            .hessian_matvec(&v)
            .expect("minus matvec")
            .expect("minus matvec");
        let fd: Array1<f64> = (&hv_plus - &hv_minus).mapv(|x| x / (2.0 * eps));

        for i in 0..p {
            let scale = want[i].abs().max(1.0);
            // O(ε²) centered FD on a smooth Hessian gives ~1e-7 relative error
            // at ε=1e-5; loose 5e-5 tolerance covers the dominant truncation
            // term plus the inflation by `||v||·||d_beta||`.
            let tol = 5e-5 * scale + 5e-7;
            assert!(
                (want[i] - fd[i]).abs() <= tol,
                "dH FD mismatch at {i}: op={:.6e}, fd={:.6e}, tol={:.6e}",
                want[i],
                fd[i],
                tol,
            );
        }
    }

    /// FD check for the cached CTN barrier d²H operator (fourth-derivative
    /// formula `D²(∇²B)[u,w]v = 6 μ Dᵀ((Du)(Dw)(Dv)/c⁴)`).
    ///
    /// Centered FD on the dH operator along `dir_w` recovers d²H[u, w] · v;
    /// this exercises both the cached `inv_hp_qu` and the chained Khatri–Rao
    /// apply on the perturbed iterate.
    #[test]
    fn ctn_d2h_operator_matches_fd_under_beta_perturbation() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let dir_u = toy_probe_vector(p, 501);
        let dir_w = toy_probe_vector(p, 502);
        let v = toy_probe_vector(p, 503);

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let want = workspace
            .second_directional_derivative_operator(&dir_u, &dir_w)
            .expect("d2H op call")
            .expect("d2H op present")
            .mul_vec(&v);

        let eps = 1e-5;
        let make_state = |scale: f64| ParameterBlockState {
            beta: &state.beta + &(dir_w.mapv(|b| scale * b)),
            eta: state.eta.clone(),
        };
        let plus_ws = family
            .exact_newton_joint_hessian_workspace(
                std::slice::from_ref(&make_state(eps)),
                &[spec.clone()],
            )
            .expect("plus ws")
            .expect("plus ws present");
        let minus_ws = family
            .exact_newton_joint_hessian_workspace(
                std::slice::from_ref(&make_state(-eps)),
                &[spec.clone()],
            )
            .expect("minus ws")
            .expect("minus ws present");
        let dh_plus = plus_ws
            .directional_derivative_operator(&dir_u)
            .expect("plus dH op call")
            .expect("plus dH op present")
            .mul_vec(&v);
        let dh_minus = minus_ws
            .directional_derivative_operator(&dir_u)
            .expect("minus dH op call")
            .expect("minus dH op present")
            .mul_vec(&v);
        let fd: Array1<f64> = (&dh_plus - &dh_minus).mapv(|x| x / (2.0 * eps));

        for i in 0..p {
            let scale = want[i].abs().max(1.0);
            let tol = 5e-5 * scale + 5e-7;
            assert!(
                (want[i] - fd[i]).abs() <= tol,
                "d2H FD mismatch at {i}: op={:.6e}, fd={:.6e}, tol={:.6e}",
                want[i],
                fd[i],
                tol,
            );
        }
    }

    /// FD check for the CTN barrier `∇²B v` operator itself: centered FD on the
    /// log-likelihood gradient w.r.t. β reproduces `H(β) v` (to within FD
    /// truncation). This is the `μ Dᵀ((Dv)/c²)` formula plus the
    /// β-independent `X_val^T W X_val` term.
    #[test]
    fn ctn_hessian_matvec_matches_grad_fd() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let v = toy_probe_vector(p, 601);

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let hv = workspace
            .hessian_matvec(&v)
            .expect("matvec call")
            .expect("matvec present");

        let eps = 1e-6;
        // CTN's `evaluate()` returns the score (gradient of log-likelihood)
        // through the working-set; the joint Hessian is `-d²ℓ/dβ²`, so
        // `H · v ≈ -[grad(β + εv) - grad(β - εv)] / (2ε)`.
        let make_state = |scale: f64| ParameterBlockState {
            beta: &state.beta + &(v.mapv(|b| scale * b)),
            eta: state.eta.clone(),
        };
        let grad_at = |st: &ParameterBlockState| -> Array1<f64> {
            let eval = family
                .evaluate(std::slice::from_ref(st))
                .expect("evaluate must succeed");
            match &eval.blockworking_sets[0] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient.clone(),
                _ => panic!("CTN must report ExactNewton working set"),
            }
        };
        let grad_plus = grad_at(&make_state(eps));
        let grad_minus = grad_at(&make_state(-eps));
        // The score is +∂ℓ/∂β, and H = -∂²ℓ/∂β². Centered FD on the score gives
        // dscore/dβ · v = -H · v, so we negate to compare against `hv`.
        let fd: Array1<f64> = (&grad_plus - &grad_minus).mapv(|x| -x / (2.0 * eps));

        for i in 0..p {
            let scale = hv[i].abs().max(1.0);
            let tol = 1e-4 * scale + 1e-6;
            assert!(
                (hv[i] - fd[i]).abs() <= tol,
                "Hv FD mismatch at {i}: op={:.6e}, fd={:.6e}, tol={:.6e}",
                hv[i],
                fd[i],
                tol,
            );
        }
    }

    #[test]
    fn ctn_scop_gradient_matches_loglikelihood_fd() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();

        let analytic = family
            .exact_newton_joint_gradient_evaluation(std::slice::from_ref(&state), &[spec])
            .expect("SCOP analytic gradient evaluation")
            .expect("SCOP analytic gradient must be present");
        assert_eq!(analytic.gradient.len(), p);

        let eps = 1e-6;
        for coord in 0..p {
            let mut beta_plus = state.beta.clone();
            beta_plus[coord] += eps;
            let plus_state = ParameterBlockState {
                beta: beta_plus,
                eta: state.eta.clone(),
            };
            let ll_plus = family
                .log_likelihood_only(std::slice::from_ref(&plus_state))
                .expect("positive perturbation remains feasible");

            let mut beta_minus = state.beta.clone();
            beta_minus[coord] -= eps;
            let minus_state = ParameterBlockState {
                beta: beta_minus,
                eta: state.eta.clone(),
            };
            let ll_minus = family
                .log_likelihood_only(std::slice::from_ref(&minus_state))
                .expect("negative perturbation remains feasible");

            let fd = (ll_plus - ll_minus) / (2.0 * eps);
            let scale = fd.abs().max(analytic.gradient[coord].abs()).max(1.0);
            let tol = 5e-6 * scale + 5e-8;
            assert!(
                (analytic.gradient[coord] - fd).abs() <= tol,
                "SCOP gradient FD mismatch at {coord}: analytic={:.6e}, fd={:.6e}, tol={:.6e}",
                analytic.gradient[coord],
                fd,
                tol,
            );
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
            .exact_newton_joint_gradient_evaluation(std::slice::from_ref(&state), &[spec.clone()])
            .expect("gradient-only call")
            .expect("gradient-only result must be present");
        assert!(
            (want_ll - gradient_eval.log_likelihood).abs()
                <= 1e-12 * want_ll.abs().max(1.0) + 1e-12,
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

    /// Pairwise oracle for Phase-2 outer-HVP cross-checking.
    ///
    /// Builds the toy CTN fixture (n=4, p_resp=2, p_cov=2, ψ_dim=2), calls the
    /// existing pairwise body `exact_newton_joint_psisecond_order_terms(i, j)`
    /// for every (i, j) pair, computes the directional contraction
    /// `Σ_j v_j · pair(i, j)` for a fixed direction `v`, and writes the full
    /// likelihood-only result to `/tmp/ctn_pairwise_oracle.json`.
    ///
    /// Independent verification path for the SCOP CTN HVP work. The old Python
    /// scripts used the pre-SCOP linear tensor likelihood and were removed so
    /// they cannot be mistaken for ground truth.
    ///
    /// Run via:
    ///     cargo test --release ctn_pairwise_oracle_dumps_json -- --nocapture
    #[test]
    fn ctn_pairwise_oracle_dumps_json() {
        let psi = array![0.15, -0.10];
        let v = array![0.4, -0.7];

        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let block_states = vec![state.clone()];
        let specs = vec![spec.clone()];
        let beta = state.beta.clone();

        let psi_dim = psi.len();
        let p_total = beta.len();
        let n_obs = family.weights.as_ref().len();

        eprintln!(
            "[oracle] toy CTN: n={n_obs}, p_resp=2, p_cov=2, p_total={p_total}, ψ_dim={psi_dim}"
        );
        eprintln!("[oracle] β = {:?}", beta.as_slice().unwrap());
        eprintln!("[oracle] ψ = {:?}", psi.as_slice().unwrap());
        eprintln!("[oracle] v = {:?}", v.as_slice().unwrap());

        // The CTN psi-psi second-order kernel can return the dense block
        // either eagerly (`hessian_psi_psi`, p_total×p_total) or lazily
        // (`hessian_psi_psi_operator`, materialized via `to_dense`). The
        // oracle dump records the dense numbers, so materialize on demand.
        let dense_pair_hessian = |terms: &ExactNewtonJointPsiSecondOrderTerms| -> Array2<f64> {
            if terms.hessian_psi_psi.nrows() > 0 {
                terms.hessian_psi_psi.clone()
            } else {
                terms
                    .hessian_psi_psi_operator
                    .as_ref()
                    .expect("CTN psi-psi must expose either dense Hessian or operator")
                    .to_dense()
            }
        };

        // Per-pair pairwise body — likelihood pieces only (no penalty/logdet).
        let mut pair_records = Vec::new();
        for i in 0..psi_dim {
            for j in 0..psi_dim {
                let terms = family
                    .exact_newton_joint_psisecond_order_terms(
                        &block_states,
                        &specs,
                        &derivative_blocks,
                        i,
                        j,
                    )
                    .expect("pairwise call ok")
                    .expect("pairwise returns Some for valid i,j");
                let g_inf = terms
                    .score_psi_psi
                    .iter()
                    .fold(0.0f64, |m, x| m.max(x.abs()));
                let b_dense = dense_pair_hessian(&terms);
                let b_inf = b_dense.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                eprintln!(
                    "[oracle] pair (i={i}, j={j}): a={:.10}, ‖g‖∞={:.6e}, ‖b_mat‖∞={:.6e}",
                    terms.objective_psi_psi, g_inf, b_inf,
                );
                pair_records.push(serde_json::json!({
                    "i": i,
                    "j": j,
                    "a": terms.objective_psi_psi,
                    "g": terms.score_psi_psi.to_vec(),
                    "b_mat": b_dense.iter().copied().collect::<Vec<f64>>(),
                    "b_mat_shape": [b_dense.nrows(), b_dense.ncols()],
                }));
            }
        }

        // Directional contraction: Σ_j v_j · pair(i, j) for each free axis i.
        let mut a_dir = Array1::<f64>::zeros(psi_dim);
        let mut g_dir = Array2::<f64>::zeros((psi_dim, p_total));
        let mut b_dir = vec![Array2::<f64>::zeros((p_total, p_total)); psi_dim];
        for i in 0..psi_dim {
            for j in 0..psi_dim {
                let terms = family
                    .exact_newton_joint_psisecond_order_terms(
                        &block_states,
                        &specs,
                        &derivative_blocks,
                        i,
                        j,
                    )
                    .expect("pairwise call ok")
                    .expect("pairwise returns Some for valid i,j");
                a_dir[i] += v[j] * terms.objective_psi_psi;
                let mut g_row = g_dir.slice_mut(s![i, ..]);
                g_row.scaled_add(v[j], &terms.score_psi_psi);
                b_dir[i].scaled_add(v[j], &dense_pair_hessian(&terms));
            }
        }

        eprintln!("[oracle] directional contraction Σ_j v_j · pair(i, j):");
        for i in 0..psi_dim {
            eprintln!(
                "[oracle]   i={i}: a_dir={:.10}, ‖g_dir‖∞={:.6e}, ‖b_dir‖∞={:.6e}",
                a_dir[i],
                g_dir.row(i).iter().fold(0.0f64, |m, x| m.max(x.abs())),
                b_dir[i].iter().fold(0.0f64, |m, x| m.max(x.abs())),
            );
        }

        let directional_records: Vec<_> = (0..psi_dim)
            .map(|i| {
                serde_json::json!({
                    "i": i,
                    "a_dir": a_dir[i],
                    "g_dir": g_dir.row(i).to_vec(),
                    "b_dir": b_dir[i].iter().copied().collect::<Vec<f64>>(),
                    "b_dir_shape": [p_total, p_total],
                })
            })
            .collect();

        let blob = serde_json::json!({
            "config": {
                "n": n_obs,
                "p_resp": 2,
                "p_cov": 2,
                "p_total": p_total,
                "psi_dim": psi_dim,
                "beta": beta.to_vec(),
                "psi": psi.to_vec(),
                "v": v.to_vec(),
            },
            "pairwise": pair_records,
            "directional_contraction": directional_records,
            "note": "Likelihood-only pieces from exact_newton_joint_psisecond_order_terms. \
                     Penalty/logdet contributions are added by the unified evaluator's \
                     outer_hessian_entry. Cross-check this against sympy-shadow's symbolic \
                     derivation of the same likelihood quantities at the same toy config.",
        });

        let path = "/tmp/ctn_pairwise_oracle.json";
        std::fs::write(
            path,
            serde_json::to_string_pretty(&blob).expect("serialize ok"),
        )
        .expect("write ok");
        eprintln!("[oracle] wrote {path}");

        // Sanity assertions: nothing NaN, directional contraction is consistent
        // with element-wise summation.
        assert!(a_dir.iter().all(|x| x.is_finite()));
        assert!(g_dir.iter().all(|x| x.is_finite()));
        assert!(b_dir.iter().all(|m| m.iter().all(|x| x.is_finite())));
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
        // Per-axis trait method routes through the directional accumulator with
        // a unit basis vector e_axis. The accumulator skips zero entries, so
        // this loops once over `axis` and yields the same p-vector that
        // `lifted_transpose(_, axis, v)` would, while keeping the directional
        // kernel as a live production caller (substrate for the future HVP).
        let mut unit = Array1::<f64>::zeros(self.covariate_derivs.len());
        unit[axis] = 1.0;
        self.transpose_directional(&unit.view(), v)
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let mut unit = Array1::<f64>::zeros(self.covariate_derivs.len());
        unit[axis] = 1.0;
        self.forward_directional(&unit.view(), u)
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        let mut unit = Array1::<f64>::zeros(q);
        unit[axis] = 1.0;
        self.transpose_second_directional(&unit.view(), &unit.view(), v)
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        let mut unit_d = Array1::<f64>::zeros(q);
        let mut unit_e = Array1::<f64>::zeros(q);
        unit_d[axis_d] = 1.0;
        unit_e[axis_e] = 1.0;
        self.transpose_second_directional(&unit_d.view(), &unit_e.view(), v)
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        // Bilinear directional accumulator with `v_psi = w_psi = e_axis` skips
        // every (j,k) pair except (axis, axis), so this is numerically equivalent
        // to `lifted_forward_second(_, axis, axis, u)`.
        let q = self.covariate_derivs.len();
        let mut unit = Array1::<f64>::zeros(q);
        unit[axis] = 1.0;
        self.forward_second_directional(&unit.view(), &unit.view(), u)
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        let mut unit_d = Array1::<f64>::zeros(q);
        let mut unit_e = Array1::<f64>::zeros(q);
        unit_d[axis_d] = 1.0;
        unit_e[axis_e] = 1.0;
        self.forward_second_directional(&unit_d.view(), &unit_e.view(), u)
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let cov = self.cov_first_axis_row_chunk(axis, rows.clone())?;
        self.lifted_row_chunk_from_cov(rows, &cov)
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let cov = self.cov_second_axis_row_chunk(axis, axis, rows.clone())?;
        self.lifted_row_chunk_from_cov(rows, &cov)
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let cov = self.cov_second_axis_row_chunk(axis_d, axis_e, rows.clone())?;
        self.lifted_row_chunk_from_cov(rows, &cov)
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
/// - factored tensor penalty derivatives matching the SCOP penalty layout:
///   `E_shape ⊗ ∂S_cov/∂ψ` at the same covariate penalty index `m`.
pub fn build_tensor_psi_derivatives(
    family: &TransformationNormalFamily,
    covariate_psi_derivs: &[CustomFamilyBlockPsiDerivative],
) -> Result<Vec<CustomFamilyBlockPsiDerivative>, String> {
    let p_resp = family.response_val_basis.ncols();
    let n_axes = covariate_psi_derivs.len();
    let mut shape_resp = Array2::<f64>::eye(p_resp);
    shape_resp[[0, 0]] = 0.0;
    let shared_operator: Arc<dyn CustomFamilyPsiDerivativeOperator> =
        Arc::new(TensorKroneckerPsiOperator {
            response_val_basis: Arc::new(family.response_val_basis.clone()),
            covariate_design: family.covariate_design.clone(),
            covariate_derivs: covariate_psi_derivs.to_vec(),
            covariate_first_cache: Arc::new(
                (0..n_axes).map(|_| Mutex::new(None)).collect::<Vec<_>>(),
            ),
        });

    let mut derivs = Vec::with_capacity(n_axes);
    for a in 0..n_axes {
        let cov_deriv = &covariate_psi_derivs[a];
        let s_psi_penalty_components = cov_deriv
            .s_psi_penalty_components
            .as_ref()
            .map(|components| lift_covariate_penalty_derivative_components(components, &shape_resp))
            .transpose()?
            .or_else(|| {
                cov_deriv.s_psi_components.as_ref().map(|components| {
                    lift_dense_covariate_penalty_derivative_components(components, &shape_resp)
                })
            });
        let s_psi_psi_penalty_components = cov_deriv
            .s_psi_psi_penalty_components
            .as_ref()
            .map(|rows| {
                rows.iter()
                    .map(|cov_pen_pairs| -> Result<_, String> {
                        lift_covariate_penalty_derivative_components(cov_pen_pairs, &shape_resp)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .or_else(|| {
                cov_deriv.s_psi_psi_components.as_ref().map(|rows| {
                    rows.iter()
                        .map(|cov_pen_pairs| {
                            lift_dense_covariate_penalty_derivative_components(
                                cov_pen_pairs,
                                &shape_resp,
                            )
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

fn lift_dense_covariate_penalty_derivative_components(
    components: &[(usize, Array2<f64>)],
    shape_resp: &Array2<f64>,
) -> Vec<(usize, PenaltyMatrix)> {
    let mut out = Vec::with_capacity(components.len());
    for &(idx, ref ds_cov) in components {
        push_lifted_covariate_penalty_component(&mut out, idx, ds_cov.clone(), shape_resp);
    }
    out
}

fn lift_covariate_penalty_derivative_components(
    components: &[(usize, PenaltyMatrix)],
    shape_resp: &Array2<f64>,
) -> Result<Vec<(usize, PenaltyMatrix)>, String> {
    let mut out = Vec::with_capacity(components.len());
    for (idx, ds_cov) in components {
        push_lifted_covariate_penalty_component(
            &mut out,
            *idx,
            extract_covariate_penalty_factor(ds_cov)?,
            shape_resp,
        );
    }
    Ok(out)
}

fn push_lifted_covariate_penalty_component(
    out: &mut Vec<(usize, PenaltyMatrix)>,
    cov_penalty_idx: usize,
    ds_cov: Array2<f64>,
    shape_resp: &Array2<f64>,
) {
    out.push((
        cov_penalty_idx,
        PenaltyMatrix::KroneckerFactored {
            left: shape_resp.clone(),
            right: ds_cov,
        },
    ));
}

#[derive(Clone)]
struct TransformationExactGeometryCache {
    key: Vec<u64>,
    covariate_spec_resolved: TermCollectionSpec,
    covariate_design: TermCollectionDesign,
    family: TransformationNormalFamily,
    blocks: Vec<ParameterBlockSpec>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
}

impl TransformationExactGeometryCache {
    fn update_initial_beta(&mut self, beta_hint: Option<&Array1<f64>>) {
        let Some(spec) = self.blocks.first_mut() else {
            return;
        };
        if let Some(hint) = beta_hint
            && hint.len() == spec.design.ncols()
        {
            spec.initial_beta = Some(hint.clone());
        }
    }

    fn update_initial_log_lambdas(&mut self, log_lambdas: &Array1<f64>) -> Result<(), String> {
        let spec = self
            .blocks
            .first_mut()
            .ok_or_else(|| "missing transformation block spec".to_string())?;
        if log_lambdas.len() != spec.initial_log_lambdas.len() {
            return Err(format!(
                "transformation final fit rho length mismatch: got {}, expected {}",
                log_lambdas.len(),
                spec.initial_log_lambdas.len()
            ));
        }
        spec.initial_log_lambdas = log_lambdas.clone();
        Ok(())
    }
}

fn transformation_spatial_geometry_key(
    spec: &TermCollectionSpec,
    spatial_terms: &[usize],
) -> Result<Vec<u64>, String> {
    let mut key = Vec::new();
    key.push(spatial_terms.len() as u64);
    for &term_idx in spatial_terms {
        let term = spec.smooth_terms.get(term_idx).ok_or_else(|| {
            format!(
                "transformation spatial geometry key term index {term_idx} out of range for {} smooth terms",
                spec.smooth_terms.len()
            )
        })?;
        key.push(term_idx as u64);

        // The CTN exact-family cache is valid only for an identical covariate
        // geometry. Length-scale and anisotropy scalars are not enough: the
        // family also embeds frozen centers, input standardization scales,
        // identifiability transforms, and active penalty topology. Serialize
        // the already-frozen term and store the exact bytes in the key so a
        // cache hit means the saved prediction design will replay the same
        // matrix used by the final inner fit.
        let payload = serde_json::to_vec(term).map_err(|err| {
            format!("failed to serialize transformation spatial geometry term {term_idx}: {err}")
        })?;
        key.push(payload.len() as u64);
        for chunk in payload.chunks(8) {
            let mut bytes = [0u8; 8];
            for (dst, src) in bytes.iter_mut().zip(chunk.iter().copied()) {
                *dst = src;
            }
            key.push(u64::from_le_bytes(bytes));
        }
    }
    Ok(key)
}

// ---------------------------------------------------------------------------
// Top-level fit function
// ---------------------------------------------------------------------------

/// Result of `fit_transformation_normal`.
#[derive(Clone)]
pub struct TransformationNormalFitResult {
    pub family: TransformationNormalFamily,
    pub fit: UnifiedFitResult,
    pub covariate_spec_resolved: TermCollectionSpec,
    pub covariate_design: TermCollectionDesign,
    pub score_calibration: TransformationScoreCalibration,
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
    let covariate_spec = covariate_spec.clone();

    // 1. Build a bootstrap covariate design first so the response basis can
    // adapt to the tensor width instead of always using the global default.
    let boot_design = build_term_collection_design(covariate_data, &covariate_spec)
        .map_err(|e| format!("failed to build bootstrap covariate design: {e}"))?;
    let boot_spec = freeze_term_collection_from_design(&covariate_spec, &boot_design)
        .map_err(|e| format!("failed to freeze bootstrap covariate spatial basis centers: {e}"))?;
    let mut effective_config = config.clone();
    effective_config.response_num_internal_knots =
        effective_response_num_internal_knots(config, response.len(), boot_design.design.ncols());

    // 2. Build response basis ONCE — it is independent of κ once the effective
    // response complexity has been chosen.
    let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
        build_response_basis(response, &effective_config)?;

    // 3. Check whether spatial κ optimization is needed.
    let spatial_terms = spatial_length_scale_term_indices(&covariate_spec);

    if spatial_terms.is_empty() || !kappa_options.enabled {
        // ------------------------------------------------------------------
        // NO κ: build family directly, fit, return.
        // ------------------------------------------------------------------
        let cov_design = boot_design;
        let cov_spec_resolved = boot_spec;

        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            response,
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
        let (fit, score_calibration) =
            calibrate_transformation_scores(&family, fit, covariate_data, &cov_spec_resolved)?;

        return Ok(TransformationNormalFitResult {
            family,
            fit,
            covariate_spec_resolved: cov_spec_resolved,
            covariate_design: cov_design,
            score_calibration,
        });
    }

    // ------------------------------------------------------------------
    // YES κ: use the N-block spatial length-scale optimizer (1 block).
    // ------------------------------------------------------------------

    let kappa0 = SpatialLogKappaCoords::from_length_scales_aniso(
        &covariate_spec,
        &spatial_terms,
        kappa_options,
    )
    .reseed_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        kappa_options,
    );
    let kappa_dims = kappa0.dims_per_term().to_vec();
    let kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        &kappa_dims,
        kappa_options,
    );
    let kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        covariate_data,
        &covariate_spec,
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
        response,
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
    let rho0 = probe_block.initial_log_lambdas.clone();
    let rho_floor = -12.0;
    let rho_lower = Array1::<f64>::from_elem(n_penalties, rho_floor);
    let rho_upper = Array1::<f64>::from_elem(n_penalties, 12.0);
    let analytic_gradient = analytic_psi_available;
    let analytic_hessian = false;

    let (rho0_min, rho0_max) = if rho0.is_empty() {
        (0.0, 0.0)
    } else {
        (
            rho0.iter().copied().fold(f64::INFINITY, f64::min),
            rho0.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    log::info!(
        "[transformation-normal] skipping baseline custom-family prefit before exact joint optimization \
         (rho_dim={}, log_kappa_dim={}, rho0_range=[{:.3}, {:.3}]); using CTN warm start and penalty-scale rho seed",
        n_penalties,
        kappa0.len(),
        rho0_min,
        rho0_max,
    );

    if !analytic_psi_available {
        return Err(
            "transformation-normal spatial length-scale optimization requires analytic spatial psi derivatives"
                .to_string(),
        );
    }

    // Shared mutable state for warm-starting across optimizer iterations.
    let beta_hint: RefCell<Option<Array1<f64>>> = RefCell::new(None);

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
                response,
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
        let effective_spec = freeze_term_collection_from_design(spec, design)
            .map_err(|e| format!("failed to freeze transformation geometry key: {e}"))?;
        let key = transformation_spatial_geometry_key(&effective_spec, &spatial_terms_for_cache)?;
        let needs_rebuild = exact_geometry_cache
            .borrow()
            .as_ref()
            .map(|cached| cached.key != key)
            .unwrap_or(true);
        if !needs_rebuild {
            return Ok(());
        }

        let geom_start = std::time::Instant::now();
        let exact_design = build_term_collection_design(covariate_data, &effective_spec)
            .map_err(|e| format!("failed to rebuild frozen transformation geometry: {e}"))?;
        let family = make_family(&exact_design)?;
        let cov_psi_derivs =
            build_block_spatial_psi_derivatives(covariate_data, &effective_spec, &exact_design)?
                .ok_or_else(|| {
                    "missing covariate spatial psi derivatives for transformation model".to_string()
                })?;
        let tensor_derivs = build_tensor_psi_derivatives(&family, &cov_psi_derivs)?;

        log::debug!(
            "[transformation-normal] rebuilt exact geometry cache for {} spatial terms in {:.3}s",
            spatial_terms_for_cache.len(),
            geom_start.elapsed().as_secs_f64(),
        );

        exact_geometry_cache.replace(Some(TransformationExactGeometryCache {
            key,
            covariate_spec_resolved: effective_spec,
            covariate_design: exact_design,
            blocks: vec![family.block_spec()],
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
        // planner cannot pick EFS / Hybrid-EFS. The exact Hessian operator is
        // numerically too brittle for CTN spatial search: finite cost steps can
        // produce non-finite logdet cross traces and enormous gradients. Route
        // this family through the analytic-gradient BFGS path; the gradient-only
        // CTN evaluator is intentionally cheap and no longer assembles the dense
        // SCOP Hessian.
        true,
        CTN_ALLOW_TAU_TAU_GRADIENT_ONLY_PREFERENCE,
        // fit_fn
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            geometry.update_initial_log_lambdas(&rho)?;
            geometry.update_initial_beta(beta_hint.borrow().as_ref());
            let fit = fit_custom_family_fixed_log_lambdas(
                &geometry.family,
                &geometry.blocks,
                &options,
                None,
                0,
                0.0,
                true,
            )
            .map_err(|e| format!("transformation fit_fn: {e}"))?;
            if let Some(block) = fit.block_states.first() {
                *geometry
                    .family
                    .row_quantity_cache
                    .lock()
                    .expect("CTN row quantity cache mutex poisoned") = None;
                let final_rows = geometry.family.row_quantities(&block.beta)?;
                let max_abs_h = final_rows
                    .h
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max);
                let cov_chunk = geometry
                    .family
                    .covariate_design
                    .try_row_chunk(0..response.len())
                    .map_err(|err| {
                        format!("final CTN covariate design validation failed: {err}")
                    })?;
                let max_abs_cov = cov_chunk.iter().copied().map(f64::abs).fold(0.0, f64::max);
                log::info!(
                    "[transformation-normal] final fixed-rho CTN validation: max_abs_h={:.6e} max_abs_covariate_basis={:.6e}",
                    max_abs_h,
                    max_abs_cov
                );
            }
            // Update warm start hints.
            if let Some(block) = fit.block_states.first() {
                *beta_hint.borrow_mut() = Some(block.beta.clone());
            }
            Ok(TransformationNormalFitResult {
                family: geometry.family.clone(),
                fit,
                covariate_spec_resolved: geometry.covariate_spec_resolved.clone(),
                covariate_design: geometry.covariate_design.clone(),
                score_calibration: TransformationScoreCalibration {
                    feature_cols: Vec::new(),
                    feature_center: Vec::new(),
                    feature_scale: Vec::new(),
                    rbf_centers: Vec::new(),
                    rbf_bandwidth: 1.0,
                    location_beta: Vec::new(),
                    log_scale_beta: Vec::new(),
                    global_mean: 0.0,
                    global_sd: 1.0,
                },
            })
        },
        // exact_fn
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], eval_mode| {
            use crate::solver::estimate::reml::unified::EvalMode;
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            geometry.update_initial_beta(beta_hint.borrow().as_ref());
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();

            let eval = evaluate_custom_family_joint_hyper(
                &geometry.family,
                &geometry.blocks,
                &options,
                &rho,
                &geometry.derivative_blocks,
                None,
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

            if matches!(eval_mode, EvalMode::ValueGradientHessian)
                && !eval.outer_hessian.is_analytic()
            {
                return Err(
                    "transformation exact joint objective did not return an outer Hessian"
                        .to_string(),
                );
            }

            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            geometry.update_initial_beta(beta_hint.borrow().as_ref());
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let eval = evaluate_custom_family_joint_hyper_efs(
                &geometry.family,
                &geometry.blocks,
                &options,
                &rho,
                &geometry.derivative_blocks,
                None,
            )
            .map_err(|e| format!("transformation exact_efs_fn: {e}"))?;
            Ok(eval.efs_eval)
        },
    )?;

    let mut fit = solved.fit;
    let (calibrated_fit, score_calibration) = calibrate_transformation_scores(
        &fit.family,
        fit.fit.clone(),
        covariate_data,
        &fit.covariate_spec_resolved,
    )?;
    fit.fit = calibrated_fit;
    fit.score_calibration = score_calibration;
    Ok(fit)
}
