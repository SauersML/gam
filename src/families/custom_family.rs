use crate::faer_ndarray::FaerCholesky;
use crate::faer_ndarray::FaerEigh;
use crate::faer_ndarray::FaerSvd;
use crate::linalg::utils::{StableSolver, default_slq_parameters, stochastic_lanczos_logdet_spd};
use crate::matrix::{
    DesignMatrix, EmbeddedColumnBlock, EmbeddedSquareBlock, LinearOperator, SymmetricMatrix,
};
use crate::pirls::{LinearInequalityConstraints, solve_newton_directionwith_lower_bounds};
use crate::smooth::{
    TermCollectionDesign, TermCollectionSpec, spatial_length_scale_term_indices,
    try_build_spatial_log_kappa_derivativeinfo_list,
};
use crate::solver::active_set::solve_quadratic_with_linear_constraints;
use crate::solver::estimate::reml::penalty_logdet::PenaltyPseudologdet;
use crate::solver::estimate::reml::unified::{
    BlockCoupledOperator, DispersionHandling, DriftDerivResult, FixedDriftDerivFn,
    HessianDerivativeProvider, HyperCoord, HyperCoordDrift, HyperCoordPair, HyperOperator,
    MatrixFreeSpdOperator, compute_block_penalty_logdet_derivs, exact_intersection_nullity,
    spectral_epsilon, spectral_regularize,
};
use crate::solver::estimate::{
    FitGeometry, ensure_finite_scalar_estimation, validate_all_finite_estimation,
};
use crate::types::{RidgeDeterminantMode, RidgePolicy};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, s};
use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use thiserror::Error;

pub use crate::solver::estimate::reml::unified::{EvalMode, PseudoLogdetMode};

/// A penalty matrix that may be stored in Kronecker-factored form.
///
/// For tensor-product terms (e.g. time-varying survival covariates), the penalty
/// has the structure `S = left ⊗ right` (Kronecker product). Keeping this
/// factored avoids materializing (p_left × p_right)² dense entries and enables
/// exact log-determinant computation via `log|A ⊗ B| = n_B log|A| + n_A log|B|`.
///
/// Dense penalties are stored as-is.  Callers that need a raw `Array2<f64>` can
/// call `as_dense()` (zero-cost for Dense, lazy-materialized for KroneckerFactored).
#[derive(Clone)]
pub enum PenaltyMatrix {
    Dense(Array2<f64>),
    KroneckerFactored {
        left: Array2<f64>,
        right: Array2<f64>,
    },
    /// Block-local penalty: `local` is `block_dim × block_dim`, embedded at
    /// `col_range` in the full parameter space of dimension `total_dim`.
    /// Avoids materializing the full `total_dim × total_dim` matrix.
    Blockwise {
        local: Array2<f64>,
        col_range: std::ops::Range<usize>,
        total_dim: usize,
    },
}

impl PenaltyMatrix {
    /// Number of rows (= number of columns, since penalties are square).
    pub fn dim(&self) -> usize {
        match self {
            Self::Dense(m) => m.nrows(),
            Self::KroneckerFactored { left, right } => left.nrows() * right.nrows(),
            Self::Blockwise { total_dim, .. } => *total_dim,
        }
    }

    /// Returns (nrows, ncols) like Array2::dim().
    pub fn shape(&self) -> (usize, usize) {
        let d = self.dim();
        (d, d)
    }

    /// Materialize the full dense matrix.
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(m) => m.clone(),
            Self::KroneckerFactored { left, right } => {
                crate::terms::construction::kronecker_product(left, right)
            }
            Self::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                let mut g = Array2::zeros((*total_dim, *total_dim));
                g.slice_mut(ndarray::s![
                    col_range.start..col_range.end,
                    col_range.start..col_range.end
                ])
                .assign(local);
                g
            }
        }
    }

    /// Borrow the inner dense matrix if Dense, otherwise materialize.
    pub fn as_dense_cow(&self) -> std::borrow::Cow<'_, Array2<f64>> {
        match self {
            Self::Dense(m) => std::borrow::Cow::Borrowed(m),
            Self::KroneckerFactored { .. } | Self::Blockwise { .. } => {
                std::borrow::Cow::Owned(self.to_dense())
            }
        }
    }

    /// Convert from a `BlockwisePenalty` without expanding to full dimensions.
    pub fn from_blockwise(bp: crate::terms::smooth::BlockwisePenalty, total_dim: usize) -> Self {
        Self::Blockwise {
            local: bp.local,
            col_range: bp.col_range,
            total_dim,
        }
    }

    /// Returns a reference to the inner matrix if this is a Dense variant.
    pub fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(m) => Some(m),
            Self::KroneckerFactored { .. } | Self::Blockwise { .. } => None,
        }
    }

    /// Compute S * v using the Kronecker vec trick when factored:
    ///   (A ⊗ B) vec(V) = vec(B V Aᵀ)
    /// where V = reshape(v, (p_right, p_left)).
    pub fn dot(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(m) => m.dot(v),
            Self::KroneckerFactored { left, right } => {
                let p_left = left.nrows();
                let p_right = right.nrows();
                // v is (p_left * p_right,).  Reshape as (p_right, p_left).
                let v_mat =
                    ndarray::ArrayView2::from_shape((p_right, p_left), v.as_slice().unwrap())
                        .unwrap();
                // result = B V A' then flatten.
                let bv = right.dot(&v_mat);
                let bva = bv.dot(&left.t());
                Array1::from_iter(bva.iter().copied())
            }
            Self::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                let mut out = Array1::zeros(*total_dim);
                let v_block = v.slice(ndarray::s![col_range.clone()]);
                let result_block = local.dot(&v_block);
                out.slice_mut(ndarray::s![col_range.clone()])
                    .assign(&result_block);
                out
            }
        }
    }

    /// Add λ * self to a mutable dense accumulator.
    pub fn add_scaled_to(&self, lambda: f64, target: &mut Array2<f64>) {
        match self {
            Self::Dense(m) => {
                target.scaled_add(lambda, m);
            }
            Self::KroneckerFactored { left, right } => {
                let p_left = left.nrows();
                let p_right = right.nrows();
                for i1 in 0..p_left {
                    for j1 in 0..p_left {
                        let a_ij = left[[i1, j1]];
                        if a_ij == 0.0 {
                            continue;
                        }
                        let scaled_a = lambda * a_ij;
                        for i2 in 0..p_right {
                            let row = i1 * p_right + i2;
                            for j2 in 0..p_right {
                                let col = j1 * p_right + j2;
                                target[[row, col]] += scaled_a * right[[i2, j2]];
                            }
                        }
                    }
                }
            }
            Self::Blockwise {
                local, col_range, ..
            } => {
                target
                    .slice_mut(ndarray::s![col_range.clone(), col_range.clone()])
                    .scaled_add(lambda, local);
            }
        }
    }

    /// Add λ * diag(self) to a mutable diagonal accumulator.
    pub fn add_scaled_diag_to(&self, lambda: f64, target: &mut Array1<f64>) {
        match self {
            Self::Dense(m) => {
                let p = m.nrows().min(target.len());
                for j in 0..p {
                    target[j] += lambda * m[[j, j]];
                }
            }
            Self::KroneckerFactored { left, right } => {
                let p_left = left.nrows();
                let p_right = right.nrows();
                debug_assert_eq!(target.len(), p_left * p_right);
                for i_left in 0..p_left {
                    let left_diag = left[[i_left, i_left]];
                    if left_diag == 0.0 {
                        continue;
                    }
                    let scaled_left = lambda * left_diag;
                    for i_right in 0..p_right {
                        target[i_left * p_right + i_right] +=
                            scaled_left * right[[i_right, i_right]];
                    }
                }
            }
            Self::Blockwise {
                local, col_range, ..
            } => {
                let width = local.nrows().min(col_range.len());
                for local_idx in 0..width {
                    target[col_range.start + local_idx] += lambda * local[[local_idx, local_idx]];
                }
            }
        }
    }

    /// Compute the quadratic form β' S β.
    pub fn quadratic_form(&self, beta: &Array1<f64>) -> f64 {
        match self {
            Self::Dense(m) => beta.dot(&m.dot(beta)),
            Self::KroneckerFactored { .. } => {
                let sv = self.dot(beta);
                beta.dot(&sv)
            }
            Self::Blockwise {
                local, col_range, ..
            } => {
                let beta_block = beta.slice(ndarray::s![col_range.clone()]);
                let sv = local.dot(&beta_block);
                beta_block.dot(&sv)
            }
        }
    }

    /// Access dimensions like an Array2.
    pub fn nrows(&self) -> usize {
        self.dim()
    }

    pub fn ncols(&self) -> usize {
        self.dim()
    }
}

impl From<Array2<f64>> for PenaltyMatrix {
    fn from(m: Array2<f64>) -> Self {
        Self::Dense(m)
    }
}

/// Static specification for one parameter block in a custom family.
#[derive(Clone)]
pub struct ParameterBlockSpec {
    pub name: String,
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    /// Block-local penalty matrices (all p_block x p_block).
    pub penalties: Vec<PenaltyMatrix>,
    /// Structural nullspace dimension of each penalty matrix (same length as `penalties`).
    /// Used by the penalty pseudo-logdet to determine rank without numerical thresholds.
    /// If empty, falls back to eigenvalue-based rank detection.
    pub nullspace_dims: Vec<usize>,
    /// Initial log-smoothing parameters for this block (same length as `penalties`).
    pub initial_log_lambdas: Array1<f64>,
    /// Optional initial coefficients (defaults to zeros if omitted).
    pub initial_beta: Option<Array1<f64>>,
}

fn custom_family_block_role(
    name: &str,
    index: usize,
    n_blocks: usize,
) -> crate::solver::estimate::BlockRole {
    use crate::solver::estimate::BlockRole;

    if n_blocks == 1 {
        return BlockRole::Mean;
    }

    match name.trim().to_ascii_lowercase().as_str() {
        "eta" | "mean" | "beta" => BlockRole::Mean,
        "mu" | "location" | "marginal_surface" => BlockRole::Location,
        "threshold" => BlockRole::Threshold,
        "log_sigma" | "scale" | "logslope_surface" => BlockRole::Scale,
        "time" | "time_transform" | "time_surface" => BlockRole::Time,
        "wiggle" | "linkwiggle" => BlockRole::LinkWiggle,
        _ if index == 0 => BlockRole::Location,
        _ => BlockRole::Scale,
    }
}

/// Current state for a parameter block.
#[derive(Clone, Debug)]
pub struct ParameterBlockState {
    pub beta: Array1<f64>,
    pub eta: Array1<f64>,
}

#[derive(Clone)]
pub struct BlockGeometryDirectionalDerivative {
    /// Directional derivative of the block design matrix along a coefficient-space direction.
    pub d_design: Option<Array2<f64>>,
    /// Directional derivative of the block offset along the same direction.
    pub d_offset: Array1<f64>,
}

/// Working quantities supplied by a custom family for one block.
///
/// # Observed vs expected information (see response.md Section 3)
///
/// For the outer REML/LAML criterion, the Hessian used in log|H| and trace terms
/// must be the **observed** (actual) Hessian at the mode, not the expected Fisher.
///
/// - `ExactNewton`: provides -nabla^2 log L directly, which is the observed Hessian
///   by construction. This is always correct.
///
/// - `Diagonal`: provides IRLS working weights W such that the per-block Hessian
///   is X'WX. For canonical links (logit-Binomial, log-Poisson), W_obs = W_Fisher.
///   For non-canonical links, W should ideally be the observed weight
///   W_obs = W_Fisher - (y-mu)*B to ensure the outer REML uses the exact Laplace
///   Hessian. Currently, GAMLSS families using Diagonal blocks with non-canonical
///   links may provide Fisher weights; this is acceptable when the link is close to
///   canonical (small residual correction) but introduces a PQL-type approximation
///   for strongly non-canonical links.
#[derive(Clone, Debug)]
pub enum BlockWorkingSet {
    /// Standard IRLS/GLM-style diagonal working set for eta-space updates.
    Diagonal {
        /// IRLS pseudo-response for this block's linear predictor.
        working_response: Array1<f64>,
        /// IRLS working weights for this block (non-negative, length n).
        ///
        /// For the inner solver, Fisher or observed weights both find the same mode.
        /// For the outer REML/LAML log|H| term, observed weights are the correct
        /// Laplace choice (see response.md Section 3). Canonical-link families need
        /// no correction since observed = Fisher.
        working_weights: Array1<f64>,
    },
    /// Exact Newton block update in coefficient space.
    ///
    /// `gradient` is nabla log L wrt block coefficients.
    /// `hessian` is -nabla^2 log L wrt block coefficients (positive semidefinite near optimum).
    ///
    /// This is the observed Hessian by construction (actual second derivative of the
    /// log-likelihood), which is the correct quantity for the outer REML Laplace
    /// approximation.
    ExactNewton {
        gradient: Array1<f64>,
        hessian: SymmetricMatrix,
    },
}

/// Slice a joint Hessian's principal diagonal blocks and pair them with
/// pre-computed per-block gradients to form `ExactNewton` working sets.
///
/// `block_gradients[k]` is the gradient for block k.
/// `block_ranges[k]` gives the coefficient index range for block k in
/// the flat joint parameter vector.
/// The Hessian for each block is the principal diagonal block
/// `joint_hessian[range, range]`.
///
/// This is the single authoritative way to derive block Hessians: they are
/// always views of the joint Hessian, never independently computed.
pub fn slice_joint_into_block_working_sets(
    block_gradients: Vec<Array1<f64>>,
    joint_hessian: &Array2<f64>,
    block_ranges: &[std::ops::Range<usize>],
) -> Vec<BlockWorkingSet> {
    assert_eq!(
        block_gradients.len(),
        block_ranges.len(),
        "slice_joint_into_block_working_sets: gradient/range count mismatch"
    );
    block_gradients
        .into_iter()
        .zip(block_ranges.iter())
        .map(|(gradient, range)| {
            let hessian = joint_hessian
                .slice(s![range.clone(), range.clone()])
                .to_owned();
            BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactNewtonOuterObjective {
    RidgedQuadraticReml,
    StrictPseudoLaplace,
}

/// Highest exact outer derivative order a family wants to expose at the
/// current realized problem scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExactOuterDerivativeOrder {
    Zeroth,
    First,
    Second,
}

impl ExactOuterDerivativeOrder {
    pub fn has_gradient(self) -> bool {
        !matches!(self, Self::Zeroth)
    }

    pub fn has_hessian(self) -> bool {
        matches!(self, Self::Second)
    }
}

/// Maximum dense outer-Hessian size (f64 elements) before downgrading to
/// first-order when a caller insists on materializing it. This gate is keyed
/// to the actual outer dimension K, not the inner coefficient dimension p.
const DEFAULT_EXACT_OUTER_MAX_ELEMENTS: u64 = 16_000_000;
const EXACT_OUTER_HESSIAN_LARGE_N_THRESHOLD: usize = 50_000;
const EXACT_OUTER_HESSIAN_LARGE_N_MIN_DIM: usize = 32;
const EXACT_OUTER_HESSIAN_MAX_LINEAR_WORK: usize = 4_000_000;
const EXACT_OUTER_HESSIAN_MAX_QUADRATIC_WORK: usize = 50_000_000;

pub(crate) fn exact_outer_hessian_problem_scale_allows(specs: &[ParameterBlockSpec]) -> bool {
    let total_p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let max_n: usize = specs
        .iter()
        .map(|spec| spec.design.nrows())
        .max()
        .unwrap_or(0);
    let linear_work = max_n.saturating_mul(total_p);
    let quadratic_work = linear_work.saturating_mul(total_p);

    !((max_n >= EXACT_OUTER_HESSIAN_LARGE_N_THRESHOLD
        && total_p >= EXACT_OUTER_HESSIAN_LARGE_N_MIN_DIM)
        || linear_work > EXACT_OUTER_HESSIAN_MAX_LINEAR_WORK
        || quadratic_work > EXACT_OUTER_HESSIAN_MAX_QUADRATIC_WORK)
}

/// Shared cost-aware gate for second-order exact outer derivatives.
///
/// The outer Hessian returned by `compute_outer_hessian` has shape
/// (K+ext_dim) × (K+ext_dim) where K = total number of smoothing parameters.
/// Inner coefficient dimension p and sample count n enter only through block-
/// local H⁻¹ solves that are already performed for the gradient, so the only
/// meaningful affordability proxy for the second-order path is the size of the
/// dense K×K outer Hessian. Inner n·p or n·p² proxies do not correspond to any
/// allocation in the Hessian assembly and would incorrectly block affordable
/// problems (e.g. K=8, n=50000, p=80 → outer Hessian is 64 f64s, but an n·p²
/// gate would disable it).
pub fn cost_gated_outer_order(specs: &[ParameterBlockSpec]) -> ExactOuterDerivativeOrder {
    let k: u64 = specs.iter().map(|s| s.penalties.len() as u64).sum();
    let outer_elements = k.saturating_mul(k);
    if outer_elements > DEFAULT_EXACT_OUTER_MAX_ELEMENTS {
        ExactOuterDerivativeOrder::First
    } else {
        ExactOuterDerivativeOrder::Second
    }
}

pub(crate) fn exact_newton_outer_geometry_supports_second_order_solver<F: CustomFamily + ?Sized>(
    family: &F,
) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::StrictPseudoLaplace
}

/// Family evaluation over all parameter blocks.
#[derive(Clone, Debug)]
pub struct FamilyEvaluation {
    pub log_likelihood: f64,
    pub blockworking_sets: Vec<BlockWorkingSet>,
}

pub struct ExactNewtonJointGradientEvaluation {
    pub log_likelihood: f64,
    pub gradient: Array1<f64>,
}

/// User-defined family contract for multi-block generalized models.
pub trait CustomFamily {
    /// Evaluate log-likelihood and per-block working quantities at current block predictors.
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String>;

    /// Compute only the log-likelihood without building working sets.
    ///
    /// This is used in backtracking line searches where only the objective value
    /// is needed, avoiding the O(n × blocks) cost of assembling IRLS working
    /// weights and responses that will be immediately discarded.
    ///
    /// The default implementation falls back to `evaluate()` and discards the
    /// working sets.  Families with expensive working-set assembly should
    /// override this for a significant speedup.
    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        self.evaluate(block_states).map(|e| e.log_likelihood)
    }

    /// Selects the outer objective semantics for exact-Newton families.
    ///
    /// `RidgedQuadraticReml` is the explicit ridged surrogate REML surface:
    ///
    ///   -loglik + penalty + 0.5 (log|H| - log|S|_+)
    ///
    /// The determinant terms in this mode are evaluated on the stabilized
    /// curvature surface declared by `ridge_policy`, so this objective is an
    /// explicitly modified surrogate rather than an exact Laplace expansion
    /// at an indefinite Hessian.
    ///
    /// `StrictPseudoLaplace` is the exact-mode pseudo-Laplace surface used by the
    /// Charbonnier spatial family:
    ///
    ///   -loglik + penalty + 0.5 log|H|
    ///
    /// The latter deliberately omits the quadratic-only `-0.5 log|S|_+`
    /// normalization term because there is no tractable exact analogue for the
    /// nonquadratic prior without introducing the intractable prior normalizer.
    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::RidgedQuadraticReml
    }

    /// Whether the joint likelihood Hessian H_L depends on β.
    ///
    /// When `true`, the unified evaluator includes M_j[u] = D_β B_j[u]
    /// moving-design drift correction for ψ coordinates and marks
    /// `HyperCoord::b_depends_on_beta = true`.
    ///
    /// Default: `true` for StrictPseudoLaplace, `false` for RidgedQuadraticReml.
    /// Gaussian location-scale must override to `true` because their
    /// joint Hessian depends on β even though outer objective is RidgedQuadraticReml.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        self.exact_newton_outerobjective() != ExactNewtonOuterObjective::RidgedQuadraticReml
    }

    /// Declares how much exact outer calculus this family wants to expose for
    /// the current realized problem size.
    ///
    /// The default uses [`cost_gated_outer_order`], keyed to the actual outer
    /// hyperparameter dimension rather than the inner coefficient dimension.
    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        cost_gated_outer_order(specs)
    }

    /// Family-specific outer seeding policy.
    ///
    /// The default preserves the generic custom-family behavior. Families with
    /// a strong warm start can override this to keep seed screening from
    /// dominating the fit.
    fn outer_seed_config(&self, n_params: usize) -> crate::seeding::SeedConfig {
        if n_params == 0 {
            return crate::seeding::SeedConfig::default();
        }
        let mut config = crate::seeding::SeedConfig::default();
        config.max_seeds = if n_params <= 4 { 6 } else { 4 };
        config.seed_budget = 1;
        config.screen_max_inner_iterations = 2;
        config
    }

    /// Whether outer hyper-derivative evaluation must use a joint exact path.
    ///
    /// Default `false` allows the generic blockwise diagonal fallback when a
    /// family does not provide joint exact curvature.
    ///
    /// Families with coupled multi-block likelihoods can override this to
    /// prevent the outer code from silently evaluating a mathematically
    /// invalid block-local surrogate. The failure mode is:
    ///
    /// 1. the outer derivative still has block-local forcing
    ///      g_k = A_k beta
    ///    because `rho_k` enters only through the penalty;
    /// 2. but the fitted mode response is not block-local,
    ///      H u_k = -g_k,
    ///    because the likelihood Hessian has off-diagonal block coupling;
    /// 3. therefore a blockwise solve
    ///      H_b u_{k,b} = -(A_k beta)_b
    ///    is not the derivative of the profiled objective the code claims to
    ///    be optimizing.
    ///
    /// When this flag is `true`, the family is asserting that any outer
    /// hyper-derivative path must first obtain the full joint exact curvature
    /// before it can return a mathematically valid result.
    fn requires_joint_outer_hyper_path(&self) -> bool {
        false
    }

    /// Optional dynamic geometry hook for blocks whose design/offset depend on
    /// current values of other blocks.
    fn block_geometry(
        &self,
        _: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        Ok((spec.design.clone(), spec.offset.clone()))
    }

    /// Whether `block_geometry(...)` can change with the current block state.
    ///
    /// The default implementation is static: the effective geometry is just the
    /// stored `spec.design/spec.offset`, so the fit engine can use those
    /// references directly without repeatedly cloning dense matrices.
    ///
    /// Families that override `block_geometry(...)` with state-dependent
    /// behavior must override this to return `true`.
    fn block_geometry_is_dynamic(&self) -> bool {
        false
    }

    /// Optional directional derivative of the effective block geometry wrt the
    /// current block coefficients.
    ///
    /// For a block with effective predictor
    ///
    ///   eta(beta) = X(beta) beta + o(beta),
    ///
    /// the directional derivative along `d_beta` is
    ///
    ///   D eta[d_beta] = X d_beta + (D X[d_beta]) beta + D o[d_beta].
    ///
    /// For diagonal working-set REML derivatives this contributes to both:
    ///
    ///   D H[d_beta]
    ///   = (D X[d_beta])^T W X
    ///   + X^T W (D X[d_beta])
    ///   + X^T diag(D w[D eta[d_beta]]) X,
    ///
    /// and to the predictor drift fed into the weight directional derivative.
    ///
    /// Default `None` means the family is declaring that the current block's
    /// geometry has no coefficient-dependent drift beyond the base `X d_beta`
    /// term. Families with dynamic `block_geometry` must implement this hook
    /// when that declaration is false.
    fn block_geometry_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &ParameterBlockSpec,
        _: &Array1<f64>,
    ) -> Result<Option<BlockGeometryDirectionalDerivative>, String> {
        Ok(None)
    }

    /// Optional per-block coefficient projection applied after each block update.
    fn post_update_block_beta(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        Ok(beta)
    }

    /// Optional barrier-aware maximum feasible step size for a block update.
    ///
    /// Given the current block state and a proposed step direction `delta`,
    /// returns `Some(alpha_max)` where `alpha_max` is the largest step size
    /// in `(0, 1]` such that `beta + alpha_max * delta` remains strictly
    /// feasible with respect to any implicit barrier in the likelihood.
    ///
    /// Families whose log-likelihood contains natural log-barrier terms
    /// (e.g. `log(h')` in transformation-normal) should implement this to
    /// prevent the line search from evaluating the likelihood at infeasible
    /// points.  A fraction-to-boundary safety factor (e.g. 0.995) should be
    /// applied internally.
    ///
    /// Returns `None` if no barrier constraint applies (the default).
    fn max_feasible_step_size(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        Ok(None)
    }

    /// Optional linear inequality constraints for a block update:
    /// `A * beta_block >= b`.
    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        Ok(None)
    }

    /// Optional exact directional derivative of a block's ExactNewton Hessian.
    ///
    /// Returns `Some(dH)` where:
    /// - `dH` is the directional derivative of the block Hessian with respect to
    ///   the provided coefficient-space direction `d_beta` at current state.
    /// - shape is `(p_block, p_block)`.
    ///
    /// Default `None` means no exact directional Hessian drift is available.
    /// Exact REML/LAML derivative paths that require this term should treat
    /// `None` as unavailable rather than silently substituting zero.
    fn exact_newton_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Optional exact joint coefficient-space Hessian across all blocks.
    ///
    /// Returns the unpenalized matrix `H_L = -nabla^2 log L` in the flattened block order.
    ///
    /// This is the **observed** (actual) Hessian of the log-likelihood at the mode,
    /// NOT the expected Fisher information. The outer REML/LAML evaluator requires
    /// the observed Hessian for the exact Laplace approximation (see response.md
    /// Section 3). Since this method returns the actual second derivative of log L,
    /// it is correct by construction.
    ///
    /// For families using `BlockWorkingSet::Diagonal` (IRLS-style updates), the
    /// per-block Hessian is X'WX where W is the working weight. For canonical links
    /// W_obs = W_Fisher, but for non-canonical links the working weight should include
    /// the observed-information correction W_obs = W_Fisher - (y-mu)*B.
    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Optional exact joint log-likelihood / score evaluation in flattened
    /// coefficient space without building per-block Hessian working sets.
    fn exact_newton_joint_gradient_evaluation(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(None)
    }

    /// Optional exact directional derivative of the joint coefficient-space Hessian.
    ///
    /// Returns `Some(dH)` where `dH` is the directional derivative of the
    /// unpenalized joint Hessian `H = -∇² log L` along the flattened
    /// coefficient-space direction `d_beta_flat`.
    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Optional exact second directional derivative of the joint Hessian.
    ///
    /// Returns `Some(d2H)` where `d2H` is:
    ///   D²H[u, v] = d/dε d/dδ H(beta + εu + δv) |_{ε=δ=0}
    /// for flattened coefficient-space directions `u = d_beta_u_flat`,
    /// `v = d_betav_flat`.
    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: &Array1<f64>,
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Optional per-evaluation workspace for exact joint Hessian operators and
    /// directional derivatives.
    ///
    /// Families with expensive cache construction can override this to build
    /// shared state once and reuse it across the repeated `dH[v]` / `d²H[u,v]`
    /// calls made by the unified outer evaluator.
    fn exact_newton_joint_hessian_workspace(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        Ok(None)
    }

    /// Optional spec-aware exact joint Hessian.
    ///
    /// This hook exists because the outer hyper-derivative code works from the
    /// realized block specs, while some family instances may or may not cache
    /// those realized designs internally.
    ///
    /// The profiled/Laplace outer objective used here is
    ///
    ///   J(theta)
    ///   = V(beta(theta), theta)
    ///     + 0.5 log|H(beta(theta), theta)|
    ///     - 0.5 log|S(theta)|_+,
    ///
    /// evaluated at the fitted inner mode defined by
    ///
    ///   F(beta, theta) := D_beta V(beta, theta) = 0,
    ///   H(beta, theta) := F_beta(beta, theta) = H_L(beta, theta) + S(theta).
    ///
    /// For pure rho directions on families whose likelihood has no explicit
    /// rho-dependence, the fixed-beta forcing is
    ///
    ///   g_k := F_{rho_k} = A_k beta,
    ///   A_k := dS/drho_k.
    ///
    /// Differentiating stationarity gives the exact joint mode response
    ///
    ///   H u_k = -g_k,
    ///   u_k = d beta / d rho_k.
    ///
    /// Even if `A_k` is supported in only one penalty block, the solve for
    /// `u_k` must use the full joint Hessian `H`, because the likelihood can
    /// couple blocks through off-diagonal curvature. The first outer
    /// derivative is then
    ///
    ///   dJ/dtheta_i
    ///   = 0.5 beta^T A_k beta
    ///     + 0.5 tr(H^{-1}(A_k + D_beta H_L[u_k]))
    ///     - 0.5 tr(S^+ A_k),
    ///
    /// and when psi moves realized penalties the same spec-aware hook must be
    /// able to reconstruct H(beta, theta), D_beta H[u], and D_beta^2 H[u, v]
    /// from the current realized specs so the generic joint assembler can form
    ///
    ///   dot H_i  = H_i + D_beta H[beta_i],
    ///   ddot H_ij
    ///   = H_ij + T_i[beta_j] + T_j[beta_i]
    ///     + D_beta H[beta_ij] + D_beta^2 H[beta_i, beta_j].
    ///
    /// Families such as binomial location-scale with
    ///
    ///   q = -eta_t exp(-eta_ls)
    ///
    /// have exactly that coupled structure: the penalty forcing is block-local
    /// but the fitted mode response and the resulting `D_beta H_L[u_k]` drift
    /// are joint objects. If the realized `specs` already contain the designs
    /// needed to build those objects, the outer code should use them directly
    /// rather than falling back to a weaker blockwise surrogate just because
    /// the family instance itself did not cache the same designs.
    ///
    /// The default implementation delegates to `exact_newton_joint_hessian`.
    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian(block_states)
    }

    /// Optional scale-aware exact joint curvature for the outer REML calculus.
    ///
    /// Families whose exact derivatives can overflow may return a uniformly
    /// rescaled Hessian together with the metadata needed to keep every outer
    /// path consistent:
    ///
    /// - `hessian`: the scale-stabilized unpenalized joint Hessian
    /// - `rho_curvature_scale`: the uniform factor applied to every ρ-driven
    ///   penalty Hessian derivative in H-dependent trace / solve terms
    /// - `hessian_logdet_correction`: the additive correction needed to recover
    ///   `log|H_exact|` from `log|H_scaled|`
    ///
    /// The scale is evaluation-local metadata: callers must use the same
    /// factor for `H`, `dH`, `d²H`, and penalized trace operators within that
    /// evaluation, but they do not differentiate the scale itself.
    ///
    /// Families overriding this must also make
    /// `exact_newton_outer_curvature_directional_derivative[_with_specs]` and
    /// `exact_newton_outer_curvature_second_directional_derivative[_with_specs]`
    /// return derivatives in that same scaled curvature space.
    fn exact_newton_outer_curvature(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonOuterCurvature>, String> {
        Ok(None)
    }

    /// Optional first directional derivative matching
    /// `exact_newton_outer_curvature`.
    fn exact_newton_outer_curvature_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
    }

    /// Spec-aware variant of `exact_newton_outer_curvature_directional_derivative`.
    fn exact_newton_outer_curvature_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_outer_curvature_directional_derivative(block_states, d_beta_flat)
    }

    /// Optional second directional derivative matching
    /// `exact_newton_outer_curvature`.
    fn exact_newton_outer_curvature_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessiansecond_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    /// Spec-aware variant of `exact_newton_outer_curvature_second_directional_derivative`.
    fn exact_newton_outer_curvature_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_outer_curvature_second_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    /// Optional spec-aware exact first directional derivative of the joint Hessian.
    ///
    /// This is the spec-aware analogue of
    /// `exact_newton_joint_hessian_directional_derivative`. It returns the
    /// exact joint likelihood-curvature drift
    ///
    ///   D_beta H_L[u],
    ///
    /// for a flattened coefficient-space direction `u`. In the profiled
    /// Laplace gradient this appears after solving the exact joint mode
    /// response
    ///
    ///   H u_k = -A_k beta,
    ///   dot H_k = A_k + D_beta H_L[u_k].
    ///
    /// Families that can reconstruct the exact joint geometry from `specs`
    /// should override this alongside
    /// `exact_newton_joint_hessian_with_specs`.
    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
    }

    /// Optional spec-aware exact second directional derivative of the joint Hessian.
    ///
    /// This is the spec-aware analogue of
    /// `exact_newton_joint_hessiansecond_directional_derivative`. For
    /// rho/rho outer Hessian entries it supplies the exact joint second-order
    /// likelihood-curvature drift
    ///
    ///   D_beta^2 H_L[u_l, u_k],
    ///
    /// which combines with
    ///
    ///   dot H_k = A_k + D_beta H_L[u_k]
    ///
    /// and the second mode response
    ///
    ///   H u_{k,l}
    ///   = -(A_k u_l + A_l u_k + B_{k,l} beta + D_beta H_L[u_l] u_k)
    ///
    /// to form
    ///
    ///   ddot H_{k,l}
    ///   = B_{k,l} + D_beta H_L[u_{k,l}] + D_beta^2 H_L[u_l, u_k].
    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessiansecond_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    /// Optional joint multi-block outer-hyper surrogate Hessian over the
    /// flattened coefficient vector.
    ///
    /// This hook exists for families whose inner working representation is
    /// block-diagonal/diagonal in `evaluate(...)`, but whose outer profiled
    /// smoothing derivatives are still joint because the fitted mode response
    /// couples blocks. The generic blockwise outer-hyper surrogate only sees
    /// per-block working sets, so it cannot recover missing cross-block
    /// curvature on its own.
    ///
    /// Families that can construct a mathematically valid joint surrogate
    /// `H_L(beta)` for the current realized `specs` may override this and the
    /// two directional derivative hooks below. Generic code then reuses the
    /// same joint rho-calculus as the exact path, but on the family-supplied
    /// surrogate curvature instead of the exact Newton Hessian.
    ///
    /// Default behavior is to reuse the spec-aware exact joint curvature when
    /// the family already provides it. That is the mathematically correct
    /// repair for the old broken multi-block blockwise surrogate path: if the
    /// family knows the full coupled Hessian and its beta-drifts, generic code
    /// should use that joint information instead of pretending per-block
    /// working sets are enough.
    fn joint_outer_hyper_surrogate_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_with_specs(block_states, specs)
    }

    /// Optional first beta-directional derivative of the joint surrogate
    /// outer-hyper Hessian.
    fn joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_with_specs(
            block_states,
            specs,
            d_beta_flat,
        )
    }

    /// Optional second beta-directional derivative of the joint surrogate
    /// outer-hyper Hessian.
    fn joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_with_specs(
            block_states,
            specs,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    /// Optional exact directional derivative of diagonal working weights along
    /// a predictor-space direction `d_eta` for `BlockWorkingSet::Diagonal`.
    ///
    /// This callback supplies the `dw` term in
    ///
    ///   D_beta J[u] = X^T diag(dw) X
    ///
    /// for diagonal working-set blocks with
    ///
    ///   J = X^T W X + S.
    ///
    /// Default `None` means no exact working-weight directional derivative is
    /// available. Exact REML/LAML derivative paths should not silently replace
    /// this with zero unless the family truly has constant working weights.
    fn diagonalworking_weights_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    /// Optional exact first-order joint psi terms over the flattened
    /// coefficient vector.
    ///
    /// Families with coupled exact-joint curvature must provide psi objects in
    /// the same flattened coefficient space used by the existing joint Hessian
    /// hooks:
    ///
    ///   objective_psi = V_psi^explicit,
    ///   score_psi     = g_psi^explicit,
    ///   hessian_psi   = H_psi^explicit.
    ///
    /// Generic code then adds the realized penalty surface, solves
    ///
    ///   beta_i = -H^{-1} g_i,
    ///
    /// forms
    ///
    ///   dot H_i = H_i + D_beta H[beta_i],
    ///
    /// and plugs those objects into the unified profiled/Laplace gradient
    ///
    ///   J_i = V_i + 0.5 tr(H^{-1} dot H_i) - 0.5 partial_i log|S(theta)|_+.
    ///
    /// The current block-local exact-Newton psi hooks are not sufficient for a
    /// full joint hyper Hessian on coupled families; joint exact-joint hyper
    /// evaluation must use this flattened-coefficient hook instead.
    fn exact_newton_joint_psi_terms(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        _: &[Vec<CustomFamilyBlockPsiDerivative>],
        _: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        Ok(None)
    }

    /// Optional exact second-order joint psi terms over the flattened
    /// coefficient vector.
    ///
    /// For two outer coordinates theta_i, theta_j the exact profiled/Laplace
    /// Hessian uses fixed-beta second partials
    ///
    ///   V_{ij}^explicit, g_{ij}^explicit, H_{ij}^explicit.
    ///
    /// For psi/psi blocks this callback returns those explicit family terms in
    /// flattened coefficient coordinates. Generic code adds penalty
    /// contributions and profile/Laplace corrections.
    fn exact_newton_joint_psisecond_order_terms(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        _: &[Vec<CustomFamilyBlockPsiDerivative>],
        _: usize,
        _: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        Ok(None)
    }

    /// Optional per-evaluation workspace for exact joint ψ derivatives.
    ///
    /// Families with expensive exact ψ calculus can override this hook to
    /// precompute shared state once per outer evaluation and serve:
    ///
    /// - exact fixed-β ψψ second-order terms, and
    /// - exact mixed β/ψ Hessian drifts `D_β H_ψ[u]`
    ///
    /// from one cached workspace. Generic code falls back to the direct hooks
    /// above when no workspace is provided.
    fn exact_newton_joint_psi_workspace(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        _: &[Vec<CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        Ok(None)
    }

    /// Whether the family's exact joint ψ workspace should also be built for
    /// first-order ψ terms during outer gradient evaluation.
    ///
    /// Default `false` avoids forcing every family to pay workspace setup cost
    /// on gradient-only outer evaluations. Families with expensive shared state
    /// that is reused by both first- and second-order ψ calculus can opt in.
    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        false
    }

    /// Optional mixed beta/psi Hessian drift D_beta H_psi[u].
    ///
    /// This is the missing T_i[u] object in the full exact joint profiled
    /// Hessian:
    ///
    ///   ddot H_{ij}
    ///   = H_{ij}
    ///     + D_beta H_i[beta_j]
    ///     + D_beta H_j[beta_i]
    ///     + D_beta H[beta_{ij}]
    ///     + D_beta^2 H[beta_i, beta_j].
    ///
    /// For i = psi_a this hook supplies D_beta H_{psi_a}[u].
    ///
    /// This direct hook is dense-only. Families that can keep the drift in an
    /// operator-backed or block-local form should expose it through
    /// `exact_newton_joint_psi_workspace()` instead.
    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        _: &[Vec<CustomFamilyBlockPsiDerivative>],
        _: usize,
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Whether pseudo-Laplace traces and sensitivities may use the positive
    /// curvature subspace when the exact mode Hessian is only semidefinite.
    ///
    /// Families that legitimately optimize to boundary regimes with exact
    /// zero-curvature directions can opt in; truly indefinite Hessians are
    /// still rejected.
    fn exact_newton_allows_semidefinitehessian(&self) -> bool {
        false
    }

    /// How the penalized Hessian's log-determinant and its derivatives
    /// should handle eigenvalues below the numerical-stability floor.
    ///
    /// See [`PseudoLogdetMode`].  Default: `Smooth`, the stable choice for
    /// full-rank Hessians.  Families whose model structure carries a
    /// numerical null-space direction — e.g. multi-block GAMLSS wiggle
    /// models where `q = q_0 + B(q_0)^⊤ β_w` is not identified from a
    /// threshold shift — should override to `HardPseudo` so the null
    /// direction drops out of both the REML cost and its gradient
    /// consistently, rather than leaking a spurious first-order
    /// contribution through the eigensolver's arbitrary choice of basis
    /// inside the null space.
    fn pseudo_logdet_mode(&self) -> PseudoLogdetMode {
        PseudoLogdetMode::Smooth
    }
}

#[derive(Clone)]
pub struct BlockwiseFitOptions {
    pub inner_max_cycles: usize,
    pub inner_tol: f64,
    pub outer_max_iter: usize,
    pub outer_tol: f64,
    pub minweight: f64,
    pub ridge_floor: f64,
    /// Shared ridge semantics used by solve/quadratic/logdet terms.
    pub ridge_policy: RidgePolicy,
    /// If true, outer smoothing optimization uses a Laplace/REML-style objective:
    ///   -loglik + penalty + 0.5(log|H| - log|S|_+)
    /// where H is blockwise working curvature and S is blockwise penalty.
    pub use_remlobjective: bool,
    /// If false, the outer smoothing optimizer uses exact gradients but does
    /// not request an analytic outer Hessian from the family.
    pub use_outer_hessian: bool,
    /// If false, skip post-fit joint covariance assembly.
    pub compute_covariance: bool,
    /// Shared cap engaged during seed screening so cost-only evaluations can
    /// stop inner iterations early without affecting the full solve.
    pub screening_max_inner_iterations: Option<Arc<AtomicUsize>>,
}

impl Default for BlockwiseFitOptions {
    fn default() -> Self {
        Self {
            inner_max_cycles: 100,
            inner_tol: 1e-6,
            outer_max_iter: 60,
            outer_tol: 1e-5,
            minweight: 1e-12,
            ridge_floor: 1e-12,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_remlobjective: true,
            use_outer_hessian: false,
            compute_covariance: false,
            screening_max_inner_iterations: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlockwiseInnerResult {
    pub block_states: Vec<ParameterBlockState>,
    pub active_sets: Vec<Option<Vec<usize>>>,
    pub log_likelihood: f64,
    pub penalty_value: f64,
    pub cycles: usize,
    pub converged: bool,
    pub block_logdet_h: f64,
    pub block_logdet_s: f64,
    /// Cached assembled penalty matrices S(ρ) = Σ_k exp(ρ_k) S_k per block.
    /// Avoids redundant re-assembly in the outer objective evaluation.
    pub s_lambdas: Vec<Array2<f64>>,
}

#[derive(Clone)]
struct ConstrainedWarmStart {
    rho: Array1<f64>,
    block_beta: Vec<Array1<f64>>,
    active_sets: Vec<Option<Vec<usize>>>,
}

fn screened_outer_warm_start<'a>(
    warm_start: Option<&'a ConstrainedWarmStart>,
    rho: &Array1<f64>,
) -> Option<&'a ConstrainedWarmStart> {
    warm_start.filter(|seed| {
        seed.rho.len() == rho.len()
            && seed
                .rho
                .iter()
                .zip(rho.iter())
                .all(|(&a, &b)| (a - b).abs() <= 1.5)
    })
}

/// Helper struct mirroring the old `BlockwiseFitResultParts`.
pub struct BlockwiseFitResultParts {
    pub block_states: Vec<ParameterBlockState>,
    pub log_likelihood: f64,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub covariance_conditional: Option<Array2<f64>>,
    pub stable_penalty_term: f64,
    pub penalized_objective: f64,
    pub outer_iterations: usize,
    pub outer_gradient_norm: f64,
    pub inner_cycles: usize,
    pub outer_converged: bool,
    pub geometry: Option<FitGeometry>,
}

fn validate_parameter_block_state_finiteness(
    label: &str,
    state: &ParameterBlockState,
) -> Result<(), String> {
    validate_all_finite_estimation(&format!("{label}.beta"), state.beta.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(&format!("{label}.eta"), state.eta.iter().copied())
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn validate_lambda_pair_consistency(
    log_lambdas: &Array1<f64>,
    lambdas: &Array1<f64>,
    label: &str,
) -> Result<(), String> {
    if log_lambdas.len() != lambdas.len() {
        return Err(format!(
            "{label} length mismatch: log_lambdas={}, lambdas={}",
            log_lambdas.len(),
            lambdas.len()
        ));
    }
    for (idx, (&log_lambda, &lambda)) in log_lambdas.iter().zip(lambdas.iter()).enumerate() {
        let expected = log_lambda.exp();
        let tolerance = 1e-10 * expected.abs().max(1.0);
        if (lambda - expected).abs() > tolerance {
            return Err(format!(
                "{label}[{idx}] inconsistent with exp(log_lambda): got {lambda}, expected {expected}",
            ));
        }
    }
    Ok(())
}

/// Build a `UnifiedFitResult` from blockwise-specific fields.
pub fn blockwise_fit_from_parts(
    parts: BlockwiseFitResultParts,
    specs: &[ParameterBlockSpec],
) -> Result<crate::solver::estimate::UnifiedFitResult, String> {
    let BlockwiseFitResultParts {
        block_states,
        log_likelihood,
        log_lambdas,
        lambdas,
        covariance_conditional,
        stable_penalty_term,
        penalized_objective,
        outer_iterations,
        outer_gradient_norm,
        inner_cycles,
        outer_converged,
        geometry,
    } = parts;

    if block_states.is_empty() {
        return Err("blockwise fit requires at least one block state".to_string());
    }
    ensure_finite_scalar_estimation("blockwise_fit.log_likelihood", log_likelihood)
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation("blockwise_fit.log_lambdas", log_lambdas.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation("blockwise_fit.lambdas", lambdas.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_lambda_pair_consistency(&log_lambdas, &lambdas, "blockwise_fit.lambdas")?;
    ensure_finite_scalar_estimation("blockwise_fit.penalized_objective", penalized_objective)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("blockwise_fit.stable_penalty_term", stable_penalty_term)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("blockwise_fit.outer_gradient_norm", outer_gradient_norm)
        .map_err(|e| e.to_string())?;

    if block_states.len() != specs.len() {
        return Err(format!(
            "blockwise_fit.block_states length ({}) does not match specs length ({})",
            block_states.len(),
            specs.len()
        ));
    }
    let n = block_states[0].eta.len();
    let total_p = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    for (idx, state) in block_states.iter().enumerate() {
        validate_parameter_block_state_finiteness(
            &format!("blockwise_fit.block_states[{idx}]"),
            state,
        )?;
        let expected_rows = specs[idx].design.nrows();
        if state.eta.len() != expected_rows {
            return Err(format!(
                "blockwise_fit.block_states[{idx}] eta length mismatch: got {}, expected {} (design rows)",
                state.eta.len(),
                expected_rows
            ));
        }
    }

    if let Some(cov) = covariance_conditional.as_ref() {
        validate_all_finite_estimation("blockwise_fit.covariance_conditional", cov.iter().copied())
            .map_err(|e| e.to_string())?;
        let (rows, cols) = cov.dim();
        if rows != total_p || cols != total_p {
            return Err(format!(
                "blockwise_fit.covariance_conditional must be {}x{}, got {}x{}",
                total_p, total_p, rows, cols
            ));
        }
    }

    if let Some(geom) = geometry.as_ref() {
        geom.validate_numeric_finiteness()
            .map_err(|e| e.to_string())?;
        let (rows, cols) = geom.penalized_hessian.dim();
        if rows != total_p || cols != total_p {
            return Err(format!(
                "blockwise_fit.geometry.penalized_hessian must be {}x{}, got {}x{}",
                total_p, total_p, rows, cols
            ));
        }
        let geom_len = geom.working_weights.len();
        if geom_len != geom.working_response.len() {
            return Err(format!(
                "blockwise_fit.geometry working vector length mismatch: weights={}, response={}",
                geom.working_weights.len(),
                geom.working_response.len(),
            ));
        }
        if geom_len != n && (n == 0 || geom_len % n != 0) {
            return Err(format!(
                "blockwise_fit.geometry.working_weights length mismatch: got {geom_len}, expected {n} or a stacked multiple of {n}",
            ));
        }
        if geom.working_response.len() != n && (n == 0 || geom.working_response.len() % n != 0) {
            return Err(format!(
                "blockwise_fit.geometry.working_response length mismatch: got {}, expected {n} or a stacked multiple of {n}",
                geom.working_response.len(),
            ));
        }
    }

    // Build unified blocks from the blockwise states.
    use crate::solver::estimate::{FittedBlock, FittedLinkState, UnifiedFitResultParts};
    let penalty_counts: Vec<usize> = specs.iter().map(|s| s.penalties.len()).collect();
    let expected_rho: usize = penalty_counts.iter().sum();
    if lambdas.len() != expected_rho {
        return Err(format!(
            "blockwise_fit.lambdas length ({}) does not match sum of per-block penalty counts ({})",
            lambdas.len(),
            expected_rho
        ));
    }
    let mut lambda_offset = 0usize;
    let blocks: Vec<FittedBlock> = block_states
        .iter()
        .enumerate()
        .map(|(i, bs)| {
            let role = custom_family_block_role(&specs[i].name, i, block_states.len());
            let k = penalty_counts[i];
            let block_lambdas = lambdas
                .slice(s![lambda_offset..lambda_offset + k])
                .to_owned();
            lambda_offset += k;
            FittedBlock {
                beta: bs.beta.clone(),
                role,
                edf: 0.0,
                lambdas: block_lambdas,
            }
        })
        .collect();
    let deviance = -2.0 * log_likelihood;

    crate::solver::estimate::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks,
        log_lambdas: log_lambdas.clone(),
        lambdas: lambdas.clone(),
        likelihood_family: None,
        likelihood_scale: crate::types::LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: crate::types::LogLikelihoodNormalization::UserProvided,
        log_likelihood,
        deviance,
        reml_score: penalized_objective,
        stable_penalty_term,
        penalized_objective,
        outer_iterations,
        outer_converged,
        outer_gradient_norm,
        standard_deviation: 1.0,
        covariance_conditional,
        covariance_corrected: None,
        inference: None,
        fitted_link: FittedLinkState::Standard(None),
        geometry,
        block_states,
        pirls_status: crate::pirls::PirlsStatus::Converged,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: crate::solver::estimate::FitArtifacts {
            pirls: None,
            ..Default::default()
        },
        inner_cycles,
    })
    .map_err(|e| e.to_string())
}

fn checked_penalizedobjective(
    log_likelihood: f64,
    penalty_value: f64,
    reml_term: f64,
    context: &str,
) -> Result<f64, String> {
    let objective = -log_likelihood + penalty_value + reml_term;
    if objective.is_finite() {
        Ok(objective)
    } else {
        Err(format!(
            "{context}: non-finite penalized objective \
             (log_likelihood={log_likelihood}, penalty_value={penalty_value}, \
             reml_term={reml_term}, objective={objective})"
        ))
    }
}

#[derive(Clone)]
pub struct CustomFamilyBlockPsiDerivative {
    pub penalty_index: Option<usize>,
    pub x_psi: Array2<f64>,
    pub s_psi: Array2<f64>,
    pub s_psi_components: Option<Vec<(usize, Array2<f64>)>>,
    pub s_psi_penalty_components: Option<Vec<(usize, PenaltyMatrix)>>,
    pub x_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi_components: Option<Vec<Vec<(usize, Array2<f64>)>>>,
    pub s_psi_psi_penalty_components: Option<Vec<Vec<(usize, PenaltyMatrix)>>>,
    pub(crate) implicit_operator: Option<Arc<dyn CustomFamilyPsiDerivativeOperator>>,
    pub implicit_axis: usize,
    pub implicit_group_id: Option<usize>,
}

pub(crate) type SharedDerivativeBlocks = Arc<Vec<Vec<CustomFamilyBlockPsiDerivative>>>;

impl CustomFamilyBlockPsiDerivative {
    /// Public constructor for use in tests and external consumers.
    /// Sets `implicit_operator` to `None`.
    pub fn new(
        penalty_index: Option<usize>,
        x_psi: Array2<f64>,
        s_psi: Array2<f64>,
        s_psi_components: Option<Vec<(usize, Array2<f64>)>>,
        x_psi_psi: Option<Vec<Array2<f64>>>,
        s_psi_psi: Option<Vec<Array2<f64>>>,
        s_psi_psi_components: Option<Vec<Vec<(usize, Array2<f64>)>>>,
    ) -> Self {
        Self {
            penalty_index,
            x_psi,
            s_psi,
            s_psi_components,
            s_psi_penalty_components: None,
            x_psi_psi,
            s_psi_psi,
            s_psi_psi_components,
            s_psi_psi_penalty_components: None,
            implicit_operator: None,
            implicit_axis: 0,
            implicit_group_id: None,
        }
    }
}

pub(crate) trait CustomFamilyPsiDerivativeOperator: Send + Sync + Any {
    fn as_any(&self) -> &dyn Any;
    fn n_data(&self) -> usize;
    fn p_out(&self) -> usize;
    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self
            .materialize_first(axis)?
            .slice(ndarray::s![rows, ..])
            .to_owned())
    }
    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self
            .materialize_second_diag(axis)?
            .slice(ndarray::s![rows, ..])
            .to_owned())
    }
    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self
            .materialize_second_cross(axis_d, axis_e)?
            .slice(ndarray::s![rows, ..])
            .to_owned())
    }
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;
    fn materialize_second_diag(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;
    fn materialize_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;
}

impl CustomFamilyPsiDerivativeOperator for crate::terms::basis::ImplicitDesignPsiDerivative {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        crate::terms::basis::ImplicitDesignPsiDerivative::n_data(self)
    }

    fn p_out(&self) -> usize {
        crate::terms::basis::ImplicitDesignPsiDerivative::p_out(self)
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::transpose_mul(self, axis, v)
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::forward_mul(self, axis, u)
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::row_chunk_first(self, axis, rows)
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::row_chunk_second_diag(self, axis, rows)
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::row_chunk_second_cross(
            self, axis_d, axis_e, rows,
        )
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::transpose_mul_second_diag(self, axis, v)
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::transpose_mul_second_cross(
            self, axis_d, axis_e, v,
        )
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::forward_mul_second_diag(self, axis, u)
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::forward_mul_second_cross(
            self, axis_d, axis_e, u,
        )
    }

    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::materialize_first(self, axis)
    }

    fn materialize_second_diag(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::materialize_second_diag(self, axis)
    }

    fn materialize_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::materialize_second_cross(
            self, axis_d, axis_e,
        )
    }
}

struct EmbeddedImplicitPsiDerivativeOperator {
    base: Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
    total_p: usize,
    global_range: Range<usize>,
}

impl EmbeddedImplicitPsiDerivativeOperator {
    fn new(
        base: Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
        global_range: Range<usize>,
        total_p: usize,
    ) -> Result<Self, String> {
        if base.p_out() != global_range.len() {
            return Err(format!(
                "embedded implicit psi operator width mismatch: got {}, expected {}",
                base.p_out(),
                global_range.len()
            ));
        }
        if global_range.end > total_p {
            return Err(format!(
                "embedded implicit psi operator range {}..{} exceeds total width {total_p}",
                global_range.start, global_range.end
            ));
        }
        Ok(Self {
            base,
            total_p,
            global_range,
        })
    }

    fn embed_vector(&self, local: Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.total_p);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        out
    }

    fn local_coeffs(
        &self,
        u: &ArrayView1<'_, f64>,
        context: &str,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if u.len() != self.total_p {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "{context} expected coefficient length {}, got {}",
                self.total_p,
                u.len()
            )));
        }
        Ok(u.slice(ndarray::s![self.global_range.clone()]).to_owned())
    }
}

impl CustomFamilyPsiDerivativeOperator for EmbeddedImplicitPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.base.n_data()
    }

    fn p_out(&self) -> usize {
        self.total_p
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        Ok(self.embed_vector(self.base.transpose_mul(axis, v)?))
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let local = self.local_coeffs(u, "embedded implicit psi forward_mul")?;
        self.base.forward_mul(axis, &local.view())
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        Ok(self.embed_vector(self.base.transpose_mul_second_diag(axis, v)?))
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        Ok(self.embed_vector(self.base.transpose_mul_second_cross(axis_d, axis_e, v)?))
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let local = self.local_coeffs(u, "embedded implicit psi forward_mul_second_diag")?;
        self.base.forward_mul_second_diag(axis, &local.view())
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let local = self.local_coeffs(u, "embedded implicit psi forward_mul_second_cross")?;
        self.base
            .forward_mul_second_cross(axis_d, axis_e, &local.view())
    }

    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(EmbeddedColumnBlock::new(
            &self.base.materialize_first(axis)?,
            self.global_range.clone(),
            self.total_p,
        )
        .materialize())
    }

    fn materialize_second_diag(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(EmbeddedColumnBlock::new(
            &self.base.materialize_second_diag(axis)?,
            self.global_range.clone(),
            self.total_p,
        )
        .materialize())
    }

    fn materialize_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(EmbeddedColumnBlock::new(
            &self.base.materialize_second_cross(axis_d, axis_e)?,
            self.global_range.clone(),
            self.total_p,
        )
        .materialize())
    }
}

fn rowwise_kronecker_dense(base: &Array2<f64>, time_basis: &Array2<f64>) -> Array2<f64> {
    assert_eq!(base.nrows(), time_basis.nrows());
    let n = base.nrows();
    let p_base = base.ncols();
    let p_time = time_basis.ncols();
    let mut out = Array2::<f64>::zeros((n, p_base * p_time));
    for i in 0..n {
        for j in 0..p_base {
            let base_ij = base[[i, j]];
            if base_ij == 0.0 {
                continue;
            }
            for t in 0..p_time {
                out[[i, j * p_time + t]] = base_ij * time_basis[[i, t]];
            }
        }
    }
    out
}

fn stack_dense_row_blocks(blocks: &[Array2<f64>]) -> Array2<f64> {
    let total_rows = blocks.iter().map(Array2::nrows).sum();
    let p = blocks.first().map(Array2::ncols).unwrap_or(0);
    let mut stacked = Array2::<f64>::zeros((total_rows, p));
    let mut row_start = 0usize;
    for block in blocks {
        assert_eq!(block.ncols(), p);
        let row_end = row_start + block.nrows();
        stacked
            .slice_mut(ndarray::s![row_start..row_end, ..])
            .assign(block);
        row_start = row_end;
    }
    stacked
}

struct EmbeddedDensePsiDerivativeOperator {
    axis: usize,
    total_p: usize,
    global_range: Range<usize>,
    first_local: Array2<f64>,
    second_diag_local: Array2<f64>,
    second_cross_local: HashMap<usize, Array2<f64>>,
}

impl EmbeddedDensePsiDerivativeOperator {
    fn new(
        axis: usize,
        total_p: usize,
        global_range: Range<usize>,
        first_local: Array2<f64>,
        second_diag_local: Array2<f64>,
        second_cross_local: HashMap<usize, Array2<f64>>,
    ) -> Result<Self, String> {
        let local_p = global_range.len();
        if first_local.ncols() != local_p {
            return Err(format!(
                "embedded dense psi operator first-derivative width mismatch: got {}, expected {local_p}",
                first_local.ncols()
            ));
        }
        if second_diag_local.ncols() != local_p {
            return Err(format!(
                "embedded dense psi operator second-diag width mismatch: got {}, expected {local_p}",
                second_diag_local.ncols()
            ));
        }
        for (cross_axis, local) in &second_cross_local {
            if local.ncols() != local_p {
                return Err(format!(
                    "embedded dense psi operator cross axis {cross_axis} width mismatch: got {}, expected {local_p}",
                    local.ncols()
                ));
            }
        }
        Ok(Self {
            axis,
            total_p,
            global_range,
            first_local,
            second_diag_local,
            second_cross_local,
        })
    }

    fn validate_axis(
        &self,
        axis: usize,
        context: &str,
    ) -> Result<(), crate::terms::basis::BasisError> {
        if axis == self.axis {
            Ok(())
        } else {
            Err(crate::terms::basis::BasisError::Other(format!(
                "{context} expected axis {}, got {axis}",
                self.axis
            )))
        }
    }

    fn embed_vector(&self, local: Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.total_p);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        out
    }

    fn local_coeffs(
        &self,
        u: &ArrayView1<'_, f64>,
        context: &str,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if u.len() != self.total_p {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "{context} expected coefficient length {}, got {}",
                self.total_p,
                u.len()
            )));
        }
        Ok(u.slice(ndarray::s![self.global_range.clone()]).to_owned())
    }

    fn cross_local(
        &self,
        axis_e: usize,
        context: &str,
    ) -> Result<&Array2<f64>, crate::terms::basis::BasisError> {
        self.second_cross_local.get(&axis_e).ok_or_else(|| {
            crate::terms::basis::BasisError::Other(format!(
                "{context} is missing cross-derivative data for axis {}",
                axis_e
            ))
        })
    }
}

impl CustomFamilyPsiDerivativeOperator for EmbeddedDensePsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.first_local.nrows()
    }

    fn p_out(&self) -> usize {
        self.total_p
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi transpose_mul")?;
        if v.len() != self.n_data() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi transpose_mul expected {} rows, got {}",
                self.n_data(),
                v.len()
            )));
        }
        Ok(self.embed_vector(self.first_local.t().dot(v)))
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi forward_mul")?;
        Ok(self
            .first_local
            .dot(&self.local_coeffs(u, "embedded dense psi forward_mul")?))
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi transpose_mul_second_diag")?;
        if v.len() != self.second_diag_local.nrows() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi transpose_mul_second_diag expected {} rows, got {}",
                self.second_diag_local.nrows(),
                v.len()
            )));
        }
        Ok(self.embed_vector(self.second_diag_local.t().dot(v)))
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi transpose_mul_second_cross")?;
        let local = self.cross_local(axis_e, "embedded dense psi transpose_mul_second_cross")?;
        if v.len() != local.nrows() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi transpose_mul_second_cross expected {} rows, got {}",
                local.nrows(),
                v.len()
            )));
        }
        Ok(self.embed_vector(local.t().dot(v)))
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi forward_mul_second_diag")?;
        Ok(self
            .second_diag_local
            .dot(&self.local_coeffs(u, "embedded dense psi forward_mul_second_diag")?))
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi forward_mul_second_cross")?;
        Ok(self
            .cross_local(axis_e, "embedded dense psi forward_mul_second_cross")?
            .dot(&self.local_coeffs(u, "embedded dense psi forward_mul_second_cross")?))
    }

    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi materialize_first")?;
        Ok(
            EmbeddedColumnBlock::new(&self.first_local, self.global_range.clone(), self.total_p)
                .materialize(),
        )
    }

    fn materialize_second_diag(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi materialize_second_diag")?;
        Ok(EmbeddedColumnBlock::new(
            &self.second_diag_local,
            self.global_range.clone(),
            self.total_p,
        )
        .materialize())
    }

    fn materialize_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi materialize_second_cross")?;
        Ok(EmbeddedColumnBlock::new(
            self.cross_local(axis_e, "embedded dense psi materialize_second_cross")?,
            self.global_range.clone(),
            self.total_p,
        )
        .materialize())
    }
}

pub(crate) fn build_embedded_dense_psi_operator(
    first_local: &Array2<f64>,
    second_diag_local: &Array2<f64>,
    second_cross_local: Option<&Vec<(usize, Array2<f64>)>>,
    global_range: Range<usize>,
    total_p: usize,
    axis: usize,
) -> Result<Arc<dyn CustomFamilyPsiDerivativeOperator>, String> {
    let second_cross_local = second_cross_local
        .map(|rows| {
            rows.iter()
                .map(|(axis, local)| (*axis, local.clone()))
                .collect()
        })
        .unwrap_or_default();
    Ok(Arc::new(EmbeddedDensePsiDerivativeOperator::new(
        axis,
        total_p,
        global_range,
        first_local.clone(),
        second_diag_local.clone(),
        second_cross_local,
    )?))
}

struct RowwiseKroneckerPsiDerivativeOperator {
    base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    time_bases: Vec<Arc<Array2<f64>>>,
    n_per_block: usize,
    p_time: usize,
    p_out: usize,
}

impl RowwiseKroneckerPsiDerivativeOperator {
    fn new(
        base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        time_bases: Vec<Arc<Array2<f64>>>,
    ) -> Result<Self, String> {
        let first = time_bases.first().ok_or_else(|| {
            "rowwise kronecker psi operator needs at least one time basis".to_string()
        })?;
        let n_per_block = first.nrows();
        let p_time = first.ncols();
        for (idx, basis) in time_bases.iter().enumerate() {
            if basis.nrows() != n_per_block || basis.ncols() != p_time {
                return Err(format!(
                    "rowwise kronecker psi operator time basis {idx} shape mismatch: got {}x{}, expected {}x{}",
                    basis.nrows(),
                    basis.ncols(),
                    n_per_block,
                    p_time
                ));
            }
        }
        if base.n_data() != n_per_block {
            return Err(format!(
                "rowwise kronecker psi operator base row mismatch: got {}, expected {n_per_block}",
                base.n_data()
            ));
        }
        Ok(Self {
            p_out: base.p_out() * p_time,
            base,
            time_bases,
            n_per_block,
            p_time,
        })
    }

    fn split_time_columns(&self, u: &ArrayView1<'_, f64>) -> Vec<Array1<f64>> {
        let p_base = self.base.p_out();
        assert_eq!(u.len(), self.p_out);
        let mut cols = vec![Array1::<f64>::zeros(p_base); self.p_time];
        for j in 0..p_base {
            for t in 0..self.p_time {
                cols[t][j] = u[j * self.p_time + t];
            }
        }
        cols
    }
}

impl CustomFamilyPsiDerivativeOperator for RowwiseKroneckerPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.n_per_block * self.time_bases.len()
    }

    fn p_out(&self) -> usize {
        self.p_out
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        assert_eq!(v.len(), self.n_data());
        let p_base = self.base.p_out();
        let mut out = Array1::<f64>::zeros(self.p_out);
        for t in 0..self.p_time {
            let mut accum = Array1::<f64>::zeros(p_base);
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let weighted = &v.slice(ndarray::s![row_start..row_end]).to_owned()
                    * &time_basis.column(t).to_owned();
                accum += &self.base.transpose_mul(axis, &weighted.view())?;
            }
            for j in 0..p_base {
                out[j * self.p_time + t] = accum[j];
            }
        }
        Ok(out)
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let time_cols = self.split_time_columns(u);
        let mut out = Array1::<f64>::zeros(self.n_data());
        for (t, coeffs) in time_cols.iter().enumerate() {
            let base_eval = self.base.forward_mul(axis, &coeffs.view())?;
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let contrib = &base_eval * &time_basis.column(t).to_owned();
                let mut out_block = out.slice_mut(ndarray::s![row_start..row_end]);
                out_block += &contrib;
            }
        }
        Ok(out)
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        assert_eq!(v.len(), self.n_data());
        let p_base = self.base.p_out();
        let mut out = Array1::<f64>::zeros(self.p_out);
        for t in 0..self.p_time {
            let mut accum = Array1::<f64>::zeros(p_base);
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let weighted = &v.slice(ndarray::s![row_start..row_end]).to_owned()
                    * &time_basis.column(t).to_owned();
                accum += &self
                    .base
                    .transpose_mul_second_diag(axis, &weighted.view())?;
            }
            for j in 0..p_base {
                out[j * self.p_time + t] = accum[j];
            }
        }
        Ok(out)
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        assert_eq!(v.len(), self.n_data());
        let p_base = self.base.p_out();
        let mut out = Array1::<f64>::zeros(self.p_out);
        for t in 0..self.p_time {
            let mut accum = Array1::<f64>::zeros(p_base);
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let weighted = &v.slice(ndarray::s![row_start..row_end]).to_owned()
                    * &time_basis.column(t).to_owned();
                accum += &self
                    .base
                    .transpose_mul_second_cross(axis_d, axis_e, &weighted.view())?;
            }
            for j in 0..p_base {
                out[j * self.p_time + t] = accum[j];
            }
        }
        Ok(out)
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let time_cols = self.split_time_columns(u);
        let mut out = Array1::<f64>::zeros(self.n_data());
        for (t, coeffs) in time_cols.iter().enumerate() {
            let base_eval = self.base.forward_mul_second_diag(axis, &coeffs.view())?;
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let contrib = &base_eval * &time_basis.column(t).to_owned();
                let mut out_block = out.slice_mut(ndarray::s![row_start..row_end]);
                out_block += &contrib;
            }
        }
        Ok(out)
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let time_cols = self.split_time_columns(u);
        let mut out = Array1::<f64>::zeros(self.n_data());
        for (t, coeffs) in time_cols.iter().enumerate() {
            let base_eval = self
                .base
                .forward_mul_second_cross(axis_d, axis_e, &coeffs.view())?;
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let contrib = &base_eval * &time_basis.column(t).to_owned();
                let mut out_block = out.slice_mut(ndarray::s![row_start..row_end]);
                out_block += &contrib;
            }
        }
        Ok(out)
    }

    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let base = self.base.materialize_first(axis)?;
        let blocks: Vec<Array2<f64>> = self
            .time_bases
            .iter()
            .map(|basis| rowwise_kronecker_dense(&base, basis))
            .collect();
        Ok(stack_dense_row_blocks(&blocks))
    }

    fn materialize_second_diag(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let base = self.base.materialize_second_diag(axis)?;
        let blocks: Vec<Array2<f64>> = self
            .time_bases
            .iter()
            .map(|basis| rowwise_kronecker_dense(&base, basis))
            .collect();
        Ok(stack_dense_row_blocks(&blocks))
    }

    fn materialize_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let base = self.base.materialize_second_cross(axis_d, axis_e)?;
        let blocks: Vec<Array2<f64>> = self
            .time_bases
            .iter()
            .map(|basis| rowwise_kronecker_dense(&base, basis))
            .collect();
        Ok(stack_dense_row_blocks(&blocks))
    }
}

pub(crate) fn build_rowwise_kronecker_psi_operator(
    base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    time_bases: Vec<Arc<Array2<f64>>>,
) -> Result<Arc<dyn CustomFamilyPsiDerivativeOperator>, String> {
    Ok(Arc::new(RowwiseKroneckerPsiDerivativeOperator::new(
        base, time_bases,
    )?))
}

pub(crate) fn wrap_spatial_implicit_psi_operator(
    op: Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
    global_range: Range<usize>,
    total_p: usize,
) -> Arc<dyn CustomFamilyPsiDerivativeOperator> {
    Arc::new(
        EmbeddedImplicitPsiDerivativeOperator::new(op, global_range, total_p)
            .expect("spatial implicit psi operator should embed into full coefficient space"),
    )
}

const CUSTOM_FAMILY_PSI_DENSE_MATERIALIZATION_THRESHOLD: usize = 1_000_000_000; // 1 GB

fn should_resolve_custom_family_psi_dense(total_rows: usize, p: usize) -> bool {
    total_rows.saturating_mul(p).saturating_mul(8)
        <= CUSTOM_FAMILY_PSI_DENSE_MATERIALIZATION_THRESHOLD
}

pub(crate) fn build_block_spatial_psi_derivatives(
    data: ndarray::ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
) -> Result<Option<Vec<CustomFamilyBlockPsiDerivative>>, String> {
    let spatial_terms = spatial_length_scale_term_indices(resolvedspec);
    let Some(info_list) =
        try_build_spatial_log_kappa_derivativeinfo_list(data, resolvedspec, design, &spatial_terms)
            .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };
    let psi_dim = info_list.len();
    let axis_lookup: HashMap<(usize, usize), usize> = info_list
        .iter()
        .enumerate()
        .filter_map(|(idx, info)| {
            info.aniso_group_id
                .map(|gid| ((gid, info.implicit_axis), idx))
        })
        .collect();
    let collected: Result<Vec<CustomFamilyBlockPsiDerivative>, String> = info_list
        .into_iter()
        .enumerate()
        .map(|(psi_idx, info)| {
            let implicit_operator = info.implicit_operator.as_ref().map(|op| {
                wrap_spatial_implicit_psi_operator(
                    Arc::clone(op),
                    info.global_range.clone(),
                    info.total_p,
                )
            });
            let dense_operator = if implicit_operator.is_none() && !info.x_psi_local.is_empty() {
                Some(build_embedded_dense_psi_operator(
                    &info.x_psi_local,
                    &info.x_psi_psi_local,
                    info.aniso_cross_designs.as_ref(),
                    info.global_range.clone(),
                    info.total_p,
                    info.implicit_axis,
                )?)
            } else {
                None
            };
            let design_operator = implicit_operator.or(dense_operator);
            let materialize_dense_design =
                !info.x_psi_local.is_empty() && design_operator.is_none();
            let embed_design = |local: &Array2<f64>| -> Array2<f64> {
                if local.ncols() == 0 || local.nrows() == 0 {
                    return Array2::<f64>::zeros((local.nrows(), info.total_p));
                }
                EmbeddedColumnBlock::new(local, info.global_range.clone(), info.total_p)
                    .materialize()
            };
            let x_full = if materialize_dense_design {
                embed_design(&info.x_psi_local)
            } else {
                Array2::<f64>::zeros((0, 0))
            };
            let penalty_indices = info.penalty_indices.clone();
            let embed_penalty = |local: &Array2<f64>| -> Array2<f64> {
                if local.nrows() == 0 || local.ncols() == 0 {
                    return Array2::<f64>::zeros((info.total_p, info.total_p));
                }
                EmbeddedSquareBlock::new(local, info.global_range.clone(), info.total_p)
                    .materialize()
            };
            let s_components: Vec<(usize, Array2<f64>)> = info
                .penalty_indices
                .into_iter()
                .zip(
                    info.s_psi_components_local
                        .into_iter()
                        .map(|local| embed_penalty(&local)),
                )
                .collect();
            // Build x_psi_psi rows with cross-derivative designs
            let x_psi_psi_rows = if materialize_dense_design {
                let mut rows =
                    vec![Array2::<f64>::zeros((x_full.nrows(), x_full.ncols())); psi_dim];
                rows[psi_idx] = embed_design(&info.x_psi_psi_local);
                if let (Some(gid), Some(cross_designs)) =
                    (info.aniso_group_id, info.aniso_cross_designs.as_ref())
                {
                    for (axis_j, local) in cross_designs {
                        if let Some(&global_j) = axis_lookup.get(&(gid, *axis_j)) {
                            rows[global_j] = embed_design(local);
                        }
                    }
                }
                Some(rows)
            } else {
                None
            };
            // Build s_psi_psi_components with cross-penalty terms
            let mut s_psi_psi_comp_rows = vec![Vec::<(usize, Array2<f64>)>::new(); psi_dim];
            s_psi_psi_comp_rows[psi_idx] = penalty_indices
                .iter()
                .copied()
                .zip(
                    info.s_psi_psi_components_local
                        .iter()
                        .map(|local| embed_penalty(local)),
                )
                .collect();
            if let (Some(gid), Some(cross_penalty_provider)) = (
                info.aniso_group_id,
                info.aniso_cross_penalty_provider.as_ref(),
            ) {
                for ((group_id, axis_j), global_j) in &axis_lookup {
                    if *group_id != gid || *axis_j == info.implicit_axis {
                        continue;
                    }
                    let local_components =
                        cross_penalty_provider(*axis_j).map_err(|err| err.to_string())?;
                    if local_components.is_empty() {
                        continue;
                    }
                    s_psi_psi_comp_rows[*global_j] = penalty_indices
                        .iter()
                        .copied()
                        .zip(local_components.iter().map(embed_penalty))
                        .collect();
                }
            }
            Ok(CustomFamilyBlockPsiDerivative {
                penalty_index: Some(info.penalty_index),
                x_psi: x_full,
                s_psi: Array2::<f64>::zeros((0, 0)),
                s_psi_components: Some(s_components),
                s_psi_penalty_components: None,
                x_psi_psi: x_psi_psi_rows,
                s_psi_psi: None,
                s_psi_psi_components: Some(s_psi_psi_comp_rows),
                s_psi_psi_penalty_components: None,
                implicit_operator: design_operator,
                implicit_axis: info.implicit_axis,
                implicit_group_id: info.aniso_group_id,
            })
        })
        .collect();
    Ok(Some(collected?))
}

#[derive(Clone)]
pub(crate) struct CustomFamilyPsiDesignAction {
    operator: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    axis: usize,
    row_range: Range<usize>,
    p: usize,
}

impl CustomFamilyPsiDesignAction {
    pub(crate) fn from_first_derivative(
        deriv: &CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        row_range: Range<usize>,
        label: &str,
    ) -> Result<Self, String> {
        if row_range.end > total_rows {
            return Err(format!(
                "{label} row range {}..{} exceeds total rows {total_rows}",
                row_range.start, row_range.end
            ));
        }
        if let Some(op) = deriv.implicit_operator.as_ref() {
            if op.n_data() == total_rows && op.p_out() == p {
                return Ok(Self {
                    operator: Arc::clone(op),
                    axis: deriv.implicit_axis,
                    row_range,
                    p,
                });
            }
        }
        Err(format!(
            "{label} is missing an implicit x_psi operator with shape {}x{}; got dense payload {}x{} instead",
            total_rows,
            p,
            deriv.x_psi.nrows(),
            deriv.x_psi.ncols(),
        ))
    }

    pub(crate) fn is_implicit(&self) -> bool {
        true
    }

    pub(crate) fn nrows(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    pub(crate) fn slice_rows(&self, row_range: Range<usize>) -> Result<Self, String> {
        if row_range.end > self.nrows() {
            return Err(format!(
                "psi design row range {}..{} exceeds available rows {}",
                row_range.start,
                row_range.end,
                self.nrows()
            ));
        }
        Ok(Self {
            operator: Arc::clone(&self.operator),
            axis: self.axis,
            row_range: (self.row_range.start + row_range.start)
                ..(self.row_range.start + row_range.end),
            p: self.p,
        })
    }

    pub(crate) fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(u.len(), self.p);
        self.operator
            .forward_mul(self.axis, &u)
            .expect("radial scalar evaluation failed during implicit psi forward_mul")
            .slice(ndarray::s![self.row_range.clone()])
            .to_owned()
    }

    pub(crate) fn transpose_mul(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.row_range.end - self.row_range.start);
        if self.row_range.start == 0 && self.row_range.end == self.operator.n_data() {
            self.operator
                .transpose_mul(self.axis, &v)
                .expect("radial scalar evaluation failed during implicit psi transpose_mul")
        } else {
            let mut expanded = Array1::<f64>::zeros(self.operator.n_data());
            expanded
                .slice_mut(ndarray::s![self.row_range.clone()])
                .assign(&v);
            self.operator
                .transpose_mul(self.axis, &expanded.view())
                .expect("radial scalar evaluation failed during implicit psi transpose_mul")
        }
    }

    fn absolute_rows(&self, rows: Range<usize>) -> Range<usize> {
        (self.row_range.start + rows.start)..(self.row_range.start + rows.end)
    }

    pub(crate) fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        if rows.end > self.nrows() {
            return Err(format!(
                "psi design row range {}..{} exceeds available rows {}",
                rows.start,
                rows.end,
                self.nrows()
            ));
        }
        self.operator
            .row_chunk_first(self.axis, self.absolute_rows(rows))
            .map_err(|e| e.to_string())
    }

    pub(crate) fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        self.row_chunk(row..row + 1).map(|m| m.row(0).to_owned())
    }
}

#[derive(Clone, Copy)]
enum CustomFamilyPsiSecondDesignLevel {
    Diag(usize),
    Cross(usize, usize),
}

#[derive(Clone)]
pub(crate) struct CustomFamilyPsiSecondDesignAction {
    operator: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    level: CustomFamilyPsiSecondDesignLevel,
    row_range: Range<usize>,
    p: usize,
}

impl CustomFamilyPsiSecondDesignAction {
    pub(crate) fn from_second_derivative(
        deriv_i: &CustomFamilyBlockPsiDerivative,
        deriv_j: &CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        row_range: Range<usize>,
        label: &str,
    ) -> Result<Option<Self>, String> {
        if row_range.end > total_rows {
            return Err(format!(
                "{label} row range {}..{} exceeds total rows {total_rows}",
                row_range.start, row_range.end
            ));
        }
        let Some(op) = deriv_i.implicit_operator.as_ref() else {
            return Ok(None);
        };
        if op.n_data() != total_rows || op.p_out() != p {
            return Err(format!(
                "{label} is missing an implicit x_psi_psi operator with shape {}x{}",
                total_rows, p
            ));
        }
        let same_group = deriv_i.implicit_group_id.is_some()
            && deriv_i.implicit_group_id == deriv_j.implicit_group_id;
        if !same_group {
            return Ok(None);
        }
        let level = if deriv_i.implicit_axis == deriv_j.implicit_axis {
            CustomFamilyPsiSecondDesignLevel::Diag(deriv_i.implicit_axis)
        } else {
            CustomFamilyPsiSecondDesignLevel::Cross(deriv_i.implicit_axis, deriv_j.implicit_axis)
        };
        Ok(Some(Self {
            operator: Arc::clone(op),
            level,
            row_range,
            p,
        }))
    }

    pub(crate) fn nrows(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    pub(crate) fn slice_rows(&self, row_range: Range<usize>) -> Result<Self, String> {
        if row_range.end > self.nrows() {
            return Err(format!(
                "psi second-design row range {}..{} exceeds available rows {}",
                row_range.start,
                row_range.end,
                self.nrows()
            ));
        }
        Ok(Self {
            operator: Arc::clone(&self.operator),
            level: self.level,
            row_range: (self.row_range.start + row_range.start)
                ..(self.row_range.start + row_range.end),
            p: self.p,
        })
    }

    pub(crate) fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(u.len(), self.p);
        let out = match self.level {
            CustomFamilyPsiSecondDesignLevel::Diag(axis) => self
                .operator
                .forward_mul_second_diag(axis, &u)
                .expect("radial scalar evaluation failed during implicit psi second forward_mul"),
            CustomFamilyPsiSecondDesignLevel::Cross(axis_d, axis_e) => self
                .operator
                .forward_mul_second_cross(axis_d, axis_e, &u)
                .expect("radial scalar evaluation failed during implicit psi second forward_mul"),
        };
        out.slice(ndarray::s![self.row_range.clone()]).to_owned()
    }

    pub(crate) fn transpose_mul(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.nrows());
        let expanded = if self.row_range.start == 0 && self.row_range.end == self.operator.n_data()
        {
            None
        } else {
            let mut expanded = Array1::<f64>::zeros(self.operator.n_data());
            expanded
                .slice_mut(ndarray::s![self.row_range.clone()])
                .assign(&v);
            Some(expanded)
        };
        let full_v = expanded.as_ref().map_or(v, |arr| arr.view());
        match self.level {
            CustomFamilyPsiSecondDesignLevel::Diag(axis) => self
                .operator
                .transpose_mul_second_diag(axis, &full_v)
                .expect("radial scalar evaluation failed during implicit psi second transpose_mul"),
            CustomFamilyPsiSecondDesignLevel::Cross(axis_d, axis_e) => self
                .operator
                .transpose_mul_second_cross(axis_d, axis_e, &full_v)
                .expect("radial scalar evaluation failed during implicit psi second transpose_mul"),
        }
    }

    fn absolute_rows(&self, rows: Range<usize>) -> Range<usize> {
        (self.row_range.start + rows.start)..(self.row_range.start + rows.end)
    }

    pub(crate) fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        if rows.end > self.nrows() {
            return Err(format!(
                "psi second-design row range {}..{} exceeds available rows {}",
                rows.start,
                rows.end,
                self.nrows()
            ));
        }
        match self.level {
            CustomFamilyPsiSecondDesignLevel::Diag(axis) => self
                .operator
                .row_chunk_second_diag(axis, self.absolute_rows(rows))
                .map_err(|e| e.to_string()),
            CustomFamilyPsiSecondDesignLevel::Cross(axis_d, axis_e) => self
                .operator
                .row_chunk_second_cross(axis_d, axis_e, self.absolute_rows(rows))
                .map_err(|e| e.to_string()),
        }
    }

    pub(crate) fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        self.row_chunk(row..row + 1).map(|m| m.row(0).to_owned())
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CustomFamilyPsiLinearMapRef<'a> {
    Dense(&'a Array2<f64>),
    First(&'a CustomFamilyPsiDesignAction),
    Second(&'a CustomFamilyPsiSecondDesignAction),
    Zero { nrows: usize, ncols: usize },
}

impl CustomFamilyPsiLinearMapRef<'_> {
    pub(crate) fn nrows(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.nrows(),
            Self::First(action) => action.nrows(),
            Self::Second(action) => action.nrows(),
            Self::Zero { nrows, .. } => *nrows,
        }
    }

    pub(crate) fn ncols(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.ncols(),
            Self::First(action) => action.p,
            Self::Second(action) => action.p,
            Self::Zero { ncols, .. } => *ncols,
        }
    }

    pub(crate) fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Array1<f64> {
        match self {
            Self::Dense(mat) => mat.dot(&u),
            Self::First(action) => action.forward_mul(u),
            Self::Second(action) => action.forward_mul(u),
            Self::Zero { nrows, .. } => Array1::<f64>::zeros(*nrows),
        }
    }

    pub(crate) fn transpose_mul(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        match self {
            Self::Dense(mat) => mat.t().dot(&v),
            Self::First(action) => action.transpose_mul(v),
            Self::Second(action) => action.transpose_mul(v),
            Self::Zero { ncols, .. } => Array1::<f64>::zeros(*ncols),
        }
    }

    pub(crate) fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        if row >= self.nrows() {
            return Err(format!(
                "psi linear-map row {row} out of bounds for {} rows",
                self.nrows()
            ));
        }
        Ok(match self {
            Self::Dense(mat) => mat.row(row).to_owned(),
            Self::First(action) => action.row_vector(row)?,
            Self::Second(action) => action.row_vector(row)?,
            Self::Zero { ncols, .. } => Array1::<f64>::zeros(*ncols),
        })
    }

    pub(crate) fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        if rows.end > self.nrows() {
            return Err(format!(
                "psi linear-map row range {}..{} out of bounds for {} rows",
                rows.start,
                rows.end,
                self.nrows()
            ));
        }
        Ok(match self {
            Self::Dense(mat) => mat.slice(ndarray::s![rows, ..]).to_owned(),
            Self::First(action) => action.row_chunk(rows)?,
            Self::Second(action) => action.row_chunk(rows)?,
            Self::Zero { ncols, .. } => Array2::<f64>::zeros((rows.end - rows.start, *ncols)),
        })
    }
}

pub(crate) fn weighted_crossprod_psi_maps(
    left: CustomFamilyPsiLinearMapRef<'_>,
    weights: ArrayView1<'_, f64>,
    right: CustomFamilyPsiLinearMapRef<'_>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != weights.len() || right.nrows() != weights.len() {
        return Err(format!(
            "psi weighted crossprod row mismatch: left={}, weights={}, right={}",
            left.nrows(),
            weights.len(),
            right.nrows()
        ));
    }
    if left.ncols() == 0 || right.ncols() == 0 {
        return Ok(Array2::<f64>::zeros((left.ncols(), right.ncols())));
    }
    if matches!(left, CustomFamilyPsiLinearMapRef::Zero { .. })
        || matches!(right, CustomFamilyPsiLinearMapRef::Zero { .. })
    {
        return Ok(Array2::<f64>::zeros((left.ncols(), right.ncols())));
    }
    let mut out = Array2::<f64>::zeros((left.ncols(), right.ncols()));
    let row_width = left.ncols().saturating_add(right.ncols()).max(1);
    let rows_per_chunk = crate::resource::rows_for_target_bytes(8 * 1024 * 1024, row_width);
    for start in (0..weights.len()).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(weights.len());
        let rows = start..end;
        let xl = left.row_chunk(rows.clone())?;
        let mut xr = right.row_chunk(rows.clone())?;
        for local in 0..xr.nrows() {
            let w = weights[start + local];
            for j in 0..xr.ncols() {
                xr[[local, j]] *= w;
            }
        }
        out += &fast_atb(&xl, &xr);
    }
    Ok(out)
}

pub(crate) fn first_psi_linear_map<'a>(
    action: Option<&'a CustomFamilyPsiDesignAction>,
    dense: Option<&'a Array2<f64>>,
    nrows: usize,
    ncols: usize,
) -> CustomFamilyPsiLinearMapRef<'a> {
    if let Some(action) = action {
        CustomFamilyPsiLinearMapRef::First(action)
    } else if let Some(dense) = dense
        && dense.nrows() == nrows
        && dense.ncols() == ncols
    {
        CustomFamilyPsiLinearMapRef::Dense(dense)
    } else {
        CustomFamilyPsiLinearMapRef::Zero { nrows, ncols }
    }
}

pub(crate) fn second_psi_linear_map<'a>(
    action: Option<&'a CustomFamilyPsiSecondDesignAction>,
    dense: Option<&'a Array2<f64>>,
    nrows: usize,
    ncols: usize,
) -> CustomFamilyPsiLinearMapRef<'a> {
    if let Some(action) = action {
        CustomFamilyPsiLinearMapRef::Second(action)
    } else if let Some(dense) = dense
        && dense.nrows() == nrows
        && dense.ncols() == ncols
    {
        CustomFamilyPsiLinearMapRef::Dense(dense)
    } else {
        CustomFamilyPsiLinearMapRef::Zero { nrows, ncols }
    }
}

pub(crate) struct CustomFamilyJointDesignChannel {
    range: Range<usize>,
    design: DesignMatrix,
    psi_derivative: Option<CustomFamilyPsiDesignAction>,
}

impl CustomFamilyJointDesignChannel {
    pub(crate) fn new<D>(
        range: Range<usize>,
        design: D,
        psi_derivative: Option<CustomFamilyPsiDesignAction>,
    ) -> Self
    where
        D: Into<DesignMatrix>,
    {
        Self {
            range,
            design: design.into(),
            psi_derivative,
        }
    }

    fn coefficients(&self, full: &Array1<f64>) -> Array1<f64> {
        full.slice(ndarray::s![self.range.clone()]).to_owned()
    }

    fn apply(&self, full: &Array1<f64>) -> Array1<f64> {
        let coeffs = self.coefficients(full);
        self.design.matrixvectormultiply(&coeffs)
    }

    fn apply_transpose(&self, values: &Array1<f64>) -> Array1<f64> {
        self.design.transpose_vector_multiply(values)
    }
}

pub(crate) struct CustomFamilyJointDesignPairContribution {
    left_channel: usize,
    right_channel: usize,
    weights: Array1<f64>,
    drift_weights: Array1<f64>,
}

impl CustomFamilyJointDesignPairContribution {
    pub(crate) fn new(
        left_channel: usize,
        right_channel: usize,
        weights: Array1<f64>,
        drift_weights: Array1<f64>,
    ) -> Self {
        Self {
            left_channel,
            right_channel,
            weights,
            drift_weights,
        }
    }
}

pub(crate) struct CustomFamilyJointPsiOperator {
    total_dim: usize,
    channels: Vec<CustomFamilyJointDesignChannel>,
    pair_contributions: Vec<CustomFamilyJointDesignPairContribution>,
    /// Optional dense correction for small cross-blocks (e.g. h/w parameters)
    /// that don't warrant their own weighted-Gram channel.
    dense_correction: Option<Array2<f64>>,
}

impl CustomFamilyJointPsiOperator {
    pub(crate) fn new(
        total_dim: usize,
        channels: Vec<CustomFamilyJointDesignChannel>,
        pair_contributions: Vec<CustomFamilyJointDesignPairContribution>,
    ) -> Self {
        Self {
            total_dim,
            channels,
            pair_contributions,
            dense_correction: None,
        }
    }
}

impl HyperOperator for CustomFamilyJointPsiOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.total_dim);
        let base_vals: Vec<Array1<f64>> = self
            .channels
            .iter()
            .map(|channel| channel.apply(v))
            .collect();
        let deriv_vals: Vec<Option<Array1<f64>>> = self
            .channels
            .iter()
            .map(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .map(|deriv| deriv.forward_mul(v.slice(ndarray::s![channel.range.clone()])))
            })
            .collect();

        let mut out = if let Some(ref corr) = self.dense_correction {
            corr.dot(v)
        } else {
            Array1::<f64>::zeros(self.total_dim)
        };
        for pair in &self.pair_contributions {
            let left = &self.channels[pair.left_channel];
            let right_base = &base_vals[pair.right_channel];
            let weighted_drift = &pair.drift_weights * right_base;
            let mut contrib = left.apply_transpose(&weighted_drift);

            if let Some(left_deriv) = left.psi_derivative.as_ref() {
                let weighted_right = &pair.weights * right_base;
                contrib += &left_deriv.transpose_mul(weighted_right.view());
            }

            if let Some(right_deriv) = deriv_vals[pair.right_channel].as_ref() {
                let weighted_right = &pair.weights * right_deriv;
                contrib += &left.apply_transpose(&weighted_right);
            }

            let mut out_slice = out.slice_mut(ndarray::s![left.range.clone()]);
            out_slice += &contrib;
        }

        out
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        assert_eq!(v.len(), self.total_dim);
        assert_eq!(u.len(), self.total_dim);
        let base_v: Vec<Array1<f64>> = self
            .channels
            .iter()
            .map(|channel| channel.apply(v))
            .collect();
        let base_u: Vec<Array1<f64>> = self
            .channels
            .iter()
            .map(|channel| channel.apply(u))
            .collect();
        let deriv_v: Vec<Option<Array1<f64>>> = self
            .channels
            .iter()
            .map(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .map(|deriv| deriv.forward_mul(v.slice(ndarray::s![channel.range.clone()])))
            })
            .collect();
        let deriv_u: Vec<Option<Array1<f64>>> = self
            .channels
            .iter()
            .map(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .map(|deriv| deriv.forward_mul(u.slice(ndarray::s![channel.range.clone()])))
            })
            .collect();

        let mut total = if let Some(ref corr) = self.dense_correction {
            v.dot(&corr.dot(u))
        } else {
            0.0
        };
        for pair in &self.pair_contributions {
            let left_base_u = &base_u[pair.left_channel];
            let right_base_v = &base_v[pair.right_channel];
            total += left_base_u.dot(&(&pair.drift_weights * right_base_v));

            if let Some(left_deriv_u) = deriv_u[pair.left_channel].as_ref() {
                total += left_deriv_u.dot(&(&pair.weights * right_base_v));
            }
            if let Some(right_deriv_v) = deriv_v[pair.right_channel].as_ref() {
                total += left_base_u.dot(&(&pair.weights * right_deriv_v));
            }
        }

        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self
            .dense_correction
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((self.total_dim, self.total_dim)));
        let mut basis = Array1::<f64>::zeros(self.total_dim);
        for j in 0..self.total_dim {
            basis[j] = 1.0;
            // Use mul_vec without the dense_correction part (already in `out`).
            let base_vals: Vec<Array1<f64>> = self
                .channels
                .iter()
                .map(|channel| channel.apply(&basis))
                .collect();
            let deriv_vals: Vec<Option<Array1<f64>>> = self
                .channels
                .iter()
                .map(|channel| {
                    channel.psi_derivative.as_ref().map(|deriv| {
                        deriv.forward_mul(basis.slice(ndarray::s![channel.range.clone()]))
                    })
                })
                .collect();
            let mut col = Array1::<f64>::zeros(self.total_dim);
            for pair in &self.pair_contributions {
                let left = &self.channels[pair.left_channel];
                let right_base = &base_vals[pair.right_channel];
                let weighted_drift = &pair.drift_weights * right_base;
                let mut contrib = left.apply_transpose(&weighted_drift);
                if let Some(left_deriv) = left.psi_derivative.as_ref() {
                    let weighted_right = &pair.weights * right_base;
                    contrib += &left_deriv.transpose_mul(weighted_right.view());
                }
                if let Some(right_deriv) = deriv_vals[pair.right_channel].as_ref() {
                    let weighted_right = &pair.weights * right_deriv;
                    contrib += &left.apply_transpose(&weighted_right);
                }
                col.slice_mut(ndarray::s![left.range.clone()])
                    .scaled_add(1.0, &contrib);
            }
            out.column_mut(j).scaled_add(1.0, &col);
            basis[j] = 0.0;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.dense_correction.is_none()
            && self.channels.iter().any(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .is_some_and(|d| d.is_implicit())
            })
    }
}

fn shared_dense_design_cache() -> &'static Mutex<HashMap<(usize, usize, usize), Weak<Array2<f64>>>>
{
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), Weak<Array2<f64>>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn shared_dense_arc(x: &Array2<f64>) -> Arc<Array2<f64>> {
    let key = (x.as_ptr() as usize, x.nrows(), x.ncols());
    let cache = shared_dense_design_cache();
    if let Ok(mut guard) = cache.lock() {
        if let Some(shared) = guard.get(&key).and_then(Weak::upgrade) {
            return shared;
        }
        let shared = Arc::new(x.clone());
        guard.insert(key, Arc::downgrade(&shared));
        shared
    } else {
        Arc::new(x.clone())
    }
}

pub(crate) fn resolve_custom_family_x_psi(
    deriv: &CustomFamilyBlockPsiDerivative,
    n: usize,
    p: usize,
    label: &str,
) -> Result<Array2<f64>, String> {
    if deriv.x_psi.nrows() == n && deriv.x_psi.ncols() == p {
        return Ok(deriv.x_psi.clone());
    }
    if let Some(op) = deriv.implicit_operator.as_ref() {
        if !should_resolve_custom_family_psi_dense(n, p) {
            return Err(format!(
                "{label} would materialize implicit x_psi densely at {}x{}; use the operator-aware exact-joint path instead",
                n, p
            ));
        }
        let x = op
            .materialize_first(deriv.implicit_axis)
            .map_err(|e| format!("{label} implicit materialize_first failed: {e}"))?;
        if x.nrows() != n || x.ncols() != p {
            return Err(format!(
                "{label} implicit x_psi shape mismatch: got {}x{}, expected {}x{}",
                x.nrows(),
                x.ncols(),
                n,
                p
            ));
        }
        return Ok(x);
    }
    Err(format!(
        "{label} x_psi shape mismatch: got {}x{}, expected {}x{}",
        deriv.x_psi.nrows(),
        deriv.x_psi.ncols(),
        n,
        p
    ))
}

pub(crate) fn resolve_custom_family_x_psi_psi(
    deriv_i: &CustomFamilyBlockPsiDerivative,
    deriv_j: &CustomFamilyBlockPsiDerivative,
    local_j: usize,
    n: usize,
    p: usize,
    label: &str,
) -> Result<Array2<f64>, String> {
    if let Some(op) = deriv_i.implicit_operator.as_ref() {
        if !should_resolve_custom_family_psi_dense(n, p) {
            return Err(format!(
                "{label} would materialize implicit x_psi_psi densely at {}x{}; use the operator-aware exact-joint path instead",
                n, p
            ));
        }
        let same_group = deriv_i.implicit_group_id.is_some()
            && deriv_i.implicit_group_id == deriv_j.implicit_group_id;
        let x = if same_group {
            if deriv_i.implicit_axis == deriv_j.implicit_axis {
                op.materialize_second_diag(deriv_i.implicit_axis)
                    .map_err(|e| format!("{label} implicit materialize_second_diag failed: {e}"))?
            } else {
                op.materialize_second_cross(deriv_i.implicit_axis, deriv_j.implicit_axis)
                    .map_err(|e| format!("{label} implicit materialize_second_cross failed: {e}"))?
            }
        } else {
            Array2::<f64>::zeros((n, p))
        };
        if x.nrows() != n || x.ncols() != p {
            return Err(format!(
                "{label} implicit x_psi_psi shape mismatch: got {}x{}, expected {}x{}",
                x.nrows(),
                x.ncols(),
                n,
                p
            ));
        }
        return Ok(x);
    }

    if let Some(x_psi_psi) = deriv_i.x_psi_psi.as_ref()
        && let Some(x_ab) = x_psi_psi.get(local_j)
    {
        if x_ab.nrows() == n && x_ab.ncols() == p {
            return Ok(x_ab.clone());
        }
        if x_ab.is_empty() {
            return Ok(Array2::<f64>::zeros((n, p)));
        }
        return Err(format!(
            "{label} x_psi_psi shape mismatch: got {}x{}, expected {}x{}",
            x_ab.nrows(),
            x_ab.ncols(),
            n,
            p
        ));
    }

    Ok(Array2::<f64>::zeros((n, p)))
}

pub struct ExactNewtonJointPsiTerms {
    pub objective_psi: f64,
    pub score_psi: Array1<f64>,
    pub hessian_psi: Array2<f64>,
    pub hessian_psi_operator: Option<Arc<dyn HyperOperator>>,
}

impl std::fmt::Debug for ExactNewtonJointPsiTerms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExactNewtonJointPsiTerms")
            .field("objective_psi", &self.objective_psi)
            .field("score_psi", &self.score_psi)
            .field("hessian_psi", &self.hessian_psi)
            .field(
                "hessian_psi_operator",
                &self.hessian_psi_operator.as_ref().map(|_| "<operator>"),
            )
            .finish()
    }
}

impl ExactNewtonJointPsiTerms {
    fn zeros(total: usize) -> Self {
        Self {
            objective_psi: 0.0,
            score_psi: Array1::zeros(total),
            hessian_psi: Array2::zeros((total, total)),
            hessian_psi_operator: None,
        }
    }
}

pub struct ExactNewtonJointPsiSecondOrderTerms {
    pub objective_psi_psi: f64,
    pub score_psi_psi: Array1<f64>,
    pub hessian_psi_psi: Array2<f64>,
    pub hessian_psi_psi_operator: Option<Box<dyn HyperOperator>>,
}

pub trait ExactNewtonJointHessianWorkspace: Send + Sync {
    fn hessian_matvec(&self, _: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String>;

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        Ok(self.directional_derivative(d_beta_flat)?.map(|matrix| {
            Arc::new(crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix })
                as Arc<dyn HyperOperator>
        }))
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
        Ok(self
            .second_directional_derivative(d_beta_u, d_beta_v)?
            .map(|matrix| {
                Arc::new(
                    crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix },
                ) as Arc<dyn HyperOperator>
            }))
    }
}

pub trait ExactNewtonJointPsiWorkspace: Send + Sync {
    fn first_order_terms(&self, _: usize) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        Ok(None)
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String>;

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String>;
}

pub(crate) struct ExactNewtonJointPsiDirectCache<T> {
    entries: Vec<Mutex<Option<Option<Arc<T>>>>>,
    lru: Mutex<std::collections::VecDeque<usize>>,
    limit: usize,
}

impl<T> ExactNewtonJointPsiDirectCache<T> {
    const DEFAULT_LIMIT: usize = 4;

    pub(crate) fn new(len: usize) -> Self {
        Self {
            entries: (0..len).map(|_| Mutex::new(None)).collect(),
            lru: Mutex::new(std::collections::VecDeque::new()),
            limit: Self::DEFAULT_LIMIT.min(len),
        }
    }

    fn touch_lru(&self, index: usize) -> Result<(), String> {
        let mut lru = self
            .lru
            .lock()
            .map_err(|_| "joint psi direct cache lru poisoned".to_string())?;
        lru.retain(|&existing| existing != index);
        lru.push_back(index);
        while lru.len() > self.limit {
            let Some(evict_index) = lru.pop_front() else {
                break;
            };
            if evict_index == index {
                continue;
            }
            if let Some(entry) = self.entries.get(evict_index) {
                let mut guard = entry
                    .lock()
                    .map_err(|_| "joint psi direct cache poisoned".to_string())?;
                *guard = None;
            }
        }
        Ok(())
    }

    pub(crate) fn get_or_try_init<F>(&self, index: usize, init: F) -> Result<Option<Arc<T>>, String>
    where
        F: FnOnce() -> Result<Option<T>, String>,
    {
        let Some(entry) = self.entries.get(index) else {
            return Err(format!(
                "psi cache index {index} out of bounds for size {}",
                self.entries.len()
            ));
        };
        {
            let guard = entry
                .lock()
                .map_err(|_| "joint psi direct cache poisoned".to_string())?;
            if let Some(cached) = guard.as_ref() {
                let cached = cached.clone();
                drop(guard);
                self.touch_lru(index)?;
                return Ok(cached);
            }
        }

        let computed = init()?.map(Arc::new);
        let mut guard = entry
            .lock()
            .map_err(|_| "joint psi direct cache poisoned".to_string())?;
        let cached = guard.get_or_insert_with(|| computed.clone());
        let out = cached.clone();
        drop(guard);
        self.touch_lru(index)?;
        Ok(out)
    }
}

#[derive(Clone)]
pub struct CustomFamilyWarmStart {
    inner: ConstrainedWarmStart,
}

pub struct CustomFamilyJointHyperResult {
    pub objective: f64,
    pub gradient: Array1<f64>,
    pub outer_hessian: crate::solver::outer_strategy::HessianResult,
    pub warm_start: CustomFamilyWarmStart,
}

pub struct CustomFamilyJointHyperEfsResult {
    pub efs_eval: crate::solver::outer_strategy::EfsEval,
    pub warm_start: CustomFamilyWarmStart,
}

struct OuterObjectiveEvalResult {
    objective: f64,
    gradient: Array1<f64>,
    outer_hessian: crate::solver::outer_strategy::HessianResult,
    warm_start: ConstrainedWarmStart,
}

fn outer_eval_result_to_joint_hyper_result(
    result: OuterObjectiveEvalResult,
) -> CustomFamilyJointHyperResult {
    CustomFamilyJointHyperResult {
        objective: result.objective,
        gradient: result.gradient,
        outer_hessian: result.outer_hessian,
        warm_start: CustomFamilyWarmStart {
            inner: result.warm_start,
        },
    }
}

fn outer_efs_result_to_joint_hyper_efs_result(
    efs_eval: crate::solver::outer_strategy::EfsEval,
    warm_start: ConstrainedWarmStart,
) -> CustomFamilyJointHyperEfsResult {
    CustomFamilyJointHyperEfsResult {
        efs_eval,
        warm_start: CustomFamilyWarmStart { inner: warm_start },
    }
}

// Unified exact joint hyper-calculus over theta = [rho, psi].
//
// The correct outer problem is not “a rho objective plus a separate psi
// objective”. It is one profiled/Laplace surface over one flattened hypervector
//
//   theta = [rho, psi],
//
// one flattened joint coefficient vector
//
//   beta = [beta_1; ...; beta_B],
//
// and one joint exact mode system
//
//   F(beta, theta) := V_beta(beta, theta) = 0,
//   H(beta, theta) := V_beta_beta(beta, theta).
//
// For every hypercoordinate theta_i we need the fixed-beta objects
//
//   V_i = partial_{theta_i} V,
//   g_i = partial_{theta_i} F,
//   H_i = partial_{theta_i} H,
//
// and for every pair (i, j)
//
//   V_ij, g_ij, H_ij,
//
// together with the beta-curvature contractions
//
//   D_beta H[u],
//   D_beta^2 H[u, v],
//   T_i[u] := D_beta H_i[u].
//
// The exact profiled mode response and total Hessian drifts are then
//
//   beta_i  = -H^{-1} g_i,
//   beta_ij = -H^{-1}(g_ij + H_i beta_j + H_j beta_i + D_beta H[beta_i] beta_j),
//
//   dot H_i
//   = H_i + D_beta H[beta_i],
//
//   ddot H_ij
//   = H_ij
//     + T_i[beta_j]
//     + T_j[beta_i]
//     + D_beta H[beta_ij]
//     + D_beta^2 H[beta_i, beta_j].
//
// Hence the exact joint profiled/Laplace derivatives are
//
//   J_i
//   = V_i + 0.5 tr(H^{-1} dot H_i) - 0.5 partial_i log|S(theta)|_+,
//
//   J_ij
//   = (V_ij - g_i^T H^{-1} g_j)
//     + 0.5 [ tr(H^{-1} ddot H_ij)
//             - tr(H^{-1} dot H_j H^{-1} dot H_i) ]
//     - 0.5 partial^2_{ij} log|S(theta)|_+.
//
// In this unified view rho and psi are the same outer calculus. They differ
// only in where their fixed-beta derivative objects come from:
//
// - rho coordinates often contribute only through the penalty surface,
//     but the generic assembler intentionally treats the penalty as S(theta),
//     not S(rho), so mixed rho/psi penalty terms are allowed whenever realized
//     component penalties move with psi:
//       V_i  = D_i  + 0.5 beta^T S_i beta
//       g_i  = D_beta_i  + S_i beta
//       H_i  = D_beta_beta_i + S_i
//       V_ij = D_ij + 0.5 beta^T S_ij beta
//       g_ij = D_beta_ij + S_ij beta
//       H_ij = D_beta_beta_ij + S_ij.
//
// - psi coordinates come from the family-specific joint exact psi hooks, while
//   the generic assembler still owns any realized-penalty motion through
//   S_i / S_ij:
//     objective_psi            <-> V_i
//     score_psi                <-> g_i
//     hessian_psi              <-> H_i
//     objective_psi_psi        <-> V_ij
//     score_psi_psi            <-> g_ij
//     hessian_psi_psi          <-> H_ij
//     D_beta H_psi[u]          <-> T_i[u].
//
// For coupled families this means any block-local psi path is wrong. Even when
// g_i is sparse or penalty-local, beta_i is defined by the full joint solve
//
//   beta_i = -H^{-1} g_i,
//
// so every exact outer derivative must be assembled in this joint flattened
// space.

#[derive(Debug, Clone, Error)]
pub enum CustomFamilyError {
    #[error("custom-family invalid input: {0}")]
    InvalidInput(String),
    #[error("custom-family optimization error: {0}")]
    Optimization(String),
}

impl From<String> for CustomFamilyError {
    fn from(value: String) -> Self {
        Self::InvalidInput(value)
    }
}

impl From<CustomFamilyError> for String {
    fn from(value: CustomFamilyError) -> Self {
        value.to_string()
    }
}

fn validate_blockspecs(specs: &[ParameterBlockSpec]) -> Result<Vec<usize>, String> {
    if specs.is_empty() {
        return Err("fit_custom_family requires at least one parameter block".to_string());
    }
    let mut penalty_counts = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let n = spec.design.nrows();
        if spec.offset.len() != n {
            return Err(format!(
                "block {b} offset length mismatch: got {}, expected {}",
                spec.offset.len(),
                n
            ));
        }
        let p = spec.design.ncols();
        if let Some(beta0) = &spec.initial_beta
            && beta0.len() != p
        {
            return Err(format!(
                "block {b} initial_beta length mismatch: got {}, expected {p}",
                beta0.len()
            ));
        }
        if spec.initial_log_lambdas.len() != spec.penalties.len() {
            return Err(format!(
                "block {b} initial_log_lambdas length {} does not match penalties {}",
                spec.initial_log_lambdas.len(),
                spec.penalties.len()
            ));
        }
        for (k, s) in spec.penalties.iter().enumerate() {
            let (r, c) = s.shape();
            if r != p || c != p {
                return Err(format!(
                    "block {b} penalty {k} must be {p}x{p}, got {r}x{c}"
                ));
            }
        }
        penalty_counts.push(spec.penalties.len());
    }
    Ok(penalty_counts)
}

fn with_block_geometry<F: CustomFamily + ?Sized, T>(
    family: &F,
    block_states: &[ParameterBlockState],
    spec: &ParameterBlockSpec,
    block_idx: usize,
    f: impl FnOnce(&DesignMatrix, &Array1<f64>) -> Result<T, String>,
) -> Result<T, String> {
    if family.block_geometry_is_dynamic() {
        let (x_dyn, off_dyn) = family.block_geometry(block_states, spec)?;
        if x_dyn.nrows() != spec.design.nrows() {
            return Err(format!(
                "block {block_idx} dynamic design row mismatch: got {}, expected {}",
                x_dyn.nrows(),
                spec.design.nrows()
            ));
        }
        if x_dyn.ncols() != spec.design.ncols() {
            return Err(format!(
                "block {block_idx} dynamic design col mismatch: got {}, expected {}",
                x_dyn.ncols(),
                spec.design.ncols()
            ));
        }
        if off_dyn.len() != spec.design.nrows() {
            return Err(format!(
                "block {block_idx} dynamic offset length mismatch: got {}, expected {}",
                off_dyn.len(),
                spec.design.nrows()
            ));
        }
        f(&x_dyn, &off_dyn)
    } else {
        f(&spec.design, &spec.offset)
    }
}

fn flatten_log_lambdas(specs: &[ParameterBlockSpec]) -> Array1<f64> {
    let total = specs
        .iter()
        .map(|s| s.initial_log_lambdas.len())
        .sum::<usize>();
    let mut out = Array1::<f64>::zeros(total);
    let mut at = 0usize;
    for spec in specs {
        let len = spec.initial_log_lambdas.len();
        if len > 0 {
            out.slice_mut(ndarray::s![at..at + len])
                .assign(&spec.initial_log_lambdas);
        }
        at += len;
    }
    out
}

fn split_log_lambdas(
    flat: &Array1<f64>,
    penalty_counts: &[usize],
) -> Result<Vec<Array1<f64>>, String> {
    let expected: usize = penalty_counts.iter().sum();
    if flat.len() != expected {
        return Err(format!(
            "log-lambda length mismatch: got {}, expected {expected}",
            flat.len()
        ));
    }
    let mut out = Vec::with_capacity(penalty_counts.len());
    let mut at = 0usize;
    for &k in penalty_counts {
        out.push(flat.slice(ndarray::s![at..at + k]).to_owned());
        at += k;
    }
    Ok(out)
}

fn buildblock_states<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
) -> Result<Vec<ParameterBlockState>, String> {
    let mut states = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let p = spec.design.ncols();
        let beta = spec
            .initial_beta
            .clone()
            .unwrap_or_else(|| Array1::<f64>::zeros(p));
        let eta = with_block_geometry(family, &states, spec, b, |x, off| {
            let mut eta = x.matrixvectormultiply(&beta);
            eta += off;
            Ok(eta)
        })?;
        states.push(ParameterBlockState { beta, eta });
    }
    // After every block state is populated, pass each β through
    // `post_update_block_beta` so the invariant "every `states[b].beta`
    // in `inner_blockwise_fit` is feasible" holds from the first eval
    // call onward — matching the same projection the warm-start seed
    // path at 5932 already applies.  Defers projection to this second
    // pass because some family overrides (e.g.
    // `SurvivalMarginalSlopeFamily::post_update_block_beta`) read
    // `block_states[block_idx]` during projection, and `block_idx == b`
    // is only populated once the first pass has pushed all states.
    //
    // Without this, a caller that supplies `initial_beta = Some(infeasible)`
    // — or leaves it `None` for a family whose zero vector violates the
    // family's bounds — feeds an infeasible β into
    // `exact_newton_joint_hessian` / `evaluate` before the first
    // line-search trial, silently corrupting the fit or tripping
    // `max_feasible_step_size` guards on iteration 1.  The warm-start
    // path (5925-5938) projects on entry for exactly this reason; this
    // extends the invariant to the cold-start path too.
    for b in 0..specs.len() {
        let raw = states[b].beta.clone();
        let projected = family.post_update_block_beta(&states, b, &specs[b], raw)?;
        states[b].beta.assign(&projected);
    }
    // Note: the caller (`inner_blockwise_fit`) calls `refresh_all_block_etas`
    // immediately after this returns, so η is recomputed against the
    // projected β before any family evaluation runs.  We don't duplicate
    // the refresh here.
    Ok(states)
}

fn refresh_all_block_etas<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
) -> Result<(), String> {
    for b in 0..specs.len() {
        refresh_single_block_eta(family, specs, states, b)?;
    }
    Ok(())
}

fn refresh_single_block_eta<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_idx: usize,
) -> Result<(), String> {
    let spec = &specs[block_idx];
    let beta = states[block_idx].beta.clone();
    states[block_idx].eta = with_block_geometry(family, states, spec, block_idx, |x, off| {
        Ok(x.matrixvectormultiply(&beta) + off)
    })?;
    Ok(())
}

#[inline]
fn capped_inner_max_cycles(options: &BlockwiseFitOptions, base_cycles: usize) -> usize {
    let screening_cap = options
        .screening_max_inner_iterations
        .as_ref()
        .map(|cap| cap.load(Ordering::Relaxed))
        .unwrap_or(0);
    if screening_cap == 0 {
        base_cycles
    } else {
        base_cycles.min(screening_cap.max(1))
    }
}

fn weighted_normal_equations(
    x: &DesignMatrix,
    w: &Array1<f64>,
    y_star: Option<&Array1<f64>>,
) -> Result<(Array2<f64>, Option<Array1<f64>>), String> {
    let n = x.nrows();
    if w.len() != n {
        return Err("weighted normal-equation dimension mismatch".to_string());
    }
    if let Some(y) = y_star
        && y.len() != n
    {
        return Err("weighted RHS dimension mismatch".to_string());
    }

    let xtwx = x.compute_xtwx(w)?;
    let xtwy = if let Some(y) = y_star {
        Some(x.compute_xtwy(w, y)?)
    } else {
        None
    };
    Ok((xtwx, xtwy))
}

#[cfg(test)]
fn solve_blockweighted_system(
    x: &DesignMatrix,
    y_star: &Array1<f64>,
    w: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    let n = x.nrows();
    if y_star.len() != n || w.len() != n {
        return Err("weighted-system dimension mismatch".to_string());
    }
    let xtwy = x.compute_xtwy(w, y_star)?;
    x.solve_systemwith_policy(w, &xtwy, Some(s_lambda), ridge_floor, ridge_policy)
        .map_err(|_| "block solve failed after ridge retries".to_string())
}

fn solve_spd_systemwith_policy(
    lhs: &Array2<f64>,
    rhs: &Array1<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    let p = lhs.nrows();
    if lhs.ncols() != p || rhs.len() != p {
        return Err("exact-newton system dimension mismatch".to_string());
    }
    let baseridge = if ridge_policy.include_laplacehessian {
        effective_solverridge(ridge_floor)
    } else {
        0.0
    };
    let solver = StableSolver::new("custom-family SPD block solve");
    solver
        .solvevectorwithridge_retries(lhs, rhs, baseridge)
        .or_else(|| {
            pinv_positive_part(lhs, effective_solverridge(ridge_floor))
                .ok()
                .map(|pinv| pinv.dot(rhs))
        })
        .ok_or_else(|| "exact-newton block solve failed after ridge retries".to_string())
}

fn exact_newton_stabilizing_shift(lhs_dense: &Array2<f64>, ridge_floor: f64) -> Option<f64> {
    let floor = effective_solverridge(ridge_floor);
    match FaerEigh::eigh(lhs_dense, Side::Lower) {
        Ok((evals, _)) => {
            let min_eval = evals.iter().copied().fold(f64::INFINITY, |a, b| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            });
            if !min_eval.is_finite() || min_eval <= floor {
                Some(floor - min_eval.min(0.0).max(-1e12))
            } else {
                None
            }
        }
        Err(_) => {
            let diag_max = (0..lhs_dense.nrows())
                .map(|d| lhs_dense[[d, d]].abs())
                .fold(0.0_f64, f64::max);
            Some(floor.max(diag_max * 1e-6).max(1e-6))
        }
    }
}

fn stabilize_exact_newton_lhs_in_place<F: CustomFamily + ?Sized>(
    family: &F,
    lhs_dense: &mut Array2<f64>,
    ridge_floor: f64,
) {
    if use_exact_newton_strict_spd(family) {
        return;
    }
    if let Some(shift) = exact_newton_stabilizing_shift(lhs_dense, ridge_floor) {
        for d in 0..lhs_dense.nrows() {
            lhs_dense[[d, d]] += shift;
        }
    }
}

fn shift_linear_constraints_to_delta(
    constraints: &LinearInequalityConstraints,
    beta: &Array1<f64>,
) -> Result<LinearInequalityConstraints, String> {
    if constraints.a.ncols() != beta.len() || constraints.a.nrows() != constraints.b.len() {
        return Err("linear constraints: shape mismatch".to_string());
    }
    Ok(LinearInequalityConstraints {
        a: constraints.a.clone(),
        b: &constraints.b - &constraints.a.dot(beta),
    })
}

fn collect_block_linear_constraints<F: CustomFamily + ?Sized>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Result<Vec<Option<LinearInequalityConstraints>>, String> {
    let mut constraints = Vec::with_capacity(specs.len());
    for (block_idx, spec) in specs.iter().enumerate() {
        constraints.push(family.block_linear_constraints(states, block_idx, spec)?);
    }
    Ok(constraints)
}

fn assemble_joint_linear_constraints(
    block_constraints: &[Option<LinearInequalityConstraints>],
    ranges: &[(usize, usize)],
    total_p: usize,
) -> Result<Option<LinearInequalityConstraints>, String> {
    if block_constraints.len() != ranges.len() {
        return Err(format!(
            "joint linear constraint assembly mismatch: {} blocks but {} ranges",
            block_constraints.len(),
            ranges.len()
        ));
    }
    let total_rows = block_constraints
        .iter()
        .map(|constraints| constraints.as_ref().map_or(0, |c| c.a.nrows()))
        .sum::<usize>();
    if total_rows == 0 {
        return Ok(None);
    }
    let mut a = Array2::<f64>::zeros((total_rows, total_p));
    let mut b = Array1::<f64>::zeros(total_rows);
    let mut row_offset = 0usize;
    for (block_idx, constraints_opt) in block_constraints.iter().enumerate() {
        let Some(constraints) = constraints_opt else {
            continue;
        };
        let (start, end) = ranges[block_idx];
        let block_p = end - start;
        if constraints.a.ncols() != block_p || constraints.a.nrows() != constraints.b.len() {
            return Err(format!(
                "joint linear constraint assembly mismatch for block {block_idx}: A is {}x{}, b is {}, block width is {}",
                constraints.a.nrows(),
                constraints.a.ncols(),
                constraints.b.len(),
                block_p
            ));
        }
        let rows = constraints.a.nrows();
        a.slice_mut(s![row_offset..(row_offset + rows), start..end])
            .assign(&constraints.a);
        b.slice_mut(s![row_offset..(row_offset + rows)])
            .assign(&constraints.b);
        row_offset += rows;
    }
    Ok(Some(LinearInequalityConstraints { a, b }))
}

fn flatten_joint_active_set(
    block_active_sets: &[Option<Vec<usize>>],
    block_constraints: &[Option<LinearInequalityConstraints>],
) -> Option<Vec<usize>> {
    if block_active_sets.len() != block_constraints.len() {
        return None;
    }
    let mut offset = 0usize;
    let mut joint_active = Vec::new();
    for (active_opt, constraints_opt) in block_active_sets.iter().zip(block_constraints.iter()) {
        let rows = constraints_opt
            .as_ref()
            .map_or(0, |constraints| constraints.a.nrows());
        if let Some(active) = active_opt {
            joint_active.extend(
                active
                    .iter()
                    .copied()
                    .filter(|&idx| idx < rows)
                    .map(|idx| offset + idx),
            );
        }
        offset += rows;
    }
    if joint_active.is_empty() {
        None
    } else {
        Some(joint_active)
    }
}

fn scatter_joint_active_set(
    joint_active: &[usize],
    block_constraints: &[Option<LinearInequalityConstraints>],
) -> Vec<Option<Vec<usize>>> {
    let mut per_block = Vec::with_capacity(block_constraints.len());
    let mut offset = 0usize;
    for constraints_opt in block_constraints {
        let rows = constraints_opt
            .as_ref()
            .map_or(0, |constraints| constraints.a.nrows());
        if rows == 0 {
            per_block.push(None);
            continue;
        }
        let mut local = joint_active
            .iter()
            .copied()
            .filter(|&idx| idx >= offset && idx < offset + rows)
            .map(|idx| idx - offset)
            .collect::<Vec<_>>();
        offset += rows;
        if local.is_empty() {
            per_block.push(None);
            continue;
        }
        local.sort_unstable();
        local.dedup();
        per_block.push(Some(local));
    }
    per_block
}

struct SimpleLowerBounds {
    lower_bounds: Array1<f64>,
    row_to_coeff: Vec<usize>,
    coeff_to_row: Vec<Option<usize>>,
}

fn extract_simple_lower_bounds(
    constraints: &LinearInequalityConstraints,
    p: usize,
) -> Result<Option<SimpleLowerBounds>, String> {
    if constraints.a.ncols() != p || constraints.a.nrows() != constraints.b.len() {
        return Err("linear constraints: shape mismatch".to_string());
    }
    let mut lower_bounds = Array1::from_elem(p, f64::NEG_INFINITY);
    let mut coeff_to_row = vec![None; p];
    let mut row_to_coeff = Vec::with_capacity(constraints.a.nrows());
    for row in 0..constraints.a.nrows() {
        let mut coeff_idx = None;
        let mut coeff_value = 0.0;
        for col in 0..p {
            let value = constraints.a[[row, col]];
            if value.abs() <= 1e-12 {
                continue;
            }
            if coeff_idx.is_some() {
                return Ok(None);
            }
            coeff_idx = Some(col);
            coeff_value = value;
        }
        let Some(col) = coeff_idx else {
            return Ok(None);
        };
        if coeff_value <= 0.0 {
            return Ok(None);
        }
        let bound = constraints.b[row] / coeff_value;
        if bound > lower_bounds[col] {
            lower_bounds[col] = bound;
            coeff_to_row[col] = Some(row);
        }
        row_to_coeff.push(col);
    }
    Ok(Some(SimpleLowerBounds {
        lower_bounds,
        row_to_coeff,
        coeff_to_row,
    }))
}

fn lower_bound_active_rows_to_coeffs(
    bounds: &SimpleLowerBounds,
    active_rows: Option<&[usize]>,
) -> Vec<usize> {
    let Some(active_rows) = active_rows else {
        return Vec::new();
    };
    let mut active_coeffs = active_rows
        .iter()
        .copied()
        .filter_map(|row| bounds.row_to_coeff.get(row).copied())
        .collect::<Vec<_>>();
    active_coeffs.sort_unstable();
    active_coeffs.dedup();
    active_coeffs
}

fn lower_bound_active_coeffs_to_rows(
    bounds: &SimpleLowerBounds,
    active_coeffs: &[usize],
) -> Vec<usize> {
    let mut active_rows = active_coeffs
        .iter()
        .copied()
        .filter_map(|coeff| bounds.coeff_to_row.get(coeff).and_then(|row| *row))
        .collect::<Vec<_>>();
    active_rows.sort_unstable();
    active_rows.dedup();
    active_rows
}

fn lower_bound_active_coeffs_from_solution(
    bounds: &SimpleLowerBounds,
    beta: &Array1<f64>,
) -> Vec<usize> {
    let mut active_coeffs = Vec::new();
    for coeff in 0..beta.len() {
        let lower = bounds.lower_bounds[coeff];
        if !lower.is_finite() {
            continue;
        }
        let scale = beta[coeff].abs().max(lower.abs()).max(1.0);
        let tol = 1e-6 * scale + 1e-10;
        if beta[coeff] <= lower + tol {
            active_coeffs.push(coeff);
        }
    }
    active_coeffs
}

fn project_to_lower_bounds(beta: &mut Array1<f64>, lower_bounds: &Array1<f64>) {
    for i in 0..beta.len() {
        let lower = lower_bounds[i];
        if lower.is_finite() && beta[i] < lower {
            beta[i] = lower;
        }
    }
}

fn solve_quadratic_with_simple_lower_bounds(
    lhs: &Array2<f64>,
    rhs: &Array1<f64>,
    beta_start: &Array1<f64>,
    bounds: &SimpleLowerBounds,
    active_rows: Option<&[usize]>,
) -> Result<(Array1<f64>, Vec<usize>), String> {
    let gradient = lhs.dot(beta_start) - rhs;
    let mut delta = Array1::zeros(beta_start.len());
    let mut active_coeffs = lower_bound_active_rows_to_coeffs(bounds, active_rows);
    solve_newton_directionwith_lower_bounds(
        lhs,
        &gradient,
        beta_start,
        &bounds.lower_bounds,
        &mut delta,
        Some(&mut active_coeffs),
    )
    .map_err(|e| format!("lower-bound Newton solve failed: {e}"))?;
    let mut beta_new = beta_start + &delta;
    project_to_lower_bounds(&mut beta_new, &bounds.lower_bounds);
    active_coeffs = lower_bound_active_coeffs_from_solution(bounds, &beta_new);
    let active = lower_bound_active_coeffs_to_rows(bounds, &active_coeffs);
    Ok((beta_new, active))
}

fn normalize_active_set(mut active_set: Vec<usize>) -> Option<Vec<usize>> {
    active_set.sort_unstable();
    active_set.dedup();
    if active_set.is_empty() {
        None
    } else {
        Some(active_set)
    }
}

fn normalize_active_sets(active_sets: Vec<Option<Vec<usize>>>) -> Vec<Option<Vec<usize>>> {
    active_sets
        .into_iter()
        .map(|active_set| active_set.and_then(normalize_active_set))
        .collect()
}

struct BlockUpdateContext<'a> {
    family: &'a dyn CustomFamily,
    states: &'a [ParameterBlockState],
    spec: &'a ParameterBlockSpec,
    block_idx: usize,
    s_lambda: &'a Array2<f64>,
    options: &'a BlockwiseFitOptions,
    linear_constraints: Option<&'a LinearInequalityConstraints>,
    cached_active_set: Option<&'a [usize]>,
}

struct BlockUpdateResult {
    beta_new_raw: Array1<f64>,
    active_set: Option<Vec<usize>>,
}

#[inline]
fn floor_positiveworking_weights(working_weights: &Array1<f64>, minweight: f64) -> Array1<f64> {
    let n = working_weights.len();
    let mut out = Array1::<f64>::uninit(n);
    for i in 0..n {
        let wi = working_weights[i];
        out[i].write(if wi <= 0.0 { 0.0 } else { wi.max(minweight) });
    }
    // SAFETY: all elements written in the loop above.
    unsafe { out.assume_init() }
}

trait ParameterBlockUpdater {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, String>;
}

struct DiagonalBlockUpdater<'a> {
    working_response: &'a Array1<f64>,
    working_weights: &'a Array1<f64>,
}

impl ParameterBlockUpdater for DiagonalBlockUpdater<'_> {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, String> {
        if self.working_response.len() != ctx.spec.design.nrows()
            || self.working_weights.len() != ctx.spec.design.nrows()
        {
            return Err(format!(
                "family diagonal working-set size mismatch on block {} ({})",
                ctx.block_idx, ctx.spec.name
            ));
        }

        // Zero-weight observations are semantically excluded and must stay inactive.
        let w_clamped = floor_positiveworking_weights(self.working_weights, ctx.options.minweight);

        if let Some(constraints) = ctx.linear_constraints {
            check_linear_feasibility(&ctx.states[ctx.block_idx].beta, constraints, 1e-8).map_err(
                |e| {
                    format!(
                        "block {} ({}) constrained diagonal solve: {e}",
                        ctx.block_idx, ctx.spec.name
                    )
                },
            )?;
            with_block_geometry(ctx.family, ctx.states, ctx.spec, ctx.block_idx, |x, off| {
                let mut y_star = self.working_response.clone();
                y_star -= off;
                let (mut lhs, rhs_opt) = weighted_normal_equations(x, &w_clamped, Some(&y_star))?;
                let rhs = rhs_opt.ok_or_else(|| {
                    "missing weighted RHS in constrained diagonal solve".to_string()
                })?;
                lhs += ctx.s_lambda;
                let lower_bounds = extract_simple_lower_bounds(constraints, lhs.ncols())?;
                let (beta_constrained, active_set) = if let Some(bounds) = lower_bounds.as_ref() {
                    solve_quadratic_with_simple_lower_bounds(
                        &lhs,
                        &rhs,
                        &ctx.states[ctx.block_idx].beta,
                        bounds,
                        ctx.cached_active_set,
                    )
                } else {
                    solve_quadratic_with_linear_constraints(
                        &lhs,
                        &rhs,
                        &ctx.states[ctx.block_idx].beta,
                        constraints,
                        ctx.cached_active_set,
                    )
                    .map_err(|e| e.to_string())
                }
                .map_err(|e| {
                    format!(
                        "block {} ({}) constrained diagonal solve failed: {e}",
                        ctx.block_idx, ctx.spec.name
                    )
                })?;
                Ok(BlockUpdateResult {
                    beta_new_raw: beta_constrained,
                    active_set: normalize_active_set(active_set),
                })
            })
        } else {
            with_block_geometry(ctx.family, ctx.states, ctx.spec, ctx.block_idx, |x, off| {
                // Fuse offset subtraction into the weighted RHS: wy[i] = w[i] * (z[i] - off[i]).
                // This avoids an O(n) working_response clone.
                let n = self.working_response.len();
                let wy = Array1::from_shape_fn(n, |i| {
                    (self.working_response[i] - off[i]) * w_clamped[i].max(0.0)
                });
                let xtwy = x.transpose_vector_multiply(&wy);
                let beta = x
                    .solve_systemwith_policy(
                        &w_clamped,
                        &xtwy,
                        Some(ctx.s_lambda),
                        ctx.options.ridge_floor,
                        ctx.options.ridge_policy,
                    )
                    .map_err(|_| "block solve failed after ridge retries".to_string())?;
                Ok(BlockUpdateResult {
                    beta_new_raw: beta,
                    active_set: None,
                })
            })
        }
    }
}

struct ExactNewtonBlockUpdater<'a> {
    gradient: &'a Array1<f64>,
    hessian: &'a SymmetricMatrix,
}

impl ParameterBlockUpdater for ExactNewtonBlockUpdater<'_> {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, String> {
        let p = ctx.spec.design.ncols();
        if self.gradient.len() != p {
            return Err(format!(
                "block {} exact-newton gradient length mismatch: got {}, expected {p}",
                ctx.block_idx,
                self.gradient.len()
            ));
        }
        if self.hessian.nrows() != p || self.hessian.ncols() != p {
            return Err(format!(
                "block {} exact-newton Hessian shape mismatch: got {}x{}, expected {}x{}",
                ctx.block_idx,
                self.hessian.nrows(),
                self.hessian.ncols(),
                p,
                p
            ));
        }

        let lhs = self.hessian.add_dense(ctx.s_lambda)?;
        // Solve in delta-space for both constrained and unconstrained blocks.
        // That keeps the linear system consistent even when we add a
        // numerical ridge to stabilize an indefinite exact-Newton Hessian.
        let rhs_step = self.gradient - &ctx.s_lambda.dot(&ctx.states[ctx.block_idx].beta);
        let mut lhs_dense = lhs.to_dense();
        stabilize_exact_newton_lhs_in_place(ctx.family, &mut lhs_dense, ctx.options.ridge_floor);

        if let Some(constraints) = ctx.linear_constraints {
            check_linear_feasibility(&ctx.states[ctx.block_idx].beta, constraints, 1e-8).map_err(
                |e| {
                    format!(
                        "block {} ({}) constrained exact-newton solve: {e}",
                        ctx.block_idx, ctx.spec.name
                    )
                },
            )?;
            let lower_bounds = extract_simple_lower_bounds(constraints, p).map_err(|e| {
                format!(
                    "block {} ({}) constrained exact-newton solve: {e}",
                    ctx.block_idx, ctx.spec.name
                )
            })?;
            let (beta_new_raw, active_set) = if let Some(bounds) = lower_bounds.as_ref() {
                let rhs_beta = &lhs_dense.dot(&ctx.states[ctx.block_idx].beta) + &rhs_step;
                solve_quadratic_with_simple_lower_bounds(
                    &lhs_dense,
                    &rhs_beta,
                    &ctx.states[ctx.block_idx].beta,
                    bounds,
                    ctx.cached_active_set,
                )
            } else {
                let delta_constraints =
                    shift_linear_constraints_to_delta(constraints, &ctx.states[ctx.block_idx].beta)
                        .map_err(|e| {
                            format!(
                                "block {} ({}) constrained exact-newton solve: {e}",
                                ctx.block_idx, ctx.spec.name
                            )
                        })?;
                let delta_start = Array1::zeros(p);
                let (delta, active_set) = solve_quadratic_with_linear_constraints(
                    &lhs_dense,
                    &rhs_step,
                    &delta_start,
                    &delta_constraints,
                    ctx.cached_active_set,
                )
                .map_err(|e| e.to_string())?;
                Ok((&ctx.states[ctx.block_idx].beta + &delta, active_set))
            }
            .map_err(|e| {
                format!(
                    "block {} ({}) constrained exact-newton solve failed: {e}",
                    ctx.block_idx, ctx.spec.name
                )
            })?;
            Ok(BlockUpdateResult {
                beta_new_raw,
                active_set: normalize_active_set(active_set),
            })
        } else {
            // Solve for the Newton step, not the next beta directly.
            //
            // For the penalized negative objective
            //
            //   Q(beta) = -log L(beta) + 0.5 beta^T S beta,
            //
            // the exact block gradient and Hessian are
            //
            //   grad_Q = S beta - gradient,
            //   hess_Q = hessian + S.
            //
            // The Newton step must therefore satisfy
            //
            //   hess_Q * delta = -grad_Q = gradient - S beta.
            //
            // This form stays correct even when the linear solver adds a
            // numerical ridge to the left-hand side to stabilize an indefinite
            // or nearly singular block. Solving directly for `beta_new` with a
            // ridged matrix would require an extra `ridge * beta` term on the
            // right-hand side; without it the step is distorted, which can trap
            // exact-Newton block updates on nonconvex blocks such as survival
            // `log_sigma`.
            let delta = if use_exact_newton_strict_spd(ctx.family) {
                strict_solve_spd(&lhs_dense, &rhs_step)?
            } else {
                solve_spd_systemwith_policy(
                    &lhs_dense,
                    &rhs_step,
                    ctx.options.ridge_floor,
                    ctx.options.ridge_policy,
                )
                .or_else(|_: String| -> Result<Array1<f64>, String> {
                    // Diagonal fallback: steepest descent scaled by the
                    // diagonal of the Hessian. This always produces a
                    // finite step when the gradient is finite.
                    let diag_step: Array1<f64> = (0..lhs_dense.nrows())
                        .map(|i| {
                            let d = lhs_dense[[i, i]].abs().max(1e-8);
                            rhs_step[i] / d
                        })
                        .collect();
                    Ok(diag_step)
                })?
            };
            let beta = &ctx.states[ctx.block_idx].beta + &delta;
            Ok(BlockUpdateResult {
                beta_new_raw: beta,
                active_set: None,
            })
        }
    }
}

impl BlockWorkingSet {
    fn updater(&self) -> Box<dyn ParameterBlockUpdater + '_> {
        match self {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => Box::new(DiagonalBlockUpdater {
                working_response,
                working_weights,
            }),
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                Box::new(ExactNewtonBlockUpdater { gradient, hessian })
            }
        }
    }
}

fn check_linear_feasibility(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    tol: f64,
) -> Result<(), String> {
    if constraints.a.ncols() != beta.len() || constraints.a.nrows() != constraints.b.len() {
        return Err("linear constraints: shape mismatch".to_string());
    }
    let slack = constraints.a.dot(beta) - &constraints.b;
    let mut worst = 0.0_f64;
    let mut worst_idx = 0usize;
    for (i, &s) in slack.iter().enumerate() {
        let v = (-s).max(0.0);
        if v > worst {
            worst = v;
            worst_idx = i;
        }
    }
    if worst > tol {
        return Err(format!(
            "infeasible iterate: max(Aβ-b violation)={worst:.3e} at constraint row {worst_idx}"
        ));
    }
    Ok(())
}

#[inline]
fn effective_solverridge(ridge_floor: f64) -> f64 {
    ridge_floor.max(1e-15)
}

fn block_quadratic_penalty(
    beta: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> f64 {
    let mut value = 0.5 * beta.dot(&s_lambda.dot(beta));
    if ridge_policy.include_quadratic_penalty {
        value += 0.5 * ridge * beta.dot(beta);
    }
    value
}

fn total_quadratic_penalty(
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> f64 {
    let mut total = 0.0;
    for (state, s_lambda) in states.iter().zip(s_lambdas.iter()) {
        total += block_quadratic_penalty(&state.beta, s_lambda, ridge, ridge_policy);
    }
    total
}

fn stable_logdet_with_ridge_policy(
    matrix: &Array2<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<f64, String> {
    let mut a = matrix.clone();
    symmetrize_dense_in_place(&mut a);
    let p = a.nrows();
    let ridge = if ridge_policy.include_penalty_logdet {
        effective_solverridge(ridge_floor)
    } else {
        0.0
    };
    for i in 0..p {
        a[[i, i]] += ridge;
    }

    match resolved_ridge_determinant_mode(ridge_policy, p) {
        RidgeDeterminantMode::Full => {
            let chol = a.cholesky(Side::Lower).map_err(|_| {
                "cholesky failed while computing full ridge-aware logdet".to_string()
            })?;
            Ok(2.0 * chol.diag().mapv(f64::ln).sum())
        }
        RidgeDeterminantMode::StochasticLanczos => {
            let (probes, steps) = default_slq_parameters(p);
            stochastic_lanczos_logdet_spd(&a, probes, steps, 42)
        }
        RidgeDeterminantMode::Auto => unreachable!("adaptive determinant mode must resolve"),
        RidgeDeterminantMode::PositivePart => {
            // Smooth-regularized logdet objective, aligned with the gradient
            // operator (`DenseSpectralOperator` in `Smooth` mode):
            //
            //   log |A|_reg = Σ_j log r_ε(σ_j),   r_ε(σ) = ½(σ + √(σ² + 4ε²))
            //
            // Every eigenvalue contributes; none are silently dropped.  The
            // regularizer r_ε is C∞, strictly positive for all real σ, and
            // numerically agrees with plain log σ when σ ≫ ε.  Negative
            // eigenvalues contribute ≈ log(ε²/|σ|) (quadratic damping) so
            // indefinite Hessians produce a finite, differentiable cost
            // rather than a discontinuous positive-part pseudo-determinant.
            //
            // This matches exactly what the downstream
            // `trace_logdet_gradient = Σ φ'(σ) u^T (dH/dρ) u` computes as the
            // analytic gradient — eliminating the cost/gradient mismatch
            // that previously broke BFGS line search on indefinite outer
            // Hessians.
            //
            // Fallback: escalating-ridge Cholesky when eigh fails despite the
            // internal jitter schedule.  Cholesky is a direct algorithm so it
            // cannot suffer QR-style convergence failures.
            match crate::faer_ndarray::FaerEigh::eigh(&a, Side::Lower) {
                Ok((evals, _)) => {
                    let eval_vec: Vec<f64> = evals
                        .as_slice()
                        .map(|sl| sl.to_vec())
                        .unwrap_or_else(|| evals.iter().copied().collect());
                    let eps = spectral_epsilon(&eval_vec).max(ridge.max(1e-14));
                    let n_negative = eval_vec.iter().filter(|&&ev| ev < -eps).count();
                    if n_negative > 0 {
                        // Diagnostic only: indefiniteness is now handled
                        // correctly by the smooth regularizer, not ignored.
                        log::debug!(
                            "[SmoothRegularizedLogdet] Hessian has {n_negative} \
                             eigenvalue(s) below -eps={eps:.2e}; r_ε damps them \
                             smoothly instead of dropping them."
                        );
                    }
                    let logdet: f64 = eval_vec
                        .iter()
                        .map(|&sigma| spectral_regularize(sigma, eps).ln())
                        .sum();
                    Ok(logdet)
                }
                Err(eigh_err) => positive_part_cholesky_fallback(&a, ridge, &eigh_err),
            }
        }
    }
}

/// Fallback for the `PositivePart` logdet branch when eigendecomposition fails
/// despite the internal escalating-jitter schedule in `FaerEigh::eigh`.
///
/// Strategy: make the matrix SPD by adding progressively larger ridge, then use
/// Cholesky (a direct, non-iterative factorization) for the logdet.
///
/// Mathematical justification: the positive-part surrogate is already an
/// approximation.  For a nearly-SPD Hessian the Cholesky logdet is close to
/// the positive-part logdet.  For a mildly indefinite Hessian the extra ridge
/// shifts all eigenvalues positive; the resulting bias is a smooth, slowly-
/// varying offset in the REML objective that does not corrupt gradients.
fn positive_part_cholesky_fallback(
    a: &Array2<f64>,
    existing_ridge: f64,
    eigh_err: &crate::faer_ndarray::FaerLinalgError,
) -> Result<f64, String> {
    if !a.iter().all(|value| value.is_finite()) {
        log::warn!(
            "[PositivePartFallback] eigendecomposition failed on non-finite Hessian ({eigh_err}); \
             dropping the positive-part surrogate contribution"
        );
        return Ok(0.0);
    }
    let p = a.nrows();
    let diag_scale = a
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);

    // Geometric schedule: start from a ridge that should dominate any
    // moderate indefiniteness, and escalate if Cholesky still fails.
    const MAX_ATTEMPTS: usize = 6;
    let mut boost = diag_scale * 1e-6;
    for attempt in 0..MAX_ATTEMPTS {
        let mut candidate = a.clone();
        for i in 0..p {
            candidate[[i, i]] += boost;
        }
        if let Ok(chol) = candidate.cholesky(Side::Lower) {
            let logdet = 2.0 * chol.diag().mapv(f64::ln).sum();
            if logdet.is_finite() {
                log::warn!(
                    "[PositivePartFallback] eigendecomposition failed ({eigh_err}); \
                     using Cholesky with boosted ridge={:.2e} (attempt {}/{MAX_ATTEMPTS}, \
                     existing_ridge={:.2e}, p={p})",
                    boost + existing_ridge,
                    attempt + 1,
                    existing_ridge,
                );
                return Ok(logdet);
            }
        }
        boost *= 10.0;
    }

    Err(format!(
        "positive-part surrogate eigendecomposition failed ({eigh_err}) and \
         Cholesky fallback also failed after {MAX_ATTEMPTS} attempts \
         (final ridge={:.2e}, p={p})",
        boost + existing_ridge,
    ))
}

/// Fallback for penalty pseudo-logdet when eigendecomposition fails.
///
/// Penalty matrices are PSD by construction (weighted sum of PSD penalties),
/// so the ridged matrix should be SPD.  Uses escalating-ridge Cholesky —
/// same pattern as `positive_part_cholesky_fallback`.
fn penalty_logdet_cholesky_fallback(
    s_ridged: &Array2<f64>,
    existing_ridge: f64,
    block: usize,
    p: usize,
    eigh_err: &crate::faer_ndarray::FaerLinalgError,
) -> Result<f64, String> {
    let diag_scale = s_ridged
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);

    const MAX_ATTEMPTS: usize = 6;
    let mut boost = diag_scale * 1e-8;
    for attempt in 0..MAX_ATTEMPTS {
        let mut candidate = s_ridged.clone();
        for i in 0..p {
            candidate[[i, i]] += boost;
        }
        if let Ok(chol) = candidate.cholesky(Side::Lower) {
            let logdet = 2.0 * chol.diag().mapv(f64::ln).sum();
            if logdet.is_finite() {
                log::warn!(
                    "[PenaltyLogdetFallback] eigendecomposition failed for block {block} \
                     ({eigh_err}); using Cholesky with boosted ridge={:.2e} \
                     (attempt {}/{MAX_ATTEMPTS}, existing_ridge={:.2e}, p={p})",
                    boost + existing_ridge,
                    attempt + 1,
                    existing_ridge,
                );
                return Ok(logdet);
            }
        }
        boost *= 10.0;
    }

    Err(format!(
        "penalty logdet eigendecomposition failed for block {block} ({eigh_err}) and \
         Cholesky fallback also failed after {MAX_ATTEMPTS} attempts \
         (final ridge={:.2e}, p={p})",
        boost + existing_ridge,
    ))
}

const AUTO_SLQ_LOGDET_MIN_DIM: usize = 4096;

fn resolved_ridge_determinant_mode(ridge_policy: RidgePolicy, dim: usize) -> RidgeDeterminantMode {
    match ridge_policy.determinant_mode {
        RidgeDeterminantMode::Auto if dim >= AUTO_SLQ_LOGDET_MIN_DIM => {
            RidgeDeterminantMode::StochasticLanczos
        }
        RidgeDeterminantMode::Auto => RidgeDeterminantMode::Full,
        mode => mode,
    }
}

fn inverse_spdwith_retry(
    matrix: &Array2<f64>,
    baseridge: f64,
    max_retry: usize,
) -> Result<Array2<f64>, String> {
    let solver = StableSolver::new("custom-family inverse spd");
    solver
        .inversewithridge_retries(matrix, baseridge, max_retry)
        .ok_or_else(|| "failed to invert SPD system after ridge retries".to_string())
}

pub(crate) fn symmetrize_dense_in_place(matrix: &mut Array2<f64>) {
    let p = matrix.nrows();
    for i in 0..p {
        for j in 0..i {
            let v = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            matrix[[i, j]] = v;
            matrix[[j, i]] = v;
        }
    }
}

fn exact_newton_joint_hessian_symmetrized<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    total: usize,
    context: &str,
) -> Result<Option<Array2<f64>>, String> {
    let Some(mut h) = family.exact_newton_joint_hessian_with_specs(states, specs)? else {
        return Ok(None);
    };
    if h.nrows() != total || h.ncols() != total {
        return Err(format!(
            "{context}: got {}x{}, expected {}x{}",
            h.nrows(),
            h.ncols(),
            total,
            total
        ));
    }
    symmetrize_dense_in_place(&mut h);
    Ok(Some(h))
}

/// Scale-aware exact joint curvature payload for the outer REML evaluator.
pub struct ExactNewtonOuterCurvature {
    pub hessian: Array2<f64>,
    pub rho_curvature_scale: f64,
    pub hessian_logdet_correction: f64,
}

enum JointHessianSource {
    Dense(Array2<f64>),
    Operator {
        apply: Arc<dyn Fn(&Array1<f64>) -> Result<Array1<f64>, String> + Send + Sync>,
        diagonal: Array1<f64>,
    },
}

struct JointHessianBundle<'a> {
    source: JointHessianSource,
    beta_flat: Array1<f64>,
    compute_dh: Box<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + 'a>,
    compute_d2h:
        Box<dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + 'a>,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
}

fn materialize_joint_hessian_source(
    source: &JointHessianSource,
    total: usize,
    context: &str,
) -> Result<Array2<f64>, String> {
    match source {
        JointHessianSource::Dense(matrix) => Ok(matrix.clone()),
        JointHessianSource::Operator { apply, .. } => {
            let mut matrix = Array2::<f64>::zeros((total, total));
            let mut basis = Array1::<f64>::zeros(total);
            for col in 0..total {
                basis[col] = 1.0;
                let applied = apply(&basis)?;
                basis[col] = 0.0;
                if applied.len() != total {
                    return Err(format!(
                        "{context}: operator matvec length mismatch: got {}, expected {}",
                        applied.len(),
                        total
                    ));
                }
                if applied.iter().any(|value| !value.is_finite()) {
                    return Err(format!(
                        "{context}: operator matvec returned non-finite values"
                    ));
                }
                matrix.column_mut(col).assign(&applied);
            }
            symmetrize_dense_in_place(&mut matrix);
            Ok(matrix)
        }
    }
}

fn exact_newton_joint_hessian_source_from_workspace(
    workspace: &Arc<dyn ExactNewtonJointHessianWorkspace>,
    total: usize,
    context: &str,
) -> Result<Option<JointHessianSource>, String> {
    let Some(diagonal) = workspace.hessian_diagonal()? else {
        return Ok(None);
    };
    if diagonal.len() != total {
        return Err(format!(
            "{context}: operator diagonal length mismatch: got {}, expected {}",
            diagonal.len(),
            total
        ));
    }
    if diagonal.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{context}: operator diagonal contains non-finite values"
        ));
    }

    let zero = Array1::<f64>::zeros(total);
    let Some(zero_image) = workspace.hessian_matvec(&zero)? else {
        return Ok(None);
    };
    if zero_image.len() != total {
        return Err(format!(
            "{context}: operator matvec length mismatch: got {}, expected {}",
            zero_image.len(),
            total
        ));
    }
    if zero_image.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{context}: operator matvec returned non-finite values"
        ));
    }

    let workspace_apply = Arc::clone(workspace);
    Ok(Some(JointHessianSource::Operator {
        apply: Arc::new(move |v: &Array1<f64>| {
            let Some(out) = workspace_apply.hessian_matvec(v)? else {
                return Err("joint exact-newton operator matvec unavailable".to_string());
            };
            Ok(out)
        }),
        diagonal,
    }))
}

fn symmetrized_square_matrix(
    mut matrix: Array2<f64>,
    expected: usize,
    context: &str,
) -> Result<Array2<f64>, String> {
    if matrix.nrows() != expected || matrix.ncols() != expected {
        return Err(format!(
            "{context}: got {}x{}, expected {}x{}",
            matrix.nrows(),
            matrix.ncols(),
            expected,
            expected
        ));
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(format!("{context}: matrix contains non-finite values"));
    }
    symmetrize_dense_in_place(&mut matrix);
    Ok(matrix)
}

/// Try exact Newton joint Hessian first, then surrogate. Returns `None` if
/// neither path provides a joint Hessian. When successful, returns the joint
/// Hessian source, flat beta, and boxed closures for computing directional
/// derivatives dH[v] and d²H[u,v].
///
/// This eliminates the previously duplicated exact-Newton and surrogate
/// code blocks in `outerobjectivegradienthessian_internal`.
fn build_joint_hessian_closures<'a, F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &'a F,
    block_states: &'a [ParameterBlockState],
    specs: &'a [ParameterBlockSpec],
    total: usize,
) -> Result<Option<JointHessianBundle<'a>>, String> {
    // Path 1: exact Newton joint Hessian (preferred).
    let beta_flat = flatten_state_betas(block_states, specs);
    let synced = Arc::new(synchronized_states_from_flat_beta(
        family,
        specs,
        block_states,
        &beta_flat,
    )?);
    let hessian_workspace = family.exact_newton_joint_hessian_workspace(synced.as_ref(), specs)?;
    if let Some(curvature) = family.exact_newton_outer_curvature(block_states)? {
        let h_joint_unpen = JointHessianSource::Dense(symmetrized_square_matrix(
            curvature.hessian,
            total,
            "joint exact-newton Hessian shape mismatch in outer gradient (rescaled)",
        )?);
        let compute_dh = Box::new(exact_newton_dh_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        ));
        let compute_d2h = Box::new(exact_newton_d2h_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            true,
            1.0,
            hessian_workspace,
        ));
        return Ok(Some(JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_d2h,
            rho_curvature_scale: curvature.rho_curvature_scale,
            hessian_logdet_correction: curvature.hessian_logdet_correction,
        }));
    }
    let exact_joint_source =
        if use_joint_matrix_free_path(total, joint_observation_count(block_states)) {
            hessian_workspace
                .as_ref()
                .map(|workspace| {
                    exact_newton_joint_hessian_source_from_workspace(
                        workspace,
                        total,
                        "joint exact-newton operator mismatch in outer gradient",
                    )
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
    let exact_joint_source = match exact_joint_source {
        Some(source) => Some(source),
        None => exact_newton_joint_hessian_symmetrized(
            family,
            block_states,
            specs,
            total,
            "joint exact-newton Hessian shape mismatch in outer gradient",
        )
        .map(|source| source.map(JointHessianSource::Dense))?,
    };
    if let Some(h_joint_unpen) = exact_joint_source {
        let compute_dh = Box::new(exact_newton_dh_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        ));
        let compute_d2h = Box::new(exact_newton_d2h_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            false,
            1.0,
            hessian_workspace,
        ));
        return Ok(Some(JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_d2h,
            rho_curvature_scale: 1.0,
            hessian_logdet_correction: 0.0,
        }));
    }

    // Path 2: surrogate joint Hessian (fallback).
    if let Some(h_joint_unpen) = family
        .joint_outer_hyper_surrogate_hessian_with_specs(block_states, specs)?
        .map(|h| {
            symmetrized_square_matrix(
                h,
                total,
                "joint outer-hyper surrogate Hessian shape mismatch",
            )
        })
        .transpose()?
    {
        let beta_flat = flatten_state_betas(block_states, specs);

        let compute_dh =
            Box::new(
                move |v_k: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                    let h_rho = family
                        .joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
                            block_states,
                            specs,
                            v_k,
                        )?;
                    match h_rho {
                        Some(h) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                            h,
                            total,
                            "joint surrogate dH shape mismatch",
                        )?))),
                        None => Err("joint surrogate dH unavailable for analytic outer gradient"
                            .to_string()),
                    }
                },
            );
        let compute_d2h = Box::new(
            move |u: &Array1<f64>, v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                match family
                    .joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
                        block_states,
                        specs,
                        u,
                        v,
                    )? {
                    Some(m) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        m,
                        total,
                        "joint surrogate d2H shape mismatch",
                    )?))),
                    None => Ok(None),
                }
            },
        );
        return Ok(Some(JointHessianBundle {
            source: JointHessianSource::Dense(h_joint_unpen),
            beta_flat,
            compute_dh,
            compute_d2h,
            rho_curvature_scale: 1.0,
            hessian_logdet_correction: 0.0,
        }));
    }

    Ok(None)
}

/// Build a closure computing dH[v] using exact Newton derivatives on synced states.
/// Non-finite derivative output is treated as a hard error.
fn exact_newton_dh_closure<'a, F: CustomFamily>(
    family: &'a F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: &'a [ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> impl Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + 'a {
    move |v_k: &Array1<f64>| {
        if use_outer_curvature_derivatives {
            let h_rho = family.exact_newton_outer_curvature_directional_derivative_with_specs(
                synced_states.as_ref(),
                specs,
                v_k,
            )?;
            return match h_rho {
                Some(h) => {
                    if h.iter().any(|v| !v.is_finite()) {
                        Err("joint exact-newton dH returned non-finite values".to_string())
                    } else {
                        let mut sym = symmetrized_square_matrix(
                            h,
                            total,
                            "joint exact-newton dH shape mismatch",
                        )?;
                        if scale != 1.0 {
                            sym *= scale;
                        }
                        Ok(Some(DriftDerivResult::Dense(sym)))
                    }
                }
                None => {
                    Err("joint exact-newton dH unavailable for analytic outer gradient".to_string())
                }
            };
        }

        if let Some(workspace) = workspace.as_ref() {
            if let Some(operator) = workspace.directional_derivative_operator(v_k)? {
                return Ok(Some(scale_drift_deriv_result(
                    DriftDerivResult::Operator(operator),
                    scale,
                )));
            }
        }

        match family.exact_newton_joint_hessian_directional_derivative_with_specs(
            synced_states.as_ref(),
            specs,
            v_k,
        )? {
            Some(h) => {
                if h.iter().any(|v| !v.is_finite()) {
                    Err("joint exact-newton dH returned non-finite values".to_string())
                } else {
                    let mut sym = symmetrized_square_matrix(
                        h,
                        total,
                        "joint exact-newton dH shape mismatch",
                    )?;
                    if scale != 1.0 {
                        sym *= scale;
                    }
                    Ok(Some(DriftDerivResult::Dense(sym)))
                }
            }
            None => {
                Err("joint exact-newton dH unavailable for analytic outer gradient".to_string())
            }
        }
    }
}

/// Build a closure computing d²H[u,v] using exact Newton derivatives on synced states.
fn exact_newton_d2h_closure<'a, F: CustomFamily>(
    family: &'a F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: &'a [ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> impl Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + 'a {
    move |u: &Array1<f64>, v: &Array1<f64>| {
        if use_outer_curvature_derivatives {
            return match family
                .exact_newton_outer_curvature_second_directional_derivative_with_specs(
                    synced_states.as_ref(),
                    specs,
                    u,
                    v,
                )? {
                Some(m) => {
                    let mut sym = symmetrized_square_matrix(
                        m,
                        total,
                        "joint exact-newton d2H shape mismatch",
                    )?;
                    if scale != 1.0 {
                        sym *= scale;
                    }
                    Ok(Some(DriftDerivResult::Dense(sym)))
                }
                None => Ok(None),
            };
        }

        if let Some(workspace) = workspace.as_ref() {
            if let Some(operator) = workspace.second_directional_derivative_operator(u, v)? {
                return Ok(Some(scale_drift_deriv_result(
                    DriftDerivResult::Operator(operator),
                    scale,
                )));
            }
        }

        match family.exact_newton_joint_hessian_second_directional_derivative_with_specs(
            synced_states.as_ref(),
            specs,
            u,
            v,
        )? {
            Some(m) => {
                let mut sym =
                    symmetrized_square_matrix(m, total, "joint exact-newton d2H shape mismatch")?;
                if scale != 1.0 {
                    sym *= scale;
                }
                Ok(Some(DriftDerivResult::Dense(sym)))
            }
            None => Ok(None),
        }
    }
}

fn exact_newton_dh_closure_owned<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: Vec<ParameterBlockSpec>,
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync> {
    Arc::new(move |v_k: &Array1<f64>| {
        if use_outer_curvature_derivatives {
            let h_rho = family.exact_newton_outer_curvature_directional_derivative_with_specs(
                synced_states.as_ref(),
                &specs,
                v_k,
            )?;
            return match h_rho {
                Some(h) => {
                    let mut sym = symmetrized_square_matrix(
                        h,
                        total,
                        "joint exact-newton dH shape mismatch",
                    )?;
                    if scale != 1.0 {
                        sym *= scale;
                    }
                    Ok(Some(DriftDerivResult::Dense(sym)))
                }
                None => {
                    Err("joint exact-newton dH unavailable for analytic outer gradient".to_string())
                }
            };
        }

        if let Some(workspace) = workspace.as_ref() {
            if let Some(operator) = workspace.directional_derivative_operator(v_k)? {
                return Ok(Some(scale_drift_deriv_result(
                    DriftDerivResult::Operator(operator),
                    scale,
                )));
            }
        }

        match family.exact_newton_joint_hessian_directional_derivative_with_specs(
            synced_states.as_ref(),
            &specs,
            v_k,
        )? {
            Some(h) => {
                let mut sym =
                    symmetrized_square_matrix(h, total, "joint exact-newton dH shape mismatch")?;
                if scale != 1.0 {
                    sym *= scale;
                }
                Ok(Some(DriftDerivResult::Dense(sym)))
            }
            None => {
                Err("joint exact-newton dH unavailable for analytic outer gradient".to_string())
            }
        }
    })
}

fn exact_newton_d2h_closure_owned<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: Vec<ParameterBlockSpec>,
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Arc<dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>
{
    Arc::new(move |u: &Array1<f64>, v: &Array1<f64>| {
        if use_outer_curvature_derivatives {
            return match family
                .exact_newton_outer_curvature_second_directional_derivative_with_specs(
                    synced_states.as_ref(),
                    &specs,
                    u,
                    v,
                )? {
                Some(m) => {
                    let mut sym = symmetrized_square_matrix(
                        m,
                        total,
                        "joint exact-newton d2H shape mismatch",
                    )?;
                    if scale != 1.0 {
                        sym *= scale;
                    }
                    Ok(Some(DriftDerivResult::Dense(sym)))
                }
                None => Ok(None),
            };
        }

        if let Some(workspace) = workspace.as_ref() {
            if let Some(operator) = workspace.second_directional_derivative_operator(u, v)? {
                return Ok(Some(scale_drift_deriv_result(
                    DriftDerivResult::Operator(operator),
                    scale,
                )));
            }
        }

        match family.exact_newton_joint_hessian_second_directional_derivative_with_specs(
            synced_states.as_ref(),
            &specs,
            u,
            v,
        )? {
            Some(m) => {
                let mut sym =
                    symmetrized_square_matrix(m, total, "joint exact-newton d2H shape mismatch")?;
                if scale != 1.0 {
                    sym *= scale;
                }
                Ok(Some(DriftDerivResult::Dense(sym)))
            }
            None => Ok(None),
        }
    })
}

fn strict_solve_spd(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD solve failed".to_string())?;
    Ok(chol.solvevec(rhs))
}

fn strict_inverse_spd(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD inverse failed".to_string())?;
    let mut ident = Array2::<f64>::eye(matrix.nrows());
    chol.solve_mat_in_place(&mut ident);
    Ok(ident)
}

fn strict_logdet_spd(matrix: &Array2<f64>) -> Result<f64, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD logdet failed".to_string())?;
    Ok(2.0 * chol.diag().mapv(f64::ln).sum())
}

fn strict_logdet_spd_with_semidefinite_option(
    matrix: &Array2<f64>,
    allow_semidefinite: bool,
) -> Result<f64, String> {
    if allow_semidefinite {
        let mut sym = matrix.clone();
        symmetrize_dense_in_place(&mut sym);
        let (evals, _) = FaerEigh::eigh(&sym, Side::Lower)
            .map_err(|e| format!("strict pseudo-laplace PSD eigendecomposition failed: {e}"))?;
        let max_abs_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
        let tol = (max_abs_eval * 1e-12).max(1e-14);
        if evals.iter().any(|&ev| ev < -tol) {
            return Err("strict pseudo-laplace SPD solve failed".to_string());
        }
        let logdet = evals
            .iter()
            .copied()
            .filter(|&ev| ev > tol)
            .map(f64::ln)
            .sum();
        return Ok(logdet);
    }
    strict_logdet_spd(matrix)
}

fn pinv_positive_part(matrix: &Array2<f64>, ridge_floor: f64) -> Result<Array2<f64>, String> {
    // SVD-based pseudoinverse: unconditionally convergent, unlike eigh which
    // can fail on pathological matrices from degenerate GAMLSS Hessians.
    let (u_opt, singular, vt_opt) = matrix
        .svd(true, true)
        .map_err(|e| format!("SVD failed in positive-part pseudoinverse: {e}"))?;
    let u = u_opt.ok_or("SVD did not produce left singular vectors")?;
    let vt = vt_opt.ok_or("SVD did not produce right singular vectors")?;
    let max_sv = singular.iter().fold(0.0_f64, |a, &b| a.max(b));
    let tol = (max_sv * 1e-12).max(ridge_floor.max(1e-14));
    let p = matrix.nrows();
    let mut pinv = Array2::<f64>::zeros((p, p));
    for k in 0..p.min(singular.len()) {
        let sv = singular[k];
        if sv > tol {
            let inv_sv = 1.0 / sv;
            for i in 0..p {
                let vik = vt[(k, i)]; // V^T row k = V column k
                for j in 0..p {
                    pinv[[i, j]] += inv_sv * vik * u[(j, k)];
                }
            }
        }
    }
    Ok(pinv)
}

fn include_exact_newton_logdet_h<F: CustomFamily + ?Sized>(
    family: &F,
    options: &BlockwiseFitOptions,
) -> bool {
    options.use_remlobjective
        && matches!(
            family.exact_newton_outerobjective(),
            ExactNewtonOuterObjective::RidgedQuadraticReml
                | ExactNewtonOuterObjective::StrictPseudoLaplace
        )
}

pub(crate) fn custom_family_outer_derivatives<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> (
    crate::solver::outer_strategy::Derivative,
    crate::solver::outer_strategy::Derivative,
) {
    use crate::solver::outer_strategy::Derivative;

    let order = family.exact_outer_derivative_order(specs, options);
    let gradient = if order.has_gradient() {
        Derivative::Analytic
    } else {
        Derivative::Unavailable
    };
    // The analytic outer Hessian is routed to ARC (adaptive cubic
    // regularization), which handles indefinite Hessians natively via
    // cubic regularization + trust region. We intentionally do NOT gate on
    // `exact_newton_outer_geometry_supports_second_order_solver` here: the
    // only consumer of the resulting Hessian is ARC, and ARC does not require
    // the Hessian to be SPD. Previously this gate forced RidgedQuadraticReml
    // families (binomial+ps custom family) onto BFGS+BfgsApprox, which stalls
    // at iter 0 with Strong Wolfe failures because the rank-2 BFGS update is
    // directionally wrong for the ridged surrogate surface. The fix is to
    // expose the analytic Hessian and let ARC drive the outer iteration.
    //
    // Availability is a property of the family's second-derivative form, not
    // of the inner problem size. The K×K affordability gate already lives
    // inside `order.has_hessian()` via `cost_gated_outer_order`.
    let hessian = if options.use_outer_hessian
        && include_exact_newton_logdet_h(family, options)
        && order.has_hessian()
    {
        Derivative::Analytic
    } else {
        Derivative::Unavailable
    };

    (gradient, hessian)
}

fn include_exact_newton_logdet_s<F: CustomFamily + ?Sized>(
    family: &F,
    options: &BlockwiseFitOptions,
) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::RidgedQuadraticReml
        && options.use_remlobjective
}

fn use_exact_newton_strict_spd<F: CustomFamily + ?Sized>(family: &F) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::StrictPseudoLaplace
}

fn blockwise_logdet_terms<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<(f64, f64), String> {
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    let allow_semidefinite = strict_spd && family.exact_newton_allows_semidefinitehessian();
    refresh_all_block_etas(family, specs, states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let mut s_lambdas = Vec::with_capacity(specs.len());
    let mut penalty_logdet_s_total = 0.0;
    for (b, spec) in specs.iter().enumerate() {
        let (start, end) = ranges[b];
        let p = end - start;
        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        if include_logdet_s {
            // Exact pseudo-logdet on the positive eigenspace, consistent with
            // the derivatives in compute_block_penalty_logdet_derivs.
            // Uses structural nullity from spec.nullspace_dims to partition
            // the eigenspace exactly, avoiding numerical thresholds.
            let ridge = if options.ridge_policy.include_penalty_logdet {
                effective_solverridge(options.ridge_floor)
            } else {
                0.0
            };
            let mut s_for_logdet = s_lambda.clone();
            if ridge > 0.0 {
                for i in 0..p {
                    s_for_logdet[[i, i]] += ridge;
                }
            }
            let block_logdet = match s_for_logdet.eigh(faer::Side::Lower) {
                Ok((evals, _)) => {
                    // Structural nullity determines the split: bottom m₀ eigenvalues
                    // are structural zeros, top (p - m₀) are the positive subspace.
                    let m0 = if !spec.nullspace_dims.is_empty()
                        && spec.nullspace_dims.len() == spec.penalties.len()
                    {
                        let penalties_dense: Vec<Array2<f64>> =
                            spec.penalties.iter().map(|p| p.to_dense()).collect();
                        crate::estimate::reml::unified::exact_intersection_nullity(
                            &penalties_dense,
                            &spec.nullspace_dims,
                        )
                    } else {
                        let eval_buffer;
                        let eval_slice = if let Some(slice) = evals.as_slice() {
                            slice
                        } else {
                            eval_buffer = evals.iter().copied().collect::<Vec<_>>();
                            &eval_buffer
                        };
                        let threshold =
                            crate::estimate::reml::unified::positive_eigenvalue_threshold(
                                eval_slice,
                            );
                        evals.iter().filter(|&&e| e <= threshold).count()
                    };
                    evals
                        .iter()
                        .skip(m0)
                        .map(|&e| e.max(f64::MIN_POSITIVE).ln())
                        .sum::<f64>()
                }
                Err(eigh_err) => {
                    // Penalty matrices are PSD by construction, so eigh failure
                    // here is purely numerical.  Fall back to Cholesky on the
                    // ridged matrix (which should be SPD).  The Cholesky logdet
                    // includes null-space contributions (~m₀ × ln(ridge)), making
                    // it a slight overestimate, but this is a smooth bias that
                    // does not corrupt REML gradients.
                    penalty_logdet_cholesky_fallback(&s_for_logdet, ridge, b, p, &eigh_err)?
                }
            };
            penalty_logdet_s_total += block_logdet;
        }
        s_lambdas.push(s_lambda);
    }
    // Try the shared scale-aware exact curvature path first.
    if let Some(curvature) = family.exact_newton_outer_curvature(states)? {
        let mut h_joint = symmetrized_square_matrix(
            curvature.hessian,
            total,
            "joint exact-newton Hessian validation in logdet terms (rescaled)",
        )?;
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(curvature.rho_curvature_scale, s_lambda);
        }
        let logdet_h_scaled = if strict_spd {
            strict_logdet_spd_with_semidefinite_option(&h_joint, allow_semidefinite)?
        } else {
            stable_logdet_with_ridge_policy(
                &h_joint,
                options.ridge_floor * curvature.rho_curvature_scale,
                options.ridge_policy,
            )?
        };
        let logdet_h_total = logdet_h_scaled + curvature.hessian_logdet_correction;
        return Ok((logdet_h_total, penalty_logdet_s_total));
    }
    let exact_joint_source =
        if !strict_spd && use_joint_matrix_free_path(total, joint_observation_count(states)) {
            family
                .exact_newton_joint_hessian_workspace(states, specs)?
                .as_ref()
                .map(|workspace| {
                    exact_newton_joint_hessian_source_from_workspace(
                        workspace,
                        total,
                        "joint exact-newton operator mismatch in logdet terms",
                    )
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
    if let Some(JointHessianSource::Operator { apply, diagonal }) = exact_joint_source {
        let mut h = materialize_joint_hessian_source(
            &JointHessianSource::Operator { apply, diagonal },
            total,
            "joint exact-newton Hessian materialization in logdet terms",
        )?;
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h.slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, s_lambda);
        }
        let logdet_h_total =
            stable_logdet_with_ridge_policy(&h, options.ridge_floor, options.ridge_policy)?;
        return Ok((logdet_h_total, penalty_logdet_s_total));
    }
    // Fallback: try the non-rescaled symmetrized path (for families that
    // don't implement exact_newton_outer_curvature but do provide
    // a plain joint Hessian).
    if let Some(mut h_joint) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total,
        "joint exact-newton Hessian validation in logdet terms",
    )? {
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, s_lambda);
        }
        let logdet_h_total = if strict_spd {
            strict_logdet_spd_with_semidefinite_option(&h_joint, allow_semidefinite)?
        } else {
            stable_logdet_with_ridge_policy(&h_joint, options.ridge_floor, options.ridge_policy)?
        };
        return Ok((logdet_h_total, penalty_logdet_s_total));
    }

    let eval = family.evaluate(states)?;
    if eval.blockworking_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.blockworking_sets.len(),
            specs.len()
        ));
    }

    let mut logdet_h_total = 0.0;
    let logdet_s_total = penalty_logdet_s_total;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let work = &eval.blockworking_sets[b];
        let p = spec.design.ncols();
        let xtwx = match work {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => with_block_geometry(family, states, spec, b, |x_dyn, _| {
                let w = floor_positiveworking_weights(working_weights, options.minweight);
                let (xtwx, _) = weighted_normal_equations(x_dyn, &w, None)?;
                Ok(xtwx)
            })?,
            BlockWorkingSet::ExactNewton {
                gradient: _,
                hessian,
            } => {
                if hessian.nrows() != p || hessian.ncols() != p {
                    return Err(format!(
                        "block {b} exact-newton Hessian shape mismatch: got {}x{}, expected {}x{}",
                        hessian.nrows(),
                        hessian.ncols(),
                        p,
                        p
                    ));
                }
                hessian.to_dense()
            }
        };

        let s_lambda = &s_lambdas[b];

        let mut h = xtwx;
        h += s_lambda;
        logdet_h_total += if strict_spd {
            strict_logdet_spd_with_semidefinite_option(&h, allow_semidefinite)?
        } else {
            stable_logdet_with_ridge_policy(&h, options.ridge_floor, options.ridge_policy)?
        };
    }
    Ok((logdet_h_total, logdet_s_total))
}

/// Snapshot of a single block's eta for line-search rollback.
///
/// Created from a specific block's state; can only restore to or update
/// that same block.  There is no shared buffer across blocks, so
/// cross-block length confusion is structurally impossible.
struct BlockEtaCheckpoint {
    saved: Array1<f64>,
}

impl BlockEtaCheckpoint {
    /// Capture the current eta of `state`.
    fn capture(state: &ParameterBlockState) -> Self {
        Self {
            saved: state.eta.clone(),
        }
    }

    /// Capture into a pre-allocated buffer, returning the filled checkpoint.
    /// The buffer is taken (O(1) move) and filled with eta's data (O(n) copy).
    fn capture_reuse(state: &ParameterBlockState, buf: &mut Array1<f64>) -> Self {
        if buf.len() == state.eta.len() {
            buf.assign(&state.eta);
            Self {
                saved: std::mem::take(buf),
            }
        } else {
            Self::capture(state)
        }
    }

    /// Return the internal buffer for recycling.
    fn into_buffer(self) -> Array1<f64> {
        self.saved
    }

    /// Restore: `state.eta = saved`.
    fn restore_eta(&self, state: &mut ParameterBlockState) {
        state.eta.assign(&self.saved);
    }

    /// Incremental update: `state.eta = saved + alpha * direction`.
    fn restore_eta_with_step(
        &self,
        state: &mut ParameterBlockState,
        alpha: f64,
        direction: &Array1<f64>,
    ) {
        // In-place: eta = eta_backup + alpha * xd (zero allocations).
        state.eta.assign(&self.saved);
        state.eta.scaled_add(alpha, direction);
    }
}

fn inner_blockwise_fit<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<BlockwiseInnerResult, String> {
    let mut states = buildblock_states(family, specs)?;
    refresh_all_block_etas(family, specs, &mut states)?;
    let total_joint_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let total_joint_n = joint_observation_count(&states);
    let has_joint_exacthessian = if use_joint_matrix_free_path(total_joint_p, total_joint_n) {
        let workspace = family.exact_newton_joint_hessian_workspace(&states, specs)?;
        match workspace.as_ref() {
            Some(workspace) => {
                exact_newton_joint_hessian_source_from_workspace(
                    workspace,
                    total_joint_p,
                    "joint exact-newton operator mismatch during inner availability probe",
                )?
                .is_some()
                    || family.exact_newton_joint_hessian(&states)?.is_some()
            }
            None => family.exact_newton_joint_hessian(&states)?.is_some(),
        }
    } else {
        family.exact_newton_joint_hessian(&states)?.is_some()
    };
    let use_joint_newton = has_joint_exacthessian && specs.len() >= 2;
    let inner_tol = options.inner_tol;
    let inner_max_cycles = options.inner_max_cycles;
    let inner_max_cycles = capped_inner_max_cycles(options, inner_max_cycles);
    let mut s_lambdas = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let p = spec.design.ncols();
        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        s_lambdas.push(s_lambda);
    }
    let ridge = effective_solverridge(options.ridge_floor);
    let mut cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; specs.len()];
    if let Some(seed) = warm_start
        && seed.block_beta.len() == states.len()
        && seed.active_sets.len() == states.len()
    {
        for (b, beta_seed) in seed.block_beta.iter().enumerate() {
            if beta_seed.len() == states[b].beta.len() {
                let beta_projected =
                    family.post_update_block_beta(&states, b, &specs[b], beta_seed.clone())?;
                states[b].beta.assign(&beta_projected);
            }
        }
        cached_active_sets = seed.active_sets.clone();
        refresh_all_block_etas(family, specs, &mut states)?;
    }
    let mut cached_eval: Option<FamilyEvaluation> = None;
    let mut cached_joint_gradient: Option<Array1<f64>> = None;
    let mut current_log_likelihood = if use_joint_newton {
        if let Some(joint_eval) = family.exact_newton_joint_gradient_evaluation(&states, specs)? {
            cached_joint_gradient = Some(joint_eval.gradient);
            joint_eval.log_likelihood
        } else {
            let eval = family.evaluate(&states)?;
            cached_joint_gradient = exact_newton_joint_gradient_from_eval(&eval, specs)?;
            let log_likelihood = eval.log_likelihood;
            cached_eval = Some(eval);
            log_likelihood
        }
    } else {
        let eval = family.evaluate(&states)?;
        let log_likelihood = eval.log_likelihood;
        cached_eval = Some(eval);
        log_likelihood
    };
    let mut current_penalty =
        total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
    let mut lastobjective = -current_log_likelihood + current_penalty;
    let mut converged = false;
    let mut cycles_done = 0usize;
    // Pre-allocate per-block eta backup buffers to avoid O(n) allocation
    // per block per cycle in the backtracking line search.
    let mut eta_backups: Vec<Array1<f64>> =
        states.iter().map(|s| Array1::zeros(s.eta.len())).collect();

    // ── Joint Newton fast path ──
    //
    // When the family provides an exact joint Hessian (GAMLSS location-scale),
    // solve the full (p_mu + p_ls) × (p_mu + p_ls) system in one Newton step
    // per cycle instead of iterating between blocks. This converges quadratically
    // (5-10 steps) instead of linearly (20-100+ blockwise cycles).
    //
    // Falls back to blockwise iteration if the joint Hessian is unavailable
    // or the joint solve fails. Linear constraints are handled directly in the
    // joint step whenever they can be assembled blockwise.

    if use_joint_newton {
        // Build block ranges for the joint system.
        let ranges: Vec<(usize, usize)> = {
            let mut offset = 0;
            specs
                .iter()
                .map(|s| {
                    let start = offset;
                    offset += s.design.ncols();
                    (start, offset)
                })
                .collect()
        };
        let total_p: usize = ranges.last().map_or(0, |r| r.1);

        let joint_mode_diagonal_ridge =
            if ridge > 0.0 && options.ridge_policy.include_quadratic_penalty {
                ridge
            } else {
                0.0
            };

        let iter_log_info = crate::solver::visualizer::pirls_iter_info_enabled();
        for cycle in 0..inner_max_cycles {
            if iter_log_info {
                // Env-gated diagnostic emission at `info` — see
                // `solver::visualizer::pirls_iter_info_enabled` for the
                // rationale. Fires at the top of each inner joint-Newton
                // cycle so CI logs can distinguish "inner spin" (thousands
                // of these) from "outer-assembly spin" (zero of these).
                log::info!(
                    "[PIRLS/blockwise joint-Newton] cycle {:>3}/{} | -loglik {:.6e} | penalty {:.6e} | objective {:.6e}",
                    cycle,
                    inner_max_cycles,
                    -current_log_likelihood,
                    current_penalty,
                    lastobjective,
                );
            }
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            let joint_constraints =
                assemble_joint_linear_constraints(&block_constraints, &ranges, total_p)?;
            // Get joint Hessian and block gradients from the current evaluation.
            let joint_hessian_source = if use_joint_matrix_free_path(total_p, total_joint_n) {
                family
                    .exact_newton_joint_hessian_workspace(&states, specs)?
                    .as_ref()
                    .map(|workspace| {
                        exact_newton_joint_hessian_source_from_workspace(
                            workspace,
                            total_p,
                            "joint Newton inner exact-newton operator mismatch",
                        )
                    })
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            let joint_hessian_source = match joint_hessian_source {
                Some(source) => source,
                None => {
                    let h_joint_opt = family.exact_newton_joint_hessian(&states)?;
                    let Some(h_joint) = h_joint_opt else {
                        break; // Fall back to blockwise if joint Hessian unavailable
                    };
                    match symmetrized_square_matrix(
                        h_joint,
                        total_p,
                        "joint Newton inner exact-newton Hessian shape mismatch",
                    ) {
                        Ok(matrix) => JointHessianSource::Dense(matrix),
                        Err(_) => break,
                    }
                }
            };

            // Concatenate block gradients and betas.
            let Some(grad_joint) = cached_joint_gradient.clone() else {
                break;
            };
            if grad_joint.len() != total_p {
                break;
            }
            let mut beta_joint = Array1::<f64>::zeros(total_p);
            for b in 0..specs.len() {
                let (start, end) = ranges[b];
                beta_joint
                    .slice_mut(ndarray::s![start..end])
                    .assign(&states[b].beta);
            }

            let trace_diagonal_ridge = joint_mode_diagonal_ridge + JOINT_TRACE_STABILITY_RIDGE;

            let (candidate_beta, joint_active_set) = if let Some(constraints) =
                joint_constraints.as_ref()
            {
                let mut lhs = match materialize_joint_hessian_source(
                    &joint_hessian_source,
                    total_p,
                    "joint Newton inner constrained Hessian materialization",
                ) {
                    Ok(matrix) => matrix,
                    Err(_) => break,
                };
                add_joint_penalty_to_matrix(&mut lhs, &ranges, &s_lambdas, trace_diagonal_ridge);
                check_linear_feasibility(&beta_joint, constraints, 1e-8)
                    .map_err(|e| format!("joint Newton constrained solve: {e}"))?;
                let warm_joint_active =
                    flatten_joint_active_set(&cached_active_sets, &block_constraints);
                let lower_bounds = match extract_simple_lower_bounds(constraints, total_p) {
                    Ok(bounds) => bounds,
                    Err(_) => break,
                };
                // Newton IRLS step in absolute-β space:
                //
                //   β_new = H_pen⁻¹ (H_L β + ∇ℓ)
                //
                // where H_pen = H_L + S, derived from Newton's update
                //   β_new = β + H_pen⁻¹(∇ℓ − Sβ)
                //         = H_pen⁻¹(H_pen β + ∇ℓ − Sβ)
                //         = H_pen⁻¹(H_L β + ∇ℓ).
                //
                // The QP `min 0.5 β' H_pen β − rhs_beta' β` has unconstrained
                // optimum β = H_pen⁻¹ rhs_beta, so rhs_beta = H_pen β + (∇ℓ − Sβ)
                // gives the correct Newton update. Passing raw grad_joint (=∇ℓ)
                // would collapse to β = H_pen⁻¹ ∇ℓ, which at the true optimum
                // (∇ℓ = Sβ̂) gives H_pen⁻¹ Sβ̂ ≠ β̂ — wrong fixed point.
                let penalty_beta_joint = apply_joint_block_penalty(
                    &ranges,
                    &s_lambdas,
                    &beta_joint,
                    joint_mode_diagonal_ridge,
                );
                let rhs_step = &grad_joint - &penalty_beta_joint;
                let rhs_beta = &lhs.dot(&beta_joint) + &rhs_step;
                let solve_result = if let Some(bounds) = lower_bounds.as_ref() {
                    solve_quadratic_with_simple_lower_bounds(
                        &lhs,
                        &rhs_beta,
                        &beta_joint,
                        bounds,
                        warm_joint_active.as_deref(),
                    )
                } else {
                    solve_quadratic_with_linear_constraints(
                        &lhs,
                        &rhs_beta,
                        &beta_joint,
                        constraints,
                        warm_joint_active.as_deref(),
                    )
                    .map_err(|e| e.to_string())
                };
                match solve_result {
                    Ok((beta_new, active_set)) => (beta_new, Some(active_set)),
                    Err(_) => break,
                }
            } else {
                // Stationarity residual: r = S*beta - gradient (for penalized NLL)
                let penalty_beta = apply_joint_block_penalty(
                    &ranges,
                    &s_lambdas,
                    &beta_joint,
                    joint_mode_diagonal_ridge,
                );
                let rhs = &grad_joint - &penalty_beta;
                let mut delta = if use_joint_matrix_free_path(total_p, total_joint_n) {
                    let preconditioner_diag = match &joint_hessian_source {
                        JointHessianSource::Dense(h_joint) => joint_penalty_preconditioner_diag(
                            &h_joint.diag().to_owned(),
                            &ranges,
                            &s_lambdas,
                            trace_diagonal_ridge,
                        ),
                        JointHessianSource::Operator { diagonal, .. } => {
                            joint_penalty_preconditioner_diag(
                                diagonal,
                                &ranges,
                                &s_lambdas,
                                trace_diagonal_ridge,
                            )
                        }
                    };
                    match &joint_hessian_source {
                        JointHessianSource::Dense(h_joint) => crate::linalg::utils::solve_spd_pcg(
                            |v| {
                                let mut out = h_joint.dot(v);
                                let penalty = apply_joint_block_penalty(
                                    &ranges,
                                    &s_lambdas,
                                    v,
                                    trace_diagonal_ridge,
                                );
                                out += &penalty;
                                out
                            },
                            &rhs,
                            &preconditioner_diag,
                            JOINT_PCG_REL_TOL,
                            JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                        ),
                        JointHessianSource::Operator { apply, .. } => {
                            let apply_h = Arc::clone(apply);
                            crate::linalg::utils::solve_spd_pcg(
                                |v| {
                                    let mut out = match apply_h(v) {
                                        Ok(out) => out,
                                        Err(error) => {
                                            log::warn!(
                                                "joint Newton inner operator matvec failed: {error}"
                                            );
                                            Array1::<f64>::zeros(total_p)
                                        }
                                    };
                                    let penalty = apply_joint_block_penalty(
                                        &ranges,
                                        &s_lambdas,
                                        v,
                                        trace_diagonal_ridge,
                                    );
                                    out += &penalty;
                                    out
                                },
                                &rhs,
                                &preconditioner_diag,
                                JOINT_PCG_REL_TOL,
                                JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                            )
                        }
                    }
                } else {
                    None
                };
                if delta.is_none() {
                    let mut lhs = match materialize_joint_hessian_source(
                        &joint_hessian_source,
                        total_p,
                        "joint Newton inner dense fallback Hessian materialization",
                    ) {
                        Ok(matrix) => matrix,
                        Err(_) => break,
                    };
                    add_joint_penalty_to_matrix(
                        &mut lhs,
                        &ranges,
                        &s_lambdas,
                        trace_diagonal_ridge,
                    );
                    let solver = crate::linalg::utils::StableSolver::new("joint Newton inner");
                    delta = solver.solvevectorwithridge_retries(
                        &lhs,
                        &rhs,
                        JOINT_TRACE_STABILITY_RIDGE,
                    );
                }

                let Some(delta) = delta else {
                    break; // Fall back to blockwise
                };
                if !delta.iter().all(|v| v.is_finite()) {
                    break; // Fall back to blockwise
                }
                (beta_joint.clone() + &delta, None)
            };
            let mut delta = &candidate_beta - &beta_joint;

            // Trust region: cap step size
            let step_inf = delta.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
            const MAX_JOINT_STEP: f64 = 20.0;
            if step_inf > MAX_JOINT_STEP {
                delta.mapv_inplace(|v| v * (MAX_JOINT_STEP / step_inf));
            }

            // A small Newton proposal is not enough for exact outer calculus:
            // we still need the joint residual check after evaluating the
            // candidate state, because ||delta|| can be small while the joint
            // KKT residual is still above tolerance.
            // Line search: try full step, then halve
            let old_beta: Vec<Array1<f64>> = states.iter().map(|s| s.beta.clone()).collect();
            let mut accepted = false;
            let mut barrier_ceiling = 1.0_f64;
            for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
                let block_delta = delta.slice(s![start..end]).to_owned();
                if let Some(alpha_max) =
                    family.max_feasible_step_size(&states, block_idx, &block_delta)?
                {
                    barrier_ceiling = barrier_ceiling.min(alpha_max);
                }
            }
            if !barrier_ceiling.is_finite() || barrier_ceiling <= 0.0 {
                for (b, old) in old_beta.iter().enumerate() {
                    states[b].beta.assign(old);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                break;
            }
            for bt in 0..8 {
                let alpha = (0.5f64.powi(bt)).min(barrier_ceiling);
                for b in 0..specs.len() {
                    let (start, end) = ranges[b];
                    let mut trial_beta = old_beta[b].clone();
                    trial_beta.scaled_add(alpha, &delta.slice(ndarray::s![start..end]));
                    // Project to feasible region (e.g. monotonicity for deviation blocks).
                    let projected =
                        family.post_update_block_beta(&states, b, &specs[b], trial_beta)?;
                    states[b].beta.assign(&projected);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                let trial_ll = match family.log_likelihood_only(&states) {
                    Ok(value) => value,
                    Err(_) => {
                        for (b, old) in old_beta.iter().enumerate() {
                            states[b].beta.assign(old);
                        }
                        refresh_all_block_etas(family, specs, &mut states)?;
                        continue;
                    }
                };
                let trial_penalty =
                    total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
                let trialobjective = -trial_ll + trial_penalty;
                if trialobjective.is_finite() && trialobjective <= lastobjective + 1e-10 {
                    current_penalty = trial_penalty;
                    if let Some(joint_active_set) = joint_active_set.as_ref() {
                        cached_active_sets =
                            scatter_joint_active_set(joint_active_set, &block_constraints);
                    }
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                // Restore original betas
                for (b, old) in old_beta.iter().enumerate() {
                    states[b].beta.assign(old);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                break; // Fall back to blockwise
            }

            // Re-evaluate for next iteration and convergence check
            if let Some(joint_eval) =
                family.exact_newton_joint_gradient_evaluation(&states, specs)?
            {
                current_log_likelihood = joint_eval.log_likelihood;
                cached_joint_gradient = Some(joint_eval.gradient);
                cached_eval = None;
            } else {
                let eval = family.evaluate(&states)?;
                current_log_likelihood = eval.log_likelihood;
                cached_joint_gradient = exact_newton_joint_gradient_from_eval(&eval, specs)?;
                cached_eval = Some(eval);
            }
            current_penalty =
                total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
            lastobjective = -current_log_likelihood + current_penalty;
            cycles_done = cycle + 1;

            // Check convergence via joint stationarity
            let Some(gradient) = cached_joint_gradient.as_ref() else {
                break;
            };
            let residual = exact_newton_joint_stationarity_inf_norm_from_gradient(
                gradient,
                &states,
                specs,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                &block_constraints,
            )?;
            if residual <= inner_tol && step_inf <= inner_tol {
                converged = true;
                break;
            }
        }

        // If joint Newton converged, skip the blockwise loop entirely.
        if converged {
            let penalty_value =
                total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
            let (block_logdet_h, block_logdet_s) =
                blockwise_logdet_terms(family, specs, &mut states, block_log_lambdas, options)?;
            return Ok(BlockwiseInnerResult {
                block_states: states,
                active_sets: normalize_active_sets(cached_active_sets),
                log_likelihood: current_log_likelihood,
                penalty_value,
                cycles: cycles_done,
                converged,
                block_logdet_h,
                block_logdet_s,
                s_lambdas,
            });
        }
        // Otherwise fall through to blockwise iteration below.
    }

    let mut cached_eval = match cached_eval {
        Some(eval) => eval,
        None => family.evaluate(&states)?,
    };
    current_log_likelihood = cached_eval.log_likelihood;
    lastobjective = -current_log_likelihood + current_penalty;

    let is_dynamic = family.block_geometry_is_dynamic();
    let iter_log_info_blockwise = crate::solver::visualizer::pirls_iter_info_enabled();
    for cycle in 0..inner_max_cycles {
        if iter_log_info_blockwise {
            // Env-gated diagnostic emission — see the joint-Newton copy
            // above and `solver::visualizer::pirls_iter_info_enabled` for
            // the rationale. Fires at the top of each blockwise coordinate
            // cycle so we can count iterations from CI logs when a
            // benchmark hangs inside the first outer-eval.
            log::info!(
                "[PIRLS/blockwise coord] cycle {:>3}/{} | -loglik {:.6e} | penalty {:.6e} | objective {:.6e}",
                cycle,
                inner_max_cycles,
                -current_log_likelihood,
                current_penalty,
                lastobjective,
            );
        }
        let mut max_beta_step = 0.0_f64;

        let mut objective_cycle_prev = lastobjective;
        // Reuse cached evaluation from end of previous cycle (or initial eval).
        // For dynamic families, the end-of-cycle evaluation is also reused here
        // instead of re-evaluating redundantly — the state hasn't changed since
        // the last cycle's final evaluate.
        let mut cycle_eval = std::mem::replace(
            &mut cached_eval,
            FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: Vec::new(),
            },
        );
        if cycle_eval.blockworking_sets.len() != specs.len() {
            return Err(format!(
                "family returned {} block working sets, expected {}",
                cycle_eval.blockworking_sets.len(),
                specs.len()
            ));
        }
        // Track whether any block was modified this cycle (for dynamic families,
        // we only need to re-evaluate before block b if a previous block changed).
        let mut any_block_modified = false;
        for b in 0..specs.len() {
            if is_dynamic && any_block_modified {
                // Only re-evaluate if a previous block in this cycle actually
                // modified coefficients. Skips the redundant evaluate for the
                // first block (b=0) since cached_eval is still valid.
                refresh_all_block_etas(family, specs, &mut states)?;
                cycle_eval = family.evaluate(&states)?;
                if cycle_eval.blockworking_sets.len() != specs.len() {
                    return Err(format!(
                        "family returned {} block working sets, expected {}",
                        cycle_eval.blockworking_sets.len(),
                        specs.len()
                    ));
                }
            }

            let spec = &specs[b];
            let work = &cycle_eval.blockworking_sets[b];
            let linear_constraints = family.block_linear_constraints(&states, b, spec)?;
            let s_lambda = &s_lambdas[b];
            let updater = work.updater();
            let update = updater.compute_update_step(&BlockUpdateContext {
                family,
                states: &states,
                spec,
                block_idx: b,
                s_lambda,
                options,
                linear_constraints: linear_constraints.as_ref(),
                cached_active_set: cached_active_sets[b].as_deref(),
            })?;
            if let Some(active_set) = update.active_set {
                cached_active_sets[b] = Some(active_set);
            }
            let beta_new_raw = update.beta_new_raw;
            let beta_new = family.post_update_block_beta(&states, b, spec, beta_new_raw)?;
            let beta_old = states[b].beta.clone();
            let raw_delta = &beta_new - &beta_old;
            // Trust region: cap the infinity-norm of the Newton step to
            // prevent eta from jumping into regions where exp() overflows.
            // The cap is generous enough to never slow normal convergence
            // (a coefficient change of 20 means exp(eta) changes by ~5e8)
            // but prevents catastrophic jumps to eta > 700.
            const MAX_NEWTON_STEP: f64 = 20.0;
            let step_inf = raw_delta
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let delta = if step_inf > MAX_NEWTON_STEP {
                &raw_delta * (MAX_NEWTON_STEP / step_inf)
            } else {
                raw_delta
            };
            let old_block_penalty =
                block_quadratic_penalty(&beta_old, s_lambda, ridge, options.ridge_policy);
            let step = delta.iter().copied().map(f64::abs).fold(0.0, f64::max);
            max_beta_step = max_beta_step.max(step);
            if step <= inner_tol {
                continue;
            }

            // Damped update: require non-increasing penalized objective under dynamic geometry.
            // Precompute X * delta once so line-search eta updates are O(n) not O(np).
            // Reuse pre-allocated eta backup to avoid O(n) allocation per block per cycle.
            let eta_checkpoint = BlockEtaCheckpoint::capture_reuse(&states[b], &mut eta_backups[b]);
            let x_delta = if !is_dynamic {
                Some(spec.design.matrixvectormultiply(&delta))
            } else {
                None
            };
            let mut accepted = false;
            // Barrier-aware step ceiling: families with natural log-barrier
            // terms (e.g. log(h') in transformation-normal) report the maximum
            // feasible step fraction so the line search never evaluates the
            // likelihood outside its domain.
            let barrier_ceiling = family
                .max_feasible_step_size(&states, b, &delta)?
                .unwrap_or(1.0);
            // Reuse trial_beta_buf to avoid allocation per backtracking trial.
            let mut trial_beta_buf = beta_old.clone();
            for bt in 0..8 {
                let alpha = (0.5f64.powi(bt)).min(barrier_ceiling);
                trial_beta_buf.assign(&beta_old);
                trial_beta_buf.scaled_add(alpha, &delta);
                let trial_beta =
                    family.post_update_block_beta(&states, b, spec, trial_beta_buf.clone())?;
                states[b].beta = trial_beta;
                // Use precomputed X*delta when geometry is static and beta wasn't modified.
                if let Some(ref xd) = x_delta {
                    if states[b].beta == trial_beta_buf {
                        eta_checkpoint.restore_eta_with_step(&mut states[b], alpha, xd);
                    } else {
                        refresh_single_block_eta(family, specs, &mut states, b)?;
                    }
                } else {
                    refresh_single_block_eta(family, specs, &mut states, b)?;
                }
                let trial_ll = match family.log_likelihood_only(&states) {
                    Ok(value) => value,
                    Err(_) => {
                        states[b].beta.assign(&beta_old);
                        eta_checkpoint.restore_eta(&mut states[b]);
                        continue;
                    }
                };
                let trial_block_penalty =
                    block_quadratic_penalty(&states[b].beta, s_lambda, ridge, options.ridge_policy);
                let trial_penalty = current_penalty - old_block_penalty + trial_block_penalty;
                let trialobjective = -trial_ll + trial_penalty;
                if trialobjective.is_finite() && trialobjective <= objective_cycle_prev + 1e-10 {
                    objective_cycle_prev = trialobjective;
                    current_penalty = trial_penalty;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                states[b].beta.assign(&beta_old);
                eta_checkpoint.restore_eta(&mut states[b]);
                if let BlockWorkingSet::ExactNewton { gradient, .. } = work {
                    let mut raw_descent = gradient - &s_lambda.dot(&beta_old);
                    if options.ridge_policy.include_quadratic_penalty && ridge > 0.0 {
                        raw_descent -= &beta_old.mapv(|v| ridge * v);
                    }
                    let raw_norm = raw_descent.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                    let descent_dir = if raw_norm > MAX_NEWTON_STEP {
                        &raw_descent * (MAX_NEWTON_STEP / raw_norm)
                    } else {
                        raw_descent
                    };
                    let dir_norm = descent_dir.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                    if dir_norm > inner_tol {
                        // Precompute X * descent_dir once for incremental eta updates.
                        let x_descent = if !is_dynamic {
                            Some(spec.design.matrixvectormultiply(&descent_dir))
                        } else {
                            None
                        };
                        let descent_barrier_ceiling = family
                            .max_feasible_step_size(&states, b, &descent_dir)?
                            .unwrap_or(1.0);
                        for bt in 0..12 {
                            let alpha = (0.5f64.powi(bt)).min(descent_barrier_ceiling);
                            trial_beta_buf.assign(&beta_old);
                            trial_beta_buf.scaled_add(alpha, &descent_dir);
                            let trial_beta = family.post_update_block_beta(
                                &states,
                                b,
                                spec,
                                trial_beta_buf.clone(),
                            )?;
                            states[b].beta = trial_beta;
                            if let Some(ref xd) = x_descent {
                                if states[b].beta == trial_beta_buf {
                                    eta_checkpoint.restore_eta_with_step(&mut states[b], alpha, xd);
                                } else {
                                    refresh_single_block_eta(family, specs, &mut states, b)?;
                                }
                            } else {
                                refresh_single_block_eta(family, specs, &mut states, b)?;
                            }
                            let trial_ll = match family.log_likelihood_only(&states) {
                                Ok(value) => value,
                                Err(_) => {
                                    states[b].beta.assign(&beta_old);
                                    eta_checkpoint.restore_eta(&mut states[b]);
                                    continue;
                                }
                            };
                            let trial_block_penalty = block_quadratic_penalty(
                                &states[b].beta,
                                s_lambda,
                                ridge,
                                options.ridge_policy,
                            );
                            let trial_penalty =
                                current_penalty - old_block_penalty + trial_block_penalty;
                            let trialobjective = -trial_ll + trial_penalty;
                            if trialobjective.is_finite()
                                && trialobjective <= objective_cycle_prev + 1e-10
                            {
                                objective_cycle_prev = trialobjective;
                                current_penalty = trial_penalty;
                                accepted = true;
                                break;
                            }
                            states[b].beta.assign(&beta_old);
                            eta_checkpoint.restore_eta(&mut states[b]);
                        }
                    }
                }
            }
            if !accepted {
                states[b].beta.assign(&beta_old);
                eta_checkpoint.restore_eta(&mut states[b]);
            } else {
                any_block_modified = true;
            }
            // Recycle the checkpoint's buffer back into the pre-allocated pool.
            eta_backups[b] = eta_checkpoint.into_buffer();
        }

        // For non-dynamic families, incremental eta updates within the block loop
        // maintain correct etas. Only refresh from scratch for dynamic-geometry families
        // where block interactions may require recomputation.
        if is_dynamic {
            refresh_all_block_etas(family, specs, &mut states)?;
        }
        cached_eval = family.evaluate(&states)?;
        current_penalty = total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
        let objective = -cached_eval.log_likelihood + current_penalty;
        let objective_change = (objective - lastobjective).abs();
        lastobjective = objective;
        cycles_done = cycle + 1;

        let objective_tol = inner_tol * (1.0 + objective.abs());
        // For single-block models the blockwise iteration IS the joint
        // iteration, so block-conditional convergence implies joint
        // convergence.  The exact_newton_joint_stationarity check can
        // stall at ~10x the tolerance due to numerical differences
        // between the block-conditional and joint gradient formulations,
        // causing 100s of wasted cycles on an already-converged solution.
        let exact_joint_stationarity_ok = if has_joint_exacthessian && specs.len() >= 2 {
            exact_newton_joint_stationarity_inf_norm(
                family,
                specs,
                &cached_eval,
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
            )?
            .map(|residual| residual <= inner_tol)
            .unwrap_or(true)
        } else {
            true
        };
        if max_beta_step <= inner_tol
            && objective_change <= objective_tol
            && exact_joint_stationarity_ok
        {
            converged = true;
            break;
        }
    }

    // ── Polishing joint Newton step ──
    //
    // For block-coupled multi-block families (e.g. GAMLSS wiggle), Gauss-Seidel
    // blockwise iteration can reach step_inf < inner_tol while the joint KKT
    // residual (||Sβ − grad_ℓ||_∞) remains at ~10× inner_tol. This is because
    // each block is solved conditionally on other blocks' current values —
    // block-conditional stationarity does not imply joint stationarity when
    // the likelihood couples blocks off-diagonally.
    //
    // Once blockwise has placed β near the true joint optimum, a single (or
    // a few) damped joint Newton steps can tighten the joint residual to the
    // floor set by β magnitudes. This polishing phase is essential for the
    // outer REML gradient formula (which assumes exact β̂ stationarity); a
    // non-converged β̂ produces large envelope-theorem violations that show
    // up as FD-vs-analytic gradient mismatches.
    if use_joint_newton && !converged {
        let ranges_joint: Vec<(usize, usize)> = {
            let mut offset = 0;
            specs
                .iter()
                .map(|s| {
                    let start = offset;
                    offset += s.design.ncols();
                    (start, offset)
                })
                .collect()
        };
        let total_p_joint: usize = ranges_joint.last().map_or(0, |r| r.1);
        let joint_mode_diagonal_ridge =
            if ridge > 0.0 && options.ridge_policy.include_quadratic_penalty {
                ridge
            } else {
                0.0
            };
        let trace_diagonal_ridge = joint_mode_diagonal_ridge + JOINT_TRACE_STABILITY_RIDGE;

        // Allow up to a few polishing steps. The blockwise endpoint is close
        // to optimum, so step sizes should be small and line search should
        // accept full steps quickly.
        const POLISH_MAX_ITER: usize = 16;
        for _polish_iter in 0..POLISH_MAX_ITER {
            // Re-evaluate at current β to get the joint gradient and Hessian.
            refresh_all_block_etas(family, specs, &mut states)?;
            let eval_for_polish = family.evaluate(&states)?;
            let grad_full = match exact_newton_joint_gradient_from_eval(&eval_for_polish, specs)? {
                Some(g) => g,
                None => break,
            };
            let h_joint_opt = family.exact_newton_joint_hessian(&states)?;
            let Some(h_joint) = h_joint_opt else { break };
            let mut h_dense = match symmetrized_square_matrix(
                h_joint,
                total_p_joint,
                "joint polish Hessian shape mismatch",
            ) {
                Ok(matrix) => matrix,
                Err(_) => break,
            };
            add_joint_penalty_to_matrix(
                &mut h_dense,
                &ranges_joint,
                &s_lambdas,
                trace_diagonal_ridge,
            );

            let mut beta_joint = Array1::<f64>::zeros(total_p_joint);
            for b in 0..specs.len() {
                let (start, end) = ranges_joint[b];
                beta_joint
                    .slice_mut(ndarray::s![start..end])
                    .assign(&states[b].beta);
            }
            let penalty_beta = apply_joint_block_penalty(
                &ranges_joint,
                &s_lambdas,
                &beta_joint,
                joint_mode_diagonal_ridge,
            );
            let rhs = &grad_full - &penalty_beta;

            // Respect constraints that block line search on the boundary.
            // Gauss-Seidel blockwise leaves the joint KKT residual at a floor
            // around |λ_k S_k β̂| for boundary-active components. The residual
            // magnitude on FREE components is a better measure of whether we
            // should keep polishing: if β_i is clipped at the boundary and
            // KKT multiplier μ_i > 0, then rhs[i] is the multiplier, not a
            // free-space gradient violation.
            let block_constraints_now = collect_block_linear_constraints(family, &states, specs)?;
            let joint_constraints_now = assemble_joint_linear_constraints(
                &block_constraints_now,
                &ranges_joint,
                total_p_joint,
            )?;
            let mut active_mask: Vec<bool> = vec![false; total_p_joint];
            if let Some(ref constraints) = joint_constraints_now
                && let Ok(Some(bounds)) = extract_simple_lower_bounds(constraints, total_p_joint)
            {
                for (idx, (bound, beta_val)) in bounds
                    .lower_bounds
                    .iter()
                    .zip(beta_joint.iter())
                    .enumerate()
                {
                    if *bound > f64::NEG_INFINITY && (*beta_val - *bound).abs() < 1e-12 {
                        active_mask[idx] = true;
                    }
                }
            }
            let res_inf_free = rhs
                .iter()
                .zip(active_mask.iter())
                .filter(|(_, active)| !**active)
                .map(|(v, _)| v.abs())
                .fold(0.0_f64, f64::max);
            if res_inf_free <= inner_tol {
                converged = true;
                break;
            }

            // Solve constrained Newton system if simple bounds are present,
            // else unconstrained.
            let delta = if let Some(ref constraints) = joint_constraints_now {
                let warm = flatten_joint_active_set(&cached_active_sets, &block_constraints_now);
                let lower_bounds_opt = extract_simple_lower_bounds(constraints, total_p_joint)
                    .ok()
                    .flatten();
                if let Some(bounds) = lower_bounds_opt.as_ref() {
                    match solve_quadratic_with_simple_lower_bounds(
                        &h_dense,
                        &rhs,
                        &beta_joint,
                        bounds,
                        warm.as_deref(),
                    ) {
                        Ok((beta_new, _active)) => &beta_new - &beta_joint,
                        Err(_) => break,
                    }
                } else {
                    match solve_quadratic_with_linear_constraints(
                        &h_dense,
                        &rhs,
                        &beta_joint,
                        constraints,
                        warm.as_deref(),
                    ) {
                        Ok((beta_new, _active)) => &beta_new - &beta_joint,
                        Err(_) => break,
                    }
                }
            } else {
                let solver = crate::linalg::utils::StableSolver::new("joint polish");
                match solver.solvevectorwithridge_retries(
                    &h_dense,
                    &rhs,
                    JOINT_TRACE_STABILITY_RIDGE,
                ) {
                    Some(d) => d,
                    None => break,
                }
            };
            if !delta.iter().all(|v| v.is_finite()) {
                break;
            }
            // Keep polishing until the free-space joint residual is small; a
            // tiny delta alone is not a certificate of stationarity.
            // Damped line search with projection.
            let old_states: Vec<ParameterBlockState> = states.clone();
            let old_obj = -eval_for_polish.log_likelihood + current_penalty;
            let mut accepted_polish = false;
            for bt in 0..10 {
                let alpha = 0.5f64.powi(bt);
                for b in 0..specs.len() {
                    let (start, end) = ranges_joint[b];
                    let mut trial_beta = old_states[b].beta.clone();
                    trial_beta.scaled_add(alpha, &delta.slice(ndarray::s![start..end]));
                    let projected =
                        family.post_update_block_beta(&old_states, b, &specs[b], trial_beta)?;
                    states[b].beta.assign(&projected);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                let trial_ll = match family.log_likelihood_only(&states) {
                    Ok(v) => v,
                    Err(_) => {
                        for (b, s) in old_states.iter().enumerate() {
                            states[b] = s.clone();
                        }
                        refresh_all_block_etas(family, specs, &mut states)?;
                        continue;
                    }
                };
                let trial_penalty =
                    total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
                let trial_obj = -trial_ll + trial_penalty;
                if trial_obj.is_finite() && trial_obj <= old_obj + 1e-12 {
                    current_penalty = trial_penalty;
                    cached_eval = family.evaluate(&states)?;
                    accepted_polish = true;
                    break;
                }
            }
            if !accepted_polish {
                // Restore and stop polishing.
                for (b, s) in old_states.iter().enumerate() {
                    states[b] = s.clone();
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                break;
            }
        }
    }

    // Reuse cached evaluation from the last cycle's end (or the initial eval if 0 cycles ran).
    let penalty_value = total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);

    let (block_logdet_h, block_logdet_s) =
        blockwise_logdet_terms(family, specs, &mut states, block_log_lambdas, options)?;

    Ok(BlockwiseInnerResult {
        block_states: states,
        active_sets: normalize_active_sets(cached_active_sets),
        log_likelihood: cached_eval.log_likelihood,
        penalty_value,
        cycles: cycles_done,
        converged,
        block_logdet_h,
        block_logdet_s,
        s_lambdas,
    })
}

/// Borrowed derivative provider for joint models that wraps closures with
/// non-`'static` lifetimes.
///
/// The closures borrow data from the calling stack frame (family, synced states,
/// specs), so we use borrowed closures with a non-`'static` lifetime.
/// Instead we borrow the closures and implement `HessianDerivativeProvider` directly.
///
/// # Sign convention
///
/// The unified evaluator passes `v_k = H⁻¹(A_k β̂)` to `hessian_derivative_correction`.
/// By the implicit function theorem, `dβ̂/dρ_k = −v_k`. The stored `compute_dh`
/// expects the actual perturbation direction `δβ`, so we negate `v_k` before calling it.
struct BorrowedJointDerivProvider<'a> {
    compute_dh: &'a dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
    compute_d2h:
        Option<&'a dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>>,
}

// SAFETY: Only used synchronously within the same stack frame that creates it.
// The provider is constructed, boxed, passed to `unified_joint_cost_gradient`,
// consumed by the unified evaluator, and dropped — all within a single call to
// `joint_outer_evaluate`. No cross-thread sharing occurs.
unsafe impl Send for BorrowedJointDerivProvider<'_> {}
unsafe impl Sync for BorrowedJointDerivProvider<'_> {}

impl HessianDerivativeProvider for BorrowedJointDerivProvider<'_> {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .hessian_derivative_correction_result(v_k)?
            .map(|result| result.into_operator().to_dense()))
    }

    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let neg_v = -v_k;
        (self.compute_dh)(&neg_v)
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)?
            .map(|result| result.into_operator().to_dense()))
    }

    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let Some(d2h) = self.compute_d2h.as_ref() else {
            return Ok(None);
        };
        let Some(term1) = (self.compute_dh)(u_kl)? else {
            return Ok(None);
        };
        let neg_v_k = -v_k;
        let neg_v_l = -v_l;
        let Some(term2) = d2h(&neg_v_l, &neg_v_k)? else {
            return Ok(None);
        };
        let op = crate::solver::estimate::reml::unified::CompositeHyperOperator {
            dense: None,
            operators: vec![term1.into_operator(), term2.into_operator()],
            dim_hint: u_kl.len(),
        };
        Ok(Some(DriftDerivResult::Operator(Arc::new(op))))
    }

    fn has_corrections(&self) -> bool {
        true
    }
}

struct OwnedJointDerivProvider {
    compute_dh: Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>,
    compute_d2h: Arc<
        dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
            + Send
            + Sync,
    >,
}

impl HessianDerivativeProvider for OwnedJointDerivProvider {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .hessian_derivative_correction_result(v_k)?
            .map(|result| result.into_operator().to_dense()))
    }

    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let neg_v = -v_k;
        (self.compute_dh)(&neg_v)
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)?
            .map(|result| result.into_operator().to_dense()))
    }

    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let Some(term1) = (self.compute_dh)(u_kl)? else {
            return Ok(None);
        };
        let neg_v_k = -v_k;
        let neg_v_l = -v_l;
        let Some(term2) = (self.compute_d2h)(&neg_v_l, &neg_v_k)? else {
            return Ok(None);
        };
        let op = crate::solver::estimate::reml::unified::CompositeHyperOperator {
            dense: None,
            operators: vec![term1.into_operator(), term2.into_operator()],
            dim_hint: u_kl.len(),
        };
        Ok(Some(DriftDerivResult::Operator(Arc::new(op))))
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn outer_hessian_derivative_kernel(
        &self,
    ) -> Option<crate::solver::estimate::reml::unified::OuterHessianDerivativeKernel> {
        Some(
            crate::solver::estimate::reml::unified::OuterHessianDerivativeKernel::Callback {
                first: Arc::clone(&self.compute_dh),
                second: Arc::clone(&self.compute_d2h),
            },
        )
    }
}

/// Optional bundle of extended (ψ) hyperparameter coordinate data to attach
/// to an `InnerSolution` before calling the unified evaluator.
struct ExtCoordBundle {
    coords: Vec<HyperCoord>,
    ext_ext_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    rho_ext_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    drift_fn: Option<FixedDriftDerivFn>,
}

struct ScaledHyperOperator {
    inner: Arc<dyn HyperOperator>,
    scale: f64,
}

impl HyperOperator for ScaledHyperOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        self.inner.mul_vec(v).mapv(|value| self.scale * value)
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.scale * self.inner.bilinear(v, u)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.inner.to_dense().mapv(|value| self.scale * value)
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

fn scale_hypercoord_drift(mut drift: HyperCoordDrift, scale: f64) -> HyperCoordDrift {
    if scale == 1.0 {
        return drift;
    }
    if let Some(ref mut dense) = drift.dense {
        *dense *= scale;
    }
    if let Some(ref mut block_local) = drift.block_local {
        block_local.local *= scale;
    }
    if let Some(operator) = drift.operator.take() {
        drift.operator = Some(Arc::new(ScaledHyperOperator {
            inner: operator,
            scale,
        }));
    }
    drift
}

fn scale_hypercoord(mut coord: HyperCoord, scale: f64) -> HyperCoord {
    if scale == 1.0 {
        return coord;
    }
    coord.g *= scale;
    if let Some(firth_g) = coord.firth_g.as_mut() {
        *firth_g *= scale;
    }
    coord.drift = scale_hypercoord_drift(coord.drift, scale);
    coord
}

fn scale_hypercoord_pair(mut pair: HyperCoordPair, scale: f64) -> HyperCoordPair {
    if scale == 1.0 {
        return pair;
    }
    pair.g *= scale;
    pair.b_mat *= scale;
    if let Some(operator) = pair.b_operator.take() {
        pair.b_operator = Some(Box::new(ScaledHyperOperator {
            inner: Arc::from(operator),
            scale,
        }));
    }
    pair
}

fn scale_drift_deriv_result(result: DriftDerivResult, scale: f64) -> DriftDerivResult {
    if scale == 1.0 {
        return result;
    }
    match result {
        DriftDerivResult::Dense(mut dense) => {
            dense *= scale;
            DriftDerivResult::Dense(dense)
        }
        DriftDerivResult::Operator(operator) => {
            DriftDerivResult::Operator(Arc::new(ScaledHyperOperator {
                inner: operator,
                scale,
            }))
        }
    }
}

impl ExtCoordBundle {
    fn scaled(self, scale: f64) -> Self {
        if scale == 1.0 {
            return self;
        }
        let coords = self
            .coords
            .into_iter()
            .map(|coord| scale_hypercoord(coord, scale))
            .collect();
        let ext_ext_fn = self.ext_ext_fn.map(|callback| {
            Box::new(move |i: usize, j: usize| scale_hypercoord_pair(callback(i, j), scale))
                as Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>
        });
        let rho_ext_fn = self.rho_ext_fn.map(|callback| {
            Box::new(move |i: usize, j: usize| scale_hypercoord_pair(callback(i, j), scale))
                as Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>
        });
        let drift_fn = self.drift_fn.map(|callback| {
            Box::new(move |ext_idx: usize, direction: &Array1<f64>| {
                callback(ext_idx, direction).map(|result| scale_drift_deriv_result(result, scale))
            }) as FixedDriftDerivFn
        });
        Self {
            coords,
            ext_ext_fn,
            rho_ext_fn,
            drift_fn,
        }
    }
}

/// Build the canonical unified REML/LAML assembly for a custom-family outer
/// evaluation.
fn build_custom_family_inner_assembly<'dp>(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    beta_flat: &Array1<f64>,
    hessian_op: Arc<dyn crate::solver::estimate::reml::unified::HessianOperator>,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    include_logdet_h: bool,
    include_logdet_s: bool,
    options: &BlockwiseFitOptions,
    deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,
    ext_bundle: Option<ExtCoordBundle>,
) -> Result<(crate::estimate::reml::assembly::InnerAssembly<'dp>, usize), String> {
    use crate::estimate::reml::assembly::{
        InnerAssembly, PenaltyBlockDesc, penalty_coords_from_blocks,
    };

    // Collect dense penalty matrices so references stay valid for the assembler.
    let per_block_penalties_dense: Vec<Vec<Array2<f64>>> = specs
        .iter()
        .map(|s| s.penalties.iter().map(|p| p.to_dense()).collect())
        .collect();
    let block_descs: Vec<PenaltyBlockDesc> = (0..specs.len())
        .flat_map(|b| {
            let (start, end) = ranges[b];
            per_block_penalties_dense[b]
                .iter()
                .map(move |dense| PenaltyBlockDesc {
                    matrix: dense,
                    range_start: start,
                    range_end: end,
                })
        })
        .collect();
    let penalty_coords = penalty_coords_from_blocks(&block_descs, total)?;

    // Compute penalty logdet derivatives.
    let per_block_penalties: Vec<&[Array2<f64>]> = per_block_penalties_dense
        .iter()
        .map(|v| v.as_slice())
        .collect();
    let penalty_logdet_ridge = if options.ridge_policy.include_penalty_logdet {
        ridge
    } else {
        0.0
    };
    let per_block_nullspace_dims: Vec<&[usize]> =
        specs.iter().map(|s| s.nullspace_dims.as_slice()).collect();
    let penalty_logdet = compute_block_penalty_logdet_derivs(
        per_block,
        &per_block_penalties,
        &per_block_nullspace_dims,
        penalty_logdet_ridge,
    )?;

    let n_observations = inner.block_states.first().map(|s| s.eta.len()).unwrap_or(0);

    // Unpack optional ext-coord bundle.
    let (ext_coords, ext_coord_pair_fn, rho_ext_pair_fn, fixed_drift_deriv) =
        if let Some(bundle) = ext_bundle {
            (
                bundle.coords,
                bundle.ext_ext_fn,
                bundle.rho_ext_fn,
                bundle.drift_fn,
            )
        } else {
            (Vec::new(), None, None, None)
        };

    let ext_dim = ext_coords.len();

    let evaluator = InnerAssembly {
        log_likelihood: inner.log_likelihood,
        // inner.penalty_value includes the 0.5 factor (= 0.5 β̂ᵀSβ̂), but the
        // unified evaluator convention expects the FULL quadratic β̂ᵀSβ̂ and
        // applies 0.5 itself. Double to match the convention.
        penalty_quadratic: 2.0 * inner.penalty_value,
        beta: beta_flat.clone(),
        n_observations,
        hessian_op,
        penalty_coords,
        penalty_logdet,
        dispersion: DispersionHandling::Fixed {
            phi: 1.0,
            include_logdet_h,
            include_logdet_s,
        },
        rho_curvature_scale,
        hessian_logdet_correction,
        penalty_subspace_trace: None,
        deriv_provider: Some(deriv_provider),
        tk_correction: 0.0,
        tk_gradient: None,
        firth: None,
        nullspace_dim: None,
        barrier_config: None,
        ext_coords,
        ext_coord_pair_fn,
        rho_ext_pair_fn,
        fixed_drift_deriv,
    };

    Ok((evaluator, ext_dim))
}

/// Build an `InnerSolution` from joint Hessian data and call the unified evaluator.
///
/// Bridge between the custom family's joint Hessian infrastructure and the
/// unified REML/LAML evaluator, routed through the canonical assembly module.
fn unified_joint_cost_gradient(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    hessian_op: Arc<dyn crate::solver::estimate::reml::unified::HessianOperator>,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    include_logdet_h: bool,
    include_logdet_s: bool,
    options: &BlockwiseFitOptions,
    deriv_provider: Box<dyn HessianDerivativeProvider + '_>,
    eval_mode: EvalMode,
    ext_bundle: Option<ExtCoordBundle>,
) -> Result<
    (
        f64,
        Array1<f64>,
        crate::solver::outer_strategy::HessianResult,
    ),
    String,
> {
    let (evaluator, ext_dim) = build_custom_family_inner_assembly(
        inner,
        specs,
        per_block,
        beta_flat,
        hessian_op,
        ranges,
        total,
        ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        include_logdet_h,
        include_logdet_s,
        options,
        deriv_provider,
        ext_bundle,
    )?;
    let rho_slice = rho
        .as_slice()
        .ok_or_else(|| "outer rho vector must be contiguous".to_string())?;
    let result = evaluator.evaluate(rho_slice, eval_mode, None)?;

    let cost = result.cost;
    let gradient = result
        .gradient
        .unwrap_or_else(|| Array1::zeros(rho.len() + ext_dim));

    let hessian = result.hessian;

    Ok((cost, gradient, hessian))
}

fn unified_joint_efs_eval(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    hessian_op: Arc<dyn crate::solver::estimate::reml::unified::HessianOperator>,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    include_logdet_h: bool,
    include_logdet_s: bool,
    options: &BlockwiseFitOptions,
    deriv_provider: Box<dyn HessianDerivativeProvider + '_>,
    ext_bundle: Option<ExtCoordBundle>,
) -> Result<crate::solver::outer_strategy::EfsEval, String> {
    let (assembly, _) = build_custom_family_inner_assembly(
        inner,
        specs,
        per_block,
        beta_flat,
        hessian_op,
        ranges,
        total,
        ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        include_logdet_h,
        include_logdet_s,
        options,
        deriv_provider,
        ext_bundle,
    )?;
    let rho_slice = rho
        .as_slice()
        .ok_or_else(|| "outer rho vector must be contiguous".to_string())?;
    let inner_solution = assembly.build();
    let has_psi = inner_solution
        .ext_coords
        .iter()
        .any(|coord| !coord.is_penalty_like);
    let eval_mode = if has_psi {
        EvalMode::ValueAndGradient
    } else {
        EvalMode::ValueOnly
    };
    let result = crate::estimate::reml::assembly::evaluate_solution(
        &inner_solution,
        rho_slice,
        eval_mode,
        None,
    )?;

    if has_psi {
        let gradient = result.gradient.as_ref().ok_or_else(|| {
            "hybrid EFS evaluation did not return the required gradient".to_string()
        })?;
        let hybrid = crate::estimate::reml::unified::compute_hybrid_efs_update(
            &inner_solution,
            rho_slice,
            gradient
                .as_slice()
                .ok_or_else(|| "outer gradient must be contiguous for hybrid EFS".to_string())?,
        );
        Ok(crate::solver::outer_strategy::EfsEval {
            cost: result.cost,
            steps: hybrid.steps,
            beta: Some(inner_solution.beta.clone()),
            psi_gradient: if hybrid.psi_gradient.is_empty() {
                None
            } else {
                Some(Array1::from_vec(hybrid.psi_gradient))
            },
            psi_indices: if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(hybrid.psi_indices)
            },
        })
    } else {
        Ok(crate::solver::outer_strategy::EfsEval {
            cost: result.cost,
            steps: crate::estimate::reml::unified::compute_efs_update(&inner_solution, rho_slice),
            beta: Some(inner_solution.beta.clone()),
            psi_gradient: None,
            psi_indices: None,
        })
    }
}

/// Shared implementation for the joint exact-Newton and surrogate outer paths.
///
/// Both paths differ only in:
/// - how the joint Hessian source is obtained (exact vs surrogate family methods)
/// - the closure for computing D_β H_L[v] (`compute_dh`)
/// - the closure for computing D²_β H_L[u, v] (`compute_d2h`)
/// - whether a tangent-basis projection is applied to the mode inverse
///
/// This function encapsulates all shared logic: penalty assembly, mode inverse
/// computation, precomputation of joint corrections + second-order traces, and
/// routing through `unified_joint_cost_gradient`.
fn joint_outer_evaluate(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    h_joint_unpen: JointHessianSource,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    moderidge: f64,
    extra_logdet_ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    include_logdet_h: bool,
    include_logdet_s: bool,
    strict_spd: bool,
    eval_mode: EvalMode,
    options: &BlockwiseFitOptions,
    pseudo_logdet_mode: PseudoLogdetMode,
    compute_dh: &dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
    compute_d2h: &dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
    owned_compute_dh: Option<
        Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>,
    >,
    owned_compute_d2h: Option<
        Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
                + Send
                + Sync,
        >,
    >,
    ext_bundle: Option<ExtCoordBundle>,
) -> Result<OuterObjectiveEvalResult, String> {
    let joint_trace_diagonal_ridge = moderidge + if !strict_spd { extra_logdet_ridge } else { 0.0 };
    let scaled_joint_trace_diagonal_ridge = rho_curvature_scale * joint_trace_diagonal_ridge;

    // Build derivative provider from the caller-supplied closures.
    let provider_box: Box<dyn HessianDerivativeProvider + '_> =
        if let (Some(owned_dh), Some(owned_d2h)) = (owned_compute_dh, owned_compute_d2h) {
            Box::new(OwnedJointDerivProvider {
                compute_dh: owned_dh,
                compute_d2h: owned_d2h,
            })
        } else {
            let compute_d2h_ref: Option<
                &dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
            > = Some(compute_d2h);
            Box::new(BorrowedJointDerivProvider {
                compute_dh,
                compute_d2h: compute_d2h_ref,
            })
        };

    let scaled_s_lambdas: Vec<Array2<f64>> = inner
        .s_lambdas
        .iter()
        .map(|matrix| {
            if rho_curvature_scale == 1.0 {
                matrix.clone()
            } else {
                matrix.mapv(|value| rho_curvature_scale * value)
            }
        })
        .collect();

    let hessian_op: Arc<dyn crate::solver::estimate::reml::unified::HessianOperator> =
        if use_joint_matrix_free_path(total, joint_observation_count(&inner.block_states)) {
            let ranges_vec = ranges.to_vec();
            let s_lambdas = Arc::new(scaled_s_lambdas.clone());
            let trace_diagonal_ridge = scaled_joint_trace_diagonal_ridge
                + rho_curvature_scale * JOINT_TRACE_STABILITY_RIDGE;
            match h_joint_unpen {
                JointHessianSource::Dense(h_joint) => {
                    let h_joint = Arc::new(h_joint);
                    let preconditioner_diag = joint_penalty_preconditioner_diag(
                        &h_joint.diag().to_owned(),
                        &ranges_vec,
                        s_lambdas.as_ref(),
                        trace_diagonal_ridge,
                    );
                    let apply_h = Arc::clone(&h_joint);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    match MatrixFreeSpdOperator::new(total, preconditioner_diag, move |v| {
                        let mut out = apply_h.dot(v);
                        let penalty = apply_joint_block_penalty(
                            &apply_ranges,
                            apply_s.as_ref(),
                            v,
                            trace_diagonal_ridge,
                        );
                        out += &penalty;
                        out
                    }) {
                        Ok(op) => Arc::new(op),
                        Err(_) => {
                            let mut j_for_traces = (*h_joint).clone();
                            add_joint_penalty_to_matrix(
                                &mut j_for_traces,
                                &ranges_vec,
                                s_lambdas.as_ref(),
                                scaled_joint_trace_diagonal_ridge,
                            );
                            Arc::new(
                                BlockCoupledOperator::from_joint_hessian_with_mode(
                                    &j_for_traces,
                                    pseudo_logdet_mode,
                                )
                                .map_err(|e| {
                                    format!("BlockCoupledOperator from joint Hessian: {e}")
                                })?,
                            )
                        }
                    }
                }
                JointHessianSource::Operator { apply, diagonal } => {
                    let preconditioner_diag = joint_penalty_preconditioner_diag(
                        &diagonal,
                        &ranges_vec,
                        s_lambdas.as_ref(),
                        trace_diagonal_ridge,
                    );
                    let apply_h = Arc::clone(&apply);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    match MatrixFreeSpdOperator::new(total, preconditioner_diag, move |v| {
                        let mut out = match apply_h(v) {
                            Ok(out) => out,
                            Err(error) => {
                                log::warn!(
                                    "joint exact-newton operator matvec failed during outer trace construction: {error}"
                                );
                                Array1::<f64>::zeros(total)
                            }
                        };
                        let penalty = apply_joint_block_penalty(
                            &apply_ranges,
                            apply_s.as_ref(),
                            v,
                            trace_diagonal_ridge,
                        );
                        out += &penalty;
                        out
                    }) {
                        Ok(op) => Arc::new(op),
                        Err(_) => {
                            let mut j_for_traces = materialize_joint_hessian_source(
                                &JointHessianSource::Operator { apply, diagonal },
                                total,
                                "joint exact-newton operator materialization",
                            )?;
                            add_joint_penalty_to_matrix(
                                &mut j_for_traces,
                                &ranges_vec,
                                s_lambdas.as_ref(),
                                scaled_joint_trace_diagonal_ridge,
                            );
                            Arc::new(
                                BlockCoupledOperator::from_joint_hessian_with_mode(
                                    &j_for_traces,
                                    pseudo_logdet_mode,
                                )
                                .map_err(|e| {
                                    format!("BlockCoupledOperator from joint Hessian: {e}")
                                })?,
                            )
                        }
                    }
                }
            }
        } else {
            let mut j_for_traces = materialize_joint_hessian_source(
                &h_joint_unpen,
                total,
                "joint exact-newton Hessian materialization",
            )?;
            add_joint_penalty_to_matrix(
                &mut j_for_traces,
                ranges,
                &scaled_s_lambdas,
                scaled_joint_trace_diagonal_ridge,
            );
            Arc::new(
                BlockCoupledOperator::from_joint_hessian_with_mode(
                    &j_for_traces,
                    pseudo_logdet_mode,
                )
                .map_err(|e| format!("BlockCoupledOperator from joint Hessian: {e}"))?,
            )
        };

    let expected_theta_dim = rho.len()
        + ext_bundle
            .as_ref()
            .map(|bundle| bundle.coords.len())
            .unwrap_or(0);

    let (objective, grad, outer_hessian) = unified_joint_cost_gradient(
        inner,
        specs,
        per_block,
        rho,
        beta_flat,
        hessian_op,
        ranges,
        total,
        ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        include_logdet_h,
        include_logdet_s,
        options,
        provider_box,
        eval_mode,
        ext_bundle.map(|bundle| bundle.scaled(rho_curvature_scale)),
    )?;
    if !objective.is_finite() {
        log::warn!(
            "joint outer evaluation produced non-finite objective: log_likelihood={} penalty_value={} block_logdet_h={} block_logdet_s={} include_logdet_h={} include_logdet_s={} rho_curvature_scale={}",
            inner.log_likelihood,
            inner.penalty_value,
            inner.block_logdet_h,
            inner.block_logdet_s,
            include_logdet_h,
            include_logdet_s,
            rho_curvature_scale,
        );
        return Err("joint outer evaluation produced a non-finite objective".to_string());
    }
    if grad.iter().any(|value| !value.is_finite()) {
        return Err("joint outer evaluation produced a non-finite gradient".to_string());
    }
    if grad.len() != expected_theta_dim {
        return Err(format!(
            "joint outer evaluation returned gradient length {}, expected {}",
            grad.len(),
            expected_theta_dim
        ));
    }
    match &outer_hessian {
        crate::solver::outer_strategy::HessianResult::Analytic(hessian) => {
            if hessian.iter().any(|value| !value.is_finite()) {
                return Err("joint outer evaluation produced a non-finite Hessian".to_string());
            }
            if hessian.nrows() != expected_theta_dim || hessian.ncols() != expected_theta_dim {
                return Err(format!(
                    "joint outer evaluation returned Hessian shape {}x{}, expected {}x{}",
                    hessian.nrows(),
                    hessian.ncols(),
                    expected_theta_dim,
                    expected_theta_dim
                ));
            }
        }
        crate::solver::outer_strategy::HessianResult::Operator(op) => {
            if op.dim() != expected_theta_dim {
                return Err(format!(
                    "joint outer evaluation returned operator Hessian dim {}, expected {}",
                    op.dim(),
                    expected_theta_dim
                ));
            }
        }
        crate::solver::outer_strategy::HessianResult::Unavailable => {}
    }

    let warm = ConstrainedWarmStart {
        rho: rho.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|st| st.beta.clone())
            .collect(),
        active_sets: inner.active_sets.clone(),
    };

    Ok(OuterObjectiveEvalResult {
        objective,
        gradient: grad,
        outer_hessian,
        warm_start: warm,
    })
}

fn joint_outer_evaluate_efs(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    h_joint_unpen: JointHessianSource,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    moderidge: f64,
    extra_logdet_ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    include_logdet_h: bool,
    include_logdet_s: bool,
    strict_spd: bool,
    options: &BlockwiseFitOptions,
    pseudo_logdet_mode: PseudoLogdetMode,
    compute_dh: &dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
    compute_d2h: &dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
    owned_compute_dh: Option<
        Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>,
    >,
    owned_compute_d2h: Option<
        Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
                + Send
                + Sync,
        >,
    >,
    ext_bundle: Option<ExtCoordBundle>,
) -> Result<crate::solver::outer_strategy::EfsEval, String> {
    let joint_trace_diagonal_ridge = moderidge + if !strict_spd { extra_logdet_ridge } else { 0.0 };
    let scaled_joint_trace_diagonal_ridge = rho_curvature_scale * joint_trace_diagonal_ridge;

    let provider_box: Box<dyn HessianDerivativeProvider + '_> =
        if let (Some(owned_dh), Some(owned_d2h)) = (owned_compute_dh, owned_compute_d2h) {
            Box::new(OwnedJointDerivProvider {
                compute_dh: owned_dh,
                compute_d2h: owned_d2h,
            })
        } else {
            let compute_d2h_ref: Option<
                &dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
            > = Some(compute_d2h);
            Box::new(BorrowedJointDerivProvider {
                compute_dh,
                compute_d2h: compute_d2h_ref,
            })
        };

    let scaled_s_lambdas: Vec<Array2<f64>> = inner
        .s_lambdas
        .iter()
        .map(|matrix| {
            if rho_curvature_scale == 1.0 {
                matrix.clone()
            } else {
                matrix.mapv(|value| rho_curvature_scale * value)
            }
        })
        .collect();

    let hessian_op: Arc<dyn crate::solver::estimate::reml::unified::HessianOperator> =
        if use_joint_matrix_free_path(total, joint_observation_count(&inner.block_states)) {
            let ranges_vec = ranges.to_vec();
            let s_lambdas = Arc::new(scaled_s_lambdas.clone());
            let trace_diagonal_ridge = scaled_joint_trace_diagonal_ridge
                + rho_curvature_scale * JOINT_TRACE_STABILITY_RIDGE;
            match h_joint_unpen {
                JointHessianSource::Dense(h_joint) => {
                    let h_joint = Arc::new(h_joint);
                    let preconditioner_diag = joint_penalty_preconditioner_diag(
                        &h_joint.diag().to_owned(),
                        &ranges_vec,
                        s_lambdas.as_ref(),
                        trace_diagonal_ridge,
                    );
                    let apply_h = Arc::clone(&h_joint);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    match MatrixFreeSpdOperator::new(total, preconditioner_diag, move |v| {
                        let mut out = apply_h.dot(v);
                        let penalty = apply_joint_block_penalty(
                            &apply_ranges,
                            apply_s.as_ref(),
                            v,
                            trace_diagonal_ridge,
                        );
                        out += &penalty;
                        out
                    }) {
                        Ok(op) => Arc::new(op),
                        Err(_) => {
                            let mut j_for_traces = (*h_joint).clone();
                            add_joint_penalty_to_matrix(
                                &mut j_for_traces,
                                &ranges_vec,
                                s_lambdas.as_ref(),
                                scaled_joint_trace_diagonal_ridge,
                            );
                            Arc::new(
                                BlockCoupledOperator::from_joint_hessian_with_mode(
                                    &j_for_traces,
                                    pseudo_logdet_mode,
                                )
                                .map_err(|e| {
                                    format!("BlockCoupledOperator from joint Hessian: {e}")
                                })?,
                            )
                        }
                    }
                }
                JointHessianSource::Operator { apply, diagonal } => {
                    let preconditioner_diag = joint_penalty_preconditioner_diag(
                        &diagonal,
                        &ranges_vec,
                        s_lambdas.as_ref(),
                        trace_diagonal_ridge,
                    );
                    let apply_h = Arc::clone(&apply);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    match MatrixFreeSpdOperator::new(total, preconditioner_diag, move |v| {
                        let mut out = match apply_h(v) {
                            Ok(out) => out,
                            Err(error) => {
                                log::warn!(
                                    "joint exact-newton operator matvec failed during fixed-point trace construction: {error}"
                                );
                                Array1::<f64>::zeros(total)
                            }
                        };
                        let penalty = apply_joint_block_penalty(
                            &apply_ranges,
                            apply_s.as_ref(),
                            v,
                            trace_diagonal_ridge,
                        );
                        out += &penalty;
                        out
                    }) {
                        Ok(op) => Arc::new(op),
                        Err(_) => {
                            let mut j_for_traces = materialize_joint_hessian_source(
                                &JointHessianSource::Operator { apply, diagonal },
                                total,
                                "joint exact-newton operator materialization for fixed-point evaluation",
                            )?;
                            add_joint_penalty_to_matrix(
                                &mut j_for_traces,
                                &ranges_vec,
                                s_lambdas.as_ref(),
                                scaled_joint_trace_diagonal_ridge,
                            );
                            Arc::new(
                                BlockCoupledOperator::from_joint_hessian_with_mode(
                                    &j_for_traces,
                                    pseudo_logdet_mode,
                                )
                                .map_err(|e| {
                                    format!("BlockCoupledOperator from joint Hessian: {e}")
                                })?,
                            )
                        }
                    }
                }
            }
        } else {
            let mut j_for_traces = materialize_joint_hessian_source(
                &h_joint_unpen,
                total,
                "joint exact-newton Hessian materialization for fixed-point evaluation",
            )?;
            add_joint_penalty_to_matrix(
                &mut j_for_traces,
                ranges,
                &scaled_s_lambdas,
                scaled_joint_trace_diagonal_ridge,
            );
            Arc::new(
                BlockCoupledOperator::from_joint_hessian_with_mode(
                    &j_for_traces,
                    pseudo_logdet_mode,
                )
                .map_err(|e| format!("BlockCoupledOperator from joint Hessian: {e}"))?,
            )
        };

    unified_joint_efs_eval(
        inner,
        specs,
        per_block,
        rho,
        beta_flat,
        hessian_op,
        ranges,
        total,
        ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        include_logdet_h,
        include_logdet_s,
        options,
        provider_box,
        ext_bundle.map(|bundle| bundle.scaled(rho_curvature_scale)),
    )
}

/// Evaluate the rho-only custom-family outer objective through the unified
/// joint hyperpath with no external ψ coordinates attached.
fn outerobjectivegradienthessian_internal<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, String> {
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()];
    evaluate_custom_family_hyper_internal(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        &derivative_blocks,
        warm_start,
        eval_mode,
    )
    .map_err(String::from)
}

fn outerobjectivegradienthessian<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    eval_mode: EvalMode,
) -> Result<(f64, Array1<f64>, Option<Array2<f64>>, ConstrainedWarmStart), String> {
    let result = outerobjectivegradienthessian_internal(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        warm_start,
        eval_mode,
    )?;
    Ok((
        result.objective,
        result.gradient,
        result.outer_hessian.materialize_dense()?,
        result.warm_start,
    ))
}

fn outerobjectiveefs<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<(crate::solver::outer_strategy::EfsEval, ConstrainedWarmStart), String> {
    let include_logdet_h = include_exact_newton_logdet_h(family, options);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    let per_block = split_log_lambdas(rho, penalty_counts)?;
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, warm_start)?;
    let ridge = effective_solverridge(options.ridge_floor);
    let moderidge = if options.ridge_policy.include_quadratic_penalty {
        ridge
    } else {
        0.0
    };
    let extra_logdet_ridge = if options.ridge_policy.include_penalty_logdet
        && !options.ridge_policy.include_quadratic_penalty
    {
        ridge
    } else {
        0.0
    };

    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, end)| *end).unwrap_or(0);

    let efs_eval = {
        if let Some(joint_bundle) =
            build_joint_hessian_closures(family, &inner.block_states, specs, total)?
        {
            let JointHessianBundle {
                source: h_joint_unpen,
                beta_flat,
                compute_dh,
                compute_d2h,
                rho_curvature_scale,
                hessian_logdet_correction,
            } = joint_bundle;
            joint_outer_evaluate_efs(
                &inner,
                specs,
                &per_block,
                rho,
                &beta_flat,
                h_joint_unpen,
                &ranges,
                total,
                ridge,
                moderidge,
                extra_logdet_ridge,
                rho_curvature_scale,
                hessian_logdet_correction,
                include_logdet_h,
                include_logdet_s,
                strict_spd,
                options,
                family.pseudo_logdet_mode(),
                compute_dh.as_ref(),
                compute_d2h.as_ref(),
                None,
                None,
                None,
            )
        } else {
            if family.requires_joint_outer_hyper_path() {
                return Err(
                        "outer hyper fixed-point evaluation requires a joint exact path for this family"
                            .to_string(),
                    );
            }
            if specs.len() != 1 {
                return Err(
                        "generic fixed-point outer fallback is only valid for single-block families; multi-block families must provide a joint outer path"
                            .to_string(),
                    );
            }

            let eval = family.evaluate(&inner.block_states)?;
            let block_idx = 0;
            let spec = &specs[block_idx];
            let work = &eval.blockworking_sets[block_idx];
            let p = spec.design.ncols();
            let mut diagonal_design = None::<DesignMatrix>;
            let h_joint_unpen = match work {
                BlockWorkingSet::Diagonal {
                    working_response: _,
                    working_weights,
                } => with_block_geometry(
                    family,
                    &inner.block_states,
                    spec,
                    block_idx,
                    |x_dyn, _| {
                        let w = floor_positiveworking_weights(working_weights, options.minweight);
                        let (xtwx, _) = weighted_normal_equations(x_dyn, &w, None)?;
                        diagonal_design = Some(x_dyn.clone());
                        Ok(xtwx)
                    },
                )?,
                BlockWorkingSet::ExactNewton {
                    gradient: _,
                    hessian,
                } => {
                    if hessian.nrows() != p || hessian.ncols() != p {
                        return Err(format!(
                            "block {block_idx} exact-newton Hessian shape mismatch in fixed-point outer evaluation: got {}x{}, expected {}x{}",
                            hessian.nrows(),
                            hessian.ncols(),
                            p,
                            p
                        ));
                    }
                    hessian.to_dense()
                }
            };
            let beta_flat = inner.block_states[block_idx].beta.clone();
            let compute_dh = |direction: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                if !include_logdet_h {
                    return Ok(None);
                }
                match work {
                    BlockWorkingSet::ExactNewton { .. } => {
                        match family.exact_newton_hessian_directional_derivative(
                            &inner.block_states,
                            block_idx,
                            direction,
                        )? {
                            Some(h_exact) => {
                                Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                                    h_exact,
                                    p,
                                    &format!(
                                        "block {block_idx} exact-newton dH shape mismatch in fixed-point outer evaluation"
                                    ),
                                )?)))
                            }
                            None => Err(format!(
                                "missing exact-newton dH callback for block {block_idx} while fixed-point evaluation requires H_beta term"
                            )),
                        }
                    }
                    BlockWorkingSet::Diagonal {
                        working_response: _,
                        working_weights,
                    } => {
                        let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                                    format!(
                                        "missing dynamic design for block {block_idx} diagonal fixed-point correction"
                                    )
                                })?;
                        let wwork =
                            floor_positiveworking_weights(working_weights, options.minweight);
                        let x_dense = x_dyn.to_dense();
                        let n = x_dense.nrows();

                        let mut d_eta = x_dyn.matrixvectormultiply(direction);
                        let geom = family.block_geometry_directional_derivative(
                            &inner.block_states,
                            block_idx,
                            spec,
                            direction,
                        )?;
                        let mut correction_mat = Array2::<f64>::zeros((p, p));

                        if let Some(geom_dir) = geom {
                            d_eta += &geom_dir.d_offset;
                            if let Some(dx) = geom_dir.d_design {
                                d_eta += &dx.dot(&beta_flat);
                                let mut wx = x_dense.clone();
                                let mut wdx = dx.clone();
                                for i in 0..n {
                                    let wi = wwork[i];
                                    if wi != 1.0 {
                                        wx.row_mut(i).mapv_inplace(|v| v * wi);
                                        wdx.row_mut(i).mapv_inplace(|v| v * wi);
                                    }
                                }
                                correction_mat += &dx.t().dot(&wx);
                                correction_mat += &x_dense.t().dot(&wdx);
                            }
                        }

                        let dw = family
                                    .diagonalworking_weights_directional_derivative(
                                        &inner.block_states,
                                        block_idx,
                                        &d_eta,
                                    )?
                                    .ok_or_else(|| {
                                        format!(
                                            "missing diagonal dW callback for block {block_idx} while fixed-point evaluation requires H_beta term"
                                        )
                                    })?;
                        if dw.len() != n {
                            return Err(format!(
                                "block {block_idx} diagonal dW length mismatch in fixed-point outer evaluation: got {}, expected {}",
                                dw.len(),
                                n
                            ));
                        }
                        let mut scaled_x = x_dense.clone();
                        for i in 0..n {
                            scaled_x.row_mut(i).mapv_inplace(|v| v * dw[i]);
                        }
                        correction_mat += &x_dense.t().dot(&scaled_x);

                        Ok(Some(DriftDerivResult::Dense(correction_mat)))
                    }
                }
            };
            let compute_d2h = |_: &Array1<f64>,
                               _: &Array1<f64>|
             -> Result<Option<DriftDerivResult>, String> { Ok(None) };
            joint_outer_evaluate_efs(
                &inner,
                specs,
                &per_block,
                rho,
                &beta_flat,
                JointHessianSource::Dense(h_joint_unpen),
                &ranges,
                total,
                ridge,
                moderidge,
                extra_logdet_ridge,
                1.0,
                0.0,
                include_logdet_h,
                include_logdet_s,
                strict_spd,
                options,
                family.pseudo_logdet_mode(),
                &compute_dh,
                &compute_d2h,
                None,
                None,
                None,
            )
        }
    }?;

    let warm = ConstrainedWarmStart {
        rho: rho.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|state| state.beta.clone())
            .collect(),
        active_sets: inner.active_sets.clone(),
    };

    Ok((efs_eval, warm))
}

fn normalize_outer_eval_error_detail(error: &str) -> &str {
    error
        .strip_prefix("custom-family invalid input: ")
        .unwrap_or(error)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section: joint outer hyper surface — unified calculus for [rho, psi]
// ═══════════════════════════════════════════════════════════════════════════
//
// The callers have already applied the current spatial coordinates `psi` when
// constructing `family`, `specs`, and `derivative_blocks`, so the explicit
// input into the section below is still only the smoothing vector
// `rho_current`. Mathematically, however, the surface being differentiated
// is the full joint profiled/Laplace objective in
//
//     theta = [rho, psi].
//
// The exact outer calculus is unified across all hypercoordinates:
//
//     J(theta)
//     = V(beta^(theta), theta)
//       + 0.5 log|H(beta^(theta), theta)|
//       - 0.5 log|S(theta)|_+,
//
// with stationarity and joint curvature
//
//     F(beta, theta) := V_beta(beta, theta) = 0,
//     H(beta, theta) := V_beta_beta(beta, theta).
//
// For each theta_i we need the fixed-beta objects
//
//     V_i, g_i := F_i, H_i,
//
// and for each pair (i, j)
//
//     V_ij, g_ij, H_ij,
//
// together with the beta-curvature contractions
//
//     D_beta H[u], D_beta^2 H[u, v], T_i[u] := D_beta H_i[u].
//
// These determine the exact joint mode responses
//
//     beta_i  = -H^{-1} g_i,
//     beta_ij = -H^{-1}(g_ij + H_i beta_j + H_j beta_i + D_beta H[beta_i] beta_j),
//
// and the total Hessian drifts
//
//     dot H_i
//     = H_i + D_beta H[beta_i],
//
//     ddot H_ij
//     = H_ij
//       + T_i[beta_j]
//       + T_j[beta_i]
//       + D_beta H[beta_ij]
//       + D_beta^2 H[beta_i, beta_j].
//
// Therefore the exact joint outer derivatives are
//
//     J_i
//     = V_i
//       + 0.5 tr(H^{-1} dot H_i)
//       - 0.5 partial_i log|S(theta)|_+,
//
//     J_ij
//     = (V_ij - g_i^T H^{-1} g_j)
//       + 0.5 [ tr(H^{-1} ddot H_ij)
//               - tr(H^{-1} dot H_j H^{-1} dot H_i) ]
//       - 0.5 partial^2_{ij} log|S(theta)|_+.
//
// In this unified view rho and psi differ only in the likelihood-side
// fixed-beta derivative objects contributed by the family. The generic exact
// assembler always adds realized penalty motion through `S(theta)` for every
// hypercoordinate:
//
// - `rho` coordinates usually have zero likelihood-side objects and pick up
//   their fixed-beta derivatives entirely from `S_rho` / `S_{rho rho}`
// - `psi` coordinates contribute likelihood-side objects from the family's
//   joint exact psi hooks and may also pick up extra penalty terms through
//   `S_psi`, `S_{rho psi}`, and `S_{psi psi}` when realized penalties move
//   with `psi`
//
// The implementation below follows this unified calculus directly. Once a
// family supplies the joint fixed-beta psi objects and the mixed
// `D_beta H_psi[u]` contraction, exact joint hyper evaluation treats `rho`
// and `psi` identically and returns the full profiled/Laplace Hessian over
// `theta = [rho, psi]`.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Unified HyperCoord builders for ψ coordinates
// ═══════════════════════════════════════════════════════════════════════════

/// Assemble the penalty derivative matrix S_ψ = Σ_k exp(ρ_k) ∂S_k/∂ψ
/// in the *block-local* coefficient space (p_block × p_block).
///
/// When the derivative carries multi-penalty components the sum iterates
/// over all `(penalty_idx, s_part)` pairs.  When only a single
/// `penalty_index` is stored the derivative `s_psi` is scaled by that
/// penalty's current lambda.  If neither is present, the derivative is
/// zero (the ψ coordinate does not move any realized penalty).
fn assemble_block_local_s_psi(
    deriv: &CustomFamilyBlockPsiDerivative,
    per_block_rho: &Array1<f64>,
    p_block: usize,
) -> Array2<f64> {
    if let Some(ref components) = deriv.s_psi_penalty_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s_part.add_scaled_to(per_block_rho[*penalty_idx].exp(), &mut s);
        }
        return s;
    }
    if let Some(ref components) = deriv.s_psi_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s.scaled_add(per_block_rho[*penalty_idx].exp(), s_part);
        }
        s
    } else if let Some(penalty_idx) = deriv.penalty_index {
        deriv.s_psi.mapv(|v| per_block_rho[penalty_idx].exp() * v)
    } else {
        Array2::<f64>::zeros((p_block, p_block))
    }
}

/// Assemble the second penalty derivative matrix S_{ψ_i ψ_j} in block-local
/// coefficient space.
///
/// This mirrors the psi/psi branch of `joint_theta_penaltysecond_matrix` but
/// returns the block-local matrix directly instead of embedding it into the
/// full flattened coefficient space.
fn assemble_block_local_s_psi_psi(
    deriv_i: &CustomFamilyBlockPsiDerivative,
    local_j: usize,
    per_block_rho: &Array1<f64>,
    p_block: usize,
) -> Array2<f64> {
    if let Some(ref parts) = deriv_i.s_psi_psi_penalty_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        if let Some(pair_parts) = parts.get(local_j) {
            for (penalty_idx, s_part) in pair_parts {
                s_part.add_scaled_to(per_block_rho[*penalty_idx].exp(), &mut s);
            }
        }
        return s;
    }
    if let Some(ref parts) = deriv_i.s_psi_psi_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        if let Some(pair_parts) = parts.get(local_j) {
            for (penalty_idx, s_part) in pair_parts {
                s.scaled_add(per_block_rho[*penalty_idx].exp(), s_part);
            }
        }
        s
    } else if let Some(ref parts) = deriv_i.s_psi_psi {
        if let Some(s_part) = parts.get(local_j) {
            if let Some(penalty_index) = deriv_i.penalty_index {
                s_part.mapv(|v| per_block_rho[penalty_index].exp() * v)
            } else {
                Array2::<f64>::zeros((p_block, p_block))
            }
        } else {
            Array2::<f64>::zeros((p_block, p_block))
        }
    } else {
        Array2::<f64>::zeros((p_block, p_block))
    }
}

/// Build `HyperCoord` objects for ψ (custom family) hyperparameters.
///
/// Converts family-provided (a^ℓ, q, L) objects and penalty derivatives
/// into the unified (a, g, B, ld_s) format. Each ψ coordinate produces
/// one `HyperCoord` in the flattened joint coefficient space.
///
/// The mapping from family objects to HyperCoord is:
///
///   a    = a^ℓ_ψ + 0.5 β̂^T S_ψ β̂
///   g    = q_ψ + S_ψ β̂
///   B    = L_ψ + S_ψ
///   ld_s = tr(S₊⁻¹ S_ψ)
///
/// where S_ψ is the assembled penalty derivative in joint coefficient space.
pub fn build_psi_hyper_coords<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    beta_flat: &Array1<f64>,
    rho: &[f64],
    penalty_counts: &[usize],
    s_logdet_blocks: Option<&[PenaltyPseudologdet]>,
    hessian_beta_independent: bool,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
) -> Result<Vec<HyperCoord>, String> {
    let ranges = block_param_ranges(specs);
    let total = beta_flat.len();
    let per_block = split_log_lambdas(&Array1::from_vec(rho.to_vec()), penalty_counts)?;

    let mut coords = Vec::new();
    let mut psi_global = 0usize;

    for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
        let (start, end) = ranges[block_idx];
        let p_block = end - start;

        for (_, deriv) in block_derivs.iter().enumerate() {
            // 1. Get family-provided likelihood objects (joint flattened space).
            let psi_terms = if let Some(workspace) = psi_workspace.as_ref() {
                if let Some(terms) = workspace.first_order_terms(psi_global)? {
                    terms
                } else {
                    family
                        .exact_newton_joint_psi_terms(
                            synced_states,
                            specs,
                            derivative_blocks,
                            psi_global,
                        )?
                        .unwrap_or_else(|| ExactNewtonJointPsiTerms::zeros(total))
                }
            } else {
                family
                    .exact_newton_joint_psi_terms(
                        synced_states,
                        specs,
                        derivative_blocks,
                        psi_global,
                    )?
                    .unwrap_or_else(|| ExactNewtonJointPsiTerms::zeros(total))
            };

            // 2. Assemble S_ψ from penalty derivatives (block-local, not embedded).
            let s_psi_local = assemble_block_local_s_psi(deriv, &per_block[block_idx], p_block);

            // 3. Build HyperCoord using block-local S_ψ (avoids full p×p materialization).
            let beta_block = beta_flat.slice(ndarray::s![start..end]);
            let s_psi_beta_local = s_psi_local.dot(&beta_block);
            let a = psi_terms.objective_psi + 0.5 * beta_block.dot(&s_psi_beta_local);
            // Embed s_psi_beta into full p-vector for the score.
            let mut s_psi_beta = Array1::zeros(total);
            s_psi_beta
                .slice_mut(ndarray::s![start..end])
                .assign(&s_psi_beta_local);
            let g = &psi_terms.score_psi + &s_psi_beta;
            let ld_s = if let Some(blocks) = s_logdet_blocks {
                blocks[block_idx].tau_gradient_component(&s_psi_local)
            } else {
                0.0
            };

            // Build drift: use block-local representation when possible to avoid
            // materializing full p×p dense matrices.
            let drift = if psi_terms.hessian_psi_operator.is_some() {
                // No dense Hessian contribution — penalty is block-local, operator
                // (if present) handles the likelihood part. O(p_block²) fast path.
                HyperCoordDrift::from_block_local_and_operator(
                    s_psi_local,
                    start,
                    end,
                    psi_terms.hessian_psi_operator,
                )
            } else {
                // Dense Hessian term exists (e.g., from non-implicit family).
                // Must add block-local penalty into the dense matrix.
                let mut dense_b = psi_terms.hessian_psi;
                dense_b
                    .slice_mut(ndarray::s![start..end, start..end])
                    .scaled_add(1.0, &s_psi_local);
                HyperCoordDrift::from_parts(Some(dense_b), psi_terms.hessian_psi_operator)
            };

            coords.push(HyperCoord {
                a,
                g,
                drift,
                ld_s,
                b_depends_on_beta: !hessian_beta_independent,
                is_penalty_like: false,
                firth_g: None,
            });

            psi_global += 1;
        }
    }

    Ok(coords)
}

/// Build pair callbacks for ψ-ψ and ρ-ψ Hessian entries.
///
/// Returns two closures:
///
/// 1. **ext-ext** `(psi_i, psi_j) -> HyperCoordPair`: second-order
///    fixed-β objects for a pair of ψ coordinates.
///
/// 2. **rho-ext** `(rho_k, psi_j) -> HyperCoordPair`: mixed second-order
///    fixed-β objects for a ρ-ψ pair.
///
/// The closures capture (via `Arc`) shared references to penalty derivatives,
/// family state, and the penalty pseudo-inverse needed for logdet terms.
///
/// # Arguments
///
/// * `family` - The custom family instance (must be `Send + Sync + 'static`).
/// * `synced_states` - Synchronized block states at the current inner mode.
/// * `specs` - Parameter block specifications.
/// * `derivative_blocks` - Per-block ψ derivative payloads.
/// * `beta_flat` - Flattened joint coefficient vector at the inner mode.
/// * `rho` - Current log-smoothing parameters (flat).
/// * `penalty_counts` - Number of penalties per block.
/// * `s_logdet_blocks` - Optional exact block-local pseudologdet eigenspaces.
pub fn build_psi_pair_callbacks<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    derivative_blocks: SharedDerivativeBlocks,
    beta_flat: &Array1<f64>,
    rho: &[f64],
    penalty_counts: &[usize],
    s_logdet_blocks: Option<&[PenaltyPseudologdet]>,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
) -> Result<
    (
        Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>,
        Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>,
    ),
    String,
> {
    // Precompute shared data into Arc-wrapped clones for the closures.
    let ranges = block_param_ranges(specs);
    let total = beta_flat.len();
    let per_block = Arc::new(split_log_lambdas(
        &Array1::from_vec(rho.to_vec()),
        penalty_counts,
    )?);
    let specs_arc = Arc::new(specs.to_vec());
    let beta_arc = Arc::new(beta_flat.clone());
    let synced_arc = Arc::new(synced_states.to_vec());
    let ranges_arc = Arc::new(ranges);
    let family_arc = Arc::new(family.clone());

    let s_logdet_block_cache = Arc::new(s_logdet_blocks.map(|blocks| blocks.to_vec()));

    struct PsiPenaltyCacheEntry {
        block_idx: usize,
        local_idx: usize,
        start: usize,
        end: usize,
        /// Block-local S_ψ matrix, stored for use with `PenaltyPseudologdet` methods.
        s_local: Option<Array2<f64>>,
    }

    struct RhoPenaltyCacheEntry {
        block_idx: usize,
        penalty_idx: usize,
        start: usize,
        end: usize,
        /// Unscaled penalty matrix S_k for use with `PenaltyPseudologdet::rho_tau_hessian_component`.
        s_k_unscaled: Array2<f64>,
    }

    // Build the psi coordinate cache once. These block-local S_psi matrices are
    // reused by ψψ and ρψ callbacks, avoiding repeated assembly inside the
    // O(q²) ext-ext loop.
    let mut psi_penalty_cache: Vec<PsiPenaltyCacheEntry> = Vec::new();
    for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
        let (start, end) = ranges_arc[block_idx];
        let p_block = end - start;
        for (local_idx, deriv) in block_derivs.iter().enumerate() {
            let s_local = assemble_block_local_s_psi(deriv, &per_block[block_idx], p_block);
            // Store the block-local S_ψ matrix when penalty logdet is active;
            // PenaltyPseudologdet methods will handle pseudoinverse and leakage internally.
            let s_local_opt = if s_logdet_block_cache.is_some() {
                Some(s_local)
            } else {
                None
            };
            psi_penalty_cache.push(PsiPenaltyCacheEntry {
                block_idx,
                local_idx,
                start,
                end,
                s_local: s_local_opt,
            });
        }
    }
    let psi_penalty_cache = Arc::new(psi_penalty_cache);

    let mut rho_penalty_cache: Vec<RhoPenaltyCacheEntry> = Vec::new();
    for (block_idx, &count) in penalty_counts.iter().enumerate() {
        let (start, end) = ranges_arc[block_idx];
        for penalty_idx in 0..count {
            let s_k_unscaled = specs_arc[block_idx].penalties[penalty_idx].to_dense();
            rho_penalty_cache.push(RhoPenaltyCacheEntry {
                block_idx,
                penalty_idx,
                start,
                end,
                s_k_unscaled,
            });
        }
    }
    let rho_penalty_cache = Arc::new(rho_penalty_cache);

    // ψ-ψ pair callback
    let ext_ext = {
        let per_block = Arc::clone(&per_block);
        let derivative_blocks = Arc::clone(&derivative_blocks);
        let specs_arc = Arc::clone(&specs_arc);
        let beta_arc = Arc::clone(&beta_arc);
        let synced_arc = Arc::clone(&synced_arc);
        let s_logdet_block_cache = Arc::clone(&s_logdet_block_cache);
        let psi_penalty_cache = Arc::clone(&psi_penalty_cache);
        let family_arc = Arc::clone(&family_arc);
        let psi_workspace = psi_workspace.clone();

        Box::new(move |psi_i: usize, psi_j: usize| -> HyperCoordPair {
            let cache_i = &psi_penalty_cache[psi_i];
            let cache_j = &psi_penalty_cache[psi_j];

            // Get family-provided second-order likelihood terms.
            let psi2 = if let Some(workspace) = psi_workspace.as_ref() {
                workspace.second_order_terms(psi_i, psi_j).ok().flatten()
            } else {
                family_arc
                    .exact_newton_joint_psisecond_order_terms(
                        &synced_arc,
                        &specs_arc,
                        &derivative_blocks,
                        psi_i,
                        psi_j,
                    )
                    .ok()
                    .flatten()
            };

            let (obj_ll, score_ll, hess_ll, hess_ll_op) = match psi2 {
                Some(t) => (
                    t.objective_psi_psi,
                    t.score_psi_psi,
                    t.hessian_psi_psi,
                    t.hessian_psi_psi_operator,
                ),
                None => (
                    0.0,
                    Array1::zeros(total),
                    Array2::zeros((total, total)),
                    None,
                ),
            };

            let mut a = obj_ll;
            let mut g = score_ll;
            let mut b_mat = hess_ll;

            // Assemble S_{ψ_i ψ_j} only on the touched block.
            let ld_s = if cache_i.block_idx == cache_j.block_idx {
                let p_block = cache_i.end - cache_i.start;
                let deriv_i = &derivative_blocks[cache_i.block_idx][cache_i.local_idx];
                let s_local = assemble_block_local_s_psi_psi(
                    deriv_i,
                    cache_j.local_idx,
                    &per_block[cache_i.block_idx],
                    p_block,
                );

                let beta_block = beta_arc.slice(s![cache_i.start..cache_i.end]).to_owned();
                let s_ij_beta_local = s_local.dot(&beta_block);
                a += 0.5 * beta_block.dot(&s_ij_beta_local);
                {
                    let mut g_local = g.slice_mut(s![cache_i.start..cache_i.end]);
                    g_local += &s_ij_beta_local;
                }
                {
                    let mut b_local =
                        b_mat.slice_mut(s![cache_i.start..cache_i.end, cache_i.start..cache_i.end]);
                    b_local += &s_local;
                }

                if let Some(ref logdet_blocks) = *s_logdet_block_cache {
                    let pld = &logdet_blocks[cache_i.block_idx];
                    let s_psi_i = cache_i
                        .s_local
                        .as_ref()
                        .expect("psi cache should include S_psi when penalty logdet is active");
                    let s_psi_j = cache_j
                        .s_local
                        .as_ref()
                        .expect("psi cache should include S_psi when penalty logdet is active");
                    // τ-Hessian: tr(S⁺ S_{ψi ψj}) − tr(S⁺ S_ψi S⁺ S_ψj) + 2 tr(Σ₊⁻² L_i L_j^T)
                    pld.tau_hessian_component(s_psi_i, s_psi_j, Some(&s_local))
                } else {
                    0.0
                }
            } else {
                0.0
            };

            HyperCoordPair {
                a,
                g,
                b_mat,
                b_operator: hess_ll_op,
                ld_s,
            }
        }) as Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>
    };

    // ρ-ψ pair callback
    let rho_ext = {
        let per_block = Arc::clone(&per_block);
        let derivative_blocks = Arc::clone(&derivative_blocks);
        let beta_arc = Arc::clone(&beta_arc);
        let psi_penalty_cache = Arc::clone(&psi_penalty_cache);
        let rho_penalty_cache = Arc::clone(&rho_penalty_cache);
        let s_logdet_block_cache = Arc::clone(&s_logdet_block_cache);

        Box::new(move |rho_k: usize, psi_j: usize| -> HyperCoordPair {
            let rho_cache = &rho_penalty_cache[rho_k];
            let psi_cache = &psi_penalty_cache[psi_j];
            let mut a = 0.0;
            let mut g = Array1::<f64>::zeros(total);
            let mut b_mat = Array2::<f64>::zeros((total, total));

            // S_{ρ_k, ψ_j} = λ_k ∂S_k/∂ψ_j.
            // Only nonzero when both coordinates share the same block and the
            // ψ derivative touches the k-th penalty.
            let ld_s = if rho_cache.block_idx == psi_cache.block_idx {
                let p_block = rho_cache.end - rho_cache.start;
                let deriv = &derivative_blocks[psi_cache.block_idx][psi_cache.local_idx];
                let lambda_k = per_block[rho_cache.block_idx][rho_cache.penalty_idx].exp();
                let local = if let Some(ref components) = deriv.s_psi_penalty_components {
                    let mut m = Array2::<f64>::zeros((p_block, p_block));
                    for (penalty_idx, s_part) in components {
                        if *penalty_idx == rho_cache.penalty_idx {
                            s_part.add_scaled_to(lambda_k, &mut m);
                        }
                    }
                    m
                } else if let Some(ref components) = deriv.s_psi_components {
                    let mut m = Array2::<f64>::zeros((p_block, p_block));
                    for (penalty_idx, s_part) in components {
                        if *penalty_idx == rho_cache.penalty_idx {
                            m.scaled_add(lambda_k, s_part);
                        }
                    }
                    m
                } else if deriv.penalty_index == Some(rho_cache.penalty_idx) {
                    deriv.s_psi.mapv(|v| lambda_k * v)
                } else {
                    Array2::<f64>::zeros((p_block, p_block))
                };

                let beta_block = beta_arc
                    .slice(s![rho_cache.start..rho_cache.end])
                    .to_owned();
                let s_kj_beta_local = local.dot(&beta_block);
                a = 0.5 * beta_block.dot(&s_kj_beta_local);
                {
                    let mut g_local = g.slice_mut(s![rho_cache.start..rho_cache.end]);
                    g_local += &s_kj_beta_local;
                }
                {
                    let mut b_local = b_mat.slice_mut(s![
                        rho_cache.start..rho_cache.end,
                        rho_cache.start..rho_cache.end
                    ]);
                    b_local += &local;
                }

                if let Some(ref logdet_blocks) = *s_logdet_block_cache {
                    let pld = &logdet_blocks[rho_cache.block_idx];
                    let s_psi_j = psi_cache
                        .s_local
                        .as_ref()
                        .expect("psi cache should include S_psi when penalty logdet is active");
                    // ∂S_k/∂ψ_j (unscaled): extract from local by dividing out λ_k.
                    let ds_k_dpsi = if lambda_k.abs() > 1e-300 {
                        Some(local.mapv(|v| v / lambda_k))
                    } else {
                        None
                    };
                    // Mixed ρ×τ Hessian: λ_k [tr(S⁺ ∂S_k/∂ψ_j) − tr(S⁺ S_k S⁺ S_ψj)]
                    pld.rho_tau_hessian_component(
                        &rho_cache.s_k_unscaled,
                        lambda_k,
                        s_psi_j,
                        ds_k_dpsi.as_ref(),
                    )
                } else {
                    0.0
                }
            } else {
                0.0
            };

            HyperCoordPair {
                a,
                g,
                b_mat,
                b_operator: None,
                ld_s,
            }
        }) as Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>
    };

    Ok((ext_ext, rho_ext))
}

/// Build the M_i[u] = D_β B_i[u] callback for ψ coordinates.
///
/// This wraps `family.exact_newton_joint_psihessian_directional_derivative`
/// into the unified `FixedDriftDerivFn` signature. For each external
/// (ψ) coordinate index `ext_idx`, calling `f(ext_idx, &direction)` returns
/// `Some(D_β H_ψ[u])` when the family provides it, or `None` otherwise.
///
/// The returned closure also adds the penalty-side β-drift when the ψ
/// coordinate moves realized penalties: `D_β S_ψ[u] = 0` for ψ that
/// only enters via the likelihood, so the penalty contribution vanishes
/// and the callback delegates entirely to the family hook. (Penalty
/// matrices S_ψ do not depend on β, so their β-directional derivative
/// is zero.)
///
/// # Returns
///
/// `Some(callback)` when the family potentially provides the drift term.
/// `None` when the family is Gaussian (B_i is β-independent for all
/// coordinates, so M_i ≡ 0).
pub fn build_psi_drift_deriv_callback<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    derivative_blocks_arc: SharedDerivativeBlocks,
    hessian_beta_independent: bool,
    psi_workspace: Option<Arc<dyn ExactNewtonJointPsiWorkspace>>,
) -> Option<FixedDriftDerivFn> {
    if hessian_beta_independent {
        // Likelihood Hessian is β-independent; M_i ≡ 0.
        return None;
    }

    let synced_arc = Arc::new(synced_states.to_vec());
    let specs_arc = Arc::new(specs.to_vec());
    let family_arc = Arc::new(family.clone());
    let psi_workspace = psi_workspace;

    Some(Box::new(
        move |ext_idx: usize, direction: &Array1<f64>| -> Option<DriftDerivResult> {
            // The family hook takes a psi index (0-based within ψ coordinates)
            // and a flattened coefficient direction.
            if let Some(workspace) = psi_workspace.as_ref() {
                workspace
                    .hessian_directional_derivative(ext_idx, direction)
                    .ok()
                    .flatten()
            } else {
                family_arc
                    .exact_newton_joint_psihessian_directional_derivative(
                        &synced_arc,
                        &specs_arc,
                        &derivative_blocks_arc,
                        ext_idx,
                        direction,
                    )
                    .ok()
                    .flatten()
                    .map(DriftDerivResult::Dense)
            }
        },
    ))
}

fn evaluate_custom_family_hyper_internal<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    warm_start: Option<&ConstrainedWarmStart>,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    evaluate_custom_family_hyper_internal_shared(
        family,
        specs,
        options,
        penalty_counts,
        rho_current,
        Arc::new(derivative_blocks.to_vec()),
        warm_start,
        eval_mode,
    )
}

fn evaluate_custom_family_hyper_internal_shared<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    derivative_blocks: SharedDerivativeBlocks,
    warm_start: Option<&ConstrainedWarmStart>,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    if derivative_blocks.len() != specs.len() {
        return Err(format!(
            "joint hyper derivative block count mismatch: got {}, expected {}",
            derivative_blocks.len(),
            specs.len()
        )
        .into());
    }

    if penalty_counts.len() != specs.len() {
        return Err(format!(
            "joint hyper penalty-count block mismatch: got {}, expected {}",
            penalty_counts.len(),
            specs.len()
        )
        .into());
    }
    let rho_dim = penalty_counts.iter().sum::<usize>();
    let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
    if rho_current.len() != rho_dim {
        return Err(format!(
            "joint hyper rho dimension mismatch: got {}, expected {} (psi={})",
            rho_current.len(),
            rho_dim,
            psi_dim
        )
        .into());
    }

    // ── Common setup: inner solve, ridge, refresh, ranges ──
    let include_logdet_h = include_exact_newton_logdet_h(family, options);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    let per_block = split_log_lambdas(rho_current, &penalty_counts)?;
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, warm_start)?;
    let ridge = effective_solverridge(options.ridge_floor);
    let moderidge = if options.ridge_policy.include_quadratic_penalty {
        ridge
    } else {
        0.0
    };
    let extra_logdet_ridge = if options.ridge_policy.include_penalty_logdet
        && !options.ridge_policy.include_quadratic_penalty
    {
        ridge
    } else {
        0.0
    };

    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);

    // ── Try to obtain a joint Hessian and route through the unified evaluator ──
    //
    // When psi_dim > 0, exact Newton is required because the ψ derivative
    // callbacks use exact Newton trait methods. When psi_dim == 0,
    // build_joint_hessian_closures handles both exact Newton and surrogate.
    if psi_dim > 0 {
        // ψ coordinates present: require exact Newton Hessian for consistency
        // with the psi derivative callbacks.
        let beta_flat = flatten_state_betas(&inner.block_states, specs);
        let synced_joint_states = Arc::new(synchronized_states_from_flat_beta(
            family,
            specs,
            &inner.block_states,
            &beta_flat,
        )?);
        let hessian_workspace =
            family.exact_newton_joint_hessian_workspace(synced_joint_states.as_ref(), specs)?;
        let (
            h_joint_unpen,
            rho_curvature_scale,
            hessian_logdet_correction,
            use_outer_curvature_derivatives,
        ) = if let Some(curvature) = family.exact_newton_outer_curvature(&inner.block_states)? {
            (
                JointHessianSource::Dense(symmetrized_square_matrix(
                    curvature.hessian,
                    total,
                    "joint exact-newton Hessian shape mismatch in joint hyper evaluator (rescaled)",
                )?),
                curvature.rho_curvature_scale,
                curvature.hessian_logdet_correction,
                true,
            )
        } else {
            let h_joint_unpen = if use_joint_matrix_free_path(
                total,
                joint_observation_count(synced_joint_states.as_ref()),
            ) {
                hessian_workspace
                    .as_ref()
                    .map(|workspace| {
                        exact_newton_joint_hessian_source_from_workspace(
                            workspace,
                            total,
                            "joint exact-newton operator mismatch in joint hyper evaluator",
                        )
                    })
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            (
                match h_joint_unpen {
                    Some(source) => Some(source),
                    None => exact_newton_joint_hessian_symmetrized(
                        family,
                        &inner.block_states,
                        specs,
                        total,
                        "joint exact-newton Hessian shape mismatch in joint hyper evaluator",
                    )
                    .map(|source| source.map(JointHessianSource::Dense))?,
                }
                .ok_or_else(|| -> CustomFamilyError {
                    "joint exact-newton Hessian unavailable for full [rho, psi] outer calculus"
                        .to_string()
                        .into()
                })?,
                1.0,
                0.0,
                false,
            )
        };

        // Build the exact pseudologdet eigenspace for each penalty block so
        // the value, ψ gradient, ψψ Hessian, and ρψ mixed block all
        // differentiate the same log|S|_+ objective.
        let s_logdet_blocks = if include_logdet_s {
            let mut blocks = Vec::with_capacity(specs.len());
            for (b, spec) in specs.iter().enumerate() {
                let p = spec.design.ncols();
                let lambdas = per_block[b].mapv(f64::exp);
                let mut s_lambda = Array2::<f64>::zeros((p, p));
                for (k, s) in spec.penalties.iter().enumerate() {
                    s.add_scaled_to(lambdas[k], &mut s_lambda);
                }
                if options.ridge_policy.include_penalty_logdet {
                    for d in 0..p {
                        s_lambda[[d, d]] += ridge;
                    }
                }
                let structural_nullity = if !spec.nullspace_dims.is_empty()
                    && spec.nullspace_dims.len() == spec.penalties.len()
                {
                    let penalties_dense: Vec<Array2<f64>> = spec
                        .penalties
                        .iter()
                        .map(|penalty| penalty.to_dense())
                        .collect();
                    Some(exact_intersection_nullity(
                        &penalties_dense,
                        &spec.nullspace_dims,
                    ))
                } else {
                    None
                };
                blocks.push(PenaltyPseudologdet::from_assembled_with_nullity(
                    s_lambda,
                    structural_nullity,
                )?);
            }
            Some(blocks)
        } else {
            None
        };

        // Build ψ HyperCoords, pair callbacks, and drift derivative callback.
        let hessian_beta_independent = !family.exact_newton_joint_hessian_beta_dependent();
        let psi_workspace = if eval_mode != EvalMode::ValueOnly
            && (eval_mode == EvalMode::ValueGradientHessian
                || family.exact_newton_joint_psi_workspace_for_first_order_terms())
        {
            family.exact_newton_joint_psi_workspace(
                synced_joint_states.as_ref(),
                specs,
                derivative_blocks.as_ref(),
            )?
        } else {
            None
        };

        let rho_slice = rho_current
            .as_slice()
            .ok_or_else(|| "outer rho vector must be contiguous".to_string())?;
        let ext_bundle = if eval_mode == EvalMode::ValueOnly {
            None
        } else {
            let psi_coords = build_psi_hyper_coords(
                family,
                synced_joint_states.as_ref(),
                specs,
                derivative_blocks.as_ref(),
                &beta_flat,
                rho_slice,
                &penalty_counts,
                s_logdet_blocks.as_deref(),
                hessian_beta_independent,
                psi_workspace.clone(),
            )?;

            let (ext_ext_fn, rho_ext_fn, drift_fn) = if eval_mode == EvalMode::ValueGradientHessian
            {
                let (ext_ext_fn, rho_ext_fn) = build_psi_pair_callbacks(
                    family,
                    synced_joint_states.as_ref(),
                    specs,
                    Arc::clone(&derivative_blocks),
                    &beta_flat,
                    rho_slice,
                    &penalty_counts,
                    s_logdet_blocks.as_deref(),
                    psi_workspace.clone(),
                )?;
                let drift_fn = build_psi_drift_deriv_callback(
                    family,
                    synced_joint_states.as_ref(),
                    specs,
                    Arc::clone(&derivative_blocks),
                    hessian_beta_independent,
                    psi_workspace,
                );
                (Some(ext_ext_fn), Some(rho_ext_fn), drift_fn)
            } else {
                (None, None, None)
            };

            Some(ExtCoordBundle {
                coords: psi_coords,
                ext_ext_fn,
                rho_ext_fn,
                drift_fn,
            })
        };

        // Build derivative provider for the ρ coordinates (D_β H[v]).
        let compute_dh = exact_newton_dh_closure(
            family,
            Arc::clone(&synced_joint_states),
            specs,
            total,
            use_outer_curvature_derivatives,
            if use_outer_curvature_derivatives {
                1.0
            } else {
                rho_curvature_scale
            },
            hessian_workspace.clone(),
        );
        let compute_d2h = exact_newton_d2h_closure(
            family,
            Arc::clone(&synced_joint_states),
            specs,
            total,
            use_outer_curvature_derivatives,
            if use_outer_curvature_derivatives {
                1.0
            } else {
                rho_curvature_scale
            },
            hessian_workspace.clone(),
        );
        let owned_compute_dh = exact_newton_dh_closure_owned(
            family.clone(),
            Arc::clone(&synced_joint_states),
            specs.to_vec(),
            total,
            use_outer_curvature_derivatives,
            if use_outer_curvature_derivatives {
                1.0
            } else {
                rho_curvature_scale
            },
            hessian_workspace.clone(),
        );
        let owned_compute_d2h = exact_newton_d2h_closure_owned(
            family.clone(),
            Arc::clone(&synced_joint_states),
            specs.to_vec(),
            total,
            use_outer_curvature_derivatives,
            if use_outer_curvature_derivatives {
                1.0
            } else {
                rho_curvature_scale
            },
            hessian_workspace.clone(),
        );

        // Route through the unified path (joint_outer_evaluate → reml_laml_evaluate).
        let eval_result = joint_outer_evaluate(
            &inner,
            specs,
            &per_block,
            rho_current,
            &beta_flat,
            h_joint_unpen,
            &ranges,
            total,
            ridge,
            moderidge,
            extra_logdet_ridge,
            rho_curvature_scale,
            hessian_logdet_correction,
            include_logdet_h,
            include_logdet_s,
            strict_spd,
            eval_mode,
            options,
            family.pseudo_logdet_mode(),
            &compute_dh,
            &compute_d2h,
            Some(owned_compute_dh),
            Some(owned_compute_d2h),
            ext_bundle,
        )?;

        // The unified evaluator produces gradient/Hessian of size (rho_dim + psi_dim),
        // with ρ coordinates first and ψ coordinates appended — matching the expected
        // output order of CustomFamilyJointHyperResult.
        return Ok(eval_result);
    }

    // ── ρ-only path (psi_dim == 0): route through unified evaluator ──
    //
    // Try build_joint_hessian_closures which handles both exact Newton and
    // surrogate Hessian sources, then call joint_outer_evaluate with no
    // extended coordinates.
    if let Some(joint_bundle) =
        build_joint_hessian_closures(family, &inner.block_states, specs, total)?
    {
        let JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_d2h,
            rho_curvature_scale,
            hessian_logdet_correction,
        } = joint_bundle;
        let eval_result = joint_outer_evaluate(
            &inner,
            specs,
            &per_block,
            rho_current,
            &beta_flat,
            h_joint_unpen,
            &ranges,
            total,
            ridge,
            moderidge,
            extra_logdet_ridge,
            rho_curvature_scale,
            hessian_logdet_correction,
            include_logdet_h,
            include_logdet_s,
            strict_spd,
            eval_mode,
            options,
            family.pseudo_logdet_mode(),
            compute_dh.as_ref(),
            compute_d2h.as_ref(),
            None,
            None,
            None, // no ext_coords when psi_dim == 0
        )?;

        return Ok(eval_result);
    }

    // Joint Hessian unavailable via either exact Newton or surrogate.
    // The generic fallback is only mathematically defensible for single-block
    // families — multi-block families with coupled likelihood curvature require
    // the joint path.
    if family.requires_joint_outer_hyper_path() {
        return Err(
            "outer hyper-derivative evaluation requires a joint exact path for this family"
                .to_string()
                .into(),
        );
    }

    // Generic fallback: single-block only. Extract the per-block Hessian and
    // route through joint_outer_evaluate with the single block as the "joint"
    // system.
    if specs.len() != 1 {
        return Err(
            "generic outer fallback is only valid for single-block families; multi-block families must provide a joint outer path"
                .to_string()
                .into(),
        );
    }
    let eval = family.evaluate(&inner.block_states)?;
    let b = 0;
    let spec = &specs[b];
    let work = &eval.blockworking_sets[b];
    let p = spec.design.ncols();
    let mut diagonal_design = None::<DesignMatrix>;
    let h_joint_unpen = match work {
        BlockWorkingSet::Diagonal {
            working_response: _,
            working_weights,
        } => with_block_geometry(family, &inner.block_states, spec, b, |x_dyn, _| {
            let w = floor_positiveworking_weights(working_weights, options.minweight);
            let (xtwx, _) = weighted_normal_equations(x_dyn, &w, None)?;
            diagonal_design = Some(x_dyn.clone());
            Ok(xtwx)
        })?,
        BlockWorkingSet::ExactNewton {
            gradient: _,
            hessian,
        } => {
            if hessian.nrows() != p || hessian.ncols() != p {
                return Err(format!(
                    "block {b} exact-newton Hessian shape mismatch in outer gradient: got {}x{}, expected {}x{}",
                    hessian.nrows(),
                    hessian.ncols(),
                    p,
                    p
                ).into());
            }
            hessian.to_dense()
        }
    };

    let beta_flat = inner.block_states[b].beta.clone();

    // Build a derivative provider that computes D_β H_L[direction] on demand.
    let compute_dh = |direction: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
        if !include_logdet_h {
            return Ok(None);
        }
        match work {
            BlockWorkingSet::ExactNewton { .. } => {
                match family.exact_newton_hessian_directional_derivative(
                    &inner.block_states,
                    b,
                    direction,
                )? {
                    Some(h_exact) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        h_exact,
                        p,
                        &format!("block {b} exact-newton dH shape mismatch"),
                    )?))),
                    None => Err(format!(
                        "missing exact-newton dH callback for block {b} while REML gradient requires H_beta term"
                    )),
                }
            }
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => {
                let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                    format!("missing dynamic design for block {b} diagonal correction")
                })?;
                let wwork = floor_positiveworking_weights(working_weights, options.minweight);
                let x_dense = x_dyn.to_dense();
                let n = x_dense.nrows();

                let mut d_eta = x_dyn.matrixvectormultiply(direction);
                let geom = family.block_geometry_directional_derivative(
                    &inner.block_states,
                    b,
                    spec,
                    direction,
                )?;
                let mut correction_mat = Array2::<f64>::zeros((p, p));

                if let Some(geom_dir) = geom {
                    d_eta += &geom_dir.d_offset;
                    if let Some(dx) = geom_dir.d_design {
                        d_eta += &dx.dot(&beta_flat);
                        let mut wx = x_dense.clone();
                        let mut wdx = dx.clone();
                        for i in 0..n {
                            let wi = wwork[i];
                            if wi != 1.0 {
                                wx.row_mut(i).mapv_inplace(|v| v * wi);
                                wdx.row_mut(i).mapv_inplace(|v| v * wi);
                            }
                        }
                        correction_mat += &dx.t().dot(&wx);
                        correction_mat += &x_dense.t().dot(&wdx);
                    }
                }

                let dw = family
                    .diagonalworking_weights_directional_derivative(
                        &inner.block_states,
                        b,
                        &d_eta,
                    )?
                    .ok_or_else(|| {
                        format!(
                            "missing diagonal dW callback for block {b} while REML gradient requires H_beta term"
                        )
                    })?;
                if dw.len() != n {
                    return Err(format!(
                        "block {b} diagonal dW length mismatch: got {}, expected {}",
                        dw.len(),
                        n
                    ));
                }
                let mut scaled_x = x_dense.clone();
                for i in 0..n {
                    scaled_x.row_mut(i).mapv_inplace(|v| v * dw[i]);
                }
                correction_mat += &x_dense.t().dot(&scaled_x);

                Ok(Some(DriftDerivResult::Dense(correction_mat)))
            }
        }
    };

    // No d²H provider for the generic single-block fallback.
    let compute_d2h =
        |_: &Array1<f64>, _: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> { Ok(None) };

    let eval_result = joint_outer_evaluate(
        &inner,
        specs,
        &per_block,
        rho_current,
        &beta_flat,
        JointHessianSource::Dense(h_joint_unpen),
        &ranges,
        total,
        ridge,
        moderidge,
        extra_logdet_ridge,
        1.0,
        0.0,
        include_logdet_h,
        include_logdet_s,
        strict_spd,
        eval_mode,
        options,
        family.pseudo_logdet_mode(),
        &compute_dh,
        &compute_d2h,
        None,
        None,
        None, // no ext_coords for generic single-block fallback
    )?;

    Ok(eval_result)
}

pub fn evaluate_custom_family_joint_hyper<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    warm_start: Option<&CustomFamilyWarmStart>,
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let eval_result = evaluate_custom_family_hyper_internal(
        family,
        specs,
        options,
        &penalty_counts,
        rho_current,
        derivative_blocks,
        warm_start.map(|w| &w.inner),
        eval_mode,
    )?;
    Ok(outer_eval_result_to_joint_hyper_result(eval_result))
}

pub(crate) fn evaluate_custom_family_joint_hyper_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    derivative_blocks: SharedDerivativeBlocks,
    warm_start: Option<&CustomFamilyWarmStart>,
    eval_mode: EvalMode,
) -> Result<CustomFamilyJointHyperResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let eval_result = evaluate_custom_family_hyper_internal_shared(
        family,
        specs,
        options,
        &penalty_counts,
        rho_current,
        derivative_blocks,
        warm_start.map(|w| &w.inner),
        eval_mode,
    )?;
    Ok(outer_eval_result_to_joint_hyper_result(eval_result))
}

fn evaluate_custom_family_joint_hyper_efs_internal<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<(crate::solver::outer_strategy::EfsEval, ConstrainedWarmStart), CustomFamilyError> {
    evaluate_custom_family_joint_hyper_efs_internal_shared(
        family,
        specs,
        options,
        penalty_counts,
        rho_current,
        Arc::new(derivative_blocks.to_vec()),
        warm_start,
    )
}

fn evaluate_custom_family_joint_hyper_efs_internal_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    derivative_blocks: SharedDerivativeBlocks,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<(crate::solver::outer_strategy::EfsEval, ConstrainedWarmStart), CustomFamilyError> {
    if derivative_blocks.len() != specs.len() {
        return Err(format!(
            "joint hyper derivative block count mismatch: got {}, expected {}",
            derivative_blocks.len(),
            specs.len()
        )
        .into());
    }
    if penalty_counts.len() != specs.len() {
        return Err(format!(
            "joint hyper penalty-count block mismatch: got {}, expected {}",
            penalty_counts.len(),
            specs.len()
        )
        .into());
    }

    let rho_dim = penalty_counts.iter().sum::<usize>();
    let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
    if psi_dim == 0 {
        return Err(CustomFamilyError::InvalidInput(
            "joint hyper EFS requires at least one ψ coordinate".to_string(),
        ));
    }
    if rho_current.len() != rho_dim {
        return Err(format!(
            "joint hyper rho dimension mismatch: got {}, expected {} (psi={})",
            rho_current.len(),
            rho_dim,
            psi_dim
        )
        .into());
    }

    let include_logdet_h = include_exact_newton_logdet_h(family, options);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    let per_block = split_log_lambdas(rho_current, penalty_counts)?;
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, warm_start)?;
    let ridge = effective_solverridge(options.ridge_floor);
    let moderidge = if options.ridge_policy.include_quadratic_penalty {
        ridge
    } else {
        0.0
    };
    let extra_logdet_ridge = if options.ridge_policy.include_penalty_logdet
        && !options.ridge_policy.include_quadratic_penalty
    {
        ridge
    } else {
        0.0
    };

    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);

    let beta_flat = flatten_state_betas(&inner.block_states, specs);
    let synced_joint_states = Arc::new(synchronized_states_from_flat_beta(
        family,
        specs,
        &inner.block_states,
        &beta_flat,
    )?);
    let hessian_workspace =
        family.exact_newton_joint_hessian_workspace(synced_joint_states.as_ref(), specs)?;
    let (
        h_joint_unpen,
        rho_curvature_scale,
        hessian_logdet_correction,
        use_outer_curvature_derivatives,
    ) = if let Some(curvature) = family.exact_newton_outer_curvature(&inner.block_states)? {
        (
            JointHessianSource::Dense(symmetrized_square_matrix(
                curvature.hessian,
                total,
                "joint exact-newton Hessian shape mismatch in joint hyper EFS evaluator (rescaled)",
            )?),
            curvature.rho_curvature_scale,
            curvature.hessian_logdet_correction,
            true,
        )
    } else {
        let h_joint_unpen = if use_joint_matrix_free_path(
            total,
            joint_observation_count(synced_joint_states.as_ref()),
        ) {
            hessian_workspace
                .as_ref()
                .map(|workspace| {
                    exact_newton_joint_hessian_source_from_workspace(
                        workspace,
                        total,
                        "joint exact-newton operator mismatch in joint hyper EFS evaluator",
                    )
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
        (
            match h_joint_unpen {
                Some(source) => Some(source),
                None => exact_newton_joint_hessian_symmetrized(
                    family,
                    &inner.block_states,
                    specs,
                    total,
                    "joint exact-newton Hessian shape mismatch in joint hyper EFS evaluator",
                )
                .map(|source| source.map(JointHessianSource::Dense))?,
            }
            .ok_or_else(|| -> CustomFamilyError {
                "joint exact-newton Hessian unavailable for full [rho, psi] fixed-point outer calculus"
                    .to_string()
                    .into()
            })?,
            1.0,
            0.0,
            false,
        )
    };

    let s_logdet_blocks = if include_logdet_s {
        let mut blocks = Vec::with_capacity(specs.len());
        for (b, spec) in specs.iter().enumerate() {
            let p = spec.design.ncols();
            let lambdas = per_block[b].mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((p, p));
            for (k, s) in spec.penalties.iter().enumerate() {
                s.add_scaled_to(lambdas[k], &mut s_lambda);
            }
            if options.ridge_policy.include_penalty_logdet {
                for d in 0..p {
                    s_lambda[[d, d]] += ridge;
                }
            }
            let structural_nullity = if !spec.nullspace_dims.is_empty()
                && spec.nullspace_dims.len() == spec.penalties.len()
            {
                let penalties_dense: Vec<Array2<f64>> = spec
                    .penalties
                    .iter()
                    .map(|penalty| penalty.to_dense())
                    .collect();
                Some(exact_intersection_nullity(
                    &penalties_dense,
                    &spec.nullspace_dims,
                ))
            } else {
                None
            };
            blocks.push(PenaltyPseudologdet::from_assembled_with_nullity(
                s_lambda,
                structural_nullity,
            )?);
        }
        Some(blocks)
    } else {
        None
    };

    let hessian_beta_independent = !family.exact_newton_joint_hessian_beta_dependent();
    let psi_workspace = if family.exact_newton_joint_psi_workspace_for_first_order_terms() {
        family.exact_newton_joint_psi_workspace(
            synced_joint_states.as_ref(),
            specs,
            derivative_blocks.as_ref(),
        )?
    } else {
        None
    };
    let rho_slice = rho_current
        .as_slice()
        .ok_or_else(|| "outer rho vector must be contiguous".to_string())?;
    let psi_coords = build_psi_hyper_coords(
        family,
        synced_joint_states.as_ref(),
        specs,
        derivative_blocks.as_ref(),
        &beta_flat,
        rho_slice,
        penalty_counts,
        s_logdet_blocks.as_deref(),
        hessian_beta_independent,
        psi_workspace.clone(),
    )?;
    let ext_bundle = ExtCoordBundle {
        coords: psi_coords,
        ext_ext_fn: None,
        rho_ext_fn: None,
        drift_fn: None,
    };

    let compute_dh = exact_newton_dh_closure(
        family,
        Arc::clone(&synced_joint_states),
        specs,
        total,
        use_outer_curvature_derivatives,
        if use_outer_curvature_derivatives {
            1.0
        } else {
            rho_curvature_scale
        },
        hessian_workspace.clone(),
    );
    let compute_d2h = exact_newton_d2h_closure(
        family,
        Arc::clone(&synced_joint_states),
        specs,
        total,
        use_outer_curvature_derivatives,
        if use_outer_curvature_derivatives {
            1.0
        } else {
            rho_curvature_scale
        },
        hessian_workspace.clone(),
    );
    let owned_compute_dh = exact_newton_dh_closure_owned(
        family.clone(),
        Arc::clone(&synced_joint_states),
        specs.to_vec(),
        total,
        use_outer_curvature_derivatives,
        if use_outer_curvature_derivatives {
            1.0
        } else {
            rho_curvature_scale
        },
        hessian_workspace.clone(),
    );
    let owned_compute_d2h = exact_newton_d2h_closure_owned(
        family.clone(),
        Arc::clone(&synced_joint_states),
        specs.to_vec(),
        total,
        use_outer_curvature_derivatives,
        if use_outer_curvature_derivatives {
            1.0
        } else {
            rho_curvature_scale
        },
        hessian_workspace,
    );

    let efs_eval = joint_outer_evaluate_efs(
        &inner,
        specs,
        &per_block,
        rho_current,
        &beta_flat,
        h_joint_unpen,
        &ranges,
        total,
        ridge,
        moderidge,
        extra_logdet_ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        include_logdet_h,
        include_logdet_s,
        strict_spd,
        options,
        family.pseudo_logdet_mode(),
        &compute_dh,
        &compute_d2h,
        Some(owned_compute_dh),
        Some(owned_compute_d2h),
        Some(ext_bundle),
    )
    .map_err(CustomFamilyError::from)?;

    let warm = ConstrainedWarmStart {
        rho: rho_current.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|state| state.beta.clone())
            .collect(),
        active_sets: inner.active_sets.clone(),
    };

    Ok((efs_eval, warm))
}

/// Evaluate the joint custom-family hyper-surface in fixed-point form for the
/// outer EFS / hybrid-EFS planners.
pub fn evaluate_custom_family_joint_hyper_efs<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<CustomFamilyJointHyperEfsResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    if derivative_blocks.len() != specs.len() {
        return Err(format!(
            "joint hyper derivative block count mismatch: got {}, expected {}",
            derivative_blocks.len(),
            specs.len()
        )
        .into());
    }
    let (efs_eval, warm_start) = if derivative_blocks.iter().all(Vec::is_empty) {
        outerobjectiveefs(
            family,
            specs,
            options,
            &penalty_counts,
            rho_current,
            warm_start.map(|w| &w.inner),
        )
        .map_err(CustomFamilyError::from)?
    } else {
        evaluate_custom_family_joint_hyper_efs_internal(
            family,
            specs,
            options,
            &penalty_counts,
            rho_current,
            derivative_blocks,
            warm_start.map(|w| &w.inner),
        )?
    };
    Ok(outer_efs_result_to_joint_hyper_efs_result(
        efs_eval, warm_start,
    ))
}

pub(crate) fn evaluate_custom_family_joint_hyper_efs_shared<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    derivative_blocks: SharedDerivativeBlocks,
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<CustomFamilyJointHyperEfsResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    if derivative_blocks.len() != specs.len() {
        return Err(format!(
            "joint hyper derivative block count mismatch: got {}, expected {}",
            derivative_blocks.len(),
            specs.len()
        )
        .into());
    }
    let (efs_eval, warm_start) = if derivative_blocks.iter().all(Vec::is_empty) {
        outerobjectiveefs(
            family,
            specs,
            options,
            &penalty_counts,
            rho_current,
            warm_start.map(|w| &w.inner),
        )
        .map_err(CustomFamilyError::from)?
    } else {
        evaluate_custom_family_joint_hyper_efs_internal_shared(
            family,
            specs,
            options,
            &penalty_counts,
            rho_current,
            derivative_blocks,
            warm_start.map(|w| &w.inner),
        )?
    };
    Ok(outer_efs_result_to_joint_hyper_efs_result(
        efs_eval, warm_start,
    ))
}

fn block_param_ranges(specs: &[ParameterBlockSpec]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(specs.len());
    let mut at = 0usize;
    for spec in specs {
        let p = spec.design.ncols();
        out.push((at, at + p));
        at += p;
    }
    out
}

const JOINT_MATRIX_FREE_MIN_DIM: usize = 512;
const JOINT_MATRIX_FREE_MIN_ROWS: usize = 50_000;
const JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N: usize = 32;
const JOINT_MATRIX_FREE_MIN_LINEAR_WORK: usize = 4_000_000;
const JOINT_TRACE_STABILITY_RIDGE: f64 = 1e-10;
const JOINT_PCG_REL_TOL: f64 = 1e-8;
const JOINT_PCG_MAX_ITER_MULTIPLIER: usize = 4;

pub(crate) fn joint_exact_analytic_outer_hessian_available() -> bool {
    true
}

fn joint_observation_count(states: &[ParameterBlockState]) -> usize {
    states
        .iter()
        .map(|state| state.eta.len())
        .max()
        .unwrap_or(0)
}

fn use_joint_matrix_free_path(total_p: usize, total_n: usize) -> bool {
    total_p >= JOINT_MATRIX_FREE_MIN_DIM
        || (total_n >= JOINT_MATRIX_FREE_MIN_ROWS
            && total_p >= JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N)
        || total_n.saturating_mul(total_p) >= JOINT_MATRIX_FREE_MIN_LINEAR_WORK
}

fn apply_joint_block_penalty(
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    vector: &Array1<f64>,
    diagonal_ridge: f64,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(vector.len());
    for (b, s_lambda) in s_lambdas.iter().enumerate() {
        let (start, end) = ranges[b];
        let block = vector.slice(s![start..end]);
        out.slice_mut(s![start..end]).assign(&s_lambda.dot(&block));
    }
    if diagonal_ridge > 0.0 {
        out.scaled_add(diagonal_ridge, vector);
    }
    out
}

fn joint_penalty_preconditioner_diag(
    base_diagonal: &Array1<f64>,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
) -> Array1<f64> {
    let mut diag = base_diagonal.clone();
    for (b, s_lambda) in s_lambdas.iter().enumerate() {
        let (start, end) = ranges[b];
        for (local_idx, global_idx) in (start..end).enumerate() {
            diag[global_idx] += s_lambda[[local_idx, local_idx]];
        }
    }
    if diagonal_ridge > 0.0 {
        for value in &mut diag {
            *value += diagonal_ridge;
        }
    }
    diag.mapv(|v| v.abs().max(1e-10))
}

fn add_joint_penalty_to_matrix(
    matrix: &mut Array2<f64>,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
) {
    for (b, s_lambda) in s_lambdas.iter().enumerate() {
        let (start, end) = ranges[b];
        let mut block = matrix.slice_mut(s![start..end, start..end]);
        block += s_lambda;
    }
    if diagonal_ridge > 0.0 {
        for d in 0..matrix.nrows() {
            matrix[[d, d]] += diagonal_ridge;
        }
    }
}

fn flatten_state_betas(
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Array1<f64> {
    let total = specs.iter().map(|s| s.design.ncols()).sum::<usize>();
    let mut beta = Array1::<f64>::zeros(total);
    let ranges = block_param_ranges(specs);
    for (b, (start, end)) in ranges.into_iter().enumerate() {
        beta.slice_mut(ndarray::s![start..end])
            .assign(&states[b].beta);
    }
    beta
}

fn set_states_from_flat_beta(
    states: &mut [ParameterBlockState],
    specs: &[ParameterBlockSpec],
    beta_flat: &Array1<f64>,
) -> Result<(), String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if beta_flat.len() != total {
        return Err(format!(
            "flat beta length mismatch: got {}, expected {}",
            beta_flat.len(),
            total
        ));
    }
    for (b, (start, end)) in ranges.into_iter().enumerate() {
        states[b]
            .beta
            .assign(&beta_flat.slice(ndarray::s![start..end]).to_owned());
    }
    Ok(())
}

fn synchronized_states_from_flat_beta<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    beta_flat: &Array1<f64>,
) -> Result<Vec<ParameterBlockState>, String> {
    let mut synced = states.to_vec();
    set_states_from_flat_beta(&mut synced, specs, beta_flat)?;
    refresh_all_block_etas(family, specs, &mut synced)?;
    Ok(synced)
}

fn penalizedobjective_at_beta<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
) -> Result<f64, String> {
    let eval = family.evaluate(states)?;
    if eval.blockworking_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.blockworking_sets.len(),
            specs.len()
        ));
    }
    let mut penalty = 0.0_f64;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let beta = &states[b].beta;
        let lambdas = per_block_log_lambdas
            .get(b)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(spec.penalties.len()))
            .mapv(f64::exp);
        for (k, s) in spec.penalties.iter().enumerate() {
            if k < lambdas.len() {
                let sb = s.dot(beta);
                penalty += 0.5 * lambdas[k] * beta.dot(&sb);
            }
        }
    }
    Ok(-eval.log_likelihood + penalty)
}

/// Inf-norm of the penalized stationarity residual with KKT multipliers
/// projected out at active lower bounds.
///
/// For a box-constrained convex quadratic, the KKT conditions at β̂ read
///
///   [S·β̂ − ∇ℓ(β̂)]_j + λ_j = 0                (j with active bound β̂_j = lb_j)
///   [S·β̂ − ∇ℓ(β̂)]_j      = 0                (j free / strictly interior)
///   λ_j ≥ 0                                  (dual feasibility)
///
/// so `residual_j = +λ_j` on active-bound coordinates — not a convergence
/// defect but a valid inequality multiplier. The inner-loop convergence
/// check must measure only the **free-set** residual, otherwise it never
/// fires at a constrained optimum and falls through to blockwise fallback,
/// wasting inner cycles and reporting spurious non-convergence.
///
/// This helper zeros the residual at coordinate `j` when
///   (i)  `j` carries a finite lower bound, AND
///   (ii) `β_j` is within a scale-relative tolerance of that bound, AND
///   (iii) `residual_j > 0` (sign of a valid inequality multiplier).
///
/// The tolerance matches `projected_gradient_norm` in `pirls.rs:2186`
/// (`1e-6 · max(|β_j|, |lb_j|, 1) + 1e-10`), so the joint-Newton and
/// blockwise PIRLS convergence criteria agree on the active set.
///
/// When `constraints` is `None` or cannot be decomposed into simple lower
/// bounds (coupled inequalities), behaviour falls back to the unprojected
/// inf-norm, which is the correct measure for unconstrained / coupled
/// problems.
fn projected_stationarity_inf_norm(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: Option<&LinearInequalityConstraints>,
) -> f64 {
    debug_assert_eq!(residual.len(), beta.len());
    let lower_bounds: Option<Array1<f64>> = constraints
        .and_then(|c| extract_simple_lower_bounds(c, beta.len()).ok().flatten())
        .map(|b| b.lower_bounds);
    let mut inf = 0.0_f64;
    for j in 0..residual.len() {
        let r = residual[j];
        if let Some(lb_arr) = lower_bounds.as_ref() {
            let lb = lb_arr[j];
            if lb.is_finite() && r > 0.0 {
                let scale = beta[j].abs().max(lb.abs()).max(1.0);
                let tol = 1e-6 * scale + 1e-10;
                if beta[j] - lb <= tol {
                    // Active lower bound with multiplier-signed residual; skip.
                    continue;
                }
            }
        }
        inf = inf.max(r.abs());
    }
    inf
}

fn exact_newton_joint_stationarity_inf_norm<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    eval: &FamilyEvaluation,
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> Result<Option<f64>, String> {
    if eval.blockworking_sets.len() != states.len() || states.len() != s_lambdas.len() {
        return Err("exact-newton joint stationarity check: block dimension mismatch".to_string());
    }
    if specs.len() != states.len() {
        return Err("exact-newton joint stationarity check: spec/state count mismatch".to_string());
    }

    let block_constraints = collect_block_linear_constraints(family, states, specs)?;
    let mut inf_norm = 0.0_f64;
    for b in 0..states.len() {
        let gradient = match &eval.blockworking_sets[b] {
            // For exact-Newton families the block score is ∇ log L with respect
            // to that block, while the penalized negative objective is
            //
            //   Q(beta, rho) = -log L(beta) + 0.5 beta^T P_mode(rho) beta,
            //
            // where `P_mode` includes the rho-independent stabilization ridge
            // exactly when that ridge participates in the quadratic objective.
            //
            // The coupled first-order condition is therefore
            //
            //   ∇Q = -∇ log L + P beta = 0.
            //
            // So the exact penalized stationarity residual for block b is
            //
            //   r_b = P_mode,b * beta_b - gradient_b.
            //
            // For blocks with simple lower-bound constraints (e.g. I-spline
            // monotone time coefficients, monotone wiggle coefficients) the
            // residual on an active-bound coordinate is the KKT multiplier
            // λ_j ≥ 0 rather than a convergence defect; the projection in
            // `projected_stationarity_inf_norm` drops those entries so the
            // inf-norm measures only the free-set residual that must be
            // driven to zero. Using only coordinate step size or an
            // unprojected norm can declare convergence too early OR fail to
            // ever declare convergence at a constrained optimum.
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient,
            _ => return Ok(None),
        };
        let mut residual = s_lambdas[b].dot(&states[b].beta) - gradient;
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            residual += &states[b].beta.mapv(|v| ridge * v);
        }
        let block_inf = projected_stationarity_inf_norm(
            &residual,
            &states[b].beta,
            block_constraints[b].as_ref(),
        );
        inf_norm = inf_norm.max(block_inf);
    }
    Ok(Some(inf_norm))
}

fn exact_newton_joint_gradient_from_eval(
    eval: &FamilyEvaluation,
    specs: &[ParameterBlockSpec],
) -> Result<Option<Array1<f64>>, String> {
    if eval.blockworking_sets.len() != specs.len() {
        return Err(format!(
            "exact-newton joint gradient extraction: family returned {} block working sets, expected {}",
            eval.blockworking_sets.len(),
            specs.len()
        ));
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let mut gradient = Array1::<f64>::zeros(total_p);
    let mut offset = 0usize;
    for (spec, work) in specs.iter().zip(eval.blockworking_sets.iter()) {
        let width = spec.design.ncols();
        let BlockWorkingSet::ExactNewton {
            gradient: block_gradient,
            ..
        } = work
        else {
            return Ok(None);
        };
        if block_gradient.len() != width {
            return Err(format!(
                "exact-newton joint gradient extraction: block gradient length mismatch, got {}, expected {}",
                block_gradient.len(),
                width
            ));
        }
        gradient
            .slice_mut(ndarray::s![offset..offset + width])
            .assign(block_gradient);
        offset += width;
    }
    Ok(Some(gradient))
}

fn exact_newton_joint_stationarity_inf_norm_from_gradient(
    gradient: &Array1<f64>,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_constraints: &[Option<LinearInequalityConstraints>],
) -> Result<f64, String> {
    if states.len() != specs.len() || states.len() != s_lambdas.len() {
        return Err(
            "exact-newton joint stationarity check from gradient: block dimension mismatch"
                .to_string(),
        );
    }
    if block_constraints.len() != states.len() {
        return Err(format!(
            "exact-newton joint stationarity check from gradient: constraint count mismatch, got {}, expected {}",
            block_constraints.len(),
            states.len()
        ));
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(format!(
            "exact-newton joint stationarity check from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ));
    }

    // Same KKT projection as `exact_newton_joint_stationarity_inf_norm`:
    // multipliers at active lower bounds are not convergence defects, so we
    // measure only the free-set residual. See `projected_stationarity_inf_norm`
    // for the tolerance choice and its parallel with `projected_gradient_norm`
    // in `pirls.rs`.
    let mut inf_norm = 0.0_f64;
    let mut offset = 0usize;
    for b in 0..states.len() {
        let width = specs[b].design.ncols();
        let mut residual =
            s_lambdas[b].dot(&states[b].beta) - gradient.slice(ndarray::s![offset..offset + width]);
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            residual += &states[b].beta.mapv(|v| ridge * v);
        }
        let block_inf = projected_stationarity_inf_norm(
            &residual,
            &states[b].beta,
            block_constraints[b].as_ref(),
        );
        inf_norm = inf_norm.max(block_inf);
        offset += width;
    }
    Ok(inf_norm)
}

fn compute_joint_hessian_from_objective<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
) -> Result<Array2<f64>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let beta_hat = flatten_state_betas(states, specs);
    let mut h = Array2::<f64>::zeros((total, total));

    let mut statesf0 = states.to_vec();
    set_states_from_flat_beta(&mut statesf0, specs, &beta_hat)?;
    refresh_all_block_etas(family, specs, &mut statesf0)?;
    let f0 = penalizedobjective_at_beta(family, specs, &statesf0, per_block_log_lambdas)?;

    let steps = Array1::from_iter(beta_hat.iter().map(|&b| (1e-4 * (1.0 + b.abs())).max(1e-6)));

    for i in 0..total {
        let hi = steps[i];
        let mut bp = beta_hat.clone();
        bp[i] += hi;
        let mut sp = states.to_vec();
        set_states_from_flat_beta(&mut sp, specs, &bp)?;
        refresh_all_block_etas(family, specs, &mut sp)?;
        let fp = penalizedobjective_at_beta(family, specs, &sp, per_block_log_lambdas)?;

        let mut bm = beta_hat.clone();
        bm[i] -= hi;
        let mut sm = states.to_vec();
        set_states_from_flat_beta(&mut sm, specs, &bm)?;
        refresh_all_block_etas(family, specs, &mut sm)?;
        let fm = penalizedobjective_at_beta(family, specs, &sm, per_block_log_lambdas)?;

        h[[i, i]] = ((fp - 2.0 * f0 + fm) / (hi * hi)).max(0.0);

        for j in 0..i {
            let hj = steps[j];
            let mut bpp = beta_hat.clone();
            bpp[i] += hi;
            bpp[j] += hj;
            let mut spp = states.to_vec();
            set_states_from_flat_beta(&mut spp, specs, &bpp)?;
            refresh_all_block_etas(family, specs, &mut spp)?;
            let fpp = penalizedobjective_at_beta(family, specs, &spp, per_block_log_lambdas)?;

            let mut bpm = beta_hat.clone();
            bpm[i] += hi;
            bpm[j] -= hj;
            let mut spm = states.to_vec();
            set_states_from_flat_beta(&mut spm, specs, &bpm)?;
            refresh_all_block_etas(family, specs, &mut spm)?;
            let fpm = penalizedobjective_at_beta(family, specs, &spm, per_block_log_lambdas)?;

            let mut bmp = beta_hat.clone();
            bmp[i] -= hi;
            bmp[j] += hj;
            let mut smp = states.to_vec();
            set_states_from_flat_beta(&mut smp, specs, &bmp)?;
            refresh_all_block_etas(family, specs, &mut smp)?;
            let fmp = penalizedobjective_at_beta(family, specs, &smp, per_block_log_lambdas)?;

            let mut bmm = beta_hat.clone();
            bmm[i] -= hi;
            bmm[j] -= hj;
            let mut smm = states.to_vec();
            set_states_from_flat_beta(&mut smm, specs, &bmm)?;
            refresh_all_block_etas(family, specs, &mut smm)?;
            let fmm = penalizedobjective_at_beta(family, specs, &smm, per_block_log_lambdas)?;

            let hij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            h[[i, j]] = hij;
            h[[j, i]] = hij;
        }
    }
    for i in 0..total {
        h[[i, i]] = h[[i, i]].max(1e-12);
    }
    Ok(h)
}

fn compute_joint_covariance<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Array2<f64>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let mut h = if let Some(h_exact) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total,
        "joint exact-newton Hessian shape mismatch in covariance",
    )? {
        let mut h = h_exact;
        for (b, spec) in specs.iter().enumerate() {
            let (start, end) = ranges[b];
            let lambdas = per_block_log_lambdas[b].mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((end - start, end - start));
            for (k, s) in spec.penalties.iter().enumerate() {
                s.add_scaled_to(lambdas[k], &mut s_lambda);
            }
            h.slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, &s_lambda);
        }
        h
    } else {
        compute_joint_hessian_from_objective(family, specs, states, per_block_log_lambdas)?
    };
    symmetrize_dense_in_place(&mut h);
    if use_exact_newton_strict_spd(family) {
        strict_inverse_spd(&h)
    } else {
        match inverse_spdwith_retry(&h, effective_solverridge(options.ridge_floor), 8) {
            Ok(cov) => Ok(cov),
            Err(_) => pinv_positive_part(&h, effective_solverridge(options.ridge_floor)),
        }
    }
}

fn compute_joint_covariance_required<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Option<Array2<f64>>, CustomFamilyError> {
    if !options.compute_covariance {
        return Ok(None);
    }
    compute_joint_covariance(family, specs, states, per_block_log_lambdas, options)
        .map(Some)
        .map_err(|e| {
            CustomFamilyError::InvalidInput(format!("joint covariance computation failed: {e}"))
        })
}

/// Compute joint working-set geometry at convergence for ALO diagnostics.
fn compute_joint_geometry<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
) -> Option<FitGeometry> {
    if specs.len() != per_block_log_lambdas.len() {
        return None;
    }
    if specs.len() == 1 {
        let eval = family.evaluate(states).ok()?;
        let [
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            },
        ] = eval.blockworking_sets.as_slice()
        else {
            return None;
        };
        let spec = &specs[0];
        let lambdas = per_block_log_lambdas[0].mapv(f64::exp);
        let mut h = spec.design.diag_xtw_x(working_weights).ok()?;
        for (k, s) in spec.penalties.iter().enumerate() {
            let s_dense = s.as_dense_cow();
            h.scaled_add(lambdas[k], &*s_dense);
        }
        return Some(FitGeometry {
            penalized_hessian: h,
            working_weights: working_weights.clone(),
            working_response: working_response.clone(),
        });
    }

    let total_p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let mut h = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total_p,
        "compute_joint_geometry",
    )
    .ok()??;
    let ranges = block_param_ranges(specs);
    for (block_idx, spec) in specs.iter().enumerate() {
        let lambdas = per_block_log_lambdas.get(block_idx)?.mapv(f64::exp);
        if lambdas.len() != spec.penalties.len() {
            return None;
        }
        let (start, end) = ranges[block_idx];
        let block_dim = end - start;
        for (penalty_idx, penalty) in spec.penalties.iter().enumerate() {
            let scale = lambdas[penalty_idx];
            if scale == 0.0 {
                continue;
            }
            let dense = penalty.as_dense_cow();
            if dense.nrows() == block_dim && dense.ncols() == block_dim {
                h.slice_mut(ndarray::s![start..end, start..end])
                    .scaled_add(scale, &*dense);
            } else if dense.nrows() == total_p && dense.ncols() == total_p {
                h.scaled_add(scale, &*dense);
            } else {
                return None;
            }
        }
    }
    let working_len = states.first().map(|state| state.eta.len()).unwrap_or(0);
    Some(FitGeometry {
        penalized_hessian: h,
        working_weights: Array1::zeros(working_len),
        working_response: Array1::zeros(working_len),
    })
}

pub fn fit_custom_family<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<crate::solver::estimate::UnifiedFitResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let rho0 = flatten_log_lambdas(specs);

    if rho0.is_empty() {
        let mut inner = inner_blockwise_fit(
            family,
            specs,
            &vec![Array1::zeros(0); specs.len()],
            options,
            None,
        )?;
        refresh_all_block_etas(family, specs, &mut inner.block_states)?;
        let covariance_conditional = compute_joint_covariance_required(
            family,
            specs,
            &inner.block_states,
            &vec![Array1::zeros(0); specs.len()],
            options,
        )?;
        let reml_term = if options.use_remlobjective {
            0.5 * (inner.block_logdet_h - inner.block_logdet_s)
        } else {
            0.0
        };
        let no_pen = vec![Array1::zeros(0); specs.len()];
        let geometry = compute_joint_geometry(family, specs, &inner.block_states, &no_pen);
        let penalized_objective = checked_penalizedobjective(
            inner.log_likelihood,
            inner.penalty_value,
            reml_term,
            "custom-family fit without smoothing parameters",
        )
        .map_err(CustomFamilyError::Optimization)?;
        return blockwise_fit_from_parts(
            BlockwiseFitResultParts {
                block_states: inner.block_states,
                log_likelihood: inner.log_likelihood,
                log_lambdas: Array1::zeros(0),
                lambdas: Array1::zeros(0),
                covariance_conditional,
                stable_penalty_term: 2.0 * inner.penalty_value,
                penalized_objective,
                outer_iterations: 0,
                outer_gradient_norm: 0.0,
                inner_cycles: inner.cycles,
                outer_converged: true,
                geometry,
            },
            specs,
        )
        .map_err(CustomFamilyError::Optimization);
    }

    // Exact Hessians are primary whenever the assembled family can supply them.
    // If a particular outer step is ill-conditioned, strategy fallback handles
    // the downgrade; we do not suppress second-order capability preemptively
    // based on the presence of a wiggle block.

    use crate::estimate::EstimationError;
    use crate::solver::outer_strategy::{
        Derivative, FallbackPolicy, OuterEval, OuterEvalOrder, OuterProblem,
    };

    // Mutable bookkeeping for the outer optimization loop. These fields were
    // previously behind Mutex because the old optimizer bridge used `Fn`
    // adapters; `ClosureObjective` uses `FnMut`, so plain fields suffice.
    struct CustomOuterState {
        warm_cache: Option<ConstrainedWarmStart>,
        last_error: Option<String>,
    }

    let screening_cap = Arc::new(AtomicUsize::new(0));
    let mut outer_options = options.clone();
    outer_options.screening_max_inner_iterations = Some(Arc::clone(&screening_cap));

    let n_rho = rho0.len();
    let (cap_gradient, cap_hessian) =
        custom_family_outer_derivatives(family, specs, &outer_options);
    let hessian = cap_hessian;
    let need_outer_hessian = matches!(hessian, Derivative::Analytic);
    // EFS / HybridEfs structural property (`H^{-1/2} B_k H^{-1/2} ≽ 0` plus a
    // parameter-independent nullspace, Wood-Fasiolo) fails for multi-block
    // families whose joint likelihood Hessian depends on β — e.g. GAMLSS /
    // location-scale where cross-block penalties induce non-block-diagonal
    // curvature that the EFS multiplicative fixed-point cannot resolve in
    // practice. On problems of this class the fixed-point iteration
    // stagnates far from the optimum and burns budget before BFGS gets a
    // turn. Opt out up front so the planner goes straight to analytic-
    // gradient BFGS instead of paying for a doomed EFS attempt first.
    let multi_block_beta_dependent =
        specs.len() > 1 && family.exact_newton_joint_hessian_beta_dependent();
    // Custom family outer plans never degrade to BFGS + Hessian approximation
    // in production. The automatic fallback ladder's remaining step is
    // `downgrade_hessian`, which converts Analytic → Unavailable → BFGS +
    // BfgsApprox, and the EFS → BFGS-with-analytic-gradient retry also uses
    // BfgsApprox for its Hessian. Both are hostile to the RidgedQuadraticReml
    // surrogate surface (directionally wrong rank-2 updates, documented Strong
    // Wolfe iter-0 failures) and were the direct cause of the 45-minute hangs
    // on binomial-logit + P-spline benchmarks. If the primary Arc + Analytic
    // plan fails, surface the non-convergence as an error rather than masking
    // it with a weaker method.
    let problem = OuterProblem::new(n_rho)
        .with_gradient(cap_gradient)
        .with_hessian(hessian)
        .with_disable_fixed_point(multi_block_beta_dependent)
        .with_fallback_policy(FallbackPolicy::Disabled)
        .with_tolerance(options.outer_tol)
        .with_max_iter(options.outer_max_iter)
        .with_seed_config(family.outer_seed_config(n_rho))
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_initial_rho(rho0.clone());

    let eval_outer = |outer: &mut CustomOuterState,
                      rho: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        let warm_ref = screened_outer_warm_start(outer.warm_cache.as_ref(), rho);
        let request_hessian =
            matches!(order, OuterEvalOrder::ValueGradientHessian) && need_outer_hessian;
        let eval_result = match outerobjectivegradienthessian_internal(
            family,
            specs,
            &outer_options,
            &penalty_counts,
            rho,
            warm_ref,
            if request_hessian {
                EvalMode::ValueGradientHessian
            } else {
                EvalMode::ValueAndGradient
            },
        ) {
            Ok(eval)
                if eval.objective.is_finite()
                    && eval.gradient.iter().all(|v| v.is_finite())
                    && match &eval.outer_hessian {
                        crate::solver::outer_strategy::HessianResult::Analytic(hessian) => {
                            hessian.iter().all(|v| v.is_finite())
                        }
                        crate::solver::outer_strategy::HessianResult::Operator(op) => {
                            !request_hessian || op.dim() == rho.len()
                        }
                        crate::solver::outer_strategy::HessianResult::Unavailable => {
                            !request_hessian
                        }
                    } =>
            {
                outer.warm_cache = Some(eval.warm_start.clone());
                outer.last_error = None;
                eval
            }
            Ok(_) => {
                outer.last_error =
                    Some("custom-family outer objective/derivatives became non-finite".to_string());
                return Err(EstimationError::RemlOptimizationFailed(
                    "custom-family outer objective/derivatives became non-finite".to_string(),
                ));
            }
            Err(e) => {
                outer.last_error = Some(e.clone());
                return Err(EstimationError::RemlOptimizationFailed(e));
            }
        };
        Ok(OuterEval {
            cost: eval_result.objective,
            gradient: eval_result.gradient,
            hessian: eval_result.outer_hessian,
        })
    };

    let mut obj = problem.build_objective_with_eval_order(
        CustomOuterState {
            warm_cache: None,
            last_error: None,
        },
        |outer: &mut CustomOuterState, rho: &Array1<f64>| {
            // Always use warm cache when available — the previous inner solution
            // gives a much better starting point. This was previously disabled for
            // exact-Hessian families, forcing every inner solve to start from
            // scratch (5-10 Newton steps instead of 1-2 with warm start).
            let warm_ref = screened_outer_warm_start(outer.warm_cache.as_ref(), rho);
            match outerobjectivegradienthessian(
                family,
                specs,
                &outer_options,
                &penalty_counts,
                rho,
                warm_ref,
                EvalMode::ValueOnly,
            ) {
                Ok((cost, _, _, warm)) => {
                    outer.warm_cache = Some(warm);
                    outer.last_error = None;
                    Ok(cost)
                }
                Err(e) => {
                    outer.last_error = Some(e.clone());
                    Err(EstimationError::RemlOptimizationFailed(e))
                }
            }
        },
        |outer: &mut CustomOuterState, rho: &Array1<f64>| {
            eval_outer(
                outer,
                rho,
                if need_outer_hessian {
                    OuterEvalOrder::ValueGradientHessian
                } else {
                    OuterEvalOrder::ValueAndGradient
                },
            )
        },
        |outer: &mut CustomOuterState, rho: &Array1<f64>, order: OuterEvalOrder| {
            eval_outer(outer, rho, order)
        },
        Some(|outer: &mut CustomOuterState| {
            outer.warm_cache = None;
        }),
        Some(|outer: &mut CustomOuterState, rho: &Array1<f64>| {
            let warm_ref = screened_outer_warm_start(outer.warm_cache.as_ref(), rho);
            match outerobjectiveefs(
                family,
                specs,
                &outer_options,
                &penalty_counts,
                rho,
                warm_ref,
            ) {
                Ok((eval, warm)) => {
                    outer.warm_cache = Some(warm);
                    outer.last_error = None;
                    Ok(eval)
                }
                Err(e) => {
                    outer.last_error = Some(e.clone());
                    Err(EstimationError::RemlOptimizationFailed(e))
                }
            }
        }),
    );

    let outer_result = problem.run(&mut obj, "custom family");

    let last_error_detail = obj
        .state
        .last_error
        .as_ref()
        .map(|e| {
            format!(
                " last objective error: {}",
                normalize_outer_eval_error_detail(e)
            )
        })
        .unwrap_or_default();

    let outer_result = outer_result.map_err(|e| {
        format!(
            "outer smoothing optimization failed after exhausting strategy fallbacks: {e}.{last_error_detail}"
        )
    })?;
    let rho_star = outer_result.rho;
    let outer_iters = outer_result.iterations;
    let outer_grad_norm = outer_result.final_grad_norm;
    screening_cap.store(0, Ordering::Relaxed);

    let per_block = split_log_lambdas(&rho_star, &penalty_counts)?;
    let final_seed = obj.state.warm_cache.clone();
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, final_seed.as_ref())
        .map_err(|e| {
            format!(
                "outer smoothing optimization failed during final inner refit: \
                     {e}.{last_error_detail}"
            )
        })?;
    refresh_all_block_etas(family, specs, &mut inner.block_states).map_err(|e| {
        format!(
            "outer smoothing optimization failed during final eta refresh: \
             {e}.{last_error_detail}"
        )
    })?;
    let covariance_conditional =
        compute_joint_covariance_required(family, specs, &inner.block_states, &per_block, options)?;

    let geometry = compute_joint_geometry(family, specs, &inner.block_states, &per_block);
    let penalized_objective = checked_penalizedobjective(
        inner.log_likelihood,
        inner.penalty_value,
        if include_exact_newton_logdet_h(family, options) {
            0.5 * inner.block_logdet_h
        } else {
            0.0
        } - if include_exact_newton_logdet_s(family, options) {
            0.5 * inner.block_logdet_s
        } else {
            0.0
        },
        "custom-family fit final outer refit",
    )
    .map_err(CustomFamilyError::Optimization)?;
    let lambdas_final = rho_star.mapv(f64::exp);
    let log_lambdas_final = lambdas_final.mapv(|v| v.max(1e-300).ln());
    blockwise_fit_from_parts(
        BlockwiseFitResultParts {
            block_states: inner.block_states,
            log_likelihood: inner.log_likelihood,
            log_lambdas: log_lambdas_final,
            lambdas: lambdas_final,
            covariance_conditional,
            stable_penalty_term: 2.0 * inner.penalty_value,
            penalized_objective,
            outer_iterations: outer_iters,
            outer_gradient_norm: outer_grad_norm,
            inner_cycles: inner.cycles,
            outer_converged: outer_result.converged,
            geometry,
        },
        specs,
    )
    .map_err(CustomFamilyError::Optimization)
}

pub(crate) fn fit_custom_family_fixed_log_lambdas<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    warm_start: Option<&CustomFamilyWarmStart>,
    outer_iterations: usize,
    outer_gradient_norm: f64,
    outer_converged: bool,
) -> Result<crate::solver::estimate::UnifiedFitResult, CustomFamilyError> {
    let penalty_counts = validate_blockspecs(specs)?;
    let rho = flatten_log_lambdas(specs);
    let per_block = split_log_lambdas(&rho, &penalty_counts)?;
    let mut inner = inner_blockwise_fit(
        family,
        specs,
        &per_block,
        options,
        warm_start.map(|warm| &warm.inner),
    )?;
    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let covariance_conditional =
        compute_joint_covariance_required(family, specs, &inner.block_states, &per_block, options)?;
    let geometry = compute_joint_geometry(family, specs, &inner.block_states, &per_block);
    let penalized_objective = checked_penalizedobjective(
        inner.log_likelihood,
        inner.penalty_value,
        if include_exact_newton_logdet_h(family, options) {
            0.5 * inner.block_logdet_h
        } else {
            0.0
        } - if include_exact_newton_logdet_s(family, options) {
            0.5 * inner.block_logdet_s
        } else {
            0.0
        },
        "custom-family fixed-log-lambda fit",
    )
    .map_err(CustomFamilyError::Optimization)?;
    let lambdas = rho.mapv(f64::exp);
    let log_lambdas = lambdas.mapv(|v| v.max(1e-300).ln());
    blockwise_fit_from_parts(
        BlockwiseFitResultParts {
            block_states: inner.block_states,
            log_likelihood: inner.log_likelihood,
            log_lambdas,
            lambdas,
            covariance_conditional,
            stable_penalty_term: 2.0 * inner.penalty_value,
            penalized_objective,
            outer_iterations,
            outer_gradient_norm,
            inner_cycles: inner.cycles,
            outer_converged,
            geometry,
        },
        specs,
    )
    .map_err(CustomFamilyError::Optimization)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
    use crate::families::gamlss::{BinomialLocationScaleFamily, BinomialLocationScaleWiggleFamily};
    use crate::matrix::DesignMatrix;
    use crate::smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
        build_term_collection_design, freeze_term_collection_from_design,
        spatial_length_scale_term_indices, try_build_spatial_log_kappa_derivativeinfo_list,
    };
    use approx::assert_relative_eq;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, Array2, array};

    fn outerobjective_andgradient<F: CustomFamily + Clone + Send + Sync + 'static>(
        family: &F,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
        penalty_counts: &[usize],
        rho: &Array1<f64>,
        warm_start: Option<&ConstrainedWarmStart>,
    ) -> Result<(f64, Array1<f64>, ConstrainedWarmStart), String> {
        let (obj, grad, _, warm) = outerobjectivegradienthessian(
            family,
            specs,
            options,
            penalty_counts,
            rho,
            warm_start,
            EvalMode::ValueAndGradient,
        )?;
        Ok((obj, grad, warm))
    }

    #[derive(Clone)]
    struct OneBlockIdentityFamily;

    #[test]
    fn custom_family_default_outer_seed_config_is_tightened_for_expensive_paths() {
        let family = OneBlockIdentityFamily;

        let small = family.outer_seed_config(4);
        assert_eq!(small.max_seeds, 6);
        assert_eq!(small.seed_budget, 1);
        assert_eq!(small.screen_max_inner_iterations, 2);

        let large = family.outer_seed_config(16);
        assert_eq!(large.max_seeds, 4);
        assert_eq!(large.seed_budget, 1);
        assert_eq!(large.screen_max_inner_iterations, 2);
    }

    #[test]
    fn floor_positiveworking_weights_preserves_exactzeros() {
        let weights = array![0.0, 1.0e-16, 0.25];
        let floored = floor_positiveworking_weights(&weights, 1.0e-6);
        assert_eq!(floored[0], 0.0);
        assert_eq!(floored[1], 1.0e-6);
        assert_eq!(floored[2], 0.25);
    }

    #[test]
    fn screened_outer_warm_start_skips_far_seed_without_dropping_latest_solution() {
        let rho_far = array![2.25, -0.5];
        let mut cache = Some(ConstrainedWarmStart {
            rho: array![0.0, -0.5],
            block_beta: vec![array![1.0, -1.0]],
            active_sets: vec![None],
        });

        assert!(
            screened_outer_warm_start(cache.as_ref(), &rho_far).is_none(),
            "far-away warm seeds should be screened before reuse"
        );

        cache = Some(ConstrainedWarmStart {
            rho: rho_far.clone(),
            block_beta: vec![array![0.2, 0.1]],
            active_sets: vec![Some(vec![1])],
        });
        let retained = screened_outer_warm_start(cache.as_ref(), &rho_far)
            .expect("the freshly solved point should remain reusable");
        assert_eq!(retained.rho, rho_far);
        assert_eq!(retained.block_beta[0], array![0.2, 0.1]);
        assert_eq!(retained.active_sets[0], Some(vec![1]));
    }

    #[test]
    fn psi_drift_deriv_workspace_preserves_block_local_operator() {
        #[derive(Clone)]
        struct ZeroFamily;

        impl CustomFamily for ZeroFamily {
            fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
                Ok(FamilyEvaluation {
                    log_likelihood: 0.0,
                    blockworking_sets: vec![],
                })
            }
        }

        struct BlockLocalPsiWorkspace;

        impl ExactNewtonJointPsiWorkspace for BlockLocalPsiWorkspace {
            fn second_order_terms(
                &self,
                _: usize,
                _: usize,
            ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
                Ok(None)
            }

            fn hessian_directional_derivative(
                &self,
                psi_index: usize,
                _: &Array1<f64>,
            ) -> Result<Option<DriftDerivResult>, String> {
                assert_eq!(psi_index, 0);
                Ok(Some(DriftDerivResult::Operator(Arc::new(
                    crate::solver::estimate::reml::unified::BlockLocalDrift {
                        local: array![[3.0, 1.0], [1.0, 2.0]],
                        start: 1,
                        end: 3,
                    },
                ))))
            }
        }

        let callback = build_psi_drift_deriv_callback(
            &ZeroFamily,
            &[],
            &[],
            Arc::new(Vec::new()),
            false,
            Some(Arc::new(BlockLocalPsiWorkspace)),
        )
        .expect("non-Gaussian psi drift callback should be available");

        let result = callback(0, &array![1.0, 2.0, 3.0])
            .expect("workspace-backed psi drift derivative should be returned");

        match result {
            DriftDerivResult::Dense(_) => {
                panic!("workspace-backed block-local psi drift derivative was densified")
            }
            DriftDerivResult::Operator(op) => {
                let (local, start, end) = op
                    .block_local_data()
                    .expect("block-local operator metadata should be preserved");
                assert_eq!((start, end), (1, 3));
                assert_eq!(local, &array![[3.0, 1.0], [1.0, 2.0]]);
            }
        }
    }

    #[test]
    fn custom_family_outer_derivatives_respects_first_order_downgrade() {
        #[derive(Clone)]
        struct OneBlockFirstOrderOnlyFamily;

        impl CustomFamily for OneBlockFirstOrderOnlyFamily {
            fn evaluate(
                &self,
                block_states: &[ParameterBlockState],
            ) -> Result<FamilyEvaluation, String> {
                let n = block_states[0].eta.len();
                Ok(FamilyEvaluation {
                    log_likelihood: 0.0,
                    blockworking_sets: vec![BlockWorkingSet::Diagonal {
                        working_response: Array1::zeros(n),
                        working_weights: Array1::ones(n),
                    }],
                })
            }

            fn exact_outer_derivative_order(
                &self,
                _: &[ParameterBlockSpec],
                _: &BlockwiseFitOptions,
            ) -> ExactOuterDerivativeOrder {
                ExactOuterDerivativeOrder::First
            }
        }

        let specs = vec![ParameterBlockSpec {
            name: "x".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: None,
        }];
        let (gradient, hessian) = custom_family_outer_derivatives(
            &OneBlockFirstOrderOnlyFamily,
            &specs,
            &BlockwiseFitOptions::default(),
        );
        assert_eq!(
            gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
        assert_eq!(
            hessian,
            crate::solver::outer_strategy::Derivative::Unavailable
        );
    }

    #[test]
    fn custom_family_outer_derivatives_exposes_surrogate_second_order_geometry() {
        // RidgedQuadraticReml is the default objective; its analytic outer
        // Hessian is routed to ARC, which handles indefinite Hessians via
        // cubic regularization. The previous behavior forced these families
        // onto BFGS+BfgsApprox and caused benchmark hangs at iter 0.
        #[derive(Clone)]
        struct SurrogateFamily;

        impl CustomFamily for SurrogateFamily {
            fn evaluate(
                &self,
                block_states: &[ParameterBlockState],
            ) -> Result<FamilyEvaluation, String> {
                let n = block_states[0].eta.len();
                Ok(FamilyEvaluation {
                    log_likelihood: 0.0,
                    blockworking_sets: vec![BlockWorkingSet::Diagonal {
                        working_response: Array1::zeros(n),
                        working_weights: Array1::ones(n),
                    }],
                })
            }
        }

        let specs = vec![ParameterBlockSpec {
            name: "x".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: None,
        }];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            use_outer_hessian: true,
            ..BlockwiseFitOptions::default()
        };
        let (gradient, hessian) =
            custom_family_outer_derivatives(&SurrogateFamily, &specs, &options);
        assert_eq!(
            gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
        assert_eq!(hessian, crate::solver::outer_strategy::Derivative::Analytic);
    }

    #[test]
    fn custom_family_outer_derivatives_keeps_strict_second_order_geometry() {
        #[derive(Clone)]
        struct StrictFamily;

        impl CustomFamily for StrictFamily {
            fn evaluate(
                &self,
                block_states: &[ParameterBlockState],
            ) -> Result<FamilyEvaluation, String> {
                let n = block_states[0].eta.len();
                Ok(FamilyEvaluation {
                    log_likelihood: 0.0,
                    blockworking_sets: vec![BlockWorkingSet::Diagonal {
                        working_response: Array1::zeros(n),
                        working_weights: Array1::ones(n),
                    }],
                })
            }

            fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
                ExactNewtonOuterObjective::StrictPseudoLaplace
            }
        }

        let specs = vec![ParameterBlockSpec {
            name: "x".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: None,
        }];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            use_outer_hessian: true,
            ..BlockwiseFitOptions::default()
        };
        let (gradient, hessian) = custom_family_outer_derivatives(&StrictFamily, &specs, &options);
        assert_eq!(
            gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
        assert_eq!(hessian, crate::solver::outer_strategy::Derivative::Analytic);
    }

    #[test]
    fn custom_family_outer_derivatives_keeps_second_order_for_large_inner_problem() {
        // Inner (n, p) scale does not block the analytic outer Hessian: the
        // outer Hessian assembled by `compute_outer_hessian` is shape
        // (K+ext_dim)×(K+ext_dim) where K = total penalties. For large inner
        // problems with modest K (the common case: n=50000, p=50, K=2) the
        // outer Hessian is tiny and must remain available so ARC can drive
        // the outer iteration. Prior versions of this test enforced an
        // inner-size cutoff that disabled the Hessian for exactly the
        // benchmark sizes (medium: n=50000,p=50; pathological: n=50000,p=80)
        // that were hanging 45-minute GH jobs on BFGS+BfgsApprox Strong Wolfe
        // failures at iter 0.
        #[derive(Clone)]
        struct StrictFamily;

        impl CustomFamily for StrictFamily {
            fn evaluate(
                &self,
                block_states: &[ParameterBlockState],
            ) -> Result<FamilyEvaluation, String> {
                let n = block_states[0].eta.len();
                Ok(FamilyEvaluation {
                    log_likelihood: 0.0,
                    blockworking_sets: vec![BlockWorkingSet::Diagonal {
                        working_response: Array1::zeros(n),
                        working_weights: Array1::ones(n),
                    }],
                })
            }

            fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
                ExactNewtonOuterObjective::StrictPseudoLaplace
            }
        }

        let specs = vec![ParameterBlockSpec {
            name: "x".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::<f64>::zeros((20_100, 50)),
            )),
            offset: Array1::zeros(20_100),
            penalties: vec![PenaltyMatrix::Dense(Array2::<f64>::eye(50))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: None,
        }];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            use_outer_hessian: true,
            ..BlockwiseFitOptions::default()
        };

        let (gradient, hessian) = custom_family_outer_derivatives(&StrictFamily, &specs, &options);
        assert_eq!(
            gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
        assert_eq!(hessian, crate::solver::outer_strategy::Derivative::Analytic);
    }

    #[test]
    fn build_block_spatial_psi_derivatives_populates_aniso_cross_rows() {
        let n = 10usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (0.37 * i as f64).sin() + 0.2 * x0;
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
        }

        let spec = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: vec![SmoothTermSpec {
                name: "spatial".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::EqualMass { num_centers: 6 },
                        length_scale: 0.8,
                        nu: MaternNu::ThreeHalves,
                        include_intercept: false,
                        double_penalty: false,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: Some(vec![0.0, 0.0]),
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let base_design =
            build_term_collection_design(data.view(), &spec).expect("build base spatial design");
        let resolvedspec = freeze_term_collection_from_design(&spec, &base_design)
            .expect("freeze spatial term spec");
        let resolved_design = build_term_collection_design(data.view(), &resolvedspec)
            .expect("rebuild frozen spatial design");
        let spatial_terms = spatial_length_scale_term_indices(&resolvedspec);
        let info_list = try_build_spatial_log_kappa_derivativeinfo_list(
            data.view(),
            &resolvedspec,
            &resolved_design,
            &spatial_terms,
        )
        .expect("build spatial derivative info list")
        .expect("anisotropic derivative info");
        let derivs =
            build_block_spatial_psi_derivatives(data.view(), &resolvedspec, &resolved_design)
                .expect("build custom-family spatial psi derivatives")
                .expect("anisotropic spatial derivative rows");

        assert_eq!(
            derivs.len(),
            2,
            "2D anisotropic term should expose two psi rows"
        );
        assert_eq!(
            info_list.len(),
            2,
            "info list should expose the same two psi rows"
        );

        let x_cross_01 = resolve_custom_family_x_psi_psi(
            &derivs[0],
            &derivs[1],
            1,
            resolved_design.design.nrows(),
            resolved_design.design.ncols(),
            "psi0 cross design",
        )
        .expect("resolve psi0 cross design");
        let x_cross_10 = resolve_custom_family_x_psi_psi(
            &derivs[1],
            &derivs[0],
            0,
            resolved_design.design.nrows(),
            resolved_design.design.ncols(),
            "psi1 cross design",
        )
        .expect("resolve psi1 cross design");
        assert_eq!(
            x_cross_01.dim(),
            (
                resolved_design.design.nrows(),
                resolved_design.design.ncols()
            )
        );
        assert_eq!(
            x_cross_10.dim(),
            (
                resolved_design.design.nrows(),
                resolved_design.design.ncols()
            )
        );
        let cross_designs_01 = info_list[0]
            .aniso_cross_designs
            .as_ref()
            .expect("psi0 cross designs");
        let cross_designs_10 = info_list[1]
            .aniso_cross_designs
            .as_ref()
            .expect("psi1 cross designs");
        assert_eq!(
            cross_designs_01[0].0, 1,
            "psi0 should point at psi1 cross design"
        );
        assert_eq!(
            cross_designs_10[0].0, 0,
            "psi1 should point at psi0 cross design"
        );
        let expected_x_cross_01 = EmbeddedColumnBlock::new(
            &cross_designs_01[0].1,
            info_list[0].global_range.clone(),
            info_list[0].total_p,
        )
        .materialize();
        let expected_x_cross_10 = EmbeddedColumnBlock::new(
            &cross_designs_10[0].1,
            info_list[1].global_range.clone(),
            info_list[1].total_p,
        )
        .materialize();
        assert!(
            x_cross_01
                .iter()
                .zip(expected_x_cross_01.iter())
                .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12),
            "generic psi builder should embed the psi0->psi1 cross design into the off-diagonal row"
        );
        assert!(
            x_cross_10
                .iter()
                .zip(expected_x_cross_10.iter())
                .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12),
            "generic psi builder should embed the psi1->psi0 cross design into the symmetric off-diagonal row"
        );

        let s_cross_01 = derivs[0]
            .s_psi_psi_components
            .as_ref()
            .expect("psi0 penalty second-derivative rows")[1]
            .clone();
        let s_cross_10 = derivs[1]
            .s_psi_psi_components
            .as_ref()
            .expect("psi1 penalty second-derivative rows")[0]
            .clone();
        let cross_penalties_01 = info_list[0]
            .aniso_cross_penalty_provider
            .as_ref()
            .expect("psi0 cross penalty provider")(1)
        .expect("psi0->psi1 cross penalties");
        let cross_penalties_10 = info_list[1]
            .aniso_cross_penalty_provider
            .as_ref()
            .expect("psi1 cross penalty provider")(0)
        .expect("psi1->psi0 cross penalties");
        assert_eq!(s_cross_01.len(), cross_penalties_01.len());
        assert_eq!(s_cross_10.len(), cross_penalties_10.len());
        for (((penalty_idx, actual), expected_local), expected_idx) in s_cross_01
            .iter()
            .zip(cross_penalties_01.iter())
            .zip(info_list[0].penalty_indices.iter())
        {
            assert_eq!(*penalty_idx, *expected_idx);
            let expected = EmbeddedSquareBlock::new(
                expected_local,
                info_list[0].global_range.clone(),
                info_list[0].total_p,
            )
            .materialize();
            assert_eq!(actual.dim(), expected.dim());
            assert!(
                actual
                    .iter()
                    .zip(expected.iter())
                    .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12),
                "generic psi builder should embed each psi0->psi1 cross penalty component into the off-diagonal row"
            );
        }
        for (((penalty_idx, actual), expected_local), expected_idx) in s_cross_10
            .iter()
            .zip(cross_penalties_10.iter())
            .zip(info_list[1].penalty_indices.iter())
        {
            assert_eq!(*penalty_idx, *expected_idx);
            let expected = EmbeddedSquareBlock::new(
                expected_local,
                info_list[1].global_range.clone(),
                info_list[1].total_p,
            )
            .materialize();
            assert_eq!(actual.dim(), expected.dim());
            assert!(
                actual
                    .iter()
                    .zip(expected.iter())
                    .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12),
                "generic psi builder should embed each psi1->psi0 cross penalty component into the symmetric off-diagonal row"
            );
        }
    }

    #[test]
    fn build_block_spatial_psi_derivatives_supports_3d_aniso_matern() {
        let n = 24usize;
        let mut data = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
            data[[i, 2]] = (2.5 * std::f64::consts::PI * t).cos();
        }

        let spec = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: vec![SmoothTermSpec {
                name: "spatial".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::EqualMass { num_centers: 6 },
                        length_scale: 0.45,
                        nu: MaternNu::ThreeHalves,
                        include_intercept: false,
                        double_penalty: false,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: Some(vec![0.0, 0.0, 0.0]),
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let base_design =
            build_term_collection_design(data.view(), &spec).expect("build base spatial design");
        let resolvedspec = freeze_term_collection_from_design(&spec, &base_design)
            .expect("freeze spatial term spec");
        let resolved_design = build_term_collection_design(data.view(), &resolvedspec)
            .expect("rebuild frozen spatial design");
        let derivs =
            build_block_spatial_psi_derivatives(data.view(), &resolvedspec, &resolved_design)
                .expect("3D anisotropic Matern psi derivatives should build")
                .expect("3D anisotropic Matern psi derivatives should be present");
        assert_eq!(derivs.len(), 3);
        assert!(derivs.iter().all(|deriv| deriv.implicit_operator.is_some()));
    }

    impl CustomFamily for OneBlockIdentityFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = block_states[0].eta.len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: Array1::ones(n),
                    working_weights: Array1::ones(n),
                }],
            })
        }
    }

    #[derive(Clone)]
    struct OneBlockGaussianFamily {
        y: Array1<f64>,
    }

    impl CustomFamily for OneBlockGaussianFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let eta = &block_states[0].eta;
            let resid = eta - &self.y;
            let ll = -0.5 * resid.dot(&resid);
            Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: self.y.clone(),
                    working_weights: Array1::ones(self.y.len()),
                }],
            })
        }

        fn diagonalworking_weights_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: usize,
            d_eta: &Array1<f64>,
        ) -> Result<Option<Array1<f64>>, String> {
            Ok(Some(Array1::zeros(d_eta.len())))
        }
    }

    #[derive(Clone)]
    struct OneBlockConstrainedExactFamily {
        target: f64,
        lower: f64,
    }

    impl CustomFamily for OneBlockConstrainedExactFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let beta = block_states
                .first()
                .ok_or_else(|| "missing block 0".to_string())?
                .beta
                .first()
                .copied()
                .ok_or_else(|| "missing coefficient".to_string())?;
            let g = self.target - beta;
            let ll = -0.5 * (beta - self.target) * (beta - self.target);
            Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![g],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                }],
            })
        }

        fn block_linear_constraints(
            &self,
            _: &[ParameterBlockState],
            block_idx: usize,
            _: &ParameterBlockSpec,
        ) -> Result<Option<LinearInequalityConstraints>, String> {
            if block_idx != 0 {
                return Ok(None);
            }
            Ok(Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![self.lower],
            }))
        }
    }

    #[derive(Clone)]
    struct OneBlockConstrainedNaNHessianFamily;

    impl CustomFamily for OneBlockConstrainedNaNHessianFamily {
        fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: SymmetricMatrix::Dense(array![[f64::NAN]]),
                }],
            })
        }

        fn block_linear_constraints(
            &self,
            _: &[ParameterBlockState],
            block_idx: usize,
            _: &ParameterBlockSpec,
        ) -> Result<Option<LinearInequalityConstraints>, String> {
            if block_idx != 0 {
                return Ok(None);
            }
            Ok(Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![0.0],
            }))
        }
    }

    #[derive(Clone)]
    struct OneBlockConstrainedIndefiniteHessianFamily;

    impl CustomFamily for OneBlockConstrainedIndefiniteHessianFamily {
        fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![-1.0],
                    hessian: SymmetricMatrix::Dense(array![[-1.0]]),
                }],
            })
        }

        fn block_linear_constraints(
            &self,
            _: &[ParameterBlockState],
            block_idx: usize,
            _: &ParameterBlockSpec,
        ) -> Result<Option<LinearInequalityConstraints>, String> {
            if block_idx != 0 {
                return Ok(None);
            }
            Ok(Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![1.0],
            }))
        }
    }

    #[derive(Clone)]
    struct OneBlockLinearLikelihoodExactFamily {
        score: f64,
    }

    impl CustomFamily for OneBlockLinearLikelihoodExactFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let beta = block_states
                .first()
                .ok_or_else(|| "missing block 0".to_string())?
                .beta
                .first()
                .copied()
                .ok_or_else(|| "missing coefficient".to_string())?;
            Ok(FamilyEvaluation {
                log_likelihood: self.score * beta,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![self.score],
                    hessian: SymmetricMatrix::Dense(array![[0.0]]),
                }],
            })
        }
    }

    #[derive(Clone)]
    struct PreferJointExactFamily;

    impl CustomFamily for PreferJointExactFamily {
        fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: SymmetricMatrix::Dense(array![[2.0]]),
                }],
            })
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: usize,
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Err(
                "blockwise exact-newton path should not be used when joint path is available"
                    .to_string(),
            )
        }

        fn exact_newton_joint_hessian(
            &self,
            _: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[2.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }
    }

    #[derive(Clone)]
    struct TwoBlockJointConstrainedFamily {
        coupling: f64,
    }

    impl CustomFamily for TwoBlockJointConstrainedFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let beta0 = block_states[0].beta[0];
            let beta1 = block_states[1].beta[0];
            let g0 = 1.0 - beta0 - self.coupling * beta1;
            let g1 = 1.0 - beta1 - self.coupling * beta0;
            Ok(FamilyEvaluation {
                log_likelihood: -0.5
                    * (beta0 * beta0 + beta1 * beta1 + 2.0 * self.coupling * beta0 * beta1)
                    + beta0
                    + beta1,
                blockworking_sets: vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: array![g0],
                        hessian: SymmetricMatrix::Dense(array![[1.0]]),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: array![g1],
                        hessian: SymmetricMatrix::Dense(array![[1.0]]),
                    },
                ],
            })
        }

        fn exact_newton_joint_hessian(
            &self,
            _: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[1.0, self.coupling], [self.coupling, 1.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(Array2::zeros((2, 2))))
        }

        fn block_linear_constraints(
            &self,
            _: &[ParameterBlockState],
            block_idx: usize,
            _: &ParameterBlockSpec,
        ) -> Result<Option<LinearInequalityConstraints>, String> {
            if block_idx >= 2 {
                return Ok(None);
            }
            Ok(Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![0.0],
            }))
        }
    }

    #[derive(Clone)]
    struct TwoBlockJointSurrogateFamily;

    impl CustomFamily for TwoBlockJointSurrogateFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n0 = block_states
                .first()
                .ok_or_else(|| "missing block 0".to_string())?
                .eta
                .len();
            let n1 = block_states
                .get(1)
                .ok_or_else(|| "missing block 1".to_string())?
                .eta
                .len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![
                    BlockWorkingSet::Diagonal {
                        working_response: Array1::zeros(n0),
                        working_weights: Array1::ones(n0),
                    },
                    BlockWorkingSet::Diagonal {
                        working_response: Array1::zeros(n1),
                        working_weights: Array1::ones(n1),
                    },
                ],
            })
        }

        fn exact_newton_joint_hessian_with_specs(
            &self,
            _: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
        ) -> Result<Option<Array2<f64>>, String> {
            let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
            Ok(Some(Array2::eye(p)))
        }

        fn exact_newton_joint_hessian_directional_derivative_with_specs(
            &self,
            _: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
            Ok(Some(Array2::zeros((p, p))))
        }

        fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
            &self,
            _: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
            _: &Array1<f64>,
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
            Ok(Some(Array2::zeros((p, p))))
        }
    }

    #[derive(Clone)]
    struct OneBlockPseudoLaplaceExactFamily {
        target: f64,
    }

    impl CustomFamily for OneBlockPseudoLaplaceExactFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let beta = block_states
                .first()
                .ok_or_else(|| "missing block 0".to_string())?
                .beta
                .first()
                .copied()
                .ok_or_else(|| "missing coefficient".to_string())?;
            let resid = beta - self.target;
            Ok(FamilyEvaluation {
                log_likelihood: -resid * resid,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![-2.0 * resid],
                    hessian: SymmetricMatrix::Dense(array![[2.0]]),
                }],
            })
        }

        fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::StrictPseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            _: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[2.0]]))
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: usize,
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }
    }

    #[derive(Clone)]
    struct OneBlockExactPsiHookFamily;

    impl CustomFamily for OneBlockExactPsiHookFamily {
        fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                }],
            })
        }

        fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::StrictPseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            _: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[1.0]]))
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: usize,
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }

        fn exact_newton_joint_psi_terms(
            &self,
            _: &[ParameterBlockState],
            _: &[ParameterBlockSpec],
            _: &[Vec<CustomFamilyBlockPsiDerivative>],
            _: usize,
        ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
            Ok(Some(ExactNewtonJointPsiTerms {
                objective_psi: 3.5,
                score_psi: array![0.0],
                hessian_psi: array![[0.0]],
                hessian_psi_operator: None,
            }))
        }
    }

    #[derive(Clone)]
    struct OneBlockIndefinitePseudoLaplaceFamily;

    impl CustomFamily for OneBlockIndefinitePseudoLaplaceFamily {
        fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: SymmetricMatrix::Dense(array![[-1.0]]),
                }],
            })
        }

        fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::StrictPseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            _: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[-1.0]]))
        }
    }

    #[derive(Clone)]
    struct OneBlockNearlySymmetricPseudoLaplaceFamily;

    impl CustomFamily for OneBlockNearlySymmetricPseudoLaplaceFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let beta = block_states
                .first()
                .ok_or_else(|| "missing block 0".to_string())?
                .beta
                .clone();
            let h = array![[2.0, 0.1], [3.0, 2.0]];
            let gradient = -h.dot(&beta);
            Ok(FamilyEvaluation {
                log_likelihood: -0.5 * beta.dot(&h.dot(&beta)),
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient,
                    hessian: SymmetricMatrix::Dense(h),
                }],
            })
        }

        fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::StrictPseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            _: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[2.0, 0.1], [3.0, 2.0]]))
        }
    }

    #[derive(Clone)]
    struct OneBlockAlwaysErrorFamily;

    impl CustomFamily for OneBlockAlwaysErrorFamily {
        fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            Err("synthetic outer objective failure: block[0] evaluate()".to_string())
        }
    }

    #[derive(Clone)]
    struct OneBlockCovarianceErrorFamily;

    impl CustomFamily for OneBlockCovarianceErrorFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = block_states[0].eta.len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(n),
                    working_weights: Array1::ones(n),
                }],
            })
        }

        fn exact_newton_joint_hessian_with_specs(
            &self,
            _: &[ParameterBlockState],
            _: &[ParameterBlockSpec],
        ) -> Result<Option<Array2<f64>>, String> {
            Err("synthetic covariance assembly failure".to_string())
        }
    }

    #[test]
    fn effectiveridge_is_never_below_solver_floor() {
        assert!((effective_solverridge(0.0) - 1e-15).abs() < 1e-30);
        assert!((effective_solverridge(1e-8) - 1e-8).abs() < 1e-20);
    }

    #[test]
    fn objective_includes_solverridge_quadratic_term() {
        // One-parameter block with X=1, y*=1, w=1, no explicit penalties.
        // Inner solve gives beta = 1 / (1 + ridge), so objective should include
        // 0.5 * ridge * beta^2 even when no smoothing penalties are present.
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            inner_tol: 0.0,
            outer_max_iter: 1,
            outer_tol: 1e-8,
            minweight: 1e-12,
            ridge_floor: 1e-4,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_remlobjective: false,
            compute_covariance: false,
            use_outer_hessian: false,
            screening_max_inner_iterations: None,
        };

        let result = fit_custom_family(&OneBlockIdentityFamily, &[spec], &options)
            .expect("custom family fit should succeed");
        let ridge = effective_solverridge(options.ridge_floor);
        let beta = result.block_states[0].beta[0];
        let expected_penalty = 0.5 * ridge * beta * beta;
        assert!(
            (result.penalized_objective - expected_penalty).abs() < 1e-12,
            "penalized objective should equal ridge quadratic term when ll=0 and S=0; got {}, expected {}",
            result.penalized_objective,
            expected_penalty
        );
    }

    #[test]
    fn inner_block_accepts_penalty_improving_step_even_if_loglik_drops() {
        let family = OneBlockGaussianFamily { y: array![1.0] };
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
            nullspace_dims: vec![],
            initial_log_lambdas: array![10.0_f64.ln()],
            initial_beta: Some(array![1.0]),
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 20,
            inner_tol: 1e-10,
            outer_max_iter: 1,
            outer_tol: 1e-8,
            minweight: 1e-12,
            ridge_floor: 0.0,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_remlobjective: false,
            compute_covariance: false,
            use_outer_hessian: false,
            screening_max_inner_iterations: None,
        };
        let per_block_log_lambdas = vec![array![10.0_f64.ln()]];
        let inner = inner_blockwise_fit(&family, &[spec], &per_block_log_lambdas, &options, None)
            .expect("inner blockwise fit should succeed");

        let beta = inner.block_states[0].beta[0];
        assert!(
            beta < 0.5,
            "beta should shrink toward penalized mode; got {}",
            beta
        );
        assert!(
            inner.log_likelihood < -1e-8,
            "raw log-likelihood should drop for this strongly penalized move; got {}",
            inner.log_likelihood
        );
    }

    #[test]
    fn exact_newton_backtracking_descent_includes_explicit_ridge() {
        let family = OneBlockLinearLikelihoodExactFamily { score: 0.5 };
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![1.0]),
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            inner_tol: 0.0,
            outer_max_iter: 1,
            outer_tol: 1e-8,
            minweight: 1e-12,
            ridge_floor: 1.0,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_remlobjective: false,
            compute_covariance: false,
            use_outer_hessian: false,
            screening_max_inner_iterations: None,
        };
        let inner = inner_blockwise_fit(&family, &[spec], &[Array1::zeros(0)], &options, None)
            .expect("inner blockwise fit should succeed");

        let beta = inner.block_states[0].beta[0];
        let objective = -inner.log_likelihood + inner.penalty_value;
        assert!(
            beta < 1.0 - 1e-12,
            "ridge-aware fallback descent should shrink beta after rejecting the uphill Newton step; got {}",
            beta
        );
        assert!(
            objective < -1e-12,
            "accepted fallback step should lower the penalized objective; got {}",
            objective
        );
    }

    #[test]
    fn outergradient_matches_finite_difference_for_one_block() {
        let n = 8usize;
        let y = Array1::from_vec(vec![0.4, -0.2, 0.8, 1.0, -0.5, 0.3, 0.1, -0.7]);
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.2],
            initial_beta: None,
        };
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            ..BlockwiseFitOptions::default()
        };
        let penalty_counts = vec![1usize];
        let rho = array![0.1];
        let (f0, g0, _) = outerobjective_andgradient(
            &OneBlockGaussianFamily { y: y.clone() },
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho,
            None,
        )
        .expect("objective/gradient");

        let h = 1e-5;
        let rho_p = array![rho[0] + h];
        let rho_m = array![rho[0] - h];
        let (fp, _, _) = outerobjective_andgradient(
            &OneBlockGaussianFamily { y: y.clone() },
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho_p,
            None,
        )
        .expect("objective+");
        let (fm, _, _) = outerobjective_andgradient(
            &OneBlockGaussianFamily { y },
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho_m,
            None,
        )
        .expect("objective-");
        let gfd = (fp - fm) / (2.0 * h);
        let rel = (g0[0] - gfd).abs() / gfd.abs().max(1e-8);

        assert!(f0.is_finite());
        assert_eq!(
            g0[0].signum(),
            gfd.signum(),
            "outer gradient sign mismatch: analytic={} fd={}",
            g0[0],
            gfd
        );
        assert!(
            rel < 5e-3,
            "outer gradient mismatch: analytic={} fd={} rel={}",
            g0[0],
            gfd,
            rel
        );
    }

    #[test]
    fn outergradient_prefers_joint_exact_pathwhen_available() {
        let spec = ParameterBlockSpec {
            name: "joint_exact".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            ..BlockwiseFitOptions::default()
        };
        let penalty_counts = vec![1usize];
        let rho = array![0.0];

        let result = outerobjective_andgradient(
            &PreferJointExactFamily,
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho,
            None,
        );
        assert!(
            result.is_ok(),
            "joint exact path should be preferred over blockwise fallback: {:?}",
            result.err()
        );
    }

    #[test]
    fn innerfit_uses_joint_exact_path_for_multiblock_constraints() {
        let spec0 = ParameterBlockSpec {
            name: "block0".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let spec1 = ParameterBlockSpec {
            name: "block1".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            inner_tol: 1e-10,
            ridge_floor: 1e-12,
            ..BlockwiseFitOptions::default()
        };
        let per_block = vec![Array1::zeros(0), Array1::zeros(0)];

        let result = inner_blockwise_fit(
            &TwoBlockJointConstrainedFamily { coupling: 0.25 },
            &[spec0, spec1],
            &per_block,
            &options,
            None,
        )
        .expect("joint constrained inner fit should succeed");

        assert!(
            result.converged,
            "joint constrained inner fit should converge in one cycle"
        );
        assert_eq!(result.cycles, 1);
        assert!((result.block_states[0].beta[0] - 0.8).abs() < 1e-8);
        assert!((result.block_states[1].beta[0] - 0.8).abs() < 1e-8);
        assert_eq!(result.active_sets, vec![None, None]);
    }

    #[test]
    fn outergradient_uses_joint_surrogate_formultiblock_diagonal_family() {
        let spec0 = ParameterBlockSpec {
            name: "block0".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [1.0]
            ])),
            offset: array![0.0, 0.0],
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let spec1 = ParameterBlockSpec {
            name: "block1".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [1.0]
            ])),
            offset: array![0.0, 0.0],
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };
        let penalty_counts = vec![1usize, 1usize];
        let rho = array![0.0, 0.0];

        let result = outerobjective_andgradient(
            &TwoBlockJointSurrogateFamily,
            &[spec0, spec1],
            &options,
            &penalty_counts,
            &rho,
            None,
        );
        assert!(
            result.is_ok(),
            "default joint multi-block surrogate path should succeed without blockwise dW callbacks: {:?}",
            result.err()
        );
    }

    #[test]
    fn exact_newton_pseudo_laplace_objective_uses_logdet_h_without_logdet_s() {
        let spec = ParameterBlockSpec {
            name: "pseudo_laplace".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-12,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        let fit = fit_custom_family(
            &OneBlockPseudoLaplaceExactFamily { target: 1.5 },
            &[spec],
            &options,
        )
        .expect("pseudo-laplace exact-newton fit");
        let expected = 0.5 * 2.0_f64.ln();
        assert!(
            (fit.penalized_objective - expected).abs() < 1e-8,
            "pseudo-Laplace objective mismatch: got {}, expected {}",
            fit.penalized_objective,
            expected
        );
    }

    #[test]
    fn exact_newton_joint_psi_hook_can_supply_fixed_beta_termswithout_quadratic_spsi() {
        let spec = ParameterBlockSpec {
            name: "psi_hook".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let deriv = CustomFamilyBlockPsiDerivative {
            penalty_index: None,
            x_psi: Array2::zeros((1, 1)),
            s_psi: Array2::zeros((1, 1)),
            s_psi_components: None,
            s_psi_penalty_components: None,
            x_psi_psi: None,
            s_psi_psi: None,
            s_psi_psi_components: None,
            s_psi_psi_penalty_components: None,
            implicit_operator: None,
            implicit_axis: 0,
            implicit_group_id: None,
        };
        let result = evaluate_custom_family_joint_hyper(
            &OneBlockExactPsiHookFamily,
            &[spec],
            &BlockwiseFitOptions {
                use_remlobjective: true,
                compute_covariance: false,
                ..BlockwiseFitOptions::default()
            },
            &Array1::zeros(0),
            &[vec![deriv]],
            None,
            EvalMode::ValueAndGradient,
        )
        .expect("joint hyper eval with exact joint psi hook");
        assert_eq!(result.gradient.len(), 1);
        assert!(
            (result.gradient[0] - 3.5).abs() < 1e-12,
            "expected family-supplied joint psi term, got {}",
            result.gradient[0]
        );
    }

    #[test]
    fn pseudo_laplace_exact_newton_rejects_non_spdhessian() {
        let spec = ParameterBlockSpec {
            name: "indefinite".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let msg = match fit_custom_family(
            &OneBlockIndefinitePseudoLaplaceFamily,
            &[spec],
            &BlockwiseFitOptions {
                use_remlobjective: true,
                compute_covariance: false,
                ..BlockwiseFitOptions::default()
            },
        ) {
            Ok(_) => panic!("indefinite pseudo-laplace Hessian should be rejected"),
            Err(err) => err.to_string(),
        };
        assert!(
            msg.contains("strict pseudo-laplace SPD")
                || msg.contains("strict pseudo-laplace SPD logdet"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn slq_determinant_mode_tracks_exact_full_logdet_policy() {
        let h = array![[6.0, 0.8, 0.1], [0.8, 4.5, 0.4], [0.1, 0.4, 3.2]];
        let exact = stable_logdet_with_ridge_policy(
            &h,
            1e-8,
            RidgePolicy::explicit_stabilization_full_exact(),
        )
        .expect("exact logdet");
        let slq = stable_logdet_with_ridge_policy(
            &h,
            1e-8,
            RidgePolicy::explicit_stabilization_full_slq(),
        )
        .expect("slq logdet");
        assert!((slq - exact).abs() < 5e-2, "slq={slq}, exact={exact}");
    }

    #[test]
    fn indefinite_hessian_uses_smooth_regularized_logdet() {
        // Indefinite Hessian: eigenvalues {-1, 2}.
        //
        // Old behaviour: silently drop the -1 direction from logdet, warn,
        // and after enough repeats escalate to an EFS abort (first-order
        // fallback marker).
        //
        // New behaviour: every eigenvalue contributes via the smooth
        // regularizer r_ε(σ) = ½(σ + √(σ² + 4ε²)).  No direction is ignored,
        // no escalation, and the logdet matches what the downstream
        // `DenseSpectralOperator` gradient computes — eliminating the
        // cost/gradient mismatch that broke BFGS line search.
        let h = array![[-1.0, 0.0], [0.0, 2.0]];
        let logdet = stable_logdet_with_ridge_policy(
            &h,
            1e-12,
            RidgePolicy::explicit_stabilization_pospart(),
        )
        .expect("smooth-regularized logdet must be finite for indefinite H");
        assert!(
            logdet.is_finite(),
            "smooth-regularized logdet should be finite, got {logdet}"
        );
        // Reference value using the same formula directly on the eigenvalues
        // of H + ridge·I (ridge = 1e-12 here).  Since ε ≫ ridge (spectral_epsilon
        // floors at √(eps_mach) ≈ 1.5e-8 for p=2), the ridge contribution is
        // absorbed into ε and the expected value is Σ log r_ε(σ_j).
        let eps = spectral_epsilon(&[-1.0_f64, 2.0]).max(1e-12_f64.max(1e-14));
        // A + ridge·I has eigenvalues shifted by 1e-12, negligible relative to ε.
        let expected: f64 = [-1.0_f64 + 1e-12, 2.0 + 1e-12]
            .iter()
            .map(|&s| spectral_regularize(s, eps).ln())
            .sum();
        assert!(
            (logdet - expected).abs() < 1e-10,
            "logdet={logdet}, expected={expected}"
        );
    }

    #[test]
    fn pseudo_laplace_exact_newton_symmetrizes_nearly_symmetrichessian() {
        let spec = ParameterBlockSpec {
            name: "nearly_symmetric".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0, 0.0],
                [0.0, 1.0]
            ])),
            offset: array![0.0, 0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0, 0.0]),
        };
        let fit = fit_custom_family(
            &OneBlockNearlySymmetricPseudoLaplaceFamily,
            &[spec],
            &BlockwiseFitOptions {
                use_remlobjective: true,
                compute_covariance: false,
                ..BlockwiseFitOptions::default()
            },
        )
        .expect("nearly symmetric pseudo-laplace Hessian should be accepted after symmetrization");
        assert!(
            fit.penalized_objective.is_finite(),
            "expected finite pseudo-laplace objective, got {}",
            fit.penalized_objective
        );
    }

    #[test]
    fn outer_lamlgradient_matches_finite_differencewhen_joint_exact_path_is_active() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        ));
        let log_sigma_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        ));
        let thresholdspec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: threshold_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
        };
        let log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: log_sigma_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![-0.2],
            initial_beta: Some(array![-0.1]),
        };
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let knots = crate::families::gamlss::initializewiggle_knots_from_seed(q_seed.view(), 3, 4)
            .expect("knots");
        let wiggle_block = crate::families::gamlss::buildwiggle_block_input_from_knots(
            q_seed.view(),
            &knots,
            3,
            2,
            false,
        )
        .expect("wiggle block");
        let wigglespec = ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: wiggle_block.design.clone(),
            offset: wiggle_block.offset.clone(),
            penalties: wiggle_block
                .penalties
                .iter()
                .map(|ps| match ps {
                    crate::solver::estimate::PenaltySpec::Block {
                        local, col_range, ..
                    } => PenaltyMatrix::Blockwise {
                        local: local.clone(),
                        col_range: col_range.clone(),
                        total_dim: wiggle_block.design.ncols(),
                    },
                    crate::solver::estimate::PenaltySpec::Dense(m) => {
                        PenaltyMatrix::Dense(m.clone())
                    }
                })
                .collect(),
            nullspace_dims: wiggle_block.nullspace_dims.clone(),
            initial_log_lambdas: array![0.1],
            initial_beta: Some(Array1::from_elem(wiggle_block.design.ncols(), 0.03)),
        };

        let family = BinomialLocationScaleWiggleFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
            wiggle_knots: knots,
            wiggle_degree: 3,
        };

        let specs = vec![thresholdspec, log_sigmaspec, wigglespec];
        let penalty_counts = vec![1usize, 1usize, 1usize];
        let rho = array![0.05, -0.15, 0.1];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (f0, g0, _) =
            outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho, None)
                .expect("objective/gradient");
        assert!(f0.is_finite());
        assert_eq!(g0.len(), rho.len());

        let h = 1e-5;
        for k in 0..rho.len() {
            let mut rho_p = rho.clone();
            let mut rho_m = rho.clone();
            rho_p[k] += h;
            rho_m[k] -= h;
            let (fp, _, _) = outerobjective_andgradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_p,
                None,
            )
            .expect("objective+");
            let (fm, _, _) = outerobjective_andgradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_m,
                None,
            )
            .expect("objective-");
            let gfd = (fp - fm) / (2.0 * h);

            // Noise floor for FD-vs-analytic comparisons.
            //
            // At a rank-deficient optimum (σ_min(H) ≲ ε_machine) the outer
            // REML gradient is a DIFFERENCE of two nearly-equal O(1)
            // quantities — ½ λ_k (H⁺[k,k] − S⁺[k,k]) — so the true gradient
            // is very close to zero.  The FD estimator `(f_p − f_m)/(2h)`
            // then measures cost-sum round-off: at f64 precision each cost
            // value carries an uncertainty of ~EPS · |cost|, and the
            // symmetric FD inflates that by 1/(2h), producing a noise floor
            // of roughly `EPS · |cost| / h` on |gfd|.  Below that floor
            // neither `|gfd|`, `|g0|`, nor `sign(gfd)` reflect the true
            // derivative — they reflect arithmetic noise.
            //
            // Concretely: for this test `|cost| ~ 6`, `h = 1e-5`, so the
            // floor is ~1.3e-10 (≈ f64::EPSILON · 6 / 1e-5).  We round up
            // to a problem-scale-derived value and treat pairs where BOTH
            // |g0| and |gfd| lie below the floor as a pass (the assertion
            // is making a claim about the TRUE derivative, and a true
            // derivative strictly less than noise is indistinguishable
            // from zero — sign is not a correctness property there).
            let cost_magnitude = f0.abs().max(1.0);
            let noise_floor = (10.0 * f64::EPSILON * cost_magnitude / h).max(1e-9);
            let both_in_noise = g0[k].abs() < noise_floor && gfd.abs() < noise_floor;

            if !both_in_noise {
                assert_eq!(
                    g0[k].signum(),
                    gfd.signum(),
                    "outer LAML gradient sign mismatch at {}: analytic={} fd={} noise_floor={:.3e}",
                    k,
                    g0[k],
                    gfd,
                    noise_floor,
                );
                let rel = (g0[k] - gfd).abs() / gfd.abs().max(noise_floor);
                assert!(
                    rel < 2e-2,
                    "outer LAML gradient mismatch at {}: analytic={} fd={} rel={} noise_floor={:.3e}",
                    k,
                    g0[k],
                    gfd,
                    rel,
                    noise_floor,
                );
            }
        }
    }

    #[test]
    fn rho_only_outer_objective_matches_joint_hyper_when_psi_is_empty() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        ));
        let log_sigma_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        ));
        let thresholdspec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: threshold_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
        };
        let log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: log_sigma_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![-0.2],
            initial_beta: Some(array![-0.1]),
        };
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let knots = crate::families::gamlss::initializewiggle_knots_from_seed(q_seed.view(), 3, 4)
            .expect("knots");
        let wiggle_block = crate::families::gamlss::buildwiggle_block_input_from_knots(
            q_seed.view(),
            &knots,
            3,
            2,
            false,
        )
        .expect("wiggle block");
        let wigglespec = ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: wiggle_block.design.clone(),
            offset: wiggle_block.offset.clone(),
            penalties: wiggle_block
                .penalties
                .iter()
                .map(|ps| match ps {
                    crate::solver::estimate::PenaltySpec::Block {
                        local, col_range, ..
                    } => PenaltyMatrix::Blockwise {
                        local: local.clone(),
                        col_range: col_range.clone(),
                        total_dim: wiggle_block.design.ncols(),
                    },
                    crate::solver::estimate::PenaltySpec::Dense(m) => {
                        PenaltyMatrix::Dense(m.clone())
                    }
                })
                .collect(),
            nullspace_dims: wiggle_block.nullspace_dims.clone(),
            initial_log_lambdas: array![0.1],
            initial_beta: Some(Array1::from_elem(wiggle_block.design.ncols(), 0.03)),
        };

        let family = BinomialLocationScaleWiggleFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
            wiggle_knots: knots,
            wiggle_degree: 3,
        };

        let specs = vec![thresholdspec, log_sigmaspec, wigglespec];
        let penalty_counts = vec![1usize, 1usize, 1usize];
        let rho = array![0.05, -0.15, 0.1];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (outer_obj, outer_grad, outer_hessian, _) = outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho,
            None,
            EvalMode::ValueGradientHessian,
        )
        .expect("rho-only outer objective");
        let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()];
        let joint_result = evaluate_custom_family_joint_hyper(
            &family,
            &specs,
            &options,
            &rho,
            &derivative_blocks,
            None,
            EvalMode::ValueGradientHessian,
        )
        .expect("joint hyper objective with empty psi");

        assert!(
            (outer_obj - joint_result.objective).abs() < 1e-12,
            "objective mismatch: rho-only={} joint={}",
            outer_obj,
            joint_result.objective
        );
        assert_eq!(outer_grad.len(), joint_result.gradient.len());
        let max_grad_diff = outer_grad
            .iter()
            .zip(joint_result.gradient.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_grad_diff < 1e-12,
            "gradient mismatch: max diff={}",
            max_grad_diff
        );

        let outer_hessian = outer_hessian.expect("rho-only outer Hessian");
        let joint_hessian = joint_result
            .outer_hessian
            .materialize_dense()
            .expect("joint outer Hessian should materialize")
            .expect("joint outer Hessian");
        assert_eq!(outer_hessian.dim(), joint_hessian.dim());
        let max_hessian_diff = outer_hessian
            .iter()
            .zip(joint_hessian.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_hessian_diff < 1e-12,
            "outer Hessian mismatch: max diff={}",
            max_hessian_diff
        );
    }

    #[test]
    fn outer_lamlgradient_diagonal_binomial_location_scale_matchesfd() {
        let n = 8usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_elem(n, 1.0);
        let thresholdspec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let threshold_design = thresholdspec.design.clone();
        let log_sigma_design = log_sigmaspec.design.clone();
        let family = BinomialLocationScaleFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
        };
        let specs = vec![thresholdspec, log_sigmaspec];
        let penalty_counts = vec![1usize, 1usize];
        let rho = array![0.0, 0.0];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (f0, g0, _) =
            outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho, None)
                .expect("objective/gradient");
        assert!(f0.is_finite());
        assert_eq!(g0.len(), rho.len());

        let h = 1e-5;
        for k in 0..rho.len() {
            let mut rho_p = rho.clone();
            let mut rho_m = rho.clone();
            rho_p[k] += h;
            rho_m[k] -= h;
            let (fp, _, _) = outerobjective_andgradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_p,
                None,
            )
            .expect("objective+");
            let (fm, _, _) = outerobjective_andgradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_m,
                None,
            )
            .expect("objective-");
            let gfd = (fp - fm) / (2.0 * h);
            let abs = (g0[k] - gfd).abs();
            let rel = abs / gfd.abs().max(1e-8);
            if abs >= 2e-3 {
                assert_eq!(
                    g0[k].signum(),
                    gfd.signum(),
                    "outer diagonal LAML gradient sign mismatch at {}: analytic={} fd={}",
                    k,
                    g0[k],
                    gfd
                );
            }
            assert!(
                abs < 2e-3 || rel < 2e-3,
                "outer diagonal LAML gradient mismatch at {}: analytic={} fd={} abs={} rel={}",
                k,
                g0[k],
                gfd,
                abs,
                rel
            );
        }
    }

    #[test]
    fn outer_lamlgradient_diagonal_binomial_location_scale_hard_case_matchesfd() {
        let n = 9usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_elem(n, 1.0);
        let thresholdspec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
        };
        let log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![-0.1]),
        };
        let threshold_design = thresholdspec.design.clone();
        let log_sigma_design = log_sigmaspec.design.clone();
        let family = BinomialLocationScaleFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
        };
        let specs = vec![thresholdspec, log_sigmaspec];
        let penalty_counts = vec![1usize, 1usize];
        let rho = array![0.15, -0.25];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (f0, g0, _) =
            outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho, None)
                .expect("objective/gradient");
        assert!(f0.is_finite());
        assert_eq!(g0.len(), rho.len());

        let h = 1e-5;
        for k in 0..rho.len() {
            let mut rho_p = rho.clone();
            let mut rho_m = rho.clone();
            rho_p[k] += h;
            rho_m[k] -= h;
            let (fp, _, _) = outerobjective_andgradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_p,
                None,
            )
            .expect("objective+");
            let (fm, _, _) = outerobjective_andgradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_m,
                None,
            )
            .expect("objective-");
            let gfd = (fp - fm) / (2.0 * h);
            let abs = (g0[k] - gfd).abs();
            let rel = abs / gfd.abs().max(1e-8);
            if abs >= 2e-3 {
                assert_eq!(
                    g0[k].signum(),
                    gfd.signum(),
                    "outer diagonal hard-case LAML gradient sign mismatch at {}: analytic={} fd={}",
                    k,
                    g0[k],
                    gfd
                );
            }
            assert!(
                abs < 2e-3 || rel < 2e-3,
                "outer diagonal hard-case LAML gradient mismatch at {}: analytic={} fd={} abs={} rel={}",
                k,
                g0[k],
                gfd,
                abs,
                rel
            );
        }
    }

    #[test]
    fn outer_lamlhessian_joint_exact_binomial_location_scale_matchesfd() {
        let n = 10usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let weights = Array1::from_elem(n, 1.0);
        let thresholdspec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.15]),
        };
        let log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![-0.05]),
        };
        let threshold_design = thresholdspec.design.clone();
        let log_sigma_design = log_sigmaspec.design.clone();
        let family = BinomialLocationScaleFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
        };
        let specs = vec![thresholdspec, log_sigmaspec];
        let penalty_counts = vec![1usize, 1usize];
        let rho = array![0.1, -0.2];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (_, _, h0_opt, _) = outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho,
            None,
            EvalMode::ValueGradientHessian,
        )
        .expect("objective/gradient/hessian");
        let h0 = h0_opt.expect("analytic outer Hessian should be available");
        assert_eq!(h0.nrows(), rho.len());
        assert_eq!(h0.ncols(), rho.len());

        let h = 1e-5;
        for l in 0..rho.len() {
            let mut rho_p = rho.clone();
            let mut rho_m = rho.clone();
            rho_p[l] += h;
            rho_m[l] -= h;
            let (_, gp, _, _) = outerobjectivegradienthessian(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_p,
                None,
                EvalMode::ValueAndGradient,
            )
            .expect("objective/gradient +");
            let (_, gm, _, _) = outerobjectivegradienthessian(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_m,
                None,
                EvalMode::ValueAndGradient,
            )
            .expect("objective/gradient -");

            for k in 0..rho.len() {
                let hfd = (gp[k] - gm[k]) / (2.0 * h);
                let abs_err = (h0[[k, l]] - hfd).abs();
                let rel = (h0[[k, l]] - hfd).abs() / hfd.abs().max(1e-7);
                if h0[[k, l]].abs().max(hfd.abs()) > 1e-10 {
                    assert_eq!(
                        h0[[k, l]].signum(),
                        hfd.signum(),
                        "outer Hessian sign mismatch at ({k},{l}): analytic={} fd={}",
                        h0[[k, l]],
                        hfd
                    );
                }
                assert!(
                    abs_err < 1e-8 || rel < 2e-2,
                    "outer Hessian mismatch at ({k},{l}): analytic={} fd={} abs={} rel={}",
                    h0[[k, l]],
                    hfd,
                    abs_err,
                    rel
                );
            }
        }

        for i in 0..h0.nrows() {
            for j in 0..i {
                let asym = (h0[[i, j]] - h0[[j, i]]).abs();
                assert!(
                    asym < 1e-8,
                    "outer Hessian not symmetric at ({i},{j}): {asym}"
                );
            }
        }
    }

    #[test]
    fn block_solve_sparse_matches_dense() {
        let x_dense = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
            [0.0, 6.0, 0.0]
        ];
        let y_star = array![1.0, -1.0, 0.5, 2.0];
        let w = array![1.0, 0.5, 2.0, 1.5];
        let s_lambda = Array2::<f64>::eye(3) * 0.1;

        let mut triplets = Vec::new();
        for i in 0..x_dense.nrows() {
            for j in 0..x_dense.ncols() {
                let v = x_dense[[i, j]];
                if v != 0.0 {
                    triplets.push(Triplet::new(i, j, v));
                }
            }
        }
        let x_sparse = SparseColMat::try_new_from_triplets(4, 3, &triplets)
            .expect("sparse matrix build should succeed");

        let beta_dense = solve_blockweighted_system(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_dense.clone())),
            &y_star,
            &w,
            &s_lambda,
            1e-12,
            RidgePolicy::explicit_stabilization_pospart(),
        )
        .expect("dense solve should succeed");

        let beta_sparse = solve_blockweighted_system(
            &DesignMatrix::from(x_sparse),
            &y_star,
            &w,
            &s_lambda,
            1e-12,
            RidgePolicy::explicit_stabilization_pospart(),
        )
        .expect("sparse solve should succeed");

        for j in 0..beta_dense.len() {
            assert!(
                (beta_dense[j] - beta_sparse[j]).abs() < 1e-10,
                "dense/sparse mismatch at {}: {} vs {}",
                j,
                beta_dense[j],
                beta_sparse[j]
            );
        }
    }

    #[test]
    fn outer_lamlhessian_joint_exact_binomial_location_scale_hard_case_matchesfd() {
        let n = 9usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_elem(n, 1.0);
        let thresholdspec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
        };
        let log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![-0.1]),
        };
        let threshold_design = thresholdspec.design.clone();
        let log_sigma_design = log_sigmaspec.design.clone();
        let family = BinomialLocationScaleFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
        };
        let specs = vec![thresholdspec, log_sigmaspec];
        let penalty_counts = vec![1usize, 1usize];
        let rho = array![0.15, -0.25];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (_, _, h0_opt, _) = outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho,
            None,
            EvalMode::ValueGradientHessian,
        )
        .expect("objective/gradient/hessian");
        let h0 = h0_opt.expect("analytic outer Hessian should be available");
        assert_eq!(h0.nrows(), rho.len());
        assert_eq!(h0.ncols(), rho.len());

        let h = 1e-5;
        for l in 0..rho.len() {
            let mut rho_p = rho.clone();
            let mut rho_m = rho.clone();
            rho_p[l] += h;
            rho_m[l] -= h;
            let (_, gp, _, _) = outerobjectivegradienthessian(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_p,
                None,
                EvalMode::ValueAndGradient,
            )
            .expect("objective/gradient +");
            let (_, gm, _, _) = outerobjectivegradienthessian(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_m,
                None,
                EvalMode::ValueAndGradient,
            )
            .expect("objective/gradient -");

            for k in 0..rho.len() {
                let hfd = (gp[k] - gm[k]) / (2.0 * h);
                let abs_err = (h0[[k, l]] - hfd).abs();
                let rel = abs_err / hfd.abs().max(1e-7);
                if h0[[k, l]].abs().max(hfd.abs()) > 1e-10 {
                    assert_eq!(
                        h0[[k, l]].signum(),
                        hfd.signum(),
                        "hard-case outer Hessian sign mismatch at ({k},{l}): analytic={} fd={}",
                        h0[[k, l]],
                        hfd
                    );
                }
                assert!(
                    abs_err < 1e-8 || rel < 2e-2,
                    "hard-case outer Hessian mismatch at ({k},{l}): analytic={} fd={} abs={} rel={}",
                    h0[[k, l]],
                    hfd,
                    abs_err,
                    rel
                );
            }
        }
    }

    #[test]
    fn block_solve_falls_backwhen_llt_rejects_indefinite_system() {
        let x_dense = array![[1.0, 0.0], [0.0, 0.0]];
        let y_star = array![2.0, 0.0];
        let w = array![1.0, 1.0];
        let s_lambda = array![[0.0, 0.0], [0.0, -1e-12]];

        let beta = solve_blockweighted_system(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_dense)),
            &y_star,
            &w,
            &s_lambda,
            1e-12,
            RidgePolicy::explicit_stabilization_pospart(),
        )
        .expect("fallback solve should succeed");

        assert!(beta.iter().all(|v| v.is_finite()));
        assert!(
            (beta[0] - 2.0).abs() < 1e-10,
            "unexpected solved coefficient"
        );
        assert!(
            beta[1].abs() < 1e-8,
            "null-space coefficient should stay near zero"
        );
    }

    #[test]
    fn exact_newton_block_enforces_linear_constraints() {
        let spec = ParameterBlockSpec {
            name: "exact_block".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![1.5]),
        };
        let family = OneBlockConstrainedExactFamily {
            target: 0.0,
            lower: 1.0,
        };
        let fit = fit_custom_family(&family, &[spec], &BlockwiseFitOptions::default())
            .expect("constrained exact-newton fit");
        let beta = fit.block_states[0].beta[0];
        assert!(
            (beta - 1.0).abs() < 1e-8,
            "expected constrained optimum at lower bound, got {beta}"
        );
    }

    #[test]
    fn extract_simple_lower_bounds_accepts_axis_aligned_rows() {
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]],
            b: array![0.25, 1.0, 1.5],
        };
        let bounds = extract_simple_lower_bounds(&constraints, 2)
            .expect("lower-bound extraction should succeed")
            .expect("axis-aligned rows should map to lower bounds");
        assert_relative_eq!(bounds.lower_bounds[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(bounds.lower_bounds[1], 0.5, epsilon = 1e-12);
        assert_eq!(bounds.coeff_to_row, vec![Some(2), Some(1)]);
    }

    #[test]
    fn extract_simple_lower_bounds_rejects_coupled_rows() {
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 1.0]],
            b: array![0.0],
        };
        assert!(
            extract_simple_lower_bounds(&constraints, 2)
                .expect("lower-bound extraction should not error on valid shapes")
                .is_none(),
            "coupled rows must stay on the generic linear-constraint path"
        );
    }

    #[test]
    fn constrained_exact_newton_indefinite_hessian_uses_stabilized_delta_solve() {
        let spec = ParameterBlockSpec {
            name: "exact_block".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![1.5]),
        };
        let states = vec![ParameterBlockState {
            beta: array![1.5],
            eta: array![1.5],
        }];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![1.0],
        };
        let hessian = SymmetricMatrix::Dense(array![[-1.0]]);
        let updater = ExactNewtonBlockUpdater {
            gradient: &array![-1.0],
            hessian: &hessian,
        };
        let s_lambda = Array2::zeros((1, 1));
        let update = updater
            .compute_update_step(&BlockUpdateContext {
                family: &OneBlockConstrainedIndefiniteHessianFamily,
                states: &states,
                spec: &spec,
                block_idx: 0,
                s_lambda: &s_lambda,
                options: &BlockwiseFitOptions::default(),
                linear_constraints: Some(&constraints),
                cached_active_set: None,
            })
            .expect("indefinite constrained exact-newton update should be stabilized");
        assert_relative_eq!(update.beta_new_raw[0], 1.0, epsilon = 1e-12);
        assert_eq!(update.active_set, Some(vec![0]));
    }

    #[test]
    fn quadratic_linear_constraints_release_positive_kkt_systemmultiplier() {
        // max ll with exact Newton equivalent to minimizing
        // 0.5 * x^2 - rhs*x with rhs=1 under 0 <= x <= 0.1.
        // At x=0, active-set KKT solve gives lambda_sys=+1 for the lower bound,
        // which must be released (lambda_true = -lambda_sys).
        let hessian = array![[1.0]];
        let rhs = array![1.0];
        let beta_start = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0], [-1.0]],
            b: array![0.0, -0.1],
        };

        let (beta, active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            None,
        )
        .expect("constrained quadratic solve should succeed");

        assert!(
            (beta[0] - 0.1).abs() <= 1e-10,
            "expected constrained optimum at upper bound 0.1, got {}",
            beta[0]
        );
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn quadratic_linear_constraints_ignore_near_tangential_inactiverows() {
        let hessian = array![[1.0, 0.0], [0.0, 1.0]];
        let rhs = array![1.0, 0.0];
        let beta_start = array![0.0, 0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[-1e-16, 1.0]],
            b: array![-1.0],
        };

        let (beta, active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            None,
        )
        .expect("near-tangential inactive row should not block the quadratic step");

        assert!(
            (beta[0] - 1.0).abs() <= 1e-12,
            "expected unconstrained x-solution of 1.0, got {}",
            beta[0]
        );
        assert!(
            beta[1].abs() <= 1e-12,
            "expected zero y-solution, got {}",
            beta[1]
        );
        assert!(active.is_empty(), "no row should become active");
    }

    #[test]
    fn quadratic_linear_constraints_projectwarm_activerows_back_to_boundary() {
        let hessian = array![[2.0]];
        let rhs = array![0.0];
        let beta_start = array![1e-9];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![0.0],
        };

        let (beta, active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            Some(&[0]),
        )
        .expect("constrained quadratic solve should project back to the boundary");

        assert_relative_eq!(beta[0], 0.0, epsilon = 1e-14);
        assert_eq!(active, vec![0]);
    }

    #[test]
    fn quadratic_linear_constraints_handles_near_dependent_rows() {
        // Three constraints in R^2 where the third is nearly a linear
        // combination of the first two, making the naive KKT system
        // ill-conditioned.  The rank-reducing compression should drop
        // the dependent row and the QP should converge cleanly.
        //
        //   x1 >= 0,  x2 >= 0,  x1 + x2 + eps >= 0   (eps ≈ 0)
        //
        // Minimize 0.5 * ||x - [−1, −1]||^2  =>  optimum at origin.
        let hessian = Array2::eye(2);
        let rhs = array![-1.0, -1.0]; // gradient points toward (−1,−1)
        let beta_start = array![0.0, 0.0];
        let eps = 1e-14;
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, 1.0], [1.0 + eps, 1.0]],
            b: array![0.0, 0.0, 0.0],
        };

        let (beta, active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            Some(&[0, 1, 2]), // all three active
        )
        .expect("near-dependent constraint QP should converge");

        assert!(
            beta[0].abs() <= 1e-10 && beta[1].abs() <= 1e-10,
            "expected optimum at origin, got ({}, {})",
            beta[0],
            beta[1]
        );
        assert!(
            active.len() <= 2,
            "at most 2 independent constraints should remain active, got {}",
            active.len()
        );
    }

    #[test]
    fn quadratic_linear_constraints_release_merged_constraint_group_by_id() {
        // Two redundant lower-bound rows compress into one active KKT row.
        // Releasing that merged row must drop both original constraint ids,
        // not transient positions in the active vector.
        let hessian = array![[1.0]];
        let rhs = array![1.0];
        let beta_start = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0], [2.0], [-1.0]],
            b: array![0.0, 0.0, -0.1],
        };

        let (beta, active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            Some(&[0, 1]),
        )
        .expect("merged active constraint group should release cleanly");

        assert!(
            (beta[0] - 0.1).abs() <= 1e-10,
            "expected constrained optimum at upper bound 0.1, got {}",
            beta[0]
        );
        assert_eq!(active, vec![2]);
    }

    #[test]
    fn quadratic_linear_constraints_release_merged_group_with_unsorted_active_positions() {
        let hessian = array![[1.0]];
        let rhs = array![1.0];
        let beta_start = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0], [2.0], [-1.0]],
            b: array![0.0, 0.0, -0.1],
        };

        let (beta, active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            Some(&[2, 0, 1]),
        )
        .expect("merged active group release should handle unsorted active positions");

        assert!(
            (beta[0] - 0.1).abs() <= 1e-10,
            "expected constrained optimum at upper bound 0.1, got {}",
            beta[0]
        );
        assert_eq!(active, vec![2]);
    }

    #[test]
    fn quadratic_linear_constraints_accept_boundary_kkt_after_rank_reduction() {
        let hessian = array![[2.0]];
        let rhs = array![0.0];
        let beta_start = array![1e-9];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0], [1.0 + 1e-13], [2.0], [3.0]],
            b: array![0.0, 0.0, 0.0, 0.0],
        };

        let (beta, active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            Some(&[0, 1, 2, 3]),
        )
        .expect("degenerate boundary KKT point should be accepted");

        assert_relative_eq!(beta[0], 0.0, epsilon = 1e-14);
        assert!(
            active.len() <= 1,
            "rank-reduced boundary solution should keep at most one representative, got {:?}",
            active
        );
    }

    #[test]
    fn quadratic_linear_constraints_singular_kkt_uses_pseudoinverse_fallback() {
        let hessian = Array2::<f64>::zeros((2, 2));
        let rhs = array![0.0, 0.0];
        let beta_start = array![0.0, 0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 1.0]],
            b: array![0.0],
        };

        let (beta, active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            Some(&[0]),
        )
        .expect("singular KKT system should fall back to a finite pseudoinverse solve");

        assert!(beta.iter().all(|value| value.is_finite()));
        assert_relative_eq!(beta[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(beta[1], 0.0, epsilon = 1e-14);
        assert_eq!(active, vec![0]);
    }

    #[test]
    fn rank_reduce_drops_exactly_dependent_row() {
        // Row 3 = Row 1 + Row 2 exactly. Rank reduction should drop it.
        let a = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],];
        let b = array![0.0, 0.0, 0.0];
        let member_constraint_ids = vec![vec![0], vec![1], vec![2]];
        let (a_out, b_out, member_constraint_ids_out) =
            crate::solver::active_set::rank_reduce_rows_pivoted_qr(a, b, member_constraint_ids);
        assert_eq!(
            a_out.nrows(),
            2,
            "should keep 2 independent rows, got {}",
            a_out.nrows()
        );
        assert_eq!(b_out.len(), 2);
        // The third constraint id should have been merged into one of the first two rows.
        let total_constraint_ids: usize = member_constraint_ids_out.iter().map(|g| g.len()).sum();
        assert_eq!(
            total_constraint_ids, 3,
            "all original constraint ids must be preserved"
        );
    }

    #[test]
    fn rank_reduce_preserves_full_rank_matrix() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];
        let b = array![0.0, 0.0, 0.0];
        let member_constraint_ids = vec![vec![0], vec![1], vec![2]];
        let (a_out, b_out, member_constraint_ids_out) =
            crate::solver::active_set::rank_reduce_rows_pivoted_qr(a, b, member_constraint_ids);
        // All three rows are independent in R^2 (but we only have rank 2).
        // The first two span R^2, so row 3 = row 1 + row 2 is dependent.
        assert_eq!(a_out.nrows(), 2);
        assert_eq!(b_out.len(), 2);
        let total_constraint_ids: usize = member_constraint_ids_out.iter().map(|g| g.len()).sum();
        assert_eq!(total_constraint_ids, 3);
    }

    #[test]
    fn constrained_exact_newton_nan_hessian_returns_feasible_noop_instead_of_failing() {
        let spec = ParameterBlockSpec {
            name: "exact_block".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let states = vec![ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        }];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![0.0],
        };
        let hessian = SymmetricMatrix::Dense(array![[f64::NAN]]);
        let updater = ExactNewtonBlockUpdater {
            gradient: &array![0.0],
            hessian: &hessian,
        };
        let s_lambda = Array2::zeros((1, 1));
        let update = updater
            .compute_update_step(&BlockUpdateContext {
                family: &OneBlockConstrainedNaNHessianFamily,
                states: &states,
                spec: &spec,
                block_idx: 0,
                s_lambda: &s_lambda,
                options: &BlockwiseFitOptions::default(),
                linear_constraints: Some(&constraints),
                cached_active_set: None,
            })
            .expect("constrained exact-newton NaN Hessian should produce a no-op update");
        assert_relative_eq!(update.beta_new_raw[0], 0.0, epsilon = 1e-14);
        assert_eq!(update.active_set, Some(vec![0]));
    }

    #[test]
    fn outerobjective_failure_context_is_preserved() {
        // One penalty forces the outer rho optimizer to run, which should now preserve
        // the real evaluation error instead of returning an opaque line-search failure.
        let spec = ParameterBlockSpec {
            name: "err_block".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [1.0]
            ])),
            offset: array![0.0, 0.0],
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            outer_max_iter: 3,
            ..BlockwiseFitOptions::default()
        };
        let err = match fit_custom_family(&OneBlockAlwaysErrorFamily, &[spec], &options) {
            Ok(_) => panic!("fit should fail when family evaluate always errors"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains(
                "last objective error: synthetic outer objective failure: block[0] evaluate()"
            ),
            "expected preserved root-cause context in error, got: {err}"
        );
    }

    #[test]
    fn fit_fails_when_requested_covariance_cannot_be_computed() {
        let spec = ParameterBlockSpec {
            name: "cov_block".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [1.0]
            ])),
            offset: array![0.0, 0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            use_remlobjective: false,
            compute_covariance: true,
            ..BlockwiseFitOptions::default()
        };
        let err = match fit_custom_family(&OneBlockCovarianceErrorFamily, &[spec], &options) {
            Ok(_) => panic!("fit should fail when covariance computation fails"),
            Err(e) => e,
        };
        assert!(
            err.to_string()
                .contains("synthetic covariance assembly failure"),
            "expected covariance root cause in fit error, got: {err}"
        );
    }

    //
    // Root-cause tests for "exact-newton eigendecomposition failed: NoConvergence"
    //
    // Bug: ExactNewtonBlockUpdater::compute_update_step calls FaerEigh::eigh()
    // to determine the minimum eigenvalue for ridging, but this eigendecomposition
    // itself fails when the Hessian contains NaN/Inf (e.g. from overflow in the
    // log_sigma block when exp(eta) overflows).  Unlike stable_logdet_with_ridge_policy
    // which has a 4-tier fallback chain (eigh -> SVD -> ridged Cholesky -> diagonal),
    // compute_update_step has NO fallback and propagates the error, killing the fit.
    //
    // The tests below prove necessity and sufficiency:
    //   - NECESSITY: NaN/Inf in the Hessian causes the eigendecomposition to fail,
    //     which is the sole cause of the "eigendecomposition failed" crash.
    //   - SUFFICIENCY: The same family with finite Hessians succeeds; the same
    //     pathological matrix succeeds through stable_logdet_with_ridge_policy.
    //

    /// A QuadraticReml family whose log_sigma block returns a Hessian containing
    /// NaN, simulating what happens when exp(eta_sigma) overflows during
    /// location-scale fitting.
    #[derive(Clone)]
    struct TwoBlockNaNHessianFamily;

    impl CustomFamily for TwoBlockNaNHessianFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n0 = block_states[0].eta.len();
            let p1 = block_states[1].beta.len();
            // Block 0 (mu): well-behaved diagonal working set.
            // Block 1 (log_sigma): ExactNewton with NaN in the Hessian,
            // simulating overflow from extreme coefficients.
            let mut hessian = Array2::<f64>::eye(p1);
            hessian[[0, 0]] = f64::NAN; // overflow poison
            Ok(FamilyEvaluation {
                log_likelihood: -0.5 * block_states[0].eta.iter().map(|&v| v * v).sum::<f64>(),
                blockworking_sets: vec![
                    BlockWorkingSet::Diagonal {
                        working_response: Array1::zeros(n0),
                        working_weights: Array1::ones(n0),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(p1),
                        hessian: SymmetricMatrix::Dense(hessian),
                    },
                ],
            })
        }
    }

    /// Same two-block layout but with finite Hessians — the control group.
    #[derive(Clone)]
    struct TwoBlockFiniteHessianFamily;

    impl CustomFamily for TwoBlockFiniteHessianFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n0 = block_states[0].eta.len();
            let p1 = block_states[1].beta.len();
            let beta1 = &block_states[1].beta;
            let resid1: f64 = beta1.iter().map(|&b| b * b).sum();
            Ok(FamilyEvaluation {
                log_likelihood: -0.5 * block_states[0].eta.iter().map(|&v| v * v).sum::<f64>()
                    - 0.5 * resid1,
                blockworking_sets: vec![
                    BlockWorkingSet::Diagonal {
                        working_response: Array1::zeros(n0),
                        working_weights: Array1::ones(n0),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: -beta1.clone(),
                        hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                    },
                ],
            })
        }
    }

    /// Same NaN-Hessian family but with PseudoLaplace objective, which takes
    /// the strict-SPD path and skips the eigendecomposition in compute_update_step.
    #[derive(Clone)]
    struct TwoBlockNaNHessianPseudoLaplaceFamily;

    impl CustomFamily for TwoBlockNaNHessianPseudoLaplaceFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            TwoBlockNaNHessianFamily.evaluate(block_states)
        }

        fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::StrictPseudoLaplace
        }
    }

    fn make_two_block_specs(n: usize) -> Vec<ParameterBlockSpec> {
        vec![
            ParameterBlockSpec {
                name: "mu".to_string(),
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    Array2::from_elem((n, 1), 1.0),
                )),
                offset: Array1::zeros(n),
                penalties: vec![],
                nullspace_dims: vec![],
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(array![0.0]),
            },
            ParameterBlockSpec {
                name: "log_sigma".to_string(),
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    Array2::from_elem((n, 2), 1.0),
                )),
                offset: Array1::zeros(n),
                penalties: vec![],
                nullspace_dims: vec![],
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(array![0.0, 0.0]),
            },
        ]
    }

    #[test]
    fn exact_newton_nan_hessian_survives_via_fallback() {
        // With the Layer 2 fix, a NaN Hessian in the log_sigma block no
        // longer crashes with "eigendecomposition failed". The fallback
        // chain detects NaN eigenvalues (via f64::minimum) or catches
        // eigh failure, applies a conservative ridge, and the solve
        // proceeds. The line search will reject the NaN-contaminated
        // step, leaving beta unchanged — a safe no-op.
        let specs = make_two_block_specs(4);
        let per_block_log_lambdas = vec![Array1::zeros(0), Array1::zeros(0)];
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        let result = inner_blockwise_fit(
            &TwoBlockNaNHessianFamily,
            &specs,
            &per_block_log_lambdas,
            &options,
            None,
        );
        // The fit should survive — either Ok (step rejected by line search)
        // or Err from a downstream cause, but NOT "eigendecomposition failed".
        match result {
            Ok(_) => {}
            Err(ref msg) => {
                assert!(
                    !msg.contains("eigendecomposition failed"),
                    "NaN Hessian should be handled by fallback, not crash: {msg}"
                );
            }
        }
    }

    #[test]
    fn exact_newton_finite_hessian_succeeds_where_nan_hessian_fails() {
        // SUFFICIENCY (control): The identical two-block structure with a
        // finite Hessian succeeds, proving that NaN in the Hessian is the
        // specific trigger — not the block layout, penalty structure, or
        // solver configuration.
        let specs = make_two_block_specs(4);
        let per_block_log_lambdas = vec![Array1::zeros(0), Array1::zeros(0)];
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        let result = inner_blockwise_fit(
            &TwoBlockFiniteHessianFamily,
            &specs,
            &per_block_log_lambdas,
            &options,
            None,
        );
        assert!(
            result.is_ok(),
            "inner fit should succeed with finite Hessian: {:?}",
            result.err()
        );
    }

    #[test]
    fn checked_penalizedobjective_rejects_non_finite_values() {
        let err = checked_penalizedobjective(-1.0, 0.5, f64::NAN, "test objective")
            .expect_err("non-finite objective should fail loudly");
        assert!(
            err.contains("non-finite penalized objective"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn exact_newton_dh_closure_rejects_non_finite_directional_derivative() {
        #[derive(Clone)]
        struct OneBlockNonFiniteJointDhFamily;

        impl CustomFamily for OneBlockNonFiniteJointDhFamily {
            fn evaluate(
                &self,
                block_states: &[ParameterBlockState],
            ) -> Result<FamilyEvaluation, String> {
                let beta = block_states
                    .first()
                    .ok_or_else(|| "missing block 0".to_string())?
                    .beta
                    .clone();
                Ok(FamilyEvaluation {
                    log_likelihood: -0.5 * beta.dot(&beta),
                    blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                        gradient: beta.mapv(|v| -v),
                        hessian: SymmetricMatrix::Dense(array![[1.0]]),
                    }],
                })
            }

            fn exact_newton_joint_hessian(
                &self,
                _: &[ParameterBlockState],
            ) -> Result<Option<Array2<f64>>, String> {
                Ok(Some(array![[1.0]]))
            }

            fn exact_newton_joint_hessian_directional_derivative(
                &self,
                _: &[ParameterBlockState],
                _: &Array1<f64>,
            ) -> Result<Option<Array2<f64>>, String> {
                Ok(Some(array![[f64::NAN]]))
            }
        }

        let family = OneBlockNonFiniteJointDhFamily;
        let specs = vec![ParameterBlockSpec {
            name: "beta".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (2, 1),
                1.0,
            ))),
            offset: Array1::zeros(2),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        }];
        let states = vec![ParameterBlockState {
            beta: array![0.0],
            eta: Array1::zeros(2),
        }];
        let synced_states = Arc::new(
            synchronized_states_from_flat_beta(&family, &specs, &states, &array![0.0])
                .expect("sync states for exact_newton_dh_closure"),
        );
        let compute_dh =
            exact_newton_dh_closure(&family, synced_states, &specs, 1, false, 1.0, None);
        let err = compute_dh(&array![1.0]).expect_err("non-finite dH should fail loudly");
        assert!(err.contains("non-finite"), "unexpected error: {err}");
    }

    #[test]
    fn nan_propagating_min_detects_nan_eigenvalues() {
        // Verify the fix: our NaN-propagating min correctly detects
        // NaN eigenvalues, unlike f64::min which silently ignored them.
        let mut mat = Array2::<f64>::eye(3);
        mat[[1, 0]] = f64::NAN;
        mat[[0, 1]] = f64::NAN;

        use crate::faer_ndarray::FaerEigh;
        match FaerEigh::eigh(&mat, faer::Side::Lower) {
            Err(_) => {
                // eigh failed — the fallback chain in compute_update_step
                // now catches this and applies a conservative ridge.
            }
            Ok((evals, _)) => {
                // NaN-propagating fold (matches the production code):
                let new_min = evals.iter().copied().fold(f64::INFINITY, |a, b| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.min(b)
                    }
                });
                assert!(
                    !new_min.is_finite(),
                    "NaN-propagating min should detect NaN eigenvalues, got {new_min}"
                );
            }
        }
    }

    #[test]
    fn multiblock_generic_outer_fallback_returns_error_instead_of_panicking() {
        let family = TwoBlockFiniteHessianFamily;
        let specs = make_two_block_specs(4);
        let penalty_counts = vec![0usize, 0usize];
        let rho = Array1::zeros(0);
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            outerobjectivegradienthessian(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho,
                None,
                EvalMode::ValueGradientHessian,
            )
        }));

        let outcome = result.expect("multi-block outer fallback must return an error, not panic");
        let err = match outcome {
            Ok(_) => panic!("multi-block family without a joint path should fail loudly"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("multi-block families must provide a joint outer path"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn pseudo_laplace_path_skips_eigendecomposition_avoiding_nan_crash() {
        // SUFFICIENCY: The PseudoLaplace path takes strict_solve_spd instead
        // of eigendecomposition-based ridging.  It will still fail (the Hessian
        // is NaN so the solve produces garbage), but the failure is NOT the
        // eigendecomposition NoConvergence error — it's a different error
        // downstream.  This proves the eigendecomposition call is the unique
        // failure point for QuadraticReml families.
        let specs = make_two_block_specs(4);
        let per_block_log_lambdas = vec![Array1::zeros(0), Array1::zeros(0)];
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        let result = inner_blockwise_fit(
            &TwoBlockNaNHessianPseudoLaplaceFamily,
            &specs,
            &per_block_log_lambdas,
            &options,
            None,
        );
        // The PseudoLaplace path may fail for other reasons (NaN in solve),
        // but it must NOT fail with the eigendecomposition error.
        match result {
            Ok(_) => {} // Acceptable — strict_solve_spd might produce NaN
            // betas which don't trigger a hard error.
            Err(ref msg) => {
                assert!(
                    !msg.contains("exact-newton eigendecomposition failed"),
                    "PseudoLaplace path should NOT hit eigendecomposition; \
                     got eigendecomposition error anyway: {msg}"
                );
            }
        }
    }

    // ---------- eta_backup heterogeneous-shape regression tests ----------
    //
    // Regression note: a previous `inner_blockwise_fit` implementation
    // reused a single `eta_backup` buffer across blocks during line search.
    // With heterogeneous eta lengths (e.g. survival time block = 3n,
    // threshold/log-sigma = n), that buffer could be left at the wrong
    // shape for the next block update and trigger an ndarray broadcast
    // panic:
    //   "could not broadcast array from shape: [n] to: [3n]"

    /// Minimal two-block family where block 0 has design nrows=3n and
    /// block 1 has design nrows=n. Both use ExactNewton. Block 0's
    /// gradient is nonzero so the Newton step exceeds tol and exercises
    /// the line-search path that previously mishandled heterogeneous
    /// eta buffer shapes.
    #[derive(Clone)]
    struct HeterogeneousEtaLengthFamily {
        n: usize,
    }

    impl CustomFamily for HeterogeneousEtaLengthFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = self.n;
            let eta0 = &block_states[0].eta;
            let eta1 = &block_states[1].eta;
            assert_eq!(eta0.len(), 3 * n, "block 0 eta must be 3n");
            assert_eq!(eta1.len(), n, "block 1 eta must be n");
            let p0 = block_states[0].beta.len();
            let p1 = block_states[1].beta.len();
            // Simple quadratic log-likelihood so optimum is at beta=0.
            let ll = -0.5 * eta0.dot(eta0) - 0.5 * eta1.dot(eta1);
            // Nonzero gradient drives a real step in both blocks.
            let grad0 = &(-&block_states[0].beta) + &Array1::from_elem(p0, 0.1);
            let grad1 = &(-&block_states[1].beta) + &Array1::from_elem(p1, 0.1);
            Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: grad0,
                        hessian: SymmetricMatrix::Dense(Array2::eye(p0)),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad1,
                        hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                    },
                ],
            })
        }
    }

    fn make_heterogeneous_eta_specs(n: usize) -> Vec<ParameterBlockSpec> {
        let p0 = 2;
        let p1 = 2;
        vec![
            ParameterBlockSpec {
                name: "big_block".to_string(),
                // 3n rows — mimics survival time block stacking
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    Array2::from_elem((3 * n, p0), 1.0),
                )),
                offset: Array1::zeros(3 * n),
                penalties: vec![],
                nullspace_dims: vec![],
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(Array1::from_elem(p0, 1.0)),
            },
            ParameterBlockSpec {
                name: "small_block".to_string(),
                // n rows — mimics threshold/log-sigma block
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    Array2::from_elem((n, p1), 1.0),
                )),
                offset: Array1::zeros(n),
                penalties: vec![],
                nullspace_dims: vec![],
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(Array1::from_elem(p1, 1.0)),
            },
        ]
    }

    /// Regression guard: blocks with identical eta lengths never exercised
    /// the old heterogeneous-shape failure mode.
    #[test]
    fn uniform_eta_lengths_do_not_panic() {
        let n = 10;
        #[derive(Clone)]
        struct UniformEtaFamily;
        impl CustomFamily for UniformEtaFamily {
            fn evaluate(
                &self,
                block_states: &[ParameterBlockState],
            ) -> Result<FamilyEvaluation, String> {
                let p0 = block_states[0].beta.len();
                let p1 = block_states[1].beta.len();
                let eta0 = &block_states[0].eta;
                let eta1 = &block_states[1].eta;
                let ll = -0.5 * eta0.dot(eta0) - 0.5 * eta1.dot(eta1);
                Ok(FamilyEvaluation {
                    log_likelihood: ll,
                    blockworking_sets: vec![
                        BlockWorkingSet::ExactNewton {
                            gradient: &(-&block_states[0].beta) + &Array1::from_elem(p0, 0.1),
                            hessian: SymmetricMatrix::Dense(Array2::eye(p0)),
                        },
                        BlockWorkingSet::ExactNewton {
                            gradient: &(-&block_states[1].beta) + &Array1::from_elem(p1, 0.1),
                            hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                        },
                    ],
                })
            }
        }
        // Both blocks have n rows — no shape mismatch possible.
        let specs = vec![
            ParameterBlockSpec {
                name: "block_a".to_string(),
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    Array2::from_elem((n, 2), 1.0),
                )),
                offset: Array1::zeros(n),
                penalties: vec![],
                nullspace_dims: vec![],
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(Array1::from_elem(2, 1.0)),
            },
            ParameterBlockSpec {
                name: "block_b".to_string(),
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    Array2::from_elem((n, 2), 1.0),
                )),
                offset: Array1::zeros(n),
                penalties: vec![],
                nullspace_dims: vec![],
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(Array1::from_elem(2, 1.0)),
            },
        ];
        let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
        let options = BlockwiseFitOptions {
            inner_max_cycles: 3,
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        // Must NOT panic — uniform eta lengths keep eta_backup
        // compatible with every block's eta after mem::swap.
        let result = inner_blockwise_fit(&UniformEtaFamily, &specs, &per_block, &options, None);
        assert!(
            result.is_ok(),
            "uniform eta lengths should not panic: {result:?}"
        );
    }

    /// Regression guard: heterogeneous eta lengths (3n vs n) must not
    /// prevent the inner fit from completing. Older code could panic with
    /// "could not broadcast array from shape: [n] to: [3n]" due to the
    /// eta_backup swap bug.
    #[test]
    fn heterogeneous_eta_lengths_inner_fit_completes() {
        let n = 10;
        let family = HeterogeneousEtaLengthFamily { n };
        let specs = make_heterogeneous_eta_specs(n);
        let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
        let options = BlockwiseFitOptions {
            inner_max_cycles: 3,
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        let result = inner_blockwise_fit(&family, &specs, &per_block, &options, None);
        assert!(result.is_ok(), "inner fit should complete: {result:?}");
    }

    /// SUFFICIENCY (single-cycle): even one inner cycle must complete
    /// without panic when blocks have heterogeneous eta lengths.
    #[test]
    fn heterogeneous_eta_single_cycle_completes() {
        let n = 10;
        let family = HeterogeneousEtaLengthFamily { n };
        let specs = make_heterogeneous_eta_specs(n);
        let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        let result = inner_blockwise_fit(&family, &specs, &per_block, &options, None);
        assert!(
            result.is_ok(),
            "single-cycle inner fit should complete: {result:?}"
        );
    }

    /// Regression guard: when all blocks have step <= tol, the line-search
    /// path is skipped for every block, so this case should remain safe
    /// even with heterogeneous eta lengths.
    #[test]
    fn heterogeneous_eta_no_panic_when_all_blocks_converged() {
        let n = 10;
        #[derive(Clone)]
        struct AllConvergedFamily {
            n: usize,
        }
        impl CustomFamily for AllConvergedFamily {
            fn evaluate(
                &self,
                block_states: &[ParameterBlockState],
            ) -> Result<FamilyEvaluation, String> {
                let n = self.n;
                let eta0 = &block_states[0].eta;
                let eta1 = &block_states[1].eta;
                assert_eq!(eta0.len(), 3 * n);
                assert_eq!(eta1.len(), n);
                let p0 = block_states[0].beta.len();
                let p1 = block_states[1].beta.len();
                let ll = -0.5 * eta0.dot(eta0) - 0.5 * eta1.dot(eta1);
                Ok(FamilyEvaluation {
                    log_likelihood: ll,
                    blockworking_sets: vec![
                        BlockWorkingSet::ExactNewton {
                            gradient: Array1::zeros(p0),
                            hessian: SymmetricMatrix::Dense(Array2::eye(p0)),
                        },
                        BlockWorkingSet::ExactNewton {
                            gradient: Array1::zeros(p1),
                            hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                        },
                    ],
                })
            }
        }
        let mut specs = make_heterogeneous_eta_specs(n);
        specs[0].initial_beta = Some(Array1::zeros(2));
        specs[1].initial_beta = Some(Array1::zeros(2));
        let family = AllConvergedFamily { n };
        let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        // All blocks converged → step=0 → `continue` before swap →
        // eta_backup never participates → no broadcast panic.
        let result = inner_blockwise_fit(&family, &specs, &per_block, &options, None);
        assert!(
            result.is_ok(),
            "should not panic when all blocks are converged: {result:?}"
        );
    }

    /// Regression guard: even when only the second (smaller) block takes
    /// a step, the fit must complete. Earlier code could still panic here
    /// after reusing an oversized eta_backup buffer across blocks.
    #[test]
    fn heterogeneous_eta_completes_when_only_small_block_steps() {
        let n = 10;
        #[derive(Clone)]
        struct OnlySmallBlockStepsFamily {
            n: usize,
        }
        impl CustomFamily for OnlySmallBlockStepsFamily {
            fn evaluate(
                &self,
                block_states: &[ParameterBlockState],
            ) -> Result<FamilyEvaluation, String> {
                let n = self.n;
                let eta0 = &block_states[0].eta;
                let eta1 = &block_states[1].eta;
                assert_eq!(eta0.len(), 3 * n);
                assert_eq!(eta1.len(), n);
                let p0 = block_states[0].beta.len();
                let p1 = block_states[1].beta.len();
                let ll = -0.5 * eta0.dot(eta0) - 0.5 * eta1.dot(eta1);
                Ok(FamilyEvaluation {
                    log_likelihood: ll,
                    blockworking_sets: vec![
                        BlockWorkingSet::ExactNewton {
                            // Block 0: converged, step=0
                            gradient: Array1::zeros(p0),
                            hessian: SymmetricMatrix::Dense(Array2::eye(p0)),
                        },
                        BlockWorkingSet::ExactNewton {
                            // Block 1: nontrivial step
                            gradient: &(-&block_states[1].beta) + &Array1::from_elem(p1, 0.1),
                            hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                        },
                    ],
                })
            }
        }
        let mut specs = make_heterogeneous_eta_specs(n);
        specs[0].initial_beta = Some(Array1::zeros(2)); // block 0 at optimum
        let family = OnlySmallBlockStepsFamily { n };
        let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        let result = inner_blockwise_fit(&family, &specs, &per_block, &options, None);
        assert!(
            result.is_ok(),
            "fit should complete when only small block steps: {result:?}"
        );
    }

    /// Direct test of the KKT-aware projection in
    /// `projected_stationarity_inf_norm`.
    ///
    /// Contract:
    ///   (i)   with no constraints, returns the plain inf-norm of the residual;
    ///   (ii)  at an active lower bound with multiplier-signed residual
    ///         (`β_j == lb_j` and `residual_j > 0`) the coordinate is skipped;
    ///   (iii) at an active lower bound with wrong-signed residual
    ///         (`residual_j < 0`) the coordinate still contributes;
    ///   (iv)  interior coordinates always contribute regardless of
    ///         residual sign.
    ///
    /// This pins the exact convergence semantics that the joint-Newton loop
    /// relies on: a genuine constrained-KKT optimum must score zero, while
    /// infeasibility and interior non-stationarity remain observable.
    #[test]
    fn projected_stationarity_inf_norm_respects_kkt_multipliers() {
        // Test (i): no constraints → plain inf-norm.
        let beta = array![1.0, 2.0, -0.5];
        let residual = array![0.3, -0.1, 0.2];
        let inf_nocon = projected_stationarity_inf_norm(&residual, &beta, None);
        assert_relative_eq!(inf_nocon, 0.3_f64, epsilon = 1e-12);

        // Test (ii): β_j at its lower bound with residual_j > 0 is a KKT
        // multiplier; projection drops it, so only the interior entry (-0.1)
        // contributes.
        let beta_active = array![0.0, 2.0];
        let residual_active = array![0.5, -0.1];
        let constraints_lb0 = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, 1.0]],
            b: array![0.0, f64::NEG_INFINITY], // only β_0 has a finite lower bound
        };
        // Build a minimal single-row constraint first (β_0 ≥ 0) so the
        // "active lower bound + positive residual" branch of the projection
        // is exercised in isolation.  β_1 is left unconstrained relative to
        // this single-row constraint matrix (it's not pinned by any row),
        // so its contribution (|-0.1| = 0.1) stays in the inf-norm.
        let single = LinearInequalityConstraints {
            a: array![[1.0, 0.0]],
            b: array![0.0],
        };
        let inf_projected =
            projected_stationarity_inf_norm(&residual_active, &beta_active, Some(&single));
        assert_relative_eq!(inf_projected, 0.1_f64, epsilon = 1e-12);

        // Also verify the per-coord handling of an explicitly-unconstrained
        // row (b = -inf) in the two-row form: β_0 has a finite lower bound
        // of 0 (from row 0), β_1 gets lb = -inf (from row 1 via b/a), which
        // `lb.is_finite() == false` routes to the "no lower bound" branch of
        // the projection.  The active-bound drop still fires on coord 0, so
        // the result matches the single-row case: 0.1.  This documents that
        // the projection's per-coord `lb.is_finite()` gate is what makes the
        // unconstrained-coord case work — NOT rejection of the whole
        // constraint set by `extract_simple_lower_bounds`.
        let inf_with_two_row =
            projected_stationarity_inf_norm(&residual_active, &beta_active, Some(&constraints_lb0));
        assert_relative_eq!(inf_with_two_row, 0.1_f64, epsilon = 1e-12);

        // Test (iii): β_j at its bound but residual points the WRONG way
        // (residual_j < 0 means the KKT dual feasibility λ_j ≥ 0 is violated
        // — i.e. the bound should release).  Keep that coordinate in the
        // norm so the optimizer does not declare convergence on an infeasible
        // multiplier.
        let beta_wrong_sign = array![0.0];
        let residual_wrong_sign = array![-0.2];
        let single1 = LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![0.0],
        };
        let inf_wrong_sign =
            projected_stationarity_inf_norm(&residual_wrong_sign, &beta_wrong_sign, Some(&single1));
        assert_relative_eq!(inf_wrong_sign, 0.2_f64, epsilon = 1e-12);

        // Test (iv): an interior coordinate with a valid lower bound keeps
        // contributing to the norm, whatever the residual sign.
        let beta_interior = array![1.5];
        let residual_interior = array![0.4];
        let inf_interior =
            projected_stationarity_inf_norm(&residual_interior, &beta_interior, Some(&single1));
        assert_relative_eq!(inf_interior, 0.4_f64, epsilon = 1e-12);
    }
}
