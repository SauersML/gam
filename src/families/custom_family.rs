use crate::faer_ndarray::FaerEigh;
use crate::faer_ndarray::{FaerCholesky, fast_atb, fast_av};
use crate::linalg::utils::StableSolver;
use crate::matrix::{
    DesignMatrix, EmbeddedColumnBlock, EmbeddedSquareBlock, LinearOperator, SymmetricMatrix,
};
use crate::pirls::{LinearInequalityConstraints, solve_newton_directionwith_lower_bounds};
use crate::resource::{DerivativeStorageMode, ResourcePolicy};
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
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, s};
use std::any::Any;
use std::cell::RefCell;
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
///   For supported non-canonical diagonal links, W must be the observed weight
///   W_obs = W_Fisher - (y-mu)*B so the outer REML uses the exact Laplace
///   Hessian. The matching [`CustomFamily::diagonalworking_weights_directional_derivative`]
///   callback must differentiate the same observed W surface; silently using Fisher
///   weights or zero `dW` would change the criterion into a PQL-type surrogate.
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

/// Exact outer derivative order for families that expose second-order
/// coefficient geometry.
///
/// This used to be a cost gate that demoted large biobank-scale problems to
/// first-order BFGS. That was a policy leak into the math layer: if the family
/// supplies analytic dense Hessian blocks or an analytic profiled-Hessian HVP,
/// the outer optimizer should see the exact second-order objective. Runtime
/// representation choices (dense vs operator) belong below this declaration,
/// not in a first-order downgrade.
pub fn exact_outer_order_from_capability(
    _specs: &[ParameterBlockSpec],
    _coefficient_cost: u64,
) -> ExactOuterDerivativeOrder {
    ExactOuterDerivativeOrder::Second
}

/// Capability-aware variant of [`exact_outer_order_from_capability`].
///
/// Kept as the public declaration helper for existing family impls, but it no
/// longer gates by cost. Once a caller has established dense or HVP analytic
/// second-order support, the correct derivative order is `Second`.
pub fn exact_outer_order_with_outer_hvp(
    specs: &[ParameterBlockSpec],
    coefficient_cost: u64,
    _outer_hyper_hessian_hvp_available: bool,
) -> ExactOuterDerivativeOrder {
    exact_outer_order_from_capability(specs, coefficient_cost)
}

/// Default coefficient-space Hessian cost: `Σ_b n_b · p_b²`, summed across
/// blocks. Represents the work to assemble or apply the dense block-diagonal
/// inner Hessian once.
pub fn default_coefficient_hessian_cost(specs: &[ParameterBlockSpec]) -> u64 {
    specs
        .iter()
        .map(|s| {
            let n = s.design.nrows() as u64;
            let p = s.design.ncols() as u64;
            n.saturating_mul(p.saturating_mul(p))
        })
        .fold(0u64, |acc, c| acc.saturating_add(c))
}

/// Joint-coupled coefficient-space Hessian cost: `n · (Σ_b p_b)²`. The honest
/// per-evaluation work for any family whose row likelihood couples every block
/// (every observation contributes a rank-`m` outer-product update to the full
/// joint Hessian over `Σ p_b` coefficients), as opposed to the block-diagonal
/// `default_coefficient_hessian_cost` which assumes each `X_b' W_b X_b` is
/// assembled independently.
///
/// Used by all GAMLSS, marginal-slope, and joint-latent families. CTN does
/// not delegate here — it uses its Khatri–Rao factor dimensions internally.
pub fn joint_coupled_coefficient_hessian_cost(n: u64, specs: &[ParameterBlockSpec]) -> u64 {
    let p_total: u64 = specs
        .iter()
        .map(|s| s.design.ncols() as u64)
        .fold(0u64, |acc, p| acc.saturating_add(p));
    n.saturating_mul(p_total.saturating_mul(p_total))
}

/// Default coefficient-space gradient cost: half the Hessian cost.
///
/// The first-order analytic gradient in the unified evaluator runs the same
/// inner Newton solve as the second-order path but skips the `K`-fold
/// pairwise Hessian assembly (`B_{j,k}` blocks) and the `K`-fold inner
/// derivative solves; what remains is the inner solve plus a single
/// gradient-only sweep through the data. Empirically this is roughly half
/// the per-evaluation arithmetic of forming the dense Hessian, hence the
/// `/2` default. Families whose gradient assembly differs structurally
/// (e.g. matrix-free Hv operators with no dense Hessian assembly to halve)
/// should override [`CustomFamily::coefficient_gradient_cost`] explicitly.
pub fn default_coefficient_gradient_cost(specs: &[ParameterBlockSpec]) -> u64 {
    default_coefficient_hessian_cost(specs) / 2
}

/// Bound first-order outer iterations when each analytic-gradient evaluation is
/// already biobank-scale work. This is only applied after the planner has
/// selected a gradient-only route; second-order/ARC plans keep their requested
/// iteration budget.
pub fn cost_gated_first_order_max_iter(
    requested: usize,
    coefficient_gradient_cost: u64,
    has_outer_hessian: bool,
) -> usize {
    const FIRST_ORDER_OUTER_WORK_BUDGET: u64 = 80_000_000_000;
    const MIN_FIRST_ORDER_ITERS: usize = 4;

    if has_outer_hessian || requested <= 1 || coefficient_gradient_cost == 0 {
        return requested;
    }

    let affordable = (FIRST_ORDER_OUTER_WORK_BUDGET / coefficient_gradient_cost) as usize;
    requested.min(affordable.max(MIN_FIRST_ORDER_ITERS))
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

/// Batched per-θ_j contributions to the analytic outer gradient.
///
/// Used by [`CustomFamily::batched_outer_gradient_terms`] to amortize the
/// joint-Hessian factorization across all K hyperparameters: instead of
/// computing each `tr(H⁻¹ · Ḣ_j)` independently (K independent solves), the
/// family factors `H` once, computes per-row leverages `L_i = Z_i H⁻¹ Z_iᵀ`,
/// and accumulates all K traces in a single streaming pass.
///
/// All three vectors have length equal to the total number of outer
/// hyperparameters (K = `rho.len() + Σ derivative_blocks[b].len()`), in the
/// same coordinate order as the unified evaluator's gradient: ρ-coords first,
/// ψ-coords appended.
///
/// # Assembly formula
///
/// The caller assembles the outer gradient as
///
/// ```text
///   grad[j] = objective_theta[j]
///           + 0.5 * trace_h_inv_hdot[j]
///           - 0.5 * trace_s_pinv_sdot[j]
/// ```
///
/// matching the three-term convention in [`outer_gradient_entry`] (penalty +
/// trace − det).
pub struct BatchedOuterHessianTerms {
    /// Exact profiled outer Hessian over θ = (ρ, ψ), assembled or exposed in
    /// operator form by the family in one amortized evaluation.
    pub outer_hessian: crate::solver::outer_strategy::HessianResult,
}

pub struct BatchedOuterGradientTerms {
    /// Explicit ∂J/∂θ_j contributions evaluated at the converged β̂ holding
    /// β fixed (i.e. the part that does NOT flow through H or S):
    ///
    /// * For ρ-coords: `½ β̂ᵀ A_k β̂` (penalty quadratic).
    /// * For ψ-coords: `V_i^explicit + g_i^explicit · β̂` style contributions.
    pub objective_theta: Array1<f64>,
    /// `tr(H⁻¹ · ∂H/∂θ_j)` for each j, with H = -∇²log L + S the full
    /// penalized Hessian at the mode.
    pub trace_h_inv_hdot: Array1<f64>,
    /// `tr(S⁺ · ∂S/∂θ_j)` for each j (penalty pseudo-logdet first derivative).
    pub trace_s_pinv_sdot: Array1<f64>,
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

    /// Options-aware log-likelihood evaluation for line search.
    ///
    /// Default forwards to [`log_likelihood_only`] and ignores `_options`.
    /// Families that consult `options.outer_score_subsample` (or other
    /// per-call options that affect the LL value) must override this so the
    /// joint-Newton line search and the post-accept gradient reload agree
    /// on which row subset is being evaluated. Biobank-scale outer-only
    /// callers (including the joint-Newton line-search screening path) can
    /// override this to evaluate a deterministic paired Horvitz-Thompson
    /// estimate without constructing a full exact-Newton workspace.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        _options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        self.log_likelihood_only(block_states)
    }

    /// Whether `log_likelihood_only_with_options` can use
    /// `BlockwiseFitOptions::early_exit_threshold` to reject line-search trials
    /// without computing the full log-likelihood.
    fn supports_log_likelihood_early_exit(&self) -> bool {
        false
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

    /// Per-evaluation arithmetic cost of forming or applying the inner
    /// coefficient-space Hessian once, in flop-equivalent units. This is used
    /// for diagnostics, seed-budget policy, and first-order iteration caps
    /// when a family genuinely lacks analytic second-order support. It is not
    /// allowed to hide an analytic Hessian from the outer optimizer.
    ///
    /// The default returns `Σ_b n_b · p_b²` via [`default_coefficient_hessian_cost`],
    /// which is the honest assembly cost only when the joint Hessian is
    /// **block-diagonal** — i.e. the inner solver assembles each block's
    /// `X_b' W_b X_b` independently, with no cross-block coupling per row.
    /// Families whose row likelihood couples all blocks (every row contributes
    /// a rank-`m` outer-product update to the full joint Hessian over
    /// `Σ p_b` coefficients) **must** override and delegate to
    /// [`joint_coupled_coefficient_hessian_cost`] (or the equivalent factored
    /// form for tensor designs), otherwise the default undercounts the
    /// cross-block outer-product terms `2·Σ_{a<b} n·p_a·p_b`.
    ///
    /// Concretely:
    ///
    /// * **Block-diagonal** (default OK): `LatentBinaryFamily` collects
    ///   separate `hess_time` and `hess_mean` per row, never forming an
    ///   off-diagonal contribution.
    /// * **Joint-coupled** (override via [`joint_coupled_coefficient_hessian_cost`]):
    ///   GAMLSS location-scale, GAMLSS wiggle variants, marginal-slope families
    ///   (Bernoulli, Survival), `LatentSurvivalFamily`,
    ///   `SurvivalLocationScaleFamily` — every row contributes to the full
    ///   `(Σ p_b)²` joint Hessian via Jacobian pullback of a multi-dimensional
    ///   primary kernel.
    /// * **Single-block** (default OK): tensor designs whose `design.ncols()`
    ///   already equals `p_total` (e.g. CTN's Khatri–Rao `n × (p_resp·p_cov)`);
    ///   `n · p²` reduces correctly to `n · p_resp² · p_cov²`.
    /// * **Matrix-free Hessian operator**: families that expose
    ///   [`Self::exact_newton_joint_hessian_workspace`] with operator-form
    ///   directional derivatives (CTN at biobank scale) may instead return
    ///   the per-`Hv` matvec cost (e.g. `n·(p_resp + p_cov)` for Khatri–Rao)
    ///   so the gate reflects the operator path rather than the dense
    ///   build that the unified evaluator skips.
    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        default_coefficient_hessian_cost(specs)
    }

    /// Per-evaluation arithmetic cost of one analytic-gradient outer
    /// evaluation, in flop-equivalent units. Used only when the family
    /// genuinely has no analytic outer Hessian and the planner must use a
    /// first-order optimizer.
    ///
    /// The default returns `coefficient_hessian_cost / 2` (see
    /// [`default_coefficient_gradient_cost`]). Families whose gradient
    /// assembly differs structurally should override; in particular,
    /// joint-coupled families that override `coefficient_hessian_cost` to
    /// `joint_coupled_coefficient_hessian_cost(n, specs)` automatically
    /// inherit the corresponding gradient cost via this default — no
    /// per-family override is required for the GAMLSS / marginal-slope /
    /// joint-latent path.
    fn coefficient_gradient_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        self.coefficient_hessian_cost(specs) / 2
    }

    /// Declares how much exact outer calculus this family wants to expose for
    /// the current realized problem size.
    ///
    /// The default exposes exact second-order calculus whenever the family
    /// advertises either dense outer Hessian blocks or profiled outer-Hessian
    /// HVPs. Large problems must stay exact and select an operator
    /// representation; they are not demoted to first-order optimizers.
    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        let coefficient_work = self
            .coefficient_hessian_cost(specs)
            .max(self.coefficient_gradient_cost(specs));
        if !self.outer_hyper_hessian_dense_available(specs)
            && !self.outer_hyper_hessian_hvp_available(specs)
        {
            return ExactOuterDerivativeOrder::First;
        }
        exact_outer_order_with_outer_hvp(
            specs,
            coefficient_work,
            self.outer_hyper_hessian_hvp_available(specs),
        )
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

    /// Optional exact second directional derivative of a block's ExactNewton Hessian.
    ///
    /// Returns `Some(d2H)` where:
    /// - `d2H` is `D²_beta H_L[u, v]` for the provided block-local
    ///   coefficient-space directions.
    /// - shape is `(p_block, p_block)`.
    ///
    /// Generic single-block REML/LAML Hessian evaluation requires this term for
    /// `BlockWorkingSet::ExactNewton` blocks; `None` means the exact second
    /// Hessian drift is unavailable.
    fn exact_newton_hessian_second_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &Array1<f64>,
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
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Default block-diagonal assembly from per-block ExactNewton hessians.
        // This is the inner-fit-side default and is *intentionally* not gated
        // by `likelihood_blocks_uncoupled()`: the inner joint-Newton loop only
        // uses this Hessian as a Newton-direction surrogate that is
        // immediately validated by the line-search + objective decrease, so
        // even if the family is coupled, an under-resolved block-diagonal
        // direction will simply backtrack instead of corrupting the outer
        // REML score.  The strict coupling gate lives one layer up, on
        // `exact_newton_joint_hessian_with_specs`, where outer REML trace
        // algebra would silently produce wrong answers from a missing
        // cross-block term.
        exact_newton_joint_hessian_from_exact_blocks(self, block_states)
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
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        exact_newton_joint_hessian_directional_derivative_from_blocks(
            self,
            block_states,
            d_beta_flat,
        )
    }

    /// Optional exact second directional derivative of the joint Hessian.
    ///
    /// Returns `Some(d2H)` where `d2H` is:
    ///   D²H[u, v] = d/dε d/dδ H(beta + εu + δv) |_{ε=δ=0}
    /// for flattened coefficient-space directions `u = d_beta_u_flat`,
    /// `v = d_betav_flat`.
    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        exact_newton_joint_hessiansecond_directional_derivative_from_blocks(
            self,
            block_states,
            d_beta_u_flat,
            d_betav_flat,
        )
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

    /// Outer-aware variant of `exact_newton_joint_hessian_workspace`.
    ///
    /// Families that consume the optional outer-only stratified row subsample
    /// (`options.outer_score_subsample`) override this method so the joint
    /// Hessian workspace can be constructed with the subsample mask attached.
    /// Generic families can stick with the default implementation, which
    /// simply forwards to the legacy no-options method and ignores the
    /// options. This keeps full backward compatibility with existing
    /// implementors while letting the marginal-slope families thread the
    /// subsample down into the cached per-evaluation joint-Hessian directional
    /// derivative paths.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        _options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        self.exact_newton_joint_hessian_workspace(states, specs)
    }

    /// Optional line-search evaluator that returns both the exact
    /// log-likelihood and a reusable joint-Hessian workspace built during the
    /// same row sweep.
    fn joint_line_search_log_likelihood_workspace(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> Result<Option<(f64, Arc<dyn ExactNewtonJointHessianWorkspace>)>, String> {
        Ok(None)
    }

    /// Optional batched analytic-gradient hook.
    ///
    /// Returns the K per-θ_j gradient contributions ([`BatchedOuterGradientTerms`])
    /// in one amortized pass when the family can factor its joint Hessian
    /// once and stream row-block leverages instead of computing each
    /// `tr(H⁻¹ · ∂H/∂θ_j)` independently.
    ///
    /// # Cost amortization
    ///
    /// Generic per-θ_j path: `O(K · n · p²)` (K independent dense traces).
    /// Batched path: `O(n · p²)` (single factor + leverage stream)
    ///                 + `O(K · n · m²)` (per-row block-diagonal accumulators
    ///                   with `m` = per-row predictor dimension; m = 2 for
    ///                   GAMLSS location-scale, 1 for scalar GLMs).
    ///
    /// At biobank scale with K ≈ 15, p ≈ 64, m = 2 the batched path is
    /// ≈ K·p²/(p² + K·m²) ≈ 15× cheaper.
    ///
    /// # Default
    ///
    /// Returns `Ok(None)`. The unified outer gradient evaluator falls back
    /// to its generic per-coordinate path. Families with row-coupled
    /// likelihoods (GAMLSS location-scale, marginal-slope) should override.
    ///
    /// Implementations may return `Ok(None)` for ψ-coordinates whose
    /// design-drift is too involved for a batched leverage form, letting
    /// the generic path handle those cases.
    fn batched_outer_gradient_terms(
        &self,
        _block_states: &[ParameterBlockState],
        _specs: &[ParameterBlockSpec],
        _derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        _rho: &Array1<f64>,
        _hessian_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    ) -> Result<Option<BatchedOuterGradientTerms>, String> {
        Ok(None)
    }

    /// Optional batched analytic-Hessian / HVP hook.
    ///
    /// This is the Hessian-side analogue of
    /// [`Self::batched_outer_gradient_terms`]: families that can share a
    /// single factorization, row-leverage stream, or directional θθ kernel
    /// across all explicit outer-Hessian terms return the exact profiled
    /// Hessian here.  The evaluator uses this hook only for Hessian-capable
    /// families and only after the inner mode has been fitted; default
    /// `None` leaves unsupported families on their existing exact path.
    fn batched_outer_hessian_terms(
        &self,
        _block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        _derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        _rho: &Array1<f64>,
        _hessian_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    ) -> Result<Option<BatchedOuterHessianTerms>, String> {
        Ok(self
            .outer_hyper_hessian_operator(specs)
            .map(|operator| BatchedOuterHessianTerms {
                outer_hessian: crate::solver::outer_strategy::HessianResult::Operator(operator),
            }))
    }

    /// Explicit name for the inner coefficient-space Hessian HVP capability.
    ///
    /// Kept separate from outer hyper-Hessian capabilities so CTN/GAMLSS row
    /// operators do not accidentally advertise pairwise θθ calculus as cheap.
    fn inner_coefficient_hessian_hvp_available(&self, _specs: &[ParameterBlockSpec]) -> bool {
        false
    }

    fn inner_joint_workspace_gradient_available(&self, _specs: &[ParameterBlockSpec]) -> bool {
        false
    }

    fn inner_joint_workspace_log_likelihood_available(
        &self,
        _specs: &[ParameterBlockSpec],
    ) -> bool {
        false
    }

    /// True only when the family has a real profiled outer Hessian-vector
    /// product over θ = (ρ, ψ), without enumerating all θ_i θ_j pairs.
    fn outer_hyper_hessian_hvp_available(&self, _specs: &[ParameterBlockSpec]) -> bool {
        false
    }

    /// True when the family can expose the dense profiled outer Hessian.
    /// Generic custom-family pairwise derivative paths default to dense
    /// availability; families with only inner HVP support should override this
    /// if dense θθ assembly is not a valid capability for their path.
    fn outer_hyper_hessian_dense_available(&self, _specs: &[ParameterBlockSpec]) -> bool {
        true
    }

    /// Family-supplied exact outer Hessian operator over θ = (ρ, ψ).
    ///
    /// When a family can produce the full profiled outer Hessian as a
    /// matrix-free Hv operator — using its own directional θθ kernels and
    /// trace algebra rather than the generic per-pair enumeration — it
    /// overrides this method and returns `Some(op)`.  The unified REML/LAML
    /// evaluator wires the operator into [`HessianResult::Operator`] via
    /// the [`HessianDerivativeProvider::family_outer_hessian_operator`] hook
    /// the family installs on its provider; consumers see a generic
    /// `Arc<dyn OuterHessianOperator>` (matvec / dim / mul_mat /
    /// is_cheap_to_materialize).
    ///
    /// Default returns `None`, leaving the family on the existing pairwise
    /// assembly path.  This is the architectural contract for CTN, survival
    /// (Gompertz-Makeham + timewiggle), GAMLSS location-scale, and
    /// Bernoulli marginal-slope families to plug their directional
    /// outer-HVP operators into the same surface.
    fn outer_hyper_hessian_operator(
        &self,
        _specs: &[ParameterBlockSpec],
    ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
        None
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
    ///
    /// For multi-block families, the working-set fallback only fires when the
    /// family has explicitly declared its blocks are uncoupled in the
    /// likelihood Hessian via `likelihood_blocks_uncoupled() = true`.  This
    /// is critical: `exact_newton_joint_hessian_from_working_sets` produces a
    /// strictly block-diagonal joint Hessian, which silently drops cross-block
    /// `∂²L/∂β_a∂β_b` terms for coupled likelihoods (GAMLSS μ-σ, marginal
    /// slope, survival location-scale, etc.).  Default `false` ⇒ multi-block
    /// custom families must override `exact_newton_joint_hessian` (or
    /// `exact_newton_outer_curvature`) and the higher layer surfaces a loud
    /// "joint outer path required" error rather than silently using
    /// block-diagonal curvature.
    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        // Multi-axis dispatch over the joint Hessian source:
        //
        // * Single-block, or family declared `likelihood_blocks_uncoupled` —
        //   the working-sets block-diagonal IS exact (no cross-block coupling
        //   exists), so it's a valid fallback when the family override
        //   returns None.
        //
        // * Multi-block coupled with `has_explicit_joint_hessian = true` —
        //   the family override IS the only trusted joint Hessian.  If it
        //   returns None (e.g. dense form too large for memory at biobank
        //   scale), propagate None.  Substituting the working-sets
        //   block-diagonal would silently drop the cross-block
        //   ∂²L/∂β_a∂β_b curvature the family is the only source of —
        //   exactly the corruption this gate exists to prevent.
        //
        // * Multi-block coupled, no explicit override — refuse entirely so
        //   the multi-block error surfaces upstream.
        if specs.len() <= 1 || self.likelihood_blocks_uncoupled() {
            match self.exact_newton_joint_hessian(block_states)? {
                Some(hessian) => Ok(Some(hessian)),
                None => exact_newton_joint_hessian_from_working_sets(self, block_states, specs),
            }
        } else if self.has_explicit_joint_hessian() {
            self.exact_newton_joint_hessian(block_states)
        } else {
            Ok(None)
        }
    }

    /// Whether the family's log-likelihood Hessian is block-diagonal in the
    /// joint coefficient vector — i.e. `∂²L/∂β_a∂β_b = 0` for every pair of
    /// distinct blocks `a ≠ b`.  Default `false` (assume coupling, the safe
    /// answer); families whose blocks share no η/W coupling override to
    /// `true` to opt into the default working-set joint-Hessian assembly for
    /// multi-block specs.
    fn likelihood_blocks_uncoupled(&self) -> bool {
        false
    }

    /// Whether the family has an explicit override of `exact_newton_joint_hessian`
    /// (or its `_with_specs` variant) that returns the *true* coupled joint
    /// Hessian rather than the trait's block-diagonal default.
    ///
    /// Default `false`.  Production families that override
    /// `exact_newton_joint_hessian` with their analytic coupled curvature must
    /// set this to `true` so the outer-REML path can trust the override
    /// downstream of `exact_newton_joint_hessian_with_specs`.  The trait can't
    /// detect override status by reflection, so this marker is the contract
    /// signal.
    fn has_explicit_joint_hessian(&self) -> bool {
        false
    }

    /// Internal helper: do the outer-REML `_with_specs` defaults trust the
    /// inner-fit's block-diagonal-from-blocks output for this family?
    ///
    /// Trustworthy iff:
    /// - single-block (no cross-block coupling possible), or
    /// - the family has declared its blocks uncoupled in the likelihood
    ///   Hessian (`likelihood_blocks_uncoupled` ⇒ block-diagonal IS exact),
    ///   or
    /// - the family has an explicit joint-Hessian override
    ///   (`has_explicit_joint_hessian` ⇒ what we receive from
    ///   `exact_newton_joint_hessian` is the true coupled Hessian, not the
    ///   block-diagonal default).
    fn outer_default_trustworthy_for_joint_hessian(&self, specs: &[ParameterBlockSpec]) -> bool {
        specs.len() <= 1 || self.likelihood_blocks_uncoupled() || self.has_explicit_joint_hessian()
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
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Same trust dispatch as `exact_newton_joint_hessian_with_specs` —
        // the default `_directional_derivative` and `_from_working_sets`
        // both build a block-diagonal `D_β H[u]`, which silently drops the
        // cross-block `∂²L_ab/∂β_a∂β_b · u_b` rows that drive the outer
        // mode-response correction for coupled families.
        if specs.len() <= 1 || self.likelihood_blocks_uncoupled() {
            match self
                .exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)?
            {
                Some(dh) => Ok(Some(dh)),
                None => exact_newton_joint_hessian_directional_derivative_from_working_sets(
                    self,
                    block_states,
                    specs,
                    d_beta_flat,
                ),
            }
        } else if self.has_explicit_joint_hessian() {
            self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
        } else {
            Ok(None)
        }
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
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Same trust dispatch as the Hessian / first-derivative paths.  The
        // delegated `exact_newton_joint_hessiansecond_directional_derivative`
        // default is block-diagonal-from-blocks, which is silently wrong for
        // outer trace assembly on coupled families.  Unlike the lower-order
        // paths, there is no working-sets fallback — both trusted branches
        // call the same delegate, so a single helper predicate suffices.
        if !self.outer_default_trustworthy_for_joint_hessian(specs) {
            return Ok(None);
        }
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

    /// Optional exact second directional derivative of diagonal working weights.
    ///
    /// This callback supplies the `d²w` term for static-design single-block
    /// generic fallback Hessian drift:
    ///
    ///   D²_beta H_L[u, v] = X^T diag(D²w[D eta_u, D eta_v]) X.
    ///
    /// Families with coefficient-dependent block geometry must use an exact
    /// Newton Hessian path or a joint outer path until second-order geometry
    /// hooks are available; the generic diagonal fallback will reject nonzero
    /// first-order geometry while building `d²H`.
    fn diagonalworking_weights_second_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &Array1<f64>,
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

    /// Outer-aware variant of `exact_newton_joint_psi_workspace`.
    ///
    /// Families that consume the optional outer-only stratified row subsample
    /// (`options.outer_score_subsample`) override this method so the workspace
    /// can be constructed with the subsample mask attached. Generic families
    /// can stick with the default implementation, which simply forwards to
    /// the legacy no-options method and ignores the options. This keeps full
    /// backward compatibility with existing implementors while letting the
    /// marginal-slope families thread the subsample down into the cached
    /// per-evaluation ψ calculus.
    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        _options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        self.exact_newton_joint_psi_workspace(states, specs, derivs)
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
    /// If true, the joint-Newton line search may reuse an exact-Newton
    /// workspace to read trial log-likelihood values. This preserves the
    /// legacy path for A/B regression tests, but defaults to false because
    /// rejected backtracking attempts discard that workspace and can dominate
    /// FLEX marginal-slope fits — the cheap scalar log-likelihood is
    /// sufficient for the accept/reject decision; the full state is built
    /// only after a step is accepted.
    pub line_search_prefer_workspace: bool,
    /// Optional line-search objective ceiling for lazy log-likelihood-only
    /// evaluations. Families whose per-row log-likelihood contributions are
    /// non-positive may stop once the partial negative log-likelihood is already
    /// above this ceiling, because the unvisited rows cannot improve the trial
    /// objective enough to be accepted. Default `None` preserves exact full-sum
    /// behavior and is the only mode used outside backtracking rejection tests.
    pub early_exit_threshold: Option<f64>,
    /// Optional stratified row subsample used by outer-only score/gradient
    /// passes. When `Some(s)`, outer score/gradient hot loops should iterate
    /// only over `s.mask` and rescale per-row contributions by
    /// `s.weight_scale = n_full / mask.len()`. Inner-PIRLS and final
    /// covariance passes always run on the full data, so this field is
    /// consulted only by outer-only call sites. Default `None` preserves
    /// the legacy full-data behavior. Wrapping in `Arc` keeps `Clone` cheap
    /// across the many places `BlockwiseFitOptions` is duplicated per-eval.
    pub outer_score_subsample:
        Option<Arc<crate::families::marginal_slope_shared::OuterScoreSubsample>>,
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
            // Default ON: families expose exact outer Hessians whenever their
            // analytic dense or operator representation is implemented.
            use_outer_hessian: true,
            compute_covariance: false,
            screening_max_inner_iterations: None,
            line_search_prefer_workspace: false,
            early_exit_threshold: None,
            outer_score_subsample: None,
        }
    }
}

#[derive(Clone)]
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
    pub joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
}

impl std::fmt::Debug for BlockwiseInnerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockwiseInnerResult")
            .field("block_states", &self.block_states)
            .field("active_sets", &self.active_sets)
            .field("log_likelihood", &self.log_likelihood)
            .field("penalty_value", &self.penalty_value)
            .field("cycles", &self.cycles)
            .field("converged", &self.converged)
            .field("block_logdet_h", &self.block_logdet_h)
            .field("block_logdet_s", &self.block_logdet_s)
            .field("s_lambdas", &self.s_lambdas)
            .field(
                "joint_workspace",
                &self.joint_workspace.as_ref().map(|_| "<workspace>"),
            )
            .finish()
    }
}

#[derive(Clone)]
struct ConstrainedWarmStart {
    rho: Array1<f64>,
    block_beta: Vec<Array1<f64>>,
    active_sets: Vec<Option<Vec<usize>>>,
    cached_inner: Option<CachedInnerMode>,
}

#[derive(Clone)]
struct CachedInnerMode {
    log_likelihood: f64,
    penalty_value: f64,
    cycles: usize,
    converged: bool,
    block_logdet_h: f64,
    block_logdet_s: f64,
    joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
}

fn screened_outer_warm_start<'a>(
    warm_start: Option<&'a ConstrainedWarmStart>,
    rho: &Array1<f64>,
) -> Option<&'a ConstrainedWarmStart> {
    warm_start.filter(|seed| seed.rho.len() == rho.len())
}

fn warm_start_matches_block_log_lambdas(
    seed: &ConstrainedWarmStart,
    block_log_lambdas: &[Array1<f64>],
) -> bool {
    let expected = block_log_lambdas
        .iter()
        .map(|values| values.len())
        .sum::<usize>();
    if seed.rho.len() != expected {
        return false;
    }
    let mut offset = 0usize;
    for block in block_log_lambdas {
        let end = offset + block.len();
        if seed.rho.slice(s![offset..end]) != block.view() {
            return false;
        }
        offset = end;
    }
    true
}

fn cached_inner_mode_from_result(result: &BlockwiseInnerResult) -> CachedInnerMode {
    CachedInnerMode {
        log_likelihood: result.log_likelihood,
        penalty_value: result.penalty_value,
        cycles: result.cycles,
        converged: result.converged,
        block_logdet_h: result.block_logdet_h,
        block_logdet_s: result.block_logdet_s,
        joint_workspace: result.joint_workspace.clone(),
    }
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
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;
    /// Single-row specialization of `row_chunk_first`. Default implementation
    /// delegates to `row_chunk_first(axis, row..row+1)` and copies the
    /// resulting row into the output buffer; implementations that can avoid
    /// the temporary matrix allocation should override this method.
    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        let chunk = self.row_chunk_first(axis, row..row + 1)?;
        out.assign(&chunk.row(0));
        Ok(())
    }
    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;
    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;

    /// Optional upcast to the dense materialization surface. Production exact
    /// paths should prefer the analytic matvec / row-chunk methods above and
    /// avoid forming the full derivative matrix; implementations that *do*
    /// support dense materialization (used by diagnostics, tests, and
    /// small-data fallbacks) should override this to return `Some(self)`.
    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        None
    }
}

/// Diagnostic / small-data extension that exposes dense materialization of
/// `\partial X / \partial \psi`. Production exact-Hessian code MUST NOT depend
/// on dense second-derivative materialization; second-order paths use the
/// row-chunk and matvec methods on [`CustomFamilyPsiDerivativeOperator`].
pub(crate) trait MaterializablePsiDerivativeOperator:
    CustomFamilyPsiDerivativeOperator
{
    fn materialize_first(
        &self,
        axis: usize,
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
        let f: fn(
            &crate::terms::basis::ImplicitDesignPsiDerivative,
            usize,
            Range<usize>,
        ) -> Result<Array2<f64>, crate::terms::basis::BasisError> =
            crate::terms::basis::ImplicitDesignPsiDerivative::row_chunk_first;
        f(self, axis, rows)
    }

    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::row_vector_first_into(
            self, axis, row, out,
        )
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let f: fn(
            &crate::terms::basis::ImplicitDesignPsiDerivative,
            usize,
            Range<usize>,
        ) -> Result<Array2<f64>, crate::terms::basis::BasisError> =
            crate::terms::basis::ImplicitDesignPsiDerivative::row_chunk_second_diag;
        f(self, axis, rows)
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let f: fn(
            &crate::terms::basis::ImplicitDesignPsiDerivative,
            usize,
            usize,
            Range<usize>,
        ) -> Result<Array2<f64>, crate::terms::basis::BasisError> =
            crate::terms::basis::ImplicitDesignPsiDerivative::row_chunk_second_cross;
        f(self, axis_d, axis_e, rows)
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

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for crate::terms::basis::ImplicitDesignPsiDerivative {
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::materialize_first(self, axis)
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

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let local = self.base.row_chunk_first(axis, rows)?;
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        out.fill(0.0);
        let local_slice = out.slice_mut(ndarray::s![self.global_range.clone()]);
        self.base.row_vector_first_into(axis, row, local_slice)
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let local = self.base.row_chunk_second_diag(axis, rows)?;
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let local = self.base.row_chunk_second_cross(axis_d, axis_e, rows)?;
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for EmbeddedImplicitPsiDerivativeOperator {
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
}

/// Non-allocating zero operator for `\partial X / \partial \psi` derivative
/// blocks whose ψ coordinate does not move the design matrix at all (e.g.
/// the spatial-adaptive overlay's mass / tension / stiffness / ε
/// hyperparameters, which act through the penalty stack alone).
///
/// All matvec/transpose_mul methods return zero vectors of the correct
/// length, all row-chunk methods return chunk-sized zero matrices. The
/// operator never allocates an `(n, p)` dense buffer, which saves ~1.45 GiB
/// at the biobank-scale spatial-adaptive overlay (n ≈ 320 000, p ≈ 101,
/// six hyperparameters).
pub(crate) struct ZeroPsiDerivativeOperator {
    n: usize,
    p: usize,
}

impl ZeroPsiDerivativeOperator {
    pub(crate) fn new(n: usize, p: usize) -> Self {
        Self { n, p }
    }
}

impl CustomFamilyPsiDerivativeOperator for ZeroPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.n
    }

    fn p_out(&self) -> usize {
        self.p
    }

    fn transpose_mul(
        &self,
        _axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        debug_assert_eq!(v.len(), self.n);
        Ok(Array1::<f64>::zeros(self.p))
    }

    fn forward_mul(
        &self,
        _axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        debug_assert_eq!(u.len(), self.p);
        Ok(Array1::<f64>::zeros(self.n))
    }

    fn transpose_mul_second_diag(
        &self,
        _axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        debug_assert_eq!(v.len(), self.n);
        Ok(Array1::<f64>::zeros(self.p))
    }

    fn transpose_mul_second_cross(
        &self,
        _axis_d: usize,
        _axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        debug_assert_eq!(v.len(), self.n);
        Ok(Array1::<f64>::zeros(self.p))
    }

    fn forward_mul_second_diag(
        &self,
        _axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        debug_assert_eq!(u.len(), self.p);
        Ok(Array1::<f64>::zeros(self.n))
    }

    fn forward_mul_second_cross(
        &self,
        _axis_d: usize,
        _axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        debug_assert_eq!(u.len(), self.p);
        Ok(Array1::<f64>::zeros(self.n))
    }

    fn row_chunk_first(
        &self,
        _axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p)))
    }

    fn row_vector_first_into(
        &self,
        _axis: usize,
        _row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        out.fill(0.0);
        Ok(())
    }

    fn row_chunk_second_diag(
        &self,
        _axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p)))
    }

    fn row_chunk_second_cross(
        &self,
        _axis_d: usize,
        _axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p)))
    }
}

fn rowwise_kronecker_dense(base: &Array2<f64>, time_basis: &Array2<f64>) -> Array2<f64> {
    assert_eq!(base.nrows(), time_basis.nrows());
    let n = base.nrows();
    let p_base = base.ncols();
    let p_time = time_basis.ncols();
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    // Row-wise Khatri-Rao of (base ⊗ time_basis) per observation:
    // out[i, j*p_time + t] = base[i, j] * time_basis[i, t]. Independent
    // across rows.
    let row_data: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let base_row = base.row(i);
            let time_row = time_basis.row(i);
            let mut row_vec = vec![0.0f64; p_base * p_time];
            for j in 0..p_base {
                let base_ij = base_row[j];
                if base_ij == 0.0 {
                    continue;
                }
                let off = j * p_time;
                for t in 0..p_time {
                    row_vec[off + t] = base_ij * time_row[t];
                }
            }
            row_vec.into_iter()
        })
        .collect();
    Array2::<f64>::from_shape_vec((n, p_base * p_time), row_data)
        .expect("row Khatri-Rao shape consistent")
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

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi row_chunk_first")?;
        let local = self.first_local.slice(ndarray::s![rows, ..]).to_owned();
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi row_vector_first_into")?;
        if row >= self.first_local.nrows() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi row_vector_first_into row {row} out of bounds for {}",
                self.first_local.nrows()
            )));
        }
        if out.len() != self.total_p {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi row_vector_first_into expected length {}, got {}",
                self.total_p,
                out.len()
            )));
        }
        out.fill(0.0);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&self.first_local.row(row));
        Ok(())
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi row_chunk_second_diag")?;
        let local = self
            .second_diag_local
            .slice(ndarray::s![rows, ..])
            .to_owned();
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi row_chunk_second_cross")?;
        let local = self
            .cross_local(axis_e, "embedded dense psi row_chunk_second_cross")?
            .slice(ndarray::s![rows, ..])
            .to_owned();
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for EmbeddedDensePsiDerivativeOperator {
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

    fn lifted_row_chunk_with_base<F>(
        &self,
        rows: Range<usize>,
        mut base_chunk: F,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>
    where
        F: FnMut(Range<usize>) -> Result<Array2<f64>, crate::terms::basis::BasisError>,
    {
        if rows.start > rows.end || rows.end > self.n_data() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "rowwise kronecker psi row chunk {}..{} out of bounds for {} rows",
                rows.start,
                rows.end,
                self.n_data()
            )));
        }
        if rows.is_empty() {
            return Ok(Array2::<f64>::zeros((0, self.p_out)));
        }

        let first_block = rows.start / self.n_per_block;
        let last_block = (rows.end - 1) / self.n_per_block;
        let mut blocks = Vec::with_capacity(last_block + 1 - first_block);
        for block_idx in first_block..=last_block {
            let block_global_start = block_idx * self.n_per_block;
            let local_start = rows.start.saturating_sub(block_global_start);
            let local_end = (rows.end - block_global_start).min(self.n_per_block);
            let local_rows = local_start..local_end;
            let base = base_chunk(local_rows.clone())?;
            let time = self.time_bases[block_idx]
                .slice(ndarray::s![local_rows, ..])
                .to_owned();
            blocks.push(rowwise_kronecker_dense(&base, &time));
        }
        Ok(stack_dense_row_blocks(&blocks))
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

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.lifted_row_chunk_with_base(rows, |local_rows| {
            self.base.row_chunk_first(axis, local_rows)
        })
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.lifted_row_chunk_with_base(rows, |local_rows| {
            self.base.row_chunk_second_diag(axis, local_rows)
        })
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.lifted_row_chunk_with_base(rows, |local_rows| {
            self.base.row_chunk_second_cross(axis_d, axis_e, local_rows)
        })
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for RowwiseKroneckerPsiDerivativeOperator {
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let base_mat = self.base.as_materializable().ok_or_else(|| {
            crate::terms::basis::BasisError::Other(
                "rowwise kronecker psi operator: base operator does not support materialization"
                    .to_string(),
            )
        })?;
        let base = base_mat.materialize_first(axis)?;
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
        if row >= self.nrows() {
            return Err(format!(
                "psi design row {row} exceeds available rows {}",
                self.nrows()
            ));
        }
        let absolute_row = self.row_range.start + row;
        let mut out = Array1::<f64>::zeros(self.p);
        self.operator
            .row_vector_first_into(self.axis, absolute_row, out.view_mut())
            .map_err(|e| e.to_string())?;
        Ok(out)
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

#[derive(Clone)]
pub(crate) enum PsiDesignMap {
    Zero {
        nrows: usize,
        ncols: usize,
    },
    Dense {
        matrix: Arc<Array2<f64>>,
    },
    First {
        action: CustomFamilyPsiDesignAction,
    },
    Second {
        action: CustomFamilyPsiSecondDesignAction,
    },
}

impl PsiDesignMap {
    pub(crate) fn ncols(&self) -> usize {
        match self {
            Self::Zero { ncols, .. } => *ncols,
            Self::Dense { matrix } => matrix.ncols(),
            Self::First { action } => action.p,
            Self::Second { action } => action.p,
        }
    }

    pub(crate) fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        match self {
            Self::Zero { nrows, .. } => Ok(Array1::<f64>::zeros(*nrows)),
            Self::Dense { matrix } => Ok(matrix.dot(&u)),
            Self::First { action } => Ok(action.forward_mul(u)),
            Self::Second { action } => Ok(action.forward_mul(u)),
        }
    }

    pub(crate) fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        let ncols = self.ncols();
        match self {
            Self::Zero { .. } => Ok(Array2::<f64>::zeros((rows.end - rows.start, ncols))),
            Self::Dense { matrix } => Ok(matrix.slice(ndarray::s![rows, ..]).to_owned()),
            Self::First { action } => action.row_chunk(rows),
            Self::Second { action } => action.row_chunk(rows),
        }
    }

    pub(crate) fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        match self {
            Self::Zero { ncols, .. } => Ok(Array1::<f64>::zeros(*ncols)),
            Self::Dense { matrix } => Ok(matrix.row(row).to_owned()),
            Self::First { action } => action.row_vector(row),
            Self::Second { action } => action.row_vector(row),
        }
    }

    /// Borrow this map as a `CustomFamilyPsiLinearMapRef`, handling every
    /// variant. This is the zero-allocation replacement for the pattern
    /// `first_psi_linear_map(action.as_ref(), dense.as_ref(), n, p)`.
    pub(crate) fn as_linear_map_ref(&self) -> CustomFamilyPsiLinearMapRef<'_> {
        match self {
            Self::Zero { nrows, ncols } => CustomFamilyPsiLinearMapRef::Zero {
                nrows: *nrows,
                ncols: *ncols,
            },
            Self::Dense { matrix } => CustomFamilyPsiLinearMapRef::Dense(matrix.as_ref()),
            Self::First { action } => CustomFamilyPsiLinearMapRef::First(action),
            Self::Second { action } => CustomFamilyPsiLinearMapRef::Second(action),
        }
    }

    /// Return a reference to the first-derivative operator action if this map
    /// holds one. Useful for callers that need to pass ownership of the action
    /// into downstream operator builders.
    pub(crate) fn as_first_action(&self) -> Option<&CustomFamilyPsiDesignAction> {
        match self {
            Self::First { action } => Some(action),
            _ => None,
        }
    }

    /// Clone the first-derivative operator action if this map holds one.
    pub(crate) fn cloned_first_action(&self) -> Option<CustomFamilyPsiDesignAction> {
        self.as_first_action().cloned()
    }
}

fn is_zero_array(a: &Array2<f64>) -> bool {
    a.iter().all(|x| *x == 0.0)
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
    let p_left = left.ncols();
    let p_right = right.ncols();
    if p_left == 0 || p_right == 0 {
        return Ok(Array2::<f64>::zeros((p_left, p_right)));
    }
    // Zero fast path: either operand being the Zero variant makes the full product zero.
    if matches!(left, CustomFamilyPsiLinearMapRef::Zero { .. })
        || matches!(right, CustomFamilyPsiLinearMapRef::Zero { .. })
    {
        return Ok(Array2::<f64>::zeros((p_left, p_right)));
    }

    let mut out = Array2::<f64>::zeros((p_left, p_right));
    // Stream row chunks of both operands so the weighted intermediate is never
    // materialized at full n x p_right size. Chunk size is governed by the
    // resource policy's row_chunk_target_bytes.
    let policy = ResourcePolicy::default_library();
    let rows_per_chunk = crate::resource::rows_for_target_bytes(
        policy.row_chunk_target_bytes,
        p_left.saturating_add(p_right).max(1),
    );

    let n = weights.len();
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let rows = start..end;
        let xl = left.row_chunk(rows.clone())?;
        let mut xr = right.row_chunk(rows.clone())?;
        for local in 0..xr.nrows() {
            let w = weights[start + local];
            if w != 1.0 {
                for j in 0..p_right {
                    xr[[local, j]] *= w;
                }
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
    fn dim(&self) -> usize {
        self.total_dim
    }

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

pub(crate) fn resolve_custom_family_x_psi_map(
    deriv: &CustomFamilyBlockPsiDerivative,
    n: usize,
    p: usize,
    row_range: Range<usize>,
    label: &str,
    policy: &ResourcePolicy,
) -> Result<PsiDesignMap, String> {
    if row_range.end > n {
        return Err(format!(
            "{label}: row range {}..{} exceeds total rows {n}",
            row_range.start, row_range.end
        ));
    }

    // Prefer operator action when dimensions match.
    if let Some(op) = deriv.implicit_operator.as_ref() {
        if op.n_data() == n && op.p_out() == p {
            return Ok(PsiDesignMap::First {
                action: CustomFamilyPsiDesignAction::from_first_derivative(
                    deriv, n, p, row_range, label,
                )?,
            });
        }
    }

    // Dense fallback guarded by policy.
    if deriv.x_psi.nrows() == n && deriv.x_psi.ncols() == p {
        match policy.derivative_storage_mode {
            DerivativeStorageMode::AnalyticOperatorRequired => {
                if is_zero_array(&deriv.x_psi) {
                    return Ok(PsiDesignMap::Zero {
                        nrows: row_range.end - row_range.start,
                        ncols: p,
                    });
                }
                return Err(format!(
                    "{label}: dense x_psi fallback disabled by AnalyticOperatorRequired"
                ));
            }
            DerivativeStorageMode::MaterializeIfSmall | DerivativeStorageMode::DiagnosticsOnly => {
                let matrix = if row_range.start == 0 && row_range.end == n {
                    Arc::new(deriv.x_psi.clone())
                } else {
                    Arc::new(
                        deriv
                            .x_psi
                            .slice(ndarray::s![row_range.clone(), ..])
                            .to_owned(),
                    )
                };
                return Ok(PsiDesignMap::Dense { matrix });
            }
        }
    }

    // Empty / zero sentinel.
    if deriv.x_psi.nrows() == 0 || deriv.x_psi.ncols() == 0 {
        return Ok(PsiDesignMap::Zero {
            nrows: row_range.end - row_range.start,
            ncols: p,
        });
    }

    Err(format!(
        "{label}: x_psi shape {:?} does not match ({n}, {p})",
        deriv.x_psi.dim()
    ))
}

pub(crate) fn resolve_custom_family_x_psi_psi_map(
    deriv_i: &CustomFamilyBlockPsiDerivative,
    deriv_j: &CustomFamilyBlockPsiDerivative,
    local_j: usize,
    n: usize,
    p: usize,
    row_range: Range<usize>,
    label: &str,
    policy: &ResourcePolicy,
) -> Result<PsiDesignMap, String> {
    if row_range.end > n {
        return Err(format!(
            "{label}: row range {}..{} exceeds total rows {n}",
            row_range.start, row_range.end
        ));
    }

    // Prefer operator action when dimensions match.
    if let Some(op) = deriv_i.implicit_operator.as_ref() {
        if op.n_data() == n && op.p_out() == p {
            let same_group = deriv_i.implicit_group_id.is_some()
                && deriv_i.implicit_group_id == deriv_j.implicit_group_id;
            if !same_group {
                return Ok(PsiDesignMap::Zero {
                    nrows: row_range.end - row_range.start,
                    ncols: p,
                });
            }
            match CustomFamilyPsiSecondDesignAction::from_second_derivative(
                deriv_i,
                deriv_j,
                n,
                p,
                row_range.clone(),
                label,
            )? {
                Some(action) => {
                    return Ok(PsiDesignMap::Second { action });
                }
                None => {
                    return Ok(PsiDesignMap::Zero {
                        nrows: row_range.end - row_range.start,
                        ncols: p,
                    });
                }
            }
        }
    }

    // Dense fallback guarded by policy, reading from the per-second-derivative
    // slot `x_psi_psi[local_j]` if provided.
    if let Some(x_psi_psi) = deriv_i.x_psi_psi.as_ref()
        && let Some(x_ab) = x_psi_psi.get(local_j)
    {
        if x_ab.nrows() == n && x_ab.ncols() == p {
            match policy.derivative_storage_mode {
                DerivativeStorageMode::AnalyticOperatorRequired => {
                    if is_zero_array(x_ab) {
                        return Ok(PsiDesignMap::Zero {
                            nrows: row_range.end - row_range.start,
                            ncols: p,
                        });
                    }
                    return Err(format!(
                        "{label}: dense x_psi_psi fallback disabled by AnalyticOperatorRequired"
                    ));
                }
                DerivativeStorageMode::MaterializeIfSmall
                | DerivativeStorageMode::DiagnosticsOnly => {
                    let matrix = if row_range.start == 0 && row_range.end == n {
                        Arc::new(x_ab.clone())
                    } else {
                        Arc::new(x_ab.slice(ndarray::s![row_range.clone(), ..]).to_owned())
                    };
                    return Ok(PsiDesignMap::Dense { matrix });
                }
            }
        }
        if x_ab.is_empty() {
            return Ok(PsiDesignMap::Zero {
                nrows: row_range.end - row_range.start,
                ncols: p,
            });
        }
        return Err(format!(
            "{label}: x_psi_psi shape {:?} does not match ({n}, {p})",
            x_ab.dim()
        ));
    }

    // No operator, no dense slot: treat as zero.
    Ok(PsiDesignMap::Zero {
        nrows: row_range.end - row_range.start,
        ncols: p,
    })
}

#[derive(Clone)]
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
    /// Pre-build any per-row jet caches the workspace will hand to the
    /// outer-eval directional-derivative path. Called once when the
    /// `compute_dh` / `compute_d2h` closures are wired up at top-level
    /// rayon, *before* the outer ext-coordinate `par_iter` enters. The
    /// alternative — letting the cache materialise lazily on first call
    /// from inside the outer `par_iter` — collapses the build's own
    /// `par_iter` to a single worker (the seven other workers are parked
    /// on the cache's `OnceLock`). Default impl is a no-op for workspaces
    /// with no per-row jet cache.
    ///
    /// Deliberately not called from PIRLS-side workspaces (which never
    /// invoke `directional_derivative_operator` and would pay the prime
    /// cost without ever consuming the cache).
    fn warm_up_outer_caches(&self) -> Result<(), String> {
        Ok(())
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        Ok(None)
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(None)
    }

    fn hessian_matvec(&self, _: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    /// Write-into variant of `hessian_matvec`. The default implementation
    /// delegates to the legacy owned-return form and copies the result into
    /// `out`, providing back-compat without per-impl work. Concrete impls in
    /// the inner-Newton biobank-scale hot path (Bernoulli marginal-slope and
    /// survival marginal-slope) override this to write directly into the
    /// caller-owned buffer, eliminating per-PCG-iter `Array1` allocations.
    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        match self.hessian_matvec(v)? {
            Some(result) => {
                if result.len() != out.len() {
                    return Err(format!(
                        "hessian_matvec_into: result length {} != out length {}",
                        result.len(),
                        out.len()
                    ));
                }
                out.assign(&result);
                Ok(true)
            }
            None => Ok(false),
        }
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
    pub(crate) fn new(len: usize) -> Self {
        Self {
            entries: (0..len).map(|_| Mutex::new(None)).collect(),
            lru: Mutex::new(std::collections::VecDeque::new()),
            limit: len,
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

impl CustomFamilyWarmStart {
    pub(crate) fn compatible_with_rho(&self, rho: &Array1<f64>) -> bool {
        screened_outer_warm_start(Some(&self.inner), rho).is_some()
    }
}

pub struct CustomFamilyJointHyperResult {
    pub objective: f64,
    pub gradient: Array1<f64>,
    pub outer_hessian: crate::solver::outer_strategy::HessianResult,
    pub warm_start: CustomFamilyWarmStart,
    /// `false` when the inner blockwise/Newton solve hit its divergence
    /// early-exit or its max-cycle cap. Envelope-theorem outer gradients
    /// and analytic outer Hessians are valid only at a stationary β̂ —
    /// callers that consume `gradient`/`outer_hessian` MUST gate on this
    /// flag and treat non-converged evaluations as inexact (e.g. let ARC
    /// back off the trust region) rather than feeding pathological
    /// derivatives into the outer optimizer.
    pub inner_converged: bool,
}

pub struct CustomFamilyJointHyperEfsResult {
    pub efs_eval: crate::solver::outer_strategy::EfsEval,
    pub warm_start: CustomFamilyWarmStart,
    /// See [`CustomFamilyJointHyperResult::inner_converged`]. EFS gradients
    /// also assume a stationary inner solve.
    pub inner_converged: bool,
}

struct OuterObjectiveEvalResult {
    objective: f64,
    gradient: Array1<f64>,
    outer_hessian: crate::solver::outer_strategy::HessianResult,
    warm_start: ConstrainedWarmStart,
    inner_converged: bool,
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
        inner_converged: result.inner_converged,
    }
}

struct OwnedDenseOuterHessianOperator {
    matrix: Array2<f64>,
}

impl crate::solver::outer_strategy::OuterHessianOperator for OwnedDenseOuterHessianOperator {
    fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.matrix.ncols() {
            return Err(format!(
                "batched dense outer Hessian matvec length mismatch: got {}, expected {}",
                v.len(),
                self.matrix.ncols()
            ));
        }
        Ok(self.matrix.dot(v))
    }

    fn is_cheap_to_materialize(&self) -> bool {
        true
    }
}

fn custom_family_batched_outer_hessian_operator<F: CustomFamily>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    rho: &Array1<f64>,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    eval_mode: EvalMode,
) -> Result<Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>>, String> {
    if eval_mode != EvalMode::ValueGradientHessian {
        return Ok(None);
    }
    let Some(terms) =
        family.batched_outer_hessian_terms(states, specs, derivative_blocks, rho, workspace)?
    else {
        return Ok(None);
    };
    match terms.outer_hessian {
        crate::solver::outer_strategy::HessianResult::Operator(operator) => Ok(Some(operator)),
        crate::solver::outer_strategy::HessianResult::Analytic(matrix) => {
            Ok(Some(Arc::new(OwnedDenseOuterHessianOperator { matrix })))
        }
        crate::solver::outer_strategy::HessianResult::Unavailable => Ok(None),
    }
}

fn outer_efs_result_to_joint_hyper_efs_result(
    efs_eval: crate::solver::outer_strategy::EfsEval,
    warm_start: ConstrainedWarmStart,
    inner_converged: bool,
) -> CustomFamilyJointHyperEfsResult {
    CustomFamilyJointHyperEfsResult {
        efs_eval,
        warm_start: CustomFamilyWarmStart { inner: warm_start },
        inner_converged,
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
    if family.block_geometry_is_dynamic() {
        for b in 0..specs.len() {
            refresh_single_block_eta(family, specs, states, b)?;
        }
        return Ok(());
    }

    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let refreshed_etas: Vec<Array1<f64>> = (0..specs.len())
        .into_par_iter()
        .map(|b| specs[b].design.matrixvectormultiply(&states[b].beta) + &specs[b].offset)
        .collect();

    for (state, eta) in states.iter_mut().zip(refreshed_etas) {
        state.eta = eta;
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
    let mut out = Array1::<f64>::zeros(working_weights.len());
    ndarray::Zip::from(&mut out)
        .and(working_weights)
        .par_for_each(|o, &wi| *o = if wi <= 0.0 { 0.0 } else { wi.max(minweight) });
    out
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
                // Strict-mode Newton step uses the LM δ-ridge continuation:
                // a single near-zero eigenvalue from numerical noise in
                // H_β should not bounce the entire seed evaluation. The
                // bare strict_solve_spd contract is preserved (still used
                // by other paths and the existing test
                // `pseudo_laplace_path_skips_eigendecomposition_avoiding_nan_crash`);
                // here we pay an O(p³) extra Cholesky attempt when needed
                // to keep adaptive optimization moving.
                let (step, lm_stats) =
                    strict_solve_spd_with_lm_continuation(&lhs_dense, &rhs_step)?;
                if lm_stats.escalations > 0 {
                    log::debug!(
                        "[strict-spd-lm] block={} ({}): δ-ridge continuation succeeded \
                         after {} escalation(s) at δ={:.3e}",
                        ctx.block_idx,
                        ctx.spec.name,
                        lm_stats.escalations,
                        lm_stats.delta_used,
                    );
                }
                step
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

const TOTAL_QUADRATIC_PENALTY_PAR_MIN_BLOCKS: usize = 4;
// Avoid Rayon overhead for a few tiny blocks; this approximates the dense
// mat-vec work in βᵀSβ before splitting independent block penalties.
const TOTAL_QUADRATIC_PENALTY_PAR_MIN_DENSE_WORK: usize = 16_384;

fn total_quadratic_penalty_parallel_worthwhile(
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
) -> bool {
    let n_blocks = states.len().min(s_lambdas.len());
    if n_blocks < TOTAL_QUADRATIC_PENALTY_PAR_MIN_BLOCKS || rayon::current_num_threads() <= 1 {
        return false;
    }

    states
        .iter()
        .zip(s_lambdas.iter())
        .map(|(state, s_lambda)| {
            let p = state.beta.len().min(s_lambda.ncols());
            p.saturating_mul(s_lambda.nrows())
        })
        .try_fold(0usize, |acc, work| {
            let next = acc.saturating_add(work);
            (next < TOTAL_QUADRATIC_PENALTY_PAR_MIN_DENSE_WORK).then_some(next)
        })
        .is_none()
}

fn total_quadratic_penalty(
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> f64 {
    if total_quadratic_penalty_parallel_worthwhile(states, s_lambdas) {
        use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

        states
            .par_iter()
            .zip(s_lambdas.par_iter())
            .map(|(state, s_lambda)| {
                block_quadratic_penalty(&state.beta, s_lambda, ridge, ridge_policy)
            })
            .reduce(|| 0.0, |left, right| left + right)
    } else {
        states
            .iter()
            .zip(s_lambdas.iter())
            .map(|(state, s_lambda)| {
                block_quadratic_penalty(&state.beta, s_lambda, ridge, ridge_policy)
            })
            .sum()
    }
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
        RidgeDeterminantMode::Auto => unreachable!("adaptive determinant mode must resolve"),
        RidgeDeterminantMode::PositivePart => {
            if let Some((row, col, value)) = a
                .indexed_iter()
                .find_map(|((row, col), &value)| (!value.is_finite()).then_some((row, col, value)))
            {
                return Err(format!(
                    "smooth-regularized logdet Hessian contains non-finite entry at ({row}, {col}): {value}"
                ));
            }
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
                Err(eigh_err) => Err(format!(
                    "smooth-regularized logdet eigendecomposition failed: {eigh_err}"
                )),
            }
        }
    }
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

fn resolved_ridge_determinant_mode(ridge_policy: RidgePolicy, dim: usize) -> RidgeDeterminantMode {
    let _ = dim;
    match ridge_policy.determinant_mode {
        RidgeDeterminantMode::Auto => RidgeDeterminantMode::Full,
        mode => mode,
    }
}

fn inverse_spdwith_retry(
    matrix: &Array2<f64>,
    baseridge: f64,
    max_retry: usize,
) -> Result<Array2<f64>, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    for attempt in 0..=max_retry {
        let ridge = if attempt == 0 {
            0.0
        } else {
            baseridge * 10.0_f64.powi((attempt - 1) as i32)
        };
        let mut candidate = sym.clone();
        if ridge > 0.0 {
            for i in 0..candidate.nrows() {
                candidate[[i, i]] += ridge;
            }
        }
        if let Ok(chol) = candidate.cholesky(Side::Lower) {
            let mut ident = Array2::<f64>::eye(candidate.nrows());
            chol.solve_mat_in_place(&mut ident);
            symmetrize_dense_in_place(&mut ident);
            return Ok(ident);
        }
    }
    Err("failed to invert SPD system after Cholesky ridge retries".to_string())
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

fn validate_flat_direction_length(
    direction: &Array1<f64>,
    expected: usize,
    context: &str,
) -> Result<(), String> {
    if direction.len() != expected {
        return Err(format!(
            "{context}: direction length mismatch: got {}, expected {expected}",
            direction.len()
        ));
    }
    Ok(())
}

fn exact_newton_joint_hessian_from_exact_blocks<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
) -> Result<Option<Array2<f64>>, String> {
    let evaluation = family.evaluate(block_states)?;
    if evaluation.blockworking_sets.len() != block_states.len() {
        return Err(format!(
            "exact_newton_joint_hessian default: working-set count {} != block count {}",
            evaluation.blockworking_sets.len(),
            block_states.len()
        ));
    }
    if evaluation
        .blockworking_sets
        .iter()
        .any(|working_set| !matches!(working_set, BlockWorkingSet::ExactNewton { .. }))
    {
        return Ok(None);
    }

    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, (state, working_set)) in block_states
        .iter()
        .zip(evaluation.blockworking_sets.iter())
        .enumerate()
    {
        let p_block = state.beta.len();
        let end = start + p_block;
        let BlockWorkingSet::ExactNewton { hessian, .. } = working_set else {
            unreachable!("non-ExactNewton working sets were filtered above");
        };
        let dense = hessian.to_dense();
        if dense.nrows() != p_block || dense.ncols() != p_block {
            return Err(format!(
                "exact_newton_joint_hessian default: block {block_idx} Hessian shape {}x{} != expected {p_block}x{p_block}",
                dense.nrows(),
                dense.ncols()
            ));
        }
        joint.slice_mut(s![start..end, start..end]).assign(&dense);
        start = end;
    }
    Ok(Some(joint))
}

fn exact_newton_joint_hessian_from_working_sets<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Result<Option<Array2<f64>>, String> {
    if block_states.len() != specs.len() {
        return Err(format!(
            "exact_newton_joint_hessian_with_specs default: block state count {} != spec count {}",
            block_states.len(),
            specs.len()
        ));
    }
    let evaluation = family.evaluate(block_states)?;
    if evaluation.blockworking_sets.len() != block_states.len() {
        return Err(format!(
            "exact_newton_joint_hessian_with_specs default: working-set count {} != block count {}",
            evaluation.blockworking_sets.len(),
            block_states.len()
        ));
    }

    let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, ((state, spec), working_set)) in block_states
        .iter()
        .zip(specs.iter())
        .zip(evaluation.blockworking_sets.iter())
        .enumerate()
    {
        let p_block = spec.design.ncols();
        if state.beta.len() != p_block {
            return Err(format!(
                "exact_newton_joint_hessian_with_specs default: block {block_idx} beta length {} != design cols {p_block}",
                state.beta.len()
            ));
        }
        let end = start + p_block;
        let dense = match working_set {
            BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
            BlockWorkingSet::Diagonal {
                working_weights, ..
            } => spec.design.diag_xtw_x(working_weights)?,
        };
        if dense.nrows() != p_block || dense.ncols() != p_block {
            return Err(format!(
                "exact_newton_joint_hessian_with_specs default: block {block_idx} Hessian shape {}x{} != expected {p_block}x{p_block}",
                dense.nrows(),
                dense.ncols()
            ));
        }
        joint.slice_mut(s![start..end, start..end]).assign(&dense);
        start = end;
    }
    Ok(Some(joint))
}

fn exact_newton_joint_hessian_directional_derivative_from_blocks<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    d_beta_flat: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    validate_flat_direction_length(
        d_beta_flat,
        total,
        "exact_newton_joint_hessian_directional_derivative default",
    )?;
    if !family.exact_newton_joint_hessian_beta_dependent() {
        return Ok(Some(Array2::zeros((total, total))));
    }

    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, state) in block_states.iter().enumerate() {
        let p_block = state.beta.len();
        let end = start + p_block;
        let d_beta_block = d_beta_flat.slice(s![start..end]).to_owned();
        let Some(local) = family.exact_newton_hessian_directional_derivative(
            block_states,
            block_idx,
            &d_beta_block,
        )?
        else {
            return Ok(None);
        };
        if local.nrows() != p_block || local.ncols() != p_block {
            return Err(format!(
                "exact_newton_joint_hessian_directional_derivative default: block {block_idx} dH shape {}x{} != expected {p_block}x{p_block}",
                local.nrows(),
                local.ncols()
            ));
        }
        joint.slice_mut(s![start..end, start..end]).assign(&local);
        start = end;
    }
    Ok(Some(joint))
}

/// Block-diagonal aggregator for the joint second directional derivative.
///
/// Mirrors `exact_newton_joint_hessian_directional_derivative_from_blocks`:
/// for a beta-independent joint Hessian the answer is identically zero;
/// otherwise we ask each block for `D²H_b[u_b, v_b]` via
/// `exact_newton_hessian_second_directional_derivative` and place those
/// per-block contributions on the joint diagonal.
///
/// The previous default returned `Some(zeros)` for beta-independent and
/// `None` (no aggregation at all) for beta-dependent families, silently
/// dropping the per-block `d²H` overrides that families like
/// `OneBlockQuarticExactFamily` provide for the outer Hessian's drift
/// contribution.  Aggregating here mirrors the first-derivative path so
/// outer REML receives the curvature term whenever the per-block
/// `exact_newton_hessian_second_directional_derivative` is implemented.
fn exact_newton_joint_hessiansecond_directional_derivative_from_blocks<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    d_beta_u_flat: &Array1<f64>,
    d_betav_flat: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    validate_flat_direction_length(d_beta_u_flat, total, "joint exact-newton d2H u")?;
    validate_flat_direction_length(d_betav_flat, total, "joint exact-newton d2H v")?;
    if !family.exact_newton_joint_hessian_beta_dependent() {
        return Ok(Some(Array2::zeros((total, total))));
    }

    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, state) in block_states.iter().enumerate() {
        let p_block = state.beta.len();
        let end = start + p_block;
        let u_block = d_beta_u_flat.slice(s![start..end]).to_owned();
        let v_block = d_betav_flat.slice(s![start..end]).to_owned();
        let Some(local) = family.exact_newton_hessian_second_directional_derivative(
            block_states,
            block_idx,
            &u_block,
            &v_block,
        )?
        else {
            return Ok(None);
        };
        if local.nrows() != p_block || local.ncols() != p_block {
            return Err(format!(
                "exact_newton_joint_hessiansecond_directional_derivative default: block {block_idx} d2H shape {}x{} != expected {p_block}x{p_block}",
                local.nrows(),
                local.ncols()
            ));
        }
        joint.slice_mut(s![start..end, start..end]).assign(&local);
        start = end;
    }
    Ok(Some(joint))
}

fn exact_newton_joint_hessian_directional_derivative_from_working_sets<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    d_beta_flat: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    if block_states.len() != specs.len() {
        return Err(format!(
            "exact_newton_joint_hessian_directional_derivative_with_specs default: block state count {} != spec count {}",
            block_states.len(),
            specs.len()
        ));
    }
    let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    validate_flat_direction_length(
        d_beta_flat,
        total,
        "exact_newton_joint_hessian_directional_derivative_with_specs default",
    )?;
    if !family.exact_newton_joint_hessian_beta_dependent() {
        return Ok(Some(Array2::zeros((total, total))));
    }

    let evaluation = family.evaluate(block_states)?;
    if evaluation.blockworking_sets.len() != block_states.len() {
        return Err(format!(
            "exact_newton_joint_hessian_directional_derivative_with_specs default: working-set count {} != block count {}",
            evaluation.blockworking_sets.len(),
            block_states.len()
        ));
    }

    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, ((state, spec), working_set)) in block_states
        .iter()
        .zip(specs.iter())
        .zip(evaluation.blockworking_sets.iter())
        .enumerate()
    {
        let p_block = spec.design.ncols();
        let end = start + p_block;
        let d_beta_block = d_beta_flat.slice(s![start..end]).to_owned();
        let local = match working_set {
            BlockWorkingSet::ExactNewton { .. } => family
                .exact_newton_hessian_directional_derivative(
                    block_states,
                    block_idx,
                    &d_beta_block,
                )?,
            BlockWorkingSet::Diagonal {
                working_weights, ..
            } => {
                let mut d_eta = spec.design.apply(&d_beta_block);
                let mut geometry_correction = Array2::<f64>::zeros((p_block, p_block));
                if let Some(geometry) = family.block_geometry_directional_derivative(
                    block_states,
                    block_idx,
                    spec,
                    &d_beta_block,
                )? {
                    if geometry.d_offset.len() != d_eta.len() {
                        return Err(format!(
                            "exact_newton_joint_hessian_directional_derivative_with_specs default: block {block_idx} geometry offset derivative length {} != eta length {}",
                            geometry.d_offset.len(),
                            d_eta.len()
                        ));
                    }
                    d_eta += &geometry.d_offset;
                    if let Some(d_design) = geometry.d_design {
                        if d_design.nrows() != spec.design.nrows() || d_design.ncols() != p_block {
                            return Err(format!(
                                "exact_newton_joint_hessian_directional_derivative_with_specs default: block {block_idx} d_design shape {}x{} != expected {}x{}",
                                d_design.nrows(),
                                d_design.ncols(),
                                spec.design.nrows(),
                                p_block
                            ));
                        }
                        d_eta += &d_design.dot(&state.beta);

                        let x_dense = spec.design.to_dense();
                        let mut weighted_x = x_dense.clone();
                        let mut weighted_dx = d_design.clone();
                        ndarray::Zip::from(weighted_x.rows_mut())
                            .and(weighted_dx.rows_mut())
                            .and(working_weights.view())
                            .for_each(|mut wx_row, mut wdx_row, &wi| {
                                wx_row.mapv_inplace(|value| value * wi);
                                wdx_row.mapv_inplace(|value| value * wi);
                            });
                        geometry_correction += &fast_atb(&d_design, &weighted_x);
                        geometry_correction += &fast_atb(&x_dense, &weighted_dx);
                    }
                }
                family
                    .diagonalworking_weights_directional_derivative(
                        block_states,
                        block_idx,
                        &d_eta,
                    )?
                    .map(|dw| {
                        let mut local = spec.design.diag_xtw_x(&dw)?;
                        local += &geometry_correction;
                        Ok::<Array2<f64>, String>(local)
                    })
                    .transpose()?
            }
        };
        let Some(local) = local else {
            return Ok(None);
        };
        if local.nrows() != p_block || local.ncols() != p_block {
            return Err(format!(
                "exact_newton_joint_hessian_directional_derivative_with_specs default: block {block_idx} dH shape {}x{} != expected {p_block}x{p_block}",
                local.nrows(),
                local.ncols()
            ));
        }
        joint.slice_mut(s![start..end, start..end]).assign(&local);
        start = end;
    }
    Ok(Some(joint))
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
        /// Write-into matvec used by the inner-Newton PCG hot path so the
        /// matvec result no longer allocates an `Array1<f64>` per CG iter.
        /// At biobank scale (~6400 inner CG iters per outer iter, p~200) this
        /// removes thousands of small Vec<f64> allocations from the tightest
        /// loop. Wired from `workspace.hessian_matvec_into`.
        apply_into: Arc<dyn Fn(&Array1<f64>, &mut Array1<f64>) -> Result<(), String> + Send + Sync>,
        diagonal: Array1<f64>,
    },
}

const EXACT_JOINT_HESSIAN_DENSE_MAX_BYTES: usize = 512 * 1024 * 1024;

fn exact_joint_hessian_dense_bytes(total: usize) -> Result<usize, String> {
    total
        .checked_mul(total)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| format!("joint Hessian dense byte count overflow for dim={total}"))
}

fn ensure_exact_joint_hessian_dense_budget(total: usize, context: &str) -> Result<(), String> {
    let bytes = exact_joint_hessian_dense_bytes(total)?;
    if bytes > EXACT_JOINT_HESSIAN_DENSE_MAX_BYTES {
        return Err(format!(
            "{context}: exact dense joint Hessian requires {:.2} GiB for dim={total}, \
             exceeding the {:.2} GiB cap; refusing approximate determinant algebra",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            EXACT_JOINT_HESSIAN_DENSE_MAX_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
        ));
    }
    Ok(())
}

struct JointHessianBundle<'a> {
    source: JointHessianSource,
    beta_flat: Array1<f64>,
    compute_dh: Box<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + 'a>,
    compute_d2h:
        Box<dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + 'a>,
    owned_compute_dh:
        Option<Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>>,
    owned_compute_d2h: Option<
        Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
                + Send
                + Sync,
        >,
    >,
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
            ensure_exact_joint_hessian_dense_budget(total, context)?;
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
    if let Some(mut hessian) = workspace.hessian_dense()? {
        if hessian.nrows() != total || hessian.ncols() != total {
            return Err(format!(
                "{context}: dense Hessian shape mismatch: got {}x{}, expected {total}x{total}",
                hessian.nrows(),
                hessian.ncols()
            ));
        }
        if hessian.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "{context}: dense Hessian contains non-finite values"
            ));
        }
        symmetrize_dense_in_place(&mut hessian);
        return Ok(Some(JointHessianSource::Dense(hessian)));
    }

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
    let workspace_apply_into = Arc::clone(workspace);
    let context_apply: Arc<str> = Arc::from(context);
    let context_apply_into = Arc::clone(&context_apply);
    Ok(Some(JointHessianSource::Operator {
        apply: Arc::new(move |v: &Array1<f64>| {
            if v.len() != total {
                return Err(format!(
                    "{}: operator input length mismatch: got {}, expected {total}",
                    &*context_apply,
                    v.len()
                ));
            }
            let Some(out) = workspace_apply.hessian_matvec(v)? else {
                return Err("joint exact-newton operator matvec unavailable".to_string());
            };
            if out.len() != total {
                return Err(format!(
                    "{}: operator matvec length mismatch: got {}, expected {total}",
                    &*context_apply,
                    out.len()
                ));
            }
            if out.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "{}: operator matvec returned non-finite values",
                    &*context_apply
                ));
            }
            Ok(out)
        }),
        apply_into: Arc::new(move |v: &Array1<f64>, out: &mut Array1<f64>| {
            if v.len() != total || out.len() != total {
                return Err(format!(
                    "{}: operator input/output length mismatch: v={} out={} expected={total}",
                    &*context_apply_into,
                    v.len(),
                    out.len()
                ));
            }
            if !workspace_apply_into.hessian_matvec_into(v, out)? {
                return Err("joint exact-newton operator matvec unavailable".to_string());
            }
            if out.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "{}: operator matvec returned non-finite values",
                    &*context_apply_into
                ));
            }
            Ok(())
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
    options: &BlockwiseFitOptions,
    preferred_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Result<Option<JointHessianBundle<'a>>, String> {
    // Path 1: exact Newton joint Hessian (preferred).
    let beta_flat = flatten_state_betas(block_states, specs);
    let synced = Arc::new(synchronized_states_from_flat_beta(
        family,
        specs,
        block_states,
        &beta_flat,
    )?);
    let hessian_workspace = match preferred_workspace {
        Some(workspace) => Some(workspace),
        None => family.exact_newton_joint_hessian_workspace_with_options(
            synced.as_ref(),
            specs,
            options,
        )?,
    };
    // Outer-eval entry: prime any per-row jet caches the workspace will hand
    // to the directional-derivative path. Runs at top-level rayon (we are
    // outside the ext-coord `par_iter` here), so the cache build's own
    // `par_iter` enjoys full thread-pool parallelism. PIRLS-side workspace
    // construction skips this priming because PIRLS never invokes
    // `directional_derivative_operator`.
    if let Some(workspace) = hessian_workspace.as_ref() {
        workspace.warm_up_outer_caches()?;
    }
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
            hessian_workspace.clone(),
        ));
        let owned_compute_dh = exact_newton_dh_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        );
        let owned_compute_d2h = exact_newton_d2h_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        );
        return Ok(Some(JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_d2h,
            owned_compute_dh: Some(owned_compute_dh),
            owned_compute_d2h: Some(owned_compute_d2h),
            rho_curvature_scale: curvature.rho_curvature_scale,
            hessian_logdet_correction: curvature.hessian_logdet_correction,
        }));
    }
    let exact_joint_source = if let Some(workspace) = hessian_workspace.as_ref() {
        exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total,
            "joint exact-newton operator mismatch in outer gradient",
        )?
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
            hessian_workspace.clone(),
        ));
        let owned_compute_dh = exact_newton_dh_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        );
        let owned_compute_d2h = exact_newton_d2h_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        );
        return Ok(Some(JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_d2h,
            owned_compute_dh: Some(owned_compute_dh),
            owned_compute_d2h: Some(owned_compute_d2h),
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
        let family_owned = family.clone();
        let states_owned = block_states.to_vec();
        let specs_owned = specs.to_vec();
        let owned_compute_dh =
            Arc::new(
                move |v_k: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                    match family_owned
                        .joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
                            &states_owned,
                            &specs_owned,
                            v_k,
                        )? {
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
        let family_owned = family.clone();
        let states_owned = block_states.to_vec();
        let specs_owned = specs.to_vec();
        let owned_compute_d2h = Arc::new(
            move |u: &Array1<f64>, v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                match family_owned
                    .joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
                        &states_owned,
                        &specs_owned,
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
            owned_compute_dh: Some(owned_compute_dh),
            owned_compute_d2h: Some(owned_compute_d2h),
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

/// Statistics about a Levenberg-Marquardt-style δ-ridge SPD continuation.
/// Recorded by `strict_solve_spd_with_lm_continuation` and surfaced for
/// diagnostics — a recurring need for nontrivial ridges signals fragile
/// curvature that the controller may need to escalate.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StrictSpdLmStats {
    /// δ value finally used (0.0 means the bare strict solve succeeded).
    pub(crate) delta_used: f64,
    /// Number of escalations performed before Cholesky succeeded.
    pub(crate) escalations: usize,
}

/// Strict-mode SPD solve with internal Levenberg-Marquardt δ-ridge
/// continuation: solves `(H + δI) x = b` with δ escalated geometrically
/// until the Cholesky succeeds.  The bare `strict_solve_spd` is unchanged —
/// callers that need strict semantics keep them.  Callers that want
/// fail-soft Newton on a fragile geometry (e.g. spatial-adaptive seed
/// evaluation) use this wrapper to avoid bouncing the entire seed on a
/// numerically-indefinite block.
///
/// Schedule: δ₀ = max(ε · ‖H‖₁ / p, 1e-12); growth ×10 per step; capped
/// at MAX_ESCALATIONS escalations.  The cap prevents runaway curvature
/// from producing arbitrary ridges; if the cap is hit, the bare strict
/// error propagates so the caller can route to a different optimization
/// path (e.g. sparse/gradient-only standard REML at full data).
pub(crate) fn strict_solve_spd_with_lm_continuation(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Result<(Array1<f64>, StrictSpdLmStats), String> {
    const MAX_ESCALATIONS: usize = 16;
    const RIDGE_GROWTH: f64 = 10.0;

    if let Ok(solution) = strict_solve_spd(matrix, rhs) {
        return Ok((solution, StrictSpdLmStats::default()));
    }

    let p = matrix.nrows();
    if p == 0 {
        return Ok((Array1::<f64>::zeros(0), StrictSpdLmStats::default()));
    }
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let trace_scale = (0..p).map(|i| sym[[i, i]].abs()).sum::<f64>() / (p as f64);
    let delta0 = (f64::EPSILON * trace_scale.max(1.0)).max(1e-12);

    let mut delta = delta0;
    for escalation in 1..=MAX_ESCALATIONS {
        let mut ridged = sym.clone();
        for i in 0..p {
            ridged[[i, i]] += delta;
        }
        match ridged.cholesky(Side::Lower) {
            Ok(chol) => {
                return Ok((
                    chol.solvevec(rhs),
                    StrictSpdLmStats {
                        delta_used: delta,
                        escalations: escalation,
                    },
                ));
            }
            Err(_) => {
                delta *= RIDGE_GROWTH;
            }
        }
    }

    // δ-ridge schedule exhausted; fall back to rank-aware eigen-floor solve.
    // Floors every eigenvalue at `eps_floor = 1e-12 · max|λ|` so the resulting
    // step is `Q diag(1/Λ̃) Qᵀ rhs` — well-conditioned modes solved exactly,
    // rank-deficient directions resolved with controlled curvature. Mirrors
    // the eigen-floor fallback in `strict_inverse_spd_with_lm_continuation`
    // and gives the pilot a warm geometry to hand to operator Newton/ARC
    // instead of bouncing all the way out to a cold full-data run.
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(|e| {
        format!(
            "strict pseudo-laplace SPD solve failed even with LM δ-ridge continuation \
             (escalated {MAX_ESCALATIONS} times to δ={delta:.3e}, trace_scale={trace_scale:.3e}); \
             eigen-floor fallback also failed: {e}"
        )
    })?;
    let max_abs_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let eps_floor = (1e-12 * max_abs_eval).max(1e-300);
    // x = Q diag(1/Λ̃) Qᵀ rhs.
    let mut q_t_rhs = Array1::<f64>::zeros(p);
    for k in 0..p {
        let mut acc = 0.0;
        for i in 0..p {
            acc += evecs[[i, k]] * rhs[i];
        }
        q_t_rhs[k] = acc;
    }
    for k in 0..p {
        q_t_rhs[k] /= evals[k].max(eps_floor);
    }
    let mut x = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut acc = 0.0;
        for k in 0..p {
            acc += evecs[[i, k]] * q_t_rhs[k];
        }
        x[i] = acc;
    }
    Ok((
        x,
        StrictSpdLmStats {
            delta_used: delta,
            escalations: MAX_ESCALATIONS + 1,
        },
    ))
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

/// Strict-mode SPD inverse with the same Levenberg-Marquardt δ-ridge
/// continuation as [`strict_solve_spd_with_lm_continuation`]: when bare
/// Cholesky on `H` rejects, factor `H + δI` with δ escalated geometrically,
/// then invert.  If the cap is exhausted, fall back to a symmetric
/// eigendecomposition with an eigenvalue floor `eps_floor = 1e-12 · max|λ|`
/// applied uniformly, so the reconstructed inverse equals the
/// pseudo-Laplace inverse on well-conditioned modes plus a floor-controlled
/// regularization on rank-deficient directions.
///
/// Mirrors [`strict_solve_spd_with_lm_continuation`] for the inverse path:
/// preserves strict semantics (a valid SPD inverse with bounded, recorded
/// perturbation) while preventing fragile pilot Hessians from collapsing
/// the spatial-adaptive seed and routing the optimizer into a cold
/// full-data run.
pub(crate) fn strict_inverse_spd_with_lm_continuation(
    matrix: &Array2<f64>,
) -> Result<(Array2<f64>, StrictSpdLmStats), String> {
    const MAX_ESCALATIONS: usize = 16;
    const RIDGE_GROWTH: f64 = 10.0;

    if let Ok(inv) = strict_inverse_spd(matrix) {
        return Ok((inv, StrictSpdLmStats::default()));
    }

    let p = matrix.nrows();
    if p == 0 {
        return Ok((Array2::<f64>::zeros((0, 0)), StrictSpdLmStats::default()));
    }
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let trace_scale = (0..p).map(|i| sym[[i, i]].abs()).sum::<f64>() / (p as f64);
    let delta0 = (f64::EPSILON * trace_scale.max(1.0)).max(1e-12);

    let mut delta = delta0;
    for escalation in 1..=MAX_ESCALATIONS {
        let mut ridged = sym.clone();
        for i in 0..p {
            ridged[[i, i]] += delta;
        }
        if let Ok(chol) = ridged.cholesky(Side::Lower) {
            let mut ident = Array2::<f64>::eye(p);
            chol.solve_mat_in_place(&mut ident);
            symmetrize_dense_in_place(&mut ident);
            return Ok((
                ident,
                StrictSpdLmStats {
                    delta_used: delta,
                    escalations: escalation,
                },
            ));
        }
        delta *= RIDGE_GROWTH;
    }

    // δ-ridge schedule exhausted; fall back to eigen-floor reconstruction.
    // Clamp every eigenvalue from below by eps_floor = 1e-12 · max|λ|, then
    // reconstruct V · diag(1/λ_clamped) · Vᵀ. Preserves all well-conditioned
    // modes exactly and replaces rank-deficient directions with a
    // controlled high-curvature pseudoinverse.
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(|e| {
        format!(
            "strict pseudo-laplace SPD inverse failed even with LM δ-ridge continuation \
             (escalated {MAX_ESCALATIONS} times to δ={delta:.3e}, trace_scale={trace_scale:.3e}); \
             eigen-floor fallback also failed: {e}"
        )
    })?;
    let max_abs_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let eps_floor = (1e-12 * max_abs_eval).max(1e-300);
    let mut inv = Array2::<f64>::zeros((p, p));
    for (k, &ev) in evals.iter().enumerate() {
        let ev_clamped = ev.max(eps_floor);
        let inv_ev = 1.0 / ev_clamped;
        for i in 0..p {
            let vi = evecs[[i, k]];
            for j in 0..p {
                inv[[i, j]] += inv_ev * vi * evecs[[j, k]];
            }
        }
    }
    symmetrize_dense_in_place(&mut inv);
    Ok((
        inv,
        StrictSpdLmStats {
            delta_used: delta,
            escalations: MAX_ESCALATIONS + 1,
        },
    ))
}

fn strict_logdet_spd(matrix: &Array2<f64>) -> Result<f64, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD logdet failed".to_string())?;
    Ok(2.0 * chol.diag().mapv(f64::ln).sum())
}

/// Strict-mode logdet with the same Levenberg-Marquardt δ-ridge continuation
/// schedule as [`strict_solve_spd_with_lm_continuation`]: when the bare
/// Cholesky on `H` rejects, we factor `H + δI` with δ escalated geometrically
/// until success.  Returns `log|H + δI|` and the δ used.
///
/// The mathematical content is the regularized pseudo-Laplace logdet,
/// `log|H + δI|`, not the bare `log|H|` — but this is precisely the quantity
/// the controller needs at indefinite seeds to make a step decision.  Without
/// continuation, every seed where H_β has a marginally-non-positive eigenvalue
/// is rejected, the spatial-adaptive pilot loses its entire seed budget, and
/// the controller fall-opens into a full-data path that was never gated for
/// it (the Matern adaptive seed-failure → 14.56 GiB OOM in the recovery
/// branch).  The δ that finally factors is recorded in `StrictSpdLmStats`
/// so the outer controller can detect a recurring need for nontrivial ridges
/// and react (e.g. tighten step length, fall through to a sparse path, or
/// surface a structural diagnostic to the user).
pub(crate) fn strict_logdet_spd_with_lm_continuation(
    matrix: &Array2<f64>,
) -> Result<(f64, StrictSpdLmStats), String> {
    const MAX_ESCALATIONS: usize = 16;
    const RIDGE_GROWTH: f64 = 10.0;

    if let Ok(logdet) = strict_logdet_spd(matrix) {
        return Ok((logdet, StrictSpdLmStats::default()));
    }

    let p = matrix.nrows();
    if p == 0 {
        return Ok((0.0, StrictSpdLmStats::default()));
    }
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let trace_scale = (0..p).map(|i| sym[[i, i]].abs()).sum::<f64>() / (p as f64);
    let delta0 = (f64::EPSILON * trace_scale.max(1.0)).max(1e-12);

    let mut delta = delta0;
    for escalation in 1..=MAX_ESCALATIONS {
        let mut ridged = sym.clone();
        for i in 0..p {
            ridged[[i, i]] += delta;
        }
        match ridged.cholesky(Side::Lower) {
            Ok(chol) => {
                let logdet = 2.0 * chol.diag().mapv(f64::ln).sum();
                return Ok((
                    logdet,
                    StrictSpdLmStats {
                        delta_used: delta,
                        escalations: escalation,
                    },
                ));
            }
            Err(_) => {
                delta *= RIDGE_GROWTH;
            }
        }
    }
    // δ-ridge schedule exhausted; eigen-floor fallback.
    // `log|C̃| = Σ log Λ̃_i` with `Λ̃_i = max(Λ_i, ε λ_max)`.
    let (evals, _) = FaerEigh::eigh(&sym, Side::Lower).map_err(|e| {
        format!(
            "strict pseudo-laplace SPD logdet failed even with LM δ-ridge continuation \
             (escalated {MAX_ESCALATIONS} times to δ={delta:.3e}, trace_scale={trace_scale:.3e}); \
             eigen-floor fallback also failed: {e}"
        )
    })?;
    let max_abs_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let eps_floor = (1e-12 * max_abs_eval).max(1e-300);
    let logdet: f64 = evals.iter().map(|&ev| ev.max(eps_floor).ln()).sum();
    Ok((
        logdet,
        StrictSpdLmStats {
            delta_used: delta,
            escalations: MAX_ESCALATIONS + 1,
        },
    ))
}

fn strict_logdet_spd_with_semidefinite_option(
    matrix: &Array2<f64>,
    allow_semidefinite: bool,
    accumulation_depth: usize,
) -> Result<f64, String> {
    if allow_semidefinite {
        let mut sym = matrix.clone();
        symmetrize_dense_in_place(&mut sym);
        let (evals, _) = FaerEigh::eigh(&sym, Side::Lower)
            .map_err(|e| format!("strict pseudo-laplace PSD eigendecomposition failed: {e}"))?;
        let p = sym.nrows();
        let max_abs_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
        // Bauer-Fike: |δσ| ≤ p·‖δH‖_∞; n-term fma roundoff gives ‖δH‖_∞ ≤ ε·n·‖H‖,
        // so σ_noise ≤ ε·n·p·‖H‖₂. Tenfold slack absorbs sign cancellations,
        // and a 100·ε floor handles the ‖H‖→0 limit.
        let eps = f64::EPSILON;
        let eps_np = eps * (accumulation_depth as f64) * (p as f64);
        let tol = (10.0 * eps_np * max_abs_eval).max(100.0 * eps);
        if evals.iter().any(|&ev| ev < -tol) {
            let min_eval = evals.iter().copied().fold(f64::INFINITY, f64::min);
            let below = evals.iter().filter(|&&ev| ev < -tol).count();
            return Err(format!(
                "strict pseudo-laplace SPD solve failed: {below} eigenvalue(s) below -tol \
                 (min(λ)={min_eval:.6e}, max|λ|={max_abs_eval:.6e}, tol={tol:.6e}, εnp={eps_np:.6e})"
            ));
        }
        let logdet = evals
            .iter()
            .copied()
            .filter(|&ev| ev > tol)
            .map(f64::ln)
            .sum();
        return Ok(logdet);
    }
    let (logdet, stats) = strict_logdet_spd_with_lm_continuation(matrix)?;
    if stats.escalations > 0 {
        log::debug!(
            "[strict-spd] logdet δ-ridge continuation: δ={:.3e}, escalations={}, p={}",
            stats.delta_used,
            stats.escalations,
            matrix.nrows(),
        );
    }
    Ok(logdet)
}

fn pinv_positive_part(matrix: &Array2<f64>, ridge_floor: f64) -> Result<Array2<f64>, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let (eigenvalues, eigenvectors) = sym
        .eigh(Side::Lower)
        .map_err(|e| format!("positive-part covariance eigendecomposition failed: {e}"))?;
    let max_abs_eigenvalue = eigenvalues.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let tol = (max_abs_eigenvalue * 1e-12).max(ridge_floor.max(1e-14));
    let p = matrix.nrows();
    let mut pinv = Array2::<f64>::zeros((p, p));
    for (k, &ev) in eigenvalues.iter().enumerate() {
        if ev <= tol {
            continue;
        }
        let inv_ev = 1.0 / ev;
        for i in 0..p {
            let vi = eigenvectors[[i, k]];
            for j in 0..p {
                pinv[[i, j]] += inv_ev * vi * eigenvectors[[j, k]];
            }
        }
    }
    symmetrize_dense_in_place(&mut pinv);
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
    crate::solver::outer_strategy::DeclaredHessianForm,
) {
    use crate::solver::outer_strategy::{DeclaredHessianForm, Derivative};

    let order = family.exact_outer_derivative_order(specs, options);
    let gradient = if order.has_gradient() {
        Derivative::Analytic
    } else {
        Derivative::Unavailable
    };
    // The analytic outer Hessian is routed to ARC whenever the realized family
    // exposes second-order calculus. Matrix-free Hessian support is a
    // representation capability used by the evaluator; it must not be hidden
    // from the outer optimizer by a cost-based first-order policy.
    let hessian = if options.use_outer_hessian
        && include_exact_newton_logdet_h(family, options)
        && order.has_hessian()
    {
        DeclaredHessianForm::Either
    } else {
        DeclaredHessianForm::Unavailable
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
    blockwise_logdet_terms_with_workspace(family, specs, states, block_log_lambdas, options, None)
}

fn blockwise_logdet_terms_with_workspace<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    preferred_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Result<(f64, f64), String> {
    let include_logdet_h = include_exact_newton_logdet_h(family, options);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    if !include_logdet_h && !include_logdet_s {
        return Ok((0.0, 0.0));
    }
    let strict_spd = use_exact_newton_strict_spd(family);
    let allow_semidefinite = strict_spd && family.exact_newton_allows_semidefinitehessian();
    refresh_all_block_etas(family, specs, states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let compute_block_logdet_term = |b: usize| -> Result<(Array2<f64>, f64), String> {
        let spec = &specs[b];
        let (start, end) = ranges[b];
        let p = end - start;
        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        let block_logdet = if include_logdet_s {
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
            match s_for_logdet.eigh(faer::Side::Lower) {
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
            }
        } else {
            0.0
        };
        Ok((s_lambda, block_logdet))
    };

    // Per-block penalty assembly and eigendecomposition are independent.
    // Use rayon only from non-rayon callers so inner operator/eigendecomp work
    // does not nest under an existing worker. Collecting an indexed range into
    // a Vec preserves block order; totals are accumulated sequentially below
    // to keep floating-point summation deterministic.
    let block_terms: Vec<Result<(Array2<f64>, f64), String>> =
        if specs.len() > 1 && rayon::current_thread_index().is_none() {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..specs.len())
                .into_par_iter()
                .map(compute_block_logdet_term)
                .collect()
        } else {
            (0..specs.len()).map(compute_block_logdet_term).collect()
        };
    let mut s_lambdas = Vec::with_capacity(block_terms.len());
    let mut penalty_logdet_s_total = 0.0;
    for block_term in block_terms {
        let (s_lambda, block_logdet) = block_term?;
        s_lambdas.push(s_lambda);
        penalty_logdet_s_total += block_logdet;
    }
    if !include_logdet_h {
        return Ok((0.0, penalty_logdet_s_total));
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
            strict_logdet_spd_with_semidefinite_option(
                &h_joint,
                allow_semidefinite,
                joint_observation_count(states),
            )?
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
    let exact_joint_source = if let Some(workspace) = preferred_workspace.as_ref() {
        exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total,
            "joint exact-newton operator mismatch in logdet terms",
        )?
    } else if !strict_spd && use_joint_matrix_free_path(total, joint_observation_count(states)) {
        family
            .exact_newton_joint_hessian_workspace_with_options(states, specs, options)?
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
    if let Some(source) = exact_joint_source {
        // Exact determinant of H + S_λ for operator-backed coefficient Hessians.
        //
        // The REML gradient and Hessian use analytic trace identities such as
        // ∂ log|A(θ)| = tr(A⁻¹ A_θ).  Mixing an approximate determinant with
        // exact traces violates that identity and gives ARC a Hessian for a
        // different objective.  Materializing the coefficient Hessian by
        // canonical-basis HVPs keeps the objective/derivative pair exact.  At
        // biobank CTN scale `total` is a few hundred, so this is sub-MiB; the
        // materializer below refuses oversized systems before allocation.
        let mut h_joint = materialize_joint_hessian_source(
            &source,
            total,
            "joint exact-newton operator dense logdet materialization",
        )?;
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, s_lambda);
        }
        let logdet_h_total = if strict_spd {
            strict_logdet_spd_with_semidefinite_option(
                &h_joint,
                allow_semidefinite,
                joint_observation_count(states),
            )?
        } else {
            stable_logdet_with_ridge_policy(&h_joint, options.ridge_floor, options.ridge_policy)?
        };
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
            strict_logdet_spd_with_semidefinite_option(
                &h_joint,
                allow_semidefinite,
                joint_observation_count(states),
            )?
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
            strict_logdet_spd_with_semidefinite_option(
                &h,
                allow_semidefinite,
                joint_observation_count(states),
            )?
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

#[derive(Clone, Copy, Debug)]
struct JointTrustRegionUpdate {
    rho: f64,
    radius: f64,
    accepted: bool,
}

fn update_joint_trust_region_radius(
    old_radius: f64,
    step_norm: f64,
    actual_reduction: f64,
    predicted_reduction: f64,
) -> JointTrustRegionUpdate {
    let rho = if predicted_reduction > 0.0 && predicted_reduction.is_finite() {
        actual_reduction / predicted_reduction
    } else {
        f64::NEG_INFINITY
    };
    let accepted = rho.is_finite() && rho > 0.0 && actual_reduction > 0.0;
    let mut radius = old_radius;
    if !accepted || rho < 0.25 {
        radius *= 0.25;
    } else if rho > 0.75 && step_norm >= 0.99 * old_radius {
        radius *= 2.0;
    }
    if !radius.is_finite() || radius <= 0.0 {
        radius = 1.0e-12;
    }
    JointTrustRegionUpdate {
        rho,
        radius: radius.clamp(1.0e-12, 1.0e6),
        accepted,
    }
}

fn joint_trust_region_step_norm(delta: &Array1<f64>) -> f64 {
    delta.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn truncate_joint_step_to_radius(delta: &mut Array1<f64>, radius: f64) -> f64 {
    let norm = joint_trust_region_step_norm(delta);
    if norm.is_finite() && norm > radius && radius > 0.0 {
        delta.mapv_inplace(|v| v * (radius / norm));
        radius
    } else {
        norm
    }
}

fn apply_joint_penalized_hessian_into(
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    vector: &Array1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), String> {
    match source {
        JointHessianSource::Dense(h_joint) => {
            crate::faer_ndarray::fast_av_view_into(h_joint, vector, out.view_mut());
        }
        JointHessianSource::Operator { apply_into, .. } => {
            apply_into(vector, out)?;
        }
    }
    let mut penalty = Array1::<f64>::zeros(vector.len());
    apply_joint_block_penalty_into(ranges, s_lambdas, vector, diagonal_ridge, &mut penalty);
    *out += &penalty;
    Ok(())
}

fn joint_quadratic_predicted_reduction(
    rhs: &Array1<f64>,
    hpen_delta: &Array1<f64>,
    delta: &Array1<f64>,
) -> f64 {
    rhs.dot(delta) - 0.5 * delta.dot(hpen_delta)
}

fn joint_preconditioned_descent_delta(
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let base_diagonal = match source {
        JointHessianSource::Dense(h_joint) => h_joint.diag().to_owned(),
        JointHessianSource::Operator { diagonal, .. } => diagonal.clone(),
    };
    let preconditioner =
        joint_penalty_preconditioner_diag(&base_diagonal, ranges, s_lambdas, diagonal_ridge);
    let mut delta = rhs / &preconditioner;
    if !delta.iter().all(|v| v.is_finite()) || rhs.dot(&delta) <= 0.0 {
        delta.assign(rhs);
    }
    let directional = rhs.dot(&delta);
    if directional.is_finite() && directional > 0.0 {
        let mut hpen_delta = Array1::<f64>::zeros(rhs.len());
        apply_joint_penalized_hessian_into(
            source,
            ranges,
            s_lambdas,
            diagonal_ridge,
            &delta,
            &mut hpen_delta,
        )?;
        let curvature = delta.dot(&hpen_delta);
        if curvature.is_finite() && curvature > 0.0 {
            let alpha = (directional / curvature).clamp(1.0e-12, 1.0);
            delta.mapv_inplace(|v| alpha * v);
        }
    }
    Ok(delta)
}

fn joint_line_search_log_likelihood<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    states: &[ParameterBlockState],
    prefer_workspace: bool,
) -> Result<(f64, Option<Arc<dyn ExactNewtonJointHessianWorkspace>>), String> {
    if prefer_workspace
        && let Some((log_likelihood, workspace)) =
            family.joint_line_search_log_likelihood_workspace(states, specs, options)?
    {
        return Ok((log_likelihood, Some(workspace)));
    }
    if (!family.supports_log_likelihood_early_exit() || options.early_exit_threshold.is_none())
        && prefer_workspace
        && family.inner_joint_workspace_log_likelihood_available(specs)
        && let Some(workspace) =
            family.exact_newton_joint_hessian_workspace_with_options(states, specs, options)?
    {
        if let Some(log_likelihood) = workspace.joint_log_likelihood_evaluation()? {
            return Ok((log_likelihood, Some(workspace)));
        }
    }
    family
        .log_likelihood_only_with_options(states, options)
        .map(|log_likelihood| (log_likelihood, None))
}

fn coefficient_line_search_options(
    options: &BlockwiseFitOptions,
    early_exit_threshold: f64,
) -> BlockwiseFitOptions {
    let mut line_search_options = options.clone();
    line_search_options.outer_score_subsample = None;
    line_search_options.early_exit_threshold = Some(early_exit_threshold);
    line_search_options
}

type JointGradientLoad = (
    f64,
    Option<Array1<f64>>,
    Option<FamilyEvaluation>,
    Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
);

fn load_joint_gradient_evaluation<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    states: &[ParameterBlockState],
    prefer_workspace: bool,
    preferred_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Result<JointGradientLoad, String> {
    let workspace = match preferred_workspace {
        Some(workspace) => Some(workspace),
        None if prefer_workspace && family.inner_joint_workspace_gradient_available(specs) => {
            family.exact_newton_joint_hessian_workspace_with_options(states, specs, options)?
        }
        None => None,
    };
    if let Some(workspace_ref) = workspace.as_ref()
        && let Some(joint_eval) = workspace_ref.joint_gradient_evaluation()?
    {
        return Ok((
            joint_eval.log_likelihood,
            Some(joint_eval.gradient),
            None,
            Some(Arc::clone(workspace_ref)),
        ));
    }
    if let Some(joint_eval) = family.exact_newton_joint_gradient_evaluation(states, specs)? {
        return Ok((
            joint_eval.log_likelihood,
            Some(joint_eval.gradient),
            None,
            workspace,
        ));
    }
    let eval = family.evaluate(states)?;
    let log_likelihood = eval.log_likelihood;
    let gradient = exact_newton_joint_gradient_from_eval(&eval, specs)?;
    Ok((log_likelihood, gradient, Some(eval), workspace))
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
    let matrix_free_joint_requested = use_joint_matrix_free_path(total_joint_p, total_joint_n);
    let has_workspace_source = family.inner_coefficient_hessian_hvp_available(specs);
    let has_joint_exacthessian = if has_workspace_source {
        true
    } else {
        family.exact_newton_joint_hessian(&states)?.is_some()
    };
    // Multi-block families have always taken the joint path when an exact
    // joint Hessian is available. Single-block families also take it when a
    // coefficient-Hessian workspace is wired; dense vs. operator form is a
    // later representation choice, not a cache-construction gate.
    let use_joint_newton = has_joint_exacthessian && (specs.len() >= 2 || has_workspace_source);
    let joint_workspace_requested = use_joint_newton && has_workspace_source;
    let inner_tol = options.inner_tol;
    let inner_max_cycles = options.inner_max_cycles;
    let inner_max_cycles = capped_inner_max_cycles(options, inner_max_cycles);
    // Each block's assembled penalty matrix depends only on that block's
    // penalties and smoothing parameters. Build these setup matrices in
    // parallel, but keep the coordinate-descent and line-search loops below
    // strictly serial because each accepted block update changes the state seen
    // by later blocks.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let s_lambdas = (0..specs.len())
        .into_par_iter()
        .map(|b| {
            let spec = &specs[b];
            let Some(block_log_lambda) = block_log_lambdas.get(b) else {
                return Err(format!(
                    "missing log-smoothing parameter vector for block {b}"
                ));
            };
            if block_log_lambda.len() != spec.penalties.len() {
                return Err(format!(
                    "block {b} log-smoothing parameter length {} does not match penalties {}",
                    block_log_lambda.len(),
                    spec.penalties.len()
                ));
            }

            let p = spec.design.ncols();
            let lambdas = block_log_lambda.mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((p, p));
            for (k, s) in spec.penalties.iter().enumerate() {
                s.add_scaled_to(lambdas[k], &mut s_lambda);
            }
            Ok(s_lambda)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let ridge = effective_solverridge(options.ridge_floor);
    let mut cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; specs.len()];
    if let Some(seed) = warm_start
        && seed.block_beta.len() == states.len()
        && seed.active_sets.len() == states.len()
    {
        if warm_start_matches_block_log_lambdas(seed, block_log_lambdas)
            && let Some(cached) = seed.cached_inner.as_ref()
            && cached.converged
            && seed
                .block_beta
                .iter()
                .zip(&states)
                .all(|(beta_seed, state)| beta_seed.len() == state.beta.len())
        {
            for (state, beta_seed) in states.iter_mut().zip(&seed.block_beta) {
                state.beta.assign(beta_seed);
            }
            cached_active_sets = seed.active_sets.clone();
            refresh_all_block_etas(family, specs, &mut states)?;
            log::info!(
                "[PIRLS/joint-Newton warm-start] reused cached same-rho inner mode | cycles={} logdet_h={:.6e} logdet_s={:.6e}",
                cached.cycles,
                cached.block_logdet_h,
                cached.block_logdet_s,
            );
            return Ok(BlockwiseInnerResult {
                block_states: states,
                active_sets: normalize_active_sets(cached_active_sets),
                log_likelihood: cached.log_likelihood,
                penalty_value: cached.penalty_value,
                cycles: cached.cycles,
                converged: cached.converged,
                block_logdet_h: cached.block_logdet_h,
                block_logdet_s: cached.block_logdet_s,
                s_lambdas,
                joint_workspace: cached.joint_workspace.clone(),
            });
        }
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
    let (
        mut current_log_likelihood,
        mut cached_eval,
        mut cached_joint_gradient,
        mut cached_joint_workspace,
    ) = if use_joint_newton {
        let (log_likelihood, gradient, eval, workspace) = load_joint_gradient_evaluation(
            family,
            specs,
            options,
            &states,
            joint_workspace_requested,
            None,
        )?;
        (log_likelihood, eval, gradient, workspace)
    } else {
        let eval = family.evaluate(&states)?;
        let log_likelihood = eval.log_likelihood;
        (log_likelihood, Some(eval), None, None)
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

        // Divergence-detection state for the joint-Newton path. Mirrors
        // the blockwise sibling later in this function: a near-null
        // direction in the joint Hessian (e.g. BernoulliMarginalSlope's
        // linkwiggle empirical anchor drifting from fit-time η₀) makes
        // Newton clamp at MAX_JOINT_STEP every cycle while -loglik stays
        // frozen and β grows linearly along the null mode. Without an
        // early-exit the loop runs to inner_max_cycles producing
        // identical -loglik values, burning ~50s per ρ-cost call at
        // biobank scale and stacking up to a 2400s timeout.
        let mut prev_log_likelihood_for_joint_divergence_check = current_log_likelihood;
        let mut consecutive_joint_frozen_loglik_cycles: usize = 0;
        const JOINT_DIVERGENCE_FROZEN_LOGLIK_CYCLES: usize = 8;

        // Cross-cycle convergence carry-over: set at the end of every
        // accepted cycle so the next cycle's line-search-failure path
        // can distinguish a true KKT optimum on a rank-deficient
        // Hessian (no meaningful trial step, even though step_inf is
        // O(1) along the null mode) from genuine non-convergence.
        let mut last_cycle_residual_below_tol = false;
        let mut last_cycle_obj_change_below_tol = false;

        let mut joint_trust_radius = 1.0_f64;
        for cycle in 0..inner_max_cycles {
            // Fires at the top of each inner joint-Newton cycle so CI logs can
            // distinguish "inner spin" (thousands of these) from "outer-assembly
            // spin" (zero of these). Emitted at info-level so the silent-grind
            // failure mode (e.g. CI hitting cmd timeout while PIRLS quietly
            // chews on the first outer-iter at biobank scale) is visible
            // without enabling debug logs.
            log::info!(
                "[PIRLS/blockwise joint-Newton] cycle {:>3}/{} | -loglik {:.6e} | penalty {:.6e} | objective {:.6e}",
                cycle,
                inner_max_cycles,
                -current_log_likelihood,
                current_penalty,
                lastobjective,
            );
            // Per-cycle phase-timing accumulators. Surface where the inner
            // joint-Newton spends time so a 18-min silent cycle 0 (the
            // bernoulli marginal-slope FLEX biobank failure mode) becomes a
            // logged timeline at the end of the cycle. Phases:
            //   * hessian: joint Hessian source build (matrix-free workspace
            //     OR dense fallback assembly)
            //   * pcg:     matrix-free QP solve via solve_spd_pcg_with_info_into
            //              (already logs its own diagnostics; we accumulate
            //              here for the end-of-cycle summary)
            //   * line_search: backtracking step-size search (up to 8 attempts)
            //   * grad_reload: post-accept joint gradient + workspace refresh
            let cycle_started = std::time::Instant::now();
            let hessian_started = std::time::Instant::now();
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            let joint_constraints =
                assemble_joint_linear_constraints(&block_constraints, &ranges, total_p)?;
            // Get joint Hessian and block gradients from the current evaluation.
            let mut hessian_workspace_for_cycle: Option<
                Arc<dyn ExactNewtonJointHessianWorkspace>,
            > = None;
            let joint_hessian_source = if joint_workspace_requested {
                let workspace = match cached_joint_workspace.take() {
                    Some(workspace) => Some(workspace),
                    None => family.exact_newton_joint_hessian_workspace_with_options(
                        &states, specs, options,
                    )?,
                };
                hessian_workspace_for_cycle = workspace.clone();
                workspace
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
            let joint_hessian_is_dense =
                matches!(&joint_hessian_source, JointHessianSource::Dense(_));

            let solve_joint_constraints_dense =
                !matrix_free_joint_requested || joint_hessian_is_dense;
            let (candidate_beta, joint_active_set) = if solve_joint_constraints_dense
                && let Some(constraints) = joint_constraints.as_ref()
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
                let pcg_started = std::time::Instant::now();
                let pcg_requested = matrix_free_joint_requested && !joint_hessian_is_dense;
                let mut delta = if pcg_requested {
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
                    // Pre-allocate the penalty workspace ONCE outside the
                    // PCG closure so each CG iter (called hundreds-to-
                    // thousands of times per outer iter at biobank scale)
                    // reuses the buffer instead of allocating per call.
                    // RefCell because solve_spd_pcg* expects `Fn` (immutable
                    // borrow of captures) and we need interior mutability
                    // to write into the workspace.
                    let penalty_workspace = RefCell::new(Array1::<f64>::zeros(total_p));
                    match &joint_hessian_source {
                        JointHessianSource::Dense(h_joint) => {
                            crate::linalg::utils::solve_spd_pcg_with_info_into(
                                |v, out| {
                                    // h_joint * v -> out (faer-backed, no alloc)
                                    crate::faer_ndarray::fast_av_view_into(
                                        h_joint,
                                        v,
                                        out.view_mut(),
                                    );
                                    let mut pen = penalty_workspace.borrow_mut();
                                    apply_joint_block_penalty_into(
                                        &ranges,
                                        &s_lambdas,
                                        v,
                                        trace_diagonal_ridge,
                                        &mut pen,
                                    );
                                    *out += &*pen;
                                },
                                &rhs,
                                &preconditioner_diag,
                                JOINT_PCG_REL_TOL,
                                JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                            )
                            .map(|(solution, info)| {
                                log_joint_pcg_diagnostics(
                                    cycle,
                                    total_p,
                                    total_joint_n,
                                    &preconditioner_diag,
                                    &info,
                                );
                                solution
                            })
                        }
                        JointHessianSource::Operator { apply_into, .. } => {
                            let apply_h_into = Arc::clone(apply_into);
                            crate::linalg::utils::solve_spd_pcg_with_info_into(
                                |v, out| {
                                    if let Err(error) = apply_h_into(v, out) {
                                        log::warn!(
                                            "joint Newton inner operator matvec failed: {error}"
                                        );
                                        out.fill(0.0);
                                    }
                                    let mut pen = penalty_workspace.borrow_mut();
                                    apply_joint_block_penalty_into(
                                        &ranges,
                                        &s_lambdas,
                                        v,
                                        trace_diagonal_ridge,
                                        &mut pen,
                                    );
                                    *out += &*pen;
                                },
                                &rhs,
                                &preconditioner_diag,
                                JOINT_PCG_REL_TOL,
                                JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                            )
                            .map(|(solution, info)| {
                                log_joint_pcg_diagnostics(
                                    cycle,
                                    total_p,
                                    total_joint_n,
                                    &preconditioner_diag,
                                    &info,
                                );
                                solution
                            })
                        }
                    }
                } else {
                    None
                };
                if pcg_requested {
                    log::info!(
                        "[PIRLS/joint-PCG] cycle {:>3} | n={} p={} solved={} elapsed={:.3}s",
                        cycle,
                        total_joint_n,
                        total_p,
                        delta.is_some(),
                        pcg_started.elapsed().as_secs_f64()
                    );
                }
                if delta.is_none() {
                    if pcg_requested {
                        break;
                    }
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
            // Hessian-source build (and any QP solve immediately above) are
            // done by the time we reach `delta`. Capture the wall-clock
            // before the line-search phase so the end-of-cycle summary can
            // attribute time correctly between the Hessian/QP and the
            // backtracking step search.
            let hessian_and_qp_elapsed = hessian_started.elapsed();
            let line_search_started = std::time::Instant::now();
            let mut delta = &candidate_beta - &beta_joint;

            // Trust-region globalization for the joint Newton proposal.  The
            // previous implementation used up to eight backtracking likelihood
            // evaluations (each can build the exact joint workspace at biobank
            // scale).  Here the step is truncated before evaluation and the
            // single trial objective is accepted only when the actual decrease
            // is positive relative to the local quadratic model.
            let mut step_inf = delta.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
            const MAX_JOINT_STEP: f64 = 20.0;
            if step_inf > MAX_JOINT_STEP {
                delta.mapv_inplace(|v| v * (MAX_JOINT_STEP / step_inf));
                step_inf = MAX_JOINT_STEP;
            }

            let old_beta: Vec<Array1<f64>> = states.iter().map(|s| s.beta.clone()).collect();
            let old_objective = lastobjective;
            let mut accepted = false;
            let mut accepted_joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>> =
                None;
            let mut line_search_attempts = 0usize;

            // Pure Newton must take a full step on the first cycle of an
            // exact quadratic problem (i.e. converge in one cycle when the
            // model is exact). The trust-region globalization above must not
            // truncate the very first proposal merely because the hard-coded
            // initial radius (1.0) is smaller than the natural Newton-step
            // 2-norm. Bumping the radius up to the post-barrier Newton-step
            // norm on cycle 0 preserves quadratic convergence on
            // well-conditioned problems while leaving the standard adaptive
            // shrink/expand for subsequent cycles. The MAX_JOINT_STEP cap on
            // |delta|_inf above remains the actual safeguard against runaway
            // proposals.
            if cycle == 0 {
                let initial_step_norm = joint_trust_region_step_norm(&delta);
                if initial_step_norm.is_finite() && initial_step_norm > joint_trust_radius {
                    joint_trust_radius = initial_step_norm;
                }
            }

            let penalty_beta = apply_joint_block_penalty(
                &ranges,
                &s_lambdas,
                &beta_joint,
                joint_mode_diagonal_ridge,
            );
            let rhs = &grad_joint - &penalty_beta;
            let beta_inf = states
                .iter()
                .flat_map(|s| s.beta.iter().copied())
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let step_tol = inner_tol * (1.0 + beta_inf);
            let objective_tol = inner_tol * (1.0 + old_objective.abs());
            let residual_tol = objective_tol;
            let current_stationarity_residual =
                exact_newton_joint_stationarity_inf_norm_from_gradient(
                    &grad_joint,
                    &states,
                    specs,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    &block_constraints,
                )?;
            if current_stationarity_residual <= residual_tol
                && last_cycle_obj_change_below_tol
                && step_inf <= step_tol
            {
                log::info!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | pre-line-search converged: step_inf={:.3e} (tol={:.3e}) | residual={:.3e} (tol={:.3e}) | previous objective change below tol",
                    cycle,
                    step_inf,
                    step_tol,
                    current_stationarity_residual,
                    residual_tol,
                );
                cached_joint_workspace = hessian_workspace_for_cycle;
                cycles_done = cycle;
                converged = true;
                break;
            }

            // Trust-region retries preserve the objective-decrease guarantee
            // when the initial radius is too optimistic. If the Newton proposal
            // is not a descent direction for the penalized quadratic model,
            // switch once to a diagonally preconditioned gradient step and keep
            // the same exact full-objective accept/reject test.
            const JOINT_TRUST_MAX_ATTEMPTS: usize = 24;
            let mut search_delta = delta.clone();
            let mut tried_preconditioned_descent = false;
            let mut model_rejects = 0usize;
            let mut likelihood_rejects = 0usize;
            let mut objective_rejects = 0usize;
            let mut first_likelihood_reject: Option<String> = None;
            for trust_attempt in 0..JOINT_TRUST_MAX_ATTEMPTS {
                line_search_attempts = trust_attempt + 1;
                accepted_joint_workspace = None;
                let mut trial_delta = search_delta.clone();
                truncate_joint_step_to_radius(&mut trial_delta, joint_trust_radius);
                let mut barrier_ceiling = 1.0_f64;
                for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
                    let block_delta = trial_delta.slice(s![start..end]).to_owned();
                    if let Some(alpha_max) =
                        family.max_feasible_step_size(&states, block_idx, &block_delta)?
                    {
                        barrier_ceiling = barrier_ceiling.min(alpha_max);
                    }
                }
                if !barrier_ceiling.is_finite() || barrier_ceiling <= 0.0 {
                    joint_trust_radius = (0.25 * joint_trust_radius).max(1.0e-12);
                    continue;
                }
                if barrier_ceiling < 1.0 {
                    trial_delta.mapv_inplace(|v| v * barrier_ceiling);
                }
                let step_norm = joint_trust_region_step_norm(&trial_delta);
                let mut hpen_delta = Array1::<f64>::zeros(total_p);
                if apply_joint_penalized_hessian_into(
                    &joint_hessian_source,
                    &ranges,
                    &s_lambdas,
                    trace_diagonal_ridge,
                    &trial_delta,
                    &mut hpen_delta,
                )
                .is_err()
                {
                    break;
                }
                let predicted_reduction =
                    joint_quadratic_predicted_reduction(&rhs, &hpen_delta, &trial_delta);
                if !predicted_reduction.is_finite() || predicted_reduction <= 0.0 {
                    model_rejects += 1;
                    if !tried_preconditioned_descent {
                        match joint_preconditioned_descent_delta(
                            &joint_hessian_source,
                            &ranges,
                            &s_lambdas,
                            trace_diagonal_ridge,
                            &rhs,
                        ) {
                            Ok(descent_delta) => {
                                search_delta = descent_delta;
                            }
                            Err(_) => {
                                joint_trust_radius = (0.25 * joint_trust_radius).max(1.0e-12);
                            }
                        }
                        tried_preconditioned_descent = true;
                    } else {
                        joint_trust_radius = (0.25 * joint_trust_radius).max(1.0e-12);
                    }
                    continue;
                }

                for b in 0..specs.len() {
                    let (start, end) = ranges[b];
                    let mut trial_beta = old_beta[b].clone();
                    trial_beta += &trial_delta.slice(ndarray::s![start..end]);
                    let projected =
                        family.post_update_block_beta(&states, b, &specs[b], trial_beta)?;
                    states[b].beta.assign(&projected);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                let trial_penalty =
                    total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
                // Families that can build a reusable workspace while computing
                // the line-search likelihood do so here. Rejected attempts can
                // still stop through the early-exit threshold; accepted full
                // sweeps keep their exact row cache for the gradient reload.
                let line_search_options =
                    coefficient_line_search_options(options, old_objective + 1e-10);
                let trial_ll = match joint_line_search_log_likelihood(
                    family,
                    specs,
                    &line_search_options,
                    &states,
                    joint_workspace_requested || options.line_search_prefer_workspace,
                ) {
                    Ok((value, workspace)) => {
                        accepted_joint_workspace = workspace;
                        value
                    }
                    Err(e) => {
                        likelihood_rejects += 1;
                        if first_likelihood_reject.is_none() {
                            first_likelihood_reject = Some(e);
                        }
                        for (b, old) in old_beta.iter().enumerate() {
                            states[b].beta.assign(old);
                        }
                        refresh_all_block_etas(family, specs, &mut states)?;
                        joint_trust_radius = (0.25 * joint_trust_radius).max(1.0e-12);
                        continue;
                    }
                };
                let trialobjective = -trial_ll + trial_penalty;
                let actual_reduction = old_objective - trialobjective;
                let trust_update = update_joint_trust_region_radius(
                    joint_trust_radius,
                    step_norm,
                    actual_reduction,
                    predicted_reduction,
                );
                let old_radius = joint_trust_radius;
                joint_trust_radius = trust_update.radius;
                log::info!(
                    "[PIRLS/joint-Newton/trust-region] cycle={} attempt={} accepted={} rho={:.3e} actual_reduction={:.3e} predicted_reduction={:.3e} radius={:.3e}->{:.3e} step_norm={:.3e} step_inf={:.3e}",
                    cycle,
                    line_search_attempts,
                    trust_update.accepted,
                    trust_update.rho,
                    actual_reduction,
                    predicted_reduction,
                    old_radius,
                    joint_trust_radius,
                    step_norm,
                    step_inf,
                );
                if trialobjective.is_finite()
                    && trust_update.accepted
                    && trialobjective <= old_objective + 1e-10
                {
                    current_penalty = trial_penalty;
                    if let Some(joint_active_set) = joint_active_set.as_ref() {
                        cached_active_sets =
                            scatter_joint_active_set(joint_active_set, &block_constraints);
                    }
                    accepted = true;
                    break;
                }
                for (b, old) in old_beta.iter().enumerate() {
                    states[b].beta.assign(old);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                objective_rejects += 1;
            }
            let line_search_elapsed = line_search_started.elapsed();
            if !accepted {
                log::info!(
                    "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=false hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} reject_model={} reject_likelihood={} reject_objective={} first_likelihood_reject={} grad_reload=0.000s total={:.3}s",
                    cycle,
                    hessian_and_qp_elapsed.as_secs_f64(),
                    line_search_elapsed.as_secs_f64(),
                    line_search_attempts,
                    model_rejects,
                    likelihood_rejects,
                    objective_rejects,
                    first_likelihood_reject.as_deref().unwrap_or("none"),
                    cycle_started.elapsed().as_secs_f64(),
                );
                // Restore original betas
                for (b, old) in old_beta.iter().enumerate() {
                    states[b].beta.assign(old);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                // If the previous cycle's bookkeeping certified KKT
                // stationarity (residual ≤ tol and objective change ≤
                // tol), the line-search failure here is round-off on a
                // rank-deficient null mode rather than non-convergence:
                // the proposed `H⁻¹ g` step stays O(1) along the null
                // direction at the optimum, every trial moves β along
                // it without changing the objective, and round-off
                // flips the sign of `actual − predicted` so the
                // sufficient-decrease check rejects every trial. The
                // iterate ALREADY satisfies the first-order optimality
                // conditions; we accept that as convergence rather
                // than fail the outer "inner solve did not converge"
                // panic on a fully resolved fit.
                if last_cycle_residual_below_tol && last_cycle_obj_change_below_tol {
                    converged = true;
                }
                break; // Fall back to blockwise
            }

            let grad_reload_started = std::time::Instant::now();
            let (log_likelihood, gradient, eval, workspace) = load_joint_gradient_evaluation(
                family,
                specs,
                options,
                &states,
                joint_workspace_requested,
                accepted_joint_workspace.take(),
            )?;
            let grad_reload_elapsed = grad_reload_started.elapsed();
            log::info!(
                "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=true hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} grad_reload={:.3}s total={:.3}s",
                cycle,
                hessian_and_qp_elapsed.as_secs_f64(),
                line_search_elapsed.as_secs_f64(),
                line_search_attempts,
                grad_reload_elapsed.as_secs_f64(),
                cycle_started.elapsed().as_secs_f64(),
            );
            current_log_likelihood = log_likelihood;
            cached_joint_gradient = gradient;
            cached_eval = eval;
            cached_joint_workspace = workspace;
            current_penalty =
                total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
            lastobjective = -current_log_likelihood + current_penalty;
            let objective_change = (lastobjective - old_objective).abs();
            let accepted_step_inf = states
                .iter()
                .zip(old_beta.iter())
                .flat_map(|(state, old)| {
                    state
                        .beta
                        .iter()
                        .zip(old.iter())
                        .map(|(new, old)| (new - old).abs())
                })
                .fold(0.0_f64, f64::max);
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

            // Scale-aware tolerances. The objective check was already
            // relative (`inner_tol * (1 + |obj|)`), but the step and
            // residual checks were absolute against the bare `inner_tol`
            // — at biobank scale (n ≈ 320k), β iterates can keep moving
            // by ~1e-5 per cycle along the monotonicity-feasible
            // manifold even after the likelihood has gone flat, and the
            // joint gradient ‖·‖_∞ is O(|obj|), not O(1). Running
            // 50-100 cycles past objective convergence is the
            // dominant inner-PIRLS cost at biobank scale. Switching to
            // relative scaling (`inner_tol * (1 + ‖β‖_∞)` for steps,
            // `inner_tol * (1 + |obj|)` for the gradient residual)
            // exits PIRLS as soon as the optimum is statistically
            // resolved, without loosening behavior at small n where
            // ‖β‖_∞ ≈ 1 and |obj| ≈ 1 give tolerances within 2× of
            // the historical absolute 1e-6.
            let beta_inf = states
                .iter()
                .flat_map(|s| s.beta.iter().copied())
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let step_tol = inner_tol * (1.0 + beta_inf);
            let objective_tol = inner_tol * (1.0 + lastobjective.abs());
            let residual_tol = objective_tol;

            // Per-cycle observability for the convergence test. Surfaces
            // WHICH criterion is binding (proposed step, accepted step,
            // residual, objective change) at every iteration so CI logs
            // distinguish "Newton hasn't proposed a small step yet"
            // (algorithm still working) from "step is small but residual
            // won't drop below tol" (tolerance scaling problem). Without
            // this, the only visible signal is the objective itself,
            // which is insufficient to choose the right algorithmic
            // remedy.
            log::info!(
                "[PIRLS/joint-Newton convergence] cycle {:>3} | step_inf={:.3e} (tol={:.3e}) | accepted_step_inf={:.3e} | residual={:.3e} (tol={:.3e}) | obj_change={:.3e} (tol={:.3e}) | beta_inf={:.3e}",
                cycle,
                step_inf,
                step_tol,
                accepted_step_inf,
                residual,
                residual_tol,
                objective_change,
                objective_tol,
                beta_inf,
            );

            // Divergence early-exit. See the rationale block above the
            // loop. We track current_log_likelihood (not the smoothed
            // `objective_change`) because under a near-null mode the
            // penalty drifts cycle-to-cycle even when the data fit is
            // genuinely stagnant — `objective_change` would mask the
            // divergence whereas a frozen log-likelihood is unambiguous.
            let loglik_change_for_joint_divergence_check =
                (current_log_likelihood - prev_log_likelihood_for_joint_divergence_check).abs();
            let loglik_frozen_tol_for_joint_divergence_check =
                inner_tol * (1.0 + current_log_likelihood.abs());
            let step_clamped_for_joint_divergence_check = step_inf >= 0.95 * MAX_JOINT_STEP;
            if loglik_change_for_joint_divergence_check
                <= loglik_frozen_tol_for_joint_divergence_check
                && step_clamped_for_joint_divergence_check
            {
                consecutive_joint_frozen_loglik_cycles += 1;
            } else {
                consecutive_joint_frozen_loglik_cycles = 0;
            }
            prev_log_likelihood_for_joint_divergence_check = current_log_likelihood;
            if consecutive_joint_frozen_loglik_cycles >= JOINT_DIVERGENCE_FROZEN_LOGLIK_CYCLES {
                log::warn!(
                    "[PIRLS/joint-Newton convergence] divergence early-exit at cycle {} | -loglik={:.6e} frozen for {} consecutive cycles | step_inf={:.3e} (clamped at cap {}) | step_tol={:.3e}; near-null Hessian direction detected — returning unconverged so the outer optimizer backs off this region instead of running to inner_max_cycles.",
                    cycle,
                    -current_log_likelihood,
                    consecutive_joint_frozen_loglik_cycles,
                    step_inf,
                    MAX_JOINT_STEP,
                    step_tol,
                );
                converged = false;
                break;
            }

            // KKT convergence: small residual plus EITHER a small
            // Newton step (tight quadratic-rate convergence, lets β
            // polish to machine precision) OR confirmed stagnation
            // (`accepted_step_inf <= step_tol` AND `objective_change
            // <= objective_tol`, the rank-deficient null-mode case).
            // The conjunction in the second branch matters: a single
            // small `objective_change` on a quadratic-rate cycle is
            // normal Newton progress, not stagnation, and treating it
            // alone as convergence stops the iteration with β still
            // O(residual_tol) away from the optimum — visible as a
            // sign mismatch against finite-differenced outer LAML
            // gradients in the autodiff cross-checks.
            if residual <= residual_tol
                && (step_inf <= step_tol
                    || (accepted_step_inf <= step_tol && objective_change <= objective_tol))
            {
                converged = true;
                break;
            }
            if accepted_step_inf <= step_tol && objective_change <= objective_tol {
                if residual <= residual_tol {
                    converged = true;
                }
                break;
            }
            // Carry the KKT-stationarity signal into the next cycle so
            // the line-search-failure path above can recognise a true
            // KKT optimum on a rank-deficient null mode. See that path
            // for the full rationale.
            last_cycle_residual_below_tol = residual <= residual_tol;
            last_cycle_obj_change_below_tol = objective_change <= objective_tol;
        }

        // If joint Newton converged, skip the blockwise loop entirely.
        if converged {
            let penalty_value =
                total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
            let (block_logdet_h, block_logdet_s) = blockwise_logdet_terms_with_workspace(
                family,
                specs,
                &mut states,
                block_log_lambdas,
                options,
                cached_joint_workspace.clone(),
            )?;
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
                joint_workspace: cached_joint_workspace.clone(),
            });
        }
        if cycles_done >= inner_max_cycles {
            let penalty_value =
                total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
            let (block_logdet_h, block_logdet_s) = blockwise_logdet_terms_with_workspace(
                family,
                specs,
                &mut states,
                block_log_lambdas,
                options,
                cached_joint_workspace.clone(),
            )?;
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
                joint_workspace: cached_joint_workspace.clone(),
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

    // Divergence-detection state for the blockwise loop.
    //
    // Some family parameterizations (e.g. BernoulliMarginalSlopeFamily with
    // linkwiggle + scorewarp) carry a near-null direction in the joint
    // Hessian when the link-deviation basis's empirical anchor — fixed at
    // the rigid-pilot η₀ when the basis is constructed — drifts during
    // PIRLS as the location/spatial blocks update η₀. The Newton step
    // becomes dominated by that null direction and is clamped at
    // MAX_NEWTON_STEP every cycle while β grows linearly along it; the
    // log-likelihood stays frozen, only the penalty changes (slowly).
    // Without an early-exit the loop runs to inner_max_cycles producing
    // the same -loglik over and over, which at biobank scale (each cycle
    // ~0.5s) burns ~50s per ρ-cost call and stacks up to a 2400s timeout.
    //
    // Detect the pattern and bail with `converged = false` so the cost
    // call returns Err / +∞, BFGS κ-optim backs off the divergent ρ
    // region, and the outer loop progresses instead of grinding.
    let mut prev_log_likelihood_for_divergence_check = cached_eval.log_likelihood;
    let mut consecutive_frozen_loglik_cycles: usize = 0;
    // Mirrors the `MAX_NEWTON_STEP` const used by the inner ExactNewton
    // path lower in this function. Hoisted here so the divergence check
    // does not need to reach into a nested scope.
    const NEWTON_STEP_CAP_FOR_DIVERGENCE: f64 = 20.0;
    const DIVERGENCE_FROZEN_LOGLIK_CYCLES: usize = 8;

    let is_dynamic = family.block_geometry_is_dynamic();
    for cycle in 0..inner_max_cycles {
        // Fires at the top of each blockwise coordinate cycle so we can count
        // iterations from CI logs when a benchmark hangs inside the first
        // outer-eval. Emitted at info-level: same rationale as the joint-Newton
        // sibling above — silent-grind diagnosis without debug logs.
        log::info!(
            "[PIRLS/blockwise coord] cycle {:>3}/{} | -loglik {:.6e} | penalty {:.6e} | objective {:.6e}",
            cycle,
            inner_max_cycles,
            -current_log_likelihood,
            current_penalty,
            lastobjective,
        );
        let mut max_proposed_beta_step = 0.0_f64;
        let mut max_accepted_beta_step = 0.0_f64;

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
            max_proposed_beta_step = max_proposed_beta_step.max(step);
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
                let trial_block_penalty =
                    block_quadratic_penalty(&states[b].beta, s_lambda, ridge, options.ridge_policy);
                let trial_penalty = current_penalty - old_block_penalty + trial_block_penalty;
                let line_search_options = coefficient_line_search_options(
                    options,
                    objective_cycle_prev - trial_penalty + 1e-10,
                );
                let trial_ll =
                    match family.log_likelihood_only_with_options(&states, &line_search_options) {
                        Ok(value) => value,
                        Err(_) => {
                            states[b].beta.assign(&beta_old);
                            eta_checkpoint.restore_eta(&mut states[b]);
                            continue;
                        }
                    };
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
                            let trial_block_penalty = block_quadratic_penalty(
                                &states[b].beta,
                                s_lambda,
                                ridge,
                                options.ridge_policy,
                            );
                            let trial_penalty =
                                current_penalty - old_block_penalty + trial_block_penalty;
                            let line_search_options = coefficient_line_search_options(
                                options,
                                objective_cycle_prev - trial_penalty + 1e-10,
                            );
                            let trial_ll = match family
                                .log_likelihood_only_with_options(&states, &line_search_options)
                            {
                                Ok(value) => value,
                                Err(_) => {
                                    states[b].beta.assign(&beta_old);
                                    eta_checkpoint.restore_eta(&mut states[b]);
                                    continue;
                                }
                            };
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
                let accepted_step = states[b]
                    .beta
                    .iter()
                    .zip(beta_old.iter())
                    .map(|(new, old)| (new - old).abs())
                    .fold(0.0_f64, f64::max);
                max_accepted_beta_step = max_accepted_beta_step.max(accepted_step);
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

        // Scale-aware tolerances — see the matching joint-Newton path
        // above for the rationale. At biobank scale absolute step/residual
        // tolerances against `inner_tol = 1e-6` keep this loop spinning
        // long after the objective has gone flat.
        let beta_inf = states
            .iter()
            .flat_map(|s| s.beta.iter().copied())
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let step_tol = inner_tol * (1.0 + beta_inf);
        let objective_tol = inner_tol * (1.0 + objective.abs());
        let residual_tol = objective_tol;
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
            .map(|residual| residual <= residual_tol)
            .unwrap_or(true)
        } else {
            true
        };
        log::info!(
            "[PIRLS/blockwise convergence] cycle {:>3} | max_proposed_step={:.3e} (tol={:.3e}) | max_accepted_step={:.3e} | obj_change={:.3e} (tol={:.3e}) | beta_inf={:.3e} | joint_stationarity_ok={}",
            cycle,
            max_proposed_beta_step,
            step_tol,
            max_accepted_beta_step,
            objective_change,
            objective_tol,
            beta_inf,
            exact_joint_stationarity_ok,
        );

        // Divergence early-exit. See the rationale block at the top of
        // this loop. We treat "log-likelihood unchanged + Newton step
        // pinned at the trust-region cap" as a near-null direction
        // signature and break out unconverged once it persists for
        // DIVERGENCE_FROZEN_LOGLIK_CYCLES consecutive iterations. Tracking
        // log-likelihood (not objective) is essential: when the null mode
        // dominates, only the penalty drifts cycle-to-cycle, so
        // `objective_change` stays above tol while -loglik is genuinely
        // frozen.
        let loglik_change_for_divergence_check =
            (cached_eval.log_likelihood - prev_log_likelihood_for_divergence_check).abs();
        let loglik_frozen_tol_for_divergence_check =
            inner_tol * (1.0 + cached_eval.log_likelihood.abs());
        let step_clamped_for_divergence_check =
            max_proposed_beta_step >= 0.95 * NEWTON_STEP_CAP_FOR_DIVERGENCE;
        if loglik_change_for_divergence_check <= loglik_frozen_tol_for_divergence_check
            && step_clamped_for_divergence_check
        {
            consecutive_frozen_loglik_cycles += 1;
        } else {
            consecutive_frozen_loglik_cycles = 0;
        }
        prev_log_likelihood_for_divergence_check = cached_eval.log_likelihood;
        if consecutive_frozen_loglik_cycles >= DIVERGENCE_FROZEN_LOGLIK_CYCLES {
            log::warn!(
                "[PIRLS/blockwise convergence] divergence early-exit at cycle {} | -loglik={:.6e} frozen for {} consecutive cycles | max_proposed_step={:.3e} (clamped at cap {}) | step_tol={:.3e}; near-null Hessian direction detected — returning unconverged so the outer optimizer backs off this region instead of running to inner_max_cycles.",
                cycle,
                -cached_eval.log_likelihood,
                consecutive_frozen_loglik_cycles,
                max_proposed_beta_step,
                NEWTON_STEP_CAP_FOR_DIVERGENCE,
                step_tol,
            );
            converged = false;
            break;
        }

        if max_accepted_beta_step <= step_tol && objective_change <= objective_tol {
            if exact_joint_stationarity_ok || max_proposed_beta_step <= step_tol {
                converged = true;
            }
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
    // non-converged β̂ produces large envelope-theorem violations in the
    // analytic outer gradient.
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
            // Scale-aware residual tolerance — the joint stationarity
            // residual ‖∇ℓ − Sβ‖_∞ scales with |obj| (≈ O(n) at biobank
            // scale), so the historical absolute `inner_tol = 1e-6` is
            // unachievable here even at the true minimum. Same rationale
            // as the joint-Newton convergence test above.
            let polish_obj = -cached_eval.log_likelihood + current_penalty;
            let polish_residual_tol = inner_tol * (1.0 + polish_obj.abs());
            if res_inf_free <= polish_residual_tol {
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
        joint_workspace: None,
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
    compute_d2h: &'a dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>,
    family_outer_hessian_operator:
        Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>>,
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

    fn family_outer_hessian_operator(
        &self,
    ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
        self.family_outer_hessian_operator.clone()
    }
}

struct OwnedJointDerivProvider {
    compute_dh: Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>,
    compute_d2h: Arc<
        dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
            + Send
            + Sync,
    >,
    family_outer_hessian_operator:
        Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>>,
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

    fn family_outer_hessian_operator(
        &self,
    ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
        self.family_outer_hessian_operator.clone()
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
    fn dim(&self) -> usize {
        self.inner.dim()
    }

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
    if let Some(tk_eta_fixed) = coord.tk_eta_fixed.as_mut() {
        *tk_eta_fixed *= scale;
    }
    if let Some(tk_x_fixed) = coord.tk_x_fixed.as_mut() {
        *tk_x_fixed *= scale;
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
    let per_block_penalties_dense: Vec<Vec<Array2<f64>>> = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        (0..specs.len())
            .into_par_iter()
            .map(|b| specs[b].penalties.iter().map(|p| p.to_dense()).collect())
            .collect()
    };
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
    // Always evaluate gradient: the universal-form EFS step
    // `Δρ = log(1 − 2·g_full / q_eff)` reads it directly from the cost
    // gradient slot, so out-of-band cost terms (TK, prior, Firth,
    // barrier, SAS log-δ ridge) shift the multiplicative target through
    // their gradient contribution without needing per-augmentation
    // post-corrections.
    let eval_mode = EvalMode::ValueAndGradient;
    let result = crate::estimate::reml::assembly::evaluate_solution(
        &inner_solution,
        rho_slice,
        eval_mode,
        None,
    )?;

    let gradient = result
        .gradient
        .as_ref()
        .ok_or_else(|| "EFS evaluation did not return the required gradient".to_string())?;
    let gradient_slice = gradient
        .as_slice()
        .ok_or_else(|| "outer gradient must be contiguous for EFS".to_string())?;

    if has_psi {
        let hybrid = crate::estimate::reml::unified::compute_hybrid_efs_update(
            &inner_solution,
            rho_slice,
            gradient_slice,
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
            steps: crate::estimate::reml::unified::compute_efs_update(
                &inner_solution,
                rho_slice,
                gradient_slice,
            ),
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
    batched_outer_hessian_operator: Option<
        Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>,
    >,
) -> Result<OuterObjectiveEvalResult, String> {
    let joint_trace_diagonal_ridge = moderidge + if !strict_spd { extra_logdet_ridge } else { 0.0 };
    let scaled_joint_trace_diagonal_ridge = rho_curvature_scale * joint_trace_diagonal_ridge;

    // Build derivative provider from the caller-supplied closures.
    let provider_box: Box<dyn HessianDerivativeProvider + '_> =
        if let (Some(owned_dh), Some(owned_d2h)) = (owned_compute_dh, owned_compute_d2h) {
            Box::new(OwnedJointDerivProvider {
                compute_dh: owned_dh,
                compute_d2h: owned_d2h,
                family_outer_hessian_operator: batched_outer_hessian_operator.clone(),
            })
        } else {
            Box::new(BorrowedJointDerivProvider {
                compute_dh,
                compute_d2h,
                family_outer_hessian_operator: batched_outer_hessian_operator.clone(),
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
                    let apply_h = Arc::clone(&h_joint);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    Arc::new(MatrixFreeSpdOperator::new(total, move |v| {
                        let mut out = apply_h.dot(v);
                        let penalty = apply_joint_block_penalty(
                            &apply_ranges,
                            apply_s.as_ref(),
                            v,
                            trace_diagonal_ridge,
                        );
                        out += &penalty;
                        out
                    }))
                }
                JointHessianSource::Operator { apply, .. } => {
                    let apply_h = Arc::clone(&apply);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    Arc::new(MatrixFreeSpdOperator::new(total, move |v| {
                        let mut out = match apply_h(v) {
                            Ok(out) => out,
                            Err(error) => {
                                log::warn!(
                                    "joint exact-newton operator matvec failed during outer trace construction: {error}"
                                );
                                Array1::<f64>::from_elem(total, f64::NAN)
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
                    }))
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
        cached_inner: Some(cached_inner_mode_from_result(inner)),
    };

    Ok(OuterObjectiveEvalResult {
        objective,
        gradient: grad,
        outer_hessian,
        warm_start: warm,
        inner_converged: inner.converged,
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
                family_outer_hessian_operator: None,
            })
        } else {
            Box::new(BorrowedJointDerivProvider {
                compute_dh,
                compute_d2h,
                family_outer_hessian_operator: None,
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
                    let apply_h = Arc::clone(&h_joint);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    Arc::new(MatrixFreeSpdOperator::new(total, move |v| {
                        let mut out = apply_h.dot(v);
                        let penalty = apply_joint_block_penalty(
                            &apply_ranges,
                            apply_s.as_ref(),
                            v,
                            trace_diagonal_ridge,
                        );
                        out += &penalty;
                        out
                    }))
                }
                JointHessianSource::Operator { apply, .. } => {
                    let apply_h = Arc::clone(&apply);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    Arc::new(MatrixFreeSpdOperator::new(total, move |v| {
                        let mut out = match apply_h(v) {
                            Ok(out) => out,
                            Err(error) => {
                                log::warn!(
                                    "joint exact-newton operator matvec failed during fixed-point trace construction: {error}"
                                );
                                Array1::<f64>::from_elem(total, f64::NAN)
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
                    }))
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

#[cfg(test)]
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

/// Test-only helper exposing the value+gradient outer evaluation entry point
/// to other modules' test code.  Used by
/// the batched-gradient tests in `families/gamlss.rs` to pin the batched
/// override against a test oracle
/// without re-implementing the inner-fit / penalty-pseudo-logdet plumbing.
/// Returns only `(value, gradient)`; the warm-start is internal state with a
/// private type and is dropped at the boundary.
#[cfg(test)]
pub(crate) fn test_outerobjective_andgradient<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
) -> Result<(f64, Array1<f64>), String> {
    let (obj, grad, _, _) = outerobjectivegradienthessian(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        None,
        EvalMode::ValueAndGradient,
    )?;
    Ok((obj, grad))
}

fn outerobjectiveefs<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<
    (
        crate::solver::outer_strategy::EfsEval,
        ConstrainedWarmStart,
        bool,
    ),
    String,
> {
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
        if let Some(joint_bundle) = build_joint_hessian_closures(
            family,
            &inner.block_states,
            specs,
            total,
            options,
            inner.joint_workspace.clone(),
        )? {
            let JointHessianBundle {
                source: h_joint_unpen,
                beta_flat,
                compute_dh,
                compute_d2h,
                owned_compute_dh,
                owned_compute_d2h,
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
                owned_compute_dh,
                owned_compute_d2h,
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
                                d_eta += &fast_av(&dx, &beta_flat);
                                let mut wx = x_dense.clone();
                                let mut wdx = dx.clone();
                                ndarray::Zip::from(wx.rows_mut())
                                    .and(wdx.rows_mut())
                                    .and(wwork.view())
                                    .par_for_each(|mut wxr, mut wdxr, &wi| {
                                        if wi != 1.0 {
                                            wxr.mapv_inplace(|v| v * wi);
                                            wdxr.mapv_inplace(|v| v * wi);
                                        }
                                    });
                                correction_mat += &fast_atb(&dx, &wx);
                                correction_mat += &fast_atb(&x_dense, &wdx);
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
                        ndarray::Zip::from(scaled_x.rows_mut())
                            .and(&dw)
                            .par_for_each(|mut sr, &dwi| sr.mapv_inplace(|v| v * dwi));
                        correction_mat += &fast_atb(&x_dense, &scaled_x);

                        Ok(Some(DriftDerivResult::Dense(correction_mat)))
                    }
                }
            };
            let compute_d2h = |u: &Array1<f64>,
                               v: &Array1<f64>|
             -> Result<Option<DriftDerivResult>, String> {
                if !include_logdet_h {
                    return Ok(None);
                }
                match work {
                    BlockWorkingSet::ExactNewton { .. } => {
                        match family.exact_newton_hessian_second_directional_derivative(
                            &inner.block_states,
                            block_idx,
                            u,
                            v,
                        )? {
                            Some(h_exact) => {
                                Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                                    h_exact,
                                    p,
                                    &format!(
                                        "block {block_idx} exact-newton d2H shape mismatch in fixed-point outer evaluation"
                                    ),
                                )?)))
                            }
                            None => Err(format!(
                                "missing exact-newton d2H callback for block {block_idx} while fixed-point evaluation requires H_beta_beta term"
                            )),
                        }
                    }
                    BlockWorkingSet::Diagonal { .. } => {
                        let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                            format!(
                                "missing dynamic design for block {block_idx} diagonal fixed-point second correction"
                            )
                        })?;
                        let x_dense = x_dyn.to_dense();
                        let n = x_dense.nrows();
                        let reject_second_order_geometry =
                            |label: &str,
                             geom: Option<BlockGeometryDirectionalDerivative>|
                             -> Result<(), String> {
                                if let Some(geom_dir) = geom {
                                    let has_offset =
                                        geom_dir.d_offset.iter().any(|value| *value != 0.0);
                                    if geom_dir.d_design.is_some() || has_offset {
                                        return Err(format!(
                                            "block {block_idx} diagonal d2H requires second-order block-geometry derivatives for {label}; use an exact-newton or joint outer path"
                                        ));
                                    }
                                }
                                Ok(())
                            };
                        reject_second_order_geometry(
                            "first direction",
                            family.block_geometry_directional_derivative(
                                &inner.block_states,
                                block_idx,
                                spec,
                                u,
                            )?,
                        )?;
                        reject_second_order_geometry(
                            "second direction",
                            family.block_geometry_directional_derivative(
                                &inner.block_states,
                                block_idx,
                                spec,
                                v,
                            )?,
                        )?;
                        let d_eta_u = x_dyn.matrixvectormultiply(u);
                        let d_eta_v = x_dyn.matrixvectormultiply(v);
                        let d2w = family
                            .diagonalworking_weights_second_directional_derivative(
                                &inner.block_states,
                                block_idx,
                                &d_eta_u,
                                &d_eta_v,
                            )?
                            .ok_or_else(|| {
                                format!(
                                    "missing diagonal d2W callback for block {block_idx} while fixed-point evaluation requires H_beta_beta term"
                                )
                            })?;
                        if d2w.len() != n {
                            return Err(format!(
                                "block {block_idx} diagonal d2W length mismatch in fixed-point outer evaluation: got {}, expected {}",
                                d2w.len(),
                                n
                            ));
                        }
                        let mut scaled_x = x_dense.clone();
                        ndarray::Zip::from(scaled_x.rows_mut())
                            .and(&d2w)
                            .par_for_each(|mut sr, &d2wi| sr.mapv_inplace(|value| value * d2wi));
                        Ok(Some(DriftDerivResult::Dense(fast_atb(&x_dense, &scaled_x))))
                    }
                }
            };
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
        cached_inner: Some(cached_inner_mode_from_result(&inner)),
    };

    Ok((efs_eval, warm, inner.converged))
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

    let build_psi_hyper_coords_start = std::time::Instant::now();
    let total_axes: usize = derivative_blocks.iter().map(|b| b.len()).sum();

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
                    total,
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
                tk_eta_fixed: None,
                tk_x_fixed: None,
            });

            psi_global += 1;
        }
    }

    log::info!(
        "[STAGE] build_psi_hyper_coords axis_count={} workspace_present={} elapsed={:.3}s",
        total_axes,
        psi_workspace.is_some(),
        build_psi_hyper_coords_start.elapsed().as_secs_f64(),
    );

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
            let mut b_operator = hess_ll_op;

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
                // The S_{ψ_i ψ_j} block contribution attaches to the dense
                // Hessian when the family returned a dense `b_mat`, and to
                // the operator-backed Hessian (via a `BlockLocalDrift`
                // composite) when the family returned `hessian_psi_psi`
                // empty alongside an operator. Slicing into a `(0, 0)`
                // dense matrix would otherwise panic in the matrix-free
                // path that survival-marginal-slope and other operator-
                // backed families use.
                if b_mat.nrows() > 0 {
                    let mut b_local =
                        b_mat.slice_mut(s![cache_i.start..cache_i.end, cache_i.start..cache_i.end]);
                    b_local += &s_local;
                } else {
                    let block_drift: Arc<dyn HyperOperator> =
                        Arc::new(crate::solver::estimate::reml::unified::BlockLocalDrift {
                            local: s_local.clone(),
                            start: cache_i.start,
                            end: cache_i.end,
                            total_dim: total,
                        });
                    b_operator = Some(match b_operator.take() {
                        Some(existing) => {
                            let existing_arc: Arc<dyn HyperOperator> = Arc::from(existing);
                            Box::new(
                                crate::solver::estimate::reml::unified::CompositeHyperOperator {
                                    dense: None,
                                    operators: vec![existing_arc, block_drift],
                                    dim_hint: total,
                                },
                            ) as Box<dyn HyperOperator>
                        }
                        None => Box::new(crate::solver::estimate::reml::unified::BlockLocalDrift {
                            local: s_local.clone(),
                            start: cache_i.start,
                            end: cache_i.end,
                            total_dim: total,
                        }) as Box<dyn HyperOperator>,
                    });
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
                b_operator,
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
    let cthf_internal_psi_branch_start = std::time::Instant::now();
    if psi_dim > 0 {
        log::info!(
            "[STAGE] cthf_internal psi_dim={} eval_mode={:?} pre_unified elapsed={:.3}s",
            psi_dim,
            eval_mode,
            cthf_internal_psi_branch_start.elapsed().as_secs_f64(),
        );
        // ψ coordinates present: require exact Newton Hessian for consistency
        // with the psi derivative callbacks.
        let beta_flat = flatten_state_betas(&inner.block_states, specs);
        let synced_joint_states = Arc::new(synchronized_states_from_flat_beta(
            family,
            specs,
            &inner.block_states,
            &beta_flat,
        )?);
        let hessian_workspace = match inner.joint_workspace.clone() {
            Some(workspace) => Some(workspace),
            None => family.exact_newton_joint_hessian_workspace_with_options(
                synced_joint_states.as_ref(),
                specs,
                options,
            )?,
        };
        // Outer-eval entry: prime per-row jet caches before the ext-coord
        // par_iter — see `warm_up_outer_caches` doc.
        if let Some(workspace) = hessian_workspace.as_ref() {
            workspace.warm_up_outer_caches()?;
        }
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
            let h_joint_unpen = if let Some(workspace) = hessian_workspace.as_ref() {
                exact_newton_joint_hessian_source_from_workspace(
                    workspace,
                    total,
                    "joint exact-newton operator mismatch in joint hyper evaluator",
                )?
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
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let block_results: Vec<Result<PenaltyPseudologdet, String>> = (0..specs.len())
                .into_par_iter()
                .map(|b| {
                    let spec = &specs[b];
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
                    PenaltyPseudologdet::from_assembled_with_nullity(s_lambda, structural_nullity)
                })
                .collect();
            let blocks: Result<Vec<_>, _> = block_results.into_iter().collect();
            Some(blocks?)
        } else {
            None
        };

        // Build ψ HyperCoords, pair callbacks, and drift derivative callback.
        let hessian_beta_independent = !family.exact_newton_joint_hessian_beta_dependent();
        let psi_workspace = if eval_mode != EvalMode::ValueOnly
            && (eval_mode == EvalMode::ValueGradientHessian
                || family.exact_newton_joint_psi_workspace_for_first_order_terms())
        {
            family.exact_newton_joint_psi_workspace_with_options(
                synced_joint_states.as_ref(),
                specs,
                derivative_blocks.as_ref(),
                options,
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
            custom_family_batched_outer_hessian_operator(
                family,
                synced_joint_states.as_ref(),
                specs,
                derivative_blocks.as_ref(),
                rho_current,
                hessian_workspace.clone(),
                eval_mode,
            )?,
        )?;

        // The unified evaluator produces gradient/Hessian of size (rho_dim + psi_dim),
        // with ρ coordinates first and ψ coordinates appended — matching the expected
        // output order of CustomFamilyJointHyperResult.
        log::info!(
            "[STAGE] cthf_internal psi_dim={} eval_mode={:?} post_unified elapsed={:.3}s",
            psi_dim,
            eval_mode,
            cthf_internal_psi_branch_start.elapsed().as_secs_f64(),
        );
        return Ok(eval_result);
    }

    // ── ρ-only path (psi_dim == 0): route through unified evaluator ──
    //
    // Batched fast-path: if the family overrides `batched_outer_gradient_terms`
    // and we are in `ValueAndGradient` mode, factor H once at the family level
    // and amortize all K trace computations in a single streaming pass.
    // The cost is still computed via the unified `ValueOnly` path; only the
    // gradient assembly is replaced. See `BatchedOuterGradientTerms`.
    if eval_mode == EvalMode::ValueAndGradient {
        let beta_flat_for_batch = flatten_state_betas(&inner.block_states, specs);
        let synced_states_for_batch = synchronized_states_from_flat_beta(
            family,
            specs,
            &inner.block_states,
            &beta_flat_for_batch,
        )?;
        let workspace_for_batch = match inner.joint_workspace.clone() {
            Some(workspace) => Some(workspace),
            None => family
                .exact_newton_joint_hessian_workspace_with_options(
                    &synced_states_for_batch,
                    specs,
                    options,
                )
                .ok()
                .flatten(),
        };
        let derivative_blocks_for_batch =
            vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()];
        if let Ok(Some(batch)) = family.batched_outer_gradient_terms(
            &synced_states_for_batch,
            specs,
            &derivative_blocks_for_batch,
            rho_current,
            workspace_for_batch.clone(),
        ) {
            // Sanity check: batched output must match (rho_dim + psi_dim).
            let expected = rho_dim + psi_dim;
            if batch.objective_theta.len() == expected
                && batch.trace_h_inv_hdot.len() == expected
                && batch.trace_s_pinv_sdot.len() == expected
            {
                if let Some(joint_bundle_value_only) = build_joint_hessian_closures(
                    family,
                    &inner.block_states,
                    specs,
                    total,
                    options,
                    inner.joint_workspace.clone(),
                )? {
                    let JointHessianBundle {
                        source: h_joint_unpen,
                        beta_flat,
                        compute_dh,
                        compute_d2h,
                        owned_compute_dh: _,
                        owned_compute_d2h: _,
                        rho_curvature_scale,
                        hessian_logdet_correction,
                        ..
                    } = joint_bundle_value_only;
                    let value_only = joint_outer_evaluate(
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
                        EvalMode::ValueOnly,
                        options,
                        family.pseudo_logdet_mode(),
                        compute_dh.as_ref(),
                        compute_d2h.as_ref(),
                        None,
                        None,
                        None,
                        None,
                    )?;
                    // Assemble the gradient via the universal three-term formula:
                    //   grad[k] = obj_θ[k] + 0.5 * tr(H⁻¹ Ḣ_k) - 0.5 * tr(S⁺ Ṡ_k).
                    // This matches `outer_gradient_entry` for fixed-dispersion
                    // families. Profiled-Gaussian families (which scale the
                    // penalty term by dp_cgrad / profiled_scale) currently fall
                    // back to the generic path because they should not opt in
                    // to this hook.
                    let mut gradient = Array1::<f64>::zeros(expected);
                    for j in 0..expected {
                        let trace_term = if include_logdet_h {
                            0.5 * batch.trace_h_inv_hdot[j]
                        } else {
                            0.0
                        };
                        let det_term = if include_logdet_s {
                            0.5 * batch.trace_s_pinv_sdot[j]
                        } else {
                            0.0
                        };
                        gradient[j] = batch.objective_theta[j] + trace_term - det_term;
                    }
                    return Ok(OuterObjectiveEvalResult {
                        objective: value_only.objective,
                        gradient,
                        outer_hessian: crate::solver::outer_strategy::HessianResult::Unavailable,
                        warm_start: value_only.warm_start,
                        inner_converged: inner.converged,
                    });
                }
            }
        }
    }

    // Try build_joint_hessian_closures which handles both exact Newton and
    // surrogate Hessian sources, then call joint_outer_evaluate with no
    // extended coordinates.
    if let Some(joint_bundle) = build_joint_hessian_closures(
        family,
        &inner.block_states,
        specs,
        total,
        options,
        inner.joint_workspace.clone(),
    )? {
        let JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_d2h,
            owned_compute_dh,
            owned_compute_d2h,
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
            owned_compute_dh,
            owned_compute_d2h,
            None, // no ext_coords when psi_dim == 0
            custom_family_batched_outer_hessian_operator(
                family,
                &inner.block_states,
                specs,
                derivative_blocks.as_ref(),
                rho_current,
                inner.joint_workspace.clone(),
                eval_mode,
            )?,
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
                        ndarray::Zip::from(wx.rows_mut())
                            .and(wdx.rows_mut())
                            .and(wwork.view())
                            .par_for_each(|mut wxr, mut wdxr, &wi| {
                                if wi != 1.0 {
                                    wxr.mapv_inplace(|v| v * wi);
                                    wdxr.mapv_inplace(|v| v * wi);
                                }
                            });
                        // Same X'(W·Y) pattern as the parallel sibling at
                        // line ~9258; route through faer for SIMD GEMM
                        // (n × p² flops at biobank moderate scale).
                        correction_mat += &fast_atb(&dx, &wx);
                        correction_mat += &fast_atb(&x_dense, &wdx);
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
                ndarray::Zip::from(scaled_x.rows_mut())
                    .and(&dw)
                    .par_for_each(|mut sr, &dwi| sr.mapv_inplace(|v| v * dwi));
                // X'(diag(dW)·X) outer correction term — faer route, same
                // rationale as above.
                correction_mat += &fast_atb(&x_dense, &scaled_x);

                Ok(Some(DriftDerivResult::Dense(correction_mat)))
            }
        }
    };

    // Build a derivative provider that computes D²_β H_L[u, v] on demand.
    let compute_d2h = |u: &Array1<f64>,
                       v: &Array1<f64>|
     -> Result<Option<DriftDerivResult>, String> {
        if !include_logdet_h {
            return Ok(None);
        }
        match work {
            BlockWorkingSet::ExactNewton { .. } => {
                match family.exact_newton_hessian_second_directional_derivative(
                    &inner.block_states,
                    b,
                    u,
                    v,
                )? {
                    Some(h_exact) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        h_exact,
                        p,
                        &format!("block {b} exact-newton d2H shape mismatch"),
                    )?))),
                    None => Err(format!(
                        "missing exact-newton d2H callback for block {b} while REML Hessian requires H_beta_beta term"
                    )),
                }
            }
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights: _,
            } => {
                let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                    format!("missing dynamic design for block {b} diagonal second correction")
                })?;
                let x_dense = x_dyn.to_dense();
                let n = x_dense.nrows();

                let reject_second_order_geometry = |label: &str,
                                                    geom: Option<
                    BlockGeometryDirectionalDerivative,
                >|
                 -> Result<(), String> {
                    if let Some(geom_dir) = geom {
                        let has_offset = geom_dir.d_offset.iter().any(|value| *value != 0.0);
                        if geom_dir.d_design.is_some() || has_offset {
                            return Err(format!(
                                "block {b} diagonal d2H requires second-order block-geometry derivatives for {label}; use an exact-newton or joint outer path"
                            ));
                        }
                    }
                    Ok(())
                };
                reject_second_order_geometry(
                    "first direction",
                    family.block_geometry_directional_derivative(
                        &inner.block_states,
                        b,
                        spec,
                        u,
                    )?,
                )?;
                reject_second_order_geometry(
                    "second direction",
                    family.block_geometry_directional_derivative(
                        &inner.block_states,
                        b,
                        spec,
                        v,
                    )?,
                )?;

                let d_eta_u = x_dyn.matrixvectormultiply(u);
                let d_eta_v = x_dyn.matrixvectormultiply(v);
                let d2w = family
                    .diagonalworking_weights_second_directional_derivative(
                        &inner.block_states,
                        b,
                        &d_eta_u,
                        &d_eta_v,
                    )?
                    .ok_or_else(|| {
                        format!(
                            "missing diagonal d2W callback for block {b} while REML Hessian requires H_beta_beta term"
                        )
                    })?;
                if d2w.len() != n {
                    return Err(format!(
                        "block {b} diagonal d2W length mismatch: got {}, expected {}",
                        d2w.len(),
                        n
                    ));
                }
                let mut scaled_x = x_dense.clone();
                ndarray::Zip::from(scaled_x.rows_mut())
                    .and(&d2w)
                    .par_for_each(|mut sr, &d2wi| sr.mapv_inplace(|value| value * d2wi));
                Ok(Some(DriftDerivResult::Dense(fast_atb(&x_dense, &scaled_x))))
            }
        }
    };

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
        custom_family_batched_outer_hessian_operator(
            family,
            &inner.block_states,
            specs,
            derivative_blocks.as_ref(),
            rho_current,
            inner.joint_workspace.clone(),
            eval_mode,
        )?,
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
) -> Result<
    (
        crate::solver::outer_strategy::EfsEval,
        ConstrainedWarmStart,
        bool,
    ),
    CustomFamilyError,
> {
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
) -> Result<
    (
        crate::solver::outer_strategy::EfsEval,
        ConstrainedWarmStart,
        bool,
    ),
    CustomFamilyError,
> {
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
    let hessian_workspace = family.exact_newton_joint_hessian_workspace_with_options(
        synced_joint_states.as_ref(),
        specs,
        options,
    )?;
    // Outer-eval entry: prime per-row jet caches before the ext-coord
    // par_iter — see `warm_up_outer_caches` doc.
    if let Some(workspace) = hessian_workspace.as_ref() {
        workspace.warm_up_outer_caches()?;
    }
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
        let h_joint_unpen = if let Some(workspace) = hessian_workspace.as_ref() {
            exact_newton_joint_hessian_source_from_workspace(
                workspace,
                total,
                "joint exact-newton operator mismatch in joint hyper EFS evaluator",
            )?
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
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let block_results: Vec<Result<PenaltyPseudologdet, String>> = (0..specs.len())
            .into_par_iter()
            .map(|b| {
                let spec = &specs[b];
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
                PenaltyPseudologdet::from_assembled_with_nullity(s_lambda, structural_nullity)
            })
            .collect();
        let blocks: Result<Vec<_>, _> = block_results.into_iter().collect();
        Some(blocks?)
    } else {
        None
    };

    let hessian_beta_independent = !family.exact_newton_joint_hessian_beta_dependent();
    let psi_workspace = if family.exact_newton_joint_psi_workspace_for_first_order_terms() {
        family.exact_newton_joint_psi_workspace_with_options(
            synced_joint_states.as_ref(),
            specs,
            derivative_blocks.as_ref(),
            options,
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
        cached_inner: Some(cached_inner_mode_from_result(&inner)),
    };

    Ok((efs_eval, warm, inner.converged))
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
    let (efs_eval, warm_start, inner_converged) = if derivative_blocks.iter().all(Vec::is_empty) {
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
        efs_eval,
        warm_start,
        inner_converged,
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
    let (efs_eval, warm_start, inner_converged) = if derivative_blocks.iter().all(Vec::is_empty) {
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
        efs_eval,
        warm_start,
        inner_converged,
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
const JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N: usize = 128;
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

/// Whether the unified evaluator will pick the matrix-free joint Hessian path
/// for a problem of size `(total_p, total_n)`. Exposed at crate scope so
/// families with matrix-free operators can branch their `coefficient_hessian_cost`
/// estimate on the same predicate the evaluator will use at fit time.
///
/// For biobank-scale row counts with only tens of coefficients, exact
/// materialization is bounded by `total_p` Hessian-vector products and then a
/// tiny dense factorization. That is cheaper and more predictable than PCG when
/// each matrix-free product streams all rows through expensive FLEX marginal-
/// slope kernels and the initial joint Hessian is ill-conditioned. Keep the
/// matrix-free route for genuinely wide joint systems, where `total_p` dense
/// products and factorization dominate.
pub(crate) fn use_joint_matrix_free_path(total_p: usize, total_n: usize) -> bool {
    total_p >= JOINT_MATRIX_FREE_MIN_DIM
        || (total_n >= JOINT_MATRIX_FREE_MIN_ROWS
            && total_p >= JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N)
        || (total_p >= JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N
            && total_n.saturating_mul(total_p) >= JOINT_MATRIX_FREE_MIN_LINEAR_WORK)
}

fn apply_joint_block_penalty(
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    vector: &Array1<f64>,
    diagonal_ridge: f64,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(vector.len());
    apply_joint_block_penalty_into(ranges, s_lambdas, vector, diagonal_ridge, &mut out);
    out
}

/// In-place variant of [`apply_joint_block_penalty`]. Caller supplies the
/// output buffer to eliminate per-call allocation.
///
/// Uses `fast_av_view_into` to write directly into the per-block slice of
/// `out`, avoiding the per-block intermediate `Array1` from `fast_av`. At
/// biobank scale this is invoked inside the PCG matvec closure (called
/// once per CG iter, hundreds-to-thousands of times per outer iter per
/// the perf-scout report).
fn apply_joint_block_penalty_into(
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    vector: &Array1<f64>,
    diagonal_ridge: f64,
    out: &mut Array1<f64>,
) {
    debug_assert_eq!(out.len(), vector.len());
    debug_assert!(s_lambdas.len() <= ranges.len());
    out.fill(0.0);

    if s_lambdas.len() <= 1 {
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            let block = vector.slice(s![start..end]);
            let mut out_slice = out.slice_mut(s![start..end]);
            crate::linalg::faer_ndarray::fast_av_view_into(s_lambda, &block, out_slice.view_mut());
        }
        if diagonal_ridge > 0.0 {
            out.scaled_add(diagonal_ridge, vector);
        }
        return;
    }

    if out.as_slice_mut().is_none() {
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            let block = vector.slice(s![start..end]);
            let mut out_slice = out.slice_mut(s![start..end]);
            crate::linalg::faer_ndarray::fast_av_view_into(s_lambda, &block, out_slice.view_mut());
        }
        if diagonal_ridge > 0.0 {
            out.scaled_add(diagonal_ridge, vector);
        }
        return;
    }

    {
        let out_values = out
            .as_slice_mut()
            .expect("joint penalty output should be contiguous");
        let mut out_blocks = Vec::with_capacity(s_lambdas.len());
        let mut remaining = out_values;
        let mut cursor = 0usize;
        for &(start, end) in ranges.iter().take(s_lambdas.len()) {
            debug_assert!(start >= cursor);
            debug_assert!(end >= start);
            let (_, after_gap) = remaining.split_at_mut(start - cursor);
            let (out_block, after_block) = after_gap.split_at_mut(end - start);
            out_blocks.push(out_block);
            remaining = after_block;
            cursor = end;
        }

        use rayon::prelude::*;

        out_blocks
            .into_par_iter()
            .enumerate()
            .for_each(|(b, out_block)| {
                let (start, end) = ranges[b];
                let block = vector.slice(s![start..end]);
                let out_view = ArrayViewMut1::from(out_block.as_mut());
                crate::linalg::faer_ndarray::fast_av_view_into(&s_lambdas[b], &block, out_view);
            });
    }

    if diagonal_ridge > 0.0 {
        if let (Some(out_values), Some(vector_values)) = (out.as_slice_mut(), vector.as_slice()) {
            use rayon::prelude::*;

            out_values
                .par_iter_mut()
                .zip(vector_values.par_iter())
                .for_each(|(out_value, vector_value)| {
                    *out_value += diagonal_ridge * *vector_value;
                });
        } else {
            out.scaled_add(diagonal_ridge, vector);
        }
    }
}

/// Penalty-aware Jacobi preconditioner used by every matrix-free PCG path
/// in the inner coefficient solve.
///
/// Builds `|diag(H) + Σ_k diag(S_k(λ)) + ridge|`, clamped at 1e-10. This is
/// the diagonal of the full penalized joint Hessian `H + Σ_k λ_k S_k`, so it
/// already incorporates contributions from every penalty operator the model
/// uses — including the cubic-Duchon `[mass, tension, stiffness]` triple
/// (orders [1,2,3] in `WigglePenaltyConfig::cubic_triple_operator_default`).
/// Design docs sometimes call this the "triple-operator penalty
/// preconditioner" for that reason; in code it is the single, unified
/// preconditioner shared by all PCG callsites.
///
/// Callers in the PIRLS inner Newton PCG path feed the result as the diagonal
/// rescale every CG iteration.
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

fn log_joint_pcg_diagnostics(
    cycle: usize,
    total_p: usize,
    total_n: usize,
    preconditioner_diag: &Array1<f64>,
    info: &crate::linalg::utils::PcgSolveInfo,
) {
    let (diag_min, diag_max) = preconditioner_diag.iter().fold(
        (f64::INFINITY, 0.0_f64),
        |(min_value, max_value), &value| {
            if value.is_finite() {
                (min_value.min(value), max_value.max(value))
            } else {
                (min_value, max_value)
            }
        },
    );
    let diag_ratio = if diag_min.is_finite() && diag_min > 0.0 && diag_max.is_finite() {
        Some(diag_max / diag_min)
    } else {
        None
    };
    log::info!(
        "[PIRLS/blockwise joint-Newton/PCG] cycle={} p={} n={} iters={} rel_res={:.3e} res0={:.3e} res_final={:.3e} res_ratio={:.3e} ritz_cond~{} jacobi_diag_ratio~{}",
        cycle,
        total_p,
        total_n,
        info.iterations,
        info.relative_residual_norm,
        info.initial_residual_norm,
        info.final_residual_norm,
        info.residual_reduction,
        info.condition_estimate
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "NA".to_string()),
        diag_ratio
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "NA".to_string()),
    );
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

fn compute_joint_covariance<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Array2<f64>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let Some(mut h) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total,
        "joint exact-newton Hessian shape mismatch in covariance",
    )?
    else {
        return Err(
            "joint covariance requires an exact analytic Hessian; objective perturbation is forbidden"
                .to_string(),
        );
    };
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
    symmetrize_dense_in_place(&mut h);
    if use_exact_newton_strict_spd(family) {
        let (inv, stats) = strict_inverse_spd_with_lm_continuation(&h)?;
        if stats.escalations > 0 {
            log::debug!(
                "[strict-spd] inverse δ-ridge continuation: δ={:.3e}, escalations={}, p={}",
                stats.delta_used,
                stats.escalations,
                h.nrows(),
            );
        }
        Ok(inv)
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
) -> Result<Option<FitGeometry>, String> {
    if specs.len() != per_block_log_lambdas.len() {
        return Ok(None);
    }
    if specs.len() == 1 {
        let eval = family.evaluate(states).ok();
        let Some(eval) = eval else {
            return Ok(None);
        };
        let [
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            },
        ] = eval.blockworking_sets.as_slice()
        else {
            return Ok(None);
        };
        let spec = &specs[0];
        let lambdas = per_block_log_lambdas[0].mapv(f64::exp);
        let Some(mut h) = spec.design.diag_xtw_x(working_weights).ok() else {
            return Ok(None);
        };
        for (k, s) in spec.penalties.iter().enumerate() {
            let s_dense = s.as_dense_cow();
            h.scaled_add(lambdas[k], &*s_dense);
        }
        return Ok(Some(FitGeometry {
            penalized_hessian: h,
            working_weights: working_weights.clone(),
            working_response: working_response.clone(),
        }));
    }

    let requires_explicit_joint_hessian = specs.iter().enumerate().any(|(idx, spec)| {
        custom_family_block_role(&spec.name, idx, specs.len())
            == crate::solver::estimate::BlockRole::LinkWiggle
    });
    let total_p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let Some(mut h) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total_p,
        "compute_joint_geometry",
    )?
    else {
        if requires_explicit_joint_hessian {
            return Err(
                "link-wiggle fits require an exact explicit joint Hessian for posterior sampling"
                    .to_string(),
            );
        }
        return Ok(None);
    };
    let ranges = block_param_ranges(specs);
    for (block_idx, spec) in specs.iter().enumerate() {
        let Some(block_log_lambdas) = per_block_log_lambdas.get(block_idx) else {
            return Ok(None);
        };
        let lambdas = block_log_lambdas.mapv(f64::exp);
        if lambdas.len() != spec.penalties.len() {
            return Ok(None);
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
                return Ok(None);
            }
        }
    }
    let working_len = states.first().map(|state| state.eta.len()).unwrap_or(0);
    Ok(Some(FitGeometry {
        penalized_hessian: h,
        working_weights: Array1::zeros(working_len),
        working_response: Array1::zeros(working_len),
    }))
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
        let geometry = compute_joint_geometry(family, specs, &inner.block_states, &no_pen)
            .map_err(CustomFamilyError::Optimization)?;
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
    if options.inner_max_cycles <= 1 && options.outer_max_iter <= 1 {
        log::info!(
            "[OUTER] custom family: skipping smoothing outer solve for explicit one-cycle inner probe"
        );
        let per_block = split_log_lambdas(&rho0, &penalty_counts)?;
        let mut inner = inner_blockwise_fit(family, specs, &per_block, options, None)?;
        refresh_all_block_etas(family, specs, &mut inner.block_states)
            .map_err(CustomFamilyError::Optimization)?;
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
            "custom-family explicit one-cycle inner probe",
        )
        .map_err(CustomFamilyError::Optimization)?;
        let lambdas = rho0.mapv(f64::exp);
        let log_lambdas = lambdas.mapv(|v| v.max(1e-300).ln());
        return blockwise_fit_from_parts(
            BlockwiseFitResultParts {
                block_states: inner.block_states,
                log_likelihood: inner.log_likelihood,
                log_lambdas,
                lambdas,
                covariance_conditional: None,
                stable_penalty_term: 2.0 * inner.penalty_value,
                penalized_objective,
                outer_iterations: 0,
                outer_gradient_norm: 0.0,
                inner_cycles: inner.cycles,
                outer_converged: inner.converged,
                geometry: None,
            },
            specs,
        )
        .map_err(CustomFamilyError::Optimization);
    }

    use crate::estimate::EstimationError;
    use crate::solver::outer_strategy::{FallbackPolicy, OuterEval, OuterEvalOrder, OuterProblem};

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
    let need_outer_hessian = hessian.is_analytic();
    let outer_max_iter = cost_gated_first_order_max_iter(
        options.outer_max_iter,
        family.coefficient_gradient_cost(specs),
        need_outer_hessian,
    );
    if outer_max_iter < options.outer_max_iter {
        log::info!(
            "[OUTER] custom family: first-order work gate reduced outer_max_iter {} -> {}",
            options.outer_max_iter,
            outer_max_iter,
        );
    }
    // EFS / HybridEfs structural property (`H^{-1/2} B_k H^{-1/2} ≽ 0` plus a
    // parameter-independent nullspace, Wood-Fasiolo) fails for multi-block
    // families whose joint likelihood Hessian depends on β. Disable
    // fixed-point only for genuinely first-order capabilities; exact-Hessian
    // capabilities route to ARC before EFS is considered.
    let multi_block_beta_dependent =
        specs.len() > 1 && family.exact_newton_joint_hessian_beta_dependent();
    // Exact-Hessian plans must fail on their own terms rather than silently
    // retrying on a quasi-Newton surface. First-order-only families keep the
    // automatic cascade because there is no second-order geometry to discard.
    let fallback_policy = if need_outer_hessian {
        FallbackPolicy::Disabled
    } else {
        FallbackPolicy::Automatic
    };
    let problem = OuterProblem::new(n_rho)
        .with_gradient(cap_gradient)
        .with_hessian(hessian.into())
        .with_disable_fixed_point(multi_block_beta_dependent)
        .with_fallback_policy(fallback_policy)
        .with_tolerance(options.outer_tol)
        .with_max_iter(outer_max_iter)
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
            Ok(eval) if !eval.inner_converged => {
                outer.warm_cache = Some(eval.warm_start.clone());
                outer.last_error = Some("custom-family inner solve did not converge".to_string());
                return Err(EstimationError::RemlOptimizationFailed(
                    "custom-family inner solve did not converge".to_string(),
                ));
            }
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
            match outerobjectivegradienthessian_internal(
                family,
                specs,
                &outer_options,
                &penalty_counts,
                rho,
                warm_ref,
                EvalMode::ValueOnly,
            ) {
                Ok(eval) if eval.inner_converged && eval.objective.is_finite() => {
                    outer.warm_cache = Some(eval.warm_start);
                    outer.last_error = None;
                    Ok(eval.objective)
                }
                Ok(eval) => {
                    outer.warm_cache = Some(eval.warm_start);
                    outer.last_error = Some(
                        "custom-family value-only inner solve did not converge or objective was non-finite"
                            .to_string(),
                    );
                    Err(EstimationError::RemlOptimizationFailed(
                        "custom-family value-only inner solve did not converge or objective was non-finite"
                            .to_string(),
                    ))
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
                Ok((eval, warm, true)) => {
                    outer.warm_cache = Some(warm);
                    outer.last_error = None;
                    Ok(eval)
                }
                Ok((_eval, warm, false)) => {
                    outer.warm_cache = Some(warm);
                    outer.last_error =
                        Some("custom-family EFS inner solve did not converge".to_string());
                    Err(EstimationError::RemlOptimizationFailed(
                        "custom-family EFS inner solve did not converge".to_string(),
                    ))
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

    let geometry = compute_joint_geometry(family, specs, &inner.block_states, &per_block)
        .map_err(CustomFamilyError::Optimization)?;
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
    if !inner.converged {
        return Err(CustomFamilyError::Optimization(format!(
            "fixed-log-lambda inner solve did not converge after {} cycles",
            inner.cycles
        )));
    }
    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let covariance_conditional =
        compute_joint_covariance_required(family, specs, &inner.block_states, &per_block, options)?;
    let geometry = compute_joint_geometry(family, specs, &inner.block_states, &per_block)
        .map_err(CustomFamilyError::Optimization)?;
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
    #[derive(Clone)]
    struct BatchedOuterHessianTestFamily {
        matrix: Array2<f64>,
    }

    struct TestOuterHessianOperator {
        matrix: Array2<f64>,
    }

    impl crate::solver::outer_strategy::OuterHessianOperator for TestOuterHessianOperator {
        fn dim(&self) -> usize {
            self.matrix.nrows()
        }

        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            Ok(self.matrix.dot(v))
        }

        fn is_cheap_to_materialize(&self) -> bool {
            true
        }
    }

    impl CustomFamily for BatchedOuterHessianTestFamily {
        fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![],
            })
        }

        fn outer_hyper_hessian_hvp_available(&self, _specs: &[ParameterBlockSpec]) -> bool {
            true
        }

        fn outer_hyper_hessian_operator(
            &self,
            _specs: &[ParameterBlockSpec],
        ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
            Some(Arc::new(TestOuterHessianOperator {
                matrix: self.matrix.clone(),
            }))
        }
    }

    #[test]
    fn batched_outer_hessian_terms_materialize_to_exact_small_matrix() {
        let exact = array![[4.0, -1.0], [-1.0, 3.0]];
        let family = BatchedOuterHessianTestFamily {
            matrix: exact.clone(),
        };
        let terms = family
            .batched_outer_hessian_terms(&[], &[], &[], &array![0.0, 0.0], None)
            .expect("batched Hessian hook succeeds")
            .expect("test family exposes batched HVP terms");
        let operator = match terms.outer_hessian {
            crate::solver::outer_strategy::HessianResult::Operator(operator) => operator,
            _ => panic!("batched hook should expose an operator"),
        };
        let dense = operator
            .mul_mat(Array2::<f64>::eye(2).view())
            .expect("operator materializes on small exact case");
        assert_eq!(dense, exact);
    }

    #[test]
    fn batched_outer_hessian_operator_selected_only_for_hessian_eval() {
        let family = BatchedOuterHessianTestFamily {
            matrix: array![[2.0, 0.5], [0.5, 5.0]],
        };
        let selected = custom_family_batched_outer_hessian_operator(
            &family,
            &[],
            &[],
            &[],
            &array![1.0, -1.0],
            None,
            EvalMode::ValueGradientHessian,
        )
        .expect("selection check succeeds");
        assert!(
            selected.is_some(),
            "supported Hessian/HVP families should select the batched operator path"
        );

        let not_selected = custom_family_batched_outer_hessian_operator(
            &family,
            &[],
            &[],
            &[],
            &array![1.0, -1.0],
            None,
            EvalMode::ValueAndGradient,
        )
        .expect("non-Hessian selection check succeeds");
        assert!(
            not_selected.is_none(),
            "batched Hessian terms must not run for gradient-only evaluations"
        );
    }

    use super::*;
    use crate::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
    use crate::families::gamlss::{BinomialLocationScaleFamily, BinomialLocationScaleWiggleFamily};
    use crate::matrix::DesignMatrix;
    use crate::smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
        build_term_collection_design, freeze_term_collection_from_design,
        spatial_length_scale_term_indices, try_build_spatial_log_kappa_derivativeinfo_list,
    };
    use crate::testing::binomial_location_scale_base_fixture;
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

    struct BinomialLocationScaleWiggleOuterFixture {
        family: BinomialLocationScaleWiggleFamily,
        specs: Vec<ParameterBlockSpec>,
        penalty_counts: Vec<usize>,
        rho: Array1<f64>,
        options: BlockwiseFitOptions,
    }

    fn binomial_location_scale_wiggle_outer_fixture() -> BinomialLocationScaleWiggleOuterFixture {
        let base = binomial_location_scale_base_fixture();
        let q_seed = Array1::linspace(-1.4, 1.4, base.n);
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
            y: base.y,
            weights: base.weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            threshold_design: Some(base.threshold_design),
            log_sigma_design: Some(base.log_sigma_design),
            wiggle_knots: knots,
            wiggle_degree: 3,
            policy: crate::resource::ResourcePolicy::default_library(),
        };
        BinomialLocationScaleWiggleOuterFixture {
            family,
            specs: vec![base.threshold_spec, base.log_sigma_spec, wigglespec],
            penalty_counts: vec![1usize, 1usize, 1usize],
            rho: array![0.05, -0.15, 0.1],
            options: BlockwiseFitOptions {
                use_remlobjective: true,
                ridge_floor: 1e-10,
                outer_max_iter: 1,
                ..BlockwiseFitOptions::default()
            },
        }
    }

    #[derive(Clone)]
    struct OneBlockIdentityFamily;

    #[test]
    fn joint_coupled_coefficient_hessian_cost_matches_n_times_p_total_squared() {
        // Three blocks p_b = (12, 20, 8), n=200. Joint-coupled cost is
        // n·(Σp_b)² = 200·40² = 320_000. Block-diagonal default with the
        // same designs would give n·Σp_b² = 200·(144+400+64) = 121_600.
        // The cross-block fill 2·n·(p_t·p_m + p_t·p_l + p_m·p_l) =
        // 2·200·(240+96+160) = 198_400 accounts for the difference.
        let mk_spec = |p: usize| ParameterBlockSpec {
            name: "test".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
                200, p,
            )))),
            offset: Array1::zeros(200),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };
        let specs = vec![mk_spec(12), mk_spec(20), mk_spec(8)];
        assert_eq!(
            joint_coupled_coefficient_hessian_cost(200, &specs),
            200 * 40 * 40
        );
        assert_eq!(
            default_coefficient_hessian_cost(&specs),
            200 * (144 + 400 + 64)
        );
        assert!(
            joint_coupled_coefficient_hessian_cost(200, &specs)
                > default_coefficient_hessian_cost(&specs)
        );
    }

    #[test]
    fn biobank_exact_adaptive_hessian_order_stays_second_order() {
        let n_train = 320_000u64;
        let p = 101usize;
        let retained_rho_dim = 3usize;
        let spec = ParameterBlockSpec {
            name: "matern60".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
                1, p,
            )))),
            offset: Array1::zeros(1),
            penalties: (0..retained_rho_dim)
                .map(|_| PenaltyMatrix::Dense(Array2::eye(p)))
                .collect(),
            nullspace_dims: vec![0; retained_rho_dim],
            initial_log_lambdas: Array1::zeros(retained_rho_dim),
            initial_beta: None,
        };
        let coefficient_hessian_cost = n_train * (p as u64) * (p as u64);

        assert_eq!(coefficient_hessian_cost, 3_264_320_000);
        assert_eq!(
            retained_rho_dim as u64 * coefficient_hessian_cost,
            9_792_960_000
        );
        assert_eq!(
            exact_outer_order_from_capability(&[spec], coefficient_hessian_cost),
            ExactOuterDerivativeOrder::Second
        );
    }

    #[test]
    fn use_joint_matrix_free_path_triggers_at_each_documented_threshold() {
        // p ≥ 512 is sufficient regardless of n.
        assert!(use_joint_matrix_free_path(512, 1));
        assert!(use_joint_matrix_free_path(2048, 4));
        assert!(!use_joint_matrix_free_path(511, 1));

        // n ≥ 50_000 AND p ≥ 128: both must hold. This keeps p≈51 FLEX
        // marginal-slope biobank fits on the bounded dense-materialized path.
        assert!(use_joint_matrix_free_path(128, 50_000));
        assert!(!use_joint_matrix_free_path(127, 50_000));
        assert!(!use_joint_matrix_free_path(128, 31_249));
        assert!(!use_joint_matrix_free_path(51, 320_000));

        // n · p ≥ 4_000_000 is the linear-work fallback, but only after the
        // same moderate-p guard; below that, materializing `p` columns is a
        // deterministic small-p bound on expensive row-kernel HVPs.
        assert!(use_joint_matrix_free_path(128, 31_250));
        assert!(!use_joint_matrix_free_path(127, 31_497));

        // Below every threshold: dense path.
        assert!(!use_joint_matrix_free_path(8, 100));
        assert!(!use_joint_matrix_free_path(64, 1000));
    }

    #[test]
    #[ignore = "biobank-shape routing/timing guard; cheap but excluded from default unit lane"]
    fn biobank_shape_margslope_flex_cycle0_uses_bounded_dense_route() {
        let total_p = 51;
        let total_n = 320_000;
        let max_pcg_hvps_before_fix = JOINT_PCG_MAX_ITER_MULTIPLIER * total_p;

        assert_eq!(max_pcg_hvps_before_fix, 204);
        assert!(
            !use_joint_matrix_free_path(total_p, total_n),
            "p=51/n=320k should materialize exactly 51 columns instead of risking up to {max_pcg_hvps_before_fix} expensive PCG matvecs in cycle 0"
        );
    }

    struct CountingHessianWorkspace {
        dense_calls: Arc<AtomicUsize>,
        matvec_calls: Arc<AtomicUsize>,
    }

    impl ExactNewtonJointHessianWorkspace for CountingHessianWorkspace {
        fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
            self.dense_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Some(Array2::eye(2)))
        }

        fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
            self.matvec_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Some(v.clone()))
        }

        fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
            Ok(Some(Array1::ones(2)))
        }

        fn directional_derivative(&self, _: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
            Ok(None)
        }
    }

    #[test]
    fn workspace_hessian_source_prefers_dense_without_zero_matvec_probe() {
        let dense_calls = Arc::new(AtomicUsize::new(0));
        let matvec_calls = Arc::new(AtomicUsize::new(0));
        let workspace: Arc<dyn ExactNewtonJointHessianWorkspace> =
            Arc::new(CountingHessianWorkspace {
                dense_calls: Arc::clone(&dense_calls),
                matvec_calls: Arc::clone(&matvec_calls),
            });

        let source =
            exact_newton_joint_hessian_source_from_workspace(&workspace, 2, "counting workspace")
                .expect("hessian source should build")
                .expect("hessian source should be present");

        assert_eq!(dense_calls.load(Ordering::Relaxed), 1);
        assert_eq!(matvec_calls.load(Ordering::Relaxed), 0);
        match source {
            JointHessianSource::Dense(hessian) => assert_eq!(hessian, Array2::<f64>::eye(2)),
            JointHessianSource::Operator { .. } => panic!("dense source was not preferred"),
        }
        assert_eq!(matvec_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn default_coefficient_gradient_cost_is_half_of_hessian_cost() {
        // The gradient-only sweep through the inner Newton solve does
        // roughly half the per-evaluation arithmetic of the full Hessian
        // assembly path (skips K-fold pairwise B_{j,k} blocks and K-fold
        // inner derivative solves). The default trait method preserves
        // this 2× ratio; families that override `coefficient_hessian_cost`
        // (e.g. GAMLSS via `joint_coupled_coefficient_hessian_cost`)
        // automatically inherit a consistent gradient-cost scaling without
        // a per-family override.
        let mk_spec = |n: usize, p: usize| ParameterBlockSpec {
            name: "test".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
                n, p,
            )))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };
        let specs = vec![mk_spec(500, 10), mk_spec(500, 14)];
        let h_cost = default_coefficient_hessian_cost(&specs);
        let g_cost = default_coefficient_gradient_cost(&specs);
        assert_eq!(h_cost, 500 * 100 + 500 * 196);
        assert_eq!(g_cost, h_cost / 2);
    }

    #[test]
    fn first_order_outer_iter_gate_caps_expensive_gradient_paths() {
        assert_eq!(
            cost_gated_first_order_max_iter(60, 10_000_000_000, false),
            8
        );
        assert_eq!(
            cost_gated_first_order_max_iter(60, 100_000_000_000, false),
            4
        );
        assert_eq!(
            cost_gated_first_order_max_iter(60, 100_000_000_000, true),
            60
        );
    }

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
    fn screened_outer_warm_start_reuses_any_matching_rho_dimension() {
        let rho_far = array![2.25, -0.5];
        let cache = Some(ConstrainedWarmStart {
            rho: array![0.0, -0.5],
            block_beta: vec![array![1.0, -1.0]],
            active_sets: vec![None],
            cached_inner: None,
        });

        let retained = screened_outer_warm_start(cache.as_ref(), &rho_far)
            .expect("matching-dimension warm starts should remain reusable");
        assert_eq!(retained.rho, array![0.0, -0.5]);
        assert_eq!(retained.block_beta[0], array![1.0, -1.0]);
        assert_eq!(retained.active_sets[0], None);
    }

    #[test]
    fn public_warm_start_compatibility_checks_rho_dimension() {
        let warm = CustomFamilyWarmStart {
            inner: ConstrainedWarmStart {
                rho: array![0.0, -0.5],
                block_beta: vec![array![1.0, -1.0]],
                active_sets: vec![None],
                cached_inner: None,
            },
        };

        assert!(warm.compatible_with_rho(&array![0.75, -0.5]));
        assert!(warm.compatible_with_rho(&array![1.75, -0.5]));
        assert!(!warm.compatible_with_rho(&array![0.0]));
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
                        total_dim: 3,
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
    fn custom_family_outer_derivatives_respects_missing_second_order_capability() {
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
            crate::solver::outer_strategy::DeclaredHessianForm::Unavailable
        );
    }

    #[derive(Clone)]
    struct DefaultDiagonalExactHookFamily;

    impl CustomFamily for DefaultDiagonalExactHookFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let eta = block_states[0].eta.clone();
            let weights = eta.mapv(|value| 2.0 + value * value);
            Ok(FamilyEvaluation {
                log_likelihood: -0.5 * eta.dot(&eta),
                blockworking_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(eta.len()),
                    working_weights: weights,
                }],
            })
        }

        fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
            true
        }

        fn diagonalworking_weights_directional_derivative(
            &self,
            block_states: &[ParameterBlockState],
            _: usize,
            d_eta: &Array1<f64>,
        ) -> Result<Option<Array1<f64>>, String> {
            Ok(Some((&block_states[0].eta * d_eta) * 2.0))
        }

        fn exact_newton_joint_hessiansecond_directional_derivative(
            &self,
            block_states: &[ParameterBlockState],
            u: &Array1<f64>,
            v: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let spec = default_diagonal_exact_hook_spec();
            let u_eta = spec.design.apply(u);
            let v_eta = spec.design.apply(v);
            assert_eq!(block_states[0].eta.len(), u_eta.len());
            spec.design.diag_xtw_x(&((&u_eta * &v_eta) * 2.0)).map(Some)
        }
    }

    fn default_diagonal_exact_hook_spec() -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: "default_exact".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0, 0.5],
                [0.0, 1.0],
                [2.0, -1.0]
            ])),
            offset: Array1::zeros(3),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(2))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2, -0.1]),
        }
    }

    #[test]
    fn default_custom_family_exact_hessian_hooks_assemble_diagonal_working_sets() {
        let family = DefaultDiagonalExactHookFamily;
        let spec = default_diagonal_exact_hook_spec();
        let beta = array![0.2, -0.1];
        let eta = spec.design.apply(&beta);
        let states = vec![ParameterBlockState {
            beta: beta.clone(),
            eta: eta.clone(),
        }];

        let h = family
            .exact_newton_joint_hessian_with_specs(&states, &[spec.clone()])
            .expect("default joint Hessian hook should succeed")
            .expect("diagonal working sets should assemble an exact joint Hessian");
        let expected_h = spec
            .design
            .diag_xtw_x(&eta.mapv(|value| 2.0 + value * value))
            .unwrap();
        assert_eq!(h, expected_h);

        let direction = array![0.3, -0.4];
        let dh = family
            .exact_newton_joint_hessian_directional_derivative_with_specs(
                &states,
                &[spec.clone()],
                &direction,
            )
            .expect("default joint dH hook should succeed")
            .expect("diagonal weight derivative should assemble an exact joint dH");
        let d_eta = spec.design.apply(&direction);
        let expected_dh = spec.design.diag_xtw_x(&((&eta * &d_eta) * 2.0)).unwrap();
        assert_eq!(dh, expected_dh);

        let d2h = family
            .exact_newton_joint_hessiansecond_directional_derivative(&states, &direction, &beta)
            .expect("family second directional hook should succeed")
            .expect("second directional hook should be exact");
        let beta_eta = spec.design.apply(&beta);
        let expected_d2h = spec
            .design
            .diag_xtw_x(&((&d_eta * &beta_eta) * 2.0))
            .unwrap();
        assert_eq!(d2h, expected_d2h);
    }

    #[test]
    fn default_custom_family_exact_hessian_hooks_drive_profiled_outer_hessian() {
        let spec = default_diagonal_exact_hook_spec();
        let result = evaluate_custom_family_joint_hyper(
            &DefaultDiagonalExactHookFamily,
            &[spec],
            &BlockwiseFitOptions {
                use_remlobjective: true,
                use_outer_hessian: true,
                compute_covariance: false,
                inner_max_cycles: 1,
                ..BlockwiseFitOptions::default()
            },
            &array![0.0],
            &[vec![]],
            None,
            EvalMode::ValueGradientHessian,
        )
        .expect("profiled outer Hessian should use default exact Hessian hooks");

        assert_eq!(result.gradient.len(), 1);
        match result.outer_hessian {
            crate::solver::outer_strategy::HessianResult::Analytic(hessian) => {
                assert_eq!(hessian.dim(), (1, 1));
                assert!(hessian[[0, 0]].is_finite());
            }
            _ => panic!("outer Hessian should be analytic"),
        }
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
        assert_eq!(
            hessian,
            crate::solver::outer_strategy::DeclaredHessianForm::Either
        );
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
        assert_eq!(
            hessian,
            crate::solver::outer_strategy::DeclaredHessianForm::Either
        );
    }

    #[derive(Clone)]
    struct OneBlockQuarticExactFamily {
        linear: f64,
        curvature: f64,
        second_scale: f64,
    }

    impl CustomFamily for OneBlockQuarticExactFamily {
        fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
            // h(β) = 1 + curvature·β² genuinely depends on β; the default
            // (false for RidgedQuadraticReml) would short-circuit the joint
            // d²H aggregator to zeros and drop the per-block override below
            // before it ever reaches the outer Hessian's drift contribution.
            true
        }

        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let beta = block_states[0].beta[0];
            let log_likelihood =
                self.linear * beta - 0.5 * beta * beta - self.curvature * beta.powi(4) / 12.0;
            let gradient = self.linear - beta - self.curvature * beta.powi(3) / 3.0;
            let hessian = 1.0 + self.curvature * beta * beta;
            Ok(FamilyEvaluation {
                log_likelihood,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![gradient],
                    hessian: SymmetricMatrix::Dense(array![[hessian]]),
                }],
            })
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            block_states: &[ParameterBlockState],
            block_idx: usize,
            direction: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert_eq!(block_idx, 0);
            let beta = block_states[0].beta[0];
            Ok(Some(array![[2.0 * self.curvature * beta * direction[0]]]))
        }

        fn exact_newton_hessian_second_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            block_idx: usize,
            u: &Array1<f64>,
            v: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert_eq!(block_idx, 0);
            let value = 2.0 * self.curvature * self.second_scale * u[0] * v[0];
            Ok(Some(array![[value]]))
        }
    }

    #[test]
    fn generic_single_block_fallback_includes_nonzero_d2h_drift() {
        let spec = ParameterBlockSpec {
            name: "quartic".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.75]),
        };
        let options = BlockwiseFitOptions {
            inner_tol: 1e-11,
            use_remlobjective: true,
            use_outer_hessian: true,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        let penalty_counts = vec![1];
        let rho = array![0.0];

        let with_d2 = evaluate_custom_family_hyper_internal(
            &OneBlockQuarticExactFamily {
                linear: 3.0,
                curvature: 0.5,
                second_scale: 1.0,
            },
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho,
            &[vec![]],
            None,
            EvalMode::ValueGradientHessian,
        )
        .expect("single-block fallback with exact d2H should evaluate");
        let without_d2_contribution = evaluate_custom_family_hyper_internal(
            &OneBlockQuarticExactFamily {
                linear: 3.0,
                curvature: 0.5,
                second_scale: 0.0,
            },
            &[spec],
            &options,
            &penalty_counts,
            &rho,
            &[vec![]],
            None,
            EvalMode::ValueGradientHessian,
        )
        .expect("single-block fallback with zero d2H should evaluate");

        let h_with = with_d2.outer_hessian.unwrap_analytic();
        let h_without = without_d2_contribution.outer_hessian.unwrap_analytic();
        let d2h_delta = h_with[[0, 0]] - h_without[[0, 0]];
        assert!(
            d2h_delta.abs() > 1e-8,
            "expected nonzero outer Hessian contribution from d2H; with={:?}, without={:?}",
            h_with,
            h_without
        );
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
        assert_eq!(
            hessian,
            crate::solver::outer_strategy::DeclaredHessianForm::Either
        );
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

        let policy = ResourcePolicy::permissive_small_data();
        let x_cross_01_map = resolve_custom_family_x_psi_psi_map(
            &derivs[0],
            &derivs[1],
            1,
            resolved_design.design.nrows(),
            resolved_design.design.ncols(),
            0..resolved_design.design.nrows(),
            "psi0 cross design",
            &policy,
        )
        .expect("resolve psi0 cross design");
        let x_cross_10_map = resolve_custom_family_x_psi_psi_map(
            &derivs[1],
            &derivs[0],
            0,
            resolved_design.design.nrows(),
            resolved_design.design.ncols(),
            0..resolved_design.design.nrows(),
            "psi1 cross design",
            &policy,
        )
        .expect("resolve psi1 cross design");
        let x_cross_01 = x_cross_01_map
            .row_chunk(0..resolved_design.design.nrows())
            .expect("materialize psi0 cross design");
        let x_cross_10 = x_cross_10_map
            .row_chunk(0..resolved_design.design.nrows())
            .expect("materialize psi1 cross design");
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

        fn diagonalworking_weights_second_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: usize,
            d_eta_u: &Array1<f64>,
            _: &Array1<f64>,
        ) -> Result<Option<Array1<f64>>, String> {
            Ok(Some(Array1::zeros(d_eta_u.len())))
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
            line_search_prefer_workspace: false,
            early_exit_threshold: None,
            outer_score_subsample: None,
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
            line_search_prefer_workspace: false,
            early_exit_threshold: None,
            outer_score_subsample: None,
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
            line_search_prefer_workspace: false,
            early_exit_threshold: None,
            outer_score_subsample: None,
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
    fn pseudo_laplace_exact_newton_lm_continuation_recovers_marginal_indefiniteness() {
        // Marginally-indefinite Hessian (here a 1×1 block with H=-1) used to
        // be rejected outright by the strict pseudo-Laplace path; the
        // adaptive-seed pilot would lose its entire seed budget and the
        // controller fall-opened into an unguarded full-data path. With
        // `strict_logdet_spd_with_lm_continuation` the strict-mode logdet
        // factors `H + δI` along the same δ-ridge schedule the strict solve
        // already uses, so a marginally-indefinite block is saved by a
        // small, principled regularization and the seed proceeds.
        //
        // For the 1×1 H=[[-1]] family below: trace_scale=1, δ₀=1e-12,
        // δ grows ×10 per escalation, so δ=10 (escalation 14) lifts H to
        // H+δI=[[9]] which Cholesky factors cleanly (logdet=log 9≈2.197).
        // The fit therefore succeeds where it formerly errored.
        let spec = ParameterBlockSpec {
            name: "marginally_indefinite".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let result = fit_custom_family(
            &OneBlockIndefinitePseudoLaplaceFamily,
            &[spec],
            &BlockwiseFitOptions {
                use_remlobjective: true,
                compute_covariance: false,
                ..BlockwiseFitOptions::default()
            },
        );
        assert!(
            result.is_ok(),
            "LM δ-ridge continuation should save the marginally-indefinite \
             Hessian H=[[-1]]; got error: {:?}",
            result.err(),
        );
    }

    #[test]
    fn strict_logdet_lm_continuation_accepts_marginal_indefiniteness() {
        // Direct test of the continuation primitive: H=[[-1]] is rejected
        // by bare Cholesky but the LM δ-ridge schedule lifts it to
        // H+δI=[[9]] at δ=10, yielding logdet=log(9). The reported stats
        // must show a non-trivial escalation count so a recurring need
        // for nontrivial ridges is detectable by the controller.
        let h = array![[-1.0_f64]];
        let (logdet, stats) = strict_logdet_spd_with_lm_continuation(&h)
            .expect("LM continuation must save 1×1 marginally-indefinite H");
        assert!(
            (logdet - (9.0_f64).ln()).abs() < 1e-12,
            "logdet={logdet}, expected log(9)={}",
            (9.0_f64).ln(),
        );
        assert!(stats.escalations > 0, "δ-ridge escalation must have fired");
        assert!(
            stats.delta_used >= 1.0,
            "δ_used={} must exceed |min_eig|=1",
            stats.delta_used,
        );
    }

    #[test]
    fn strict_logdet_lm_continuation_passes_through_when_h_is_already_spd() {
        // When H is already SPD the bare strict logdet succeeds and the
        // continuation must report no escalation (δ_used=0) — i.e. callers
        // pay no extra cost on the well-conditioned hot path.
        let h = array![[2.0_f64, 0.5], [0.5, 3.0]];
        let (logdet, stats) = strict_logdet_spd_with_lm_continuation(&h)
            .expect("strict logdet must succeed on SPD H");
        let expected = (2.0_f64 * 3.0 - 0.5 * 0.5).ln();
        assert!(
            (logdet - expected).abs() < 1e-12,
            "logdet={logdet}, expected={expected}",
        );
        assert_eq!(stats.escalations, 0);
        assert_eq!(stats.delta_used, 0.0);
    }

    #[test]
    fn auto_determinant_mode_is_exact_full_logdet_policy() {
        let h = array![[6.0, 0.8, 0.1], [0.8, 4.5, 0.4], [0.1, 0.4, 3.2]];
        let exact = stable_logdet_with_ridge_policy(
            &h,
            1e-8,
            RidgePolicy::explicit_stabilization_full_exact(),
        )
        .expect("exact logdet");
        let auto =
            stable_logdet_with_ridge_policy(&h, 1e-8, RidgePolicy::explicit_stabilization_full())
                .expect("auto logdet");
        assert!((auto - exact).abs() < 1e-12, "auto={auto}, exact={exact}");
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
        let BinomialLocationScaleWiggleOuterFixture {
            family,
            specs,
            penalty_counts,
            rho,
            options: base_options,
        } = binomial_location_scale_wiggle_outer_fixture();
        // FD/analytic noise floor below is `EPS·|cost|/h`, valid only when PIRLS
        // converges to f64 precision; HardPseudo + σ_min~1e-10 amplifies the
        // default 1e-6 inner residual into ~1e-7 cost slack that lifts both
        // estimators above the machine-precision floor.
        let options = BlockwiseFitOptions {
            inner_tol: 1e-12,
            inner_max_cycles: 500,
            ..base_options
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
        let BinomialLocationScaleWiggleOuterFixture {
            family,
            specs,
            penalty_counts,
            rho,
            options,
        } = binomial_location_scale_wiggle_outer_fixture();

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
            policy: crate::resource::ResourcePolicy::default_library(),
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
            policy: crate::resource::ResourcePolicy::default_library(),
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
        // Asymmetric y (6 ones / 4 zeros). A balanced 5/5 vector forces
        // β̂_threshold = 0 by probit-link symmetry, which makes the joint
        // observed Hessian block-diagonal in (threshold, log_sigma) at the
        // inner mode. The outer LAML Hessian off-diagonals are then ~1e-11,
        // below the central-FD noise floor (≈ pirls_tol / h) at h=1e-5, so
        // FD-vs-analytic agreement cannot be enforced. Asymmetric y gives
        // β̂_threshold ≠ 0, coupling the (β_0, β_1) blocks through the
        // observed-information weights and making all four entries validatable.
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
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
            policy: crate::resource::ResourcePolicy::default_library(),
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
            policy: crate::resource::ResourcePolicy::default_library(),
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

    // Exact analytic Hessians must be finite. Non-finite Hessians are rejected
    // loudly instead of being masked by a surrogate update.

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
    fn exact_newton_nan_hessian_fails_loudly_before_eigendecomposition() {
        // Exact Newton Hessians are part of the mathematical contract.  A
        // NaN in a block Hessian means the family derivative is invalid; we
        // should reject it at the logdet boundary instead of hiding it behind
        // a conservative eigendecomposition fallback.
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
        let err = result.expect_err("NaN exact Hessian must fail loudly");
        assert!(
            err.contains("smooth-regularized logdet Hessian contains non-finite entry"),
            "expected explicit non-finite Hessian error, got: {err}"
        );
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

    /// Regression check: when `strict_solve_spd_with_lm_continuation` is given a
    /// strongly negative-definite matrix whose `|λ_min|` exceeds the LM δ-ridge
    /// schedule's terminal δ (≈ ε · trace_scale · 10¹⁶), the bare schedule can't
    /// rescue Cholesky and the terminal eigen-floor fallback must return a
    /// finite solution equal to `Q diag(1/Λ̃) Qᵀ rhs`, with
    /// `Λ̃_i = max(Λ_i, ε λ_max)`.
    ///
    /// We also exercise the schedule-success path with a milder matrix to lock
    /// in that the eigen-floor doesn't perturb the LM-δ output for cases the
    /// schedule can already handle.
    #[test]
    fn strict_solve_spd_falls_back_to_eigen_floor_on_indefinite_matrix() {
        // δ schedule from `delta0 = max(ε·tr/p, 1e-12)`, growth 10×, 16 steps.
        // With `tr = 4·1e30` we get `delta0 ≈ ε·1e30 ≈ 2.2e14`; terminal δ at
        // escalation 16 is `2.2e14 · 1e16 = 2.2e30`. Set `λ_min ≈ -1e32` to
        // outpace the schedule and force the eigen-floor branch.
        let p = 4usize;
        let mut h = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            h[[i, i]] = -1e32 - (i as f64) * 1e30;
        }
        h[[0, 1]] = 5e29;
        h[[1, 0]] = 5e29;
        let rhs = Array1::from_vec(vec![1e30, -5e29, 2.5e29, 7.5e29]);

        let (x, stats) = strict_solve_spd_with_lm_continuation(&h, &rhs)
            .expect("eigen-floor fallback must succeed on the negative-definite matrix");
        assert!(
            stats.escalations > 16,
            "expected eigen-floor terminal fallback (escalations > MAX_ESCALATIONS), got {}",
            stats.escalations,
        );
        for &v in x.iter() {
            assert!(
                v.is_finite(),
                "eigen-floor solve returned non-finite component {v}"
            );
        }

        // Reconstruct the analytic floored solve and compare component-wise.
        let mut sym = h.clone();
        symmetrize_dense_in_place(&mut sym);
        let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).expect("eigh");
        let max_abs_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let eps_floor = (1e-12 * max_abs_eval).max(1e-300);
        let mut want = Array1::<f64>::zeros(p);
        for k in 0..p {
            let mut q_t_rhs = 0.0;
            for i in 0..p {
                q_t_rhs += evecs[[i, k]] * rhs[i];
            }
            let scaled = q_t_rhs / evals[k].max(eps_floor);
            for i in 0..p {
                want[i] += evecs[[i, k]] * scaled;
            }
        }
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - x[i]).abs() <= tol,
                "eigen-floor solve component {i}: want={:.6e}, got={:.6e}",
                want[i],
                x[i],
            );
        }
    }

    /// Companion regression check for `strict_logdet_spd_with_lm_continuation`:
    /// on a strongly negative-definite matrix the LM δ-ridge schedule cannot
    /// factor `H + δI` (`|λ_min|` exceeds the terminal δ), so the eigen-floor
    /// fallback must return `Σ log Λ̃_i` with `Λ̃_i = max(Λ_i, ε λ_max)`.
    #[test]
    fn strict_logdet_spd_falls_back_to_eigen_floor_on_indefinite_matrix() {
        let p = 4usize;
        let mut h = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            h[[i, i]] = -1e32 - (i as f64) * 1e30;
        }
        let (logdet, stats) = strict_logdet_spd_with_lm_continuation(&h)
            .expect("eigen-floor logdet fallback must succeed");
        assert!(
            stats.escalations > 16,
            "expected eigen-floor terminal fallback for logdet, got escalations={}",
            stats.escalations,
        );
        let mut sym = h.clone();
        symmetrize_dense_in_place(&mut sym);
        let (evals, _) = FaerEigh::eigh(&sym, Side::Lower).expect("eigh");
        let max_abs_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let eps_floor = (1e-12 * max_abs_eval).max(1e-300);
        let want: f64 = evals.iter().map(|&ev| ev.max(eps_floor).ln()).sum();
        let tol = 1e-10 * want.abs().max(1.0) + 1e-10;
        assert!(
            (want - logdet).abs() <= tol,
            "eigen-floor logdet: want={:.6e}, got={:.6e}",
            want,
            logdet,
        );
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

    #[test]
    fn zero_psi_derivative_operator_acts_as_zero_map() {
        let n = 17usize;
        let p = 5usize;
        let op = ZeroPsiDerivativeOperator::new(n, p);

        assert_eq!(op.n_data(), n);
        assert_eq!(op.p_out(), p);

        let u = Array1::from_iter((0..p).map(|k| 1.0 + k as f64));
        let v = Array1::from_iter((0..n).map(|k| 1.0 - 0.5 * k as f64));

        let fwd = op.forward_mul(0, &u.view()).expect("forward_mul");
        assert_eq!(fwd.len(), n);
        assert!(fwd.iter().all(|x| *x == 0.0));

        let trn = op.transpose_mul(0, &v.view()).expect("transpose_mul");
        assert_eq!(trn.len(), p);
        assert!(trn.iter().all(|x| *x == 0.0));

        let fwd2 = op
            .forward_mul_second_diag(0, &u.view())
            .expect("forward_mul_second_diag");
        assert_eq!(fwd2.len(), n);
        assert!(fwd2.iter().all(|x| *x == 0.0));

        let trn2 = op
            .transpose_mul_second_diag(0, &v.view())
            .expect("transpose_mul_second_diag");
        assert_eq!(trn2.len(), p);
        assert!(trn2.iter().all(|x| *x == 0.0));

        let fwd_cross = op
            .forward_mul_second_cross(0, 1, &u.view())
            .expect("forward_mul_second_cross");
        assert_eq!(fwd_cross.len(), n);
        assert!(fwd_cross.iter().all(|x| *x == 0.0));

        let trn_cross = op
            .transpose_mul_second_cross(0, 1, &v.view())
            .expect("transpose_mul_second_cross");
        assert_eq!(trn_cross.len(), p);
        assert!(trn_cross.iter().all(|x| *x == 0.0));

        let chunk = op.row_chunk_first(0, 3..7).expect("row_chunk_first");
        assert_eq!(chunk.dim(), (4, p));
        assert!(chunk.iter().all(|x| *x == 0.0));

        let chunk_diag = op
            .row_chunk_second_diag(0, 0..n)
            .expect("row_chunk_second_diag");
        assert_eq!(chunk_diag.dim(), (n, p));
        assert!(chunk_diag.iter().all(|x| *x == 0.0));

        let chunk_cross = op
            .row_chunk_second_cross(0, 1, 1..3)
            .expect("row_chunk_second_cross");
        assert_eq!(chunk_cross.dim(), (2, p));
        assert!(chunk_cross.iter().all(|x| *x == 0.0));

        let mut row = Array1::from_elem(p, 9.5);
        op.row_vector_first_into(0, 4, row.view_mut())
            .expect("row_vector_first_into");
        assert!(row.iter().all(|x| *x == 0.0));

        // The operator must not advertise dense materialization — production
        // hot paths rely on this to avoid forming an (n, p) buffer.
        assert!(op.as_materializable().is_none());
    }

    /// At biobank scale (n=320 000, p=101) a dense `Array2::zeros((n, p))`
    /// for an unused ψ-derivative slot consumes ≈ 0.24 GiB; the spatial-
    /// adaptive baseline used to allocate one per ψ coordinate (≈ 1.4 GiB
    /// of guaranteed-zero memory at six coords). Replacing the dense zero
    /// matrix with a `(0, 0)` shape sentinel — without an implicit
    /// operator — must still resolve to `PsiDesignMap::Zero` so callers
    /// see exact-zero semantics with O(1) memory.
    #[test]
    fn spatial_adaptive_zero_xpsi_uses_zero_map_without_dense_allocation() {
        let n = 320_000usize;
        let p = 101usize;
        let deriv = CustomFamilyBlockPsiDerivative {
            penalty_index: None,
            x_psi: Array2::<f64>::zeros((0, 0)),
            s_psi: Array2::<f64>::zeros((0, 0)),
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
        let policy = ResourcePolicy::default_library();
        let map = resolve_custom_family_x_psi_map(
            &deriv,
            n,
            p,
            0..n,
            "spatial-adaptive zero sentinel",
            &policy,
        )
        .expect("resolve x_psi map for (0, 0)-sentinel deriv");
        match map {
            PsiDesignMap::Zero { nrows, ncols } => {
                assert_eq!(nrows, n);
                assert_eq!(ncols, p);
            }
            other => panic!(
                "(0, 0) x_psi sentinel must resolve to PsiDesignMap::Zero, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn zero_psi_derivative_operator_resolves_to_zero_design_map() {
        let n = 12usize;
        let p = 4usize;
        let zero_op: Arc<dyn CustomFamilyPsiDerivativeOperator> =
            Arc::new(ZeroPsiDerivativeOperator::new(n, p));
        let deriv = CustomFamilyBlockPsiDerivative {
            penalty_index: None,
            x_psi: Array2::<f64>::zeros((0, 0)),
            s_psi: Array2::<f64>::zeros((0, 0)),
            s_psi_components: None,
            s_psi_penalty_components: None,
            x_psi_psi: None,
            s_psi_psi: None,
            s_psi_psi_components: None,
            s_psi_psi_penalty_components: None,
            implicit_operator: Some(Arc::clone(&zero_op)),
            implicit_axis: 0,
            implicit_group_id: None,
        };
        let policy = ResourcePolicy::default_library();
        let map = resolve_custom_family_x_psi_map(&deriv, n, p, 0..n, "zero", &policy)
            .expect("resolve x_psi map");
        let u = Array1::from_iter((0..p).map(|k| 1.0 + k as f64));
        let fwd = map.forward_mul(u.view()).expect("forward_mul map");
        assert_eq!(fwd.len(), n);
        assert!(fwd.iter().all(|x| *x == 0.0));

        let chunk = map.row_chunk(2..5).expect("row_chunk map");
        assert_eq!(chunk.dim(), (3, p));
        assert!(chunk.iter().all(|x| *x == 0.0));

        let map_second =
            resolve_custom_family_x_psi_psi_map(&deriv, &deriv, 0, n, p, 0..n, "zero", &policy)
                .expect("resolve x_psi_psi map");
        let fwd_second = map_second
            .forward_mul(u.view())
            .expect("forward_mul second");
        assert_eq!(fwd_second.len(), n);
        assert!(fwd_second.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn rowwise_kronecker_psi_row_chunks_are_window_consistent() {
        let first = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let second_diag = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]];
        let second_cross = array![[-1.0, 0.25], [-1.5, 0.5], [-2.0, 0.75]];
        let base = build_embedded_dense_psi_operator(
            &first,
            &second_diag,
            Some(&vec![(1, second_cross.clone())]),
            0..2,
            2,
            0,
        )
        .expect("embedded dense base");
        let time_a = Arc::new(array![[1.0, 0.0], [0.5, 1.0], [1.5, -0.5]]);
        let time_b = Arc::new(array![[0.25, 2.0], [-1.0, 0.75], [0.0, 1.25]]);
        let op = build_rowwise_kronecker_psi_operator(base, vec![time_a, time_b])
            .expect("rowwise kronecker psi operator");
        let mat = op
            .as_materializable()
            .expect("rowwise operator dense reference");
        let rows = 1..5;

        let first_dense = mat.materialize_first(0).expect("dense first");
        let first_chunk = op.row_chunk_first(0, rows.clone()).expect("chunk first");
        assert_eq!(
            first_chunk,
            first_dense.slice(ndarray::s![rows.clone(), ..]).to_owned()
        );

        let diag_full = op
            .row_chunk_second_diag(0, 0..op.n_data())
            .expect("full row-chunk diag");
        let diag_chunk = op
            .row_chunk_second_diag(0, rows.clone())
            .expect("chunk diag");
        assert_eq!(
            diag_chunk,
            diag_full.slice(ndarray::s![rows.clone(), ..]).to_owned()
        );

        let cross_full = op
            .row_chunk_second_cross(0, 1, 0..op.n_data())
            .expect("full row-chunk cross");
        let cross_chunk = op
            .row_chunk_second_cross(0, 1, rows.clone())
            .expect("chunk cross");
        assert_eq!(
            cross_chunk,
            cross_full.slice(ndarray::s![rows, ..]).to_owned()
        );
    }

    #[test]
    fn joint_trust_region_radius_update_accept_reject_logic() {
        let accepted = update_joint_trust_region_radius(1.0, 1.0, 2.0, 2.0);
        assert!(accepted.accepted);
        assert!((accepted.rho - 1.0).abs() < 1.0e-12);
        assert!((accepted.radius - 2.0).abs() < 1.0e-12);

        let rejected = update_joint_trust_region_radius(1.0, 0.5, -0.1, 2.0);
        assert!(!rejected.accepted);
        assert!(rejected.rho < 0.0);
        assert!((rejected.radius - 0.25).abs() < 1.0e-12);

        let poor = update_joint_trust_region_radius(1.0, 0.5, 0.1, 1.0);
        assert!(poor.accepted);
        assert!((poor.rho - 0.1).abs() < 1.0e-12);
        assert!((poor.radius - 0.25).abs() < 1.0e-12);
    }

    #[test]
    fn joint_trust_region_rosenbrock_like_quadratic_is_armijo_safe() {
        // Local Rosenbrock-at-the-valley quadratic in variables (x, y):
        // f ≈ 0.5 * [dx, dy]' H [dx, dy], H = [[802, -400], [-400, 200]].
        // Add a tiny ridge to make the test SPD and use a gradient whose full
        // Newton step crosses the radius, exercising truncation before the
        // objective is evaluated.
        let h = array![[802.0, -400.0], [-400.0, 200.1]];
        let unconstrained = array![1.0, 1.0];
        let gradient = -h.dot(&unconstrained);
        let rhs = -&gradient;
        let mut step = unconstrained.clone();
        let step_norm = truncate_joint_step_to_radius(&mut step, 0.25);
        assert!(step_norm <= 0.25 + 1.0e-12);
        assert!(joint_trust_region_step_norm(&unconstrained) > 0.25);

        let h_step = h.dot(&step);
        let predicted = joint_quadratic_predicted_reduction(&rhs, &h_step, &step);
        let old_objective = 0.0;
        let trial_objective = gradient.dot(&step) + 0.5 * step.dot(&h_step);
        let actual = old_objective - trial_objective;
        assert!(predicted > 0.0);
        assert!((predicted - actual).abs() < 1.0e-10);

        let update = update_joint_trust_region_radius(0.25, step_norm, actual, predicted);
        assert!(update.accepted);
        assert!(trial_objective < old_objective);
    }
}
