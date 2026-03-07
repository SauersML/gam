use crate::faer_ndarray::FaerCholesky;
use crate::faer_ndarray::{FaerArrayView, FaerEigh};
use crate::linalg::utils::StableSolver;
use crate::matrix::DesignMatrix;
use crate::pirls::LinearInequalityConstraints;
use crate::solver::opt_objective::CachedSecondOrderObjective;
use crate::types::{LinkFunction, RidgeDeterminantMode, RidgePolicy};
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{Lblt as FaerLblt, Solve as FaerSolve};
use ndarray::{Array1, Array2};
use thiserror::Error;
use opt::{
    Bounds, MaxIterations, NewtonTrustRegion, NewtonTrustRegionError, ObjectiveEvalError,
    Tolerance,
};

/// Optional known link metadata when a family uses a learnable wiggle correction.
#[derive(Debug, Clone, Copy)]
pub struct KnownLinkWiggle {
    pub base_link: LinkFunction,
    pub wiggle_block: Option<usize>,
}

/// Static specification for one parameter block in a custom family.
#[derive(Clone)]
pub struct ParameterBlockSpec {
    pub name: String,
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    /// Block-local penalty matrices (all p_block x p_block).
    pub penalties: Vec<Array2<f64>>,
    /// Initial log-smoothing parameters for this block (same length as `penalties`).
    pub initial_log_lambdas: Array1<f64>,
    /// Optional initial coefficients (defaults to zeros if omitted).
    pub initial_beta: Option<Array1<f64>>,
}

/// Current state for a parameter block.
#[derive(Clone)]
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
#[derive(Clone)]
pub enum BlockWorkingSet {
    /// Standard IRLS/GLM-style diagonal working set for eta-space updates.
    Diagonal {
        /// IRLS pseudo-response for this block's linear predictor.
        working_response: Array1<f64>,
        /// IRLS working weights for this block (non-negative, length n).
        working_weights: Array1<f64>,
    },
    /// Exact Newton block update in coefficient space.
    ///
    /// `gradient` is ∇ log L wrt block coefficients.
    /// `hessian` is -∇² log L wrt block coefficients (positive semidefinite near optimum).
    ExactNewton {
        gradient: Array1<f64>,
        hessian: Array2<f64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactNewtonOuterObjective {
    QuadraticReml,
    PseudoLaplace,
}

/// Family evaluation over all parameter blocks.
#[derive(Clone)]
pub struct FamilyEvaluation {
    pub log_likelihood: f64,
    pub block_working_sets: Vec<BlockWorkingSet>,
}

/// User-defined family contract for multi-block generalized models.
pub trait CustomFamily {
    /// Evaluate log-likelihood and per-block working quantities at current block predictors.
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String>;

    /// Selects the outer objective semantics for exact-Newton families.
    ///
    /// `QuadraticReml` is the classical blockwise REML/LAML surface:
    ///
    ///   -loglik + penalty + 0.5 (log|H| - log|S|_+)
    ///
    /// `PseudoLaplace` is the exact-mode pseudo-Laplace surface used by the
    /// Charbonnier spatial family:
    ///
    ///   -loglik + penalty + 0.5 log|H|
    ///
    /// The latter deliberately omits the quadratic-only `-0.5 log|S|_+`
    /// normalization term because there is no tractable exact analogue for the
    /// nonquadratic prior without introducing the intractable prior normalizer.
    fn exact_newton_outer_objective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::QuadraticReml
    }

    /// Optional metadata describing a known link with learnable wiggle.
    fn known_link_wiggle(&self) -> Option<KnownLinkWiggle> {
        None
    }

    /// Optional dynamic geometry hook for blocks whose design/offset depend on
    /// current values of other blocks.
    fn block_geometry(
        &self,
        _block_states: &[ParameterBlockState],
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
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _spec: &ParameterBlockSpec,
        _d_beta: &Array1<f64>,
    ) -> Result<Option<BlockGeometryDirectionalDerivative>, String> {
        Ok(None)
    }

    /// Optional per-block coefficient projection applied after each block update.
    fn post_update_beta(&self, beta: Array1<f64>) -> Result<Array1<f64>, String> {
        Ok(beta)
    }

    /// Optional linear inequality constraints for a block update:
    /// `A * beta_block >= b`.
    fn block_linear_constraints(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _spec: &ParameterBlockSpec,
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
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Optional exact joint coefficient-space Hessian across all blocks.
    ///
    /// Returns the unpenalized matrix `H = -∇² log L` in the flattened block order.
    fn exact_newton_joint_hessian(
        &self,
        _block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Optional exact directional derivative of the joint coefficient-space Hessian.
    ///
    /// Returns `Some(dH)` where `dH` is the directional derivative of the
    /// unpenalized joint Hessian `H = -∇² log L` along the flattened
    /// coefficient-space direction `d_beta_flat`.
    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Optional exact second directional derivative of the joint Hessian.
    ///
    /// Returns `Some(d2H)` where `d2H` is:
    ///   D²H[u, v] = d/dε d/dδ H(beta + εu + δv) |_{ε=δ=0}
    /// for flattened coefficient-space directions `u = d_beta_u_flat`,
    /// `v = d_beta_v_flat`.
    fn exact_newton_joint_hessian_second_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _d_beta_u_flat: &Array1<f64>,
        _d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
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
    fn diagonal_working_weights_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    /// Optional exact psi-gradient for blocks using `BlockWorkingSet::ExactNewton`.
    ///
    /// Families that expose an exact coefficient-space Hessian rather than a
    /// diagonal IRLS working set cannot reuse the diagonal psi-gradient algebra
    /// below. They must provide the derivative of the active outer objective
    /// directly for the externally realized psi direction encoded in `ctx.deriv`.
    ///
    /// Returning `None` means the family does not provide an analytic exact-Newton
    /// psi gradient for this block.
    fn exact_newton_block_psi_gradient(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _ctx: ExactNewtonPsiGradientContext<'_>,
    ) -> Result<Option<f64>, String> {
        Ok(None)
    }
}

#[derive(Clone)]
pub struct BlockwiseFitOptions {
    pub inner_max_cycles: usize,
    pub inner_tol: f64,
    pub outer_max_iter: usize,
    pub outer_tol: f64,
    pub min_weight: f64,
    pub ridge_floor: f64,
    /// Shared ridge semantics used by solve/quadratic/logdet terms.
    pub ridge_policy: RidgePolicy,
    /// If true, outer smoothing optimization uses a Laplace/REML-style objective:
    ///   -loglik + penalty + 0.5(log|H| - log|S|_+)
    /// where H is blockwise working curvature and S is blockwise penalty.
    pub use_reml_objective: bool,
    /// If false, skip post-fit joint covariance assembly.
    pub compute_covariance: bool,
}

impl Default for BlockwiseFitOptions {
    fn default() -> Self {
        Self {
            inner_max_cycles: 100,
            inner_tol: 1e-6,
            outer_max_iter: 60,
            outer_tol: 1e-5,
            min_weight: 1e-12,
            ridge_floor: 1e-12,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_reml_objective: true,
            compute_covariance: true,
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
}

#[derive(Clone)]
struct ConstrainedWarmStart {
    rho: Array1<f64>,
    block_beta: Vec<Array1<f64>>,
    active_sets: Vec<Option<Vec<usize>>>,
}

#[derive(Clone)]
pub struct BlockwiseFitResult {
    pub block_states: Vec<ParameterBlockState>,
    pub log_likelihood: f64,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub covariance_conditional: Option<Array2<f64>>,
    pub penalized_objective: f64,
    pub outer_iterations: usize,
    pub outer_final_gradient_norm: f64,
    pub inner_cycles: usize,
    pub converged: bool,
}

#[derive(Clone)]
pub struct CustomFamilyBlockPsiDerivative {
    pub penalty_index: usize,
    pub x_psi: Array2<f64>,
    pub s_psi: Array2<f64>,
    pub s_psi_components: Option<Vec<(usize, Array2<f64>)>>,
}

#[derive(Clone, Copy)]
pub struct ExactNewtonPsiGradientContext<'a> {
    pub spec: &'a ParameterBlockSpec,
    pub deriv: &'a CustomFamilyBlockPsiDerivative,
    pub lambdas: &'a Array1<f64>,
    pub options: &'a BlockwiseFitOptions,
}

#[derive(Clone)]
pub struct CustomFamilyWarmStart {
    inner: ConstrainedWarmStart,
}

pub struct CustomFamilyJointHyperResult {
    pub objective: f64,
    pub gradient: Array1<f64>,
    pub outer_hessian: Option<Array2<f64>>,
    pub warm_start: CustomFamilyWarmStart,
}

struct OuterObjectiveEvalResult {
    objective: f64,
    gradient: Array1<f64>,
    outer_hessian: Option<Array2<f64>>,
    warm_start: ConstrainedWarmStart,
    inner: BlockwiseInnerResult,
}

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

fn validate_block_specs(specs: &[ParameterBlockSpec]) -> Result<Vec<usize>, String> {
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
            let (r, c) = s.dim();
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

fn build_block_states<F: CustomFamily>(
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
            let mut eta = x.matrix_vector_multiply(&beta);
            eta += off;
            Ok(eta)
        })?;
        states.push(ParameterBlockState { beta, eta });
    }
    Ok(states)
}

fn refresh_all_block_etas<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
) -> Result<(), String> {
    for b in 0..specs.len() {
        let spec = &specs[b];
        let beta = states[b].beta.clone();
        states[b].eta = with_block_geometry(family, states, spec, b, |x, off| {
            Ok(x.matrix_vector_multiply(&beta) + off)
        })?;
    }
    Ok(())
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

fn solve_block_weighted_system(
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

    let (xtwx_base, xtwy_opt) = weighted_normal_equations(x, w, Some(y_star))?;
    let xtwy = xtwy_opt.ok_or_else(|| "missing weighted RHS in block solve".to_string())?;
    let base_ridge = if ridge_policy.include_laplace_hessian {
        effective_solver_ridge(ridge_floor)
    } else {
        0.0
    };

    let mut xtwx = xtwx_base.clone();
    xtwx += s_lambda;
    let solver = StableSolver::new("custom-family weighted block solve");
    solver
        .solve_vector_with_ridge_retries(&xtwx, &xtwy, base_ridge)
        .ok_or_else(|| "block solve failed after ridge retries".to_string())
}

fn solve_spd_system_with_policy(
    lhs: &Array2<f64>,
    rhs: &Array1<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    let p = lhs.nrows();
    if lhs.ncols() != p || rhs.len() != p {
        return Err("exact-newton system dimension mismatch".to_string());
    }
    let base_ridge = if ridge_policy.include_laplace_hessian {
        effective_solver_ridge(ridge_floor)
    } else {
        0.0
    };
    let solver = StableSolver::new("custom-family SPD block solve");
    solver
        .solve_vector_with_ridge_retries(lhs, rhs, base_ridge)
        .ok_or_else(|| "exact-newton block solve failed after ridge retries".to_string())
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

        let w_clamped = self
            .working_weights
            .mapv(|wi| wi.max(ctx.options.min_weight));

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
                let (beta_constrained, active_set) = solve_quadratic_with_linear_constraints(
                    &lhs,
                    &rhs,
                    &ctx.states[ctx.block_idx].beta,
                    constraints,
                    ctx.cached_active_set,
                )
                .map_err(|e| {
                    format!(
                        "block {} ({}) constrained diagonal solve failed: {e}",
                        ctx.block_idx, ctx.spec.name
                    )
                })?;
                Ok(BlockUpdateResult {
                    beta_new_raw: beta_constrained,
                    active_set: Some(active_set),
                })
            })
        } else {
            with_block_geometry(ctx.family, ctx.states, ctx.spec, ctx.block_idx, |x, off| {
                let mut y_star = self.working_response.clone();
                y_star -= off;
                let beta = solve_block_weighted_system(
                    x,
                    &y_star,
                    &w_clamped,
                    ctx.s_lambda,
                    ctx.options.ridge_floor,
                    ctx.options.ridge_policy,
                )?;
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
    hessian: &'a Array2<f64>,
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

        let mut lhs = self.hessian.clone();
        lhs += ctx.s_lambda;
        let mut rhs = self.hessian.dot(&ctx.states[ctx.block_idx].beta);
        rhs += self.gradient;

        if let Some(constraints) = ctx.linear_constraints {
            check_linear_feasibility(&ctx.states[ctx.block_idx].beta, constraints, 1e-8).map_err(
                |e| {
                    format!(
                        "block {} ({}) constrained exact-newton solve: {e}",
                        ctx.block_idx, ctx.spec.name
                    )
                },
            )?;
            let (beta_constrained, active_set) = solve_quadratic_with_linear_constraints(
                &lhs,
                &rhs,
                &ctx.states[ctx.block_idx].beta,
                constraints,
                ctx.cached_active_set,
            )
            .map_err(|e| {
                format!(
                    "block {} ({}) constrained exact-newton solve failed: {e}",
                    ctx.block_idx, ctx.spec.name
                )
            })?;
            Ok(BlockUpdateResult {
                beta_new_raw: beta_constrained,
                active_set: Some(active_set),
            })
        } else {
            let beta = if use_exact_newton_strict_spd(ctx.family) {
                strict_solve_spd(&lhs, &rhs)?
            } else {
                solve_spd_system_with_policy(
                    &lhs,
                    &rhs,
                    ctx.options.ridge_floor,
                    ctx.options.ridge_policy,
                )?
            };
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

fn solve_kkt_step(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    active_a: &Array2<f64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let p = hessian.nrows();
    if hessian.ncols() != p || gradient.len() != p || active_a.ncols() != p {
        return Err("constrained KKT step: dimension mismatch".to_string());
    }
    let m = active_a.nrows();
    if m == 0 {
        let rhs = gradient.mapv(|v| -v);
        let solver = StableSolver::new("custom-family unconstrained kkt step");
        let direction = solver
            .solve_vector_with_ridge_retries(hessian, &rhs, 0.0)
            .ok_or_else(|| {
                "constrained unconstrained-step solve produced non-finite values".to_string()
            })?;
        if !direction.iter().all(|v| v.is_finite()) {
            return Err(
                "constrained unconstrained-step solve produced non-finite values".to_string(),
            );
        }
        return Ok((direction, Array1::zeros(0)));
    }

    let mut kkt = Array2::<f64>::zeros((p + m, p + m));
    kkt.slice_mut(ndarray::s![0..p, 0..p]).assign(hessian);
    kkt.slice_mut(ndarray::s![0..p, p..(p + m)])
        .assign(&active_a.t());
    kkt.slice_mut(ndarray::s![p..(p + m), 0..p])
        .assign(active_a);

    let mut rhs = Array1::<f64>::zeros(p + m);
    for i in 0..p {
        rhs[i] = -gradient[i];
    }

    let kkt_view = FaerArrayView::new(&kkt);
    let lb = FaerLblt::new(kkt_view.as_ref(), Side::Lower);
    let mut rhs_mat = FaerMat::zeros(p + m, 1);
    for i in 0..(p + m) {
        rhs_mat[(i, 0)] = rhs[i];
    }
    lb.solve_in_place(rhs_mat.as_mut());

    let mut direction = Array1::<f64>::zeros(p);
    let mut lambda = Array1::<f64>::zeros(m);
    for i in 0..p {
        direction[i] = rhs_mat[(i, 0)];
    }
    for i in 0..m {
        lambda[i] = rhs_mat[(p + i, 0)];
    }
    if !direction.iter().all(|v| v.is_finite()) || !lambda.iter().all(|v| v.is_finite()) {
        return Err("constrained KKT step produced non-finite values".to_string());
    }
    Ok((direction, lambda))
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

fn solve_quadratic_with_linear_constraints(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    beta_start: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    warm_active_set: Option<&[usize]>,
) -> Result<(Array1<f64>, Vec<usize>), String> {
    let p = hessian.nrows();
    let m = constraints.a.nrows();
    if hessian.ncols() != p || rhs.len() != p || beta_start.len() != p {
        return Err("constrained quadratic solve: system dimension mismatch".to_string());
    }
    if constraints.a.ncols() != p || constraints.b.len() != m {
        return Err("constrained quadratic solve: constraint dimension mismatch".to_string());
    }
    let tol_active = 1e-10;
    let tol_step = 1e-12;
    let tol_dual = 1e-10;
    let feas_tol = 1e-8;

    check_linear_feasibility(beta_start, constraints, feas_tol)?;

    // Fast path: unconstrained optimum is feasible.
    if let Ok(beta_unc) = solve_spd_system_with_policy(
        hessian,
        rhs,
        0.0,
        RidgePolicy::explicit_stabilization_pospart(),
    ) {
        let slack = constraints.a.dot(&beta_unc) - &constraints.b;
        if slack.iter().all(|&s| s >= -feas_tol) {
            return Ok((beta_unc, Vec::new()));
        }
    }

    let mut x = beta_start.to_owned();
    let mut slack = constraints.a.dot(&x) - &constraints.b;
    let mut active: Vec<usize> = Vec::new();
    let mut is_active = vec![false; m];
    if let Some(seed) = warm_active_set {
        for &idx in seed {
            if idx < m && !is_active[idx] {
                active.push(idx);
                is_active[idx] = true;
            }
        }
    }
    for i in 0..m {
        if slack[i] <= tol_active && !is_active[i] {
            active.push(i);
            is_active[i] = true;
        }
    }
    for &idx in &active {
        is_active[idx] = true;
    }

    for _ in 0..((p + m + 8) * 4) {
        let gradient = hessian.dot(&x) - rhs;
        let mut a_w = Array2::<f64>::zeros((active.len(), p));
        for (r, &idx) in active.iter().enumerate() {
            a_w.row_mut(r).assign(&constraints.a.row(idx));
        }
        let (direction, lambda_sys) = solve_kkt_step(hessian, &gradient, &a_w)?;
        let step_norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        if step_norm <= tol_step {
            if active.is_empty() {
                return Ok((x, active));
            }
            // solve_kkt_step returns multipliers from:
            //   H d + A_w^T lambda_sys = -gradient.
            // For constraints A*x >= b, true multipliers satisfy
            //   gradient + H d = A_w^T lambda_true, so lambda_true = -lambda_sys.
            // Release active rows with lambda_true < 0.
            let mut most_negative_true = -tol_dual;
            let mut remove_pos: Option<usize> = None;
            for (pos, &lam_sys) in lambda_sys.iter().enumerate() {
                let lam_true = -lam_sys;
                if lam_true < most_negative_true {
                    most_negative_true = lam_true;
                    remove_pos = Some(pos);
                }
            }
            if let Some(pos) = remove_pos {
                let idx = active.remove(pos);
                is_active[idx] = false;
                continue;
            }
            return Ok((x, active));
        }

        let mut alpha = 1.0_f64;
        let mut entering: Option<usize> = None;
        for i in 0..m {
            if is_active[i] {
                continue;
            }
            let ai = constraints.a.row(i);
            let ai_d = ai.dot(&direction);
            if ai_d < -1e-14 {
                let cand = (slack[i] / (-ai_d)).max(0.0);
                if cand < alpha {
                    alpha = cand;
                    entering = Some(i);
                }
            }
        }

        x += &direction.mapv(|v| alpha * v);
        slack = constraints.a.dot(&x) - &constraints.b;

        if let Some(idx) = entering
            && !is_active[idx]
        {
            active.push(idx);
            is_active[idx] = true;
        }
    }

    Err("constrained quadratic active-set solver failed to converge".to_string())
}

#[inline]
fn effective_solver_ridge(ridge_floor: f64) -> f64 {
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
    let p = a.nrows();
    for i in 0..p {
        for j in 0..i {
            let v = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
    }
    let ridge = if ridge_policy.include_penalty_logdet {
        effective_solver_ridge(ridge_floor)
    } else {
        0.0
    };
    for i in 0..p {
        a[[i, i]] += ridge;
    }

    match ridge_policy.determinant_mode {
        RidgeDeterminantMode::Full => {
            let chol = a.clone().cholesky(Side::Lower).map_err(|_| {
                "cholesky failed while computing full ridge-aware logdet".to_string()
            })?;
            Ok(2.0 * chol.diag().mapv(f64::ln).sum())
        }
        RidgeDeterminantMode::PositivePart => {
            let (evals, _) = crate::faer_ndarray::FaerEigh::eigh(&a, Side::Lower)
                .map_err(|e| format!("eigh failed while computing logdet: {e}"))?;
            let floor = ridge.max(1e-14);
            let mut logdet = 0.0;
            for &ev in &evals {
                if ev > floor {
                    logdet += ev.ln();
                }
            }
            Ok(logdet)
        }
    }
}

fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let (r, c) = a.dim();
    let mut t = 0.0;
    for i in 0..r {
        for j in 0..c {
            t += a[[i, j]] * b[[j, i]];
        }
    }
    t
}

fn trace_jinv_a_jinv_b(j_inv: &Array2<f64>, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    // tr(J^{-1} A J^{-1} B) computed without forming explicit Kronecker products.
    let tmp = j_inv.dot(a);
    let tmp = tmp.dot(j_inv);
    trace_product(&tmp, b)
}

fn leverage_quadratic_forms(x: &DesignMatrix, h_inv: &Array2<f64>) -> Array1<f64> {
    let x_dense_arc = x.to_dense_arc();
    let x_dense = x_dense_arc.as_ref();
    let x_hinv = x_dense.dot(h_inv);
    let mut out = Array1::<f64>::zeros(x_dense.nrows());
    for i in 0..x_dense.nrows() {
        out[i] = x_hinv.row(i).dot(&x_dense.row(i));
    }
    out
}

fn inverse_spd_with_retry(
    matrix: &Array2<f64>,
    base_ridge: f64,
    max_retry: usize,
) -> Result<Array2<f64>, String> {
    let solver = StableSolver::new("custom-family inverse spd");
    solver
        .inverse_with_ridge_retries(matrix, base_ridge, max_retry)
        .ok_or_else(|| "failed to invert SPD system after ridge retries".to_string())
}

fn strict_solve_spd(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
    let chol = matrix
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD solve failed".to_string())?;
    Ok(chol.solve_vec(rhs))
}

fn strict_inverse_spd(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let chol = matrix
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD inverse failed".to_string())?;
    let ident = Array2::<f64>::eye(matrix.nrows());
    Ok(chol.solve_mat(&ident))
}

fn strict_logdet_spd(matrix: &Array2<f64>) -> Result<f64, String> {
    let chol = matrix
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD logdet failed".to_string())?;
    Ok(2.0 * chol.diag().mapv(f64::ln).sum())
}

fn pinv_positive_part(matrix: &Array2<f64>, ridge_floor: f64) -> Result<Array2<f64>, String> {
    let (evals, evecs) = FaerEigh::eigh(matrix, Side::Lower)
        .map_err(|e| format!("eigh failed in positive-part pseudoinverse: {e}"))?;
    let p = matrix.nrows();
    let max_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let tol = (max_eval * 1e-12).max(ridge_floor.max(1e-14));
    let mut pinv = Array2::<f64>::zeros((p, p));
    for k in 0..p {
        let ev = evals[k];
        if ev > tol {
            let inv_ev = 1.0 / ev;
            for i in 0..p {
                let uik = evecs[(i, k)];
                for j in 0..p {
                    pinv[[i, j]] += inv_ev * uik * evecs[(j, k)];
                }
            }
        }
    }
    Ok(pinv)
}

fn include_exact_newton_logdet_h<F: CustomFamily + ?Sized>(family: &F) -> bool {
    matches!(
        family.exact_newton_outer_objective(),
        ExactNewtonOuterObjective::QuadraticReml | ExactNewtonOuterObjective::PseudoLaplace
    )
}

fn include_exact_newton_logdet_s<F: CustomFamily + ?Sized>(family: &F, options: &BlockwiseFitOptions) -> bool {
    family.exact_newton_outer_objective() == ExactNewtonOuterObjective::QuadraticReml
        && options.use_reml_objective
}

fn use_exact_newton_strict_spd<F: CustomFamily + ?Sized>(family: &F) -> bool {
    family.exact_newton_outer_objective() == ExactNewtonOuterObjective::PseudoLaplace
}

fn blockwise_logdet_terms<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<(f64, f64), String> {
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    refresh_all_block_etas(family, specs, states)?;
    if let Some(h_joint) = family.exact_newton_joint_hessian(states)? {
        let ranges = block_param_ranges(specs);
        let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
        if h_joint.nrows() != total || h_joint.ncols() != total {
            return Err(format!(
                "joint exact-newton Hessian shape mismatch in logdet terms: got {}x{}, expected {}x{}",
                h_joint.nrows(),
                h_joint.ncols(),
                total,
                total
            ));
        }
        let mut s_joint = Array2::<f64>::zeros((total, total));
        let mut logdet_s_total = 0.0;
        for (b, spec) in specs.iter().enumerate() {
            let (start, end) = ranges[b];
            let p = end - start;
            let lambdas = block_log_lambdas[b].mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((p, p));
            for (k, s) in spec.penalties.iter().enumerate() {
                s_lambda.scaled_add(lambdas[k], s);
            }
            s_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .assign(&s_lambda);
            if include_logdet_s {
                logdet_s_total += stable_logdet_with_ridge_policy(
                    &s_lambda,
                    options.ridge_floor,
                    options.ridge_policy,
                )?;
            }
        }
        let mut h = h_joint;
        h += &s_joint;
        let logdet_h_total = if strict_spd {
            strict_logdet_spd(&h)?
        } else {
            stable_logdet_with_ridge_policy(&h, options.ridge_floor, options.ridge_policy)?
        };
        return Ok((logdet_h_total, logdet_s_total));
    }

    let eval = family.evaluate(states)?;
    if eval.block_working_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.block_working_sets.len(),
            specs.len()
        ));
    }

    let mut logdet_h_total = 0.0;
    let mut logdet_s_total = 0.0;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let work = &eval.block_working_sets[b];
        let p = spec.design.ncols();
        let xtwx = match work {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => with_block_geometry(family, states, spec, b, |x_dyn, _| {
                let w = working_weights.mapv(|wi| wi.max(options.min_weight));
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
                hessian.clone()
            }
        };

        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s_lambda.scaled_add(lambdas[k], s);
        }

        let mut h = xtwx;
        h += &s_lambda;
        logdet_h_total += if strict_spd {
            strict_logdet_spd(&h)?
        } else {
            stable_logdet_with_ridge_policy(&h, options.ridge_floor, options.ridge_policy)?
        };
        if include_logdet_s {
            logdet_s_total += stable_logdet_with_ridge_policy(
                &s_lambda,
                options.ridge_floor,
                options.ridge_policy,
            )?;
        }
    }
    Ok((logdet_h_total, logdet_s_total))
}

fn inner_blockwise_fit<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<BlockwiseInnerResult, String> {
    let mut states = build_block_states(family, specs)?;
    refresh_all_block_etas(family, specs, &mut states)?;
    let mut s_lambdas = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let p = spec.design.ncols();
        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s_lambda.scaled_add(lambdas[k], s);
        }
        s_lambdas.push(s_lambda);
    }
    let ridge = effective_solver_ridge(options.ridge_floor);
    let mut cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; specs.len()];
    if let Some(seed) = warm_start
        && seed.block_beta.len() == states.len()
        && seed.active_sets.len() == states.len()
    {
        for (b, beta_seed) in seed.block_beta.iter().enumerate() {
            if beta_seed.len() == states[b].beta.len() {
                states[b].beta.assign(beta_seed);
            }
        }
        cached_active_sets = seed.active_sets.clone();
        refresh_all_block_etas(family, specs, &mut states)?;
    }
    let initial_eval = family.evaluate(&states)?;
    let mut current_penalty =
        total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
    let mut last_objective = -initial_eval.log_likelihood + current_penalty;
    let mut converged = false;
    let mut cycles_done = 0usize;

    for cycle in 0..options.inner_max_cycles {
        let mut max_beta_step = 0.0_f64;

        let mut objective_cycle_prev = last_objective;
        for b in 0..specs.len() {
            // Keep all blocks synchronized with any dynamic geometry.
            refresh_all_block_etas(family, specs, &mut states)?;
            let eval = family.evaluate(&states)?;
            if eval.block_working_sets.len() != specs.len() {
                return Err(format!(
                    "family returned {} block working sets, expected {}",
                    eval.block_working_sets.len(),
                    specs.len()
                ));
            }

            let spec = &specs[b];
            let work = &eval.block_working_sets[b];
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
            let beta_new = family.post_update_beta(beta_new_raw)?;
            let beta_old = states[b].beta.clone();
            let delta = &beta_new - &beta_old;
            let old_block_penalty =
                block_quadratic_penalty(&beta_old, s_lambda, ridge, options.ridge_policy);
            let step = delta.iter().copied().map(f64::abs).fold(0.0, f64::max);
            max_beta_step = max_beta_step.max(step);
            if step <= options.inner_tol {
                continue;
            }

            // Damped update: require non-increasing penalized objective under dynamic geometry.
            let mut accepted = false;
            for bt in 0..8 {
                let alpha = 0.5f64.powi(bt);
                let trial_beta_raw = &beta_old + &delta.mapv(|v| alpha * v);
                let trial_beta = family.post_update_beta(trial_beta_raw)?;
                states[b].beta = trial_beta;
                refresh_all_block_etas(family, specs, &mut states)?;
                let trial_eval = family.evaluate(&states)?;
                let trial_block_penalty =
                    block_quadratic_penalty(&states[b].beta, s_lambda, ridge, options.ridge_policy);
                let trial_penalty = current_penalty - old_block_penalty + trial_block_penalty;
                let trial_objective = -trial_eval.log_likelihood + trial_penalty;
                if trial_objective.is_finite() && trial_objective <= objective_cycle_prev + 1e-10 {
                    objective_cycle_prev = trial_objective;
                    current_penalty = trial_penalty;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                states[b].beta = beta_old;
                refresh_all_block_etas(family, specs, &mut states)?;
            }
        }

        refresh_all_block_etas(family, specs, &mut states)?;
        let eval = family.evaluate(&states)?;
        current_penalty = total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
        let objective = -eval.log_likelihood + current_penalty;
        let objective_change = (objective - last_objective).abs();
        last_objective = objective;
        cycles_done = cycle + 1;

        let objective_tol = options.inner_tol * (1.0 + objective.abs());
        if max_beta_step <= options.inner_tol && objective_change <= objective_tol {
            converged = true;
            break;
        }
    }

    let final_eval = family.evaluate(&states)?;
    let penalty_value = total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);

    let (block_logdet_h, block_logdet_s) =
        blockwise_logdet_terms(family, specs, &mut states, block_log_lambdas, options)?;

    Ok(BlockwiseInnerResult {
        block_states: states,
        active_sets: cached_active_sets,
        log_likelihood: final_eval.log_likelihood,
        penalty_value,
        cycles: cycles_done,
        converged,
        block_logdet_h,
        block_logdet_s,
    })
}

/// Fit a custom multi-block family.
///
/// Inner loop: cyclic blockwise penalized weighted regressions.
/// Outer loop: trust-region optimization of all log-smoothing parameters using
/// exact cost/gradient samples.

fn outer_objective_gradient_hessian_internal<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    need_hessian: bool,
) -> Result<OuterObjectiveEvalResult, String> {
    let include_logdet_h = include_exact_newton_logdet_h(family);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    let per_block = split_log_lambdas(rho, penalty_counts)?;
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, warm_start)?;
    // Outer objective at fixed rho:
    //   V(rho) = -ell(beta^) + 0.5 * beta^T P(rho) beta^
    //            + 0.5 * log|J(beta^,rho)| - 0.5 * log|P(rho)|_+,
    // where beta^ solves g(beta,rho)=0 with
    //   g(beta,rho) = -∇_beta ell(beta) + P(rho) beta,
    //   J(beta,rho) = H(beta) + P(rho),  H(beta) = -∇^2_beta ell(beta).
    //
    // The exact outer gradient used below is:
    //   V_k = 0.5*beta^T A_k beta^
    //       + 0.5*tr(J^{-1}(A_k + D H[u_k]))
    //       - 0.5*tr(P_+^{-1} A_k),
    // with A_k=dP/drho_k and u_k=d beta^/drho_k.
    let objective = -inner.log_likelihood
        + inner.penalty_value
        + if include_logdet_h {
            0.5 * inner.block_logdet_h
        } else {
            0.0
        }
        - if include_logdet_s {
            0.5 * inner.block_logdet_s
        } else {
            0.0
        };
    let mut grad = Array1::<f64>::zeros(rho.len());
    let mut outer_hessian: Option<Array2<f64>> = None;

    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let eval = family.evaluate(&inner.block_states)?;
    if let Some(h_joint_unpen) = family.exact_newton_joint_hessian(&inner.block_states)? {
        let ranges = block_param_ranges(specs);
        let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
        if h_joint_unpen.nrows() != total || h_joint_unpen.ncols() != total {
            return Err(format!(
                "joint exact-newton Hessian shape mismatch in outer gradient: got {}x{}, expected {}x{}",
                h_joint_unpen.nrows(),
                h_joint_unpen.ncols(),
                total,
                total
            ));
        }
        let beta_flat = flatten_state_betas(&inner.block_states, specs);
        let mut s_joint = Array2::<f64>::zeros((total, total));
        let mut s_pinv_joint = if include_logdet_s {
            Some(Array2::<f64>::zeros((total, total)))
        } else {
            None
        };
        for (b, spec) in specs.iter().enumerate() {
            let (start, end) = ranges[b];
            let p = end - start;
            let lambdas = per_block[b].mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((p, p));
            for (k, s) in spec.penalties.iter().enumerate() {
                s_lambda.scaled_add(lambdas[k], s);
            }
            s_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .assign(&s_lambda);
            if let Some(s_pinv) = s_pinv_joint.as_mut() {
                let s_part = pinv_positive_part(
                    &s_lambda,
                    if options.ridge_policy.include_penalty_logdet {
                        effective_solver_ridge(options.ridge_floor)
                    } else {
                        options.ridge_floor
                    },
                )?;
                s_pinv
                    .slice_mut(ndarray::s![start..end, start..end])
                    .assign(&s_part);
            }
        }
        // Inner stationarity for the coupled blocks is:
        //   g(beta,rho) = -∇_beta ell(beta) + P(rho) beta = 0.
        // Differentiating w.r.t. rho_k gives the exact sensitivity system:
        //   J * u_k = -(dP/drho_k) beta,  with u_k = d beta / d rho_k.
        // We use A_k := dP/drho_k and J = H(beta) + P(rho), where
        // H(beta) = -∇^2_beta ell(beta) is the full joint likelihood curvature.
        //
        // Here `h_joint_unpen` is H(beta) and `s_joint` is P(rho), so
        // `j_joint = h_joint_unpen + s_joint` is the coupled Jacobian/Hessian used
        // in both sensitivity solves and log|J| trace terms.
        let mut j_joint = h_joint_unpen.clone();
        j_joint += &s_joint;

        // Log-determinant traces may follow a ridge-augmented surface depending
        // on ridge policy, but coupled beta sensitivities below are always solved
        // against the full joint system J.
        //
        // Exact REML/LAML first derivative for one smoothing coordinate rho_k:
        //
        //   V_k
        //   = 0.5 beta^T A_k beta
        //   + 0.5 tr(J_trace^{-1} J_k^total)
        //   - 0.5 tr(S_+^{-1} A_k),
        //
        // where
        //
        //   A_k = dP/drho_k,
        //   J_mode  = H(beta^) + P,
        //   J_trace = J_mode (+ optional logdet-only ridge),
        //   J_k^total = A_k + D_beta H[u_k],
        //   J_mode u_k = -A_k beta^.
        //
        // The key algebraic separation is:
        // - `u_k` must come from the mode system `J_mode`
        // - the trace may be evaluated on the ridge-adjusted `J_trace`
        //   if the chosen determinant surface includes that stabilization.
        let mut j_for_traces = j_joint.clone();
        if !strict_spd && options.ridge_policy.include_penalty_logdet {
            let ridge = effective_solver_ridge(options.ridge_floor);
            for d in 0..total {
                j_for_traces[[d, d]] += ridge;
            }
        }
        let h_inv = if strict_spd {
            strict_inverse_spd(&j_for_traces)?
        } else {
            match inverse_spd_with_retry(
                &j_for_traces,
                effective_solver_ridge(options.ridge_floor),
                8,
            ) {
                Ok(inv) => inv,
                Err(_) => {
                    // Joint exact geometry can be mildly indefinite away from the inner mode.
                    // Fall back to a positive-part pseudoinverse to keep outer gradients defined.
                    pinv_positive_part(&j_for_traces, options.ridge_floor)?
                }
            }
        };
        let mut at = 0usize;
        let mut a_terms: Vec<Array2<f64>> = Vec::new();
        let mut a_beta_terms: Vec<Array1<f64>> = Vec::new();
        let mut u_terms: Vec<Array1<f64>> = Vec::new();
        let mut j_terms: Vec<Array2<f64>> = Vec::new();
        for b in 0..specs.len() {
            let spec = &specs[b];
            let (start, end) = ranges[b];
            let beta_block = beta_flat.slice(ndarray::s![start..end]).to_owned();
            let lambdas = per_block[b].mapv(f64::exp);
            for (k, s_k) in spec.penalties.iter().enumerate() {
                let mut a_k = Array2::<f64>::zeros((total, total));
                let local = s_k.mapv(|v| lambdas[k] * v);
                a_k.slice_mut(ndarray::s![start..end, start..end])
                    .assign(&local);
                let a_k_beta = a_k.dot(&beta_flat);
                // Penalty-quadratic derivative:
                //   d/drho_k [0.5 * beta^T P beta] = 0.5 * beta^T A_k beta
                // at the inner mode (envelope theorem), with A_k = dP/drho_k.
                let g_pen = 0.5 * beta_block.dot(&local.dot(&beta_block));
                let keep_second_order = need_hessian || include_logdet_h || include_logdet_s;
                let u_k = if keep_second_order {
                    let rhs_k = -&a_k_beta;
                    if strict_spd {
                        strict_solve_spd(&j_joint, &rhs_k)?
                    } else {
                        match solve_spd_system_with_policy(
                            &j_joint,
                            &rhs_k,
                            options.ridge_floor,
                            options.ridge_policy,
                        ) {
                            Ok(sol) => sol,
                            Err(_) => pinv_positive_part(&j_joint, options.ridge_floor)
                                .map(|j_pinv| j_pinv.dot(&rhs_k))
                                .map_err(|_| {
                                    "failed to solve joint mode sensitivity system for REML gradient"
                                        .to_string()
                                })?,
                        }
                    }
                } else {
                    Array1::<f64>::zeros(total)
                };
                let g = if include_logdet_h || include_logdet_s {
                    let g_logs = if include_logdet_s {
                        0.5
                            * trace_product(
                                s_pinv_joint.as_ref().ok_or_else(|| {
                                    "missing joint S^+ for REML gradient".to_string()
                                })?,
                                &a_k,
                            )
                    } else {
                        0.0
                    };
                    let u_norm = u_k.dot(&u_k).sqrt();
                    // Exact log-determinant derivative structure:
                    //
                    //   d/drho_k log|J_trace|
                    //   = tr(J_trace^{-1} J_k^total),
                    //
                    // with
                    //
                    //   J_k^total = A_k + D_beta H[u_k].
                    //
                    // The first term A_k is the explicit penalty drift. The
                    // second term is the implicit beta-path correction: moving
                    // rho_k changes the fitted mode beta^, and therefore changes
                    // the likelihood curvature H(beta^).
                    //
                    // This branch preserves full cross-block coupling because
                    // u_k is solved against the full joint mode system, not a
                    // blockwise approximation.
                    let mut d_j_k = a_k.clone();
                    if u_norm > 1e-14 {
                        // D_beta H[u_k] is the directional derivative of the
                        // joint likelihood Hessian with respect to beta along
                        // u_k = d beta^ / d rho_k.
                        let h_rho = family
                            .exact_newton_joint_hessian_directional_derivative(
                                &inner.block_states,
                                &u_k,
                            )?
                            .ok_or_else(|| {
                                "joint exact-newton dH unavailable for analytic outer gradient"
                                    .to_string()
                            })?;
                        if !h_rho.iter().all(|v| v.is_finite()) {
                            return Err(
                                "joint exact-newton dH contains non-finite values".to_string()
                            );
                        }
                        if h_rho.nrows() != total || h_rho.ncols() != total {
                            return Err(format!(
                                "joint exact-newton dH shape mismatch: got {}x{}, expected {}x{}",
                                h_rho.nrows(),
                                h_rho.ncols(),
                                total,
                                total
                            ));
                        }
                        d_j_k += &h_rho;
                    }
                    let g_logj = if include_logdet_h {
                        0.5 * trace_product(&h_inv, &d_j_k)
                    } else {
                        0.0
                    };
                    if need_hessian {
                        a_terms.push(a_k.clone());
                        a_beta_terms.push(a_k_beta.clone());
                        u_terms.push(u_k.clone());
                        j_terms.push(d_j_k);
                    }
                    g_pen + g_logj - g_logs
                } else {
                    if need_hessian {
                        a_terms.push(a_k.clone());
                        a_beta_terms.push(a_k_beta.clone());
                        u_terms.push(u_k.clone());
                    }
                    g_pen
                };
                grad[at + k] = g;
            }
            at += spec.penalties.len();
        }
        if need_hessian {
            let mut hess = Array2::<f64>::zeros((rho.len(), rho.len()));
            let mut hess_available = true;
            for k in 0..rho.len() {
                for l in k..rho.len() {
                    // Outer Hessian entry (rho = log lambda coordinates):
                    //   V_{k,l} =
                    //     beta^T A_k u_l + 0.5*delta_{k,l}*beta^T A_k beta
                    //   + 0.5*( -tr(J^{-1}J_l J^{-1}J_k) + tr(J^{-1}J_{k,l}) )
                    //   + 0.5*tr(P^+ A_l P^+ A_k) - 0.5*delta_{k,l}*tr(P^+ A_k),
                    // where A_k = dP/drho_k = lambda_k S_k and
                    // delta_{k,l} A_k appears because dA_k/drho_l = delta_{k,l} A_k.
                    let delta_kl = if k == l { 1.0 } else { 0.0 };
                    let mut v_kl = a_beta_terms[k].dot(&u_terms[l])
                        + 0.5 * delta_kl * beta_flat.dot(&a_beta_terms[k]);
                    if include_logdet_h || include_logdet_s {
                        let tr_prod = trace_jinv_a_jinv_b(&h_inv, &j_terms[l], &j_terms[k]);
                        // Second-order sensitivity:
                        //   J u_{k,l} = -(J_l u_k + A_k u_l + delta_{k,l} A_k beta).
                        let rhs_kl = -(a_terms[k].dot(&u_terms[l])
                            + j_terms[l].dot(&u_terms[k])
                            + delta_kl * a_beta_terms[k].clone());
                        let u_kl = if strict_spd {
                            strict_solve_spd(&j_joint, &rhs_kl)?
                        } else {
                            match solve_spd_system_with_policy(
                                &j_joint,
                                &rhs_kl,
                                options.ridge_floor,
                                options.ridge_policy,
                            ) {
                                Ok(sol) => sol,
                                Err(_) => pinv_positive_part(&j_joint, options.ridge_floor)
                                    .map(|j_pinv| j_pinv.dot(&rhs_kl))
                                    .map_err(|_| {
                                        "failed to solve joint second-order mode sensitivity system for REML Hessian"
                                            .to_string()
                                    })?,
                            }
                        };
                        let dh_u_kl = match family
                            .exact_newton_joint_hessian_directional_derivative(
                                &inner.block_states,
                                &u_kl,
                            )? {
                            Some(v) => v,
                            None => {
                                hess_available = false;
                                break;
                            }
                        };
                        let d2h_ul_uk = match family
                            .exact_newton_joint_hessian_second_directional_derivative(
                                &inner.block_states,
                                &u_terms[l],
                                &u_terms[k],
                            )? {
                            Some(v) => v,
                            None => {
                                hess_available = false;
                                break;
                            }
                        };
                        // J_{k,l} chain rule with log-lambda correction:
                        //   J_{k,l} = dH[u_{k,l}] + d^2H[u_l,u_k] + delta_{k,l} A_k.
                        let mut j_kl = dh_u_kl;
                        j_kl += &d2h_ul_uk;
                        if delta_kl > 0.0 {
                            j_kl += &a_terms[k];
                        }
                        let tr_second = trace_product(&h_inv, &j_kl);
                        let tr_h = if include_logdet_h {
                            0.5 * (-tr_prod + tr_second)
                        } else {
                            0.0
                        };
                        let tr_p = if include_logdet_s {
                            0.5 * trace_jinv_a_jinv_b(
                                s_pinv_joint.as_ref().ok_or_else(|| {
                                    "missing joint S^+ for REML Hessian".to_string()
                                })?,
                                &a_terms[l],
                                &a_terms[k],
                            ) - 0.5
                                * delta_kl
                                * trace_product(
                                    s_pinv_joint.as_ref().ok_or_else(|| {
                                        "missing joint S^+ for REML Hessian".to_string()
                                    })?,
                                    &a_terms[k],
                                )
                        } else {
                            0.0
                        };
                        v_kl += tr_h + tr_p;
                    }
                    hess[[k, l]] = v_kl;
                    hess[[l, k]] = v_kl;
                }
                if !hess_available {
                    break;
                }
            }
            if hess_available {
                outer_hessian = Some(hess);
            }
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
        return Ok(OuterObjectiveEvalResult {
            objective,
            gradient: grad,
            outer_hessian,
            warm_start: warm,
            inner,
        });
    }
    let mut at = 0usize;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let work = &eval.block_working_sets[b];
        let p = spec.design.ncols();
        let mut diagonal_design = None::<DesignMatrix>;
        let xtwx = match work {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => with_block_geometry(family, &inner.block_states, spec, b, |x_dyn, _| {
                let w = working_weights.mapv(|wi| wi.max(options.min_weight));
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
                    ));
                }
                hessian.clone()
            }
        };

        let lambdas = per_block[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s_lambda.scaled_add(lambdas[k], s);
        }

        let mut h_mode = xtwx;
        h_mode += &s_lambda;
        let mut h_for_logdet = h_mode.clone();
        if !strict_spd && options.ridge_policy.include_penalty_logdet {
            let ridge = effective_solver_ridge(options.ridge_floor);
            for d in 0..p {
                h_for_logdet[[d, d]] += ridge;
            }
        }
        let h_inv = if strict_spd {
            strict_inverse_spd(&h_for_logdet)?
        } else {
            inverse_spd_with_retry(
                &h_for_logdet,
                effective_solver_ridge(options.ridge_floor),
                8,
            )?
        };
        let diagonal_leverages = diagonal_design
            .as_ref()
            .map(|x_dyn| leverage_quadratic_forms(x_dyn, &h_inv));

        let mut s_for_logdet = s_lambda.clone();
        if options.ridge_policy.include_penalty_logdet {
            let ridge = effective_solver_ridge(options.ridge_floor);
            for d in 0..p {
                s_for_logdet[[d, d]] += ridge;
            }
        }
        let s_pinv = if include_logdet_s {
            Some(pinv_positive_part(&s_for_logdet, options.ridge_floor)?)
        } else {
            None
        };

        let beta = &inner.block_states[b].beta;
        for (k, s_k) in spec.penalties.iter().enumerate() {
            let a_k = s_k.mapv(|v| lambdas[k] * v);
            let a_k_beta = a_k.dot(beta);
            let g_pen = 0.5 * beta.dot(&a_k_beta);
            let g = if include_logdet_h || include_logdet_s {
                // Here H is per-block penalized likelihood curvature:
                //   H = -∇^2_{beta_b} ell + S_lambda.
                // For each smoothing coordinate in this block:
                //   d/drho_k [0.5 log|H|] = 0.5 tr(H^{-1}(A_k + D H[u_k])),
                //   d/drho_k [0.5 log|S|_+] = 0.5 tr(S_+^{-1} A_k).
                //
                // The code computes:
                //   g_logh  = 0.5 tr(H^{-1} A_k),
                //   g_hbeta = 0.5 tr(H^{-1} D H[u_k]),
                //   g_logs  = 0.5 tr(S_+^{-1} A_k),
                // then combines them with g_pen.
                //
                // For exact second derivatives wrt rho in this block, the same pattern as
                // the joint branch applies:
                //   H_{k,l} contribution needs D H[u_{k,l}] and D^2 H[u_l, u_k].
                let g_logh = if include_logdet_h {
                    0.5 * trace_product(&h_inv, &a_k)
                } else {
                    0.0
                };
                let g_logs = if include_logdet_s {
                    0.5
                        * trace_product(
                            s_pinv
                                .as_ref()
                                .ok_or_else(|| "missing S^+ for REML gradient".to_string())?,
                            &a_k,
                        )
                } else {
                    0.0
                };
                // Exact derivative of the log|H| term:
                //   0.5 * tr(H^{-1} * dH/drho_k),
                // with
                //   dH/drho_k = A_k + D H[u_k],
                //   A_k = dS/drho_k = lambda_k * S_k,
                //   H u_k = -A_k beta.
                //
                // `g_logh` is the explicit part 0.5*tr(H^{-1} A_k), and `g_hbeta` below
                // is the implicit beta-path part 0.5*tr(H^{-1} D H[u_k]).
                // Smoothing-coordinate sensitivity solve:
                //
                //   rho_k = log lambda_k,
                //   A_k   = dS/drho_k = lambda_k S_k,
                //   J     = H(beta^) + S.
                //
                // Differentiating stationarity g(beta^,rho)=0 gives
                //
                //   J u_k = -A_k beta^,
                //
                // so u_k = d beta^ / d rho_k must be solved against the mode
                // Hessian `h_mode`. The log-determinant may use a ridge-adjusted
                // matrix for trace evaluation, but that ridge is not part of the
                // stationarity surface and must not enter the sensitivity solve.
                let rhs = -&a_k_beta;
                let u_k = if strict_spd {
                    strict_solve_spd(&h_mode, &rhs)?
                } else {
                    solve_spd_system_with_policy(
                        &h_mode,
                        &rhs,
                        options.ridge_floor,
                        options.ridge_policy,
                    )
                    .or_else(|_| {
                        pinv_positive_part(&h_mode, options.ridge_floor)
                            .map(|h_pinv| h_pinv.dot(&rhs))
                    })?
                };
                let u_norm = u_k.dot(&u_k).sqrt();
                let g_hbeta = if !include_logdet_h || u_norm <= 1e-14 {
                    0.0
                } else {
                    match work {
                        BlockWorkingSet::ExactNewton { .. } => {
                            if let Some(h_exact) = family
                                .exact_newton_hessian_directional_derivative(
                                    &inner.block_states,
                                    b,
                                    &u_k,
                                )?
                            {
                                if h_exact.nrows() != p || h_exact.ncols() != p {
                                    return Err(format!(
                                        "block {b} exact-newton dH shape mismatch: got {}x{}, expected {}x{}",
                                        h_exact.nrows(),
                                        h_exact.ncols(),
                                        p,
                                        p
                                    ));
                                }
                                0.5 * trace_product(&h_inv, &h_exact)
                            } else {
                                return Err(format!(
                                    "missing exact-newton dH callback for block {b} while REML gradient requires H_beta term"
                                ));
                            }
                        }
                        BlockWorkingSet::Diagonal { .. } => {
                            // For diagonal working sets:
                            //   H = X' diag(w) X + S
                            //   D H[u_k]
                            //   = (D X[u_k])' W X
                            //   + X' W (D X[u_k])
                            //   + X' diag(dw) X,
                            //   dw = D_beta w [D eta[u_k]].
                            // Then
                            //   tr(H^{-1} D H[u_k]) = sum_i q_i * dw_i,
                            // where q_i = x_i' H^{-1} x_i (precomputed leverages),
                            // plus the explicit geometry-drift trace from `D X[u_k]`.
                            let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                                format!(
                                    "missing dynamic design for block {b} diagonal H_rho trace term"
                                )
                            })?;
                            let w_work = match work {
                                BlockWorkingSet::Diagonal {
                                    working_response: _,
                                    working_weights,
                                } => working_weights.mapv(|wi| wi.max(options.min_weight)),
                                BlockWorkingSet::ExactNewton { .. } => unreachable!(),
                            };
                            let leverages = diagonal_leverages.as_ref().ok_or_else(|| {
                                format!(
                                    "missing leverage cache for block {b} diagonal H_rho trace term"
                                )
                            })?;
                            let x_dense = x_dyn.to_dense();
                            let mut d_eta = x_dyn.matrix_vector_multiply(&u_k);
                            let geom_trace = apply_geometry_direction_to_eta_and_trace(
                                &x_dense,
                                beta,
                                &w_work,
                                &h_inv,
                                &mut d_eta,
                                family.block_geometry_directional_derivative(
                                    &inner.block_states,
                                    b,
                                    spec,
                                    &u_k,
                                )?,
                            )?;
                            if let Some(dw) = family
                                .diagonal_working_weights_directional_derivative(
                                    &inner.block_states,
                                    b,
                                    &d_eta,
                                )?
                            {
                                if dw.len() != leverages.len() {
                                    return Err(format!(
                                        "block {b} diagonal dW length mismatch: got {}, expected {}",
                                        dw.len(),
                                        leverages.len()
                                    ));
                                }
                                geom_trace + 0.5 * leverages.dot(&dw)
                            } else {
                                return Err(format!(
                                    "missing diagonal dW callback for block {b} while REML gradient requires H_beta term"
                                ));
                            }
                        }
                    }
                };

                g_pen + g_logh + g_hbeta - g_logs
            } else {
                g_pen
            };
            grad[at + k] = g;
        }
        at += spec.penalties.len();
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
        inner,
    })
}

fn outer_objective_gradient_hessian<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    need_hessian: bool,
) -> Result<(f64, Array1<f64>, Option<Array2<f64>>, ConstrainedWarmStart), String> {
    let result = outer_objective_gradient_hessian_internal(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        warm_start,
        need_hessian,
    )?;
    Ok((
        result.objective,
        result.gradient,
        result.outer_hessian,
        result.warm_start,
    ))
}

#[cfg(test)]
fn outer_objective_and_gradient<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<(f64, Array1<f64>, ConstrainedWarmStart), String> {
    let (obj, grad, _hess, warm) = outer_objective_gradient_hessian(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        warm_start,
        false,
    )?;
    Ok((obj, grad, warm))
}

fn compute_custom_family_block_psi_gradients<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    per_block: &[Array1<f64>],
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    inner: &mut BlockwiseInnerResult,
) -> Result<Array1<f64>, String> {
    let include_logdet_h = include_exact_newton_logdet_h(family);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let eval = family.evaluate(&inner.block_states)?;
    if eval.block_working_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.block_working_sets.len(),
            specs.len()
        ));
    }

    let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
    let mut grad = Array1::<f64>::zeros(psi_dim);
    let ridge = effective_solver_ridge(options.ridge_floor);
    let mut at = 0usize;

    for (b, spec) in specs.iter().enumerate() {
        let psi_terms = &derivative_blocks[b];
        if psi_terms.is_empty() {
            continue;
        }

        let p = spec.design.ncols();
        let beta = &inner.block_states[b].beta;
        let eta = &inner.block_states[b].eta;
        let lambdas = per_block[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s_lambda.scaled_add(lambdas[k], s);
        }

        let mut s_for_logdet = s_lambda.clone();
        if options.ridge_policy.include_penalty_logdet {
            for d in 0..p {
                s_for_logdet[[d, d]] += ridge;
            }
        }
        let s_pinv = if include_logdet_s {
            Some(pinv_positive_part(&s_for_logdet, options.ridge_floor)?)
        } else {
            None
        };

        let work = &eval.block_working_sets[b];
        let x_dense = with_block_geometry(family, &inner.block_states, spec, b, |x_dyn, _| {
            Ok(x_dyn.to_dense())
        })?;

        let (w, u, h_mode, h_inv) = match work {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => with_block_geometry(family, &inner.block_states, spec, b, |x_dyn, _| {
                if working_response.len() != x_dyn.nrows() || working_weights.len() != x_dyn.nrows()
                {
                    return Err(format!(
                        "family diagonal working-set size mismatch on block {b} ({})",
                        spec.name
                    ));
                }
                let w = working_weights.mapv(|wi| wi.max(options.min_weight));
                let u = &w * &(working_response - eta);
                let (xtwx, _) = weighted_normal_equations(x_dyn, &w, None)?;
                let mut h_for_logdet = xtwx;
                h_for_logdet += &s_lambda;
                let h_mode = h_for_logdet.clone();
                if !strict_spd && options.ridge_policy.include_penalty_logdet {
                    for d in 0..p {
                        h_for_logdet[[d, d]] += ridge;
                    }
                }
                let h_inv = if strict_spd {
                    strict_inverse_spd(&h_for_logdet)?
                } else {
                    inverse_spd_with_retry(&h_for_logdet, ridge, 8)?
                };
                Ok((w, u, h_mode, h_inv))
            })?,
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                if hessian.nrows() != p || hessian.ncols() != p {
                    return Err(format!(
                        "exact-newton block {b} Hessian shape mismatch in psi gradient: got {}x{}, expected {}x{}",
                        hessian.nrows(),
                        hessian.ncols(),
                        p,
                        p
                    ));
                }
                let mut h_mode = hessian.clone();
                h_mode += &s_lambda;
                let mut h_for_logdet = h_mode.clone();
                if !strict_spd && options.ridge_policy.include_penalty_logdet {
                    for d in 0..p {
                        h_for_logdet[[d, d]] += ridge;
                    }
                }
                let h_inv = if strict_spd {
                    strict_inverse_spd(&h_for_logdet)?
                } else {
                    inverse_spd_with_retry(&h_for_logdet, ridge, 8)?
                };

                for deriv in psi_terms {
                    // Exact-Newton generic psi-gradient path for penalty-only psi
                    // directions.
                    //
                    // This covers the mathematically important case used by the
                    // spatial Charbonnier hyperparameters:
                    //   - the likelihood has no direct psi dependence,
                    //   - the design does not move with psi (X_psi = 0),
                    //   - only the penalty surface changes.
                    //
                    // Then the exact outer derivative of
                    //
                    //   V(psi) = -loglik(beta^(psi)) + 0.5 beta^T S(psi) beta
                    //            + 0.5 log|H(beta^(psi), psi)|
                    //            - 0.5 log|S(psi)|_+
                    //
                    // reduces to:
                    //
                    //   explicit  = 0.5 beta^T S_psi beta
                    //   H_mode u  = S_psi beta
                    //   Hdot      = S_psi + D_beta H[-u]
                    //   dV/dpsi   = explicit + 0.5 tr(H^{-1} Hdot)
                    //               - 0.5 tr(S_+^{-1} S_psi).
                    //
                    // Since beta_psi = -u and the family callback already returns
                    // D_beta H[direction], we form that exact quantity directly.
                    let xpsi_nonzero = deriv.x_psi.iter().any(|v| *v != 0.0);
                    let generic_supported = !xpsi_nonzero;
                    let family_value = family.exact_newton_block_psi_gradient(
                        &inner.block_states,
                        b,
                        ExactNewtonPsiGradientContext {
                            spec,
                            deriv,
                            lambdas: &lambdas,
                            options,
                        },
                    )?;
                    let s_psi_total = if let Some(parts) = deriv.s_psi_components.as_ref() {
                        let mut total = Array2::<f64>::zeros(deriv.s_psi.raw_dim());
                        for (k, s_part) in parts {
                            if *k >= lambdas.len() {
                                if family_value.is_some() {
                                    continue;
                                }
                                return Err(format!(
                                    "psi derivative component penalty index {} out of bounds for exact-newton block {}",
                                    k, b
                                ));
                            }
                            total.scaled_add(lambdas[*k], s_part);
                        }
                        total
                    } else if deriv.penalty_index < lambdas.len() {
                        deriv.s_psi.mapv(|v| lambdas[deriv.penalty_index] * v)
                    } else if family_value.is_some() {
                        Array2::<f64>::zeros(deriv.s_psi.raw_dim())
                    } else {
                        return Err(format!(
                            "psi derivative penalty index {} out of bounds for exact-newton block {}",
                            deriv.penalty_index, b
                        ));
                    };
                    let value = if let Some(value) = family_value {
                        Ok(value)
                    } else if generic_supported {
                        let explicit = 0.5 * beta.dot(&s_psi_total.dot(beta));
                        let rhs = s_psi_total.dot(beta);
                        let u_psi = if strict_spd {
                            strict_solve_spd(&h_mode, &rhs)?
                        } else {
                            solve_spd_system_with_policy(
                                &h_mode,
                                &rhs,
                                options.ridge_floor,
                                options.ridge_policy,
                            )
                            .or_else(|_| {
                                pinv_positive_part(&h_mode, options.ridge_floor)
                                    .map(|h_pinv| h_pinv.dot(&rhs))
                            })?
                        };
                        let beta_psi = -&u_psi;
                        let mut h_dot = s_psi_total.clone();
                        if beta_psi.dot(&beta_psi).sqrt() > 1e-14 {
                            let h_beta = family
                                .exact_newton_hessian_directional_derivative(
                                    &inner.block_states,
                                    b,
                                    &beta_psi,
                                )?
                                .ok_or_else(|| {
                                    format!(
                                        "missing exact-newton dH callback for psi gradient on block {b}"
                                    )
                                })?;
                            if h_beta.nrows() != p || h_beta.ncols() != p {
                                return Err(format!(
                                    "exact-newton psi dH shape mismatch on block {b}: got {}x{}, expected {}x{}",
                                    h_beta.nrows(),
                                    h_beta.ncols(),
                                    p,
                                    p
                                ));
                            }
                            h_dot += &h_beta;
                        }
                        let h_trace = if include_logdet_h {
                            0.5 * trace_product(&h_inv, &h_dot)
                        } else {
                            0.0
                        };
                        let pseudo_det = if include_logdet_s {
                            -0.5
                                * trace_product(
                                    s_pinv.as_ref().ok_or_else(|| {
                                        "missing S^+ for exact-newton psi gradient".to_string()
                                    })?,
                                    &s_psi_total,
                                )
                        } else {
                            0.0
                        };
                        Ok(explicit + h_trace + pseudo_det)
                    } else {
                        Err(format!(
                            "analytic psi gradient is not implemented for exact-newton block {} ({})",
                            b, spec.name
                        ))
                    }?;
                    let _ = gradient;
                    grad[at] = value;
                    at += 1;
                }
                continue;
            }
        };

        let mut wx = x_dense.clone();
        for i in 0..wx.nrows() {
            let wi = w[i];
            if wi != 1.0 {
                let mut row = wx.row_mut(i);
                row *= wi;
            }
        }

        for deriv in psi_terms {
            if deriv.x_psi.nrows() != x_dense.nrows() || deriv.x_psi.ncols() != x_dense.ncols() {
                return Err(format!(
                    "X_psi shape mismatch on block {}: expected {}x{}, got {}x{}",
                    b,
                    x_dense.nrows(),
                    x_dense.ncols(),
                    deriv.x_psi.nrows(),
                    deriv.x_psi.ncols()
                ));
            }
            if deriv.s_psi.nrows() != p || deriv.s_psi.ncols() != p {
                return Err(format!(
                    "S_psi shape mismatch on block {}: expected {}x{}, got {}x{}",
                    b,
                    p,
                    p,
                    deriv.s_psi.nrows(),
                    deriv.s_psi.ncols()
                ));
            }

            let xpsi_beta = deriv.x_psi.dot(beta);
            let w_psi = family.diagonal_working_weights_directional_derivative(
                &inner.block_states,
                b,
                &xpsi_beta,
            )?;
            if options.use_reml_objective && w_psi.is_none() {
                return Err(format!(
                    "missing explicit psi dW callback for block {b} while REML psi gradient requires X^T diag(w_psi) X"
                ));
            }
            let s_psi_total = if let Some(parts) = deriv.s_psi_components.as_ref() {
                let mut total = Array2::<f64>::zeros(deriv.s_psi.raw_dim());
                for (k, s_part) in parts {
                    if *k >= lambdas.len() {
                        return Err(format!(
                            "psi derivative component penalty index {} out of bounds for block {}",
                            k, b
                        ));
                    }
                    total.scaled_add(lambdas[*k], s_part);
                }
                total
            } else {
                if deriv.penalty_index >= lambdas.len() {
                    return Err(format!(
                        "psi derivative penalty index {} out of bounds for block {}",
                        deriv.penalty_index, b
                    ));
                }
                deriv.s_psi.mapv(|v| lambdas[deriv.penalty_index] * v)
            };
            // Geometry/penalty derivative psi at the fitted inner mode beta^:
            //
            //   V(psi) = f(beta^(psi), psi)
            //          + 0.5 log|J(beta^(psi), psi)|
            //          - 0.5 log|S(psi)|_+,
            //
            // where
            //   f(beta,psi) = -ell(beta,psi) + 0.5 beta^T S(psi) beta.
            //
            // By the envelope theorem, the fit block is the explicit partial:
            //
            //   d/dpsi f(beta^(psi),psi) = f_psi(beta^,psi).
            //
            // In the diagonal working-set notation used here,
            //   u = W(z - eta)
            // is the predictor-space score contribution, and
            //   eta_psi = X_psi beta
            // for this derivative object. Therefore
            //
            //   f_psi = -u^T eta_psi + 0.5 beta^T S_psi beta.
            let explicit = -u.dot(&xpsi_beta) + 0.5 * beta.dot(&s_psi_total.dot(beta));

            let mut wx_psi = deriv.x_psi.clone();
            for i in 0..wx_psi.nrows() {
                let wi = w[i];
                if wi != 1.0 {
                    let mut row = wx_psi.row_mut(i);
                    row *= wi;
                }
            }
            // Explicit curvature drift at fixed beta:
            //
            //   J = X^T W X + S
            //
            // so
            //
            //   J_psi^explicit
            //   = X_psi^T W X
            //   + X^T W X_psi
            //   + X^T diag(w_psi) X
            //   + S_psi.
            //
            // The first two terms are assembled through `wx` / `wx_psi`, the
            // third via `h_w_psi`, and the last by adding `s_psi_total`.
            let mut h_psi = deriv.x_psi.t().dot(&wx);
            h_psi += &x_dense.t().dot(&wx_psi);
            if let Some(dw) = w_psi {
                if dw.len() != x_dense.nrows() {
                    return Err(format!(
                        "block {b} psi dW length mismatch: got {}, expected {}",
                        dw.len(),
                        x_dense.nrows()
                    ));
                }
                let mut h_w_psi = Array2::<f64>::zeros((p, p));
                for i in 0..x_dense.nrows() {
                    let wi = dw[i];
                    if wi == 0.0 {
                        continue;
                    }
                    for a in 0..p {
                        let xa = x_dense[[i, a]];
                        for bb in a..p {
                            h_w_psi[[a, bb]] += wi * xa * x_dense[[i, bb]];
                        }
                    }
                }
                for a in 0..p {
                    for bb in 0..a {
                        h_w_psi[[a, bb]] = h_w_psi[[bb, a]];
                    }
                }
                h_psi += &h_w_psi;
            }
            h_psi += &s_psi_total;
            let trace_term = if options.use_reml_objective {
                // Exact REML trace derivative:
                //
                //   d/dpsi [0.5 log|J|]
                //   = 0.5 tr(J^{-1}(J_psi^explicit + D_beta J[u_psi])).
                let mut trace_acc = 0.5 * trace_product(&h_inv, &h_psi);
                // Implicit mode-sensitivity correction:
                //   0.5 * tr(J^{-1} D_beta J[u_psi]), with
                //   J * u_psi = -g_psi.
                let g_psi = {
                    let eta_psi = xpsi_beta.clone();
                    // Differentiate block stationarity
                    //
                    //   0 = g(beta^(psi),psi) = X^T u - S beta
                    //
                    // at fixed beta to obtain
                    //
                    //   g_psi = X_psi^T u - X^T W eta_psi - S_psi beta.
                    let mut out = deriv.x_psi.t().dot(&u);
                    out -= &x_dense.t().dot(&(&w * &eta_psi));
                    out -= &s_psi_total.dot(beta);
                    out
                };
                let rhs = -&g_psi;
                let u_psi = solve_spd_system_with_policy(
                    &h_mode,
                    &rhs,
                    options.ridge_floor,
                    options.ridge_policy,
                )
                .or_else(|_| {
                    pinv_positive_part(&h_mode, options.ridge_floor).map(|h_pinv| h_pinv.dot(&rhs))
                })?;
                let mut d_eta_mode = x_dense.dot(&u_psi);
                trace_acc += apply_geometry_direction_to_eta_and_trace(
                    &x_dense,
                    beta,
                    &w,
                    &h_inv,
                    &mut d_eta_mode,
                    family.block_geometry_directional_derivative(
                        &inner.block_states,
                        b,
                        spec,
                        &u_psi,
                    )?,
                )?;
                if let Some(dw_mode) = family.diagonal_working_weights_directional_derivative(
                    &inner.block_states,
                    b,
                    &d_eta_mode,
                )? {
                    if dw_mode.len() != x_dense.nrows() {
                        return Err(format!(
                            "block {b} implicit psi dW length mismatch: got {}, expected {}",
                            dw_mode.len(),
                            x_dense.nrows()
                        ));
                    }
                    let mut leverage = Array1::<f64>::zeros(x_dense.nrows());
                    for i in 0..x_dense.nrows() {
                        let xi = x_dense.row(i).to_owned();
                        leverage[i] = xi.dot(&h_inv.dot(&xi));
                    }
                    // For diagonal working weights,
                    //
                    //   D_beta J[u_psi] = X^T diag(dw_mode) X,
                    //
                    // hence
                    //
                    //   tr(J^{-1} D_beta J[u_psi])
                    //   = sum_i (x_i^T J^{-1} x_i) dw_mode_i.
                    trace_acc += 0.5 * leverage.dot(&dw_mode);
                } else if u_psi.dot(&u_psi).sqrt() > 1e-14 {
                    return Err(format!(
                        "missing implicit psi dW callback for block {b} while REML psi gradient requires D_beta J[u_psi]"
                    ));
                }
                trace_acc
            } else {
                0.0
            };
            let pseudo_det_term = if options.use_reml_objective {
                // Positive-part penalty determinant correction:
                //
                //   d/dpsi [-0.5 log|S|_+]
                //   = -0.5 tr(S_+^{-1} S_psi).
                -0.5 * trace_product(
                    s_pinv
                        .as_ref()
                        .ok_or_else(|| "missing S^+ for psi gradient".to_string())?,
                    &s_psi_total,
                )
            } else {
                0.0
            };
            grad[at] = explicit + trace_term + pseudo_det_term;
            at += 1;
        }
    }

    Ok(grad)
}

fn apply_geometry_direction_to_eta_and_trace(
    x_dense: &Array2<f64>,
    beta: &Array1<f64>,
    w: &Array1<f64>,
    h_inv: &Array2<f64>,
    base_d_eta: &mut Array1<f64>,
    geom_dir: Option<BlockGeometryDirectionalDerivative>,
) -> Result<f64, String> {
    // Geometry drift helper for diagonal REML derivatives.
    //
    // For dynamic block geometry,
    //
    //   eta(beta) = X(beta) beta + o(beta),
    //
    // so along a coefficient-space direction `u`
    //
    //   D eta[u] = X u + (D X[u]) beta + D o[u].
    //
    // The diagonal Hessian used by the REML trace algebra is
    //
    //   H(beta) = X(beta)^T W(beta) X(beta) + S.
    //
    // Differentiating that composite object along `u` gives
    //
    //   D H[u]
    //   = (D X[u])^T W X
    //   + X^T W (D X[u])
    //   + X^T diag(D w[D eta[u]]) X.
    //
    // The first two terms are explicit geometry-curvature drift. The third is
    // the usual weight-direction term, but its argument is the *full*
    // predictor derivative D eta[u], not just X u.
    //
    // This helper therefore does two jobs:
    //
    //   1. augment `base_d_eta` by
    //        (D X[u]) beta + D o[u],
    //      so the family's dW callback sees the correct predictor direction;
    //
    //   2. return the explicit trace contribution
    //        0.5 tr(H^{-1}[(D X[u])^T W X + X^T W (D X[u])]).
    //
    // Without both pieces, a dynamic `block_geometry` family would be
    // differentiating the wrong Hessian even if its dW callback were exact.
    let Some(geom) = geom_dir else {
        return Ok(0.0);
    };
    if geom.d_offset.len() != x_dense.nrows() {
        return Err(format!(
            "geometry directional offset length mismatch: got {}, expected {}",
            geom.d_offset.len(),
            x_dense.nrows()
        ));
    }
    *base_d_eta += &geom.d_offset;
    let mut trace_term = 0.0;
    if let Some(dx) = geom.d_design {
        if dx.nrows() != x_dense.nrows() || dx.ncols() != x_dense.ncols() {
            return Err(format!(
                "geometry directional design shape mismatch: got {}x{}, expected {}x{}",
                dx.nrows(),
                dx.ncols(),
                x_dense.nrows(),
                x_dense.ncols()
            ));
        }
        *base_d_eta += &dx.dot(beta);
        let mut wx = x_dense.clone();
        let mut wdx = dx.clone();
        for i in 0..x_dense.nrows() {
            let wi = w[i];
            if wi != 1.0 {
                let mut wx_row = wx.row_mut(i);
                wx_row *= wi;
                let mut wdx_row = wdx.row_mut(i);
                wdx_row *= wi;
            }
        }
        let mut d_h_geom = dx.t().dot(&wx);
        d_h_geom += &x_dense.t().dot(&wdx);
        trace_term = 0.5 * trace_product(h_inv, &d_h_geom);
    }
    Ok(trace_term)
}

/// Evaluate the joint outer hyper surface for the *currently realized* custom-family state.
///
/// This function is intentionally not a pure map from a full `theta=[rho, psi]` vector.
/// The geometry/penalty `psi` coordinates must already have been applied by the caller when
/// constructing:
///
/// - `family`
/// - `specs`
/// - `derivative_blocks`
///
/// Therefore the explicit input here is only the current smoothing coordinate `rho`.
/// The returned gradient is still stacked as `[dV/drho, dV/dpsi]`, but the `psi` part is the
/// derivative of the scalar objective for the *externally realized* `psi` state encoded in those
/// inputs, not with respect to any unseen local `theta` tail.
pub fn evaluate_custom_family_joint_hyper<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_current: &Array1<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    warm_start: Option<&CustomFamilyWarmStart>,
    need_hessian: bool,
) -> Result<CustomFamilyJointHyperResult, CustomFamilyError> {
    if derivative_blocks.len() != specs.len() {
        return Err(format!(
            "joint hyper derivative block count mismatch: got {}, expected {}",
            derivative_blocks.len(),
            specs.len()
        )
        .into());
    }

    let penalty_counts = validate_block_specs(specs)?;
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

    let warm_inner = warm_start.map(|w| &w.inner);
    let mut result = outer_objective_gradient_hessian_internal(
        family,
        specs,
        options,
        &penalty_counts,
        rho_current,
        warm_inner,
        need_hessian,
    )?;
    let per_block = split_log_lambdas(rho_current, &penalty_counts)?;
    let psi_grad = compute_custom_family_block_psi_gradients(
        family,
        specs,
        options,
        &per_block,
        derivative_blocks,
        &mut result.inner,
    )?;

    let mut gradient = Array1::<f64>::zeros(rho_dim + psi_dim);
    gradient
        .slice_mut(ndarray::s![..rho_dim])
        .assign(&result.gradient);
    gradient.slice_mut(ndarray::s![rho_dim..]).assign(&psi_grad);

    Ok(CustomFamilyJointHyperResult {
        objective: result.objective,
        gradient,
        // Exact outer Hessian derivation currently covers rho-block only
        // (smoothing-parameter coordinates). For joint theta=[rho,psi],
        // a full exact Hessian would also require psi/psi and rho/psi second
        // derivatives, which are not provided by this path.
        outer_hessian: if need_hessian && psi_dim == 0 {
            result.outer_hessian
        } else {
            None
        },
        warm_start: CustomFamilyWarmStart {
            inner: result.warm_start,
        },
    })
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

fn penalized_objective_at_beta<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
) -> Result<f64, String> {
    let eval = family.evaluate(states)?;
    if eval.block_working_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.block_working_sets.len(),
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

fn compute_joint_hessian_from_objective<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
) -> Result<Array2<f64>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let beta_hat = flatten_state_betas(states, specs);
    let mut h = Array2::<f64>::zeros((total, total));

    let mut states_f0 = states.to_vec();
    set_states_from_flat_beta(&mut states_f0, specs, &beta_hat)?;
    refresh_all_block_etas(family, specs, &mut states_f0)?;
    let f0 = penalized_objective_at_beta(family, specs, &states_f0, per_block_log_lambdas)?;

    let steps = Array1::from_iter(beta_hat.iter().map(|&b| (1e-4 * (1.0 + b.abs())).max(1e-6)));

    for i in 0..total {
        let hi = steps[i];
        let mut bp = beta_hat.clone();
        bp[i] += hi;
        let mut sp = states.to_vec();
        set_states_from_flat_beta(&mut sp, specs, &bp)?;
        refresh_all_block_etas(family, specs, &mut sp)?;
        let fp = penalized_objective_at_beta(family, specs, &sp, per_block_log_lambdas)?;

        let mut bm = beta_hat.clone();
        bm[i] -= hi;
        let mut sm = states.to_vec();
        set_states_from_flat_beta(&mut sm, specs, &bm)?;
        refresh_all_block_etas(family, specs, &mut sm)?;
        let fm = penalized_objective_at_beta(family, specs, &sm, per_block_log_lambdas)?;

        h[[i, i]] = ((fp - 2.0 * f0 + fm) / (hi * hi)).max(0.0);

        for j in 0..i {
            let hj = steps[j];
            let mut bpp = beta_hat.clone();
            bpp[i] += hi;
            bpp[j] += hj;
            let mut spp = states.to_vec();
            set_states_from_flat_beta(&mut spp, specs, &bpp)?;
            refresh_all_block_etas(family, specs, &mut spp)?;
            let fpp = penalized_objective_at_beta(family, specs, &spp, per_block_log_lambdas)?;

            let mut bpm = beta_hat.clone();
            bpm[i] += hi;
            bpm[j] -= hj;
            let mut spm = states.to_vec();
            set_states_from_flat_beta(&mut spm, specs, &bpm)?;
            refresh_all_block_etas(family, specs, &mut spm)?;
            let fpm = penalized_objective_at_beta(family, specs, &spm, per_block_log_lambdas)?;

            let mut bmp = beta_hat.clone();
            bmp[i] -= hi;
            bmp[j] += hj;
            let mut smp = states.to_vec();
            set_states_from_flat_beta(&mut smp, specs, &bmp)?;
            refresh_all_block_etas(family, specs, &mut smp)?;
            let fmp = penalized_objective_at_beta(family, specs, &smp, per_block_log_lambdas)?;

            let mut bmm = beta_hat.clone();
            bmm[i] -= hi;
            bmm[j] -= hj;
            let mut smm = states.to_vec();
            set_states_from_flat_beta(&mut smm, specs, &bmm)?;
            refresh_all_block_etas(family, specs, &mut smm)?;
            let fmm = penalized_objective_at_beta(family, specs, &smm, per_block_log_lambdas)?;

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

fn compute_joint_covariance<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Array2<f64>, String> {
    let mut h = if let Some(h_exact) = family.exact_newton_joint_hessian(states)? {
        let ranges = block_param_ranges(specs);
        let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
        if h_exact.nrows() != total || h_exact.ncols() != total {
            return Err(format!(
                "joint exact-newton Hessian shape mismatch in covariance: got {}x{}, expected {}x{}",
                h_exact.nrows(),
                h_exact.ncols(),
                total,
                total
            ));
        }
        let mut h = h_exact;
        for (b, spec) in specs.iter().enumerate() {
            let (start, end) = ranges[b];
            let lambdas = per_block_log_lambdas[b].mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((end - start, end - start));
            for (k, s) in spec.penalties.iter().enumerate() {
                s_lambda.scaled_add(lambdas[k], s);
            }
            h.slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, &s_lambda);
        }
        h
    } else {
        compute_joint_hessian_from_objective(family, specs, states, per_block_log_lambdas)?
    };
    let p_total = h.nrows();
    for i in 0..p_total {
        for j in 0..i {
            let v = 0.5 * (h[[i, j]] + h[[j, i]]);
            h[[i, j]] = v;
            h[[j, i]] = v;
        }
    }
    if use_exact_newton_strict_spd(family) {
        strict_inverse_spd(&h)
    } else {
        match inverse_spd_with_retry(&h, effective_solver_ridge(options.ridge_floor), 8) {
            Ok(cov) => Ok(cov),
            Err(_) => pinv_positive_part(&h, effective_solver_ridge(options.ridge_floor)),
        }
    }
}

pub fn fit_custom_family<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, CustomFamilyError> {
    let penalty_counts = validate_block_specs(specs)?;
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
        let covariance_conditional = if options.compute_covariance {
            compute_joint_covariance(
                family,
                specs,
                &inner.block_states,
                &vec![Array1::zeros(0); specs.len()],
                options,
            )
            .ok()
        } else {
            None
        };
        let reml_term = if options.use_reml_objective {
            0.5 * (inner.block_logdet_h - inner.block_logdet_s)
        } else {
            0.0
        };
        return Ok(BlockwiseFitResult {
            block_states: inner.block_states,
            log_likelihood: inner.log_likelihood,
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            covariance_conditional,
            penalized_objective: -inner.log_likelihood + inner.penalty_value + reml_term,
            outer_iterations: 0,
            outer_final_gradient_norm: 0.0,
            inner_cycles: inner.cycles,
            converged: inner.converged,
        });
    }

    let warm_cache = std::sync::Mutex::new(None::<ConstrainedWarmStart>);
    let last_outer_error = std::sync::Mutex::new(None::<String>);
    let lower = Array1::<f64>::from_elem(rho0.len(), -30.0);
    let upper = Array1::<f64>::from_elem(rho0.len(), 30.0);
    let mut last_eval: Option<(Array1<f64>, f64, Array1<f64>, Option<Array2<f64>>)> = None;
    let objective = CachedSecondOrderObjective::new(
        |x: &Array1<f64>| {
            let cached = warm_cache.lock().ok().and_then(|g| g.clone());
            let (obj, grad, hess_opt) = match outer_objective_gradient_hessian(
                family,
                specs,
                options,
                &penalty_counts,
                x,
                cached.as_ref(),
                true,
            ) {
                Ok((obj, grad, hess_opt, warm))
                    if obj.is_finite()
                        && grad.iter().all(|v| v.is_finite())
                        && hess_opt
                            .as_ref()
                            .map(|h| h.iter().all(|v| v.is_finite()))
                            .unwrap_or(true) =>
                {
                    if let Ok(mut guard) = warm_cache.lock() {
                        let seed_ok = cached
                            .as_ref()
                            .map(|c| {
                                c.rho.len() == x.len()
                                    && c.rho
                                        .iter()
                                        .zip(x.iter())
                                        .all(|(&a, &b)| (a - b).abs() <= 1.5)
                            })
                            .unwrap_or(true);
                        if seed_ok {
                            *guard = Some(warm);
                        } else {
                            *guard = None;
                        }
                    }
                    if let Ok(mut guard) = last_outer_error.lock() {
                        *guard = None;
                    }
                    (obj, grad, hess_opt)
                }
                Ok((_obj, _grad, _hess_opt, _warm)) => {
                    if let Ok(mut guard) = last_outer_error.lock() {
                        *guard = Some(
                            "custom-family outer objective/derivatives became non-finite"
                                .to_string(),
                        );
                    }
                    return Err(ObjectiveEvalError::recoverable(
                        "custom-family outer objective/derivatives became non-finite",
                    ));
                }
                Err(e) => {
                    if let Ok(mut guard) = last_outer_error.lock() {
                        *guard = Some(e);
                    }
                    return Err(ObjectiveEvalError::recoverable(
                        "custom-family outer objective/gradient evaluation failed",
                    ));
                }
            };
            last_eval = Some((x.clone(), obj, grad.clone(), hess_opt.clone()));
            Ok((obj, grad, hess_opt))
        },
        1e-4,
    );
    let mut solver = NewtonTrustRegion::new(rho0.clone(), objective)
        .with_bounds(
            Bounds::new(lower, upper, 1e-6).expect("custom-family rho bounds must be valid"),
        )
        .with_tolerance(
            Tolerance::new(options.outer_tol).expect("custom-family tolerance must be valid"),
        )
        .with_max_iterations(
            MaxIterations::new(options.outer_max_iter)
                .expect("custom-family max iterations must be valid"),
        );
    let last_eval_error = || {
        last_outer_error
            .lock()
            .ok()
            .and_then(|g| (*g).clone())
            .map(|e| format!(" last objective error: {e}"))
            .unwrap_or_default()
    };
    let sol = match solver.run() {
        Ok(sol) => sol,
        Err(NewtonTrustRegionError::MaxIterationsReached { last_solution }) => {
            if last_solution.final_value.is_finite()
                && last_solution.final_gradient_norm.is_finite()
            {
                log::warn!(
                    "Outer smoothing hit max iterations; using best-so-far solution (iter={}, f={:.6e}, ||g||={:.3e}).",
                    last_solution.iterations,
                    last_solution.final_value,
                    last_solution.final_gradient_norm
                );
                *last_solution
            } else {
                return Err(format!(
                    "outer smoothing optimization failed: MaxIterationsReached.{details}",
                    details = last_eval_error()
                )
                .into());
            }
        }
        Err(e) => {
            return Err(format!(
                "outer smoothing optimization failed: {e:?}.{details}",
                details = last_eval_error()
            )
            .into());
        }
    };

    let rho_star = sol.final_point;
    let per_block = split_log_lambdas(&rho_star, &penalty_counts)?;
    let final_seed = warm_cache.lock().ok().and_then(|g| g.clone());
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, final_seed.as_ref())
        .map_err(|e| {
            format!(
                "outer smoothing optimization failed during final inner refit: {e}.{details}",
                details = last_eval_error()
            )
        })?;
    refresh_all_block_etas(family, specs, &mut inner.block_states).map_err(|e| {
        format!(
            "outer smoothing optimization failed during final eta refresh: {e}.{details}",
            details = last_eval_error()
        )
    })?;
    let covariance_conditional = if options.compute_covariance {
        compute_joint_covariance(family, specs, &inner.block_states, &per_block, options).ok()
    } else {
        None
    };

    Ok(BlockwiseFitResult {
        block_states: inner.block_states,
        log_likelihood: inner.log_likelihood,
        log_lambdas: rho_star.clone(),
        lambdas: rho_star.mapv(f64::exp),
        covariance_conditional,
        penalized_objective: -inner.log_likelihood
            + inner.penalty_value
            + if include_exact_newton_logdet_h(family) {
                0.5 * inner.block_logdet_h
            } else {
                0.0
            }
            - if include_exact_newton_logdet_s(family, options) {
                0.5 * inner.block_logdet_s
            } else {
                0.0
            },
        outer_iterations: sol.iterations,
        outer_final_gradient_norm: sol.final_gradient_norm,
        inner_cycles: inner.cycles,
        converged: inner.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::gamlss::{
        BinomialLocationScaleProbitFamily, BinomialLocationScaleProbitWiggleFamily,
    };
    use crate::matrix::DesignMatrix;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, Array2, array};

    #[derive(Clone)]
    struct OneBlockIdentityFamily;

    impl CustomFamily for OneBlockIdentityFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = block_states[0].eta.len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                block_working_sets: vec![BlockWorkingSet::Diagonal {
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
                block_working_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: self.y.clone(),
                    working_weights: Array1::ones(self.y.len()),
                }],
            })
        }

        fn diagonal_working_weights_directional_derivative(
            &self,
            _block_states: &[ParameterBlockState],
            _block_idx: usize,
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
                block_working_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![g],
                    hessian: array![[1.0]],
                }],
            })
        }

        fn block_linear_constraints(
            &self,
            _block_states: &[ParameterBlockState],
            block_idx: usize,
            _spec: &ParameterBlockSpec,
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
    struct PreferJointExactFamily;

    impl CustomFamily for PreferJointExactFamily {
        fn evaluate(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                block_working_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: array![[2.0]],
                }],
            })
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            _block_states: &[ParameterBlockState],
            _block_idx: usize,
            _d_beta: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Err(
                "blockwise exact-newton path should not be used when joint path is available"
                    .to_string(),
            )
        }

        fn exact_newton_joint_hessian(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[2.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _block_states: &[ParameterBlockState],
            _d_beta_flat: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
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
                block_working_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![-2.0 * resid],
                    hessian: array![[2.0]],
                }],
            })
        }

        fn exact_newton_outer_objective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::PseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[2.0]]))
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            _block_states: &[ParameterBlockState],
            _block_idx: usize,
            _d_beta: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _block_states: &[ParameterBlockState],
            _d_beta_flat: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }
    }

    #[derive(Clone)]
    struct OneBlockExactPsiHookFamily;

    impl CustomFamily for OneBlockExactPsiHookFamily {
        fn evaluate(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                block_working_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: array![[1.0]],
                }],
            })
        }

        fn exact_newton_outer_objective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::PseudoLaplace
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            _block_states: &[ParameterBlockState],
            _block_idx: usize,
            _d_beta: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }

        fn exact_newton_block_psi_gradient(
            &self,
            _block_states: &[ParameterBlockState],
            _block_idx: usize,
            _ctx: ExactNewtonPsiGradientContext<'_>,
        ) -> Result<Option<f64>, String> {
            Ok(Some(3.5))
        }
    }

    #[derive(Clone)]
    struct OneBlockIndefinitePseudoLaplaceFamily;

    impl CustomFamily for OneBlockIndefinitePseudoLaplaceFamily {
        fn evaluate(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                block_working_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: array![[-1.0]],
                }],
            })
        }

        fn exact_newton_outer_objective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::PseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[-1.0]]))
        }
    }

    #[derive(Clone)]
    struct OneBlockAlwaysErrorFamily;

    impl CustomFamily for OneBlockAlwaysErrorFamily {
        fn evaluate(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            Err("synthetic outer objective failure: block[0] evaluate()".to_string())
        }
    }

    #[test]
    fn effective_ridge_is_never_below_solver_floor() {
        assert!((effective_solver_ridge(0.0) - 1e-15).abs() < 1e-30);
        assert!((effective_solver_ridge(1e-8) - 1e-8).abs() < 1e-20);
    }

    #[test]
    fn objective_includes_solver_ridge_quadratic_term() {
        // One-parameter block with X=1, y*=1, w=1, no explicit penalties.
        // Inner solve gives beta = 1 / (1 + ridge), so objective should include
        // 0.5 * ridge * beta^2 even when no smoothing penalties are present.
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            inner_tol: 0.0,
            outer_max_iter: 1,
            outer_tol: 1e-8,
            min_weight: 1e-12,
            ridge_floor: 1e-4,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_reml_objective: false,
            compute_covariance: false,
        };

        let result = fit_custom_family(&OneBlockIdentityFamily, &[spec], &options)
            .expect("custom family fit should succeed");
        let ridge = effective_solver_ridge(options.ridge_floor);
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
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![array![[1.0]]],
            initial_log_lambdas: array![10.0_f64.ln()],
            initial_beta: Some(array![1.0]),
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 20,
            inner_tol: 1e-10,
            outer_max_iter: 1,
            outer_tol: 1e-8,
            min_weight: 1e-12,
            ridge_floor: 0.0,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_reml_objective: false,
            compute_covariance: false,
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
    fn outer_gradient_matches_finite_difference_for_one_block() {
        let n = 8usize;
        let y = Array1::from_vec(vec![0.4, -0.2, 0.8, 1.0, -0.5, 0.3, 0.1, -0.7]);
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.2],
            initial_beta: None,
        };
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
            ridge_floor: 1e-10,
            ..BlockwiseFitOptions::default()
        };
        let penalty_counts = vec![1usize];
        let rho = array![0.1];
        let (f0, g0, _) = outer_objective_and_gradient(
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
        let (fp, _, _) = outer_objective_and_gradient(
            &OneBlockGaussianFamily { y: y.clone() },
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho_p,
            None,
        )
        .expect("objective+");
        let (fm, _, _) = outer_objective_and_gradient(
            &OneBlockGaussianFamily { y },
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho_m,
            None,
        )
        .expect("objective-");
        let g_fd = (fp - fm) / (2.0 * h);
        let rel = (g0[0] - g_fd).abs() / g_fd.abs().max(1e-8);

        assert!(f0.is_finite());
        assert_eq!(
            g0[0].signum(),
            g_fd.signum(),
            "outer gradient sign mismatch: analytic={} fd={}",
            g0[0],
            g_fd
        );
        assert!(
            rel < 5e-3,
            "outer gradient mismatch: analytic={} fd={} rel={}",
            g0[0],
            g_fd,
            rel
        );
    }

    #[test]
    fn outer_gradient_prefers_joint_exact_path_when_available() {
        let spec = ParameterBlockSpec {
            name: "joint_exact".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
            ridge_floor: 1e-10,
            ..BlockwiseFitOptions::default()
        };
        let penalty_counts = vec![1usize];
        let rho = array![0.0];

        let result = outer_objective_and_gradient(
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
    fn exact_newton_pseudo_laplace_objective_uses_logdet_h_without_logdet_s() {
        let spec = ParameterBlockSpec {
            name: "pseudo_laplace".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
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
    fn exact_newton_family_hook_can_supply_psi_gradient_without_quadratic_spsi() {
        let spec = ParameterBlockSpec {
            name: "psi_hook".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let deriv = CustomFamilyBlockPsiDerivative {
            penalty_index: 0,
            x_psi: Array2::zeros((1, 1)),
            s_psi: Array2::zeros((1, 1)),
            s_psi_components: None,
        };
        let result = evaluate_custom_family_joint_hyper(
            &OneBlockExactPsiHookFamily,
            &[spec],
            &BlockwiseFitOptions {
                use_reml_objective: true,
                compute_covariance: false,
                ..BlockwiseFitOptions::default()
            },
            &Array1::zeros(0),
            &[vec![deriv]],
            None,
            false,
        )
        .expect("joint hyper eval with exact psi hook");
        assert_eq!(result.gradient.len(), 1);
        assert!(
            (result.gradient[0] - 3.5).abs() < 1e-12,
            "expected family-supplied psi gradient, got {}",
            result.gradient[0]
        );
    }

    #[test]
    fn pseudo_laplace_exact_newton_rejects_non_spd_hessian() {
        let spec = ParameterBlockSpec {
            name: "indefinite".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let msg = match fit_custom_family(
            &OneBlockIndefinitePseudoLaplaceFamily,
            &[spec],
            &BlockwiseFitOptions {
                use_reml_objective: true,
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
    fn outer_laml_gradient_matches_finite_difference_when_joint_exact_path_is_active() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let log_sigma_design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let threshold_spec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: threshold_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
        };
        let log_sigma_spec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: log_sigma_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![-0.2],
            initial_beta: Some(array![-0.1]),
        };
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let knots = crate::families::gamlss::initialize_wiggle_knots_from_seed(q_seed.view(), 3, 4)
            .expect("knots");
        let wiggle_block = crate::families::gamlss::build_wiggle_block_input_from_knots(
            q_seed.view(),
            &knots,
            3,
            2,
            false,
        )
        .expect("wiggle block");
        let wiggle_spec = ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: wiggle_block.design.clone(),
            offset: wiggle_block.offset.clone(),
            penalties: wiggle_block.penalties.clone(),
            initial_log_lambdas: array![0.1],
            initial_beta: Some(Array1::from_elem(wiggle_block.design.ncols(), 0.03)),
        };

        let family = BinomialLocationScaleProbitWiggleFamily {
            y,
            weights,
            sigma_min: 0.05,
            sigma_max: 4.0,
            link_kind: crate::types::InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
            wiggle_knots: knots,
            wiggle_degree: 3,
        };

        let specs = vec![threshold_spec, log_sigma_spec, wiggle_spec];
        let penalty_counts = vec![1usize, 1usize, 1usize];
        let rho = array![0.05, -0.15, 0.1];
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (f0, g0, _) =
            outer_objective_and_gradient(&family, &specs, &options, &penalty_counts, &rho, None)
                .expect("objective/gradient");
        assert!(f0.is_finite());
        assert_eq!(g0.len(), rho.len());

        let h = 1e-5;
        for k in 0..rho.len() {
            let mut rho_p = rho.clone();
            let mut rho_m = rho.clone();
            rho_p[k] += h;
            rho_m[k] -= h;
            let (fp, _, _) = outer_objective_and_gradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_p,
                None,
            )
            .expect("objective+");
            let (fm, _, _) = outer_objective_and_gradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_m,
                None,
            )
            .expect("objective-");
            let g_fd = (fp - fm) / (2.0 * h);
            let rel = (g0[k] - g_fd).abs() / g_fd.abs().max(1e-8);
            assert_eq!(
                g0[k].signum(),
                g_fd.signum(),
                "outer LAML gradient sign mismatch at {}: analytic={} fd={}",
                k,
                g0[k],
                g_fd
            );
            assert!(
                rel < 2e-2,
                "outer LAML gradient mismatch at {}: analytic={} fd={} rel={}",
                k,
                g0[k],
                g_fd,
                rel
            );
        }
    }

    #[test]
    fn outer_laml_gradient_diagonal_binomial_location_scale_matches_fd() {
        let n = 9usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_elem(n, 1.0);
        let threshold_spec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
        };
        let log_sigma_spec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![-0.1]),
        };
        let family = BinomialLocationScaleProbitFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            sigma_min: 0.05,
            sigma_max: 4.0,
            threshold_design: Some(threshold_spec.design.clone()),
            log_sigma_design: Some(log_sigma_spec.design.clone()),
        };
        let specs = vec![threshold_spec, log_sigma_spec];
        let penalty_counts = vec![1usize, 1usize];
        let rho = array![0.15, -0.25];
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (f0, g0, _) =
            outer_objective_and_gradient(&family, &specs, &options, &penalty_counts, &rho, None)
                .expect("objective/gradient");
        assert!(f0.is_finite());
        assert_eq!(g0.len(), rho.len());

        let h = 1e-5;
        for k in 0..rho.len() {
            let mut rho_p = rho.clone();
            let mut rho_m = rho.clone();
            rho_p[k] += h;
            rho_m[k] -= h;
            let (fp, _, _) = outer_objective_and_gradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_p,
                None,
            )
            .expect("objective+");
            let (fm, _, _) = outer_objective_and_gradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_m,
                None,
            )
            .expect("objective-");
            let g_fd = (fp - fm) / (2.0 * h);
            let rel = (g0[k] - g_fd).abs() / g_fd.abs().max(1e-8);
            assert_eq!(
                g0[k].signum(),
                g_fd.signum(),
                "outer diagonal LAML gradient sign mismatch at {}: analytic={} fd={}",
                k,
                g0[k],
                g_fd
            );
            assert!(
                rel < 2e-2,
                "outer diagonal LAML gradient mismatch at {}: analytic={} fd={} rel={}",
                k,
                g0[k],
                g_fd,
                rel
            );
        }
    }

    #[test]
    fn outer_laml_hessian_joint_exact_binomial_location_scale_matches_fd() {
        let n = 10usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let weights = Array1::from_elem(n, 1.0);
        let threshold_spec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.15]),
        };
        let log_sigma_spec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![-0.05]),
        };
        let family = BinomialLocationScaleProbitFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            sigma_min: 0.05,
            sigma_max: 4.0,
            threshold_design: Some(threshold_spec.design.clone()),
            log_sigma_design: Some(log_sigma_spec.design.clone()),
        };
        let specs = vec![threshold_spec, log_sigma_spec];
        let penalty_counts = vec![1usize, 1usize];
        let rho = array![0.1, -0.2];
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (_f0, g0, h0_opt, _) = outer_objective_gradient_hessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho,
            None,
            true,
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
            let (_fp, gp, _, _) = outer_objective_gradient_hessian(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_p,
                None,
                false,
            )
            .expect("objective/gradient +");
            let (_fm, gm, _, _) = outer_objective_gradient_hessian(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_m,
                None,
                false,
            )
            .expect("objective/gradient -");

            for k in 0..rho.len() {
                let h_fd = (gp[k] - gm[k]) / (2.0 * h);
                let abs_err = (h0[[k, l]] - h_fd).abs();
                let rel = (h0[[k, l]] - h_fd).abs() / h_fd.abs().max(1e-7);
                assert_eq!(
                    h0[[k, l]].signum(),
                    h_fd.signum(),
                    "outer Hessian sign mismatch at ({k},{l}): analytic={} fd={}",
                    h0[[k, l]],
                    h_fd
                );
                assert!(
                    abs_err < 1e-8 || rel < 2e-2,
                    "outer Hessian mismatch at ({k},{l}): analytic={} fd={} abs={} rel={}",
                    h0[[k, l]],
                    h_fd,
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
        let _ = g0;
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

        let beta_dense = solve_block_weighted_system(
            &DesignMatrix::Dense(x_dense.clone()),
            &y_star,
            &w,
            &s_lambda,
            1e-12,
            RidgePolicy::explicit_stabilization_pospart(),
        )
        .expect("dense solve should succeed");

        let beta_sparse = solve_block_weighted_system(
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
    fn block_solve_falls_back_when_llt_rejects_indefinite_system() {
        let x_dense = array![[1.0, 0.0], [0.0, 0.0]];
        let y_star = array![2.0, 0.0];
        let w = array![1.0, 1.0];
        let s_lambda = array![[0.0, 0.0], [0.0, -1e-12]];

        let beta = solve_block_weighted_system(
            &DesignMatrix::Dense(x_dense),
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
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
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
    fn quadratic_linear_constraints_release_positive_kkt_system_multiplier() {
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
    fn outer_objective_failure_context_is_preserved() {
        // One penalty forces the outer rho optimizer to run, which should now preserve
        // the real evaluation error instead of returning an opaque line-search failure.
        let spec = ParameterBlockSpec {
            name: "err_block".to_string(),
            design: DesignMatrix::Dense(array![[1.0], [1.0]]),
            offset: array![0.0, 0.0],
            penalties: vec![Array2::eye(1)],
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
}
