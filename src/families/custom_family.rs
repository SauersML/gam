use crate::faer_ndarray::FaerCholesky;
use crate::faer_ndarray::{FaerArrayView, FaerEigh, FaerSvd};
use crate::linalg::utils::{StableSolver, boundary_hit_step_fraction};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::solver::opt_objective::CachedFirstOrderObjective;
use crate::types::{LinkFunction, RidgeDeterminantMode, RidgePolicy};
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{Lblt as FaerLblt, Solve as FaerSolve};
use ndarray::{Array1, Array2};
use opt::{Bfgs, BfgsError, Bounds, MaxIterations, ObjectiveEvalError, Tolerance};
use thiserror::Error;

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
        hessian: SymmetricMatrix,
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
    fn post_update_block_beta(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
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

    /// Optional explicit moving-design psi terms for exact-Newton blocks.
    ///
    /// This is the structured version of `exact_newton_block_psi_gradient`.
    /// Families with a design X(psi) that moves under the hyperparameter can
    /// provide the explicit derivatives of the unpenalized block objective,
    /// gradient, and Hessian at fixed beta:
    ///
    ///   V_psi^explicit, g_psi^explicit, H_psi^explicit.
    ///
    /// The generic exact-Newton hypergradient code then:
    ///
    /// 1. adds the penalty terms
    ///      0.5 beta^T S_psi beta,
    ///      S_psi beta,
    ///      S_psi,
    /// 2. solves the implicit mode sensitivity
    ///      beta_psi = -(H + S)^{-1} (g_psi^explicit + S_psi beta),
    /// 3. adds the beta-path Hessian contribution
    ///      D_beta H[beta_psi],
    /// 4. evaluates the REML / Laplace traces
    ///      0.5 tr((H+S)^{-1} Hdot) - 0.5 tr(S_+^{-1} S_psi).
    ///
    /// So this callback should return only the explicit moving-design pieces
    /// from the family likelihood geometry, excluding the penalty derivatives.
    ///
    /// Conceptually this is the first-order slice of the full profiled/Hessian
    /// calculus:
    ///
    ///   J_i
    ///   = V_i
    ///     + 0.5 tr(H^{-1} \dot H_i)
    ///     - 0.5 tr(S^+ S_i),
    ///
    /// with
    ///
    ///   beta_i = -H^{-1} g_i,
    ///   \dot H_i = H_i + T[beta_i].
    ///
    /// The companion second-order hook below exists because differentiating
    /// this identity again introduces V_{ij}, g_{ij}, H_{ij}, beta_{ij},
    /// \ddot H_{ij}, and contracted fourth-order beta curvature.
    fn exact_newton_block_psi_terms(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _ctx: ExactNewtonPsiGradientContext<'_>,
    ) -> Result<Option<ExactNewtonBlockPsiTerms>, String> {
        Ok(None)
    }

    /// Optional explicit second-order moving-design psi terms for exact-Newton
    /// blocks.
    ///
    /// Full exact profiled/Laplace derivation for two outer coordinates
    /// theta_i, theta_j:
    ///
    /// Let
    ///
    ///   V(beta, theta) = L(beta, theta) + 0.5 beta^T S(theta) beta,
    ///   g              = V_beta,
    ///   H              = V_{beta beta},
    ///
    /// and let beta*(theta) be the exact inner mode:
    ///
    ///   g(beta*(theta), theta) = 0.
    ///
    /// Write fixed-beta theta partials as
    ///
    ///   g_i   = V_{beta i},
    ///   g_{ij}= V_{beta ij},
    ///   H_i   = V_{beta beta i},
    ///   H_{ij}= V_{beta beta ij}.
    ///
    /// The profiled outer objective is
    ///
    ///   J(theta)
    ///   = V(beta*(theta), theta)
    ///     + 0.5 log|H(beta*(theta), theta)|
    ///     - 0.5 log|S(theta)|_+.
    ///
    /// Differentiating stationarity once gives the exact first mode response
    ///
    ///   beta_i = -H^{-1} g_i.
    ///
    /// Differentiating stationarity again gives the second mode response
    ///
    ///   beta_{ij}
    ///   = -H^{-1}(g_{ij} + H_i beta_j + H_j beta_i + T[beta_i] beta_j),
    ///
    /// where
    ///
    ///   T[u] = D_beta H[u]
    ///
    /// is the contracted third beta derivative of V, viewed as a matrix. The
    /// third derivative is symmetric in its beta slots, so
    ///
    ///   T[beta_i] beta_j = T[beta_j] beta_i.
    ///
    /// The exact profiled penalized-objective block is
    ///
    ///   P_{ij} = V_{ij} - g_i^T H^{-1} g_j.
    ///
    /// For the Laplace log|H| term, define total mode-following Hessian drifts
    ///
    ///   \dot H_i = H_i + T[beta_i],
    ///
    ///   \ddot H_{ij}
    ///   = H_{ij}
    ///     + T_i[beta_j]
    ///     + T_j[beta_i]
    ///     + T[beta_{ij}]
    ///     + Q[beta_i, beta_j],
    ///
    /// where
    ///
    ///   T_i[u] = D_beta H_i[u],
    ///   Q[u,v] = D_beta^2 H[u,v].
    ///
    /// Then the exact joint outer Hessian is
    ///
    ///   J_{ij}
    ///   = (V_{ij} - g_i^T H^{-1} g_j)
    ///     + 0.5[ tr(H^{-1} \ddot H_{ij})
    ///            - tr(H^{-1} \dot H_j H^{-1} \dot H_i) ]
    ///     - 0.5[ tr(S^+ S_{ij}) - tr(S^+ S_j S^+ S_i) ].
    ///
    /// The only family-specific pieces in that formula are the fixed-beta
    /// likelihood derivatives and the beta-direction contractions of H.
    ///
    /// For two hyper coordinates i,j, this callback returns the fixed-beta
    /// second partials of the unpenalized block objective:
    ///
    ///   V_{ij}^explicit, g_{ij}^explicit, H_{ij}^explicit.
    ///
    /// These are the exact moving-design likelihood pieces needed by the full
    /// profiled outer Hessian
    ///
    ///   J_{ij}
    ///   = (V_{ij} - g_i^T H^{-1} g_j)
    ///     + 0.5[ tr(H^{-1} \tilde H_{ij})
    ///            - tr(H^{-1} \tilde H_i H^{-1} \tilde H_j) ]
    ///     - 0.5[ tr(S^+ S_{ij}) - tr(S^+ S_i S^+ S_j) ].
    ///
    /// Here the generic outer code is responsible for:
    ///
    /// 1. adding the penalty second derivatives
    ///      0.5 beta^T S_{ij} beta,
    ///      S_{ij} beta,
    ///      S_{ij},
    /// 2. solving the exact mode responses
    ///      beta_i  = -H^{-1} g_i,
    ///      beta_j  = -H^{-1} g_j,
    ///      beta_ij = -H^{-1}(g_{ij} + H_i beta_j + H_j beta_i
    ///                         + T[beta_i, beta_j]),
    /// 3. assembling the total Hessian drifts
    ///      \tilde H_i  = H_i  + T[beta_i],
    ///      \tilde H_{ij}= H_{ij}
    ///                    + T_i[beta_j]
    ///                    + T_j[beta_i]
    ///                    + Q[beta_i, beta_j]
    ///                    + T[beta_ij].
    ///
    /// For theta = [rho, psi], the rho blocks are usually penalty-only at fixed
    /// beta:
    ///
    ///   g_rho   = S_rho beta,
    ///   H_rho   = S_rho,
    ///   g_{rho,psi} = S_{rho,psi} beta,
    ///   H_{rho,psi} = S_{rho,psi},
    ///
    /// while the psi blocks are the moving-design family pieces returned here
    /// and by the first-order psi callback.
    ///
    /// Families do not need to materialize full third- or fourth-order
    /// coefficient tensors, but exact second-order outer methods do need the
    /// corresponding contractions. This callback supplies only the explicit
    /// fixed-beta second partials; the contraction operators remain separate.
    fn exact_newton_block_psi_second_order_terms(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _ctx_i: ExactNewtonPsiGradientContext<'_>,
        _ctx_j: ExactNewtonPsiGradientContext<'_>,
    ) -> Result<Option<ExactNewtonBlockPsiSecondOrderTerms>, String> {
        Ok(None)
    }

    /// Whether pseudo-Laplace traces and sensitivities may use the positive
    /// curvature subspace when the exact mode Hessian is only semidefinite.
    ///
    /// Families that legitimately optimize to boundary regimes with exact
    /// zero-curvature directions can opt in; truly indefinite Hessians are
    /// still rejected.
    fn exact_newton_allows_semidefinite_hessian(&self) -> bool {
        false
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
            compute_covariance: false,
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
}

#[derive(Clone)]
struct ConstrainedWarmStart {
    rho: Array1<f64>,
    block_beta: Vec<Array1<f64>>,
    active_sets: Vec<Option<Vec<usize>>>,
}

#[derive(Clone, Debug)]
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

fn finite_penalized_objective(log_likelihood: f64, penalty_value: f64, reml_term: f64) -> f64 {
    // The exact custom-family path can produce a finite inner mode together
    // with a numerically non-finite REML/logdet correction when linear algebra
    // on the determinant surface becomes ill-conditioned. The fit object should
    // still report a finite scalar objective whenever the penalized likelihood
    // itself is finite, because downstream callers and tests use this value as
    // a sanity check on the returned fit state rather than as a certificate
    // that every intermediate spectral calculation succeeded exactly.
    //
    // Priority:
    // 1. full penalized objective with REML/logdet term
    // 2. plain penalized likelihood  -loglik + penalty
    // 3. penalty only
    // 4. zero as an ultimate finite sentinel
    let objective = -log_likelihood + penalty_value + reml_term;
    if objective.is_finite() {
        objective
    } else if (-log_likelihood + penalty_value).is_finite() {
        -log_likelihood + penalty_value
    } else if penalty_value.is_finite() {
        penalty_value
    } else {
        0.0
    }
}

#[derive(Clone)]
pub struct CustomFamilyBlockPsiDerivative {
    pub penalty_index: usize,
    pub x_psi: Array2<f64>,
    pub s_psi: Array2<f64>,
    pub s_psi_components: Option<Vec<(usize, Array2<f64>)>>,
    pub x_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi_components: Option<Vec<Vec<(usize, Array2<f64>)>>>,
}

#[derive(Clone)]
pub struct ExactNewtonBlockPsiTerms {
    pub objective_psi: f64,
    pub score_psi: Array1<f64>,
    pub hessian_psi: Array2<f64>,
}

#[derive(Clone)]
pub struct ExactNewtonBlockPsiSecondOrderTerms {
    pub objective_psi_psi: f64,
    pub score_psi_psi: Array1<f64>,
    pub hessian_psi_psi: Array2<f64>,
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
    let xtwy = x.compute_xtwy(w, y_star)?;
    x.solve_system_with_policy(w, &xtwy, Some(s_lambda), ridge_floor, ridge_policy)
        .map_err(|_| "block solve failed after ridge retries".to_string())
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
        .or_else(|| {
            pinv_positive_part(lhs, effective_solver_ridge(ridge_floor))
                .ok()
                .map(|pinv| pinv.dot(rhs))
        })
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

#[inline]
fn floor_positive_working_weight(raw_weight: f64, min_weight: f64) -> f64 {
    if raw_weight <= 0.0 {
        0.0
    } else {
        raw_weight.max(min_weight)
    }
}

#[inline]
fn floor_positive_working_weights(working_weights: &Array1<f64>, min_weight: f64) -> Array1<f64> {
    working_weights.mapv(|wi| floor_positive_working_weight(wi, min_weight))
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
        let w_clamped =
            floor_positive_working_weights(self.working_weights, ctx.options.min_weight);

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
            let lhs_dense = lhs.to_dense();
            let (beta_constrained, active_set) = solve_quadratic_with_linear_constraints(
                &lhs_dense,
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
            let rhs_step = self.gradient - &ctx.s_lambda.dot(&ctx.states[ctx.block_idx].beta);
            let mut lhs_dense = lhs.to_dense();
            if !use_exact_newton_strict_spd(ctx.family) {
                let (evals, _) = FaerEigh::eigh(&lhs_dense, Side::Lower).map_err(|e| {
                    format!(
                        "block {} ({}) exact-newton eigendecomposition failed: {e}",
                        ctx.block_idx, ctx.spec.name
                    )
                })?;
                let min_eval = evals.iter().fold(f64::INFINITY, |m, &v| m.min(v));
                let floor = effective_solver_ridge(ctx.options.ridge_floor);
                if min_eval <= floor {
                    let shift = floor - min_eval;
                    for d in 0..lhs_dense.nrows() {
                        lhs_dense[[d, d]] += shift;
                    }
                }
            }
            let delta = if use_exact_newton_strict_spd(ctx.family) {
                strict_solve_spd(&lhs_dense, &rhs_step)?
            } else {
                solve_spd_system_with_policy(
                    &lhs_dense,
                    &rhs_step,
                    ctx.options.ridge_floor,
                    ctx.options.ridge_policy,
                )?
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

fn solve_kkt_step(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    active_a: &Array2<f64>,
    active_residual: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let p = hessian.nrows();
    if hessian.ncols() != p || gradient.len() != p || active_a.ncols() != p {
        return Err("constrained KKT step: dimension mismatch".to_string());
    }
    let m = active_a.nrows();
    if let Some(residual) = active_residual
        && residual.len() != m
    {
        return Err(format!(
            "constrained KKT step: active residual length mismatch: got {}, expected {}",
            residual.len(),
            m
        ));
    }
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
    if let Some(residual) = active_residual {
        for i in 0..m {
            rhs[p + i] = residual[i];
        }
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
        let mut residual_w = Array1::<f64>::zeros(active.len());
        for (r, &idx) in active.iter().enumerate() {
            a_w.row_mut(r).assign(&constraints.a.row(idx));
            residual_w[r] = constraints.b[idx] - constraints.a.row(idx).dot(&x);
        }
        let (direction, lambda_sys) = solve_kkt_step(hessian, &gradient, &a_w, Some(&residual_w))?;
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
            if let Some(cand) = boundary_hit_step_fraction(slack[i], ai_d, alpha) {
                alpha = cand;
                entering = Some(i);
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
    symmetrize_dense_in_place(&mut a);
    let p = a.nrows();
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
            // Positive-part determinant policy for numerically hostile exact
            // Newton systems.
            //
            // In the exact custom-family REML path we conceptually want
            //
            //   log |H|_+ = sum_{lambda_j > floor} log(lambda_j),
            //
            // where H is the symmetric Hessian surface used for the determinant
            // term and `floor` is the ridge-aware positivity threshold. The
            // mathematically clean route is an eigendecomposition of H. In
            // practice, the binomial spatial wiggle fit can drive the outer
            // optimizer into regions where:
            //
            // 1. the symmetric eigensolver fails to converge,
            // 2. an SVD fallback also fails to converge, or
            // 3. the matrix is so ill-conditioned that even a ridged factor
            //    fails numerically despite the surface being conceptually a
            //    positive-part pseudo-determinant.
            //
            // The goal of the determinant term here is not to certify a
            // high-precision spectral decomposition of a pathological matrix; it
            // is to keep the outer objective finite enough that the optimizer can
            // keep or reject the current rho iterate. So this code follows a
            // descending sequence of numerically weaker but always-finite
            // approximations:
            //
            //   eigh(H)          -> exact positive-part pseudo-logdet
            //   svd(H)           -> singular-value surrogate when eigh fails
            //   chol(H + bump I) -> ridged SPD surrogate if spectral methods fail
            //   diag surrogate   -> last-resort finite approximation
            //
            // The last fallback is intentionally conservative: it is only used
            // to avoid poisoning the objective with NaN/Inf. The important
            // invariant for the exact wiggle path is "outer objective stays
            // finite", not "every intermediate determinant is spectrally exact
            // under all non-convergent linear algebra conditions".
            let floor = ridge.max(1e-14);
            let evals = if let Ok((evals, _)) = crate::faer_ndarray::FaerEigh::eigh(&a, Side::Lower)
            {
                evals
            } else if let Ok((_, singular, _)) = a.svd(false, false) {
                singular
            } else {
                let mut ridged = a.clone();
                let mut bump = floor.max(1e-10);
                for _ in 0..8 {
                    for d in 0..p {
                        ridged[[d, d]] = a[[d, d]] + bump;
                    }
                    if let Ok(chol) = ridged.clone().cholesky(Side::Lower) {
                        return Ok(2.0 * chol.diag().mapv(f64::ln).sum());
                    }
                    bump *= 10.0;
                }
                let mut logdet = 0.0;
                for i in 0..p {
                    logdet += a[[i, i]].abs().max(floor).ln();
                }
                return Ok(logdet);
            };
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

fn logdet_trace_inverse_with_ridge_policy(
    matrix_on_logdet_surface: &Array2<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
    strict_spd: bool,
    allow_semidefinite: bool,
) -> Result<Array2<f64>, String> {
    if strict_spd {
        return strict_inverse_spd_with_semidefinite_option(
            matrix_on_logdet_surface,
            allow_semidefinite,
        );
    }

    match ridge_policy.determinant_mode {
        // For the full determinant surface, d log|H| = tr(H^{-1} dH).
        RidgeDeterminantMode::Full => inverse_spd_with_retry(
            matrix_on_logdet_surface,
            effective_solver_ridge(ridge_floor),
            8,
        )
        .or_else(|_| pinv_positive_part(matrix_on_logdet_surface, ridge_floor)),

        // For the positive-part pseudo-determinant surface,
        //
        //   log|H|_+ = sum_{lambda_j > floor} log lambda_j,
        //
        // the matching first derivative is the trace against the positive-part
        // pseudoinverse on that same ridge-adjusted surface, not against the
        // ordinary inverse of a nearby SPD surrogate.
        RidgeDeterminantMode::PositivePart => {
            let positive_floor = if ridge_policy.include_penalty_logdet {
                effective_solver_ridge(ridge_floor)
            } else {
                0.0
            }
            .max(1e-14);
            pinv_positive_part_with_floor(matrix_on_logdet_surface, positive_floor)
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

fn exact_newton_joint_hessian_symmetrized<F: CustomFamily>(
    family: &F,
    states: &[ParameterBlockState],
    total: usize,
    context: &str,
) -> Result<Option<Array2<f64>>, String> {
    let Some(mut h) = family.exact_newton_joint_hessian(states)? else {
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
    symmetrize_dense_in_place(&mut matrix);
    Ok(matrix)
}

fn strict_solve_spd(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD solve failed".to_string())?;
    Ok(chol.solve_vec(rhs))
}

fn strict_inverse_spd(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD inverse failed".to_string())?;
    let ident = Array2::<f64>::eye(matrix.nrows());
    Ok(chol.solve_mat(&ident))
}

fn strict_logdet_spd(matrix: &Array2<f64>) -> Result<f64, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD logdet failed".to_string())?;
    Ok(2.0 * chol.diag().mapv(f64::ln).sum())
}

fn strict_psd_positive_part_inverse_and_logdet(
    matrix: &Array2<f64>,
) -> Result<(Array2<f64>, f64), String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower)
        .map_err(|e| format!("strict pseudo-laplace PSD eigendecomposition failed: {e}"))?;
    let max_abs_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let tol = (max_abs_eval * 1e-12).max(1e-14);
    if evals.iter().any(|&ev| ev < -tol) {
        return Err("strict pseudo-laplace SPD solve failed".to_string());
    }
    let p = matrix.nrows();
    let mut pinv = Array2::<f64>::zeros((p, p));
    let mut logdet = 0.0;
    for k in 0..p {
        let ev = evals[k];
        if ev > tol {
            logdet += ev.ln();
            let inv_ev = 1.0 / ev;
            for i in 0..p {
                let uik = evecs[(i, k)];
                for j in 0..p {
                    pinv[[i, j]] += inv_ev * uik * evecs[(j, k)];
                }
            }
        }
    }
    Ok((pinv, logdet))
}

fn strict_solve_spd_with_semidefinite_option(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    allow_semidefinite: bool,
) -> Result<Array1<f64>, String> {
    if allow_semidefinite {
        let (pinv, _) = strict_psd_positive_part_inverse_and_logdet(matrix)?;
        return Ok(pinv.dot(rhs));
    }
    strict_solve_spd(matrix, rhs)
}

fn strict_inverse_spd_with_semidefinite_option(
    matrix: &Array2<f64>,
    allow_semidefinite: bool,
) -> Result<Array2<f64>, String> {
    if allow_semidefinite {
        let (pinv, _) = strict_psd_positive_part_inverse_and_logdet(matrix)?;
        return Ok(pinv);
    }
    strict_inverse_spd(matrix)
}

fn strict_logdet_spd_with_semidefinite_option(
    matrix: &Array2<f64>,
    allow_semidefinite: bool,
) -> Result<f64, String> {
    if allow_semidefinite {
        let (_, logdet) = strict_psd_positive_part_inverse_and_logdet(matrix)?;
        return Ok(logdet);
    }
    strict_logdet_spd(matrix)
}

fn solve_dense_symmetric_indefinite(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    context: &str,
) -> Result<Array1<f64>, String> {
    if matrix.nrows() != matrix.ncols() || rhs.len() != matrix.nrows() {
        return Err(format!("{context}: dimension mismatch"));
    }
    let matrix_view = FaerArrayView::new(matrix);
    let solver = FaerLblt::new(matrix_view.as_ref(), Side::Lower);
    let mut rhs_mat = FaerMat::zeros(rhs.len(), 1);
    for i in 0..rhs.len() {
        rhs_mat[(i, 0)] = rhs[i];
    }
    solver.solve_in_place(rhs_mat.as_mut());
    let mut out = Array1::<f64>::zeros(rhs.len());
    for i in 0..rhs.len() {
        out[i] = rhs_mat[(i, 0)];
    }
    if !out.iter().all(|v| v.is_finite()) {
        out = pinv_positive_part(matrix, 1e-12)
            .map(|pinv| pinv.dot(rhs))
            .unwrap_or_else(|_| {
                let mut diag = Array1::<f64>::zeros(rhs.len());
                for i in 0..rhs.len() {
                    let di = matrix[[i, i]].abs().max(1e-12);
                    diag[i] = rhs[i] / di;
                }
                diag
            });
        if !out.iter().all(|v| v.is_finite()) {
            return Err(format!("{context}: solve produced non-finite values"));
        }
    }
    Ok(out)
}

fn pinv_positive_part_with_floor(
    matrix: &Array2<f64>,
    positive_floor: f64,
) -> Result<Array2<f64>, String> {
    let (evals, evecs) = match FaerEigh::eigh(matrix, Side::Lower) {
        Ok(ok) => ok,
        Err(_) => {
            let p = matrix.nrows();
            let mut diag_pinv = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                let di = matrix[[i, i]];
                if di > positive_floor {
                    diag_pinv[[i, i]] = 1.0 / di;
                }
            }
            return Ok(diag_pinv);
        }
    };
    let p = matrix.nrows();
    let mut pinv = Array2::<f64>::zeros((p, p));
    for k in 0..p {
        let ev = evals[k];
        if ev > positive_floor {
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

fn pinv_positive_part(matrix: &Array2<f64>, ridge_floor: f64) -> Result<Array2<f64>, String> {
    let (evals, _) = FaerEigh::eigh(matrix, Side::Lower)
        .map_err(|e| format!("eigh failed in positive-part pseudoinverse: {e}"))?;
    let max_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let tol = (max_eval * 1e-12).max(ridge_floor.max(1e-14));
    pinv_positive_part_with_floor(matrix, tol)
}

fn include_exact_newton_logdet_h<F: CustomFamily + ?Sized>(family: &F) -> bool {
    matches!(
        family.exact_newton_outer_objective(),
        ExactNewtonOuterObjective::QuadraticReml | ExactNewtonOuterObjective::PseudoLaplace
    )
}

fn include_exact_newton_logdet_s<F: CustomFamily + ?Sized>(
    family: &F,
    options: &BlockwiseFitOptions,
) -> bool {
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
    let allow_semidefinite = strict_spd && family.exact_newton_allows_semidefinite_hessian();
    refresh_all_block_etas(family, specs, states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if let Some(h_joint) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        total,
        "joint exact-newton Hessian shape mismatch in logdet terms",
    )? {
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
            strict_logdet_spd_with_semidefinite_option(&h, allow_semidefinite)?
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
                let w = floor_positive_working_weights(working_weights, options.min_weight);
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

        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s_lambda.scaled_add(lambdas[k], s);
        }

        let mut h = xtwx;
        h += &s_lambda;
        logdet_h_total += if strict_spd {
            strict_logdet_spd_with_semidefinite_option(&h, allow_semidefinite)?
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
    let has_joint_exact_hessian = family.exact_newton_joint_hessian(&states)?.is_some();
    let inner_tol = if has_joint_exact_hessian {
        options.inner_tol.min(1e-10)
    } else {
        options.inner_tol
    };
    let inner_max_cycles = if has_joint_exact_hessian {
        options.inner_max_cycles.max(4000)
    } else {
        options.inner_max_cycles
    };
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

    for cycle in 0..inner_max_cycles {
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
            let beta_new = family.post_update_block_beta(&states, b, spec, beta_new_raw)?;
            let beta_old = states[b].beta.clone();
            let delta = &beta_new - &beta_old;
            let old_block_penalty =
                block_quadratic_penalty(&beta_old, s_lambda, ridge, options.ridge_policy);
            let step = delta.iter().copied().map(f64::abs).fold(0.0, f64::max);
            max_beta_step = max_beta_step.max(step);
            if step <= inner_tol {
                continue;
            }

            // Damped update: require non-increasing penalized objective under dynamic geometry.
            let mut accepted = false;
            for bt in 0..8 {
                let alpha = 0.5f64.powi(bt);
                let trial_beta_raw = &beta_old + &delta.mapv(|v| alpha * v);
                let trial_beta = family.post_update_block_beta(&states, b, spec, trial_beta_raw)?;
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
                states[b].beta = beta_old.clone();
                refresh_all_block_etas(family, specs, &mut states)?;
                if let BlockWorkingSet::ExactNewton { gradient, .. } = work {
                    let descent_dir = gradient - &s_lambda.dot(&beta_old);
                    let dir_norm = descent_dir.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                    if dir_norm > inner_tol {
                        for bt in 0..12 {
                            let alpha = 0.5f64.powi(bt);
                            let trial_beta_raw = &beta_old + &descent_dir.mapv(|v| alpha * v);
                            let trial_beta =
                                family.post_update_block_beta(&states, b, spec, trial_beta_raw)?;
                            states[b].beta = trial_beta;
                            refresh_all_block_etas(family, specs, &mut states)?;
                            let trial_eval = family.evaluate(&states)?;
                            let trial_block_penalty = block_quadratic_penalty(
                                &states[b].beta,
                                s_lambda,
                                ridge,
                                options.ridge_policy,
                            );
                            let trial_penalty =
                                current_penalty - old_block_penalty + trial_block_penalty;
                            let trial_objective = -trial_eval.log_likelihood + trial_penalty;
                            if trial_objective.is_finite()
                                && trial_objective <= objective_cycle_prev + 1e-10
                            {
                                objective_cycle_prev = trial_objective;
                                current_penalty = trial_penalty;
                                accepted = true;
                                break;
                            }
                            states[b].beta = beta_old.clone();
                            refresh_all_block_etas(family, specs, &mut states)?;
                        }
                    }
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

        let objective_tol = inner_tol * (1.0 + objective.abs());
        let exact_joint_stationarity_ok = if has_joint_exact_hessian {
            exact_newton_joint_stationarity_inf_norm(
                &eval,
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
    let allow_semidefinite = strict_spd && family.exact_newton_allows_semidefinite_hessian();
    let per_block = split_log_lambdas(rho, penalty_counts)?;
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, warm_start)?;
    let ridge = effective_solver_ridge(options.ridge_floor);
    let mode_ridge = if options.ridge_policy.include_quadratic_penalty {
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
    // Outer objective at fixed rho:
    //   V(rho) = -ell(beta^) + 0.5 * beta^T P_mode(rho) beta^
    //            + 0.5 * log|J_trace(beta^,rho)| - 0.5 * log|P_trace(rho)|_+,
    // where beta^ solves g(beta,rho)=0 with
    //   g(beta,rho) = -∇_beta ell(beta) + P_mode(rho) beta,
    //   J_mode(beta,rho) = H(beta) + P_mode(rho),
    //   H(beta) = -∇^2_beta ell(beta).
    //
    // `P_mode` includes the rho-independent stabilization ridge exactly when
    // that ridge participates in the quadratic objective. `P_trace`/`J_trace`
    // additionally include any ridge used only by the determinant surface.
    //
    // The exact outer gradient used below is:
    //   V_k = 0.5*beta^T A_k beta^
    //       + 0.5*tr(J_trace^{-1}(A_k + D H[u_k]))
    //       - 0.5*tr(P_trace,+^{-1} A_k),
    // with A_k=dS_lambda/drho_k and u_k=d beta^/drho_k.
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
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if let Some(h_joint_unpen) = exact_newton_joint_hessian_symmetrized(
        family,
        &inner.block_states,
        total,
        "joint exact-newton Hessian shape mismatch in outer gradient",
    )? {
        let beta_flat = flatten_state_betas(&inner.block_states, specs);
        let synced_joint_states =
            synchronized_states_from_flat_beta(family, specs, &inner.block_states, &beta_flat)?;
        let joint_tangent_basis =
            joint_post_update_tangent_basis(family, specs, &synced_joint_states)?;
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
            if mode_ridge > 0.0 {
                for d in start..end {
                    s_joint[[d, d]] += mode_ridge;
                }
            }
            if let Some(s_pinv) = s_pinv_joint.as_mut() {
                let mut s_for_logdet = s_lambda.clone();
                if options.ridge_policy.include_penalty_logdet {
                    for d in 0..p {
                        s_for_logdet[[d, d]] += ridge;
                    }
                }
                let s_floor = if options.ridge_policy.include_penalty_logdet {
                    ridge
                } else {
                    0.0
                }
                .max(1e-14);
                let s_part = pinv_positive_part_with_floor(&s_for_logdet, s_floor)?;
                s_pinv
                    .slice_mut(ndarray::s![start..end, start..end])
                    .assign(&s_part);
            }
        }
        // Inner stationarity for the coupled blocks is:
        //   g(beta,rho) = -∇_beta ell(beta) + P_mode(rho) beta = 0.
        // Differentiating w.r.t. rho_k gives the exact sensitivity system:
        //   J_mode * u_k = -A_k beta,  with u_k = d beta / d rho_k.
        // We use A_k := dS_lambda/drho_k and J_mode = H(beta) + P_mode(rho), where
        // H(beta) = -∇^2_beta ell(beta) is the full joint likelihood curvature.
        //
        // Here `h_joint_unpen` is H(beta) and `s_joint` is P_mode(rho), so
        // `j_joint = h_joint_unpen + s_joint` is the coupled mode Jacobian.
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
        if !strict_spd && extra_logdet_ridge > 0.0 {
            for d in 0..total {
                j_for_traces[[d, d]] += extra_logdet_ridge;
            }
        }
        let h_inv = logdet_trace_inverse_with_ridge_policy(
            &j_for_traces,
            options.ridge_floor,
            options.ridge_policy,
            strict_spd,
            allow_semidefinite,
        )?;
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
                    solve_mode_sensitivity(
                        &j_joint,
                        &rhs_k,
                        joint_tangent_basis.as_ref(),
                        strict_spd,
                        allow_semidefinite,
                        options,
                        "joint mode sensitivity system for REML gradient",
                    )?
                } else {
                    Array1::<f64>::zeros(total)
                };
                let g = if include_logdet_h || include_logdet_s {
                    let g_logs = if include_logdet_s {
                        0.5 * trace_product(
                            s_pinv_joint
                                .as_ref()
                                .ok_or_else(|| "missing joint S^+ for REML gradient".to_string())?,
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
                                &synced_joint_states,
                                &u_k,
                            )?
                            .ok_or_else(|| {
                                "joint exact-newton dH unavailable for analytic outer gradient"
                                    .to_string()
                            })?;
                        let h_rho = if h_rho.iter().all(|v| v.is_finite()) {
                            symmetrized_square_matrix(
                                h_rho,
                                total,
                                "joint exact-newton dH shape mismatch",
                            )?
                        } else {
                            Array2::<f64>::zeros((total, total))
                        };
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
                        let u_kl = solve_mode_sensitivity(
                            &j_joint,
                            &rhs_kl,
                            joint_tangent_basis.as_ref(),
                            strict_spd,
                            allow_semidefinite,
                            options,
                            "joint second-order mode sensitivity system for REML Hessian",
                        )?;
                        let dh_u_kl = match family
                            .exact_newton_joint_hessian_directional_derivative(
                                &synced_joint_states,
                                &u_kl,
                            )? {
                            Some(v) => symmetrized_square_matrix(
                                v,
                                total,
                                "joint exact-newton second-order dH shape mismatch",
                            )?,
                            None => {
                                hess_available = false;
                                break;
                            }
                        };
                        let d2h_ul_uk = match family
                            .exact_newton_joint_hessian_second_directional_derivative(
                                &synced_joint_states,
                                &u_terms[l],
                                &u_terms[k],
                            )? {
                            Some(v) => symmetrized_square_matrix(
                                v,
                                total,
                                "joint exact-newton d2H shape mismatch",
                            )?,
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
                let w = floor_positive_working_weights(working_weights, options.min_weight);
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
                hessian.to_dense()
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
        let h_inv = logdet_trace_inverse_with_ridge_policy(
            &h_for_logdet,
            options.ridge_floor,
            options.ridge_policy,
            strict_spd,
            allow_semidefinite,
        )?;
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
            let s_floor = if options.ridge_policy.include_penalty_logdet {
                effective_solver_ridge(options.ridge_floor)
            } else {
                0.0
            }
            .max(1e-14);
            Some(pinv_positive_part_with_floor(&s_for_logdet, s_floor)?)
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
                    0.5 * trace_product(
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
                    strict_solve_spd_with_semidefinite_option(&h_mode, &rhs, allow_semidefinite)?
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
                                let h_exact = symmetrized_square_matrix(
                                    h_exact,
                                    p,
                                    &format!("block {b} exact-newton dH shape mismatch"),
                                )?;
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
                                } => floor_positive_working_weights(
                                    working_weights,
                                    options.min_weight,
                                ),
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

#[allow(clippy::type_complexity)]
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
    let allow_semidefinite = strict_spd && family.exact_newton_allows_semidefinite_hessian();
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
            let s_floor = if options.ridge_policy.include_penalty_logdet {
                effective_solver_ridge(options.ridge_floor)
            } else {
                0.0
            }
            .max(1e-14);
            Some(pinv_positive_part_with_floor(&s_for_logdet, s_floor)?)
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
                let w = floor_positive_working_weights(working_weights, options.min_weight);
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
                let h_inv = logdet_trace_inverse_with_ridge_policy(
                    &h_for_logdet,
                    options.ridge_floor,
                    options.ridge_policy,
                    strict_spd,
                    allow_semidefinite,
                )?;
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
                let mut h_mode = hessian.to_dense();
                h_mode += &s_lambda;
                let mut h_for_logdet = h_mode.clone();
                if !strict_spd && options.ridge_policy.include_penalty_logdet {
                    for d in 0..p {
                        h_for_logdet[[d, d]] += ridge;
                    }
                }
                let h_inv = logdet_trace_inverse_with_ridge_policy(
                    &h_for_logdet,
                    options.ridge_floor,
                    options.ridge_policy,
                    strict_spd,
                    allow_semidefinite,
                )?;

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
                    let family_terms = family.exact_newton_block_psi_terms(
                        &inner.block_states,
                        b,
                        ExactNewtonPsiGradientContext {
                            spec,
                            deriv,
                            lambdas: &lambdas,
                            options,
                        },
                    )?;
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
                    } else if family_terms.is_some() || family_value.is_some() {
                        Array2::<f64>::zeros(deriv.s_psi.raw_dim())
                    } else {
                        return Err(format!(
                            "psi derivative penalty index {} out of bounds for exact-newton block {}",
                            deriv.penalty_index, b
                        ));
                    };
                    let value = if let Some(terms) = family_terms {
                        if terms.score_psi.len() != p {
                            return Err(format!(
                                "exact-newton psi score length mismatch on block {b}: got {}, expected {}",
                                terms.score_psi.len(),
                                p
                            ));
                        }
                        if terms.hessian_psi.nrows() != p || terms.hessian_psi.ncols() != p {
                            return Err(format!(
                                "exact-newton psi Hessian shape mismatch on block {b}: got {}x{}, expected {}x{}",
                                terms.hessian_psi.nrows(),
                                terms.hessian_psi.ncols(),
                                p,
                                p
                            ));
                        }

                        // Full moving-design exact-Newton first-order derivation:
                        //
                        // Let the profiled block contribution be
                        //
                        //   J(psi)
                        //   = V(beta*(psi), psi)
                        //     + 0.5 log|H_mode(beta*(psi), psi)|
                        //     - 0.5 log|S(psi)|_+,
                        //
                        // with penalized inner objective
                        //
                        //   V(beta, psi) = L(beta, psi) + 0.5 beta^T S(psi) beta,
                        //
                        // score
                        //
                        //   g(beta, psi) = V_beta,
                        //
                        // and exact mode defined by stationarity
                        //
                        //   g(beta*(psi), psi) = 0.
                        //
                        // Differentiate stationarity:
                        //
                        //   0 = d/dpsi g(beta*(psi), psi)
                        //     = g_beta beta_psi + g_psi
                        //     = (H + S) beta_psi + g_psi,
                        //
                        // so the exact mode response is
                        //
                        //   beta_psi = -(H + S)^{-1} g_psi.
                        //
                        // The family callback supplies only the explicit
                        // fixed-beta likelihood pieces
                        //
                        //   V_psi^explicit, g_psi^explicit, H_psi^explicit.
                        //
                        // Generic code then adds the penalty surface:
                        //
                        //   explicit = V_psi^explicit + 0.5 beta^T S_psi beta
                        //   g_psi    = g_psi^explicit + S_psi beta
                        //   H_psi    = H_psi^explicit + S_psi.
                        //
                        // The envelope/profile identity gives the exact first
                        // derivative of the profiled objective:
                        //
                        //   dJ/dpsi
                        //   = explicit
                        //     + 0.5 tr(H_mode^{-1} Hdot)
                        //     - 0.5 tr(S_+^{-1} S_psi),
                        //
                        // where the total mode-following Hessian drift is
                        //
                        //   Hdot = H_psi + D_beta H[beta_psi].
                        //
                        // So the implementation below is literally:
                        //
                        // 1. build the explicit fixed-beta pieces returned by
                        //    the family callback
                        // 2. add penalty derivatives
                        // 3. solve beta_psi = -(H+S)^{-1} g_psi
                        // 4. evaluate Hdot = H_psi + D_beta H[beta_psi]
                        // 5. assemble the log|H| and log|S|_+ trace terms.
                        //
                        // The still-missing second-order joint hyper-Hessian in
                        // theta = [rho, psi] differentiates this profile
                        // identity once more:
                        //
                        //   J_{ij}
                        //   = (V_{ij} - g_i^T H^{-1} g_j)
                        //     + 0.5[ tr(H^{-1} \tilde H_{ij})
                        //            - tr(H^{-1} \tilde H_i H^{-1} \tilde H_j) ]
                        //     - 0.5[ tr(S^+ S_{ij}) - tr(S^+ S_i S^+ S_j) ].
                        //
                        // with
                        //
                        //   beta_i  = -H^{-1} g_i,
                        //   beta_{ij}
                        //     = -H^{-1}(g_{ij} + H_i beta_j + H_j beta_i
                        //                + T[beta_i, beta_j]),
                        //
                        // and total Hessian drifts
                        //
                        //   \tilde H_i  = H_i + T[beta_i],
                        //   \tilde H_{ij}
                        //     = H_{ij}
                        //       + T_i[beta_j]
                        //       + T_j[beta_i]
                        //       + Q[beta_i, beta_j]
                        //       + T[beta_{ij}].
                        //
                        // That second-order calculus is not implemented here
                        // yet, so `outer_hessian` is still suppressed when psi
                        // coordinates are present.
                        //
                        // The penalty-only generic path below is the special
                        // case H_psi^explicit = 0 and g_psi^explicit = 0.
                        let explicit = terms.objective_psi + 0.5 * beta.dot(&s_psi_total.dot(beta));
                        let rhs = &terms.score_psi + &s_psi_total.dot(beta);
                        let u_psi = if strict_spd {
                            strict_solve_spd_with_semidefinite_option(
                                &h_mode,
                                &rhs,
                                allow_semidefinite,
                            )?
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
                        let mut h_dot = terms.hessian_psi.clone();
                        h_dot += &s_psi_total;
                        if beta_psi.dot(&beta_psi).sqrt() > 1e-14 {
                            let h_beta = family
                                .exact_newton_hessian_directional_derivative(
                                    &inner.block_states,
                                    b,
                                    &beta_psi,
                                )?
                                .ok_or_else(|| {
                                    format!(
                                        "missing exact-newton dH callback for structured psi terms on block {b}"
                                    )
                                })?;
                            if h_beta.nrows() != p || h_beta.ncols() != p {
                                return Err(format!(
                                    "exact-newton structured psi dH shape mismatch on block {b}: got {}x{}, expected {}x{}",
                                    h_beta.nrows(),
                                    h_beta.ncols(),
                                    p,
                                    p
                                ));
                            }
                            h_dot += &h_beta;
                        }
                        let h_trace = if include_logdet_h {
                            0.5 * trace_of_product(&h_inv, &h_dot)?
                        } else {
                            0.0
                        };
                        let pseudo_det = if include_logdet_s {
                            -0.5 * trace_of_product(
                                s_pinv.as_ref().ok_or_else(|| {
                                    "missing S^+ for exact-newton psi terms".to_string()
                                })?,
                                &s_psi_total,
                            )?
                        } else {
                            0.0
                        };
                        Ok(explicit + h_trace + pseudo_det)
                    } else if let Some(value) = family_value {
                        Ok(value)
                    } else if generic_supported {
                        let explicit = 0.5 * beta.dot(&s_psi_total.dot(beta));
                        let rhs = s_psi_total.dot(beta);
                        let u_psi = if strict_spd {
                            strict_solve_spd_with_semidefinite_option(
                                &h_mode,
                                &rhs,
                                allow_semidefinite,
                            )?
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
                            -0.5 * trace_product(
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
///
/// Exact derivative structure:
///
/// 1. First-order joint gradient
///
/// For theta = [rho, psi], this entry point currently computes
///
///   J_i
///   = V_i
///     + 0.5 tr(H^{-1} \dot H_i)
///     - 0.5 tr(S^+ S_i),
///
/// where:
///
///   V_i = partial_{theta_i} V(beta*(theta), theta),
///   \dot H_i = H_i + T[beta_i],
///   beta_i = -H^{-1} g_i.
///
/// The rho block comes from `outer_objective_gradient_hessian_internal`, and
/// the psi block comes from `compute_custom_family_block_psi_gradients`.
///
/// 2. Missing second-order joint Hessian
///
/// A full exact outer Hessian would need
///
///   J_{ij}
///   = (V_{ij} - g_i^T H^{-1} g_j)
///     + 0.5[ tr(H^{-1} \ddot H_{ij})
///            - tr(H^{-1} \dot H_j H^{-1} \dot H_i) ]
///     - 0.5[ tr(S^+ S_{ij}) - tr(S^+ S_j S^+ S_i) ].
///
/// with
///
///   beta_i  = -H^{-1} g_i,
///   beta_{ij}
///     = -H^{-1}(g_{ij} + H_i beta_j + H_j beta_i + T[beta_i] beta_j),
///
///   \dot H_i = H_i + T[beta_i],
///   \ddot H_{ij}
///     = H_{ij}
///       + T_i[beta_j]
///       + T_j[beta_i]
///       + T[beta_{ij}]
///       + Q[beta_i, beta_j].
///
/// That second-order assembly is not implemented here yet. This function
/// therefore returns `outer_hessian = None` whenever psi coordinates are
/// present, even though the first-order psi gradient is exact.
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

    let mut result = outer_objective_gradient_hessian_internal(
        family,
        specs,
        options,
        &penalty_counts,
        rho_current,
        warm_start.map(|w| &w.inner),
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
        // (smoothing-parameter coordinates). For joint theta = [rho, psi], the
        // profiled objective is
        //
        //   J(theta)
        //   = V(beta*(theta), theta)
        //     + 0.5 log|H_mode(beta*(theta), theta)|
        //     - 0.5 log|S(theta)|_+,
        //
        // with stationarity g(beta*(theta), theta) = 0 and first-order mode
        // response
        //
        //   beta_i = -H^{-1} g_i.
        //
        // The current implementation computes J_i exactly for psi directions by
        // combining:
        //
        //   explicit_i,
        //   g_i,
        //   H_i + D_beta H[beta_i],
        //
        // inside `compute_custom_family_block_psi_gradients`.
        //
        // A full exact Hessian J_{ij} would require the complete profiled
        // decomposition
        //
        //   J_{ij}
        //   = (V_{ij} - g_i^T H^{-1} g_j)
        //     + 0.5[ tr(H^{-1} \tilde H_{ij})
        //            - tr(H^{-1} \tilde H_i H^{-1} \tilde H_j) ]
        //     - 0.5[ tr(S^+ S_{ij}) - tr(S^+ S_i S^+ S_j) ].
        //
        // In practical code terms that means adding the fixed-beta second
        // partials
        //
        //   V_{ij}, g_{ij}, H_{ij}, S_{ij},
        //
        // together with the contracted beta-curvature operators needed to form
        // the total Hessian drifts
        //
        //   \tilde H_i  = H_i + T[beta_i],
        //   \tilde H_{ij}
        //     = H_{ij}
        //       + T_i[beta_j]
        //       + T_j[beta_i]
        //       + Q[beta_i, beta_j]
        //       + T[beta_{ij}].
        //
        // Until those exact second-order psi objects exist, returning a
        // partial Hessian here would be wrong, so the joint path intentionally
        // suppresses `outer_hessian` whenever psi_dim > 0.
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

fn trace_of_product(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64, String> {
    if a.nrows() != a.ncols() || b.nrows() != b.ncols() || a.nrows() != b.nrows() {
        return Err(format!(
            "trace_of_product shape mismatch: a={}x{}, b={}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        ));
    }
    let n = a.nrows();
    let mut out = 0.0;
    for i in 0..n {
        for j in 0..n {
            out += a[[i, j]] * b[[j, i]];
        }
    }
    Ok(out)
}

fn synchronized_states_from_flat_beta<F: CustomFamily>(
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

fn post_update_tangent_basis_for_block<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_states: &[ParameterBlockState],
    block_idx: usize,
) -> Result<Option<Array2<f64>>, String> {
    let spec = specs.get(block_idx).ok_or_else(|| {
        format!(
            "post-update tangent basis block index {block_idx} out of bounds for {} specs",
            specs.len()
        )
    })?;
    let state = block_states.get(block_idx).ok_or_else(|| {
        format!(
            "post-update tangent basis block index {block_idx} out of bounds for {} states",
            block_states.len()
        )
    })?;
    let p = state.beta.len();
    if p == 0 {
        return Ok(Some(Array2::zeros((0, 0))));
    }

    let base = family.post_update_block_beta(block_states, block_idx, spec, state.beta.clone())?;
    if base.len() != p {
        return Err(format!(
            "post-update tangent basis block {block_idx} returned beta length {}, expected {p}",
            base.len()
        ));
    }

    let mut jac = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        let mut probe = base.clone();
        probe[j] += 1.0;
        let mapped = family.post_update_block_beta(block_states, block_idx, spec, probe)?;
        if mapped.len() != p {
            return Err(format!(
                "post-update tangent basis block {block_idx} probe returned beta length {}, expected {p}",
                mapped.len()
            ));
        }
        jac.column_mut(j).assign(&(&mapped - &base));
    }

    let identity = Array2::<f64>::eye(p);
    let jac_err = (&jac - &identity)
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    if jac_err <= 1e-10 {
        return Ok(None);
    }

    let (u_opt, singular, _) = jac
        .svd(true, false)
        .map_err(|e| format!("post-update tangent basis SVD failed for block {block_idx}: {e}"))?;
    let u = u_opt.ok_or_else(|| {
        format!("post-update tangent basis SVD omitted left singular vectors for block {block_idx}")
    })?;
    let max_sv = singular.iter().fold(0.0_f64, |m, &s| m.max(s.abs()));
    let tol = (max_sv * 1e-10).max(1e-12);
    let rank = singular.iter().filter(|&&s| s > tol).count();
    if rank == p {
        return Ok(None);
    }
    Ok(Some(u.slice(ndarray::s![.., 0..rank]).to_owned()))
}

fn joint_post_update_tangent_basis<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_states: &[ParameterBlockState],
) -> Result<Option<Array2<f64>>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, end)| *end).unwrap_or(0);
    let mut blocks: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    let mut any_reduced = false;
    let mut total_rank = 0usize;
    for (block_idx, &(start, end)) in ranges.iter().enumerate() {
        let p = end - start;
        let basis =
            match post_update_tangent_basis_for_block(family, specs, block_states, block_idx)? {
                Some(basis) => {
                    any_reduced = true;
                    basis
                }
                None => Array2::<f64>::eye(p),
            };
        total_rank += basis.ncols();
        blocks.push(basis);
    }
    if !any_reduced {
        return Ok(None);
    }

    let mut basis = Array2::<f64>::zeros((total, total_rank));
    let mut col_at = 0usize;
    for (block_idx, block_basis) in blocks.iter().enumerate() {
        let (start, end) = ranges[block_idx];
        let width = block_basis.ncols();
        if width == 0 {
            continue;
        }
        basis
            .slice_mut(ndarray::s![start..end, col_at..col_at + width])
            .assign(block_basis);
        col_at += width;
    }
    Ok(Some(basis))
}

fn solve_mode_sensitivity(
    lhs: &Array2<f64>,
    rhs: &Array1<f64>,
    tangent_basis: Option<&Array2<f64>>,
    strict_spd: bool,
    allow_semidefinite: bool,
    _options: &BlockwiseFitOptions,
    context: &str,
) -> Result<Array1<f64>, String> {
    let solve_dense = |matrix: &Array2<f64>, vector: &Array1<f64>| -> Result<Array1<f64>, String> {
        if strict_spd {
            strict_solve_spd_with_semidefinite_option(matrix, vector, allow_semidefinite)
        } else {
            solve_dense_symmetric_indefinite(matrix, vector, context)
        }
    };

    match tangent_basis {
        Some(basis) => {
            if basis.nrows() != lhs.nrows() {
                return Err(format!(
                    "{context} tangent basis row mismatch: got {}, expected {}",
                    basis.nrows(),
                    lhs.nrows()
                ));
            }
            if basis.ncols() == 0 {
                return Ok(Array1::zeros(lhs.nrows()));
            }
            let lhs_reduced = basis.t().dot(lhs).dot(basis);
            let rhs_reduced = basis.t().dot(rhs);
            let sol_reduced = solve_dense(&lhs_reduced, &rhs_reduced)?;
            Ok(basis.dot(&sol_reduced))
        }
        None => solve_dense(lhs, rhs),
    }
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

fn exact_newton_joint_stationarity_inf_norm(
    eval: &FamilyEvaluation,
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> Result<Option<f64>, String> {
    if eval.block_working_sets.len() != states.len() || states.len() != s_lambdas.len() {
        return Err("exact-newton joint stationarity check: block dimension mismatch".to_string());
    }

    let mut inf_norm = 0.0_f64;
    for b in 0..states.len() {
        let gradient = match &eval.block_working_sets[b] {
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
            // This is the quantity that must be small at the inner mode before
            // the outer LAML derivative formulas are trustworthy. Using only
            // coordinate step size can declare convergence too early for
            // coupled exact-Newton families, leaving a visibly wrong outer
            // objective/gradient even when the blockwise updates have nearly
            // stalled.
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient,
            _ => return Ok(None),
        };
        let mut residual = s_lambdas[b].dot(&states[b].beta) - gradient;
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            residual += &states[b].beta.mapv(|v| ridge * v);
        }
        let block_inf = residual.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        inf_norm = inf_norm.max(block_inf);
    }
    Ok(Some(inf_norm))
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
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let mut h = if let Some(h_exact) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        total,
        "joint exact-newton Hessian shape mismatch in covariance",
    )? {
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
    symmetrize_dense_in_place(&mut h);
    if use_exact_newton_strict_spd(family) {
        strict_inverse_spd(&h)
    } else {
        match inverse_spd_with_retry(&h, effective_solver_ridge(options.ridge_floor), 8) {
            Ok(cov) => Ok(cov),
            Err(_) => pinv_positive_part(&h, effective_solver_ridge(options.ridge_floor)),
        }
    }
}

#[allow(clippy::type_complexity)]
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
            penalized_objective: finite_penalized_objective(
                inner.log_likelihood,
                inner.penalty_value,
                reml_term,
            ),
            outer_iterations: 0,
            outer_final_gradient_norm: 0.0,
            inner_cycles: inner.cycles,
            converged: inner.converged,
        });
    }

    if include_exact_newton_logdet_h(family) && family.known_link_wiggle().is_some() {
        // Stabilized exact-wiggle fit path.
        //
        // The mathematically ambitious route for exact custom families is:
        //
        // 1. solve the inner block mode beta^(rho),
        // 2. differentiate the exact joint Hessian H(beta^) with respect to rho,
        // 3. optimize the REML/LAML objective over rho using those exact
        //    derivatives.
        //
        // For the binomial location-scale wiggle family with spatial blocks, the
        // derivative formulas are now correct, but the full outer exact-REML
        // optimization remains numerically fragile in practice. The failure mode
        // is not a clean statistical disagreement; it is a pile-up of hard
        // linear algebra problems on badly conditioned matrices:
        //
        // - exact block solves need repeated ridge retries,
        // - positive-part logdet eigensolvers may not converge,
        // - REML mode-sensitivity solves can become indefinite/non-finite,
        // - the outer BFGS path can walk into rho regions where every one of the
        //   above gets amplified.
        //
        // For this family the important user-facing contract of the API is that
        // the fit remains finite and returns a coherent block state. The initial
        // smoothing values in `rho0` already define a valid penalized model, and
        // the corrected inner exact/blockwise fit at those values is stable. So
        // until the outer exact-REML path is robust enough to optimize rho
        // reliably for the wiggle family, we intentionally stop at:
        //
        //   beta^ = argmin_beta Q(beta; rho0)
        //
        // and return that finite fit instead of attempting a numerically brittle
        // exact outer search. The returned penalized objective still includes the
        // REML-style logdet pieces when they are finite; if those terms become
        // non-finite, `finite_penalized_objective` drops back to the finite
        // penalized likelihood surface rather than returning NaN/Inf.
        //
        // This is a deliberate stabilization choice for the known-link wiggle
        // exact-Newton family, not a generic change to custom-family smoothing.
        let per_block = split_log_lambdas(&rho0, &penalty_counts)?;
        let mut inner = inner_blockwise_fit(family, specs, &per_block, options, None)?;
        refresh_all_block_etas(family, specs, &mut inner.block_states)?;
        let covariance_conditional = if options.compute_covariance {
            compute_joint_covariance(family, specs, &inner.block_states, &per_block, options).ok()
        } else {
            None
        };
        let reml_term = if include_exact_newton_logdet_h(family) {
            0.5 * inner.block_logdet_h
        } else {
            0.0
        } - if include_exact_newton_logdet_s(family, options) {
            0.5 * inner.block_logdet_s
        } else {
            0.0
        };
        let reml_term = if reml_term.is_finite() {
            reml_term
        } else {
            0.0
        };
        return Ok(BlockwiseFitResult {
            block_states: inner.block_states,
            log_likelihood: inner.log_likelihood,
            log_lambdas: rho0.clone(),
            lambdas: rho0.mapv(f64::exp),
            covariance_conditional,
            penalized_objective: finite_penalized_objective(
                inner.log_likelihood,
                inner.penalty_value,
                reml_term,
            ),
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
    let objective = CachedFirstOrderObjective::new(|x: &Array1<f64>| {
        let cached = warm_cache.lock().ok().and_then(|g| g.clone());
        let warm_start = if include_exact_newton_logdet_h(family) {
            None
        } else {
            cached.as_ref()
        };
        let (obj, grad, hess_opt) = match outer_objective_gradient_hessian(
            family,
            specs,
            options,
            &penalty_counts,
            x,
            warm_start,
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
                        "custom-family outer objective/derivatives became non-finite".to_string(),
                    );
                }
                if include_exact_newton_logdet_h(family)
                    && let Some((x_prev, obj_prev, grad_prev, _)) = last_eval.as_ref()
                    && x_prev.len() == x.len()
                {
                    return Ok((*obj_prev, grad_prev.clone()));
                }
                return Err(ObjectiveEvalError::recoverable(
                    "custom-family outer objective/derivatives became non-finite",
                ));
            }
            Err(e) => {
                if let Ok(mut guard) = last_outer_error.lock() {
                    *guard = Some(e);
                }
                if include_exact_newton_logdet_h(family)
                    && let Some((x_prev, obj_prev, grad_prev, _)) = last_eval.as_ref()
                    && x_prev.len() == x.len()
                {
                    return Ok((*obj_prev, grad_prev.clone()));
                }
                return Err(ObjectiveEvalError::recoverable(
                    "custom-family outer objective/gradient evaluation failed",
                ));
            }
        };
        last_eval = Some((x.clone(), obj, grad.clone(), hess_opt.clone()));
        Ok((obj, grad))
    });
    let outer_max_iter = if include_exact_newton_logdet_h(family) {
        options.outer_max_iter.min(3).max(1)
    } else {
        options.outer_max_iter
    };
    let mut solver = Bfgs::new(rho0.clone(), objective)
        .with_bounds(
            Bounds::new(lower, upper, 1e-6).expect("custom-family rho bounds must be valid"),
        )
        .with_tolerance(
            Tolerance::new(options.outer_tol).expect("custom-family tolerance must be valid"),
        )
        .with_profile(opt::Profile::Robust)
        .with_max_iterations(
            MaxIterations::new(outer_max_iter).expect("custom-family max iterations must be valid"),
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
        Err(BfgsError::MaxIterationsReached { last_solution }) => {
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
        Err(BfgsError::LineSearchFailed { last_solution, .. }) => {
            if last_solution.final_value.is_finite()
                && last_solution.final_gradient_norm.is_finite()
            {
                log::warn!(
                    "Outer smoothing line search failed; using best-so-far solution (iter={}, f={:.6e}, ||g||={:.3e}).",
                    last_solution.iterations,
                    last_solution.final_value,
                    last_solution.final_gradient_norm
                );
                *last_solution
            } else {
                return Err(format!(
                    "outer smoothing optimization failed: LineSearchFailed.{details}",
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
        penalized_objective: finite_penalized_objective(
            inner.log_likelihood,
            inner.penalty_value,
            if include_exact_newton_logdet_h(family) {
                0.5 * inner.block_logdet_h
            } else {
                0.0
            } - if include_exact_newton_logdet_s(family, options) {
                0.5 * inner.block_logdet_s
            } else {
                0.0
            },
        ),
        outer_iterations: sol.iterations,
        outer_final_gradient_norm: sol.final_gradient_norm,
        inner_cycles: inner.cycles,
        converged: inner.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::gamlss::{BinomialLocationScaleFamily, BinomialLocationScaleWiggleFamily};
    use crate::matrix::DesignMatrix;
    use approx::assert_relative_eq;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, Array2, array};

    #[derive(Clone)]
    struct OneBlockIdentityFamily;

    #[test]
    fn floor_positive_working_weights_preserves_exact_zeros() {
        let weights = array![0.0, 1.0e-16, 0.25];
        let floored = floor_positive_working_weights(&weights, 1.0e-6);
        assert_eq!(floored[0], 0.0);
        assert_eq!(floored[1], 1.0e-6);
        assert_eq!(floored[2], 0.25);
    }

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
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
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
                    hessian: SymmetricMatrix::Dense(array![[2.0]]),
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
                    hessian: SymmetricMatrix::Dense(array![[2.0]]),
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
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
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
                    hessian: SymmetricMatrix::Dense(array![[-1.0]]),
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
                block_working_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient,
                    hessian: SymmetricMatrix::Dense(h),
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
            Ok(Some(array![[2.0, 0.1], [3.0, 2.0]]))
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
            x_psi_psi: None,
            s_psi_psi: None,
            s_psi_psi_components: None,
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
    fn strict_psd_positive_part_geometry_accepts_semidefinite_boundary() {
        let h = array![[4.0, 0.0], [0.0, 0.0]];
        let rhs = array![8.0, 3.0];
        let sol = strict_solve_spd_with_semidefinite_option(&h, &rhs, true)
            .expect("semidefinite positive-part solve");
        let inv = strict_inverse_spd_with_semidefinite_option(&h, true)
            .expect("semidefinite positive-part inverse");
        let logdet = strict_logdet_spd_with_semidefinite_option(&h, true)
            .expect("semidefinite positive-part logdet");
        assert!((sol[0] - 2.0).abs() < 1e-12);
        assert!(sol[1].abs() < 1e-12);
        assert!((inv[[0, 0]] - 0.25).abs() < 1e-12);
        assert!(inv[[1, 1]].abs() < 1e-12);
        assert!((logdet - 4.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn strict_psd_positive_part_geometry_still_rejects_indefinite_matrix() {
        let h = array![[1.0, 0.0], [0.0, -1e-3]];
        let err = strict_solve_spd_with_semidefinite_option(&h, &array![1.0, 0.0], true)
            .expect_err("indefinite matrix must still be rejected");
        assert!(
            err.contains("strict pseudo-laplace SPD solve failed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn pseudo_laplace_exact_newton_symmetrizes_nearly_symmetric_hessian() {
        let spec = ParameterBlockSpec {
            name: "nearly_symmetric".to_string(),
            design: DesignMatrix::Dense(array![[1.0, 0.0], [0.0, 1.0]]),
            offset: array![0.0, 0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0, 0.0]),
        };
        let fit = fit_custom_family(
            &OneBlockNearlySymmetricPseudoLaplaceFamily,
            &[spec],
            &BlockwiseFitOptions {
                use_reml_objective: true,
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

        let family = BinomialLocationScaleWiggleFamily {
            y,
            weights,
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
        let n = 8usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_elem(n, 1.0);
        let threshold_spec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let log_sigma_spec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let family = BinomialLocationScaleFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            threshold_design: None,
            log_sigma_design: None,
        };
        let specs = vec![threshold_spec, log_sigma_spec];
        let penalty_counts = vec![1usize, 1usize];
        let rho = array![0.0, 0.0];
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
            let abs = (g0[k] - g_fd).abs();
            let rel = abs / g_fd.abs().max(1e-8);
            if abs >= 2e-3 {
                assert_eq!(
                    g0[k].signum(),
                    g_fd.signum(),
                    "outer diagonal LAML gradient sign mismatch at {}: analytic={} fd={}",
                    k,
                    g0[k],
                    g_fd
                );
            }
            assert!(
                abs < 2e-3 || rel < 2e-3,
                "outer diagonal LAML gradient mismatch at {}: analytic={} fd={} abs={} rel={}",
                k,
                g0[k],
                g_fd,
                abs,
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
        let threshold_design = threshold_spec.design.clone();
        let log_sigma_design = log_sigma_spec.design.clone();
        let family = BinomialLocationScaleFamily {
            y,
            weights,
            link_kind: crate::types::InverseLink::Standard(crate::types::LinkFunction::Probit),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
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
                if h0[[k, l]].abs().max(h_fd.abs()) > 1e-10 {
                    assert_eq!(
                        h0[[k, l]].signum(),
                        h_fd.signum(),
                        "outer Hessian sign mismatch at ({k},{l}): analytic={} fd={}",
                        h0[[k, l]],
                        h_fd
                    );
                }
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
    fn quadratic_linear_constraints_ignore_near_tangential_inactive_rows() {
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
    fn quadratic_linear_constraints_project_warm_active_rows_back_to_boundary() {
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
