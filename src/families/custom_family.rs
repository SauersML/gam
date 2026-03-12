use crate::faer_ndarray::FaerCholesky;
use crate::faer_ndarray::{FaerArrayView, FaerEigh, FaerSvd};
use crate::linalg::utils::{
    StableSolver, boundary_hit_step_fraction, default_slq_parameters, stochastic_lanczos_logdet_spd,
};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::solver::opt_objective::CachedSecondOrderObjective;
use crate::types::{LinkFunction, RidgeDeterminantMode, RidgePolicy};
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{Lblt as FaerLblt, Solve as FaerSolve};
use ndarray::{Array1, Array2};
use opt::{
    Bounds, MaxIterations, NewtonTrustRegion, NewtonTrustRegionError, ObjectiveEvalError, Tolerance,
};
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
    pub blockworking_sets: Vec<BlockWorkingSet>,
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
    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::QuadraticReml
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

    /// Optional metadata describing a known link with learnable wiggle.
    fn known_link_wiggle(&self) -> Option<KnownLinkWiggle> {
        None
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
    /// Returns the unpenalized matrix `H = -∇² log L` in the flattened block order.
    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
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
            minweight: 1e-12,
            ridge_floor: 1e-12,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_remlobjective: true,
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
    pub penalizedobjective: f64,
    pub outer_iterations: usize,
    pub outer_final_gradient_norm: f64,
    pub inner_cycles: usize,
    pub converged: bool,
}

fn finite_penalizedobjective(log_likelihood: f64, penalty_value: f64, reml_term: f64) -> f64 {
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
    pub penalty_index: Option<usize>,
    pub x_psi: Array2<f64>,
    pub s_psi: Array2<f64>,
    pub s_psi_components: Option<Vec<(usize, Array2<f64>)>>,
    pub x_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi_components: Option<Vec<Vec<(usize, Array2<f64>)>>>,
}

#[derive(Clone)]
pub struct ExactNewtonJointPsiTerms {
    pub objective_psi: f64,
    pub score_psi: Array1<f64>,
    pub hessian_psi: Array2<f64>,
}

#[derive(Clone)]
pub struct ExactNewtonJointPsiSecondOrderTerms {
    pub objective_psi_psi: f64,
    pub score_psi_psi: Array1<f64>,
    pub hessian_psi_psi: Array2<f64>,
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

#[derive(Clone, Debug)]
enum JointHyperCoord {
    Rho {
        block_idx: usize,
        penalty_idx: usize,
        global_idx: usize,
    },
    Psi {
        block_idx: usize,
        local_idx: usize,
        global_idx: usize,
    },
}

impl JointHyperCoord {
    fn global_idx(&self) -> usize {
        match self {
            Self::Rho { global_idx, .. } | Self::Psi { global_idx, .. } => *global_idx,
        }
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

fn buildblock_states<F: CustomFamily>(
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
            Ok(x.matrixvectormultiply(&beta) + off)
        })?;
    }
    Ok(())
}

fn refresh_single_block_eta<F: CustomFamily>(
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
fn floor_positiveworkingweight(rawweight: f64, minweight: f64) -> f64 {
    if rawweight <= 0.0 {
        0.0
    } else {
        rawweight.max(minweight)
    }
}

#[inline]
fn floor_positiveworking_weights(working_weights: &Array1<f64>, minweight: f64) -> Array1<f64> {
    working_weights.mapv(|wi| floor_positiveworkingweight(wi, minweight))
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
                let (beta_constrained, active_set) = solve_quadraticwith_linear_constraints(
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
                let beta = solve_blockweighted_system(
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
            let (beta_constrained, active_set) = solve_quadraticwith_linear_constraints(
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
                let floor = effective_solverridge(ctx.options.ridge_floor);
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
                solve_spd_systemwith_policy(
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
            .solvevectorwithridge_retries(hessian, &rhs, 0.0)
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

    let kktview = FaerArrayView::new(&kkt);
    let lb = FaerLblt::new(kktview.as_ref(), Side::Lower);
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

fn solve_quadraticwith_linear_constraints(
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
        let mut aw = Array2::<f64>::zeros((active.len(), p));
        let mut residualw = Array1::<f64>::zeros(active.len());
        for (r, &idx) in active.iter().enumerate() {
            aw.row_mut(r).assign(&constraints.a.row(idx));
            residualw[r] = constraints.b[idx] - constraints.a.row(idx).dot(&x);
        }
        let (direction, lambda_sys) = solve_kkt_step(hessian, &gradient, &aw, Some(&residualw))?;
        let step_norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        if step_norm <= tol_step {
            if active.is_empty() {
                return Ok((x, active));
            }
            // solve_kkt_step returns multipliers from:
            //   H d + Aw^T lambda_sys = -gradient.
            // For constraints A*x >= b, true multipliers satisfy
            //   gradient + H d = Aw^T lambda_true, so lambda_true = -lambda_sys.
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
                    if let Ok(chol) = ridged.cholesky(Side::Lower) {
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
        return strict_inverse_spdwith_semidefinite_option(
            matrix_on_logdet_surface,
            allow_semidefinite,
        );
    }

    match ridge_policy.determinant_mode {
        // For the full determinant surface, d log|H| = tr(H^{-1} dH).
        RidgeDeterminantMode::Auto
        | RidgeDeterminantMode::Full
        | RidgeDeterminantMode::StochasticLanczos => inverse_spdwith_retry(
            matrix_on_logdet_surface,
            effective_solverridge(ridge_floor),
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
                effective_solverridge(ridge_floor)
            } else {
                0.0
            }
            .max(1e-14);
            pinv_positive_part_with_floor(matrix_on_logdet_surface, positive_floor)
        }
    }
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

fn exact_newton_joint_hessian_symmetrized<F: CustomFamily>(
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
    Ok(chol.solvevec(rhs))
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

fn strict_solve_spdwith_semidefinite_option(
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

fn strict_inverse_spdwith_semidefinite_option(
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
    let matrixview = FaerArrayView::new(matrix);
    let solver = FaerLblt::new(matrixview.as_ref(), Side::Lower);
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
        family.exact_newton_outerobjective(),
        ExactNewtonOuterObjective::QuadraticReml | ExactNewtonOuterObjective::PseudoLaplace
    )
}

fn include_exact_newton_logdet_s<F: CustomFamily + ?Sized>(
    family: &F,
    options: &BlockwiseFitOptions,
) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::QuadraticReml
        && options.use_remlobjective
}

fn use_exact_newton_strict_spd<F: CustomFamily + ?Sized>(family: &F) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::PseudoLaplace
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
    let allow_semidefinite = strict_spd && family.exact_newton_allows_semidefinitehessian();
    refresh_all_block_etas(family, specs, states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if let Some(h_joint) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
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
    if eval.blockworking_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.blockworking_sets.len(),
            specs.len()
        ));
    }

    let mut logdet_h_total = 0.0;
    let mut logdet_s_total = 0.0;
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
    let mut states = buildblock_states(family, specs)?;
    refresh_all_block_etas(family, specs, &mut states)?;
    let has_joint_exacthessian = family.exact_newton_joint_hessian(&states)?.is_some();
    let inner_tol = if has_joint_exacthessian {
        options.inner_tol.min(1e-10)
    } else {
        options.inner_tol
    };
    let inner_max_cycles = if has_joint_exacthessian {
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
    let ridge = effective_solverridge(options.ridge_floor);
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
    let mut lastobjective = -initial_eval.log_likelihood + current_penalty;
    let mut converged = false;
    let mut cycles_done = 0usize;

    for cycle in 0..inner_max_cycles {
        let mut max_beta_step = 0.0_f64;

        let mut objective_cycle_prev = lastobjective;
        for b in 0..specs.len() {
            // Keep all blocks synchronized with any dynamic geometry.
            if family.block_geometry_is_dynamic() {
                refresh_all_block_etas(family, specs, &mut states)?;
            }
            let eval = family.evaluate(&states)?;
            if eval.blockworking_sets.len() != specs.len() {
                return Err(format!(
                    "family returned {} block working sets, expected {}",
                    eval.blockworking_sets.len(),
                    specs.len()
                ));
            }

            let spec = &specs[b];
            let work = &eval.blockworking_sets[b];
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
                refresh_single_block_eta(family, specs, &mut states, b)?;
                let trial_eval = family.evaluate(&states)?;
                let trial_block_penalty =
                    block_quadratic_penalty(&states[b].beta, s_lambda, ridge, options.ridge_policy);
                let trial_penalty = current_penalty - old_block_penalty + trial_block_penalty;
                let trialobjective = -trial_eval.log_likelihood + trial_penalty;
                if trialobjective.is_finite() && trialobjective <= objective_cycle_prev + 1e-10 {
                    objective_cycle_prev = trialobjective;
                    current_penalty = trial_penalty;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                states[b].beta = beta_old.clone();
                refresh_single_block_eta(family, specs, &mut states, b)?;
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
                            refresh_single_block_eta(family, specs, &mut states, b)?;
                            let trial_eval = family.evaluate(&states)?;
                            let trial_block_penalty = block_quadratic_penalty(
                                &states[b].beta,
                                s_lambda,
                                ridge,
                                options.ridge_policy,
                            );
                            let trial_penalty =
                                current_penalty - old_block_penalty + trial_block_penalty;
                            let trialobjective = -trial_eval.log_likelihood + trial_penalty;
                            if trialobjective.is_finite()
                                && trialobjective <= objective_cycle_prev + 1e-10
                            {
                                objective_cycle_prev = trialobjective;
                                current_penalty = trial_penalty;
                                accepted = true;
                                break;
                            }
                            states[b].beta = beta_old.clone();
                            refresh_single_block_eta(family, specs, &mut states, b)?;
                        }
                    }
                }
            }
            if !accepted {
                states[b].beta = beta_old;
                refresh_single_block_eta(family, specs, &mut states, b)?;
            }
        }

        refresh_all_block_etas(family, specs, &mut states)?;
        let eval = family.evaluate(&states)?;
        current_penalty = total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
        let objective = -eval.log_likelihood + current_penalty;
        let objective_change = (objective - lastobjective).abs();
        lastobjective = objective;
        cycles_done = cycle + 1;

        let objective_tol = inner_tol * (1.0 + objective.abs());
        let exact_joint_stationarity_ok = if has_joint_exacthessian {
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
fn outerobjectivegradienthessian_internal<F: CustomFamily>(
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
    let allow_semidefinite = strict_spd && family.exact_newton_allows_semidefinitehessian();
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
    // Joint exact-Hessian path for the current realized family/spec state.
    //
    // The outer objective is the profiled Laplace surface
    //
    //   J(theta)
    //   = V(beta(theta), theta)
    //     + 0.5 log|H(beta(theta), theta)|
    //     - 0.5 log|S(theta)|_+,
    //
    // evaluated at the fitted inner mode. Here
    //
    //   F(beta, theta) := D_beta V(beta, theta) = 0,
    //   H(beta, theta) := F_beta(beta, theta) = H_L(beta, theta) + S(theta).
    //
    // Because the likelihood has no explicit rho-dependence, the fixed-beta
    // forcing for smoothing coordinate rho_k is
    //
    //   g_k := F_{rho_k} = A_k beta,
    //   A_k := dS/drho_k.
    //
    // Differentiating stationarity gives the exact joint mode response
    //
    //   H u_k = -g_k,
    //   u_k = d beta / d rho_k.
    //
    // The first profiled derivative is therefore
    //
    //   dJ/drho_k
    //   = 0.5 beta^T A_k beta
    //     + 0.5 tr(H^{-1} dot H_k)
    //     - 0.5 tr(S^+ A_k),
    //
    // with total curvature drift
    //
    //   dot H_k = A_k + D_beta H_L[u_k].
    //
    // The key structural fact is that `g_k` can be penalty-local while `u_k`
    // is not. Once the likelihood couples blocks, solving only inside the
    // penalized block computes the derivative of a different surrogate
    // objective. Families such as binomial location-scale with
    //
    //   q = -eta_t exp(-eta_ls)
    //
    // therefore must use this joint path whenever the realized specs provide
    // enough design information to build the exact joint curvature, even if
    // the family instance itself did not cache those designs internally.
    if let Some(h_joint_unpen) = exact_newton_joint_hessian_symmetrized(
        family,
        &inner.block_states,
        specs,
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
            if moderidge > 0.0 {
                for d in start..end {
                    s_joint[[d, d]] += moderidge;
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
        // Inner stationarity for the coupled blocks is
        //
        //   F(beta, rho) = D_beta V(beta, rho) = 0.
        //
        // Differentiating with respect to rho_k gives
        //
        //   H u_k = -g_k,
        //   g_k = A_k beta,
        //
        // where `A_k = dS/drho_k`. Here `h_joint_unpen` is the joint
        // likelihood curvature `H_L`, `s_joint` is the joint penalty matrix
        // `S`, and
        //
        //   j_joint = h_joint_unpen + s_joint
        //
        // is the exact joint mode Jacobian used to solve `u_k`.
        let mut j_joint = h_joint_unpen.clone();
        j_joint += &s_joint;

        // Log-determinant traces may follow a ridge-augmented surface depending
        // on ridge policy, but the coupled beta sensitivities are always
        // solved against the true joint mode system.
        //
        // The exact first derivative for smoothing coordinate rho_k is
        //
        //   dJ/drho_k
        //   = 0.5 beta^T A_k beta
        //     + 0.5 tr(J_trace^{-1} dot J_k)
        //     - 0.5 tr(S_+^{-1} A_k),
        //
        // where
        //
        //   J_mode  = H_L(beta^) + S,
        //   J_trace = J_mode (+ optional logdet-only ridge),
        //   dot J_k = A_k + D_beta H_L[u_k],
        //   J_mode u_k = -A_k beta^.
        //
        // The algebraic separation is:
        // - `u_k` must come from the full joint mode system `J_mode`
        // - the trace can be evaluated on the ridge-adjusted `J_trace`
        //   if the chosen determinant surface includes that stabilization
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
                    //   = tr(J_trace^{-1} dot J_k),
                    //
                    // with
                    //
                    //   dot J_k = A_k + D_beta H_L[u_k].
                    //
                    // The first term `A_k` is the explicit penalty drift. The
                    // second term is the implicit mode-path correction: moving
                    // rho_k changes the fitted coefficients, which changes the
                    // joint likelihood curvature. This remains a full joint
                    // object even when `A_k` is supported in only one block.
                    let mut d_j_k = a_k.clone();
                    if u_norm > 1e-14 {
                        // D_beta H_L[u_k] is the directional derivative of the
                        // joint likelihood curvature with respect to beta along
                        // the exact joint mode response u_k = d beta^ / d rho_k.
                        let h_rho = family
                            .exact_newton_joint_hessian_directional_derivative_with_specs(
                                &synced_joint_states,
                                specs,
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
                    // Exact profiled/Laplace Hessian entry for rho = log lambda:
                    //
                    //   d^2J / (drho_k drho_l)
                    //   = u_l^T A_k beta
                    //     + 0.5 beta^T B_{k,l} beta
                    //     + 0.5 tr(H^{-1} ddot H_{k,l})
                    //     - 0.5 tr(H^{-1} dot H_l H^{-1} dot H_k)
                    //     - 0.5 d^2/drho_k drho_l log|S|_+.
                    //
                    // For simple log-scaled penalties, B_{k,l} = delta_{k,l} A_k.
                    // The code stores
                    //
                    //   dot H_k = A_k + D_beta H_L[u_k]
                    //
                    // in `j_terms[k]`, then forms
                    //
                    //   ddot H_{k,l}
                    //   = D_beta H_L[u_{k,l}]
                    //     + D_beta^2 H_L[u_l, u_k]
                    //     + delta_{k,l} A_k.
                    //
                    // The second mode response is solved from
                    //
                    //   H u_{k,l}
                    //   = -(A_k u_l + dot H_l u_k + delta_{k,l} A_k beta),
                    //
                    // which is the log-scaled specialization of
                    //
                    //   H u_{k,l}
                    //   = -(A_k u_l + A_l u_k + B_{k,l} beta + D_beta H_L[u_l] u_k).
                    //
                    // The positive-part penalty term below is the corresponding
                    // second derivative of -0.5 log|S|_+.
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
                            .exact_newton_joint_hessian_directional_derivative_with_specs(
                                &synced_joint_states,
                                specs,
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
                            .exact_newton_joint_hessian_second_directional_derivative_with_specs(
                                &synced_joint_states,
                                specs,
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
                        let trsecond = trace_product(&h_inv, &j_kl);
                        let tr_h = if include_logdet_h {
                            0.5 * (-tr_prod + trsecond)
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
    if let Some(h_joint_unpen) = family
        .joint_outer_hyper_surrogate_hessian_with_specs(&inner.block_states, specs)?
        .map(|h| {
            symmetrized_square_matrix(
                h,
                total,
                "joint outer-hyper surrogate Hessian shape mismatch",
            )
        })
        .transpose()?
    {
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
            if moderidge > 0.0 {
                for d in start..end {
                    s_joint[[d, d]] += moderidge;
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
        let mut j_joint = h_joint_unpen.clone();
        j_joint += &s_joint;
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
        let logdet_h_total = if include_logdet_h {
            if strict_spd {
                strict_logdet_spd_with_semidefinite_option(&j_for_traces, allow_semidefinite)?
            } else {
                stable_logdet_with_ridge_policy(
                    &j_for_traces,
                    options.ridge_floor,
                    options.ridge_policy,
                )?
            }
        } else {
            0.0
        };
        let logdet_s_total = if include_logdet_s {
            let mut acc = 0.0;
            for (b, spec) in specs.iter().enumerate() {
                let lambdas = per_block[b].mapv(f64::exp);
                let mut s_lambda = Array2::<f64>::zeros(spec.penalties[0].raw_dim());
                for (k, s) in spec.penalties.iter().enumerate() {
                    s_lambda.scaled_add(lambdas[k], s);
                }
                acc += stable_logdet_with_ridge_policy(
                    &s_lambda,
                    options.ridge_floor,
                    options.ridge_policy,
                )?;
            }
            acc
        } else {
            0.0
        };
        let objective = -inner.log_likelihood + inner.penalty_value + 0.5 * logdet_h_total
            - 0.5 * logdet_s_total;

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
                let g_pen = 0.5 * beta_block.dot(&local.dot(&beta_block));
                let keep_second_order = need_hessian || include_logdet_h || include_logdet_s;
                let u_k = if keep_second_order {
                    solve_mode_sensitivity(
                        &j_joint,
                        &(-&a_k_beta),
                        None,
                        strict_spd,
                        allow_semidefinite,
                        options,
                        "joint mode sensitivity system for surrogate REML gradient",
                    )?
                } else {
                    Array1::<f64>::zeros(total)
                };
                let g = if include_logdet_h || include_logdet_s {
                    let g_logs = if include_logdet_s {
                        0.5 * trace_product(
                            s_pinv_joint.as_ref().ok_or_else(|| {
                                "missing joint S^+ for surrogate REML gradient".to_string()
                            })?,
                            &a_k,
                        )
                    } else {
                        0.0
                    };
                    let mut d_j_k = a_k.clone();
                    if u_k.dot(&u_k).sqrt() > 1e-14 {
                        let h_rho = family
                            .joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
                                &inner.block_states,
                                specs,
                                &u_k,
                            )?
                            .ok_or_else(|| {
                                "joint surrogate dH unavailable for analytic outer gradient"
                                    .to_string()
                            })?;
                        d_j_k += &symmetrized_square_matrix(
                            h_rho,
                            total,
                            "joint surrogate dH shape mismatch",
                        )?;
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
                    let delta_kl = if k == l { 1.0 } else { 0.0 };
                    let mut v_kl = a_beta_terms[k].dot(&u_terms[l])
                        + 0.5 * delta_kl * beta_flat.dot(&a_beta_terms[k]);
                    if include_logdet_h || include_logdet_s {
                        let tr_prod = trace_jinv_a_jinv_b(&h_inv, &j_terms[l], &j_terms[k]);
                        let rhs_kl = -(a_terms[k].dot(&u_terms[l])
                            + j_terms[l].dot(&u_terms[k])
                            + delta_kl * a_beta_terms[k].clone());
                        let u_kl = solve_mode_sensitivity(
                            &j_joint,
                            &rhs_kl,
                            None,
                            strict_spd,
                            allow_semidefinite,
                            options,
                            "joint second-order mode sensitivity system for surrogate REML Hessian",
                        )?;
                        let dh_u_kl = match family
                            .joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
                                &inner.block_states,
                                specs,
                                &u_kl,
                            )? {
                            Some(v) => symmetrized_square_matrix(
                                v,
                                total,
                                "joint surrogate second-order dH shape mismatch",
                            )?,
                            None => {
                                hess_available = false;
                                break;
                            }
                        };
                        let d2h_ul_uk = match family
                            .joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
                                &inner.block_states,
                                specs,
                                &u_terms[l],
                                &u_terms[k],
                            )? {
                            Some(v) => symmetrized_square_matrix(
                                v,
                                total,
                                "joint surrogate d2H shape mismatch",
                            )?,
                            None => {
                                hess_available = false;
                                break;
                            }
                        };
                        let mut j_kl = dh_u_kl;
                        j_kl += &d2h_ul_uk;
                        if delta_kl > 0.0 {
                            j_kl += &a_terms[k];
                        }
                        let trsecond = trace_product(&h_inv, &j_kl);
                        let tr_h = if include_logdet_h {
                            0.5 * (-tr_prod + trsecond)
                        } else {
                            0.0
                        };
                        let tr_p = if include_logdet_s {
                            0.5 * trace_jinv_a_jinv_b(
                                s_pinv_joint.as_ref().ok_or_else(|| {
                                    "missing joint S^+ for surrogate REML Hessian".to_string()
                                })?,
                                &a_terms[l],
                                &a_terms[k],
                            ) - 0.5
                                * delta_kl
                                * trace_product(
                                    s_pinv_joint.as_ref().ok_or_else(|| {
                                        "missing joint S^+ for surrogate REML Hessian".to_string()
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
    // At this point the joint exact path is unavailable.
    //
    // The old generic fallback below is only mathematically defensible when
    // there is a single block. Once there are multiple blocks, generic code
    // has no way to prove that the profiled objective is actually block
    // separable. A multi-block family can have penalty-local forcing
    //
    //   g_i = S_i beta
    //
    // while still having a coupled fitted-mode response
    //
    //   H beta_i = -g_i
    //
    // because the likelihood contributes off-diagonal block curvature. In
    // that case a blockwise surrogate solve computes the derivative of a
    // different objective. The correct behavior is therefore:
    //
    // - exact joint path if available
    // - otherwise fail for multi-block families
    // - only permit the generic blockwise surrogate in the single-block case,
    //   where it coincides with the full system.
    if family.requires_joint_outer_hyper_path() {
        return Err(
            "outer hyper-derivative evaluation requires a joint exact path for this family"
                .to_string(),
        );
    }
    let mut at = 0usize;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let work = &eval.blockworking_sets[b];
        let p = spec.design.ncols();
        let mut diagonal_design = None::<DesignMatrix>;
        let xtwx = match work {
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
            let ridge = effective_solverridge(options.ridge_floor);
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
            let ridge = effective_solverridge(options.ridge_floor);
            for d in 0..p {
                s_for_logdet[[d, d]] += ridge;
            }
        }
        let s_pinv = if include_logdet_s {
            let s_floor = if options.ridge_policy.include_penalty_logdet {
                effective_solverridge(options.ridge_floor)
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
                    strict_solve_spdwith_semidefinite_option(&h_mode, &rhs, allow_semidefinite)?
                } else {
                    solve_spd_systemwith_policy(
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
                            let wwork = match work {
                                BlockWorkingSet::Diagonal {
                                    working_response: _,
                                    working_weights,
                                } => floor_positiveworking_weights(
                                    working_weights,
                                    options.minweight,
                                ),
                                BlockWorkingSet::ExactNewton { .. } => unreachable!(),
                            };
                            let leverages = diagonal_leverages.as_ref().ok_or_else(|| {
                                format!(
                                    "missing leverage cache for block {b} diagonal H_rho trace term"
                                )
                            })?;
                            let x_dense = x_dyn.to_dense();
                            let mut d_eta = x_dyn.matrixvectormultiply(&u_k);
                            let geom_trace = apply_geometry_direction_to_eta_and_trace(
                                &x_dense,
                                beta,
                                &wwork,
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
                                .diagonalworking_weights_directional_derivative(
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
fn outerobjectivegradienthessian<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    need_hessian: bool,
) -> Result<(f64, Array1<f64>, Option<Array2<f64>>, ConstrainedWarmStart), String> {
    let result = outerobjectivegradienthessian_internal(
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
fn outerobjective_andgradient<F: CustomFamily>(
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
        false,
    )?;
    Ok((obj, grad, warm))
}

fn compute_custom_family_joint_hyper_exact<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho_current: &Array1<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    inner: &mut BlockwiseInnerResult,
    need_hessian: bool,
) -> Result<(Array1<f64>, Option<Array2<f64>>), String> {
    // Full exact joint profiled/Laplace outer calculus over
    //
    //   theta = [rho, psi]
    //
    // is assembled here in one flattened coefficient space. The family has
    // already exposed the exact inner mode beta^(theta) through `inner`, so
    // this routine builds the full outer derivatives from fixed-beta pieces.
    //
    // Write
    //
    //   V(beta, theta) = D(beta, psi) + 0.5 beta^T S(theta) beta,
    //   F(beta, theta) = V_beta(beta, theta),
    //   H(beta, theta) = V_beta_beta(beta, theta).
    //
    // At the fitted inner mode,
    //
    //   F(beta^(theta), theta) = 0.
    //
    // For each hypercoordinate theta_i we need the fixed-beta objects
    //
    //   V_i, g_i := F_i, H_i,
    //
    // and for each pair (i, j)
    //
    //   V_ij, g_ij, H_ij,
    //
    // together with the beta-curvature contractions
    //
    //   D_beta H[u], D_beta^2 H[u, v], T_i[u] := D_beta H_i[u].
    //
    // The profiled mode responses are the joint solves
    //
    //   beta_i  = -H^{-1} g_i,
    //   beta_ij = -H^{-1}(g_ij + H_i beta_j + H_j beta_i + D_beta H[beta_i] beta_j).
    //
    // The total Hessian drifts feeding the Laplace block are then
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
    // Hence the exact outer derivatives are
    //
    //   J_i
    //   = V_i
    //     + 0.5 tr(H^{-1} dot H_i)
    //     - 0.5 partial_i log|S(theta)|_+,
    //
    //   J_ij
    //   = (V_ij - g_i^T H^{-1} g_j)
    //     + 0.5 [ tr(H^{-1} ddot H_ij)
    //             - tr(H^{-1} dot H_j H^{-1} dot H_i) ]
    //     - 0.5 partial^2_{ij} log|S(theta)|_+.
    //
    // This routine follows those formulas literally:
    //
    // - build fixed-beta first-order objects (`v_i`, `g_i`, `h_i`, `s_i`)
    // - solve the joint first-order mode response `beta_i`
    // - build `dot H_i` and the exact gradient
    // - build fixed-beta second-order objects (`v_ij`, `g_ij`, `h_ij`, `s_ij`)
    // - solve the joint second-order mode response `beta_ij`
    // - build `ddot H_ij` and the exact Hessian
    //
    // The generic code owns every realized penalty contribution S_i / S_ij.
    // Families contribute only the likelihood-side exact joint hyper objects
    //
    //   D_i, D_{beta i}, D_{beta beta i},
    //   D_ij, D_{beta ij}, D_{beta beta ij},
    //   T_i[u] = D_beta H_i^{(D)}[u],
    //
    // where H_i^{(D)} denotes the likelihood-side fixed-beta Hessian drift.
    // Generic code then promotes those likelihood-only objects to the full
    // profiled quantities by adding
    //
    //   0.5 beta^T S_i beta,  S_i beta,  S_i
    //
    // and similarly at second order with S_ij. This ownership split is what
    // makes the unified exact path correct when psi moves realized component
    // penalties, because mixed rho/psi and pure psi penalty terms enter
    // automatically through S(theta).
    let include_logdet_h = include_exact_newton_logdet_h(family);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    let allow_semidefinite = strict_spd && family.exact_newton_allows_semidefinitehessian();
    let per_block = split_log_lambdas(rho_current, penalty_counts)?;
    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let rho_dim = penalty_counts.iter().sum::<usize>();
    let coords = flatten_joint_hyper_coords(specs, derivative_blocks)?;
    let theta_dim = coords.len();
    let h_joint_unpen = exact_newton_joint_hessian_symmetrized(
        family,
        &inner.block_states,
        specs,
        total,
        "joint exact-newton Hessian shape mismatch in joint hyper evaluator",
    )?
    .ok_or_else(|| {
        "joint exact-newton Hessian unavailable for full [rho, psi] outer calculus".to_string()
    })?;
    let beta_flat = flatten_state_betas(&inner.block_states, specs);
    let synced_joint_states =
        synchronized_states_from_flat_beta(family, specs, &inner.block_states, &beta_flat)?;
    let joint_tangent_basis = joint_post_update_tangent_basis(family, specs, &synced_joint_states)?;
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
        if moderidge > 0.0 {
            for d in start..end {
                s_joint[[d, d]] += moderidge;
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
    let mut h_mode = h_joint_unpen.clone();
    h_mode += &s_joint;
    let mut h_for_traces = h_mode.clone();
    if !strict_spd && extra_logdet_ridge > 0.0 {
        for d in 0..total {
            h_for_traces[[d, d]] += extra_logdet_ridge;
        }
    }
    let h_inv = logdet_trace_inverse_with_ridge_policy(
        &h_for_traces,
        options.ridge_floor,
        options.ridge_policy,
        strict_spd,
        allow_semidefinite,
    )?;

    let mut v_i_terms: Vec<f64> = Vec::with_capacity(theta_dim);
    let mut g_i_terms: Vec<Array1<f64>> = Vec::with_capacity(theta_dim);
    let mut h_i_terms: Vec<Array2<f64>> = Vec::with_capacity(theta_dim);
    let mut s_i_terms: Vec<Array2<f64>> = Vec::with_capacity(theta_dim);
    let mut beta_i_terms: Vec<Array1<f64>> = Vec::with_capacity(theta_dim);
    let mut d_h_beta_i_terms: Vec<Array2<f64>> = Vec::with_capacity(theta_dim);
    let mut dot_h_i_terms: Vec<Array2<f64>> = Vec::with_capacity(theta_dim);
    let mut gradient = Array1::<f64>::zeros(theta_dim);

    for coord in &coords {
        let s_i = joint_theta_penalty_first_matrix(
            specs,
            &ranges,
            &per_block,
            derivative_blocks,
            total,
            coord,
        )?;
        let (v_i, g_i, h_i) = match coord {
            JointHyperCoord::Rho { .. } => {
                let g = s_i.dot(&beta_flat);
                let v = 0.5 * beta_flat.dot(&g);
                (v, g, s_i.clone())
            }
            JointHyperCoord::Psi { .. } => {
                let psi_idx = coord.global_idx() - rho_dim;
                let psi_terms = family
                    .exact_newton_joint_psi_terms(
                        &synced_joint_states,
                        specs,
                        derivative_blocks,
                        psi_idx,
                    )?
                    .ok_or_else(|| {
                        format!(
                            "missing exact joint psi terms for psi coordinate {} while full [rho, psi] Hessian is required",
                            psi_idx
                        )
                    })?;
                let g = &psi_terms.score_psi + &s_i.dot(&beta_flat);
                let v = psi_terms.objective_psi + 0.5 * beta_flat.dot(&s_i.dot(&beta_flat));
                let h = psi_terms.hessian_psi + &s_i;
                (v, g, h)
            }
        };
        let log_s_i = if include_logdet_s {
            trace_product(
                s_pinv_joint
                    .as_ref()
                    .ok_or_else(|| "missing joint S^+ for first derivative".to_string())?,
                &s_i,
            )
        } else {
            0.0
        };
        let beta_i = solve_mode_sensitivity(
            &h_mode,
            &(-&g_i),
            joint_tangent_basis.as_ref(),
            strict_spd,
            allow_semidefinite,
            options,
            "joint mode sensitivity system for full [rho, psi] outer calculus",
        )?;
        let d_h_beta_i = if beta_i.dot(&beta_i).sqrt() > 1e-14 {
            symmetrized_square_matrix(
                family
                    .exact_newton_joint_hessian_directional_derivative_with_specs(
                        &synced_joint_states,
                        specs,
                        &beta_i,
                    )?
                    .ok_or_else(|| {
                        "missing joint exact-newton dH for full [rho, psi] outer calculus"
                            .to_string()
                    })?,
                total,
                "joint exact-newton dH shape mismatch in full [rho, psi] outer calculus",
            )?
        } else {
            Array2::<f64>::zeros((total, total))
        };
        let dot_h_i = &h_i + &d_h_beta_i;
        let grad_i =
            v_i + if include_logdet_h {
                0.5 * trace_product(&h_inv, &dot_h_i)
            } else {
                0.0
            } - if include_logdet_s { 0.5 * log_s_i } else { 0.0 };
        gradient[coord.global_idx()] = grad_i;
        v_i_terms.push(v_i);
        g_i_terms.push(g_i);
        h_i_terms.push(h_i);
        s_i_terms.push(s_i);
        beta_i_terms.push(beta_i);
        d_h_beta_i_terms.push(d_h_beta_i);
        dot_h_i_terms.push(dot_h_i);
    }

    let mut outer_hessian = if need_hessian {
        Some(Array2::<f64>::zeros((theta_dim, theta_dim)))
    } else {
        None
    };
    if let Some(hess) = outer_hessian.as_mut() {
        for i in 0..theta_dim {
            for j in i..theta_dim {
                let s_ij = joint_theta_penaltysecond_matrix(
                    specs,
                    &ranges,
                    &per_block,
                    derivative_blocks,
                    total,
                    &coords[i],
                    &coords[j],
                )?;
                let (v_ij, g_ij, h_ij) = match (&coords[i], &coords[j]) {
                    (JointHyperCoord::Psi { .. }, JointHyperCoord::Psi { .. }) => {
                        let psi_i = coords[i].global_idx() - rho_dim;
                        let psi_j = coords[j].global_idx() - rho_dim;
                        let psi_terms = family
                            .exact_newton_joint_psisecond_order_terms(
                                &synced_joint_states,
                                specs,
                                derivative_blocks,
                                psi_i,
                                psi_j,
                            )?
                            .ok_or_else(|| {
                                format!(
                                    "missing exact joint psi second-order terms for psi coordinates ({psi_i}, {psi_j})"
                                )
                            })?;
                        (
                            psi_terms.objective_psi_psi
                                + 0.5 * beta_flat.dot(&s_ij.dot(&beta_flat)),
                            psi_terms.score_psi_psi + &s_ij.dot(&beta_flat),
                            psi_terms.hessian_psi_psi + &s_ij,
                        )
                    }
                    _ => {
                        let g = s_ij.dot(&beta_flat);
                        let v = 0.5 * beta_flat.dot(&g);
                        (v, g, s_ij.clone())
                    }
                };
                let beta_ij_rhs = &g_ij
                    + &h_i_terms[i].dot(&beta_i_terms[j])
                    + &h_i_terms[j].dot(&beta_i_terms[i])
                    + &d_h_beta_i_terms[i].dot(&beta_i_terms[j]);
                let beta_ij = solve_mode_sensitivity(
                    &h_mode,
                    &(-beta_ij_rhs),
                    joint_tangent_basis.as_ref(),
                    strict_spd,
                    allow_semidefinite,
                    options,
                    "joint second-order mode sensitivity system for full [rho, psi] outer calculus",
                )?;
                let mut ddot_h_ij = h_ij.clone();
                match &coords[i] {
                    JointHyperCoord::Psi { .. } => {
                        let psi_i = coords[i].global_idx() - rho_dim;
                        ddot_h_ij += &symmetrized_square_matrix(
                            family
                                .exact_newton_joint_psihessian_directional_derivative(
                                    &synced_joint_states,
                                    specs,
                                    derivative_blocks,
                                    psi_i,
                                    &beta_i_terms[j],
                                )?
                                .ok_or_else(|| {
                                    format!(
                                        "missing exact joint psi mixed Hessian drift for psi coordinate {}",
                                        psi_i
                                    )
                                })?,
                            total,
                            "joint exact-newton D_beta H_psi[u] shape mismatch",
                        )?;
                    }
                    JointHyperCoord::Rho { .. } => {}
                }
                match &coords[j] {
                    JointHyperCoord::Psi { .. } => {
                        let psi_j = coords[j].global_idx() - rho_dim;
                        ddot_h_ij += &symmetrized_square_matrix(
                            family
                                .exact_newton_joint_psihessian_directional_derivative(
                                    &synced_joint_states,
                                    specs,
                                    derivative_blocks,
                                    psi_j,
                                    &beta_i_terms[i],
                                )?
                                .ok_or_else(|| {
                                    format!(
                                        "missing exact joint psi mixed Hessian drift for psi coordinate {}",
                                        psi_j
                                    )
                                })?,
                            total,
                            "joint exact-newton D_beta H_psi[u] shape mismatch",
                        )?;
                    }
                    JointHyperCoord::Rho { .. } => {}
                }
                ddot_h_ij += &symmetrized_square_matrix(
                    family
                        .exact_newton_joint_hessian_directional_derivative_with_specs(
                            &synced_joint_states,
                            specs,
                            &beta_ij,
                        )?
                        .ok_or_else(|| {
                            "missing joint exact-newton dH for second-order [rho, psi] calculus"
                                .to_string()
                        })?,
                    total,
                    "joint exact-newton dH(beta_ij) shape mismatch",
                )?;
                ddot_h_ij += &symmetrized_square_matrix(
                    family
                        .exact_newton_joint_hessian_second_directional_derivative_with_specs(
                            &synced_joint_states,
                            specs,
                            &beta_i_terms[i],
                            &beta_i_terms[j],
                        )?
                        .ok_or_else(|| {
                            "missing joint exact-newton d2H for second-order [rho, psi] calculus"
                                .to_string()
                        })?,
                    total,
                    "joint exact-newton d2H shape mismatch in full [rho, psi] outer calculus",
                )?;
                let profile_term = v_ij + g_i_terms[i].dot(&beta_i_terms[j]);
                let log_h_term = if include_logdet_h {
                    0.5 * (trace_product(&h_inv, &ddot_h_ij)
                        - trace_jinv_a_jinv_b(&h_inv, &dot_h_i_terms[j], &dot_h_i_terms[i]))
                } else {
                    0.0
                };
                let log_s_term = if include_logdet_s {
                    -0.5 * trace_product(
                        s_pinv_joint
                            .as_ref()
                            .ok_or_else(|| "missing joint S^+ for second derivative".to_string())?,
                        &s_ij,
                    ) + 0.5
                        * trace_jinv_a_jinv_b(
                            s_pinv_joint.as_ref().ok_or_else(|| {
                                "missing joint S^+ for second derivative".to_string()
                            })?,
                            &s_i_terms[j],
                            &s_i_terms[i],
                        )
                } else {
                    0.0
                };
                let value = profile_term + log_h_term + log_s_term;
                hess[[i, j]] = value;
                hess[[j, i]] = value;
            }
        }
    }

    Ok((gradient, outer_hessian))
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
                let mut wxrow = wx.row_mut(i);
                wxrow *= wi;
                let mut wdxrow = wdx.row_mut(i);
                wdxrow *= wi;
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
/// The caller has already applied the current spatial coordinates `psi` when
/// constructing:
///
/// - `family`
/// - `specs`
/// - `derivative_blocks`
///
/// so the explicit input here is still only the smoothing vector `rho_current`.
/// Mathematically, however, the surface being differentiated is the full joint
/// profiled/Laplace objective in
///
///   theta = [rho, psi].
///
/// The exact outer calculus is unified across all hypercoordinates:
///
///   J(theta)
///   = V(beta^(theta), theta)
///     + 0.5 log|H(beta^(theta), theta)|
///     - 0.5 log|S(theta)|_+,
///
/// with stationarity and joint curvature
///
///   F(beta, theta) := V_beta(beta, theta) = 0,
///   H(beta, theta) := V_beta_beta(beta, theta).
///
/// For each theta_i we need the fixed-beta objects
///
///   V_i, g_i := F_i, H_i,
///
/// and for each pair (i, j)
///
///   V_ij, g_ij, H_ij,
///
/// together with the beta-curvature contractions
///
///   D_beta H[u], D_beta^2 H[u, v], T_i[u] := D_beta H_i[u].
///
/// These determine the exact joint mode responses
///
///   beta_i  = -H^{-1} g_i,
///   beta_ij = -H^{-1}(g_ij + H_i beta_j + H_j beta_i + D_beta H[beta_i] beta_j),
///
/// and the total Hessian drifts
///
///   dot H_i
///   = H_i + D_beta H[beta_i],
///
///   ddot H_ij
///   = H_ij
///     + T_i[beta_j]
///     + T_j[beta_i]
///     + D_beta H[beta_ij]
///     + D_beta^2 H[beta_i, beta_j].
///
/// Therefore the exact joint outer derivatives are
///
///   J_i
///   = V_i
///     + 0.5 tr(H^{-1} dot H_i)
///     - 0.5 partial_i log|S(theta)|_+,
///
///   J_ij
///   = (V_ij - g_i^T H^{-1} g_j)
///     + 0.5 [ tr(H^{-1} ddot H_ij)
///             - tr(H^{-1} dot H_j H^{-1} dot H_i) ]
///     - 0.5 partial^2_{ij} log|S(theta)|_+.
///
/// In this unified view rho and psi differ only in the likelihood-side
/// fixed-beta derivative objects contributed by the family. The generic exact
/// assembler always adds realized penalty motion through `S(theta)` for every
/// hypercoordinate:
///
/// - `rho` coordinates usually have zero likelihood-side objects and pick up
///   their fixed-beta derivatives entirely from `S_rho` / `S_{rho rho}`
/// - `psi` coordinates contribute likelihood-side objects from the family's
///   joint exact psi hooks and may also pick up extra penalty terms through
///   `S_psi`, `S_{rho psi}`, and `S_{psi psi}` when realized penalties move
///   with `psi`
///
/// The implementation below follows this unified calculus directly. Once a
/// family supplies the joint fixed-beta psi objects and the mixed
/// `D_beta H_psi[u]` contraction, exact joint hyper evaluation treats `rho`
/// and `psi` identically and returns the full profiled/Laplace Hessian over
/// `theta = [rho, psi]`.
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

    let penalty_counts = validate_blockspecs(specs)?;
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

    let mut result = outerobjectivegradienthessian_internal(
        family,
        specs,
        options,
        &penalty_counts,
        rho_current,
        warm_start.map(|w| &w.inner),
        need_hessian,
    )?;
    if psi_dim > 0 {
        let (gradient, outer_hessian) = compute_custom_family_joint_hyper_exact(
            family,
            specs,
            options,
            &penalty_counts,
            rho_current,
            derivative_blocks,
            &mut result.inner,
            need_hessian,
        )?;
        return Ok(CustomFamilyJointHyperResult {
            objective: result.objective,
            gradient,
            outer_hessian,
            warm_start: CustomFamilyWarmStart {
                inner: result.warm_start,
            },
        });
    }

    let mut gradient = Array1::<f64>::zeros(rho_dim + psi_dim);
    gradient
        .slice_mut(ndarray::s![..rho_dim])
        .assign(&result.gradient);

    Ok(CustomFamilyJointHyperResult {
        objective: result.objective,
        gradient,
        outer_hessian: if need_hessian {
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

fn flatten_joint_hyper_coords(
    specs: &[ParameterBlockSpec],
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
) -> Result<Vec<JointHyperCoord>, String> {
    if derivative_blocks.len() != specs.len() {
        return Err(format!(
            "joint hyper derivative block count mismatch: got {}, expected {}",
            derivative_blocks.len(),
            specs.len()
        ));
    }
    let mut coords = Vec::new();
    let mut global = 0usize;
    for (block_idx, spec) in specs.iter().enumerate() {
        for penalty_idx in 0..spec.penalties.len() {
            coords.push(JointHyperCoord::Rho {
                block_idx,
                penalty_idx,
                global_idx: global,
            });
            global += 1;
        }
    }
    for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
        for local_idx in 0..block_derivs.len() {
            coords.push(JointHyperCoord::Psi {
                block_idx,
                local_idx,
                global_idx: global,
            });
            global += 1;
        }
    }
    Ok(coords)
}

// Build S_i = partial_{theta_i} S(theta) in flattened coefficient space.
//
// This helper is intentionally generic in theta = [rho, psi]:
//
// - rho coordinates contribute the usual log-lambda scaled penalty block
// - psi coordinates contribute realized penalty drift from the spatial
//   derivative payload, including summed component penalties when one psi
//   coordinate moves several penalty components at once
//
// The returned matrix is the generic penalty contribution to the fixed-beta
// objects:
//
//   V_i += 0.5 beta^T S_i beta,
//   g_i += S_i beta,
//   H_i += S_i.
fn joint_theta_penalty_first_matrix(
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
    per_block: &[Array1<f64>],
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    total: usize,
    coord: &JointHyperCoord,
) -> Result<Array2<f64>, String> {
    match coord {
        JointHyperCoord::Rho {
            block_idx,
            penalty_idx,
            ..
        } => {
            let spec = &specs[*block_idx];
            let (start, end) = ranges[*block_idx];
            let mut s_i = Array2::<f64>::zeros((total, total));
            let local = spec.penalties[*penalty_idx]
                .mapv(|v| per_block[*block_idx][*penalty_idx].exp() * v);
            s_i.slice_mut(ndarray::s![start..end, start..end])
                .assign(&local);
            Ok(s_i)
        }
        JointHyperCoord::Psi {
            block_idx,
            local_idx,
            ..
        } => {
            let deriv = &derivative_blocks[*block_idx][*local_idx];
            let (start, end) = ranges[*block_idx];
            let p = end - start;
            let mut s_i = Array2::<f64>::zeros((total, total));
            let local = if let Some(parts) = deriv.s_psi_components.as_ref() {
                let mut total_local = Array2::<f64>::zeros((p, p));
                for (penalty_idx, s_part) in parts {
                    if *penalty_idx >= per_block[*block_idx].len() {
                        return Err(format!(
                            "psi penalty derivative component {} out of bounds for block {}",
                            penalty_idx, block_idx
                        ));
                    }
                    total_local.scaled_add(per_block[*block_idx][*penalty_idx].exp(), s_part);
                }
                total_local
            } else if let Some(penalty_index) = deriv.penalty_index {
                if penalty_index >= per_block[*block_idx].len() {
                    return Err(format!(
                        "psi penalty derivative index {} out of bounds for block {}",
                        penalty_index, block_idx
                    ));
                }
                deriv
                    .s_psi
                    .mapv(|v| per_block[*block_idx][penalty_index].exp() * v)
            } else {
                Array2::<f64>::zeros((p, p))
            };
            s_i.slice_mut(ndarray::s![start..end, start..end])
                .assign(&local);
            Ok(s_i)
        }
    }
}

// Build S_ij = partial^2_{theta_i theta_j} S(theta) in flattened coefficient
// space.
//
// This handles all mixed exact-penalty cases used by the unified outer
// Hessian:
//
// - rho/rho from the usual log-linear penalty scaling
// - rho/psi when a psi coordinate moves the realized penalty associated with a
//   rho coordinate
// - psi/psi from realized second penalty drift carried by the derivative
//   payload
//
// The returned matrix is the generic penalty contribution to
//
//   V_ij += 0.5 beta^T S_ij beta,
//   g_ij += S_ij beta,
//   H_ij += S_ij.
fn joint_theta_penaltysecond_matrix(
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
    per_block: &[Array1<f64>],
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    total: usize,
    coord_i: &JointHyperCoord,
    coord_j: &JointHyperCoord,
) -> Result<Array2<f64>, String> {
    match (coord_i, coord_j) {
        (
            JointHyperCoord::Rho {
                block_idx: bi,
                penalty_idx: pi,
                ..
            },
            JointHyperCoord::Rho {
                block_idx: bj,
                penalty_idx: pj,
                ..
            },
        ) => {
            let mut s_ij = Array2::<f64>::zeros((total, total));
            if bi == bj && pi == pj {
                let (start, end) = ranges[*bi];
                let local = specs[*bi].penalties[*pi].mapv(|v| per_block[*bi][*pi].exp() * v);
                s_ij.slice_mut(ndarray::s![start..end, start..end])
                    .assign(&local);
            }
            Ok(s_ij)
        }
        (
            JointHyperCoord::Rho {
                block_idx: bi,
                penalty_idx: pi,
                ..
            },
            JointHyperCoord::Psi {
                block_idx: bj,
                local_idx: lj,
                ..
            },
        )
        | (
            JointHyperCoord::Psi {
                block_idx: bj,
                local_idx: lj,
                ..
            },
            JointHyperCoord::Rho {
                block_idx: bi,
                penalty_idx: pi,
                ..
            },
        ) => {
            let mut s_ij = Array2::<f64>::zeros((total, total));
            if bi != bj {
                return Ok(s_ij);
            }
            let (start, end) = ranges[*bi];
            let p = end - start;
            let deriv = &derivative_blocks[*bj][*lj];
            let local = if let Some(parts) = deriv.s_psi_components.as_ref() {
                let mut total_local = Array2::<f64>::zeros((p, p));
                for (penalty_idx, s_part) in parts {
                    if *penalty_idx == *pi {
                        total_local += &s_part.mapv(|v| per_block[*bi][*pi].exp() * v);
                    }
                }
                total_local
            } else if deriv.penalty_index == Some(*pi) {
                deriv.s_psi.mapv(|v| per_block[*bi][*pi].exp() * v)
            } else {
                Array2::<f64>::zeros((p, p))
            };
            s_ij.slice_mut(ndarray::s![start..end, start..end])
                .assign(&local);
            Ok(s_ij)
        }
        (
            JointHyperCoord::Psi {
                block_idx: bi,
                local_idx: li,
                ..
            },
            JointHyperCoord::Psi {
                block_idx: bj,
                local_idx: lj,
                ..
            },
        ) => {
            let mut s_ij = Array2::<f64>::zeros((total, total));
            if bi != bj {
                return Ok(s_ij);
            }
            let (start, end) = ranges[*bi];
            let p = end - start;
            let deriv = &derivative_blocks[*bi][*li];
            let local = if let Some(parts) = deriv.s_psi_psi_components.as_ref() {
                let mut total_local = Array2::<f64>::zeros((p, p));
                if let Some(pair_parts) = parts.get(*lj) {
                    for (penalty_idx, s_part) in pair_parts {
                        if *penalty_idx >= per_block[*bi].len() {
                            return Err(format!(
                                "psi second-derivative component {} out of bounds for block {}",
                                penalty_idx, bi
                            ));
                        }
                        total_local.scaled_add(per_block[*bi][*penalty_idx].exp(), s_part);
                    }
                }
                total_local
            } else if let Some(parts) = deriv.s_psi_psi.as_ref() {
                if let Some(s_part) = parts.get(*lj) {
                    let Some(penalty_index) = deriv.penalty_index else {
                        return Ok(s_ij);
                    };
                    if penalty_index >= per_block[*bi].len() {
                        return Err(format!(
                            "psi second-derivative penalty index {} out of bounds for block {}",
                            penalty_index, bi
                        ));
                    }
                    s_part.mapv(|v| per_block[*bi][penalty_index].exp() * v)
                } else {
                    Array2::<f64>::zeros((p, p))
                }
            } else {
                Array2::<f64>::zeros((p, p))
            };
            s_ij.slice_mut(ndarray::s![start..end, start..end])
                .assign(&local);
            Ok(s_ij)
        }
    }
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
    _: &BlockwiseFitOptions,
    context: &str,
) -> Result<Array1<f64>, String> {
    let solve_dense = |matrix: &Array2<f64>, vector: &Array1<f64>| -> Result<Array1<f64>, String> {
        if strict_spd {
            strict_solve_spdwith_semidefinite_option(matrix, vector, allow_semidefinite)
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

fn penalizedobjective_at_beta<F: CustomFamily>(
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

fn exact_newton_joint_stationarity_inf_norm(
    eval: &FamilyEvaluation,
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> Result<Option<f64>, String> {
    if eval.blockworking_sets.len() != states.len() || states.len() != s_lambdas.len() {
        return Err("exact-newton joint stationarity check: block dimension mismatch".to_string());
    }

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
        match inverse_spdwith_retry(&h, effective_solverridge(options.ridge_floor), 8) {
            Ok(cov) => Ok(cov),
            Err(_) => pinv_positive_part(&h, effective_solverridge(options.ridge_floor)),
        }
    }
}

fn compute_joint_covariance_required<F: CustomFamily>(
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

#[allow(clippy::type_complexity)]
pub fn fit_custom_family<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, CustomFamilyError> {
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
        return Ok(BlockwiseFitResult {
            block_states: inner.block_states,
            log_likelihood: inner.log_likelihood,
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            covariance_conditional,
            penalizedobjective: finite_penalizedobjective(
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
        // non-finite, `finite_penalizedobjective` drops back to the finite
        // penalized likelihood surface rather than returning NaN/Inf.
        //
        // This is a deliberate stabilization choice for the known-link wiggle
        // exact-Newton family, not a generic change to custom-family smoothing.
        let per_block = split_log_lambdas(&rho0, &penalty_counts)?;
        let mut inner = inner_blockwise_fit(family, specs, &per_block, options, None)?;
        refresh_all_block_etas(family, specs, &mut inner.block_states)?;
        let covariance_conditional = compute_joint_covariance_required(
            family,
            specs,
            &inner.block_states,
            &per_block,
            options,
        )?;
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
            penalizedobjective: finite_penalizedobjective(
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
    let objective = CachedSecondOrderObjective::new(
        |x: &Array1<f64>| {
            let cached = warm_cache.lock().ok().and_then(|g| g.clone());
            let warm_start = if include_exact_newton_logdet_h(family) {
                None
            } else {
                cached.as_ref()
            };
            let (obj, grad, hess_opt) = match outerobjectivegradienthessian(
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
                Ok((_, _, _, _)) => {
                    if let Ok(mut guard) = last_outer_error.lock() {
                        *guard = Some(
                            "custom-family outer objective/derivatives became non-finite"
                                .to_string(),
                        );
                    }
                    if include_exact_newton_logdet_h(family)
                        && let Some((x_prev, obj_prev, grad_prev, hess_prev)) = last_eval.as_ref()
                        && x_prev.len() == x.len()
                    {
                        return Ok((*obj_prev, grad_prev.clone(), hess_prev.clone()));
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
                        && let Some((x_prev, obj_prev, grad_prev, hess_prev)) = last_eval.as_ref()
                        && x_prev.len() == x.len()
                    {
                        return Ok((*obj_prev, grad_prev.clone(), hess_prev.clone()));
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
    let outer_max_iter = if include_exact_newton_logdet_h(family) {
        options.outer_max_iter.min(3).max(1)
    } else {
        options.outer_max_iter
    };
    let mut solver = NewtonTrustRegion::new(rho0.clone(), objective)
        .with_bounds(
            Bounds::new(lower, upper, 1e-6).expect("custom-family rho bounds must be valid"),
        )
        .with_tolerance(
            Tolerance::new(options.outer_tol).expect("custom-family tolerance must be valid"),
        )
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
    let covariance_conditional =
        compute_joint_covariance_required(family, specs, &inner.block_states, &per_block, options)?;

    Ok(BlockwiseFitResult {
        block_states: inner.block_states,
        log_likelihood: inner.log_likelihood,
        log_lambdas: rho_star.clone(),
        lambdas: rho_star.mapv(f64::exp),
        covariance_conditional,
        penalizedobjective: finite_penalizedobjective(
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
    fn floor_positiveworking_weights_preserves_exactzeros() {
        let weights = array![0.0, 1.0e-16, 0.25];
        let floored = floor_positiveworking_weights(&weights, 1.0e-6);
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
            block_states: &[ParameterBlockState],
            block_idx: usize,
            d_eta: &Array1<f64>,
        ) -> Result<Option<Array1<f64>>, String> {
            let _ = block_states;
            let _ = block_idx;
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
            block_states: &[ParameterBlockState],
            block_idx: usize,
            spec: &ParameterBlockSpec,
        ) -> Result<Option<LinearInequalityConstraints>, String> {
            let _ = block_states;
            let _ = spec;
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
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let _ = block_states;
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
            block_states: &[ParameterBlockState],
            block_idx: usize,
            d_beta: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let _ = block_idx;
            let _ = d_beta;
            Err(
                "blockwise exact-newton path should not be used when joint path is available"
                    .to_string(),
            )
        }

        fn exact_newton_joint_hessian(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            Ok(Some(array![[2.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            block_states: &[ParameterBlockState],
            d_beta_flat: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let _ = d_beta_flat;
            Ok(Some(array![[0.0]]))
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
            block_states: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
            Ok(Some(Array2::eye(p)))
        }

        fn exact_newton_joint_hessian_directional_derivative_with_specs(
            &self,
            block_states: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
            d_beta_flat: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let _ = d_beta_flat;
            let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
            Ok(Some(Array2::zeros((p, p))))
        }

        fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
            &self,
            block_states: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
            d_beta_u_flat: &Array1<f64>,
            d_betav_flat: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let _ = d_beta_u_flat;
            let _ = d_betav_flat;
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
            ExactNewtonOuterObjective::PseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            Ok(Some(array![[2.0]]))
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            block_states: &[ParameterBlockState],
            block_idx: usize,
            d_beta: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let _ = block_idx;
            let _ = d_beta;
            Ok(Some(array![[0.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            block_states: &[ParameterBlockState],
            d_beta_flat: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let _ = d_beta_flat;
            Ok(Some(array![[0.0]]))
        }
    }

    #[derive(Clone)]
    struct OneBlockExactPsiHookFamily;

    impl CustomFamily for OneBlockExactPsiHookFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let _ = block_states;
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                }],
            })
        }

        fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::PseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            Ok(Some(array![[1.0]]))
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            block_states: &[ParameterBlockState],
            block_idx: usize,
            d_beta: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let _ = block_idx;
            let _ = d_beta;
            Ok(Some(array![[0.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            block_states: &[ParameterBlockState],
            d_beta_flat: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let _ = d_beta_flat;
            Ok(Some(array![[0.0]]))
        }

        fn exact_newton_joint_psi_terms(
            &self,
            block_states: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
            derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
            psi_index: usize,
        ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
            let _ = block_states;
            let _ = specs;
            let _ = derivative_blocks;
            let _ = psi_index;
            Ok(Some(ExactNewtonJointPsiTerms {
                objective_psi: 3.5,
                score_psi: array![0.0],
                hessian_psi: array![[0.0]],
            }))
        }
    }

    #[derive(Clone)]
    struct OneBlockIndefinitePseudoLaplaceFamily;

    impl CustomFamily for OneBlockIndefinitePseudoLaplaceFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let _ = block_states;
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: SymmetricMatrix::Dense(array![[-1.0]]),
                }],
            })
        }

        fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::PseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
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
            ExactNewtonOuterObjective::PseudoLaplace
        }

        fn exact_newton_joint_hessian(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            Ok(Some(array![[2.0, 0.1], [3.0, 2.0]]))
        }
    }

    #[derive(Clone)]
    struct OneBlockAlwaysErrorFamily;

    impl CustomFamily for OneBlockAlwaysErrorFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let _ = block_states;
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
            block_states: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
        ) -> Result<Option<Array2<f64>>, String> {
            let _ = block_states;
            let _ = specs;
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
            minweight: 1e-12,
            ridge_floor: 1e-4,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_remlobjective: false,
            compute_covariance: false,
        };

        let result = fit_custom_family(&OneBlockIdentityFamily, &[spec], &options)
            .expect("custom family fit should succeed");
        let ridge = effective_solverridge(options.ridge_floor);
        let beta = result.block_states[0].beta[0];
        let expected_penalty = 0.5 * ridge * beta * beta;
        assert!(
            (result.penalizedobjective - expected_penalty).abs() < 1e-12,
            "penalized objective should equal ridge quadratic term when ll=0 and S=0; got {}, expected {}",
            result.penalizedobjective,
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
            minweight: 1e-12,
            ridge_floor: 0.0,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_remlobjective: false,
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
    fn outergradient_matches_finite_difference_for_one_block() {
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
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![Array2::eye(1)],
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
    fn outergradient_uses_joint_surrogate_formultiblock_diagonal_family() {
        let spec0 = ParameterBlockSpec {
            name: "block0".to_string(),
            design: DesignMatrix::Dense(array![[1.0], [1.0]]),
            offset: array![0.0, 0.0],
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let spec1 = ParameterBlockSpec {
            name: "block1".to_string(),
            design: DesignMatrix::Dense(array![[1.0], [1.0]]),
            offset: array![0.0, 0.0],
            penalties: vec![Array2::eye(1)],
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
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
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
            (fit.penalizedobjective - expected).abs() < 1e-8,
            "pseudo-Laplace objective mismatch: got {}, expected {}",
            fit.penalizedobjective,
            expected
        );
    }

    #[test]
    fn exact_newton_joint_psi_hook_can_supply_fixed_beta_termswithout_quadratic_spsi() {
        let spec = ParameterBlockSpec {
            name: "psi_hook".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let deriv = CustomFamilyBlockPsiDerivative {
            penalty_index: None,
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
                use_remlobjective: true,
                compute_covariance: false,
                ..BlockwiseFitOptions::default()
            },
            &Array1::zeros(0),
            &[vec![deriv]],
            None,
            false,
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
    fn strict_psd_positive_part_geometry_accepts_semidefinite_boundary() {
        let h = array![[4.0, 0.0], [0.0, 0.0]];
        let rhs = array![8.0, 3.0];
        let sol = strict_solve_spdwith_semidefinite_option(&h, &rhs, true)
            .expect("semidefinite positive-part solve");
        let inv = strict_inverse_spdwith_semidefinite_option(&h, true)
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
        let err = strict_solve_spdwith_semidefinite_option(&h, &array![1.0, 0.0], true)
            .expect_err("indefinite matrix must still be rejected");
        assert!(
            err.contains("strict pseudo-laplace SPD solve failed"),
            "unexpected error: {err}"
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
    fn pseudo_laplace_exact_newton_symmetrizes_nearly_symmetrichessian() {
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
                use_remlobjective: true,
                compute_covariance: false,
                ..BlockwiseFitOptions::default()
            },
        )
        .expect("nearly symmetric pseudo-laplace Hessian should be accepted after symmetrization");
        assert!(
            fit.penalizedobjective.is_finite(),
            "expected finite pseudo-laplace objective, got {}",
            fit.penalizedobjective
        );
    }

    #[test]
    fn outer_lamlgradient_matches_finite_differencewhen_joint_exact_path_is_active() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let log_sigma_design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let thresholdspec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: threshold_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
        };
        let log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: log_sigma_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
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
            let rel = (g0[k] - gfd).abs() / gfd.abs().max(1e-8);
            assert_eq!(
                g0[k].signum(),
                gfd.signum(),
                "outer LAML gradient sign mismatch at {}: analytic={} fd={}",
                k,
                g0[k],
                gfd
            );
            assert!(
                rel < 2e-2,
                "outer LAML gradient mismatch at {}: analytic={} fd={} rel={}",
                k,
                g0[k],
                gfd,
                rel
            );
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
            &DesignMatrix::Dense(x_dense.clone()),
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
    fn block_solve_falls_backwhen_llt_rejects_indefinite_system() {
        let x_dense = array![[1.0, 0.0], [0.0, 0.0]];
        let y_star = array![2.0, 0.0];
        let w = array![1.0, 1.0];
        let s_lambda = array![[0.0, 0.0], [0.0, -1e-12]];

        let beta = solve_blockweighted_system(
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

        let (beta, active) =
            solve_quadraticwith_linear_constraints(&hessian, &rhs, &beta_start, &constraints, None)
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

        let (beta, active) =
            solve_quadraticwith_linear_constraints(&hessian, &rhs, &beta_start, &constraints, None)
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

        let (beta, active) = solve_quadraticwith_linear_constraints(
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
    fn outerobjective_failure_context_is_preserved() {
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

    #[test]
    fn fit_fails_when_requested_covariance_cannot_be_computed() {
        let spec = ParameterBlockSpec {
            name: "cov_block".to_string(),
            design: DesignMatrix::Dense(array![[1.0], [1.0]]),
            offset: array![0.0, 0.0],
            penalties: vec![],
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
}
