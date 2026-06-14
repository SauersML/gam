//! The `CustomFamily` trait itself plus the evaluation result structs it returns
//! (`FamilyEvaluation`, joint-gradient/batched-term carriers) and the eval-scope /
//! outer-eval-context enums that parameterize trait calls.

use super::*;

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
    /// Family-owned fingerprint for persistent coefficient warm-starts.
    ///
    /// The generic block specs contain design matrices, offsets, penalties,
    /// and dimensions, but they deliberately do not know the family response
    /// vector or likelihood-side data stored on `Self`. Reusing β across
    /// different responses is mathematically unsafe, so persistent block-level
    /// warm-starts are enabled only for families that provide a fingerprint of
    /// the data that defines their likelihood. Outer ρ cache remains available
    /// independently through `BlockwiseFitOptions::cache_session`.
    fn persistent_warm_start_fingerprint(
        &self,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Option<String> {
        assert_valid_blockspecs(specs, "persistent warm-start fingerprint");
        assert_valid_options(options, "persistent warm-start fingerprint");
        None
    }

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
    /// on which row subset is being evaluated. Large-scale outer-only
    /// callers (including the joint-Newton line-search screening path) can
    /// override this to evaluate a deterministic paired Horvitz-Thompson
    /// estimate without constructing a full exact-Newton workspace.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        assert_valid_options(options, "log_likelihood_only_with_options");
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

    /// Whether the outer REML/LAML logdet term `½ log|H + Sλ|` and its analytic
    /// trace gradient `½ tr((H+Sλ)⁺ ∂Sλ)` are evaluated over the FULL
    /// identifiable subspace `range(H + Sλ)` (mgcv's generalized determinant,
    /// gam#752) rather than the penalty-range subspace `range(Sλ)`.
    ///
    /// This is a value/gradient SUBSPACE-CONSISTENCY concern, orthogonal to
    /// whether the Hessian depends on β (`exact_newton_joint_hessian_beta_dependent`,
    /// which gates the *drift* corrections). The previous code conflated the two
    /// by gating the projected logdet on β-dependence, so `RidgedQuadraticReml`
    /// families (survival/bernoulli marginal-slope) silently used the
    /// `range(Sλ)`-only determinant: on a near-collinear penalty-null trend (the
    /// clustered-PC matern marginal-slope geometry) that DROPS the penalty-null
    /// likelihood determinant `log|U_kᵀ H U_k|` from the value while
    /// `½ log|Sλ|₊` is correctly over `range(Sλ)`, making the ρ-derivative of the
    /// REML criterion inconsistent. The outer optimizer then drives that block's
    /// λ → ∞ and the envelope gradient (valid only at a stationary β̂) freezes —
    /// the constant-‖g‖ outer stall in gam#808/#787.
    ///
    /// The generalized determinant is the correct objective in ALL cases: when
    /// `H + Sλ` is full rank it equals the ordinary logdet (the projection is a
    /// no-op, so the correction is ≈0), and when it is rank-deficient it drops
    /// only the truly unidentified `ker(H) ∩ ker(Sλ)` directions — exactly the
    /// directions `½ log|Sλ|₊` also omits, keeping value and gradient over one
    /// subspace. Always enabled by default.
    fn use_projected_penalty_logdet(&self) -> bool {
        true
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
    ///   directional derivatives (CTN at large scale) may instead return
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
    ///
    /// **Capability vs representation.** This method reports the highest
    /// analytic order this family implements. The realized policy carries
    /// work estimates for dense/operator routing and staged κ schedules, but
    /// those estimates do not downgrade a second-order family to a first-order
    /// optimizer.
    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        assert!(std::mem::size_of_val(options) > 0);
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

    /// Realized outer-derivative policy at the current problem size.
    ///
    /// Combines the capability query [`Self::exact_outer_derivative_order`]
    /// with predicted per-eval costs from [`Self::coefficient_gradient_cost`] /
    /// [`Self::coefficient_hessian_cost`] and the joint outer-coordinate
    /// dimension `rho_dim + psi_dim`. Capability decides derivative order;
    /// predicted costs inform dense/operator routing and staged κ schedules.
    ///
    /// Families with non-generic cost models (Khatri–Rao CTN, matrix-free
    /// HVP families, marginal-slope row-third workloads) should override
    /// this directly and set the `predicted_*_work` fields from their own
    /// cost model. The default uses the generic
    /// `n × (rho_dim + psi_dim) × p_total` shape via
    /// [`default_outer_derivative_policy_costs`].
    fn outer_derivative_policy(
        &self,
        specs: &[ParameterBlockSpec],
        psi_dim: usize,
        options: &BlockwiseFitOptions,
    ) -> OuterDerivativePolicy {
        let capability = self.exact_outer_derivative_order(specs, options);
        let grad_cost = self.coefficient_gradient_cost(specs);
        let hess_cost = self.coefficient_hessian_cost(specs);
        let (predicted_gradient_work, predicted_hessian_work) =
            default_outer_derivative_policy_costs(specs, psi_dim, grad_cost, hess_cost);
        OuterDerivativePolicy {
            capability,
            predicted_gradient_work,
            predicted_hessian_work,
            subsample_capable: self.outer_derivative_subsample_capable(),
        }
    }

    /// Whether this family's outer-only paths honour HT-weighted partial sums
    /// over `options.outer_score_subsample`.
    ///
    /// Default `false`: the trait's default outer-only paths
    /// (`log_likelihood_only_with_options`,
    /// `exact_newton_joint_psi_workspace_with_options`, ...) forward to the
    /// no-options variants and ignore `outer_score_subsample`. Families that
    /// override those hooks to honour HT-weighted partial sums should override
    /// this hook to return `true`; the default [`Self::outer_derivative_policy`]
    /// then threads the flag into the emitted [`OuterDerivativePolicy`].
    fn outer_derivative_subsample_capable(&self) -> bool {
        false
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

    /// Per-block output-channel assignment for the identifiability audit.
    ///
    /// Multi-parameter families (Dirichlet, beta, Gaussian/binomial
    /// location-scale, multinomial, …) drive several *independent* linear
    /// predictors `η_r = X_r β_r`, one per distributional parameter / class.
    /// Each [`ParameterBlockSpec`] feeds exactly one of those output channels.
    /// When two blocks share the same covariate basis (e.g. every Dirichlet
    /// component uses the same `[1 | B]`), their columns are *not* gauge
    /// aliases — they are block-diagonal entries of the true joint Jacobian
    /// `blkdiag(X_0, …, X_{m-1})`, full rank `Σ p_b`.
    ///
    /// The pre-fit identifiability audit can only see this block-diagonal
    /// structure through the **channel-aware** route, which requires each
    /// block to carry a multi-output `jacobian_callback` (n_outputs > 1).
    /// Families built via the canonical helpers (`build_location_scale_block`,
    /// `MultinomialFamily::build_block_specs`) wire that callback themselves;
    /// families fit through the low-level `fit_custom_family` API with
    /// hand-built specs do not, and the flat audit then mistakes the repeated
    /// shared basis for cross-block aliases and refuses a well-posed fit
    /// (issues #319 / #363 / #558).
    ///
    /// Returning `Some(channels)` — a vector of length `specs.len()` giving the
    /// zero-based output channel each block drives — lets `fit_custom_family`
    /// install the appropriate [`AdditiveBlockJacobian`] on any block that
    /// lacks an explicit callback, so the audit routes channel-aware
    /// automatically. The total channel count is `channels.iter().max() + 1`.
    ///
    /// Default: every block drives output channel 0. `wire_output_channels`
    /// recognizes this as the single-output flat route and leaves specs unchanged.
    ///
    /// When `Some`, the returned vector MUST have length equal to the number
    /// of blocks; `fit_custom_family` surfaces a structured error otherwise.
    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        Some(vec![0; specs.len()])
    }

    /// Optional dynamic geometry hook for blocks whose design/offset depend on
    /// current values of other blocks.
    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        assert!(block_states.len() <= isize::MAX as usize);
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
        block_states: &[ParameterBlockState],
        idx: usize,
        block_spec: &ParameterBlockSpec,
        arr: &Array1<f64>,
    ) -> Result<Option<BlockGeometryDirectionalDerivative>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(!block_spec.name.is_empty());
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Optional per-block coefficient projection applied after each block update.
    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(!block_spec.name.is_empty());
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
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Optional linear inequality constraints for a block update:
    /// `A * beta_block >= b`.
    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(!block_spec.name.is_empty());
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
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
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
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
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
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        Ok(None)
    }

    /// Optional block-concatenated log-likelihood gradient `g = nabla l(theta)`
    /// assembled from the SAME single source of truth as
    /// [`Self::exact_newton_joint_hessian`] (e.g. a per-row jet-tower kernel), so
    /// the damped Newton `H delta = g` is solved on a consistent (objective,
    /// gradient, Hessian) triple. The default returns `None`, leaving the caller
    /// on its legacy hand-assembled gradient.
    fn exact_newton_joint_loglik_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
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
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
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
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert_valid_options(options, "exact Newton joint Hessian workspace");
        self.exact_newton_joint_hessian_workspace(states, specs)
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
    /// At large scale with K ≈ 15, p ≈ 64, m = 2 the batched path is
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
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        rho: &Array1<f64>,
        options: &BlockwiseFitOptions,
        hessian_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    ) -> Result<Option<BatchedOuterGradientTerms>, String> {
        assert_valid_blockspecs(specs, "batched outer gradient terms");
        assert_states_match_specs(block_states, specs, "batched outer gradient terms");
        assert_derivative_blocks_match_specs(
            derivative_blocks,
            specs,
            "batched outer gradient terms",
        );
        assert_rho_matches_specs(rho, specs, "batched outer gradient terms");
        assert_valid_options(options, "batched outer gradient terms");
        validate_hessian_workspace_ready(&hessian_workspace, "batched outer gradient terms")?;
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
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        rho: &Array1<f64>,
        hessian_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    ) -> Result<Option<BatchedOuterHessianTerms>, String> {
        assert_valid_blockspecs(specs, "batched outer Hessian terms");
        assert_states_match_specs(block_states, specs, "batched outer Hessian terms");
        assert_derivative_blocks_match_specs(
            derivative_blocks,
            specs,
            "batched outer Hessian terms",
        );
        assert_rho_matches_specs(rho, specs, "batched outer Hessian terms");
        validate_hessian_workspace_ready(&hessian_workspace, "batched outer Hessian terms")?;
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
    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "inner coefficient Hessian HVP availability");
        false
    }

    fn inner_joint_workspace_gradient_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "inner joint workspace gradient availability");
        false
    }

    /// Opt families in to the matrix-free inner-Newton/PCG path on top of the
    /// generic `use_joint_matrix_free_path` heuristic.
    ///
    /// `use_joint_matrix_free_path` is tuned for families with cheap per-row
    /// work where dense `O(n·p²)` assembly is itself the bottleneck and HVPs
    /// cost the same. Families with very expensive per-row work (e.g. BMS flex
    /// streaming cell partitions + flex-jet evaluations per row) can override
    /// this to force the operator path even at moderate `p`, because each HVP
    /// reuses the row stream once and PCG converges in a handful of iters.
    /// Default `false` keeps the heuristic untouched for everyone else.
    fn prefers_matrix_free_inner_joint(
        &self,
        specs: &[ParameterBlockSpec],
        states: &[ParameterBlockState],
    ) -> bool {
        assert_valid_blockspecs(specs, "matrix-free inner-joint preference");
        assert!(states.len() <= isize::MAX as usize);
        false
    }

    fn inner_joint_workspace_log_likelihood_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "inner joint workspace log-likelihood availability");
        false
    }

    /// True only when the family has a real profiled outer Hessian-vector
    /// product over θ = (ρ, ψ), without enumerating all θ_i θ_j pairs.
    fn outer_hyper_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "outer hyper-Hessian HVP availability");
        false
    }

    /// True when the family can expose the dense profiled outer Hessian.
    /// Generic custom-family pairwise derivative paths default to dense
    /// availability; families with only inner HVP support should override this
    /// if dense θθ assembly is not a valid capability for their path.
    fn outer_hyper_hessian_dense_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "outer hyper-Hessian dense availability");
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
        specs: &[ParameterBlockSpec],
    ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
        assert_valid_blockspecs(specs, "outer hyper-Hessian operator");
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
        //   returns None (e.g. dense form too large for memory at large-scale
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
            // Multi-block coupled family that did NOT set the explicit marker.
            // The marker exists because the trait cannot reflect on whether
            // `exact_newton_joint_hessian` was overridden — its *default* impl
            // assembles a strictly block-diagonal matrix from per-block exact
            // blocks, which would silently drop cross-block ∂²L/∂β_a∂β_b
            // curvature for a coupled likelihood. But the marker is not the
            // only available signal: a family that genuinely overrides the
            // joint Hessian with true coupled curvature produces a matrix with
            // *nonzero off-diagonal blocks*, which the block-diagonal default
            // can never produce. Detect that structurally and trust it. A
            // returned matrix that is block-diagonal is indistinguishable from
            // the default for a coupled family, so it stays gated to None.
            match self.exact_newton_joint_hessian(block_states)? {
                Some(hessian) if joint_hessian_has_cross_block_coupling(&hessian, block_states) => {
                    Ok(Some(hessian))
                }
                _ => Ok(None),
            }
        }
    }

    /// Structural-coupling probe shared by the `_with_specs` joint dispatch
    /// gates: is the family's `exact_newton_joint_hessian` a genuinely coupled
    /// matrix (nonzero off-diagonal blocks), as opposed to the trait's
    /// block-diagonal default? This is the marker-free signal that lets the
    /// engine trust a coupled multi-block family that overrode the joint
    /// Hessian without hand-setting `has_explicit_joint_hessian()`. Returns
    /// `false` when no joint Hessian is available or it is block-diagonal.
    fn joint_hessian_is_structurally_coupled(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<bool, String> {
        Ok(match self.exact_newton_joint_hessian(block_states)? {
            Some(hessian) => joint_hessian_has_cross_block_coupling(&hessian, block_states),
            None => false,
        })
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

    /// Whether the family's inner/outer solves need the full-span Jeffreys
    /// curvature `H_Φ` and score `∇Φ`.
    ///
    /// Default `true` to preserve the existing separation/near-singular
    /// robustness on every family the term was historically armed for
    /// (probit/binomial, GAMLSS location-scale, BMS, survival marginal-slope).
    ///
    /// A family overrides this to `false` when it has no
    /// separation/under-identification regime by construction — the
    /// canonical case is a continuous-response monotone-transformation
    /// family like `TransformationNormalFamily`, where the Fisher information
    /// is `O(n)` on every identified direction at every working point and
    /// the Jeffreys gate would always smooth-step to zero anyway. There the
    /// term is pure overhead: each evaluation runs `p` directional
    /// derivatives of the joint Hessian (`O(n·p²)` per call for the SCOP
    /// directional derivative), called multiple times per inner cycle and
    /// once per outer evaluation. At large scale (`p=144`, `n=20000`) the
    /// overhead is the dominant per-cycle cost and exhausts the CI budget
    /// long before the inner Newton converges, while contributing
    /// essentially zero to the converged gradient and curvature.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    /// Optional Tier-B Jeffreys information matrix.
    ///
    /// Defaults to the exact joint Newton Hessian for existing families.
    /// Non-canonical Bernoulli/binomial families should override this with the
    /// expected Fisher information: Jeffreys' prior is defined from expected
    /// information, while observed information can grow in saturated
    /// misclassified tails and create an artificial prior-reward valley.
    fn joint_jeffreys_information_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_with_specs(block_states, specs)
    }

    /// First beta-directional derivative of
    /// [`Self::joint_jeffreys_information_with_specs`].
    fn joint_jeffreys_information_directional_derivative_with_specs(
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

    /// Second beta-directional derivative of
    /// [`Self::joint_jeffreys_information_with_specs`].
    fn joint_jeffreys_information_second_directional_derivative_with_specs(
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

    /// Optional contracted second beta-derivative of the observed joint
    /// Newton information:
    ///
    ///   ∇²_β tr(W H(β))
    ///
    /// for a fixed full-joint trace weight `W`.
    ///
    /// This is the wide-p route for Jeffreys' omitted second-directional
    /// completion. The default returns `None`, so callers fall back to the
    /// existing p(p+1)/2 pairwise `H''[e_a,e_b]` path.
    fn exact_newton_joint_contracted_trace_hessian(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != specs.len() {
            return Err(format!(
                "exact_newton_joint_contracted_trace_hessian default: block state count {} != spec count {}",
                block_states.len(),
                specs.len()
            ));
        }
        let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        if weight.dim() != (total, total) {
            return Err(format!(
                "exact_newton_joint_contracted_trace_hessian default: weight shape {:?} != ({total}, {total})",
                weight.dim()
            ));
        }
        for (block_idx, (state, spec)) in block_states.iter().zip(specs.iter()).enumerate() {
            let p_block = spec.design.ncols();
            if state.beta.len() != p_block {
                return Err(format!(
                    "exact_newton_joint_contracted_trace_hessian default: block {block_idx} beta length {} != design cols {p_block}",
                    state.beta.len()
                ));
            }
        }
        Ok(None)
    }

    /// Contracted second beta-derivative matching
    /// [`Self::joint_jeffreys_information_with_specs`]:
    ///
    ///   ∇²_β tr(W I_J(β)).
    ///
    /// Defaults to the observed-information contract above. Families that
    /// override the Jeffreys information with expected/Fisher information
    /// should override this too when they can compute the contraction in one
    /// pass; otherwise the default `None` preserves the pairwise `H''`
    /// fallback through
    /// [`Self::joint_jeffreys_information_second_directional_derivative_with_specs`].
    fn joint_jeffreys_information_contracted_trace_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_contracted_trace_hessian(block_states, specs, weight)
    }

    /// Whether
    /// [`Self::joint_jeffreys_information_contracted_trace_hessian_with_specs`]
    /// can supply the wide-p Jeffreys completion without the pairwise `H''`
    /// fallback. Default `false` preserves the historical width cap exactly.
    fn joint_jeffreys_information_contracted_trace_hessian_available(&self) -> bool {
        false
    }

    /// Whether [`Self::joint_jeffreys_information_with_specs`] is the SAME
    /// object as the observed joint Newton Hessian
    /// (`exact_newton_joint_hessian_with_specs`).
    ///
    /// Default `true`: the trait defaults delegate the Jeffreys information
    /// to the observed quantities, so conditioning certificates obtained from
    /// observed-Hessian matvecs transfer to the Jeffreys gate exactly.
    ///
    /// Families that override the Jeffreys information with the EXPECTED
    /// Fisher information must override this to `false`. Every matrix-free
    /// "Jeffreys provably skippable" pre-check
    /// (`jeffreys_term_skippable_via_matvec`) certifies conditioning from
    /// OBSERVED Hessian matvecs; that certificate does NOT transfer when the
    /// two informations diverge. For probit-class likelihoods the observed
    /// information GROWS (~η²) on saturated misclassified rows while the
    /// expected information DECAYS, so an observed-conditioning skip would
    /// zero the Jeffreys term exactly in the saturation regime it must police
    /// (gam#1020). When this returns `false` the pre-checks are bypassed and
    /// the exact expected-information gate always runs.
    fn joint_jeffreys_information_matches_observed_hessian(&self) -> bool {
        true
    }

    /// Whether the coupled-joint inner Newton should engage its self-vanishing
    /// Levenberg–Marquardt damping `μ` on a FULL-RANK-but-ILL-CONDITIONED
    /// penalized Hessian (cond > `COND_NEWTON_SAFETY`), not only on a
    /// rank-deficient one (`nullity > 0`). Default `false` (binary / AFT /
    /// others byte-identical). Survival marginal-slope overrides to `true`
    /// (#808: full-rank but cond ≈ 5.8e6; the self-vanishing μ shapes only the
    /// trajectory, so the converged β is unbiased and the log-slope target is
    /// preserved). Survival-local by trait override so the shared spectral-range
    /// solver stays byte-identical for every other family — in particular AFT
    /// (`survival_location_scale`), whose intercept-only-scale fits can be
    /// high-cond and which a shared (unconditional) gate would regress (#735/#736).
    fn levenberg_on_ill_conditioning(&self) -> bool {
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
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonOuterCurvature>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
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
        block_specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
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
        block_specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
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
        } else if self.has_explicit_joint_hessian()
            || self.joint_hessian_is_structurally_coupled(block_states)?
        {
            // Marked, or structurally detected coupled (see
            // `exact_newton_joint_hessian_with_specs`): the family's own
            // directional derivative is the trusted cross-block `D_β H[u]`.
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
        // The marker predicate is supplemented by the marker-free structural
        // probe so an auto-routed coupled family (one that returns a genuinely
        // off-diagonal joint Hessian without setting the explicit marker) is
        // trusted consistently across all three derivative orders.
        if !self.outer_default_trustworthy_for_joint_hessian(specs)
            && !self.joint_hessian_is_structurally_coupled(block_states)?
        {
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
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
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
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
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
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        idx: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        assert!(derivative_blocks.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
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
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        idx: usize,
        idx2: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        assert!(derivative_blocks.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(idx2 < usize::MAX);
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
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        assert!(derivative_blocks.len() <= isize::MAX as usize);
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
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        assert_valid_options(options, "exact Newton joint psi workspace");
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
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        assert!(derivative_blocks.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
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

/// Scope of an outer-evaluation context — distinguishes a real outer
/// derivative evaluation (where auto-subsample is allowed to install a
/// fresh stratified mask and emit phase prints) from an inner
/// coefficient line-search trial (where the family must reuse the outer
/// row measure, so auto-subsample must stay disabled).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum EvalScope {
    /// Real outer derivative evaluation: ρ has advanced; auto-subsample
    /// install paths may build/refresh a mask keyed on this ρ.
    OuterDerivative,
    /// Inner coefficient trial (joint-Newton / line-search) at fixed
    /// outer ρ: row measure must remain identical to the surrounding
    /// outer eval, so auto-subsample must not install a fresh mask.
    InnerCoefficient,
}

/// Context published by the outer smoothing optimizer for every
/// downstream family evaluation. Carries the current outer ρ and a
/// monotonic per-outer-eval id alongside the [`EvalScope`] tag used to
/// gate auto-subsample installation. See the
/// [`BlockwiseFitOptions::outer_eval_context`] field doc for the bug
/// this prevents.
#[derive(Clone, Debug)]
pub struct OuterEvalContext {
    pub rho: Arc<Array1<f64>>,
    pub eval_id: usize,
    pub scope: EvalScope,
}
