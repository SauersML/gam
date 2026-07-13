use super::*;

/// Specifies whether the model uses profiled scale (Gaussian REML) or
/// fixed dispersion (non-Gaussian LAML).
#[derive(Clone, Debug)]
pub enum DispersionHandling {
    /// Gaussian REML: φ̂ = D_p / (n − M_p), profiled out of the objective.
    /// The cost includes (n−M_p)/2 · log(2πφ̂) and the gradient includes
    /// the profiled scale derivative. Always includes both logdet terms.
    ProfiledGaussian,
    /// Non-Gaussian LAML or maximum penalized likelihood.
    ///
    /// `include_logdet_h` controls whether ½ log|H| is included (true for full
    /// LAML, false for MPL/PQL).
    /// `include_logdet_s` controls whether −½ log|S|₊ is included.
    ///
    /// Standard LAML: `Fixed { phi: 1.0, include_logdet_h: true, include_logdet_s: true }`
    /// MaxPenalizedLikelihood: `Fixed { phi: 1.0, include_logdet_h: false, include_logdet_s: false }`
    Fixed {
        phi: f64,
        include_logdet_h: bool,
        include_logdet_s: bool,
    },
}

/// The unified inner solution produced by any inner solver.
///
/// Contains everything the outer REML/LAML evaluator needs. Produced by:
/// - Single-block PIRLS (via `PirlsResult::into_inner_solution()`)
/// - Blockwise coupled Newton (via `BlockwiseInnerResult::into_inner_solution()`)
/// - Sparse Cholesky (via `SparsePenalizedSystem::into_inner_solution()`)
pub struct InnerSolution<'dp> {
    // === Objective ingredients ===
    /// ℓ(β̂) — log-likelihood at the converged mode.
    /// For Gaussian: −0.5 × deviance (RSS). For GLMs: actual log-likelihood.
    pub log_likelihood: f64,

    /// β̂ᵀS(ρ)β̂ — penalty quadratic form at the mode.
    pub penalty_quadratic: f64,

    // === The factorization (single source of truth for all linear algebra) ===
    /// The Hessian operator providing logdet, trace, and solve.
    /// Both cost and gradient use this same object.
    ///
    /// IMPORTANT: This MUST encode the **observed** Hessian H_obs = X'W_obs X + S
    /// at the converged mode, where W_obs includes the residual-dependent correction
    /// for non-canonical links. Using expected Fisher H_Fisher = X'W_Fisher X + S
    /// would make this a PQL surrogate rather than the exact Laplace approximation.
    /// See response.md Section 3 for the mathematical justification.
    pub hessian_op: Arc<dyn HessianFactorization>,

    // === Coefficients and penalty structure ===
    /// β̂ — coefficients at the converged mode (in the operator's native basis).
    pub beta: Array1<f64>,

    /// Penalty coordinates for the rho block.
    ///
    /// Each coordinate represents one smoothing-parameter direction
    ///   A_k = λ_k S_k
    /// through either a full-root or a block-local root.
    pub penalty_coords: Vec<PenaltyCoordinate>,

    /// Derivatives of log|S(ρ)|₊ — precomputed from penalty structure.
    pub penalty_logdet: PenaltyLogdetDerivs,

    // === Family-specific derivative info ===
    /// Provider of third-derivative corrections for non-Gaussian families.
    ///
    /// The c and d arrays (dW/deta, d^2W/deta^2) carried by this provider MUST
    /// be the **observed** derivatives, not the Fisher derivatives. For non-canonical
    /// links the observed c/d include residual-dependent corrections:
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    /// These corrections matter for the outer gradient (C[v] correction) and
    /// outer Hessian (Q[v_k, v_l] correction). See response.md Section 3.
    pub deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,

    // === Corrections ===
    /// Optional exact Jeffreys/Firth term in the active coefficient basis.
    pub firth: Option<ExactJeffreysTerm>,

    /// Additive correction for the Hessian logdet when `hessian_op` encodes a
    /// uniformly rescaled exact curvature matrix.
    pub hessian_logdet_correction: f64,

    /// When the cost uses `log|U_Sᵀ H U_S|_+` (rank-deficient LAML fix),
    /// this carries the matching projected kernel so the gradient trace
    /// `tr(K · Ḣ)` agrees with the cost's derivative.  See
    /// [`PenaltySubspaceTrace`] for the full derivation.
    pub penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,

    /// Uniform scale `s` applied to rho-coordinate penalty derivatives in the
    /// H-dependent trace / solve parts of the outer calculus.
    ///
    /// ## Contract (CRITICAL — gradient/cost consistency)
    ///
    /// `rho_curvature_scale` is NOT a free knob.  It encodes the convention
    /// that the supplied `hessian_op` represents the **rescaled** curvature
    /// `H_op = s · (∇²(-ℓ) + Σ_k e^{ρ_k} S_k)`, i.e. every contribution to
    /// the curvature (likelihood Hessian AND penalty `λ_k S_k`) has been
    /// uniformly multiplied by `s` before reaching the evaluator.  Under this
    /// convention:
    ///
    /// * `∂H_op/∂ρ_k = s · λ_k S_k` (matches the `curvature_lambdas = s · λ`
    ///   drift used inside the gradient's trace term),
    /// * `K = H_op⁻¹ = (1/s) · (∇²(-ℓ) + λS)⁻¹`,
    /// * `tr(K · ∂H_op/∂ρ_k) = tr((∇²(-ℓ) + λS)⁻¹ · λ_k S_k)` (the analytic
    ///   gradient of the **unscaled** `log|H|`),
    /// * `log|H_op| = log|∇²(-ℓ) + λS| + p · log(s)`, which the caller MUST
    ///   un-scale by supplying `hessian_logdet_correction += −p · log(s)` so
    ///   that `hop.logdet() + hessian_logdet_correction` evaluates the same
    ///   unscaled `log|H|` whose derivative the gradient trace computes.
    ///
    /// Callers that set `rho_curvature_scale ≠ 1` without ALSO pre-scaling
    /// `hessian_op` AND adding the matching `−p·log(s)` term to
    /// `hessian_logdet_correction` will get a gradient that is off by the
    /// factor `s` from `dV/dρ_k`.  The unified evaluator does **not** scale
    /// `hop` for the caller — that would defeat the purpose of the
    /// curvature-conditioning trick survival families use to keep the
    /// outer eigendecomposition numerically stable.
    ///
    /// See `survival::location_scale::exact_newton_outer_curvature` for the
    /// canonical example: `rho_curvature_scale = exp(-log_scale)` paired with
    /// `hessian_logdet_correction = p · log_scale = −p · log(scale)`.
    ///
    /// The evaluator enforces `rho_curvature_scale > 0` and finite; pass
    /// `1.0` (the documented default) when no curvature conditioning is in
    /// play.
    pub rho_curvature_scale: f64,

    /// Configured prior over rho coordinates. The evaluator receives the
    /// realized cost/gradient tuple separately; this copy lets EFS use the
    /// conjugate Gamma rate in its multiplicative denominator.
    pub rho_prior: gam_problem::RhoPrior,

    // === Model dimensions ===
    /// Number of observations.
    pub n_observations: usize,

    /// M_p: dimension of the penalty null space (unpenalized coefficients).
    pub nullspace_dim: f64,

    /// ½·Σᵢ log(wᵢ) — half the sum of log prior weights.
    ///
    /// This is the per-observation Gaussian normalization constant that the
    /// `log_likelihood` (computed by
    /// [`crate::pirls::pirls_data_log_kernel_from_eta`]) deliberately drops. The
    /// full weighted-Gaussian negative log-likelihood normalization is
    ///   ½·Σᵢ log(2π·φ/wᵢ) = (n/2)·log(2πφ) − ½·Σᵢ log(wᵢ),
    /// because `Var(yᵢ) = φ/wᵢ` under inverse-variance prior weights.
    ///
    /// Dropping `−½·Σ log(wᵢ)` does not move the ρ-argmin in exact arithmetic
    /// (it is constant in ρ), but it makes the ProfiledGaussian objective VALUE
    /// scale-dependent: under a global rescale `w → c·w` the invariance-
    /// preserving smoothing `λ → c·λ` leaves the cost SHAPE fixed but inflates
    /// its absolute value by `(n/2)·log c`. That inflation breaks the exact
    /// weight-scale invariance of the selected λ̂ / EDF / fit (issue #877).
    /// Restoring this term makes the ProfiledGaussian cost value exactly
    /// invariant to `w → c·w` (with σ̂² absorbing the c factor), matching mgcv.
    ///
    /// Only consumed by the `ProfiledGaussian` arm; the `Fixed`-dispersion arm
    /// already omits the Gaussian normalization constant by design and is not
    /// affected.
    pub gaussian_weight_log_sum_half: f64,

    /// Deviance scale `D₀` used as the *relative* reference for the smooth
    /// penalized-deviance floor (see [`crate::estimate::smooth_floor_dp`]).
    ///
    /// Set to the weighted null deviance of the Gaussian response,
    /// `D₀ = Σ wᵢ(yᵢ − ȳ_w)²`, which is the natural upper reference for
    /// `D_p` and — crucially — transforms as `D₀ → a²·D₀` under a response
    /// rescale `y → a·y`, exactly as `D_p` does. Flooring `D_p` at a fixed
    /// fraction of `D₀` therefore keeps the profiled Gaussian REML criterion
    /// exactly scale-equivariant (issue #1127); an absolute floor does not.
    ///
    /// Only consumed by the `ProfiledGaussian` arm. Defaults to `1.0`, which
    /// reproduces the historical absolute floor byte-for-byte for every caller
    /// that does not supply a response scale.
    pub dp_floor_scale: f64,

    /// How the dispersion parameter is handled.
    pub dispersion: DispersionHandling,

    // === Extended hyperparameter coordinates (ψ / τ) ===
    /// External (non-ρ) hyperparameter coordinates with their fixed-β objects.
    /// These are appended after the ρ coordinates in the gradient/Hessian output.
    pub ext_coords: Vec<HyperCoord>,

    /// Callback to compute second-order fixed-β objects for a pair (i, j)
    /// of external coordinates (or external × ρ cross pairs).
    /// Arguments: (ext_index_i, ext_index_j) → HyperCoordPair.
    /// When None, the outer Hessian is not computed for extended coordinates.
    ///
    /// `Arc`-backed ([`HyperCoordPairFn`]) so a derived solution — notably the
    /// tangent-projected solution built under active inequality constraints —
    /// can clone the same callback through to `ValueGradientHessian` assembly.
    pub ext_coord_pair_fn: Option<HyperCoordPairFn>,

    /// Callback for ρ × ext cross pairs: (rho_index, ext_index) → HyperCoordPair.
    pub rho_ext_pair_fn: Option<HyperCoordPairFn>,

    /// M_i[u] = D_β B_i[u] callback for extended coordinates.
    /// Arguments: (ext_index, direction) → correction matrix.
    ///
    /// `Arc`-backed ([`SharedFixedDriftDerivFn`]) so the tangent-projected
    /// solution can clone it through — the drift `M` is a p-space matrix the
    /// wrapper projects via `ZᵀMZ` in `trace_logdet_*`.
    pub fixed_drift_deriv: Option<SharedFixedDriftDerivFn>,

    /// Direction-contracted second-order ψ hook for the profiled θ-HVP (#740).
    /// When present, the outer-Hessian operator builder skips the `K²` per-pair
    /// `base_h2` ψψ assembly and instead applies this once per matvec to obtain
    /// every output row's `tr(K · D²_ψ H_L[ψ_i, ψ(α)])` in a single family row
    /// pass. `None` keeps the exact per-pair assembly. See
    /// [`ContractedPsiSecondOrderFn`].
    pub contracted_psi_second_order: Option<ContractedPsiSecondOrderFn>,

    /// Optional log-barrier configuration for monotonicity-constrained coefficients.
    /// When present, the barrier cost and Hessian corrections are added to the
    /// outer REML/LAML objective.
    pub barrier_config: Option<BarrierConfig>,

    /// Optional inner KKT residual `r = ∇_β L_pen(β̂)` at the converged β̂,
    /// already projected onto the free subspace (see [`ProjectedKktResidual`]
    /// for the invariant and why the type wraps this). `Some` activates the
    /// implicit-function-theorem corrections in `reml_laml_evaluate` (cost
    /// gets `−½ rᵀ H⁻¹ r`, ρ-gradient and ρρ Hessian get the matching first
    /// and second derivatives of that same scalar correction). `None` keeps
    /// the envelope-only behaviour for callers that genuinely guarantee
    /// exact KKT.
    pub kkt_residual: Option<ProjectedKktResidual>,

    /// Optional active linear-inequality constraints at the converged inner
    /// iterate. `Some(rows)` means the joint constraint matrix's row indices
    /// in `rows.active_indices` are pinned (treated as equality constraints
    /// at the cert point). The unified evaluator combines this with the
    /// `penalty_subspace_trace` to form the **constraint-aware** kernel
    /// `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S` for per-coordinate IFT mode
    /// responses `v_k = ∂β/∂ρ_k`. See [`ConstrainedSubspaceKernel`] for
    /// the full derivation and consistency with `log|U_Tᵀ H U_T|`.
    ///
    /// `None` is the legacy/unconstrained path (no active inequality
    /// constraints to project against).
    pub active_constraints: Option<Arc<ActiveLinearConstraintBlock>>,

    /// Fit-level stochastic trace state. Shared by stochastic trace batches so
    /// CRN probe prefixes stay fixed and matrix-free trace CG can warm-start
    /// from the previous solve of the same probe id.
    pub stochastic_trace_state: Arc<Mutex<StochasticTraceState>>,
}

/// Builder for `InnerSolution` that provides sensible defaults and
/// auto-computes derived quantities (nullspace_dim).
pub struct InnerSolutionBuilder<'dp> {
    // Required fields
    pub(crate) log_likelihood: f64,
    pub(crate) penalty_quadratic: f64,
    pub(crate) hessian_op: Arc<dyn HessianFactorization>,
    pub(crate) beta: Array1<f64>,
    pub(crate) penalty_coords: Vec<PenaltyCoordinate>,
    pub(crate) penalty_logdet: PenaltyLogdetDerivs,
    pub(crate) n_observations: usize,
    pub(crate) dispersion: DispersionHandling,
    // Optional fields with defaults
    pub(crate) deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,
    pub(crate) firth: Option<ExactJeffreysTerm>,
    pub(crate) hessian_logdet_correction: f64,
    pub(crate) penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,
    pub(crate) rho_curvature_scale: f64,
    pub(crate) rho_prior: gam_problem::RhoPrior,
    pub(crate) nullspace_dim_override: Option<f64>,
    // Extended hyperparameter coordinates
    pub(crate) ext_coords: Vec<HyperCoord>,
    pub(crate) ext_coord_pair_fn: Option<HyperCoordPairFn>,
    pub(crate) rho_ext_pair_fn: Option<HyperCoordPairFn>,
    pub(crate) fixed_drift_deriv: Option<SharedFixedDriftDerivFn>,
    pub(crate) contracted_psi_second_order: Option<ContractedPsiSecondOrderFn>,
    pub(crate) barrier_config: Option<BarrierConfig>,
    pub(crate) kkt_residual: Option<ProjectedKktResidual>,
    pub(crate) active_constraints: Option<Arc<ActiveLinearConstraintBlock>>,
    pub(crate) gaussian_weight_log_sum_half: f64,
    pub(crate) dp_floor_scale: f64,
}

impl<'dp> InnerSolutionBuilder<'dp> {
    /// Create a builder with the required core fields.
    pub fn new(
        log_likelihood: f64,
        penalty_quadratic: f64,
        beta: Array1<f64>,
        n_observations: usize,
        hessian_op: Arc<dyn HessianFactorization>,
        penalty_coords: Vec<PenaltyCoordinate>,
        penalty_logdet: PenaltyLogdetDerivs,
        dispersion: DispersionHandling,
    ) -> Self {
        Self {
            log_likelihood,
            penalty_quadratic,
            hessian_op,
            beta,
            penalty_coords,
            penalty_logdet,
            n_observations,
            dispersion,
            deriv_provider: Box::new(GaussianDerivatives),
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: gam_problem::RhoPrior::Flat,
            nullspace_dim_override: None,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            gaussian_weight_log_sum_half: 0.0,
            dp_floor_scale: 1.0,
        }
    }

    pub fn deriv_provider(mut self, p: Box<dyn HessianDerivativeProvider + 'dp>) -> Self {
        self.deriv_provider = p;
        self
    }

    /// Install a pre-built Jeffreys/Firth term (Tier-A operator-backed via
    /// `ExactJeffreysTerm::new`, or the Tier-B value-only carrier via
    /// `ExactJeffreysTerm::value_only`).
    pub fn firth_term(mut self, term: Option<ExactJeffreysTerm>) -> Self {
        self.firth = term;
        self
    }

    pub fn hessian_logdet_correction(mut self, correction: f64) -> Self {
        self.hessian_logdet_correction = correction;
        self
    }

    /// Install the projected-logdet trace kernel that pairs with the
    /// `hessian_logdet_correction` on a rank-deficient penalty surface.
    /// See [`PenaltySubspaceTrace`] for the derivation and when it is
    /// required for gradient consistency.
    pub fn penalty_subspace_trace(mut self, kernel: Option<Arc<PenaltySubspaceTrace>>) -> Self {
        self.penalty_subspace_trace = kernel;
        self
    }

    pub fn rho_curvature_scale(mut self, scale: f64) -> Self {
        self.rho_curvature_scale = scale;
        self
    }

    pub fn rho_prior(mut self, prior: gam_problem::RhoPrior) -> Self {
        self.rho_prior = prior;
        self
    }

    /// Override the auto-computed nullspace dimension.
    ///
    /// By default, `build()` computes nullspace_dim as
    /// `beta.len() - sum(penalty_coord.rank())`. Use this when the caller
    /// has a different authoritative value (e.g. from stored per-penalty dims).
    pub fn nullspace_dim_override(mut self, dim: f64) -> Self {
        self.nullspace_dim_override = Some(dim);
        self
    }

    pub fn ext_coords(mut self, coords: Vec<HyperCoord>) -> Self {
        self.ext_coords = coords;
        self
    }

    pub fn ext_coord_pair_fn(
        mut self,
        f: Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>,
    ) -> Self {
        // `Arc::from(Box<dyn …>)` is a zero-cost re-tag of the existing
        // allocation; storing it as `Arc` lets the projected solution clone
        // the callback through.
        self.ext_coord_pair_fn = Some(HyperCoordPairFn::from(f));
        self
    }

    pub fn rho_ext_pair_fn(
        mut self,
        f: Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>,
    ) -> Self {
        self.rho_ext_pair_fn = Some(HyperCoordPairFn::from(f));
        self
    }

    pub fn fixed_drift_deriv(mut self, f: FixedDriftDerivFn) -> Self {
        // `Arc::from(Box<dyn …>)` re-tags the existing allocation for shared
        // ownership so the projected solution can clone the callback through.
        self.fixed_drift_deriv = Some(SharedFixedDriftDerivFn::from(f));
        self
    }

    /// Install the direction-contracted second-order ψ hook (#740). When set,
    /// the outer-Hessian operator builder uses it instead of the `K²` per-pair
    /// `base_h2` ψψ assembly. See [`ContractedPsiSecondOrderFn`].
    pub fn contracted_psi_second_order(mut self, f: Option<ContractedPsiSecondOrderFn>) -> Self {
        self.contracted_psi_second_order = f;
        self
    }

    pub fn barrier_config(mut self, config: Option<BarrierConfig>) -> Self {
        self.barrier_config = config;
        self
    }

    pub fn kkt_residual(mut self, residual: Option<ProjectedKktResidual>) -> Self {
        self.kkt_residual = residual;
        self
    }

    /// Stash the active linear-inequality constraint block carried alongside the
    /// inner solution. Used by `PenaltySubspaceTrace::with_active_constraints`
    /// at REML/LAML evaluation time to form the constraint-aware kernel
    /// `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`.
    pub fn active_constraints(mut self, block: Option<Arc<ActiveLinearConstraintBlock>>) -> Self {
        self.active_constraints = block;
        self
    }

    /// Build the `InnerSolution`, auto-computing nullspace_dim from penalty coordinates.
    pub fn build(self) -> InnerSolution<'dp> {
        let beta_dim = self.beta.len();
        let penalty_dim = self.penalty_coords.len();
        assert_eq!(
            self.hessian_op.dim(),
            beta_dim,
            "InnerSolutionBuilder: Hessian dimension {} does not match beta length {}",
            self.hessian_op.dim(),
            beta_dim
        );
        for (idx, coord) in self.penalty_coords.iter().enumerate() {
            assert_eq!(
                coord.dim(),
                beta_dim,
                "InnerSolutionBuilder: penalty coordinate {idx} has dimension {} but beta length is {}",
                coord.dim(),
                beta_dim
            );
        }
        assert_eq!(
            self.penalty_logdet.first.len(),
            penalty_dim,
            "InnerSolutionBuilder: penalty logdet first-derivative length {} does not match penalty coordinate count {}",
            self.penalty_logdet.first.len(),
            penalty_dim
        );
        if let Some(second) = self.penalty_logdet.second.as_ref() {
            assert!(
                second.nrows() == penalty_dim && second.ncols() == penalty_dim,
                "InnerSolutionBuilder: penalty logdet Hessian shape {}x{} does not match penalty coordinate count {}",
                second.nrows(),
                second.ncols(),
                penalty_dim
            );
        }
        if let Some(barrier_config) = self.barrier_config.as_ref() {
            assert_eq!(
                barrier_config.constrained_indices.len(),
                barrier_config.lower_bounds.len(),
                "InnerSolutionBuilder: barrier constrained index count {} does not match lower-bound count {}",
                barrier_config.constrained_indices.len(),
                barrier_config.lower_bounds.len()
            );
            assert_eq!(
                barrier_config.constrained_indices.len(),
                barrier_config.bound_signs.len(),
                "InnerSolutionBuilder: barrier constrained index count {} does not match bound-direction count {}",
                barrier_config.constrained_indices.len(),
                barrier_config.bound_signs.len()
            );
            assert!(
                barrier_config.tau.is_finite() && barrier_config.tau >= 0.0,
                "InnerSolutionBuilder: barrier tau must be finite and non-negative, got {}",
                barrier_config.tau
            );
            for ((&idx, &lower_bound), &sign) in barrier_config
                .constrained_indices
                .iter()
                .zip(barrier_config.lower_bounds.iter())
                .zip(barrier_config.bound_signs.iter())
            {
                assert!(
                    idx < beta_dim,
                    "InnerSolutionBuilder: barrier constrained index {idx} out of bounds for beta length {beta_dim}"
                );
                assert!(
                    lower_bound.is_finite(),
                    "InnerSolutionBuilder: barrier lower bound for beta[{idx}] must be finite, got {lower_bound}"
                );
                assert!(
                    sign == 1.0 || sign == -1.0,
                    "InnerSolutionBuilder: barrier bound direction for beta[{idx}] must be ±1, got {sign}"
                );
            }
        }
        if let Some(active_constraints) = self.active_constraints.as_ref() {
            assert_eq!(
                active_constraints.a.ncols(),
                beta_dim,
                "InnerSolutionBuilder: active constraint width {} does not match beta length {}",
                active_constraints.a.ncols(),
                beta_dim
            );
        }
        let nullspace_dim = self.nullspace_dim_override.unwrap_or_else(|| {
            let penalty_rank: usize = self
                .penalty_coords
                .iter()
                .map(PenaltyCoordinate::rank)
                .sum();
            beta_dim.saturating_sub(penalty_rank) as f64
        });

        InnerSolution {
            log_likelihood: self.log_likelihood,
            penalty_quadratic: self.penalty_quadratic,
            hessian_op: self.hessian_op,
            beta: self.beta,
            penalty_coords: self.penalty_coords,
            penalty_logdet: self.penalty_logdet,
            deriv_provider: self.deriv_provider,
            firth: self.firth,
            hessian_logdet_correction: self.hessian_logdet_correction,
            penalty_subspace_trace: self.penalty_subspace_trace,
            rho_curvature_scale: self.rho_curvature_scale,
            rho_prior: self.rho_prior,
            n_observations: self.n_observations,
            nullspace_dim,
            gaussian_weight_log_sum_half: self.gaussian_weight_log_sum_half,
            dp_floor_scale: self.dp_floor_scale,
            dispersion: self.dispersion,
            ext_coords: self.ext_coords,
            ext_coord_pair_fn: self.ext_coord_pair_fn,
            rho_ext_pair_fn: self.rho_ext_pair_fn,
            fixed_drift_deriv: self.fixed_drift_deriv,
            contracted_psi_second_order: self.contracted_psi_second_order,
            barrier_config: self.barrier_config,
            kkt_residual: self.kkt_residual,
            active_constraints: self.active_constraints,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }
}
