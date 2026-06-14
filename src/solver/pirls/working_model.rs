//! The WorkingModel trait, GamWorkingModel and its PIRLS state machinery:
//! candidate screening, workspace, PIRLS options, arrow-Schur config and the
//! full GamWorkingModel inner-solve implementation.

use super::*;

pub trait WorkingModel {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError>;

    fn update_with_curvature(
        &mut self,
        beta: &Coefficients,
        curvature_kind: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        assert!(core::mem::size_of_val(&curvature_kind) > 0);
        self.update(beta)
    }

    fn update_candidate(
        &mut self,
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        self.update_with_curvature(beta, curvature)
    }

    fn screen_candidate(
        &mut self,
        beta: &Coefficients,
        arr: &Array1<f64>,
        linear_predictor: &LinearPredictor,
        curvature: HessianCurvatureKind,
    ) -> Result<CandidateEvaluation, EstimationError> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(std::mem::size_of_val(linear_predictor) > 0);
        self.update_candidate(beta, curvature)
            .map(CandidateEvaluation::Full)
    }

    fn supports_observed_information_curvature(&self) -> bool {
        false
    }
}


/// Result of a cheap LM-candidate screen: penalized objective + arithmetic
/// finiteness, without the gradient/Hessian needed for an accepted step.
#[derive(Debug, Clone)]
pub struct CandidateScreen {
    pub penalized_objective: f64,
    pub deviance: f64,
    pub penalty_term: f64,
    pub arithmetic_finite: bool,
}


/// Outcome of `WorkingModel::screen_candidate`: either a cheap screen result
/// (LM loop must upgrade with `update_with_curvature` on acceptance) or the
/// full state when screening was not applicable.
pub enum CandidateEvaluation {
    Screen(CandidateScreen),
    Full(WorkingState),
}


impl CandidateEvaluation {
    #[inline]
    fn penalized_objective(&self, firth_bias_reduction: bool) -> f64 {
        match self {
            Self::Screen(s) => s.penalized_objective,
            Self::Full(state) => {
                let mut value = state.deviance + state.penalty_term;
                if firth_bias_reduction && let Some(j) = state.jeffreys_logdet() {
                    value -= 2.0 * j;
                }
                value
            }
        }
    }

    #[inline]
    fn arithmetic_finite(&self) -> bool {
        match self {
            Self::Screen(s) => s.arithmetic_finite,
            Self::Full(state) => state.gradient.iter().all(|g| g.is_finite()),
        }
    }

    #[inline]
    fn into_full(self) -> Option<WorkingState> {
        match self {
            Self::Full(state) => Some(state),
            Self::Screen(_) => None,
        }
    }
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PirlsAcceptedStateCacheKey {
    curvature: HessianCurvatureKind,
    firth_active: bool,
    beta_bits: Vec<u64>,
    arrow_latent_bits: Option<Vec<u64>>,
}


impl PirlsAcceptedStateCacheKey {
    fn requested(
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
        options: &WorkingModelPirlsOptions,
    ) -> Self {
        Self::new(beta, curvature, options.firth_bias_reduction, options)
    }

    fn accepted(
        beta: &Coefficients,
        state: &WorkingState,
        options: &WorkingModelPirlsOptions,
    ) -> Self {
        Self::new(
            beta,
            state.hessian_curvature,
            matches!(state.firth, FirthDiagnostics::Active { .. }),
            options,
        )
    }

    fn new(
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
        firth_active: bool,
        options: &WorkingModelPirlsOptions,
    ) -> Self {
        let arrow_latent_bits = options.arrow_schur.as_ref().map(|arrow_cfg| {
            arrow_cfg.snapshot_t.as_ref()()
                .iter()
                .map(|value| value.to_bits())
                .collect()
        });
        Self {
            curvature,
            firth_active,
            beta_bits: beta.as_ref().iter().map(|value| value.to_bits()).collect(),
            arrow_latent_bits,
        }
    }
}


/// Uncertainty inputs for integrated (GHQ) IRLS updates.
#[derive(Clone, Copy)]
pub(crate) struct IntegratedWorkingInput<'a> {
    pub quadctx: &'a crate::quadrature::QuadratureContext,
    pub se: ArrayView1<'a, f64>,
    pub mixture_link_state: Option<&'a MixtureLinkState>,
    pub sas_link_state: Option<&'a SasLinkState>,
}


pub struct WorkingDerivativeBuffersMut<'a> {
    pub(crate) c: &'a mut Array1<f64>,
    pub(crate) d: &'a mut Array1<f64>,
    pub(crate) dmu_deta: &'a mut Array1<f64>,
    pub(crate) d2mu_deta2: &'a mut Array1<f64>,
    pub(crate) d3mu_deta3: &'a mut Array1<f64>,
}


/// Contiguous mutable views of the three core working buffers (`mu`, `weights`,
/// `z`) shared by every PIRLS working-state writer.
pub(crate) struct WorkingSlices<'a> {
    pub mu: &'a mut [f64],
    pub weights: &'a mut [f64],
    pub z: &'a mut [f64],
}


/// Contiguous mutable views of the Newton derivative/curvature buffers
/// (`c`, `d`, `dmu/deta` jet) shared by the full-derivative PIRLS writers.
pub(crate) struct WorkingDerivSlices<'a> {
    pub c: &'a mut [f64],
    pub d: &'a mut [f64],
    pub dmu: &'a mut [f64],
    pub d2: &'a mut [f64],
    pub d3: &'a mut [f64],
}


/// Canonical "contiguous-or-panic" unpacking of the three core working buffers.
///
/// Single source of truth for the contiguity contract and panic messages that
/// every working-state writer relies on; every writer routes through this.
#[inline]
pub(crate) fn working_slices<'a>(
    mu: &'a mut Array1<f64>,
    weights: &'a mut Array1<f64>,
    z: &'a mut Array1<f64>,
) -> WorkingSlices<'a> {
    WorkingSlices {
        mu: mu.as_slice_mut().expect("mu must be contiguous"),
        weights: weights.as_slice_mut().expect("weights must be contiguous"),
        z: z.as_slice_mut().expect("z must be contiguous"),
    }
}


/// Canonical "contiguous-or-panic" unpacking of the Newton derivative buffers.
///
/// Single source of truth for the contiguity contract and panic messages of the
/// `c`/`d`/`dmu`/`d2`/`d3` curvature buffers; every full-derivative writer routes
/// through this.
#[inline]
pub(crate) fn working_deriv_slices<'a>(
    derivs: &'a mut WorkingDerivativeBuffersMut<'_>,
) -> WorkingDerivSlices<'a> {
    WorkingDerivSlices {
        c: derivs.c.as_slice_mut().expect("c must be contiguous"),
        d: derivs.d.as_slice_mut().expect("d must be contiguous"),
        dmu: derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous"),
        d2: derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous"),
        d3: derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous"),
    }
}


#[derive(Clone, Copy)]
pub(crate) struct WorkingBernoulliGeometry {
    pub(crate) mu: f64,
    pub(crate) weight: f64,
    pub(crate) z: f64,
    pub(crate) c: f64,
    pub(crate) d: f64,
}


/// Shared likelihood interface used by PIRLS working updates.
///
/// This keeps the update/deviance math in one place so engine-level likelihoods
/// and higher-level wrappers (custom family, GAMLSS warm starts) can share a
/// consistent implementation.
pub(crate) trait WorkingLikelihood {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        priorweights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
        derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    ) -> Result<(), EstimationError>;

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        priorweights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError>;
}


impl WorkingLikelihood for GlmLikelihoodSpec {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        priorweights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
        derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    ) -> Result<(), EstimationError> {
        match (&self.spec.response, &self.spec.link, integrated.is_some()) {
            (ResponseFamily::Binomial, _, true) => {
                let integ = integrated.unwrap();
                update_glmvectors_integrated_by_family(
                    integ.quadctx,
                    y,
                    eta,
                    integ.se,
                    &self.spec,
                    priorweights,
                    mu,
                    weights,
                    z,
                    derivatives,
                    integ.mixture_link_state,
                    integ.sas_link_state,
                )?;
                Ok(())
            }
            (ResponseFamily::Binomial, link, false) => {
                if matches!(link, InverseLink::Mixture(_)) {
                    crate::bail_invalid_estim!(
                        "BinomialMixture IRLS update requires explicit mixture link state"
                            .to_string(),
                    );
                }
                update_glmvectors(
                    y,
                    eta,
                    &self.spec.link,
                    priorweights,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::Gaussian, _, _) => {
                update_glmvectors(
                    y,
                    eta,
                    &InverseLink::Standard(StandardLink::Identity),
                    priorweights,
                    mu,
                    weights,
                    z,
                    None,
                )?;
                // For Gaussian identity, the canonical IRLS working weight is
                //     w_i = prior_i * (dmu/deta)^2 / Var(Y_i | mu_i) = prior_i / phi.
                // When the scale metadata explicitly fixes phi (rather than
                // profiling sigma out), the working weights must include 1/phi
                // so that PIRLS minimises the scaled deviance / scaled negative
                // log-likelihood that the calibrator and downstream variance
                // calculations expect. `ProfiledGaussian` returns `None` here,
                // preserving the historical "weights == prior" behaviour for
                // the default profiled case.
                if let Some(phi) = self.scale.fixed_phi() {
                    if !(phi.is_finite() && phi > 0.0) {
                        crate::bail_invalid_estim!(
                            "Gaussian fixed dispersion phi must be finite and positive (got {})",
                            phi
                        );
                    }
                    if phi != 1.0 {
                        let inv_phi = 1.0 / phi;
                        weights.mapv_inplace(|w| w * inv_phi);
                    }
                }
                Ok(())
            }
            (ResponseFamily::Poisson, _, _) => {
                write_poisson_log_working_state(y, eta, priorweights, mu, weights, z, derivatives);
                Ok(())
            }
            (ResponseFamily::Tweedie { p }, _, _) => {
                let p = *p;
                write_tweedie_log_working_state(
                    y,
                    eta,
                    priorweights,
                    p,
                    fixed_glm_dispersion(self),
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::NegativeBinomial { theta, .. }, _, _) => {
                let theta = *theta;
                write_negative_binomial_log_working_state(
                    y,
                    eta,
                    priorweights,
                    theta,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::Beta { phi }, _, _) => {
                let phi = *phi;
                write_beta_logit_working_state(
                    y,
                    eta,
                    priorweights,
                    phi,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::Gamma, _, _) => {
                write_gamma_log_working_state(
                    y,
                    eta,
                    priorweights,
                    self.gamma_shape().unwrap_or(1.0),
                    mu,
                    weights,
                    z,
                    derivatives,
                );
                Ok(())
            }
            (ResponseFamily::RoystonParmar, _, _) => Err(EstimationError::InvalidInput(
                "RoystonParmar is survival-specific and not a GLM IRLS family".to_string(),
            )),
        }
    }

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        priorweights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError> {
        if matches!(self.spec.response, ResponseFamily::Tweedie { .. }) {
            validate_tweedie_responses(&y, &priorweights)?;
        }
        Ok(calculate_deviance(y, mu, self, priorweights))
    }
}


// Suggestion #6: Preallocate and reuse iteration workspaces
pub struct PirlsWorkspace {
    // Common IRLS buffers. Only O(n) state is kept persistently; any
    // design-weighted n x p scratch must be streamed through bounded chunks.
    pub wz: Array1<f64>,
    pub eta_buf: Array1<f64>,
    // Stage 2/4 assembly (use max needed sizes)
    pub scaled_matrix: Array2<f64>,    // (<= p + ebrows) x p
    pub final_aug_matrix: Array2<f64>, // (<= p + erows) x p
    // Stage 5 RHS buffers
    pub rhs_full: Array1<f64>, // length <= p + erows
    // Gradient check helpers
    pub working_residual: Array1<f64>,
    pub weighted_residual: Array1<f64>,
    // Step-halving direction (XΔβ)
    pub delta_eta: Array1<f64>,
    // Preallocated buffer for GEMV results (length p)
    pub vec_buf_p: Array1<f64>,
    // Cached sparse penalized-system workspace for sparse-native solve eligibility/assembly.
    sparse_penalized_system_cache: Option<SparsePenalizedSystemCache>,
    // Factorization scratch (avoid per-iteration allocation)
    pub factorization_scratch: MemBuffer,
    // Permutation buffers for LDLT
    pub perm: Vec<usize>,
    pub perm_inv: Vec<usize>,
    // Buffer for in-place factorization (preserves original Hessian in WorkingState)
    pub factorization_matrix: Array2<f64>,
    // Buffer for sparse matrix scaling (avoid per-iteration allocation)
    pub weighted_xvalues: Vec<f64>,
    // Dense chunk buffer for streaming X'WX assembly on very large n.
    pub weighted_x_chunk: Array2<f64>,
    // Reusable p×p buffer for Hessian assembly (avoids per-iteration allocation).
    pub hessian_buf: Array2<f64>,
    // Reusable n-length buffer for X*β matvec (avoids per-iteration allocation in update).
    pub matvec_buf: Array1<f64>,
}


impl PirlsWorkspace {
    pub fn new(n: usize, p: usize, idx: usize, idx2: usize) -> Self {
        assert!(idx < usize::MAX);
        assert!(idx2 < usize::MAX);
        // Stage buffers are allocated lazily: historically these were pre-sized to
        // worst-case dimensions, which inflates memory when many PIRLS workspaces
        // exist concurrently (e.g. parallel REML evals).
        // The active code paths resize-on-demand where needed.

        PirlsWorkspace {
            wz: Array1::zeros(n),
            eta_buf: Array1::zeros(n),
            scaled_matrix: Array2::zeros((0, 0).f()),
            final_aug_matrix: Array2::zeros((0, 0).f()),
            rhs_full: Array1::zeros(0),
            working_residual: Array1::zeros(n),
            weighted_residual: Array1::zeros(n),
            delta_eta: Array1::zeros(n),
            vec_buf_p: Array1::zeros(p),
            sparse_penalized_system_cache: None,
            // Keep scratch minimal at init; grow only if/when a factorization path
            // needs it.
            factorization_scratch: {
                let par = faer::Par::Seq;
                let req = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
                    1,
                    par,
                    Spec::new(<LltParams as Auto<f64>>::auto()),
                );
                MemBuffer::new(req)
            },
            perm: vec![0; p],
            perm_inv: vec![0; p],
            factorization_matrix: Array2::zeros((0, 0)),
            weighted_xvalues: Vec::new(),
            weighted_x_chunk: Array2::zeros((0, 0).f()),
            hessian_buf: Array2::zeros((0, 0).f()),
            matvec_buf: Array1::zeros(n),
        }
    }

    pub(super) fn add_dense_xtwx_signed(
        weights: &Array1<f64>,
        weighted_x_scratch: &mut Array2<f64>,
        x: &Array2<f64>,
        out: &mut Array2<f64>,
    ) {
        *out = crate::solver::estimate::reml::assembly::xt_diag_x_dense_into(
            x,
            weights,
            weighted_x_scratch,
        );
    }

    /// Ensure the sparse penalty cache is populated and consistent with `x` and `s_lambda`.
    fn ensure_sparse_penalty_cache(
        &mut self,
        x: &SparseColMat<usize, f64>,
        s_lambda: &Array2<f64>,
    ) -> Result<(), EstimationError> {
        let penalty_pattern = SparsePenaltyPattern::from_dense_upper(s_lambda, 1e-12);
        let rebuild = match self.sparse_penalized_system_cache.as_ref() {
            Some(cache) => !cache.matches(x, &penalty_pattern),
            None => true,
        };
        if rebuild {
            self.sparse_penalized_system_cache =
                Some(SparsePenalizedSystemCache::new(x, penalty_pattern)?);
        }
        Ok(())
    }

    pub(crate) fn sparse_penalized_system_stats(
        &mut self,
        x: &SparseColMat<usize, f64>,
        s_lambda: &Array2<f64>,
    ) -> Result<SparsePenalizedSystemStats, EstimationError> {
        self.ensure_sparse_penalty_cache(x, s_lambda)?;
        Ok(self.sparse_penalized_system_cache.as_ref().unwrap().stats())
    }

    // Phase 2 hook: numeric sparse penalized-system assembly in original coordinates.
    pub(super) fn assemble_sparse_penalized_hessian(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
        s_lambda: &Array2<f64>,
        ridge: f64,
        precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        self.ensure_sparse_penalty_cache(x, s_lambda)?;
        self.sparse_penalized_system_cache
            .as_mut()
            .unwrap()
            .assemble_upper(x, weights, ridge, precomputed_xtwx)
    }
}


#[derive(Clone, Debug)]
pub struct WorkingModelPirlsOptions {
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub adaptive_kkt_tolerance: Option<AdaptiveKktTolerance>,
    pub max_step_halving: usize,
    pub min_step_size: f64,
    pub firth_bias_reduction: bool,
    /// Optional lower bounds on coefficients (same coordinate system as `beta`).
    /// Use `-inf` for unconstrained entries.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional linear inequality constraints in current coefficient coordinates:
    ///   A * beta >= b.
    pub linear_constraints: Option<LinearInequalityConstraints>,
    /// Optional warm-start hint for the Levenberg-Marquardt damping
    /// coefficient. When set, the inner solver seeds `λ_LM` to this
    /// value instead of the default `1e-6`. Clamped on consumption to
    /// `[1e-6, 1e-3]` so a stale or pathological hint cannot poison the
    /// solve: the upper bound costs at most three damping halvings
    /// versus the cold default, which is dwarfed by the savings when
    /// the hint is informative.
    ///
    /// Used by `execute_pirls_if_needed` (in `solver::reml::runtime`)
    /// to persist the converged λ across consecutive PIRLS calls in a
    /// single REML outer optimization, so the inner Newton does not
    /// have to rediscover problem-specific damping at every accepted
    /// outer iterate.
    pub initial_lm_lambda: Option<f64>,
    /// Enable the Transtrum-Sethna geodesic-acceleration second-order
    /// correction on each accepted Levenberg-Marquardt step. When true,
    /// after the standard LM direction `δp = −(H + λ_lm·diag(H))⁻¹ g`
    /// is computed and accepted by the LM gain test, the solver computes
    /// a finite-difference estimate of the directional second derivative
    /// of the gradient along `δp`, solves a *second* linear system with
    /// the same (already-factored) Hessian, and adds the correction
    /// `δp₂` to the step only if `‖δp₂‖ ≤ α‖δp‖` (the Transtrum-Sethna
    /// 2011 acceptance criterion, α = 0.75 here). The correction costs
    /// two extra full `WorkingModel::update` calls per accepted step
    /// (for the FD evaluations); it is most useful for fits whose
    /// penalized Hessian is near-singular (latent-coordinate fits,
    /// near-collinear bases). Default `false`; opt-in until validated
    /// across the broader family of likelihoods and penalties.
    pub geodesic_acceleration: bool,
    /// Optional arrow-Schur structured-inner-solve descriptor.
    ///
    /// When `Some`, every accepted LM Newton step inside the inner loop
    /// is computed by the per-observation arrow-Schur path
    /// ([`crate::solver::arrow_schur::ArrowSchurSystem`]) instead of the
    /// β-only `solve_newton_direction_dense`. When `None`, the existing
    /// β-only path is used unchanged (back-compat: every existing call
    /// site that does not opt in is unaffected).
    ///
    /// **Scope note.** This wires the *inner* Gauss–Newton step. The REML
    /// outer-loop gradient w.r.t. `t` (which carries a shared `Schur⁻¹`
    /// factor) is a separate plumbing change owned by the REML driver and is
    /// **not** handled here.
    pub arrow_schur: Option<ArrowSchurInnerConfig>,
}


/// Per-iteration arrow-Schur builder hook.
///
/// The driver supplies a closure that, given the current `β` iterate,
/// returns a freshly-populated [`crate::solver::arrow_schur::ArrowSchurSystem`]
/// — i.e. the per-row `H_tt^(i)`, `H_tβ^(i)`, `g_t^(i)` blocks and the
/// β-block `H_ββ`, `g_β`. The driver owns the assembly because the
/// per-row Jacobians depend on the latent-coord term's basis (Duchon,
/// Sphere, …) and the analytic-penalty contributions depend on the
/// registry the outer-fit configuration owns. PIRLS only knows how to
/// *solve* the bordered system once it has been assembled.
#[derive(Clone)]
pub struct ArrowSchurInnerConfig {
    /// Number of latent rows `N`.
    pub n_rows: usize,
    /// Latent dimensionality `d`.
    pub latent_dim: usize,
    /// β dimensionality `K` (must match the inner Hessian dimension).
    pub n_beta: usize,
    /// Closure that builds the bordered system at the current `β` and
    /// current latent `t` (the latter held externally by the driver, e.g.
    /// in a `LatentCoordValues` registered alongside the working model).
    /// Returning `None` signals "fall back to the β-only path for this
    /// iteration" — useful for the seeding sweep before `t` has been
    /// initialized.
    pub build: std::sync::Arc<
        dyn Fn(&Array1<f64>) -> Option<crate::solver::arrow_schur::ArrowSchurSystem> + Send + Sync,
    >,
    /// BA Schur solve mode. `None` selects Direct for `K <= 2000` and
    /// InexactPCG above, following "Bundle Adjustment in the Large".
    pub solver_mode: Option<crate::solver::arrow_schur::ArrowSolverMode>,
    /// When set, assemble the reduced dense Schur block in row chunks.
    pub streaming_chunk_size: Option<usize>,
    /// Steihaug trust-region radius for the reduced shared step. This ports
    /// the Ceres/BA trust-region guard while retaining PIRLS's LM damping.
    pub trust_region_radius: f64,
    /// Optional β-block column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// When `Some`, the PIRLS driver calls
    /// [`crate::solver::arrow_schur::ArrowSchurSystem::set_block_offsets`] on
    /// every system returned by the `build` closure, wiring the block-Jacobi
    /// path without requiring each family's closure to call it manually.
    ///
    /// Derive from `ParameterBlockSpec` slices via
    /// [`crate::families::custom_family::block_offsets_from_specs`].  When
    /// `None`, the preconditioner falls back to scalar-diagonal Jacobi (the
    /// pre-#287 behaviour); when `Some([])` (empty slice), the same fallback
    /// applies.
    pub block_offsets: Option<Arc<[std::ops::Range<usize>]>>,
    /// Callback that the inner solver invokes after each LM-attempted
    /// joint step to write the latent tangent increment back into the
    /// driver's `LatentCoordValues` via that latent's update rule
    /// (`retract_flat_delta` for manifold latents). `delta_t` is the flat
    /// row-major increment of length `n_rows * latent_dim`.
    pub apply_delta_t: std::sync::Arc<dyn Fn(&Array1<f64>) + Send + Sync>,
    /// Snapshot the driver's latent field before an LM trial step mutates it.
    pub snapshot_t: std::sync::Arc<dyn Fn() -> Array1<f64> + Send + Sync>,
    /// Restore a snapshot produced by [`Self::snapshot_t`] after any rejected
    /// LM trial. Accepted trials deliberately do not call this hook: β and t
    /// commit together.
    pub restore_t: std::sync::Arc<dyn Fn(&Array1<f64>) + Send + Sync>,
}


impl std::fmt::Debug for ArrowSchurInnerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowSchurInnerConfig")
            .field("n_rows", &self.n_rows)
            .field("latent_dim", &self.latent_dim)
            .field("n_beta", &self.n_beta)
            .field("solver_mode", &self.solver_mode)
            .field("streaming_chunk_size", &self.streaming_chunk_size)
            .field("trust_region_radius", &self.trust_region_radius)
            .field(
                "block_offsets",
                &self.block_offsets.as_ref().map(|o| o.len()),
            )
            .finish_non_exhaustive()
    }
}


fn restore_arrow_latent_if_needed(
    options: &WorkingModelPirlsOptions,
    snapshot: Option<Array1<f64>>,
) {
    if let (Some(arrow_cfg), Some(snapshot)) = (options.arrow_schur.as_ref(), snapshot) {
        arrow_cfg.restore_t.as_ref()(&snapshot);
    }
}


pub(super) fn restore_pending_arrow_latent_if_needed(
    options: &WorkingModelPirlsOptions,
    pending_snapshot: &mut Option<Array1<f64>>,
) {
    restore_arrow_latent_if_needed(options, pending_snapshot.take());
}


pub(super) fn commit_pending_arrow_latent(pending_snapshot: &mut Option<Array1<f64>>) {
    drop(pending_snapshot.take());
}


// Fixed stabilization ridge for PIRLS/PLS. `penalty_term` carries this as
// ridge * ||beta||^2 (equivalently 0.5 * ridge * ||beta||^2 in the
// 0.5 * (deviance + penalty_term) objective), and it is constant w.r.t. rho.
//
// Math note:
//   Objective: V(ρ) includes log|H(ρ)| with H(ρ) = X' W X + S_λ(ρ) + δ I.
//   If δ = δ(ρ) is adaptive, V(ρ) is only piecewise-smooth and ∂V/∂ρ ignores
//   ∂δ/∂ρ, causing a mismatch between the optimized surface and the analytic
//   derivative surface. Using a fixed δ makes V(ρ) smooth and the standard
//   envelope-theorem gradient valid:
//     dV/dρ_k = 0.5 λ_k βᵀ S_k β + 0.5 λ_k tr(H^{-1} S_k) - 0.5 det1[k].
pub(crate) const FIXED_STABILIZATION_RIDGE: f64 = 1e-8;


pub(crate) struct GamWorkingModel<'a> {
    x_original: DesignMatrix,
    coordinate_design: WorkingCoordinateDesign,
    offset: Array1<f64>,
    y: ArrayView1<'a, f64>,
    priorweights: ArrayView1<'a, f64>,
    penalty: PirlsPenalty,
    workspace: PirlsWorkspace,
    likelihood: GlmLikelihoodSpec,
    link_kind: InverseLink,
    firth_bias_reduction: bool,
    lastmu: Array1<f64>,
    lastweights: Array1<f64>,
    lastz: Array1<f64>,
    last_c: Array1<f64>,
    last_d: Array1<f64>,
    lasthessian_weights: Array1<f64>,
    lasthessian_c: Array1<f64>,
    lasthessian_d: Array1<f64>,
    lasthessian_curvature: HessianCurvatureKind,
    last_dmu_deta: Array1<f64>,
    last_d2mu_deta2: Array1<f64>,
    last_d3mu_deta3: Array1<f64>,
    last_penalty_term: f64,
    x_original_csr: Option<SparseRowMat<usize, f64>>,
    /// Optional per-observation SE for integrated (GHQ) likelihood.
    /// When present, uses integrated family-dispatched working updates.
    covariate_se: Option<Array1<f64>>,
    /// Whether the Gamma dispersion shape has been estimated and frozen for the
    /// duration of this inner P-IRLS solve. The shape (= 1/φ) is a nuisance
    /// scale that multiplies both the working weight (`w = shape·prior`) and the
    /// reported deviance (`2·shape·Σ wᵢ dᵢ`). Re-estimating it per inner Newton/LM
    /// iterate moves the product φ·λ that the penalized argmin β̂ depends on, so
    /// the LM gain ratio compares two different objectives and the solve stalls.
    /// The shape is therefore estimated once from the warm-start η on the first
    /// curvature build and held fixed; it refreshes naturally across *outer*
    /// iterations because a fresh `GamWorkingModel` is built per inner solve.
    /// See issue #511 (regression of #359).
    gamma_shape_locked: bool,
    /// Whether the Beta-regression precision `phi` has been estimated and frozen
    /// for the duration of this inner P-IRLS solve. Like the Gamma shape, `phi`
    /// is a nuisance scale entering the working weight `w ∝ (1+phi)` and the
    /// variance `Var(y)=mu(1-mu)/(1+phi)`; re-estimating it per Newton/LM iterate
    /// moves the penalized argmin, so it is estimated once from the warm-start η
    /// and held fixed within the inner solve, refreshing across outer iterations
    /// (a fresh working model is built per inner solve). Issue #567.
    beta_phi_locked: bool,
    /// Whether the Tweedie dispersion `phi` has been estimated and frozen for the
    /// duration of this inner P-IRLS solve. Like the Gamma shape, `phi` is a
    /// nuisance scale entering only the working weight (`prior·μ^{2−p}/phi`) and
    /// not the working response, so re-estimating it per Newton/LM iterate would
    /// move the product `φ·λ` the penalized argmin β̂ depends on and stall the LM
    /// gain ratio. It is therefore estimated once from the warm-start η and held
    /// fixed within the inner solve, refreshing across outer iterations (a fresh
    /// working model is built per inner solve). Issue #771.
    tweedie_phi_locked: bool,
    /// Whether the Negative-Binomial overdispersion `theta` has been estimated
    /// and frozen for the duration of this inner P-IRLS solve. `theta` enters the
    /// working weight `W = μθ/(θ+μ)` (the NB2 Fisher information) and the working
    /// response, so — like the Beta precision, and unlike the scale-free Gamma
    /// shape — re-estimating it per Newton/LM iterate would move the penalized
    /// argmin β̂ and stall the LM gain ratio. It is therefore estimated once from
    /// the warm-start η and held fixed within the inner solve, refreshing across
    /// outer iterations (a fresh working model is built per inner solve). The
    /// converged-η joint refresh in `loop_driver` re-arms this lock so the
    /// reported `theta` is exactly the ML estimate at the reported η. Issue #802.
    negbin_theta_locked: bool,
    quadctx: crate::quadrature::QuadratureContext,
    /// Frozen-weight first-Fisher-step data-fit Gram `XᵀWX` (#1111 / #1033
    /// mechanism (c)), in the same *original* (conditioned `x_fit`) frame
    /// `penalized_hessian` forms `compute_xtwx_blas(self.x_original, ...)` in,
    /// i.e. BEFORE any Qs conjugation. When present it serves the FIRST
    /// Fisher-scoring iteration's `XᵀWX` n-free, eliding the dominant
    /// O(N·p²) weighted cross-product on a large-n GLM ψ-trial. Consumed at
    /// most once per inner solve (the first `penalized_hessian` build at the
    /// warm β); later iterations restream the true moving `W`.
    glm_first_step_gram: Option<Array2<f64>>,
    /// Set once the frozen-W first-step Gram has been consumed, so subsequent
    /// inner iterations restream `XᵀWX` from the (moving) working weights.
    glm_first_step_gram_consumed: bool,
}


pub(super) struct GamModelFinalState {
    likelihood: GlmLikelihoodSpec,
    coordinate_frame: PirlsCoordinateFrame,
    finalmu: Array1<f64>,
    finalweights: Array1<f64>,
    scoreweights: Array1<f64>,
    finalz: Array1<f64>,
    final_c: Array1<f64>,
    final_d: Array1<f64>,
    final_dmu_deta: Array1<f64>,
    final_d2mu_deta2: Array1<f64>,
    final_d3mu_deta3: Array1<f64>,
    penalty_term: f64,
}


impl<'a> GamWorkingModel<'a> {
    fn new(
        x_transformed: Option<DesignMatrix>,
        x_original: DesignMatrix,
        coordinate_frame: PirlsCoordinateFrame,
        offset: ArrayView1<f64>,
        y: ArrayView1<'a, f64>,
        priorweights: ArrayView1<'a, f64>,
        penalty: PirlsPenalty,
        workspace: PirlsWorkspace,
        likelihood: GlmLikelihoodSpec,
        link_kind: InverseLink,
        firth_bias_reduction: bool,
        transform: Option<WorkingReparamTransform>,
        quadctx: crate::quadrature::QuadratureContext,
        glm_first_step_gram: Option<Array2<f64>>,
    ) -> Self {
        let coordinate_design = match coordinate_frame {
            PirlsCoordinateFrame::OriginalSparseNative => {
                WorkingCoordinateDesign::OriginalSparseNative
            }
            PirlsCoordinateFrame::TransformedQs => {
                if let Some(x_transformed) = x_transformed {
                    WorkingCoordinateDesign::TransformedExplicit {
                        x_csr: x_transformed.to_csr_cache(),
                        x_transformed,
                    }
                } else {
                    WorkingCoordinateDesign::TransformedImplicit {
                        transform: transform.expect(
                            "TransformedQs PIRLS coordinate frame requires either x_transformed or qs",
                        ),
                    }
                }
            }
        };
        let x_original_csr = x_original.to_csr_cache();
        let n = match &coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => x_original.nrows(),
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                x_transformed.nrows()
            }
            WorkingCoordinateDesign::TransformedImplicit { .. } => x_original.nrows(),
        };
        GamWorkingModel {
            x_original,
            coordinate_design,
            offset: offset.to_owned(),
            y,
            priorweights,
            penalty,
            workspace,
            likelihood,
            link_kind,
            firth_bias_reduction,
            lastmu: Array1::zeros(n),
            lastweights: Array1::zeros(n),
            lastz: Array1::zeros(n),
            last_c: Array1::zeros(n),
            last_d: Array1::zeros(n),
            lasthessian_weights: Array1::zeros(n),
            lasthessian_c: Array1::zeros(n),
            lasthessian_d: Array1::zeros(n),
            lasthessian_curvature: HessianCurvatureKind::Fisher,
            last_dmu_deta: Array1::zeros(n),
            last_d2mu_deta2: Array1::zeros(n),
            last_d3mu_deta3: Array1::zeros(n),
            last_penalty_term: 0.0,
            x_original_csr,
            covariate_se: None,
            gamma_shape_locked: false,
            beta_phi_locked: false,
            tweedie_phi_locked: false,
            negbin_theta_locked: false,
            quadctx,
            glm_first_step_gram,
            glm_first_step_gram_consumed: false,
        }
    }

    /// Set per-observation SE for integrated (GHQ) likelihood.
    /// When set, the working model uses uncertainty-aware IRLS updates.
    fn with_covariate_se(mut self, se: Array1<f64>) -> Self {
        self.covariate_se = Some(se);
        self
    }

    /// Convert the working model into its final state for outer REML consumption.
    ///
    /// The `finalweights` field is set to `lasthessian_weights`, which are the
    /// **observed-information** weights (for non-canonical links) or Fisher weights
    /// (for canonical links where observed = Fisher). These flow into the outer
    /// REML H = X'W_obs X + S, ensuring log|H| uses the correct Laplace curvature.
    /// See response.md Section 3 for the mathematical justification.
    fn into_final_state(self) -> GamModelFinalState {
        let GamWorkingModel {
            coordinate_design,
            lastmu,
            lastweights,
            lastz,
            last_c: _,
            last_d: _,
            lasthessian_weights,
            lasthessian_c,
            lasthessian_d,
            last_dmu_deta,
            last_d2mu_deta2,
            last_d3mu_deta3,
            last_penalty_term,
            ..
        } = self;
        let coordinate_frame = match coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => {
                PirlsCoordinateFrame::OriginalSparseNative
            }
            WorkingCoordinateDesign::TransformedExplicit { .. } => {
                PirlsCoordinateFrame::TransformedQs
            }
            WorkingCoordinateDesign::TransformedImplicit { .. } => {
                PirlsCoordinateFrame::TransformedQs
            }
        };
        GamModelFinalState {
            likelihood: self.likelihood.clone(),
            coordinate_frame,
            finalmu: lastmu,
            finalweights: lasthessian_weights,
            scoreweights: lastweights,
            finalz: lastz,
            final_c: lasthessian_c,
            final_d: lasthessian_d,
            final_dmu_deta: last_dmu_deta,
            final_d2mu_deta2: last_d2mu_deta2,
            final_d3mu_deta3: last_d3mu_deta3,
            penalty_term: last_penalty_term,
        }
    }

    /// Compute X_transformed * β into a pre-allocated buffer, avoiding
    /// per-iteration allocation in the dense case.
    fn transformed_matvec_into(&self, beta: &Coefficients, out: &mut Array1<f64>) {
        self.transformed_matvec_array_into(beta.as_ref(), out);
    }

    /// View-based sibling of `transformed_matvec_into` that operates on a raw
    /// `&Array1<f64>` to avoid wrapping (and cloning into) `Coefficients` on
    /// hot LM-screen paths.
    fn transformed_matvec_array_into(&self, beta: &Array1<f64>, out: &mut Array1<f64>) {
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                if let Some(dense) = x_transformed.as_dense() {
                    fast_av_into(dense, beta, out);
                    return;
                }
                out.assign(&x_transformed.matrixvectormultiply(beta));
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                // Composed: X · (Qs · beta).  Qs·beta is p-dim (cheap),
                // then write X·(Qs·beta) directly into out when X is dense.
                let beta_orig = transform.apply(beta);
                if let Some(dense) = self.x_original.as_dense() {
                    fast_av_into(dense, &beta_orig, out);
                } else {
                    out.assign(&self.x_original.apply(&beta_orig));
                }
            }
            WorkingCoordinateDesign::OriginalSparseNative => {
                out.assign(&self.x_original.matrixvectormultiply(beta));
            }
        }
    }

    fn transformed_transpose_matvec(&self, vec: &Array1<f64>) -> Array1<f64> {
        match &self.coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => {
                self.x_original.transpose_vector_multiply(vec)
            }
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                x_transformed.transpose_vector_multiply(vec)
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                let xtv = self.x_original.transpose_vector_multiply(vec);
                transform.apply_transpose(&xtv)
            }
        }
    }

    /// Compute X^T W X via the shared dense assembly path.
    /// Falls back to the scalar loop for sparse matrices.
    pub(crate) fn compute_xtwx_blas(
        workspace: &mut PirlsWorkspace,
        design: &DesignMatrix,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        match design {
            // Only the materialized arm can use the shared dense assembly path.
            // Lazy operator-backed dense designs (TPS/Matern at large scale)
            // cannot be densified; fall through to the operator XᵀWX path.
            DesignMatrix::Dense(x) if x.is_materialized_dense() => {
                let p = x.ncols();
                let x_dense = x.to_dense_arc();
                // Reuse workspace hessian buffer to avoid per-iteration allocation.
                if workspace.hessian_buf.nrows() != p || workspace.hessian_buf.ncols() != p {
                    workspace.hessian_buf = Array2::zeros((p, p).f());
                } else {
                    workspace.hessian_buf.fill(0.0);
                }
                if crate::gpu::cuda_selected() {
                    return crate::solver::gpu::pirls_gpu::weighted_crossprod_gpu(
                        x_dense.view(),
                        weights.view(),
                    )
                    .map_err(EstimationError::InvalidInput);
                }
                crate::gpu::log_backend_inventory_once();
                // DenseXtWX has no compiled vendor backend on this path; the
                // workload-size predicate is computed only for diagnostic
                // logging via the `decide` reason channel.
                let gpu_decision = crate::gpu::decide(
                    crate::gpu::GpuKernel::DenseXtWX,
                    crate::gpu::GpuEligibility::BackendNotCompiled,
                );
                gpu_decision
                    .require_supported()
                    .map_err(EstimationError::InvalidInput)?;
                gpu_decision.log();
                if weights.iter().any(|&w| w < 0.0) {
                    // Observed-information assembly may have signed row
                    // weights.  Use Xᵀ(WX) exactly; never sqrt/clip.
                    PirlsWorkspace::add_dense_xtwx_signed(
                        weights,
                        &mut workspace.weighted_x_chunk,
                        x_dense.as_ref(),
                        &mut workspace.hessian_buf,
                    );
                } else {
                    // All weights are non-negative; the shared dense helper
                    // computes Xᵀ·diag(w)·X directly without sqrt/clip.
                    PirlsWorkspace::add_dense_xtwx_signed(
                        weights,
                        &mut workspace.weighted_x_chunk,
                        x_dense.as_ref(),
                        &mut workspace.hessian_buf,
                    );
                }
                // Move the buffer out instead of cloning — saves O(p²) memcpy.
                // Next call will reallocate (same cost as the existing zero-fill).
                Ok(std::mem::take(&mut workspace.hessian_buf))
            }
            // Observed-Hessian assembly: working weights may be signed
            // (binomial + cloglog, Gamma + identity, etc.). Route through the
            // signed-Gram API so the CSC / sparse-accumulator paths preserve
            // sign instead of silently clipping negative-curvature mass.
            _ => crate::matrix::xt_diag_x_signed(
                design,
                crate::matrix::SignedWeightsView::from_array(weights),
            )
            .map(|h| h.to_dense())
            .map_err(EstimationError::InvalidInput),
        }
    }

    fn penalized_hessian(&mut self, weights: &Array1<f64>) -> Result<Array2<f64>, EstimationError> {
        // #1111 / #1033 mechanism (c): the frozen-weight first-Fisher-step Gram
        // `XᵀWX` (in the original / `x_fit` conditioned frame) serves the FIRST
        // Fisher-scoring iteration n-free, eliding the dominant O(N·p²) weighted
        // cross-product on a large-n GLM ψ-trial. It is only correct for the
        // first build at the warm β with FISHER curvature (the frozen tensor was
        // assembled from the canonical Fisher weights), and only in the two
        // original-frame coordinate designs (TransformedImplicit conjugates the
        // original-frame Gram afterward; OriginalSparseNative is already in that
        // frame). For TransformedExplicit the streamed Gram lives in the Qs frame
        // the tensor was not built in, so that variant always restreams. Every
        // later iteration restreams the true (moving) `W`, so the converged β̂ is
        // unchanged — only the first Gram build is skipped.
        let use_frozen_first_step = !self.glm_first_step_gram_consumed
            && self.glm_first_step_gram.is_some()
            && self.lasthessian_curvature == HessianCurvatureKind::Fisher
            && !matches!(
                self.coordinate_design,
                WorkingCoordinateDesign::TransformedExplicit { .. }
            );
        if use_frozen_first_step {
            // Take the cached original-frame Gram exactly once.
            let xtwx = self
                .glm_first_step_gram
                .take()
                .expect("frozen first-step Gram present by the guard above");
            self.glm_first_step_gram_consumed = true;
            log::debug!(
                "[frozen-glm-gram] serving first Fisher-step XᵀWX n-free (p={})",
                xtwx.nrows()
            );
            return match &self.coordinate_design {
                WorkingCoordinateDesign::TransformedImplicit { transform } => {
                    let mut h = transform.conjugate_matrix(&xtwx);
                    self.penalty.add_to_hessian(&mut h);
                    Ok(h)
                }
                WorkingCoordinateDesign::OriginalSparseNative => {
                    let mut h = xtwx;
                    self.penalty.add_to_hessian(&mut h);
                    Ok(h)
                }
                WorkingCoordinateDesign::TransformedExplicit { .. } => {
                    // Excluded from `use_frozen_first_step` by the guard above
                    // (the frozen Gram lives in the original frame the explicit
                    // transform was not built in). A clean error rather than a
                    // panic if a future refactor ever lets this state through.
                    Err(EstimationError::InvalidInput(
                        "frozen first-step Gram path reached with TransformedExplicit \
                         coordinate design, which the gate excludes"
                            .to_string(),
                    ))
                }
            };
        }
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                let mut h = Self::compute_xtwx_blas(&mut self.workspace, x_transformed, weights)?;
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                let xtwx = Self::compute_xtwx_blas(&mut self.workspace, &self.x_original, weights)?;
                let mut h = transform.conjugate_matrix(&xtwx);
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
            WorkingCoordinateDesign::OriginalSparseNative => {
                let mut h =
                    Self::compute_xtwx_blas(&mut self.workspace, &self.x_original, weights)?;
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
        }
    }

    fn supports_observed_hessian_curvature(&self) -> bool {
        supports_observed_hessian_curvature_for_likelihood(&self.likelihood, &self.link_kind)
    }

    /// Compute the Hessian-side weight arrays (w, c, d) for the requested curvature kind.
    ///
    /// When `requested == Observed` and the link supports it, returns the
    /// **observed-information** weights including the residual-dependent correction:
    ///   W_obs = W_Fisher - (y - mu) * B,  B = (h'' V - h'^2 V') / (phi V^2)
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    ///
    /// For canonical links (for example logit-Binomial and log-Poisson), B = 0
    /// so observed = Fisher. Gamma-log is non-canonical and therefore needs its
    /// own observed-information correction.
    ///
    /// These arrays serve dual purpose:
    /// 1. **Inner iteration**: They define the Newton system H*delta = -g.
    ///    Fisher scoring (using W_Fisher) is also valid here since any convergent
    ///    algorithm finds the same mode.
    /// 2. **Outer REML**: They define the Laplace Hessian H_obs = X'W_obs X + S.
    ///    The outer log|H| and trace terms MUST use observed information for the
    ///    exact Laplace approximation. See response.md Section 3.
    fn update_hessian_curvature_arrays(
        &mut self,
        requested: HessianCurvatureKind,
    ) -> Result<HessianCurvatureKind, EstimationError> {
        if requested == HessianCurvatureKind::Fisher || !self.supports_observed_hessian_curvature()
        {
            self.lasthessian_weights.assign(&self.lastweights);
            self.lasthessian_c.assign(&self.last_c);
            self.lasthessian_d.assign(&self.last_d);
            return Ok(HessianCurvatureKind::Fisher);
        }

        compute_observed_hessian_curvature_arrays_into(
            &self.likelihood,
            &self.link_kind,
            &self.workspace.eta_buf,
            self.y,
            &self.lastweights,
            self.priorweights,
            &mut self.lasthessian_weights,
            &mut self.lasthessian_c,
            &mut self.lasthessian_d,
        )?;
        Ok(HessianCurvatureKind::Observed)
    }

    fn sparse_penalized_hessian(
        &mut self,
        weights: &Array1<f64>,
        ridge: f64,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        let x_sparse = self.x_original.as_sparse().ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse-native PIRLS requires a sparse original design".to_string(),
            )
        })?;
        let PirlsPenalty::Dense { s_transformed, .. } = &self.penalty else {
            crate::bail_invalid_estim!(
                "sparse-native PIRLS requires a dense transformed penalty matrix"
            );
        };
        self.workspace.assemble_sparse_penalized_hessian(
            x_sparse,
            weights,
            s_transformed,
            ridge,
            None,
        )
    }

    /// LM-screen helper: evaluates a candidate β by reusing the previous
    /// `current_eta` plus a single design-matrix matvec `X·δ`, then runs the
    /// inverse-link only far enough to recover μ, w, z and the deviance.
    /// No Hessian assembly, no derivative buffers, no Jeffreys logdet.
    ///
    /// The LM loop calls `update_with_curvature` to upgrade the screen to a
    /// full `WorkingState` only when the screen is accepted. Rejected LM
    /// candidates therefore skip the O(np²) curvature build entirely.
    fn screen_candidate_from_direction(
        &mut self,
        beta: &Coefficients,
        direction: &Array1<f64>,
        current_eta: &LinearPredictor,
    ) -> Result<CandidateScreen, EstimationError> {
        let n = self.offset.len();
        if self.workspace.eta_buf.len() != n {
            self.workspace.eta_buf = Array1::zeros(n);
        }
        if self.workspace.delta_eta.len() != n {
            self.workspace.delta_eta = Array1::zeros(n);
        }

        // Compute δη = X·direction once into the workspace, then assemble
        // η_cand = η_current + δη in parallel.
        let mut delta_eta = std::mem::take(&mut self.workspace.delta_eta);
        // Avoid wrapping/cloning `direction` into a `Coefficients` newtype just
        // to satisfy the &Coefficients overload — the view-based sibling
        // performs the identical matvec without the per-LM-attempt clone.
        self.transformed_matvec_array_into(direction, &mut delta_eta);
        Zip::from(&mut self.workspace.eta_buf)
            .and(current_eta.as_ref())
            .and(&delta_eta)
            .par_for_each(|eta, &base, &d| *eta = base + d);
        self.workspace.delta_eta = delta_eta;

        // NB: the Gamma dispersion shape is deliberately NOT re-estimated here.
        // This screen only evaluates a *trial* β to feed the LM gain-ratio
        // accept/reject test, whose predicted reduction comes from the gradient
        // and Hessian built (at the current shape) by the last accepted
        // `update_with_curvature`. Re-estimating the shape per trial — and per
        // halving attempt — silently changes the objective the screen reports
        // (deviance = 2·shape·Σ wᵢ dᵢ) relative to that predicted reduction, so
        // the gain ratio compares two different objectives, every step is
        // rejected, λ_LM runs to its ceiling, and the inner solve stalls with a
        // large residual gradient ("LM step search exhausted"). The shape is a
        // nuisance scale that must stay fixed within an inner Newton/LM step; it
        // is updated once per *accepted* iterate in `update_with_curvature`
        // (block-coordinate β | shape), exactly as mgcv holds the scale fixed
        // through the inner P-IRLS solve. See issue #511 (regression of #359).
        let integrated = self.covariate_se.as_ref().map(|se| IntegratedWorkingInput {
            quadctx: &self.quadctx,
            se: se.view(),
            mixture_link_state: self.link_kind.mixture_state(),
            sas_link_state: self.link_kind.sas_state(),
        });
        match &self.link_kind {
            InverseLink::Mixture(_)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        None,
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        None,
                    )?;
                }
            }
            InverseLink::Standard(_) => {
                self.likelihood.irls_update(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    &mut self.lastmu,
                    &mut self.lastweights,
                    &mut self.lastz,
                    integrated,
                    None,
                )?;
            }
        }

        let deviance = self
            .likelihood
            .loglik_deviance(self.y, &self.lastmu, self.priorweights)?;
        let penalty_term = self.penalty.shifted_quadratic(beta.as_ref());
        let penalized_objective = deviance + penalty_term;
        let arithmetic_finite = penalized_objective.is_finite()
            && self.workspace.eta_buf.iter().all(|v| v.is_finite())
            && self.lastmu.iter().all(|v| v.is_finite())
            && self.lastweights.iter().all(|v| v.is_finite());
        Ok(CandidateScreen {
            penalized_objective,
            deviance,
            penalty_term,
            arithmetic_finite,
        })
    }
}


impl<'a> WorkingModel for GamWorkingModel<'a> {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
    }

    fn update_with_curvature(
        &mut self,
        beta: &Coefficients,
        requested_curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        let n = self.offset.len();
        if self.workspace.eta_buf.len() != n {
            self.workspace.eta_buf = Array1::zeros(n);
        }
        if self.workspace.matvec_buf.len() != n {
            self.workspace.matvec_buf = Array1::zeros(n);
        }
        let mut matvec_tmp = std::mem::take(&mut self.workspace.matvec_buf);
        self.transformed_matvec_into(beta, &mut matvec_tmp);
        self.workspace.eta_buf.assign(&self.offset);
        self.workspace.eta_buf += &matvec_tmp;
        self.workspace.matvec_buf = matvec_tmp;

        // Estimate the Gamma dispersion shape once from the warm-start η and
        // freeze it for the remainder of this inner solve. Holding the shape
        // fixed keeps the product φ·λ constant, so the penalized argmin β̂ is a
        // stationary target and the LM gain ratio stays consistent across trial
        // and accepted iterates. The shape refreshes across outer iterations
        // because a fresh model is built per inner solve. See issue #511.
        if self.likelihood.scale.gamma_shape_is_estimated() && !self.gamma_shape_locked {
            let shape =
                estimate_gamma_shape_from_eta(self.y, &self.workspace.eta_buf, self.priorweights);
            self.likelihood = self.likelihood.clone().with_gamma_shape(shape);
            self.gamma_shape_locked = true;
        }

        // Estimate the Beta precision φ once from the warm-start η and freeze it
        // for this inner solve (issue #567). φ enters the IRLS weights and the
        // variance `Var(y)=mu(1-mu)/(1+φ)`; holding it fixed within the inner
        // solve keeps the penalized argmin β̂ stationary (mirroring the Gamma
        // shape lock above), and it refreshes across outer iterations as a fresh
        // working model is built per inner solve. With φ pinned at the seed of 1
        // the mean smooth was over-penalized / under-fit on precise data.
        if self.likelihood.scale.beta_phi_is_estimated() && !self.beta_phi_locked {
            let phi =
                estimate_beta_phi_from_eta(self.y, &self.workspace.eta_buf, self.priorweights);
            self.likelihood = self.likelihood.clone().with_beta_phi(phi);
            self.beta_phi_locked = true;
        }

        // Estimate the Tweedie dispersion φ once from the warm-start η and freeze
        // it for this inner solve (issue #771). φ enters the IRLS weight
        // `prior·μ^{2−p}/φ` (and so the covariance Vb = H⁻¹, giving SE ∝ √φ);
        // holding it fixed within the inner solve keeps the product φ·λ — hence
        // the penalized argmin β̂ — a stationary LM target (mirroring the Gamma
        // shape and Beta φ locks above), and it refreshes across outer iterations
        // as a fresh working model is built per inner solve.
        if self.likelihood.scale.tweedie_phi_is_estimated() && !self.tweedie_phi_locked {
            if let ResponseFamily::Tweedie { p } = self.likelihood.spec.response {
                let phi = estimate_tweedie_phi_from_eta(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    p,
                );
                self.likelihood = self.likelihood.clone().with_tweedie_phi(phi);
                self.tweedie_phi_locked = true;
            }
        }

        // Estimate the Negative-Binomial overdispersion `theta` once from the
        // warm-start η and freeze it for this inner solve (issue #802). `theta`
        // enters the working weight `W = μθ/(θ+μ)` (the NB2 Fisher information)
        // and the working response, so holding it fixed within the inner solve
        // keeps the penalized argmin β̂ a stationary LM target (mirroring the Beta
        // φ lock above); it refreshes across outer iterations as a fresh working
        // model is built per inner solve. With `theta` frozen at the seed every
        // coefficient/η SE ignored the data's overdispersion.
        if self.likelihood.scale.negbin_theta_is_estimated() && !self.negbin_theta_locked {
            let theta =
                estimate_negbin_theta_from_eta(self.y, &self.workspace.eta_buf, self.priorweights);
            self.likelihood = self.likelihood.clone().with_negbin_theta(theta);
            self.negbin_theta_locked = true;
        }

        // Use integrated (GHQ) likelihood if per-observation SE is available.
        // This coherently accounts for uncertainty in the base prediction.
        let integrated = self.covariate_se.as_ref().map(|se| IntegratedWorkingInput {
            quadctx: &self.quadctx,
            se: se.view(),
            mixture_link_state: self.link_kind.mixture_state(),
            sas_link_state: self.link_kind.sas_state(),
        });
        match &self.link_kind {
            InverseLink::Mixture(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                }
            }
            InverseLink::LatentCLogLog(_) | InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                }
            }
            InverseLink::Standard(_) => {
                self.likelihood.irls_update(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    &mut self.lastmu,
                    &mut self.lastweights,
                    &mut self.lastz,
                    integrated,
                    Some(WorkingDerivativeBuffersMut {
                        c: &mut self.last_c,
                        d: &mut self.last_d,
                        dmu_deta: &mut self.last_dmu_deta,
                        d2mu_deta2: &mut self.last_d2mu_deta2,
                        d3mu_deta3: &mut self.last_d3mu_deta3,
                    }),
                )?;
            }
        }
        let mut firth = FirthDiagnostics::Inactive;
        if self.firth_bias_reduction {
            if !inverse_link_has_fisher_weight_jet(&self.link_kind) {
                crate::bail_invalid_estim!(
                    "Firth/Jeffreys PIRLS requested for unsupported inverse link {:?}",
                    self.link_kind
                );
            }
            // IMPORTANT: Jeffreys/Firth bias reduction must be computed in the
            // *same coefficient basis* as the inner objective being optimized by PIRLS.
            //
            // The working response (z) and the coefficients β are in the transformed
            // basis when a reparameterization is used. The Jeffreys term is the
            // identifiable-subspace Fisher logdet evaluated on a canonical
            // orthonormal basis of the transformed design column space,
            // not a raw-coordinate logdet. Its PIRLS hat-diagonal adjustment must
            // therefore be computed from that same transformed-design Fisher
            // matrix, otherwise the inner objective and the outer LAML
            // derivatives disagree.
            //
            // This mismatch is subtle but severe: it leaves the analytic gradient
            // differentiating a *different* objective than the one PIRLS actually
            // solved, and the gradient check fails catastrophically.
            //
            // Rule: use X_transformed if available; fall back to X_original only
            // when PIRLS is operating directly in the original basis.
            let (hat_diag, jeffreys_logdet, firth_score_shift) = match &self.coordinate_design {
                WorkingCoordinateDesign::TransformedExplicit {
                    x_transformed,
                    x_csr,
                } => {
                    if x_transformed.as_sparse().is_some() {
                        let csr = x_csr.as_ref().ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "missing CSR cache for sparse transformed design".to_string(),
                            )
                        })?;
                        compute_jeffreys_pirls_diagnostics_sparse(
                            &self.link_kind,
                            csr,
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    } else {
                        let x_dense_cow = x_transformed.to_dense_cow();
                        compute_jeffreys_pirls_diagnostics(
                            &self.link_kind,
                            x_dense_cow.view(),
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    }
                }
                WorkingCoordinateDesign::TransformedImplicit { transform } => {
                    // Jeffreys/Firth MUST use a consistent basis. TransformedImplicit
                    // stores s_transformed in the Qs basis, so we need X in that
                    // same basis.  Materialize X·Qs on demand (Firth models are
                    // typically small clinical logistic regressions).
                    let x_t_dense =
                        fast_ab(&self.x_original.to_dense(), &transform.materialize_dense());
                    compute_jeffreys_pirls_diagnostics(
                        &self.link_kind,
                        x_t_dense.view(),
                        self.workspace.eta_buf.view(),
                        self.priorweights,
                    )?
                }
                WorkingCoordinateDesign::OriginalSparseNative => {
                    // s_transformed is in original coords here (qs = I).
                    if self.x_original.as_sparse().is_some() {
                        let csr = self.x_original_csr.as_ref().ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "missing CSR cache for sparse original design".to_string(),
                            )
                        })?;
                        compute_jeffreys_pirls_diagnostics_sparse(
                            &self.link_kind,
                            csr,
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    } else {
                        let x_dense = self
                            .x_original
                            .try_to_dense_arc(
                                "Firth diagnostics require dense access to the original design",
                            )
                            .map_err(EstimationError::InvalidInput)?;
                        compute_jeffreys_pirls_diagnostics(
                            &self.link_kind,
                            x_dense.view(),
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    }
                }
            };
            firth = FirthDiagnostics::Active {
                jeffreys_logdet,
                hat_diag: hat_diag.clone(),
            };
            // Apply the link-general Firth working-response shift `Δ_i` built by
            // the operator (`½ (w'_i/w_i) h_diag_i`). PIRLS then solves
            // `Xᵀ W (z* − η) = 0`, so the Firth term it adds to the score is
            // `Σ_i w_i Δ_i x_i = ½ Σ_i w'_i h_diag_i x_i = ∂Φ/∂β` — exactly the
            // Jeffreys score the outer REML differentiates. For the canonical
            // logit `Δ_i` equals the historical `h_i (½ − μ_i)/w_i`; for probit /
            // cloglog it carries the correct non-canonical `w'_i/w_i` instead of
            // the logit-pinned `(½ − μ_i)`, so the inner mode and the outer
            // objective no longer disagree.
            ndarray::Zip::from(&mut self.lastz)
                .and(&firth_score_shift)
                .and(&self.lastweights)
                .par_for_each(|zi, &delta_i, &wi| {
                    if wi > 0.0 {
                        *zi += delta_i;
                    }
                });
        }

        let z = &self.lastz;
        // Fused single-pass: compute weighted_residual = (eta - z) * w
        // and working_residual = eta - z simultaneously, avoiding two
        // separate O(n) passes and an intermediate copy.
        ndarray::Zip::from(&mut self.workspace.weighted_residual)
            .and(&mut self.workspace.working_residual)
            .and(&self.workspace.eta_buf)
            .and(z)
            .and(&self.lastweights)
            .par_for_each(|wr, r, &eta, &zi, &wi| {
                let residual = eta - zi;
                *r = residual;
                *wr = residual * wi;
            });
        let mut gradient = self.transformed_transpose_matvec(&self.workspace.weighted_residual);
        // Score norm ||X' (weighted residual)||_2 — captured before adding the
        // penalty contribution so the natural gradient scale can be assembled
        // for the scale-invariant convergence certificate.
        let score_norm = array1_l2_norm(&gradient);
        let s_beta = self.penalty.shifted_gradient(beta.as_ref());
        let s_beta_norm = array1_l2_norm(&s_beta);
        gradient += &s_beta;
        let hessian_curvature = self.update_hessian_curvature_arrays(requested_curvature)?;
        self.lasthessian_curvature = hessian_curvature;

        // Build solver-side weights in the reusable n-buffer: apply a
        // per-observation SPD floor so the Newton linear system is
        // well-conditioned, without contaminating the model weights stored in
        // `lasthessian_weights`.
        if self.workspace.matvec_buf.len() != n {
            self.workspace.matvec_buf = Array1::zeros(n);
        }
        solver_hessian_weights_into(
            &self.lasthessian_weights,
            &self.lastweights,
            &mut self.workspace.matvec_buf,
        );
        let solver_weights = std::mem::take(&mut self.workspace.matvec_buf);

        let (penalized_hessian, sparsehessian, ridge_used) = if matches!(
            self.coordinate_design,
            WorkingCoordinateDesign::OriginalSparseNative
        ) {
            // The SPD-check factor is discarded here: the downstream consumer
            // is the LM Newton step, which always factorizes
            // (H + loop_lambda · I) with a non-zero loop_lambda (initial value
            // 1e-6), so it sees a different matrix.
            let (h_sparse, _factor, ridge_used) =
                ensure_sparse_positive_definitewithridge(|ridge| {
                    self.sparse_penalized_hessian(&solver_weights, ridge)
                })?;
            (Array2::zeros((0, 0)), Some(h_sparse), ridge_used)
        } else {
            let mut penalized_hessian = self.penalized_hessian(&solver_weights)?;
            assert_symmetric_tol(&penalized_hessian, "PIRLS penalized Hessian", 1e-8);
            let ridge_used = ensure_positive_definitewithridge(
                &mut penalized_hessian,
                "PIRLS penalized Hessian",
            )?;
            (penalized_hessian, None, ridge_used)
        };
        self.workspace.matvec_buf = solver_weights;

        // Match the stabilized Hessian used by the outer LAML objective.
        // If a ridge is needed, we treat it as an explicit penalty term:
        //
        //   l_p(β; ρ) = l(β) - 0.5 * βᵀ S_λ β - 0.5 * ridge * ||β||²
        //
        // This keeps the PIRLS fixed point aligned with the stabilized Hessian
        // that drives log|H| and the implicit-gradient correction.
        let deviance = self
            .likelihood
            .loglik_deviance(self.y, &self.lastmu, self.priorweights)?;
        let log_likelihood = calculate_loglikelihood_omitting_constants(
            self.y,
            &self.lastmu,
            &self.likelihood,
            self.priorweights,
        );

        let mut penalty_term = self.penalty.shifted_quadratic(beta.as_ref());
        let mut ridge_grad_norm = 0.0;
        if ridge_used > 0.0 {
            let ridge_penalty = ridge_used * beta.as_ref().dot(beta.as_ref());
            penalty_term += ridge_penalty;
            gradient.zip_mut_with(beta.as_ref(), |g, &b| *g += ridge_used * b);
            ridge_grad_norm = ridge_used * array1_l2_norm(beta.as_ref());
        }

        self.last_penalty_term = penalty_term;
        let gradient_natural_scale = score_norm + s_beta_norm + ridge_grad_norm;

        Ok(WorkingState {
            eta: LinearPredictor::new(std::mem::replace(
                &mut self.workspace.eta_buf,
                Array1::zeros(0),
            )),
            gradient,
            hessian: match sparsehessian {
                Some(h_sparse) => crate::linalg::matrix::SymmetricMatrix::Sparse(h_sparse),
                None => crate::linalg::matrix::SymmetricMatrix::Dense(penalized_hessian),
            },

            log_likelihood,
            deviance,
            penalty_term,
            firth,
            ridge_used,
            hessian_curvature,
            gradient_natural_scale,
        })
    }

    fn update_candidate(
        &mut self,
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        if !self.firth_bias_reduction {
            return self.update_with_curvature(beta, curvature);
        }
        let firth_enabled = self.firth_bias_reduction;
        self.firth_bias_reduction = false;
        let result = self.update_with_curvature(beta, curvature);
        self.firth_bias_reduction = firth_enabled;
        result
    }

    fn screen_candidate(
        &mut self,
        beta: &Coefficients,
        direction: &Array1<f64>,
        current_eta: &LinearPredictor,
        curvature: HessianCurvatureKind,
    ) -> Result<CandidateEvaluation, EstimationError> {
        if self.firth_bias_reduction {
            return self
                .update_candidate(beta, curvature)
                .map(CandidateEvaluation::Full);
        }
        self.screen_candidate_from_direction(beta, direction, current_eta)
            .map(CandidateEvaluation::Screen)
    }

    fn supports_observed_information_curvature(&self) -> bool {
        self.supports_observed_hessian_curvature()
    }
}


// Cutoff between the dense outer-product backend and sparse SpGEMM. At p=1024
// the dense p×p output buffer is 8 MiB — L3-resident on most current targets
// and small enough that per-thread copies used during parallel reduction stay
// within an order of magnitude of the cache hierarchy.
