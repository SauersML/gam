//! Reusable inner-loop scratch (`PirlsWorkspace`), the P-IRLS options bundle
//! (`WorkingModelPirlsOptions`), the arrow-Schur structured-inner-solve
//! descriptor, and the arrow-latent snapshot/restore/commit helpers.

use super::*;

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
    pub(crate) sparse_penalized_system_cache: Option<SparsePenalizedSystemCache>,
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
    pub fn new(n: usize, p: usize, _: usize, _: usize) -> Self {
        // Default implementation ignores this parameter.
        // Default implementation ignores this parameter.
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
    pub(crate) fn ensure_sparse_penalty_cache(
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
    /// Used by `execute_pirls_if_needed` (in `solver::reml::outer_eval`)
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

pub(crate) fn restore_arrow_latent_if_needed(
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
