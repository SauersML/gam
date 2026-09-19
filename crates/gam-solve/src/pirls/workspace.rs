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
    // Gradient helper
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
    // Reusable p×p buffer for Hessian assembly (avoids per-iteration allocation).
    pub hessian_buf: Array2<f64>,
    // Reusable n-length buffer for X*β matvec (avoids per-iteration allocation in update).
    pub matvec_buf: Array1<f64>,
    // #1412: device-resident design `X` for the GPU `XᵀWX` Gram. The inner P-IRLS
    // loop rebuilds the Gram once per Newton/LM iterate with the SAME design `X`
    // (only the working weights `w` move), so re-uploading the full n×p `X` on
    // every iterate starves the device on H2D staging (measured ~98% of the
    // pipeline at <20% utilisation). This caches the device-resident `X` keyed on
    // its host data pointer + shape, so the first Gram of an inner solve uploads
    // `X` and every later iterate crosses only the n-vector `w` H2D and the p×p
    // Gram D2H. `None` whenever CUDA is unavailable / the shape is below the GPU
    // Gram threshold / the upload failed — the caller keeps its per-call path.
    pub(crate) resident_design_gram: Option<(
        usize,
        usize,
        usize,
        gam_gpu::linalg_dispatch::ResidentDesignGram,
    )>,
}

impl PirlsWorkspace {
    /// Coefficient-space scratch for an exact sufficient-statistic solve.
    ///
    /// The Gaussian value-only rho lane has already reduced every data-row
    /// contribution into `XᵀWX`, `XᵀW(y-offset)`, and the centered response
    /// norm. It never enters an IRLS iteration, so allocating the general
    /// workspace's five observation-length vectors would reintroduce O(n)
    /// work before the zero-iteration branch can consume those statistics.
    pub(crate) fn coefficient_only(p: usize) -> Self {
        Self::new(0, p)
    }

    /// Scratch for an `n`-row, `p`-coefficient PIRLS solve.
    ///
    /// Penalty-block extents are deliberately *not* parameters: every stage
    /// buffer that depends on them is allocated on demand by the path that
    /// needs it, so sizing them here only pre-committed memory no consumer read.
    pub fn new(n: usize, p: usize) -> Self {
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
            hessian_buf: Array2::zeros((0, 0).f()),
            matvec_buf: Array1::zeros(n),
            resident_design_gram: None,
        }
    }

    pub(super) fn add_dense_xtwx_signed(
        weights: &Array1<f64>,
        x: &Array2<f64>,
        out: &mut Array2<f64>,
    ) {
        *out = crate::estimate::reml::assembly::xt_diag_x_dense(x, weights);
    }

    /// Ensure the sparse penalty cache is populated and consistent with `x` and `s_lambda`.
    pub(crate) fn ensure_sparse_penalty_cache(
        &mut self,
        x: &SparseColMat<usize, f64>,
        s_lambda: &Array2<f64>,
    ) -> Result<(), EstimationError> {
        let penalty_pattern = SparsePenaltyPattern::from_dense_upper(s_lambda);
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
        Ok(self
            .sparse_penalized_system_cache
            .as_ref()
            .expect("ensure_sparse_penalty_cache installs the cache or returns Err")
            .stats())
    }

    // Phase 2 hook: numeric sparse penalized-system assembly in original coordinates.
    pub(super) fn assemble_sparse_penalized_hessian(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
        s_lambda: &Array2<f64>,
        precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        self.ensure_sparse_penalty_cache(x, s_lambda)?;
        self.sparse_penalized_system_cache
            .as_mut()
            .expect("ensure_sparse_penalty_cache installs the cache or returns Err")
            .assemble_upper(x, weights, precomputed_xtwx)
    }
}

#[derive(Clone, Debug)]
pub struct WorkingModelPirlsOptions {
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub adaptive_kkt_tolerance: Option<AdaptiveKktTolerance>,
    pub max_step_halving: usize,
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
}

