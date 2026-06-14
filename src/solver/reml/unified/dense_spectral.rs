use super::*;

/// How the penalized Hessian's log-determinant and its derivatives treat the
/// spectrum below the stability floor `ε = spectral_epsilon(·)`.
///
/// Two conventions, both mathematically internally consistent:
///
/// ## `Smooth` (default — appropriate for almost all GLM/GAM families)
///
/// Eigenvalues above the structural positive-eigenvalue threshold — the same
/// ~100·p·ε_mach·‖H‖ cutoff that `fixed_subspace_penalty_rank_and_logdet`
/// applies to `log|S|_+` — contribute to `log|H|` via the smooth regularizer
/// `r_ε(σ) = ½(σ + √(σ² + 4ε²))`.  Gradients use `φ'(σ) = 1/√(σ² + 4ε²)`
/// so that `d log|H|_reg/dρ = Σ φ'(σ_j) · u_j^T (dH/dρ) u_j` is the EXACT
/// derivative of the scalar objective `Σ log r_ε(σ_j)` over the active set.
/// For a well-conditioned H the threshold sits far below every genuine
/// eigenvalue and every pair is active, so behaviour matches the previous
/// unfiltered soft-floor formulation.  In the rank-deficient regime where
/// `rank(X) + rank(S) < p` (e.g. small-n high-dim Duchon), H has eigenvalues
/// inside the numerical noise band; those directions are also null in S, so
/// excluding them from BOTH `log|H|` and `log|S|_+` keeps the LAML ratio
/// well-defined on the identified subspace rather than driving
/// `½ log|H| − ½ log|S|_+` to −∞.
///
/// ## `HardPseudo` (opt-in for structurally rank-deficient families)
///
/// When the model is known to carry a numerical null-space direction that
/// is not informative — e.g. multi-block GAMLSS wiggle models where the
/// threshold + constant wiggle-intercept are collinear — the smooth floor
/// still contributes to `log|H|_reg` through that direction, and its
/// first-order `dσ/dρ = u^T (dH/dρ) u` estimate is unreliable because the
/// eigenvector u for a near-zero σ is a random linear combination of
/// whatever the numerical eigensolver selected inside the null space.
///
/// Under `HardPseudo`, eigenvalues satisfying `σ_j ≤ ε` are EXCLUDED from
/// `log|H|`, `tr(G_ε · A)`, `tr(H⁻¹ · ·)`, and every cross-trace.  This is
/// the exact pseudo-logdeterminant on the active eigenspace:
///
///   log|H|₊  = Σ_{σ_j > ε} log σ_j
///   d/dρ_k   = Σ_{σ_j > ε} (1/σ_j) · u_j^T (dH/dρ_k) u_j
///
/// with the smooth floor `r_ε(σ)` retained in place of `log σ` / `1/σ` so
/// there is no discontinuity as an eigenvalue crosses ε.  The key property
/// is that null-space directions drop out of both the cost and the
/// gradient in a matched way; first-order perturbation theory applies only to
/// directions that actually have curvature to perturb.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PseudoLogdetMode {
    #[default]
    Smooth,
    HardPseudo,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dense spectral HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Dense spectral Hessian operator using eigendecomposition.
///
/// Computes logdet, trace, and solve from a single eigendecomposition,
/// guaranteeing spectral consistency. Indefinite or near-singular eigenvalues
/// are handled via smooth spectral regularization `r_ε(σ)` rather than hard
/// clamping, ensuring that logdet and inverse use the same smooth mapping.
pub struct DenseSpectralOperator {
    /// Regularized eigenvalues: `r_ε(σ_i)` for each raw eigenvalue `σ_i`.
    pub(crate) reg_eigenvalues: Vec<f64>,
    /// Per-eigenvalue mask: `true` if the eigenpair participates in all
    /// traces, solves, and logdet contributions.  Under
    /// [`PseudoLogdetMode::Smooth`] every entry is `true`.  Under
    /// [`PseudoLogdetMode::HardPseudo`] entries with `σ_j ≤ ε` are `false`,
    /// so the numerical null space is excluded consistently from
    /// `log|H|_+`, its gradient, its cross-traces, AND `H⁻¹` solves
    /// (`H⁺` on the active subspace).
    pub(crate) active_mask: Vec<bool>,
    /// Eigenvectors of H (columns).
    pub(crate) eigenvectors: Array2<f64>,
    /// Precomputed: W = U diag(1/√r_ε(σ)) for efficient traces.
    /// trace(H⁻¹ A) = Σ (AW ⊙ W)
    pub(crate) w_factor: Array2<f64>,
    /// Precomputed kernel K_ab = 1 / (r_a r_b) for exact H⁻¹ cross traces in
    /// the eigenbasis.
    hinv_cross_kernel: Array2<f64>,
    /// Precomputed: G = U diag(1/√(√(σ² + 4ε²))) for logdet gradient traces.
    /// trace(G_ε(H) A) = Σ (AG ⊙ G) where G_ε uses φ'(σ) = 1/√(σ² + 4ε²).
    pub(crate) g_factor: Array2<f64>,
    /// Precomputed divided-difference kernel Γ for exact logdet Hessian cross traces
    /// in the eigenbasis.
    logdet_hessian_kernel: Array2<f64>,
    /// Precomputed log-determinant: Σ ln(r_ε(σ_i)).
    pub(crate) cached_logdet: f64,
    pub(crate) projected_factor_cache: ProjectedFactorCache,
    /// Full dimension.
    pub(crate) n_dim: usize,
}

impl DenseSpectralOperator {
    pub fn reg_eigenvalue(&self, k: usize) -> f64 {
        self.reg_eigenvalues[k]
    }
    pub fn eigenvector_entry(&self, i: usize, k: usize) -> f64 {
        self.eigenvectors[[i, k]]
    }
    /// Whether eigenpair `k` is active in the operator's logdet, traces,
    /// and solves. Under `PseudoLogdetMode::Smooth` this is always `true`;
    /// under `HardPseudo` it is `false` when `σ_k ≤ ε`.
    pub fn eigenpair_active(&self, k: usize) -> bool {
        self.active_mask[k]
    }

    /// Create from a symmetric matrix (may be indefinite or singular).
    ///
    /// The eigendecomposition is computed once. Eigenvalues are smoothly
    /// regularized via `r_ε(σ)`. All subsequent operations (logdet, trace,
    /// solve) use the regularized spectrum, ensuring mathematical consistency.
    pub fn from_symmetric(h: &Array2<f64>) -> Result<Self, String> {
        Self::from_symmetric_with_mode(h, PseudoLogdetMode::Smooth)
    }

    /// Variant of [`from_symmetric`](Self::from_symmetric) that selects the
    /// log-determinant convention.
    ///
    /// See [`PseudoLogdetMode`] for the derivation and the exact set of
    /// kernels that differ between the two modes.  At a high level:
    /// `Smooth` keeps every eigenpair in play with a soft floor, whereas
    /// `HardPseudo` masks out `σ_j ≤ ε` consistently across logdet,
    /// gradient traces, cross-traces, and the H⁻¹ kernels.
    pub fn from_symmetric_with_mode(
        h: &Array2<f64>,
        mode: PseudoLogdetMode,
    ) -> Result<Self, String> {
        use faer::Side;

        let n = h.nrows();
        if n != h.ncols() {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "HessianOperator: expected square matrix, got {}×{}",
                    n,
                    h.ncols()
                ),
            }
            .into());
        }

        let (eigenvalues, eigenvectors) = h
            .eigh(Side::Lower)
            .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

        let epsilon = spectral_epsilon(eigenvalues.as_slice().unwrap());

        // `active[j]` selects which eigenpairs participate in every trace
        // and in the cached logdet.
        //
        // `Smooth` is the regularized full-spectrum mode: every eigenpair stays
        // active and singular directions are handled only through
        // `r_ε(σ)`. This is the documented default semantics used by the
        // unified REML/LAML objective.
        //
        // `HardPseudo` is the identified-subspace mode: eigenpairs with
        // `σ_j ≤ ε` are excluded consistently from logdet, traces, and solves.
        // Families that need exact pseudo-determinant behaviour opt into this
        // mode explicitly through `pseudo_logdet_mode()`.
        let active: Vec<bool> = match mode {
            PseudoLogdetMode::Smooth => vec![true; n],
            PseudoLogdetMode::HardPseudo => eigenvalues.iter().map(|&s| s > epsilon).collect(),
        };

        // Apply smooth regularization to all eigenvalues (even inactive ones:
        // `reg_eigenvalues[j]` is still consulted by `trace_hinv_product`
        // when using `w_factor[:, j]`, but we zero-out `w_factor[:, j]` for
        // inactive eigenpairs so those entries never enter any sum).
        let reg_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .map(|&sigma| spectral_regularize(sigma, epsilon))
            .collect();

        // Build W factor for traces: W[:, j] = u_j / sqrt(r_ε(σ_j)) on
        // active eigenpairs, 0 otherwise.
        let mut w_factor = Array2::zeros((n, n));
        for j in 0..n {
            if !active[j] {
                continue;
            }
            let scale = 1.0 / reg_eigenvalues[j].sqrt();
            for row in 0..n {
                w_factor[[row, j]] = eigenvectors[[row, j]] * scale;
            }
        }

        let mut hinv_cross_kernel = Array2::zeros((n, n));
        for a in 0..n {
            if !active[a] {
                continue;
            }
            let inv_ra = 1.0 / reg_eigenvalues[a];
            for b in 0..n {
                if !active[b] {
                    continue;
                }
                hinv_cross_kernel[[a, b]] = inv_ra / reg_eigenvalues[b];
            }
        }

        // Build G factor for logdet gradient traces: G[:, j] = u_j / sqrt(√(σ_j² + 4ε²))
        // φ'(σ) = 1/√(σ² + 4ε²), so we need 1/√(φ'(σ)) = (σ² + 4ε²)^{1/4}
        // Actually: tr(G_ε A) = Σ_j φ'(σ_j) u_jᵀ A u_j = Σ (AG ⊙ G)
        // where G[:, j] = u_j · √(φ'(σ_j)) = u_j / (σ_j² + 4ε²)^{1/4}
        let four_eps_sq = 4.0 * epsilon * epsilon;
        let mut g_factor = Array2::zeros((n, n));
        for j in 0..n {
            if !active[j] {
                continue;
            }
            let sigma = eigenvalues[j];
            let phi_prime = 1.0 / (sigma * sigma + four_eps_sq).sqrt();
            let scale = phi_prime.sqrt();
            for row in 0..n {
                g_factor[[row, j]] = eigenvectors[[row, j]] * scale;
            }
        }

        let mut logdet_hessian_kernel = Array2::zeros((n, n));
        let sqrt_disc: Vec<f64> = eigenvalues
            .iter()
            .map(|&s| (s * s + four_eps_sq).sqrt())
            .collect();
        for a in 0..n {
            if !active[a] {
                continue;
            }
            let sigma_a = eigenvalues[a];
            let sqrt_a = sqrt_disc[a];
            for b in 0..n {
                if !active[b] {
                    continue;
                }
                logdet_hessian_kernel[[a, b]] = if a == b {
                    -sigma_a / (sqrt_a * sqrt_a * sqrt_a)
                } else {
                    let sigma_b = eigenvalues[b];
                    let sqrt_b = sqrt_disc[b];
                    -(sigma_a + sigma_b) / (sqrt_a * sqrt_b * (sqrt_a + sqrt_b))
                };
            }
        }

        // Precompute logdet: Σ_{active} ln(r_ε(σ_i)).
        let cached_logdet: f64 = reg_eigenvalues
            .iter()
            .zip(active.iter())
            .filter_map(|(&v, &act)| if act { Some(v.ln()) } else { None })
            .sum();

        Ok(Self {
            reg_eigenvalues,
            active_mask: active,
            eigenvectors,
            w_factor,
            hinv_cross_kernel,
            g_factor,
            logdet_hessian_kernel,
            cached_logdet,
            projected_factor_cache: ProjectedFactorCache::default(),
            n_dim: n,
        })
    }

    #[inline]
    pub(crate) fn rotate_to_eigenbasis(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let left = crate::faer_ndarray::fast_atb(&self.eigenvectors, matrix);
        crate::faer_ndarray::fast_ab(&left, &self.eigenvectors)
    }

    /// Factor `F` satisfying `trace(G_epsilon(H) A) = trace(F^T A F)`.
    ///
    /// Structured row-local operators use this to contract the logdet-gradient
    /// trace directly in row space without forming `A F` in coefficient space.
    pub fn logdet_gradient_factor(&self) -> &Array2<f64> {
        &self.g_factor
    }

    #[inline]
    pub(crate) fn trace_hinv_product_cross_rotated(
        &self,
        a_rot: &Array2<f64>,
        b_rot: &Array2<f64>,
    ) -> f64 {
        let mut result = 0.0;
        for ((kernel_row, a_row), b_col) in self
            .hinv_cross_kernel
            .rows()
            .into_iter()
            .zip(a_rot.rows().into_iter())
            .zip(b_rot.columns().into_iter())
        {
            for ((kernel, a_value), b_value) in kernel_row
                .iter()
                .copied()
                .zip(a_row.iter().copied())
                .zip(b_col.iter().copied())
            {
                result += kernel * a_value * b_value;
            }
        }
        result
    }

    #[inline]
    pub(crate) fn trace_hinv_product_cross_dense(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let a_rot = self.rotate_to_eigenbasis(a);
        if std::ptr::eq(a, b) {
            return self.trace_hinv_product_cross_rotated(&a_rot, &a_rot);
        }
        let b_rot = self.rotate_to_eigenbasis(b);
        self.trace_hinv_product_cross_rotated(&a_rot, &b_rot)
    }

    #[inline]
    pub(crate) fn projected_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let left = crate::faer_ndarray::fast_atb(&self.w_factor, matrix);
        crate::faer_ndarray::fast_ab(&left, &self.w_factor)
    }

    #[inline]
    pub(crate) fn projected_operator(
        &self,
        factor: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> Array2<f64> {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let result = op.projected_matrix_cached(factor, &self.projected_factor_cache);
            let signature = format!(
                "DenseSpectralOperator::projected_operator dim={} rank={} implicit={}",
                self.n_dim,
                factor.ncols(),
                op.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            op.projected_matrix_cached(factor, &self.projected_factor_cache)
        }
    }

    #[inline]
    pub(crate) fn trace_projected_cross(&self, left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        let mut result = 0.0;
        for (left_row, right_col) in left.rows().into_iter().zip(right.columns().into_iter()) {
            for (left_value, right_value) in left_row.iter().copied().zip(right_col.iter().copied())
            {
                result += left_value * right_value;
            }
        }
        result
    }

    #[inline]
    pub(crate) fn trace_logdet_hessian_cross_rotated(
        &self,
        h_i_rot: &Array2<f64>,
        h_j_rot: &Array2<f64>,
    ) -> f64 {
        let mut result = 0.0;
        for ((kernel_row, h_i_row), h_j_col) in self
            .logdet_hessian_kernel
            .rows()
            .into_iter()
            .zip(h_i_rot.rows().into_iter())
            .zip(h_j_rot.columns().into_iter())
        {
            for ((kernel, h_i_value), h_j_value) in kernel_row
                .iter()
                .copied()
                .zip(h_i_row.iter().copied())
                .zip(h_j_col.iter().copied())
            {
                result += kernel * h_i_value * h_j_value;
            }
        }
        result
    }
}

/// Coalesce repeated identical `[STAGE]` log lines from `DenseSpectralOperator`
/// methods. First occurrence of a (method, dims, implicit-flags) signature
/// logs immediately; identical consecutive repeats are silenced and accrue
/// into a counter, emitting heartbeat summaries at doubling cadence
/// (2, 4, 8, 16, …) and a final summary when the signature changes.
pub(crate) fn dense_spectral_stage_log(signature: &str, elapsed_s: f64) {
    use std::sync::Mutex;
    pub(crate) struct Repeat {
        signature: String,
        count: u64,
        total: f64,
        min: f64,
        max: f64,
        next_heartbeat: u64,
    }
    pub(crate) static REPEAT: Mutex<Option<Repeat>> = Mutex::new(None);

    let mut guard = match REPEAT.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };

    if let Some(state) = guard.as_mut() {
        if state.signature == signature {
            state.count += 1;
            state.total += elapsed_s;
            if elapsed_s < state.min {
                state.min = elapsed_s;
            }
            if elapsed_s > state.max {
                state.max = elapsed_s;
            }
            if state.count >= state.next_heartbeat {
                log::info!(
                    "[STAGE] {} (×{} so far, total={:.3}s min={:.3}s max={:.3}s avg={:.3}s)",
                    state.signature,
                    state.count,
                    state.total,
                    state.min,
                    state.max,
                    state.total / state.count as f64,
                );
                state.next_heartbeat = state.next_heartbeat.saturating_mul(2);
            }
            return;
        }
        // Signature changed — flush a final summary for the previous one
        // when it ran more than once (the first occurrence already logged
        // its own line, so a count of 1 needs no follow-up).
        if state.count > 1 {
            log::info!(
                "[STAGE] {} final ×{} total={:.3}s min={:.3}s max={:.3}s avg={:.3}s",
                state.signature,
                state.count,
                state.total,
                state.min,
                state.max,
                state.total / state.count as f64,
            );
        }
    }

    log::info!("[STAGE] {} elapsed={:.3}s", signature, elapsed_s);
    *guard = Some(Repeat {
        signature: signature.to_string(),
        count: 1,
        total: elapsed_s,
        min: elapsed_s,
        max: elapsed_s,
        next_heartbeat: 2,
    });
}

impl HessianOperator for DenseSpectralOperator {
    pub(crate) fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    pub(crate) fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self)
    }

    pub(crate) fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        Ok(assemble_h_raw_dense(self))
    }

    pub(crate) fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // tr(H_reg⁻¹ A) = Σ_j (1/r_ε(σ_j)) uⱼᵀAuⱼ
        // Computed as Σ (AW ⊙ W) where W = U diag(1/√r_ε(σ)).
        let aw = a.dot(&self.w_factor);
        aw.iter()
            .zip(self.w_factor.iter())
            .map(|(&a, &w)| a * w)
            .sum()
    }

    pub(crate) fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        // H_reg⁻¹ v = Σ_j (1/r_ε(σ_j)) (uⱼᵀv) uⱼ.  Inactive eigenpairs
        // (σ_j ≤ ε under `HardPseudo`) are skipped so the returned vector
        // lives entirely in the active subspace — otherwise v_k picks up a
        // huge spurious component along the numerical null space direction
        // (coefficient ~ 1/r_ε(σ_j) for σ_j ≈ 0) that is not part of the
        // IFT mode response `dβ̂/dρ` and would leak into the REML gradient.
        let mut result = Array1::zeros(self.n_dim);
        for j in 0..self.n_dim {
            if !self.active_mask[j] {
                continue;
            }
            let u = self.eigenvectors.column(j);
            let coeff = u.dot(rhs) / self.reg_eigenvalues[j];
            for row in 0..self.n_dim {
                result[row] += coeff * u[row];
            }
        }
        result
    }

    pub(crate) fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut projected = self.eigenvectors.t().dot(rhs);
        for j in 0..self.n_dim {
            if self.active_mask[j] {
                let scale = 1.0 / self.reg_eigenvalues[j];
                projected.row_mut(j).mapv_inplace(|value| value * scale);
            } else {
                // Zero out inactive eigendirections so `H⁺` acts on the
                // active subspace only (mirroring the single-vector `solve`).
                projected.row_mut(j).fill(0.0);
            }
        }
        self.eigenvectors.dot(&projected)
    }

    pub(crate) fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.trace_hinv_product_cross_dense(a, b)
    }

    pub(crate) fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let result =
                op.trace_projected_factor_cached(&self.w_factor, &self.projected_factor_cache);
            let signature = format!(
                "DenseSpectralOperator::trace_hinv_operator dim={} rank={} implicit={}",
                self.n_dim,
                self.w_factor.ncols(),
                op.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            op.trace_projected_factor_cached(&self.w_factor, &self.projected_factor_cache)
        }
    }

    pub(crate) fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        let left = self.w_factor.t().dot(matrix).dot(&self.w_factor);
        let right = self.projected_operator(&self.w_factor, op);
        self.trace_projected_cross(&left, &right)
    }

    pub(crate) fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let left_proj = self.projected_operator(&self.w_factor, left);
            let result = if std::ptr::addr_eq(left, right) {
                self.trace_projected_cross(&left_proj, &left_proj)
            } else {
                let right_proj = self.projected_operator(&self.w_factor, right);
                self.trace_projected_cross(&left_proj, &right_proj)
            };
            let signature = format!(
                "DenseSpectralOperator::trace_hinv_operator_cross dim={} rank={} left_implicit={} right_implicit={}",
                self.n_dim,
                self.w_factor.ncols(),
                left.is_implicit(),
                right.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            let left_proj = self.projected_operator(&self.w_factor, left);
            if std::ptr::addr_eq(left, right) {
                self.trace_projected_cross(&left_proj, &left_proj)
            } else {
                let right_proj = self.projected_operator(&self.w_factor, right);
                self.trace_projected_cross(&left_proj, &right_proj)
            }
        }
    }

    pub(crate) fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        // tr(G_ε(H) A) = Σ_j φ'(σ_j) uⱼᵀAuⱼ
        // where φ'(σ) = 1/√(σ² + 4ε²).
        // Computed as Σ (AG ⊙ G) where G = U diag(√φ'(σ)).
        let ag = a.dot(&self.g_factor);
        ag.iter()
            .zip(self.g_factor.iter())
            .map(|(&a, &g)| a * g)
            .sum()
    }

    pub(crate) fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        // h^G_i = ‖(X G)_{i,:}‖² where G_ε = G Gᵀ and G = self.g_factor.
        // The dominant cost at large scale is the (n × p)·(p × rank) matmul
        // — for matern60 with n=320K, p=101 that's ~3.3 GFLOPs and the
        // ndarray default `.dot()` runs single-threaded (no BLAS feature
        // enabled in this crate's build), so we route through faer's parallel
        // SIMD GEMM. For operator-backed (Lazy) designs we additionally
        // stream by row chunk so we never materialize the full (n×p) block
        // at large scale.
        let n = x.nrows();
        let p = x.ncols();
        let rank = self.g_factor.ncols();
        let mut h = Array1::<f64>::zeros(n);
        if n == 0 || p == 0 || rank == 0 {
            return h;
        }
        // Issue #922: offload this n-dependent pass to the device pool when a
        // GPU was probed and n·p² clears the dispatch floor. The result is the
        // same f64 arithmetic (X·G then row-wise ‖·‖²), just relocated across
        // every device via `scatter_batched`; any failure falls through to the
        // faer CPU stream below so the REML criterion is byte-for-byte
        // unchanged on machines without a GPU.
        if let Some(gpu) =
            crate::gpu::linalg::try_fast_spectral_leverage_diagonal(x, self.g_factor.view())
        {
            return gpu;
        }
        let chunk_rows = byte_balanced_row_chunk(p + rank, n);
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk_rows).min(n);
            let rows = x.try_row_chunk(start..end).unwrap_or_else(|err| {
                // SAFETY: `try_row_chunk` only fails on operator implementation
                // bugs — the `start..end` range is constructed from
                // `0..n = 0..x.nrows()` with `end = (start+block).min(n)`,
                // so it is always a valid sub-range of `x`. A failure here
                // means the operator violated its row-chunk contract.
                // SAFETY: row range built from 0..x.nrows(); failure means operator broke its contract.
                reml_contract_panic(format!(
                    "xt_logdet_kernel_x_diagonal: row chunk failed: {err}"
                ))
            });
            let xg = crate::faer_ndarray::fast_ab(&rows, &self.g_factor);
            for (local, row) in xg.outer_iter().enumerate() {
                h[start + local] = row.iter().map(|v| v * v).sum();
            }
            start = end;
        }
        h
    }

    pub(crate) fn trace_logdet_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(G_ε A) = Σ (A·G ⊙ G) for block-local A.
        // Only needs G[start..end, :] — O(block² × rank) instead of O(p² × rank).
        let g_block = self.g_factor.slice(ndarray::s![start..end, ..]);
        let ag = block.dot(&g_block);
        scale
            * ag.iter()
                .zip(g_block.iter())
                .map(|(&a, &g)| a * g)
                .sum::<f64>()
    }

    pub(crate) fn trace_hinv_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(H_reg⁻¹ A) = Σ (A·W ⊙ W) for block-local A.
        let w_block = self.w_factor.slice(ndarray::s![start..end, ..]);
        let aw = block.dot(&w_block);
        scale
            * aw.iter()
                .zip(w_block.iter())
                .map(|(&a, &w)| a * w)
                .sum::<f64>()
    }

    pub(crate) fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(H⁻¹ A H⁻¹ A) where A = scale · embed(block, start, end) and
        // `block` is the symmetric (b × b) local matrix.
        //
        // H⁻¹ = W W^T, so the symmetric block is
        //   H⁻¹_block = W_block · W_block^T,   W_block = W[start..end, :].
        // For block-local A, only the [start..end, start..end] sub-block of
        //   H⁻¹ A H⁻¹ A
        // contributes nonzero diagonal entries:
        //   tr(H⁻¹ A H⁻¹ A) = scale² · tr( (H⁻¹_block · B)² )
        //                    = scale² · tr( (W_block^T B W_block)² )
        // (cyclic on the rank-sized symmetric M = W_block^T B W_block, then
        // tr(M²) = ||M||_F² because B is symmetric so M is symmetric).
        let w_block = self.w_factor.slice(ndarray::s![start..end, ..]);
        let bw = block.dot(&w_block); // (b × rank)
        let m = w_block.t().dot(&bw); // (rank × rank), symmetric for symmetric block
        let scale_sq = scale * scale;
        scale_sq * m.iter().map(|&v| v * v).sum::<f64>()
    }

    pub(crate) fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let result =
                op.trace_projected_factor_cached(&self.g_factor, &self.projected_factor_cache);
            let signature = format!(
                "DenseSpectralOperator::trace_logdet_operator dim={} rank={} implicit={}",
                self.n_dim,
                self.g_factor.ncols(),
                op.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            op.trace_projected_factor_cached(&self.g_factor, &self.projected_factor_cache)
        }
    }

    pub(crate) fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        let hp_i = self.rotate_to_eigenbasis(h_i);
        if std::ptr::eq(h_i, h_j) {
            return self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_i);
        }
        let hp_j = self.rotate_to_eigenbasis(h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    pub(crate) fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        let hp_i = self.rotate_to_eigenbasis(h_i);
        let hp_j = self.projected_operator(&self.eigenvectors, h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    pub(crate) fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        let hp_i = self.projected_operator(&self.eigenvectors, h_i);
        if std::ptr::addr_eq(h_i, h_j) {
            return self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_i);
        }
        let hp_j = self.projected_operator(&self.eigenvectors, h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    pub(crate) fn trace_logdet_hessian_crosses(&self, matrices: &[&Array2<f64>]) -> Array2<f64> {
        let n = matrices.len();
        let rotated = matrices
            .iter()
            .map(|matrix| self.rotate_to_eigenbasis(matrix))
            .collect::<Vec<_>>();
        let mut out = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let value = self.trace_logdet_hessian_cross_rotated(&rotated[i], &rotated[j]);
                out[[i, j]] = value;
                out[[j, i]] = value;
            }
        }
        out
    }

    pub(crate) fn active_rank(&self) -> usize {
        self.active_mask.iter().filter(|&&active| active).count()
    }

    pub(crate) fn dim(&self) -> usize {
        self.n_dim
    }

    pub(crate) fn is_dense(&self) -> bool {
        true
    }

    pub(crate) fn prefers_stochastic_trace_estimation(&self) -> bool {
        false
    }

    pub(crate) fn logdet_traces_match_hinv_kernel(&self) -> bool {
        false
    }

    pub(crate) fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self)
    }
}
