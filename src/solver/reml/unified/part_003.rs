use dense_projection::{dense_projected_matrix, dense_trace_projected_factor};


fn reml_contract_panic(message: impl Into<String>) -> ! {
    std::panic::panic_any(message.into())
}


// ═══════════════════════════════════════════════════════════════════════════
//  Typed errors for the unified REML/LAML evaluator.
//
//  The evaluator and its helpers historically returned `Result<_, String>`.
//  Internally we now build typed errors at the leaves and convert at the
//  boundary via `From<RemlError> for String`, which is byte-equivalent to
//  the previous `format!(...)` strings so external callers continue to see
//  the same diagnostic text.
// ═══════════════════════════════════════════════════════════════════════════

/// Typed failure categories raised by the unified REML/LAML evaluator and
/// its outer-Hessian / penalty-root helpers.
///
/// Each variant carries a pre-formatted `reason` string so that the
/// `Display` impl is byte-equivalent to the original `format!(...)` text the
/// module emitted before the typed-error migration. External signatures
/// remain `Result<_, String>`; the boundary conversion goes through
/// `From<RemlError> for String`.
#[derive(Debug, Clone)]
pub enum RemlError {
    /// A length / shape disagreement between two views that should match
    /// (penalty coords vs Hessian dim, residual length vs operator dim,
    /// precomputed-correction count vs total, etc.).
    DimensionMismatch { reason: String },
    /// A scalar / vector / matrix entry that must be finite came back NaN
    /// or ±∞ (cost, gradient entry, Hessian entry, cross-trace entry).
    NonFiniteValue { reason: String },
    /// A correction path was invoked against an operator kernel that does
    /// not support it (scalar-only correction on a non-scalar kernel,
    /// callback correction on a non-callback kernel).
    InvalidKernelMode { reason: String },
    /// A caller violated the evaluator contract. These are not numerical
    /// failures; they mean an upstream solver presented an inner state with
    /// insufficient certificates for the requested derivative surface.
    ContractViolation { reason: String },
}


impl std::fmt::Display for RemlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemlError::DimensionMismatch { reason }
            | RemlError::NonFiniteValue { reason }
            | RemlError::InvalidKernelMode { reason }
            | RemlError::ContractViolation { reason } => f.write_str(reason),
        }
    }
}


impl std::error::Error for RemlError {}


impl From<RemlError> for String {
    fn from(err: RemlError) -> String {
        err.to_string()
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Debug stash: thread-local capture of (op_total, U) from the ext-grad path,
//  used by the iso-κ Duchon FD investigation test. The stash type itself and
//  the per-thread TLS live in `crate::test_support::debug_stash` so that the
//  test reader (`take_terms`) and the production writer (`store_terms`)
//  share a single source of truth — a previous incomplete refactor left
//  duplicate `TermStash` definitions in this module and in `test_support`,
//  routing writes to one TLS and reads to the other so the diagnostic tests
//  always saw empty captures.
// ═══════════════════════════════════════════════════════════════════════════

pub use crate::test_support::debug_stash;


// ═══════════════════════════════════════════════════════════════════════════
//  Core traits
// ═══════════════════════════════════════════════════════════════════════════

/// Fit-level stochastic trace state shared by all adaptive Hutchinson batches.
///
/// `monotone_probe_floor` pins the CRN prefix length across batches. The
/// `cg_warm_starts` map stores the previous H⁻¹ solve for the same deterministic
/// probe id so the next outer evaluation can initialize matrix-free trace CG
/// from the matching probe only.
#[derive(Debug, Default)]
pub struct StochasticTraceState {
    pub monotone_probe_floor: usize,
    pub cg_warm_starts: HashMap<u64, Array1<f64>>,
    pub solve_rel_tol_override: Option<f64>,
    pub last_linear_residual_norm: Option<f64>,
    pub last_probe_sigma_sq: Option<f64>,
    pub last_probe_count: usize,
}


/// Abstract interface for Hessian linear algebra operations.
///
/// All operations use the SAME internal decomposition, ensuring spectral
/// consistency between logdet (used in cost) and trace/solve (used in gradient).
///
/// Implementors:
/// - `DenseSpectralOperator`: eigendecomposition of dense H
/// - Sparse Cholesky operators (external implementations)
/// - `BlockCoupledOperator`: eigendecomposition of joint multi-block H
/// Minimum operator dimension at which the Hutch++ stochastic trace estimator is
/// preferred over materializing an implicit operator densely. Below this, the
/// `2·m_s + m_h` Hutch++ matvecs do not beat `dim` dense H⁻¹ HVPs, so the dense
/// fallback is cheaper.
const HUTCHPP_TRACE_MIN_DIM: usize = 128;


/// Build the Hutch++ stochastic-trace configuration for an operator of the given
/// dimension. The sketch dimension grows with `dim` (one column per 32 of
/// dimension, bounded to `[4, 16]`), and the probe budget tracks the sketch so
/// the estimator's variance and cost stay balanced across problem sizes. Shared
/// by every implicit-operator trace path so they cannot drift apart.
fn hutchpp_config_for_dim(dim: usize) -> StochasticTraceConfig {
    const SKETCH_DIM_PER: usize = 32;
    const SKETCH_DIM_MIN: usize = 4;
    const SKETCH_DIM_MAX: usize = 16;
    const PROBES_PER_SKETCH: usize = 4;
    const PROBES_MAX_FLOOR: usize = 32;
    const PROBES_MIN_FLOOR: usize = 8;
    let sketch = (dim / SKETCH_DIM_PER).clamp(SKETCH_DIM_MIN, SKETCH_DIM_MAX);
    let mut config = StochasticTraceConfig::default();
    config.hutchpp_sketch_dim = Some(sketch);
    config.n_probes_max = (sketch * PROBES_PER_SKETCH).max(PROBES_MAX_FLOOR);
    config.n_probes_min = sketch.max(PROBES_MIN_FLOOR);
    config
}


pub trait HessianOperator: Send + Sync {
    /// log|H|₊ — pseudo-logdet using only active eigenvalues/pivots.
    fn logdet(&self) -> f64;

    /// tr(H₊⁻¹ A) — trace of pseudo-inverse times a symmetric matrix.
    /// Uses the SAME decomposition as `logdet`.
    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64;

    /// Exact dense spectral representation, when this backend has one.
    ///
    /// Outer-Hessian assembly uses this to batch all logdet-Hessian cross
    /// traces in the eigenbasis. For CTN scale-dimension fits this avoids
    /// projecting the same implicit ψ drift once per upper-triangular pair.
    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        None
    }

    /// Assemble the raw dense Hessian represented by this backend for
    /// active-constraint tangent projection.
    ///
    /// Backends that do not store either a dense spectral decomposition or an
    /// explicit factorization should keep the default error.
    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        Err("backend does not support tangent projection".to_string())
    }

    /// tr(H₊⁻¹ B) for an operator-backed Hessian drift.
    ///
    /// Default implementation materializes `B` densely. Backends with
    /// native operator traces (notably sparse Cholesky) should override it.
    ///
    /// For HVP-only (implicit) operators on large problems we route
    /// through Hutch++ — the Meyer–Musco split estimator achieves O(1/ε)
    /// matvecs vs O(1/ε²) for plain Hutchinson, and avoids the O(p²)
    /// memory + O(p) HVP cost of materializing the operator densely.
    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        // Hutch++ fast path for the warn-and-materialize default. Only
        // backends that fall through to this default reach here;
        // backends with native operator traces override it. We require
        // an implicit operator (so materialization is expensive) and a
        // moderately-large dim (so 2 m_s + m_h matvecs beats `dim`
        // dense HVPs).
        if op.is_implicit() && self.dim() >= HUTCHPP_TRACE_MIN_DIM {
            let config = hutchpp_config_for_dim(self.dim());
            return hutchpp_estimate_trace_hinv_operator(self, op, &config);
        }
        if op.is_implicit() {
            log::warn!(
                "trace_hinv_operator: materializing implicit HyperOperator — \
                 backend should provide a matrix-free override"
            );
        }
        self.trace_hinv_product(&op.to_dense())
    }

    /// Efficient computation of tr(H₊⁻¹ Hₖ) for the third-derivative contraction.
    ///
    /// For non-Gaussian families, Hₖ = Aₖ + Xᵀ diag(c ⊙ Xvₖ) X where
    /// vₖ = H⁻¹(Aₖβ̂). This method allows backends to compute the contraction
    /// efficiently without forming the full p×p correction matrix.
    ///
    /// Default implementation: forms the correction and calls `trace_hinv_product`.
    fn trace_hinv_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        let base = self.trace_hinv_product(a_k);
        match third_deriv_correction {
            Some(c) => base + self.trace_hinv_product(c),
            None => base,
        }
    }

    /// H⁻¹ v — linear solve using the active decomposition.
    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64>;

    /// H⁻¹ M — multi-column solve.
    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64>;

    /// H⁻¹ v for stochastic trace probes.
    ///
    /// Exact backends use the normal solve. Matrix-free backends may override
    /// this to use a looser PCG tolerance when the caller's Monte Carlo error
    /// dominates the linear-solve error.
    fn stochastic_trace_solve(&self, rhs: &Array1<f64>, rel_tol: f64) -> Array1<f64> {
        assert!(
            rel_tol.is_finite() && rel_tol > 0.0,
            "stochastic trace solve tolerance must be positive and finite"
        );
        self.solve(rhs)
    }

    /// H⁻¹ v for a deterministic stochastic trace probe id.
    ///
    /// Backends with matrix-free CG may use `probe_id` to warm-start from the
    /// previous solve of the same CRN probe. The default exact backend ignores
    /// the id and uses the normal stochastic trace solve.
    fn stochastic_trace_solve_for_probe(
        &self,
        rhs: &Array1<f64>,
        rel_tol: f64,
        probe_id: u64,
        state: Option<&Arc<Mutex<StochasticTraceState>>>,
    ) -> Array1<f64> {
        // Default exact backend has no matrix-free CG, so per-probe warm
        // starts are inapplicable. If a previous matrix-free backend left
        // a warm-start vector for this `probe_id` in the shared state,
        // drop it so a later matrix-free run does not consume a vector
        // that was generated against a different operator factorization.
        if let Some(state_arc) = state
            && let Ok(mut guard) = state_arc.lock()
        {
            guard.cg_warm_starts.remove(&probe_id);
        }
        self.stochastic_trace_solve(rhs, rel_tol)
    }

    /// H⁻¹ M for stochastic trace probes.
    fn stochastic_trace_solve_multi(&self, rhs: &Array2<f64>, rel_tol: f64) -> Array2<f64> {
        assert!(
            rel_tol.is_finite() && rel_tol > 0.0,
            "stochastic trace multi-solve tolerance must be positive and finite"
        );
        self.solve_multi(rhs)
    }

    /// Whether this backend exposes a matrix-free operator usable by trace CG.
    fn has_matrix_free_trace_cg_operator(&self) -> bool {
        false
    }

    /// tr(H⁻¹ A H⁻¹ B) for dense symmetric Hessian drifts.
    ///
    /// This is the second-order trace object used by EFS denominators and the
    /// ψ-block trace Gram preconditioner. The default implementation computes
    /// both solved column stacks exactly and contracts them as
    /// `tr((H⁻¹A)(H⁻¹B))`.
    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let solved_a = self.solve_multi(a);
        if std::ptr::eq(a, b) {
            return trace_matrix_product(&solved_a, &solved_a);
        }
        let solved_b = self.solve_multi(b);
        trace_matrix_product(&solved_a, &solved_b)
    }

    /// tr(H⁻¹ A H⁻¹ B) for a dense drift `A` and an operator-backed drift `B`.
    ///
    /// Default implementation materializes the operator and dispatches to the
    /// dense cross-trace path. Matrix-free and sparse backends should override
    /// this to avoid dense operator materialization.
    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        if op.is_implicit() && self.dim() >= HUTCHPP_TRACE_MIN_DIM {
            let config = hutchpp_config_for_dim(self.dim());
            // Wrap the dense LHS in a matrix-backed HyperOperator so the
            // shared cross routine can call mul_vec_into on it.
            let lhs = DenseMatrixHyperOperator {
                matrix: matrix.clone(),
            };
            return hutchpp_estimate_trace_hinv_operator_cross(self, &lhs, op, &config);
        }
        if op.is_implicit() {
            log::warn!(
                "trace_hinv_matrix_operator_cross: materializing implicit HyperOperator — \
                 backend should provide a matrix-free override"
            );
        }
        self.trace_hinv_product_cross(matrix, &op.to_dense())
    }

    /// tr(H⁻¹ A H⁻¹ B) for operator-backed Hessian drifts.
    ///
    /// Default implementation materializes both operators densely. Backends
    /// with native operator-aware cross traces should override this.
    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        let l_implicit = left.is_implicit();
        let r_implicit = right.is_implicit();
        if (l_implicit || r_implicit) && self.dim() >= HUTCHPP_TRACE_MIN_DIM {
            let config = hutchpp_config_for_dim(self.dim());
            // Same-operator self-cross is PSD; the squared form is the
            // exact algorithm for that case (lower variance, no sign).
            if std::ptr::eq(
                left as *const dyn HyperOperator as *const (),
                right as *const dyn HyperOperator as *const (),
            ) {
                return hutchpp_estimate_trace_hinv_op_squared(self, left, &config);
            }
            return hutchpp_estimate_trace_hinv_operator_cross(self, left, right, &config);
        }
        if l_implicit || r_implicit {
            log::warn!(
                "trace_hinv_operator_cross: materializing implicit HyperOperator(s) — \
                 backend should provide a matrix-free override"
            );
        }
        self.trace_hinv_product_cross(&left.to_dense(), &right.to_dense())
    }

    /// tr(G_ε(H) A) — trace for the logdet gradient ∂_i log|R_ε(H)|.
    ///
    /// For non-spectral backends (Cholesky), G_ε = H⁻¹ and this reduces to
    /// `trace_hinv_product`. For spectral regularization, G_ε uses eigenvalues
    /// `φ'(σ_a) = 1/√(σ_a² + 4ε²)` instead of `1/r_ε(σ_a)`.
    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        self.trace_hinv_product(a)
    }

    /// diag(X · G_ε(H) · Xᵀ) — the leverage corresponding to `trace_logdet_gradient`.
    /// `trace_logdet_gradient(Xᵀ diag(w) X) = Σᵢ wᵢ · h^G[i]`.
    ///
    /// Streams the rows of `X` through the design's `try_row_chunk` so
    /// operator-backed (Lazy) designs never materialize the full (n×p)
    /// block at large scale.
    fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        assert!(self.logdet_traces_match_hinv_kernel());
        let n = x.nrows();
        let p = x.ncols();

        let block = {
            const TARGET_CHUNK_FLOATS: usize = 1 << 16;
            (TARGET_CHUNK_FLOATS / p.max(1)).clamp(1, n.max(1))
        };

        let mut h = Array1::<f64>::zeros(n);
        let mut start = 0usize;
        while start < n {
            let end = (start + block).min(n);
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
            let chunk_t = rows.t().to_owned();
            let z_chunk = self.solve_multi(&chunk_t);
            for (i, (row, z_col)) in rows
                .outer_iter()
                .zip(z_chunk.columns().into_iter())
                .enumerate()
            {
                let mut acc = 0.0;
                for (row_value, z_value) in row.iter().copied().zip(z_col.iter().copied()) {
                    acc += row_value * z_value;
                }
                h[start + i] = acc;
            }
            start = end;
        }
        h
    }

    /// tr(G_ε(H) B) for an operator-backed Hessian drift.
    ///
    /// Default implementation materializes `B` densely. For Cholesky-based
    /// backends this equals `trace_hinv_operator`.
    ///
    /// When `logdet_traces_match_hinv_kernel()` is true (Cholesky-style
    /// backends where `trace_logdet_gradient(A) = trace_hinv_product(A)`)
    /// and the operator is implicit on a moderate-or-large problem, route
    /// through Hutch++ to avoid the dense materialization. Spectral
    /// backends override this to false (their logdet trace uses
    /// regularized eigenvalue weights, not `H⁻¹`), so they keep the
    /// materialize path or provide their own override.
    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        if op.is_implicit()
            && self.dim() >= HUTCHPP_TRACE_MIN_DIM
            && self.logdet_traces_match_hinv_kernel()
        {
            let config = hutchpp_config_for_dim(self.dim());
            return hutchpp_estimate_trace_hinv_operator(self, op, &config);
        }
        if op.is_implicit() {
            log::warn!(
                "trace_logdet_operator: materializing implicit HyperOperator — \
                 backend should provide a matrix-free override"
            );
        }
        self.trace_logdet_gradient(&op.to_dense())
    }

    /// Efficient computation of tr(G_ε(H) Hₖ) for the logdet gradient,
    /// analogous to `trace_hinv_h_k` but using the logdet gradient operator.
    ///
    /// Default implementation: forms the correction and calls `trace_logdet_gradient`.
    fn trace_logdet_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        let base = self.trace_logdet_gradient(a_k);
        match third_deriv_correction {
            Some(c) => base + self.trace_logdet_gradient(c),
            None => base,
        }
    }

    /// Efficient computation of tr(G_ε(H) B_k) for an operator-backed Hessian drift,
    /// optionally plus the dense third-derivative correction.
    fn trace_logdet_h_k_operator(
        &self,
        b_k: &dyn HyperOperator,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        let base = self.trace_logdet_operator(b_k);
        match third_deriv_correction {
            Some(c) => base + self.trace_logdet_gradient(c),
            None => base,
        }
    }

    /// tr(G_ε(H) · A_block) where A_block is a p_block × p_block matrix
    /// embedded at rows/columns [start..end].
    ///
    /// This avoids materializing the full p×p matrix for block-structured
    /// penalties. The default implementation builds the full matrix and
    /// delegates to `trace_logdet_gradient`; spectral backends override
    /// this with O(p_block × active_rank) work.
    fn trace_logdet_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        let p = self.dim();
        let mut full = Array2::<f64>::zeros((p, p));
        let bs = end - start;
        for i in 0..bs {
            for j in 0..bs {
                full[[start + i, start + j]] = scale * block[[i, j]];
            }
        }
        self.trace_logdet_gradient(&full)
    }

    /// tr(H₊⁻¹ · A_block) where A_block is embedded at [start..end].
    /// Same block-local optimization as `trace_logdet_block_local`.
    fn trace_hinv_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        let p = self.dim();
        let mut full = Array2::<f64>::zeros((p, p));
        let bs = end - start;
        for i in 0..bs {
            for j in 0..bs {
                full[[start + i, start + j]] = scale * block[[i, j]];
            }
        }
        self.trace_hinv_product(&full)
    }

    /// tr(H⁻¹ A H⁻¹ A) for a block-local penalty matrix A embedded at [start..end].
    ///
    /// `block` is the p_block × p_block local penalty matrix and `scale` is the
    /// smoothing parameter (λ_k). The full A = scale · embed(block, start, end).
    ///
    /// Default implementation materializes the full p×p matrix and delegates to
    /// `trace_hinv_product_cross`. The `DenseSpectralOperator` override uses
    /// W-factor slicing for O(rank × block_size × (block_size + p)) work.
    fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        let p = self.dim();
        let bs = end - start;
        let mut full = Array2::<f64>::zeros((p, p));
        for i in 0..bs {
            for j in 0..bs {
                full[[start + i, start + j]] = scale * block[[i, j]];
            }
        }
        self.trace_hinv_product_cross(&full, &full)
    }

    /// Cross-trace for the logdet Hessian:
    /// `∂²_{ij} log|R_ε(H)| = tr(G_ε Ḧ_{ij}) + spectral_cross(Ḣ_i, Ḣ_j)`.
    ///
    /// This method computes the `spectral_cross(Ḣ_i, Ḣ_j)` part, which for
    /// non-spectral backends equals `-tr(H⁻¹ Ḣ_j H⁻¹ Ḣ_i)`.
    ///
    /// For spectral regularization, the divided-difference kernel Γ_{ab} replaces
    /// the simple product of inverses.
    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        // Default: standard formula -tr(H⁻¹ Ḣ_j H⁻¹ Ḣ_i) = -⟨Y_j^T, Y_i⟩_F
        // where Y_i = H⁻¹ Ḣ_i.
        let y_i = self.solve_multi(h_i);
        if std::ptr::eq(h_i, h_j) {
            return -trace_matrix_product(&y_i, &y_i);
        }
        let y_j = self.solve_multi(h_j);
        -trace_matrix_product(&y_j, &y_i)
    }

    /// Operator-backed mixed form of [`trace_logdet_hessian_cross`].
    ///
    /// The default materializes the operator; spectral and sparse backends
    /// override this to keep the exact analytic cross trace matrix-free.
    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.trace_logdet_hessian_cross(h_i, &h_j.to_dense())
    }

    /// Operator-backed form of [`trace_logdet_hessian_cross`].
    ///
    /// The default materializes both operators; exact backends override this
    /// when they can contract the logdet-Hessian kernel against operator
    /// projections directly.
    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.trace_logdet_hessian_cross(&h_i.to_dense(), &h_j.to_dense())
    }

    /// Batched cross traces for the logdet Hessian:
    /// `cross[i,j] = trace_logdet_hessian_cross(H_i, H_j)`.
    ///
    /// The default implementation applies `trace_logdet_hessian_cross`
    /// pairwise. Dense spectral backends override this to rotate each drift
    /// into the eigenbasis once and reuse the same divided-difference kernel
    /// across all pairs.
    fn trace_logdet_hessian_crosses(&self, matrices: &[&Array2<f64>]) -> Array2<f64> {
        let n = matrices.len();
        let mut out = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let value = self.trace_logdet_hessian_cross(matrices[i], matrices[j]);
                out[[i, j]] = value;
                out[[j, i]] = value;
            }
        }
        out
    }

    /// Number of active dimensions (rank of pseudo-inverse).
    fn active_rank(&self) -> usize;

    /// Full dimension of H.
    fn dim(&self) -> usize;

    /// Whether this operator is backed by a dense factorization.
    ///
    /// Dense operators (eigendecomposition) have O(p²) trace cost per matrix,
    /// making stochastic trace estimation worthwhile for large p.  Sparse
    /// operators (Cholesky) have O(nnz) solve cost, so exact column-by-column
    /// traces are already cheap and stochastic estimation is not needed.
    fn is_dense(&self) -> bool {
        false
    }

    /// Whether the unified evaluator should batch large trace computations
    /// through the stochastic Hutchinson path for this operator.
    ///
    /// Dense eigendecomposition backends prefer this once `p` is large because
    /// exact per-coordinate traces are O(p²). Matrix-free iterative backends
    /// have the same preference even though they do not store a dense factor.
    fn prefers_stochastic_trace_estimation(&self) -> bool {
        self.is_dense()
    }

    /// Whether stochastic Hutchinson estimates based on `H⁻¹` are valid for
    /// logdet-gradient / logdet-Hessian trace terms on this backend.
    ///
    /// This is true for plain SPD-logdet operators where
    /// `trace_logdet_gradient(A) = tr(H⁻¹ A)` and
    /// `trace_logdet_hessian_cross(A, B) = -tr(H⁻¹ A H⁻¹ B)`.
    ///
    /// Smooth spectral regularization does not satisfy those identities, so
    /// dense spectral backends must override this to `false`.
    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        true
    }

    /// Access the dense spectral backend when this operator is powered by a
    /// single eigendecomposition.
    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        None
    }
}


/// Representative curvature scale for a Hessian operator.
///
/// Returns the geometric mean of the active Hessian eigenvalues,
/// `exp(log|H|_+ / rank(H))`. This has the same physical units as a Hessian
/// diagonal entry but is basis-invariant, cheap after the operator has computed
/// its log-determinant, and well-defined for both dense spectral and
/// matrix-free operator paths.
pub fn hessian_operator_geometric_scale(op: &dyn HessianOperator) -> Option<f64> {
    let rank = op.active_rank();
    if rank == 0 {
        return None;
    }
    let logdet = op.logdet();
    if !logdet.is_finite() {
        return None;
    }
    let scale = (logdet / rank as f64).exp();
    if scale.is_finite() && scale > 0.0 {
        Some(scale)
    } else {
        None
    }
}


/// Provider of family-specific Hessian derivative information.
///
/// The REML/LAML gradient requires ∂H/∂ρₖ. For Gaussian, this is just Aₖ = λₖSₖ.
/// For non-Gaussian GLMs, the working curvature W(η) depends on β̂, so
/// ∂H/∂ρₖ = Aₖ + Xᵀ diag(c ⊙ Xvₖ) X where vₖ = −dβ̂/dρₖ.
/// For block-coupled families (GAMLSS, survival), the correction is
/// D_β H_L[−vₖ] using the joint likelihood Hessian.
///
/// This trait abstracts over all three cases.
pub trait HessianDerivativeProvider: Send + Sync {
    /// Compute the third-derivative correction to Hₖ.
    ///
    /// Given the mode response vₖ = H⁻¹(Aₖβ̂), returns the correction matrix
    /// such that Hₖ = Aₖ + correction.
    ///
    /// Returns `None` for Gaussian (c=d=0, no correction needed).
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String>;

    /// Operator-capable version of `hessian_derivative_correction`.
    ///
    /// Implementations may override this to return matrix-free or composite
    /// drifts without forcing dense materialization.
    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        Ok(self
            .hessian_derivative_correction(v_k)?
            .map(DriftDerivResult::Dense))
    }

    /// Batched first-order correction hook for families whose
    /// `D_beta H[u_k]` operators share row-local state across all smoothing
    /// coordinates. The default preserves the single-direction semantics.
    fn hessian_derivative_corrections_result(
        &self,
        v_ks: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        v_ks.iter()
            .map(|v_k| self.hessian_derivative_correction_result(v_k))
            .collect()
    }

    fn has_batched_hessian_derivative_corrections(&self) -> bool {
        false
    }

    /// Compute the second-order correction to H_{k,l} for the outer Hessian.
    ///
    /// Returns `None` if not needed or not implemented.
    fn hessian_second_derivative_correction(
        &self,
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
        arr3: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
        assert!(arr3.iter().all(|v| !v.is_nan()));
        if self.has_corrections() {
            Err(
                "HessianDerivativeProvider reports first-order corrections but does not implement second-order correction"
                    .to_string(),
            )
        } else {
            Ok(None)
        }
    }

    /// Operator-capable version of `hessian_second_derivative_correction`.
    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        Ok(self
            .hessian_second_derivative_correction(v_k, v_l, u_kl)?
            .map(DriftDerivResult::Dense))
    }

    /// Batched second-order correction hook. The K(K+1)/2 ρ-ρ pairs in
    /// `compute_outer_hessian` each call
    /// `hessian_second_derivative_correction_result(v_k, v_l, u_kl)`; for
    /// families whose `D²H[v_k, v_l]` operators share row-local state (one
    /// per-row scan across n observations that evaluates against all
    /// triples in parallel) the batched form amortises the row-walk across
    /// pairs instead of re-scanning n rows per pair. The default preserves
    /// the single-direction semantics by looping over the singular hook.
    /// Pair the override with
    /// `has_batched_hessian_second_derivative_corrections` so the unified
    /// evaluator only routes through this when a family actually fuses the
    /// per-row work.
    ///
    /// Wired into `compute_outer_hessian`'s parallel ρ-ρ pair loop: when a
    /// provider's `has_batched_hessian_second_derivative_corrections`
    /// returns `true`, the loop precomputes all K(K+1)/2 triples (one
    /// shared `hop.solve_multi` over the pair-stacked RHS), batch-calls
    /// this hook once per outer Hessian assembly, then traces the
    /// returned drifts through the projected subspace kernel before the
    /// parallel pair sweep starts. Otherwise the loop falls back to
    /// per-pair `hessian_second_derivative_correction_result`.
    fn hessian_second_derivative_corrections_result(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        triples
            .iter()
            .map(|(v_k, v_l, u_kl)| {
                self.hessian_second_derivative_correction_result(v_k, v_l, u_kl)
            })
            .collect()
    }

    fn has_batched_hessian_second_derivative_corrections(&self) -> bool {
        false
    }

    /// Whether this provider has non-trivial corrections.
    /// False for Gaussian, true for GLMs and coupled families.
    fn has_corrections(&self) -> bool;

    /// Raw ingredients for the adjoint trace optimization.
    ///
    /// When available, the evaluator can use these to compute
    /// tr(H⁻¹ C[u]) = uᵀ z_c  (O(p) dot product instead of O(p²) solve)
    /// and fourth-derivative traces directly, without the trait having to
    /// implement the optimization algorithm.
    ///
    /// Returns `None` for Gaussian (no corrections), multi-predictor,
    /// and coupled families where the optimization doesn't apply.
    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        None
    }

    /// Owned data needed for matrix-free outer Hessian-vector products.
    ///
    /// Providers that can express their second-order corrections through an
    /// owned scalar-GLM kernel or owned callback closures should override
    /// this so the unified evaluator can return an exact outer Hv operator
    /// instead of forcing dense materialization.
    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        self.scalar_glm_ingredients()
            .map(OuterHessianDerivativeKernel::from_scalar_glm)
    }

    /// Family-supplied exact outer Hessian operator over θ = (ρ, ψ).
    ///
    /// When a family can produce the full profiled outer Hessian as a
    /// matrix-free Hv operator without enumerating θ_iθ_j pairs, it returns
    /// `Some(op)` here.  The unified evaluator then short-circuits the
    /// kernel-based assembly path at
    /// [`reml_laml_evaluate`](self::reml_laml_evaluate) and routes the result
    /// straight into [`HessianResult::Operator`].
    ///
    /// Default returns `None`, in which case the evaluator falls through to
    /// the existing `outer_hessian_derivative_kernel` / `compute_outer_hessian`
    /// path.  This is the contract surface for CTN, survival, GAMLSS and
    /// other families that ship a directional outer-HVP operator.
    fn family_outer_hessian_operator(
        &self,
    ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
        None
    }
}


/// Raw ingredients for the adjoint trace optimization in scalar GLMs.
///
/// For single-predictor GLMs, the third-derivative correction is
///   C[u] = Xᵀ diag(c ⊙ Xu) X
/// and the fourth-derivative correction is
///   Q[vₖ, vₗ] = Xᵀ diag(d ⊙ (Xvₖ)(Xvₗ)) X
///
/// The evaluator uses these arrays to implement the adjoint trace trick
/// and compute fourth-derivative traces without materializing p×p matrices.
pub struct ScalarGlmIngredients<'a> {
    /// c = dW/dη, the third-derivative weight array.
    pub c_array: &'a Array1<f64>,
    /// d = d²W/dη², the fourth-derivative weight array (`None` if zero).
    pub d_array: Option<&'a Array1<f64>>,
    /// Design matrix X in the transformed basis.
    pub x: &'a DesignMatrix,
}


#[derive(Clone)]
pub enum OuterHessianDerivativeKernel {
    /// Gaussian/constant-curvature families have no likelihood drift corrections.
    /// This marker still enables the unified exact outer-HVP operator, whose
    /// penalty/logdet/profiled-dispersion terms are fully analytic and avoid
    /// dense pairwise assembly at large n.
    Gaussian,
    ScalarGlm {
        c_array: Array1<f64>,
        d_array: Option<Array1<f64>>,
        x: DesignMatrix,
    },
    Callback {
        first: Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>,
        second: Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
                + Send
                + Sync,
        >,
    },
}


impl OuterHessianDerivativeKernel {
    fn from_scalar_glm(ingredients: ScalarGlmIngredients<'_>) -> Self {
        Self::ScalarGlm {
            c_array: ingredients.c_array.clone(),
            d_array: ingredients.d_array.cloned(),
            x: ingredients.x.clone(),
        }
    }
}


/// Null implementation for Gaussian families (c=d=0).
pub struct GaussianDerivatives;


impl HessianDerivativeProvider for GaussianDerivatives {
    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        Some(OuterHessianDerivativeKernel::Gaussian)
    }

    fn hessian_derivative_correction(
        &self,
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }
    fn has_corrections(&self) -> bool {
        false
    }
}


/// Single-predictor GLM derivative provider.
///
/// For non-Gaussian single-predictor models, the third-derivative correction is:
///   Cₖ = Xᵀ diag(c ⊙ X vₖ) X
/// where c is the first eta-derivative of the working curvature W(η),
/// and vₖ = H⁻¹(Aₖβ̂) is the mode response.
///
/// When the link is not canonical — probit, cloglog, SAS, mixture, or
/// beta-logistic — `c_array` and `d_array` store the **observed-information**
/// weight derivatives (c_obs, d_obs) that include residual-dependent
/// corrections:
///
///   c_obs = c_F + h'·B − (y−μ)·B_η
///   d_obs = d_F + h''·B + 2h'·B_η − (y−μ)·B_ηη
///
/// where B = (h''V − h'²V') / (φV²).  For canonical links (logit for
/// binomial, log for Poisson), B = 0 so observed = Fisher and the arrays
/// are populated with the Fisher values unchanged. These arrays are carried
/// out of PIRLS as the accepted Hessian-side curvature surface and passed
/// through `RemlState::hessian_cd_arrays` at the construction sites in
/// `runtime.rs`.
///
/// The link-parameter ext_coord path (build_sas_link_ext_coords /
/// build_mixture_link_ext_coords) independently uses observed weight
/// derivatives computed inline.
pub struct SinglePredictorGlmDerivatives {
    /// c_array: dW_obs/dη, the first eta-derivative of the observed
    /// working curvature.  For canonical links this equals c_F.
    pub c_array: Array1<f64>,
    /// d_array: d²W_obs/dη², the second eta-derivative of the observed
    /// working curvature.  For canonical links this equals d_F.
    pub d_array: Option<Array1<f64>>,
    /// Hessian-side working weights whose active rows define the curvature
    /// surface being differentiated.
    pub hessian_weights: Array1<f64>,
    /// Design matrix X in the transformed basis.
    pub x_transformed: DesignMatrix,
}


impl HessianDerivativeProvider for SinglePredictorGlmDerivatives {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The Hessian derivative is dH/dρₖ = Aₖ + D_β(X'W_HX)[−vₖ].
        // Since vₖ = H⁻¹(Aₖβ̂) = −dβ̂/dρₖ, the β-direction is −vₖ, giving:
        //   D_β(X'W_HX)[−vₖ] = X' diag(c · X(−vₖ)) X
        //                     = −X' diag(c ⊙ Xvₖ) X
        // where c = dW_H/dη (the Hessian-side third-derivative weight array).
        //
        // This method returns the correction (dH/dρₖ − Aₖ), which is NEGATIVE.
        // Stays matrix-free: `matrixvectormultiply` and `xt_diag_x_signed_op`
        // route through the operator-backed design's chunked kernels at large-scale
        // scale, so we never materialize the full (n×p) dense block.
        let x_v = self.x_transformed.matrixvectormultiply(v_k); // X vₖ: n-vector

        let crate::pirls::DirectionalWorkingCurvature::Diagonal(mut neg_c_xv) =
            crate::pirls::directionalworking_curvature_from_c_array(
                &self.c_array,
                &self.hessian_weights,
                &x_v,
            );
        neg_c_xv.mapv_inplace(|value| -value);

        // −Xᵀ diag(c ⊙ Xvₖ) X via the design's matrix-free weighted gram.
        let result = self
            .x_transformed
            .xt_diag_x_signed_op(SignedWeightsView::from_array(&neg_c_xv))
            .map_err(|e| format!("hessian_derivative_correction xtwx: {e}"))?;

        Ok(Some(result))
    }

    /// #901 layer-2 fix: the first-order correction stays in OPERATOR form.
    ///
    /// `coord_corrections` (the ρ AND ψ logdet-gradient drifts) are built
    /// through this method; returning `DriftDerivResult::Operator` routes
    /// every downstream spectral-kernel trace through
    /// `reduce_operator`/`trace_operator`, whose `C·u_a` probes evaluate the
    /// near-null quadratic forms stably (see
    /// [`GlmCurvatureCorrectionOperator`]). The dense
    /// `hessian_derivative_correction` above remains for consumers that
    /// genuinely need the materialized block (outer-Hessian pair assembly).
    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let x_v = self.x_transformed.matrixvectormultiply(v_k);
        let crate::pirls::DirectionalWorkingCurvature::Diagonal(mut neg_c_xv) =
            crate::pirls::directionalworking_curvature_from_c_array(
                &self.c_array,
                &self.hessian_weights,
                &x_v,
            );
        neg_c_xv.mapv_inplace(|value| -value);
        Ok(Some(DriftDerivResult::Operator(Arc::new(
            GlmCurvatureCorrectionOperator {
                x_design: self.x_transformed.clone(),
                neg_c_xv,
                p: self.x_transformed.ncols(),
            },
        ))))
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Second-order correction for the outer Hessian.
        // H_{kl} includes contributions from both c (third) and d (fourth) derivatives:
        //   Xᵀ diag(c ⊙ X u_{kl} + d ⊙ (X vₖ) ⊙ (X vₗ)) X
        // Stays matrix-free via the design's `matrixvectormultiply` and
        // `xt_diag_x_signed_op` so large-scale designs never densify the (n×p)
        // block.
        let x_vk = self.x_transformed.matrixvectormultiply(v_k);
        let x_vl = self.x_transformed.matrixvectormultiply(v_l);
        let x_ukl = self.x_transformed.matrixvectormultiply(u_kl);

        let n = self.x_transformed.nrows();
        let mut weights = Array1::zeros(n);

        // c ⊙ X u_{kl}, masked the same way as the Hessian curvature surface.
        let crate::pirls::DirectionalWorkingCurvature::Diagonal(first_weights) =
            crate::pirls::directionalworking_curvature_from_c_array(
                &self.c_array,
                &self.hessian_weights,
                &x_ukl,
            );
        weights.assign(&first_weights);

        // + d ⊙ (X vₖ) ⊙ (X vₗ)
        if let Some(ref d_array) = self.d_array {
            Zip::from(&mut weights)
                .and(d_array)
                .and(&x_vk)
                .and(&x_vl)
                .and(&self.hessian_weights)
                .par_for_each(|w, &d, &xvk, &xvl, &h| {
                    if h > 0.0 {
                        let delta = d * xvk * xvl;
                        if delta.is_finite() {
                            *w += delta;
                        }
                    }
                });
        }

        // Xᵀ diag(weights) X via the design's matrix-free weighted gram.
        let result = self
            .x_transformed
            .xt_diag_x_signed_op(SignedWeightsView::from_array(&weights))
            .map_err(|e| format!("hessian_second_derivative_correction xtwx: {e}"))?;

        Ok(Some(result))
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        Some(ScalarGlmIngredients {
            c_array: &self.c_array,
            d_array: self.d_array.as_ref(),
            x: &self.x_transformed,
        })
    }
}


/// Firth-aware GLM derivative provider.
///
/// Wraps the base GLM corrections with Firth/Jeffreys Hφ corrections:
///   H_k = A_k + base_correction(v_k) − D(Hφ)[B_k]
///   H_{kl} = base_second(v_k, v_l, u_kl) − D(Hφ)[B_{kl}] − D²(Hφ)[B_k, B_l]
///
/// where B_k = −v_k (mode response) and the Firth operators use δη = X·B_k.
pub struct FirthAwareGlmDerivatives {
    pub(super) base: SinglePredictorGlmDerivatives,
    pub(super) firth_op: std::sync::Arc<super::FirthDenseOperator>,
}


impl HessianDerivativeProvider for FirthAwareGlmDerivatives {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Base GLM correction: −Xᵀ diag(c ⊙ X vₖ) X
        let base_corr = self.base.hessian_derivative_correction(v_k)?;

        // Firth correction: −D(Hφ)[B_k] where B_k = −v_k, δη_k = X·(−v_k).
        let deta_k: Array1<f64> =
            crate::faer_ndarray::fast_av(&self.firth_op.x_dense, v_k).mapv(|v| -v);
        let dir_k = self.firth_op.direction_from_deta(deta_k);
        let firth_corr = self.firth_op.hphi_direction(&dir_k);

        match base_corr {
            Some(mut bc) => {
                bc -= &firth_corr;
                Ok(Some(bc))
            }
            None => Ok(Some(-firth_corr)),
        }
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Base GLM second correction: Xᵀ diag(c ⊙ X u_{kl} + d ⊙ (X vₖ)(X vₗ)) X
        let base_corr = self
            .base
            .hessian_second_derivative_correction(v_k, v_l, u_kl)?;

        // Firth D(Hφ)[B_{kl}]: B_{kl} direction is u_kl in β-space.
        let deta_kl: Array1<f64> = crate::faer_ndarray::fast_av(&self.firth_op.x_dense, u_kl);
        let dir_kl = self.firth_op.direction_from_deta(deta_kl);
        let firth_first = self.firth_op.hphi_direction(&dir_kl);

        // Firth D²(Hφ)[B_k, B_l]: second directional derivative.
        let deta_k: Array1<f64> =
            crate::faer_ndarray::fast_av(&self.firth_op.x_dense, v_k).mapv(|v| -v);
        let dir_k = self.firth_op.direction_from_deta(deta_k);
        let deta_l: Array1<f64> =
            crate::faer_ndarray::fast_av(&self.firth_op.x_dense, v_l).mapv(|v| -v);
        let dir_l = self.firth_op.direction_from_deta(deta_l);
        let p = v_k.len();
        let eye = Array2::<f64>::eye(p);
        let firth_second = self
            .firth_op
            .hphisecond_direction_apply(&dir_k, &dir_l, &eye);

        let mut result = match base_corr {
            Some(bc) => bc,
            None => Array2::zeros((p, p)),
        };
        result -= &firth_first;
        result -= &firth_second;
        Ok(Some(result))
    }

    /// #901 layer-2: keep the base GLM cubic correction in operator form and
    /// graft the (dense, well-conditioned) Firth part on through
    /// [`CompositeHyperOperator`], mirroring `BarrierDerivativeProvider`.
    /// The roundoff-critical near-null quadratic forms live entirely in the
    /// base `Xᵀ diag(c⊙Xv) X` sandwich; the Firth `−D(Hφ)[B_k]` block stays
    /// dense as before.
    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let base = self.base.hessian_derivative_correction_result(v_k)?;

        let deta_k: Array1<f64> =
            crate::faer_ndarray::fast_av(&self.firth_op.x_dense, v_k).mapv(|v| -v);
        let dir_k = self.firth_op.direction_from_deta(deta_k);
        let neg_firth_corr = -self.firth_op.hphi_direction(&dir_k);

        match base {
            Some(DriftDerivResult::Operator(operator)) => Ok(Some(DriftDerivResult::Operator(
                Arc::new(CompositeHyperOperator {
                    dense: Some(neg_firth_corr),
                    operators: vec![operator],
                    dim_hint: self.base.x_transformed.ncols(),
                }),
            ))),
            Some(DriftDerivResult::Dense(mut dense)) => {
                dense += &neg_firth_corr;
                Ok(Some(DriftDerivResult::Dense(dense)))
            }
            None => Ok(Some(DriftDerivResult::Dense(neg_firth_corr))),
        }
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        None
    }
}


/// Exact Jeffreys/Firth term used by the unified outer evaluator.
///
/// The scalar contribution and all outer derivatives must be sourced from the
/// same operator in the same coefficient basis.
#[derive(Clone)]
pub struct ExactJeffreysTerm {
    /// Tier-A GLM dense operator carrying the value and all β-gradient
    /// machinery. `None` for the Tier-B value-only carrier (see
    /// [`ExactJeffreysTerm::value_only`]), where the coupled joint path
    /// supplies the curvature/drift terms through its own
    /// `H_Φ`-aware derivative provider and only the scalar `Φ(β̂)` needs to
    /// reach the LAML cost.
    operator: Option<std::sync::Arc<super::FirthDenseOperator>>,
    /// Tangent-projected value override. When `Some`, `value()` returns
    /// this scalar instead of the operator's full-space `½ log|J|`. This
    /// is used by `try_tangent_projected_evaluate` to substitute
    /// `½ log|ZᵀJZ|` while reusing the rest of the evaluator pipeline.
    /// The same `Arc<FirthDenseOperator>` is retained so any downstream
    /// consumer that accesses the operator (e.g. for β-gradient terms)
    /// sees the unmodified operator; only the scalar contribution to the
    /// outer LAML cost changes. For the Tier-B value-only carrier this is
    /// always `Some` (it IS the value).
    value_override: Option<f64>,
}


impl ExactJeffreysTerm {
    pub(crate) fn new(operator: std::sync::Arc<super::FirthDenseOperator>) -> Self {
        Self {
            operator: Some(operator),
            value_override: None,
        }
    }

    /// Tier-B value-only carrier: the coupled joint custom-family path folds
    /// the gated Jeffreys value `Φ(β̂) = ½ log|H_id|` into the LAML cost
    /// (`cost −= Φ`) so the outer criterion is the Laplace approximation of
    /// the SAME Firth-augmented objective `−ℓ + ½βᵀSβ − Φ` the inner Newton
    /// converged on. Without this fold the envelope identity breaks at every
    /// Firth-active mode: `∇_β(−ℓ + ½βᵀSβ)(β̂) = +∇Φ ≠ 0`, so the analytic
    /// outer gradient (which differentiates the Φ-folded criterion via the
    /// envelope) disagrees with the finite difference of the Φ-less value
    /// by `(∇Φ)ᵀ ∂β̂/∂ρ` (gam#979). The β-gradient/curvature machinery for
    /// Tier-B lives in the `H_Φ`-aware joint derivative provider, not here.
    pub(crate) fn value_only(phi: f64) -> Self {
        Self {
            operator: None,
            value_override: Some(phi),
        }
    }

    /// Construct a tangent-projected variant: wraps the same operator but
    /// returns `½ log|ZᵀJZ|` from `value()`.
    pub(crate) fn with_projected_value(
        operator: std::sync::Arc<super::FirthDenseOperator>,
        projected_value: f64,
    ) -> Self {
        Self {
            operator: Some(operator),
            value_override: Some(projected_value),
        }
    }

    #[inline]
    pub(crate) fn value(&self) -> f64 {
        self.value_override.unwrap_or_else(|| {
            self.operator
                .as_ref()
                .map_or(0.0, |operator| operator.jeffreys_logdet())
        })
    }

    #[inline]
    pub(crate) fn operator_arc(&self) -> Option<std::sync::Arc<super::FirthDenseOperator>> {
        self.operator.as_ref().map(std::sync::Arc::clone)
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Guarded scalar correction (value + ρ-gradient under ONE include flag)
// ═══════════════════════════════════════════════════════════════════════════

/// A scalar objective correction whose VALUE and analytic ρ-GRADIENT are
/// carried together and applied through a SINGLE site under a SINGLE guard.
///
/// This is the structural cure for the recurring objective↔gradient desync
/// bug class (issues #752/#748/#808 and the latent Tierney–Kadane desync):
/// when a correction's value and its derivative are added to the cost and the
/// ρ-gradient in physically separate statements — each with its own
/// hand-written `if include_logdet_h { … }` guard — the two drift apart. Here
/// the `include` flag is read ONCE and gates BOTH contributions in
/// [`GuardedCorrection::apply`], so a future edit cannot re-introduce the
/// half-applied/half-omitted state by construction.
///
/// Mirrors the already-paired `PenaltyLogdetDerivs` / `joint_jeffreys_term`
/// objects, which return value+derivative together for exactly this reason.
pub(crate) struct GuardedCorrection {
    /// Scalar contribution to the outer REML/LAML cost.
    value: f64,
    /// Contribution to the ρ-gradient (one entry per active ρ coordinate),
    /// `None` when the correction is value-only (derivative-free regime).
    gradient: Option<Array1<f64>>,
    /// The SINGLE guard. When `false`, NEITHER the value nor the gradient is
    /// applied; when `true`, BOTH are.
    include: bool,
}


impl GuardedCorrection {
    /// Construct a guarded correction from a loose `(value, gradient)` pair and
    /// the include flag that must gate both.
    pub(crate) fn new(value: f64, gradient: Option<Array1<f64>>, include: bool) -> Self {
        Self {
            value,
            gradient,
            include,
        }
    }

    /// Apply the VALUE contribution to `cost` under the single `include` guard.
    pub(crate) fn apply_value(&self, cost: &mut f64) {
        if self.include {
            *cost += self.value;
        }
    }

    /// Apply the ρ-GRADIENT contribution to the leading entries of `rho_grad`
    /// under the SAME single `include` guard read from `self`.
    pub(crate) fn apply_gradient(&self, rho_grad: &mut Array1<f64>) {
        if !self.include {
            return;
        }
        if let Some(grad) = self.gradient.as_ref() {
            let k = grad.len();
            let mut sl = rho_grad.slice_mut(ndarray::s![..k]);
            sl += grad;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Log-barrier support for constrained coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for a log-barrier penalty on constrained coefficients.
///
/// The barrier-augmented objective adds `-τ Σ_{j ∈ C} log(s_j β_j − b_j)`,
/// where `s_j = 1` for lower bounds and `s_j = -1` for upper bounds.
/// τ is an algorithmic continuation parameter — NOT a hyperparameter.
#[derive(Clone, Debug)]
pub struct BarrierConfig {
    /// Barrier strength parameter (continuation schedule drives this → 0).
    pub tau: f64,
    /// Indices of constrained coefficients in the β vector.
    pub constrained_indices: Vec<usize>,
    /// Right-hand-side `b_j` for each directional coordinate constraint.
    pub lower_bounds: Vec<f64>,
    /// Direction `s_j` for each coordinate constraint `s_j β_j >= b_j`.
    pub bound_signs: Vec<f64>,
}


impl BarrierConfig {
    /// Construct a `BarrierConfig` from linear inequality constraints `A β ≥ b`
    /// by extracting rows that represent simple coordinate bounds
    /// (`β_j ≥ b_i` or `β_j ≤ -b_i`).
    ///
    /// A row is a simple bound iff it has exactly one nonzero entry equal to ±1.0.
    /// Returns `None` if the constraints are `None` or no simple-bound rows are found.
    pub fn from_constraints(
        constraints: Option<&crate::pirls::LinearInequalityConstraints>,
    ) -> Option<Self> {
        // Tolerance for recognizing a constraint-matrix entry as exactly 0 or
        // exactly ±1, so a row qualifies as a simple coordinate bound. The
        // constraint rows are assembled exactly, so any nonzero deviation this
        // large is a genuine multi-coefficient constraint, not round-off.
        const SIMPLE_BOUND_ENTRY_TOL: f64 = 1e-14;
        // Default log-barrier strength τ used when a simple-bound BarrierConfig
        // is synthesized from constraints (a weak barrier that keeps β strictly
        // feasible without materially perturbing an interior optimum).
        const DEFAULT_BARRIER_TAU: f64 = 1e-6;
        let constraints = constraints?;
        let mut indices = Vec::new();
        let mut lower_bounds = Vec::new();
        let mut bound_signs = Vec::new();
        for i in 0..constraints.a.nrows() {
            let row = constraints.a.row(i);
            let mut single_col = None;
            let mut single_sign = 0.0_f64;
            let mut is_simple = true;
            for (j, &val) in row.iter().enumerate() {
                if val.abs() < SIMPLE_BOUND_ENTRY_TOL {
                    continue;
                }
                if ((val - 1.0).abs() < SIMPLE_BOUND_ENTRY_TOL
                    || (val + 1.0).abs() < SIMPLE_BOUND_ENTRY_TOL)
                    && single_col.is_none()
                {
                    single_col = Some(j);
                    single_sign = if val > 0.0 { 1.0 } else { -1.0 };
                } else {
                    is_simple = false;
                    break;
                }
            }
            if is_simple && let Some(col) = single_col {
                indices.push(col);
                lower_bounds.push(constraints.b[i]);
                bound_signs.push(single_sign);
            }
        }
        if indices.is_empty() {
            return None;
        }
        Some(BarrierConfig {
            tau: DEFAULT_BARRIER_TAU,
            constrained_indices: indices,
            lower_bounds,
            bound_signs,
        })
    }

    /// Compute slack values Δ_j = s_j β_j − b_j. Returns `None` if infeasible.
    pub fn slacks(&self, beta: &Array1<f64>) -> Option<Vec<f64>> {
        let mut slacks = Vec::with_capacity(self.constrained_indices.len());
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let sign = self.bound_signs[ci];
            let delta = sign * beta[idx] - self.lower_bounds[ci];
            if delta <= 0.0 {
                return None;
            }
            slacks.push(delta);
        }
        Some(slacks)
    }

    /// Add the barrier Hessian diagonal τ·D^(2) to H in-place.
    pub fn add_barrier_hessian_diagonal(
        &self,
        h: &mut Array2<f64>,
        beta: &Array1<f64>,
    ) -> Result<(), String> {
        let slacks = self
            .slacks(beta)
            .ok_or_else(|| "Barrier: infeasible point (slack ≤ 0)".to_string())?;
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            h[[idx, idx]] += self.tau / (slacks[ci] * slacks[ci]);
        }
        Ok(())
    }

    /// Compute the barrier cost `−τ Σ log(Δ_j)`.
    ///
    /// **Contract.** The log-barrier objective is, by construction, a
    /// real-valued function of β on the feasible interior that diverges to
    /// `+∞` as any slack `Δ_j = s_j β_j − b_j` approaches `0⁺`. We extend it
    /// continuously to the closed exterior `Δ_j ≤ 0` by the same limit:
    /// `barrier_cost(β) = +∞` whenever any constrained coordinate has reached
    /// or crossed its bound. This makes the barrier objective composable with
    /// generic line-search / trust-region code that compares scalar
    /// objectives — an infeasible trial step is automatically rejected by
    /// monotonicity, with no special-cased `Err` branch in every call site.
    ///
    /// We never return NaN: at `Δ_j = 0` exactly we shortcut to `+∞` rather
    /// than evaluating `ln(0) = −∞` (which would multiply with `−τ` to give
    /// `+∞` but only after a non-finite intermediate); at `Δ_j < 0` we
    /// shortcut to `+∞` rather than computing `ln(negative) = NaN`.
    pub fn barrier_cost(&self, beta: &Array1<f64>) -> f64 {
        let mut total = 0.0_f64;
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let sign = self.bound_signs[ci];
            let delta = sign * beta[idx] - self.lower_bounds[ci];
            if delta <= 0.0 {
                return f64::INFINITY;
            }
            // Δ > 0 here, so ln(Δ) is finite; contribution is finite real.
            total += delta.ln();
        }
        -self.tau * total
    }

    /// Detection of barrier-dominated geometry, where EFS — which assumes
    /// inner Hessian ≈ X'WX + S and ignores the log-barrier drift
    /// `τ / (β_j − l_j)²` on its diagonal — becomes unreliable. Returns
    /// `true` whenever at least one of the following holds (each captures a
    /// distinct failure mode of the EFS precondition):
    ///
    /// (a) **Asymmetric concentration.** With slacks Δ_j = β_j − l_j,
    /// `min_j Δ_j < ratio · median_j Δ_j`. This is a *scale-free* check
    /// using only slack ratios, so it is independent of the absolute scale
    /// of β. It catches the common pathology where one constrained
    /// coefficient runs to its bound while the rest stay healthy — that
    /// one coord's `τ/Δ²` then dominates the inner Hessian diagonal at
    /// that coord, and EFS's multiplicative update is no longer
    /// guaranteed-ascent there.
    ///
    /// (b) **Absolute saturation.** `τ / min_j Δ_j² ≥ saturation_threshold`.
    /// This is a *dimensional* check that catches the case (a) misses:
    /// when ALL slacks shrink together near the optimum, slack ratios stay
    /// near 1 but the per-coord barrier curvature still saturates. With
    /// the default `τ = 1e-6` and a `saturation_threshold` of 1.0 (the
    /// natural unit penalty scale), this fires at `Δ_min ≲ 1e-3`.
    ///
    /// Returns `true` on infeasible β (Δ_j ≤ 0).
    ///
    /// Replaces the older `barrier_curvature_is_significant(_, ref_diag, _)`,
    /// whose `ref_diag` was a representative diagonal of `X'W_HX + S` that
    /// no call site could compute correctly without surfacing the inner
    /// Hessian out to the EFS bridge.
    pub fn barrier_curvature_locally_concentrated(
        &self,
        beta: &Array1<f64>,
        ratio: f64,
        saturation_threshold: f64,
    ) -> bool {
        let Some(mut slacks) = self.slacks(beta) else {
            return true; // infeasible → conservatively unreliable
        };
        if slacks.is_empty() {
            return false;
        }
        let min_slack = slacks.iter().copied().fold(f64::INFINITY, f64::min);

        // (b) Absolute saturation: τ / Δ_min² ≥ threshold. Catches the
        // symmetric near-boundary regime that ratio-only checks miss.
        if min_slack > 0.0 && min_slack.is_finite() && saturation_threshold.is_finite() {
            let max_barrier_curv = self.tau / (min_slack * min_slack);
            if max_barrier_curv >= saturation_threshold {
                return true;
            }
        }

        // (a) Asymmetric concentration: min Δ ≪ median Δ.
        slacks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if slacks.len() % 2 == 1 {
            slacks[slacks.len() / 2]
        } else {
            let mid = slacks.len() / 2;
            0.5 * (slacks[mid - 1] + slacks[mid])
        };
        if !median.is_finite() || median <= 0.0 {
            return true;
        }
        min_slack < ratio * median
    }

    /// Check whether the barrier curvature is non-negligible relative to a
    /// reference Hessian diagonal scale.
    ///
    /// Returns `true` when `max_j τ / (β_j − l_j)² > threshold * ref_diag`,
    /// indicating that EFS (which ignores the barrier Hessian drift) would be
    /// unreliable. If β is infeasible, conservatively returns `true`.
    ///
    /// `ref_diag` should be a representative diagonal of X'W_HX + S (e.g. the
    /// median or mean). A typical `threshold` is 0.01–0.1.
    pub fn barrier_curvature_is_significant(
        &self,
        beta: &Array1<f64>,
        ref_diag: f64,
        threshold: f64,
    ) -> bool {
        let Some(slacks) = self.slacks(beta) else {
            return true; // infeasible → conservatively active
        };
        let max_barrier_curv = slacks
            .iter()
            .map(|&d| self.tau / (d * d))
            .fold(0.0_f64, f64::max);
        max_barrier_curv > threshold * ref_diag
    }
}


/// Barrier-aware Hessian derivative provider wrapping an inner provider.
///
/// Adds C_bar[u] = −2τ·diag(u ⊙ d^(3)) and Q_bar[u,v] = 6τ·diag(u ⊙ v ⊙ d^(4)).
pub struct BarrierDerivativeProvider<'a> {
    inner: &'a dyn HessianDerivativeProvider,
    tau: f64,
    constrained_indices: &'a [usize],
    bound_signs: &'a [f64],
    slacks: Vec<f64>,
    p: usize,
}


impl<'a> BarrierDerivativeProvider<'a> {
    pub fn new(
        inner: &'a dyn HessianDerivativeProvider,
        config: &'a BarrierConfig,
        beta: &Array1<f64>,
    ) -> Result<Self, String> {
        let slacks = config
            .slacks(beta)
            .ok_or_else(|| "BarrierDerivativeProvider: infeasible point".to_string())?;
        Ok(Self {
            inner,
            tau: config.tau,
            constrained_indices: &config.constrained_indices,
            bound_signs: &config.bound_signs,
            slacks,
            p: beta.len(),
        })
    }

    fn barrier_correction(&self, u: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((self.p, self.p));
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let inv_cube = 1.0 / (self.slacks[ci].powi(3));
            result[[idx, idx]] = -2.0 * self.tau * self.bound_signs[ci] * u[idx] * inv_cube;
        }
        result
    }

    fn barrier_second_correction(&self, u: &Array1<f64>, v: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((self.p, self.p));
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let inv_4 = 1.0 / (self.slacks[ci].powi(4));
            result[[idx, idx]] = 6.0 * self.tau * u[idx] * v[idx] * inv_4;
        }
        result
    }
}


impl HessianDerivativeProvider for BarrierDerivativeProvider<'_> {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The trait convention passes vₖ = H⁻¹(Aₖβ̂), but the barrier
        // third-derivative should be evaluated at the mode sensitivity
        // direction β̂_ρk = −vₖ.  barrier_correction(u) computes
        // D_β(B_ββ)[u] = −2τ u_j/gap³, so we negate vₖ to get:
        //   D_β(B_ββ)[−vₖ] = +2τ vₖ_j/gap³.
        let neg_v_k = v_k.mapv(|x| -x);
        let barrier_corr = self.barrier_correction(&neg_v_k);
        match self.inner.hessian_derivative_correction(v_k)? {
            Some(mut ic) => {
                ic += &barrier_corr;
                Ok(Some(ic))
            }
            None => Ok(Some(barrier_corr)),
        }
    }

    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let neg_v_k = v_k.mapv(|x| -x);
        let barrier_corr = self.barrier_correction(&neg_v_k);
        match self.inner.hessian_derivative_correction_result(v_k)? {
            Some(DriftDerivResult::Dense(mut dense)) => {
                dense += &barrier_corr;
                Ok(Some(DriftDerivResult::Dense(dense)))
            }
            Some(DriftDerivResult::Operator(operator)) => Ok(Some(DriftDerivResult::Operator(
                Arc::new(CompositeHyperOperator {
                    dense: Some(barrier_corr),
                    operators: vec![operator],
                    dim_hint: self.p,
                }),
            ))),
            None => Ok(Some(DriftDerivResult::Dense(barrier_corr))),
        }
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let barrier_total =
            &self.barrier_correction(u_kl) + &self.barrier_second_correction(v_k, v_l);
        match self
            .inner
            .hessian_second_derivative_correction(v_k, v_l, u_kl)?
        {
            Some(mut ic) => {
                ic += &barrier_total;
                Ok(Some(ic))
            }
            None => Ok(Some(barrier_total)),
        }
    }

    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let barrier_total =
            &self.barrier_correction(u_kl) + &self.barrier_second_correction(v_k, v_l);
        match self
            .inner
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)?
        {
            Some(DriftDerivResult::Dense(mut dense)) => {
                dense += &barrier_total;
                Ok(Some(DriftDerivResult::Dense(dense)))
            }
            Some(DriftDerivResult::Operator(operator)) => Ok(Some(DriftDerivResult::Operator(
                Arc::new(CompositeHyperOperator {
                    dense: Some(barrier_total),
                    operators: vec![operator],
                    dim_hint: self.p,
                }),
            ))),
            None => Ok(Some(DriftDerivResult::Dense(barrier_total))),
        }
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        None
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Link-wiggle derivative provider (exact second-order Hessian corrections)
// ═══════════════════════════════════════════════════════════════════════════

/// Derivative provider for link-wiggle models that restores exact second-order
/// Hessian corrections for the outer REML/LAML evaluator.
///
/// # Background
///
/// In link-wiggle models, the Gauss-Newton Hessian H = J'WJ has a coupled
/// Jacobian J that depends on the coefficients β through the link function.
/// Differentiating H twice with respect to the outer smoothing parameters
/// (via the implicit function theorem) produces FIVE distinct contributions.
/// Without these, the unified REML evaluator cannot compute the exact outer
/// Hessian, so the outer planner must downgrade to a non-analytic-Hessian
/// strategy (BFGS, or EFS / hybrid EFS when that fixed-point structure is
/// available).
///
/// This provider stores pre-computed ingredients from the converged P-IRLS
/// inner loop and implements both first-order (∂H/∂ρ_k) and second-order
/// (∂²H/∂ρ_k∂ρ_l) Hessian corrections analytically, enabling the exact
/// analytic-Hessian outer plan instead of those downgraded strategies.
///
/// # Mathematical framework (response.md Sections 3 and 6)
///
/// The link-wiggle predictor is q = g(η; θ_link) where g is a flexible
/// link function parameterized by θ_link. The joint Jacobian J maps the
/// combined parameter vector (β_base, β_link) to the predictor derivatives:
///
///   J[:,0..p_base] = diag(g'(η)) · X_base        (base block)
///   J[:,p_base..]  = B(z) · Z                      (link block)
///
/// where z = (η - min)/(max - min) is the normalized base predictor, B(z)
/// is the B-spline basis evaluated at z, and Z is the geometric constraint
/// transform ensuring monotonicity.
///
/// The Gauss-Newton Hessian is H = J'WJ where W = diag(w_i) are the
/// working weights from the negative log-likelihood second derivative.
///
/// Differentiating H with respect to ρ_k (via the chain rule through
/// the implicit function theorem β̂(ρ)) requires:
///
///   ∂H/∂ρ_k = D_β H[-v_k]  where v_k = H⁻¹(A_k β̂)
///
/// and for the second derivative:
///
///   ∂²H/∂ρ_k∂ρ_l = D_β H[u_kl] + D²_β H[-v_k, -v_l]
///
/// where u_kl = H⁻¹(−g_kl + Ḣ_l v_k + Ḣ_k v_l) is the second-order
/// IFT mode response.
///
/// # Relationship to Arbogast
///
/// The five-term decomposition arises from the Arbogast formula for the
/// second derivative of the composed map ρ → β̂(ρ) → J(β̂) → J'WJ. Each
/// differentiation of J'WJ produces terms from:
/// - Differentiating J (Jacobian drift, terms 2-4)
/// - Differentiating W (weight drift, terms 3-5)
/// - Cross terms between the two differentiations (terms 2, 3, 4)
/// - The curvature of W itself through w'' (term 5)
#[derive(Clone)]
pub struct HyperCoord {
    /// ∂_i F|_β — fixed-β cost derivative (scalar).
    pub a: f64,
    /// ∂_i (∇_β F)|_β — fixed-β score (p-vector).
    pub g: Array1<f64>,
    /// ∂_i H|_β — fixed-β Hessian drift.
    ///
    /// The drift may have a materialized dense contribution, an operator
    /// contribution, or both. This replaces the old `b_mat + optional
    /// b_operator + zero-sized placeholder` convention.
    pub drift: HyperCoordDrift,
    /// ∂_i L_δ(S) — smooth penalty pseudo-logdet first derivative.
    /// Uses (S + δI)⁻¹ instead of the hard-truncated pseudoinverse S₊⁻¹.
    pub ld_s: f64,
    /// Whether B_i depends on β (true for ψ with non-Gaussian likelihood).
    /// When true, M_i[u] = D_β B_i[u] contributes to the exact outer Hessian.
    pub b_depends_on_beta: bool,
    /// Whether this coordinate is "penalty-like" (τ) vs "design-moving" (ψ).
    ///
    /// Penalty-like coordinates (τ) have Hessian drifts derived from penalty
    /// matrix derivatives (similar to ρ coordinates), so they are PSD.
    /// Design-moving coordinates (ψ) have Hessian drifts that contain
    /// design-motion and likelihood-curvature terms and need not be PSD or even
    /// sign-definite.
    ///
    /// This flag controls eligibility for EFS (Fellner-Schall) updates.
    /// See [`compute_efs_update`] for details.
    pub is_penalty_like: bool,
    /// Fixed-β Jeffreys/Firth gradient partial `(g_Φ)_i`, when the inner
    /// objective includes the exact bias-reduction term.
    pub firth_g: Option<Array1<f64>>,
    /// Fixed-β linear predictor derivative used by the Tierney-Kadane
    /// correction's direct c/d derivative terms.
    pub tk_eta_fixed: Option<Array1<f64>>,
    /// Fixed-β design derivative used by the Tierney-Kadane correction's
    /// direct design-row derivative terms.
    pub tk_x_fixed: Option<Array2<f64>>,
}


/// Second-order fixed-β objects for a pair of outer coordinates.
///
/// Used by the outer Hessian computation. For ρ-ρ diagonal pairs, these
/// equal the first-order objects (a_kk = a_k, g_kk = g_k, B_kk = B_k).
/// For ρ-ρ off-diagonal pairs with k≠l, these are all zero.
pub struct HyperCoordPair {
    /// ∂²_ij F|_β — fixed-β cost second derivative (scalar).
    pub a: f64,
    /// ∂²_ij (∇_β F)|_β — fixed-β score second derivative (p-vector).
    pub g: Array1<f64>,
    /// ∂²_ij H|_β — fixed-β Hessian second drift (p×p matrix).
    pub b_mat: Array2<f64>,
    /// ∂²_ij H|_β — operator-valued Hessian second drift (implicit, avoids p×p).
    pub b_operator: Option<Box<dyn HyperOperator>>,
    /// ∂²_ij L_δ(S) — smooth penalty pseudo-logdet second derivative.
    /// Uses (S + δI)⁻¹ instead of the hard-truncated pseudoinverse S₊⁻¹.
    pub ld_s: f64,
}


impl HyperCoordPair {
    /// Return a zero-valued pair (used as a no-op fallback when hyper-coordinate
    /// construction is skipped for large models).
    pub fn zero() -> Self {
        Self {
            a: 0.0,
            g: Array1::zeros(0),
            b_mat: Array2::zeros((0, 0)),
            b_operator: None,
            ld_s: 0.0,
        }
    }
}


/// Callback for computing M_i[u] = D_β B_i[u], the directional derivative
/// of the fixed-β Hessian drift along direction u.
///
/// This is needed for the exact outer Hessian when B_i depends on β
/// (i.e., for ψ coordinates with non-Gaussian likelihoods).
/// For ρ coordinates, B_i = A_i is β-independent, so M_i ≡ 0.
///
/// When unavailable, the outer Hessian is approximate (fine for BFGS/ARC,
/// insufficient for exact Newton quadratic convergence).
/// Result of a fixed-drift derivative evaluation: can be dense or operator-backed.
#[derive(Clone)]
pub enum DriftDerivResult {
    Dense(Array2<f64>),
    Operator(Arc<dyn HyperOperator>),
}


impl std::fmt::Debug for DriftDerivResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(matrix) => f
                .debug_tuple("Dense")
                .field(&format_args!("{}x{}", matrix.nrows(), matrix.ncols()))
                .finish(),
            Self::Operator(_) => f
                .debug_tuple("Operator")
                .field(&"<hyper-operator>")
                .finish(),
        }
    }
}


impl DriftDerivResult {
    pub fn into_operator(self) -> Arc<dyn HyperOperator> {
        match self {
            Self::Dense(matrix) => Arc::new(DenseMatrixHyperOperator { matrix }),
            Self::Operator(operator) => operator,
        }
    }

    pub fn trace_logdet(&self, hop: &dyn HessianOperator) -> f64 {
        match self {
            Self::Dense(matrix) => hop.trace_logdet_gradient(matrix),
            Self::Operator(operator) => hop.trace_logdet_operator(operator.as_ref()),
        }
    }

    pub fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(matrix) => matrix.dot(v),
            Self::Operator(operator) => operator.mul_vec(v),
        }
    }

    pub fn trace_logdet_hessian_cross(&self, rhs: &Self, hop: &dyn HessianOperator) -> f64 {
        match (self, rhs) {
            (Self::Dense(left), Self::Dense(right)) => hop.trace_logdet_hessian_cross(left, right),
            (Self::Dense(left), Self::Operator(right)) => {
                hop.trace_logdet_hessian_cross_matrix_operator(left, right.as_ref())
            }
            (Self::Operator(left), Self::Dense(right)) => {
                hop.trace_logdet_hessian_cross_matrix_operator(right, left.as_ref())
            }
            (Self::Operator(left), Self::Operator(right)) => {
                hop.trace_logdet_hessian_cross_operator(left.as_ref(), right.as_ref())
            }
        }
    }
}


pub type FixedDriftDerivFn =
    Box<dyn Fn(usize, &Array1<f64>) -> Option<DriftDerivResult> + Send + Sync>;


/// Direction-contracted ψψ-block second-order terms for the profiled θ-HVP
/// (#740).
///
/// The argument `alpha_psi` is the ψ slice (length `ext_dim`) of one applied
/// outer direction. The result is the `α`-contraction over the ψ COLUMNS of
/// every `(ψ_i, ψ_j)` second-order term against the combined ψ-direction
/// `ψ(α) = Σ_j alpha_psi[j] ψ_j`, returned per ψ output row `i`. This covers
/// the ψψ block ONLY — the ρρ and ρψ blocks stay in the operator's precomputed
/// tables (they are cheap, `O(K·p²)`, and carry no family row pass), so each
/// block is assembled in exactly one place with no overlap.
///
/// Indexing of every field is the ψ output row (`ext_dim` of them, in the order
/// of `solution.ext_coords`):
/// - `objective[i] = Σ_j α_ψ[j] V_{ψ_i ψ_j}` (likelihood + penalty
///   `½βᵀS_{ψ_iψ_j}β`),
/// - `score.row(i) = Σ_j α_ψ[j] g_{ψ_i ψ_j}` (likelihood + penalty
///   `S_{ψ_iψ_j}β`), an `ext_dim × p` matrix,
/// - `hessian[i] = Σ_j α_ψ[j] D²_ψ H_L[ψ_i, ψ_j]` (+ penalty `S_{ψ_iψ_j}`), the
///   `base_h2` ψψ contribution as a `tr`-able drift,
/// - `ld_s[i] = Σ_j α_ψ[j] ∂²log|S|/∂ψ_i∂ψ_j`, the `pair_ld_s` ψ-row
///   contribution.
///
/// One call produces every output row in a single family row pass (the family
/// likelihood part) plus cheap block-local penalty assembly, so densifying the
/// operator costs `K` such passes instead of the dense path's `K²`. `None`
/// declines the fast path (the builder keeps the exact per-pair assembly).
pub struct ContractedPsiSecondOrder {
    pub objective: Array1<f64>,
    pub score: Array2<f64>,
    pub hessian: Vec<DriftDerivResult>,
    pub ld_s: Array1<f64>,
}


pub type ContractedPsiSecondOrderFn =
    Arc<dyn Fn(&[f64]) -> Result<Option<ContractedPsiSecondOrder>, String> + Send + Sync>;


// ═══════════════════════════════════════════════════════════════════════════
//  Implicit Hessian-drift operators for scalable anisotropic REML
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for operators that can compute B_i · v (matrix-vector product)
/// without materializing the full (p × p) B_i matrix.
///
/// This is used for anisotropic ψ coordinates where the Hessian drift
/// B_i = (∂X/∂ψ_d)^T W X + X^T W (∂X/∂ψ_d) + S_{ψ_d} involves the
/// implicit design-derivative operator. For small problems, a dense
/// fallback wraps an `Array2<f64>`.
///
/// The key integration point is the stochastic trace estimator: instead of
/// materializing B_i as a (p × p) matrix and calling `A_k · w`, we compute
/// `B_i · w` on the fly using implicit design-derivative matvecs.
pub trait HyperOperator: Send + Sync {
    /// Operator dimension `p` such that `B · v` consumes a `p`-vector and
    /// produces a `p`-vector.  No default — every impl must answer cheaply
    /// from a stored field or constructor argument.  Implementations must
    /// not materialize the operator to read a shape.
    fn dim(&self) -> usize;

    /// Compute B · v (matrix-vector product). v and result are p-vectors.
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64>;

    /// Compute B · v from a vector view.
    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        self.mul_vec(&v.to_owned())
    }

    /// Compute B · v into caller-owned storage.
    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        out.assign(&self.mul_vec_view(v));
    }

    /// Compute B · F where F is (p × k). Default dispatches per-column in
    /// parallel; matrix-free Khatri–Rao operators override this to fuse
    /// the K applies into two BLAS3 matmuls (`projected_operator` hot path).
    ///
    /// When invoked from inside an existing rayon worker (e.g. the parallel
    /// cross-trace assembly in `compute_outer_hessian`), dispatch sequentially
    /// to avoid pool oversubscription that manifested as
    /// `LockLatch::wait_and_reset` stalls on operator-backed corrections.
    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        if rayon::current_thread_index().is_some() {
            for col in 0..k {
                let bv = out.column_mut(col);
                self.mul_vec_into(factor.column(col), bv);
            }
            return out;
        }
        let cols: Vec<Array1<f64>> = (0..k)
            .into_par_iter()
            .map(|col| {
                let mut bv = Array1::<f64>::zeros(p);
                self.mul_vec_into(factor.column(col), bv.view_mut());
                bv
            })
            .collect();
        for (col, bv) in cols.into_iter().enumerate() {
            out.column_mut(col).assign(&bv);
        }
        out
    }

    /// Compute `trace(F^T B F)` for a `(p x k)` factor matrix `F`.
    ///
    /// The default uses the batched `B F` path, but structured row-coefficient
    /// operators can override this to avoid materialising the full product when
    /// callers only need the projected trace.
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        let op_factor = self.mul_mat(factor);
        factor
            .iter()
            .zip(op_factor.iter())
            .map(|(&f, &bf)| f * bf)
            .sum()
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        factor_cache: &ProjectedFactorCache,
    ) -> f64 {
        assert!(std::mem::size_of_val(factor_cache) > 0);
        self.trace_projected_factor(factor)
    }

    /// Compute the exact projected matrix `F^T B F`.
    ///
    /// The default uses the batched `B F` path. Structured operators can
    /// override this when the projection itself has a cheaper analytic form
    /// than materialising every column of `B F`. This is the quantity required
    /// by dense spectral logdet-Hessian contractions.
    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        let op_factor = self.mul_mat(factor);
        crate::faer_ndarray::fast_atb(factor, &op_factor)
    }

    /// Compute the exact projected matrix `F^T B F`, reusing caller-owned
    /// projection caches when the operator has a shared row/design factor.
    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        factor_cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        assert!(std::mem::size_of_val(factor_cache) > 0);
        self.projected_matrix(factor)
    }

    /// Fill columns `[start, start + out.ncols())` of `B` into `out`.
    ///
    /// Sparse exact traces build `B E` in column batches. Operators with
    /// materialized column storage can override this to copy columns directly
    /// instead of multiplying one basis vector at a time.
    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        let dim = out.nrows();
        assert!(start + cols <= dim);
        let mut basis = Array1::<f64>::zeros(dim);
        for local_col in 0..cols {
            let global_col = start + local_col;
            basis[global_col] = 1.0;
            self.mul_vec_into(basis.view(), out.column_mut(local_col));
            basis[global_col] = 0.0;
        }
    }

    /// Accumulate `scale * B · v` into caller-owned storage.
    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        let mut work = Array1::<f64>::zeros(out.len());
        self.mul_vec_into(v, work.view_mut());
        out.scaled_add(scale, &work);
    }

    /// Compute v^T · B · u (bilinear form).
    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let mut bv = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), bv.view_mut());
        u.dot(&bv)
    }

    /// Compute v^T · B · u without requiring owned vector inputs.
    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        let mut bv = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, bv.view_mut());
        u.dot(&bv)
    }

    /// Whether `bilinear_view` is implemented as a direct scalar contraction.
    ///
    /// The default `bilinear_view` materializes `Bv`; callers that already
    /// own reusable work buffers should keep using `mul_vec_into` unless an
    /// operator advertises a genuinely faster scalar contraction.
    fn has_fast_bilinear_view(&self) -> bool {
        false
    }

    /// Full dense materialization (fallback for exact trace computation).
    ///
    /// Callers should check `is_implicit()` first: the default implementation
    /// recovers the dense form by `dim()` calls to `mul_vec` against successive
    /// canonical basis vectors, which is the right shape for materialized
    /// operators but O(dim²) work and is not the right path for genuinely
    /// implicit ones. Implicit operators should either override `to_dense`
    /// with their structure-aware materialization or return `is_implicit() =
    /// true` so callers route around dense paths entirely.
    fn to_dense(&self) -> Array2<f64> {
        let p = self.dim();
        let mut out = Array2::<f64>::zeros((p, p));
        let mut basis = Array1::<f64>::zeros(p);
        for j in 0..p {
            basis[j] = 1.0;
            self.mul_vec_into(basis.view(), out.column_mut(j));
            basis[j] = 0.0;
        }
        out
    }

    /// Whether this operator uses implicit (non-materialized) storage.
    fn is_implicit(&self) -> bool;

    /// Downcast to `ImplicitHyperOperator` if this is one.
    ///
    /// Returns `Some` for implicit operators that use the weighted-Gram
    /// structure (A_d = X^T C_d X + P_d), `None` for dense wrappers.
    fn as_implicit(&self) -> Option<&ImplicitHyperOperator> {
        None
    }

    /// Downcast to `CompositeHyperOperator` when this operator is a linear
    /// bundle. Exact dense-spectral trace batching uses this to flatten
    /// coordinate drifts across coordinates, so one shared design projection
    /// can feed many implicit ψ/correction operators.
    fn as_composite(&self) -> Option<&CompositeHyperOperator> {
        None
    }

    /// Downcast to `WeightedHyperOperator` when this operator is a weighted
    /// linear bundle.
    fn as_weighted(&self) -> Option<&WeightedHyperOperator> {
        None
    }

    /// If this operator is block-local (nonzero only in [start..end, start..end]),
    /// returns the block range and local matrix. Enables O(p_block²) trace
    /// computations instead of O(p²).
    fn block_local_data(&self) -> Option<(&Array2<f64>, usize, usize)> {
        None
    }

    /// Test-only downcast to `SparseDirectionalHyperOperator`, used by the
    /// per-term operator decomposition diagnostic.
    fn as_sparse_directional(&self) -> Option<&SparseDirectionalHyperOperator> {
        None
    }
}


#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProjectedFactorKey {
    design_id: usize,
    factor_ptr: usize,
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
    value_hash: u64,
    value_hash2: u64,
}


impl ProjectedFactorKey {
    pub fn from_factor_view(design_id: usize, factor: ArrayView2<'_, f64>) -> Self {
        let strides = factor.strides();
        let (value_hash, value_hash2) = projected_factor_value_fingerprint(factor);
        Self {
            design_id,
            factor_ptr: factor.as_ptr() as usize,
            rows: factor.nrows(),
            cols: factor.ncols(),
            row_stride: strides[0],
            col_stride: strides[1],
            value_hash,
            value_hash2,
        }
    }
}


fn projected_factor_value_fingerprint(factor: ArrayView2<'_, f64>) -> (u64, u64) {
    let mut h1 = 0xcbf2_9ce4_8422_2325_u64;
    let mut h2 = 0x9e37_79b1_85eb_ca87_u64;
    for (idx, value) in factor.iter().enumerate() {
        let bits = value.to_bits();
        let mixed = bits.wrapping_add((idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
        h1 ^= mixed;
        h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
        h2 ^= bits.rotate_left((idx & 63) as u32);
        h2 = h2.wrapping_mul(0x94d0_49bb_1331_11eb).rotate_left(27);
    }
    (h1, h2)
}


/// Memoizer for `X · F` design-projection products keyed on a
/// `(design, factor)` fingerprint.
///
/// The cache trades memory for arithmetic: a 32-axis ψ-sweep that would
/// otherwise repeat the same `O(n · p · rank)` GEMM for every axis hits
/// the same cache slot 32 times. At large scale that is the
/// difference between minutes and seconds of design-GEMM work (see
/// [`ImplicitHyperOperator::trace_projected_factor_cached`] for the
/// usage rationale).
///
/// The cache is bounded by a byte budget. When inserting a new entry
/// would exceed the budget, the *least-recently-used* entries are
/// evicted until it fits. A budget of `0` (or `usize::MAX`) disables
/// eviction. The default is `Self::DEFAULT_BUDGET_BYTES` — large
/// enough to hold any realistic working set for in-memory problems
/// while still bounding worst-case peak resident memory at large-scale
/// scale, where a single `(n, rank) = (320K, 95)` projection consumes
/// ~243 MiB and a sweep over many distinct factors could otherwise
/// pin tens of GiB.
pub struct ProjectedFactorCache {
    inner: Mutex<ProjectedFactorCacheInner>,
}


struct ProjectedFactorCacheInner {
    entries: HashMap<ProjectedFactorKey, ProjectedFactorEntry>,
    in_progress: HashMap<ProjectedFactorKey, Arc<ProjectedFactorInProgress>>,
    next_seq: u64,
    total_bytes: usize,
    budget_bytes: usize,
}


struct ProjectedFactorInProgress {
    state: Mutex<Option<ProjectedFactorInProgressState>>,
    ready: Condvar,
    /// Number of threads currently parked inside the `Wait` branch for this
    /// in-progress slot. Producer panics-recovery tests use this to block
    /// (via [`subscriber_arrived`]) on subscriber arrival deterministically.
    waiter_count: std::sync::atomic::AtomicUsize,
    /// Notifies once a subscriber has incremented `waiter_count`. Producer
    /// panics-recovery tests park on this condvar so they don't have to
    /// spin or sleep waiting for the race window to close.
    subscriber_arrived: (Mutex<()>, Condvar),
}


enum ProjectedFactorInProgressState {
    Ready(Arc<Array2<f64>>),
    Failed,
}


struct ProjectedFactorEntry {
    value: Arc<Array2<f64>>,
    bytes: usize,
    last_used: u64,
}


impl Default for ProjectedFactorCache {
    fn default() -> Self {
        Self::with_budget(Self::DEFAULT_BUDGET_BYTES)
    }
}


impl ProjectedFactorCache {
    /// Default byte budget for the cache. Aligned with the large-scale
    /// `ResourcePolicy::max_single_materialization_bytes` (2 GiB) so
    /// production REML evaluations on typical hardware stay bounded
    /// without artificially throttling small problems whose entire
    /// working set fits trivially.
    pub const DEFAULT_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

    /// Construct a cache with an explicit byte budget. A budget of `0`
    /// disables eviction (legacy unbounded behavior); any non-zero
    /// budget enables LRU eviction once total cached bytes plus the
    /// next entry would exceed it.
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(ProjectedFactorCacheInner {
                entries: HashMap::new(),
                in_progress: HashMap::new(),
                next_seq: 0,
                total_bytes: 0,
                budget_bytes,
            }),
        }
    }

    pub fn get_or_insert_with(
        &self,
        key: ProjectedFactorKey,
        compute: impl FnOnce() -> Array2<f64>,
    ) -> Arc<Array2<f64>> {
        enum CacheLookup {
            Hit(Arc<Array2<f64>>),
            Wait(Arc<ProjectedFactorInProgress>),
            Compute(Arc<ProjectedFactorInProgress>),
        }

        let lookup = {
            let mut inner = self
                .inner
                .lock()
                .expect("projected factor cache lock poisoned");
            inner.next_seq += 1;
            let now = inner.next_seq;
            if let Some(entry) = inner.entries.get_mut(&key) {
                entry.last_used = now;
                CacheLookup::Hit(entry.value.clone())
            } else if let Some(waiter) = inner.in_progress.get(&key) {
                CacheLookup::Wait(waiter.clone())
            } else {
                let marker = Arc::new(ProjectedFactorInProgress {
                    state: Mutex::new(None),
                    ready: Condvar::new(),
                    waiter_count: std::sync::atomic::AtomicUsize::new(0),
                    subscriber_arrived: (Mutex::new(()), Condvar::new()),
                });
                inner.in_progress.insert(key, marker.clone());
                CacheLookup::Compute(marker)
            }
        };

        match lookup {
            CacheLookup::Hit(value) => value,
            CacheLookup::Wait(marker) => {
                marker
                    .waiter_count
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                let (lock, cv) = &marker.subscriber_arrived;
                // release-early-on-purpose: drop the arrival mutex before notifying the producer.
                drop(
                    lock.lock()
                        .expect("subscriber-arrived notification lock poisoned"),
                );
                cv.notify_all();
                let mut guard = marker
                    .state
                    .lock()
                    .expect("projected factor in-progress lock poisoned");
                let result = loop {
                    match guard.as_ref() {
                        Some(ProjectedFactorInProgressState::Ready(value)) => {
                            break value.clone();
                        }
                        Some(ProjectedFactorInProgressState::Failed) => {
                            marker
                                .waiter_count
                                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                            // SAFETY: a waiting consumer observed that the
                            // producer thread for this projected-factor cache
                            // slot panicked (state transitioned to `Failed`
                            // via the producer's drop guard). Propagating the
                            // panic to all waiters is the only correct
                            // recovery — silently returning a stale or
                            // half-initialized factor would corrupt every
                            // downstream REML/PIRLS computation that depends
                            // on it.
                            // SAFETY: producer thread panicked; propagating to waiters avoids returning corrupted factor.
                            reml_contract_panic("projected factor cache producer panicked")
                        }
                        None => {
                            guard = marker
                                .ready
                                .wait(guard)
                                .expect("projected factor in-progress wait poisoned");
                        }
                    }
                };
                marker
                    .waiter_count
                    .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                result
            }
            CacheLookup::Compute(marker) => {
                // Compute outside the cache mutex so expensive design GEMMs do
                // not serialize unrelated cache keys. Sibling callers for the
                // same key wait on `marker` instead of redundantly launching the
                // same projection, which is crucial when exact outer-gradient
                // coordinates are evaluated in parallel.
                let computed = match catch_unwind(AssertUnwindSafe(|| Arc::new(compute()))) {
                    Ok(value) => value,
                    Err(payload) => {
                        let mut inner = self
                            .inner
                            .lock()
                            .expect("projected factor cache lock poisoned");
                        inner.in_progress.remove(&key);
                        // release-early-on-purpose: avoid holding the cache mutex while publishing failure.
                        drop(inner);

                        let mut guard = marker
                            .state
                            .lock()
                            .expect("projected factor in-progress lock poisoned");
                        *guard = Some(ProjectedFactorInProgressState::Failed);
                        marker.ready.notify_all();
                        resume_unwind(payload);
                    }
                };
                let bytes = computed.len().saturating_mul(std::mem::size_of::<f64>());
                let mut inner = self
                    .inner
                    .lock()
                    .expect("projected factor cache lock poisoned");
                inner.next_seq += 1;
                let now = inner.next_seq;

                if inner.budget_bytes > 0 && bytes <= inner.budget_bytes {
                    while inner.total_bytes.saturating_add(bytes) > inner.budget_bytes
                        && !inner.entries.is_empty()
                    {
                        let Some(oldest_key) = inner
                            .entries
                            .iter()
                            .min_by_key(|(_, e)| e.last_used)
                            .map(|(k, _)| *k)
                        else {
                            break;
                        };
                        if let Some(removed) = inner.entries.remove(&oldest_key) {
                            inner.total_bytes = inner.total_bytes.saturating_sub(removed.bytes);
                        }
                    }
                }

                let value = if let Some(entry) = inner.entries.get_mut(&key) {
                    entry.last_used = now;
                    entry.value.clone()
                } else {
                    inner.entries.insert(
                        key,
                        ProjectedFactorEntry {
                            value: computed.clone(),
                            bytes,
                            last_used: now,
                        },
                    );
                    inner.total_bytes = inner.total_bytes.saturating_add(bytes);
                    computed
                };
                inner.in_progress.remove(&key);
                // release-early-on-purpose: avoid holding the cache mutex while notifying waiters.
                drop(inner);

                let mut guard = marker
                    .state
                    .lock()
                    .expect("projected factor in-progress lock poisoned");
                *guard = Some(ProjectedFactorInProgressState::Ready(value.clone()));
                marker.ready.notify_all();
                value
            }
        }
    }

    /// Number of entries currently cached. Intended for diagnostics
    /// and tests; production code should not branch on this.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.entries.len())
            .unwrap_or(0)
    }

    /// Total bytes resident in the cache. Intended for diagnostics
    /// and tests.
    pub fn total_bytes(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.total_bytes)
            .unwrap_or(0)
    }

    /// `true` when the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}


/// Dense matrix wrapper implementing `HyperOperator`.
#[derive(Clone)]
pub struct DenseMatrixHyperOperator {
    pub matrix: Array2<f64>,
}


impl HyperOperator for DenseMatrixHyperOperator {
    fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        self.matrix.dot(v)
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        self.matrix.dot(&v)
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
        dense_matvec_into(&self.matrix, v, out);
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let end = start + out.ncols();
        assert!(end <= self.matrix.ncols());
        out.assign(&self.matrix.slice(ndarray::s![.., start..end]));
    }

    fn scaled_add_mul_vec(&self, v: ArrayView1<'_, f64>, scale: f64, out: ArrayViewMut1<'_, f64>) {
        dense_matvec_scaled_add_into(&self.matrix, v, scale, out);
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        dense_bilinear(&self.matrix, v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        dense_bilinear(&self.matrix, v, u)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.matrix.clone()
    }

    fn is_implicit(&self) -> bool {
        false
    }
}


#[derive(Clone)]
pub struct CompositeHyperOperator {
    pub dense: Option<Array2<f64>>,
    pub operators: Vec<Arc<dyn HyperOperator>>,
    pub dim_hint: usize,
}


/// Group composite operators by shared `(implicit_deriv, x_design, w_diag)`
/// so every Duchon ψ-axis built atop the same implicit derivative runs
/// through a single row-kernel sweep via
/// `trace_projected_factor_all_axes_with_xf`. Per-axis `s_psi` and
/// `c_x_psi_beta` are threaded in individually so the batched path matches
/// the per-axis path exactly. Non-implicit operators and singleton groups
/// fall through to the original per-op trace path.
fn composite_trace_implicit_batched(
    operators: &[Arc<dyn HyperOperator>],
    factor: &Array2<f64>,
    cache: Option<&ProjectedFactorCache>,
) -> f64 {
    let mut trace = 0.0;
    let mut group_starts: Vec<Vec<usize>> = Vec::new();
    let mut handled = vec![false; operators.len()];

    for (i, op) in operators.iter().enumerate() {
        if handled[i] {
            continue;
        }
        let Some(impl_i) = op.as_implicit() else {
            continue;
        };
        let mut group = vec![i];
        handled[i] = true;
        for j in (i + 1)..operators.len() {
            if handled[j] {
                continue;
            }
            if let Some(impl_j) = operators[j].as_implicit()
                && Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                && Arc::ptr_eq(impl_i.w_diag.as_arc(), impl_j.w_diag.as_arc())
                && impl_i.p == impl_j.p
            {
                group.push(j);
                handled[j] = true;
            }
        }
        group_starts.push(group);
    }

    for group in &group_starts {
        if group.len() >= 2 {
            let lead = operators[group[0]].as_implicit().unwrap();
            let xf = match cache {
                Some(c) => lead.cached_xf(factor, c),
                None => Arc::new(lead.compute_xf(factor)),
            };
            let axes: Vec<(usize, &Array2<f64>, Option<&Array1<f64>>)> = group
                .iter()
                .map(|&k| {
                    let op = operators[k].as_implicit().unwrap();
                    (op.axis, &op.s_psi, op.c_x_psi_beta.as_deref())
                })
                .collect();
            let values = lead.trace_projected_factor_all_axes_with_xf(factor, xf.view(), &axes);
            trace += values.iter().sum::<f64>();
        } else {
            let op = &operators[group[0]];
            trace += match cache {
                Some(c) => op.trace_projected_factor_cached(factor, c),
                None => op.trace_projected_factor(factor),
            };
        }
    }

    for (i, op) in operators.iter().enumerate() {
        if handled[i] {
            continue;
        }
        trace += match cache {
            Some(c) => op.trace_projected_factor_cached(factor, c),
            None => op.trace_projected_factor(factor),
        };
    }

    trace
}


/// Vector form of the implicit-axis trace batching used by
/// [`CompositeHyperOperator`].  It returns one exact `tr(Fᵀ B_i F)` value per
/// input operator while sharing the expensive `X·F` projection and Duchon
/// row-kernel sweeps across sibling implicit ψ/ρ axes.
fn trace_projected_factors_batched(
    operators: &[Arc<dyn HyperOperator>],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<f64> {
    let mut out = vec![0.0; operators.len()];
    let mut handled = vec![false; operators.len()];

    for i in 0..operators.len() {
        if handled[i] {
            continue;
        }
        let Some(impl_i) = operators[i].as_implicit() else {
            out[i] = operators[i].trace_projected_factor_cached(factor, cache);
            handled[i] = true;
            continue;
        };

        let mut group = vec![i];
        handled[i] = true;
        for j in (i + 1)..operators.len() {
            if handled[j] {
                continue;
            }
            if let Some(impl_j) = operators[j].as_implicit()
                && Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                && Arc::ptr_eq(impl_i.w_diag.as_arc(), impl_j.w_diag.as_arc())
                && impl_i.p == impl_j.p
            {
                group.push(j);
                handled[j] = true;
            }
        }

        if group.len() >= 2 {
            let xf = impl_i.cached_xf(factor, cache);
            let axes: Vec<(usize, &Array2<f64>, Option<&Array1<f64>>)> = group
                .iter()
                .map(|&idx| {
                    let op = operators[idx].as_implicit().unwrap();
                    (op.axis, &op.s_psi, op.c_x_psi_beta.as_deref())
                })
                .collect();
            let values = impl_i.trace_projected_factor_all_axes_with_xf(factor, xf.view(), &axes);
            for (&idx, value) in group.iter().zip(values) {
                out[idx] = value;
            }
        } else {
            out[i] = operators[i].trace_projected_factor_cached(factor, cache);
        }
    }

    out
}


fn collect_projected_trace_terms<'a>(
    out_idx: usize,
    weight: f64,
    op: &'a dyn HyperOperator,
    factor: &Array2<f64>,
    dense_acc: &mut [f64],
    terms: &mut Vec<(usize, f64, &'a dyn HyperOperator)>,
) {
    if weight == 0.0 {
        return;
    }
    if let Some(composite) = op.as_composite() {
        if let Some(dense) = composite.dense.as_ref() {
            dense_acc[out_idx] += weight * dense_trace_projected_factor(dense, factor);
        }
        for inner in &composite.operators {
            collect_projected_trace_terms(
                out_idx,
                weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else if let Some(weighted) = op.as_weighted() {
        for (term_weight, inner) in &weighted.terms {
            collect_projected_trace_terms(
                out_idx,
                weight * *term_weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else {
        terms.push((out_idx, weight, op));
    }
}


fn collect_projected_matrix_terms<'a>(
    out_idx: usize,
    weight: f64,
    op: &'a dyn HyperOperator,
    factor: &Array2<f64>,
    dense_acc: &mut [Array2<f64>],
    terms: &mut Vec<(usize, f64, &'a dyn HyperOperator)>,
) {
    if weight == 0.0 {
        return;
    }
    if let Some(composite) = op.as_composite() {
        if let Some(dense) = composite.dense.as_ref() {
            dense_acc[out_idx].scaled_add(weight, &dense_projected_matrix(dense, factor));
        }
        for inner in &composite.operators {
            collect_projected_matrix_terms(
                out_idx,
                weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else if let Some(weighted) = op.as_weighted() {
        for (term_weight, inner) in &weighted.terms {
            collect_projected_matrix_terms(
                out_idx,
                weight * *term_weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else {
        terms.push((out_idx, weight, op));
    }
}


fn trace_projected_operator_terms_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_out];
    let mut handled = vec![false; terms.len()];

    for i in 0..terms.len() {
        if handled[i] {
            continue;
        }
        let Some(impl_i) = terms[i].2.as_implicit() else {
            continue;
        };
        let mut group = vec![i];
        handled[i] = true;
        for j in (i + 1)..terms.len() {
            if handled[j] {
                continue;
            }
            if let Some(impl_j) = terms[j].2.as_implicit()
                && Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                && Arc::ptr_eq(impl_i.w_diag.as_arc(), impl_j.w_diag.as_arc())
                && impl_i.p == impl_j.p
            {
                group.push(j);
                handled[j] = true;
            }
        }

        let lead = terms[group[0]].2.as_implicit().unwrap();
        let xf = lead.cached_xf(factor, cache);
        let axes: Vec<(usize, &Array2<f64>, Option<&Array1<f64>>)> = group
            .iter()
            .map(|&term_idx| {
                let op = terms[term_idx].2.as_implicit().unwrap();
                (op.axis, &op.s_psi, op.c_x_psi_beta.as_deref())
            })
            .collect();
        let values = lead.trace_projected_factor_all_axes_with_xf(factor, xf.view(), &axes);
        for (&term_idx, value) in group.iter().zip(values.iter()) {
            let (out_idx, weight, _) = terms[term_idx];
            out[out_idx] += weight * *value;
        }
    }

    for (i, (out_idx, weight, op)) in terms.iter().enumerate() {
        if handled[i] {
            continue;
        }
        out[*out_idx] += *weight * op.trace_projected_factor_cached(factor, cache);
    }

    out
}


fn projected_operator_terms_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<Array2<f64>> {
    let rank = factor.ncols();
    let mut out: Vec<Array2<f64>> = (0..n_out)
        .map(|_| Array2::<f64>::zeros((rank, rank)))
        .collect();
    for (out_idx, weight, op) in terms.iter() {
        let projected = op.projected_matrix_cached(factor, cache);
        out[*out_idx].scaled_add(*weight, &projected);
    }
    out
}


fn project_hyper_operators_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<Array2<f64>> {
    projected_operator_terms_batched(n_out, terms, factor, cache)
}


fn trace_logdet_drifts_projected_factor_batched(
    drifts: &[DriftDerivResult],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; drifts.len()];
    let mut terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
    for (idx, drift) in drifts.iter().enumerate() {
        match drift {
            DriftDerivResult::Dense(matrix) => {
                out[idx] += dense_trace_projected_factor(matrix, factor);
            }
            DriftDerivResult::Operator(op) => {
                collect_projected_trace_terms(idx, 1.0, op.as_ref(), factor, &mut out, &mut terms);
            }
        }
    }
    let batched = trace_projected_operator_terms_batched(drifts.len(), &terms, factor, cache);
    for (dst, value) in out.iter_mut().zip(batched) {
        *dst += value;
    }
    out
}


fn dense_spectral_trace_logdet_drifts_batched(
    ds: &DenseSpectralOperator,
    drifts: &[DriftDerivResult],
) -> Vec<f64> {
    trace_logdet_drifts_projected_factor_batched(drifts, &ds.g_factor, &ds.projected_factor_cache)
}


fn penalty_subspace_trace_factor(kernel: &PenaltySubspaceTrace) -> Array2<f64> {
    let (evals, evecs) = kernel
        .h_proj_inverse
        .eigh(faer::Side::Lower)
        .expect("PenaltySubspaceTrace kernel factor eigendecomposition failed");
    let r = evals.len();
    // F must satisfy F·Fᵀ = K exactly: the batched `tr(FᵀAF)` is consumed as
    // the gradient of the SAME pseudo-logdet criterion whose exact kernel the
    // per-coordinate path contracts via `h_proj_inverse` directly. The kernel
    // eigenvalues are `1/σ_a` over the kept Hessian spectrum, so their
    // dynamic range is the Hessian condition number — clamp ONLY the
    // roundoff-negative tail to zero (K is PSD by construction; a negative
    // eigenvalue is O(ε)·‖K‖ eigensolver noise, and √(max(λ,0)) is the
    // honest PSD square root). A relative floor here is NOT a stabilization:
    // raising `1/σ_max` to `√ε·r·(1/σ_min)` rewrites the criterion's
    // sensitivity along exactly the stiffest directions — where the ρ-drifts
    // `λ_k·S_k` live — inflating the analytic trace by up to `√ε·r·κ(H_pen)`
    // (O(1) once κ ≳ 1e7) while FD differentiates the true criterion. That
    // desync red-lined every iso-κ Duchon probit/logit FD test and starved
    // the spatial κ-optimizer of descent directions; Gaussian was immune
    // because the intrinsic kernel is only installed for c-nontrivial
    // families (#901).
    let mut root = evecs.clone();
    for col in 0..r {
        let scale = evals[col].max(0.0).sqrt();
        for row in 0..r {
            root[[row, col]] *= scale;
        }
    }
    crate::faer_ndarray::fast_ab(&kernel.u_s, &root)
}


fn penalty_subspace_trace_drifts_batched(
    kernel: &PenaltySubspaceTrace,
    drifts: &[DriftDerivResult],
) -> Vec<f64> {
    let factor = penalty_subspace_trace_factor(kernel);
    let cache = ProjectedFactorCache::default();
    trace_logdet_drifts_projected_factor_batched(drifts, &factor, &cache)
}


fn penalty_subspace_reduce_drifts_batched(
    kernel: &PenaltySubspaceTrace,
    drifts: &[DriftDerivResult],
) -> Vec<Array2<f64>> {
    drifts
        .iter()
        .map(|drift| match drift {
            DriftDerivResult::Dense(matrix) => kernel.reduce(matrix),
            // #901 layer-2 (outer-Hessian path): reduce the operator via
            // `U_Sᵀ·A·U_S = U_Sᵀ·A.mul_mat(U_S)` — NOT `op.to_dense()` then
            // reduce. For the GLM cubic correction `C[v] = Xᵀdiag(c⊙Xv)X` the
            // dense materialization computes near-null quadratic forms by
            // cancelling O(‖C‖) entries, and the spectral kernel's `1/σ_min`
            // then amplifies the roundoff (the +39-vs-−0.30 / ~−7.7e5 blow-up).
            // `reduce_operator` probes through the `X·U_S` matvecs instead, so
            // tiny² stays tiny — the same stability cure as the first-order
            // `trace_operator` path.
            DriftDerivResult::Operator(op) => kernel.reduce_operator(op.as_ref()),
        })
        .collect()
}


fn dense_spectral_trace_logdet_operators_batched(
    ds: &DenseSpectralOperator,
    operators: &[Arc<dyn HyperOperator>],
) -> Vec<f64> {
    if operators.is_empty() {
        return Vec::new();
    }
    if log::log_enabled!(log::Level::Info) {
        let start = std::time::Instant::now();
        let out =
            trace_projected_factors_batched(operators, &ds.g_factor, &ds.projected_factor_cache);
        let implicit_count = operators.iter().filter(|op| op.is_implicit()).count();
        dense_spectral_stage_log(
            &format!(
                "DenseSpectralOperator::trace_logdet_operators_batched dim={} rank={} ops={} implicit_ops={}",
                ds.n_dim,
                ds.g_factor.ncols(),
                operators.len(),
                implicit_count,
            ),
            start.elapsed().as_secs_f64(),
        );
        out
    } else {
        trace_projected_factors_batched(operators, &ds.g_factor, &ds.projected_factor_cache)
    }
}


impl HyperOperator for CompositeHyperOperator {
    fn as_composite(&self) -> Option<&CompositeHyperOperator> {
        Some(self)
    }

    fn dim(&self) -> usize {
        self.dim_hint
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        if self.dense.is_none() && self.operators.len() == 1 {
            self.operators[0].mul_vec_into(v, out);
            return;
        }

        out.fill(0.0);
        if let Some(dense) = self.dense.as_ref() {
            dense_matvec_into(dense, v, out.view_mut());
        }
        for op in &self.operators {
            op.scaled_add_mul_vec(v, 1.0, out.view_mut());
        }
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        if self.dense.is_none() && self.operators.len() == 1 {
            self.operators[0].mul_basis_columns_into(start, out);
            return;
        }

        out.fill(0.0);
        let cols = out.ncols();
        let end = start + cols;
        if let Some(dense) = self.dense.as_ref() {
            out += &dense.slice(ndarray::s![.., start..end]);
        }
        let mut work = Array2::<f64>::zeros((out.nrows(), cols));
        for op in &self.operators {
            op.mul_basis_columns_into(start, work.view_mut());
            out += &work;
        }
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        if self.dense.is_none() && self.operators.len() == 1 {
            self.operators[0].scaled_add_mul_vec(v, scale, out);
            return;
        }

        if let Some(dense) = self.dense.as_ref() {
            dense_matvec_scaled_add_into(dense, v, scale, out.view_mut());
        }
        for op in &self.operators {
            op.scaled_add_mul_vec(v, scale, out.view_mut());
        }
    }

    /// Forward batched apply to inner operators so their `mul_mat` overrides
    /// (matrix-free Khatri–Rao BLAS3 fuses) fire instead of the default
    /// per-column parallel matvec — which would triple-nest rayon when an
    /// inner op already parallelizes internally.
    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].mul_mat(factor);
        }
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        if let Some(dense) = self.dense.as_ref() {
            out += &dense.dot(factor);
        }
        for op in &self.operators {
            out += &op.mul_mat(factor);
        }
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].trace_projected_factor(factor);
        }

        let mut trace = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            let dense_factor = dense.dot(factor);
            trace += factor
                .iter()
                .zip(dense_factor.iter())
                .map(|(&f, &bf)| f * bf)
                .sum::<f64>();
        }
        trace += composite_trace_implicit_batched(&self.operators, factor, None);
        trace
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].trace_projected_factor_cached(factor, cache);
        }

        let mut trace = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            let dense_factor = dense.dot(factor);
            trace += factor
                .iter()
                .zip(dense_factor.iter())
                .map(|(&f, &bf)| f * bf)
                .sum::<f64>();
        }
        trace += composite_trace_implicit_batched(&self.operators, factor, Some(cache));
        trace
    }

    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].projected_matrix(factor);
        }

        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        if let Some(dense) = self.dense.as_ref() {
            let mf = crate::faer_ndarray::fast_ab(dense, factor);
            projected += &crate::faer_ndarray::fast_atb(factor, &mf);
        }
        for op in &self.operators {
            projected += &op.projected_matrix(factor);
        }
        projected
    }

    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].projected_matrix_cached(factor, cache);
        }

        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        if let Some(dense) = self.dense.as_ref() {
            let mf = crate::faer_ndarray::fast_ab(dense, factor);
            projected += &crate::faer_ndarray::fast_atb(factor, &mf);
        }
        for op in &self.operators {
            projected += &op.projected_matrix_cached(factor, cache);
        }
        projected
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let mut total = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            total += dense_bilinear(dense, v.view(), u.view());
        }
        for op in &self.operators {
            total += op.bilinear(v, u);
        }
        total
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        let mut total = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            total += dense_bilinear(dense, v, u);
        }
        for op in &self.operators {
            total += op.bilinear_view(v, u);
        }
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self
            .dense
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((self.dim_hint, self.dim_hint)));
        for op in &self.operators {
            out += &op.to_dense();
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.operators.iter().any(|op| op.is_implicit())
    }
}


/// Fixed-β Hessian drift payload for a single hyper coordinate.
///
/// Some coordinates are naturally dense. Others are most efficient as
/// operator-backed implicit drifts. A few workflows need to carry both a dense
/// correction and an operator-backed main term, so this type can represent both
/// simultaneously without relying on dummy zero-sized matrices.
/// A block-local square matrix embedded in joint p-space. Supports O(p_block²)
/// matvec without materializing to full p×p.
#[derive(Clone)]
pub struct BlockLocalDrift {
    pub local: Array2<f64>,
    pub start: usize,
    pub end: usize,
    /// Total joint dimension `p` — recorded at construction so `dim()` is
    /// `O(1)` and `to_dense` does not need a separate hint.  Must satisfy
    /// `total_dim >= end`.
    pub total_dim: usize,
}


impl HyperOperator for BlockLocalDrift {
    fn dim(&self) -> usize {
        self.total_dim
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(v.len());
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::zeros(v.len());
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        out.fill(0.0);
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let out_block = out.slice_mut(ndarray::s![self.start..self.end]);
        dense_matvec_into(&self.local, v_block, out_block);
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        out.fill(0.0);
        let global_end = start + out.ncols();
        let col_start = start.max(self.start);
        let col_end = global_end.min(self.end);
        if col_start >= col_end {
            return;
        }
        let local_col_start = col_start - self.start;
        let local_col_end = col_end - self.start;
        let out_col_start = col_start - start;
        let out_col_end = col_end - start;
        out.slice_mut(ndarray::s![
            self.start..self.end,
            out_col_start..out_col_end
        ])
        .assign(
            &self
                .local
                .slice(ndarray::s![.., local_col_start..local_col_end]),
        );
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let out_block = out.slice_mut(ndarray::s![self.start..self.end]);
        dense_matvec_scaled_add_into(&self.local, v_block, scale, out_block);
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let u_block = u.slice(ndarray::s![self.start..self.end]);
        u_block.dot(&self.local.dot(&v_block))
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let u_block = u.slice(ndarray::s![self.start..self.end]);
        let mut total = 0.0;
        for (row, u_value) in self.local.rows().into_iter().zip(u_block.iter().copied()) {
            let mut row_dot = 0.0;
            for (entry, v_value) in row.iter().copied().zip(v_block.iter().copied()) {
                row_dot += entry * v_value;
            }
            total += u_value * row_dot;
        }
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.total_dim;
        let mut out = Array2::zeros((p, p));
        out.slice_mut(ndarray::s![self.start..self.end, self.start..self.end])
            .assign(&self.local);
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }

    fn block_local_data(&self) -> Option<(&Array2<f64>, usize, usize)> {
        Some((&self.local, self.start, self.end))
    }
}


#[derive(Clone)]
pub struct HyperCoordDrift {
    /// Full p×p dense matrix (forces dense fallback when present).
    pub dense: Option<Array2<f64>>,
    /// Block-local penalty contribution (does NOT force dense fallback).
    pub block_local: Option<BlockLocalDrift>,
    /// Implicit operator (fast path).
    pub operator: Option<Arc<dyn HyperOperator>>,
}


impl HyperCoordDrift {
    pub fn none() -> Self {
        Self {
            dense: None,
            block_local: None,
            operator: None,
        }
    }

    pub fn from_dense(dense: Array2<f64>) -> Self {
        Self {
            dense: Some(dense),
            block_local: None,
            operator: None,
        }
    }

    pub fn from_operator(operator: Arc<dyn HyperOperator>) -> Self {
        Self {
            dense: None,
            block_local: None,
            operator: Some(operator),
        }
    }

    pub fn from_parts(
        dense: Option<Array2<f64>>,
        operator: Option<Arc<dyn HyperOperator>>,
    ) -> Self {
        let dense = dense.filter(|mat| !(operator.is_some() && mat.is_empty()));
        Self {
            dense,
            block_local: None,
            operator,
        }
    }

    pub fn from_block_local_and_operator(
        local: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
        operator: Option<Arc<dyn HyperOperator>>,
    ) -> Self {
        Self {
            dense: None,
            block_local: Some(BlockLocalDrift {
                local,
                start,
                end,
                total_dim,
            }),
            operator,
        }
    }

    pub fn has_operator(&self) -> bool {
        self.operator.is_some()
    }

    /// Returns true when some part of the drift can stay operator-backed.
    /// A dense correction may still be present; callers should compose it with
    /// the operator pieces instead of materializing those pieces into dense form.
    pub fn uses_operator_fast_path(&self) -> bool {
        self.operator.is_some() || self.block_local.is_some()
    }

    pub fn operator_ref(&self) -> Option<&dyn HyperOperator> {
        self.operator.as_ref().map(Arc::as_ref)
    }

    pub fn materialize(&self) -> Array2<f64> {
        let p = self.infer_dim();
        if p == 0 {
            return Array2::zeros((0, 0));
        }
        let mut out = self.dense.clone().unwrap_or_else(|| Array2::zeros((p, p)));
        if let Some(bl) = &self.block_local {
            out.slice_mut(ndarray::s![bl.start..bl.end, bl.start..bl.end])
                .scaled_add(1.0, &bl.local);
        }
        if let Some(op) = &self.operator {
            out += &op.to_dense();
        }
        out
    }

    pub fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(v.len());
        self.scaled_add_apply(v.view(), 1.0, &mut out);
        out
    }

    pub fn scaled_add_apply(&self, v: ArrayView1<'_, f64>, scale: f64, out: &mut Array1<f64>) {
        assert_eq!(v.len(), out.len());
        if scale == 0.0 {
            return;
        }
        if let Some(dense) = &self.dense {
            dense_matvec_scaled_add_into(dense, v, scale, out.view_mut());
        }
        if let Some(bl) = &self.block_local {
            let v_block = v.slice(ndarray::s![bl.start..bl.end]);
            let out_block = out.slice_mut(ndarray::s![bl.start..bl.end]);
            dense_matvec_scaled_add_into(&bl.local, v_block, scale, out_block);
        }
        if let Some(op) = &self.operator {
            op.scaled_add_mul_vec(v, scale, out.view_mut());
        }
    }

    fn infer_dim(&self) -> usize {
        if let Some(d) = &self.dense {
            return d.nrows();
        }
        if let Some(op) = &self.operator {
            return op.dim();
        }
        if let Some(bl) = &self.block_local {
            return bl.total_dim;
        }
        0
    }
}


/// Implicit Hessian-drift operator for a single anisotropic ψ_d coordinate.
///
/// Computes B_d · v on the fly:
///   B_d · v = (∂X/∂ψ_d)^T (W · (X · v)) + X^T (W · ((∂X/∂ψ_d) · v)) + S_{ψ_d} · v
///
/// The first two terms use the implicit design-derivative operator (no dense
/// (n × p) matrices), and S_{ψ_d} is a dense (p × p) penalty matrix (manageable).
///
/// Storage: the implicit operator holds O(n·k·D) radial jets, plus references
/// to an active-basis X design operator and W (the working weights). The
/// penalty matrix S_{ψ_d} is stored as a dense (p × p) matrix.
/// Thread-local scratch buffers for `ImplicitHyperOperator::mul_vec_into`.
/// Reused across PCG iterations and basis-column sweeps so each matvec
/// avoids three fresh O(n)/O(p) allocations.
mod implicit_matvec_scratch {
    use std::cell::RefCell;

    pub(super) struct Scratch {
        pub x_v: Vec<f64>,
        pub n_work: Vec<f64>,
        pub p_work: Vec<f64>,
    }

    impl Scratch {
        const fn new() -> Self {
            Self {
                x_v: Vec::new(),
                n_work: Vec::new(),
                p_work: Vec::new(),
            }
        }
    }

    thread_local! {
        static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::new()) };
    }

    pub(super) fn with<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
        SCRATCH.with(|cell| f(&mut cell.borrow_mut()))
    }
}


pub struct ImplicitHyperOperator {
    /// The implicit design-derivative operator (shared across all axes).
    pub implicit_deriv: std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
    /// Which axis this operator is for.
    pub axis: usize,
    /// The active-basis design matrix X. This may be lazy / operator-backed.
    pub(crate) x_design: std::sync::Arc<DesignMatrix>,
    /// Working weights W (diagonal, length n) — observed-information curvature,
    /// signed for non-canonical links. Carried as the owned [`crate::matrix::SignedWeightsArc`]
    /// newtype so the sign character is construction-enforced at the operator
    /// struct boundary; the function-boundary contract from `linalg/matrix.rs`
    /// is no longer reconstructable accidentally inside `mul_vec`.
    pub(crate) w_diag: crate::matrix::SignedWeightsArc,
    /// Penalty derivative matrix S_{ψ_d} (p × p), dense.
    pub s_psi: Array2<f64>,
    /// Total basis dimension p.
    pub(crate) p: usize,
    /// Non-Gaussian fixed-β third-derivative correction: c ⊙ (X_{ψ_d} β̂),
    /// length n. When present, the operator additionally applies
    /// `Xᵀ diag(c_x_psi_beta) X v` so that the full B_d formula
    /// `B_d v = (∂X/∂ψ_d)ᵀ W X v + Xᵀ W (∂X/∂ψ_d) v + Xᵀ diag(c ⊙ X_{ψ_d} β̂) X v + S_{ψ_d} v`
    /// is matrix-free for non-Gaussian likelihoods. `None` for Gaussian
    /// identity (c ≡ 0 there).
    pub c_x_psi_beta: Option<std::sync::Arc<Array1<f64>>>,
}


impl HyperOperator for ImplicitHyperOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        // Single canonical path: route every matvec through `mul_vec_into`,
        // which routes through `matvec_with_shared_xz_into`. The four terms of
        // B_d are assembled there, with the third-derivative correction added
        // by `accumulate_c_correction_xt_into` so the four matvec entry points
        // share one inner kernel.
        let mut out = Array1::<f64>::zeros(self.p);
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.p);
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
        assert_eq!(v.len(), self.p);
        let n_obs = self.w_diag.len();
        // Reuse thread-local scratch across repeated matvec calls (e.g.
        // PCG iterations, basis-column sweeps) instead of allocating
        // (2 n_obs + p) f64s every time.
        implicit_matvec_scratch::with(|s| {
            s.x_v.clear();
            s.x_v.resize(n_obs, 0.0);
            s.n_work.clear();
            s.n_work.resize(n_obs, 0.0);
            s.p_work.clear();
            s.p_work.resize(self.p, 0.0);
            let mut x_v_view = ndarray::ArrayViewMut1::from(s.x_v.as_mut_slice());
            let n_work_view = ndarray::ArrayViewMut1::from(s.n_work.as_mut_slice());
            let p_work_view = ndarray::ArrayViewMut1::from(s.p_work.as_mut_slice());
            design_matrix_apply_view_into(&self.x_design, v, x_v_view.view_mut());
            self.matvec_with_shared_xz_into(x_v_view.view(), v, out, n_work_view, p_work_view);
        });
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        assert!(start + cols <= self.p);

        let n_obs = self.w_diag.len();
        let mut basis = Array1::<f64>::zeros(self.p);
        let mut x_col = Array1::<f64>::zeros(n_obs);
        let mut dx_col = Array1::<f64>::zeros(n_obs);
        let mut weighted = Array1::<f64>::zeros(n_obs);
        let mut term = Array1::<f64>::zeros(self.p);

        for local_col in 0..cols {
            let global_col = start + local_col;
            let mut out_col = out.column_mut(local_col);
            out_col.assign(&self.s_psi.column(global_col));

            design_matrix_column_into(&self.x_design, global_col, x_col.view_mut());
            Zip::from(weighted.view_mut())
                .and(self.w_diag.view())
                .and(x_col.view())
                .par_for_each(|dst, &w, &x| *dst = w * x);
            term.assign(
                &self
                    .implicit_deriv
                    .transpose_mul(self.axis, &weighted.view())
                    .expect("radial scalar evaluation failed during implicit hyper transpose_mul"),
            );
            out_col += &term;

            basis[global_col] = 1.0;
            dx_col.assign(
                &self
                    .implicit_deriv
                    .forward_mul(self.axis, &basis.view())
                    .expect("radial scalar evaluation failed during implicit hyper forward_mul"),
            );
            basis[global_col] = 0.0;

            Zip::from(weighted.view_mut())
                .and(self.w_diag.view())
                .and(dx_col.view())
                .par_for_each(|dst, &w, &dx| *dst = w * dx);
            design_matrix_transpose_apply_view_into(
                &self.x_design,
                weighted.view(),
                term.view_mut(),
            );
            out_col += &term;

            // Non-Gaussian third-derivative correction column j: shared kernel.
            self.accumulate_c_correction_xt_into(
                x_col.view(),
                weighted.view_mut(),
                term.view_mut(),
                out_col,
            );
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.bilinear_view(v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(v.len(), self.p);
        assert_eq!(u.len(), self.p);

        let x_v = design_matrix_apply_view(&self.x_design, v);
        let x_u = design_matrix_apply_view(&self.x_design, u);
        let dx_v = self
            .implicit_deriv
            .forward_mul(self.axis, &v)
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        let dx_u = self
            .implicit_deriv
            .forward_mul(self.axis, &u)
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");

        let w = &*self.w_diag;
        let mut design = 0.0;
        for i in 0..w.len() {
            design += dx_v[i] * w[i] * x_u[i];
            design += dx_u[i] * w[i] * x_v[i];
        }

        design += self.c_correction_bilinear(&x_v, &x_u);

        let penalty = dense_bilinear(&self.s_psi, v, u);

        design + penalty
    }

    fn is_implicit(&self) -> bool {
        true
    }

    fn as_implicit(&self) -> Option<&ImplicitHyperOperator> {
        Some(self)
    }

    /// Compute `tr(F^T B F)` directly via fused chunked BLAS3 GEMMs on the
    /// shared X and the shared raw kernel matrix, bypassing the rank-many
    /// separate matvecs the default impl would run through the lazy /
    /// operator-backed design.
    ///
    /// **Why this matters:** the default trait impl is
    ///   `let bf = self.mul_mat(F); (F ⊙ bf).sum()`
    /// which calls `mul_vec_into` per column of `F` (rank columns). On a
    /// lazy Duchon / Matérn / CTN design each `mul_vec_into` triggers a
    /// full `O(n · p · kernel_eval)` row-streamed matvec — and with rank ≈ p
    /// at large-scale shape (16D-Duchon-aniso 32 ψ-axes, p ≈ 95, n = 320 K)
    /// the per-axis trace landed at ~30 s. With 32 axes per outer Hessian
    /// eval and ~5 outer iters that's the ~1 hr large-scale timeout.
    ///
    /// Algebra:
    /// ```text
    ///   B_d = D_d^T W X + X^T W D_d  + X^T diag(c) X  + S_psi
    ///   D_d = (∂X/∂ψ_d) = K_d · Z_unproject       (raw kernel · unproject)
    ///   tr(F^T B_d F) = 2 · ⟨W ⊙ DXF, XF⟩ + ⟨c ⊙ XF, XF⟩ + tr(F^T S_psi F)
    /// ```
    /// where `K_d` is the raw (n × n_knots) per-pair kernel scalar matrix
    /// for axis `d` (`q · s_combo + c · coeff_sum · φ` per (i, j) pair) and
    /// `Z_unproject` is the identifiability/padding back-projection.
    ///
    /// We compute `U_knot = unproject_matrix(F)` once at (n_knots × rank),
    /// then for each row chunk do a fused pass:
    ///   * `XF_chunk  = X_chunk · F`        (chunk × rank)  — shared-X GEMM
    ///   * `Kd_chunk  = row_chunk_first_raw`(chunk × n_knots) — raw kernel
    ///   * `DXF_chunk = Kd_chunk · U_knot`  (chunk × rank)  — single GEMM
    /// and immediately accumulate `⟨W ⊙ DXF, XF⟩` and `⟨c ⊙ XF, XF⟩` over
    /// the chunk, never materialising full XF or DXF.
    ///
    /// This replaces the previous `rank`-many `forward_mul` apply loop. On
    /// the large-scale margslope-aniso-duchon16d shard each per-axis trace
    /// drops from ~30 s to a single chunked-GEMM cost.
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return 0.0;
        }
        let xf = self.compute_xf(factor);
        self.trace_projected_factor_with_xf(factor, xf.view())
    }

    /// Cached variant — *the* hot-path optimisation for large-scale outer
    /// gradient/Hessian sweeps. Every ψ-axis built atop the same `x_design`
    /// (e.g. all 32 ψ-axes of a marginal-slope model, or the same axis hit
    /// from `g_factor` and `w_factor` traces) shares one chunked
    /// `X · F` design GEMM per `(x_design, factor)` pair via
    /// [`ProjectedFactorCache`]. With 32 axes per outer-gradient sweep and
    /// O(rank) more cross-axis traces inside the outer-Hessian build, the
    /// cache turns 32× redundant `O(n · p · rank)` GEMMs into a single one
    /// per outer iter. At large-scale shape (`n = 320 K`, `p = rank = 95`) that
    /// is the difference between minutes and seconds of design-GEMM work.
    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return 0.0;
        }
        let xf = self.cached_xf(factor, cache);
        self.trace_projected_factor_with_xf(factor, xf.view())
    }
}


/// Row-block size that keeps each streamed `n × cols` chunk near an 8 MiB
/// working set, with a 512-row floor so a wide design still makes useful BLAS-3
/// progress per block, capped at the total row count. Shared by the implicit
/// operator's row-streaming kernels so they cannot drift apart.
fn byte_balanced_row_chunk(cols: usize, n_rows: usize) -> usize {
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_CHUNK_ROWS: usize = 512;
    let bytes_per_row = cols.max(1) * std::mem::size_of::<f64>();
    (TARGET_BYTES / bytes_per_row)
        .max(MIN_CHUNK_ROWS)
        .min(n_rows)
}


impl ImplicitHyperOperator {
    /// Chunked `X · F` via faer SIMD-parallel GEMM. The chunk-row sizing
    /// targets ~8 MiB live blocks so the (chunk_n × p) row slice and
    /// (chunk_n × rank) result both stay in L2/L3 across realistic large-scale
    /// shapes; the kernel mirrors `xt_logdet_kernel_x_diagonal`'s sizing
    /// rule. Caller wraps this in [`Self::cached_xf`] when invariance
    /// across ψ-axes lets one matrix serve every axis at this `(x_design,
    /// factor)` pair.
    fn compute_xf(&self, factor: &Array2<f64>) -> Array2<f64> {
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        let mut xf = Array2::<f64>::zeros((n_obs, rank));
        let chunk_rows = byte_balanced_row_chunk(self.p + rank, n_obs);
        let mut start = 0usize;
        while start < n_obs {
            let end = (start + chunk_rows).min(n_obs);
            let rows = self
                .x_design
                .try_row_chunk(start..end)
                // SAFETY: `try_row_chunk` only fails on operator
                // implementation bugs — `start..end` is built from
                // `0..n_obs = 0..x_design.nrows()` with
                // `end = (start+chunk_rows).min(n_obs)`, so the range is
                // always a valid sub-range of `x_design`. Failure means the
                // operator broke its row-chunk contract.
                .unwrap_or_else(|err| {
                    // SAFETY: row range is a valid sub-range of x_design; failure means operator broke contract.
                    reml_contract_panic(format!(
                        "ImplicitHyperOperator::compute_xf row chunk failed: {err}"
                    ))
                });
            let block = crate::faer_ndarray::fast_ab(&rows, factor);
            xf.slice_mut(ndarray::s![start..end, ..]).assign(&block);
            start = end;
        }
        xf
    }

    /// Look up `X · F` from the [`ProjectedFactorCache`] (compute-on-miss).
    /// Cache key combines the shared `x_design` Arc pointer and the
    /// factor's value fingerprint, so two `ImplicitHyperOperator` instances
    /// built atop the same `x_design` (e.g. axis-0 and axis-1 of a 32-axis
    /// ψ-block) consult the same cache slot and hit after the first
    /// computes.
    fn cached_xf(&self, factor: &Array2<f64>, cache: &ProjectedFactorCache) -> Arc<Array2<f64>> {
        let design_id = Arc::as_ptr(&self.x_design) as usize;
        let key = ProjectedFactorKey::from_factor_view(design_id, factor.view());
        cache.get_or_insert_with(key, || self.compute_xf(factor))
    }

    /// Evaluate `tr(Fᵀ B_d F)` given a precomputed `X · F`. Pulls every
    /// per-axis-redundant `X · F` out of the inner loop so the cache (or
    /// caller-supplied matrix) covers every ψ-axis at once. The remaining
    /// per-axis work is the row-kernel build (`row_chunk_first_raw`),
    /// the `K_d · U_knot` GEMM, the fused `⟨W ⊙ DXF, XF⟩` inner products,
    /// and the small dense penalty contraction.
    fn trace_projected_factor_with_xf(&self, factor: &Array2<f64>, xf: ArrayView2<'_, f64>) -> f64 {
        let rank = factor.ncols();
        let n_obs = self.w_diag.len();
        assert_eq!(xf.dim(), (n_obs, rank));

        // Once: unproject F to raw knot space → (n_knots × rank).
        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());

        // Match the chunk sizing `xt_logdet_kernel_x_diagonal` uses so the
        // live block stays in L2/L3 across realistic large-scale shapes.
        let chunk_rows = byte_balanced_row_chunk(self.p + rank, n_obs);

        let w = self.w_diag.as_ref();
        let c_opt = self.c_x_psi_beta.as_ref().map(|arc| arc.as_ref());
        let mut design_total = 0.0_f64;
        let mut correction_total = 0.0_f64;
        let mut start = 0usize;
        while start < n_obs {
            let end = (start + chunk_rows).min(n_obs);
            let chunk_n = end - start;

            // Cached-or-precomputed X·F slice for this chunk.
            let xf_chunk = xf.slice(ndarray::s![start..end, ..]);

            // Raw kernel scalars for axis d on this chunk, then a single
            // (chunk × n_knots) · (n_knots × rank) GEMM gives DXF_chunk.
            let kd_chunk = self
                .implicit_deriv
                .row_chunk_first_raw(self.axis, start..end)
                .expect("radial scalar evaluation failed during implicit hyper forward_mul_matrix");
            let dxf_chunk = crate::faer_ndarray::fast_ab(&kd_chunk, &u_knot);

            // Fused inner-product accumulation.
            for i_local in 0..chunk_n {
                let i = start + i_local;
                let w_i = w[i];
                let dxf_row = dxf_chunk.row(i_local);
                let xf_row = xf_chunk.row(i_local);
                for k in 0..rank {
                    design_total += dxf_row[k] * w_i * xf_row[k];
                }
                if let Some(c) = c_opt {
                    let c_i = c[i];
                    for k in 0..rank {
                        let v = xf_row[k];
                        correction_total += c_i * v * v;
                    }
                }
            }
            start = end;
        }

        // Penalty trace: tr(F^T S_psi F) via dense BLAS3.
        let s_f = self.s_psi.dot(factor);
        let penalty: f64 = factor.iter().zip(s_f.iter()).map(|(&f, &s)| f * s).sum();

        2.0 * design_total + correction_total + penalty
    }

    /// Batched-axis sibling of [`Self::trace_projected_factor_with_xf`].
    /// Returns `tr(Fᵀ B_d F)` for every `(axis, s_psi, c_x_psi_beta)` triple
    /// in `axes`, sharing the unproject-and-row-sweep work across axes that
    /// only differ in their axis index / penalty matrix / correction vector.
    fn trace_projected_factor_all_axes_with_xf(
        &self,
        factor: &Array2<f64>,
        xf: ArrayView2<'_, f64>,
        axes: &[(usize, &Array2<f64>, Option<&Array1<f64>>)],
    ) -> Vec<f64> {
        let rank = factor.ncols();
        let n_obs = self.w_diag.len();
        assert_eq!(xf.dim(), (n_obs, rank));

        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());

        let chunk_rows = byte_balanced_row_chunk(self.p + rank, n_obs.max(1));

        let w = self.w_diag.as_ref();
        let mut design_totals = vec![0.0_f64; axes.len()];
        let mut correction_totals = vec![0.0_f64; axes.len()];

        let mut start = 0usize;
        while start < n_obs {
            let end = (start + chunk_rows).min(n_obs);
            let chunk_n = end - start;
            let xf_chunk = xf.slice(ndarray::s![start..end, ..]);

            for (axis_idx, (axis, _s_psi, c_opt_axis)) in axes.iter().enumerate() {
                let kd_chunk = self
                    .implicit_deriv
                    .row_chunk_first_raw(*axis, start..end)
                    .expect(
                        "radial scalar evaluation failed during \
                         trace_projected_factor_all_axes_with_xf",
                    );
                let dxf_chunk = crate::faer_ndarray::fast_ab(&kd_chunk, &u_knot);

                for i_local in 0..chunk_n {
                    let i = start + i_local;
                    let w_i = w[i];
                    let dxf_row = dxf_chunk.row(i_local);
                    let xf_row = xf_chunk.row(i_local);
                    for k in 0..rank {
                        design_totals[axis_idx] += dxf_row[k] * w_i * xf_row[k];
                    }
                    if let Some(c) = c_opt_axis {
                        let c_i = c[i];
                        for k in 0..rank {
                            let v = xf_row[k];
                            correction_totals[axis_idx] += c_i * v * v;
                        }
                    }
                }
            }
            start = end;
        }

        axes.iter()
            .enumerate()
            .map(|(idx, (_axis, s_psi, _c_opt_axis))| {
                let s_f = s_psi.dot(factor);
                let penalty: f64 = factor.iter().zip(s_f.iter()).map(|(&f, &s)| f * s).sum();
                2.0 * design_totals[idx] + correction_totals[idx] + penalty
            })
            .collect()
    }

    fn accumulate_c_correction_xt_into(
        &self,
        x_col: ArrayView1<'_, f64>,
        mut n_work: ArrayViewMut1<'_, f64>,
        mut p_work: ArrayViewMut1<'_, f64>,
        mut out_col: ArrayViewMut1<'_, f64>,
    ) {
        let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() else {
            return;
        };
        let c = c_x_psi_beta.as_ref();
        assert_eq!(x_col.len(), c.len());
        assert_eq!(n_work.len(), c.len());
        assert_eq!(p_work.len(), self.p);

        for i in 0..c.len() {
            n_work[i] = c[i] * x_col[i];
        }
        design_matrix_transpose_apply_view_into(&self.x_design, n_work.view(), p_work.view_mut());
        out_col += &p_work;
    }

    fn c_correction_bilinear(&self, x_v: &Array1<f64>, x_u: &Array1<f64>) -> f64 {
        let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() else {
            return 0.0;
        };
        x_v.iter()
            .zip(x_u.iter())
            .zip(c_x_psi_beta.iter())
            .map(|((&xv, &xu), &c)| xv * c * xu)
            .sum()
    }

    /// Compute the design-part bilinear form u^T (X^T C_d X) z using precomputed
    /// shared X-multiplies, avoiding the full B_d matvec.
    ///
    /// The design part of B_d is:
    ///   (∂X/∂ψ_d)^T W X + X^T W (∂X/∂ψ_d)
    ///
    /// For vectors z and u, the bilinear form u^T [design_part] z equals:
    ///   ((∂X/∂ψ_d) u)^T (W (Xz)) + (Xu)^T (W ((∂X/∂ψ_d) z))
    ///   = 2 * (w ⊙ y_vec)^T dx_z       [when u = u, z = z]
    ///
    /// where y_vec = X u, dx_z = (∂X/∂ψ_d) z.
    ///
    /// But the full bilinear form is NOT symmetric in its dependence on z vs u
    /// through the design derivative, so we compute both cross-terms:
    ///   dx_z^T (w ⊙ y_vec) + dx_u^T (w ⊙ x_vec)
    ///
    /// # Arguments
    /// - `x_vec`: X z (precomputed, shared across axes)
    /// - `y_vec`: X u (precomputed, shared across axes)
    /// - `z`: the probe vector (needed for forward_mul and penalty)
    /// - `u`: H⁻¹ z (needed for forward_mul and penalty)
    ///
    /// # Returns
    /// The full bilinear form u^T B_d z = design_part + penalty_part.
    pub fn bilinear_with_shared_x(
        &self,
        x_vec: &Array1<f64>,
        y_vec: &Array1<f64>,
        z: &Array1<f64>,
        u: &Array1<f64>,
    ) -> f64 {
        // Design part: dx_z^T (w ⊙ y_vec) + dx_u^T (w ⊙ x_vec)
        let dx_z = self
            .implicit_deriv
            .forward_mul(self.axis, &z.view())
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        let dx_u = self
            .implicit_deriv
            .forward_mul(self.axis, &u.view())
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");

        let mut design = 0.0f64;
        let w = &*self.w_diag;
        for i in 0..x_vec.len() {
            let wi = w[i];
            design += dx_z[i] * wi * y_vec[i];
            design += dx_u[i] * wi * x_vec[i];
        }

        // Non-Gaussian fixed-β third-derivative correction:
        //   uᵀ Xᵀ diag(c ⊙ X_{ψ_d} β̂) X z = Σ_i (X u)_i · c_x_psi_beta_i · (X z)_i
        //   = Σ_i y_vec[i] · c_x_psi_beta[i] · x_vec[i]
        if let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() {
            let c = c_x_psi_beta.as_ref();
            for i in 0..x_vec.len() {
                design += y_vec[i] * c[i] * x_vec[i];
            }
        }

        // Penalty part: u^T S_psi z
        let penalty = dense_bilinear(&self.s_psi, z.view(), u.view());

        design + penalty
    }

    /// Compute the design-part contribution to A_d z without the X^T step.
    ///
    /// Returns the n-vector C_d (X z) where C_d encodes the diagonal weighting.
    /// Specifically: (∂X/∂ψ_d)^T maps FROM n-space, but for stochastic trace
    /// estimation we need q_d = A_d z = X^T (C_d x_vec) + P_d z.
    ///
    /// This method computes q_d = A_d z using the shared x_vec = X z:
    ///   q_d = (∂X/∂ψ_d)^T (W (X z)) + X^T (W ((∂X/∂ψ_d) z)) + S_psi z
    /// which is the standard mul_vec but we can share x_vec across axes.
    pub fn matvec_with_shared_xz_into(
        &self,
        x_vec: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        mut out: ArrayViewMut1<'_, f64>,
        mut n_work: ArrayViewMut1<'_, f64>,
        mut p_work: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(z.len(), self.p);
        assert_eq!(out.len(), self.p);
        assert_eq!(n_work.len(), self.w_diag.len());
        assert_eq!(p_work.len(), self.p);

        let w = &*self.w_diag;
        for i in 0..w.len() {
            n_work[i] = w[i] * x_vec[i];
        }
        let term1 = self
            .implicit_deriv
            .transpose_mul(self.axis, &n_work.view())
            .expect("radial scalar evaluation failed during implicit hyper transpose_mul");
        out.assign(&term1);

        let dx_z = self
            .implicit_deriv
            .forward_mul(self.axis, &z)
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        for i in 0..w.len() {
            n_work[i] = w[i] * dx_z[i];
        }
        design_matrix_transpose_apply_view_into(&self.x_design, n_work.view(), p_work.view_mut());
        out += &p_work;

        dense_matvec_into(&self.s_psi, z, p_work.view_mut());
        out += &p_work;

        // Non-Gaussian fixed-β third-derivative correction.
        if let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() {
            let c = c_x_psi_beta.as_ref();
            for i in 0..w.len() {
                n_work[i] = c[i] * x_vec[i];
            }
            design_matrix_transpose_apply_view_into(
                &self.x_design,
                n_work.view(),
                p_work.view_mut(),
            );
            out += &p_work;
        }
    }
}


/// Operator-backed fixed-β Hessian drift for sparse-exact τ coordinates.
///
/// This stays in the original sparse/native coefficient basis and computes the
/// exact first-order τ Hessian drift
///   B_τ = X_τᵀ W X + Xᵀ W X_τ + Xᵀ diag(c ⊙ X_τ β̂) X + S_τ − (H_φ)_{τ}|_β
/// without materializing the full dense matrix up front.
pub struct SparseDirectionalHyperOperator {
    /// Original-basis design derivative X_τ.
    pub(crate) x_tau: super::HyperDesignDerivative,
    /// Design matrix X in the sparse-native basis.
    pub(crate) x_design: DesignMatrix,
    /// Working weights W (diagonal) — observed-information curvature, signed
    /// for non-canonical links.  Carried as the owned [`crate::matrix::SignedWeightsArc`]
    /// newtype so the sign character is construction-enforced at the operator
    /// struct boundary.
    pub(crate) w_diag: crate::matrix::SignedWeightsArc,
    /// Penalty derivative S_τ.
    pub(crate) s_tau: Array2<f64>,
    /// Fixed-β non-Gaussian curvature term c ⊙ (X_τ β̂), if applicable.
    pub(crate) c_x_tau_beta: Option<Array1<f64>>,
    /// Fixed-β Firth partial Hessian drift (H_φ)_{τ}|_β, if applicable.
    pub(crate) firth_hphi_tau_partial: Option<Array2<f64>>,
    /// Total coefficient dimension.
    pub(crate) p: usize,
}


impl HyperOperator for SparseDirectionalHyperOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p);

        // X v
        let x_v = self.x_design.matrixvectormultiply(v);

        // X_tauᵀ (W (X v))
        let w_x_v = &*self.w_diag * &x_v;
        let term1 = self
            .x_tau
            .transpose_mul_original(&w_x_v)
            .expect("SparseDirectionalHyperOperator transpose product should be shape-consistent");

        // Xᵀ (W (X_tau v))
        let x_tau_v = self
            .x_tau
            .forward_mul_original(v)
            .expect("SparseDirectionalHyperOperator forward product should be shape-consistent");
        let w_x_tau_v = &*self.w_diag * &x_tau_v;
        let term2 = self.x_design.transpose_vector_multiply(&w_x_tau_v);

        // S_tau v
        let term3 = self.s_tau.dot(v);

        let mut out = term1 + term2 + term3;

        // Non-Gaussian fixed-beta curvature: Xᵀ diag(c ⊙ X_tau β̂) X v
        if let Some(c_x_tau_beta) = self.c_x_tau_beta.as_ref() {
            let weighted = c_x_tau_beta * &x_v;
            out += &self.x_design.transpose_vector_multiply(&weighted);
        }

        // Firth fixed-beta partial: subtract (H_φ)_{τ}|_β v
        if let Some(hphi_tau_partial) = self.firth_hphi_tau_partial.as_ref() {
            out -= &hphi_tau_partial.dot(v);
        }

        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
    fn as_sparse_directional(&self) -> Option<&SparseDirectionalHyperOperator> {
        Some(self)
    }
}


/// Matrix-free GLM cubic-correction drift `C[v] = −Xᵀ diag(c ⊙ X v) X`
/// (rows masked to the active Hessian-curvature surface, sign folded into
/// the stored diagonal).
///
/// # Why this must stay an operator (#901 layer 2)
///
/// The spectral logdet kernel evaluates `tr(H⁺ · C)` as
/// `Σ_a (1/σ_a) · u_aᵀ C u_a` over the eigenpairs of `H_pen`. For a
/// near-null eigenvector (`σ_min ~ 1e−4` on the Duchon fixtures) the true
/// quadratic form is tiny — `‖X u_a‖² ≲ σ_a / w_min` — but a DENSE
/// materialization of `C` computes it as a cancellation across entries of
/// magnitude `‖C‖`, leaving roundoff `~ ε‖C‖p` that the kernel then
/// amplifies by `1/σ_min`. On the iso-κ Duchon binomial FD drivers this
/// turned a true cubic trace of `−0.30` into `+39.0`, and `~−7.7e5` on the
/// κ-scaled ψ arms where `‖C‖ ~ λ · ∂S/∂ψ` — the dominant #901 blow-up.
///
/// In operator form the kernel probes `C · u_a = −Xᵀ(d ⊙ (X u_a))`: the
/// cancellation happens inside the `X u_a` matvec (error `~ ε‖X‖‖u_a‖`),
/// and the quadratic form is the *square* of that already-small vector —
/// tiny² stays tiny, so the `1/σ_a` amplification acts on a relatively
/// accurate value. This is the same stability argument as evaluating
/// leverages via `(X u)ᵀ d (X u)` instead of `uᵀ (XᵀdX) u`.
pub struct GlmCurvatureCorrectionOperator {
    /// Design matrix X in the transformed basis (matrix-free capable).
    pub(crate) x_design: DesignMatrix,
    /// Pre-masked, sign-folded diagonal `−(c ⊙ X v)` over active rows.
    pub(crate) neg_c_xv: Array1<f64>,
    /// Total coefficient dimension.
    pub(crate) p: usize,
}


impl HyperOperator for GlmCurvatureCorrectionOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p);
        let x_v = self.x_design.matrixvectormultiply(v);
        let weighted = &self.neg_c_xv * &x_v;
        self.x_design.transpose_vector_multiply(&weighted)
    }

    fn is_implicit(&self) -> bool {
        false
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Data structures
// ═══════════════════════════════════════════════════════════════════════════

/// Exact pseudo-logdeterminant log|S|₊ and its derivatives with respect to ρ.
///
/// # Exact pseudo-logdet on the positive eigenspace
///
/// For S(ρ) = Σ exp(ρ_k) S_k with S_k ⪰ 0, the nullspace
/// N(S) = ∩_k N(S_k) is structurally fixed (independent of ρ).
/// No eigenvalue of S crosses zero during optimization, so the
/// pseudo-logdet L = Σ_{σ_i > 0} log σ_i is C∞ in ρ.
///
/// ## Computation
///
/// Eigendecompose S, identify positive eigenvalues σ_i > ε (where ε is a
/// relative threshold for numerical zero detection), then:
///
///   L(S)     = Σ_{positive} log σ_i
///   ∂_k L    = tr(S⁺ A_k)            where A_k = λ_k S_k
///   ∂²_kl L  = δ_{kl} ∂_k L − tr(S⁺ A_l S⁺ A_k)
///
/// S⁺ is the Moore-Penrose pseudoinverse restricted to the positive
/// eigenspace. These are the exact derivatives of L — no δ-regularization,
/// no nullity metadata, no chain-rule inconsistencies.
#[derive(Clone, Debug)]
pub struct PenaltyLogdetDerivs {
    /// L(S) = log|S|₊ — the exact pseudo-logdeterminant on the positive eigenspace.
    ///
    /// L(S) = Σ_{σ_i > ε} log σ_i, where ε is a relative threshold that
    /// identifies the structural nullspace directly from the eigenspectrum.
    pub value: f64,
    /// ∂/∂ρₖ L(S) — first derivatives (one per smoothing parameter).
    ///
    /// ∂_k L = tr(S⁺ Aₖ) where Aₖ = λₖ Sₖ and S⁺ is the pseudoinverse
    /// restricted to the positive eigenspace.
    pub first: Array1<f64>,
    /// ∂²/(∂ρₖ∂ρₗ) L(S) — second derivatives (for outer Hessian).
    ///
    /// ∂²_kl L = δ_{kl} ∂_k L − λₖ λₗ tr(S⁺ Sₖ S⁺ Sₗ).
    pub second: Option<Array2<f64>>,
}


/// Unified representation of a single smoothing-parameter penalty coordinate.
///
/// A rho-coordinate always contributes
///
///   A_k = λ_k S_k,
///   S_k = R_k^T R_k.
///
/// For single-block/small problems it is fine to store the full-root `R_k`
/// in the joint basis. For exact-joint multi-block paths that scaling is
/// wasteful: the root is naturally block-local. This enum lets the unified
/// evaluator consume both forms through one interface.
#[derive(Clone, Debug)]
pub enum PenaltyCoordinate {
    DenseRoot(Array2<f64>),
    DenseRootCentered {
        root: Array2<f64>,
        prior_mean: Array1<f64>,
    },
    BlockRoot {
        root: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
    },
    BlockRootCentered {
        root: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
        prior_mean: Array1<f64>,
    },
    /// Kronecker-factored penalty coordinate for tensor-product smooths.
    ///
    /// In the reparameterized (eigenbasis) representation, the penalty
    /// `I ⊗ ... ⊗ S_k ⊗ ... ⊗ I` becomes `I ⊗ ... ⊗ Λ_k ⊗ ... ⊗ I`
    /// where `Λ_k = diag(μ_{k,0}, ..., μ_{k,q_k-1})`.  This is diagonal
    /// in each mode, so apply/quadratic/trace operations avoid O(p²).
    KroneckerMarginal {
        /// Marginal eigenvalues for ALL dimensions: `eigenvalues[j]` has length `q_j`.
        eigenvalues: Vec<Array1<f64>>,
        /// Which marginal dimension this penalty coordinate corresponds to.
        dim_index: usize,
        /// Marginal basis dimensions: `[q_0, ..., q_{d-1}]`.
        marginal_dims: Vec<usize>,
        /// Total joint dimension: `∏ q_j`.
        total_dim: usize,
    },
}


impl PenaltyCoordinate {
    pub fn from_dense_root(root: Array2<f64>) -> Self {
        Self::DenseRoot(root)
    }

    pub fn from_dense_root_with_mean(root: Array2<f64>, prior_mean: Array1<f64>) -> Self {
        assert_eq!(root.ncols(), prior_mean.len());
        if prior_mean.iter().all(|&value| value == 0.0) {
            Self::DenseRoot(root)
        } else {
            Self::DenseRootCentered { root, prior_mean }
        }
    }

    pub fn from_block_root(root: Array2<f64>, start: usize, end: usize, total_dim: usize) -> Self {
        assert_eq!(
            root.ncols(),
            end.saturating_sub(start),
            "block prior root column count must match block width"
        );
        assert!(
            end <= total_dim,
            "block prior root end exceeds total dimension: start={start}, end={end}, total_dim={total_dim}, root_dim={:?}",
            root.dim()
        );
        Self::BlockRoot {
            root,
            start,
            end,
            total_dim,
        }
    }

    pub fn from_block_root_with_mean(
        root: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
        prior_mean: Array1<f64>,
    ) -> Self {
        assert_eq!(
            root.ncols(),
            end.saturating_sub(start),
            "centered block prior root column count must match block width"
        );
        assert_eq!(
            prior_mean.len(),
            end.saturating_sub(start),
            "centered block prior mean length must match block width"
        );
        assert!(
            end <= total_dim,
            "centered block prior root end exceeds total dimension: start={start}, end={end}, total_dim={total_dim}, root_dim={:?}, prior_mean_len={}",
            root.dim(),
            prior_mean.len()
        );
        if prior_mean.iter().all(|&value| value == 0.0) {
            Self::from_block_root(root, start, end, total_dim)
        } else {
            Self::BlockRootCentered {
                root,
                start,
                end,
                total_dim,
                prior_mean,
            }
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::DenseRoot(root)
            | Self::DenseRootCentered { root, .. }
            | Self::BlockRoot { root, .. }
            | Self::BlockRootCentered { root, .. } => root.nrows(),
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                ..
            } => {
                // Rank = number of nonzero marginal eigenvalues for this dim,
                // times the product of all other dims.
                let nz = eigenvalues[*dim_index]
                    .iter()
                    .filter(|&&v| v.abs() > 1e-12)
                    .count();
                let other: usize = eigenvalues
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != *dim_index)
                    .map(|(_, e)| e.len())
                    .product::<usize>()
                    .max(1);
                nz * other
            }
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => root.ncols(),
            Self::BlockRoot { total_dim, .. }
            | Self::BlockRootCentered { total_dim, .. }
            | Self::KroneckerMarginal { total_dim, .. } => *total_dim,
        }
    }

    pub fn uses_operator_fast_path(&self) -> bool {
        matches!(
            self,
            Self::BlockRoot { .. }
                | Self::BlockRootCentered { .. }
                | Self::KroneckerMarginal { .. }
        )
    }

    /// Restrict this penalty coordinate onto the free subspace spanned by the
    /// orthonormal columns of `z` (shape `p × m`, `m ≤ p`, `zᵀz = I`).
    ///
    /// When a linear-inequality active set is non-empty, the inner solve and the
    /// penalized Hessian are reduced to the free subspace `β = z β_f` of
    /// dimension `m = p − active_set_size`. The penalty must move in lockstep:
    /// the quadratic `βᵀ S_k β = β_fᵀ (zᵀ S_k z) β_f`, and since `S_k = R_kᵀ R_k`
    /// the reduced root is `R_k z` (shape `rank_k × m`). For a block-local root
    /// `R_k` acting on `β[start..end]` the same identity gives reduced dense root
    /// `R_k · z[start..end, :]`, so the reduced coordinate is always a
    /// (dimension-`m`) `DenseRoot` / `DenseRootCentered` — the block structure
    /// does not survive an arbitrary subspace rotation. A centered mean `μ_k`
    /// maps to `zᵀ μ_k`, the representation of `μ_k` in the free subspace.
    ///
    /// This keeps `dim()` equal to the reduced `beta.len()`, which
    /// `InnerSolutionBuilder::build` asserts.
    pub fn project_into_subspace(&self, z: &Array2<f64>) -> Self {
        assert_eq!(
            z.nrows(),
            self.dim(),
            "PenaltyCoordinate::project_into_subspace: free-basis row count {} does not match coordinate dimension {}",
            z.nrows(),
            self.dim()
        );
        match self {
            Self::DenseRoot(root) => Self::DenseRoot(root.dot(z)),
            Self::DenseRootCentered { root, prior_mean } => {
                Self::from_dense_root_with_mean(root.dot(z), z.t().dot(prior_mean))
            }
            Self::BlockRoot {
                root, start, end, ..
            } => {
                let z_block = z.slice(ndarray::s![*start..*end, ..]);
                Self::DenseRoot(root.dot(&z_block))
            }
            Self::BlockRootCentered {
                root,
                start,
                end,
                prior_mean,
                ..
            } => {
                let z_block = z.slice(ndarray::s![*start..*end, ..]);
                // Reduced mean: the block-local prior `μ_k` sits at
                // `β[start..end]`; lift it into the full coordinate before
                // projecting so the free-space mean is `zᵀ (E_block μ_k)`.
                let z_block_owned = z_block.to_owned();
                Self::from_dense_root_with_mean(
                    root.dot(&z_block_owned),
                    z_block_owned.t().dot(prior_mean),
                )
            }
            Self::KroneckerMarginal { .. } => reml_contract_panic(
                "PenaltyCoordinate::project_into_subspace: Kronecker-factored \
                 coordinates do not co-occur with linear-inequality active sets \
                 (box/monotone constraints lower to dense/block roots)",
            ),
        }
    }

    fn apply_root(&self, beta: &Array1<f64>) -> Array1<f64> {
        assert_eq!(beta.len(), self.dim());
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => root.dot(beta),
            Self::BlockRoot {
                root, start, end, ..
            }
            | Self::BlockRootCentered {
                root, start, end, ..
            } => root.dot(&beta.slice(ndarray::s![*start..*end])),
            Self::KroneckerMarginal { .. } => {
                // No single root for Kronecker — use apply_penalty instead.
                // SAFETY: `has_root()` returns `false` for the
                // KroneckerMarginal variant (see the `matches!` block
                // above); callers of `apply_root` are required to gate on
                // `has_root()`, so reaching this arm means a caller
                // invoked the rooted-only API on a rootless variant.
                // SAFETY: KroneckerMarginal has no root; callers must gate on has_root() before apply_root.
                reml_contract_panic(
                    "apply_root not supported for KroneckerMarginal; use apply_penalty directly",
                );
            }
        }
    }

    pub fn apply_penalty(&self, beta: &Array1<f64>, scale: f64) -> Array1<f64> {
        assert_eq!(beta.len(), self.dim());
        let mut out = Array1::<f64>::zeros(self.dim());
        self.apply_penalty_view_into(beta.view(), scale, out.view_mut());
        out
    }

    pub fn apply_penalty_view_into(
        &self,
        beta: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(beta.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        out.fill(0.0);
        self.scaled_add_penalty_view(beta, scale, out);
    }

    pub fn scaled_add_penalty_view(
        &self,
        beta: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(beta.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        if scale == 0.0 {
            return;
        }
        match self {
            Self::DenseRoot(_)
            | Self::DenseRootCentered { .. }
            | Self::BlockRoot { .. }
            | Self::BlockRootCentered { .. } => match self {
                Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                    let mut root_beta = Array1::<f64>::zeros(root.nrows());
                    dense_matvec_into(root, beta, root_beta.view_mut());
                    dense_transpose_matvec_scaled_add_into(
                        root,
                        root_beta.view(),
                        scale,
                        out.view_mut(),
                    );
                }
                Self::BlockRoot {
                    root,
                    start,
                    end,
                    total_dim: _,
                }
                | Self::BlockRootCentered {
                    root,
                    start,
                    end,
                    total_dim: _,
                    ..
                } => {
                    let beta_block = beta.slice(ndarray::s![*start..*end]);
                    let mut root_beta = Array1::<f64>::zeros(root.nrows());
                    dense_matvec_into(root, beta_block, root_beta.view_mut());
                    let out_block = out.slice_mut(ndarray::s![*start..*end]);
                    dense_transpose_matvec_scaled_add_into(
                        root,
                        root_beta.view(),
                        scale,
                        out_block,
                    );
                }
                // Outer arm guarantees only the four root-bearing variants reach here.
                Self::KroneckerMarginal { .. } => {}
            },
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                total_dim,
            } => {
                // Apply (I ⊗ ... ⊗ Λ_k ⊗ ... ⊗ I) β via mode-k scaling.
                // In the eigenbasis, Λ_k is diagonal, so this is element-wise.
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let inner_size = stride_k;
                let eigs = &eigenvalues[k];
                assert_eq!(
                    outer_size * q_k * stride_k,
                    *total_dim,
                    "KroneckerMarginal dimension mismatch in apply"
                );

                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j] * scale;
                        if mu == 0.0 {
                            continue;
                        }
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..inner_size {
                            let idx = base + inner;
                            out[idx] += mu * beta[idx];
                        }
                    }
                }
            }
        }
    }

    pub fn quadratic(&self, beta: &Array1<f64>, scale: f64) -> f64 {
        match self {
            Self::DenseRoot(_)
            | Self::DenseRootCentered { .. }
            | Self::BlockRoot { .. }
            | Self::BlockRootCentered { .. } => {
                let root_beta = self.apply_root(beta);
                scale * root_beta.dot(&root_beta)
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                ..
            } => {
                // β' (I ⊗ ... ⊗ Λ_k ⊗ ... ⊗ I) β = Σ μ_{k,j} β[...]²
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let inner_size = stride_k;
                let eigs = &eigenvalues[k];

                let mut sum = 0.0;
                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j];
                        if mu == 0.0 {
                            continue;
                        }
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..inner_size {
                            let v = beta[base + inner];
                            sum += mu * v * v;
                        }
                    }
                }
                sum * scale
            }
        }
    }

    pub fn apply_shifted_penalty(&self, beta: &Array1<f64>, scale: f64) -> Array1<f64> {
        match self {
            Self::DenseRootCentered { root, prior_mean } => {
                let centered = beta - prior_mean;
                let root_beta = root.dot(&centered);
                let mut out = root.t().dot(&root_beta);
                out *= scale;
                out
            }
            Self::BlockRootCentered {
                root,
                start,
                end,
                total_dim,
                prior_mean,
            } => {
                let mut out = Array1::<f64>::zeros(*total_dim);
                let beta_block = beta.slice(ndarray::s![*start..*end]);
                let centered = beta_block.to_owned() - prior_mean;
                let root_beta = root.dot(&centered);
                let mut block = root.t().dot(&root_beta);
                block *= scale;
                out.slice_mut(ndarray::s![*start..*end]).assign(&block);
                out
            }
            _ => self.apply_penalty(beta, scale),
        }
    }

    pub fn shifted_quadratic(&self, beta: &Array1<f64>, scale: f64) -> f64 {
        match self {
            Self::DenseRootCentered { root, prior_mean } => {
                let centered = beta - prior_mean;
                let root_beta = root.dot(&centered);
                scale * root_beta.dot(&root_beta)
            }
            Self::BlockRootCentered {
                root,
                start,
                end,
                prior_mean,
                ..
            } => {
                let beta_block = beta.slice(ndarray::s![*start..*end]);
                let centered = beta_block.to_owned() - prior_mean;
                let root_beta = root.dot(&centered);
                scale * root_beta.dot(&root_beta)
            }
            _ => self.quadratic(beta, scale),
        }
    }

    pub fn scaled_dense_matrix(&self, scale: f64) -> Array2<f64> {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                let mut out = root.t().dot(root);
                out *= scale;
                out
            }
            Self::BlockRoot {
                root,
                start,
                end,
                total_dim,
            }
            | Self::BlockRootCentered {
                root,
                start,
                end,
                total_dim,
                ..
            } => {
                let mut out = Array2::<f64>::zeros((*total_dim, *total_dim));
                let mut block = root.t().dot(root);
                block *= scale;
                out.slice_mut(ndarray::s![*start..*end, *start..*end])
                    .assign(&block);
                out
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                total_dim,
            } => {
                // Materialize diagonal penalty in eigenbasis.
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let eigs = &eigenvalues[k];
                assert_eq!(
                    outer_size * q_k * stride_k,
                    *total_dim,
                    "KroneckerMarginal dimension mismatch in to_dense"
                );

                let mut out = Array2::<f64>::zeros((*total_dim, *total_dim));
                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j] * scale;
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..stride_k {
                            let idx = base + inner;
                            out[[idx, idx]] = mu;
                        }
                    }
                }
                out
            }
        }
    }

    /// Returns the block-local scaled penalty matrix (p_block × p_block) along
    /// with the embedding range, WITHOUT materializing into total_dim × total_dim.
    /// For DenseRoot (full-rank, no block structure), returns (matrix, 0, p).
    pub fn scaled_block_local(&self, scale: f64) -> (Array2<f64>, usize, usize) {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                let mut out = root.t().dot(root);
                out *= scale;
                let p = out.nrows();
                (out, 0, p)
            }
            Self::BlockRoot {
                root, start, end, ..
            }
            | Self::BlockRootCentered {
                root, start, end, ..
            } => {
                let mut block = root.t().dot(root);
                block *= scale;
                (block, *start, *end)
            }
            Self::KroneckerMarginal { total_dim, .. } => {
                // Fallback: materialize full matrix.
                let mat = self.scaled_dense_matrix(scale);
                (mat, 0, *total_dim)
            }
        }
    }

    /// Whether this coordinate has block structure (not full-rank dense).
    pub fn is_block_local(&self) -> bool {
        matches!(
            self,
            Self::BlockRoot { .. }
                | Self::BlockRootCentered { .. }
                | Self::KroneckerMarginal { .. }
        )
    }

    /// Apply λ_k S_k to a vector v without materializing the full matrix.
    /// For BlockRoot: extracts v[start..end], multiplies by local S_k, embeds result.
    pub fn scaled_matvec(&self, v: &Array1<f64>, scale: f64) -> Array1<f64> {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                let root_v = root.dot(v);
                let mut out = root.t().dot(&root_v);
                out *= scale;
                out
            }
            Self::BlockRoot {
                root, start, end, ..
            }
            | Self::BlockRootCentered {
                root, start, end, ..
            } => {
                let mut out = Array1::zeros(v.len());
                let v_block = v.slice(ndarray::s![*start..*end]);
                let root_v = root.dot(&v_block);
                let mut block_result = root.t().dot(&root_v);
                block_result *= scale;
                out.slice_mut(ndarray::s![*start..*end])
                    .assign(&block_result);
                out
            }
            Self::KroneckerMarginal { .. } => {
                // Reuse apply_penalty which handles mode-k contraction.
                self.apply_penalty(v, scale)
            }
        }
    }
}


// PenaltyLogdetEigenspace, build_penalty_logdet_eigenspace,
// scaled_penalty_logdet_nullspace_leakage, and frobenius_inner_same_shape
// have been replaced by the canonical PenaltyPseudologdet in
// super::penalty_logdet. All callers now use that module directly.

/// Reduced trace kernel `K = U · M · Uᵀ` for pseudo-logdet REML/LAML
/// criteria: an orthonormal column basis `u_s` (p × r) plus the r × r
/// symmetric reduced kernel `h_proj_inverse`, with `tr(K · A)` evaluated as
/// `tr(M · Uᵀ A U)` so contractions run on the r-dimensional subspace.
///
/// Two producers install it, with different (documented) exactness domains:
///
/// 1. **Intrinsic spectral form (#901, the GLM dense paths in runtime.rs —
///    `intrinsic_hessian_pseudo_logdet_parts`):** `u_s = U_H`, the kept
///    eigenvectors of the penalized Hessian `H_pen`, and `h_proj_inverse =
///    diag(1/σ_a)`. Then `K = H_pen⁺` exactly, and `tr(K · Ḣ)` is the exact
///    first derivative of the cost's `log|H_pen|₊` along **every** drift
///    direction — penalty-supported or not, moving-subspace ψ drifts
///    included — because on a constant-rank stratum first-order eigenvector
///    motion cancels out of the pseudo-logdet derivative. This object can be
///    traced against the GLM IFT correction `D_β H[v] = X' diag(c ⊙ X v) X`
///    (which leaks onto `null(S)` via the intercept column) without error.
///
/// 2. **Range(Sλ) Schur block (#752, `joint_penalty_subspace_trace_parts`
///    in custom_family.rs):** `u_s` spans `range(Sλ)` and `h_proj_inverse =
///    U_Sᵀ (H+Sλ)⁺ U_S`. For penalty-supported `A` (`A = ∂Sλ/∂ρ`), the
///    identity `U_S U_Sᵀ A U_S U_Sᵀ = A` gives `tr(K · A) = tr((H+Sλ)⁺ A) =
///    d log|H+Sλ|₊/dρ` — exact for the ρ family. It is **not** exact for
///    drifts with `null(Sλ)` support (GLM cubic corrections, ψ basis
///    drifts); paths that carry such drifts must install form 1.
///
/// Historically this struct carried a third reading — `(U_Sᵀ H U_S)⁻¹`, the
/// plain projected inverse paired with the projected cost `log|U_Sᵀ H U_S|₊`.
/// That object is WRONG as a REML determinant term: splitting `H` over
/// `range(S) ⊕ ker(S)` as `[[A,B],[Bᵀ,C]]`, the projected logdet is
/// `log det A`, dropping the θ-dependent Schur curvature
/// `log det(C − BᵀA⁻¹B)` of the likelihood-identified, penalty-null block
/// (sign-flipped ρ-gradients, ~1e5 ψ blow-ups vs FD — #901). No producer
/// builds it anymore.
#[derive(Clone, Debug)]
pub struct PenaltySubspaceTrace {
    pub u_s: Array2<f64>,
    pub h_proj_inverse: Array2<f64>,
}


impl PenaltySubspaceTrace {
    /// Compute `tr(K · A)` where `K = U_S · h_proj_inverse · U_Sᵀ` — the
    /// pseudo-logdet trace kernel (see the struct doc for the two producer
    /// forms and their exactness domains).
    ///
    /// Uses the identity `tr(K · A) = tr(h_proj_inverse · U_Sᵀ A U_S)` so the
    /// reduction runs on the r × r subspace rather than materializing K.
    pub fn trace_projected_logdet(&self, a: &Array2<f64>) -> f64 {
        crate::construction::trace_penalty_covariance_in_orthogonal_basis(
            a,
            &self.u_s,
            &self.h_proj_inverse,
        )
    }

    /// Reduce a p × p matrix `A` to its r × r projection `U_Sᵀ · A · U_S`.
    ///
    /// Exposed so callers that need the same reduced matrix for both the
    /// single-trace `tr(K · A)` and the cross-trace `tr(K · A · K · B)`
    /// can avoid repeating the p × p · p × r matmuls.  Routes through
    /// faer's parallel SIMD GEMM (`fast_atb` / `fast_ab`) so the p-large
    /// contraction axis amortizes across all cores.
    pub fn reduce(&self, a: &Array2<f64>) -> Array2<f64> {
        let u_s_t_a = crate::faer_ndarray::fast_atb(&self.u_s, a);
        crate::faer_ndarray::fast_ab(&u_s_t_a, &self.u_s)
    }

    /// Compute `tr(H_proj⁻¹ · R)` given an already-reduced `R = U_Sᵀ A U_S`.
    pub fn trace_projected_logdet_reduced(&self, r_mat: &Array2<f64>) -> f64 {
        crate::construction::trace_reduced_penalty_covariance(r_mat, &self.h_proj_inverse)
    }

    /// Cross-trace given pre-reduced blocks `R_A = U_Sᵀ A U_S`, `R_B = U_Sᵀ B U_S`.
    pub fn trace_projected_logdet_cross_reduced(&self, ra: &Array2<f64>, rb: &Array2<f64>) -> f64 {
        // left = H_proj⁻¹ · R_A ;  right = H_proj⁻¹ · R_B ;  tr(left · right).
        let left = self.h_proj_inverse.dot(ra);
        let right = self.h_proj_inverse.dot(rb);
        trace_matrix_product(&left, &right)
    }

    /// Reduce a `HyperOperator` `A` to its `r × r` projection
    /// `U_Sᵀ · A · U_S` without materializing the dense `p × p` block.
    /// Uses `A.mul_mat(U_S)` so an Hv-only operator is probed in `r` matvecs
    /// (each `O(work_of_A)`), then a single `r × p × r` reduction routed
    /// through faer's parallel SIMD GEMM (`fast_atb`).
    pub fn reduce_operator<O>(&self, a: &O) -> Array2<f64>
    where
        O: HyperOperator + ?Sized,
    {
        let au = a.mul_mat(&self.u_s);
        crate::faer_ndarray::fast_atb(&self.u_s, &au)
    }

    /// `tr(K · A)` for `A` exposed only as a `HyperOperator`.  Mirrors
    /// [`Self::trace_projected_logdet`] without forcing dense materialization
    /// of `A`.
    pub fn trace_operator<O>(&self, a: &O) -> f64
    where
        O: HyperOperator + ?Sized,
    {
        self.trace_projected_logdet_reduced(&self.reduce_operator(a))
    }

    /// Projected leverage `h^{G,proj}_i = Xᵢᵀ · K · Xᵢ` for every row of `x`.
    ///
    /// Computed in bulk as `Z = X · U_S` (`n × r`) then
    /// `h^{G,proj}_i = (Z H_proj⁻¹ Zᵀ)_{ii} = Σ_{a,b} Z_{ia} (H_proj⁻¹)_{ab} Z_{ib}`,
    /// total cost `O(n · p · r + n · r²)` — strictly cheaper than `n` calls
    /// to [`Self::apply`] because the `n × p · p × r` GEMM streams the
    /// `p`-axis once.  Streams `X` through `try_row_chunk` so operator-backed
    /// (Lazy) designs at large scale never densify the full `(n × p)` block.
    pub fn xt_projected_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let r = self.u_s.ncols();
        assert_eq!(self.u_s.nrows(), p);
        assert_eq!(self.h_proj_inverse.nrows(), r);
        assert_eq!(self.h_proj_inverse.ncols(), r);

        let block = {
            const TARGET_CHUNK_FLOATS: usize = 1 << 16;
            (TARGET_CHUNK_FLOATS / p.max(1)).clamp(1, n.max(1))
        };

        let mut h = Array1::<f64>::zeros(n);
        let mut start = 0usize;
        while start < n {
            let end = (start + block).min(n);
            let rows = x.try_row_chunk(start..end).unwrap_or_else(|err| {
                // SAFETY: `start..end` is constructed from
                // `0..n = 0..x.nrows()` with `end = (start+block).min(n)`,
                // so it is always a valid sub-range of `x`. Failure means
                // the operator broke its row-chunk contract.
                // SAFETY: row range built from 0..x.nrows(); failure means operator broke its contract.
                reml_contract_panic(format!(
                    "xt_projected_kernel_x_diagonal: row chunk failed: {err}"
                ))
            });
            // Z_chunk = rows · U_S  ((end-start) × r).
            let z_chunk = crate::faer_ndarray::fast_ab(&rows, &self.u_s);
            // h_i = Σ_{a,b} Z_{ia} (H_proj⁻¹)_{ab} Z_{ib}.
            for (i, row_z) in z_chunk.outer_iter().enumerate() {
                let mut acc = 0.0;
                for (z_a, h_row) in row_z
                    .iter()
                    .copied()
                    .zip(self.h_proj_inverse.rows().into_iter())
                {
                    let mut inner = 0.0;
                    for (h_value, z_b) in h_row.iter().copied().zip(row_z.iter().copied()) {
                        inner += h_value * z_b;
                    }
                    acc += z_a * inner;
                }
                h[start + i] = acc;
            }
            start = end;
        }
        h
    }

    /// Projected bilinear pseudo-inverse `aᵀ · K⁺ · b` where
    /// `K⁺ = U_S · H_proj⁻¹ · U_Sᵀ`.
    ///
    /// Used by the rank-deficient LAML IFT correction path: when `b ∈
    /// col(S_k) ⊂ range(S_+)`, applying the projected pseudo-inverse
    /// instead of the full `H⁻¹` strips spurious null-space noise from
    /// `a` (≈ the outer-stationarity residual `r`) before the inverse,
    /// without biasing the numerator. Costs `O(p·r + r²)` versus the
    /// `O(p²·r)` full solve.
    pub fn bilinear_pseudo_inverse(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let proj_a = crate::faer_ndarray::fast_atv(&self.u_s, a);
        let proj_b = crate::faer_ndarray::fast_atv(&self.u_s, b);
        let h_proj_inv_b = self.h_proj_inverse.dot(&proj_b);
        proj_a.dot(&h_proj_inv_b)
    }

    /// Euclidean projection onto the retained penalty/Hessian range used by
    /// this projected kernel: `P_S a = U_S U_Sᵀ a`.
    pub fn project_onto_subspace(&self, a: &Array1<f64>) -> Array1<f64> {
        let proj_a = crate::faer_ndarray::fast_atv(&self.u_s, a);
        crate::faer_ndarray::fast_av(&self.u_s, &proj_a)
    }

    /// Apply the projected pseudo-inverse `K = U_S · H_proj⁻¹ · U_Sᵀ` to a
    /// vector `a`, returning the minimum-norm solution `v = K · a` of the
    /// system `H v = a` restricted to `range(S₊)`.
    ///
    /// This is the correct stand-in for `H⁻¹ · a` in all per-coordinate
    /// outer-gradient/Hessian formulas when the rank-deficient LAML fix is
    /// active (`penalty_subspace_trace = Some`). The full `H⁻¹ · a` solve
    /// amplifies any component of `a` outside `range(H_free)` by
    /// `1/σ_min(H_active_normal)` — which on large-scale survival
    /// marginal-slope is ~10¹² and propagates into outer gradients of
    /// magnitude 10¹⁴, suppressed by the envelope tripwire downstream and
    /// killing every seed before the fit can take a step. This operator may
    /// only drop components that the inner KKT certificate has already made
    /// negligible; `ProjectedKktResidual::projected_into_reduced_range` enforces
    /// that contract before the IFT correction uses this pseudo-inverse. With
    /// that guard, the returned gradient lives on the constrained manifold,
    /// matching the projected `log|U_Sᵀ H U_S|` term.
    ///
    /// Costs `O(p·r + r²)` for the two `U_S`-contractions plus the `r × r`
    /// solve — strictly cheaper than the `O(p²)` full `hop.solve_multi`
    /// when `r ≪ p`, and bounded regardless of `σ_min(H)`.
    pub fn apply_pseudo_inverse(&self, a: &Array1<f64>) -> Array1<f64> {
        // The one sensitivity operator (#935): the projected inverse action
        // `U_S · H_proj⁻¹ · U_Sᵀ · a` has a single spelling, shared with every
        // other consumer of `FittedInverse::Projected`.
        self.sensitivity().apply(a)
    }

    /// View this projected trace kernel as the unified [`FitSensitivity`]
    /// (#935) over the rank-deficient LAML convention `K = U_S · H_proj⁻¹ ·
    /// U_Sᵀ`. The trace machinery stays here; the *inverse action* is the
    /// shared operator, so no site can disagree about what `H⁻¹` means.
    pub fn sensitivity(&self) -> crate::solver::sensitivity::FitSensitivity<'_> {
        crate::solver::sensitivity::FitSensitivity::from_projected(&self.u_s, &self.h_proj_inverse)
    }

    /// Build the **constrained pseudo-inverse kernel**
    /// `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`
    /// from this penalty-projected kernel `K_S` and the *active* row block
    /// `A_act` of the joint linear inequality constraint matrix.
    ///
    /// `K_T` is the **Moore-Penrose pseudo-inverse of `H` restricted to
    /// `T = range(S₊) ∩ ker(A_act)`** — the smooth manifold the inner
    /// solver actually moves on at a constrained-stationary point. It is
    /// exactly the kernel that solves the per-coordinate saddle-point
    /// IFT system
    ///
    /// ```text
    ///   [ H   Aᵀ_act ] [ ∂β/∂ρ_k ]   [ −a_k ]
    ///   [ A_act  0   ] [ ∂λ/∂ρ_k ] = [   0  ]
    /// ```
    ///
    /// with `∂β/∂ρ_k = −K_T · a_k`. Using `K_T` for the per-coordinate
    /// mode response `v_k` makes the outer gradient the *exact* derivative
    /// of the projected Laplace cost `log|U_Tᵀ H U_T|`, where `U_T` is an
    /// orthonormal basis of `T` — the marginal-likelihood determinant the
    /// inner is actually drawing on.
    ///
    /// Returns a [`ConstrainedSubspaceKernel`] handle that caches the
    /// small `k_active × k_active` Schur complement so subsequent
    /// `apply_pseudo_inverse` calls for different RHS reuse it. When the
    /// active set is empty the handle degrades to a pass-through over
    /// `self` (no extra work).
    ///
    /// Total precompute cost: `k_active` calls to
    /// [`Self::apply_pseudo_inverse`] (one per active row) plus a
    /// `k_active × k_active` Cholesky/QR. Per-vector `apply` cost: one
    /// `K_S` apply + one `k_active × p` matvec + one small triangular
    /// solve + one `p × k_active` matvec.
    pub fn with_active_constraints<'a>(
        &'a self,
        a_act: ndarray::ArrayView2<'a, f64>,
    ) -> ConstrainedSubspaceKernel<'a> {
        let k_active = a_act.nrows();
        if k_active == 0 {
            return ConstrainedSubspaceKernel {
                kernel: self,
                z: Array2::zeros((0, self.u_s.nrows())),
                a_act,
                m_inv: Array2::zeros((0, 0)),
                k_active: 0,
            };
        }
        // Z = K_S · Aᵀ_act,  shape (p × k_active).
        let p = self.u_s.nrows();
        let mut z = Array2::<f64>::zeros((p, k_active));
        for j in 0..k_active {
            let a_row = a_act.row(j).to_owned();
            let k_s_a_row = self.apply_pseudo_inverse(&a_row);
            z.column_mut(j).assign(&k_s_a_row);
        }
        // M = A_act · Z   (shape k_active × k_active, symmetric PSD on
        // range(K_S) ∩ image(A_actᵀ); on a rank-deficient overlap we
        // add a tiny diagonal regulariser so the inversion remains
        // bounded — same noise-floor strategy as elsewhere in this
        // module).
        let mut m = a_act.dot(&z);
        // Symmetrise (numerical noise from the matmul leaves small skew).
        for i in 0..k_active {
            for j in 0..i {
                let avg = 0.5 * (m[[i, j]] + m[[j, i]]);
                m[[i, j]] = avg;
                m[[j, i]] = avg;
            }
        }
        // Eigendecomposition-based Moore-Penrose pseudo-inverse with a
        // relative spectral cutoff. This is the principled treatment of
        // rank deficiency in `A_act` when restricted to `range(S₊)`:
        // some active constraint rows may be linearly dependent after
        // projection (e.g. several monotonicity rows pinning the same
        // flat region all reduce to the same row in `range(S₊)`).
        // A plain `M⁻¹` then amplifies near-null directions; the
        // pseudo-inverse drops them at a relative threshold
        // `tol = eps · k_active · σ_max(M)`, which is the standard
        // NumPy/LAPACK convention and exactly what Codex flagged as
        // necessary in the math review.
        let (evals, evecs) = m
            .eigh(faer::Side::Lower)
            .unwrap_or_else(|_| (Array1::<f64>::zeros(k_active), Array2::<f64>::eye(k_active)));
        let sigma_max = evals.iter().copied().fold(0.0_f64, f64::max).max(0.0);
        let tol = f64::EPSILON * (k_active as f64) * sigma_max.max(1.0);
        let mut m_inv = Array2::<f64>::zeros((k_active, k_active));
        let mut dropped = 0usize;
        for q in 0..k_active {
            if evals[q] > tol {
                let inv_sigma = 1.0 / evals[q];
                // Outer product u_q u_qᵀ scaled by 1/σ_q.
                for i in 0..k_active {
                    for j in 0..k_active {
                        m_inv[[i, j]] += inv_sigma * evecs[[i, q]] * evecs[[j, q]];
                    }
                }
            } else {
                dropped += 1;
            }
        }
        if dropped > 0 {
            log::debug!(
                "[constrained-subspace kernel] dropped {} of {} active-constraint directions \
                 (rank-deficient on range(S₊)); pseudo-inverse threshold = {:.3e}",
                dropped,
                k_active,
                tol,
            );
        }
        ConstrainedSubspaceKernel {
            kernel: self,
            z,
            a_act,
            m_inv,
            k_active,
        }
    }
}


/// Per-evaluation handle that combines a penalty-projected
/// [`PenaltySubspaceTrace`] with an active inequality-constraint block,
/// producing the constraint-aware pseudo-inverse
/// `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`. See
/// [`PenaltySubspaceTrace::with_active_constraints`] for the math.
///
/// Caches the small `k_active × k_active` Schur inverse so subsequent
/// per-coordinate `apply` calls only do `O(p · k_active)` work each.
pub struct ConstrainedSubspaceKernel<'a> {
    kernel: &'a PenaltySubspaceTrace,
    /// `Z = K_S · Aᵀ_act`, shape `(p × k_active)`.
    z: Array2<f64>,
    /// Active-row block of the joint constraint matrix.
    a_act: ndarray::ArrayView2<'a, f64>,
    /// `(A_act · K_S · Aᵀ_act)⁻¹`, shape `(k_active × k_active)`.
    m_inv: Array2<f64>,
    k_active: usize,
}


impl<'a> ConstrainedSubspaceKernel<'a> {
    /// Apply `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S` to `a`. The result
    /// lies in `range(S₊) ∩ ker(A_act)` — the smooth manifold the inner
    /// solver actually moves on at a constrained-stationary point.
    pub fn apply_pseudo_inverse(&self, a: &Array1<f64>) -> Array1<f64> {
        let v_s = self.kernel.apply_pseudo_inverse(a);
        if self.k_active == 0 {
            return v_s;
        }
        // mu = M_inv · (A_act · v_s)
        let t = self.a_act.dot(&v_s);
        let mu = self.m_inv.dot(&t);
        // v = v_s - Z · mu
        let correction = self.z.dot(&mu);
        v_s - &correction
    }

    /// Whether any active constraints contribute (when false this kernel
    /// is identical to the bare [`PenaltySubspaceTrace::apply_pseudo_inverse`]).
    pub fn has_active_constraints(&self) -> bool {
        self.k_active > 0
    }
}


/// Tangency self-audit gate for the constrained mode-response arm: the
/// emitted `v = K_T · rhs` must lie in `ker(A_act)` by construction, so
/// `|A_act · v|` is compared against this fraction of the cancellation
/// scale `|A_act| · |v|` (per active row). Generous enough that legitimate
/// rank-deficient active sets (whose dropped Schur directions leave
/// ε-level residue, see [`PenaltySubspaceTrace::with_active_constraints`])
/// never trip it; the historical failure mode it guards (the d6b17a7f
/// `1/σ_min ≈ 10¹²` null-space amplification) exceeds it by six orders.
const THETA_MODE_RESPONSE_TANGENCY_GATE: f64 = 1e-6;


/// #931 migration pass 2 — the ThetaDirection shared-drift pass: the ONE
/// per-evaluation selection of the IFT mode-response kernel behind every
/// `dβ̂/dθ = −K · ∂g/∂θ` solve in the outer gradient/Hessian assembly.
///
/// Before this object existed, four sites (the gradient solve stack in
/// `reml_laml_evaluate`, the ρ- and ext-coordinate standalone fallbacks in
/// `compute_outer_hessian`, and the standalone fallback in
/// `build_outer_hessian_operator`) each re-implemented the same selection
/// rule by hand, with comments warning each other to "mirror the
/// selection exactly, otherwise the operator-form Hessian and dense
/// materialization disagree on every entry". A hand-copied convention every
/// caller must remember is precisely the objective↔gradient desync surface
/// (#748/#752/#901 class) the criterion-as-atoms architecture (#931)
/// removes. Now the rule is DECIDED in exactly one constructor and every
/// consumer is a contraction of the same kernel object — the gradient and
/// both Hessian representations structurally cannot pick different
/// inverses for the same evaluation point:
///
///   * Active inequality constraints recorded on the inner solution → the
///     lifted constrained kernel
///     `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`. The inner SCOP solver
///     clamps β̂(θ) onto `T = range(S₊) ∩ ker(A_act)`, so the true IFT
///     derivative lives in T and the lifted kernel gives the minimum-norm
///     solution there; the full solve would amplify any RHS component
///     outside `range(H_free)` by `1/σ_min(H_active_normal)` — ~10¹² on
///     large-scale survival marginal-slope (commit d6b17a7f).
///   * Otherwise → the FULL Hessian solve `v = H⁻¹ · rhs`, even when the
///     LAML cost surface uses the projected logdet `½ log|U_Sᵀ H U_S|`:
///     the inner solver converges β̂ ∈ R^p in the unconstrained full
///     space, so the IFT identity demands the full inverse, and the
///     penalty-subspace projection acts on the TRACE contraction side
///     only. Routing through bare `K_S` here would discard the
///     `null(S₊)` component of dβ̂/dθ — the near-separable ψ-gradient
///     blow-up pinned by `duchon_probit_per_row_dnu_dpsi_fd_vs_analytic`.
///
/// The two emission shapes (`respond_one` per-vector, `respond_stack`
/// batched) exist because the call sites have different RHS layouts and
/// their solve shapes must stay bit-identical to the pre-port assembly
/// (per-column GEMV vs blocked GEMM sum in different orders) — NOT because
/// a site may choose a different kernel. Both shapes dispatch on the same
/// stored decision.
///
/// This is the `Sensitivity`-operator half of the `ThetaDirection`
/// calculus sketched in `atoms.rs`: the direction's `β̇` channel is a
/// contraction of this kernel, so atoms borrowing the shared drift can no
/// longer see a different chain rule than their neighbors.
pub(crate) struct ThetaModeResponseKernel<'s> {
    hop: &'s dyn HessianOperator,
    /// `Some` exactly when the selection rule chose the lifted constrained
    /// kernel. Built once per evaluation point (one Schur-complement
    /// factorization), shared by every gradient/Hessian consumer — the
    /// pre-port code rebuilt it per consumer site.
    constrained: Option<ConstrainedSubspaceKernel<'s>>,
}


impl<'s> ThetaModeResponseKernel<'s> {
    /// The ONE place the mode-response kernel selection rule lives.
    pub(crate) fn select(
        subspace: Option<&'s PenaltySubspaceTrace>,
        active_constraints: Option<&'s ActiveLinearConstraintBlock>,
        hop: &'s dyn HessianOperator,
    ) -> Self {
        let constrained = match (subspace, active_constraints) {
            (Some(kernel), Some(block)) => {
                let ck = kernel.with_active_constraints(block.a.view());
                ck.has_active_constraints().then_some(ck)
            }
            _ => None,
        };
        Self { hop, constrained }
    }

    /// Mode response for one right-hand side: `K_T · rhs` under active
    /// constraints, `H⁻¹ · rhs` (single-RHS `solve`) otherwise. Used by the
    /// per-coordinate fallbacks whose pre-port assembly solved one vector at
    /// a time — the single-RHS shape is preserved bit-identically.
    pub(crate) fn respond_one(&self, rhs: &Array1<f64>) -> Array1<f64> {
        match self.constrained.as_ref() {
            Some(ck) => {
                let v = ck.apply_pseudo_inverse(rhs);
                self.certify_tangency(ck, &v);
                v
            }
            None => self.hop.solve(rhs),
        }
    }

    /// Mode responses for a column-stacked RHS block: per-column `K_T`
    /// applies under active constraints (the lifted kernel has no blocked
    /// form), one batched `solve_multi` otherwise (BLAS-3 / GPU batched
    /// route) — exactly the shapes the stacked call sites used pre-port.
    /// Zero RHS columns (box-masked ρ coordinates) emit exact zeros through
    /// either arm, since both kernels are linear.
    pub(crate) fn respond_stack(&self, rhs_stack: &Array2<f64>) -> Array2<f64> {
        match self.constrained.as_ref() {
            Some(ck) => {
                let mut out = Array2::<f64>::zeros(rhs_stack.raw_dim());
                for (j, col) in rhs_stack.columns().into_iter().enumerate() {
                    let v = ck.apply_pseudo_inverse(&col.to_owned());
                    self.certify_tangency(ck, &v);
                    out.column_mut(j).assign(&v);
                }
                out
            }
            None => self.hop.solve_multi(rhs_stack),
        }
    }

    /// Per-atom certify body (#934 FD-self-audit pattern, applied as an
    /// exact structural invariant): every constrained emission must lie in
    /// `ker(A_act)` — `A_act · v = 0` is the defining property of `K_T`'s
    /// range, so a violation can only mean the kernel object and the
    /// emission desynced. Checked on every constrained response (cost
    /// `O(k_active · p)`, negligible next to the apply itself) against the
    /// row-wise cancellation scale `|A_act| · |v|`; a violation does not
    /// fail the fit — it names the atom loudly in the `[CERTIFICATE]`
    /// stream, exactly like the outer-optimum criterion audit. The
    /// unconstrained arm carries no separate certify: its coherence with
    /// the criterion VALUE is audited end-to-end by the #934
    /// `CriterionCertificate` at every returned optimum.
    fn certify_tangency(&self, ck: &ConstrainedSubspaceKernel<'_>, v: &Array1<f64>) {
        let residual = ck.a_act.dot(v);
        for (row, r) in residual.iter().enumerate() {
            let scale: f64 = ck
                .a_act
                .row(row)
                .iter()
                .zip(v.iter())
                .map(|(a, x)| (a * x).abs())
                .sum();
            if r.abs() > THETA_MODE_RESPONSE_TANGENCY_GATE * (scale + f64::EPSILON) {
                log::warn!(
                    "[CERTIFICATE warning] atom \"theta_mode_response\": constrained IFT \
                     mode response left ker(A_act) — active row {row} residual {:.3e} \
                     exceeds gate {:.1e}·{:.3e}; the lifted kernel K_T and its emission \
                     have desynced (#931 pass-2 invariant)",
                    r.abs(),
                    THETA_MODE_RESPONSE_TANGENCY_GATE,
                    scale,
                );
            }
        }
    }
}


/// Subspace represented by a stored KKT residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KktResidualSubspace {
    /// Residual after active-constraint normal components have been stripped:
    /// `r_A = P_T(Sβ + Γβ - ∇ℓ)`.
    ActiveProjected,
    /// Residual additionally projected into the retained identifiable range:
    /// `r_R = R Rᵀ r_A`.
    ReducedRange,
}


/// KKT residual `r = ∇_β L_pen(β̂)` at the converged inner iterate, with its
/// exact represented subspace tagged.
///
/// The IFT correction `−½ rᵀ H⁻¹ r` in `reml_laml_evaluate` requires `r` to
/// be in the same reduced range as the inverse kernel. An active-projected
/// residual is sufficient for the full-H path when the Hessian is nonsingular.
/// When the projected pseudo-inverse path is active, the evaluator converts it
/// to `ReducedRange` before assembling the IFT cost/gradient/Hessian.
///
/// This newtype lifts the projection contract into the type system: a value
/// of this type can only be produced by projection-aware constructors, and the
/// stored subspace says whether the residual is merely active-projected or has
/// also been reduced into the identifiable range.
#[derive(Clone, Debug)]
pub struct ProjectedKktResidual {
    /// The residual vector in the full coefficient coordinates. Active and
    /// reduced-range projection zero out excluded directions rather than
    /// shortening the vector, so its length remains `p`.
    residual: Array1<f64>,
    subspace: KktResidualSubspace,
    /// The KKT-stationarity tolerance the inner solver compared the
    /// residual against when the certificate fired. `None` for legacy
    /// construction sites that haven't been threaded yet; downstream
    /// consumers fall back to `f64::NAN` in that case.
    residual_tol: Option<f64>,
    /// `total_p - active_set_size` at the producing iterate. Records
    /// the dimensionality of the subspace on which the residual is
    /// stationary, which the outer optimiser uses when scoring the
    /// joint-Newton certificate's strength.
    free_rank: Option<usize>,
}


impl ProjectedKktResidual {
    /// Construct from `r_A = P_T(Sβ + Γβ - ∇ℓ)`, with active constraint
    /// multipliers removed but before any reduced-range projection.
    pub(crate) fn from_active_projected(residual: Array1<f64>) -> Self {
        Self {
            residual,
            subspace: KktResidualSubspace::ActiveProjected,
            residual_tol: None,
            free_rank: None,
        }
    }

    /// Construct from `r_R = R Rᵀ r_A`, where `R` is the actual reduced
    /// identifiable basis used by the projected inverse kernel.
    ///
    /// This is deliberately private: callers must start with an
    /// active-projected residual and go through `projected_into_reduced_range`
    /// so the dropped null/range-excluded component is checked against the
    /// producing inner KKT tolerance.
    fn from_reduced_range(residual: Array1<f64>) -> Self {
        Self {
            residual,
            subspace: KktResidualSubspace::ReducedRange,
            residual_tol: None,
            free_rank: None,
        }
    }

    /// Attach the KKT tolerance and free-subspace rank to a previously
    /// constructed residual. Builder-style so the construction path
    /// (`from_active_projected` then `with_metadata`) reads as a single
    /// inline expression at the call site.
    pub(crate) fn with_metadata(mut self, residual_tol: f64, free_rank: usize) -> Self {
        self.residual_tol = Some(residual_tol);
        self.free_rank = Some(free_rank);
        self
    }

    /// Borrow the underlying free-space residual for the H⁻¹·r solve and
    /// its ρ-derivatives.
    pub fn as_array(&self) -> &Array1<f64> {
        &self.residual
    }

    pub fn subspace(&self) -> KktResidualSubspace {
        self.subspace
    }

    fn projected_into_reduced_range(&self, kernel: &PenaltySubspaceTrace) -> Result<Self, String> {
        match self.subspace {
            KktResidualSubspace::ReducedRange => Ok(self.clone()),
            KktResidualSubspace::ActiveProjected => {
                let reduced_residual = kernel.project_onto_subspace(&self.residual);
                let dropped_inf = self
                    .residual
                    .iter()
                    .zip(reduced_residual.iter())
                    .map(|(full, reduced)| (full - reduced).abs())
                    .fold(0.0_f64, f64::max);
                let residual_inf = self
                    .residual
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max);
                // Default mixed absolute/relative tolerance for the dropped-mass
                // gate when the caller supplies no explicit `residual_tol`:
                // ~1e-10 scaled by `1 + ‖r‖∞` so it degrades gracefully with the
                // residual magnitude.
                const DEFAULT_KKT_RESIDUAL_REL_TOL: f64 = 1e-10;
                let tol = self
                    .residual_tol
                    .unwrap_or_else(|| DEFAULT_KKT_RESIDUAL_REL_TOL * (1.0 + residual_inf));
                let gate = tol;
                if dropped_inf > gate {
                    return Err(format!(
                        "projected KKT residual contains unresolved mass outside the reduced \
                         Hessian/penalty range: |r_A - r_R|∞={dropped_inf:.3e} > tol={gate:.3e}; \
                         range-projected IFT correction is valid only after the null direction is \
                         explicitly removed/fixed or after the active-projected residual is small"
                    ));
                }
                let mut reduced = Self::from_reduced_range(reduced_residual);
                reduced.residual_tol = self.residual_tol;
                reduced.free_rank = self.free_rank;
                Ok(reduced)
            }
        }
    }

    /// The KKT-stationarity tolerance the inner solver applied at the
    /// producing iterate. Returns `None` when the residual was built
    /// from a legacy site that hasn't been threaded yet; downstream
    /// consumers should substitute `f64::NAN` in that case.
    pub fn residual_tol(&self) -> Option<f64> {
        self.residual_tol
    }

    /// Dimensionality of the free subspace: `total_p - active_set_size`
    /// at the producing iterate. `None` from legacy construction sites.
    pub fn free_rank(&self) -> Option<usize> {
        self.free_rank
    }
}


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
    pub hessian_op: Arc<dyn HessianOperator>,

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
    /// Firth-only frozen-curvature Tierney-Kadane surrogate correction.
    /// Standard non-Firth LAML leaves this at zero so the production objective
    /// stays paired with the exact analytic unified derivatives.
    pub tk_correction: f64,

    /// Gradient of the Firth-only frozen-curvature TK surrogate with respect
    /// to active outer coordinates.
    pub tk_gradient: Option<Array1<f64>>,

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
    /// See `survival_location_scale::exact_newton_outer_curvature` for the
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
    pub rho_prior: crate::types::RhoPrior,

    // === Model dimensions ===
    /// Number of observations.
    pub n_observations: usize,

    /// M_p: dimension of the penalty null space (unpenalized coefficients).
    pub nullspace_dim: f64,

    /// ½·Σᵢ log(wᵢ) — half the sum of log prior weights.
    ///
    /// This is the per-observation Gaussian normalization constant that the
    /// `log_likelihood` (computed by
    /// [`calculate_loglikelihood_omitting_constants`]) deliberately drops. The
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
    pub ext_coord_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,

    /// Callback for ρ × ext cross pairs: (rho_index, ext_index) → HyperCoordPair.
    pub rho_ext_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,

    /// M_i[u] = D_β B_i[u] callback for extended coordinates.
    /// Arguments: (ext_index, direction) → correction matrix.
    pub fixed_drift_deriv: Option<FixedDriftDerivFn>,

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


/// Active row block of the joint linear inequality constraint matrix at the
/// converged inner iterate. Carries the dense rows needed for the
/// constraint-aware pseudo-inverse `K_T` in
/// [`PenaltySubspaceTrace::with_active_constraints`]. Only the `A` rows are
/// needed by the kernel itself; if a future audit needs the RHS values, add
/// them back as a typed field then.
#[derive(Clone, Debug)]
pub struct ActiveLinearConstraintBlock {
    /// `k_active × p` matrix of active constraint rows.
    pub a: Array2<f64>,
}


/// Builder for `InnerSolution` that provides sensible defaults and
/// auto-computes derived quantities (nullspace_dim).
pub struct InnerSolutionBuilder<'dp> {
    // Required fields
    log_likelihood: f64,
    penalty_quadratic: f64,
    hessian_op: Arc<dyn HessianOperator>,
    beta: Array1<f64>,
    penalty_coords: Vec<PenaltyCoordinate>,
    penalty_logdet: PenaltyLogdetDerivs,
    n_observations: usize,
    dispersion: DispersionHandling,
    // Optional fields with defaults
    deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,
    tk_correction: f64,
    tk_gradient: Option<Array1<f64>>,
    firth: Option<ExactJeffreysTerm>,
    hessian_logdet_correction: f64,
    penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,
    rho_curvature_scale: f64,
    rho_prior: crate::types::RhoPrior,
    nullspace_dim_override: Option<f64>,
    // Extended hyperparameter coordinates
    ext_coords: Vec<HyperCoord>,
    ext_coord_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    rho_ext_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    fixed_drift_deriv: Option<FixedDriftDerivFn>,
    contracted_psi_second_order: Option<ContractedPsiSecondOrderFn>,
    barrier_config: Option<BarrierConfig>,
    kkt_residual: Option<ProjectedKktResidual>,
    active_constraints: Option<Arc<ActiveLinearConstraintBlock>>,
    gaussian_weight_log_sum_half: f64,
}


impl<'dp> InnerSolutionBuilder<'dp> {
    /// Create a builder with the required core fields.
    pub fn new(
        log_likelihood: f64,
        penalty_quadratic: f64,
        beta: Array1<f64>,
        n_observations: usize,
        hessian_op: Arc<dyn HessianOperator>,
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
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
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
        }
    }

    pub fn deriv_provider(mut self, p: Box<dyn HessianDerivativeProvider + 'dp>) -> Self {
        self.deriv_provider = p;
        self
    }

    pub fn tk(mut self, correction: f64, gradient: Option<Array1<f64>>) -> Self {
        self.tk_correction = correction;
        self.tk_gradient = gradient;
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

    pub fn rho_prior(mut self, prior: crate::types::RhoPrior) -> Self {
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
        self.ext_coord_pair_fn = Some(f);
        self
    }

    pub fn rho_ext_pair_fn(
        mut self,
        f: Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>,
    ) -> Self {
        self.rho_ext_pair_fn = Some(f);
        self
    }

    pub fn fixed_drift_deriv(mut self, f: FixedDriftDerivFn) -> Self {
        self.fixed_drift_deriv = Some(f);
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
        if let Some(tk_gradient) = self.tk_gradient.as_ref() {
            assert_eq!(
                tk_gradient.len(),
                penalty_dim,
                "InnerSolutionBuilder: TK gradient length {} does not match penalty coordinate count {}",
                tk_gradient.len(),
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
            tk_correction: self.tk_correction,
            tk_gradient: self.tk_gradient,
            firth: self.firth,
            hessian_logdet_correction: self.hessian_logdet_correction,
            penalty_subspace_trace: self.penalty_subspace_trace,
            rho_curvature_scale: self.rho_curvature_scale,
            rho_prior: self.rho_prior,
            n_observations: self.n_observations,
            nullspace_dim,
            gaussian_weight_log_sum_half: self.gaussian_weight_log_sum_half,
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


/// Evaluation mode for the unified evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Compute cost only (e.g., for line search).
    ValueOnly,
    /// Compute cost and gradient (the common case).
    ValueAndGradient,
    /// Compute cost, gradient, and outer Hessian.
    ValueGradientHessian,
}


/// Result of the unified REML/LAML evaluation.
#[derive(Debug)]
pub struct RemlLamlResult {
    /// The REML/LAML objective value (to be minimized).
    pub cost: f64,
    /// Newton-decrement energy `½ rᵀH⁻¹r` of the converged inner KKT
    /// residual at this `ρ`, where `r = ∇_β L(β̂, ρ)` and `H` is the inner
    /// Hessian. Bounds the inner sub-optimality `|V(β̂) − V(β*)| ≤
    /// ½ rᵀH⁻¹r` to first order, and is consumed by:
    ///
    /// * the HyperGradientBudget controller, which uses it as the
    ///   inner-channel energy proxy `E_inner` when estimating
    ///   `s_inner` and re-allocating per-channel tolerances; and
    /// * the trust-energy gate in the outer strategy, which shrinks the
    ///   trust radius when this energy exceeds
    ///   `TRUST_ENERGY_FACTOR × |predicted_decrease|`.
    ///
    /// `None` when the inner solve did not compute an energy estimate
    /// (e.g., projected-pseudo-inverse paths that lack a full-H solve).
    pub ift_residual_energy: Option<f64>,
    /// One-Newton-step inner polish vector `w = H⁻¹ r`, populated only
    /// when the evaluator solves against the full inner Hessian `H` (not
    /// the projected pseudo-inverse used on rank-deficient paths).
    ///
    /// Applied by the runtime as a *free* refinement of the warm-start β
    /// at the next outer iteration: `β_warm ← β̂ + w` short-circuits one
    /// PIRLS step, exploiting the Hessian factorization already paid for
    /// during the cost-side IFT correction. `None` whenever the polish
    /// step was not produced (projected-pseudo-inverse path, value-only
    /// evaluation, etc.).
    pub inner_polish_step: Option<Array1<f64>>,
    /// Gradient ∂V/∂ρ (present if mode ≥ ValueAndGradient).
    pub gradient: Option<Array1<f64>>,
    /// Outer Hessian ∂²V/∂ρ² (present if mode = ValueGradientHessian).
    pub hessian: crate::solver::outer_strategy::HessianResult,
    /// Rho-coordinate mode responses, one `K · g_j` vector per column, when
    /// they were already built for derivative corrections. Consumed by the
    /// runtime IFT mode-response cache for joint-IFT warm starts.
    pub rho_mode_response_cols: Option<Array2<f64>>,
    /// Extended-coordinate mode responses, one `K · g_j` vector per column,
    /// when extended derivative coordinates required them.
    pub ext_mode_response_cols: Option<Array2<f64>>,
}


// ═══════════════════════════════════════════════════════════════════════════
//  Soft floor for penalized deviance (Gaussian profiled scale)
// ═══════════════════════════════════════════════════════════════════════════

// Canonical definitions live in estimate.rs; re-use them here.
use crate::solver::estimate::smooth_floor_dp;


/// Ridge floor for denominator safety.
const DENOM_RIDGE: f64 = 1e-8;


fn penalty_a_k_beta(coord: &PenaltyCoordinate, beta: &Array1<f64>, lambda: f64) -> Array1<f64> {
    coord.apply_shifted_penalty(beta, lambda)
}


fn penalty_a_k_quadratic(coord: &PenaltyCoordinate, beta: &Array1<f64>, lambda: f64) -> f64 {
    coord.shifted_quadratic(beta, lambda)
}


/// Apply the curvature-conditioning scale `s = rho_curvature_scale` to a
/// raw ρ-coordinate `λ_k = exp(ρ_k)`.
///
/// Returns `s · λ_k`, which is the per-coordinate drift coefficient
/// `∂H_op/∂ρ_k = s · λ_k · S_k` under the convention documented on
/// [`InnerSolution::rho_curvature_scale`].  The matching
/// `hessian_logdet_correction = −p · log(s)` (additive in ρ, derivative
/// zero) cancels the `p · log(s)` term in `log|H_op|` so that the cost
/// the evaluator reports and the trace `tr(K · s·λ_k·S_k)` (with
/// `K = H_op⁻¹ = (1/s) · H_orig⁻¹`) both correspond to the SAME unscaled
/// `log|H_orig|` and its analytic derivative `tr(H_orig⁻¹ · λ_k S_k)`.
///
/// If you change this scaling, you MUST also update the corresponding
/// `hessian_logdet_correction` in every caller that sets
/// `rho_curvature_scale ≠ 1`, or the cost and gradient will disagree by
/// a factor `s` — see issue #200 for the failure mode.
#[inline]
fn rho_curvature_lambda(solution: &InnerSolution<'_>, lambda: f64) -> f64 {
    solution.rho_curvature_scale * lambda
}


fn penalty_coord_to_operator(coord: PenaltyCoordinate, scale: f64) -> Arc<dyn HyperOperator> {
    struct OwnedPenaltyHyperOperator {
        coord: PenaltyCoordinate,
        scale: f64,
    }

    impl HyperOperator for OwnedPenaltyHyperOperator {
        fn dim(&self) -> usize {
            self.coord.dim()
        }

        fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(v.len());
            self.mul_vec_into(v.view(), out.view_mut());
            out
        }

        fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(v.len());
            self.mul_vec_into(v, out.view_mut());
            out
        }

        fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
            self.coord.apply_penalty_view_into(v, self.scale, out);
        }

        fn scaled_add_mul_vec(
            &self,
            v: ArrayView1<'_, f64>,
            scale: f64,
            out: ArrayViewMut1<'_, f64>,
        ) {
            if scale == 0.0 {
                return;
            }
            self.coord
                .scaled_add_penalty_view(v, scale * self.scale, out);
        }

        fn to_dense(&self) -> Array2<f64> {
            self.coord.scaled_dense_matrix(self.scale)
        }

        fn is_implicit(&self) -> bool {
            false
        }
    }

    Arc::new(OwnedPenaltyHyperOperator { coord, scale })
}


fn penalty_total_drift_result(
    coord: &PenaltyCoordinate,
    scale: f64,
    correction: Option<&DriftDerivResult>,
) -> DriftDerivResult {
    match correction {
        Some(DriftDerivResult::Dense(corr)) => {
            if coord.uses_operator_fast_path() {
                DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                    dense: Some(corr.clone()),
                    operators: vec![penalty_coord_to_operator(coord.clone(), scale)],
                    dim_hint: coord.dim(),
                }))
            } else {
                let mut dense = coord.scaled_dense_matrix(scale);
                dense += corr;
                DriftDerivResult::Dense(dense)
            }
        }
        Some(DriftDerivResult::Operator(corr_op)) => {
            DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                dense: if coord.uses_operator_fast_path() {
                    None
                } else {
                    Some(coord.scaled_dense_matrix(scale))
                },
                operators: {
                    let mut ops = vec![Arc::clone(corr_op)];
                    if coord.uses_operator_fast_path() {
                        ops.push(penalty_coord_to_operator(coord.clone(), scale));
                    }
                    ops
                },
                dim_hint: coord.dim(),
            }))
        }
        None => {
            if coord.uses_operator_fast_path() {
                DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                    dense: None,
                    operators: vec![penalty_coord_to_operator(coord.clone(), scale)],
                    dim_hint: coord.dim(),
                }))
            } else {
                DriftDerivResult::Dense(coord.scaled_dense_matrix(scale))
            }
        }
    }
}


fn hyper_coord_drift_operators(drift: &HyperCoordDrift) -> Vec<Arc<dyn HyperOperator>> {
    let mut operators: Vec<Arc<dyn HyperOperator>> = Vec::new();
    if let Some(block_local) = drift.block_local.as_ref() {
        operators.push(Arc::new(block_local.clone()));
    }
    if let Some(operator) = drift.operator.as_ref() {
        operators.push(Arc::clone(operator));
    }
    operators
}


fn hyper_coord_drift_operator_arc(
    drift: &HyperCoordDrift,
    dim_hint: usize,
) -> Option<Arc<dyn HyperOperator>> {
    let mut operators = hyper_coord_drift_operators(drift);
    if operators.is_empty() {
        return None;
    }

    if drift.dense.is_none() && operators.len() == 1 {
        return Some(operators.pop().expect("single operator drift"));
    }

    Some(Arc::new(CompositeHyperOperator {
        dense: drift.dense.clone(),
        operators,
        dim_hint,
    }))
}


fn drift_parts_into_result(
    dense: Option<Array2<f64>>,
    mut operators: Vec<Arc<dyn HyperOperator>>,
    dim_hint: usize,
) -> DriftDerivResult {
    if operators.is_empty() {
        DriftDerivResult::Dense(dense.unwrap_or_else(|| Array2::<f64>::zeros((dim_hint, dim_hint))))
    } else if dense.is_none() && operators.len() == 1 {
        DriftDerivResult::Operator(operators.pop().expect("single operator drift"))
    } else {
        DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
            dense,
            operators,
            dim_hint,
        }))
    }
}


fn hyper_coord_total_drift_parts(
    drift: &HyperCoordDrift,
    correction: Option<&DriftDerivResult>,
) -> (Option<Array2<f64>>, Vec<Arc<dyn HyperOperator>>) {
    let mut dense = drift.dense.clone();
    let mut operators = hyper_coord_drift_operators(drift);
    if let Some(correction) = correction {
        match correction {
            DriftDerivResult::Dense(matrix) => {
                if let Some(existing) = dense.as_mut() {
                    *existing += matrix;
                } else {
                    dense = Some(matrix.clone());
                }
            }
            DriftDerivResult::Operator(operator) => operators.push(Arc::clone(operator)),
        }
    }
    (dense, operators)
}


fn hyper_coord_total_drift_result(
    drift: &HyperCoordDrift,
    correction: Option<&DriftDerivResult>,
    dim_hint: usize,
) -> DriftDerivResult {
    let (dense, operators) = hyper_coord_total_drift_parts(drift, correction);
    drift_parts_into_result(dense, operators, dim_hint)
}


// ─── EFS multiplicative-update helpers ───────────────────────────────────
//
// The Wood–Fasiolo Extended Fellner–Schall update is multiplicative in the
// smoothing parameter. Writing it in log coordinates `ρ = log λ`,
//
//   Δρ = log( target / q_eff )
//      = log( ( d − t ) / q_eff )
//
// where:
//   • q_eff is the penalty-quadratic contribution to the *gradient*,
//     scaled exactly the way `outer_gradient_entry` scales it. For Fixed
//     dispersion, q_eff = β̂ᵀ B β̂ = 2 a_i. For ProfiledGaussian, it picks
//     up the smooth-floor factor `dp_cgrad / φ̂` so EFS and the gradient
//     share the same stationarity equation.
//   • d = ∂ log|S_λ|₊/∂ρ_i = tr(S_λ⁺ B_i). For ρ-coords this is
//     `solution.penalty_logdet.first[idx]`; for τ-coords it is
//     `coord.ld_s`.
//   • t = tr(K · B_i) where K is the *cost's* logdet kernel — `G_ε(H)` in
//     ordinary SPD/smooth-spectral mode, or the projected
//     `U_S (U_Sᵀ H U_S)⁻¹ U_Sᵀ` under the rank-deficient LAML fix.
//
// The previous implementation used `Δρ = (2a − tr(H⁻¹B)) / tr(H⁻¹BH⁻¹B)`,
// which (a) silently dropped the `tr(S_λ⁺ B)` term, (b) used a different
// kernel from the gradient, and (c) used the Frobenius/Gram trace as a
// curvature proxy instead of the canonical EFS denominator. As a concrete
// counterexample, the scalar Gaussian/Laplace model with z = 2, λ = 1/3 is
// at the exact REML optimum (gradient = 0) but the old formula returned
// step `+8` (clamped to `+5`) — see the unit test in this module.
//
// Exactness depends on the likelihood curvature. For Gaussian/quadratic
// likelihoods, `H_obs` is beta-independent, so `C[v_k] = 0` and the
// classical explicit trace fixed point with `Ḣ_k = λ_k S_k` is exact. For
// non-Gaussian families (Cox/survival/binomial), `H_obs` depends on beta;
// the exact logdet gradient uses the total Hessian drift
// `Ḣ_k = λ_k S_k + C[v_k]`. A pure MacKay/Tipping/Wood-Fasiolo explicit
// trace update that uses only `λ_k S_k` is therefore an approximation.
//
// This code path does not use that pure explicit-trace surrogate. EFS is
// expressed in terms of the full outer gradient from `reml_laml_evaluate`;
// that gradient builds `rho_corrections`, threads them through
// `penalty_total_drift_result`, and traces the corrected `Ḣ_k`.

/// `q_eff = 2 · penalty_term` matching `outer_gradient_entry`.
#[inline]
fn efs_q_eff(a_i: f64, dispersion: &DispersionHandling, dp_cgrad: f64, phi: f64) -> f64 {
    match dispersion {
        DispersionHandling::ProfiledGaussian => 2.0 * dp_cgrad * a_i / phi,
        DispersionHandling::Fixed { .. } => 2.0 * a_i,
    }
}


fn gamma_precision_rate_for_rho(prior: &crate::types::RhoPrior, idx: usize) -> Option<f64> {
    match prior {
        crate::types::RhoPrior::GammaPrecision { rate, .. } => Some(*rate),
        crate::types::RhoPrior::Independent(priors) => {
            priors.get(idx).and_then(|prior| match prior {
                crate::types::RhoPrior::GammaPrecision { rate, .. } => Some(*rate),
                _ => None,
            })
        }
        _ => None,
    }
}


#[inline]
fn efs_q_eff_with_gamma_rate(
    base_q_eff: f64,
    lambda: f64,
    prior: &crate::types::RhoPrior,
    idx: usize,
) -> f64 {
    match gamma_precision_rate_for_rho(prior, idx) {
        Some(rate) if rate.is_finite() && rate > 0.0 => base_q_eff + 2.0 * rate * lambda,
        _ => base_q_eff,
    }
}


/// EFS step expressed in terms of the *full* outer gradient
/// `g_full = ∂V_total/∂ρ_i` and the penalty-quadratic curvature scale
/// `q_eff`:
///
/// ```text
///   Δρ = log(1 − 2·g_full / q_eff).
/// ```
///
/// This is the universal-form Wood–Fasiolo update: when the cost is base
/// REML/LAML, the canonical `g_base = (q_eff + t − d)/2` gives
/// `1 − 2·g_base/q_eff = (d − t)/q_eff` (the classical pseudoinverse-and-
/// trace form); when out-of-band terms — Tierney–Kadane corrections,
/// smoothing-parameter priors, Firth bias-reduction, monotonicity
/// barriers, the SAS log-δ ridge — enter `g_full = g_base + g_extra`,
/// the multiplicative target shifts by exactly the right amount,
/// `1 − 2·g_full/q_eff = (d − t − 2·g_extra)/q_eff`. No per-augmentation
/// post-correction is needed in `compute_efs_update` /
/// `compute_hybrid_efs_update`. The line search in the outer
/// fixed-point bridge handles the only thing this formula can't —
/// non-PSD penalty derivatives that flip the descent direction.
///
/// Three regimes:
/// - **Stable (`q_eff > 0`, `2·g_full < q_eff`)**: clamp to `±EFS_MAX_STEP`.
/// - **Over-correction (`q_eff > 0`, `2·g_full ≥ q_eff`)**: emit
///   `−EFS_MAX_STEP`; line search trims and the canonical form resumes
///   on the next iteration.
/// - **Pathological (`q_eff ≤ 0` or non-finite)**: returns `None` so the
///   caller leaves the step at zero for that coordinate.
#[inline]
fn efs_log_step_from_grad(q_eff: f64, g_full: f64) -> Option<f64> {
    if !q_eff.is_finite() || q_eff <= 0.0 || !g_full.is_finite() {
        return None;
    }
    let ratio = 1.0 - 2.0 * g_full / q_eff;
    if ratio > 0.0 {
        Some(ratio.ln().clamp(-EFS_MAX_STEP, EFS_MAX_STEP))
    } else {
        Some(-EFS_MAX_STEP)
    }
}


/// EFS profiling factors (`profiled_scale`, `dp_cgrad`) matched to the
/// gradient assembly. For Fixed dispersion both are unused; we return
/// `(phi, 0.0)` so that `efs_q_eff` simply uses `2·a_i`.
#[inline]
fn efs_profiling(solution: &InnerSolution<'_>) -> (f64, f64) {
    match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let (dp_c, dp_cgrad, _) = smooth_floor_dp(dp_raw);
            let denom = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
            (dp_c / denom, dp_cgrad)
        }
        DispersionHandling::Fixed { phi, .. } => (*phi, 0.0),
    }
}


fn trace_hinv_cached_drift_cross(
    hop: &dyn HessianOperator,
    left_dense: Option<&Array2<f64>>,
    left_op: Option<&dyn HyperOperator>,
    right_dense: Option<&Array2<f64>>,
    right_op: Option<&dyn HyperOperator>,
) -> f64 {
    match (left_op, right_op) {
        (Some(left), Some(right)) => hop.trace_hinv_operator_cross(left, right),
        (Some(left), None) => hop.trace_hinv_matrix_operator_cross(
            right_dense.expect("right dense drift should be cached"),
            left,
        ),
        (None, Some(right)) => hop.trace_hinv_matrix_operator_cross(
            left_dense.expect("left dense drift should be cached"),
            right,
        ),
        (None, None) => hop.trace_hinv_product_cross(
            left_dense.expect("left dense drift should be cached"),
            right_dense.expect("right dense drift should be cached"),
        ),
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Shared outer-derivative formulas
// ═══════════════════════════════════════════════════════════════════════════
//
// These helpers implement the analytic identities ONCE so that all
// coordinate types (ρ, τ, ψ) and all pair types (ρ-ρ, ρ-ext, ext-ext)
// go through the same formula. Any chain-rule or transformed-parameter
// fix automatically applies to every code path.

/// Compute one entry of the outer gradient.
///
/// The universal three-term formula is:
///
/// ```text
///   ∂V/∂θ_i = a_i_scaled + ½ tr(G_ε Ḣ_i) − ½ ∂_i log|S|₊
/// ```
///
/// where:
/// - `a_i` is the fixed-β cost derivative (0.5 × β̂ᵀAₖβ̂ for ρ, coord.a for ext)
/// - `trace_logdet_i` is tr(G_ε(H) Ḣ_i) (logdet gradient operator applied to
///   the total Hessian drift including IFT correction)
/// - `ld_s_i` is ∂_i log|S|₊ (penalty pseudo-logdet derivative)
///
/// The dispersion handling scales the penalty term:
/// - Profiled Gaussian: dp_cgrad × a_i / φ̂
/// - Fixed dispersion: a_i
#[inline]
fn outer_gradient_entry(
    a_i: f64,
    trace_logdet_i: f64,
    ld_s_i: f64,
    dispersion: &DispersionHandling,
    dp_cgrad: f64,
    profiled_scale: f64,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
) -> f64 {
    let penalty_term = match dispersion {
        DispersionHandling::ProfiledGaussian => dp_cgrad * a_i / profiled_scale,
        DispersionHandling::Fixed { .. } => a_i,
    };
    let trace_term = if incl_logdet_h {
        0.5 * trace_logdet_i
    } else {
        0.0
    };
    let det_term = if incl_logdet_s { 0.5 * ld_s_i } else { 0.0 };
    penalty_term + trace_term - det_term
}


/// Compute one entry of the outer Hessian.
///
/// The universal three-term formula is:
///
/// ```text
///   ∂²V/∂θ_i∂θ_j = Q_ij + L_ij + P_ij
/// ```
///
/// where:
/// - Q_ij = pair_a − g_i·v_j  (penalty quadratic second derivative, with
///   profiled Gaussian chain-rule terms from the smooth deviance floor)
/// - L_ij = ½ (cross_trace + h2_trace) (logdet Hessian)
/// - P_ij = −½ pair_ld_s  (penalty logdet second derivative)
///
/// The `cross_trace` is the exact logdet spectral cross term. For ordinary
/// SPD backends this is `−tr(H⁻¹ Ḣ_j H⁻¹ Ḣ_i)`; for smooth spectral logdet
/// regularization it is the divided-difference contraction of
/// `log r_ε(σ)`. The `h2_trace` is tr(G_ε Ḧ_ij) from the second Hessian
/// drift including IFT and fourth-derivative corrections.
#[inline]
fn outer_hessian_entry(
    a_i: f64,
    a_j: f64,
    g_i_dot_v_j: f64,
    pair_a: f64,
    cross_trace: f64,
    h2_trace: f64,
    pair_ld_s: f64,
    profiled_phi: f64,
    profiled_nu: f64,
    profiled_dp_cgrad: f64,
    profiled_dp_cgrad2: f64,
    is_profiled: bool,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
) -> f64 {
    let q_raw = pair_a - g_i_dot_v_j;
    let q = if is_profiled {
        profiled_dp_cgrad * q_raw / profiled_phi
            + 2.0
                * (profiled_dp_cgrad2 * profiled_nu * profiled_phi
                    - profiled_dp_cgrad * profiled_dp_cgrad)
                * a_i
                * a_j
                / (profiled_nu * profiled_phi * profiled_phi)
    } else {
        q_raw
    };
    let l = if incl_logdet_h {
        0.5 * (cross_trace + h2_trace)
    } else {
        0.0
    };
    let p = if incl_logdet_s { -0.5 * pair_ld_s } else { 0.0 };
    q + l + p
}


// ═══════════════════════════════════════════════════════════════════════════
//  Constraint-tangent-space projection
// ═══════════════════════════════════════════════════════════════════════════
//
// When the inner solver converges at a constrained-stationary point with a
// non-empty active inequality-constraint set `A_act β = b_act` (k_act rows),
// the Laplace approximation lives on the tangent manifold `T = β̂ + null(A_act)`.
// With orthonormal basis `Z ∈ ℝ^{p × m}` for null(A_act) (m = p − k_act), the
// principled outer LAML objective is
//
//   V_T(ρ) = -ℓ(β̂) + ½ β̂ᵀ S(λ) β̂ + ½ log|ZᵀHZ| − ½ log|Zᵀ S(λ) Z|_+ + …
//
// (β̂-quadratic terms stay in p-space; β̂ doesn't change under projection.)
// The gradient is the envelope-theorem derivative at fixed β̂:
//
//   ∂_ρ_k V_T = ½ λ_k β̂ᵀ S_k β̂ + ½ tr((ZᵀHZ)⁻¹ Zᵀ(λ_k S_k) Z)
//             − ½ λ_k tr((ZᵀSZ)⁺ ZᵀS_kZ)
//
// Refs: Wood 2011; Wood–Pya–Säfken 2016 §3; Marra–Wood 2012 §2.
//
// The implementation strategy: wrap the inner Hessian operator in a
// tangent-projected adapter that transforms its trace/solve/logdet APIs
// from p-space to tangent space, recompute `PenaltyLogdetDerivs` for
// `ZᵀS(λ)Z`, then recurse into the regular `reml_laml_evaluate` with
// `active_constraints = None`. This routes the entire downstream pipeline
// (gradient, Hessian, IFT corrections) through the projected operator
// without duplicating cost/gradient formulas.

/// Orthonormal basis `Z ∈ ℝ^{p × m}` for `null(A_act)` via eigendecomposition
/// of `A_actᵀ A_act` (PSD; `null(A_actᵀ A_act) = null(A_act)`). Returns
/// `None` when the active set is empty or the tangent space is empty
/// (k_act ≥ p).
fn compute_active_constraint_tangent_basis(a_act: &Array2<f64>) -> Option<Array2<f64>> {
    let k_act = a_act.nrows();
    let p = a_act.ncols();
    if k_act == 0 {
        return None;
    }
    // `null(A_act) = null(A_actᵀ A_act)`; eigendecompose the symmetric PSD
    // `p × p` matrix and pull the eigenvectors with σ ≤ threshold as the
    // null basis. This gives `m = p − rank(A_act)`, the correct tangent
    // dimension regardless of whether `A_act` has linearly dependent rows.
    let ata = a_act.t().dot(a_act);
    let (evals, evecs) = ata.eigh(faer::Side::Lower).ok()?;
    let evals_slice = evals.as_slice()?;
    let threshold = positive_eigenvalue_threshold(evals_slice);
    let null_count = evals_slice.iter().filter(|&&s| s <= threshold).count();
    if null_count == 0 || null_count == p {
        // `null_count == p` means A_act has no effective constraint (every
        // row is in the noise floor). Returning `None` skips the projection
        // and lets the full p-space evaluator run unmodified.
        return None;
    }
    Some(evecs.slice(ndarray::s![.., 0..null_count]).to_owned())
}


/// Dense `p × p` materialization of a penalty coordinate via canonical
/// basis vectors. `S_k e_j` is the `j`-th column of `S_k`; assembled into
/// a p × p matrix. Cost O(p² · matvec).
fn materialize_penalty_coord_dense(coord: &PenaltyCoordinate, p: usize) -> Array2<f64> {
    // Each `PenaltyCoordinate` variant already has a structure-aware
    // materializer (`scaled_dense_matrix(1.0)`):
    //   - `DenseRoot` / `DenseRootCentered` → `Rᵀ R` via faer matmul
    //     (BLAS3, parallel).
    //   - `BlockRoot` / `BlockRootCentered` → block-local `Rᵀ R` embedded
    //     into a `total_dim × total_dim` matrix.
    //   - `KroneckerMarginal` → diagonal write (no matmul needed).
    // Routing through it replaces the previous serial p-fold matvec loop
    // with the variant-appropriate O(p²) (or O(p) for Kronecker) path.
    let out = coord.scaled_dense_matrix(1.0);
    assert_eq!(out.nrows(), p, "penalty coord dim mismatch");
    assert_eq!(out.ncols(), p, "penalty coord dim mismatch");
    out
}


/// Reconstruct the *raw* Hessian `H = V · diag(σ) · Vᵀ` (pre-regularization)
/// from a `DenseSpectralOperator`. The operator stores
/// `r_ε(σ) = ½(σ + √(σ² + 4ε²))`; invert via `σ = r − ε²/r` so the tangent
/// projection `ZᵀHZ` sees the un-regularized data. The `from_symmetric`
/// call applied to that projection then performs a *single* tangent-space
/// regularization, matching `log|ZᵀHZ|` with one consistent `r_ε` instead
/// of double-regularizing (`r_ε(ZᵀV·r_ε(σ)·VᵀZ)`).
///
/// Per the math review (codex), projecting an already-regularized H_reg
/// and re-regularizing in tangent space is not exactly `log|ZᵀHZ|`; it is
/// a modified smoothed objective. Inverting `r_ε` first restores the
/// principled single-regularization identity.
fn assemble_h_raw_dense(op: &DenseSpectralOperator) -> Array2<f64> {
    let p = op.n_dim;
    // `ε = √ε_mach · p`. Same `spectral_epsilon` formula as the operator's
    // own construction; depends only on dim.
    let epsilon = f64::EPSILON.sqrt() * (p as f64).max(1.0);
    let eps_sq = epsilon * epsilon;
    if p == 0 {
        return Array2::<f64>::zeros((0, 0));
    }
    // Express `H = V · diag(σ_raw) · Vᵀ` as two BLAS3 matmuls (faer's
    // `fast_ab` / `fast_atb` are already parallelized internally),
    // replacing the previous triple-nested O(p³) loop.
    //
    //   sigma_j = r_j − ε²/r_j  for active, nonzero `r`; else 0.
    //   VS = V · diag(sigma)    (scale columns of V by sigma)
    //   H  = VS · Vᵀ            (= fast_abt(VS, V))
    let mut vs = op.eigenvectors.clone();
    for j in 0..p {
        let sigma = if op.active_mask[j] {
            let r = op.reg_eigenvalues[j];
            if r == 0.0 { 0.0 } else { r - eps_sq / r }
        } else {
            0.0
        };
        if sigma != 1.0 {
            let mut col = vs.column_mut(j);
            if sigma == 0.0 {
                col.fill(0.0);
            } else {
                col.mapv_inplace(|v| v * sigma);
            }
        }
    }
    // H = VS · Vᵀ without materializing Vᵀ.
    crate::faer_ndarray::fast_abt(&vs, &op.eigenvectors)
}


/// Tangent-projected `HessianOperator` adapter. Wraps an `m × m`
/// `H_T = ZᵀHZ` operator and exposes the `p × p` interface needed by the
/// existing evaluator pipeline. All p-space inputs are projected via `Z`
/// before being passed to the tangent operator; outputs are lifted back
/// via `Z`. By construction this is the constraint-aware pseudo-inverse
/// `H⁺_T = Z (ZᵀHZ)⁻¹ Zᵀ`, which is bounded independent of σ_min(H)
/// when σ_min(ZᵀHZ) is bounded.
struct TangentProjectedHessianOperator {
    /// Orthonormal basis for null(A_act), `p × m`.
    z: Array2<f64>,
    /// `H_T = ZᵀHZ`, re-eigendecomposed with its own `r_ε` regularization.
    h_t_op: DenseSpectralOperator,
}


impl HessianOperator for TangentProjectedHessianOperator {
    fn active_rank(&self) -> usize {
        self.h_t_op.active_rank()
    }

    fn dim(&self) -> usize {
        self.z.nrows()
    }
    fn logdet(&self) -> f64 {
        self.h_t_op.logdet()
    }
    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let r_t = self.z.t().dot(rhs);
        let q_t = self.h_t_op.solve(&r_t);
        self.z.dot(&q_t)
    }
    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let r_t = self.z.t().dot(rhs);
        let q_t = self.h_t_op.solve_multi(&r_t);
        self.z.dot(&q_t)
    }
    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // tr(Z H_T⁻¹ Zᵀ · A) = tr(H_T⁻¹ · ZᵀAZ) (cyclic permutation).
        let zaz = self.z.t().dot(a).dot(&self.z);
        self.h_t_op.trace_hinv_product(&zaz)
    }
    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        // tr(G_ε(H) · A) where H is the wrapped tangent operator.
        // d log|ZᵀHZ|/dt = tr((ZᵀHZ)⁻¹ · Zᵀ Ḣ Z) → use H_T's logdet kernel
        // applied to ZᵀḢZ.
        let zaz = self.z.t().dot(a).dot(&self.z);
        self.h_t_op.trace_logdet_gradient(&zaz)
    }
    fn is_dense(&self) -> bool {
        self.h_t_op.is_dense()
    }
    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        self.h_t_op.logdet_traces_match_hinv_kernel()
    }
    // Deliberately keep `as_dense_spectral` and `as_exact_dense_spectral`
    // at default `None`: their consumers expect a p-space spectral basis,
    // whereas the wrapped operator lives in m-dimensional tangent space.
    // Surfacing the tangent operator there would silently let downstream
    // code mix p- and m-dim eigenvectors.
}


/// Build `PenaltyLogdetDerivs` for `log|ZᵀS(λ)Z|_+`, its first
/// derivatives, and its second derivatives. The identities are the same
/// as in p-space, applied to the projected penalty:
///   value      = log|M(λ)|_+,                M(λ) = ZᵀS(λ)Z = Σ_k λ_k Zᵀ S_k Z
///   ∂_k value  = λ_k · tr(M⁺ · Zᵀ S_k Z)
///   ∂²_kl      = δ_{kl} ∂_k value − λ_k λ_l · tr(M⁺ · Zᵀ S_l Z · M⁺ · Zᵀ S_k Z)
fn tangent_penalty_logdet(
    z: &Array2<f64>,
    penalty_coords: &[PenaltyCoordinate],
    lambdas: &[f64],
    p: usize,
) -> Result<PenaltyLogdetDerivs, String> {
    let m = z.ncols();
    let k = lambdas.len();
    let zsz: Vec<Array2<f64>> = penalty_coords
        .iter()
        .map(|c| {
            let s_k_full = materialize_penalty_coord_dense(c, p);
            z.t().dot(&s_k_full).dot(z)
        })
        .collect();
    let mut s_t = Array2::<f64>::zeros((m, m));
    for k_idx in 0..k {
        s_t.scaled_add(lambdas[k_idx], &zsz[k_idx]);
    }
    let (evals, evecs) = s_t
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("tangent S eigendecomposition failed: {e}"))?;
    let evals_slice = evals.as_slice().ok_or_else(|| {
        "tangent S eigendecomposition returned non-contiguous eigenvalues".to_string()
    })?;
    let threshold = positive_eigenvalue_threshold(evals_slice);
    let value = exact_pseudo_logdet(evals_slice, threshold);
    // Build M⁺ = Σ_{σ_j > τ} u_j u_jᵀ / σ_j once for first AND second derivatives.
    let mut s_t_plus = Array2::<f64>::zeros((m, m));
    for j in 0..m {
        if evals[j] > threshold {
            let inv = 1.0 / evals[j];
            for r in 0..m {
                let factor = evecs[[r, j]] * inv;
                for c in 0..m {
                    s_t_plus[[r, c]] += factor * evecs[[c, j]];
                }
            }
        }
    }
    let mut first = Array1::<f64>::zeros(k);
    for k_idx in 0..k {
        first[k_idx] = lambdas[k_idx] * trace_matrix_product(&s_t_plus, &zsz[k_idx]);
    }
    let mut second = Array2::<f64>::zeros((k, k));
    // δ_{kl} ∂_k value contribution (from ∂_ρ_l λ_k = λ_k δ_{kl}).
    for k_idx in 0..k {
        second[[k_idx, k_idx]] += first[k_idx];
    }
    // − λ_k λ_l · tr(M⁺ · Zᵀ S_l Z · M⁺ · Zᵀ S_k Z).
    let s_plus_zsz: Vec<Array2<f64>> = zsz.iter().map(|m_k| s_t_plus.dot(m_k)).collect();
    for k_idx in 0..k {
        for l_idx in 0..=k_idx {
            let cross = trace_matrix_product(&s_plus_zsz[k_idx], &s_plus_zsz[l_idx]);
            let entry = -lambdas[k_idx] * lambdas[l_idx] * cross;
            second[[k_idx, l_idx]] += entry;
            if l_idx != k_idx {
                second[[l_idx, k_idx]] += entry;
            }
        }
    }
    Ok(PenaltyLogdetDerivs {
        value,
        first,
        second: Some(second),
    })
}


/// Borrowing adapter that lets a tangent-projected `InnerSolution` reuse
/// the original `HessianDerivativeProvider` without taking ownership.
/// The provider returns p-space drift matrices (`D_β H[v]`) which the
/// tangent-wrapped `HessianOperator` correctly projects via `ZᵀMZ` in
/// its `trace_logdet_gradient` / `trace_hinv_product` methods. So no
/// per-method projection is needed here — pure delegation suffices.
struct BorrowedDerivProvider<'a>(&'a dyn HessianDerivativeProvider);


impl<'a> HessianDerivativeProvider for BorrowedDerivProvider<'a> {
    fn hessian_derivative_correction(
        &self,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0.hessian_derivative_correction(v)
    }
    fn hessian_derivative_correction_result(
        &self,
        v: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        self.0.hessian_derivative_correction_result(v)
    }
    fn hessian_derivative_corrections_result(
        &self,
        vs: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        self.0.hessian_derivative_corrections_result(vs)
    }
    fn has_batched_hessian_derivative_corrections(&self) -> bool {
        self.0.has_batched_hessian_derivative_corrections()
    }
    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0.hessian_second_derivative_correction(v_k, v_l, u_kl)
    }
    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        self.0
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)
    }
    fn hessian_second_derivative_corrections_result(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        self.0.hessian_second_derivative_corrections_result(triples)
    }
    fn has_batched_hessian_second_derivative_corrections(&self) -> bool {
        self.0.has_batched_hessian_second_derivative_corrections()
    }
    fn has_corrections(&self) -> bool {
        self.0.has_corrections()
    }
    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        self.0.outer_hessian_derivative_kernel()
    }
    fn family_outer_hessian_operator(
        &self,
    ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
        self.0.family_outer_hessian_operator()
    }
    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        self.0.scalar_glm_ingredients()
    }
}


/// If the inner solution carries a non-empty active inequality-constraint
/// set, build a tangent-projected solution and dispatch the outer
/// derivative computation to it. Returns `Ok(None)` when no projection is
/// required (no active constraints); `Ok(Some(result))` when projection
/// succeeded and the recursive evaluate returned a value; `Err` only if
/// projection failed (e.g., dense backend required but not available).
fn try_tangent_projected_evaluate(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
) -> Result<Option<RemlLamlResult>, String> {
    let block = match solution.active_constraints.as_ref() {
        Some(b) if b.a.nrows() > 0 => b,
        _ => return Ok(None),
    };
    let p = solution.beta.len();
    if block.a.ncols() != p {
        return Err(format!(
            "active_constraints.a has {} columns but β is {}-dim",
            block.a.ncols(),
            p
        ));
    }
    // Principled pass-through / projection of optional `InnerSolution`
    // features.  The cost-side scalars (`tk_correction`, barrier cost at β̂)
    // are not in β-space and require no projection.  `tk_gradient` is a
    // ρ-vector and also passes through unchanged.  `hessian_logdet_correction`
    // encodes a p-space uniform rescale `−p·log α`; under projection the
    // equivalent correction is `−m·log α`, recovered by the scalar factor
    // `m/p`.  `barrier_config` propagates to the projected solution so the
    // barrier-derivative wrapper still augments dH/dρ; the tangent operator
    // applies `ZᵀMZ` correctly in its trace methods.
    //
    // `firth` and `ext_coords` carry p-space objects (Jeffreys `½ log|J|`,
    // ext-coord g/drift).  Both are projected here under the same
    // tangent-projected LAML setup as the rest of this routine
    // (`½ log|J| → ½ log|ZᵀJZ|`, `g → Zᵀ g`, `drift M → Zᵀ M Z`).  The
    // only remaining unsupported case is the `ValueGradientHessian`
    // mode with non-empty `ext_coord_pair_fn` / `rho_ext_pair_fn` —
    // those callbacks return objects in p-space whose projected form
    // requires composing with `Z` per-call; for `ValueAndGradient` the
    // per-coord `g`/`drift` is sufficient.
    let z = match compute_active_constraint_tangent_basis(&block.a) {
        Some(z) => z,
        None => {
            // Constraint matrix spans the full p-space — tangent manifold is
            // {β̂}. There is no degree of freedom left to optimise over and
            // the outer LAML cost is the constant `-ℓ(β̂) + ½ β̂ᵀSβ̂`. We
            // can return a degenerate result, but it's simpler and clearer
            // to surface the situation.
            return Err(format!(
                "active constraint matrix has rank {} on {}-dim space; \
                 tangent manifold is a single point ({{β̂}}), no outer \
                 derivative is defined",
                block.a.nrows(),
                p
            ));
        }
    };
    let h_full = solution
        .hessian_op
        .assemble_h_dense_for_tangent_projection()?;
    let h_t = z.t().dot(&h_full).dot(&z);
    let h_t_op = DenseSpectralOperator::from_symmetric(&h_t)
        .map_err(|e| format!("tangent H eigendecomposition failed: {e}"))?;
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let projected_logdet = tangent_penalty_logdet(&z, &solution.penalty_coords, &lambdas, p)?;
    // Project the KKT residual to lift the IFT correction into tangent
    // coordinates: with `q = H⁺_T r = Z (ZᵀHZ)⁻¹ Zᵀ r`, the formulas
    // `-½ rᵀ q` and `-aᵀ_k q + ½ qᵀ A_k q` are the same as the p-space
    // formulas (Zᵀ cancels through the operator wrapper). We pass r in
    // p-space; the wrapper does the projection internally.
    let projected_kkt = solution.kkt_residual.clone();
    let m_tangent = z.ncols();
    let wrapper = TangentProjectedHessianOperator {
        z: z.clone(),
        h_t_op,
    };
    // Rank-aware projection of a uniform-rescale correction:
    //   the p-space correction encodes `−p·log α` so that
    //   `log|H| = log|H'| − p·log α`. Under tangent projection the
    //   correction becomes `−m·log α = (m/p) · (−p·log α)`, i.e. the
    //   same scalar scaled by the rank ratio m/p.
    let projected_hlogdet_correction = if p == 0 {
        0.0
    } else {
        solution.hessian_logdet_correction * (m_tangent as f64 / p as f64)
    };
    // Construct the projected InnerSolution. The fields that must be
    // overridden are: hessian_op (now tangent-wrapped), penalty_logdet
    // (now in tangent space), hessian_logdet_correction (rank-ratio
    // rescaled to tangent space), penalty_subspace_trace (None; direct
    // tangent-H path replaces the kernel route), active_constraints
    // (None; prevents recursion). `tk_*` (ρ-/scalar-space) and
    // `barrier_config` (cost evaluated at β̂ in p-space; barrier-derivative
    // wrapper produces p-space drift that the tangent operator projects
    // via ZᵀMZ in its trace methods) pass through unchanged.
    // Active-constraint tangent projection for the Firth/Jeffreys term.
    // Replace the operator's full-space `½ log|J|` with the projected
    // `½ log|ZᵀJZ|`. The same underlying `FirthDenseOperator` is retained
    // so any downstream β-gradient consumer still sees a consistent
    // operator — only the scalar contribution to the outer LAML cost is
    // overridden. This projection-aware Firth is exact under the same
    // tangent-projected LAML setup as the rest of
    // `try_tangent_projected_evaluate` (mode = ValueAndGradient or below).
    let projected_firth = solution
        .firth
        .as_ref()
        .map(|term| match term.operator_arc() {
            Some(op_arc) => {
                let projected_value = op_arc.jeffreys_logdet_projected(z.view());
                ExactJeffreysTerm::with_projected_value(op_arc, projected_value)
            }
            // Tier-B value-only carrier: the scalar Φ(β̂) is already final (the
            // coupled joint path owns its own constraint handling upstream), so
            // the term passes through unchanged.
            None => term.clone(),
        });
    // Active-constraint tangent projection for ext coords. The tangent
    // hessian wrapper accepts p-space `g` and p-space drift `M` and
    // applies the `Zᵀ · Z` projection internally inside its `solve` /
    // `trace_logdet_gradient` / `trace_hinv_product` methods, so
    // pass-through is mathematically equivalent to projecting `g → Zᵀg`
    // and `M → ZᵀMZ` here (the wrapper composes the projections with
    // the inner H_T operator). This is the same pattern
    // `BorrowedDerivProvider` uses for the deriv-provider corrections.
    //
    // The pair callbacks (`ext_coord_pair_fn`, `rho_ext_pair_fn`) return
    // `HyperCoordPair` objects with p-space `b_mat` / `b_operator`.
    // `ValueAndGradient` mode does not contract those pair objects (they
    // only enter outer-Hessian assembly), so they are dropped (set to
    // `None`) in the projected inner solution — gradient evaluations are
    // unaffected. `ValueGradientHessian` mode would actually consume the
    // pair callbacks; the tangent hessian wrapper cannot re-project
    // their p-space second-drift outputs a posteriori, so refuse that
    // combination upfront when callbacks are present.
    if mode == EvalMode::ValueGradientHessian
        && !solution.ext_coords.is_empty()
        && (solution.ext_coord_pair_fn.is_some() || solution.rho_ext_pair_fn.is_some())
    {
        return Err(
            "active constraints + ext_coords + mode=ValueGradientHessian not yet supported; \
             fall back to ValueAndGradient. The ext-coord pair callbacks return p-space \
             second-drift objects that the tangent hessian wrapper does not re-project."
                .to_string(),
        );
    }
    let projected = InnerSolution {
        log_likelihood: solution.log_likelihood,
        penalty_quadratic: solution.penalty_quadratic,
        hessian_op: Arc::new(wrapper),
        beta: solution.beta.clone(),
        penalty_coords: solution.penalty_coords.clone(),
        penalty_logdet: projected_logdet,
        deriv_provider: Box::new(BorrowedDerivProvider(solution.deriv_provider.as_ref())),
        tk_correction: solution.tk_correction,
        tk_gradient: solution.tk_gradient.clone(),
        // Same operator, projection-aware scalar contribution.
        firth: projected_firth,
        hessian_logdet_correction: projected_hlogdet_correction,
        // Direct tangent-H path; the projected-kernel route is unused here.
        penalty_subspace_trace: None,
        rho_curvature_scale: solution.rho_curvature_scale,
        rho_prior: solution.rho_prior.clone(),
        n_observations: solution.n_observations,
        nullspace_dim: solution.nullspace_dim,
        gaussian_weight_log_sum_half: solution.gaussian_weight_log_sum_half,
        dispersion: solution.dispersion.clone(),
        // ext_coord g/drift pass-through: projection is applied by the
        // tangent hessian wrapper's trace and solve methods.
        ext_coords: solution.ext_coords.clone(),
        ext_coord_pair_fn: None,
        rho_ext_pair_fn: None,
        // Second-order pair callbacks are dropped on the projected path (same
        // reason as the ext-coord/rho pair fns: the tangent hessian wrapper
        // cannot re-project their p-space second-drift outputs).
        contracted_psi_second_order: None,
        fixed_drift_deriv: None,
        barrier_config: solution.barrier_config.clone(),
        kkt_residual: projected_kkt,
        // Prevents recursion via `try_tangent_projected_evaluate`.
        active_constraints: None,
        stochastic_trace_state: solution.stochastic_trace_state.clone(),
    };
    let result = reml_laml_evaluate(&projected, rho, mode, prior_cost_gradient)?;
    Ok(Some(result))
}
