use super::*;

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
pub(crate) const HUTCHPP_TRACE_MIN_DIM: usize = 128;

/// Build the Hutch++ stochastic-trace configuration for an operator of the given
/// dimension. The sketch dimension grows with `dim` (one column per 32 of
/// dimension, bounded to `[4, 16]`), and the probe budget tracks the sketch so
/// the estimator's variance and cost stay balanced across problem sizes. Shared
/// by every implicit-operator trace path so they cannot drift apart.
pub(crate) fn hutchpp_config_for_dim(dim: usize) -> StochasticTraceConfig {
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

    /// Efficient computation of tr(G_ε(H) Hₖ) for the logdet gradient.
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
