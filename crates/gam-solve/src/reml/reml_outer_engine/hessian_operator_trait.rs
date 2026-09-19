use super::*;

// ═══════════════════════════════════════════════════════════════════════════
//  Core traits
// ═══════════════════════════════════════════════════════════════════════════

/// Abstract interface for Hessian linear algebra operations.
///
/// All operations use the SAME internal decomposition, ensuring spectral
/// consistency between logdet (used in cost) and trace/solve (used in gradient).
///
/// Implementors:
/// - `DenseSpectralOperator`: eigendecomposition of dense H
/// - `DenseCholeskyOperator`: exact LLT of dense positive-definite H
/// - Sparse Cholesky operators (external implementations)
/// - `BlockCoupledOperator`: one LLT or spectral factorization of joint H
pub trait HessianFactorization: Send + Sync {
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

    /// The orthonormal basis and the eigenvalues this backend's `solve` divides by on it
    /// (gam#2765), read from an exact dense spectral view. A backend whose `solve` lifts through a
    /// projection, or that holds its dense matrix, names its own. The default never densifies: a
    /// sparse factorization has no dense form to decompose within its memory budget, so it names
    /// no span, and a mode solved against it is not graded for a fold.
    fn inverted_span(&self) -> Option<InvertedSpan> {
        self.as_exact_dense_spectral()
            .and_then(InvertedSpan::from_dense_spectral)
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
    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
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

    /// tr(H⁻¹ A H⁻¹ B) for dense symmetric Hessian drifts.
    ///
    /// This is the second-order trace object used by EFS denominators and the
    /// ψ-block trace Gram preconditioner. The default implementation computes
    /// both solved column stacks exactly and contracts them as
    /// `tr((H⁻¹A)(H⁻¹B))`.
    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let solved_a = self.solve_multi(a);
        if std::ptr::eq(a, b) {
            return dense::trace_product(&solved_a, &solved_a);
        }
        let solved_b = self.solve_multi(b);
        dense::trace_product(&solved_a, &solved_b)
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
        if left.is_implicit() || right.is_implicit() {
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

        let block = gam_runtime::resource::byte_balanced_row_chunk(p, n);

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
    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
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
    /// `tr(G_ε(H) · A_block)` for a PSD `A_block = rootᵀroot` given by its
    /// ROOT rather than by the squared block.
    ///
    /// Same quantity as [`Self::trace_logdet_block_local`] with
    /// `block = rootᵀroot`, and NOT the same arithmetic. The squared form
    /// evaluates `Σ_j g_jᵀ A g_j`, whose per-column error is `O(ε‖A‖‖g_j‖²)`;
    /// the `g_j` are scaled by `σ_j(H)^{-1/2}`, so the sum carries
    /// `O(ε·‖A‖/σ_min(H))` — `O(ε·κ(H))` on a trace bounded by `rank(A)`. On
    /// the `s(pc1,k=5)+s(pc2,k=5)` witness of #2644 that is `2.2e-16 · 6.2e11 /
    /// 0.16 ≈ 8.5e-4` per coordinate, against a stationarity bound of `3.0e-3`.
    ///
    /// The Gram form `‖root · G_block‖_F²` squares the residual instead
    /// (`O(ε²‖A‖/σ_min)`) and cannot come out negative. See
    /// `PenaltyCoordinate::scaled_block_root`.
    ///
    /// The default squares the root so every backend answers correctly;
    /// spectral backends override it.
    fn trace_logdet_block_root(
        &self,
        root: ndarray::ArrayView2<'_, f64>,
        start: usize,
        end: usize,
    ) -> f64 {
        let block = root.t().dot(&root);
        self.trace_logdet_block_local(&block, 1.0, start, end)
    }

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
            return -dense::trace_product(&y_i, &y_i);
        }
        let y_j = self.solve_multi(h_j);
        -dense::trace_product(&y_j, &y_i)
    }

    /// Operator-backed mixed form of `trace_logdet_hessian_cross`.
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

    /// Operator-backed form of `trace_logdet_hessian_cross`.
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

    /// A first-order bound on the forward error of [`Self::logdet`] carried from
    /// this factorization's own backward error (#2954), with the `O(‖δH‖²)`
    /// remainder dropped. `None` when the backend forms none.
    fn logdet_forward_error(&self) -> Option<f64> {
        None
    }

    /// Whether this operator is backed by a dense factorization.
    ///
    /// Dense operators (eigendecomposition) have O(p²) trace cost per matrix;
    /// sparse operators (Cholesky) have O(nnz) solve cost.
    fn is_dense(&self) -> bool {
        false
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
pub fn hessian_factorization_geometric_scale(op: &dyn HessianFactorization) -> Option<f64> {
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
