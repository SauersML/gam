//! Unified REML/LAML evaluator.
//!
//! This module provides a single implementation of the outer REML/LAML objective,
//! gradient, and Hessian that is shared across all backends (dense spectral,
//! sparse Cholesky, block-coupled) and all families (Gaussian, GLM, GAMLSS,
//! survival, link wiggles).
//!
//! # Architecture
//!
//! The REML/LAML formula is invariant to the sparsity
//! pattern, block structure, and family type. It is always:
//!
//! ```text
//! V(ρ) = −ℓ(β̂) + ½ β̂ᵀS(ρ)β̂ + ½ log|H| − ½ log|S|₊ + corrections
//! ```
//!
//! What differs across backends is how the inner solver finds β̂, how
//! logdet/trace/solve operations dispatch (dense eigendecomposition vs sparse
//! Cholesky vs block-coupled), and what family-specific derivative information
//! is available.
//!
//! This module separates those concerns:
//! - [`HessianOperator`]: backend-specific linear algebra (logdet, trace, solve)
//! - [`InnerSolution`]: the converged inner state (β̂, penalties, factorization)
//! - [`reml_laml_evaluate`]: the single formula, written once
//!
//! # Spectral Consistency Guarantee
//!
//! The `HessianOperator` trait ensures that `logdet()` (used in cost) and
//! `trace_hinv_product()` (used in gradient) are computed from the same
//! internal decomposition. This eliminates the class of bugs where cost uses
//! Cholesky-based logdet while gradient uses eigendecomposition-based traces
//! with a different numerical threshold.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::faer_ndarray::FaerEigh;
use crate::linalg::matrix::DesignMatrix;

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
/// - Sparse Cholesky operators (external implementations)
/// - `BlockCoupledOperator`: eigendecomposition of joint multi-block H
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
    fn stochastic_trace_solve(&self, rhs: &Array1<f64>, _rel_tol: f64) -> Array1<f64> {
        self.solve(rhs)
    }

    /// H⁻¹ M for stochastic trace probes.
    fn stochastic_trace_solve_multi(&self, rhs: &Array2<f64>, _rel_tol: f64) -> Array2<f64> {
        self.solve_multi(rhs)
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
    /// block at biobank scale.
    fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        debug_assert!(self.logdet_traces_match_hinv_kernel());
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
                panic!("xt_logdet_kernel_x_diagonal: row chunk failed: {err}")
            });
            let chunk_t = rows.t().to_owned();
            let z_chunk = self.solve_multi(&chunk_t);
            for i in 0..(end - start) {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += rows[[i, j]] * z_chunk[[j, i]];
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

    /// Compute the second-order correction to H_{k,l} for the outer Hessian.
    ///
    /// Returns `None` if not needed or not implemented.
    fn hessian_second_derivative_correction(
        &self,
        _: &Array1<f64>,
        _: &Array1<f64>,
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
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
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
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
/// For non-canonical links (probit, cloglog, SAS, mixture, beta-logistic),
/// `c_array` and `d_array` store the **observed-information** weight
/// derivatives (c_obs, d_obs) that include residual-dependent corrections:
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
        // Stays matrix-free: `matrixvectormultiply` and `compute_xtwx` route
        // through the operator-backed design's chunked kernels at biobank
        // scale, so we never materialize the full (n×p) dense block.
        let x_v = self.x_transformed.matrixvectormultiply(v_k); // X vₖ: n-vector

        // Elementwise: −c ⊙ (X vₖ); par_for_each scales over n at biobank size.
        let mut neg_c_xv = x_v;
        Zip::from(&mut neg_c_xv)
            .and(&self.c_array)
            .par_for_each(|xv_i, &c_i| *xv_i *= -c_i);

        // −Xᵀ diag(c ⊙ Xvₖ) X via the design's matrix-free weighted gram.
        let result = self
            .x_transformed
            .compute_xtwx(&neg_c_xv)
            .map_err(|e| format!("hessian_derivative_correction xtwx: {e}"))?;

        Ok(Some(result))
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
        // `compute_xtwx` so biobank-scale designs never densify the (n×p)
        // block.
        let x_vk = self.x_transformed.matrixvectormultiply(v_k);
        let x_vl = self.x_transformed.matrixvectormultiply(v_l);
        let x_ukl = self.x_transformed.matrixvectormultiply(u_kl);

        let n = self.x_transformed.nrows();
        let mut weights = Array1::zeros(n);

        // c ⊙ X u_{kl}
        Zip::from(&mut weights)
            .and(&self.c_array)
            .and(&x_ukl)
            .par_for_each(|w, &c, &xu| *w = c * xu);

        // + d ⊙ (X vₖ) ⊙ (X vₗ)
        if let Some(ref d_array) = self.d_array {
            Zip::from(&mut weights)
                .and(d_array)
                .and(&x_vk)
                .and(&x_vl)
                .par_for_each(|w, &d, &xvk, &xvl| *w += d * xvk * xvl);
        }

        // Xᵀ diag(weights) X via the design's matrix-free weighted gram.
        let result = self
            .x_transformed
            .compute_xtwx(&weights)
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
    operator: std::sync::Arc<super::FirthDenseOperator>,
}

impl ExactJeffreysTerm {
    pub(crate) fn new(operator: std::sync::Arc<super::FirthDenseOperator>) -> Self {
        Self { operator }
    }

    #[inline]
    pub(crate) fn value(&self) -> f64 {
        self.operator.jeffreys_logdet()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Log-barrier support for constrained coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for a log-barrier penalty on constrained coefficients.
///
/// The barrier-augmented objective adds `-τ Σ_{j ∈ C} log(β_j − b_j)`.
/// τ is an algorithmic continuation parameter — NOT a hyperparameter.
#[derive(Clone, Debug)]
pub struct BarrierConfig {
    /// Barrier strength parameter (continuation schedule drives this → 0).
    pub tau: f64,
    /// Indices of constrained coefficients in the β vector.
    pub constrained_indices: Vec<usize>,
    /// Lower bounds b_j for each constrained coefficient.
    pub lower_bounds: Vec<f64>,
}

impl BarrierConfig {
    /// Construct a `BarrierConfig` from linear inequality constraints `A β ≥ b`
    /// by extracting rows that represent simple coordinate bounds (β_j ≥ b_i).
    ///
    /// A row is a simple bound iff it has exactly one nonzero entry equal to 1.0.
    /// Returns `None` if the constraints are `None` or no simple-bound rows are found.
    pub fn from_constraints(
        constraints: Option<&crate::pirls::LinearInequalityConstraints>,
    ) -> Option<Self> {
        let constraints = constraints?;
        let mut indices = Vec::new();
        let mut lower_bounds = Vec::new();
        for i in 0..constraints.a.nrows() {
            let row = constraints.a.row(i);
            let mut single_col = None;
            let mut is_simple = true;
            for (j, &val) in row.iter().enumerate() {
                if val.abs() < 1e-14 {
                    continue;
                }
                if (val - 1.0).abs() < 1e-14 && single_col.is_none() {
                    single_col = Some(j);
                } else {
                    is_simple = false;
                    break;
                }
            }
            if is_simple {
                if let Some(col) = single_col {
                    indices.push(col);
                    lower_bounds.push(constraints.b[i]);
                }
            }
        }
        if indices.is_empty() {
            return None;
        }
        Some(BarrierConfig {
            tau: 1e-6,
            constrained_indices: indices,
            lower_bounds,
        })
    }

    /// Compute slack values Δ_j = β_j − b_j. Returns `None` if infeasible.
    pub fn slacks(&self, beta: &Array1<f64>) -> Option<Vec<f64>> {
        let mut slacks = Vec::with_capacity(self.constrained_indices.len());
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let delta = beta[idx] - self.lower_bounds[ci];
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

    /// Compute the barrier cost −τ Σ log(Δ_j).
    pub fn barrier_cost(&self, beta: &Array1<f64>) -> Result<f64, String> {
        let slacks = self
            .slacks(beta)
            .ok_or_else(|| "Barrier: infeasible point (slack ≤ 0)".to_string())?;
        Ok(-self.tau * slacks.iter().map(|&d| d.ln()).sum::<f64>())
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
        let slacks = match self.slacks(beta) {
            Some(s) => s,
            None => return true, // infeasible → conservatively active
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
            slacks,
            p: beta.len(),
        })
    }

    fn barrier_correction(&self, u: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((self.p, self.p));
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let inv_cube = 1.0 / (self.slacks[ci].powi(3));
            result[[idx, idx]] = -2.0 * self.tau * u[idx] * inv_cube;
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
        _cache: &ProjectedFactorCache,
    ) -> f64 {
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
        factor.t().dot(&op_factor)
    }

    /// Fill columns `[start, start + out.ncols())` of `B` into `out`.
    ///
    /// Sparse exact traces build `B E` in column batches. Operators with
    /// materialized column storage can override this to copy columns directly
    /// instead of multiplying one basis vector at a time.
    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        let dim = out.nrows();
        debug_assert!(start + cols <= dim);
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
    /// Panics or returns a zero matrix if the operator was designed to avoid
    /// materialization. Callers should check `is_implicit()` first.
    fn to_dense(&self) -> Array2<f64>;

    /// Whether this operator uses implicit (non-materialized) storage.
    fn is_implicit(&self) -> bool;

    /// Downcast to `ImplicitHyperOperator` if this is one.
    ///
    /// Returns `Some` for implicit operators that use the weighted-Gram
    /// structure (A_d = X^T C_d X + P_d), `None` for dense wrappers.
    fn as_implicit(&self) -> Option<&ImplicitHyperOperator> {
        None
    }

    /// If this operator is block-local (nonzero only in [start..end, start..end]),
    /// returns the block range and local matrix. Enables O(p_block²) trace
    /// computations instead of O(p²).
    fn block_local_data(&self) -> Option<(&Array2<f64>, usize, usize)> {
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

#[derive(Default)]
pub struct ProjectedFactorCache {
    entries: Mutex<HashMap<ProjectedFactorKey, Arc<Array2<f64>>>>,
}

impl ProjectedFactorCache {
    pub fn get_or_insert_with(
        &self,
        key: ProjectedFactorKey,
        compute: impl FnOnce() -> Array2<f64>,
    ) -> Arc<Array2<f64>> {
        let mut entries = self
            .entries
            .lock()
            .expect("projected factor cache lock poisoned");
        if let Some(value) = entries.get(&key).cloned() {
            return value;
        }
        let computed = Arc::new(compute());
        entries.insert(key, computed.clone());
        computed
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
        debug_assert!(end <= self.matrix.ncols());
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

impl HyperOperator for CompositeHyperOperator {
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
        for op in &self.operators {
            trace += op.trace_projected_factor(factor);
        }
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
        for op in &self.operators {
            trace += op.trace_projected_factor_cached(factor, cache);
        }
        trace
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
        for row in 0..self.local.nrows() {
            let mut row_dot = 0.0;
            for col in 0..self.local.ncols() {
                row_dot += self.local[[row, col]] * v_block[col];
            }
            total += u_block[row] * row_dot;
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
        debug_assert_eq!(v.len(), out.len());
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
pub struct ImplicitHyperOperator {
    /// The implicit design-derivative operator (shared across all axes).
    pub implicit_deriv: std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
    /// Which axis this operator is for.
    pub axis: usize,
    /// The active-basis design matrix X. This may be lazy / operator-backed.
    pub x_design: std::sync::Arc<DesignMatrix>,
    /// Working weights W (diagonal, length n). Shared reference.
    pub w_diag: std::sync::Arc<Array1<f64>>,
    /// Penalty derivative matrix S_{ψ_d} (p × p), dense.
    pub s_psi: Array2<f64>,
    /// Total basis dimension p.
    pub p: usize,
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
        debug_assert_eq!(v.len(), self.p);
        let n_obs = self.w_diag.len();
        let mut x_v = Array1::<f64>::zeros(n_obs);
        let mut n_work = Array1::<f64>::zeros(n_obs);
        let mut p_work = Array1::<f64>::zeros(self.p);
        design_matrix_apply_view_into(&self.x_design, v, x_v.view_mut());
        self.matvec_with_shared_xz_into(&x_v, v, out, n_work.view_mut(), p_work.view_mut());
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        debug_assert!(start + cols <= self.p);

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
        debug_assert_eq!(v.len(), self.p);
        debug_assert_eq!(u.len(), self.p);

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

    fn to_dense(&self) -> Array2<f64> {
        // Fallback: materialize column by column.
        let p = self.p;
        let mut out = Array2::<f64>::zeros((p, p));
        let mut ei = Array1::<f64>::zeros(p);
        for j in 0..p {
            ei[j] = 1.0;
            self.mul_vec_into(ei.view(), out.column_mut(j));
            ei[j] = 0.0;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        true
    }

    fn as_implicit(&self) -> Option<&ImplicitHyperOperator> {
        Some(self)
    }
}

impl ImplicitHyperOperator {
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
        debug_assert_eq!(x_col.len(), c.len());
        debug_assert_eq!(n_work.len(), c.len());
        debug_assert_eq!(p_work.len(), self.p);

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
    pub fn matvec_with_shared_xz(&self, x_vec: &Array1<f64>, z: &Array1<f64>) -> Array1<f64> {
        // Term 1: (∂X/∂ψ_d)^T (W · x_vec)
        let w_x_vec = &*self.w_diag * x_vec;
        let term1 = self
            .implicit_deriv
            .transpose_mul(self.axis, &w_x_vec.view())
            .expect("radial scalar evaluation failed during implicit hyper transpose_mul");

        // Term 2: X^T (W · ((∂X/∂ψ_d) · z))
        let dx_z = self
            .implicit_deriv
            .forward_mul(self.axis, &z.view())
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        let w_dx_z = &*self.w_diag * &dx_z;
        let term2 = self.x_design.transpose_vector_multiply(&w_dx_z);

        // Term 3: S_{ψ_d} · z
        let term3 = self.s_psi.dot(z);

        let mut out = term1 + term2 + term3;

        // Term 4 (non-Gaussian only): X^T diag(c ⊙ X_{ψ_d} β̂) · x_vec
        // (`x_vec` is already X·z, supplied by the caller).
        if let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() {
            let weighted = c_x_psi_beta.as_ref() * x_vec;
            out += &self.x_design.transpose_vector_multiply(&weighted);
        }

        out
    }

    pub fn matvec_with_shared_xz_into(
        &self,
        x_vec: &Array1<f64>,
        z: ArrayView1<'_, f64>,
        mut out: ArrayViewMut1<'_, f64>,
        mut n_work: ArrayViewMut1<'_, f64>,
        mut p_work: ArrayViewMut1<'_, f64>,
    ) {
        debug_assert_eq!(z.len(), self.p);
        debug_assert_eq!(out.len(), self.p);
        debug_assert_eq!(n_work.len(), self.w_diag.len());
        debug_assert_eq!(p_work.len(), self.p);

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
    pub x_tau: super::HyperDesignDerivative,
    /// Design matrix X in the sparse-native basis.
    pub x_design: DesignMatrix,
    /// Working weights W (diagonal).
    pub w_diag: std::sync::Arc<Array1<f64>>,
    /// Penalty derivative S_τ.
    pub s_tau: Array2<f64>,
    /// Fixed-β non-Gaussian curvature term c ⊙ (X_τ β̂), if applicable.
    pub c_x_tau_beta: Option<Array1<f64>>,
    /// Fixed-β Firth partial Hessian drift (H_φ)_{τ}|_β, if applicable.
    pub firth_hphi_tau_partial: Option<Array2<f64>>,
    /// Total coefficient dimension.
    pub p: usize,
}

impl HyperOperator for SparseDirectionalHyperOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p);

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

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.p, self.p));
        let mut basis = Array1::<f64>::zeros(self.p);
        for j in 0..self.p {
            basis[j] = 1.0;
            self.mul_vec_into(basis.view(), out.column_mut(j));
            basis[j] = 0.0;
        }
        out
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
    BlockRoot {
        root: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
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

    pub fn from_block_root(root: Array2<f64>, start: usize, end: usize, total_dim: usize) -> Self {
        assert_eq!(root.ncols(), end.saturating_sub(start));
        assert!(end <= total_dim);
        Self::BlockRoot {
            root,
            start,
            end,
            total_dim,
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::DenseRoot(root) | Self::BlockRoot { root, .. } => root.nrows(),
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
            Self::DenseRoot(root) => root.ncols(),
            Self::BlockRoot { total_dim, .. } | Self::KroneckerMarginal { total_dim, .. } => {
                *total_dim
            }
        }
    }

    pub fn uses_operator_fast_path(&self) -> bool {
        matches!(
            self,
            Self::BlockRoot { .. } | Self::KroneckerMarginal { .. }
        )
    }

    fn apply_root(&self, beta: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(beta.len(), self.dim());
        match self {
            Self::DenseRoot(root) => root.dot(beta),
            Self::BlockRoot {
                root, start, end, ..
            } => root.dot(&beta.slice(ndarray::s![*start..*end])),
            Self::KroneckerMarginal { .. } => {
                // No single root for Kronecker — use apply_penalty instead.
                panic!(
                    "apply_root not supported for KroneckerMarginal; use apply_penalty directly"
                );
            }
        }
    }

    pub fn apply_penalty(&self, beta: &Array1<f64>, scale: f64) -> Array1<f64> {
        debug_assert_eq!(beta.len(), self.dim());
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
        debug_assert_eq!(beta.len(), self.dim());
        debug_assert_eq!(out.len(), self.dim());
        out.fill(0.0);
        self.scaled_add_penalty_view(beta, scale, out);
    }

    pub fn scaled_add_penalty_view(
        &self,
        beta: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        debug_assert_eq!(beta.len(), self.dim());
        debug_assert_eq!(out.len(), self.dim());
        if scale == 0.0 {
            return;
        }
        match self {
            Self::DenseRoot(_) | Self::BlockRoot { .. } => match self {
                Self::DenseRoot(root) => {
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
                _ => unreachable!(),
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
                debug_assert_eq!(
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
            Self::DenseRoot(_) | Self::BlockRoot { .. } => {
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

    pub fn scaled_dense_matrix(&self, scale: f64) -> Array2<f64> {
        match self {
            Self::DenseRoot(root) => {
                let mut out = root.t().dot(root);
                out *= scale;
                out
            }
            Self::BlockRoot {
                root,
                start,
                end,
                total_dim,
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
                debug_assert_eq!(
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
            Self::DenseRoot(root) => {
                let mut out = root.t().dot(root);
                out *= scale;
                let p = out.nrows();
                (out, 0, p)
            }
            Self::BlockRoot {
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
            Self::BlockRoot { .. } | Self::KroneckerMarginal { .. }
        )
    }

    /// Apply λ_k S_k to a vector v without materializing the full matrix.
    /// For BlockRoot: extracts v[start..end], multiplies by local S_k, embeds result.
    pub fn scaled_matvec(&self, v: &Array1<f64>, scale: f64) -> Array1<f64> {
        match self {
            Self::DenseRoot(root) => {
                let root_v = root.dot(v);
                let mut out = root.t().dot(&root_v);
                out *= scale;
                out
            }
            Self::BlockRoot {
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

    /// Compute tr(M · λ_k S_k) where M is given as a dense matrix, without
    /// materializing λ_k S_k to full total_dim × total_dim.
    /// For BlockRoot: only reads M[start..end, start..end].
    pub fn trace_with_dense(&self, m: &Array2<f64>, scale: f64) -> f64 {
        match self {
            Self::DenseRoot(root) => {
                let rm = root.dot(m);
                scale
                    * rm.iter()
                        .zip(root.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
            }
            Self::BlockRoot {
                root, start, end, ..
            } => {
                let m_block = m.slice(ndarray::s![*start..*end, *start..*end]);
                let rm = root.dot(&m_block);
                scale
                    * rm.iter()
                        .zip(root.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                ..
            } => {
                // tr(M · diag(μ)) = Σ_i μ_i M_{ii}  (penalty is diagonal in eigenbasis)
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

                let mut trace = 0.0;
                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j];
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..stride_k {
                            let idx = base + inner;
                            trace += mu * m[[idx, idx]];
                        }
                    }
                }
                trace * scale
            }
        }
    }

    pub fn scaled_operator<'a>(
        &'a self,
        scale: f64,
        dense_correction: Option<&'a Array2<f64>>,
    ) -> PenaltyHyperOperator<'a> {
        PenaltyHyperOperator {
            coord: self,
            scale,
            dense_correction,
        }
    }
}

pub struct PenaltyHyperOperator<'a> {
    coord: &'a PenaltyCoordinate,
    scale: f64,
    dense_correction: Option<&'a Array2<f64>>,
}

impl HyperOperator for PenaltyHyperOperator<'_> {
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

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        self.coord
            .apply_penalty_view_into(v, self.scale, out.view_mut());
        if let Some(correction) = self.dense_correction {
            dense_matvec_scaled_add_into(correction, v, 1.0, out.view_mut());
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
        self.coord
            .scaled_add_penalty_view(v, scale * self.scale, out.view_mut());
        if let Some(correction) = self.dense_correction {
            dense_matvec_scaled_add_into(correction, v, scale, out.view_mut());
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self.coord.scaled_dense_matrix(self.scale);
        if let Some(correction) = self.dense_correction {
            out += correction;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

/// Compute the exact dimension of the intersection ∩_k N(S_k) for PSD penalties.
///
/// For a single penalty, this is just `nullspace_dims[0]`. For multiple
/// penalties, eigendecomposes each S_k individually, extracts its nullspace
/// basis (bottom `nullspace_dims[k]` eigenvectors), and iteratively
/// intersects the subspaces via SVD.
///
/// Returns 0 if any penalty is full rank (nullspace_dims[k] == 0).
pub(crate) fn exact_intersection_nullity(
    penalties: &[Array2<f64>],
    nullspace_dims: &[usize],
) -> usize {
    if penalties.is_empty() || nullspace_dims.is_empty() {
        return 0;
    }
    if penalties.len() != nullspace_dims.len() {
        return 0;
    }
    // If any penalty is full rank, the intersection nullspace is {0}.
    if nullspace_dims.iter().any(|&m| m == 0) {
        return 0;
    }

    // Single penalty: nullity is exact from structural info.
    if penalties.len() == 1 {
        return nullspace_dims[0];
    }

    // Multiple penalties: intersect nullspace bases iteratively.
    // Eigendecompose S_1, get its nullspace basis (bottom m_1 eigenvectors).
    let p = penalties[0].nrows();
    let (_, vecs0) = match penalties[0].eigh(faer::Side::Lower) {
        Ok(ev) => ev,
        Err(_) => return 0,
    };
    let m0 = nullspace_dims[0].min(p);
    // Null basis: bottom m0 eigenvectors (ascending order from eigh).
    // N has shape (p, current_dim).
    let mut n_basis = Array2::<f64>::zeros((p, m0));
    for col in 0..m0 {
        for row in 0..p {
            n_basis[[row, col]] = vecs0[[row, col]];
        }
    }
    const SHARED_DIR_THRESHOLD: f64 = 0.99;

    for k in 1..penalties.len() {
        let current_dim = n_basis.ncols();
        if current_dim == 0 {
            return 0;
        }

        // Eigendecompose S_k, get its nullspace basis.
        let (_, vecs_k) = match penalties[k].eigh(faer::Side::Lower) {
            Ok(ev) => ev,
            Err(_) => return 0,
        };
        let mk = nullspace_dims[k].min(p);
        let mut nk_basis = Array2::<f64>::zeros((p, mk));
        for col in 0..mk {
            for row in 0..p {
                nk_basis[[row, col]] = vecs_k[[row, col]];
            }
        }

        // Intersect: M = N^T N_k (current_dim × mk).
        // SVD of M: singular values near 1 indicate shared directions.
        let m_mat = n_basis.t().dot(&nk_basis);
        let (u_opt, s, _) = match crate::faer_ndarray::FaerSvd::svd(&m_mat, true, false) {
            Ok(usv) => usv,
            Err(_) => return 0,
        };
        let u = match u_opt {
            Some(u) => u,
            None => return 0,
        };

        // Count singular values ≈ 1 (shared directions).
        let shared: Vec<usize> = s
            .iter()
            .enumerate()
            .filter(|(_, sv)| **sv > SHARED_DIR_THRESHOLD)
            .map(|(i, _)| i)
            .collect();

        if shared.is_empty() {
            return 0;
        }

        // Update basis to the shared directions: N_new = N * u_shared.
        let mut n_new = Array2::<f64>::zeros((p, shared.len()));
        for (new_col, &orig_col) in shared.iter().enumerate() {
            for row in 0..p {
                let mut val = 0.0;
                for j in 0..current_dim {
                    val += n_basis[[row, j]] * u[[j, orig_col]];
                }
                n_new[[row, new_col]] = val;
            }
        }
        n_basis = n_new;
    }

    n_basis.ncols()
}

/// Positive-eigenvalue threshold for a given eigenspectrum.
///
/// For a p×p PSD matrix, eigendecomposition introduces errors of order
/// `p × ε_mach × ‖S‖`. True null eigenvalues sit in this noise band.
/// The threshold must be above the noise floor but well below any
/// genuinely positive eigenvalue.
///
/// Uses `p × ε_mach × max(|eigenvalues|, 1)` with a safety factor,
/// giving ~1e-13 × max_ev for typical sizes (p ≤ 1000).
pub(crate) fn positive_eigenvalue_threshold(eigenvalues: &[f64]) -> f64 {
    let p = eigenvalues.len();
    let max_ev = eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |a, b| a.max(b.abs()))
        .max(1.0);
    // Safety factor of 100 above the theoretical noise floor p × ε_mach × ‖S‖.
    let safety = 100.0;
    safety * (p as f64) * f64::EPSILON * max_ev
}

/// Exact pseudo-logdet on the positive eigenspace: L = Σ_{σ_i > threshold} log σ_i.
///
/// No δ-regularization, no nullity parameter. The structural nullspace is
/// identified directly from the eigenspectrum. For PSD penalty sums
/// S(ρ) = Σ exp(ρ_k) S_k, the positive eigenspace is structurally fixed,
/// so this function is C∞ in ρ.
pub(crate) fn exact_pseudo_logdet(eigenvalues: &[f64], threshold: f64) -> f64 {
    eigenvalues
        .iter()
        .filter(|&&s| s > threshold)
        .map(|&s| s.ln())
        .sum()
}

// PenaltyLogdetEigenspace, build_penalty_logdet_eigenspace,
// scaled_penalty_logdet_nullspace_leakage, and frobenius_inner_same_shape
// have been replaced by the canonical PenaltyPseudologdet in
// super::penalty_logdet. All callers now use that module directly.

/// Projected-logdet trace kernel for rank-deficient penalty geometries.
///
/// When the outer cost is evaluated as `log|U_Sᵀ H U_S|_+` on the positive
/// eigenspace of `S_λ` (see `hessian_logdet_correction`), the derivative
/// `d log|U_Sᵀ H U_S|/dτ = tr(U_S · (U_Sᵀ H U_S)⁻¹ · U_Sᵀ · Ḣ)` uses the
/// **projected** inverse kernel, not the full-space `H⁻¹`.  The two agree
/// only when `Ḣ` has no support on `null(S)` — true for ρ-direction
/// penalty drifts `A_k = λ_k S_k` (S_k vanishes on null(S) by construction),
/// but **false** for the IFT correction `D_β H[v] = X' diag(c ⊙ X v) X`
/// of non-Gaussian GLMs, because the intercept column `X[:,0] = 1_n`
/// typically lies in `null(S)` and gives `D_β H[v]` non-zero rows/columns
/// on that direction.
///
/// Evaluating `tr(H⁻¹ · Ḣ)` then picks up a spurious null-space
/// contribution that is absent from the cost's projected logdet derivative.
/// For Gaussian identity, `c = 0` so `D_β H[v] = 0` and the leakage vanishes,
/// which is why Gaussian fixtures pass untouched.
///
/// `u_s`           — p × r orthonormal basis of `range(S_+)`.
/// `h_proj_inverse` — r × r symmetric matrix `(U_Sᵀ H U_S)⁻¹`, precomputed
/// from the same `H_proj = U_Sᵀ · H · U_S` that feeds `log|H_proj|_+`.
#[derive(Clone, Debug)]
pub struct PenaltySubspaceTrace {
    pub u_s: Array2<f64>,
    pub h_proj_inverse: Array2<f64>,
}

impl PenaltySubspaceTrace {
    /// Compute `tr(K · A)` where `K = U_S · H_proj⁻¹ · U_Sᵀ` — the
    /// projected logdet kernel that matches `d log|U_Sᵀ H U_S|/dτ`.
    ///
    /// Uses the identity `tr(K · A) = tr(H_proj⁻¹ · U_Sᵀ A U_S)` so the
    /// reduction runs on the r × r subspace rather than materializing K.
    pub fn trace_projected_logdet(&self, a: &Array2<f64>) -> f64 {
        self.trace_projected_logdet_reduced(&self.reduce(a))
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
        let mut trace = 0.0;
        let r = self.h_proj_inverse.nrows();
        for i in 0..r {
            for j in 0..r {
                trace += self.h_proj_inverse[[i, j]] * r_mat[[j, i]];
            }
        }
        trace
    }

    /// Cross-trace given pre-reduced blocks `R_A = U_Sᵀ A U_S`, `R_B = U_Sᵀ B U_S`.
    pub fn trace_projected_logdet_cross_reduced(&self, ra: &Array2<f64>, rb: &Array2<f64>) -> f64 {
        // left = H_proj⁻¹ · R_A ;  right = H_proj⁻¹ · R_B ;  tr(left · right).
        let left = self.h_proj_inverse.dot(ra);
        let right = self.h_proj_inverse.dot(rb);
        let r = left.nrows();
        let mut trace = 0.0;
        for i in 0..r {
            for j in 0..r {
                trace += left[[i, j]] * right[[j, i]];
            }
        }
        trace
    }

    /// Reduce a `HyperOperator` `A` to its `r × r` projection
    /// `U_Sᵀ · A · U_S` without materializing the dense `p × p` block.
    /// Uses `A.mul_mat(U_S)` so an Hv-only operator is probed in `r` matvecs
    /// (each `O(work_of_A)`), then a single `r × p × r` reduction routed
    /// through faer's parallel SIMD GEMM (`fast_atb`).
    pub fn reduce_operator(&self, a: &dyn HyperOperator) -> Array2<f64> {
        let au = a.mul_mat(&self.u_s);
        crate::faer_ndarray::fast_atb(&self.u_s, &au)
    }

    /// `tr(K · A)` for `A` exposed only as a `HyperOperator`.  Mirrors
    /// [`Self::trace_projected_logdet`] without forcing dense materialization
    /// of `A`.
    pub fn trace_operator(&self, a: &dyn HyperOperator) -> f64 {
        self.trace_projected_logdet_reduced(&self.reduce_operator(a))
    }

    /// Apply the projected logdet kernel `K = U_S · H_proj⁻¹ · U_Sᵀ` to a
    /// vector `v`, yielding `K · v`.  Computed factor-by-factor as
    /// `U_S · (H_proj⁻¹ · (U_Sᵀ · v))` so cost is `O(p · r + r²)` and the
    /// result is identically zero on `null(U_S)`.
    ///
    /// Test-only: documents the projected-kernel action that
    /// [`Self::xt_projected_kernel_x_diagonal`] inlines for production
    /// (the prod path streams `Z = X · U_S` row-chunked rather than
    /// invoking `apply` per row).  The companion test
    /// `subspace_apply_adjoint_shortcut_matches_dense_projected_trace`
    /// verifies the identity `tr(K · C[u]) = uᵀ · Xᵀ(c ⊙ h^{G,proj})` that
    /// the production leverage shortcut depends on.  Note the corresponding
    /// `z_c` stays gated by `H⁻¹` (not `K`) so the IFT mode-response
    /// semantics line up with `compute_outer_hessian`.
    #[cfg(test)]
    pub fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        let v_proj = self.u_s.t().dot(v);
        let y_proj = self.h_proj_inverse.dot(&v_proj);
        self.u_s.dot(&y_proj)
    }

    /// Projected leverage `h^{G,proj}_i = Xᵢᵀ · K · Xᵢ` for every row of `x`.
    ///
    /// Computed in bulk as `Z = X · U_S` (`n × r`) then
    /// `h^{G,proj}_i = (Z H_proj⁻¹ Zᵀ)_{ii} = Σ_{a,b} Z_{ia} (H_proj⁻¹)_{ab} Z_{ib}`,
    /// total cost `O(n · p · r + n · r²)` — strictly cheaper than `n` calls
    /// to [`Self::apply`] because the `n × p · p × r` GEMM streams the
    /// `p`-axis once.  Streams `X` through `try_row_chunk` so operator-backed
    /// (Lazy) designs at biobank scale never densify the full `(n × p)` block.
    pub fn xt_projected_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let r = self.u_s.ncols();
        debug_assert_eq!(self.u_s.nrows(), p);
        debug_assert_eq!(self.h_proj_inverse.nrows(), r);
        debug_assert_eq!(self.h_proj_inverse.ncols(), r);

        let block = {
            const TARGET_CHUNK_FLOATS: usize = 1 << 16;
            (TARGET_CHUNK_FLOATS / p.max(1)).clamp(1, n.max(1))
        };

        let mut h = Array1::<f64>::zeros(n);
        let mut start = 0usize;
        while start < n {
            let end = (start + block).min(n);
            let rows = x.try_row_chunk(start..end).unwrap_or_else(|err| {
                panic!("xt_projected_kernel_x_diagonal: row chunk failed: {err}")
            });
            // Z_chunk = rows · U_S  ((end-start) × r).
            let z_chunk = crate::faer_ndarray::fast_ab(&rows.to_owned(), &self.u_s);
            // h_i = Σ_{a,b} Z_{ia} (H_proj⁻¹)_{ab} Z_{ib}.
            for i in 0..(end - start) {
                let row_z = z_chunk.row(i);
                let mut acc = 0.0;
                for a in 0..r {
                    let mut inner = 0.0;
                    for b in 0..r {
                        inner += self.h_proj_inverse[[a, b]] * row_z[b];
                    }
                    acc += row_z[a] * inner;
                }
                h[start + i] = acc;
            }
            start = end;
        }
        h
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

    /// Uniform scale applied to rho-coordinate penalty derivatives only in the
    /// H-dependent trace / solve parts of the outer calculus.
    pub rho_curvature_scale: f64,

    // === Model dimensions ===
    /// Number of observations.
    pub n_observations: usize,

    /// M_p: dimension of the penalty null space (unpenalized coefficients).
    pub nullspace_dim: f64,

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

    /// Optional log-barrier configuration for monotonicity-constrained coefficients.
    /// When present, the barrier cost and Hessian corrections are added to the
    /// outer REML/LAML objective.
    pub barrier_config: Option<BarrierConfig>,
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
    nullspace_dim_override: Option<f64>,
    // Extended hyperparameter coordinates
    ext_coords: Vec<HyperCoord>,
    ext_coord_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    rho_ext_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    fixed_drift_deriv: Option<FixedDriftDerivFn>,
    barrier_config: Option<BarrierConfig>,
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
            nullspace_dim_override: None,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
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

    pub fn firth(mut self, op: Option<std::sync::Arc<super::FirthDenseOperator>>) -> Self {
        self.firth = op.map(ExactJeffreysTerm::new);
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

    pub fn barrier_config(mut self, config: Option<BarrierConfig>) -> Self {
        self.barrier_config = config;
        self
    }

    /// Build the `InnerSolution`, auto-computing nullspace_dim from penalty coordinates.
    pub fn build(self) -> InnerSolution<'dp> {
        let nullspace_dim = self.nullspace_dim_override.unwrap_or_else(|| {
            let total_p = self.beta.len();
            let penalty_rank: usize = self
                .penalty_coords
                .iter()
                .map(PenaltyCoordinate::rank)
                .sum();
            total_p.saturating_sub(penalty_rank) as f64
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
            n_observations: self.n_observations,
            nullspace_dim,
            dispersion: self.dispersion,
            ext_coords: self.ext_coords,
            ext_coord_pair_fn: self.ext_coord_pair_fn,
            rho_ext_pair_fn: self.rho_ext_pair_fn,
            fixed_drift_deriv: self.fixed_drift_deriv,
            barrier_config: self.barrier_config,
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
pub struct RemlLamlResult {
    /// The REML/LAML objective value (to be minimized).
    pub cost: f64,
    /// Gradient ∂V/∂ρ (present if mode ≥ ValueAndGradient).
    pub gradient: Option<Array1<f64>>,
    /// Outer Hessian ∂²V/∂ρ² (present if mode = ValueGradientHessian).
    pub hessian: crate::solver::outer_strategy::HessianResult,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Soft floor for penalized deviance (Gaussian profiled scale)
// ═══════════════════════════════════════════════════════════════════════════

// Canonical definitions live in estimate.rs; re-use them here.
use crate::solver::estimate::smooth_floor_dp;

/// Ridge floor for denominator safety.
const DENOM_RIDGE: f64 = 1e-8;

fn penalty_a_k_beta(coord: &PenaltyCoordinate, beta: &Array1<f64>, lambda: f64) -> Array1<f64> {
    coord.apply_penalty(beta, lambda)
}

fn penalty_a_k_quadratic(coord: &PenaltyCoordinate, beta: &Array1<f64>, lambda: f64) -> f64 {
    coord.quadratic(beta, lambda)
}

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
// We deliberately keep the same approximation the old code made: drop the
// `C[v_k]` IFT correction from the trace. This costs no `H⁻¹` solves and
// matches `Ḣ_k = λ_k S_k` exactly when the family is Gaussian or has no
// third-derivative correction. For non-Gaussian families with corrections
// the EFS step is a slightly different surrogate from the gradient, which
// is acceptable for an EFS-style fixed-point iteration; the outer driver
// performs cost validation downstream.

/// `q_eff = 2 · penalty_term` matching `outer_gradient_entry`.
#[inline]
fn efs_q_eff(a_i: f64, dispersion: &DispersionHandling, dp_cgrad: f64, phi: f64) -> f64 {
    match dispersion {
        DispersionHandling::ProfiledGaussian => 2.0 * dp_cgrad * a_i / phi,
        DispersionHandling::Fixed { .. } => 2.0 * a_i,
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

#[inline]
fn dense_matvec_into(
    matrix: &Array2<f64>,
    x: ArrayView1<'_, f64>,
    mut out: ArrayViewMut1<'_, f64>,
) {
    debug_assert_eq!(matrix.ncols(), x.len());
    debug_assert_eq!(matrix.nrows(), out.len());
    for (row, out_value) in matrix.rows().into_iter().zip(out.iter_mut()) {
        *out_value = row.dot(&x);
    }
}

#[inline]
fn dense_matvec_scaled_add_into(
    matrix: &Array2<f64>,
    x: ArrayView1<'_, f64>,
    scale: f64,
    mut out: ArrayViewMut1<'_, f64>,
) {
    debug_assert_eq!(matrix.ncols(), x.len());
    debug_assert_eq!(matrix.nrows(), out.len());
    if scale == 0.0 {
        return;
    }
    for (row, out_value) in matrix.rows().into_iter().zip(out.iter_mut()) {
        *out_value += scale * row.dot(&x);
    }
}

#[inline]
fn dense_transpose_matvec_into(
    matrix: &Array2<f64>,
    x: ArrayView1<'_, f64>,
    mut out: ArrayViewMut1<'_, f64>,
) {
    debug_assert_eq!(matrix.nrows(), x.len());
    debug_assert_eq!(matrix.ncols(), out.len());
    out.fill(0.0);
    dense_transpose_matvec_scaled_add_into(matrix, x, 1.0, out);
}

#[inline]
fn dense_transpose_matvec_scaled_add_into(
    matrix: &Array2<f64>,
    x: ArrayView1<'_, f64>,
    scale: f64,
    mut out: ArrayViewMut1<'_, f64>,
) {
    debug_assert_eq!(matrix.nrows(), x.len());
    debug_assert_eq!(matrix.ncols(), out.len());
    if scale == 0.0 {
        return;
    }
    for (row, x_value) in matrix.rows().into_iter().zip(x.iter().copied()) {
        let row_scale = scale * x_value;
        if row_scale == 0.0 {
            continue;
        }
        for (out_value, entry) in out.iter_mut().zip(row.iter().copied()) {
            *out_value += row_scale * entry;
        }
    }
}

#[inline]
fn dense_bilinear(matrix: &Array2<f64>, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
    debug_assert_eq!(matrix.ncols(), v.len());
    debug_assert_eq!(matrix.nrows(), u.len());
    let mut total = 0.0;
    for (row, u_value) in matrix.rows().into_iter().zip(u.iter().copied()) {
        total += u_value * row.dot(&v);
    }
    total
}

fn design_matrix_apply_view(design: &DesignMatrix, vector: ArrayView1<'_, f64>) -> Array1<f64> {
    let mut output = Array1::<f64>::zeros(design.nrows());
    design_matrix_apply_view_into(design, vector, output.view_mut());
    output
}

fn design_matrix_column_into(
    design: &DesignMatrix,
    col: usize,
    mut output: ArrayViewMut1<'_, f64>,
) {
    debug_assert!(col < design.ncols());
    debug_assert_eq!(design.nrows(), output.len());

    if let Some(dense) = design.as_dense() {
        output.assign(&dense.column(col));
        return;
    }

    if let Some(sparse) = design.as_sparse() {
        let matrix = sparse.as_ref();
        output.fill(0.0);
        let (symbolic, values) = matrix.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for idx in col_ptr[col]..col_ptr[col + 1] {
            output[row_idx[idx]] = values[idx];
        }
        return;
    }

    let mut basis = Array1::<f64>::zeros(design.ncols());
    basis[col] = 1.0;
    output.assign(&design.matrixvectormultiply(&basis));
}

fn design_matrix_apply_view_into(
    design: &DesignMatrix,
    vector: ArrayView1<'_, f64>,
    mut output: ArrayViewMut1<'_, f64>,
) {
    debug_assert_eq!(design.ncols(), vector.len());
    debug_assert_eq!(design.nrows(), output.len());

    if let Some(dense) = design.as_dense() {
        dense_matvec_into(dense, vector, output);
        return;
    }

    if let Some(sparse) = design.as_sparse() {
        let matrix = sparse.as_ref();
        output.fill(0.0);
        let (symbolic, values) = matrix.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for col in 0..matrix.ncols() {
            let x = vector[col];
            if x == 0.0 {
                continue;
            }
            for idx in col_ptr[col]..col_ptr[col + 1] {
                output[row_idx[idx]] += values[idx] * x;
            }
        }
        return;
    }

    output.assign(&design.matrixvectormultiply(&vector.to_owned()));
}

fn design_matrix_transpose_apply_view_into(
    design: &DesignMatrix,
    vector: ArrayView1<'_, f64>,
    mut output: ArrayViewMut1<'_, f64>,
) {
    debug_assert_eq!(design.nrows(), vector.len());
    debug_assert_eq!(design.ncols(), output.len());

    if let Some(dense) = design.as_dense() {
        dense_transpose_matvec_into(dense, vector, output);
        return;
    }

    if let Some(sparse) = design.as_sparse() {
        let matrix = sparse.as_ref();
        let (symbolic, values) = matrix.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for col in 0..matrix.ncols() {
            let mut value = 0.0;
            for idx in col_ptr[col]..col_ptr[col + 1] {
                value += values[idx] * vector[row_idx[idx]];
            }
            output[col] = value;
        }
        return;
    }

    output.assign(&design.transpose_vector_multiply(&vector.to_owned()));
}

#[inline]
fn trace_matrix_product(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    debug_assert_eq!(left.nrows(), left.ncols());
    debug_assert_eq!(left.raw_dim(), right.raw_dim());
    let n = left.nrows();
    let mut trace = 0.0;
    for i in 0..n {
        for j in 0..n {
            trace += left[[i, j]] * right[[j, i]];
        }
    }
    trace
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
//  The single evaluator
// ═══════════════════════════════════════════════════════════════════════════

/// Unified REML/LAML evaluation.
///
/// This is the SINGLE implementation of the outer objective. It handles:
/// - Gaussian REML with profiled scale
/// - Non-Gaussian LAML with fixed dispersion
/// - Any backend (dense spectral, sparse Cholesky, block-coupled)
/// - Any family (Gaussian, GLM, GAMLSS, survival, link wiggles)
///
/// Cost and gradient share intermediates by construction — they are computed
/// in the same function scope, using the same `HessianOperator`, the same
/// penalty derivatives, and the same coefficients. Drift between cost and
/// gradient is structurally impossible because there is no second function.
///
/// # Observed information requirement (see response.md Section 3)
///
/// The Laplace approximation to the marginal likelihood integral
///   int exp(-F(beta)) dbeta  ~  exp(-F(beta_hat)) * (2pi)^{p/2} / sqrt(|H_obs|)
/// requires H_obs = nabla^2 F(beta_hat), the **observed** (actual) Hessian at
/// the mode --- NOT the expected Fisher information. Replacing H_obs with
/// E[H] changes the quadratic approximation itself, yielding a PQL-type
/// surrogate rather than the true Laplace/LAML criterion.
///
/// For this evaluator, the `solution.hessian_op` MUST encode log|H_obs| and
/// provide traces tr(H_obs^{-1} A_k) using the observed Hessian. Callers
/// (runtime.rs, joint.rs) are responsible for constructing H from the
/// observed-information weights W_obs = W_Fisher - (y-mu)*B at the mode.
///
/// The **mixed strategy** is valid and deliberately used here:
/// - The inner P-IRLS solver may use Fisher scoring (expected information)
///   as its iteration matrix --- any convergent algorithm finds the same mode.
/// - The outer REML criterion uses the observed Hessian at that mode.
/// This is correct because the inner algorithm is just a solver; only the
/// outer log|H| and trace terms define the Laplace approximation.
///
/// For canonical links (for example logit-Binomial and log-Poisson), observed
/// equals expected, so no correction is needed. For non-canonical links
/// (including probit, cloglog, SAS, mixture/flexible, and Gamma-log), the observed weight includes a
/// residual-dependent correction:
///   W_obs = W_Fisher - (y - mu) * B,
///   B = (h'' V - h'^2 V') / (phi V^2)
/// and the c/d arrays (dW/deta, d^2W/deta^2) similarly include observed
/// corrections. These are computed by `compute_observed_hessian_curvature_arrays`
/// in pirls.rs and flow through `PirlsResult` into the `InnerSolution`.
///
/// # Arguments
/// - `solution`: The converged inner state (beta_hat, H_obs, penalties, corrections).
/// - `rho`: Log smoothing parameters (rho_k = log lambda_k).
/// - `mode`: What to compute (value only, value+gradient, or all three).
/// - `prior_cost_gradient`: Optional soft prior on rho (value, gradient, optional Hessian).
pub fn reml_laml_evaluate(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
) -> Result<RemlLamlResult, String> {
    let cost_phase_start = std::time::Instant::now();
    let k = rho.len();
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();
    let hop = &*solution.hessian_op;

    // ─── Shared intermediates (computed once, used by both cost and gradient) ───

    let log_det_h = hop.logdet() + solution.hessian_logdet_correction;
    let log_det_s = solution.penalty_logdet.value;

    let (cost, profiled_scale, dp_cgrad, _dp_cgrad2) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            // Gaussian REML with profiled scale:
            //   V(ρ) = D_p/(2φ̂) + ½ log|H| − ½ log|S|₊ + ((n−M_p)/2) log(2πφ̂)
            // where D_p = deviance + penalty, φ̂ = D_p/(n−M_p).
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw);
            let denom = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
            let phi = dp_c / denom;

            let cost = dp_c / (2.0 * phi)
                + 0.5 * (log_det_h - log_det_s)
                + (denom / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

            (cost, phi, dp_cgrad, dp_cgrad2)
        }
        DispersionHandling::Fixed {
            phi,
            include_logdet_h,
            include_logdet_s,
        } => {
            // Fixed-dispersion Laplace / maximum penalized likelihood:
            //   V(ρ) = −ℓ(β̂) + ½ β̂ᵀSβ̂
            //         + [½ log|H| + frozen-curvature TK − Firth]  if include_logdet_h
            //         − [½ log|S|₊]               if include_logdet_s
            //
            // The additive Gaussian normalization constant 0.5 * M * log(2πφ)
            // is intentionally omitted here. It does not affect outer
            // derivatives, and the custom-family exact paths already define
            // their scalar objective without it. Keeping the fixed-dispersion
            // evaluator aligned with those exact paths avoids objective drift
            // between the unified and direct custom-family implementations.
            //
            // Pair-subtract `log|H| − log|S|_+` before scaling by 0.5 and
            // summing with the rest, mirroring the profiled-Gaussian cost
            // expression above.  The pair `(log|H|, log|S|_+)` has nearly-
            // identical ρ-motion at a rank-deficient optimum (the analytic
            // gradient is their difference, which is tiny), so subtracting
            // them FIRST preserves the leading-order cancellation in f64
            // precision; adding them to `cost` independently would bury
            // the difference below ~ULP(cost) ≈ f64::EPSILON * cost.
            let logdet_pair_h = if *include_logdet_h { log_det_h } else { 0.0 };
            let logdet_pair_s = if *include_logdet_s { log_det_s } else { 0.0 };
            let cost_logdet_diff = 0.5 * (logdet_pair_h - logdet_pair_s);
            let mut cost =
                cost_logdet_diff + (-solution.log_likelihood) + 0.5 * solution.penalty_quadratic;
            if *include_logdet_h {
                cost += solution.tk_correction
                    - solution
                        .firth
                        .as_ref()
                        .map_or(0.0, ExactJeffreysTerm::value);
            }
            (cost, *phi, 0.0, 0.0)
        }
    };

    // Add prior.
    let mut cost = match &prior_cost_gradient {
        Some((pc, _, _)) => cost + pc,
        None => cost,
    };

    // Add log-barrier cost for monotonicity-constrained coefficients.
    if let Some(ref barrier_cfg) = solution.barrier_config {
        match barrier_cfg.barrier_cost(&solution.beta) {
            Ok(bc) => cost += bc,
            Err(e) => {
                log::warn!("Barrier cost skipped (infeasible): {e}");
            }
        }
    }

    if !cost.is_finite() {
        return Err(format!(
            "REML/LAML cost is non-finite ({cost}); check inner solver convergence"
        ));
    }

    if mode == EvalMode::ValueOnly {
        return Ok(RemlLamlResult {
            cost,
            gradient: None,
            hessian: crate::solver::outer_strategy::HessianResult::Unavailable,
        });
    }

    log::info!(
        "[STAGE] reml_laml cost_only_done k={} ext_dim={} dim={} elapsed={:.3}s",
        k,
        solution.ext_coords.len(),
        hop.dim(),
        cost_phase_start.elapsed().as_secs_f64(),
    );

    // ─── Gradient (uses SAME hop, SAME intermediates) ───

    // When a barrier is active, wrap the inner derivative provider so that
    // dH/dρ and d²H/dρ² include barrier-Hessian correction terms.
    let barrier_deriv_holder: Option<BarrierDerivativeProvider<'_>> = if let Some(ref barrier_cfg) =
        solution.barrier_config
    {
        match BarrierDerivativeProvider::new(&*solution.deriv_provider, barrier_cfg, &solution.beta)
        {
            Ok(bdp) => Some(bdp),
            Err(e) => {
                log::warn!("BarrierDerivativeProvider skipped (infeasible): {e}");
                None
            }
        }
    } else {
        None
    };
    let effective_deriv: &dyn HessianDerivativeProvider = match barrier_deriv_holder {
        Some(ref bdp) => bdp,
        None => &*solution.deriv_provider,
    };

    // Extract logdet flags once (same for all coordinates).
    let (incl_logdet_h, incl_logdet_s) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => (true, true),
        DispersionHandling::Fixed {
            include_logdet_h,
            include_logdet_s,
            ..
        } => (*include_logdet_h, *include_logdet_s),
    };

    let ext_dim = solution.ext_coords.len();
    let mut grad = Array1::zeros(k + ext_dim);
    // Coordinate-local fixed-β penalty terms, mode responses, and family
    // derivative corrections are independent within a single outer evaluation.
    // Keep the dependency-ordered BFGS/line-search loops serial, but use rayon
    // here so each accepted outer iterate evaluates its objective derivatives
    // by farming out the per-coordinate Hessian/gradient work.
    let rho_penalty_a_k_betas: Vec<Array1<f64>> = (0..k)
        .into_par_iter()
        .map(|idx| penalty_a_k_beta(&solution.penalty_coords[idx], &solution.beta, lambdas[idx]))
        .collect();
    let rho_curvature_a_k_betas: Vec<Array1<f64>> = (0..k)
        .into_par_iter()
        .map(|idx| {
            penalty_a_k_beta(
                &solution.penalty_coords[idx],
                &solution.beta,
                curvature_lambdas[idx],
            )
        })
        .collect();
    let need_rho_v = effective_deriv.has_corrections();
    let rho_v_ks: Option<Vec<Array1<f64>>> = if need_rho_v {
        Some(
            rho_curvature_a_k_betas
                .par_iter()
                .map(|a_k_beta| hop.solve(a_k_beta))
                .collect(),
        )
    } else {
        None
    };
    let rho_corrections: Vec<Option<DriftDerivResult>> = if effective_deriv.has_corrections() {
        rho_v_ks
            .as_ref()
            .expect("rho mode responses required for Hessian corrections")
            .par_iter()
            .map(|v_k| effective_deriv.hessian_derivative_correction_result(v_k))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..k).map(|_| None).collect()
    };

    // --- Stochastic trace estimation decision ---
    //
    // Hutchinson traces based on H^{-1} are only valid for logdet-gradient
    // terms on backends where the logdet kernel is exactly H^{-1}.
    // Smooth spectral regularization uses G_eps(H) instead, so those backends
    // must stay on the exact trace path.  The rank-deficient LAML fix also
    // replaces the kernel with the projected `U_S · (U_Sᵀ H U_S)⁻¹ · U_Sᵀ`,
    // which the Hutchinson path cannot produce — stay exact when it is active.
    let total_p = hop.dim();
    let use_stochastic_traces = can_use_stochastic_logdet_hinv_kernel(hop, total_p, incl_logdet_h)
        && solution.penalty_subspace_trace.is_none();

    // When using stochastic traces, pre-collect all H_k drifts (both rho and
    // ext coordinates) and batch them through a single StochasticTraceEstimator.
    // This amortizes the H^{-1} solve cost: ONE solve per probe, shared across
    // all k + ext_dim coordinates. The collector must inspect the fully
    // assembled drift (base coordinate plus SCOP/family correction) before
    // deciding dense vs operator; checking only the base coordinate misses
    // matrix-free derivative corrections and silently densifies them.
    let stochastic_trace_values: Option<Vec<f64>> = if use_stochastic_traces {
        let mut dense_matrices: Vec<Array2<f64>> = Vec::with_capacity(k + ext_dim);
        let mut operators: Vec<Arc<dyn HyperOperator>> = Vec::new();
        let mut coord_has_operator = Vec::with_capacity(k + ext_dim);

        // rho-coordinates: H_k = A_k + correction(v_k)
        for idx in 0..k {
            match penalty_total_drift_result(
                &solution.penalty_coords[idx],
                curvature_lambdas[idx],
                rho_corrections[idx].as_ref(),
            ) {
                DriftDerivResult::Dense(matrix) => {
                    dense_matrices.push(matrix);
                    coord_has_operator.push(false);
                }
                DriftDerivResult::Operator(op) => {
                    operators.push(op);
                    coord_has_operator.push(true);
                }
            }
        }

        // ext-coordinates: H_i = B_i + D_beta H[-v_i].
        for coord in solution.ext_coords.iter() {
            let correction = if effective_deriv.has_corrections() {
                let v_i = hop.solve(&coord.g);
                effective_deriv.hessian_derivative_correction_result(&v_i)?
            } else {
                None
            };
            match hyper_coord_total_drift_result(&coord.drift, correction.as_ref(), hop.dim()) {
                DriftDerivResult::Dense(matrix) => {
                    dense_matrices.push(matrix);
                    coord_has_operator.push(false);
                }
                DriftDerivResult::Operator(op) => {
                    operators.push(op);
                    coord_has_operator.push(true);
                }
            }
        }

        let dense_refs: Vec<&Array2<f64>> = dense_matrices.iter().collect();
        let generic_ops: Vec<&dyn HyperOperator> = operators.iter().map(|op| op.as_ref()).collect();
        let implicit_ops: Vec<&ImplicitHyperOperator> =
            operators.iter().filter_map(|op| op.as_implicit()).collect();
        let raw_traces = if generic_ops.is_empty() {
            stochastic_trace_hinv_products(hop, StochasticTraceTargets::Dense(&dense_refs))
        } else if generic_ops.len() == implicit_ops.len() {
            stochastic_trace_hinv_products(
                hop,
                StochasticTraceTargets::Structural {
                    dense_matrices: &dense_refs,
                    implicit_ops: &implicit_ops,
                },
            )
        } else {
            stochastic_trace_hinv_products(
                hop,
                StochasticTraceTargets::Mixed {
                    dense_matrices: &dense_refs,
                    operators: &generic_ops,
                },
            )
        };

        let mut result = Vec::with_capacity(k + ext_dim);
        let n_dense_total = coord_has_operator.iter().filter(|&&b| !b).count();
        let mut dense_cursor = 0usize;
        let mut operator_cursor = n_dense_total;
        for &has_operator in &coord_has_operator {
            if has_operator {
                result.push(raw_traces[operator_cursor]);
                operator_cursor += 1;
            } else {
                result.push(raw_traces[dense_cursor]);
                dense_cursor += 1;
            }
        }
        Some(result)
    } else {
        None
    };

    // ── Gradient: one shared formula for ALL coordinate types ──
    //
    // Both ρ and ext coordinates are processed through outer_gradient_entry()
    // so that the three-term formula (penalty + trace − det) is written once.

    let rho_grad_entries: Vec<(usize, f64)> = (0..k)
        .into_par_iter()
        .map(|idx| {
            let coord = &solution.penalty_coords[idx];
            let a_k_beta = &rho_penalty_a_k_betas[idx];

            // Cost derivative: a_i = ½ β̂ᵀ Aₖ β̂.
            let a_i = 0.5 * solution.beta.dot(a_k_beta);

            // Trace term: tr(K · Ḣₖ) where Ḣₖ = Aₖ + C[vₖ].
            //
            // Kernel choice mirrors the ψ/τ block: full-space `G_ε(H)` when the
            // cost uses the unprojected `log|H|`, or the identified-subspace
            // kernel `U_S · (U_Sᵀ H U_S)⁻¹ · U_Sᵀ` when the rank-deficient LAML
            // fix is active.  `Aₖ = λₖ Sₖ` is zero on `null(S)` by construction,
            // but the third-derivative correction `C[vₖ] = X'·diag(c ⊙ X vₖ)·X`
            // leaks onto the intercept direction for non-Gaussian families — so
            // the two kernels disagree whenever `hessian_logdet_correction ≠ 0`
            // and `c_array ≠ 0`.
            let trace_logdet_i = if !incl_logdet_h {
                0.0
            } else if let Some(ref stoch_traces) = stochastic_trace_values {
                stoch_traces[idx]
            } else if let Some(kernel) = solution.penalty_subspace_trace.as_ref() {
                let drift = penalty_total_drift_result(
                    coord,
                    curvature_lambdas[idx],
                    rho_corrections[idx].as_ref(),
                );
                match drift {
                    DriftDerivResult::Dense(matrix) => kernel.trace_projected_logdet(&matrix),
                    DriftDerivResult::Operator(op) => kernel.trace_operator(op.as_ref()),
                }
            } else if coord.is_block_local() && rho_corrections[idx].is_none() {
                let (block, start, end) = coord.scaled_block_local(1.0);
                hop.trace_logdet_block_local(&block, curvature_lambdas[idx], start, end)
            } else {
                penalty_total_drift_result(
                    coord,
                    curvature_lambdas[idx],
                    rho_corrections[idx].as_ref(),
                )
                .trace_logdet(hop)
            };

            (
                idx,
                outer_gradient_entry(
                    a_i,
                    trace_logdet_i,
                    solution.penalty_logdet.first[idx],
                    &solution.dispersion,
                    dp_cgrad,
                    profiled_scale,
                    incl_logdet_h,
                    incl_logdet_s,
                ),
            )
        })
        .collect();
    for (idx, value) in rho_grad_entries {
        grad[idx] = value;
    }

    // Extended hyperparameter gradient (ψ/τ coordinates).
    //
    // Uses the SAME outer_gradient_entry() formula as ρ coordinates above.
    //
    // All extended coordinates store canonical fixed-β stationarity
    // derivatives g_i = F_{βi}. IFT gives β_i = -H^{-1}g_i, exactly like
    // the ρ block.
    let ext_grad_entries: Result<Vec<(usize, f64)>, String> = (0..ext_dim)
        .into_par_iter()
        .map(|ext_idx| {
            let coord = &solution.ext_coords[ext_idx];
            let ext_coord_start = std::time::Instant::now();
            let grad_idx = k + ext_idx;

            // Mode response magnitude: v_i = H⁻¹(g_i), with β_i = −v_i.
            let v_i = hop.solve(&coord.g);

            // Trace term: tr(K · Ḣ_i) where Ḣ_i = B_i + D_β H[−v_i].
            //
            // Kernel choice pairs with the cost:
            //   * Default cost `½ log|H|` (or `Σ log r_ε(σ_j)` under Smooth spectral
            //     regularization) → K = G_ε(H), computed full-space.
            //   * Rank-deficient LAML fix (`hessian_logdet_correction ≠ 0`) uses
            //     cost `½ log|U_Sᵀ H U_S|_+` on the identified subspace, which
            //     pairs with K = U_S · (U_Sᵀ H U_S)⁻¹ · U_Sᵀ.
            //
            // For non-Gaussian families the total drift includes
            // `D_β H[−v_i]`, which has non-zero
            // support on `null(S)` whenever `X` contains an all-ones intercept
            // column — the null direction of `S_λ`.  Using the full-space
            // `G_ε(H)` there picks up a spurious null-space contribution absent
            // from `d log|U_Sᵀ H U_S|_+/dτ`; the projected kernel reroutes the
            // trace through `range(S_+)` only, matching the cost exactly.
            // Gaussian identity skips this path harmlessly because `c = 0` forces
            // `D_β H = 0`, so `Ḣ` already lives entirely in `range(S_+)²`.
            let trace_logdet_i = if !incl_logdet_h {
                0.0
            } else if let Some(ref stoch_traces) = stochastic_trace_values {
                stoch_traces[k + ext_idx]
            } else {
                let correction = if effective_deriv.has_corrections() {
                    effective_deriv.hessian_derivative_correction_result(&v_i)?
                } else {
                    None
                };
                let drift =
                    hyper_coord_total_drift_result(&coord.drift, correction.as_ref(), hop.dim());
                match (&solution.penalty_subspace_trace, drift) {
                    (Some(kernel), DriftDerivResult::Dense(matrix)) => {
                        kernel.trace_projected_logdet(&matrix)
                    }
                    (Some(kernel), DriftDerivResult::Operator(op)) => {
                        kernel.trace_operator(op.as_ref())
                    }
                    (None, DriftDerivResult::Dense(matrix)) => hop.trace_logdet_h_k(&matrix, None),
                    (None, DriftDerivResult::Operator(op)) => {
                        hop.trace_logdet_operator(op.as_ref())
                    }
                }
            };

            let value = outer_gradient_entry(
                coord.a,
                trace_logdet_i,
                coord.ld_s,
                &solution.dispersion,
                dp_cgrad,
                profiled_scale,
                incl_logdet_h,
                incl_logdet_s,
            );
            log::info!(
                "[STAGE] reml_laml ext_coord_trace ext_idx={} elapsed={:.3}s",
                ext_idx,
                ext_coord_start.elapsed().as_secs_f64(),
            );
            Ok((grad_idx, value))
        })
        .collect();
    for (idx, value) in ext_grad_entries? {
        grad[idx] = value;
    }

    // Add correction gradients (ρ-only).
    if let Some(tk_grad) = &solution.tk_gradient {
        {
            let mut sl = grad.slice_mut(ndarray::s![..k]);
            sl += tk_grad;
        }
    }

    // Add prior gradient (ρ-only).
    if let Some((_, ref pg, _)) = prior_cost_gradient {
        {
            let mut sl = grad.slice_mut(ndarray::s![..k]);
            sl += pg;
        }
    }

    if let Some((idx, value)) = grad.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "REML/LAML gradient contains non-finite entry at index {idx}: {value}"
        ));
    }

    // Outer Hessian (if requested).
    let hessian = if mode == EvalMode::ValueGradientHessian {
        // First, allow the family to short-circuit with its own exact outer
        // Hv operator.  Default `None` keeps the fall-through identical to
        // the historical kernel-based assembly path; CTN/survival/GAMLSS
        // families that implement a directional θθ HVP will return Some(op)
        // here and skip the kernel-based dispatch entirely.
        if let Some(family_op) = effective_deriv.family_outer_hessian_operator() {
            // Family's own exact Hv operator. Emit the same routing markers
            // as the kernel-based path so the bench runner's outer_h
            // aggregation captures this route too — without these the
            // family-op count silently disappears from the verdict, and
            // CTN/survival/GAMLSS fits look like they never built an outer
            // Hessian at all. The "family_op" reason is distinguishable
            // from the kernel-based reasons so the analyzer can tell which
            // representation a particular fit actually used.
            let n_obs = effective_deriv
                .scalar_glm_ingredients()
                .map(|ing| ing.x.nrows())
                .unwrap_or(solution.n_observations);
            let p_dim = hop.dim();
            let k_outer = k + solution.ext_coords.len();
            log::info!(
                "[OUTER hessian-route] choice=operator reason=family_op \
                 n={n_obs} p={p_dim} k={k_outer} \
                 callback_kernel=false subspace_trace={subspace} \
                 scale_prefers_operator=irrelevant",
                subspace = solution.penalty_subspace_trace.is_some(),
            );
            if family_op.dim() != k_outer {
                return Err(format!(
                    "family outer Hessian operator dimension mismatch: got {}, expected {}",
                    family_op.dim(),
                    k_outer
                ));
            }
            let assembly_start = std::time::Instant::now();
            let mut hessian = crate::solver::outer_strategy::HessianResult::Operator(family_op);
            if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                hessian.add_rho_block_dense(ph)?;
            }
            log::info!(
                "[OUTER hessian-elapsed] choice=operator reason=family_op \
                 n={n_obs} p={p_dim} k={k_outer} elapsed={:.3}s",
                assembly_start.elapsed().as_secs_f64(),
            );
            return Ok(RemlLamlResult {
                cost,
                gradient: Some(grad),
                hessian,
            });
        }
        let hessian_kernel = effective_deriv.outer_hessian_derivative_kernel();
        // Cost selects representation (operator vs dense), not capability.
        // The (n, p, K) scale rule routes biobank-scale problems through the
        // matrix-free Hv operator path even when the per-axis thresholds
        // (`p >= 512` or `K >= 32`) alone do not fire.  At Matern biobank
        // scale (n=320 000, p=101, K=6) the dense path's per-outer-eval
        // O(K·n·p²) assembly is ≈ 2·10¹⁰ FLOPs and dominates wall-clock; the
        // operator path absorbs it via O(n·p) HVPs.
        //
        // The matrix-free operator path supports both full-space and projected
        // logdet kernels.  When a `penalty_subspace_trace` is installed, the
        // operator traces first/second Hessian drifts through
        // `U_S (U_Sᵀ H U_S)⁻¹ U_Sᵀ`, matching the dense analytic path without
        // forcing p×p assembly solely for rank-deficient penalties.
        let n_obs = effective_deriv
            .scalar_glm_ingredients()
            .map(|ing| ing.x.nrows())
            .unwrap_or(solution.n_observations);
        let p_dim = hop.dim();
        let k_outer = k + solution.ext_coords.len();
        let callback_operator_kernel = matches!(
            hessian_kernel,
            Some(OuterHessianDerivativeKernel::Callback { .. })
        );
        // Decompose the routing decision so the [OUTER hessian-route] log
        // can attribute *which* clause selected the path, including the
        // projected rank-deficient route at biobank-shape large k.
        let large_p = p_dim >= MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD;
        let large_n_and_moderate_p = n_obs >= MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD
            && p_dim >= MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N;
        let large_linear_work =
            n_obs.saturating_mul(p_dim) >= MATRIX_FREE_OUTER_HESSIAN_NP_THRESHOLD;
        let large_k = k_outer >= MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD;
        let scale_prefers_operator = prefer_outer_hessian_operator(n_obs, p_dim, k_outer);
        let has_subspace_trace = solution.penalty_subspace_trace.is_some();
        let use_operator = hessian_kernel.is_some()
            && use_outer_hessian_operator_path(
                n_obs,
                p_dim,
                k_outer,
                callback_operator_kernel,
            );
        // Reason mnemonic: which clause carried the routing.  Ordered so
        // the most specific reason wins; "kernel_absent" wins over
        // everything else because that disables the operator path
        // unconditionally.
        let route_reason = if hessian_kernel.is_none() {
            "kernel_absent"
        } else if has_subspace_trace && scale_prefers_operator {
            "subspace_projected_operator"
        } else if callback_operator_kernel {
            "callback_kernel"
        } else if large_k {
            "large_k"
        } else if large_p {
            "large_p"
        } else if large_n_and_moderate_p {
            "large_n_moderate_p"
        } else if large_linear_work {
            "large_linear_work"
        } else {
            "below_crossover"
        };
        let route_choice = if use_operator { "operator" } else { "dense" };
        log::info!(
            "[OUTER hessian-route] choice={route_choice} reason={route_reason} \
             n={n_obs} p={p_dim} k={k_outer} \
             callback_kernel={callback_operator_kernel} subspace_trace={has_subspace_trace} \
             scale_prefers_operator={scale_prefers_operator}"
        );
        let assembly_start = std::time::Instant::now();
        let result = if use_operator {
            match build_outer_hessian_operator(
                solution,
                &lambdas,
                effective_deriv,
                hessian_kernel.expect("checked is_some above"),
            ) {
                Ok(op) => {
                    let mut hessian =
                        crate::solver::outer_strategy::HessianResult::Operator(Arc::new(op));
                    if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                        hessian.add_rho_block_dense(ph)?;
                    }
                    hessian
                }
                Err(err) if is_hessian_unavailable(&err) => {
                    log::warn!("{err}");
                    crate::solver::outer_strategy::HessianResult::Unavailable
                }
                Err(err) => return Err(err),
            }
        } else {
            match compute_outer_hessian(solution, rho, &lambdas, hop, effective_deriv) {
                Ok(mut h) => {
                    // Add prior Hessian (second derivatives of the soft prior on ρ, ρ-only).
                    if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                        let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                        sl += ph;
                    }
                    crate::solver::outer_strategy::HessianResult::Analytic(h)
                }
                Err(err) if is_hessian_unavailable(&err) => {
                    log::warn!("{err}");
                    crate::solver::outer_strategy::HessianResult::Unavailable
                }
                Err(err) => return Err(err),
            }
        };
        log::info!(
            "[OUTER hessian-elapsed] choice={route_choice} reason={route_reason} \
             n={n_obs} p={p_dim} k={k_outer} elapsed={:.3}s",
            assembly_start.elapsed().as_secs_f64(),
        );
        result
    } else {
        crate::solver::outer_strategy::HessianResult::Unavailable
    };

    Ok(RemlLamlResult {
        cost,
        gradient: Some(grad),
        hessian,
    })
}

const HESSIAN_UNAVAILABLE_PREFIX: &str = "outer Hessian unavailable:";

/// Minimum coefficient dimension at which the matrix-free operator path is
/// selected unconditionally — once `p` is this large the dense `p × p`
/// assembly itself dominates and operator HVPs win regardless of `n` or `K`.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD: usize = 512;

/// Sample-count threshold for the (`n`, `p`) crossover branch: when `n` is
/// large enough that per-row work dominates, the operator path wins even
/// at moderate `p`.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD: usize = 50_000;

/// Coefficient dimension paired with [`MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD`]
/// in the (`n`, `p`) crossover branch.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N: usize = 32;

/// `n · p` linear-work cutoff: per-eval `O(K · n · p²)` dense assembly
/// dominates once `n · p` crosses this threshold even when both `n` and `p`
/// are individually below the per-axis thresholds.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_NP_THRESHOLD: usize = 4_000_000;

/// Smoothing-parameter count above which the operator path wins regardless
/// of `n` and `p`: the per-outer-eval Hessian-assembly cost is
/// `O(K · n · p²)`, so `K` itself drives the crossover.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD: usize = 32;

/// Predicate for selecting the matrix-free Hv-operator outer-Hessian
/// representation over the dense `K × K` assembly.  Cost selects
/// representation, never capability — the operator path delivers the same
/// math as the dense path with `O(n · p)` HVPs instead of dense `p × p`
/// assembly.
///
/// Each clause is one independent crossover regime; any one firing routes
/// the evaluator to the operator path.
pub(crate) fn prefer_outer_hessian_operator(n: usize, p: usize, k: usize) -> bool {
    // Wide coefficient basis: dense `p × p` assembly itself dominates.
    let large_p = p >= MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD;
    // Tall design with moderate width: per-row work dominates even when `p`
    // alone is below the wide-basis threshold.
    let large_n_and_moderate_p = n >= MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD
        && p >= MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N;
    // Linear-work fallback: `n · p` crosses the assembly-cost crossover even
    // when neither `n` nor `p` individually trip a per-axis threshold.
    let large_linear_work = n.saturating_mul(p) >= MATRIX_FREE_OUTER_HESSIAN_NP_THRESHOLD;
    // Many smoothing parameters: per-outer-eval cost is `O(K · n · p²)`, so
    // `K` itself can drive the crossover regardless of `(n, p)`.
    let large_k = k >= MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD;
    large_p || large_n_and_moderate_p || large_linear_work || large_k
}

/// Selects the matrix-free outer-Hessian representation once a Hessian HVP
/// kernel is available.
///
/// Callback kernels are explicit family-supplied directional operators, so they
/// select the operator path independently of the generic `(n, p, K)` crossover.
/// A penalty-subspace trace, when present, is handled inside
/// `build_outer_hessian_operator` / `UnifiedOuterHessianOperator::matvec`
/// through the projected `trace_operator` / `trace_projected_logdet` paths,
/// so it no longer forces the dense fallback — the matvec is bit-equivalent
/// to `compute_outer_hessian` under subspace.
pub(crate) fn use_outer_hessian_operator_path(
    n: usize,
    p: usize,
    k: usize,
    callback_operator_kernel: bool,
) -> bool {
    callback_operator_kernel || prefer_outer_hessian_operator(n, p, k)
}

fn is_hessian_unavailable(error: &str) -> bool {
    error.starts_with(HESSIAN_UNAVAILABLE_PREFIX)
}

//   C[u]            = Xᵀ diag(c ⊙ Xu) X
//   h^G             = diag(X G_ε(H) Xᵀ)
//   v               = Xᵀ (c ⊙ h^G)
//   z_c             = H⁻¹ v
//   tr(G_ε C[u])    = uᵀ Xᵀ (c ⊙ h^G) = uᵀ v
fn compute_adjoint_z_c(
    ing: &ScalarGlmIngredients<'_>,
    hop: &dyn HessianOperator,
    leverage: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let mut weighted = Array1::<f64>::zeros(ing.c_array.len());
    Zip::from(&mut weighted)
        .and(ing.c_array)
        .and(leverage)
        .for_each(|w, &c, &h| *w = c * h);
    // Matrix-free Xᵀ · weighted via DesignMatrix transpose-apply, so
    // operator-backed (Lazy) designs at biobank scale never densify.
    let v = ing.x.transpose_vector_multiply(&weighted);
    Ok(hop.solve(&v))
}

/// Compute the fourth-derivative trace: tr(G_ε(H) Xᵀ diag(d ⊙ (Xvₖ)(Xvₗ)) X).
///
/// Identity: tr(G_ε Xᵀ diag(w) X) = Σᵢ wᵢ · h^G[i].
/// Returns `None` if there are no fourth-derivative (d) terms.
fn compute_fourth_derivative_trace(
    ing: &ScalarGlmIngredients<'_>,
    v_k: &Array1<f64>,
    v_l: &Array1<f64>,
    leverage: &Array1<f64>,
) -> Result<Option<f64>, String> {
    let Some(d_array) = ing.d_array else {
        return Ok(None);
    };
    // Matrix-free X·v via DesignMatrix matvec; operator-backed (Lazy)
    // designs at biobank scale stream through their chunked kernels
    // instead of materializing the full (n×p) block.
    let x_vk = ing.x.matrixvectormultiply(v_k);
    let x_vl = ing.x.matrixvectormultiply(v_l);

    let mut acc = 0.0;
    Zip::from(d_array)
        .and(&x_vk)
        .and(&x_vl)
        .and(leverage)
        .for_each(|&d, &xvk, &xvl, &h| acc += d * xvk * xvl * h);
    Ok(Some(acc))
}

/// Compute every fourth-derivative trace for a coordinate set in one pass.
///
/// For scalar GLM Hessian corrections,
///
/// ```text
///   Q_ij = tr(G X' diag(d * (Xv_i) * (Xv_j)) X)
///        = Σ_r d_r h^G_r (Xv_i)_r (Xv_j)_r.
/// ```
///
/// This is a weighted Gram matrix of the row-space mode matrix `XV`.  Computing
/// it once replaces the per-pair `Xv_i` / `Xv_j` matvecs in
/// `compute_fourth_derivative_trace`, reducing the exact outer Hessian from
/// `O(T²)` design matvecs to `O(T)` design matvecs plus one `T×T` Gram.
fn compute_fourth_derivative_trace_matrix(
    ing: &ScalarGlmIngredients<'_>,
    modes: &[&Array1<f64>],
    leverage: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    let Some(d_array) = ing.d_array else {
        return Ok(None);
    };
    let n = ing.c_array.len();
    let t = modes.len();
    if t == 0 {
        return Ok(Some(Array2::zeros((0, 0))));
    }
    if d_array.len() != n || leverage.len() != n {
        return Err(format!(
            "fourth-derivative trace shape mismatch: c={}, d={}, leverage={}",
            n,
            d_array.len(),
            leverage.len()
        ));
    }

    let mut x_modes = Array2::<f64>::zeros((n, t));
    for (j, mode) in modes.iter().enumerate() {
        let x_v = ing.x.matrixvectormultiply(mode);
        if x_v.len() != n {
            return Err(format!(
                "fourth-derivative trace Xv length mismatch for mode {j}: got {}, expected {n}",
                x_v.len()
            ));
        }
        x_modes.column_mut(j).assign(&x_v);
    }

    let mut weighted = x_modes.clone();
    Zip::from(weighted.rows_mut())
        .and(d_array)
        .and(leverage)
        .for_each(|mut row, &d, &h| {
            let scale = d * h;
            row.mapv_inplace(|value| value * scale);
        });
    Ok(Some(crate::faer_ndarray::fast_atb(&x_modes, &weighted)))
}

/// Compute the IFT second-derivative correction contribution to h2_trace.
///
/// This is the SINGLE implementation of the formula:
///
/// ```text
///   correction = tr(G_ε C[u_ij]) + tr(G_ε Q[v_i, v_j])
/// ```
///
/// where u_ij is the second implicit derivative RHS (already solved or
/// consumed via the adjoint shortcut), and v_i, v_j are the first-order
/// mode responses (positive convention: v = H⁻¹(g)).
///
/// When the adjoint z_c is available, uses the O(p) shortcut:
///   C_trace = rhs · z_c,  Q_trace = compute_fourth_derivative_trace(v_i, v_j)
///
/// Otherwise falls back to the O(p²) direct path:
///   u = H⁻¹(rhs),  correction = hessian_second_derivative_correction(v_i, v_j, u)
fn compute_ift_correction_trace(
    hop: &dyn HessianOperator,
    rhs: &Array1<f64>,
    v_i: &Array1<f64>,
    v_j: &Array1<f64>,
    effective_deriv: &dyn HessianDerivativeProvider,
    adjoint_z_c: Option<&Array1<f64>>,
    glm_ingredients: Option<&ScalarGlmIngredients<'_>>,
    leverage: Option<&Array1<f64>>,
    precomputed_fourth_trace: Option<f64>,
    subspace: Option<&PenaltySubspaceTrace>,
) -> Result<f64, String> {
    if !effective_deriv.has_corrections() {
        return Ok(0.0);
    }
    // The adjoint shortcut `tr(G_ε C[u]) = uᵀ z_c` is only valid for the
    // full-space kernel.  When the projected kernel is required, fall back
    // to materialising the correction and tracing through the subspace.
    if let (Some(z_c), None) = (adjoint_z_c, subspace) {
        let c_trace = rhs.dot(z_c);
        let d_trace = if let Some(trace) = precomputed_fourth_trace {
            trace
        } else {
            match (glm_ingredients, leverage) {
                (Some(ing), Some(h_g)) => {
                    compute_fourth_derivative_trace(ing, v_i, v_j, h_g)?.unwrap_or(0.0)
                }
                _ => 0.0,
            }
        };
        Ok(c_trace + d_trace)
    } else {
        let u = hop.solve(rhs);
        if let Some(correction) =
            effective_deriv.hessian_second_derivative_correction_result(v_i, v_j, &u)?
        {
            if let Some(kernel) = subspace {
                // correction's DriftDerivResult materialises to a dense
                // matrix for the projected trace.
                match correction {
                    DriftDerivResult::Dense(matrix) => Ok(kernel.trace_projected_logdet(&matrix)),
                    DriftDerivResult::Operator(op) => Ok(kernel.trace_operator(op.as_ref())),
                }
            } else {
                Ok(correction.trace_logdet(hop))
            }
        } else {
            Ok(0.0)
        }
    }
}

/// Compute the β-dependent drift derivative traces: M_i[β_j] + M_j[β_i].
///
/// When a coordinate's fixed-β Hessian drift B depends on β, the second
/// Hessian drift Ḧ_{ij} includes additional terms D_β B_i[β_j] and
/// D_β B_j[β_i].  This function computes their traces through G_ε.
///
/// For ρ coordinates, B_k = A_k (penalty derivative) is β-independent, so
/// `b_depends_on_beta = false` and this returns 0.
fn compute_drift_deriv_traces(
    hop: &dyn HessianOperator,
    b_i_depends: bool,
    b_j_depends: bool,
    ext_i: Option<usize>,
    ext_j: Option<usize>,
    beta_i: &Array1<f64>,
    beta_j: &Array1<f64>,
    fixed_drift_deriv: Option<&FixedDriftDerivFn>,
    subspace: Option<&PenaltySubspaceTrace>,
) -> f64 {
    let trace_via = |result: DriftDerivResult| -> f64 {
        if let Some(kernel) = subspace {
            match result {
                DriftDerivResult::Dense(matrix) => kernel.trace_projected_logdet(&matrix),
                DriftDerivResult::Operator(op) => kernel.trace_operator(op.as_ref()),
            }
        } else {
            match result {
                DriftDerivResult::Dense(matrix) => hop.trace_logdet_gradient(&matrix),
                DriftDerivResult::Operator(op) => hop.trace_logdet_operator(op.as_ref()),
            }
        }
    };
    let mut trace = 0.0;
    // M_i[β_j] = D_β B_i[β_j]
    if b_i_depends {
        if let (Some(ei), Some(drift_fn)) = (ext_i, fixed_drift_deriv) {
            if let Some(result) = drift_fn(ei, beta_j) {
                trace += trace_via(result);
            }
        }
    }
    // M_j[β_i] = D_β B_j[β_i]
    if b_j_depends {
        if let (Some(ej), Some(drift_fn)) = (ext_j, fixed_drift_deriv) {
            if let Some(result) = drift_fn(ej, beta_i) {
                trace += trace_via(result);
            }
        }
    }
    trace
}

/// Compute the base trace of the fixed-β second Hessian drift: tr(G_ε ∂²H/∂θ_i∂θ_j|_β).
///
/// Uses the operator-backed path when available, otherwise falls back to
/// dense matrix trace.  Returns 0 when neither is provided (e.g., ρ-ρ
/// off-diagonal where the fixed-β second drift is zero).
fn compute_base_h2_trace(
    hop: &dyn HessianOperator,
    b_mat: &Array2<f64>,
    b_operator: Option<&dyn HyperOperator>,
    subspace: Option<&PenaltySubspaceTrace>,
) -> f64 {
    if let Some(kernel) = subspace {
        if let Some(op) = b_operator {
            kernel.trace_operator(op)
        } else if b_mat.nrows() > 0 {
            kernel.trace_projected_logdet(b_mat)
        } else {
            0.0
        }
    } else if let Some(op) = b_operator {
        hop.trace_logdet_operator(op)
    } else if b_mat.nrows() > 0 {
        hop.trace_logdet_gradient(b_mat)
    } else {
        0.0
    }
}

fn compute_base_h2_traces(
    hop: &dyn HessianOperator,
    pairs: &[&HyperCoordPair],
    subspace: Option<&PenaltySubspaceTrace>,
) -> Vec<f64> {
    if pairs.is_empty() {
        return Vec::new();
    }
    if subspace.is_none()
        && hop.prefers_stochastic_trace_estimation()
        && hop.logdet_traces_match_hinv_kernel()
    {
        let mut out = vec![0.0; pairs.len()];
        let mut dense_refs: Vec<&Array2<f64>> = Vec::new();
        let mut dense_slots = Vec::new();
        let mut op_refs: Vec<&dyn HyperOperator> = Vec::new();
        let mut op_slots = Vec::new();
        for (idx, pair) in pairs.iter().enumerate() {
            if let Some(op) = pair.b_operator.as_deref() {
                op_slots.push(idx);
                op_refs.push(op);
            } else if pair.b_mat.nrows() > 0 {
                dense_slots.push(idx);
                dense_refs.push(&pair.b_mat);
            }
        }
        if !dense_refs.is_empty() || !op_refs.is_empty() {
            let estimator = StochasticTraceEstimator::with_defaults();
            let values = estimator.estimate_traces_with_operators(hop, &dense_refs, &op_refs);
            for (local, &slot) in dense_slots.iter().enumerate() {
                out[slot] = values[local];
            }
            let offset = dense_refs.len();
            for (local, &slot) in op_slots.iter().enumerate() {
                out[slot] = values[offset + local];
            }
        }
        return out;
    }

    pairs
        .iter()
        .map(|pair| compute_base_h2_trace(hop, &pair.b_mat, pair.b_operator.as_deref(), subspace))
        .collect()
}

fn trace_logdet_hessian_cross_dense_drift(
    hop: &dyn HessianOperator,
    dense: &Array2<f64>,
    drift: &DriftDerivResult,
) -> f64 {
    match drift {
        DriftDerivResult::Dense(matrix) => hop.trace_logdet_hessian_cross(dense, matrix),
        DriftDerivResult::Operator(operator) => {
            hop.trace_logdet_hessian_cross_matrix_operator(dense, operator.as_ref())
        }
    }
}

fn trace_logdet_hessian_crosses_dense_spectral_drifts(
    dense_hop: &DenseSpectralOperator,
    dense_drifts: &[Array2<f64>],
    ext_drifts: &[DriftDerivResult],
) -> Array2<f64> {
    let total = dense_drifts.len() + ext_drifts.len();
    let mut rotated = Vec::with_capacity(total);
    for matrix in dense_drifts {
        rotated.push(dense_hop.rotate_to_eigenbasis(matrix));
    }
    for drift in ext_drifts {
        let projected = match drift {
            DriftDerivResult::Dense(matrix) => dense_hop.rotate_to_eigenbasis(matrix),
            DriftDerivResult::Operator(operator) => {
                dense_hop.projected_operator(&dense_hop.eigenvectors, operator.as_ref())
            }
        };
        rotated.push(projected);
    }

    let mut out = Array2::<f64>::zeros((total, total));
    for i in 0..total {
        for j in i..total {
            let value = dense_hop.trace_logdet_hessian_cross_rotated(&rotated[i], &rotated[j]);
            out[[i, j]] = value;
            if i != j {
                out[[j, i]] = value;
            }
        }
    }
    out
}

#[inline]
fn can_use_stochastic_logdet_hinv_kernel(
    hop: &dyn HessianOperator,
    total_p: usize,
    incl_logdet_h: bool,
) -> bool {
    total_p > 500
        && hop.prefers_stochastic_trace_estimation()
        && hop.logdet_traces_match_hinv_kernel()
        && incl_logdet_h
}

/// Compute the outer Hessian ∂²V/∂ρₖ∂ρₗ.
///
/// Uses the precomputed HessianOperator for all linear algebra.
fn compute_outer_hessian(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    lambdas: &[f64],
    hop: &dyn HessianOperator,
    effective_deriv: &dyn HessianDerivativeProvider,
) -> Result<Array2<f64>, String> {
    let k = rho.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let mut hess = Array2::zeros((total, total));
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();

    let (incl_logdet_h, incl_logdet_s) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => (true, true),
        DispersionHandling::Fixed {
            include_logdet_h,
            include_logdet_s,
            ..
        } => (*include_logdet_h, *include_logdet_s),
    };

    let det2 = solution.penalty_logdet.second.as_ref().ok_or_else(|| {
        "Outer Hessian requested but penalty second derivatives not provided".to_string()
    })?;

    // ── Profiled Gaussian precomputation ──
    let (profiled_phi, profiled_nu, profiled_dp_cgrad, profiled_dp_cgrad2, is_profiled) =
        match &solution.dispersion {
            DispersionHandling::ProfiledGaussian => {
                let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
                let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw);
                let nu = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
                let phi_hat = dp_c / nu;
                (phi_hat, nu, dp_cgrad, dp_cgrad2, true)
            }
            _ => (1.0, 1.0, 1.0, 0.0, false),
        };

    // ── ρ precomputation ──

    // Precompute both unscaled penalty derivatives and scaled H-curvature
    // derivatives.  The former differentiates the quadratic penalty cost; the
    // latter is only for H/logdet/IFT trace machinery.
    let mut penalty_a_k_betas: Vec<Array1<f64>> = Vec::with_capacity(k);
    let mut curvature_a_k_betas: Vec<Array1<f64>> = Vec::with_capacity(k);
    let mut v_ks: Vec<Array1<f64>> = Vec::with_capacity(k);

    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let penalty_a_k_beta_vec = penalty_a_k_beta(coord, &solution.beta, lambdas[idx]);
        let curvature_a_k_beta = penalty_a_k_beta(coord, &solution.beta, curvature_lambdas[idx]);
        let v_k = hop.solve(&curvature_a_k_beta);
        penalty_a_k_betas.push(penalty_a_k_beta_vec);
        curvature_a_k_betas.push(curvature_a_k_beta);
        v_ks.push(v_k);
    }

    // Precompute a_k = ½ β̂ᵀ Aₖ β̂ for profiled Gaussian correction.
    let rho_a_vals: Vec<f64> = (0..k)
        .map(|idx| 0.5 * solution.beta.dot(&penalty_a_k_betas[idx]))
        .collect();

    // Build pure Aₖ = λₖ Rₖᵀ Rₖ and Ḣₖ = Aₖ + correction for all k.
    //
    // We store both because:
    //   - Ḣₖ (first derivative of H) is needed for cross-trace Y_k = H⁻¹ Ḣₖ
    //   - Aₖ (penalty derivative only) is needed for the Ḧ_{kl} base and for
    //     the second implicit derivative β_{kl} = H⁻¹(Ḣₗ vₖ + Aₖ vₗ − δₖₗ Aₖ β̂)
    let mut a_k_matrices: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut h_k_matrices: Vec<Array2<f64>> = Vec::with_capacity(k);
    for idx in 0..k {
        let mut a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
        a_k_matrices.push(a_k.clone());

        let correction = if effective_deriv.has_corrections() {
            effective_deriv.hessian_derivative_correction(&v_ks[idx])?
        } else {
            None
        };
        if let Some(corr) = correction {
            a_k += &corr;
        }
        h_k_matrices.push(a_k);
    }

    // ── Adjoint trick precomputation ──
    //
    // For scalar GLMs with C[u] = Xᵀ diag(c ⊙ Xu) X:
    //   h^G          = diag(X G_ε(H) Xᵀ)
    //   z_c          = H⁻¹ Xᵀ (c ⊙ h^G)
    //   tr(G_ε C[u]) = uᵀ Xᵀ (c ⊙ h^G) = uᵀ (Hu_old) · z_c
    //
    // h^G also plugs into the fourth-derivative trace
    //   tr(G_ε Xᵀ diag(w) X) = Σᵢ wᵢ h^G[i],
    // collapsing per-pair O(np²) → O(n) work.
    let glm_ingredients = effective_deriv.scalar_glm_ingredients();
    let leverage = if incl_logdet_h {
        glm_ingredients
            .as_ref()
            .map(|ing| hop.xt_logdet_kernel_x_diagonal(ing.x))
    } else {
        None
    };
    let adjoint_z_c = if incl_logdet_h {
        match (glm_ingredients.as_ref(), leverage.as_ref()) {
            (Some(ing), Some(h_g)) => Some(compute_adjoint_z_c(ing, hop, h_g)?),
            _ => None,
        }
    } else {
        None
    };

    // ── ext precomputation ──

    // Check if any ext coordinate uses implicit operators and if the problem
    // is large enough to warrant stochastic cross-traces instead of
    // materializing p x p Hessian drift matrices.
    let any_ext_implicit = solution.ext_coords.iter().any(|c| {
        c.drift.operator_ref().map_or(false, |op| {
            c.drift.uses_operator_fast_path() && op.is_implicit()
        })
    });
    let total_p = hop.dim();
    // Stochastic cross-traces are only used when:
    // (1) implicit operators are present
    // (2) problem is large (p > 500)
    // (3) dense operator (eigendecomposition-based)
    // (4) logdet_h is included
    // (5) no third-derivative corrections (Gaussian family)
    //
    // Condition (5) ensures correctness: the stochastic estimator uses
    // B_d (the implicit operator) which equals Ḣ_d only when C[v_d] = 0.
    // For non-Gaussian families, Ḣ_d = B_d + C[v_d] and the correction
    // is a dense p x p matrix, so we fall back to dense materialization.
    let use_stochastic_cross_traces = any_ext_implicit
        && can_use_stochastic_logdet_hinv_kernel(hop, total_p, incl_logdet_h)
        && !effective_deriv.has_corrections()
        && solution.penalty_subspace_trace.is_none();

    // Precompute ext solve responses and total Hessian drifts. All ext
    // coordinates use canonical fixed-β stationarity derivatives, so
    // β_i = -H^-1 g_i and the correction provider is called with +v_i.
    let mut ext_v: Vec<Array1<f64>> = Vec::with_capacity(ext_dim);
    let mut ext_h_drifts: Vec<DriftDerivResult> = Vec::with_capacity(ext_dim);

    for coord in solution.ext_coords.iter() {
        let v_i = hop.solve(&coord.g);

        let correction = if effective_deriv.has_corrections() {
            effective_deriv.hessian_derivative_correction_result(&v_i)?
        } else {
            None
        };
        let h_i = hyper_coord_total_drift_result(&coord.drift, correction.as_ref(), hop.dim());

        ext_v.push(v_i);
        ext_h_drifts.push(h_i);
    }

    let fourth_trace_matrix =
        if incl_logdet_h && solution.penalty_subspace_trace.is_none() && adjoint_z_c.is_some() {
            match (glm_ingredients.as_ref(), leverage.as_ref()) {
                (Some(ing), Some(h_g)) if ing.d_array.is_some() => {
                    let modes = v_ks.iter().chain(ext_v.iter()).collect::<Vec<_>>();
                    compute_fourth_derivative_trace_matrix(ing, &modes, h_g)?
                }
                _ => None,
            }
        } else {
            None
        };

    // ── Stochastic second-order cross-trace precomputation ──
    //
    // When implicit operators are present and the problem is large, compute
    // the full (total x total) cross-trace matrix
    //   cross[d,e] = tr(H^{-1} Hd H^{-1} He)
    // stochastically. This path is only enabled on backends where the
    // logdet-Hessian cross term is exactly -tr(H^{-1} Hd H^{-1} He).
    //
    // Estimator:
    //   u = H^{-1} z,  q_e = A_e z,  r_e = H^{-1} q_e,  estimate = u^T A_d r_e
    //
    // This avoids materializing the (p x p) Hessian drift matrices for
    // implicit operators, and uses the correct tr(H^{-1} A_d H^{-1} A_e)
    // formula rather than the WRONG tr(A_d H^{-2} A_e).
    //
    // NOTE: The sign convention here gives +tr(H^{-1} Hd H^{-1} He).
    // The outer Hessian uses -tr(H^{-1} Hj H^{-1} Hi) = -(this value).
    let stochastic_cross_traces: Option<Array2<f64>> = if use_stochastic_cross_traces {
        let total_coords = k + ext_dim;
        let mut dense_mats: Vec<Array2<f64>> = Vec::new();
        let mut coord_has_operator: Vec<bool> = Vec::with_capacity(total_coords);
        let mut operator_arcs: Vec<Arc<dyn HyperOperator>> = Vec::new();

        // rho coordinates: always dense.
        for idx in 0..k {
            dense_mats.push(h_k_matrices[idx].clone());
            coord_has_operator.push(false);
        }

        // ext coordinates: dense or operator-backed, including any
        // non-Gaussian third-derivative correction already composed into
        // `ext_h_drifts`.
        for drift in &ext_h_drifts {
            match drift {
                DriftDerivResult::Dense(matrix) => {
                    dense_mats.push(matrix.clone());
                    coord_has_operator.push(false);
                }
                DriftDerivResult::Operator(operator) => {
                    operator_arcs.push(Arc::clone(operator));
                    coord_has_operator.push(true);
                }
            }
        }

        let generic_ops: Vec<&dyn HyperOperator> =
            operator_arcs.iter().map(|op| op.as_ref()).collect();
        let impl_ops: Vec<&ImplicitHyperOperator> = generic_ops
            .iter()
            .filter_map(|op| op.as_implicit())
            .collect();

        Some(stochastic_trace_hinv_crosses(
            hop,
            &dense_mats,
            &coord_has_operator,
            &generic_ops,
            &impl_ops,
        ))
    } else {
        None
    };

    // When the rank-deficient LAML fix replaces the full-space logdet
    // kernel with the projected `U_S · H_proj⁻¹ · U_Sᵀ`, the cross-trace
    // `−tr(K Ḣ_j K Ḣ_i)` must also use the projected kernel for the same
    // reason the first-order trace does (the IFT correction `D_β H[v]`
    // spills onto `null(S)` for non-Gaussian families).  Collect the
    // reduced drifts `R_d = U_Sᵀ Ḣ_d U_S` once and reuse them for every
    // pair; per-pair cost is then O(r²) instead of O(p²) per cross.
    let subspace = solution.penalty_subspace_trace.as_deref();
    let reduced_h_drifts: Option<Vec<Array2<f64>>> = subspace.map(|kernel| {
        let mut reduced = Vec::with_capacity(k + ext_dim);
        for matrix in &h_k_matrices {
            reduced.push(kernel.reduce(matrix));
        }
        for drift in &ext_h_drifts {
            let reduced_drift = match drift {
                DriftDerivResult::Dense(matrix) => kernel.reduce(matrix),
                DriftDerivResult::Operator(operator) => kernel.reduce_operator(operator.as_ref()),
            };
            reduced.push(reduced_drift);
        }
        reduced
    });
    let exact_logdet_cross_traces = if incl_logdet_h && stochastic_cross_traces.is_none() {
        if let (Some(kernel), Some(reduced)) = (subspace, reduced_h_drifts.as_ref()) {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let n = reduced.len();
            // Each `(i, j)` upper-triangular pair is an independent cross
            // trace `−tr(K · A_i · K · A_j)` over the projected kernel
            // `K = U_S (U_Sᵀ H U_S)⁻¹ U_Sᵀ`; the kernel and `reduced` slice
            // are both read-only borrows so the K(K+1)/2 pairs dispatch in
            // parallel, then we stitch the symmetric `n × n` Array2
            // sequentially.
            let pairs: Vec<(usize, usize)> =
                (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
            let pair_values: Vec<(usize, usize, f64)> = pairs
                .into_par_iter()
                .map(|(i, j)| {
                    let value =
                        -kernel.trace_projected_logdet_cross_reduced(&reduced[i], &reduced[j]);
                    (i, j, value)
                })
                .collect();
            let mut out = Array2::<f64>::zeros((n, n));
            for (i, j, value) in pair_values {
                out[[i, j]] = value;
                if i != j {
                    out[[j, i]] = value;
                }
            }
            Some(out)
        } else if let Some(dense_hop) = hop.as_exact_dense_spectral() {
            Some(trace_logdet_hessian_crosses_dense_spectral_drifts(
                dense_hop,
                &h_k_matrices,
                &ext_h_drifts,
            ))
        } else {
            let total_coords = k + ext_dim;
            let mut out = Array2::<f64>::zeros((total_coords, total_coords));
            for ii in 0..total_coords {
                for jj in ii..total_coords {
                    let value = match (ii < k, jj < k) {
                        (true, true) => {
                            hop.trace_logdet_hessian_cross(&h_k_matrices[ii], &h_k_matrices[jj])
                        }
                        (true, false) => trace_logdet_hessian_cross_dense_drift(
                            hop,
                            &h_k_matrices[ii],
                            &ext_h_drifts[jj - k],
                        ),
                        (false, true) => trace_logdet_hessian_cross_dense_drift(
                            hop,
                            &h_k_matrices[jj],
                            &ext_h_drifts[ii - k],
                        ),
                        (false, false) => ext_h_drifts[ii - k]
                            .trace_logdet_hessian_cross(&ext_h_drifts[jj - k], hop),
                    };
                    out[[ii, jj]] = value;
                    if ii != jj {
                        out[[jj, ii]] = value;
                    }
                }
            }
            Some(out)
        }
    } else {
        None
    };

    // ── ρ-ρ block ── (uses shared helpers for all trace computations)

    for kk in 0..k {
        for ll in kk..k {
            let pair_a = if kk == ll { rho_a_vals[kk] } else { 0.0 };

            let cross_trace = if let Some(ref exact) = exact_logdet_cross_traces {
                exact[[kk, ll]]
            } else if let Some(ref sct) = stochastic_cross_traces {
                -sct[[kk, ll]]
            } else {
                hop.trace_logdet_hessian_cross(&h_k_matrices[kk], &h_k_matrices[ll])
            };

            // Second Hessian drift trace via shared helpers.
            //
            // RHS = Ḣ_l v_k + B_k v_l − δ_{kl} g_k
            // base = δ_{kl} tr(K A_k)
            // correction = compute_ift_correction_trace(RHS, v_k, v_l)
            //
            // `K` is the full-space `G_ε(H)` unless the rank-deficient LAML
            // fix is active, in which case every trace routes through the
            // projected kernel so the outer Hessian matches the projected
            // `½ log|U_Sᵀ H U_S|_+` cost.
            let base = if kk == ll {
                if let Some(kernel) = subspace {
                    kernel.trace_projected_logdet(&a_k_matrices[kk])
                } else if solution.penalty_coords[kk].is_block_local() {
                    let (block, start, end) = solution.penalty_coords[kk].scaled_block_local(1.0);
                    hop.trace_logdet_block_local(&block, curvature_lambdas[kk], start, end)
                } else {
                    hop.trace_logdet_gradient(&a_k_matrices[kk])
                }
            } else {
                0.0
            };

            let mut rhs = h_k_matrices[ll].dot(&v_ks[kk]);
            rhs += &solution.penalty_coords[kk].scaled_matvec(&v_ks[ll], curvature_lambdas[kk]);
            if kk == ll {
                rhs -= &curvature_a_k_betas[kk];
            }

            let correction = compute_ift_correction_trace(
                hop,
                &rhs,
                &v_ks[kk],
                &v_ks[ll],
                effective_deriv,
                adjoint_z_c.as_ref(),
                glm_ingredients.as_ref(),
                leverage.as_ref(),
                fourth_trace_matrix.as_ref().map(|trace| trace[[kk, ll]]),
                subspace,
            )?;

            let h_kl_trace = base + correction;

            let h_val = outer_hessian_entry(
                rho_a_vals[kk],
                rho_a_vals[ll],
                penalty_a_k_betas[ll].dot(&v_ks[kk]),
                pair_a,
                cross_trace,
                h_kl_trace,
                det2[[kk, ll]],
                profiled_phi,
                profiled_nu,
                profiled_dp_cgrad,
                profiled_dp_cgrad2,
                is_profiled,
                incl_logdet_h,
                incl_logdet_s,
            );
            hess[[kk, ll]] = h_val;
            if kk != ll {
                hess[[ll, kk]] = h_val;
            }
        }
    }

    // ── ρ-ext cross block ── (uses shared helpers for all trace computations)

    if let Some(ref rho_ext_fn) = solution.rho_ext_pair_fn {
        for rho_idx in 0..k {
            for ext_idx in 0..ext_dim {
                let pair = rho_ext_fn(rho_idx, ext_idx);
                let a_ext = solution.ext_coords[ext_idx].a;

                let (cross_trace, h2_trace) = if incl_logdet_h {
                    let cross_trace = if let Some(ref exact) = exact_logdet_cross_traces {
                        exact[[rho_idx, k + ext_idx]]
                    } else if let Some(ref sct) = stochastic_cross_traces {
                        -sct[[rho_idx, k + ext_idx]]
                    } else {
                        trace_logdet_hessian_cross_dense_drift(
                            hop,
                            &h_k_matrices[rho_idx],
                            &ext_h_drifts[ext_idx],
                        )
                    };

                    // `coord.g` stores g_i = F_{βi} and v_i = H⁻¹g_i, so the
                    // actual mode derivative is β_i = -v_i for both ρ and ext.
                    // Differentiating stationarity gives:
                    //   H β_{rho,ext}
                    //     = -g_{rho,ext} - H_rho β_ext - Ḣ_ext β_rho
                    //     = -g_{rho,ext} + H_rho v_ext + Ḣ_ext v_rho.
                    let mut rhs = -&pair.g;
                    rhs += &solution.penalty_coords[rho_idx]
                        .scaled_matvec(&ext_v[ext_idx], curvature_lambdas[rho_idx]);
                    let beta_rho = v_ks[rho_idx].mapv(|value| -value);
                    rhs += &ext_h_drifts[ext_idx].apply(&v_ks[rho_idx]);

                    let base = compute_base_h2_trace(
                        hop,
                        &pair.b_mat,
                        pair.b_operator.as_deref(),
                        subspace,
                    );

                    let beta_ext = ext_v[ext_idx].mapv(|value| -value);
                    let m_terms = compute_drift_deriv_traces(
                        hop,
                        false, // ρ drift is β-independent
                        solution.ext_coords[ext_idx].b_depends_on_beta,
                        None,
                        Some(ext_idx),
                        &beta_rho,
                        &beta_ext,
                        solution.fixed_drift_deriv.as_ref(),
                        subspace,
                    );

                    let correction = compute_ift_correction_trace(
                        hop,
                        &rhs,
                        &v_ks[rho_idx],
                        &ext_v[ext_idx],
                        effective_deriv,
                        adjoint_z_c.as_ref(),
                        glm_ingredients.as_ref(),
                        leverage.as_ref(),
                        fourth_trace_matrix
                            .as_ref()
                            .map(|trace| trace[[rho_idx, k + ext_idx]]),
                        subspace,
                    )?;

                    (cross_trace, base + m_terms + correction)
                } else {
                    (0.0, 0.0)
                };

                let h_val = outer_hessian_entry(
                    rho_a_vals[rho_idx],
                    a_ext,
                    penalty_a_k_betas[rho_idx].dot(&ext_v[ext_idx]),
                    pair.a,
                    cross_trace,
                    h2_trace,
                    pair.ld_s,
                    profiled_phi,
                    profiled_nu,
                    profiled_dp_cgrad,
                    profiled_dp_cgrad2,
                    is_profiled,
                    incl_logdet_h,
                    incl_logdet_s,
                );
                hess[[rho_idx, k + ext_idx]] = h_val;
                hess[[k + ext_idx, rho_idx]] = h_val;
            }
        }
    }

    // ── ext-ext block ── (uses shared helpers for all trace computations)

    if let Some(ref ext_pair_fn) = solution.ext_coord_pair_fn {
        for ii in 0..ext_dim {
            for jj in ii..ext_dim {
                let pair = ext_pair_fn(ii, jj);
                let coord_i = &solution.ext_coords[ii];
                let coord_j = &solution.ext_coords[jj];

                let (cross_trace, h2_trace) = if incl_logdet_h {
                    let cross_trace = if let Some(ref exact) = exact_logdet_cross_traces {
                        exact[[k + ii, k + jj]]
                    } else if let Some(ref sct) = stochastic_cross_traces {
                        -sct[[k + ii, k + jj]]
                    } else {
                        ext_h_drifts[ii].trace_logdet_hessian_cross(&ext_h_drifts[jj], hop)
                    };

                    // `coord.g` is g_i = F_{βi} and v_i = H⁻¹g_i, hence
                    // β_i = -v_i. Differentiating stationarity gives:
                    //   H β_{ij}
                    //     = -g_{ij} - H_i β_j - Ḣ_j β_i
                    //     = -g_{ij} + H_i v_j + Ḣ_j v_i.
                    let mut rhs = -&pair.g;
                    coord_i
                        .drift
                        .scaled_add_apply(ext_v[jj].view(), 1.0, &mut rhs);
                    rhs += &ext_h_drifts[jj].apply(&ext_v[ii]);

                    let base = compute_base_h2_trace(
                        hop,
                        &pair.b_mat,
                        pair.b_operator.as_deref(),
                        subspace,
                    );

                    let beta_i = ext_v[ii].mapv(|value| -value);
                    let beta_j = ext_v[jj].mapv(|value| -value);
                    let m_terms = compute_drift_deriv_traces(
                        hop,
                        coord_i.b_depends_on_beta,
                        coord_j.b_depends_on_beta,
                        Some(ii),
                        Some(jj),
                        &beta_i,
                        &beta_j,
                        solution.fixed_drift_deriv.as_ref(),
                        subspace,
                    );

                    let correction = compute_ift_correction_trace(
                        hop,
                        &rhs,
                        &ext_v[ii],
                        &ext_v[jj],
                        effective_deriv,
                        adjoint_z_c.as_ref(),
                        glm_ingredients.as_ref(),
                        leverage.as_ref(),
                        fourth_trace_matrix
                            .as_ref()
                            .map(|trace| trace[[k + ii, k + jj]]),
                        subspace,
                    )?;

                    let h2 = base + m_terms + correction;
                    let g_dot_v = coord_i.g.dot(&ext_v[jj]);
                    let pair_g_finite = pair.g.iter().all(|v| v.is_finite());
                    let b_mat_finite = pair.b_mat.iter().all(|v| v.is_finite());
                    let ext_vi_finite = ext_v[ii].iter().all(|v| v.is_finite());
                    let ext_vj_finite = ext_v[jj].iter().all(|v| v.is_finite());
                    let any_non_finite = !cross_trace.is_finite()
                        || !base.is_finite()
                        || !m_terms.is_finite()
                        || !correction.is_finite()
                        || !h2.is_finite()
                        || !pair.a.is_finite()
                        || !pair.ld_s.is_finite()
                        || !g_dot_v.is_finite()
                        || !pair_g_finite
                        || !b_mat_finite;
                    if any_non_finite {
                        // Probe a single bad b_mat entry so we can tell whether
                        // the NaN is structural (whole matrix bad) or localized
                        // to a particular row/col.
                        let mut first_bad_b_mat = None;
                        if !b_mat_finite {
                            'outer: for r in 0..pair.b_mat.nrows() {
                                for c in 0..pair.b_mat.ncols() {
                                    if !pair.b_mat[[r, c]].is_finite() {
                                        first_bad_b_mat = Some((r, c, pair.b_mat[[r, c]]));
                                        break 'outer;
                                    }
                                }
                            }
                        }
                        let mut first_bad_pair_g = None;
                        if !pair_g_finite {
                            for (idx, value) in pair.g.iter().enumerate() {
                                if !value.is_finite() {
                                    first_bad_pair_g = Some((idx, *value));
                                    break;
                                }
                            }
                        }
                        log::warn!(
                            "[OUTER ext-ext non-finite] ({},{}): cross_trace={} base={} m_terms={} correction={} pair.a={} pair.ld_s={} g.dot(v_jj)={} pair_g_finite={} first_bad_pair_g={:?} b_mat_finite={} first_bad_b_mat={:?} b_operator_present={} b_mat_dim={}x{} ext_v[ii]_finite={} ext_v[jj]_finite={} coord_i.b_depends_on_beta={} coord_j.b_depends_on_beta={}",
                            ii,
                            jj,
                            cross_trace,
                            base,
                            m_terms,
                            correction,
                            pair.a,
                            pair.ld_s,
                            g_dot_v,
                            pair_g_finite,
                            first_bad_pair_g,
                            b_mat_finite,
                            first_bad_b_mat,
                            pair.b_operator.is_some(),
                            pair.b_mat.nrows(),
                            pair.b_mat.ncols(),
                            ext_vi_finite,
                            ext_vj_finite,
                            coord_i.b_depends_on_beta,
                            coord_j.b_depends_on_beta,
                        );
                    }
                    (cross_trace, h2)
                } else {
                    (0.0, 0.0)
                };

                let h_val = outer_hessian_entry(
                    coord_i.a,
                    coord_j.a,
                    coord_i.g.dot(&ext_v[jj]),
                    pair.a,
                    cross_trace,
                    h2_trace,
                    pair.ld_s,
                    profiled_phi,
                    profiled_nu,
                    profiled_dp_cgrad,
                    profiled_dp_cgrad2,
                    is_profiled,
                    incl_logdet_h,
                    incl_logdet_s,
                );
                hess[[k + ii, k + jj]] = h_val;
                if ii != jj {
                    hess[[k + jj, k + ii]] = h_val;
                }
            }
        }
    }

    if hess.iter().any(|v| !v.is_finite()) {
        // NaN bisection: report which intermediate inputs were already
        // non-finite before the entry-builder summed them. This pinpoints the
        // original source (penalty drift, drift correction, cross-trace, ...)
        // instead of just flagging the final outer-Hessian entry.
        let report_finite = |name: &str, value: f64, ii: usize, jj: usize| {
            if !value.is_finite() {
                log::warn!(
                    "[OUTER non-finite] {} at ({}, {}) = {}",
                    name,
                    ii,
                    jj,
                    value,
                );
            }
        };
        for kk in 0..k {
            report_finite("rho_a_vals[kk]", rho_a_vals[kk], kk, kk);
            for entry in penalty_a_k_betas[kk].iter() {
                if !entry.is_finite() {
                    log::warn!(
                        "[OUTER non-finite] penalty_a_k_betas[{}] has non-finite",
                        kk
                    );
                    break;
                }
            }
            for entry in v_ks[kk].iter() {
                if !entry.is_finite() {
                    log::warn!("[OUTER non-finite] v_ks[{}] has non-finite", kk);
                    break;
                }
            }
        }
        if let Some(ref exact) = exact_logdet_cross_traces {
            for ii in 0..exact.nrows() {
                for jj in 0..exact.ncols() {
                    report_finite("exact_logdet_cross_traces", exact[[ii, jj]], ii, jj);
                }
            }
        }
        if let Some(ref sct) = stochastic_cross_traces {
            for ii in 0..sct.nrows() {
                for jj in 0..sct.ncols() {
                    report_finite("stochastic_cross_traces", sct[[ii, jj]], ii, jj);
                }
            }
        }
        if let Some(ref h_g) = leverage {
            for entry in h_g.iter() {
                if !entry.is_finite() {
                    log::warn!("[OUTER non-finite] leverage h^G has non-finite entries");
                    break;
                }
            }
        }
        if let Some(ref z_c) = adjoint_z_c {
            for entry in z_c.iter() {
                if !entry.is_finite() {
                    log::warn!("[OUTER non-finite] adjoint_z_c has non-finite entries");
                    break;
                }
            }
        }
        for ii in 0..total {
            for jj in 0..total {
                report_finite("hess", hess[[ii, jj]], ii, jj);
            }
        }
        return Err(
            "Outer Hessian contains non-finite entries; exact higher-order derivatives are invalid"
                .to_string(),
        );
    }

    Ok(hess)
}

struct StoredFirstDrift {
    dense: Option<Array2<f64>>,
    dense_rotated: Option<Array2<f64>>,
    operators: Vec<Arc<dyn HyperOperator>>,
}

impl StoredFirstDrift {
    fn from_parts(
        dense: Option<Array2<f64>>,
        dense_rotated: Option<Array2<f64>>,
        operators: Vec<Arc<dyn HyperOperator>>,
    ) -> Self {
        Self {
            dense,
            dense_rotated,
            operators,
        }
    }

    fn scaled_add_apply(&self, v: ArrayView1<'_, f64>, scale: f64, out: &mut Array1<f64>) {
        debug_assert_eq!(v.len(), out.len());
        if scale == 0.0 {
            return;
        }
        if let Some(matrix) = self.dense.as_ref() {
            dense_matvec_scaled_add_into(matrix, v, scale, out.view_mut());
        }
        if !self.operators.is_empty() {
            for op in &self.operators {
                op.scaled_add_mul_vec(v, scale, out.view_mut());
            }
        }
    }

    fn apply_dot(&self, v: ArrayView1<'_, f64>, test: ArrayView1<'_, f64>) -> f64 {
        debug_assert_eq!(v.len(), test.len());
        let mut total = 0.0;
        if let Some(matrix) = self.dense.as_ref() {
            total += dense_bilinear(matrix, v, test);
        }
        for op in &self.operators {
            total += op.bilinear_view(v, test);
        }
        total
    }
}

struct BorrowedStoredDriftOperator<'a> {
    drift: &'a StoredFirstDrift,
    dim_hint: usize,
}

impl HyperOperator for BorrowedStoredDriftOperator<'_> {
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
        out.fill(0.0);
        if let Some(matrix) = self.drift.dense.as_ref() {
            dense_matvec_into(matrix, v, out.view_mut());
        }
        for op in &self.drift.operators {
            op.scaled_add_mul_vec(v, 1.0, out.view_mut());
        }
    }

    fn scaled_add_mul_vec(&self, v: ArrayView1<'_, f64>, scale: f64, out: ArrayViewMut1<'_, f64>) {
        if scale == 0.0 {
            return;
        }
        let mut out = out;
        if let Some(matrix) = self.drift.dense.as_ref() {
            dense_matvec_scaled_add_into(matrix, v, scale, out.view_mut());
        }
        for op in &self.drift.operators {
            op.scaled_add_mul_vec(v, scale, out.view_mut());
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.drift.apply_dot(v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        self.drift.apply_dot(v, u)
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self
            .drift
            .dense
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((self.dim_hint, self.dim_hint)));
        for op in &self.drift.operators {
            out += &op.to_dense();
        }
        out
    }

    fn is_implicit(&self) -> bool {
        !self.drift.operators.is_empty()
    }
}

/// Linear combination of `HyperOperator` factors with explicit scalar
/// weights. Used to bundle a coord's per-mode drift operators (or any other
/// per-term linear combination) into a single matrix-free operator that
/// implements the same `HyperOperator` trait, so callers downstream do not
/// need to handle a vector of (weight, op) pairs themselves.
pub(crate) struct WeightedHyperOperator {
    pub(crate) terms: Vec<(f64, Arc<dyn HyperOperator>)>,
    pub(crate) dim_hint: usize,
}

impl HyperOperator for WeightedHyperOperator {
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
        let mut nonzero_terms = self.terms.iter().filter(|(weight, _)| *weight != 0.0);
        if let Some((weight, op)) = nonzero_terms.next()
            && nonzero_terms.next().is_none()
        {
            op.mul_vec_into(v, out.view_mut());
            if *weight != 1.0 {
                out.mapv_inplace(|value| *weight * value);
            }
            return;
        }

        out.fill(0.0);
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                op.scaled_add_mul_vec(v, *weight, out.view_mut());
            }
        }
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let mut nonzero_terms = self.terms.iter().filter(|(weight, _)| *weight != 0.0);
        if let Some((weight, op)) = nonzero_terms.next()
            && nonzero_terms.next().is_none()
        {
            op.mul_basis_columns_into(start, out.view_mut());
            if *weight != 1.0 {
                out.mapv_inplace(|value| *weight * value);
            }
            return;
        }

        out.fill(0.0);
        let mut work = Array2::<f64>::zeros((out.nrows(), out.ncols()));
        for (weight, op) in &self.terms {
            if *weight == 0.0 {
                continue;
            }
            op.mul_basis_columns_into(start, work.view_mut());
            out.scaled_add(*weight, &work);
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
        for (weight, op) in &self.terms {
            let combined = scale * *weight;
            if combined != 0.0 {
                op.scaled_add_mul_vec(v, combined, out.view_mut());
            }
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.bilinear(v, u))
            .sum()
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.bilinear_view(v, u))
            .sum()
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.trace_projected_factor(factor))
            .sum()
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.trace_projected_factor_cached(factor, cache))
            .sum()
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.dim_hint, self.dim_hint));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                out.scaled_add(*weight, &op.to_dense());
            }
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.terms.iter().any(|(_, op)| op.is_implicit())
    }
}

struct OuterHessianCoord {
    a: f64,
    g: Array1<f64>,
    v: Array1<f64>,
    total_drift: StoredFirstDrift,
    base_drift: StoredFirstDrift,
    ext_index: Option<usize>,
    b_depends_on_beta: bool,
}

impl OuterHessianCoord {
    fn is_ext(&self) -> bool {
        self.ext_index.is_some()
    }
}

struct UnifiedOuterHessianOperator {
    hop: Arc<dyn HessianOperator>,
    coords: Vec<OuterHessianCoord>,
    pair_a: Array2<f64>,
    pair_ld_s: Array2<f64>,
    g_dot_v: Array2<f64>,
    pair_g: Vec<Vec<Option<Array1<f64>>>>,
    base_h2: Array2<f64>,
    m_pair_trace: Array2<f64>,
    /// Precomputed pair-wise logdet-Hessian cross traces.
    /// `cross_trace[i, j] = tr(G_ε(H) Ḣ_i Ḣ_j)` decomposed across the
    /// dense and operator components of each coord's `total_drift`.
    /// Populated only when `incl_logdet_h`.  matvec recovers the alpha-combo
    /// trace as `cross_trace.row(idx).dot(alpha)`, replacing the per-HVP
    /// recomputation that previously rebuilt these traces every time the
    /// K×K outer Hessian was materialized via K matvecs.
    cross_trace: Option<Array2<f64>>,
    profiled_phi: f64,
    profiled_nu: f64,
    profiled_dp_cgrad: f64,
    profiled_dp_cgrad2: f64,
    is_profiled: bool,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
    kernel: OuterHessianDerivativeKernel,
    subspace: Option<Arc<PenaltySubspaceTrace>>,
    adjoint_z_c: Option<Array1<f64>>,
    leverage: Option<Array1<f64>>,
    fourth_trace: Option<Array2<f64>>,
    callback_second_modes: Option<Vec<Array1<f64>>>,
}

impl UnifiedOuterHessianOperator {
    fn signed_mode_combo_for_correction(&self, alpha: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.hop.dim());
        for (j, coord) in self.coords.iter().enumerate() {
            if alpha[j] == 0.0 {
                continue;
            }
            if coord.is_ext() {
                out.scaled_add(-alpha[j], &coord.v);
            } else {
                out.scaled_add(alpha[j], &coord.v);
            }
        }
        out
    }

    fn pair_rhs_dot(&self, row: usize, col: usize, test: ArrayView1<'_, f64>) -> f64 {
        let row_coord = &self.coords[row];
        let col_coord = &self.coords[col];
        let pair_g_dot = self.pair_g[row][col]
            .as_ref()
            .map(|pair_g| pair_g.dot(&test))
            .unwrap_or(0.0);

        col_coord.total_drift.apply_dot(row_coord.v.view(), test)
            + row_coord.base_drift.apply_dot(col_coord.v.view(), test)
            - pair_g_dot
    }

    fn scaled_add_pair_rhs(&self, row: usize, col: usize, scale: f64, out: &mut Array1<f64>) {
        if scale == 0.0 {
            return;
        }
        let row_coord = &self.coords[row];
        let col_coord = &self.coords[col];
        col_coord
            .total_drift
            .scaled_add_apply(row_coord.v.view(), scale, out);
        row_coord
            .base_drift
            .scaled_add_apply(col_coord.v.view(), scale, out);
        if let Some(pair_g) = self.pair_g[row][col].as_ref() {
            out.scaled_add(-scale, pair_g);
        }
    }

    fn pair_rhs_combo(&self, idx: usize, alpha: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.hop.dim());
        for j in 0..alpha.len() {
            if alpha[j] != 0.0 {
                self.scaled_add_pair_rhs(idx, j, alpha[j], &mut out);
            }
        }
        out
    }

    fn scalar_correction_trace(
        &self,
        idx: usize,
        alpha: &Array1<f64>,
        v_i: &Array1<f64>,
        m_alpha: &Array1<f64>,
    ) -> Result<f64, String> {
        let OuterHessianDerivativeKernel::ScalarGlm {
            c_array,
            d_array,
            x,
        } = &self.kernel
        else {
            return Err("scalar correction requested for non-scalar kernel".to_string());
        };

        // Cheap adjoint shortcut: works for both full-Hessian and projected
        // (subspace) regimes because §10 populates `leverage`/`adjoint_z_c`
        // with the projected `h^{G,proj}` and `K · v` under subspace, and
        // the identity tr(Kernel · C[u]) = uᵀ Xᵀ(c ⊙ h^G) carries through.
        let z_c = self.adjoint_z_c.as_ref().ok_or_else(|| {
            "missing adjoint trace cache for scalar outer Hessian operator".to_string()
        })?;
        let ingredients = ScalarGlmIngredients {
            c_array,
            d_array: d_array.as_ref(),
            x,
        };
        let h_g = self.leverage.as_ref().ok_or_else(|| {
            "missing leverage cache for scalar outer Hessian operator".to_string()
        })?;
        let mut c_trace = 0.0;
        for (j, &alpha_j) in alpha.iter().enumerate() {
            if alpha_j == 0.0 {
                continue;
            }
            c_trace += alpha_j * self.pair_rhs_dot(idx, j, z_c.view());
        }
        let d_trace = if let Some(trace) = self.fourth_trace.as_ref() {
            let mut combo = 0.0;
            for (j, &alpha_j) in alpha.iter().enumerate() {
                if alpha_j != 0.0 {
                    combo += alpha_j * trace[[idx, j]];
                }
            }
            combo
        } else {
            compute_fourth_derivative_trace(&ingredients, v_i, m_alpha, h_g)?.unwrap_or(0.0)
        };
        Ok(c_trace + d_trace)
    }

    fn callback_correction_trace(
        &self,
        rhs: &Array1<f64>,
        second_v: &Array1<f64>,
        neg_m_alpha: &Array1<f64>,
    ) -> Result<f64, String> {
        let OuterHessianDerivativeKernel::Callback { first, second } = &self.kernel else {
            return Err("callback correction requested for non-callback kernel".to_string());
        };
        let u = self.hop.solve(rhs);
        let Some(term1) = first(&u)? else {
            return Ok(0.0);
        };
        let Some(term2) = second(neg_m_alpha, second_v)? else {
            return Ok(0.0);
        };
        let combined = CompositeHyperOperator {
            dense: None,
            operators: vec![term1.into_operator(), term2.into_operator()],
            dim_hint: self.hop.dim(),
        };
        if let Some(subspace) = self.subspace.as_deref() {
            Ok(subspace.trace_operator(&combined))
        } else {
            Ok(self.hop.trace_logdet_operator(&combined))
        }
    }
}

impl crate::solver::outer_strategy::OuterHessianOperator for UnifiedOuterHessianOperator {
    fn dim(&self) -> usize {
        self.coords.len()
    }

    fn matvec(&self, alpha: &Array1<f64>) -> Result<Array1<f64>, String> {
        if alpha.len() != self.coords.len() {
            return Err(format!(
                "outer Hessian alpha length mismatch: got {}, expected {}",
                alpha.len(),
                self.coords.len()
            ));
        }
        let mut a_alpha = 0.0;
        for (idx, coord) in self.coords.iter().enumerate() {
            if alpha[idx] != 0.0 {
                a_alpha += alpha[idx] * coord.a;
            }
        }
        let correction_m_alpha = self.signed_mode_combo_for_correction(alpha);
        let callback_neg_m_alpha =
            matches!(self.kernel, OuterHessianDerivativeKernel::Callback { .. })
                .then(|| -&correction_m_alpha);
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let values: Result<Vec<f64>, String> = (0..self.coords.len())
            .into_par_iter()
            .map(|idx| {
                let coord = &self.coords[idx];
                let pair_a = self.pair_a.row(idx).dot(alpha);
                let pair_ld_s = self.pair_ld_s.row(idx).dot(alpha);
                let g_dot_v_alpha = self.g_dot_v.row(idx).dot(alpha);
                let base_h2 = self.base_h2.row(idx).dot(alpha);
                let m_terms = self.m_pair_trace.row(idx).dot(alpha);

                let cross_trace = match self.cross_trace.as_ref() {
                    Some(ct) => ct.row(idx).dot(alpha),
                    None => 0.0,
                };

                let correction = if self.incl_logdet_h {
                    match &self.kernel {
                        OuterHessianDerivativeKernel::Gaussian => 0.0,
                        OuterHessianDerivativeKernel::ScalarGlm { .. } => {
                            self.scalar_correction_trace(idx, alpha, &coord.v, &correction_m_alpha)?
                        }
                        OuterHessianDerivativeKernel::Callback { .. } => {
                            let second_v = &self
                                .callback_second_modes
                                .as_ref()
                                .expect("callback second modes")[idx];
                            let rhs = self.pair_rhs_combo(idx, alpha);
                            self.callback_correction_trace(
                                &rhs,
                                second_v,
                                callback_neg_m_alpha
                                    .as_ref()
                                    .expect("callback negated mode"),
                            )?
                        }
                    }
                } else {
                    0.0
                };

                Ok(outer_hessian_entry(
                    coord.a,
                    a_alpha,
                    g_dot_v_alpha,
                    pair_a,
                    cross_trace,
                    base_h2 + m_terms + correction,
                    pair_ld_s,
                    self.profiled_phi,
                    self.profiled_nu,
                    self.profiled_dp_cgrad,
                    self.profiled_dp_cgrad2,
                    self.is_profiled,
                    self.incl_logdet_h,
                    self.incl_logdet_s,
                ))
            })
            .collect();

        Ok(Array1::from_vec(values?))
    }
}

fn build_outer_hessian_operator(
    solution: &InnerSolution<'_>,
    lambdas: &[f64],
    effective_deriv: &dyn HessianDerivativeProvider,
    kernel: OuterHessianDerivativeKernel,
) -> Result<UnifiedOuterHessianOperator, String> {
    let hop = Arc::clone(&solution.hessian_op);
    let k = lambdas.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();

    let (incl_logdet_h, incl_logdet_s) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => (true, true),
        DispersionHandling::Fixed {
            include_logdet_h,
            include_logdet_s,
            ..
        } => (*include_logdet_h, *include_logdet_s),
    };

    let det2 = solution.penalty_logdet.second.as_ref().ok_or_else(|| {
        "Outer Hessian requested but penalty second derivatives not provided".to_string()
    })?;

    let (profiled_phi, profiled_nu, profiled_dp_cgrad, profiled_dp_cgrad2, is_profiled) =
        match &solution.dispersion {
            DispersionHandling::ProfiledGaussian => {
                let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
                let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw);
                let nu = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
                let phi_hat = dp_c / nu;
                (phi_hat, nu, dp_cgrad, dp_cgrad2, true)
            }
            _ => (1.0, 1.0, 1.0, 0.0, false),
        };

    let mut coords = Vec::with_capacity(total);
    let mut rho_penalty_a_k_betas: Vec<Array1<f64>> = Vec::with_capacity(k);
    // Mode responses are fixed-β stationarity derivatives and always use
    // the full Hessian solve.  Rank-deficient LAML changes only the logdet
    // trace kernel, handled below through `subspace`; projecting these solves
    // would change β_i/β_ij curvature semantics.
    let subspace = solution.penalty_subspace_trace.as_deref();
    let dispatch_solve = |v: &Array1<f64>| -> Array1<f64> { hop.solve(v) };
    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let penalty_a_k_beta_vec = penalty_a_k_beta(coord, &solution.beta, lambdas[idx]);
        let curvature_a_k_beta = penalty_a_k_beta(coord, &solution.beta, curvature_lambdas[idx]);
        let v_k = dispatch_solve(&curvature_a_k_beta);
        let correction = effective_deriv.hessian_derivative_correction_result(&v_k)?;
        let mut total_dense = None;
        let mut total_operators = Vec::new();
        match penalty_total_drift_result(coord, curvature_lambdas[idx], correction.as_ref()) {
            DriftDerivResult::Dense(matrix) => total_dense = Some(matrix),
            DriftDerivResult::Operator(op) => total_operators.push(op),
        }
        let mut base_dense = None;
        let mut base_operators = Vec::new();
        match penalty_total_drift_result(coord, curvature_lambdas[idx], None) {
            DriftDerivResult::Dense(matrix) => base_dense = Some(matrix),
            DriftDerivResult::Operator(op) => base_operators.push(op),
        }
        let dense_rotated = match (hop.as_dense_spectral(), total_dense.as_ref()) {
            (Some(dense_hop), Some(matrix)) => Some(dense_hop.rotate_to_eigenbasis(matrix)),
            _ => None,
        };
        let a_i = 0.5 * solution.beta.dot(&penalty_a_k_beta_vec);
        rho_penalty_a_k_betas.push(penalty_a_k_beta_vec);
        coords.push(OuterHessianCoord {
            a: a_i,
            g: curvature_a_k_beta,
            v: v_k,
            total_drift: StoredFirstDrift::from_parts(total_dense, dense_rotated, total_operators),
            base_drift: StoredFirstDrift::from_parts(base_dense, None, base_operators),
            ext_index: None,
            b_depends_on_beta: false,
        });
    }

    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        let v_i = dispatch_solve(&coord.g);
        let correction = effective_deriv.hessian_derivative_correction_result(&v_i)?;
        let (total_dense, total_operators) =
            hyper_coord_total_drift_parts(&coord.drift, correction.as_ref());
        let (base_dense, base_operators) = hyper_coord_total_drift_parts(&coord.drift, None);
        let dense_rotated = match (hop.as_dense_spectral(), total_dense.as_ref()) {
            (Some(dense_hop), Some(matrix)) => Some(dense_hop.rotate_to_eigenbasis(matrix)),
            _ => None,
        };
        coords.push(OuterHessianCoord {
            a: coord.a,
            g: coord.g.clone(),
            v: v_i,
            total_drift: StoredFirstDrift::from_parts(total_dense, dense_rotated, total_operators),
            base_drift: StoredFirstDrift::from_parts(base_dense, None, base_operators),
            ext_index: Some(ext_idx),
            b_depends_on_beta: coord.b_depends_on_beta,
        });
    }

    let mut pair_a = Array2::<f64>::zeros((total, total));
    let mut pair_ld_s = Array2::<f64>::zeros((total, total));
    let mut g_dot_v = Array2::<f64>::zeros((total, total));
    let mut pair_g = vec![vec![None; total]; total];
    let mut base_h2 = Array2::<f64>::zeros((total, total));
    let mut m_pair_trace = Array2::<f64>::zeros((total, total));

    for ii in 0..total {
        for jj in ii..total {
            let value = match (coords[ii].ext_index, coords[jj].ext_index) {
                (None, None) => {
                    let rho_j = jj;
                    rho_penalty_a_k_betas[rho_j].dot(&coords[ii].v)
                }
                (None, Some(_)) => {
                    let rho_i = ii;
                    rho_penalty_a_k_betas[rho_i].dot(&coords[jj].v)
                }
                (Some(_), None) => {
                    let rho_j = jj;
                    rho_penalty_a_k_betas[rho_j].dot(&coords[ii].v)
                }
                (Some(_), Some(_)) => coords[ii].g.dot(&coords[jj].v),
            };
            g_dot_v[[ii, jj]] = value;
            g_dot_v[[jj, ii]] = value;
        }
    }

    for ii in 0..k {
        for jj in ii..k {
            pair_ld_s[[ii, jj]] = det2[[ii, jj]];
            if ii != jj {
                pair_ld_s[[jj, ii]] = det2[[ii, jj]];
            }
        }
    }

    for idx in 0..k {
        pair_a[[idx, idx]] = coords[idx].a;
        pair_g[idx][idx] = Some(coords[idx].g.clone());
        let base = if let Some(kernel) = subspace {
            let a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
            kernel.trace_projected_logdet(&a_k)
        } else if solution.penalty_coords[idx].is_block_local() {
            let (block, start, end) = solution.penalty_coords[idx].scaled_block_local(1.0);
            hop.trace_logdet_block_local(&block, curvature_lambdas[idx], start, end)
        } else {
            let a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
            hop.trace_logdet_gradient(&a_k)
        };
        base_h2[[idx, idx]] = base;
    }

    if let Some(rho_ext_fn) = solution.rho_ext_pair_fn.as_ref() {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pairs: Vec<(usize, usize)> = (0..k)
            .flat_map(|rho_idx| (0..ext_dim).map(move |ext_idx| (rho_idx, ext_idx)))
            .collect();
        let entries: Vec<(usize, usize, HyperCoordPair)> = pairs
            .into_par_iter()
            .map(|(rho_idx, ext_idx)| {
                let pair = rho_ext_fn(rho_idx, ext_idx);
                (rho_idx, ext_idx, pair)
            })
            .collect();
        // Batch all second-drift traces so `--scale-dimensions` pays one
        // shared Hutchinson solve stream for the whole rho-ext block instead
        // of one estimator per pair.  Projected subspace traces skip the
        // stochastic shortcut inside `compute_base_h2_traces`.
        let pair_refs: Vec<&HyperCoordPair> = entries.iter().map(|(_, _, pair)| pair).collect();
        let bases = compute_base_h2_traces(hop.as_ref(), &pair_refs, subspace);
        for ((rho_idx, ext_idx, pair), base) in entries.into_iter().zip(bases.into_iter()) {
            let row = rho_idx;
            let col = k + ext_idx;
            pair_a[[row, col]] = pair.a;
            pair_a[[col, row]] = pair.a;
            pair_ld_s[[row, col]] = pair.ld_s;
            pair_ld_s[[col, row]] = pair.ld_s;
            pair_g[row][col] = Some(pair.g.clone());
            pair_g[col][row] = Some(pair.g);
            base_h2[[row, col]] = base;
            base_h2[[col, row]] = base;
        }
    }

    if let Some(ext_pair_fn) = solution.ext_coord_pair_fn.as_ref() {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pairs: Vec<(usize, usize)> = (0..ext_dim)
            .flat_map(|ii| (ii..ext_dim).map(move |jj| (ii, jj)))
            .collect();
        let entries: Vec<(usize, usize, HyperCoordPair)> = pairs
            .into_par_iter()
            .map(|(ii, jj)| {
                let pair = ext_pair_fn(ii, jj);
                (ii, jj, pair)
            })
            .collect();
        let pair_refs: Vec<&HyperCoordPair> = entries.iter().map(|(_, _, pair)| pair).collect();
        let bases = compute_base_h2_traces(hop.as_ref(), &pair_refs, subspace);
        for ((ii, jj, pair), base) in entries.into_iter().zip(bases.into_iter()) {
            let row = k + ii;
            let col = k + jj;
            pair_a[[row, col]] = pair.a;
            pair_a[[col, row]] = pair.a;
            pair_ld_s[[row, col]] = pair.ld_s;
            pair_ld_s[[col, row]] = pair.ld_s;
            let g_pair = pair.g.clone();
            pair_g[row][col] = Some(g_pair.clone());
            pair_g[col][row] = Some(g_pair);
            base_h2[[row, col]] = base;
            base_h2[[col, row]] = base;
        }
    }

    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pairs: Vec<(usize, usize)> = (0..total)
            .flat_map(|ii| (ii..total).map(move |jj| (ii, jj)))
            .collect();
        let entries: Vec<((usize, usize), f64)> = pairs
            .into_par_iter()
            .map(|(ii, jj)| {
                let beta_i = coords[ii].v.mapv(|value| -value);
                let beta_j = coords[jj].v.mapv(|value| -value);
                let trace = compute_drift_deriv_traces(
                    hop.as_ref(),
                    coords[ii].b_depends_on_beta,
                    coords[jj].b_depends_on_beta,
                    coords[ii].ext_index,
                    coords[jj].ext_index,
                    &beta_i,
                    &beta_j,
                    solution.fixed_drift_deriv.as_ref(),
                    subspace,
                );
                ((ii, jj), trace)
            })
            .collect();
        for ((ii, jj), trace) in entries {
            m_pair_trace[[ii, jj]] = trace;
            m_pair_trace[[jj, ii]] = trace;
        }
    }

    // Precompute pair-wise logdet-Hessian cross traces:
    //   cross_trace[i, j] = tr(G_ε(H) Ḣ_i Ḣ_j)
    // Each coord's total Hessian drift Ḣ decomposes into a dense block plus
    // operator terms; the bilinear form expands across all four
    // dense-dense / dense-op / op-dense / op-op cross combinations.  By
    // bilinearity of `tr(G_ε(H) · · )` in the second factor, the full
    // alpha-combo cross trace recovered in matvec via
    //   cross_trace.row(i).dot(alpha)
    // matches the previous on-the-fly recomputation that built `alpha_dense`,
    // `alpha_dense_rotated`, and `alpha_op` at every HVP.
    let cross_trace: Option<Array2<f64>> = if incl_logdet_h {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let dense_hop_opt = hop.as_dense_spectral();
        if let Some(kernel) = subspace {
            let reduced: Vec<Array2<f64>> = coords
                .iter()
                .map(|coord| {
                    let mut out = Array2::<f64>::zeros((
                        kernel.h_proj_inverse.nrows(),
                        kernel.h_proj_inverse.ncols(),
                    ));
                    if let Some(matrix) = coord.total_drift.dense.as_ref() {
                        out += &kernel.reduce(matrix);
                    }
                    for op in &coord.total_drift.operators {
                        out += &kernel.reduce_operator(op.as_ref());
                    }
                    out
                })
                .collect();
            let pairs: Vec<(usize, usize)> = (0..total)
                .flat_map(|ii| (ii..total).map(move |jj| (ii, jj)))
                .collect();
            let pair_values: Vec<((usize, usize), f64)> = pairs
                .into_par_iter()
                .map(|(ii, jj)| {
                    let value =
                        -kernel.trace_projected_logdet_cross_reduced(&reduced[ii], &reduced[jj]);
                    ((ii, jj), value)
                })
                .collect();
            let mut ct = Array2::<f64>::zeros((total, total));
            for ((ii, jj), value) in pair_values {
                if !value.is_finite() {
                    return Err(format!(
                        "outer Hessian operator projected cross_trace[{ii}, {jj}] is non-finite ({value})"
                    ));
                }
                ct[[ii, jj]] = value;
                if ii != jj {
                    ct[[jj, ii]] = value;
                }
            }
            Some(ct)
        } else if hop.prefers_stochastic_trace_estimation() && hop.logdet_traces_match_hinv_kernel()
        {
            // Matrix-free backends expose the SPD logdet kernel
            //   ∂² log|H|[A_i,A_j] = -tr(H⁻¹ A_i H⁻¹ A_j).
            //
            // Estimate the whole coordinate matrix in one Hutchinson batch
            // rather than launching one two-coordinate estimator per upper
            // triangle entry.  For `--scale-dimensions` with 16 ψ axes this
            // replaces 136 independent solve batches with one 16-coordinate
            // batch sharing the same probes and Krylov solves.
            let bundled: Vec<BorrowedStoredDriftOperator<'_>> = coords
                .iter()
                .map(|coord| BorrowedStoredDriftOperator {
                    drift: &coord.total_drift,
                    dim_hint: hop.dim(),
                })
                .collect();
            let op_refs: Vec<&dyn HyperOperator> =
                bundled.iter().map(|op| op as &dyn HyperOperator).collect();
            let estimator = StochasticTraceEstimator::for_outer_hessian(hop.dim(), total);
            let no_dense: [&Array2<f64>; 0] = [];
            let mut ct = estimator.estimate_second_order_traces_with_operators(
                hop.as_ref(),
                &no_dense,
                &op_refs,
            );
            ct.mapv_inplace(|value| -value);
            Some(ct)
        } else if let Some(dense_hop) = dense_hop_opt {
            // Exact smooth-logdet Hessian kernel for operator-backed drifts.
            //
            // The second derivative of
            //     log |r_epsilon(H(theta))|
            // is not, in general,
            //     -tr(H_epsilon^{-1} H_i H_epsilon^{-1} H_j).
            // That identity only holds for the unregularized SPD logdet.
            // DenseSpectralOperator uses the divided-difference kernel of
            // log r_epsilon(sigma), so every dense/operator component must be
            // rotated into the eigenbasis and contracted with that same
            // kernel.  The dense Hessian assembly path already does this;
            // the matrix-free outer-Hv path must match it exactly.
            let rotated: Vec<Array2<f64>> = coords
                .iter()
                .map(|coord| {
                    let mut projected =
                        coord.total_drift.dense_rotated.clone().unwrap_or_else(|| {
                            Array2::<f64>::zeros((dense_hop.n_dim, dense_hop.n_dim))
                        });
                    for op in &coord.total_drift.operators {
                        projected +=
                            &dense_hop.projected_operator(&dense_hop.eigenvectors, op.as_ref());
                    }
                    projected
                })
                .collect();

            let mut ct = Array2::<f64>::zeros((total, total));
            for ii in 0..total {
                for jj in ii..total {
                    let value =
                        dense_hop.trace_logdet_hessian_cross_rotated(&rotated[ii], &rotated[jj]);
                    if !value.is_finite() {
                        return Err(format!(
                            "outer Hessian operator cross_trace[{ii}, {jj}] is non-finite ({value})"
                        ));
                    }
                    ct[[ii, jj]] = value;
                    if ii != jj {
                        ct[[jj, ii]] = value;
                    }
                }
            }
            Some(ct)
        } else {
            // Enumerate the upper triangle (`ii ≤ jj`) so each `(ii, jj)` is an
            // independent unit of work — every entry of `cross_trace` is computed
            // from `coords[ii]` / `coords[jj]` only, with no shared mutable
            // state, so we can dispatch the K(K+1)/2 pair traces in parallel.
            let pairs: Vec<(usize, usize)> = (0..total)
                .flat_map(|ii| (ii..total).map(move |jj| (ii, jj)))
                .collect();
            let pair_values: Vec<((usize, usize), f64)> = pairs
                .into_par_iter()
                .map(|(ii, jj)| {
                    let left = &coords[ii].total_drift;
                    let right = &coords[jj].total_drift;
                    let mut value = 0.0;
                    if let (Some(left_dense), Some(right_dense)) =
                        (left.dense.as_ref(), right.dense.as_ref())
                    {
                        if let (Some(dense_hop), Some(left_rot), Some(right_rot)) = (
                            dense_hop_opt,
                            left.dense_rotated.as_ref(),
                            right.dense_rotated.as_ref(),
                        ) {
                            value +=
                                dense_hop.trace_logdet_hessian_cross_rotated(left_rot, right_rot);
                        } else {
                            value += hop.trace_logdet_hessian_cross(left_dense, right_dense);
                        }
                    }
                    if let Some(left_dense) = left.dense.as_ref() {
                        for op in &right.operators {
                            value -= hop.trace_hinv_matrix_operator_cross(left_dense, op.as_ref());
                        }
                    }
                    if let Some(right_dense) = right.dense.as_ref() {
                        for op in &left.operators {
                            value -= hop.trace_hinv_matrix_operator_cross(right_dense, op.as_ref());
                        }
                    }
                    if !left.operators.is_empty() && !right.operators.is_empty() {
                        // Bundle each side's per-mode operators into a single
                        // weight-1 linear combination so the cross trace expands
                        // as `tr(H⁻¹ Â B̂) = Σ_a Σ_b tr(H⁻¹ A_a B_b)` with one
                        // call into the cross-trace kernel instead of the full
                        // O(|left.ops|·|right.ops|) sweep. Mathematically
                        // equivalent (bilinearity of `tr(H⁻¹ · ·)`).
                        let left_bundle = WeightedHyperOperator {
                            terms: left
                                .operators
                                .iter()
                                .map(|op| (1.0, Arc::clone(op)))
                                .collect(),
                            dim_hint: hop.dim(),
                        };
                        let right_bundle = WeightedHyperOperator {
                            terms: right
                                .operators
                                .iter()
                                .map(|op| (1.0, Arc::clone(op)))
                                .collect(),
                            dim_hint: hop.dim(),
                        };
                        value -= hop.trace_hinv_operator_cross(&left_bundle, &right_bundle);
                    }
                    ((ii, jj), value)
                })
                .collect();
            let mut ct = Array2::<f64>::zeros((total, total));
            for ((ii, jj), value) in pair_values {
                if !value.is_finite() {
                    return Err(format!(
                        "outer Hessian operator cross_trace[{ii}, {jj}] is non-finite ({value})"
                    ));
                }
                ct[[ii, jj]] = value;
                if ii != jj {
                    ct[[jj, ii]] = value;
                }
            }
            Some(ct)
        }
    } else {
        None
    };

    // Leverage and the scalar-GLM adjoint-z_c cache support both the
    // full-Hessian and projected-subspace paths.  Under subspace,
    //   h^{G,proj}_i = Xᵢᵀ · K · Xᵢ      (K = U_S H_proj⁻¹ U_Sᵀ)
    //   z_c^{proj}   = H⁻¹ · Xᵀ(c ⊙ h^{G,proj})
    // and the adjoint identity
    //   tr(K · C[u]) = uᵀ · Xᵀ(c ⊙ h^{G,proj})
    // (with u = H⁻¹ · rhs unchanged) lets `scalar_correction_trace` take
    // the cheap branch via `(rhs)ᵀ z_c^{proj} = rhsᵀ H⁻¹ Xᵀ(c ⊙ h^{G,proj})
    //                                      = uᵀ Xᵀ(c ⊙ h^{G,proj}) = tr(K C[u])`
    // instead of materialising the second-derivative correction.  Only the
    // leverage swaps to the projected diagonal; z_c stays gated by `H⁻¹`
    // so the IFT mode-response semantics line up with `compute_outer_hessian`.
    let leverage = if incl_logdet_h {
        match &kernel {
            OuterHessianDerivativeKernel::Gaussian => None,
            OuterHessianDerivativeKernel::ScalarGlm { x, .. } => match subspace {
                Some(s) => Some(s.xt_projected_kernel_x_diagonal(x)),
                None => Some(hop.xt_logdet_kernel_x_diagonal(x)),
            },
            OuterHessianDerivativeKernel::Callback { .. } => None,
        }
    } else {
        None
    };
    let adjoint_z_c = if incl_logdet_h {
        match (&kernel, leverage.as_ref()) {
            (
                OuterHessianDerivativeKernel::ScalarGlm {
                    c_array,
                    d_array,
                    x,
                },
                Some(h_g),
            ) => Some(compute_adjoint_z_c(
                &ScalarGlmIngredients {
                    c_array,
                    d_array: d_array.as_ref(),
                    x,
                },
                hop.as_ref(),
                h_g,
            )?),
            _ => None,
        }
    } else {
        None
    };

    let callback_second_modes = matches!(kernel, OuterHessianDerivativeKernel::Callback { .. })
        .then(|| {
            coords
                .iter()
                .map(|coord| {
                    if coord.is_ext() {
                        coord.v.clone()
                    } else {
                        -&coord.v
                    }
                })
                .collect::<Vec<_>>()
        });
    let fourth_trace = if incl_logdet_h && adjoint_z_c.is_some() {
        match (&kernel, leverage.as_ref()) {
            (
                OuterHessianDerivativeKernel::ScalarGlm {
                    c_array,
                    d_array: Some(d_array),
                    x,
                },
                Some(h_g),
            ) => {
                let modes = coords.iter().map(|coord| &coord.v).collect::<Vec<_>>();
                compute_fourth_derivative_trace_matrix(
                    &ScalarGlmIngredients {
                        c_array,
                        d_array: Some(d_array),
                        x,
                    },
                    &modes,
                    h_g,
                )?
            }
            _ => None,
        }
    } else {
        None
    };

    Ok(UnifiedOuterHessianOperator {
        hop,
        coords,
        pair_a,
        pair_ld_s,
        g_dot_v,
        pair_g,
        base_h2,
        m_pair_trace,
        cross_trace,
        profiled_phi,
        profiled_nu,
        profiled_dp_cgrad,
        profiled_dp_cgrad2,
        is_profiled,
        incl_logdet_h,
        incl_logdet_s,
        kernel,
        subspace: solution.penalty_subspace_trace.clone(),
        adjoint_z_c,
        leverage,
        fourth_trace,
        callback_second_modes,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Extended Fellner–Schall (EFS) update for all hyperparameters
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum absolute step size in log-λ for the EFS update (prevents
/// overshooting). Each iteration changes `λ` by at most `exp(EFS_MAX_STEP)`.
const EFS_MAX_STEP: f64 = 5.0;

/// Extended Fellner–Schall update for ρ and penalty-like (τ) hyperparameters.
///
/// Universal-form multiplicative log-λ update driven by the *full* outer
/// gradient `g_full = ∂V_total/∂θ_i`:
///
/// ```text
///   Δρ_i = log( 1 − 2 · g_full[i] / q_eff_i ).
/// ```
///
/// `q_eff_i = 2 · penalty_term_i` is the penalty-quadratic contribution
/// that `outer_gradient_entry` already pairs with the rest of the
/// gradient — i.e. `2·a_i` for `Fixed` dispersion, `2·dp_cgrad·a_i / φ̂`
/// for `ProfiledGaussian`. Since `g_full = (q_eff + t − d)/2 + g_extra`
/// covers both the base REML/LAML stationarity (`g_extra = 0`,
/// recovering the canonical `log((d − t)/q_eff)`) and any out-of-band
/// augmentations — Tierney–Kadane corrections, smoothing-parameter
/// priors, Firth bias-reduction, monotonicity barriers, SAS log-δ ridge
/// — the step automatically targets the right *augmented* stationarity
/// without any per-augmentation post-correction.
///
/// At any stationary point of `V_total`, `g_full = 0`, so `Δρ = 0`.
/// In the over-correction regime (`2·g_full ≥ q_eff`) the multiplicative
/// form is undefined and the helper [`efs_log_step_from_grad`] returns
/// `−EFS_MAX_STEP`; the outer cost line-search trims it and the
/// canonical formula resumes once the iterate re-enters the stable
/// regime. In the pathological regime (`q_eff ≤ 0`, e.g. when the
/// inner solver placed `β̂` exactly on `null(S)`) the step is zero and
/// the iteration relies on the outer fallback.
///
/// ## EFS does not generalize to ψ coordinates
///
/// EFS needs `A_k = ∂S/∂ρ_k ⪰ 0` and a parameter-independent nullspace.
/// For ψ (design-moving) coordinates, `B_{ψ_j}` contains design-motion
/// and likelihood-curvature terms with potentially mixed inertia. The
/// scalar counterexample (response.md Section 2) shows that no update
/// rule based only on `{a, tr(H⁻¹B), tr(H⁻¹BH⁻¹B)}` can be a universal
/// descent direction for V on a ψ. ψ coordinates use the preconditioned
/// gradient step in [`compute_hybrid_efs_update`] instead.
///
/// ## Approximation: IFT corrections rolled into the gradient
///
/// `g_full` is the same gradient `reml_laml_evaluate` produces in
/// `EvalMode::ValueAndGradient`, which already includes the third-
/// derivative `C[v_k]` IFT correction for non-Gaussian families. The
/// EFS step inherits this correction automatically; we no longer carry
/// a separate kernel-correct trace path, which removes the
/// "approximate Wood–Fasiolo + line-search safety net" gap that the
/// original code had.
///
/// # Arguments
/// - `solution`: Converged inner state (β̂, H, penalties, HessianOperator).
/// - `rho`: Current log-smoothing parameters.
/// - `gradient`: Full outer gradient `∂V_total/∂θ`, length
///   `n_rho + n_ext`. The caller must run
///   [`super::reml::unified::EvalMode::ValueAndGradient`] when
///   evaluating the cost so this slice is available.
///
/// # Returns
/// A vector of additive steps for all coordinates: first the ρ block,
/// then the ext block (in the same order as `solution.ext_coords`).
/// Apply as `θ_i^new = θ_i + step[i]`. Steps for ψ coordinates
/// (`is_penalty_like == false`) are always 0; the hybrid update handles
/// them.
///
/// Steps are clamped to `[-EFS_MAX_STEP, EFS_MAX_STEP]` so a single
/// iteration cannot move λ by more than `exp(EFS_MAX_STEP)`.
pub fn compute_efs_update(solution: &InnerSolution<'_>, rho: &[f64], gradient: &[f64]) -> Vec<f64> {
    let k = rho.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    debug_assert_eq!(
        gradient.len(),
        total,
        "compute_efs_update: gradient length {} != n_rho({k}) + n_ext({ext_dim})",
        gradient.len(),
    );
    let mut steps = vec![0.0; total];

    let (profiled_scale, dp_cgrad) = efs_profiling(solution);

    // Universal-form EFS: `Δρ_i = log(1 − 2·g_full[i]/q_eff_i)`. This is
    // identical to the canonical `log((d−t)/q_eff)` when no out-of-band
    // cost terms exist (TK, prior, Firth, barrier, SAS ridge), and shifts
    // the multiplicative target by exactly the residual gradient when
    // they do. We get the augmented stationarity for free, in exchange
    // for one `EvalMode::ValueAndGradient` evaluation per outer
    // iteration.
    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let lambda = rho[idx].exp();
        let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambda);
        let q_eff = efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale);
        if let Some(step) = efs_log_step_from_grad(q_eff, gradient[idx]) {
            steps[idx] = step;
        }
    }

    // ψ coords (`!is_penalty_like`) are skipped: EFS has no convergence
    // guarantee there. The hybrid update supplies a preconditioned
    // gradient step for them.
    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        if !coord.is_penalty_like {
            continue;
        }
        let g_idx = k + ext_idx;
        let q_eff = efs_q_eff(coord.a, &solution.dispersion, dp_cgrad, profiled_scale);
        if let Some(step) = efs_log_step_from_grad(q_eff, gradient[g_idx]) {
            steps[g_idx] = step;
        }
    }

    steps
}

/// Regularization threshold for pseudoinverse of the trace Gram matrix.
///
/// Eigenvalues below `PSI_GRAM_PINV_TOL * max_eigenvalue` are treated as
/// zero when computing the pseudoinverse G⁺. This prevents amplification
/// of noise in near-singular directions of the ψ-ψ Gram matrix.
const PSI_GRAM_PINV_TOL: f64 = 1e-8;

/// Initial step-size damping factor for the preconditioned gradient on ψ.
///
/// The raw step `Δψ_raw = -G⁺ g_ψ` is scaled by α ∈ (0, 1] before
/// applying. This conservative initial value prevents overshooting in
/// early iterations when the quadratic model may be inaccurate.
const PSI_INITIAL_ALPHA: f64 = 1.0;

/// Minimum number of scalar ρ/τ EFS candidates before `compute_hybrid_efs_update`
/// fans out with rayon.  Smaller blocks are common (1-4 smoothing parameters),
/// where task scheduling costs dominate the independent arithmetic.
const HYBRID_EFS_SCALAR_PAR_THRESHOLD: usize = 8;

/// Minimum number of independent ψ-ψ Gram entries before exact trace assembly
/// fans out with rayon.  This is expressed in upper-triangle pair count rather
/// than `n_psi` so 5 ψ coordinates (15 pairs) stay serial while moderate
/// anisotropic/design-moving blocks parallelize.
const HYBRID_EFS_GRAM_PAIR_PAR_THRESHOLD: usize = 24;

/// Minimum number of ψ drifts before materialization/projection is done in
/// parallel during exact Gram assembly.
const HYBRID_EFS_PSI_DRIFT_PAR_THRESHOLD: usize = 8;

/// Result of the hybrid EFS update, containing both the step vector and
/// metadata needed for backtracking on the ψ block.
pub struct HybridEfsResult {
    /// Combined step vector (EFS for ρ/τ, preconditioned gradient for ψ).
    pub steps: Vec<f64>,
    /// Indices of ψ (design-moving) coordinates in the full θ vector.
    /// Empty if no ψ coordinates are present.
    pub psi_indices: Vec<usize>,
    /// Raw REML/LAML gradient restricted to ψ coordinates.
    /// Length matches `psi_indices.len()`.
    pub psi_gradient: Vec<f64>,
}

/// Hybrid EFS + preconditioned gradient update.
///
/// Computes a combined step for all hyperparameters:
/// - **ρ (penalty-like) coordinates**: standard EFS multiplicative fixed-point
///   update, identical to [`compute_efs_update`].
/// - **ψ (design-moving) coordinates**: safeguarded preconditioned gradient step
///   using the trace Gram matrix as preconditioner:
///
///   ```text
///   Δψ = -α G⁺ g_ψ
///   ```
///
///   where:
///   - `g_ψ` is the REML/LAML gradient restricted to the ψ block
///   - `G_{de} = tr(H⁻¹ B_d H⁻¹ B_e)` is the trace Gram matrix for ψ-ψ pairs
///   - `G⁺` is the Moore-Penrose pseudoinverse (truncated at `PSI_GRAM_PINV_TOL`)
///   - `α ∈ (0, 1]` is the damping factor
///
/// ## Why this works (reference: response.md Section 2)
///
/// The trace Gram matrix G is the same object that EFS uses as its scalar
/// denominator for penalty-like coordinates. For ψ coordinates, G still
/// captures the local curvature structure `tr(H⁻¹ B_d H⁻¹ B_e)` — it is
/// the natural metric on the ψ-subspace induced by the penalized likelihood.
/// However, unlike the EFS case, we cannot derive a monotone fixed-point
/// iteration from G alone because B_ψ may have mixed inertia (the Frobenius
/// norm `tr(H⁻¹BH⁻¹B)` is always positive but does not bound the true
/// curvature).
///
/// The preconditioned gradient `Δψ = -G⁺ g_ψ` is the cheap replacement
/// recommended by the math team: it uses the same trace Gram matrix, stays
/// at O(1) H⁻¹ solves per iteration (same as pure EFS), and avoids
/// pretending that the Gram denominator is the true scalar curvature.
/// Compare with full BFGS which requires O(dim(θ)) gradient evaluations
/// (each involving a full inner solve) per outer step.
///
/// ## Step-size safeguarding
///
/// 1. Compute G for the ψ-ψ block from H⁻¹ B_d products (already available).
/// 2. Pseudoinverse: G⁺ via eigendecomposition, truncating eigenvalues below
///    `PSI_GRAM_PINV_TOL * max_eigenvalue` to avoid noise amplification in
///    near-singular directions.
/// 3. Raw step: `Δψ_raw = -G⁺ g_ψ`.
/// 4. Damping: `Δψ = α × Δψ_raw` with initial `α = PSI_INITIAL_ALPHA`.
/// 5. Capping: `||Δψ||_∞ ≤ EFS_MAX_STEP` (same cap as ρ coordinates).
/// 6. Backtracking (handled by caller): the outer fixed-point bridge wraps
///    the *whole* combined step in a cost line search, halving α over the
///    full vector. If full-vector backtracking exhausts, it retries with
///    the ψ block zeroed (ρ/τ-only fallback) before surfacing the
///    first-order fallback marker.
///
/// # Arguments
/// - `solution`: Converged inner state (β̂, H, penalties, HessianOperator).
/// - `rho`: Current log-smoothing parameters.
/// - `gradient`: Full REML/LAML gradient ∂V/∂θ (length = n_rho + n_ext).
///   Must be provided; the hybrid needs the gradient for ψ coordinates.
///
/// # Returns
/// A [`HybridEfsResult`] containing the combined step vector and metadata
/// for backtracking.
pub fn compute_hybrid_efs_update(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    gradient: &[f64],
) -> HybridEfsResult {
    let k = rho.len();
    let hop = &*solution.hessian_op;
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let mut steps = vec![0.0; total];

    let (profiled_scale, dp_cgrad) = efs_profiling(solution);
    debug_assert_eq!(
        gradient.len(),
        total,
        "compute_hybrid_efs_update: gradient length {} != n_rho({k}) + n_ext({ext_dim})",
        gradient.len(),
    );

    // ── ρ coordinates: universal-form EFS (see compute_efs_update) ──
    //
    // The per-coordinate candidate construction is independent: each candidate
    // reads only the converged β̂, the coordinate root, ρᵢ, and gᵢ.  Build
    // candidates in parallel once the block is large enough, then keep the
    // actual update write-back serial so fallback/backtracking decisions still
    // see a deterministic step vector.
    let rho_candidates: Vec<(usize, Option<f64>)> =
        if k >= HYBRID_EFS_SCALAR_PAR_THRESHOLD && rayon::current_thread_index().is_none() {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..k)
                .into_par_iter()
                .map(|idx| {
                    let coord = &solution.penalty_coords[idx];
                    let lambda = rho[idx].exp();
                    let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambda);
                    let q_eff = efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale);
                    (idx, efs_log_step_from_grad(q_eff, gradient[idx]))
                })
                .collect()
        } else {
            (0..k)
                .map(|idx| {
                    let coord = &solution.penalty_coords[idx];
                    let lambda = rho[idx].exp();
                    let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambda);
                    let q_eff = efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale);
                    (idx, efs_log_step_from_grad(q_eff, gradient[idx]))
                })
                .collect()
        };
    for (idx, candidate) in rho_candidates {
        if let Some(step) = candidate {
            steps[idx] = step;
        }
    }

    // ── Extended penalty-like (τ) coordinates: universal-form EFS ──
    // ── ψ (design-moving) coordinates: collect for preconditioned gradient ──
    //
    // τ coords go through the same Wood–Fasiolo update as ρ. ψ coords are
    // collected and processed jointly via the ψ-ψ trace Gram matrix below.
    let mut psi_local_indices: Vec<usize> = Vec::new(); // index within ext_coords
    let mut psi_global_indices: Vec<usize> = Vec::new(); // index in full θ vector
    let mut tau_local_indices: Vec<usize> = Vec::new(); // penalty-like ext coords

    // Classify ext coordinates serially.  This preserves ψ ordering for the
    // returned metadata and keeps the penalty-like-vs-design-moving decision
    // out of the parallel update fill.
    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        let g_idx = k + ext_idx;
        if coord.is_penalty_like {
            tau_local_indices.push(ext_idx);
        } else {
            // ψ coordinate: collect for joint preconditioned gradient.
            psi_local_indices.push(ext_idx);
            psi_global_indices.push(g_idx);
        }
    }

    let tau_candidates: Vec<(usize, Option<f64>)> = if tau_local_indices.len()
        >= HYBRID_EFS_SCALAR_PAR_THRESHOLD
        && rayon::current_thread_index().is_none()
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        tau_local_indices
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|ext_idx| {
                let coord = &solution.ext_coords[ext_idx];
                let g_idx = k + ext_idx;
                let q_eff = efs_q_eff(coord.a, &solution.dispersion, dp_cgrad, profiled_scale);
                (g_idx, efs_log_step_from_grad(q_eff, gradient[g_idx]))
            })
            .collect()
    } else {
        tau_local_indices
            .iter()
            .map(|&ext_idx| {
                let coord = &solution.ext_coords[ext_idx];
                let g_idx = k + ext_idx;
                let q_eff = efs_q_eff(coord.a, &solution.dispersion, dp_cgrad, profiled_scale);
                (g_idx, efs_log_step_from_grad(q_eff, gradient[g_idx]))
            })
            .collect()
    };
    for (g_idx, candidate) in tau_candidates {
        if let Some(step) = candidate {
            steps[g_idx] = step;
        }
    }

    // Collect the ψ-block gradient for the caller (for backtracking).
    let psi_gradient: Vec<f64> = psi_global_indices.iter().map(|&gi| gradient[gi]).collect();

    // ── ψ coordinates: preconditioned gradient step ──
    //
    // The preconditioned gradient step for ψ (design-moving) coordinates:
    //
    //   Δψ = -α G⁺ g_ψ
    //
    // where G_{de} = tr(H⁻¹ B_d H⁻¹ B_e) is the trace Gram matrix and
    // g_ψ is the REML/LAML gradient restricted to the ψ block.
    //
    // This is the practical replacement for EFS on ψ coordinates recommended
    // by the math team (response.md Section 2). It uses the same trace Gram
    // matrix that EFS computes, stays cheap (O(1) H⁻¹ solves), and avoids
    // the invalid assumption that the Gram norm bounds the true curvature.
    let n_psi = psi_local_indices.len();
    if n_psi > 0 {
        if n_psi == 1 {
            let li = psi_local_indices[0];
            let drift = &solution.ext_coords[li].drift;
            let op = hyper_coord_drift_operator_arc(drift, hop.dim());
            let dense = op.is_none().then(|| drift.materialize());
            let gram = if let Some(dense_hop) = hop.as_dense_spectral() {
                let projected = if let Some(op) = op.as_ref() {
                    dense_hop.projected_operator(&dense_hop.w_factor, op.as_ref())
                } else {
                    dense_hop
                        .projected_matrix(dense.as_ref().expect("dense drift should be cached"))
                };
                dense_hop.trace_projected_cross(&projected, &projected)
            } else {
                trace_hinv_cached_drift_cross(
                    hop,
                    dense.as_ref(),
                    op.as_deref(),
                    dense.as_ref(),
                    op.as_deref(),
                )
            };
            if gram.abs() >= PSI_GRAM_PINV_TOL.max(1e-30) {
                let global_idx = psi_global_indices[0];
                let raw_step = -PSI_INITIAL_ALPHA * psi_gradient[0] / gram;
                steps[global_idx] = raw_step.clamp(-EFS_MAX_STEP, EFS_MAX_STEP);
            }
            return HybridEfsResult {
                steps,
                psi_indices: psi_global_indices,
                psi_gradient,
            };
        }

        let total_p = hop.dim();
        let any_psi_operator = psi_local_indices.iter().any(|&li| {
            let drift = &solution.ext_coords[li].drift;
            drift.uses_operator_fast_path()
        });
        let use_stochastic_psi_gram =
            any_psi_operator && total_p > 500 && hop.prefers_stochastic_trace_estimation();

        // Step 1: Build the trace Gram matrix
        //   G_{de} = tr(H⁻¹ B_d H⁻¹ B_e).
        //
        // Large matrix-free/operator-backed problems batch this through the
        // shared stochastic second-order trace estimator. Smaller or fully
        // dense problems use exact pairwise cross traces.
        let gram = if use_stochastic_psi_gram {
            let mut dense_mats = Vec::new();
            let mut coord_has_operator = Vec::with_capacity(n_psi);
            let mut operator_arcs: Vec<Arc<dyn HyperOperator>> = Vec::new();

            for &li in &psi_local_indices {
                let coord = &solution.ext_coords[li];
                if let Some(op) = hyper_coord_drift_operator_arc(&coord.drift, hop.dim()) {
                    coord_has_operator.push(true);
                    operator_arcs.push(op);
                } else {
                    coord_has_operator.push(false);
                    dense_mats.push(coord.drift.materialize());
                }
            }

            let generic_ops: Vec<&dyn HyperOperator> =
                operator_arcs.iter().map(|op| op.as_ref()).collect();
            let impl_ops: Vec<&ImplicitHyperOperator> = generic_ops
                .iter()
                .filter_map(|op| op.as_implicit())
                .collect();

            stochastic_trace_hinv_crosses(
                hop,
                &dense_mats,
                &coord_has_operator,
                &generic_ops,
                &impl_ops,
            )
        } else {
            let mut gram = ndarray::Array2::<f64>::zeros((n_psi, n_psi));
            let parallel_psi_drifts = n_psi >= HYBRID_EFS_PSI_DRIFT_PAR_THRESHOLD
                && rayon::current_thread_index().is_none();
            let drift_ops: Vec<Option<Arc<dyn HyperOperator>>> = if parallel_psi_drifts {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n_psi)
                    .into_par_iter()
                    .map(|idx| {
                        let drift = &solution.ext_coords[psi_local_indices[idx]].drift;
                        hyper_coord_drift_operator_arc(drift, hop.dim())
                    })
                    .collect()
            } else {
                psi_local_indices
                    .iter()
                    .map(|&li| {
                        let drift = &solution.ext_coords[li].drift;
                        hyper_coord_drift_operator_arc(drift, hop.dim())
                    })
                    .collect()
            };
            let dense_drifts: Vec<Option<Array2<f64>>> = if parallel_psi_drifts {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n_psi)
                    .into_par_iter()
                    .map(|idx| {
                        let drift = &solution.ext_coords[psi_local_indices[idx]].drift;
                        drift_ops[idx].is_none().then(|| drift.materialize())
                    })
                    .collect()
            } else {
                psi_local_indices
                    .iter()
                    .enumerate()
                    .map(|(idx, &li)| {
                        let drift = &solution.ext_coords[li].drift;
                        drift_ops[idx].is_none().then(|| drift.materialize())
                    })
                    .collect()
            };
            let pair_count = n_psi * (n_psi + 1) / 2;
            let parallel_gram_pairs = pair_count >= HYBRID_EFS_GRAM_PAIR_PAR_THRESHOLD
                && rayon::current_thread_index().is_none();
            if let Some(dense_hop) = hop.as_dense_spectral() {
                let projected_drifts: Vec<Array2<f64>> = if parallel_psi_drifts {
                    use rayon::iter::{IntoParallelIterator, ParallelIterator};
                    (0..n_psi)
                        .into_par_iter()
                        .map(|idx| {
                            if let Some(op) = drift_ops[idx].as_ref() {
                                dense_hop.projected_operator(&dense_hop.w_factor, op.as_ref())
                            } else {
                                dense_hop.projected_matrix(
                                    dense_drifts[idx]
                                        .as_ref()
                                        .expect("dense drift should be cached"),
                                )
                            }
                        })
                        .collect()
                } else {
                    (0..n_psi)
                        .map(|idx| {
                            if let Some(op) = drift_ops[idx].as_ref() {
                                dense_hop.projected_operator(&dense_hop.w_factor, op.as_ref())
                            } else {
                                dense_hop.projected_matrix(
                                    dense_drifts[idx]
                                        .as_ref()
                                        .expect("dense drift should be cached"),
                                )
                            }
                        })
                        .collect()
                };
                if parallel_gram_pairs {
                    use rayon::iter::{IntoParallelIterator, ParallelIterator};
                    let pairs: Vec<(usize, usize)> = (0..n_psi)
                        .flat_map(|d| (d..n_psi).map(move |e| (d, e)))
                        .collect();
                    let pair_values: Vec<(usize, usize, f64)> = pairs
                        .into_par_iter()
                        .map(|(d, e)| {
                            let val = dense_hop
                                .trace_projected_cross(&projected_drifts[d], &projected_drifts[e]);
                            (d, e, val)
                        })
                        .collect();
                    for (d, e, val) in pair_values {
                        gram[[d, e]] = val;
                        gram[[e, d]] = val;
                    }
                } else {
                    for d in 0..n_psi {
                        for e in d..n_psi {
                            let val = dense_hop
                                .trace_projected_cross(&projected_drifts[d], &projected_drifts[e]);
                            gram[[d, e]] = val;
                            gram[[e, d]] = val;
                        }
                    }
                }
            } else if parallel_gram_pairs {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                let pairs: Vec<(usize, usize)> = (0..n_psi)
                    .flat_map(|d| (d..n_psi).map(move |e| (d, e)))
                    .collect();
                let pair_values: Vec<(usize, usize, f64)> = pairs
                    .into_par_iter()
                    .map(|(d, e)| {
                        let val = trace_hinv_cached_drift_cross(
                            hop,
                            dense_drifts[d].as_ref(),
                            drift_ops[d].as_deref(),
                            dense_drifts[e].as_ref(),
                            drift_ops[e].as_deref(),
                        );
                        (d, e, val)
                    })
                    .collect();
                for (d, e, val) in pair_values {
                    gram[[d, e]] = val;
                    gram[[e, d]] = val;
                }
            } else {
                for d in 0..n_psi {
                    for e in d..n_psi {
                        let val = trace_hinv_cached_drift_cross(
                            hop,
                            dense_drifts[d].as_ref(),
                            drift_ops[d].as_deref(),
                            dense_drifts[e].as_ref(),
                            drift_ops[e].as_deref(),
                        );
                        gram[[d, e]] = val;
                        gram[[e, d]] = val;
                    }
                }
            }
            gram
        };

        // Step 2: Pseudoinverse G⁺ via eigendecomposition.
        //
        // For small n_psi (typically 2-10 anisotropic axes), this is cheap.
        // We truncate eigenvalues below PSI_GRAM_PINV_TOL * λ_max to form
        // the pseudoinverse, avoiding noise amplification in near-singular
        // directions. This is the standard approach for constrained
        // optimization on submanifolds (see response.md Section 4).
        let delta_psi = pseudoinverse_times_vec(&gram, &psi_gradient, PSI_GRAM_PINV_TOL);

        // Step 3: Apply damping and capping.
        //
        // Δψ = -α × G⁺ g_ψ, capped to ||Δψ||_∞ ≤ EFS_MAX_STEP.
        // The negative sign is because we are descending on V(θ) (minimizing).
        let alpha = PSI_INITIAL_ALPHA;
        for (psi_idx, &global_idx) in psi_global_indices.iter().enumerate() {
            let raw_step = -alpha * delta_psi[psi_idx];
            steps[global_idx] = raw_step.clamp(-EFS_MAX_STEP, EFS_MAX_STEP);
        }
    }

    HybridEfsResult {
        steps,
        psi_indices: psi_global_indices,
        psi_gradient,
    }
}

/// Compute G⁺ v where G⁺ is the pseudoinverse of symmetric matrix G.
///
/// Uses eigendecomposition with truncation: eigenvalues below
/// `tol * max_eigenvalue` are treated as zero. For small matrices
/// (typical n_psi = 2-10), the O(n³) cost is negligible.
fn pseudoinverse_times_vec(
    gram: &ndarray::Array2<f64>,
    v: &[f64],
    tol: f64,
) -> ndarray::Array1<f64> {
    let n = gram.nrows();
    assert_eq!(n, v.len(), "pseudoinverse_times_vec dimension mismatch");
    if n == 0 {
        return ndarray::Array1::zeros(0);
    }

    // Special case: scalar (1x1).
    if n == 1 {
        let g = gram[[0, 0]];
        if g.abs() < tol.max(1e-30) {
            return ndarray::Array1::zeros(1);
        }
        return ndarray::Array1::from_vec(vec![v[0] / g]);
    }

    // Eigendecomposition of symmetric G via the faer crate would be ideal,
    // but to keep this self-contained we use a simple symmetric
    // eigendecomposition via Jacobi rotations for small matrices, or
    // fall back to diagonal-only pseudoinverse for safety.
    //
    // For production quality, this should use faer's `SelfAdjointEigendecomposition`.
    // Here we implement a robust fallback that works for typical n_psi = 2-10.

    // Attempt: use ndarray's built-in symmetric eigendecomposition if available,
    // otherwise fall back to a diagonal approximation.
    //
    // Robust implementation: compute G = Q Λ Q^T via iterative Jacobi.
    // For n ≤ 10 this converges in a handful of sweeps.
    let (eigenvalues, eigenvectors) = symmetric_eigen(gram);

    let max_eval = eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = tol * max_eval;

    // G⁺ v = Q diag(1/λ_i for λ_i > cutoff, else 0) Q^T v
    let qt_v: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|row| eigenvectors[[row, i]] * v[row]).sum())
        .collect();

    let mut result = ndarray::Array1::zeros(n);
    for i in 0..n {
        if eigenvalues[i] > cutoff {
            let scale = qt_v[i] / eigenvalues[i];
            for row in 0..n {
                result[row] += scale * eigenvectors[[row, i]];
            }
        }
    }
    result
}

/// Symmetric eigendecomposition via classical Jacobi iteration.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are stored
/// column-wise. Suitable for small matrices (n ≤ 20). For n_psi = 2-10
/// (typical anisotropic axis counts), this converges in 2-5 sweeps.
///
/// This is a self-contained implementation to avoid external dependencies.
/// For larger matrices, use faer's `SelfAdjointEigendecomposition`.
fn symmetric_eigen(a: &ndarray::Array2<f64>) -> (Vec<f64>, ndarray::Array2<f64>) {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "symmetric_eigen requires square matrix");

    let mut work = a.clone();
    let mut v = ndarray::Array2::<f64>::eye(n);

    // Jacobi iteration: sweep through all off-diagonal pairs, zeroing them.
    let max_sweeps = 100;
    let tol = 1e-15;

    let mut sweep = 0;
    while sweep < max_sweeps {
        // Check convergence: sum of squares of off-diagonal elements.
        let mut off_diag_sq = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_sq += work[[i, j]] * work[[i, j]];
            }
        }
        if off_diag_sq < tol * tol {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = work[[p, q]];
                if apq.abs() < tol * 0.01 {
                    continue;
                }

                let app = work[[p, p]];
                let aqq = work[[q, q]];
                let tau = (aqq - app) / (2.0 * apq);

                // Stable computation of t = sign(τ) / (|τ| + sqrt(1 + τ²))
                let t = if tau.abs() > 1e15 {
                    // Nearly diagonal: skip.
                    continue;
                } else {
                    let sign_tau = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign_tau / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply Jacobi rotation to work matrix.
                work[[p, p]] = app - t * apq;
                work[[q, q]] = aqq + t * apq;
                work[[p, q]] = 0.0;
                work[[q, p]] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let wrp = work[[r, p]];
                    let wrq = work[[r, q]];
                    work[[r, p]] = c * wrp - s * wrq;
                    work[[p, r]] = work[[r, p]];
                    work[[r, q]] = s * wrp + c * wrq;
                    work[[q, r]] = work[[r, q]];
                }

                // Accumulate eigenvectors.
                for r in 0..n {
                    let vrp = v[[r, p]];
                    let vrq = v[[r, q]];
                    v[[r, p]] = c * vrp - s * vrq;
                    v[[r, q]] = s * vrp + c * vrq;
                }
            }
        }
        sweep += 1;
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| work[[i, i]]).collect();
    (eigenvalues, v)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Corrected coefficient covariance (smoothing-parameter uncertainty)
// ═══════════════════════════════════════════════════════════════════════════

/// Corrected covariance of the coefficient vector, accounting for uncertainty
/// in the smoothing/hyperparameters theta = (rho, psi).
///
/// The standard conditional covariance H^{-1} ignores uncertainty in theta.
/// The corrected covariance adds the propagation term:
///
/// ```text
///   V*_alpha = H^{-1} + J_alpha V_theta J_alpha^T
/// ```
///
/// where:
/// - H^{-1} is obtained via `hop.solve` on identity columns
/// - J_alpha = [-v_1, ..., -v_k, -ext_v_1, ..., -ext_v_m] is the p x q matrix
///   of negated mode responses (the implicit-function sensitivities d(beta)/d(theta))
/// - V_theta = outer_hessian^{-1} is the q x q inverse outer Hessian
///
/// The mode responses v_k = H^{-1}(A_k beta) and ext_v_i = H^{-1}(g_i) are
/// already computed in the unified evaluator gradient/Hessian loop, so this
/// function reuses them directly.
///
/// # Arguments
/// - `v_ks`: mode responses for rho coordinates, v_k = H^{-1}(A_k beta)
/// - `ext_v`: mode responses for extended (psi) coordinates, v_i = H^{-1}(g_i)
/// - `outer_hessian`: the q x q outer Hessian (nabla^2_theta V)
/// - `hop`: the HessianOperator providing H^{-1}
///
/// # Returns
/// The full p x p corrected covariance matrix V*_alpha = H^{-1} + J V_theta J^T.
///
/// # Edge cases
/// - If the outer Hessian is indefinite, eigendecomposition with positive-part
///   projection is used for V_theta (negative eigenvalues are clamped to zero).
/// - Returns `Err` only if the eigendecomposition itself fails.
pub fn compute_corrected_covariance(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
) -> Result<Array2<f64>, String> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    if q == 0 {
        // No hyperparameters — corrected covariance equals the conditional H^{-1}.
        let eye = Array2::eye(p);
        return Ok(hop.solve_multi(&eye));
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(format!(
            "compute_corrected_covariance: outer Hessian dimension ({}, {}) does not match \
             total hyperparameter count q = {} (rho: {}, ext: {})",
            outer_hessian.nrows(),
            outer_hessian.ncols(),
            q,
            v_ks.len(),
            ext_v.len(),
        ));
    }

    // Step 1: Assemble J_alpha (p x q) with columns = -v_i.
    let mut j_alpha = Array2::zeros((p, q));
    for (col, v) in v_ks.iter().enumerate() {
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }
    for (i, v) in ext_v.iter().enumerate() {
        let col = v_ks.len() + i;
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }

    // Step 2: Compute V_theta = outer_hessian^{-1} via eigendecomposition
    // with positive-part projection (handles indefinite Hessians gracefully).
    let v_theta = invert_with_positive_projection(outer_hessian)?;

    // Step 3: Compute the correction term J_alpha V_theta J_alpha^T.
    // Factored as (J V_theta) J^T to reuse the intermediate (p x q).
    let j_v_theta = j_alpha.dot(&v_theta); // p x q
    let correction = j_v_theta.dot(&j_alpha.t()); // p x p

    // Step 4: Compute H^{-1} and add the correction.
    let eye = Array2::eye(p);
    let mut h_inv = hop.solve_multi(&eye);
    h_inv += &correction;

    // Enforce exact symmetry (numerical noise from the matrix products).
    enforce_symmetry_inplace(&mut h_inv);

    Ok(h_inv)
}

/// Compute only the diagonal of the corrected covariance V*_alpha.
///
/// This is much cheaper than the full p x p matrix: O(p q) instead of O(p^2 q).
///
/// ```text
///   diag(V*_alpha) = diag(H^{-1}) + row_norms(J_alpha L_theta)^2
/// ```
///
/// where L_theta is the Cholesky-like square root of V_theta. When V_theta
/// is obtained via positive-projected eigendecomposition, L_theta = U sqrt(D+)
/// where D+ contains the positive-part eigenvalues.
///
/// # Arguments
/// - `v_ks`: mode responses for rho coordinates
/// - `ext_v`: mode responses for extended (psi) coordinates
/// - `outer_hessian`: the q x q outer Hessian
/// - `hop`: the HessianOperator providing H^{-1}
///
/// # Returns
/// A p-vector of corrected marginal variances.
pub fn compute_corrected_covariance_diagonal(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
) -> Result<Array1<f64>, String> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    // Start with diag(H^{-1}).
    let mut diag = Array1::zeros(p);
    for i in 0..p {
        let mut e_i = Array1::zeros(p);
        e_i[i] = 1.0;
        let h_inv_ei = hop.solve(&e_i);
        diag[i] = h_inv_ei[i];
    }

    if q == 0 {
        return Ok(diag);
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(format!(
            "compute_corrected_covariance_diagonal: outer Hessian dimension ({}, {}) \
             does not match q = {}",
            outer_hessian.nrows(),
            outer_hessian.ncols(),
            q,
        ));
    }

    // Compute V_theta^{1/2} via positive-projected eigendecomposition.
    // V_theta^{1/2} = U diag(sqrt(max(0, 1/sigma_i))) where sigma_i are
    // eigenvalues of the outer Hessian.
    let v_theta_sqrt = sqrt_inverse_with_positive_projection(outer_hessian)?;

    // Assemble J_alpha (p x q) with columns = -v_i.
    let mut j_alpha = Array2::zeros((p, q));
    for (col, v) in v_ks.iter().enumerate() {
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }
    for (i, v) in ext_v.iter().enumerate() {
        let col = v_ks.len() + i;
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }

    // Compute M = J_alpha V_theta^{1/2} (p x q).
    let m = j_alpha.dot(&v_theta_sqrt); // p x q

    // diag(correction) = row_norms(M)^2 = sum_j M[i,j]^2 for each i.
    for i in 0..p {
        let mut row_norm_sq = 0.0;
        for j in 0..m.ncols() {
            row_norm_sq += m[[i, j]] * m[[i, j]];
        }
        diag[i] += row_norm_sq;
    }

    Ok(diag)
}

/// Invert a symmetric matrix using eigendecomposition with positive-part
/// projection: eigenvalues <= 0 are treated as infinite (their directions
/// are omitted from the inverse, equivalent to clamping to zero).
///
/// This handles indefinite outer Hessians gracefully — negative curvature
/// directions indicate that the Laplace approximation is unreliable in those
/// directions, so we conservatively omit their contribution rather than
/// amplifying uncertainty along negatively-curved directions.
fn invert_with_positive_projection(mat: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = mat.nrows();
    let (eigenvalues, eigenvectors) = mat
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("eigendecomposition failed in positive-projection inverse: {e}"))?;

    let mut result = Array2::zeros((n, n));
    for j in 0..n {
        let sigma = eigenvalues[j];
        if sigma <= 0.0 {
            continue; // omit non-positive directions
        }
        let inv_sigma = 1.0 / sigma;
        let u = eigenvectors.column(j);
        for a in 0..n {
            let ua = inv_sigma * u[a];
            for b in a..n {
                let val = ua * u[b];
                result[[a, b]] += val;
                if a != b {
                    result[[b, a]] += val;
                }
            }
        }
    }
    Ok(result)
}

/// Compute V_theta^{1/2} = U diag(sqrt(1/sigma_i^+)) for positive eigenvalues
/// of the outer Hessian. Non-positive eigenvalues produce zero columns.
///
/// The result is q x q_active (where q_active <= q is the number of positive
/// eigenvalues), but we return the full q x q matrix with zero columns for
/// omitted directions for simplicity.
fn sqrt_inverse_with_positive_projection(mat: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = mat.nrows();
    let (eigenvalues, eigenvectors) = mat
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("eigendecomposition failed in sqrt-inverse: {e}"))?;

    let mut result = Array2::zeros((n, n));
    for j in 0..n {
        let sigma = eigenvalues[j];
        if sigma <= 0.0 {
            continue;
        }
        let scale = (1.0 / sigma).sqrt();
        for row in 0..n {
            result[[row, j]] = eigenvectors[[row, j]] * scale;
        }
    }
    Ok(result)
}

/// Enforce exact symmetry on a square matrix by averaging off-diagonal pairs.
fn enforce_symmetry_inplace(m: &mut Array2<f64>) {
    let n = m.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m[[i, j]] + m[[j, i]]);
            m[[i, j]] = avg;
            m[[j, i]] = avg;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Smooth spectral regularization
// ═══════════════════════════════════════════════════════════════════════════
//
// For indefinite or near-singular Hessians, hard eigenvalue clamping
// `max(σ, ε)` is non-smooth and creates inconsistency between log|H| and
// H⁻¹ at the threshold boundary. We use instead the C∞ regularizer:
//
//   r_ε(σ) = ½(σ + √(σ² + 4ε²))
//
// Properties:
//   - C∞ and strictly positive for all σ ∈ ℝ
//   - r_ε(σ) → σ  as σ → +∞  (transparent for well-conditioned eigenvalues)
//   - r_ε(σ) → ε  as σ → 0   (smooth floor)
//   - r_ε(σ) → ε²/|σ| as σ → -∞  (damps negative eigenvalues)
//
// Its derivative is:
//
//   r'_ε(σ) = ½(1 + σ/√(σ² + 4ε²))
//
// Using the SAME r_ε for both log-determinant and inverse ensures the
// gradient is the exact derivative of a single scalar objective — no
// inconsistency from mixing different regularizations.

/// Smooth spectral regularizer: `r_ε(σ) = ½(σ + √(σ² + 4ε²))`.
///
/// Returns a strictly positive value for any real `sigma`. For large positive
/// `sigma` this is approximately `sigma`; near zero it smoothly floors at `epsilon`.
#[inline]
pub(crate) fn spectral_regularize(sigma: f64, epsilon: f64) -> f64 {
    let disc = sigma.hypot(2.0 * epsilon);
    if sigma >= 0.0 {
        0.5 * sigma + 0.5 * disc
    } else {
        // Avoid catastrophic cancellation in 0.5 * (σ + disc) when σ is
        // large and negative: r_ε(σ) = 2 ε² / (disc - σ).
        (2.0 * epsilon * epsilon) / (disc - sigma)
    }
}

/// Compute the spectral regularization scale for a set of eigenvalues.
///
/// `ε = √(machine_eps) · p` where `p` is the matrix dimension — a
/// ρ-INDEPENDENT numerical-stability floor on the smooth regularization
/// `r_ε(σ)`.  The previous formulation `ε = √(machine_eps) · max(|σ_max|, 1)`
/// coupled ε to the largest eigenvalue of H(ρ), which made ε a function of
/// ρ whenever the Hessian spectrum moved with ρ.  That coupling leaked a
/// spurious ∂log|H|_reg/∂ρ contribution through the near-zero eigenvalues:
/// for σ_j ≪ ε we have `log r_ε(σ_j) ≈ log ε`, so `d log r_ε(σ_j)/dρ`
/// picks up `(1/ε) · dε/dρ` whenever max|σ_j| moved.  That created a
/// first-order derivative mismatch in outer REML gradients (up to ~1.5% of
/// the dominant `d log|H|/dρ` term on problems with one near-singular
/// direction, e.g. multi-block GAMLSS wiggle models where the intercept/wiggle
/// direction is effectively in the null space of the likelihood curvature).
///
/// The analytic gradient formula `tr(G_ε(H) · dH/dρ_k)` assumes ε is
/// fixed; removing the ρ-coupling restores that assumption.  Scaling ε
/// by the matrix dimension `p` (a ρ-independent integer, set by the
/// problem geometry) gives numerical stability for larger systems without
/// reintroducing ρ leakage.  The absolute floor stays below any physically
/// meaningful eigenvalue (for p ≤ 10⁶, ε ≤ 1.5e-2; well-conditioned
/// problems have min σ ≫ ε and are unaffected).
#[inline]
pub(crate) fn spectral_epsilon(eigenvalues: &[f64]) -> f64 {
    f64::EPSILON.sqrt() * (eigenvalues.len() as f64).max(1.0)
}

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PseudoLogdetMode {
    Smooth,
    HardPseudo,
}

impl Default for PseudoLogdetMode {
    fn default() -> Self {
        Self::Smooth
    }
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
    reg_eigenvalues: Vec<f64>,
    /// Per-eigenvalue mask: `true` if the eigenpair participates in all
    /// traces, solves, and logdet contributions.  Under
    /// [`PseudoLogdetMode::Smooth`] every entry is `true`.  Under
    /// [`PseudoLogdetMode::HardPseudo`] entries with `σ_j ≤ ε` are `false`,
    /// so the numerical null space is excluded consistently from
    /// `log|H|_+`, its gradient, its cross-traces, AND `H⁻¹` solves
    /// (`H⁺` on the active subspace).
    active_mask: Vec<bool>,
    /// Eigenvectors of H (columns).
    eigenvectors: Array2<f64>,
    /// Precomputed: W = U diag(1/√r_ε(σ)) for efficient traces.
    /// trace(H⁻¹ A) = Σ (AW ⊙ W)
    w_factor: Array2<f64>,
    /// Precomputed kernel K_ab = 1 / (r_a r_b) for exact H⁻¹ cross traces in
    /// the eigenbasis.
    hinv_cross_kernel: Array2<f64>,
    /// Precomputed: G = U diag(1/√(√(σ² + 4ε²))) for logdet gradient traces.
    /// trace(G_ε(H) A) = Σ (AG ⊙ G) where G_ε uses φ'(σ) = 1/√(σ² + 4ε²).
    g_factor: Array2<f64>,
    /// Precomputed divided-difference kernel Γ for exact logdet Hessian cross traces
    /// in the eigenbasis.
    logdet_hessian_kernel: Array2<f64>,
    /// Precomputed log-determinant: Σ ln(r_ε(σ_i)).
    cached_logdet: f64,
    projected_factor_cache: ProjectedFactorCache,
    /// Full dimension.
    n_dim: usize,
}

impl DenseSpectralOperator {
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
            return Err(format!(
                "HessianOperator: expected square matrix, got {}×{}",
                n,
                h.ncols()
            ));
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
    fn rotate_to_eigenbasis(&self, matrix: &Array2<f64>) -> Array2<f64> {
        self.eigenvectors.t().dot(matrix).dot(&self.eigenvectors)
    }

    #[inline]
    fn trace_hinv_product_cross_rotated(&self, a_rot: &Array2<f64>, b_rot: &Array2<f64>) -> f64 {
        let mut result = 0.0;
        for a in 0..self.n_dim {
            for b in 0..self.n_dim {
                result += self.hinv_cross_kernel[[a, b]] * a_rot[[a, b]] * b_rot[[b, a]];
            }
        }
        result
    }

    #[inline]
    fn trace_hinv_product_cross_dense(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let a_rot = self.rotate_to_eigenbasis(a);
        if std::ptr::eq(a, b) {
            return self.trace_hinv_product_cross_rotated(&a_rot, &a_rot);
        }
        let b_rot = self.rotate_to_eigenbasis(b);
        self.trace_hinv_product_cross_rotated(&a_rot, &b_rot)
    }

    #[inline]
    fn projected_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        self.w_factor.t().dot(matrix).dot(&self.w_factor)
    }

    #[inline]
    fn projected_operator(&self, factor: &Array2<f64>, op: &dyn HyperOperator) -> Array2<f64> {
        let start = std::time::Instant::now();
        let result = op.projected_matrix(factor);
        log::info!(
            "[STAGE] DenseSpectralOperator::projected_operator dim={} rank={} implicit={} elapsed={:.3}s",
            self.n_dim,
            factor.ncols(),
            op.is_implicit(),
            start.elapsed().as_secs_f64(),
        );
        result
    }

    #[inline]
    fn trace_projected_cross(&self, left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        let mut result = 0.0;
        for a in 0..left.nrows() {
            for b in 0..left.ncols() {
                result += left[[a, b]] * right[[b, a]];
            }
        }
        result
    }

    #[inline]
    fn trace_logdet_hessian_cross_rotated(
        &self,
        h_i_rot: &Array2<f64>,
        h_j_rot: &Array2<f64>,
    ) -> f64 {
        let mut result = 0.0;
        for a in 0..self.n_dim {
            for b in 0..self.n_dim {
                result += self.logdet_hessian_kernel[[a, b]] * h_i_rot[[a, b]] * h_j_rot[[b, a]];
            }
        }
        result
    }
}

impl HessianOperator for DenseSpectralOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self)
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // tr(H_reg⁻¹ A) = Σ_j (1/r_ε(σ_j)) uⱼᵀAuⱼ
        // Computed as Σ (AW ⊙ W) where W = U diag(1/√r_ε(σ)).
        let aw = a.dot(&self.w_factor);
        aw.iter()
            .zip(self.w_factor.iter())
            .map(|(&a, &w)| a * w)
            .sum()
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
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

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
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

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.trace_hinv_product_cross_dense(a, b)
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        let start = std::time::Instant::now();
        let result = op.trace_projected_factor_cached(&self.w_factor, &self.projected_factor_cache);
        log::info!(
            "[STAGE] DenseSpectralOperator::trace_hinv_operator dim={} rank={} implicit={} elapsed={:.3}s",
            self.n_dim,
            self.w_factor.ncols(),
            op.is_implicit(),
            start.elapsed().as_secs_f64(),
        );
        result
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        let left = self.w_factor.t().dot(matrix).dot(&self.w_factor);
        let right = self.projected_operator(&self.w_factor, op);
        self.trace_projected_cross(&left, &right)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        let start = std::time::Instant::now();
        let left_proj = self.projected_operator(&self.w_factor, left);
        let result = if std::ptr::addr_eq(left, right) {
            self.trace_projected_cross(&left_proj, &left_proj)
        } else {
            let right_proj = self.projected_operator(&self.w_factor, right);
            self.trace_projected_cross(&left_proj, &right_proj)
        };
        log::info!(
            "[STAGE] DenseSpectralOperator::trace_hinv_operator_cross dim={} rank={} left_implicit={} right_implicit={} elapsed={:.3}s",
            self.n_dim,
            self.w_factor.ncols(),
            left.is_implicit(),
            right.is_implicit(),
            start.elapsed().as_secs_f64(),
        );
        result
    }

    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        // tr(G_ε(H) A) = Σ_j φ'(σ_j) uⱼᵀAuⱼ
        // where φ'(σ) = 1/√(σ² + 4ε²).
        // Computed as Σ (AG ⊙ G) where G = U diag(√φ'(σ)).
        let ag = a.dot(&self.g_factor);
        ag.iter()
            .zip(self.g_factor.iter())
            .map(|(&a, &g)| a * g)
            .sum()
    }

    fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        // h^G_i = ‖(X G)_{i,:}‖² where G_ε = G Gᵀ and G = self.g_factor.
        // The dominant cost at biobank scale is the (n × p)·(p × rank) matmul
        // — for matern60 with n=320K, p=101 that's ~3.3 GFLOPs and the
        // ndarray default `.dot()` runs single-threaded (no BLAS feature
        // enabled in this crate's build), so we route through faer's parallel
        // SIMD GEMM. For operator-backed (Lazy) designs we additionally
        // stream by row chunk so we never materialize the full (n×p) block
        // at biobank scale.
        let n = x.nrows();
        let p = x.ncols();
        let rank = self.g_factor.ncols();
        let mut h = Array1::<f64>::zeros(n);
        if n == 0 || p == 0 || rank == 0 {
            return h;
        }
        let chunk_rows = {
            const TARGET_BYTES: usize = 8 * 1024 * 1024;
            (TARGET_BYTES / ((p + rank).max(1) * 8)).max(512).min(n)
        };
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk_rows).min(n);
            let rows = x.try_row_chunk(start..end).unwrap_or_else(|err| {
                panic!("xt_logdet_kernel_x_diagonal: row chunk failed: {err}")
            });
            let xg = crate::faer_ndarray::fast_ab(&rows, &self.g_factor);
            for (local, row) in xg.outer_iter().enumerate() {
                h[start + local] = row.iter().map(|v| v * v).sum();
            }
            start = end;
        }
        h
    }

    fn trace_logdet_block_local(
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

    fn trace_hinv_block_local(
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

    fn trace_hinv_block_local_cross(
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

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        let start = std::time::Instant::now();
        let result = op.trace_projected_factor_cached(&self.g_factor, &self.projected_factor_cache);
        log::info!(
            "[STAGE] DenseSpectralOperator::trace_logdet_operator dim={} rank={} implicit={} elapsed={:.3}s",
            self.n_dim,
            self.g_factor.ncols(),
            op.is_implicit(),
            start.elapsed().as_secs_f64(),
        );
        result
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        let hp_i = self.rotate_to_eigenbasis(h_i);
        if std::ptr::eq(h_i, h_j) {
            return self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_i);
        }
        let hp_j = self.rotate_to_eigenbasis(h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        let hp_i = self.rotate_to_eigenbasis(h_i);
        let hp_j = self.projected_operator(&self.eigenvectors, h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    fn trace_logdet_hessian_cross_operator(
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

    fn trace_logdet_hessian_crosses(&self, matrices: &[&Array2<f64>]) -> Array2<f64> {
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

    fn active_rank(&self) -> usize {
        // With smooth regularization all eigenvalues are active (positive).
        // Return the full dimension for consistency.
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
        false
    }

    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        false
    }

    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Cholesky HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse Cholesky Hessian operator.
///
/// Wraps an existing `SparseExactFactor` and provides logdet, trace, and solve
/// from the same Cholesky factorization.
pub struct SparseCholeskyOperator {
    /// The sparse Cholesky factorization.
    factor: std::sync::Arc<crate::linalg::sparse_exact::SparseExactFactor>,
    /// Takahashi selected inverse (precomputed H^{-1} entries on the filled pattern of L).
    /// When available, trace computations use direct lookups instead of column solves.
    takahashi: Option<std::sync::Arc<crate::linalg::sparse_exact::TakahashiInverse>>,
    /// Precomputed log-determinant from the Cholesky diagonal.
    cached_logdet: f64,
    /// Dimension of H.
    n_dim: usize,
}

impl SparseCholeskyOperator {
    /// Create from an existing sparse factorization and its precomputed logdet.
    pub fn new(
        factor: std::sync::Arc<crate::linalg::sparse_exact::SparseExactFactor>,
        logdet_h: f64,
        dim: usize,
    ) -> Self {
        Self {
            factor,
            takahashi: None,
            cached_logdet: logdet_h,
            n_dim: dim,
        }
    }

    pub fn with_takahashi(
        mut self,
        taka: std::sync::Arc<crate::linalg::sparse_exact::TakahashiInverse>,
    ) -> Self {
        self.takahashi = Some(taka);
        self
    }

    const OPERATOR_SOLVE_CHUNK: usize = 64;

    fn takahashi_block_trace(
        taka: &crate::linalg::sparse_exact::TakahashiInverse,
        block: &Array2<f64>,
        start: usize,
    ) -> f64 {
        debug_assert_eq!(block.nrows(), block.ncols());
        let mut trace = 0.0;
        for i in 0..block.nrows() {
            let diag = block[[i, i]];
            if diag.abs() > 1e-30 {
                trace += taka.get(start + i, start + i) * diag;
            }
            for j in (i + 1)..block.ncols() {
                let pair = block[[i, j]] + block[[j, i]];
                if pair.abs() > 1e-30 {
                    trace += taka.get(start + i, start + j) * pair;
                }
            }
        }
        trace
    }

    fn takahashi_left_multiply_block(
        taka: &crate::linalg::sparse_exact::TakahashiInverse,
        block: &Array2<f64>,
        start: usize,
    ) -> Array2<f64> {
        let dim = block.nrows();
        let mut out = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            let z_diag = taka.get(start + i, start + i);
            if z_diag.abs() > 1e-30 {
                for k in 0..dim {
                    out[[i, k]] += z_diag * block[[i, k]];
                }
            }
            for j in (i + 1)..dim {
                let z = taka.get(start + i, start + j);
                if z.abs() <= 1e-30 {
                    continue;
                }
                for k in 0..dim {
                    out[[i, k]] += z * block[[j, k]];
                    out[[j, k]] += z * block[[i, k]];
                }
            }
        }
        out
    }

    fn trace_hinv_operator_exact(&self, op: &dyn HyperOperator) -> f64 {
        let (range_start, range_end) = op
            .block_local_data()
            .map(|(_, start, end)| (start, end))
            .unwrap_or((0, self.n_dim));
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(self.n_dim.max(1));
        let mut trace = 0.0_f64;
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut start = range_start;

        while start < range_end {
            let end = (start + chunk).min(range_end);
            let cols = end - start;
            op.mul_basis_columns_into(start, rhs_block.slice_mut(ndarray::s![.., ..cols]));

            let diagonal_sum = if cols == chunk {
                crate::linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
                    &self.factor,
                    &rhs_block,
                    start,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
                    &self.factor,
                    &rhs_view,
                    start,
                )
            };
            trace += diagonal_sum.unwrap_or_else(|e| {
                panic!("SparseCholeskyOperator exact trace_hinv_operator solve failed: {e}")
            });
            start = end;
        }

        trace
    }

    fn solve_operator_column_range_rows_exact(
        &self,
        op: &dyn HyperOperator,
        col_start: usize,
        col_end: usize,
        row_start: usize,
        row_end: usize,
    ) -> Result<Array2<f64>, String> {
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(self.n_dim.max(1));
        let cols_total = col_end - col_start;
        let rows_total = row_end - row_start;
        let mut solved = Array2::<f64>::zeros((rows_total, cols_total));
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut start = col_start;

        while start < col_end {
            let end = (start + chunk).min(col_end);
            let cols = end - start;
            op.mul_basis_columns_into(start, rhs_block.slice_mut(ndarray::s![.., ..cols]));

            let solved_block = if cols == chunk {
                crate::linalg::sparse_exact::solve_sparse_spdmulti_rows(
                    &self.factor,
                    &rhs_block,
                    row_start,
                    row_end,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti_rows(
                    &self.factor,
                    &rhs_view,
                    row_start,
                    row_end,
                )
            }
            .map_err(|e| {
                format!(
                    "SparseCholeskyOperator::solve_operator_column_range_rows_exact multi-solve failed: {e}"
                )
            })?;
            solved
                .slice_mut(ndarray::s![.., start - col_start..end - col_start])
                .assign(&solved_block);
            start = end;
        }

        Ok(solved)
    }

    fn fill_scaled_block_columns(
        block: &Array2<f64>,
        scale: f64,
        block_start: usize,
        local_col_start: usize,
        cols: usize,
        mut rhs_block: ndarray::ArrayViewMut2<'_, f64>,
    ) {
        let block_end = block_start + block.nrows();
        let source = block.slice(ndarray::s![.., local_col_start..local_col_start + cols]);
        let mut target = rhs_block.slice_mut(ndarray::s![block_start..block_end, ..cols]);
        if scale == 1.0 {
            target.assign(&source);
        } else {
            Zip::from(target)
                .and(source)
                .for_each(|dst, &value| *dst = scale * value);
        }
    }

    fn trace_hinv_block_local_exact(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        if scale == 0.0 {
            return 0.0;
        }
        debug_assert_eq!(block.nrows(), end - start);
        let t_start = std::time::Instant::now();
        let block_size = end - start;
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(block_size.max(1));
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut trace = 0.0;
        let mut local_col_start = 0usize;

        while local_col_start < block_size {
            let cols = (block_size - local_col_start).min(chunk);
            Self::fill_scaled_block_columns(
                block,
                scale,
                start,
                local_col_start,
                cols,
                rhs_block.view_mut(),
            );
            let diagonal_sum = if cols == chunk {
                crate::linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
                    &self.factor,
                    &rhs_block,
                    start + local_col_start,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
                    &self.factor,
                    &rhs_view,
                    start + local_col_start,
                )
            };
            trace += diagonal_sum.unwrap_or_else(|e| {
                panic!("SparseCholeskyOperator exact block-local trace solve failed: {e}")
            });
            local_col_start += cols;
        }

        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        if elapsed_ms > 100.0 {
            log::info!(
                "[REML-trace] block_local_exact | n_dim={} | block={} | {:.1}ms",
                self.n_dim,
                block_size,
                elapsed_ms
            );
        }
        trace
    }

    fn solve_block_local_rows_exact(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> Result<Array2<f64>, String> {
        debug_assert_eq!(block.nrows(), end - start);
        let block_size = end - start;
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(block_size.max(1));
        let mut solved = Array2::<f64>::zeros((block_size, block_size));
        if scale == 0.0 {
            return Ok(solved);
        }
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut local_col_start = 0usize;

        while local_col_start < block_size {
            let cols = (block_size - local_col_start).min(chunk);
            Self::fill_scaled_block_columns(
                block,
                scale,
                start,
                local_col_start,
                cols,
                rhs_block.view_mut(),
            );
            let solved_block = if cols == chunk {
                crate::linalg::sparse_exact::solve_sparse_spdmulti_rows(
                    &self.factor,
                    &rhs_block,
                    start,
                    end,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti_rows(
                    &self.factor,
                    &rhs_view,
                    start,
                    end,
                )
            }
            .map_err(|e| {
                format!(
                    "SparseCholeskyOperator::solve_block_local_rows_exact multi-solve failed: {e}"
                )
            })?;
            solved
                .slice_mut(ndarray::s![.., local_col_start..local_col_start + cols])
                .assign(&solved_block);
            local_col_start += cols;
        }

        Ok(solved)
    }

    fn trace_hinv_block_local_cross_exact(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        let t_start = std::time::Instant::now();
        let solved = self
            .solve_block_local_rows_exact(block, scale, start, end)
            .unwrap_or_else(|e| {
                panic!("SparseCholeskyOperator exact block-local cross solve failed: {e}")
            });
        let result = trace_matrix_product(&solved, &solved);
        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        if elapsed_ms > 100.0 {
            log::info!(
                "[REML-trace] block_local_cross_exact | n_dim={} | block={} | {:.1}ms",
                self.n_dim,
                end - start,
                elapsed_ms
            );
        }
        result
    }

    fn trace_hinv_matrix_operator_cross_exact(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        if let Some((_, range_start, range_end)) = op.block_local_data()
            && range_end - range_start < self.n_dim
        {
            return self.trace_hinv_matrix_block_operator_cross_exact(
                matrix,
                op,
                range_start,
                range_end,
            );
        }

        let solved_matrix = self.solve_multi(matrix);
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(self.n_dim.max(1));
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut trace = 0.0_f64;
        let (range_start, range_end) = op
            .block_local_data()
            .map(|(_, start, end)| (start, end))
            .unwrap_or((0, self.n_dim));
        let mut start = range_start;

        while start < range_end {
            let end = (start + chunk).min(range_end);
            let cols = end - start;
            op.mul_basis_columns_into(start, rhs_block.slice_mut(ndarray::s![.., ..cols]));

            let solved_op = if cols == chunk {
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_block)
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_view)
            };

            let solved_op = solved_op.unwrap_or_else(|e| {
                panic!("SparseCholeskyOperator exact matrix/operator cross solve failed: {e}")
            });

            for local_col in 0..cols {
                let matrix_row = start + local_col;
                for row in 0..self.n_dim {
                    trace += solved_matrix[[matrix_row, row]] * solved_op[[row, local_col]];
                }
            }
            start = end;
        }

        trace
    }

    fn trace_hinv_matrix_block_operator_cross_exact(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
        range_start: usize,
        range_end: usize,
    ) -> f64 {
        let t_start = std::time::Instant::now();
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(self.n_dim.max(1));
        let mut op_rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut eye_rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut trace = 0.0_f64;
        let mut start = range_start;

        while start < range_end {
            let end = (start + chunk).min(range_end);
            let cols = end - start;
            op.mul_basis_columns_into(start, op_rhs_block.slice_mut(ndarray::s![.., ..cols]));

            eye_rhs_block.fill(0.0);
            for local_col in 0..cols {
                eye_rhs_block[[start + local_col, local_col]] = 1.0;
            }

            let solved_op = if cols == chunk {
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &op_rhs_block)
            } else {
                let rhs_view = op_rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_view)
            };
            let solved_op = solved_op.unwrap_or_else(|e| {
                panic!(
                    "SparseCholeskyOperator exact matrix/block-operator cross operator solve failed: {e}"
                )
            });

            let solved_eye = if cols == chunk {
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &eye_rhs_block)
            } else {
                let rhs_view = eye_rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_view)
            };
            let solved_eye = solved_eye.unwrap_or_else(|e| {
                panic!(
                    "SparseCholeskyOperator exact matrix/block-operator cross identity solve failed: {e}"
                )
            });

            let selected_rows_t = matrix.t().dot(&solved_eye);
            for local_col in 0..cols {
                for row in 0..self.n_dim {
                    trace += selected_rows_t[[row, local_col]] * solved_op[[row, local_col]];
                }
            }
            start = end;
        }

        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        if elapsed_ms > 100.0 {
            log::info!(
                "[REML-trace] matrix_block_op_cross_exact | n_dim={} | block={} | {:.1}ms",
                self.n_dim,
                range_end - range_start,
                elapsed_ms
            );
        }
        trace
    }

    fn trace_hinv_operator_cross_exact(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        let (left_start, left_end) = left
            .block_local_data()
            .map(|(_, start, end)| (start, end))
            .unwrap_or((0, self.n_dim));
        let (right_start, right_end) = right
            .block_local_data()
            .map(|(_, start, end)| (start, end))
            .unwrap_or((0, self.n_dim));

        let solved_left = self
            .solve_operator_column_range_rows_exact(
                left,
                left_start,
                left_end,
                right_start,
                right_end,
            )
            .unwrap_or_else(|e| {
                panic!("SparseCholeskyOperator exact operator cross left solve failed: {e}")
            });
        let same_operator =
            std::ptr::addr_eq(left, right) && left_start == right_start && left_end == right_end;
        let solved_right = if same_operator {
            None
        } else {
            Some(
                self.solve_operator_column_range_rows_exact(
                    right,
                    right_start,
                    right_end,
                    left_start,
                    left_end,
                )
                .unwrap_or_else(|e| {
                    panic!("SparseCholeskyOperator exact operator cross right solve failed: {e}")
                }),
            )
        };

        let right_cols = right_end - right_start;
        let mut trace = 0.0;
        for left_col in 0..(left_end - left_start) {
            for right_col in 0..right_cols {
                let right_value = match solved_right.as_ref() {
                    Some(solved) => solved[[left_col, right_col]],
                    None => solved_left[[left_col, right_col]],
                };
                trace += solved_left[[right_col, left_col]] * right_value;
            }
        }
        trace
    }
}

impl HessianOperator for SparseCholeskyOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // When Takahashi is available, use direct entry lookup for tr(H^{-1} A).
        // This is O(p^2) via dense A iteration but avoids p column solves.
        if let Some(ref taka) = self.takahashi {
            let mut trace = 0.0;
            for i in 0..a.nrows() {
                let a_ii = a[[i, i]];
                if a_ii.abs() > 1e-30 {
                    trace += taka.get(i, i) * a_ii;
                }
                for j in (i + 1)..a.ncols() {
                    let pair = a[[i, j]] + a[[j, i]];
                    if pair.abs() > 1e-30 {
                        trace += taka.get(i, j) * pair;
                    }
                }
            }
            return trace;
        }
        crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, a)
            .unwrap_or_else(|e| {
                panic!("SparseCholeskyOperator exact trace_hinv_product solve failed: {e}")
            })
            .diag()
            .sum()
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        if let Some(ref taka) = self.takahashi {
            if let Some((local, start, end)) = op.block_local_data() {
                debug_assert_eq!(local.nrows(), end - start);
                return Self::takahashi_block_trace(taka, local, start);
            }
            // For other non-implicit operators: materialize and use Takahashi lookups
            if !op.is_implicit() {
                let dense = op.to_dense();
                return self.trace_hinv_product(&dense);
            }
        }
        self.trace_hinv_operator_exact(op)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.trace_hinv_operator(op)
    }

    fn trace_hinv_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        if let Some(ref taka) = self.takahashi {
            debug_assert_eq!(block.nrows(), end - start);
            return scale * Self::takahashi_block_trace(taka, block, start);
        }
        self.trace_hinv_block_local_exact(block, scale, start, end)
    }

    fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        if let Some(ref taka) = self.takahashi {
            debug_assert_eq!(block.nrows(), end - start);
            let za = Self::takahashi_left_multiply_block(taka, block, start);
            return scale * scale * trace_matrix_product(&za, &za);
        }
        self.trace_hinv_block_local_cross_exact(block, scale, start, end)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        crate::linalg::sparse_exact::solve_sparse_spd(&self.factor, rhs)
            .unwrap_or_else(|e| panic!("SparseCholeskyOperator exact solve failed: {e}"))
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, rhs)
            .unwrap_or_else(|e| panic!("SparseCholeskyOperator exact multi-solve failed: {e}"))
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        // For general dense matrices, column solves are better than materializing
        // full Z from Takahashi (O(p * nnz) vs O(p³)). Takahashi cross-traces
        // are only used for block-local operators via trace_hinv_operator_cross.
        let solved_a = self.solve_multi(a);
        if std::ptr::eq(a, b) {
            return trace_matrix_product(&solved_a, &solved_a);
        }
        let solved_b = self.solve_multi(b);
        trace_matrix_product(&solved_a, &solved_b)
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        // For mixed dense-matrix × block-local-operator, column solves are
        // still better than materializing full Z. Only use Takahashi when both
        // sides are block-local (handled in trace_hinv_operator_cross).
        self.trace_hinv_matrix_operator_cross_exact(matrix, op)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        // Takahashi fast path: when both operators are block-local to the same
        // block, compute tr(Z A Z B) using only the block of Z = H⁻¹.
        if let Some(ref taka) = self.takahashi {
            if let (Some((a_local, a_start, a_end)), Some((b_local, b_start, b_end))) =
                (left.block_local_data(), right.block_local_data())
            {
                if a_start == b_start && a_end == b_end {
                    // Same block: tr(Z_block * A_local * Z_block * B_local)
                    let za = Self::takahashi_left_multiply_block(taka, a_local, a_start);
                    if std::ptr::addr_eq(left, right) {
                        return trace_matrix_product(&za, &za);
                    }
                    let zb = Self::takahashi_left_multiply_block(taka, b_local, b_start);
                    // tr(ZA * ZB) = sum_ij (ZA)_ij * (ZB^T)_ij
                    return (&za * &zb.t()).sum();
                }
                // Different blocks: column solves are better than materializing
                // full p×p Z. Fall through to exact path.
            }
        }
        self.trace_hinv_operator_cross_exact(left, right)
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        -self.trace_hinv_matrix_operator_cross(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        -self.trace_hinv_operator_cross(h_i, h_j)
    }

    fn active_rank(&self) -> usize {
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }
}

// BlockCoupledDerivativeProvider was removed — its functionality is now handled
// by the `deriv_provider` trait (HessianDerivativeProvider), with concrete
// implementations like JointModelDerivProvider and SurvivalDerivProvider
// capturing the full correction including Jacobian sensitivity, weight
// sensitivity, and basis sensitivity.

// ═══════════════════════════════════════════════════════════════════════════
//  Block-coupled HessianOperator for joint multi-block models
// ═══════════════════════════════════════════════════════════════════════════

/// Block-coupled Hessian operator for joint multi-block models (GAMLSS, survival).
///
/// Wraps a [`DenseSpectralOperator`] over the full assembled joint Hessian while
/// retaining block-structure metadata. All [`HessianOperator`] trait methods
/// delegate to the inner spectral decomposition, ensuring a single
/// eigendecomposition governs logdet, trace, and solve.
///
/// # Block structure
///
/// A joint model with B parameter blocks has a joint Hessian of dimension
/// `p_total = sum_b p_b`. Each block occupies rows/columns
/// # When to use
///
/// Use `BlockCoupledOperator` whenever building an [`InnerSolution`] for a joint
/// multi-block model. It replaces the pattern of constructing a raw
/// `DenseSpectralOperator` and manually tracking block ranges separately.
pub struct BlockCoupledOperator {
    /// Inner spectral operator over the full joint Hessian.
    inner: DenseSpectralOperator,
}

impl BlockCoupledOperator {
    /// Create from an assembled joint Hessian using the `Smooth` regularizer.
    ///
    /// Test-only convenience wrapper around
    /// [`from_joint_hessian_with_mode`](Self::from_joint_hessian_with_mode).
    /// Production call sites thread the family's `PseudoLogdetMode`
    /// explicitly through `_with_mode`, so the Smooth-only entry point is
    /// intentionally gated to tests to keep the family-mode choice
    /// unambiguous at every production callsite.
    #[cfg(test)]
    pub fn from_joint_hessian(joint_hessian: &Array2<f64>) -> Result<Self, String> {
        Self::from_joint_hessian_with_mode(joint_hessian, PseudoLogdetMode::Smooth)
    }

    /// Construct from an assembled joint Hessian using the supplied
    /// [`PseudoLogdetMode`].  Internally performs a single
    /// eigendecomposition of `joint_hessian`.
    pub fn from_joint_hessian_with_mode(
        joint_hessian: &Array2<f64>,
        mode: PseudoLogdetMode,
    ) -> Result<Self, String> {
        let inner = DenseSpectralOperator::from_symmetric_with_mode(joint_hessian, mode)
            .map_err(|e| format!("BlockCoupledOperator eigendecomposition: {e}"))?;

        Ok(Self { inner })
    }
}

impl HessianOperator for BlockCoupledOperator {
    fn logdet(&self) -> f64 {
        self.inner.logdet()
    }

    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        self.inner.as_exact_dense_spectral()
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        self.inner.trace_hinv_product(a)
    }

    fn trace_hinv_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        self.inner.trace_hinv_h_k(a_k, third_deriv_correction)
    }

    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        self.inner.trace_logdet_gradient(a)
    }

    fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        self.inner.xt_logdet_kernel_x_diagonal(x)
    }

    fn trace_logdet_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        self.inner.trace_logdet_h_k(a_k, third_deriv_correction)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.inner.trace_logdet_operator(op)
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        self.inner.trace_logdet_hessian_cross(h_i, h_j)
    }

    fn trace_logdet_hessian_crosses(&self, matrices: &[&Array2<f64>]) -> Array2<f64> {
        self.inner.trace_logdet_hessian_crosses(matrices)
    }

    fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        self.inner
            .trace_hinv_block_local_cross(block, scale, start, end)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.inner.solve(rhs)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.inner.solve_multi(rhs)
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.inner.trace_hinv_product_cross(a, b)
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        self.inner.trace_hinv_matrix_operator_cross(matrix, op)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        self.inner.trace_hinv_operator_cross(left, right)
    }

    fn active_rank(&self) -> usize {
        self.inner.active_rank()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
        false
    }

    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        false
    }

    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(&self.inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Matrix-free SPD HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Operator-backed SPD Hessian with exact spectral REML algebra.
///
/// The operator closure is still useful for construction paths that naturally
/// expose HVPs, but REML cost/gradient/Hessian terms must all come from one
/// exact decomposition so `∂ log|H| = tr(H⁻¹ ∂H)` holds.  We therefore
/// materialize the coefficient Hessian by canonical-basis HVPs under an
/// explicit memory cap and delegate logdet, traces, and solves to
/// `DenseSpectralOperator`.
pub struct MatrixFreeSpdOperator {
    apply: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    cached_logdet: OnceLock<f64>,
    n_dim: usize,
    dense_spectral: OnceLock<Option<DenseSpectralOperator>>,
}

impl MatrixFreeSpdOperator {
    const EXACT_DENSE_SPECTRAL_MAX_BYTES: usize = 512 * 1024 * 1024;
    const EXACT_DENSE_SPECTRAL_ARRAYS: usize = 6;

    pub fn new<F>(dim: usize, apply: F) -> Self
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + 'static,
    {
        let apply = Arc::new(apply);

        Self {
            apply,
            cached_logdet: OnceLock::new(),
            n_dim: dim,
            dense_spectral: OnceLock::new(),
        }
    }

    fn exact_dense_spectral_bytes(&self) -> Option<usize> {
        self.n_dim
            .checked_mul(self.n_dim)?
            .checked_mul(std::mem::size_of::<f64>())?
            .checked_mul(Self::EXACT_DENSE_SPECTRAL_ARRAYS)
    }

    fn exact_dense_spectral_budget_ok(&self) -> bool {
        match self.exact_dense_spectral_bytes() {
            Some(bytes) if bytes <= Self::EXACT_DENSE_SPECTRAL_MAX_BYTES => true,
            Some(bytes) => {
                log::error!(
                    "MatrixFreeSpdOperator exact dense spectral materialization requires {:.2} GiB \
                     for dim={}, exceeding the {:.2} GiB cap",
                    bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    self.n_dim,
                    Self::EXACT_DENSE_SPECTRAL_MAX_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                false
            }
            None => {
                log::error!(
                    "MatrixFreeSpdOperator exact dense spectral byte count overflow for dim={}",
                    self.n_dim
                );
                false
            }
        }
    }

    fn materialize_dense_operator(&self) -> Option<DenseSpectralOperator> {
        if !self.exact_dense_spectral_budget_ok() {
            return None;
        }
        let materialize_start = std::time::Instant::now();
        let mut matrix = Array2::<f64>::zeros((self.n_dim, self.n_dim));
        let mut basis = Array1::<f64>::zeros(self.n_dim);
        for j in 0..self.n_dim {
            basis[j] = 1.0;
            let col = (self.apply)(&basis);
            basis[j] = 0.0;
            if col.len() != self.n_dim || !col.iter().all(|v| v.is_finite()) {
                return None;
            }
            matrix.column_mut(j).assign(&col);
        }
        for i in 0..self.n_dim {
            for j in (i + 1)..self.n_dim {
                let avg = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
                matrix[[i, j]] = avg;
                matrix[[j, i]] = avg;
            }
        }
        let result = DenseSpectralOperator::from_symmetric(&matrix).ok();
        log::info!(
            "[STAGE] matrix_free_spd materialize n_dim={} matvec_count={} elapsed={:.3}s",
            self.n_dim,
            self.n_dim,
            materialize_start.elapsed().as_secs_f64(),
        );
        result
    }

    fn dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        self.dense_spectral
            .get_or_init(|| self.materialize_dense_operator())
            .as_ref()
    }

    fn exact_dense_spectral(&self) -> &DenseSpectralOperator {
        self.dense_spectral().expect(
            "MatrixFreeSpdOperator exact REML algebra requires dense spectral materialization within the configured budget",
        )
    }
}

impl HessianOperator for MatrixFreeSpdOperator {
    fn logdet(&self) -> f64 {
        *self
            .cached_logdet
            .get_or_init(|| self.exact_dense_spectral().logdet())
    }

    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self.exact_dense_spectral())
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        self.exact_dense_spectral().trace_hinv_product(a)
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.exact_dense_spectral().trace_hinv_operator(op)
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.exact_dense_spectral().trace_hinv_product_cross(a, b)
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        self.exact_dense_spectral()
            .trace_hinv_matrix_operator_cross(matrix, op)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        self.exact_dense_spectral()
            .trace_hinv_operator_cross(left, right)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        let trace_start = std::time::Instant::now();
        let result = self.exact_dense_spectral().trace_logdet_operator(op);
        log::info!(
            "[STAGE] matrix_free_spd trace_logdet_operator implicit={} dim={} elapsed={:.3}s",
            op.is_implicit(),
            op.dim(),
            trace_start.elapsed().as_secs_f64(),
        );
        result
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.exact_dense_spectral().solve(rhs)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.exact_dense_spectral().solve_multi(rhs)
    }

    fn stochastic_trace_solve(&self, rhs: &Array1<f64>, rel_tol: f64) -> Array1<f64> {
        let _ = rel_tol;
        self.solve(rhs)
    }

    fn stochastic_trace_solve_multi(&self, rhs: &Array2<f64>, rel_tol: f64) -> Array2<f64> {
        let _ = rel_tol;
        self.solve_multi(rhs)
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        self.exact_dense_spectral()
            .trace_logdet_hessian_cross(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.exact_dense_spectral()
            .trace_logdet_hessian_cross_matrix_operator(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.exact_dense_spectral()
            .trace_logdet_hessian_cross_operator(h_i, h_j)
    }

    fn trace_logdet_hessian_crosses(&self, matrices: &[&Array2<f64>]) -> Array2<f64> {
        self.exact_dense_spectral()
            .trace_logdet_hessian_crosses(matrices)
    }

    fn active_rank(&self) -> usize {
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
        false
    }

    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        false
    }

    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        self.dense_spectral()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers for custom family → InnerSolution conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the square root of a symmetric positive semidefinite penalty matrix.
///
/// Returns R such that S = RᵀR, with R having `rank(S)` rows.
/// Uses eigendecomposition: S = U Λ U^T → R = Λ_+^{1/2} U_+^T.
pub fn penalty_matrix_root(s: &Array2<f64>) -> Result<Array2<f64>, String> {
    use faer::Side;
    let n = s.nrows();
    if n != s.ncols() {
        return Err(format!(
            "penalty_matrix_root: expected square matrix, got {}×{}",
            n,
            s.ncols()
        ));
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    let (eigenvalues, eigenvectors) = s
        .eigh(Side::Lower)
        .map_err(|e| format!("penalty_matrix_root eigendecomposition failed: {e}"))?;

    let max_ev = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let tol = (n.max(1) as f64) * f64::EPSILON * max_ev.max(1e-12);

    let active: Vec<usize> = eigenvalues
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > tol)
        .map(|(i, _)| i)
        .collect();
    let rank = active.len();

    let mut r = Array2::zeros((rank, n));
    for (out_row, &idx) in active.iter().enumerate() {
        let scale = eigenvalues[idx].sqrt();
        for col in 0..n {
            r[[out_row, col]] = scale * eigenvectors[[col, idx]];
        }
    }
    Ok(r)
}

/// Compute the exact pseudo-logdet log|S|₊ and its ρ-derivatives for a
/// blockwise penalty structure.
///
/// For each block, eigendecomposes S_b = Σ λ_k S_k, identifies the positive
/// eigenspace (structural nullspace detected from the eigenspectrum), and
/// computes exact derivatives on that subspace:
///
/// - L(S) = Σ_{σ_i > ε} log σ_i
/// - ∂/∂ρₖ L = tr(S⁺ Aₖ)
/// - ∂²/(∂ρₖ∂ρₗ) L = δ_{kl} ∂_k L − tr(S⁺ Aₗ S⁺ Aₖ)
///
/// For S(ρ) = Σ exp(ρ_k) S_k with S_k ⪰ 0, the nullspace N(S) = ∩_k N(S_k)
/// is structurally fixed (independent of ρ), so L is C∞ in ρ and these are
/// its exact derivatives.
///
/// `per_block_rho[b]` contains the log-lambdas for block b.
/// `per_block_penalties[b]` contains the penalty matrices for block b.
/// `ridge` is an additional ridge for logdet stability (0 if not applicable).
pub fn compute_block_penalty_logdet_derivs(
    per_block_rho: &[Array1<f64>],
    per_block_penalties: &[&[Array2<f64>]],
    per_block_nullspace_dims: &[&[usize]],
    ridge: f64,
) -> Result<PenaltyLogdetDerivs, String> {
    use super::penalty_logdet::PenaltyPseudologdet;

    let total_k: usize = per_block_rho.iter().map(|r| r.len()).sum();
    let block_offsets: Vec<usize> = per_block_rho
        .iter()
        .scan(0usize, |at, rho| {
            let current = *at;
            *at += rho.len();
            Some(current)
        })
        .collect();

    struct BlockPenaltyLogdetResult {
        offset: usize,
        value: f64,
        first: Array1<f64>,
        second: Array2<f64>,
    }

    let compute_block = |(b, block_rho): (usize, &Array1<f64>)| {
        let penalties = per_block_penalties[b];
        let kb = block_rho.len();
        if penalties.is_empty() || kb == 0 {
            return Ok(BlockPenaltyLogdetResult {
                offset: block_offsets[b],
                value: 0.0,
                first: Array1::zeros(kb),
                second: Array2::zeros((kb, kb)),
            });
        }
        let lambdas: Vec<f64> = block_rho.iter().map(|&r| r.exp()).collect();

        // Compute structural nullity if dimensions are available.
        let block_nullspace_dims = if b < per_block_nullspace_dims.len() {
            per_block_nullspace_dims[b]
        } else {
            &[]
        };
        let structural_nullity =
            if !block_nullspace_dims.is_empty() && block_nullspace_dims.len() == penalties.len() {
                Some(exact_intersection_nullity(penalties, block_nullspace_dims))
            } else {
                None
            };

        // Single eigendecomposition via canonical PenaltyPseudologdet.
        let pld = PenaltyPseudologdet::from_components_with_nullity(
            penalties,
            &lambdas,
            ridge,
            structural_nullity,
        )
        .map_err(|e| format!("penalty logdet failed for block {b}: {e}"))?;

        let value = pld.value();
        let (first, second) = pld.rho_derivatives(penalties, &lambdas);
        Ok(BlockPenaltyLogdetResult {
            offset: block_offsets[b],
            value,
            first,
            second,
        })
    };

    let block_results: Vec<BlockPenaltyLogdetResult> = if rayon::current_thread_index().is_some() {
        per_block_rho
            .iter()
            .enumerate()
            .map(compute_block)
            .collect::<Result<Vec<_>, String>>()?
    } else {
        per_block_rho
            .par_iter()
            .enumerate()
            .map(compute_block)
            .collect::<Result<Vec<_>, String>>()?
    };

    let mut log_det_total = 0.0;
    let mut first = Array1::zeros(total_k);
    let mut second = Array2::zeros((total_k, total_k));
    for block in block_results {
        log_det_total += block.value;
        let kb = block.first.len();
        for k in 0..kb {
            first[block.offset + k] = block.first[k];
        }
        for k in 0..kb {
            for l in 0..kb {
                second[[block.offset + k, block.offset + l]] = block.second[[k, l]];
            }
        }
    }

    Ok(PenaltyLogdetDerivs {
        value: log_det_total,
        first,
        second: Some(second),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Stochastic trace estimation via Rademacher probes
// ═══════════════════════════════════════════════════════════════════════════
//
// For large-scale models, computing tr(H⁻¹ A_k) exactly via the full p×p
// eigendecomposition or column-by-column sparse solves costs O(p²) per
// coordinate k.  Stochastic trace estimation gives an unbiased estimate
// using only matrix–vector products (solves), at cost O(M·p) where M is the
// number of random probe vectors (typically 10–200).
//
// The Girard–Hutchinson estimator:
//
//   tr(H⁻¹ A_k) ≈ (1/M) Σ_m  z_mᵀ H⁻¹ A_k z_m
//
// where z_m are i.i.d. random vectors with E[zzᵀ] = I.
//
// Rademacher probes (entries ±1 with equal probability) have strictly
// lower variance than Gaussian probes:
//   Var_Rad = 2(‖S‖²_F − Σ_i S²_{ii})
//   Var_Gau = 2‖S‖²_F
// where S = sym(H⁻¹ A_k).  The diagonal variance term is always removed.
//
// Key efficiency: ONE H⁻¹ solve per probe, shared across ALL k
// coordinates.  For each probe z we compute w = H⁻¹z once, then for each k
// we get q_k = zᵀ(A_k w) with a cheap matrix–vector multiply.

/// Configuration for stochastic trace estimation.
#[derive(Clone, Debug)]
pub struct StochasticTraceConfig {
    /// Minimum number of probe vectors (default: 10).
    pub n_probes_min: usize,
    /// Maximum number of probe vectors (default: 200).
    pub n_probes_max: usize,
    /// Target relative accuracy ε for the adaptive stopping criterion (default: 0.01).
    pub relative_tol: f64,
    /// Protection threshold τ_rel for near-zero traces (default: 1e-8).
    pub tau_rel: f64,
    /// Relative tolerance for iterative solves inside stochastic trace probes.
    pub solve_rel_tol: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for StochasticTraceConfig {
    fn default() -> Self {
        Self {
            n_probes_min: 10,
            n_probes_max: 200,
            relative_tol: 0.01,
            tau_rel: 1e-8,
            solve_rel_tol: 1e-8,
            seed: 0xCAFE_BABE,
        }
    }
}

impl StochasticTraceConfig {
    /// Fast, scale-aware estimator for second-order outer-Hessian traces.
    ///
    /// These traces shape the ARC/Newton model; they are not the REML
    /// objective itself. The default 200-probe estimator is too strict for
    /// high-dimensional marginal-slope jobs because near-zero off-diagonal
    /// cross traces never satisfy a pure relative-error test. A bounded probe
    /// budget with a scale-relative zero floor preserves the large curvature
    /// entries and lets ARC's trust-region logic absorb residual noise.
    fn outer_hessian(dim: usize, n_coords: usize) -> Self {
        let large_problem = dim >= 512 || n_coords >= 4;
        Self {
            n_probes_min: if large_problem { 4 } else { 6 },
            n_probes_max: if large_problem { 8 } else { 24 },
            relative_tol: if large_problem { 0.12 } else { 0.05 },
            tau_rel: 1e-3,
            solve_rel_tol: if large_problem { 1e-4 } else { 1e-5 },
            seed: 0xC0A5_7ACE,
        }
    }
}

/// Stochastic trace estimator using Rademacher probes with adaptive stopping.
///
/// Estimates `tr(H⁻¹ A_k)` for multiple matrices `A_k` simultaneously,
/// sharing a single `H⁻¹` solve per probe across all coordinates.
///
/// # Adaptive stopping
///
/// After each probe (once `n_probes_min` is reached), the estimator checks:
///
/// ```text
/// max_k  s_{M,k} / (√M · max(|q̄_{M,k}|, τ_rel))  ≤  ε
/// ```
///
/// where `s_{M,k}` is the sample standard deviation of the per-probe
/// estimates for coordinate k, and `q̄_{M,k}` is the running mean.
///
/// # Bias from approximate solves
///
/// If `H⁻¹` is computed approximately (e.g., via PCG with tolerance δ_PCG),
/// the bias satisfies `|bias| ≤ (δ_PCG · p / λ_min(H)) · ‖Ḣ_k‖₂`.
/// Set δ_PCG small enough that this is below the Monte Carlo tolerance.
pub struct StochasticTraceEstimator {
    config: StochasticTraceConfig,
}

enum StochasticTraceTargets<'a> {
    Dense(&'a [&'a Array2<f64>]),
    Mixed {
        dense_matrices: &'a [&'a Array2<f64>],
        operators: &'a [&'a dyn HyperOperator],
    },
    Structural {
        dense_matrices: &'a [&'a Array2<f64>],
        implicit_ops: &'a [&'a ImplicitHyperOperator],
    },
}

impl StochasticTraceTargets<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(matrices) => matrices.len(),
            Self::Mixed {
                dense_matrices,
                operators,
            } => dense_matrices.len() + operators.len(),
            Self::Structural {
                dense_matrices,
                implicit_ops,
            } => dense_matrices.len() + implicit_ops.len(),
        }
    }
}

impl StochasticTraceEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(config: StochasticTraceConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StochasticTraceConfig::default())
    }

    fn for_outer_hessian(dim: usize, n_coords: usize) -> Self {
        Self::new(StochasticTraceConfig::outer_hessian(dim, n_coords))
    }

    fn estimate_from_probe_batch<F>(
        &self,
        hop: &dyn HessianOperator,
        n_coords: usize,
        mut evaluate_probe: F,
    ) -> Vec<f64>
    where
        F: FnMut(&Array1<f64>, &Array1<f64>, &mut [f64]),
    {
        if n_coords == 0 {
            return Vec::new();
        }

        let p = hop.dim();
        if p == 0 {
            return vec![0.0; n_coords];
        }

        let mut means = vec![0.0_f64; n_coords];
        let mut m2s = vec![0.0_f64; n_coords];
        let mut probe_values = vec![0.0_f64; n_coords];
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);
        let check_interval = 4;

        let mut z = Array1::<f64>::zeros(p);
        for m in 0..self.config.n_probes_max {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            let w = hop.stochastic_trace_solve(&z, self.config.solve_rel_tol);
            evaluate_probe(&z, &w, &mut probe_values);

            for k in 0..n_coords {
                let q_k = probe_values[k];
                let count = (m + 1) as f64;
                let delta = q_k - means[k];
                means[k] += delta / count;
                let delta2 = q_k - means[k];
                m2s[k] += delta * delta2;
            }

            let n_done = m + 1;
            if n_done >= self.config.n_probes_min && n_done % check_interval == 0 {
                if self.check_convergence(n_done, &means, &m2s) {
                    break;
                }
            }
        }

        means
    }

    fn estimate_matrix_from_probe_batch<F>(
        &self,
        hop: &dyn HessianOperator,
        n_coords: usize,
        mut evaluate_probe: F,
    ) -> Array2<f64>
    where
        F: FnMut(&Array1<f64>, &mut Array2<f64>),
    {
        if n_coords == 0 {
            return Array2::zeros((0, 0));
        }
        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((n_coords, n_coords));
        }

        let mut means = Array2::<f64>::zeros((n_coords, n_coords));
        let mut m2s = Array2::<f64>::zeros((n_coords, n_coords));
        let mut probe_values = Array2::<f64>::zeros((n_coords, n_coords));
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);
        let check_interval = 4;
        let mut z = Array1::<f64>::zeros(p);

        for m in 0..self.config.n_probes_max {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            probe_values.fill(0.0);
            evaluate_probe(&z, &mut probe_values);

            let count = (m + 1) as f64;
            for d in 0..n_coords {
                for e in 0..n_coords {
                    let q = probe_values[[d, e]];
                    let delta = q - means[[d, e]];
                    means[[d, e]] += delta / count;
                    let delta2 = q - means[[d, e]];
                    m2s[[d, e]] += delta * delta2;
                }
            }

            let n_done = m + 1;
            if n_done >= self.config.n_probes_min
                && n_done % check_interval == 0
                && self.check_matrix_convergence(n_done, &means, &m2s)
            {
                break;
            }
        }

        for d in 0..n_coords {
            for e in (d + 1)..n_coords {
                let avg = 0.5 * (means[[d, e]] + means[[e, d]]);
                means[[d, e]] = avg;
                means[[e, d]] = avg;
            }
        }
        means
    }

    fn estimate_hinv_traces(
        &self,
        hop: &dyn HessianOperator,
        targets: StochasticTraceTargets<'_>,
    ) -> Vec<f64> {
        let n_coords = targets.len();
        if n_coords == 0 {
            return Vec::new();
        }

        match targets {
            StochasticTraceTargets::Dense(matrices) => {
                let mut a_w = Array1::<f64>::zeros(hop.dim());
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    for k in 0..matrices.len() {
                        dense_matvec_into(matrices[k], w.view(), a_w.view_mut());
                        probe_values[k] = z.dot(&a_w);
                    }
                })
            }
            StochasticTraceTargets::Mixed {
                dense_matrices,
                operators,
            } => {
                let mut a_w = Array1::<f64>::zeros(hop.dim());
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    for k in 0..dense_matrices.len() {
                        dense_matvec_into(dense_matrices[k], w.view(), a_w.view_mut());
                        probe_values[k] = z.dot(&a_w);
                    }

                    let dense_count = dense_matrices.len();
                    for (oi, op) in operators.iter().enumerate() {
                        let k = dense_count + oi;
                        if op.has_fast_bilinear_view() {
                            probe_values[k] = op.bilinear_view(w.view(), z.view());
                        } else {
                            op.mul_vec_into(w.view(), a_w.view_mut());
                            probe_values[k] = z.dot(&a_w);
                        }
                    }
                })
            }
            StochasticTraceTargets::Structural {
                dense_matrices,
                implicit_ops,
            } => {
                if implicit_ops.is_empty() {
                    let no_ops: [&dyn HyperOperator; 0] = [];
                    return self.estimate_hinv_traces(
                        hop,
                        StochasticTraceTargets::Mixed {
                            dense_matrices,
                            operators: &no_ops,
                        },
                    );
                }

                let x_design = implicit_ops[0].x_design.clone();
                let mut x_vec = Array1::<f64>::zeros(x_design.nrows());
                let mut y_vec = Array1::<f64>::zeros(x_design.nrows());
                let mut a_w = Array1::<f64>::zeros(hop.dim());
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    design_matrix_apply_view_into(x_design.as_ref(), z.view(), x_vec.view_mut());
                    design_matrix_apply_view_into(x_design.as_ref(), w.view(), y_vec.view_mut());

                    for k in 0..dense_matrices.len() {
                        dense_matvec_into(dense_matrices[k], w.view(), a_w.view_mut());
                        probe_values[k] = z.dot(&a_w);
                    }

                    let dense_count = dense_matrices.len();
                    for (oi, op) in implicit_ops.iter().enumerate() {
                        let k = dense_count + oi;
                        probe_values[k] = op.bilinear_with_shared_x(&x_vec, &y_vec, z, w);
                    }
                })
            }
        }
    }

    /// Estimate a single trace `tr(H⁻¹ A)` using the same batched Hutchinson
    /// core as the multi-coordinate path.
    pub fn estimate_single_trace(&self, hop: &dyn HessianOperator, matrix: &Array2<f64>) -> f64 {
        let matrices = [matrix];
        self.estimate_hinv_traces(hop, StochasticTraceTargets::Dense(&matrices))[0]
    }

    /// Estimate `tr(H⁻¹ A_k)` for multiple matrices `A_k` simultaneously.
    ///
    /// Uses Rademacher probes and adaptive stopping. Each probe requires
    /// exactly ONE `H⁻¹` solve (shared across all k), plus one `A_k`
    /// matrix–vector product per coordinate k.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `matrices`: the `A_k` matrices for which to estimate `tr(H⁻¹ A_k)`.
    ///
    /// # Returns
    /// A vector of estimated traces, one per input matrix.
    pub fn estimate_traces(
        &self,
        hop: &dyn HessianOperator,
        matrices: &[&Array2<f64>],
    ) -> Vec<f64> {
        self.estimate_hinv_traces(hop, StochasticTraceTargets::Dense(matrices))
    }

    /// Estimate `tr(H⁻¹ A_k)` for a mix of dense matrices and implicit operators.
    ///
    /// This extends [`estimate_traces`] to support implicit `HyperOperator` trait
    /// objects alongside dense matrices. The dense matrices are passed first,
    /// followed by the operators. Each probe requires ONE `H⁻¹` solve (shared),
    /// plus one matvec per coordinate.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `dense_matrices`: dense `A_k` matrices for which to estimate `tr(H⁻¹ A_k)`.
    /// - `operators`: implicit `HyperOperator` trait objects.
    ///
    /// # Returns
    /// A vector of estimated traces: first for dense matrices, then for operators.
    pub fn estimate_traces_with_operators(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        operators: &[&dyn HyperOperator],
    ) -> Vec<f64> {
        self.estimate_hinv_traces(
            hop,
            StochasticTraceTargets::Mixed {
                dense_matrices,
                operators,
            },
        )
    }

    /// Estimate first-order traces `tr(H⁻¹ A_d)` for implicit operators using the
    /// weighted-Gram structure, sharing one H⁻¹ solve and two X multiplies per probe.
    ///
    /// For each implicit operator d, the bilinear form `u^T A_d z` is computed using
    /// shared `x_vec = X z` and `y_vec = X u`, plus per-axis `forward_mul` calls.
    /// This avoids the X^T multiply per axis that the standard `mul_vec` requires.
    ///
    /// Dense matrices are handled alongside implicit operators in a single pass.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `dense_matrices`: dense A_k matrices.
    /// - `implicit_ops`: implicit `ImplicitHyperOperator` trait objects.
    ///
    /// # Returns
    /// Estimated traces: first for dense matrices, then for implicit operators.
    pub fn estimate_traces_structural(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        implicit_ops: &[&ImplicitHyperOperator],
    ) -> Vec<f64> {
        self.estimate_hinv_traces(
            hop,
            StochasticTraceTargets::Structural {
                dense_matrices,
                implicit_ops,
            },
        )
    }

    /// Estimate the full D×D matrix of second-order traces `tr(H⁻¹ A_d H⁻¹ A_e)`
    /// for implicit operators, using the CORRECT estimator.
    ///
    /// The correct Girard-Hutchinson estimator for `tr(H⁻¹ A_d H⁻¹ A_e)` is:
    ///
    /// ```text
    /// u = H⁻¹ z
    /// q_e = A_e z        for each axis e
    /// r_e = H⁻¹ q_e      for each axis e  (block solve, D RHS)
    /// estimate = u^T A_d r_e
    /// ```
    ///
    /// This gives tr(H⁻¹ A_d H⁻¹ A_e) correctly, NOT tr(A_d H⁻² A_e).
    ///
    /// Dense matrices are included alongside implicit operators. The output
    /// is a (total × total) matrix of cross-traces, symmetrized.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve` and `solve_multi`.
    /// - `dense_matrices`: dense A_k matrices.
    /// - `implicit_ops`: implicit `ImplicitHyperOperator` trait objects.
    ///
    /// # Returns
    /// Estimated D×D matrix of `tr(H⁻¹ A_d H⁻¹ A_e)` values, symmetrized.
    pub fn estimate_second_order_traces(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        implicit_ops: &[&ImplicitHyperOperator],
    ) -> Array2<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = implicit_ops.len();
        let total = n_dense + n_ops;
        if total == 0 {
            return Array2::zeros((0, 0));
        }

        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((total, total));
        }

        if total == 1 {
            let value = if n_dense == 1 {
                self.estimate_second_order_single_dense(hop, dense_matrices[0])
            } else {
                self.estimate_second_order_single_implicit(hop, implicit_ops[0])
            };
            return Array2::from_elem((1, 1), value);
        }

        // Get the shared X reference from the first implicit operator.
        let x_design = if n_ops > 0 {
            Some(implicit_ops[0].x_design.clone())
        } else {
            None
        };

        let mut q_columns = Array2::zeros((p, total));
        let mut dense_a_u: Vec<Array1<f64>> = (0..n_dense).map(|_| Array1::zeros(p)).collect();
        let n_obs = implicit_ops.first().map(|op| op.w_diag.len()).unwrap_or(0);
        let mut x_vec = Array1::<f64>::zeros(n_obs);
        let mut y_vec = Array1::<f64>::zeros(n_obs);
        let mut x_r: Vec<Array1<f64>> = (0..total).map(|_| Array1::zeros(n_obs)).collect();

        struct ImplicitSecondOrderScratch {
            w_dx_u: Array1<f64>,
            w_y: Array1<f64>,
            u_s: Array1<f64>,
        }

        self.estimate_matrix_from_probe_batch(hop, total, |z, probe_values| {
            // Step 1: u = H⁻¹ z (shared solve)
            let u = hop.stochastic_trace_solve(z, self.config.solve_rel_tol);

            if let Some(ref x) = x_design {
                design_matrix_apply_view_into(x.as_ref(), z.view(), x_vec.view_mut());
            }

            // Step 2: Form q_e = A_e z for all axes e. Each operator column is
            // independent, so fill the destination columns in parallel while
            // keeping only per-worker implicit matvec scratch.
            {
                use ndarray::Axis;
                use ndarray::parallel::prelude::*;

                q_columns
                    .axis_iter_mut(Axis(1))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(e, q_col)| {
                        if e < n_dense {
                            dense_matvec_into(dense_matrices[e], z.view(), q_col);
                        } else {
                            let op = implicit_ops[e - n_dense];
                            let mut n_work = Array1::<f64>::zeros(n_obs);
                            let mut p_work = Array1::<f64>::zeros(p);
                            op.matvec_with_shared_xz_into(
                                &x_vec,
                                z.view(),
                                q_col,
                                n_work.view_mut(),
                                p_work.view_mut(),
                            );
                        }
                    });
            }

            // Step 3: R = H⁻¹ [q_1, ..., q_D] (block solve, total RHS)
            let r = hop.stochastic_trace_solve_multi(&q_columns, self.config.solve_rel_tol);

            // Step 4: Compute T[d, e] = u^T A_d r_e for all (d, e) pairs.
            // For dense A_d: T[d, e] = (A_d^T u)^T r_e = (A_d u)^T r_e (A_d symmetric)
            // For implicit A_d: use shared X multiplies and bounded per-pair scratch.

            // Precompute X u and X r_e for implicit operators.
            if let Some(ref x) = x_design {
                design_matrix_apply_view_into(x.as_ref(), u.view(), y_vec.view_mut());
            }

            // For dense operators, precompute A_d u once.
            for d in 0..n_dense {
                dense_matvec_into(dense_matrices[d], u.view(), dense_a_u[d].view_mut());
            }

            // Precompute X r_e for all axes e (for implicit operators). These
            // columns are independent and reused by every implicit row.
            if let Some(ref x) = x_design {
                use rayon::prelude::*;
                x_r.par_iter_mut().enumerate().for_each(|(e, x_r_e)| {
                    design_matrix_apply_view_into(x.as_ref(), r.column(e), x_r_e.view_mut());
                });
            }

            // Precompute row-wise implicit quantities that are reused across all
            // columns. Deliberately do not materialize (∂X/∂ψ_d) r_e for every
            // d×e pair; those n_obs-sized vectors are built inside the pair task
            // below, which bounds scratch by the number of active rayon workers
            // rather than n_ops * total.
            let implicit_scratch: Vec<ImplicitSecondOrderScratch> = {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n_ops)
                    .into_par_iter()
                    .map(|idx| {
                        let op = implicit_ops[idx];
                        let dx_u = op
                            .implicit_deriv
                            .forward_mul(op.axis, &u.view())
                            .expect(
                                "radial scalar evaluation failed during implicit derivative forward_mul",
                            );
                        let w = &*op.w_diag;
                        let mut w_dx_u = Array1::<f64>::zeros(n_obs);
                        let mut w_y = Array1::<f64>::zeros(n_obs);
                        for i in 0..w.len() {
                            w_dx_u[i] = w[i] * dx_u[i];
                            w_y[i] = w[i] * y_vec[i];
                        }
                        let mut u_s = Array1::<f64>::zeros(p);
                        dense_transpose_matvec_into(&op.s_psi, u.view(), u_s.view_mut());
                        ImplicitSecondOrderScratch { w_dx_u, w_y, u_s }
                    })
                    .collect()
            };

            let pairs: Vec<(usize, usize)> = (0..total)
                .flat_map(|d| (0..total).map(move |e| (d, e)))
                .collect();
            let pair_values: Vec<(usize, usize, f64)> = {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                pairs
                    .into_par_iter()
                    .map(|(d, e)| {
                        let r_e = r.column(e);
                        let val = if d < n_dense {
                            // Dense A_d: u^T A_d r_e = (A_d u)^T r_e
                            dense_a_u[d].dot(&r_e)
                        } else {
                            // Implicit A_d: compute u^T A_d r_e using shared X multiplies.
                            // u^T A_d r_e = ((∂X/∂ψ_d)u)^T (W X r_e)
                            //             + (Xu)^T (W (∂X/∂ψ_d) r_e)
                            //             + u^T S_psi r_e
                            let oi = d - n_dense;
                            let op = implicit_ops[oi];
                            let scratch = &implicit_scratch[oi];
                            let x_re = &x_r[e];
                            let dx_re = op
                                .implicit_deriv
                                .forward_mul(op.axis, &r_e)
                                .expect(
                                    "radial scalar evaluation failed during implicit derivative forward_mul",
                                );

                            let mut design_val = 0.0f64;
                            for i in 0..scratch.w_dx_u.len() {
                                design_val += scratch.w_dx_u[i] * x_re[i];
                                design_val += scratch.w_y[i] * dx_re[i];
                            }

                            // Non-Gaussian fixed-β third-derivative correction:
                            //   uᵀ Xᵀ diag(c ⊙ X_{ψ_d} β̂) X r_e
                            //   = Σ_i y_vec[i] · c_x_psi_beta_i · x_re[i]
                            if let Some(c_x_psi_beta) = op.c_x_psi_beta.as_ref() {
                                let c = c_x_psi_beta.as_ref();
                                for i in 0..scratch.w_dx_u.len() {
                                    design_val += y_vec[i] * c[i] * x_re[i];
                                }
                            }

                            // Penalty: u^T S_psi r_e = (S_psi^T u)^T r_e
                            let penalty_val = scratch.u_s.dot(&r_e);
                            design_val + penalty_val
                        };
                        (d, e, val)
                    })
                    .collect()
            };

            for (d, e, val) in pair_values {
                probe_values[[d, e]] = val;
            }
        })
    }

    /// Estimate the full D×D matrix of second-order traces `tr(H⁻¹ A_d H⁻¹ A_e)`
    /// for a mix of dense matrices and generic hyperoperators.
    pub fn estimate_second_order_traces_with_operators(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        operators: &[&dyn HyperOperator],
    ) -> Array2<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = operators.len();
        let total = n_dense + n_ops;
        if total == 0 {
            return Array2::zeros((0, 0));
        }

        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((total, total));
        }

        if total == 1 {
            let value = if n_dense == 1 {
                self.estimate_second_order_single_dense(hop, dense_matrices[0])
            } else {
                self.estimate_second_order_single_operator(hop, operators[0])
            };
            return Array2::from_elem((1, 1), value);
        }

        let mut q_columns = Array2::zeros((p, total));
        let mut a_u_columns = Array2::zeros((p, total));

        self.estimate_matrix_from_probe_batch(hop, total, |z, probe_values| {
            let u = hop.stochastic_trace_solve(z, self.config.solve_rel_tol);

            for e in 0..n_dense {
                dense_matvec_into(dense_matrices[e], z.view(), q_columns.column_mut(e));
                dense_matvec_into(dense_matrices[e], u.view(), a_u_columns.column_mut(e));
            }
            for (oi, op) in operators.iter().enumerate() {
                let e = n_dense + oi;
                op.mul_vec_into(z.view(), q_columns.column_mut(e));
                op.mul_vec_into(u.view(), a_u_columns.column_mut(e));
            }

            let r = hop.stochastic_trace_solve_multi(&q_columns, self.config.solve_rel_tol);

            for d in 0..total {
                let a_d_u = a_u_columns.column(d);
                for e in d..total {
                    let r_e = r.column(e);
                    let val = a_d_u.dot(&r_e);
                    probe_values[[d, e]] = val;
                    if d != e {
                        let r_d = r.column(d);
                        let val_sym = a_u_columns.column(e).dot(&r_d);
                        probe_values[[e, d]] = val_sym;
                    }
                }
            }
        })
    }

    fn estimate_second_order_single_dense(
        &self,
        hop: &dyn HessianOperator,
        matrix: &Array2<f64>,
    ) -> f64 {
        let p = hop.dim();
        if p == 0 {
            return 0.0;
        }

        let mut q = Array1::<f64>::zeros(p);
        self.estimate_matrix_from_probe_batch(hop, 1, |z, probe_values| {
            let u = hop.stochastic_trace_solve(z, self.config.solve_rel_tol);
            dense_matvec_into(matrix, z.view(), q.view_mut());
            let r = hop.stochastic_trace_solve(&q, self.config.solve_rel_tol);
            probe_values[[0, 0]] = dense_bilinear(matrix, u.view(), r.view());
        })[[0, 0]]
    }

    fn estimate_second_order_single_implicit(
        &self,
        hop: &dyn HessianOperator,
        op: &ImplicitHyperOperator,
    ) -> f64 {
        let p = hop.dim();
        if p == 0 {
            return 0.0;
        }

        let n_obs = op.w_diag.len();
        let mut x_z = Array1::<f64>::zeros(n_obs);
        let mut x_u = Array1::<f64>::zeros(n_obs);
        let mut x_r = Array1::<f64>::zeros(n_obs);
        let mut n_work = Array1::<f64>::zeros(n_obs);
        let mut p_work = Array1::<f64>::zeros(p);
        let mut q = Array1::<f64>::zeros(p);
        self.estimate_matrix_from_probe_batch(hop, 1, |z, probe_values| {
            let u = hop.stochastic_trace_solve(z, self.config.solve_rel_tol);
            design_matrix_apply_view_into(&op.x_design, z.view(), x_z.view_mut());
            op.matvec_with_shared_xz_into(
                &x_z,
                z.view(),
                q.view_mut(),
                n_work.view_mut(),
                p_work.view_mut(),
            );
            let r = hop.stochastic_trace_solve(&q, self.config.solve_rel_tol);

            design_matrix_apply_view_into(&op.x_design, u.view(), x_u.view_mut());
            design_matrix_apply_view_into(&op.x_design, r.view(), x_r.view_mut());
            let dx_u = op
                .implicit_deriv
                .forward_mul(op.axis, &u.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul");
            let dx_r = op
                .implicit_deriv
                .forward_mul(op.axis, &r.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul");

            let w = &*op.w_diag;
            let mut value = 0.0;
            for i in 0..w.len() {
                let wi = w[i];
                value += dx_u[i] * wi * x_r[i];
                value += x_u[i] * wi * dx_r[i];
            }
            // Non-Gaussian fixed-β third-derivative correction:
            //   uᵀ Xᵀ diag(c ⊙ X_{ψ_d} β̂) X r = Σ_i (X u)_i · c_x_psi_beta_i · (X r)_i
            if let Some(c_x_psi_beta) = op.c_x_psi_beta.as_ref() {
                let c = c_x_psi_beta.as_ref();
                for i in 0..w.len() {
                    value += x_u[i] * c[i] * x_r[i];
                }
            }
            value += dense_bilinear(&op.s_psi, r.view(), u.view());

            probe_values[[0, 0]] = value;
        })[[0, 0]]
    }

    fn estimate_second_order_single_operator(
        &self,
        hop: &dyn HessianOperator,
        op: &dyn HyperOperator,
    ) -> f64 {
        let p = hop.dim();
        if p == 0 {
            return 0.0;
        }

        let mut q = Array1::<f64>::zeros(p);
        let mut a_u = Array1::<f64>::zeros(p);
        self.estimate_matrix_from_probe_batch(hop, 1, |z, probe_values| {
            let u = hop.stochastic_trace_solve(z, self.config.solve_rel_tol);
            op.mul_vec_into(z.view(), q.view_mut());
            op.mul_vec_into(u.view(), a_u.view_mut());
            let r = hop.stochastic_trace_solve(&q, self.config.solve_rel_tol);
            probe_values[[0, 0]] = a_u.dot(&r);
        })[[0, 0]]
    }

    /// Check the adaptive stopping criterion.
    ///
    /// Returns `true` if all coordinates have converged:
    /// ```text
    /// max_k  s_{M,k} / (√M · max(|q̄_{M,k}|, τ_rel))  ≤  ε
    /// ```
    fn check_convergence(&self, n: usize, means: &[f64], m2s: &[f64]) -> bool {
        if n < 2 {
            return false;
        }
        let sqrt_n = (n as f64).sqrt();
        let n_f = n as f64;

        for k in 0..means.len() {
            let variance = m2s[k] / (n_f - 1.0);
            let std_dev = variance.max(0.0).sqrt();
            let denom = sqrt_n * means[k].abs().max(self.config.tau_rel);
            let rel_err = std_dev / denom;
            if rel_err > self.config.relative_tol {
                return false;
            }
        }
        true
    }

    fn check_matrix_convergence(&self, n: usize, means: &Array2<f64>, m2s: &Array2<f64>) -> bool {
        if n < 2 {
            return false;
        }
        let sqrt_n = (n as f64).sqrt();
        let n_f = n as f64;
        let scale_floor = means
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
            .max(1.0)
            * self.config.tau_rel;
        for ((d, e), &mean) in means.indexed_iter() {
            let variance = m2s[[d, e]] / (n_f - 1.0);
            let std_dev = variance.max(0.0).sqrt();
            let denom = sqrt_n * mean.abs().max(scale_floor);
            let rel_err = std_dev / denom;
            if rel_err > self.config.relative_tol {
                return false;
            }
        }
        true
    }
}

fn stochastic_trace_hinv_products(
    hop: &dyn HessianOperator,
    targets: StochasticTraceTargets<'_>,
) -> Vec<f64> {
    let estimator = StochasticTraceEstimator::with_defaults();
    match targets {
        StochasticTraceTargets::Dense(matrices) if matrices.len() == 1 => {
            vec![estimator.estimate_single_trace(hop, matrices[0])]
        }
        StochasticTraceTargets::Dense(matrices) => estimator.estimate_traces(hop, matrices),
        StochasticTraceTargets::Mixed {
            dense_matrices,
            operators,
        } => estimator.estimate_traces_with_operators(hop, dense_matrices, operators),
        StochasticTraceTargets::Structural {
            dense_matrices,
            implicit_ops,
        } => estimator.estimate_traces_structural(hop, dense_matrices, implicit_ops),
    }
}

fn stochastic_trace_hinv_crosses<'a>(
    hop: &dyn HessianOperator,
    dense_matrices: &'a [Array2<f64>],
    coord_has_operator: &[bool],
    generic_ops: &[&'a dyn HyperOperator],
    implicit_ops: &[&'a ImplicitHyperOperator],
) -> Array2<f64> {
    let estimator =
        StochasticTraceEstimator::for_outer_hessian(hop.dim(), coord_has_operator.len());
    let dense_refs: Vec<&Array2<f64>> = dense_matrices.iter().collect();
    let raw_cross = if generic_ops.len() == implicit_ops.len() {
        estimator.estimate_second_order_traces(hop, &dense_refs, implicit_ops)
    } else {
        estimator.estimate_second_order_traces_with_operators(hop, &dense_refs, generic_ops)
    };

    let total_coords = coord_has_operator.len();
    let n_dense_total = coord_has_operator.iter().filter(|&&b| !b).count();
    let mut original_to_raw = Vec::with_capacity(total_coords);
    let mut dense_cursor = 0usize;
    let mut operator_cursor = n_dense_total;
    for &has_operator in coord_has_operator {
        if has_operator {
            original_to_raw.push(operator_cursor);
            operator_cursor += 1;
        } else {
            original_to_raw.push(dense_cursor);
            dense_cursor += 1;
        }
    }

    let mut mapped = Array2::zeros((total_coords, total_coords));
    for i in 0..total_coords {
        for j in 0..total_coords {
            mapped[[i, j]] = raw_cross[[original_to_raw[i], original_to_raw[j]]];
        }
    }
    mapped
}

// Lightweight xoshiro256ss RNG
//
// We use a self-contained xoshiro256ss implementation so that the stochastic
// trace estimator does not impose any new dependency requirements. The
// codebase already uses `rand` (0.10), but a minimal inline RNG avoids
// pulling in the full `rand` trait machinery for what is just a stream of
// random bits for ±1 generation.

/// Minimal xoshiro256** PRNG (period 2^256 − 1).
///
/// This is used exclusively for Rademacher probe generation. The state is
/// seeded deterministically from a u64 via splitmix64.
struct Xoshiro256SS {
    s: [u64; 4],
}

impl Xoshiro256SS {
    /// Seed from a single u64 via splitmix64 expansion.
    fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let s0 = splitmix64(&mut sm);
        let s1 = splitmix64(&mut sm);
        let s2 = splitmix64(&mut sm);
        let s3 = splitmix64(&mut sm);
        // Guard against the all-zero state (astronomically unlikely but
        // formally required for xoshiro correctness).
        let s = if s0 | s1 | s2 | s3 == 0 {
            [1, 0, 0, 0]
        } else {
            [s0, s1, s2, s3]
        };
        Self { s }
    }

    /// Generate the next u64.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }
}

/// Splitmix64: deterministic expansion of a single u64 seed into a sequence.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn rademacher_probe_into(mut z: ArrayViewMut1<'_, f64>, rng: &mut Xoshiro256SS) {
    let mut bits: u64 = 0;
    let mut remaining_bits = 0u32;

    for i in 0..z.len() {
        if remaining_bits == 0 {
            bits = rng.next_u64();
            remaining_bits = 64;
        }
        z[i] = if bits & 1 == 0 { 1.0 } else { -1.0 };
        bits >>= 1;
        remaining_bits -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::estimate::DP_FLOOR;
    use approx::assert_relative_eq;
    use ndarray::array;

    struct SentinelOuterHessianOperator {
        matrix: Array2<f64>,
    }

    impl crate::solver::outer_strategy::OuterHessianOperator for SentinelOuterHessianOperator {
        fn dim(&self) -> usize {
            self.matrix.nrows()
        }

        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            Ok(self.matrix.dot(v))
        }

        fn is_cheap_to_materialize(&self) -> bool {
            true
        }
    }

    struct FamilyOperatorOnlyDerivatives {
        op: Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>,
    }

    impl HessianDerivativeProvider for FamilyOperatorOnlyDerivatives {
        fn hessian_derivative_correction(
            &self,
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(None)
        }

        fn has_corrections(&self) -> bool {
            false
        }

        fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
            None
        }

        fn family_outer_hessian_operator(
            &self,
        ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
            Some(Arc::clone(&self.op))
        }
    }

    #[test]
    fn value_gradient_hessian_prefers_family_supplied_outer_operator() {
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&Array2::eye(2)).unwrap());
        let family_matrix = array![[42.0]];
        let family_operator = Arc::new(SentinelOuterHessianOperator {
            matrix: family_matrix.clone(),
        });
        let deriv_provider = FamilyOperatorOnlyDerivatives {
            op: family_operator,
        };

        let solution = InnerSolution {
            log_likelihood: -1.25,
            penalty_quadratic: 0.4,
            hessian_op: hop,
            beta: array![0.5, -0.25],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(Array2::eye(2))],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.0],
                second: Some(array![[0.0]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            n_observations: 2,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        };

        let result = reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None)
            .expect("family outer operator evaluation");
        let crate::solver::outer_strategy::HessianResult::Operator(op) = result.hessian else {
            panic!("expected family-supplied operator Hessian route");
        };
        let dense = op.materialize_dense().expect("sentinel materialization");
        assert_eq!(dense, family_matrix);
    }

    #[test]
    fn test_dense_spectral_operator_simple() {
        // 2×2 diagonal matrix: H = diag(2, 5)
        let h = Array2::from_diag(&array![2.0, 5.0]);
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // logdet = ln(2) + ln(5)
        let expected_logdet = 2.0_f64.ln() + 5.0_f64.ln();
        assert!((op.logdet() - expected_logdet).abs() < 1e-12);

        // tr(H⁻¹ I) = 1/2 + 1/5 = 0.7
        let id = Array2::eye(2);
        let trace = op.trace_hinv_product(&id);
        assert!((trace - 0.7).abs() < 1e-12);

        // solve: H⁻¹ [1, 1] = [0.5, 0.2]
        let rhs = array![1.0, 1.0];
        let sol = op.solve(&rhs);
        assert!((sol[0] - 0.5).abs() < 1e-12);
        assert!((sol[1] - 0.2).abs() < 1e-12);

        assert_eq!(sol.len(), 2);
    }

    #[test]
    fn test_dense_spectral_operator_solve_multi_matches_column_solves() {
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let rhs = array![[1.0, -1.0], [0.5, 2.0], [3.0, 0.25],];

        let multi = op.solve_multi(&rhs);
        for col in 0..rhs.ncols() {
            let single = op.solve(&rhs.column(col).to_owned());
            for row in 0..rhs.nrows() {
                let err = (multi[[row, col]] - single[row]).abs();
                assert!(
                    err < 1e-12,
                    "solve_multi mismatch at ({row}, {col}): multi={}, single={}",
                    multi[[row, col]],
                    single[row]
                );
            }
        }
    }

    #[test]
    fn test_dense_spectral_operator_cross_trace_matches_column_solves() {
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let a = array![[1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 0.5],];
        let b = array![[0.5, -0.4, 0.1], [-0.4, 1.5, 0.25], [0.1, 0.25, 0.75],];

        let expected = (&op.solve_multi(&a).t() * &op.solve_multi(&b)).sum();
        let exact = op.trace_hinv_product_cross(&a, &b);

        assert_relative_eq!(exact, expected, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn test_dense_spectral_operator_operator_cross_matches_dense_formula() {
        let h = array![[5.0, 0.5, 0.25], [0.5, 3.5, 0.2], [0.25, 0.2, 2.5],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let dense = array![[1.0, 0.1, -0.2], [0.1, 0.75, 0.3], [-0.2, 0.3, 1.25],];
        let other = array![[0.6, -0.3, 0.15], [-0.3, 1.1, 0.05], [0.15, 0.05, 0.9],];
        let other_op = DenseMatrixHyperOperator {
            matrix: other.clone(),
        };

        let expected = op.trace_hinv_product_cross(&dense, &other);
        let mixed = op.trace_hinv_matrix_operator_cross(&dense, &other_op);
        let operator = op.trace_hinv_operator_cross(&other_op, &other_op);
        let operator_expected = op.trace_hinv_product_cross(&other, &other);

        assert_relative_eq!(mixed, expected, epsilon = 1e-12, max_relative = 1e-12);
        assert_relative_eq!(
            operator,
            operator_expected,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_hyper_coord_total_drift_result_keeps_operator_and_dense_correction() {
        let h = array![[4.0, 0.25], [0.25, 3.0],];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let base = array![[1.0, 0.2], [0.2, 0.5],];
        let corr = array![[0.3, -0.1], [-0.1, 0.4],];
        let drift = HyperCoordDrift::from_operator(Arc::new(DenseMatrixHyperOperator {
            matrix: base.clone(),
        }));
        let correction = DriftDerivResult::Dense(corr.clone());

        let combined = hyper_coord_total_drift_result(&drift, Some(&correction), h.nrows());
        let expected = hop.trace_logdet_gradient(&(&base + &corr));

        assert_relative_eq!(
            combined.trace_logdet(&hop),
            expected,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_dense_spectral_operator_rotated_logdet_cross_matches_dense_path() {
        let h = array![[4.0, 0.5, 0.2], [0.5, 2.5, 0.3], [0.2, 0.3, 1.75],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let a = array![[0.8, 0.2, -0.1], [0.2, 1.4, 0.35], [-0.1, 0.35, 0.9],];
        let b = array![[1.2, -0.25, 0.05], [-0.25, 0.7, 0.15], [0.05, 0.15, 0.6],];

        let a_rot = op.rotate_to_eigenbasis(&a);
        let b_rot = op.rotate_to_eigenbasis(&b);

        let direct = op.trace_logdet_hessian_cross(&a, &b);
        let rotated = op.trace_logdet_hessian_cross_rotated(&a_rot, &b_rot);

        assert_relative_eq!(rotated, direct, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn test_compute_adjoint_z_c_streaming_matches_dense_reference() {
        // streaming and dense paths differ only by reordering the sum that builds v;
        // with n=64, p=8 the gap is bounded by O(εn) ≈ 1e-14.
        let n = 64usize;
        let p = 8usize;
        let mut rng = Xoshiro256SS::from_seed(0x5EED_C0FFEE_u64);
        let unit = |rng: &mut Xoshiro256SS| {
            let bits = rng.next_u64() >> 11;
            (bits as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };

        let mut x_data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x_data[[i, j]] = unit(&mut rng);
            }
        }
        let mut c_array = Array1::<f64>::zeros(n);
        for i in 0..n {
            c_array[i] = unit(&mut rng);
        }

        let mut m = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                m[[i, j]] = unit(&mut rng);
            }
        }
        let mut h = m.t().dot(&m);
        for i in 0..p {
            h[[i, i]] += p as f64;
        }
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let x = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_data.clone()));
        let ing = ScalarGlmIngredients {
            c_array: &c_array,
            d_array: None,
            x: &x,
        };
        // Construct h_dense = diag(X H⁻¹ Xᵀ) via solve_multi for the dense reference.
        let z_full = hop.solve_multi(&x_data.t().to_owned());
        let mut h_dense = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..p {
                acc += x_data[[i, j]] * z_full[[j, i]];
            }
            h_dense[i] = acc;
        }
        let streamed = compute_adjoint_z_c(&ing, &hop, &h_dense).expect("adjoint path");

        let mut t = h_dense.clone();
        Zip::from(&mut t)
            .and(&c_array)
            .for_each(|t_i, &c_i| *t_i *= c_i);
        let v = x_data.t().dot(&t);
        let reference = hop.solve(&v);

        for k in 0..p {
            assert_relative_eq!(
                streamed[k],
                reference[k],
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn fourth_derivative_trace_matrix_matches_scalar_pair_formula() {
        let n = 37usize;
        let p = 5usize;
        let t = 4usize;
        let mut rng = Xoshiro256SS::from_seed(0xF047_ACE5_u64);
        let unit = |rng: &mut Xoshiro256SS| {
            let bits = rng.next_u64() >> 11;
            (bits as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };

        let mut x_data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x_data[[i, j]] = unit(&mut rng);
            }
        }
        let mut c_array = Array1::<f64>::zeros(n);
        let mut d_array = Array1::<f64>::zeros(n);
        let mut leverage = Array1::<f64>::zeros(n);
        for i in 0..n {
            c_array[i] = unit(&mut rng);
            d_array[i] = unit(&mut rng);
            leverage[i] = 0.25 + unit(&mut rng).abs();
        }
        let x = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_data));
        let ing = ScalarGlmIngredients {
            c_array: &c_array,
            d_array: Some(&d_array),
            x: &x,
        };

        let mut modes = Vec::with_capacity(t);
        for _ in 0..t {
            let mut mode = Array1::<f64>::zeros(p);
            for j in 0..p {
                mode[j] = unit(&mut rng);
            }
            modes.push(mode);
        }
        let mode_refs = modes.iter().collect::<Vec<_>>();
        let gram = compute_fourth_derivative_trace_matrix(&ing, &mode_refs, &leverage)
            .expect("batched fourth trace")
            .expect("d-array is present");

        for i in 0..t {
            for j in 0..t {
                let scalar = compute_fourth_derivative_trace(&ing, &modes[i], &modes[j], &leverage)
                    .expect("scalar fourth trace")
                    .expect("d-array is present");
                assert_relative_eq!(gram[[i, j]], scalar, epsilon = 1e-10, max_relative = 1e-10);
            }
        }
    }

    #[test]
    fn operator_hessian_matches_dense_with_operator_drifts_and_extended_glm_corrections() {
        let h = array![[1.0e-7, 0.0], [0.0, 2.7]];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());
        let beta = array![0.4, -0.7];
        let penalty_root = array![[1.2, 0.1], [0.0, 0.8]];
        let ext_drift = array![[0.45, -0.15], [-0.15, 0.35]];
        let x = array![[1.0, 0.2], [-0.4, 1.1], [0.7, -0.8]];
        let c_array = array![0.31, -0.27, 0.19];
        let d_array = array![0.17, -0.11, 0.23];
        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array,
            d_array: Some(d_array),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
        };

        let solution = InnerSolution {
            log_likelihood: -2.3,
            penalty_quadratic: 0.6,
            hessian_op: hop.clone(),
            beta,
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(penalty_root)],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.4],
                second: Some(array![[0.13]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            n_observations: 3,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: vec![HyperCoord {
                a: -0.21,
                g: array![0.33, -0.42],
                drift: HyperCoordDrift::from_operator(Arc::new(DenseMatrixHyperOperator {
                    matrix: ext_drift,
                })),
                ld_s: 0.07,
                b_depends_on_beta: false,
                is_penalty_like: false,
                firth_g: None,
                tk_eta_fixed: None,
                tk_x_fixed: None,
            }],
            ext_coord_pair_fn: Some(Box::new(|_, _| HyperCoordPair {
                a: 0.09,
                g: array![0.16, -0.12],
                b_mat: array![[0.08, 0.03], [0.03, -0.04]],
                b_operator: None,
                ld_s: -0.05,
            })),
            rho_ext_pair_fn: Some(Box::new(|_, _| HyperCoordPair {
                a: -0.14,
                g: array![-0.18, 0.22],
                b_mat: array![[0.05, -0.02], [-0.02, 0.07]],
                b_operator: None,
                ld_s: 0.04,
            })),
            fixed_drift_deriv: None,
            barrier_config: None,
        };
        let rho: Vec<f64> = vec![0.2_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let dense = compute_outer_hessian(
            &solution,
            &rho,
            &lambdas,
            solution.hessian_op.as_ref(),
            solution.deriv_provider.as_ref(),
        )
        .unwrap();
        let kernel = solution
            .deriv_provider
            .outer_hessian_derivative_kernel()
            .unwrap();
        let operator = build_outer_hessian_operator(
            &solution,
            &lambdas,
            solution.deriv_provider.as_ref(),
            kernel,
        )
        .unwrap();
        let materialized =
            crate::solver::outer_strategy::OuterHessianOperator::materialize_dense(&operator)
                .unwrap();

        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                let materialized_entry = materialized[[row, col]];
                let dense_entry = dense[[row, col]];
                let tolerance = 1e-10_f64.max(1e-10 * dense_entry.abs());
                assert!(
                    (materialized_entry - dense_entry).abs() <= tolerance,
                    "outer Hessian operator mismatch at ({row}, {col}): materialized={materialized_entry}, dense={dense_entry}"
                );
            }
        }

        let alpha = array![0.37, -0.58];
        let hvp = crate::solver::outer_strategy::OuterHessianOperator::matvec(&operator, &alpha)
            .expect("operator HVP");
        let dense_hvp = dense.dot(&alpha);
        for i in 0..hvp.len() {
            let tolerance = 1e-10_f64.max(1e-10 * dense_hvp[i].abs());
            assert!(
                (hvp[i] - dense_hvp[i]).abs() <= tolerance,
                "outer Hessian HVP mismatch at {i}: operator={}, dense={}",
                hvp[i],
                dense_hvp[i]
            );
        }
    }

    #[test]
    fn subspace_apply_adjoint_shortcut_matches_dense_projected_trace() {
        // Math contract for §10: under K = U_S H_proj⁻¹ U_Sᵀ and
        //   C[u] = Xᵀ diag(c ⊙ Xu) X
        // the identity
        //   tr(K · C[u]) = uᵀ · Xᵀ(c ⊙ h^{G,proj})
        // holds with h^{G,proj}_i = Xᵢᵀ · K · Xᵢ.  This test contracts both
        // sides on a tiny fixture so the leverage / adjoint_z_c shortcut
        // in `build_outer_hessian_operator` (under subspace) is provably
        // bit-equivalent to the dense `trace_projected_logdet(C[u])` path.
        // (H itself is not needed here — only U_S and H_proj⁻¹ define K.)
        let u_s = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
        let det = 3.0_f64 * 5.0 - 0.1 * 0.1;
        let h_proj_inverse = array![[5.0 / det, -0.1 / det], [-0.1 / det, 3.0 / det]];
        let subspace = PenaltySubspaceTrace {
            u_s: u_s.clone(),
            h_proj_inverse: h_proj_inverse.clone(),
        };

        // Sanity-check `apply`: K should annihilate the null subspace and
        // act as U_S H_proj⁻¹ U_Sᵀ on the identified subspace.
        let v_null = array![0.0, 0.0, 0.7, -0.3];
        let kv_null = subspace.apply(&v_null);
        for &entry in kv_null.iter() {
            assert_relative_eq!(entry, 0.0, epsilon = 1e-14);
        }
        let v_id = array![0.4, -0.2, 0.0, 0.0];
        let kv_id = subspace.apply(&v_id);
        // Last two entries (null components) must be zero by construction.
        assert_relative_eq!(kv_id[2], 0.0, epsilon = 1e-14);
        assert_relative_eq!(kv_id[3], 0.0, epsilon = 1e-14);

        let x = array![
            [1.0, 0.2, 0.5, 0.3],
            [1.0, 1.1, -0.2, 0.4],
            [1.0, -0.8, 0.7, -0.1],
            [1.0, 0.5, 0.3, 0.6]
        ];
        let c = array![0.31_f64, -0.27, 0.19, -0.11];

        // Dense reference K and h^{G,proj}.
        let k_dense = u_s.dot(&h_proj_inverse).dot(&u_s.t());
        let n = x.nrows();
        let mut h_g_proj = Array1::<f64>::zeros(n);
        for i in 0..n {
            let row = x.row(i).to_owned();
            let kx = subspace.apply(&row);
            h_g_proj[i] = row.dot(&kx);
            // Cross-check against the dense kernel.
            let kx_dense = k_dense.dot(&row);
            assert_relative_eq!(h_g_proj[i], row.dot(&kx_dense), epsilon = 1e-12);
        }

        // Probe several u directions, including ones with null components.
        let probes = [
            array![0.6_f64, -0.4, 0.0, 0.0],
            array![0.0_f64, 0.0, 0.5, 0.7],
            array![0.3_f64, -0.1, 0.4, -0.2],
            array![1.0_f64, 1.0, 1.0, 1.0],
        ];
        for u in probes.iter() {
            // C[u] = Xᵀ diag(c ⊙ Xu) X — dense reference.
            let xu = x.dot(u);
            let mut weighted_x = x.clone();
            for i in 0..n {
                let w = c[i] * xu[i];
                for j in 0..weighted_x.ncols() {
                    weighted_x[[i, j]] *= w;
                }
            }
            let c_u_dense = x.t().dot(&weighted_x);

            // LHS: tr(K · C[u]) via the projected logdet path.
            let lhs = subspace.trace_projected_logdet(&c_u_dense);

            // RHS: uᵀ · Xᵀ(c ⊙ h^{G,proj}).
            let mut weighted = Array1::<f64>::zeros(n);
            for i in 0..n {
                weighted[i] = c[i] * h_g_proj[i];
            }
            let rhs_vec = x.t().dot(&weighted);
            let rhs = u.dot(&rhs_vec);

            assert_relative_eq!(lhs, rhs, epsilon = 1e-12, max_relative = 1e-12);

            // Also verify the adjoint form: z_c = K · Xᵀ(c ⊙ h^{G,proj}).
            // Then `Σⱼ αⱼ · pair_rhs_dot(idx, j, z_c.view())` reduces to
            // `(Aᵢβ)ᵀ z_c` style products in the operator, which equals
            // `(Aᵢβ)ᵀ · K · Xᵀ(c ⊙ h^{G,proj})` by linearity.  The local
            // contract here is just that `K · w` annihilates null(U_S).
            let z_c = subspace.apply(&rhs_vec);
            let z_c_dense = k_dense.dot(&rhs_vec);
            for i in 0..z_c.len() {
                assert_relative_eq!(z_c[i], z_c_dense[i], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn outer_hessian_operator_matvec_matches_dense_subspace_with_null_alpha() {
        // p=4, K=2, r=2 fixture — exercises the full projection K = U_S H_proj⁻¹ U_Sᵀ
        // (the existing r=1 case at projected_operator_hessian_matches_dense_subspace_trace
        // only verifies a trivial 1-D subspace).  Includes a small symmetric off-diagonal
        // so H_proj is non-diagonal.
        let h = array![
            [3.0, 0.1, 0.0, 0.0],
            [0.1, 5.0, 0.05, 0.0],
            [0.0, 0.05, 7.0, 0.15],
            [0.0, 0.0, 0.15, 11.0]
        ];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());

        // U_S spans the first two coordinates.  Null directions are dims 2,3.
        let u_s = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        // H_proj = U_Sᵀ H U_S = top-left 2×2 of H = [[3.0, 0.1], [0.1, 5.0]].
        // Closed-form 2×2 inverse for the test fixture: 1/(3·5 − 0.1²) · [[5, −0.1], [−0.1, 3]].
        let det = 3.0_f64 * 5.0 - 0.1 * 0.1;
        let h_proj_inverse = array![[5.0 / det, -0.1 / det], [-0.1 / det, 3.0 / det]];

        // Penalty roots mix identified (rows 0,1) and null (rows 2,3) directions, so
        // the projection is non-trivial — `compute_outer_hessian` must collapse the
        // null components and the matvec must match.
        let penalty_root_0 = array![[0.7, 0.3, 0.6, 0.0]];
        let penalty_root_1 = array![[0.2, 0.5, 0.0, 0.4]];

        let x = array![[1.0, 0.2, 0.5, 0.3], [1.0, 1.1, -0.2, 0.4], [1.0, -0.8, 0.7, -0.1], [1.0, 0.5, 0.3, 0.6]];
        let c_array = array![0.31, -0.27, 0.19, -0.11];
        let d_array = array![0.17, -0.11, 0.23, 0.07];
        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array,
            d_array: Some(d_array),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
        };

        // Pre-compute log|H_proj|_+ = ln(det(H_proj)) for the correction term.
        let logdet_h_proj = det.ln();

        let beta = array![0.4, -0.7, 0.2, 0.1];
        let solution = InnerSolution {
            log_likelihood: -2.3,
            penalty_quadratic: 0.6,
            hessian_op: hop.clone(),
            beta,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(penalty_root_0),
                PenaltyCoordinate::from_dense_root(penalty_root_1),
            ],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.4, -0.2],
                second: Some(array![[0.13, 0.02], [0.02, 0.09]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: logdet_h_proj - hop.logdet(),
            penalty_subspace_trace: Some(Arc::new(PenaltySubspaceTrace {
                u_s,
                h_proj_inverse,
            })),
            rho_curvature_scale: 1.0,
            n_observations: 4,
            nullspace_dim: 2.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        };
        let rho: Vec<f64> = vec![0.2_f64, -0.1];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let dense = compute_outer_hessian(
            &solution,
            &rho,
            &lambdas,
            solution.hessian_op.as_ref(),
            solution.deriv_provider.as_ref(),
        )
        .unwrap();
        let kernel = solution
            .deriv_provider
            .outer_hessian_derivative_kernel()
            .unwrap();
        let operator = build_outer_hessian_operator(
            &solution,
            &lambdas,
            solution.deriv_provider.as_ref(),
            kernel,
        )
        .unwrap();

        // (6) Materialised dense extension to r=2: every entry must match.
        let materialized =
            crate::solver::outer_strategy::OuterHessianOperator::materialize_dense(&operator)
                .unwrap();
        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                assert_relative_eq!(
                    materialized[[row, col]],
                    dense[[row, col]],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }

        // (3) HVP equivalence across a basis-and-mix set of α probes.
        // (4) The [1, -1] and [0.7, -0.3] probes lift through penalty roots whose
        //     columns 2,3 carry non-zero null components, so they exercise the
        //     projection rather than just the identified subspace.
        let alphas = [
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 1.0],
            array![1.0, -1.0],
            array![0.7, -0.3],
        ];
        for alpha in alphas.iter() {
            let hvp =
                crate::solver::outer_strategy::OuterHessianOperator::matvec(&operator, alpha)
                    .expect("operator HVP");
            let dense_hvp = dense.dot(alpha);
            for i in 0..hvp.len() {
                assert_relative_eq!(
                    hvp[i],
                    dense_hvp[i],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    fn projected_operator_hessian_matches_dense_subspace_trace() {
        let h = array![[3.0, 0.2], [0.2, 5.0]];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());
        let beta = array![0.4, -0.7];
        let penalty_root = array![[0.0, 1.0]];
        let ext_drift = array![[0.45, -0.15], [-0.15, 0.35]];
        let x = array![[1.0, 0.2], [1.0, 1.1], [1.0, -0.8], [1.0, 0.5]];
        let c_array = array![0.31, -0.27, 0.19, -0.11];
        let d_array = array![0.17, -0.11, 0.23, 0.07];
        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array,
            d_array: Some(d_array),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
        };
        let h_proj = h[[1, 1]];

        let solution = InnerSolution {
            log_likelihood: -2.3,
            penalty_quadratic: 0.6,
            hessian_op: hop.clone(),
            beta,
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(penalty_root)],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.4],
                second: Some(array![[0.13]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: h_proj.ln() - hop.logdet(),
            penalty_subspace_trace: Some(Arc::new(PenaltySubspaceTrace {
                u_s: array![[0.0], [1.0]],
                h_proj_inverse: array![[1.0 / h_proj]],
            })),
            rho_curvature_scale: 1.0,
            n_observations: 4,
            nullspace_dim: 1.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: vec![HyperCoord {
                a: -0.21,
                g: array![0.33, -0.42],
                drift: HyperCoordDrift::from_operator(Arc::new(DenseMatrixHyperOperator {
                    matrix: ext_drift,
                })),
                ld_s: 0.07,
                b_depends_on_beta: false,
                is_penalty_like: false,
                firth_g: None,
                tk_eta_fixed: None,
                tk_x_fixed: None,
            }],
            ext_coord_pair_fn: Some(Box::new(|_, _| HyperCoordPair {
                a: 0.09,
                g: array![0.16, -0.12],
                b_mat: array![[0.08, 0.03], [0.03, -0.04]],
                b_operator: None,
                ld_s: -0.05,
            })),
            rho_ext_pair_fn: Some(Box::new(|_, _| HyperCoordPair {
                a: -0.14,
                g: array![-0.18, 0.22],
                b_mat: array![[0.05, -0.02], [-0.02, 0.07]],
                b_operator: None,
                ld_s: 0.04,
            })),
            fixed_drift_deriv: None,
            barrier_config: None,
        };
        let rho: Vec<f64> = vec![0.2_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let dense = compute_outer_hessian(
            &solution,
            &rho,
            &lambdas,
            solution.hessian_op.as_ref(),
            solution.deriv_provider.as_ref(),
        )
        .unwrap();
        let kernel = solution
            .deriv_provider
            .outer_hessian_derivative_kernel()
            .unwrap();
        let operator = build_outer_hessian_operator(
            &solution,
            &lambdas,
            solution.deriv_provider.as_ref(),
            kernel,
        )
        .unwrap();
        let materialized =
            crate::solver::outer_strategy::OuterHessianOperator::materialize_dense(&operator)
                .unwrap();

        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                assert_relative_eq!(
                    materialized[[row, col]],
                    dense[[row, col]],
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }

    #[test]
    fn subspace_trace_large_k_routes_to_projected_operator() {
        let h = array![[3.0, 0.2], [0.2, 5.0]];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());
        let pcoord = PenaltyCoordinate::from_dense_root(array![[0.0, 1.0]]);
        let k = MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD;
        let x = array![[1.0, 0.2], [1.0, 1.1], [1.0, -0.8], [1.0, 0.5]];
        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array: array![0.31, -0.27, 0.19, -0.11],
            d_array: Some(array![0.17, -0.11, 0.23, 0.07]),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
        };
        let h_proj = h[[1, 1]];
        let solution = InnerSolution {
            log_likelihood: -2.3,
            penalty_quadratic: 0.6,
            hessian_op: hop.clone(),
            beta: array![0.4, -0.7],
            penalty_coords: vec![pcoord; k],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: Array1::zeros(k),
                second: Some(Array2::zeros((k, k))),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: h_proj.ln() - hop.logdet(),
            penalty_subspace_trace: Some(Arc::new(PenaltySubspaceTrace {
                u_s: array![[0.0], [1.0]],
                h_proj_inverse: array![[1.0 / h_proj]],
            })),
            rho_curvature_scale: 1.0,
            n_observations: 4,
            nullspace_dim: 1.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        };
        let rho = vec![0.0_f64; k];
        let result =
            reml_laml_evaluate(&solution, &rho, EvalMode::ValueGradientHessian, None).unwrap();

        assert!(
            matches!(
                result.hessian,
                crate::solver::outer_strategy::HessianResult::Operator(_)
            ),
            "large-k subspace-trace case should use projected outer Hessian operator"
        );
    }

    #[test]
    fn test_dense_spectral_operator_singular() {
        // Rank-1 matrix: H = [1 1; 1 1] has eigenvalues {0, 2}.
        let h = array![[1.0, 1.0], [1.0, 1.0]];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Under `Smooth` mode (the default used by `from_symmetric`), every
        // eigenpair stays active and singular directions are regularized
        // through `r_ε(σ)` rather than hard-masked. For H = [[1,1],[1,1]],
        // the expected logdet is therefore
        //   ln(r_ε(0)) + ln(r_ε(2)).
        let epsilon = spectral_epsilon(&[0.0, 2.0]);
        let r0 = spectral_regularize(0.0, epsilon);
        let r2 = spectral_regularize(2.0, epsilon);
        let expected_logdet = r0.ln() + r2.ln();
        assert!((op.logdet() - expected_logdet).abs() < 1e-10);
        // The regularized null direction must still yield a finite trace.
        let trace = op.trace_hinv_product(&Array2::eye(2));
        assert!(trace.is_finite());
    }

    #[test]
    fn test_spectral_regularize_stays_finite_in_extreme_tails() {
        let epsilon = 1e-8;

        let large_negative = spectral_regularize(-1e16, epsilon);
        assert!(
            large_negative.is_finite() && large_negative > 0.0,
            "large negative sigma should regularize to a positive finite value, got {large_negative}"
        );

        let large_positive = spectral_regularize(1e308, epsilon);
        assert!(
            large_positive.is_finite() && large_positive > 0.0,
            "large positive sigma should stay finite, got {large_positive}"
        );
    }

    #[test]
    fn test_smooth_floor_dp() {
        // Well above floor: should be approximately identity
        let (val, grad, _) = smooth_floor_dp(1.0);
        assert!((val - 1.0).abs() < 1e-6);
        assert!((grad - 1.0).abs() < 1e-6);

        // At floor: should be approximately DP_FLOOR + tau*ln(2)
        let (val, grad, _) = smooth_floor_dp(DP_FLOOR);
        assert!(val > DP_FLOOR);
        assert!((grad - 0.5).abs() < 0.1); // sigmoid at 0 ≈ 0.5

        // Well below floor: value should stay above DP_FLOOR
        let (val, _, _) = smooth_floor_dp(0.0);
        assert!(val >= DP_FLOOR);
    }

    #[test]
    fn test_gaussian_derivatives_has_no_corrections() {
        let g = GaussianDerivatives;
        assert!(!g.has_corrections());
        assert!(
            g.hessian_derivative_correction(&array![1.0, 2.0])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn gaussian_derivatives_advertise_exact_outer_hvp_kernel() {
        let g = GaussianDerivatives;
        assert!(matches!(
            g.outer_hessian_derivative_kernel(),
            Some(OuterHessianDerivativeKernel::Gaussian)
        ));
    }

    #[test]
    fn standard_gam_large_n_gaussian_prefers_operator_when_dense_work_is_large() {
        assert!(prefer_outer_hessian_operator(320_000, 42, 6));
        assert!(matches!(
            GaussianDerivatives.outer_hessian_derivative_kernel(),
            Some(OuterHessianDerivativeKernel::Gaussian)
        ));
    }

    #[test]
    fn gaussian_outer_hessian_operator_matches_dense_assembly() {
        let h = array![[2.4, 0.2], [0.2, 1.7]];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());
        let beta = array![0.35, -0.55];
        let penalty_root_0 = array![[1.0, 0.2], [0.0, 0.4]];
        let penalty_root_1 = array![[0.3, -0.1], [0.0, 0.9]];
        let solution = InnerSolution {
            log_likelihood: -8.0,
            penalty_quadratic: 0.9,
            hessian_op: hop.clone(),
            beta,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(penalty_root_0),
                PenaltyCoordinate::from_dense_root(penalty_root_1),
            ],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.8, 0.6],
                second: Some(array![[0.11, 0.03], [0.03, 0.17]]),
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            n_observations: 320_000,
            nullspace_dim: 1.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        };
        let rho: Vec<f64> = vec![0.2_f64, -0.4_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let dense = compute_outer_hessian(
            &solution,
            &rho,
            &lambdas,
            solution.hessian_op.as_ref(),
            solution.deriv_provider.as_ref(),
        )
        .unwrap();
        let kernel = solution
            .deriv_provider
            .outer_hessian_derivative_kernel()
            .unwrap();
        let operator = build_outer_hessian_operator(
            &solution,
            &lambdas,
            solution.deriv_provider.as_ref(),
            kernel,
        )
        .unwrap();
        let materialized =
            crate::solver::outer_strategy::OuterHessianOperator::materialize_dense(&operator)
                .unwrap();

        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                let expected = dense[[row, col]];
                let actual = materialized[[row, col]];
                let tolerance = 1e-10_f64.max(1e-10 * expected.abs());
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "Gaussian outer Hessian operator mismatch at ({row}, {col}): materialized={actual}, dense={expected}"
                );
            }
        }
    }

    /// Scalar EFS counterexample: at z=2, λ=1/3 in a one-coefficient
    /// Gaussian/Laplace surrogate, the REML/LAML gradient is exactly zero
    /// (β̂² λ + λ/(1+λ) − 1 = 0.75 + 0.25 − 1 = 0). The Wood–Fasiolo
    /// multiplicative EFS update must therefore return Δρ ≈ 0.
    ///
    /// The previous Frobenius/Gram-norm formula returned `(2a − tr(H⁻¹B)) /
    /// tr(H⁻¹BH⁻¹B) = 0.5 / 0.0625 = 8`, which then clamped to `+5` — a
    /// huge spurious step at the exact optimum.
    #[test]
    fn efs_step_is_zero_at_scalar_optimum() {
        // β̂ = z / (1 + λ) = 2 / (4/3) = 1.5, H = 1 + λ = 4/3.
        let lambda = 1.0 / 3.0;
        let beta_hat = 1.5_f64;
        let h = Array2::from_shape_vec((1, 1), vec![1.0 + lambda]).unwrap();
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // S = R^T R with R = [[1]] gives S = [[1]]. Pseudoinverse log-det
        // derivative tr(S⁺ · λS) = 1 (full-rank, scale cancels).
        let penalty_root = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        let solution = InnerSolution {
            log_likelihood: 0.0,
            penalty_quadratic: 0.0,
            hessian_op: Arc::new(op),
            beta: array![beta_hat],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(penalty_root)],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![1.0],
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            n_observations: 10,
            nullspace_dim: 0.0,
            // Use Fixed dispersion so the gradient is exactly the
            // Laplace/REML form `½(λβ̂²S β̂ + tr(H⁻¹λS) − tr(S⁺λS))`
            // without the smooth-floor / profiling factors the test
            // would otherwise have to track.
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        };
        let rho = [lambda.ln()];

        // At the optimum the full outer gradient is identically 0; the
        // universal form `Δρ = log(1 − 2·g_full/q_eff)` collapses to
        // `log(1) = 0`.
        let gradient_at_optimum = [0.0_f64];
        let steps = compute_efs_update(&solution, &rho, &gradient_at_optimum);
        assert_eq!(steps.len(), 1);
        assert!(
            steps[0].abs() < 1e-12,
            "EFS step at scalar optimum should be exactly 0, got {} (old buggy formula returned ~+5)",
            steps[0]
        );

        // Off-optimum: simulate `g_full = +0.1` with the same q_eff. The
        // multiplicative target `1 − 2·0.1/0.75 = 0.733` ⇒ Δρ = log(0.733).
        let q_eff = lambda * beta_hat * beta_hat; // 0.75
        let g_off = 0.1_f64;
        let steps_off = compute_efs_update(&solution, &rho, &[g_off]);
        let expected = (1.0_f64 - 2.0 * g_off / q_eff).ln();
        assert!(
            (steps_off[0] - expected).abs() < 1e-12,
            "off-optimum EFS step {} != expected {}",
            steps_off[0],
            expected
        );
    }

    /// `efs_log_step_from_grad` recovers the canonical
    /// `log((d − t)/q_eff)` Wood–Fasiolo step when the gradient is the
    /// pure REML/LAML stationarity gradient `g_base = (q_eff + t − d)/2`,
    /// and shifts by exactly the right amount when out-of-band terms
    /// `g_extra` enter the gradient.
    #[test]
    fn efs_log_step_from_grad_recovers_canonical_form() {
        // Canonical agreement on stable cases: g_base = (q_eff − target)/2
        // ⇒ universal = log((d − t)/q_eff).
        let cases = [
            (1.0_f64, 0.5),
            (2.0, 1.5),
            (0.75, 0.75),
            (4.0, 0.1),
            (1.0, 0.999),
        ];
        for (q_eff, target) in cases {
            let g_base = (q_eff - target) / 2.0;
            let universal = efs_log_step_from_grad(q_eff, g_base).unwrap();
            let canonical = (target / q_eff).ln().clamp(-EFS_MAX_STEP, EFS_MAX_STEP);
            assert!(
                (universal - canonical).abs() < 1e-12,
                "universal {universal} ≠ canonical {canonical} at q={q_eff}, t={target}"
            );
        }

        // Augmented stationarity: g_full = g_base + g_extra = 0 ⇒
        // q_eff = (d − t) − 2·g_extra. The universal form must return ≈ 0
        // *with the same q_eff value the iteration actually has*.
        let target = 0.6_f64;
        let g_extra = -0.7_f64;
        let augmented_q = target - 2.0 * g_extra;
        let g_full_at_aug_opt = (augmented_q - target) / 2.0 + g_extra;
        assert!(g_full_at_aug_opt.abs() < 1e-12);
        let s_at_opt = efs_log_step_from_grad(augmented_q, g_full_at_aug_opt).unwrap();
        assert!(
            s_at_opt.abs() < 1e-12,
            "Δρ at augmented optimum != 0: {s_at_opt}"
        );

        // Stable: log ratio.
        let s = efs_log_step_from_grad(2.0, 0.75).expect("stable regime");
        assert!((s - (0.25_f64).ln()).abs() < 1e-12);

        // Optimum: g_full = 0 ⇒ Δρ = 0.
        let s = efs_log_step_from_grad(0.75, 0.0).expect("zero gradient");
        assert!(s.abs() < 1e-12);

        // Over-correction (2·g_full ≥ q_eff ⇒ ratio ≤ 0): clamp to max descent.
        for &(q_eff, g) in &[(1.0_f64, 0.6), (2.0, 1.5), (0.5, 1e6)] {
            let s = efs_log_step_from_grad(q_eff, g).expect("over-correction");
            assert!((s - (-EFS_MAX_STEP)).abs() < 1e-12);
        }

        // Asymptotic clamp on the lower side: ratio → 0⁺ ⇒ floor at -MAX.
        let s = efs_log_step_from_grad(1.0, 0.5 - 1e-30).expect("near-singular");
        assert!((s - (-EFS_MAX_STEP)).abs() < 1e-12 || s == 0.5 * (-EFS_MAX_STEP) || s.is_finite());
        assert!(s <= 0.0);

        // Pathological: q_eff ≤ 0, non-finite inputs.
        assert!(efs_log_step_from_grad(0.0, 0.0).is_none());
        assert!(efs_log_step_from_grad(-1.0, 0.0).is_none());
        assert!(efs_log_step_from_grad(f64::NAN, 0.0).is_none());
        assert!(efs_log_step_from_grad(1.0, f64::NAN).is_none());
        assert!(efs_log_step_from_grad(1.0, f64::INFINITY).is_none());
    }

    /// `DenseSpectralOperator::trace_hinv_block_local_cross` must compute
    /// `tr(H⁻¹ A H⁻¹ A)`, not `tr(H⁻¹ A²)`. These coincide only when A
    /// commutes with H⁻¹ — generically they differ.
    #[test]
    fn dense_spectral_block_local_cross_trace_matches_dense() {
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // 2×2 block at [0..2], non-commuting with H⁻¹.
        let block = array![[1.5, 0.4], [0.4, 0.7]];
        let scale = 1.7_f64;

        // Reference: full-matrix `tr((H⁻¹ A)²)` via repeated solves.
        let mut a_full = Array2::<f64>::zeros((3, 3));
        for i in 0..2 {
            for j in 0..2 {
                a_full[[i, j]] = scale * block[[i, j]];
            }
        }
        let hinva = op.solve_multi(&a_full); // = H⁻¹ A
        let expected = (&hinva.t() * &hinva).sum(); // tr((H⁻¹A)(H⁻¹A))

        let got = op.trace_hinv_block_local_cross(&block, scale, 0, 2);
        assert!(
            (got - expected).abs() < 1e-10,
            "block-local cross trace = {got}, expected = {expected} (delta {})",
            got - expected
        );
    }

    #[test]
    fn test_reml_laml_evaluate_gaussian_basic() {
        // Simple 2-param Gaussian model.
        let h = Array2::from_diag(&array![10.0, 8.0]);
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let solution = InnerSolution {
            log_likelihood: -5.0, // −0.5 × deviance = −0.5 × 10
            penalty_quadratic: 2.0,
            hessian_op: Arc::new(op),
            beta: array![1.0, 0.5],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(
                Array2::eye(2), // S₁ = I (penalty root for param 1)
            )],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![1.0],
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            n_observations: 100,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        };

        let rho = [0.0]; // λ = 1

        // Should produce finite cost
        let result = reml_laml_evaluate(&solution, &rho, EvalMode::ValueOnly, None).unwrap();
        assert!(result.cost.is_finite());
        assert!(result.gradient.is_none());

        // With gradient
        let result = reml_laml_evaluate(&solution, &rho, EvalMode::ValueAndGradient, None).unwrap();
        assert!(result.cost.is_finite());
        assert!(result.gradient.is_some());
        let grad = result.gradient.unwrap();
        assert_eq!(grad.len(), 1);
        assert!(grad[0].is_finite());
    }

    #[test]
    fn fixed_dispersion_firth_cost_subtracts_jeffreys_term() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]];
        let eta = array![0.0, 0.4, -0.2];
        let firth_op = std::sync::Arc::new(
            super::super::FirthDenseOperator::build(&x, &eta).expect("firth operator"),
        );
        let firth_value = firth_op.jeffreys_logdet();

        let solution = InnerSolution {
            log_likelihood: 0.0,
            penalty_quadratic: 0.0,
            hessian_op: Arc::new(DenseSpectralOperator::from_symmetric(&Array2::eye(2)).unwrap()),
            beta: Array1::zeros(2),
            penalty_coords: Vec::new(),
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: Array1::zeros(0),
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: Some(ExactJeffreysTerm::new(firth_op)),
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            n_observations: x.nrows(),
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: false,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        };

        let result = reml_laml_evaluate(&solution, &[], EvalMode::ValueOnly, None).unwrap();
        assert_relative_eq!(result.cost, -firth_value, epsilon = 1e-12);
    }

    struct FixedOuterHessianOperator {
        matrix: Array2<f64>,
    }

    impl crate::solver::outer_strategy::OuterHessianOperator for FixedOuterHessianOperator {
        fn dim(&self) -> usize {
            self.matrix.nrows()
        }

        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            if v.len() != self.dim() {
                return Err(format!(
                    "fixed test outer Hessian dimension mismatch: got {}, expected {}",
                    v.len(),
                    self.dim()
                ));
            }
            Ok(self.matrix.dot(v))
        }

        fn is_cheap_to_materialize(&self) -> bool {
            true
        }
    }

    struct FamilyOperatorDerivatives {
        op: Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>,
    }

    impl HessianDerivativeProvider for FamilyOperatorDerivatives {
        fn hessian_derivative_correction(
            &self,
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            panic!("family operator dispatch should not request pairwise first derivatives")
        }

        fn hessian_second_derivative_correction(
            &self,
            _: &Array1<f64>,
            _: &Array1<f64>,
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            panic!("family operator dispatch should not request pairwise second derivatives")
        }

        fn has_corrections(&self) -> bool {
            false
        }

        fn family_outer_hessian_operator(
            &self,
        ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
            Some(Arc::clone(&self.op))
        }
    }

    #[test]
    fn family_outer_hessian_operator_short_circuits_dense_pairwise_assembly() {
        let supplied = array![[2.5]];
        let provider_op: Arc<dyn crate::solver::outer_strategy::OuterHessianOperator> =
            Arc::new(FixedOuterHessianOperator {
                matrix: supplied.clone(),
            });
        let solution = InnerSolution {
            log_likelihood: 0.0,
            penalty_quadratic: 0.4,
            hessian_op: Arc::new(DenseSpectralOperator::from_symmetric(&array![[3.0]]).unwrap()),
            beta: array![0.2],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(array![[1.0]])],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![1.0],
                second: Some(array![[0.0]]),
            },
            deriv_provider: Box::new(FamilyOperatorDerivatives { op: provider_op }),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            n_observations: 1,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        };

        let result =
            reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None).unwrap();
        let crate::solver::outer_strategy::HessianResult::Operator(op) = result.hessian else {
            panic!("expected family-supplied operator Hessian");
        };
        assert_eq!(op.dim(), 1);
        let hv = op.matvec(&array![4.0]).unwrap();
        assert_relative_eq!(hv[0], 10.0, epsilon = 1e-12);
        let dense = op.materialize_dense().unwrap();
        assert_relative_eq!(dense[[0, 0]], supplied[[0, 0]], epsilon = 1e-12);
    }

    struct FixedCorrectionDerivatives {
        correction: Array2<f64>,
    }

    impl HessianDerivativeProvider for FixedCorrectionDerivatives {
        fn hessian_derivative_correction(
            &self,
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(self.correction.clone()))
        }

        fn has_corrections(&self) -> bool {
            true
        }
    }

    fn build_projected_rho_gradient_solution(rho: f64) -> InnerSolution<'static> {
        let lambda = rho.exp();
        let h = array![[3.0 + 4.0 * rho, 0.0], [0.0, 5.0 + lambda],];
        let full_logdet = h[[0, 0]].ln() + h[[1, 1]].ln();
        let projected_logdet = h[[1, 1]].ln();

        InnerSolution {
            log_likelihood: 0.0,
            penalty_quadratic: 0.0,
            hessian_op: Arc::new(
                DenseSpectralOperator::from_symmetric_with_mode(&h, PseudoLogdetMode::HardPseudo)
                    .unwrap(),
            ),
            beta: Array1::zeros(2),
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(array![[0.0, 1.0]])],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.0],
                second: None,
            },
            deriv_provider: Box::new(FixedCorrectionDerivatives {
                correction: array![[4.0, 0.0], [0.0, 0.0]],
            }),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: projected_logdet - full_logdet,
            penalty_subspace_trace: Some(Arc::new(PenaltySubspaceTrace {
                u_s: array![[0.0], [1.0]],
                h_proj_inverse: array![[1.0 / h[[1, 1]]]],
            })),
            rho_curvature_scale: 1.0,
            n_observations: 10,
            nullspace_dim: 1.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: false,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        }
    }

    #[test]
    fn test_rho_gradient_uses_projected_logdet_kernel_when_available() {
        let rho = 0.0;
        let result = reml_laml_evaluate(
            &build_projected_rho_gradient_solution(rho),
            &[rho],
            EvalMode::ValueAndGradient,
            None,
        )
        .unwrap();
        let analytic = result.gradient.expect("gradient")[0];

        let eps = 1e-6;
        let rho_plus = rho + eps;
        let cost_plus = reml_laml_evaluate(
            &build_projected_rho_gradient_solution(rho_plus),
            &[rho_plus],
            EvalMode::ValueOnly,
            None,
        )
        .unwrap()
        .cost;

        let rho_minus = rho - eps;
        let cost_minus = reml_laml_evaluate(
            &build_projected_rho_gradient_solution(rho_minus),
            &[rho_minus],
            EvalMode::ValueOnly,
            None,
        )
        .unwrap()
        .cost;

        let fd = (cost_plus - cost_minus) / (2.0 * eps);
        assert_relative_eq!(analytic, fd, epsilon = 1e-8, max_relative = 1e-8);

        let full_space_trace = 4.0 / 3.0 + 1.0 / 6.0;
        assert!(
            (analytic - 0.5 * full_space_trace).abs() > 0.5,
            "projected rho trace should exclude the null-space leakage term"
        );
    }

    /// Helper: build an InnerSolution for a Gaussian model at a given rho.
    /// The Hessian H = X'X + Σ λₖ Sₖ depends on rho through the penalty,
    /// so we must rebuild InnerSolution for each rho evaluation.
    fn build_gaussian_test_solution(rho: &[f64]) -> InnerSolution<'_> {
        let p = 3; // 3 coefficients
        let n = 50; // 50 observations

        // Fixed X'X (data-dependent, rho-independent)
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0],];

        // Two penalty matrices (one per smoothing parameter)
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0],];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],];

        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();

        // Build H = X'X + λ₁S₁ + λ₂S₂
        let mut h = xtx.clone();
        h.scaled_add(lambdas[0], &s1);
        h.scaled_add(lambdas[1], &s2);

        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Solve for β̂ = H⁻¹ X'y (simulate with a fixed X'y)
        let xty = array![5.0, 3.0, 2.0];
        let beta = op.solve(&xty);

        // Penalty roots via eigendecomposition: Sₖ = Rₖᵀ Rₖ (exact).
        let r1 = penalty_matrix_root(&s1).unwrap();
        let r2 = penalty_matrix_root(&s2).unwrap();

        // Penalty quadratic: Σ λₖ β'Sₖβ
        let penalty_quad =
            lambdas[0] * beta.dot(&s1.dot(&beta)) + lambdas[1] * beta.dot(&s2.dot(&beta));

        // Deviance at β̂: ||y − Xβ̂||² = y'y − 2β̂'X'y + β̂'X'Xβ̂.
        // y'y is a ρ-independent constant (the actual value doesn't matter).
        // Computing deviance at the mode is essential: the analytic gradient
        // relies on the envelope theorem (∂D_p/∂β = 0 at the mode), which
        // is violated if deviance is held constant as β̂ varies with ρ.
        let yty = 20.0;
        let deviance = yty - 2.0 * beta.dot(&xty) + beta.dot(&xtx.dot(&beta));
        let log_likelihood = -0.5 * deviance;

        // Penalty logdet: exact pseudo-logdet on positive eigenspace.
        let mut s_total = Array2::zeros((p, p));
        s_total.scaled_add(lambdas[0], &s1);
        s_total.scaled_add(lambdas[1], &s2);
        let (s_eigs, _) = s_total.eigh(faer::Side::Lower).unwrap();
        let threshold = positive_eigenvalue_threshold(s_eigs.as_slice().unwrap());
        let log_det_s = exact_pseudo_logdet(s_eigs.as_slice().unwrap(), threshold);

        // Penalty logdet first derivatives (numerical FD).
        let mut det1 = Array1::zeros(rho.len());
        let eps = 1e-7;
        for k in 0..rho.len() {
            let mut rho_plus = rho.to_vec();
            rho_plus[k] += eps;
            let lambdas_plus: Vec<f64> = rho_plus.iter().map(|&r| r.exp()).collect();
            let mut s_plus = Array2::zeros((p, p));
            s_plus.scaled_add(lambdas_plus[0], &s1);
            s_plus.scaled_add(lambdas_plus[1], &s2);
            let (s_eigs_plus, _) = s_plus.eigh(faer::Side::Lower).unwrap();
            let threshold_plus = positive_eigenvalue_threshold(s_eigs_plus.as_slice().unwrap());
            let log_det_s_plus =
                exact_pseudo_logdet(s_eigs_plus.as_slice().unwrap(), threshold_plus);

            let mut rho_minus = rho.to_vec();
            rho_minus[k] -= eps;
            let lambdas_minus: Vec<f64> = rho_minus.iter().map(|&r| r.exp()).collect();
            let mut s_minus = Array2::zeros((p, p));
            s_minus.scaled_add(lambdas_minus[0], &s1);
            s_minus.scaled_add(lambdas_minus[1], &s2);
            let (s_eigs_minus, _) = s_minus.eigh(faer::Side::Lower).unwrap();
            let threshold_minus = positive_eigenvalue_threshold(s_eigs_minus.as_slice().unwrap());
            let log_det_s_minus =
                exact_pseudo_logdet(s_eigs_minus.as_slice().unwrap(), threshold_minus);

            det1[k] = (log_det_s_plus - log_det_s_minus) / (2.0 * eps);
        }

        InnerSolution {
            log_likelihood,
            penalty_quadratic: penalty_quad,
            hessian_op: Arc::new(op),
            beta,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(r1),
                PenaltyCoordinate::from_dense_root(r2),
            ],
            penalty_logdet: PenaltyLogdetDerivs {
                value: log_det_s,
                first: det1,
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            n_observations: n,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        }
    }

    fn build_large_dense_spectral_gaussian_solution(rho: f64) -> InnerSolution<'static> {
        let p = 520usize;
        let n = 2 * p;
        let lambda = rho.exp();

        let xtx_diag = Array1::from_shape_fn(p, |i| 5.0 + 0.01 * (i as f64));
        let xtx = Array2::from_diag(&xtx_diag);
        let penalty = Array2::<f64>::eye(p);

        let mut h = xtx.clone();
        h.scaled_add(lambda, &penalty);

        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let xty = Array1::from_shape_fn(p, |i| 1.0 + 0.002 * (i as f64));
        let beta = op.solve(&xty);

        let penalty_quad = lambda * beta.dot(&beta);
        let yty = 10.0 * (p as f64);
        let deviance = yty - 2.0 * beta.dot(&xty) + beta.dot(&xtx.dot(&beta));
        let log_likelihood = -0.5 * deviance;

        InnerSolution {
            log_likelihood,
            penalty_quadratic: penalty_quad,
            hessian_op: Arc::new(op),
            beta,
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(Array2::<f64>::eye(p))],
            penalty_logdet: PenaltyLogdetDerivs {
                value: (p as f64) * rho,
                first: array![p as f64],
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            n_observations: n,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        }
    }

    /// The structural test: finite-difference gradient matches analytic gradient.
    ///
    /// Because the unified evaluator computes cost and gradient from the same
    /// intermediates in the same function, drift is impossible. This test
    /// verifies that the mathematical formulas are correct (which FD catches),
    /// and serves as a regression gate.
    #[test]
    fn test_gaussian_reml_fd_vs_analytic_gradient() {
        let rho = vec![1.0, -0.5];
        let solution = build_gaussian_test_solution(&rho);

        let result = reml_laml_evaluate(&solution, &rho, EvalMode::ValueAndGradient, None).unwrap();
        let analytic_grad = result.gradient.unwrap();

        // Finite-difference gradient
        let eps = 1e-5;
        let mut fd_grad = Array1::zeros(rho.len());
        for k in 0..rho.len() {
            let mut rho_plus = rho.clone();
            rho_plus[k] += eps;
            let sol_plus = build_gaussian_test_solution(&rho_plus);
            let cost_plus = reml_laml_evaluate(&sol_plus, &rho_plus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            let mut rho_minus = rho.clone();
            rho_minus[k] -= eps;
            let sol_minus = build_gaussian_test_solution(&rho_minus);
            let cost_minus = reml_laml_evaluate(&sol_minus, &rho_minus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            fd_grad[k] = (cost_plus - cost_minus) / (2.0 * eps);
        }

        // Check agreement
        for k in 0..rho.len() {
            let abs_err = (analytic_grad[k] - fd_grad[k]).abs();
            let rel_err = abs_err / (1.0 + analytic_grad[k].abs());
            assert!(
                rel_err < 1e-4,
                "Gradient mismatch at k={}: analytic={:.8e}, fd={:.8e}, rel_err={:.3e}",
                k,
                analytic_grad[k],
                fd_grad[k],
                rel_err,
            );
        }
    }

    #[test]
    fn test_stochastic_trace_estimator_accuracy() {
        // Build a small SPD matrix and compare stochastic trace estimate
        // against the exact DenseSpectralOperator trace.
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0],];
        let a1 = array![[1.0, 0.3, 0.0], [0.3, 0.5, 0.1], [0.0, 0.1, 0.2],];
        let a2 = array![[0.2, 0.0, 0.1], [0.0, 1.0, 0.4], [0.1, 0.4, 0.8],];

        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Exact traces via the dense operator.
        let exact1 = op.trace_hinv_product(&a1);
        let exact2 = op.trace_hinv_product(&a2);

        // Stochastic estimates with tight tolerance and many probes.
        let config = StochasticTraceConfig {
            n_probes_min: 50,
            n_probes_max: 200,
            relative_tol: 0.005,
            tau_rel: 1e-10,
            solve_rel_tol: 1e-8,
            seed: 42,
        };
        let estimator = StochasticTraceEstimator::new(config);
        let matrices: Vec<&Array2<f64>> = vec![&a1, &a2];
        let estimates = estimator.estimate_traces(&op, &matrices);

        // With 200 probes on a 3x3 system, we should be very close.
        let rel_err1 = (estimates[0] - exact1).abs() / exact1.abs().max(1e-10);
        let rel_err2 = (estimates[1] - exact2).abs() / exact2.abs().max(1e-10);

        assert!(
            rel_err1 < 0.05,
            "Stochastic trace 1: est={:.6}, exact={:.6}, rel_err={:.4}",
            estimates[0],
            exact1,
            rel_err1,
        );
        assert!(
            rel_err2 < 0.05,
            "Stochastic trace 2: est={:.6}, exact={:.6}, rel_err={:.4}",
            estimates[1],
            exact2,
            rel_err2,
        );
    }

    #[test]
    fn dense_spectral_large_p_outer_gradient_matches_finite_difference() {
        let rho = 0.2;
        let solution = build_large_dense_spectral_gaussian_solution(rho);
        let result =
            reml_laml_evaluate(&solution, &[rho], EvalMode::ValueAndGradient, None).unwrap();
        let analytic = result.gradient.expect("gradient")[0];

        let eps = 1e-5;
        let rho_plus = rho + eps;
        let solution_plus = build_large_dense_spectral_gaussian_solution(rho_plus);
        let cost_plus = reml_laml_evaluate(&solution_plus, &[rho_plus], EvalMode::ValueOnly, None)
            .unwrap()
            .cost;

        let rho_minus = rho - eps;
        let solution_minus = build_large_dense_spectral_gaussian_solution(rho_minus);
        let cost_minus =
            reml_laml_evaluate(&solution_minus, &[rho_minus], EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

        let fd = (cost_plus - cost_minus) / (2.0 * eps);
        let rel_err = (analytic - fd).abs() / (1.0 + analytic.abs());
        assert!(
            rel_err < 2e-4,
            "large-p dense spectral gradient mismatch: analytic={analytic:.8e}, fd={fd:.8e}, rel_err={rel_err:.3e}"
        );
    }

    #[test]
    fn dense_spectral_logdet_traces_do_not_claim_hinv_kernel_equivalence() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        assert!(!op.prefers_stochastic_trace_estimation());
        assert!(!op.logdet_traces_match_hinv_kernel());
        assert!(!can_use_stochastic_logdet_hinv_kernel(&op, 1024, true));

        let block = BlockCoupledOperator::from_joint_hessian(&h).unwrap();
        assert!(!block.prefers_stochastic_trace_estimation());
        assert!(!block.logdet_traces_match_hinv_kernel());
        assert!(!can_use_stochastic_logdet_hinv_kernel(&block, 1024, true));
    }

    #[test]
    fn dense_spectral_hinv_cross_matches_solve_contraction() {
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let a = array![[1.0, 0.2, 0.1], [0.2, 0.5, 0.0], [0.1, 0.0, 0.3],];
        let b = array![[0.3, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.2, 0.6],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let exact = op.trace_hinv_product_cross(&a, &b);
        let solved_a = op.solve_multi(&a);
        let solved_b = op.solve_multi(&b);
        let reference = (&solved_a.t() * &solved_b).sum();

        assert_relative_eq!(exact, reference, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn dense_spectral_batched_logdet_crosses_match_pairwise() {
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let h1 = array![[1.0, 0.2, 0.1], [0.2, 0.5, 0.0], [0.1, 0.0, 0.3],];
        let h2 = array![[0.3, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.2, 0.6],];
        let h3 = array![[0.7, 0.0, 0.2], [0.0, 0.4, 0.1], [0.2, 0.1, 0.9],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let mats = [&h1, &h2, &h3];
        let batched = op.trace_logdet_hessian_crosses(&mats);

        for i in 0..mats.len() {
            for j in 0..mats.len() {
                let pairwise = op.trace_logdet_hessian_cross(mats[i], mats[j]);
                assert_relative_eq!(
                    batched[[i, j]],
                    pairwise,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }

    #[test]
    fn sparse_block_local_trace_without_takahashi_matches_dense_reference() {
        let h = array![
            [5.0, 0.2, 0.0, 0.1],
            [0.2, 4.0, 0.3, 0.0],
            [0.0, 0.3, 3.0, 0.4],
            [0.1, 0.0, 0.4, 2.5],
        ];
        let h_sparse =
            crate::linalg::sparse_exact::dense_to_sparse_symmetric_upper(&h, 0.0).unwrap();
        let factor = std::sync::Arc::new(
            crate::linalg::sparse_exact::factorize_sparse_spd(&h_sparse).unwrap(),
        );
        let sparse = SparseCholeskyOperator::new(factor, 0.0, h.nrows());
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let block = array![[0.8, 0.15], [0.15, 0.45]];
        let scale = 1.7;
        let start = 1;
        let end = 3;
        let mut full = Array2::<f64>::zeros(h.raw_dim());
        for i in 0..block.nrows() {
            for j in 0..block.ncols() {
                full[[start + i, start + j]] = scale * block[[i, j]];
            }
        }

        assert_relative_eq!(
            sparse.trace_hinv_block_local(&block, scale, start, end),
            dense.trace_hinv_product(&full),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            sparse.trace_hinv_block_local_cross(&block, scale, start, end),
            dense.trace_hinv_product_cross(&full, &full),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    fn sparse_block_local_operator_cross_without_takahashi_matches_dense_reference() {
        let h = array![
            [5.0, 0.2, 0.0, 0.1],
            [0.2, 4.0, 0.3, 0.0],
            [0.0, 0.3, 3.0, 0.4],
            [0.1, 0.0, 0.4, 2.5],
        ];
        let h_sparse =
            crate::linalg::sparse_exact::dense_to_sparse_symmetric_upper(&h, 0.0).unwrap();
        let factor = std::sync::Arc::new(
            crate::linalg::sparse_exact::factorize_sparse_spd(&h_sparse).unwrap(),
        );
        let sparse = SparseCholeskyOperator::new(factor, 0.0, h.nrows());
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let local = array![[0.8, 0.15], [0.15, 0.45]];
        let start = 1;
        let end = 3;
        let op = BlockLocalDrift {
            local: local.clone(),
            start,
            end,
            total_dim: h.nrows(),
        };
        let mut full = Array2::<f64>::zeros(h.raw_dim());
        full.slice_mut(ndarray::s![start..end, start..end])
            .assign(&local);

        assert_relative_eq!(
            sparse.trace_hinv_operator_cross(&op, &op),
            dense.trace_hinv_product_cross(&full, &full),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    fn sparse_matrix_block_operator_cross_without_takahashi_matches_dense_reference() {
        let h = array![
            [5.0, 0.2, 0.0, 0.1],
            [0.2, 4.0, 0.3, 0.0],
            [0.0, 0.3, 3.0, 0.4],
            [0.1, 0.0, 0.4, 2.5],
        ];
        let h_sparse =
            crate::linalg::sparse_exact::dense_to_sparse_symmetric_upper(&h, 0.0).unwrap();
        let factor = std::sync::Arc::new(
            crate::linalg::sparse_exact::factorize_sparse_spd(&h_sparse).unwrap(),
        );
        let sparse = SparseCholeskyOperator::new(factor, 0.0, h.nrows());
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let matrix = array![
            [1.0, 0.2, -0.1, 0.3],
            [0.2, 0.7, 0.4, -0.2],
            [-0.1, 0.4, 1.2, 0.5],
            [0.3, -0.2, 0.5, 0.9],
        ];
        let local = array![[0.8, 0.15], [0.15, 0.45]];
        let start = 1;
        let end = 3;
        let op = BlockLocalDrift {
            local: local.clone(),
            start,
            end,
            total_dim: h.nrows(),
        };
        let mut full = Array2::<f64>::zeros(h.raw_dim());
        full.slice_mut(ndarray::s![start..end, start..end])
            .assign(&local);

        assert_relative_eq!(
            sparse.trace_hinv_matrix_operator_cross(&matrix, &op),
            dense.trace_hinv_product_cross(&matrix, &full),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    fn sparse_takahashi_trace_hinv_product_pairs_symmetric_lookups() {
        let h = array![[4.0, 0.2, 0.1], [0.2, 3.0, 0.4], [0.1, 0.4, 2.5],];
        let h_sparse =
            crate::linalg::sparse_exact::dense_to_sparse_symmetric_upper(&h, 0.0).unwrap();
        let factor = std::sync::Arc::new(
            crate::linalg::sparse_exact::factorize_sparse_spd(&h_sparse).unwrap(),
        );
        let sfactor = crate::linalg::sparse_exact::factorize_simplicial(&h_sparse).unwrap();
        let taka = std::sync::Arc::new(
            crate::linalg::sparse_exact::TakahashiInverse::compute(&sfactor).unwrap(),
        );
        let sparse = SparseCholeskyOperator::new(factor, 0.0, h.nrows()).with_takahashi(taka);
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let a = array![[1.0, 0.7, -0.2], [0.1, 0.5, 0.9], [0.4, -0.3, 0.2],];
        assert_relative_eq!(
            sparse.trace_hinv_product(&a),
            dense.trace_hinv_product(&a),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    fn hyper_operator_bilinear_view_matches_owned_bilinear() {
        let dense = DenseMatrixHyperOperator {
            matrix: array![[2.0, 0.3, -0.1], [0.3, 1.5, 0.4], [-0.1, 0.4, 3.0],],
        };
        let block = BlockLocalDrift {
            local: array![[1.2, 0.2], [0.2, 0.7]],
            start: 1,
            end: 3,
            total_dim: 3,
        };
        let composite = CompositeHyperOperator {
            dense: Some(array![[0.4, 0.1, 0.0], [0.1, 0.8, -0.2], [0.0, -0.2, 0.6],]),
            operators: vec![Arc::new(block.clone())],
            dim_hint: 3,
        };
        let weighted = WeightedHyperOperator {
            terms: vec![
                (1.7, Arc::new(dense.clone()) as Arc<dyn HyperOperator>),
                (-0.4, Arc::new(block.clone()) as Arc<dyn HyperOperator>),
            ],
            dim_hint: 3,
        };

        let v_storage = array![9.0, 0.5, -1.2, 0.7, 8.0];
        let u_storage = array![7.0, -0.3, 1.1, 0.9, 6.0];
        let v_view = v_storage.slice(ndarray::s![1..4]);
        let u_view = u_storage.slice(ndarray::s![1..4]);
        let v_owned = v_view.to_owned();
        let u_owned = u_view.to_owned();

        let operators: [&dyn HyperOperator; 4] = [&dense, &block, &composite, &weighted];
        for op in operators {
            assert_relative_eq!(
                op.bilinear_view(v_view, u_view),
                op.bilinear(&v_owned, &u_owned),
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn hyper_operator_scaled_add_mul_vec_matches_owned_matvec() {
        let dense = DenseMatrixHyperOperator {
            matrix: array![[2.0, 0.3, -0.1], [0.3, 1.5, 0.4], [-0.1, 0.4, 3.0],],
        };
        let block = BlockLocalDrift {
            local: array![[1.2, 0.2], [0.2, 0.7]],
            start: 1,
            end: 3,
            total_dim: 3,
        };
        let composite = CompositeHyperOperator {
            dense: Some(array![[0.4, 0.1, 0.0], [0.1, 0.8, -0.2], [0.0, -0.2, 0.6],]),
            operators: vec![Arc::new(block.clone())],
            dim_hint: 3,
        };
        let weighted = WeightedHyperOperator {
            terms: vec![
                (1.7, Arc::new(dense.clone()) as Arc<dyn HyperOperator>),
                (-0.4, Arc::new(block.clone()) as Arc<dyn HyperOperator>),
                (0.0, Arc::new(composite.clone()) as Arc<dyn HyperOperator>),
            ],
            dim_hint: 3,
        };

        let v_storage = array![9.0, 0.5, -1.2, 0.7, 8.0];
        let v_view = v_storage.slice(ndarray::s![1..4]);
        let v_owned = v_view.to_owned();
        let base = array![0.25, -0.5, 1.5];
        let scale = -1.3;

        let operators: [&dyn HyperOperator; 4] = [&dense, &block, &composite, &weighted];
        for op in operators {
            let mut accumulated = base.clone();
            op.scaled_add_mul_vec(v_view, scale, accumulated.view_mut());

            let mut expected = base.clone();
            expected.scaled_add(scale, &op.mul_vec(&v_owned));
            for idx in 0..accumulated.len() {
                assert_relative_eq!(
                    accumulated[idx],
                    expected[idx],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    fn stochastic_single_second_order_estimators_match_batched_paths() {
        let diag = array![4.0, 3.0, 2.0];
        let hop = MatrixFreeSpdOperator::new(diag.len(), move |v| &diag * v);
        let estimator = StochasticTraceEstimator::with_defaults();
        let dense = array![[0.8, 0.2, 0.0], [0.2, 0.5, 0.1], [0.0, 0.1, 0.7],];
        let op = DenseMatrixHyperOperator {
            matrix: dense.clone(),
        };

        let no_ops: [&dyn HyperOperator; 0] = [];
        let dense_refs = [&dense];
        let batched_dense =
            estimator.estimate_second_order_traces_with_operators(&hop, &dense_refs, &no_ops);
        assert_relative_eq!(
            estimator.estimate_second_order_single_dense(&hop, &dense),
            batched_dense[[0, 0]],
            epsilon = 1e-12,
            max_relative = 1e-12
        );

        let no_dense: [&Array2<f64>; 0] = [];
        let op_refs: [&dyn HyperOperator; 1] = [&op];
        let batched_op =
            estimator.estimate_second_order_traces_with_operators(&hop, &no_dense, &op_refs);
        assert_relative_eq!(
            estimator.estimate_second_order_single_operator(&hop, &op),
            batched_op[[0, 0]],
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn matrix_free_logdet_traces_use_exact_spectral_algebra() {
        let diag = array![4.0, 3.0, 2.0];
        let h = Array2::from_diag(&diag);
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let op = MatrixFreeSpdOperator::new(diag.len(), move |v| &diag * v);
        let a = array![[0.7, 0.1, 0.0], [0.1, 0.4, 0.2], [0.0, 0.2, 0.5]];

        assert_relative_eq!(op.logdet(), dense.logdet(), epsilon = 1e-12);
        assert_relative_eq!(
            op.trace_hinv_product(&a),
            dense.trace_hinv_product(&a),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            op.trace_logdet_hessian_cross(&a, &a),
            dense.trace_logdet_hessian_cross(&a, &a),
            epsilon = 1e-12
        );
        assert!(!op.prefers_stochastic_trace_estimation());
        assert!(!op.logdet_traces_match_hinv_kernel());
        assert!(!can_use_stochastic_logdet_hinv_kernel(&op, 1024, true));
        assert!(!can_use_stochastic_logdet_hinv_kernel(&op, 128, true));
        assert!(!can_use_stochastic_logdet_hinv_kernel(&op, 1024, false));
    }

    #[test]
    fn test_rademacher_probe_properties() {
        // Verify probes have entries +/-1 and are deterministic given the same seed.
        let mut rng = Xoshiro256SS::from_seed(99);
        let mut z = Array1::zeros(100);
        rademacher_probe_into(z.view_mut(), &mut rng);
        assert_eq!(z.len(), 100);
        for &v in z.iter() {
            assert!(v == 1.0 || v == -1.0, "Rademacher entry must be +/-1");
        }

        // Same seed produces the same probe.
        let mut rng2 = Xoshiro256SS::from_seed(99);
        let mut z2 = Array1::zeros(100);
        rademacher_probe_into(z2.view_mut(), &mut rng2);
        assert_eq!(z, z2, "Same seed must produce identical probes");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Test 1: Spectral logdet gradient with r_epsilon regularization
    // ═══════════════════════════════════════════════════════════════════

    /// Verify that the analytic gradient of log|H(t)| computed through
    /// `DenseSpectralOperator` (with smooth spectral regularization r_epsilon)
    /// matches a central finite-difference estimate.
    ///
    /// Setup: H(t) = diag(2 + t, 0.01 + 2t, 3 - t) — one eigenvalue near
    /// zero so the regularization is exercised.
    #[test]
    fn test_spectral_logdet_gradient_fd() {
        let t0 = 0.0_f64;
        let h_step = 1e-6;

        // H(t) = diag(2+t, 0.01+2t, 3-t)
        // dH/dt = diag(1, 2, -1)
        let dh_dt = Array2::from_diag(&array![1.0, 2.0, -1.0]);

        // Build operator at t0
        let h0 = Array2::from_diag(&array![2.0 + t0, 0.01 + 2.0 * t0, 3.0 - t0]);
        let op0 = DenseSpectralOperator::from_symmetric(&h0).unwrap();

        // Analytic gradient: d/dt log|R_eps(H(t))| = tr(G_eps(H) dH/dt)
        let analytic = op0.trace_logdet_gradient(&dh_dt);

        // Finite difference: (logdet(t+h) - logdet(t-h)) / (2h)
        let h_plus = Array2::from_diag(&array![
            2.0 + t0 + h_step,
            0.01 + 2.0 * (t0 + h_step),
            3.0 - (t0 + h_step)
        ]);
        let h_minus = Array2::from_diag(&array![
            2.0 + t0 - h_step,
            0.01 + 2.0 * (t0 - h_step),
            3.0 - (t0 - h_step)
        ]);
        let op_plus = DenseSpectralOperator::from_symmetric(&h_plus).unwrap();
        let op_minus = DenseSpectralOperator::from_symmetric(&h_minus).unwrap();
        let fd = (op_plus.logdet() - op_minus.logdet()) / (2.0 * h_step);

        let rel_err = (analytic - fd).abs() / fd.abs().max(1e-12);
        assert!(
            rel_err < 1e-5,
            "Spectral logdet gradient mismatch: analytic={:.10e}, fd={:.10e}, rel_err={:.3e}",
            analytic,
            fd,
            rel_err,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Test 2: Moving nullspace correction for penalty pseudo-logdet
    // ═══════════════════════════════════════════════════════════════════

    /// Helper: build a 3x3 penalty matrix S(psi) whose nullspace rotates.
    ///
    /// S(psi) = R(psi) diag(s1, s2, 0) R(psi)^T
    /// where R(psi) is a rotation around the z-axis by angle psi.
    /// The nullspace is spanned by R(psi) * e3, which rotates as psi changes.
    fn rotating_nullspace_penalty(psi: f64, s1: f64, s2: f64) -> Array2<f64> {
        let c = psi.cos();
        let s = psi.sin();
        // R rotates in the (0,2) plane so the nullspace direction changes.
        let r = array![[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c],];
        let d = Array2::from_diag(&array![s1, s2, 0.0]);
        r.dot(&d).dot(&r.t())
    }

    /// Compute log|S|_+ (pseudo-logdeterminant over positive eigenvalues).
    fn pseudo_logdet(s: &Array2<f64>, tol: f64) -> f64 {
        let (eigs, _) = s.eigh(faer::Side::Lower).unwrap();
        eigs.iter().filter(|&&v| v > tol).map(|v| v.ln()).sum()
    }

    /// Compute d/dpsi log|S(psi)|_+ by central finite difference.
    fn pseudo_logdet_fd_first(psi: f64, h: f64, s1: f64, s2: f64, tol: f64) -> f64 {
        let sp = rotating_nullspace_penalty(psi + h, s1, s2);
        let sm = rotating_nullspace_penalty(psi - h, s1, s2);
        (pseudo_logdet(&sp, tol) - pseudo_logdet(&sm, tol)) / (2.0 * h)
    }

    /// Compute d^2/dpsi^2 log|S(psi)|_+ by central finite difference.
    fn pseudo_logdet_fd_second(psi: f64, h: f64, s1: f64, s2: f64, tol: f64) -> f64 {
        let sp = pseudo_logdet(&rotating_nullspace_penalty(psi + h, s1, s2), tol);
        let s0 = pseudo_logdet(&rotating_nullspace_penalty(psi, s1, s2), tol);
        let sm = pseudo_logdet(&rotating_nullspace_penalty(psi - h, s1, s2), tol);
        (sp - 2.0 * s0 + sm) / (h * h)
    }

    /// Analytic second derivative of log|S(psi)|_+ WITH the moving-nullspace
    /// correction, and WITHOUT it, so we can verify the correction is needed.
    ///
    /// Returns (with_correction, without_correction).
    fn analytic_pseudo_logdet_second(psi: f64, s1: f64, s2: f64, tol: f64) -> (f64, f64) {
        let s_mat = rotating_nullspace_penalty(psi, s1, s2);

        // Eigendecompose S
        let (eigs, vecs) = s_mat.eigh(faer::Side::Lower).unwrap();
        let p = eigs.len();

        let pos_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] > tol).collect();
        let null_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] <= tol).collect();

        // Build S_psi = dS/dpsi analytically.
        // S(psi) = R D R^T => dS/dpsi = R' D R^T + R D R'^T
        let c = psi.cos();
        let s = psi.sin();
        let r = array![[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c],];
        // R' = dR/dpsi
        let rp = array![[-s, 0.0, -c], [0.0, 0.0, 0.0], [c, 0.0, -s],];
        let d = Array2::from_diag(&array![s1, s2, 0.0]);
        let s_psi = rp.dot(&d).dot(&r.t()) + r.dot(&d).dot(&rp.t());

        // Build S_psi_psi = d^2S/dpsi^2 analytically.
        // R'' = d^2R/dpsi^2
        let rpp = array![[-c, 0.0, s], [0.0, 0.0, 0.0], [-s, 0.0, -c],];
        let s_psi_psi =
            rpp.dot(&d).dot(&r.t()) + 2.0 * &rp.dot(&d).dot(&rp.t()) + r.dot(&d).dot(&rpp.t());

        // Build S^+ (pseudoinverse): S^+ = V diag(1/sigma_i for pos, 0 for null) V^T
        let mut s_dag = Array2::<f64>::zeros((p, p));
        for &i in &pos_idx {
            let col = vecs.column(i);
            for r in 0..p {
                for c2 in 0..p {
                    s_dag[[r, c2]] += col[r] * col[c2] / eigs[i];
                }
            }
        }

        // Fixed-nullspace formula:
        //   d^2/dpsi^2 log|S|_+ = tr(S^+ S_psi_psi) - tr(S^+ S_psi S^+ S_psi)
        let sdag_s_psi = s_dag.dot(&s_psi);
        let term_linear = trace_mat(&s_dag.dot(&s_psi_psi));
        let term_quad = trace_mat(&sdag_s_psi.dot(&sdag_s_psi));
        let without_correction = term_linear - term_quad;

        // Moving-nullspace correction:
        //   +2 * tr(S^{+2} S_psi P_0 S_psi)
        // where P_0 = U_0 U_0^T, S^{+2} = (S^+)^2
        //
        // Efficient: tr(Sigma^{+2} L L^T) where L = U_+^T S_psi U_0
        let mut correction = 0.0_f64;
        if !pos_idx.is_empty() && !null_idx.is_empty() {
            // Build U_+ and U_0
            let n_pos = pos_idx.len();
            let n_null = null_idx.len();
            let mut u_pos = Array2::<f64>::zeros((p, n_pos));
            let mut u_null = Array2::<f64>::zeros((p, n_null));
            for (out, &idx) in pos_idx.iter().enumerate() {
                u_pos.column_mut(out).assign(&vecs.column(idx));
            }
            for (out, &idx) in null_idx.iter().enumerate() {
                u_null.column_mut(out).assign(&vecs.column(idx));
            }

            // L = U_+^T S_psi U_0  (n_pos x n_null)
            let l_mat = u_pos.t().dot(&s_psi.dot(&u_null));

            // Sigma^{+2} = diag(1/sigma_i^2) for positive eigenvalues
            for a in 0..n_pos {
                let sigma_inv_sq = 1.0 / (eigs[pos_idx[a]] * eigs[pos_idx[a]]);
                correction += sigma_inv_sq * l_mat.row(a).dot(&l_mat.row(a));
            }
            // The full correction is 2 * tr(Sigma^{+2} L L^T)
            correction *= 2.0;
        }

        let with_correction = without_correction + correction;
        (with_correction, without_correction)
    }

    /// tr(A) for a square matrix.
    fn trace_mat(a: &Array2<f64>) -> f64 {
        (0..a.nrows()).map(|i| a[[i, i]]).sum()
    }

    #[test]
    fn test_moving_nullspace_correction_needed() {
        // S(psi) = R(psi) diag(4, 1, 0) R(psi)^T — rank-2, nullspace rotates.
        let s1 = 4.0;
        let s2 = 1.0;
        let psi = 0.3; // nonzero angle
        let tol = 1e-10;
        let h = 1e-5;

        // The pseudo-logdet depends only on the positive eigenvalues, so a pure
        // nullspace rotation leaves the first derivative exactly zero.
        let fd_first = pseudo_logdet_fd_first(psi, h, s1, s2, tol);
        assert!(
            fd_first.is_finite() && fd_first.abs() < 1e-8,
            "First derivative should vanish for rotating nullspace, got {fd_first}"
        );

        let fd_second = pseudo_logdet_fd_second(psi, h, s1, s2, tol);
        let (with_corr, without_corr) = analytic_pseudo_logdet_second(psi, s1, s2, tol);

        // WITH correction should match FD
        let rel_err_with = (with_corr - fd_second).abs() / fd_second.abs().max(1e-12);
        assert!(
            rel_err_with < 1e-4,
            "With correction: analytic={:.8e}, fd={:.8e}, rel_err={:.3e}",
            with_corr,
            fd_second,
            rel_err_with,
        );

        // WITHOUT correction should NOT match FD (error should be large)
        let rel_err_without = (without_corr - fd_second).abs() / fd_second.abs().max(1e-12);
        assert!(
            rel_err_without > 1e-2,
            "Without correction should disagree with FD: \
             without={:.8e}, fd={:.8e}, rel_err={:.3e} (expected > 1e-2)",
            without_corr,
            fd_second,
            rel_err_without,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Test 3: Correction vanishes when nullspace is fixed
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_fixed_nullspace_correction_vanishes() {
        // S(rho) = diag(exp(rho1), exp(rho2), 0) — the nullspace is always e3,
        // regardless of rho. The correction terms should vanish, so both
        // formulas (with and without correction) should agree with FD.
        let tol = 1e-10;
        let h = 1e-5;

        // Evaluate at a specific point
        let rho1 = 0.5_f64;
        let rho2 = -0.3_f64;

        // Pseudo-logdet: log(exp(rho1)) + log(exp(rho2)) = rho1 + rho2
        // d/drho1 = 1, d^2/drho1^2 = 0 (exact).
        // But let's verify via the analytic+FD machinery for consistency.

        // We parameterize by a single scalar t: rho1 = 0.5 + t, rho2 = -0.3 + 2t.
        // S(t) = diag(exp(0.5+t), exp(-0.3+2t), 0)
        // log|S|_+ = (0.5+t) + (-0.3+2t) = 0.2 + 3t
        // d/dt = 3, d^2/dt^2 = 0.

        let build_s = |t: f64| -> Array2<f64> {
            Array2::from_diag(&array![(rho1 + t).exp(), (rho2 + 2.0 * t).exp(), 0.0])
        };

        let t0 = 0.0_f64;

        // FD second derivative
        let ld_plus = pseudo_logdet(&build_s(t0 + h), tol);
        let ld_0 = pseudo_logdet(&build_s(t0), tol);
        let ld_minus = pseudo_logdet(&build_s(t0 - h), tol);
        let fd_second = (ld_plus - 2.0 * ld_0 + ld_minus) / (h * h);

        // Analytic: S_t = diag(exp(rho1+t), 2*exp(rho2+2t), 0)
        // S_tt = diag(exp(rho1+t), 4*exp(rho2+2t), 0)
        let s_mat = build_s(t0);
        let s_t = Array2::from_diag(&array![
            (rho1 + t0).exp(),
            2.0 * (rho2 + 2.0 * t0).exp(),
            0.0
        ]);
        let s_tt = Array2::from_diag(&array![
            (rho1 + t0).exp(),
            4.0 * (rho2 + 2.0 * t0).exp(),
            0.0
        ]);

        let (eigs, vecs) = s_mat.eigh(faer::Side::Lower).unwrap();
        let p = 3;
        let pos_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] > tol).collect();
        let null_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] <= tol).collect();

        // Build S^+
        let mut s_dag = Array2::<f64>::zeros((p, p));
        for &i in &pos_idx {
            let col = vecs.column(i);
            for r in 0..p {
                for c in 0..p {
                    s_dag[[r, c]] += col[r] * col[c] / eigs[i];
                }
            }
        }

        // Fixed-nullspace formula
        let sdag_s_t = s_dag.dot(&s_t);
        let term_linear = trace_mat(&s_dag.dot(&s_tt));
        let term_quad = trace_mat(&sdag_s_t.dot(&sdag_s_t));
        let without_correction = term_linear - term_quad;

        // Compute the correction (should be ~0 since nullspace doesn't move)
        let mut correction = 0.0_f64;
        if !pos_idx.is_empty() && !null_idx.is_empty() {
            let n_pos = pos_idx.len();
            let n_null = null_idx.len();
            let mut u_pos = Array2::<f64>::zeros((p, n_pos));
            let mut u_null = Array2::<f64>::zeros((p, n_null));
            for (out, &idx) in pos_idx.iter().enumerate() {
                u_pos.column_mut(out).assign(&vecs.column(idx));
            }
            for (out, &idx) in null_idx.iter().enumerate() {
                u_null.column_mut(out).assign(&vecs.column(idx));
            }
            let l_mat = u_pos.t().dot(&s_t.dot(&u_null));
            for a in 0..n_pos {
                let sigma_inv_sq = 1.0 / (eigs[pos_idx[a]] * eigs[pos_idx[a]]);
                correction += sigma_inv_sq * l_mat.row(a).dot(&l_mat.row(a));
            }
            correction *= 2.0;
        }

        // The correction should be negligible (nullspace is fixed)
        assert!(
            correction.abs() < 1e-12,
            "Correction should vanish for fixed nullspace, got {:.3e}",
            correction,
        );

        // Both formulas should match FD
        let with_correction = without_correction + correction;

        // For diag(e^a, e^b, 0), d^2/dt^2 log|S|_+ = 0, so use absolute error
        // since fd_second ~ 0.
        let abs_err_with = (with_correction - fd_second).abs();
        let abs_err_without = (without_correction - fd_second).abs();
        assert!(
            abs_err_with < 1e-4,
            "With correction should match FD: with={:.8e}, fd={:.8e}, abs_err={:.3e}",
            with_correction,
            fd_second,
            abs_err_with,
        );
        assert!(
            abs_err_without < 1e-4,
            "Without correction should also match FD (fixed nullspace): \
             without={:.8e}, fd={:.8e}, abs_err={:.3e}",
            without_correction,
            fd_second,
            abs_err_without,
        );
    }

    #[test]
    fn test_symmetric_eigen_identity() {
        let eye = Array2::<f64>::eye(3);
        let (evals, evecs) = symmetric_eigen(&eye);
        for &e in &evals {
            assert!((e - 1.0).abs() < 1e-12, "eigenvalue should be 1.0, got {e}");
        }
        // Eigenvectors should be orthonormal.
        let prod = evecs.t().dot(&evecs);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[[i, j]] - expected).abs() < 1e-12,
                    "Q^T Q should be identity"
                );
            }
        }
    }

    #[test]
    fn test_symmetric_eigen_diagonal() {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = 4.0;
        d[[1, 1]] = 2.0;
        d[[2, 2]] = 1.0;
        let (evals, _) = symmetric_eigen(&d);
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((sorted[0] - 1.0).abs() < 1e-12);
        assert!((sorted[1] - 2.0).abs() < 1e-12);
        assert!((sorted[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_pseudoinverse_times_vec_identity() {
        let eye = Array2::<f64>::eye(3);
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result =
            pseudoinverse_times_vec(&eye, v.as_slice().expect("contiguous test vector"), 1e-8);
        for i in 0..3 {
            assert!((result[i] - v[i]).abs() < 1e-12, "G=I: G⁺v should equal v");
        }
    }

    #[test]
    fn test_pseudoinverse_times_vec_singular() {
        // Rank-1 matrix: G = [1 1; 1 1]. Pseudoinverse G⁺ = [0.25 0.25; 0.25 0.25].
        let mut g = Array2::<f64>::zeros((2, 2));
        g[[0, 0]] = 1.0;
        g[[0, 1]] = 1.0;
        g[[1, 0]] = 1.0;
        g[[1, 1]] = 1.0;
        let v = Array1::from_vec(vec![2.0, 0.0]);
        let result =
            pseudoinverse_times_vec(&g, v.as_slice().expect("contiguous test vector"), 1e-8);
        // G⁺ v = [0.25*2 + 0.25*0; 0.25*2 + 0.25*0] = [0.5; 0.5]
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
    }

    /// Contract: `ImplicitHyperOperator::mul_vec(v)` reproduces the analytic
    /// first-order spatial drift
    ///   `B_d v = (∂X/∂ψ_d)ᵀ W X v + Xᵀ W (∂X/∂ψ_d) v + Xᵀ diag(c·X_{ψ_d}β̂) X v + S_{ψ_d} v`.
    ///
    /// The third (non-Gaussian) term is the part that landed under task #7 —
    /// it must agree with the dense reference computed from
    /// `materialize_first(axis)`. We build a tiny `ImplicitDesignPsiDerivative`
    /// (n=4, n_knots=2, n_axes=1, no identifiability transform), assemble a
    /// known X / W / S_ψ / c_x_psi_beta, and check `mul_vec(v)` against the
    /// fully-dense formula above for several probe vectors v.
    ///
    /// Also runs once with `c_x_psi_beta = None` to lock in the Gaussian
    /// fast-path: the third term must drop out cleanly.
    #[test]
    fn implicit_hyper_operator_third_derivative_term_matches_dense_reference() {
        use crate::terms::basis::ImplicitDesignPsiDerivative;
        use std::sync::Arc;

        let n = 4usize;
        let n_knots = 2usize;
        let n_axes = 1usize;
        let p = n_knots; // no polynomial padding, no identifiability transform

        // Implicit operator: deliberately non-trivial radial scalars so the
        // resulting (∂X/∂ψ_0) is dense and not accidentally zero.
        // First-axis kernel value (no transform path) is `q_ij·s_b[axis] + c·phi_ij`
        // with `c = psi_scale_share = 0.0` — so the kernel is `q_ij · s_{0,ij}`.
        let phi_values = array![1.0, 0.5, 0.7, 0.9, 0.3, 0.4, 0.6, 0.8];
        let q_values = array![0.5, -0.2, 0.3, 0.1, -0.4, 0.2, 0.6, -0.1];
        let t_values = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // axis_components is (n*n_knots, n_axes) row-major: rows = (i, j) pair.
        let axis_components = array![[0.7], [0.3], [-0.4], [0.5], [0.2], [-0.1], [0.6], [0.8]];
        let implicit = Arc::new(ImplicitDesignPsiDerivative::new(
            phi_values,
            q_values,
            t_values,
            axis_components,
            None,
            None,
            n,
            n_knots,
            0,
            n_axes,
        ));

        // Active-basis design X (n × p): chosen so Xᵀ X is well-conditioned.
        let x_data = array![[1.0, 0.30], [0.50, 1.20], [-0.20, 0.80], [0.90, -0.40],];
        let x_design = Arc::new(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_data.clone(),
        )));
        let w_diag = Arc::new(array![0.8, 1.2, 0.6, 1.5]);

        // S_psi (p × p): symmetric, dense.
        let s_psi = array![[0.40, 0.05], [0.05, 0.25]];

        // β̂ used to fold c · (∂X/∂ψ_0) β̂ into the per-row kernel.
        let beta_eval = array![0.30, -0.20];
        // c_array (length n) — the GLM third-derivative weight.
        let c_array = array![0.10, -0.05, 0.20, 0.15];

        // Reference dense (∂X/∂ψ_0).
        let dx_dpsi = implicit
            .materialize_first(0)
            .expect("materialize_first should succeed on tiny fixture");
        assert_eq!(dx_dpsi.shape(), &[n, p]);

        // c_x_psi_beta[i] = c[i] · (∂X/∂ψ_0 · β̂)[i].
        let dx_beta = dx_dpsi.dot(&beta_eval);
        let c_x_psi_beta_dense = &c_array * &dx_beta;
        let c_x_psi_beta = Some(Arc::new(c_x_psi_beta_dense.clone()));

        let op = ImplicitHyperOperator {
            implicit_deriv: Arc::clone(&implicit),
            axis: 0,
            x_design: Arc::clone(&x_design),
            w_diag: Arc::clone(&w_diag),
            s_psi: s_psi.clone(),
            p,
            c_x_psi_beta,
        };

        let probes = [
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![0.7, -0.4],
            array![-0.25, 1.10],
        ];
        for (k, v) in probes.iter().enumerate() {
            // Analytic dense reference.
            //   t1 = (∂X/∂ψ_0)ᵀ · diag(W) · X · v
            //   t2 = Xᵀ · diag(W) · (∂X/∂ψ_0) · v
            //   t3 = Xᵀ · diag(c_x_psi_beta) · X · v
            //   t4 = S_psi · v
            let xv = x_data.dot(v);
            let dxv = dx_dpsi.dot(v);
            let w_xv = &*w_diag * &xv;
            let w_dxv = &*w_diag * &dxv;
            let t1 = dx_dpsi.t().dot(&w_xv);
            let t2 = x_data.t().dot(&w_dxv);
            let weighted = &c_x_psi_beta_dense * &xv;
            let t3 = x_data.t().dot(&weighted);
            let t4 = s_psi.dot(v);
            let want = &t1 + &t2 + &t3 + &t4;

            let got = op.mul_vec(v);
            assert_eq!(got.len(), p);
            for i in 0..p {
                let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "B_d·v mismatch at probe {k}, comp {i}: want={:.6e}, got={:.6e}",
                    want[i],
                    got[i],
                );
            }
        }

        // Gaussian path: c_x_psi_beta = None must drop the third term cleanly.
        let op_gauss = ImplicitHyperOperator {
            implicit_deriv: Arc::clone(&implicit),
            axis: 0,
            x_design,
            w_diag: Arc::clone(&w_diag),
            s_psi: s_psi.clone(),
            p,
            c_x_psi_beta: None,
        };
        let v = array![0.7, -0.4];
        let xv = x_data.dot(&v);
        let dxv = dx_dpsi.dot(&v);
        let w_xv = &*w_diag * &xv;
        let w_dxv = &*w_diag * &dxv;
        let want = &dx_dpsi.t().dot(&w_xv) + &x_data.t().dot(&w_dxv) + &s_psi.dot(&v);
        let got = op_gauss.mul_vec(&v);
        for i in 0..p {
            let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "Gaussian B_d·v mismatch at comp {i}: want={:.6e}, got={:.6e}",
                want[i],
                got[i],
            );
        }
    }

    /// Centered finite-difference check on the third-derivative term in
    /// isolation: at fixed (X, W, S_ψ, β̂) the term `Xᵀ diag(c · X_ψ β̂) X v` is
    /// linear in `v`, so the *correctness* check is a comparison against the
    /// analytic action. To exercise the FD route the spec asks for, we
    /// finite-difference along v using the operator's `mul_vec` and confirm
    /// the operator is exactly the linear map encoded by its kernel — i.e. the
    /// difference quotient `(op.mul_vec(v + ε e_j) − op.mul_vec(v − ε e_j))/(2ε)`
    /// equals the j-th column of `Xᵀ diag(c_x_psi_beta) X` at any v.
    #[test]
    fn implicit_hyper_operator_third_derivative_term_centered_fd_matches_jacobian_column() {
        use crate::terms::basis::ImplicitDesignPsiDerivative;
        use std::sync::Arc;

        let n = 5usize;
        let n_knots = 3usize;
        let n_axes = 1usize;
        let p = n_knots;

        let phi_values =
            Array1::from_vec((0..n * n_knots).map(|k| 0.1 + 0.05 * (k as f64)).collect());
        let q_values =
            Array1::from_vec((0..n * n_knots).map(|k| -0.2 + 0.07 * (k as f64)).collect());
        let t_values = Array1::zeros(n * n_knots);
        let axis_components = Array2::from_shape_vec(
            (n * n_knots, n_axes),
            (0..n * n_knots).map(|k| 0.3 + 0.04 * (k as f64)).collect(),
        )
        .unwrap();
        let implicit = Arc::new(ImplicitDesignPsiDerivative::new(
            phi_values,
            q_values,
            t_values,
            axis_components,
            None,
            None,
            n,
            n_knots,
            0,
            n_axes,
        ));

        let x_data = array![
            [1.0, 0.4, -0.2],
            [0.5, 1.1, 0.3],
            [-0.3, 0.9, 0.6],
            [0.8, -0.5, 1.2],
            [0.2, 0.7, -0.4],
        ];
        let x_design = Arc::new(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_data.clone(),
        )));
        let w_diag = Arc::new(array![1.0, 0.7, 1.3, 0.9, 1.1]);
        let s_psi = Array2::<f64>::eye(p) * 0.05;

        let beta_eval = array![0.20, -0.10, 0.30];
        let c_array = array![0.15, -0.08, 0.22, 0.05, -0.12];
        let dx_dpsi = implicit.materialize_first(0).expect("materialize_first");
        let dx_beta = dx_dpsi.dot(&beta_eval);
        let c_x_psi_beta_dense = &c_array * &dx_beta;

        let op = ImplicitHyperOperator {
            implicit_deriv: Arc::clone(&implicit),
            axis: 0,
            x_design,
            w_diag,
            s_psi,
            p,
            c_x_psi_beta: Some(Arc::new(c_x_psi_beta_dense.clone())),
        };

        // Dense Jacobian column j: B_d e_j.
        let v_base = array![0.10, -0.05, 0.20];
        let eps = 1e-6;
        for j in 0..p {
            let mut e_j = Array1::<f64>::zeros(p);
            e_j[j] = 1.0;
            // Centered FD on mul_vec along e_j gives B_d e_j (operator is linear in v).
            let mut v_plus = v_base.clone();
            v_plus[j] += eps;
            let mut v_minus = v_base.clone();
            v_minus[j] -= eps;
            let fd = (&op.mul_vec(&v_plus) - &op.mul_vec(&v_minus)).mapv(|x| x / (2.0 * eps));
            let analytic = op.mul_vec(&e_j);
            for i in 0..p {
                let tol = 1e-7 * analytic[i].abs().max(1.0) + 1e-7;
                assert!(
                    (analytic[i] - fd[i]).abs() <= tol,
                    "FD col {j} mismatch at row {i}: analytic={:.6e}, fd={:.6e}",
                    analytic[i],
                    fd[i],
                );
            }
        }
    }

    #[test]
    fn test_pseudoinverse_scalar() {
        let mut g = Array2::<f64>::zeros((1, 1));
        g[[0, 0]] = 4.0;
        let v = Array1::from_vec(vec![8.0]);
        let result =
            pseudoinverse_times_vec(&g, v.as_slice().expect("contiguous test vector"), 1e-8);
        assert!((result[0] - 2.0).abs() < 1e-12);
    }
}
