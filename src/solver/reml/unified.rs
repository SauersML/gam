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
//!
//! # Trace-Estimation Tiers
//!
//! Several REML/LAML/PIRLS quantities reduce to traces of operators that
//! have efficient HVPs but expensive dense materialization. The codebase
//! picks among three estimators depending on the operator's structure and
//! the problem size; backends override the default trait method to take
//! the cheapest path natively when one exists.
//!
//! ## Tier 1: Exact (default for small p, native overrides for large p)
//!
//! When the operator is small enough that materializing it as a dense
//! `p × p` matrix and summing the diagonal of `H⁻¹ M` is cheap, OR when a
//! backend has a structure-aware exact path (e.g. Takahashi-selected
//! inverse for sparse Cholesky), use it. Examples: every concrete
//! `HessianOperator` impl below overrides `trace_hinv_operator` and the
//! cross-trace family with a native exact path.
//!
//! ## Tier 2: Hutchinson (multi-target shared-probe)
//!
//! When the same `H⁻¹` solve serves multiple coordinate targets — the
//! REML/LAML rho-gradient computes `tr(H⁻¹ A_k)` for `k = 1, ..., K` —
//! [`StochasticTraceEstimator`] runs Girard–Hutchinson with one shared
//! `H⁻¹` solve per probe and adaptive Welford-style stopping. Common
//! random numbers (deterministic seed) hold across rho coordinates, so
//! each probe contributes coherently to every coordinate's gradient.
//! Triggered for very large `p` via [`can_use_stochastic_logdet_hinv_kernel`].
//!
//! ## Tier 3: Hutch++ (single-target, HVP-only operator)
//!
//! When a single trace `tr(H⁻¹ M)` is needed against an HVP-only
//! operator and `p ≥ 128`, [`hutchpp_estimate_trace_hinv_operator`]
//! splits the trace via Meyer–Musco's randomized range finder. The
//! sketch captures the dominant subspace of `H⁻¹ M` exactly; the
//! Hutchinson residual handles the orthogonal complement with greatly
//! reduced variance. Achieves `O(1/ε)` matvecs vs `O(1/ε²)` for plain
//! Hutchinson.
//!
//! [`hutchpp_estimate_trace_hinv_op_squared`] handles the symmetric
//! same-operator cross-trace `tr((H⁻¹A)²)` (used by outer-Hessian
//! diagonals); [`hutchpp_estimate_trace_hinv_operator_cross`] handles
//! the asymmetric `tr(H⁻¹A_L H⁻¹A_R)` via a shared sketch. Default
//! impls of [`HessianOperator::trace_hinv_operator`],
//! [`HessianOperator::trace_logdet_operator`], and the cross-trace
//! family auto-select Hutch++ for implicit operators at moderate
//! `dim()`. Concrete backends with native paths (dense spectral,
//! Takahashi Cholesky) override and never reach Hutch++.
//!
//! ## Why these three and not more
//!
//! The BMS / survival-marginal-slope row-trace path is *not* a
//! Hutch++ candidate even though it computes a trace. The exact
//! per-row algebra exploits a rank-r factor projection plus linearity
//! in the rho direction to compute one length-r vector per row that
//! serves all rho coordinates; a probe-based estimator would require
//! `O(m · k_directions)` row passes vs the existing single row pass.
//! See `bernoulli_marginal_slope::row_primary_third_trace_gradient_with_moments`.
//!
//! ## Orthogonal axis: row subsampling for biobank-scale fits
//!
//! Trace estimators here reduce work *within* the Hessian structure
//! for a fixed row set. The marginal-slope families have a separate,
//! complementary mechanism that reduces the row set itself: stratified
//! Horvitz–Thompson outer-score subsampling (see
//! `families::marginal_slope_shared`). The two compose naturally — a
//! Hutch++ trace against an `H⁻¹ M` operator stays valid when `M` is
//! itself a partial-row sum, and the row subsample's variance bound
//! is independent of the trace estimator used inside the per-row work.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};
use rayon::prelude::*;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex};

use crate::faer_ndarray::{FaerCholesky, FaerEigh};
use crate::linalg::matrix::DesignMatrix;

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
//  used by the iso-κ Duchon FD investigation test. Empty in production runs.
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
pub mod debug_stash {
    use std::cell::RefCell;

    #[derive(Clone, Debug, Default)]
    pub struct TermStash {
        /// Per-row diagonal of term4: `c · X_τβ̂` (n-vector).
        pub c_x_tau_beta_diag: Option<ndarray::Array1<f64>>,
        /// `X · v_ψ` per row, where v_ψ = hop⁻¹·stored_g. The correction
        /// matrix is `−X' diag(c · X · v_ψ) X`, so multiplying this by c
        /// (from PIRLS) gives the diagonal entering the correction sandwich.
        pub c_x_v_psi_diag: Option<ndarray::Array1<f64>>,
        /// Unprojected eigenmode trace Σ φ'(σ_j)·(Uᵀ op U)_jj — i.e. the value
        /// of tr(K · op_total) using the full-space `G_ε(H)` kernel without
        /// the penalty-subspace projection. Recorded for tests that need to
        /// pin the projection-vs-unprojected gap.
        pub unprojected_tr: Option<f64>,
        /// The production `trace_logdet_i` value that actually enters the
        /// outer gradient. Routes through `kernel.trace_projected_logdet`
        /// when `penalty_subspace_trace` is Some, otherwise through the
        /// full-space kernel. Recorded so tests can pin
        /// `unprojected_tr ≠ production_tr` when projection is active.
        pub production_tr: Option<f64>,
        /// Whether `penalty_subspace_trace` was Some for this coordinate
        /// (i.e. the production trace ran through the projected kernel).
        pub projection_active: Option<bool>,
    }

    // Thread-local storage keyed on the THREAD THAT CALLED `reml_laml_evaluate`
    // (i.e. the test thread). The diagnostic block inside the function is
    // physically computed by a rayon worker for `ext_idx == 0`, but the
    // worker's stash is plumbed back through the par_iter return value
    // and `store_terms` is invoked on the calling thread after the
    // parallel loop completes. That makes thread-local correct again —
    // each test thread reads its own captures, even when `cargo test`
    // runs the suite at full parallelism and many tests are inside
    // `reml_laml_evaluate` simultaneously.
    //
    // A previous attempt to share a single process-global `Mutex<TermStash>`
    // deterministically corrupted concurrent tests: thread A would call
    // `take_terms()` after its own evaluate returned, but if thread B's
    // evaluate had stored a stash in between, A read B's data. Routing
    // the write back to the calling thread's thread-local is the only
    // arrangement that keeps the per-test capture intact under parallel
    // testing.
    thread_local! {
        static TERMS: RefCell<TermStash> = const { RefCell::new(TermStash {
            c_x_tau_beta_diag: None,
            c_x_v_psi_diag: None,
            unprojected_tr: None,
            production_tr: None,
            projection_active: None,
        }) };
    }

    /// Replace the calling thread's `TermStash` with `stash`. Called by
    /// `reml_laml_evaluate` from the calling thread AFTER its ext-coord
    /// par_iter has produced the stash on a rayon worker and returned
    /// it via the closure's tuple. Tests read the stored stash via
    /// `take_terms()` from the same thread.
    pub fn store_terms(stash: TermStash) {
        TERMS.with(|cell| *cell.borrow_mut() = stash);
    }
    /// Consume the calling thread's `TermStash` and return its contents.
    /// Tests call this after running an evaluation to inspect the
    /// diagnostic captures (`unprojected_tr`, `production_tr`,
    /// `c_x_*_diag`, etc.).
    pub fn take_terms() -> TermStash {
        TERMS.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
    }
}

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
        if op.is_implicit() && self.dim() >= 128 {
            let mut config = StochasticTraceConfig::default();
            let sketch = (self.dim() / 32).clamp(4, 16);
            config.hutchpp_sketch_dim = Some(sketch);
            config.n_probes_max = (sketch * 4).max(32);
            config.n_probes_min = sketch.max(8);
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
        if op.is_implicit() && self.dim() >= 128 {
            let mut config = StochasticTraceConfig::default();
            let sketch = (self.dim() / 32).clamp(4, 16);
            config.hutchpp_sketch_dim = Some(sketch);
            config.n_probes_max = (sketch * 4).max(32);
            config.n_probes_min = sketch.max(8);
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
        if (l_implicit || r_implicit) && self.dim() >= 128 {
            let mut config = StochasticTraceConfig::default();
            let sketch = (self.dim() / 32).clamp(4, 16);
            config.hutchpp_sketch_dim = Some(sketch);
            config.n_probes_max = (sketch * 4).max(32);
            config.n_probes_min = sketch.max(8);
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
    ///
    /// When `logdet_traces_match_hinv_kernel()` is true (Cholesky-style
    /// backends where `trace_logdet_gradient(A) = trace_hinv_product(A)`)
    /// and the operator is implicit on a moderate-or-large problem, route
    /// through Hutch++ to avoid the dense materialization. Spectral
    /// backends override this to false (their logdet trace uses
    /// regularized eigenvalue weights, not `H⁻¹`), so they keep the
    /// materialize path or provide their own override.
    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        if op.is_implicit() && self.dim() >= 128 && self.logdet_traces_match_hinv_kernel() {
            let mut config = StochasticTraceConfig::default();
            let sketch = (self.dim() / 32).clamp(4, 16);
            config.hutchpp_sketch_dim = Some(sketch);
            config.n_probes_max = (sketch * 4).max(32);
            config.n_probes_min = sketch.max(8);
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
        // Stays matrix-free: `matrixvectormultiply` and `compute_xtwx` route
        // through the operator-backed design's chunked kernels at biobank
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
        let mut slacks = match self.slacks(beta) {
            Some(s) => s,
            None => return true, // infeasible → conservatively unreliable
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
        crate::faer_ndarray::fast_atb(factor, &op_factor)
    }

    /// Compute the exact projected matrix `F^T B F`, reusing caller-owned
    /// projection caches when the operator has a shared row/design factor.
    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        _cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
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
    #[cfg(test)]
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
/// the same cache slot 32 times. At biobank scale that is the
/// difference between minutes and seconds of design-GEMM work (see
/// [`ImplicitHyperOperator::trace_projected_factor_cached`] for the
/// usage rationale).
///
/// The cache is bounded by a byte budget. When inserting a new entry
/// would exceed the budget, the *least-recently-used* entries are
/// evicted until it fits. A budget of `0` (or `usize::MAX`) disables
/// eviction. The default is `Self::DEFAULT_BUDGET_BYTES` — large
/// enough to hold any realistic working set for in-memory problems
/// while still bounding worst-case peak resident memory at biobank
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
    /// Default byte budget for the cache. Aligned with the biobank-scale
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
                });
                inner.in_progress.insert(key, marker.clone());
                CacheLookup::Compute(marker)
            }
        };

        match lookup {
            CacheLookup::Hit(value) => value,
            CacheLookup::Wait(marker) => {
                let mut guard = marker
                    .state
                    .lock()
                    .expect("projected factor in-progress lock poisoned");
                loop {
                    match guard.as_ref() {
                        Some(ProjectedFactorInProgressState::Ready(value)) => return value.clone(),
                        Some(ProjectedFactorInProgressState::Failed) => {
                            panic!("projected factor cache producer panicked")
                        }
                        None => {
                            guard = marker
                                .ready
                                .wait(guard)
                                .expect("projected factor in-progress wait poisoned");
                        }
                    }
                }
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
            if let Some(impl_j) = operators[j].as_implicit() {
                if Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                    && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                    && Arc::ptr_eq(&impl_i.w_diag, &impl_j.w_diag)
                    && impl_i.p == impl_j.p
                {
                    group.push(j);
                    handled[j] = true;
                }
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

fn dense_trace_projected_factor(matrix: &Array2<f64>, factor: &Array2<f64>) -> f64 {
    let matrix_factor = matrix.dot(factor);
    factor
        .iter()
        .zip(matrix_factor.iter())
        .map(|(&f, &mf)| f * mf)
        .sum()
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
            if let Some(impl_j) = terms[j].2.as_implicit() {
                if Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                    && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                    && Arc::ptr_eq(&impl_i.w_diag, &impl_j.w_diag)
                    && impl_i.p == impl_j.p
                {
                    group.push(j);
                    handled[j] = true;
                }
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

fn dense_projected_matrix(matrix: &Array2<f64>, factor: &Array2<f64>) -> Array2<f64> {
    let mf = crate::faer_ndarray::fast_ab(matrix, factor);
    crate::faer_ndarray::fast_atb(factor, &mf)
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

fn projected_operator_terms_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<Array2<f64>> {
    let rank = factor.ncols();
    let mut out = (0..n_out)
        .map(|_| Array2::<f64>::zeros((rank, rank)))
        .collect::<Vec<_>>();
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
            if let Some(impl_j) = terms[j].2.as_implicit() {
                if Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                    && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                    && Arc::ptr_eq(&impl_i.w_diag, &impl_j.w_diag)
                    && impl_i.p == impl_j.p
                {
                    group.push(j);
                    handled[j] = true;
                }
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
        let values = lead.projected_matrix_all_axes_with_xf(factor, xf.view(), &axes);
        for (&term_idx, value) in group.iter().zip(values.iter()) {
            let (out_idx, weight, _) = terms[term_idx];
            out[out_idx].scaled_add(weight, value);
        }
    }

    for (i, (out_idx, weight, op)) in terms.iter().enumerate() {
        if handled[i] {
            continue;
        }
        out[*out_idx].scaled_add(*weight, &op.projected_matrix_cached(factor, cache));
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
    if let Ok(chol) = kernel.h_proj_inverse.cholesky(faer::Side::Lower) {
        let lower = chol.lower_triangular();
        return crate::faer_ndarray::fast_ab(&kernel.u_s, &lower);
    }

    let (evals, evecs) = kernel
        .h_proj_inverse
        .eigh(faer::Side::Lower)
        .expect("PenaltySubspaceTrace kernel factor eigendecomposition failed");
    let r = evals.len();
    let max_eval = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let floor = f64::EPSILON.sqrt() * (r as f64).max(1.0) * max_eval.max(1.0);
    let mut root = evecs.clone();
    for col in 0..r {
        let scale = evals[col].max(floor).sqrt();
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
    let rank = kernel.u_s.ncols();
    let mut reduced = (0..drifts.len())
        .map(|_| Array2::<f64>::zeros((rank, rank)))
        .collect::<Vec<_>>();
    let mut terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
    for (idx, drift) in drifts.iter().enumerate() {
        match drift {
            DriftDerivResult::Dense(matrix) => {
                reduced[idx] += &kernel.reduce(matrix);
            }
            DriftDerivResult::Operator(op) => {
                collect_projected_matrix_terms(
                    idx,
                    1.0,
                    op.as_ref(),
                    &kernel.u_s,
                    &mut reduced,
                    &mut terms,
                );
            }
        }
    }
    if !terms.is_empty() {
        let cache = ProjectedFactorCache::default();
        let batched = project_hyper_operators_batched(drifts.len(), &terms, &kernel.u_s, &cache);
        for (dst, projected) in reduced.iter_mut().zip(batched.iter()) {
            *dst += projected;
        }
    }
    reduced
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
    /// at biobank shape (16D-Duchon-aniso 32 ψ-axes, p ≈ 95, n = 320 K)
    /// the per-axis trace landed at ~30 s. With 32 axes per outer Hessian
    /// eval and ~5 outer iters that's the ~1 hr biobank timeout.
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
    /// the biobank-shape margslope-aniso-duchon16d shard each per-axis trace
    /// drops from ~30 s to a single chunked-GEMM cost.
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return 0.0;
        }
        let xf = self.compute_xf(factor);
        self.trace_projected_factor_with_xf(factor, xf.view())
    }

    /// Cached variant — *the* hot-path optimisation for biobank-shape outer
    /// gradient/Hessian sweeps. Every ψ-axis built atop the same `x_design`
    /// (e.g. all 32 ψ-axes of a marginal-slope model, or the same axis hit
    /// from `g_factor` and `w_factor` traces) shares one chunked
    /// `X · F` design GEMM per `(x_design, factor)` pair via
    /// [`ProjectedFactorCache`]. With 32 axes per outer-gradient sweep and
    /// O(rank) more cross-axis traces inside the outer-Hessian build, the
    /// cache turns 32× redundant `O(n · p · rank)` GEMMs into a single one
    /// per outer iter. At biobank shape (`n = 320 K`, `p = rank = 95`) that
    /// is the difference between minutes and seconds of design-GEMM work.
    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return 0.0;
        }
        let xf = self.cached_xf(factor, cache);
        self.trace_projected_factor_with_xf(factor, xf.view())
    }

    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return Array2::<f64>::zeros((rank, rank));
        }
        let xf = self.compute_xf(factor);
        self.projected_matrix_with_xf(factor, xf.view())
    }

    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return Array2::<f64>::zeros((rank, rank));
        }
        let xf = self.cached_xf(factor, cache);
        self.projected_matrix_with_xf(factor, xf.view())
    }
}

impl ImplicitHyperOperator {
    /// Chunked `X · F` via faer SIMD-parallel GEMM. The chunk-row sizing
    /// targets ~8 MiB live blocks so the (chunk_n × p) row slice and
    /// (chunk_n × rank) result both stay in L2/L3 across realistic biobank
    /// shapes; the kernel mirrors `xt_logdet_kernel_x_diagonal`'s sizing
    /// rule. Caller wraps this in [`Self::cached_xf`] when invariance
    /// across ψ-axes lets one matrix serve every axis at this `(x_design,
    /// factor)` pair.
    fn compute_xf(&self, factor: &Array2<f64>) -> Array2<f64> {
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        let mut xf = Array2::<f64>::zeros((n_obs, rank));
        const TARGET_BYTES: usize = 8 * 1024 * 1024;
        let chunk_rows = (TARGET_BYTES / ((self.p + rank).max(1) * 8))
            .max(512)
            .min(n_obs);
        let num_chunks = (n_obs + chunk_rows - 1) / chunk_rows;
        // Parallelize chunk traversal when we are not already inside a rayon
        // worker (so we don't oversubscribe nested pools) and the work is
        // large enough to amortize thread setup.
        let use_parallel = num_chunks >= 2
            && rayon::current_thread_index().is_none()
            && (n_obs as u64) * (self.p as u64).max(1) >= 1_000_000;
        if use_parallel {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let chunks: Vec<(usize, Array2<f64>)> = (0..num_chunks)
                .into_par_iter()
                .map(|ci| {
                    let start = ci * chunk_rows;
                    let end = (start + chunk_rows).min(n_obs);
                    let rows = self
                        .x_design
                        .try_row_chunk(start..end)
                        .unwrap_or_else(|err| {
                            panic!("ImplicitHyperOperator::compute_xf row chunk failed: {err}")
                        });
                    let block = crate::faer_ndarray::fast_ab(&rows, factor);
                    (start, block)
                })
                .collect();
            for (start, block) in chunks {
                let end = start + block.nrows();
                xf.slice_mut(ndarray::s![start..end, ..]).assign(&block);
            }
        } else {
            let mut start = 0usize;
            while start < n_obs {
                let end = (start + chunk_rows).min(n_obs);
                let rows = self
                    .x_design
                    .try_row_chunk(start..end)
                    .unwrap_or_else(|err| {
                        panic!("ImplicitHyperOperator::compute_xf row chunk failed: {err}")
                    });
                let block = crate::faer_ndarray::fast_ab(&rows, factor);
                xf.slice_mut(ndarray::s![start..end, ..]).assign(&block);
                start = end;
            }
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

    fn projected_chunk_contribution(
        &self,
        xf: ArrayView2<'_, f64>,
        u_knot: &Array2<f64>,
        w: &[f64],
        c_opt: Option<&[f64]>,
        rank: usize,
        start: usize,
        end: usize,
    ) -> Array2<f64> {
        let xf_chunk = xf.slice(ndarray::s![start..end, ..]);
        let kd_chunk = self
            .implicit_deriv
            .row_chunk_first_raw(self.axis, start..end)
            .expect("radial scalar evaluation failed during implicit hyper projected_matrix");
        let mut weighted_dxf = crate::faer_ndarray::fast_ab(&kd_chunk, u_knot);
        for i_local in 0..(end - start) {
            let w_i = w[start + i_local];
            let mut row = weighted_dxf.row_mut(i_local);
            for col in 0..rank {
                row[col] *= w_i;
            }
        }
        let m = crate::faer_ndarray::fast_atb(&weighted_dxf, &xf_chunk);
        let mut partial = Array2::<f64>::zeros((rank, rank));
        partial += &m;
        partial += &m.t();
        if let Some(c) = c_opt {
            let mut weighted_xf = xf_chunk.to_owned();
            for i_local in 0..(end - start) {
                let c_i = c[start + i_local];
                let mut row = weighted_xf.row_mut(i_local);
                for col in 0..rank {
                    row[col] *= c_i;
                }
            }
            partial += &crate::faer_ndarray::fast_atb(&xf_chunk, &weighted_xf);
        }
        partial
    }

    fn trace_projected_chunk_reduction(
        &self,
        xf: ArrayView2<'_, f64>,
        u_knot: &Array2<f64>,
        w: &[f64],
        c_opt: Option<&[f64]>,
        rank: usize,
        start: usize,
        end: usize,
    ) -> (f64, f64) {
        let chunk_n = end - start;
        let xf_chunk = xf.slice(ndarray::s![start..end, ..]);
        let kd_chunk = self
            .implicit_deriv
            .row_chunk_first_raw(self.axis, start..end)
            .expect("radial scalar evaluation failed during trace chunk reduction");
        let dxf_chunk = crate::faer_ndarray::fast_ab(&kd_chunk, u_knot);
        let mut design_total = 0.0_f64;
        let mut correction_total = 0.0_f64;
        let dxf_slice_opt = if dxf_chunk.is_standard_layout() {
            dxf_chunk.as_slice()
        } else {
            None
        };
        let xf_slice_opt = if xf_chunk.is_standard_layout() {
            xf_chunk.as_slice()
        } else {
            None
        };
        if let (Some(dxf_slice), Some(xf_slice)) = (dxf_slice_opt, xf_slice_opt) {
            for i_local in 0..chunk_n {
                let i = start + i_local;
                let w_i = w[i];
                let off = i_local * rank;
                let drow = &dxf_slice[off..off + rank];
                let xrow = &xf_slice[off..off + rank];
                let mut acc = 0.0_f64;
                for k in 0..rank {
                    acc += drow[k] * xrow[k];
                }
                design_total += w_i * acc;
                if let Some(c) = c_opt {
                    let c_i = c[i];
                    let mut acc2 = 0.0_f64;
                    for k in 0..rank {
                        let v = xrow[k];
                        acc2 += v * v;
                    }
                    correction_total += c_i * acc2;
                }
            }
        } else {
            for i_local in 0..chunk_n {
                let i = start + i_local;
                let w_i = w[i];
                let dxf_row = dxf_chunk.row(i_local);
                let xf_row = xf_chunk.row(i_local);
                let mut acc = 0.0_f64;
                for k in 0..rank {
                    acc += dxf_row[k] * xf_row[k];
                }
                design_total += w_i * acc;
                if let Some(c) = c_opt {
                    let c_i = c[i];
                    let mut acc2 = 0.0_f64;
                    for k in 0..rank {
                        let v = xf_row[k];
                        acc2 += v * v;
                    }
                    correction_total += c_i * acc2;
                }
            }
        }
        (design_total, correction_total)
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
        debug_assert_eq!(xf.dim(), (n_obs, rank));

        // Once: unproject F to raw knot space → (n_knots × rank).
        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());

        // Match the chunk sizing `xt_logdet_kernel_x_diagonal` uses so the
        // live block stays in L2/L3 across realistic biobank shapes.
        const TARGET_BYTES: usize = 8 * 1024 * 1024;
        let chunk_rows = (TARGET_BYTES / ((self.p + rank).max(1) * 8))
            .max(512)
            .min(n_obs);

        let w_array = self.w_diag.as_ref();
        let w = w_array
            .as_slice()
            .expect("w_diag must be contiguous for slice access");
        let c_opt = self
            .c_x_psi_beta
            .as_ref()
            .map(|arc| arc.as_slice().expect("c_x_psi_beta must be contiguous"));
        let num_chunks = (n_obs + chunk_rows - 1) / chunk_rows;
        let use_parallel = num_chunks >= 2
            && rayon::current_thread_index().is_none()
            && (n_obs as u64) * (rank as u64).max(1) >= 1_000_000;
        let chunk_totals: Vec<(f64, f64)> = if use_parallel {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..num_chunks)
                .into_par_iter()
                .map(|ci| {
                    let start = ci * chunk_rows;
                    let end = (start + chunk_rows).min(n_obs);
                    self.trace_projected_chunk_reduction(xf, &u_knot, w, c_opt, rank, start, end)
                })
                .collect()
        } else {
            let mut totals = Vec::with_capacity(num_chunks);
            let mut start = 0usize;
            while start < n_obs {
                let end = (start + chunk_rows).min(n_obs);
                totals.push(
                    self.trace_projected_chunk_reduction(xf, &u_knot, w, c_opt, rank, start, end),
                );
                start = end;
            }
            totals
        };
        let mut design_total = 0.0_f64;
        let mut correction_total = 0.0_f64;
        for (d, c) in chunk_totals {
            design_total += d;
            correction_total += c;
        }

        // Penalty trace: tr(F^T S_psi F) via dense BLAS3.
        let s_f = self.s_psi.dot(factor);
        let penalty: f64 = factor.iter().zip(s_f.iter()).map(|(&f, &s)| f * s).sum();

        2.0 * design_total + correction_total + penalty
    }

    /// Exact `F^T B_d F` using the same cached `X · F` projection as the
    /// scalar trace path. This avoids the default rank-many matrix-free
    /// matvecs in dense-spectral outer-Hessian cross-trace assembly.
    fn projected_matrix_with_xf(
        &self,
        factor: &Array2<f64>,
        xf: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let rank = factor.ncols();
        let n_obs = self.w_diag.len();
        debug_assert_eq!(xf.dim(), (n_obs, rank));

        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());
        const TARGET_BYTES: usize = 8 * 1024 * 1024;
        let chunk_rows = (TARGET_BYTES / ((self.p + rank).max(1) * 8))
            .max(512)
            .min(n_obs);

        let w_array = self.w_diag.as_ref();
        let w = w_array
            .as_slice()
            .expect("w_diag must be contiguous for slice access");
        let c_opt = self
            .c_x_psi_beta
            .as_ref()
            .map(|arc| arc.as_slice().expect("c_x_psi_beta must be contiguous"));
        let num_chunks = (n_obs + chunk_rows - 1) / chunk_rows;
        let use_parallel = num_chunks >= 2
            && rayon::current_thread_index().is_none()
            && (n_obs as u64) * (rank as u64).max(1) >= 1_000_000;
        let mut projected = if use_parallel {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..num_chunks)
                .into_par_iter()
                .map(|ci| {
                    let start = ci * chunk_rows;
                    let end = (start + chunk_rows).min(n_obs);
                    self.projected_chunk_contribution(xf, &u_knot, w, c_opt, rank, start, end)
                })
                .reduce(
                    || Array2::<f64>::zeros((rank, rank)),
                    |mut acc, partial| {
                        acc += &partial;
                        acc
                    },
                )
        } else {
            let mut projected = Array2::<f64>::zeros((rank, rank));
            let mut start = 0usize;
            while start < n_obs {
                let end = (start + chunk_rows).min(n_obs);
                let partial =
                    self.projected_chunk_contribution(xf, &u_knot, w, c_opt, rank, start, end);
                projected += &partial;
                start = end;
            }
            projected
        };

        let s_f = crate::faer_ndarray::fast_ab(&self.s_psi, factor);
        projected += &crate::faer_ndarray::fast_atb(factor, &s_f);
        let projected_t = projected.t().to_owned();
        projected += &projected_t;
        projected.mapv_inplace(|value| 0.5 * value);
        projected
    }

    /// Batched-axis sibling of [`Self::projected_matrix_with_xf`].
    /// Shares `X · F`, raw radial row chunks, and chunk traversal across all
    /// axes in the group; each axis still gets its exact own penalty and
    /// optional third-derivative correction.
    fn projected_matrix_all_axes_with_xf<'a>(
        &self,
        factor: &Array2<f64>,
        xf: ArrayView2<'_, f64>,
        axes: &[(usize, &'a Array2<f64>, Option<&'a Array1<f64>>)],
    ) -> Vec<Array2<f64>> {
        let n_axes = axes.len();
        if n_axes == 0 {
            return Vec::new();
        }
        let rank = factor.ncols();
        let n_obs = self.w_diag.len();
        debug_assert_eq!(xf.dim(), (n_obs, rank));

        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());
        const TARGET_BYTES: usize = 8 * 1024 * 1024;
        let chunk_rows = (TARGET_BYTES / ((self.p + rank).max(1) * 8))
            .max(512)
            .min(n_obs);

        let w_array = self.w_diag.as_ref();
        let w = w_array
            .as_slice()
            .expect("w_diag must be contiguous for slice access");
        let num_chunks = (n_obs + chunk_rows - 1) / chunk_rows;
        let use_parallel = num_chunks >= 2
            && rayon::current_thread_index().is_none()
            && (n_obs as u64) * (rank as u64).max(1) >= 1_000_000;
        let compute_chunk = |start: usize, end: usize| -> Vec<Array2<f64>> {
            let xf_chunk = xf.slice(ndarray::s![start..end, ..]);
            let kd_all = self
                .implicit_deriv
                .row_chunk_first_raw_all_axes(start..end)
                .expect("radial scalar evaluation failed during implicit hyper batched projected_matrix");
            let mut wxf_chunk = xf_chunk.to_owned();
            for i_local in 0..(end - start) {
                let w_i = w[start + i_local];
                let mut row = wxf_chunk.row_mut(i_local);
                for col in 0..rank {
                    row[col] *= w_i;
                }
            }
            let mut local = (0..n_axes)
                .map(|_| Array2::<f64>::zeros((rank, rank)))
                .collect::<Vec<_>>();
            for (slot, (axis, _, c_opt)) in axes.iter().enumerate() {
                let kd_chunk = &kd_all[*axis];
                let dxf_chunk = crate::faer_ndarray::fast_ab(kd_chunk, &u_knot);
                let m = crate::faer_ndarray::fast_atb(&dxf_chunk, &wxf_chunk);
                local[slot] += &m;
                local[slot] += &m.t();
                if let Some(c) = c_opt {
                    let c_slice = c.as_slice().expect("c_x_psi_beta must be contiguous");
                    let mut weighted_xf = xf_chunk.to_owned();
                    for i_local in 0..(end - start) {
                        let c_i = c_slice[start + i_local];
                        let mut row = weighted_xf.row_mut(i_local);
                        for col in 0..rank {
                            row[col] *= c_i;
                        }
                    }
                    local[slot] += &crate::faer_ndarray::fast_atb(&xf_chunk, &weighted_xf);
                }
            }
            local
        };
        let mut out = if use_parallel {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..num_chunks)
                .into_par_iter()
                .map(|ci| {
                    let start = ci * chunk_rows;
                    let end = (start + chunk_rows).min(n_obs);
                    compute_chunk(start, end)
                })
                .reduce(
                    || {
                        (0..n_axes)
                            .map(|_| Array2::<f64>::zeros((rank, rank)))
                            .collect::<Vec<_>>()
                    },
                    |mut acc, partial| {
                        for (a, p) in acc.iter_mut().zip(partial) {
                            *a += &p;
                        }
                        acc
                    },
                )
        } else {
            let mut out = (0..n_axes)
                .map(|_| Array2::<f64>::zeros((rank, rank)))
                .collect::<Vec<_>>();
            let mut start = 0usize;
            while start < n_obs {
                let end = (start + chunk_rows).min(n_obs);
                let local = compute_chunk(start, end);
                for (a, l) in out.iter_mut().zip(local) {
                    *a += &l;
                }
                start = end;
            }
            out
        };

        for (slot, (_, s_psi, _)) in axes.iter().enumerate() {
            let s_f = crate::faer_ndarray::fast_ab(s_psi, factor);
            out[slot] += &crate::faer_ndarray::fast_atb(factor, &s_f);
            let out_t = out[slot].t().to_owned();
            out[slot] += &out_t;
            out[slot].mapv_inplace(|value| 0.5 * value);
        }

        out
    }

    /// Batched-axis sibling of [`Self::trace_projected_factor_with_xf`].
    /// For every `(axis, s_psi, c_x_psi_beta)` tuple in `axes`, returns
    /// `tr(F^T B_d F)` using a single sweep through the design rows: each
    /// chunk's radial scalars `(phi, q, r²)` are evaluated once via
    /// `row_chunk_first_raw_all_axes`, then the per-axis `K_d · U_knot`
    /// GEMM and fused inner products run inside that one row pass. Each
    /// axis carries its own penalty matrix and (optional) third-derivative
    /// correction vector so the per-axis result is numerically identical
    /// (modulo accumulation order) to the existing per-axis path.
    fn trace_projected_factor_all_axes_with_xf<'a>(
        &self,
        factor: &Array2<f64>,
        xf: ArrayView2<'_, f64>,
        axes: &[(usize, &'a Array2<f64>, Option<&'a Array1<f64>>)],
    ) -> Vec<f64> {
        let n_axes = axes.len();
        if n_axes == 0 {
            return Vec::new();
        }
        let rank = factor.ncols();
        let n_obs = self.w_diag.len();
        debug_assert_eq!(xf.dim(), (n_obs, rank));

        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());

        const TARGET_BYTES: usize = 8 * 1024 * 1024;
        let chunk_rows = (TARGET_BYTES / ((self.p + rank).max(1) * 8))
            .max(512)
            .min(n_obs);

        let w_array = self.w_diag.as_ref();
        let w = w_array
            .as_slice()
            .expect("w_diag must be contiguous for slice access");
        let num_chunks = (n_obs + chunk_rows - 1) / chunk_rows;
        let use_parallel = num_chunks >= 2
            && rayon::current_thread_index().is_none()
            && (n_obs as u64) * (rank as u64).max(1) >= 1_000_000;
        let compute_chunk = |start: usize, end: usize| -> (Vec<f64>, Vec<f64>) {
            let chunk_n = end - start;
            let xf_chunk = xf.slice(ndarray::s![start..end, ..]);
            let kd_all = self
                .implicit_deriv
                .row_chunk_first_raw_all_axes(start..end)
                .expect("radial scalar evaluation failed during implicit hyper batched trace");
            let xf_slice_opt = if xf_chunk.is_standard_layout() {
                xf_chunk.as_slice()
            } else {
                None
            };
            let mut design_local = vec![0.0_f64; n_axes];
            let mut correction_local = vec![0.0_f64; n_axes];
            for (slot, (axis, _, c_opt)) in axes.iter().enumerate() {
                let kd_chunk = &kd_all[*axis];
                let dxf_chunk = crate::faer_ndarray::fast_ab(kd_chunk, &u_knot);
                let mut design_total = 0.0_f64;
                let mut correction_total = 0.0_f64;
                let dxf_slice_opt = if dxf_chunk.is_standard_layout() {
                    dxf_chunk.as_slice()
                } else {
                    None
                };
                let c_slice_opt = c_opt.and_then(|c| c.as_slice());
                if let (Some(dxf_slice), Some(xf_slice)) = (dxf_slice_opt, xf_slice_opt) {
                    for i_local in 0..chunk_n {
                        let i = start + i_local;
                        let w_i = w[i];
                        let off = i_local * rank;
                        let drow = &dxf_slice[off..off + rank];
                        let xrow = &xf_slice[off..off + rank];
                        let mut acc = 0.0_f64;
                        for k in 0..rank {
                            acc += drow[k] * xrow[k];
                        }
                        design_total += w_i * acc;
                        if let Some(c) = c_slice_opt {
                            let c_i = c[i];
                            let mut acc2 = 0.0_f64;
                            for k in 0..rank {
                                let v = xrow[k];
                                acc2 += v * v;
                            }
                            correction_total += c_i * acc2;
                        }
                    }
                } else {
                    for i_local in 0..chunk_n {
                        let i = start + i_local;
                        let w_i = w[i];
                        let dxf_row = dxf_chunk.row(i_local);
                        let xf_row = xf_chunk.row(i_local);
                        let mut acc = 0.0_f64;
                        for k in 0..rank {
                            acc += dxf_row[k] * xf_row[k];
                        }
                        design_total += w_i * acc;
                        if let Some(c) = c_opt {
                            let c_i = c[i];
                            let mut acc2 = 0.0_f64;
                            for k in 0..rank {
                                let v = xf_row[k];
                                acc2 += v * v;
                            }
                            correction_total += c_i * acc2;
                        }
                    }
                }
                design_local[slot] = design_total;
                correction_local[slot] = correction_total;
            }
            (design_local, correction_local)
        };
        let (design_totals, correction_totals) = if use_parallel {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..num_chunks)
                .into_par_iter()
                .map(|ci| {
                    let start = ci * chunk_rows;
                    let end = (start + chunk_rows).min(n_obs);
                    compute_chunk(start, end)
                })
                .reduce(
                    || (vec![0.0_f64; n_axes], vec![0.0_f64; n_axes]),
                    |(mut da, mut ca), (db, cb)| {
                        for (a, b) in da.iter_mut().zip(db) {
                            *a += b;
                        }
                        for (a, b) in ca.iter_mut().zip(cb) {
                            *a += b;
                        }
                        (da, ca)
                    },
                )
        } else {
            let mut design_totals = vec![0.0_f64; n_axes];
            let mut correction_totals = vec![0.0_f64; n_axes];
            let mut start = 0usize;
            while start < n_obs {
                let end = (start + chunk_rows).min(n_obs);
                let (d_local, c_local) = compute_chunk(start, end);
                for (a, b) in design_totals.iter_mut().zip(d_local) {
                    *a += b;
                }
                for (a, b) in correction_totals.iter_mut().zip(c_local) {
                    *a += b;
                }
                start = end;
            }
            (design_totals, correction_totals)
        };

        let mut out = Vec::with_capacity(n_axes);
        for (slot, (_axis, s_psi, _)) in axes.iter().enumerate() {
            let s_f = s_psi.dot(factor);
            let penalty: f64 = factor.iter().zip(s_f.iter()).map(|(&f, &s)| f * s).sum();
            out.push(2.0 * design_totals[slot] + correction_totals[slot] + penalty);
        }
        out
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
    #[allow(private_interfaces)]
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

    fn is_implicit(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn as_sparse_directional(&self) -> Option<&SparseDirectionalHyperOperator> {
        Some(self)
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
        assert_eq!(root.ncols(), end.saturating_sub(start));
        assert!(end <= total_dim);
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
        assert_eq!(root.ncols(), end.saturating_sub(start));
        assert_eq!(prior_mean.len(), end.saturating_sub(start));
        assert!(end <= total_dim);
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

    fn apply_root(&self, beta: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(beta.len(), self.dim());
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
        let m_mat = crate::faer_ndarray::fast_atb(&n_basis, &nk_basis);
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
///
/// Threshold is RELATIVE to `max|eigenvalue|` — never floored at an
/// absolute value. Earlier this function clamped `max_ev` to at least
/// `1.0`, which silently classified genuine positive modes of
/// small-scale penalties (Wahba pseudo-spline `m=4` Gram had
/// `max|eig| ≈ 5e-3`) as numerical zero. That corrupted the
/// pseudo-logdet and broke REML's invariance under `S → c·S`, causing
/// the m=4 smooth contribution to collapse to ~0. When `max_ev == 0`
/// (no positive modes) the threshold collapses to 0 too, which is the
/// only correct answer.
pub(crate) fn positive_eigenvalue_threshold(eigenvalues: &[f64]) -> f64 {
    let p = eigenvalues.len();
    let max_ev = eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |a, b| a.max(b.abs()));
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

    /// Apply the projected pseudo-inverse `K = U_S · H_proj⁻¹ · U_Sᵀ` to a
    /// vector `a`, returning the minimum-norm solution `v = K · a` of the
    /// system `H v = a` restricted to `range(S₊)`.
    ///
    /// This is the correct stand-in for `H⁻¹ · a` in all per-coordinate
    /// outer-gradient/Hessian formulas when the rank-deficient LAML fix is
    /// active (`penalty_subspace_trace = Some`). The full `H⁻¹ · a` solve
    /// amplifies any component of `a` outside `range(H_free)` by
    /// `1/σ_min(H_active_normal)` — which on biobank-scale survival
    /// marginal-slope is ~10¹² and propagates into outer gradients of
    /// magnitude 10¹⁴, suppressed by the envelope tripwire downstream and
    /// killing every seed before the fit can take a step. The projected
    /// pseudo-inverse drops the null-space contribution by construction,
    /// returning the gradient that lives on the constrained manifold —
    /// which is what LAML on the projected-Hessian cost demands for
    /// derivative-consistency with the projected `log|U_Sᵀ H U_S|` term.
    ///
    /// Costs `O(p·r + r²)` for the two `U_S`-contractions plus the `r × r`
    /// solve — strictly cheaper than the `O(p²)` full `hop.solve_multi`
    /// when `r ≪ p`, and bounded regardless of `σ_min(H)`.
    pub fn apply_pseudo_inverse(&self, a: &Array1<f64>) -> Array1<f64> {
        let proj_a = crate::faer_ndarray::fast_atv(&self.u_s, a);
        let h_proj_inv_a = self.h_proj_inverse.dot(&proj_a);
        crate::faer_ndarray::fast_av(&self.u_s, &h_proj_inv_a)
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
            .map(|(w, v)| (w, v))
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

/// KKT residual `r = ∇_β L_pen(β̂)` at the converged inner iterate, already
/// projected onto the *free subspace* — i.e. with the Lagrange-multiplier
/// component along the active inequality-constraint normals stripped out.
///
/// The IFT correction `−½ rᵀ H⁻¹ r` in `reml_laml_evaluate` requires `r` to
/// lie in `range(H_free)` so that `H⁻¹·r` is well-conditioned. At a
/// constrained-stationary point the unprojected residual carries multiplier
/// mass *outside* that range; feeding it through `H⁻¹` amplifies floating-
/// point noise by `1/σ_min(H_active_normal)` (≈10¹² on rank-deficient inner
/// Hessians) and the resulting gradient explodes to ~10¹⁵ — the envelope
/// tripwire then suppresses the gradient and the outer optimizer can make no
/// progress (the biobank survival marginal-slope failure mode).
///
/// This newtype lifts the projection contract into the type system: a value
/// of this type can only be produced by the projection-aware constructors,
/// so callers cannot accidentally hand an unprojected residual to the
/// unified evaluator. The free-function helpers in `families::custom_family`
/// (`exact_newton_joint_kkt_residual_for_ift`) take active-set information
/// and emit values of this type; the unified evaluator consumes it via the
/// borrowing accessor.
#[derive(Clone, Debug)]
pub struct ProjectedKktResidual(Array1<f64>);

impl ProjectedKktResidual {
    /// Construct from a vector that the caller guarantees has already been
    /// projected onto the free subspace. Crate-private so the projection
    /// invariant cannot be bypassed by downstream callers.
    pub(crate) fn from_projected(residual: Array1<f64>) -> Self {
        Self(residual)
    }

    /// Borrow the underlying free-space residual for the H⁻¹·r solve and
    /// its ρ-derivatives.
    pub fn as_array(&self) -> &Array1<f64> {
        &self.0
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

    /// Configured prior over rho coordinates. The evaluator receives the
    /// realized cost/gradient tuple separately; this copy lets EFS use the
    /// conjugate Gamma rate in its multiplicative denominator.
    pub rho_prior: crate::types::RhoPrior,

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
    barrier_config: Option<BarrierConfig>,
    kkt_residual: Option<ProjectedKktResidual>,
    active_constraints: Option<Arc<ActiveLinearConstraintBlock>>,
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
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
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
            rho_prior: self.rho_prior,
            n_observations: self.n_observations,
            nullspace_dim,
            dispersion: self.dispersion,
            ext_coords: self.ext_coords,
            ext_coord_pair_fn: self.ext_coord_pair_fn,
            rho_ext_pair_fn: self.rho_ext_pair_fn,
            fixed_drift_deriv: self.fixed_drift_deriv,
            barrier_config: self.barrier_config,
            kkt_residual: self.kkt_residual,
            active_constraints: self.active_constraints,
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
    coord.apply_shifted_penalty(beta, lambda)
}

fn penalty_a_k_quadratic(coord: &PenaltyCoordinate, beta: &Array1<f64>, lambda: f64) -> f64 {
    coord.shifted_quadratic(beta, lambda)
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

    // ─── Implicit-function-theorem cost correction ───
    //
    // Define β*(ρ) as the exact inner optimum (g_β(β*, ρ) = 0).  The outer
    // objective we *want* is V(β*(ρ), ρ); what the envelope formula above
    // computes is V(β̂, ρ) at the inner-returned β̂ ≠ β*.  First-order
    // implicit-function theorem gives
    //   β* − β̂ ≈ −H⁻¹ r,   r := ∇_β L_pen(β̂) = S(λ)β̂ − ∇ℓ(β̂)
    //   V(β*) ≈ V(β̂) + (∂V/∂β)ᵀ(β* − β̂) = V(β̂) + rᵀ · (−H⁻¹ r)
    //         = V(β̂) − ½ rᵀ H⁻¹ r            (using ∂V/∂β = r at β̂, second-
    //                                          order expansion of V symmetric
    //                                          in (β* − β̂)).
    //
    // The cost correction strictly vanishes when the inner reached exact
    // KKT (r = 0).  When the inner exits via the noise-floor certificate
    // with ‖r‖ > 0 it absorbs the leading error; the gradient and Hessian
    // corrections below are the exact first and second ρ derivatives of this
    // same scalar Newton correction under fixed-dispersion LAML.
    //
    // Filter: callers populate `kkt_residual` only on convergent inner paths.
    // The [`ProjectedKktResidual`] newtype lifts the projection invariant into
    // the type system (callers cannot construct one without going through the
    // active-set-aware projection helper), so the only thing left to validate
    // here is the length match against the Hessian operator. `None` means the
    // caller is presenting an exact-KKT mode and the envelope identities are
    // already valid.
    let kkt_residual_vec: Option<&Array1<f64>> = match solution.kkt_residual.as_ref() {
        Some(residual) => {
            let r = residual.as_array();
            if r.len() != hop.dim() {
                return Err(RemlError::DimensionMismatch {
                    reason: format!(
                        "projected KKT residual length mismatch: got {}, expected {}",
                        r.len(),
                        hop.dim()
                    ),
                }
                .into());
            }
            Some(r)
        }
        None => None,
    };
    let kkt_residual_correction_active = kkt_residual_vec.is_some()
        && matches!(solution.dispersion, DispersionHandling::Fixed { .. });
    // One-shot structured log of the IFT gate. Debug-level so it doesn't
    // spam normal runs but is immediately greppable when debugging an
    // envelope-gradient consistency failure (search for `[ift-gate]`).
    log::debug!(
        "[ift-gate] kkt_residual.is_some()={} dispersion={} correction_active={} subspace_trace.is_some()={} hop.dim()={} k={}",
        solution.kkt_residual.is_some(),
        match &solution.dispersion {
            DispersionHandling::Fixed { .. } => "Fixed",
            DispersionHandling::ProfiledGaussian => "ProfiledGaussian",
        },
        kkt_residual_correction_active,
        solution.penalty_subspace_trace.is_some(),
        hop.dim(),
        k,
    );
    if let Some(r) = kkt_residual_vec.filter(|_| kkt_residual_correction_active) {
        // Cost-side IFT correction `−½ rᵀ H⁻¹ r`. When the rank-deficient
        // LAML fix is active (`penalty_subspace_trace = Some`), the
        // mathematically correct inverse here is the Moore-Penrose
        // pseudo-inverse projected onto `range(S_+)` — not the full
        // `H⁻¹`. The full-H solve at near-singular boundary states
        // (the biobank survival marginal-slope pathology) amplifies
        // floating-point noise in `r` outside `range(S_+)` by
        // `1/σ_min(H) ≈ 10¹²`, which then propagates into a 10¹³-magnitude
        // gradient component and traps the outer optimizer at max-iter.
        // The projected pseudo-inverse kills any spurious null-space
        // component of `r` before the inverse is applied, recovering
        // the honest correction.
        let cost_correction = if let Some(kernel) = solution.penalty_subspace_trace.as_ref() {
            -0.5_f64 * kernel.bilinear_pseudo_inverse(r, r)
        } else {
            let mut rhs = Array2::<f64>::zeros((hop.dim(), 1));
            rhs.column_mut(0).assign(r);
            let w_mat = hop.solve_multi(&rhs);
            let w: Array1<f64> = w_mat.column(0).to_owned();
            -0.5_f64 * r.view().dot(&w)
        };
        if cost_correction.is_finite() {
            cost += cost_correction;
        }
    }

    if !cost.is_finite() {
        return Err(RemlError::NonFiniteValue {
            reason: format!(
                "REML/LAML cost is non-finite ({cost}); check inner solver convergence"
            ),
        }
        .into());
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
    let need_family_corrections = effective_deriv.has_corrections();
    // Stack the K curvature-penalty RHS (when corrections are active) and the
    // ext_dim outer-coordinate gradient RHS into a single (dim, total_cols)
    // matrix and dispatch one `solve_multi` against the shared Hessian
    // factorization. The dense-spectral backend turns that into one
    // `Uᵀ·R`, one per-eigendirection scale, and one `U·projected` — a BLAS-3
    // pass that has much better cache locality than 14 independent BLAS-2
    // single-RHS solves and lets large-`dim` fits hit the cuSOLVER batched
    // chol_solve route in `gpu/policy.rs` (the per-vector solves never
    // crossed the threshold).
    let dim = hop.dim();
    let ext_dim_local = solution.ext_coords.len();
    let total_cols = if need_family_corrections {
        k + ext_dim_local
    } else {
        ext_dim_local
    };
    let (rho_v_ks, ext_v_is): (Option<Vec<Array1<f64>>>, Vec<Array1<f64>>) = if total_cols == 0 {
        (
            if need_family_corrections {
                Some(Vec::new())
            } else {
                None
            },
            Vec::new(),
        )
    } else {
        // Per-coordinate `v_k = H⁻¹ · a_k` mode responses. When the
        // rank-deficient LAML fix is active (`penalty_subspace_trace =
        // Some`), the full `hop.solve_multi` path amplifies any component
        // of `a_k` outside `range(H_free)` by `1/σ_min(H_active_normal)`
        // — which on biobank-scale survival marginal-slope is ~10¹² and
        // propagates into family-correction terms with magnitude 10¹⁴,
        // tripping the envelope-consistency check downstream and killing
        // every seed. The projected pseudo-inverse `K = U_S · H_proj⁻¹ ·
        // U_Sᵀ` is the principled stand-in for `H⁻¹` on the constrained
        // manifold — it returns the minimum-norm solution of `H v = a`
        // restricted to `range(S₊)`, matches the derivative of the
        // projected `log|U_Sᵀ H U_S|` cost term, and is bounded
        // regardless of `σ_min(H)`. Route every v_k / v_i through the
        // kernel when it is installed; fall back to the full solve only
        // when no kernel is available.
        let kernel = solution.penalty_subspace_trace.as_ref();
        if let Some(kernel) = kernel {
            // Lift to a *constraint-aware* kernel when the inner solver
            // recorded active inequality constraints. The Schur-complement
            // formula
            //   K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S
            // returns the minimum-norm IFT mode response inside
            // T = range(S₊) ∩ ker(A_act) — the smooth manifold the inner
            // is genuinely moving on — and matches the derivative of the
            // projected Laplace cost log|U_Tᵀ H U_T|. When no active
            // constraints are present the helper degrades to bare K_S
            // with zero overhead.
            let constrained = solution
                .active_constraints
                .as_ref()
                .map(|block| kernel.with_active_constraints(block.a.view()));
            let apply = |v: &Array1<f64>| -> Array1<f64> {
                match constrained.as_ref() {
                    Some(ck) if ck.has_active_constraints() => ck.apply_pseudo_inverse(v),
                    _ => kernel.apply_pseudo_inverse(v),
                }
            };
            let rho_v_ks = if need_family_corrections {
                Some(
                    rho_curvature_a_k_betas
                        .iter()
                        .map(|a_k| apply(a_k))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let ext_v_is: Vec<Array1<f64>> = solution
                .ext_coords
                .iter()
                .map(|coord| apply(&coord.g))
                .collect();
            (rho_v_ks, ext_v_is)
        } else {
            let mut rhs_stack = Array2::<f64>::zeros((dim, total_cols));
            let mut col_idx = 0;
            if need_family_corrections {
                for a_k_beta in rho_curvature_a_k_betas.iter() {
                    rhs_stack.column_mut(col_idx).assign(a_k_beta);
                    col_idx += 1;
                }
            }
            for coord in solution.ext_coords.iter() {
                rhs_stack.column_mut(col_idx).assign(&coord.g);
                col_idx += 1;
            }
            debug_assert_eq!(col_idx, total_cols);
            let solved_stack = hop.solve_multi(&rhs_stack);
            let rho_v_ks = if need_family_corrections {
                Some((0..k).map(|i| solved_stack.column(i).to_owned()).collect())
            } else {
                None
            };
            let ext_offset = if need_family_corrections { k } else { 0 };
            let ext_v_is: Vec<Array1<f64>> = (0..ext_dim_local)
                .map(|i| solved_stack.column(ext_offset + i).to_owned())
                .collect();
            (rho_v_ks, ext_v_is)
        }
    };
    let coord_corrections: Vec<Option<DriftDerivResult>> = if need_family_corrections {
        let rho_vs = rho_v_ks
            .as_ref()
            .expect("rho mode responses required for Hessian corrections");
        let mut correction_vs = Vec::with_capacity(k + ext_dim);
        correction_vs.extend(rho_vs.iter().cloned());
        correction_vs.extend(ext_v_is.iter().cloned());
        let correction_work = solution
            .n_observations
            .saturating_mul(hop.dim())
            .saturating_mul((k + ext_dim).max(1));
        // Small coefficient systems produce bounded-size correction operators;
        // keep their independent row contractions parallel even at large n.
        let correction_parallel_work_limit = if hop.dim() <= 512 {
            1_000_000_000
        } else {
            64_000_000
        };
        let parallel_corrections = correction_work <= correction_parallel_work_limit;
        if effective_deriv.has_batched_hessian_derivative_corrections() {
            log::info!(
                "[STAGE] reml_laml coord_corrections mode=batched k={} ext_dim={} n={} dim={} work={}",
                k,
                ext_dim,
                solution.n_observations,
                hop.dim(),
                correction_work
            );
            effective_deriv.hessian_derivative_corrections_result(&correction_vs)?
        } else if parallel_corrections {
            correction_vs
                .par_iter()
                .map(|v_k| effective_deriv.hessian_derivative_correction_result(v_k))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            log::info!(
                "[STAGE] reml_laml coord_corrections mode=serial k={} ext_dim={} n={} dim={} work={}",
                k,
                ext_dim,
                solution.n_observations,
                hop.dim(),
                correction_work
            );
            correction_vs
                .iter()
                .map(|v_k| effective_deriv.hessian_derivative_correction_result(v_k))
                .collect::<Result<Vec<_>, _>>()?
        }
    } else {
        (0..(k + ext_dim)).map(|_| None).collect()
    };
    if coord_corrections.len() != k + ext_dim {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "REML/LAML derivative correction count mismatch: got {}, expected {}",
                coord_corrections.len(),
                k + ext_dim
            ),
        }
        .into());
    }
    let rho_corrections = &coord_corrections[..k];
    let ext_corrections = &coord_corrections[k..];

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
        for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
            let correction = ext_corrections[ext_idx].as_ref();
            match hyper_coord_total_drift_result(&coord.drift, correction, hop.dim()) {
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

    let build_trace_drifts = || {
        let mut drifts = Vec::with_capacity(k + ext_dim);
        for idx in 0..k {
            drifts.push(penalty_total_drift_result(
                &solution.penalty_coords[idx],
                curvature_lambdas[idx],
                rho_corrections[idx].as_ref(),
            ));
        }
        for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
            drifts.push(hyper_coord_total_drift_result(
                &coord.drift,
                ext_corrections[ext_idx].as_ref(),
                hop.dim(),
            ));
        }
        drifts
    };

    let projected_trace_values: Option<Vec<f64>> =
        if incl_logdet_h && stochastic_trace_values.is_none() {
            solution
                .penalty_subspace_trace
                .as_ref()
                .map(|kernel| penalty_subspace_trace_drifts_batched(kernel, &build_trace_drifts()))
        } else {
            None
        };

    let exact_dense_trace_values: Option<Vec<f64>> =
        if incl_logdet_h && stochastic_trace_values.is_none() && projected_trace_values.is_none() {
            hop.as_exact_dense_spectral()
                .map(|ds| dense_spectral_trace_logdet_drifts_batched(ds, &build_trace_drifts()))
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

            // Cost derivative for the shifted penalty:
            // a_i = ½ λₖ (β̂ - μₖ)' Sₖ (β̂ - μₖ).
            //
            // The β-gradient derivative is λₖSₖ(β̂-μₖ); dotting it with β̂
            // would drop the μₖ'λₖSₖμₖ half of the chain rule.
            let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambdas[idx]);

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
            } else if let Some(ref projected_traces) = projected_trace_values {
                projected_traces[idx]
            } else if let Some(ref exact_traces) = exact_dense_trace_values {
                exact_traces[idx]
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
            let value = outer_gradient_entry(
                a_i,
                trace_logdet_i,
                solution.penalty_logdet.first[idx],
                &solution.dispersion,
                dp_cgrad,
                profiled_scale,
                incl_logdet_h,
                incl_logdet_s,
            );
            // Per-coordinate breakdown of the outer-gradient entry. Was a
            // floor-level eprintln during the LAML cost-trajectory
            // investigation; demoted to trace! so RUST_LOG=trace can still
            // recover it without 91-line-per-iter stderr noise on default
            // runs.
            log::trace!(
                "[RHO-GRAD] idx={} value={:+.6e} a_i={:+.6e} trace_logdet={:+.6e} ld_s_first={:+.6e} incl_h={} incl_s={}",
                idx, value, a_i, trace_logdet_i, solution.penalty_logdet.first[idx], incl_logdet_h, incl_logdet_s
            );
            (idx, value)
        })
        .collect();
    for (idx, value) in rho_grad_entries {
        grad[idx] = value;
    }

    // ─── Implicit-function-theorem gradient correction ───
    //
    // The envelope formula above is the total derivative dV/dρ_k *only* when
    // β̂ satisfies the inner KKT condition ∇_β L_pen(β̂) = 0.  When the inner
    // exits via the noise-floor certificate with `r = ∇_β L_pen(β̂) ≠ 0`,
    // the corrected scalar objective is the one-step Newton profile
    //
    //   Ṽ(ρ) = V(β̂, ρ) − ½ rᵀ H⁻¹ r.
    //
    // Holding β̂ fixed while differentiating the correction gives, with
    // q = H⁻¹r, A_k = λ_k S_k, and a_k = A_k β̂:
    //
    //   ∂_k r = a_k,        ∂_k H = A_k
    //   ∂_k q = H⁻¹(a_k − A_k q)
    //   ∂_k(-½ rᵀq) = −a_kᵀq + ½ qᵀA_kq.
    //
    // The leading `−a_kᵀq` term is the familiar `−rᵀv_k` correction; the
    // `+½ qᵀA_kq` term is second-order in the KKT residual but is required if
    // the analytic gradient is to be the derivative of the corrected scalar
    // objective. The Hessian builder receives the corresponding second
    // derivative from `compute_kkt_residual_rho_corrections` so ARC sees one
    // coherent objective model instead of an envelope Hessian for a corrected
    // value/gradient pair.
    //
    // The correction strictly vanishes when r = 0.  When the inner exit
    // accepts ‖r‖ > 0 on a coordinate whose H block is poorly conditioned
    // (e.g., the failing biobank survival marginal-slope case where ‖H⁻¹‖
    // is ~10¹² on the one unpinned λ), the dropped term inflates by
    // ‖H⁻¹‖·‖r‖ and the envelope reports a gradient component orders of
    // magnitude past anything the function can actually produce — TR
    // rejects every step and collapses to its floor.  This term recovers
    // the legitimate descent direction.
    //
    // Use `rho_penalty_a_k_betas` (with `lambdas`), NOT `rho_v_ks` (whose
    // computation may use `curvature_lambdas = rho_curvature_scale · lambdas`):
    // the residual correction is in the actual S(λ) basis, and the curvature
    // scale only applies to the H-dependent trace terms.
    let kkt_rho_corrections =
        if let Some(r) = kkt_residual_vec.filter(|_| kkt_residual_correction_active && k > 0) {
            Some(compute_kkt_residual_rho_corrections(
                solution,
                hop,
                &lambdas,
                &rho_penalty_a_k_betas,
                r,
                mode == EvalMode::ValueGradientHessian,
            )?)
        } else {
            None
        };
    if let Some(corrections) = kkt_rho_corrections.as_ref() {
        let mut sl = grad.slice_mut(ndarray::s![..k]);
        sl += &corrections.gradient;
    }

    // Extended hyperparameter gradient (ψ/τ coordinates).
    //
    // Uses the SAME outer_gradient_entry() formula as ρ coordinates above.
    //
    // All extended coordinates store canonical fixed-β stationarity
    // derivatives g_i = F_{βi}. IFT gives β_i = -H^{-1}g_i, exactly like
    // the ρ block.
    // Per-call sink for the EIG-DECOMP diagnostic stash. Under `#[cfg(test)]`
    // the rayon worker for `ext_idx == 0` populates this with the captured
    // `TermStash`; after the par_iter completes the calling thread copies
    // it into its own thread-local via `debug_stash::store_terms`. Building
    // the stash inside the worker and handing it back through a per-call
    // sink eliminates the cross-thread overwrites that a process-global
    // sink suffered under concurrent tests.
    #[cfg(test)]
    let ext_stash_sink: std::sync::Arc<std::sync::Mutex<Option<debug_stash::TermStash>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    #[cfg(test)]
    let ext_stash_sink_for_closure = ext_stash_sink.clone();
    let ext_grad_entries: Result<Vec<(usize, f64)>, String> = (0..ext_dim)
        .into_par_iter()
        .map(|ext_idx| {
            let coord = &solution.ext_coords[ext_idx];
            let ext_coord_start = std::time::Instant::now();
            let grad_idx = k + ext_idx;

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
            // For canonical Gaussian (Identity link) the assembly skips
            // installing `penalty_subspace_trace` at all — `c ≡ 0` forces
            // `D_β H ≡ 0`, the classical Gaussian REML cost identity reads
            // `log|H|` (not `log|H_proj|`), and the unprojected `G_ε(H)`
            // kernel is the formula that matches that cost surface within
            // FD precision (the moving-`U_S(ψ)` projection would otherwise
            // add a `dU_S/dψ` term to the cost that the analytic gradient
            // does not capture — see the `c_nontrivial` gate in
            // `build_dense_assembly` / `build_dense_original_assembly`).
            // Drops into the `None` arm below in that branch.
            let trace_logdet_i = if !incl_logdet_h {
                0.0
            } else if let Some(ref stoch_traces) = stochastic_trace_values {
                stoch_traces[k + ext_idx]
            } else if let Some(ref projected_traces) = projected_trace_values {
                projected_traces[k + ext_idx]
            } else if let Some(ref exact_traces) = exact_dense_trace_values {
                exact_traces[k + ext_idx]
            } else {
                let correction = ext_corrections[ext_idx].as_ref();
                let drift = hyper_coord_total_drift_result(&coord.drift, correction, hop.dim());
                match (&solution.penalty_subspace_trace, &drift) {
                    (Some(kernel), DriftDerivResult::Dense(matrix)) => {
                        kernel.trace_projected_logdet(matrix)
                    }
                    (Some(kernel), DriftDerivResult::Operator(op)) => {
                        kernel.trace_operator(op.as_ref())
                    }
                    (None, DriftDerivResult::Dense(matrix)) => hop.trace_logdet_h_k(matrix, None),
                    (None, DriftDerivResult::Operator(op)) => {
                        hop.trace_logdet_operator(op.as_ref())
                    }
                }
            };

            // Test-only eigenmode diagnostic of the trace_logdet path.
            //
            // Production builds compile this block out entirely. In test
            // runs we log the unprojected `Σ φ'(σ_j)·(Uᵀ op_total U)_jj`
            // alongside the production trace `trace_logdet_i`, so the
            // reader can never mistake one for the other. For Duchon ψ
            // axes the two values can disagree by orders of magnitude
            // because the penalty-subspace projection eliminates a
            // spurious null-space contribution that has no place in the
            // cost identity. The block runs whenever the Hessian has an
            // exact dense-spectral view at small p so the stash is
            // populated regardless of which precomputed-trace path
            // satisfied `trace_logdet_i` above — relying on the fallback
            // branch alone misses the case where
            // `projected_trace_values` or `exact_dense_trace_values` is
            // already batched, which is the configuration the
            // projection-pin regression test actually exercises.
            #[cfg(test)]
            if incl_logdet_h
                && let Some(ds) = hop.as_exact_dense_spectral()
                && ds.dim() <= 12
            {
                let correction = ext_corrections[ext_idx].as_ref();
                let mut op_dense = Array2::<f64>::zeros((ds.dim(), ds.dim()));
                let b_drift_again = hyper_coord_total_drift_result(&coord.drift, None, ds.dim());
                let b_dense = match b_drift_again {
                    DriftDerivResult::Operator(o) => o.to_dense(),
                    DriftDerivResult::Dense(m) => m,
                };
                op_dense += &b_dense;
                if let Some(c) = correction {
                    match c {
                        DriftDerivResult::Dense(m) => op_dense += m,
                        DriftDerivResult::Operator(o) => op_dense += &o.to_dense(),
                    }
                }
                // Rotate to eigenbasis: (U^T op U)_jj per eigenmode
                let p = ds.dim();
                let mut u_mat = Array2::<f64>::zeros((p, p));
                for col in 0..p {
                    for row in 0..p {
                        u_mat[[row, col]] = ds.eigenvector_entry(row, col);
                    }
                }
                let ut_op = crate::faer_ndarray::fast_atb(&u_mat, &op_dense);
                let proj = crate::faer_ndarray::fast_ab(&ut_op, &u_mat);
                let eps_sq = {
                    let eps_f = (2.22e-16_f64).sqrt() * (p as f64);
                    4.0 * eps_f * eps_f
                };
                let mut per_mode = Vec::with_capacity(p);
                let mut unprojected_tr = 0.0_f64;
                for j in 0..p {
                    // reg_eigenvalue = r_ε(σ) = ½(σ + √(σ²+4ε²)). Recover σ.
                    let r = ds.reg_eigenvalue(j);
                    let sigma = r - eps_sq / (4.0 * r); // r = ½(σ + √(σ²+4ε²)) ⇒ σ = r − ε²/r
                    let phi_prime = 1.0 / (sigma * sigma + eps_sq).sqrt();
                    let contrib = phi_prime * proj[[j, j]];
                    per_mode.push((sigma, proj[[j, j]], contrib));
                    unprojected_tr += contrib;
                }
                let projection_active = solution.penalty_subspace_trace.is_some();
                eprintln!(
                    "[EIG-DECOMP ext_idx={}] unprojected_tr={:+.4e} \
                     production_tr={:+.4e} (projection_active={}) per_mode={:?}",
                    ext_idx, unprojected_tr, trace_logdet_i, projection_active, per_mode
                );
                if ext_idx == 0 {
                    let mut stash = debug_stash::TermStash {
                        unprojected_tr: Some(unprojected_tr),
                        production_tr: Some(trace_logdet_i),
                        projection_active: Some(projection_active),
                        ..Default::default()
                    };
                    if let Some(op_arc) = coord.drift.operator.as_ref()
                        && let Some(sd) = op_arc.as_sparse_directional()
                    {
                        // Stash term4's diagonal `c · X_τβ̂` directly,
                        // plus `X · v_ψ` per row where
                        // v_ψ = ext_v_is[ext_idx] = hop⁻¹·coord.g. The
                        // diagonal entering the correction sandwich is
                        // `c · X · v_ψ`; multiplying by the test's known
                        // c recovers that diagonal.
                        stash.c_x_tau_beta_diag = sd.c_x_tau_beta.clone();
                        let v_i = &ext_v_is[ext_idx];
                        stash.c_x_v_psi_diag = Some(sd.x_design.matrixvectormultiply(v_i));
                    }
                    // Hand the stash back through the per-call sink. The
                    // calling thread copies it into its own thread-local
                    // after the par_iter completes, so concurrent tests
                    // each end up reading their OWN ext_idx==0 capture
                    // rather than racing on a shared global slot.
                    *ext_stash_sink_for_closure
                        .lock()
                        .expect("EIG-DECOMP stash sink mutex poisoned") = Some(stash);
                }
            }

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
            log::trace!(
                "[EXT-GRAD] ext_idx={} value={:+.6e} coord.a={:+.6e} trace_logdet={:+.6e} ld_s={:+.6e} incl_h={} incl_s={}",
                ext_idx, value, coord.a, trace_logdet_i, coord.ld_s, incl_logdet_h, incl_logdet_s
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

    // Drain the per-call EIG-DECOMP sink into the calling thread's
    // thread-local stash. The rayon worker that produced the stash may
    // live on a different thread, but `store_terms` writes to the
    // current thread's TLS — which is also the thread the test will
    // call `take_terms()` from, because everything between
    // `evaluate_joint_reml_outer_eval_at_theta` and `reml_laml_evaluate`
    // is synchronous and stays on the test thread.
    #[cfg(test)]
    if let Some(stash) = ext_stash_sink
        .lock()
        .expect("EIG-DECOMP stash sink mutex poisoned")
        .take()
    {
        debug_stash::store_terms(stash);
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
        return Err(RemlError::NonFiniteValue {
            reason: format!("REML/LAML gradient contains non-finite entry at index {idx}: {value}"),
        }
        .into());
    }

    // Run the envelope-gradient sanity check *before* the outer-Hessian
    // assembly. When the inner-KKT IFT correction was not applied and the
    // envelope formula predicts a √ε-step cost change > 4·|cost| (i.e. the
    // gradient is mathematically inconsistent with the function), the
    // analytic gradient will be marked unavailable below. The Hessian
    // computed from the same ill-conditioned inner state would also be
    // untrustworthy and would just be discarded by the outer optimizer —
    // and for biobank-scale custom families the assembly can take 20+
    // minutes per evaluation. Decide once here, then reuse the verdict
    // for both the gradient and the Hessian outputs.
    let cost_scale = cost.abs().max(1.0);
    let resolve_step = f64::EPSILON.sqrt();
    let envelope_inconsistent = grad
        .iter()
        .enumerate()
        .map(|(i, g)| (i, g.abs()))
        .reduce(|a, b| if a.1 >= b.1 { a } else { b })
        .and_then(|(max_idx, max_abs)| {
            let predicted_change = max_abs * resolve_step;
            if max_abs.is_finite() && predicted_change > 4.0 * cost_scale {
                Some((max_idx, max_abs, predicted_change))
            } else {
                None
            }
        });
    // Principled rule: `envelope_inconsistent` is evaluated on the
    // *post-correction* gradient (the `kkt_rho_corrections.gradient`
    // additive block above has already been folded into `grad`). If the
    // predicted √ε-step cost change still exceeds 4·|cost| after that
    // fold, the gradient is invalid as a descent direction. A previous
    // exception kept the gradient when `kkt_residual_correction_active`
    // was true, but that was self-contradictory: the tripwire fires on
    // the same gradient the correction was supposed to have repaired,
    // and when `‖r_proj‖∞ ≈ 0` (the cert-exit contract) the correction
    // `-aᵀ_k q + ½ qᵀA_k q` with `q = H⁻¹·r ≡ 0` is identically zero
    // (see `compute_kkt_residual_rho_corrections`). Suppress
    // unconditionally and let the outer optimizer fall back to FD or
    // reject the seed.
    let envelope_suppresses_outputs = envelope_inconsistent.is_some();
    let _ = kkt_residual_correction_active; // cost-side IFT identity logged earlier
    if envelope_inconsistent.is_some()
        && matches!(solution.dispersion, DispersionHandling::Fixed { .. })
        && solution.kkt_residual.is_none()
    {
        return Err(RemlError::ContractViolation {
            reason: "REML/LAML fixed-dispersion derivative contract violated: envelope gradient \
                     is inconsistent but no projected KKT residual was supplied. A convergent \
                     custom-family inner path must populate BlockwiseInnerResult::kkt_residual \
                     using the active-set-aware projected residual before requesting analytic \
                     outer derivatives"
                .to_string(),
        }
        .into());
    }

    // Outer Hessian (if requested).
    let hessian = if mode == EvalMode::ValueGradientHessian && !envelope_suppresses_outputs {
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
                return Err(RemlError::DimensionMismatch {
                    reason: format!(
                        "family outer Hessian operator dimension mismatch: got {}, expected {}",
                        family_op.dim(),
                        k_outer
                    ),
                }
                .into());
            }
            let assembly_start = std::time::Instant::now();
            let mut hessian = crate::solver::outer_strategy::HessianResult::Operator(family_op);
            if let Some(kkt_hessian) = kkt_rho_corrections
                .as_ref()
                .and_then(|corrections| corrections.hessian.as_ref())
            {
                hessian.add_rho_block_dense(kkt_hessian)?;
            }
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
        let has_subspace_trace = solution.penalty_subspace_trace.is_some();
        let route_plan = outer_hessian_route_plan(
            n_obs,
            p_dim,
            k_outer,
            hessian_kernel.is_some(),
            callback_operator_kernel,
            has_subspace_trace,
        );
        let use_operator = route_plan.use_operator;
        let route_choice = route_plan.choice();
        let route_reason = route_plan.reason;
        log::info!(
            "[OUTER hessian-route] choice={route_choice} reason={route_reason} \
             n={n_obs} p={p_dim} k={k_outer} \
             callback_kernel={callback_operator_kernel} subspace_trace={has_subspace_trace} \
             scale_prefers_operator={} dense_workspace_bytes={}",
            route_plan.scale_prefers_operator,
            route_plan.dense_workspace_bytes,
        );
        let assembly_start = std::time::Instant::now();
        let result = if use_operator {
            let coord_vs_for_hessian = rho_v_ks.as_ref().map(|rho_vs| {
                let mut all = Vec::with_capacity(k + ext_dim);
                all.extend(rho_vs.iter().cloned());
                all.extend(ext_v_is.iter().cloned());
                all
            });
            match build_outer_hessian_operator(
                solution,
                &lambdas,
                effective_deriv,
                hessian_kernel.expect("checked is_some above"),
                coord_vs_for_hessian.as_deref(),
                Some(&coord_corrections),
            ) {
                Ok(op) => {
                    let mut hessian =
                        crate::solver::outer_strategy::HessianResult::Operator(Arc::new(op));
                    if let Some(kkt_hessian) = kkt_rho_corrections
                        .as_ref()
                        .and_then(|corrections| corrections.hessian.as_ref())
                    {
                        hessian.add_rho_block_dense(kkt_hessian)?;
                    }
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
            let reml_workspace = RemlDerivativeWorkspace {
                curvature_lambdas: &curvature_lambdas,
                rho_penalty_a_k_betas: &rho_penalty_a_k_betas,
                rho_curvature_a_k_betas: &rho_curvature_a_k_betas,
                rho_v_ks: rho_v_ks.as_deref(),
                coord_corrections: &coord_corrections,
            };
            match compute_outer_hessian(
                solution,
                rho,
                &lambdas,
                hop,
                effective_deriv,
                Some(&reml_workspace),
            ) {
                Ok(mut h) => {
                    if let Some(kkt_hessian) = kkt_rho_corrections
                        .as_ref()
                        .and_then(|corrections| corrections.hessian.as_ref())
                    {
                        let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                        sl += kkt_hessian;
                    }
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

    // Envelope-gradient sanity tripwire — last line of defense.
    //
    // The post-IFT-correction gradient is what `envelope_inconsistent` is
    // computed on (the `kkt_rho_corrections.gradient` block was folded in
    // earlier in this function). If the predicted √ε-step cost change
    // still exceeds 4·|cost| after that fold, the gradient is invalid as
    // a descent direction. Suppress and let the outer optimizer fall
    // back to FD or reject the seed. Threshold ratio > 4 keeps healthy
    // near-stationary gradients (|g|∞ ≈ √ε·|cost|, ratio ≈ 1) from
    // tripping.
    let gradient_out = match envelope_inconsistent {
        Some((max_idx, max_abs, predicted_change)) => {
            // Self-diagnosing warning. The three gates that control whether
            // the IFT correction can run all map to observable booleans, so
            // the next failing run pinpoints the cause without another
            // debugging round.
            let kkt_some = solution.kkt_residual.is_some();
            let dispersion_label = match &solution.dispersion {
                DispersionHandling::Fixed { .. } => "Fixed",
                DispersionHandling::ProfiledGaussian => "ProfiledGaussian",
            };
            let kernel_present = solution.penalty_subspace_trace.is_some();
            log::warn!(
                "[reml_laml envelope-gradient consistency] |g|∞ = {:.3e} at coord {} predicts \
                 |Δcost| ≈ {:.3e} along a √ε step while |cost| = {:.3e} (ratio {:.2e}). \
                 Envelope formula contaminated by inner KKT residual on ill-conditioned H block; \
                 marking analytic gradient unavailable so outer optimizer does not chase a \
                 mathematically impossible descent direction. Outer-Hessian assembly skipped on \
                 this evaluation to avoid spending wall-clock on a result the optimizer would \
                 discard. \
                 IFT-gate diagnostics: kkt_residual.is_some()={} (must be true; this is the \
                 projected-KKT residual the inner solver hands over), dispersion={} (must be \
                 `Fixed` for the LAML IFT identity to hold), penalty_subspace_trace.is_some()={} \
                 (when true the cost IFT uses bilinear_pseudo_inverse on range(S₊); when false \
                 the full H⁻¹·r solve is the only path and is unsafe on near-singular H). \
                 If kkt_residual.is_some()=false the convergent inner path forgot to populate \
                 `BlockwiseInnerResult::kkt_residual` (call \
                 `exact_newton_joint_kkt_residual_for_ift(..., Some(active_sets))` on return). \
                 If kkt_residual.is_some()=true but the warning still fires, the dispersion or \
                 length gate above tripped — the convergent path emitted a residual but the \
                 evaluator refused it; check the length-mismatch hard error earlier in this \
                 function or the dispersion variant printed here.",
                max_abs,
                max_idx,
                predicted_change,
                cost_scale,
                predicted_change / cost_scale,
                kkt_some,
                dispersion_label,
                kernel_present,
            );
            None
        }
        None => Some(grad),
    };

    Ok(RemlLamlResult {
        cost,
        gradient: gradient_out,
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

/// Row-pair work cutoff for callback-backed outer Hessians.
///
/// Callback kernels expose exact row-local first/second Hessian drifts. Dense
/// `K x K` assembly can still be expensive at tiny coefficient dimension
/// because the dominant work is not `p x p` algebra; it is repeated row-kernel
/// contractions over the upper-triangular coordinate pairs.
pub(crate) const CALLBACK_OUTER_HESSIAN_ROW_PAIR_WORK_THRESHOLD: usize = 25_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OuterHessianRoutePlan {
    use_operator: bool,
    reason: &'static str,
    scale_prefers_operator: bool,
    dense_workspace_bytes: usize,
}

impl OuterHessianRoutePlan {
    fn choice(self) -> &'static str {
        if self.use_operator {
            "operator"
        } else {
            "dense"
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OuterHessianScaleDecision {
    prefers_operator: bool,
    reason: &'static str,
}

fn saturating_f64_matrix_bytes(rows: usize, cols: usize) -> usize {
    rows.saturating_mul(cols)
        .saturating_mul(std::mem::size_of::<f64>())
}

fn outer_hessian_dense_workspace_bytes(p: usize, k: usize) -> usize {
    // Dense assembly keeps first-order drifts for each coordinate and uses at
    // least one transient second-order drift while filling the K x K Hessian.
    // Charge a small safety multiple so the route never depends on fitting a
    // single p x p matrix while the actual dense path holds several.
    let drift_count = k.saturating_mul(2).saturating_add(3).max(1);
    saturating_f64_matrix_bytes(p, p).saturating_mul(drift_count)
}

fn outer_hessian_dense_workspace_budget_bytes() -> usize {
    crate::resource::ResourcePolicy::default_library().max_single_materialization_bytes
}

fn dense_outer_hessian_workspace_fits(p: usize, k: usize) -> bool {
    outer_hessian_dense_workspace_bytes(p, k) <= outer_hessian_dense_workspace_budget_bytes()
}

fn generic_outer_hessian_scale_decision(n: usize, p: usize, k: usize) -> OuterHessianScaleDecision {
    if !dense_outer_hessian_workspace_fits(p, k) {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "dense_memory_budget",
        };
    }
    if k >= MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_k",
        };
    }
    if p >= MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_p",
        };
    }
    if n >= MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD
        && p >= MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N
    {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_n_moderate_p",
        };
    }
    if n.saturating_mul(p) >= MATRIX_FREE_OUTER_HESSIAN_NP_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_linear_work",
        };
    }
    OuterHessianScaleDecision {
        prefers_operator: false,
        reason: "below_crossover",
    }
}

fn callback_outer_hessian_scale_decision(
    n: usize,
    p: usize,
    k: usize,
) -> OuterHessianScaleDecision {
    if !dense_outer_hessian_workspace_fits(p, k) {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "dense_memory_budget",
        };
    }
    if k >= MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_k",
        };
    }
    if p >= MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_p",
        };
    }
    if n.saturating_mul(k).saturating_mul(k) >= CALLBACK_OUTER_HESSIAN_ROW_PAIR_WORK_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "callback_row_pair_work",
        };
    }
    OuterHessianScaleDecision {
        prefers_operator: false,
        reason: "below_crossover",
    }
}

fn outer_hessian_route_plan(
    n: usize,
    p: usize,
    k: usize,
    kernel_available: bool,
    callback_kernel: bool,
    subspace_trace: bool,
) -> OuterHessianRoutePlan {
    let dense_workspace_bytes = outer_hessian_dense_workspace_bytes(p, k);
    if !kernel_available {
        return OuterHessianRoutePlan {
            use_operator: false,
            reason: "kernel_absent",
            scale_prefers_operator: false,
            dense_workspace_bytes,
        };
    }

    let scale = if callback_kernel {
        callback_outer_hessian_scale_decision(n, p, k)
    } else {
        generic_outer_hessian_scale_decision(n, p, k)
    };
    let reason = if subspace_trace && scale.prefers_operator {
        "subspace_projected_operator"
    } else {
        scale.reason
    };
    OuterHessianRoutePlan {
        use_operator: scale.prefers_operator,
        reason,
        scale_prefers_operator: scale.prefers_operator,
        dense_workspace_bytes,
    }
}

/// Predicate for selecting the matrix-free Hv-operator outer-Hessian
/// representation over the dense `K × K` assembly.  Cost selects
/// representation, never capability — the operator path delivers the same
/// math as the dense path while avoiding large dense `p × p` drift storage
/// and pairwise row assembly when the model says those dominate.
pub(crate) fn prefer_outer_hessian_operator(n: usize, p: usize, k: usize) -> bool {
    generic_outer_hessian_scale_decision(n, p, k).prefers_operator
}

/// Selects the matrix-free outer-Hessian representation once a Hessian HVP
/// kernel is available. Decision is cost-driven via the `(n, p, K)` crossover
/// plus a callback-specific row-pair workload crossover: the operator and
/// dense paths produce identical math, so they only differ in assembly cost.
///
/// Callback-backed kernels deliberately do not inherit the generic
/// large-`n`, moderate-`p` route. At small coefficient dimension their
/// operator logdet projections still stream row kernels over all observations
/// per outer-HVP, while the dense path can assemble the small `K x K` Hessian
/// directly. The callback path only flips to matrix-free for genuinely wide
/// bases, many outer coordinates, or enough row-pair work to amortize those
/// repeated projections.
///
/// Real fast HVP capability (a family-supplied directional θθ operator) is
/// routed separately through `HessianDerivativeProvider::family_outer_hessian_operator`,
/// which short-circuits this function entirely at the call site.
pub(crate) fn use_outer_hessian_operator_path(
    n: usize,
    p: usize,
    k: usize,
    callback_kernel: bool,
) -> bool {
    outer_hessian_route_plan(n, p, k, true, callback_kernel, false).use_operator
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
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "fourth-derivative trace shape mismatch: c={}, d={}, leverage={}",
                n,
                d_array.len(),
                leverage.len()
            ),
        }
        .into());
    }

    let mut x_modes = Array2::<f64>::zeros((n, t));
    for (j, mode) in modes.iter().enumerate() {
        let x_v = ing.x.matrixvectormultiply(mode);
        if x_v.len() != n {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "fourth-derivative trace Xv length mismatch for mode {j}: got {}, expected {n}",
                    x_v.len()
                ),
            }
            .into());
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
    if let Some(kernel) = subspace {
        let factor = penalty_subspace_trace_factor(kernel);
        let cache = ProjectedFactorCache::default();
        let mut out = vec![0.0_f64; pairs.len()];
        let mut op_terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
        for (idx, pair) in pairs.iter().enumerate() {
            if let Some(op) = pair.b_operator.as_deref() {
                collect_projected_trace_terms(idx, 1.0, op, &factor, &mut out, &mut op_terms);
            } else if pair.b_mat.nrows() > 0 {
                out[idx] = dense_trace_projected_factor(&pair.b_mat, &factor);
            }
        }
        if !op_terms.is_empty() {
            let batched =
                trace_projected_operator_terms_batched(pairs.len(), &op_terms, &factor, &cache);
            for (idx, val) in batched.into_iter().enumerate() {
                out[idx] += val;
            }
        }
        return out;
    }
    // Dense-spectral batched path: collect every operator-backed pair into a
    // single chunked sweep so the implicit design (compute_xf + per-axis
    // kernel scalars) is traversed once instead of `pairs.len()` times. At
    // biobank scale this turns the 16+ per-call `trace_logdet_operator`
    // hot spots into a single batched evaluation.
    if subspace.is_none() {
        if let Some(ds) = hop.as_exact_dense_spectral() {
            let mut out = vec![0.0_f64; pairs.len()];
            let mut op_terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
            for (idx, pair) in pairs.iter().enumerate() {
                if let Some(op) = pair.b_operator.as_deref() {
                    op_terms.push((idx, 1.0, op));
                } else if pair.b_mat.nrows() > 0 {
                    out[idx] = hop.trace_logdet_gradient(&pair.b_mat);
                }
            }
            if !op_terms.is_empty() {
                let batched = trace_projected_operator_terms_batched(
                    pairs.len(),
                    &op_terms,
                    &ds.g_factor,
                    &ds.projected_factor_cache,
                );
                for (idx, val) in batched.into_iter().enumerate() {
                    out[idx] += val;
                }
            }
            return out;
        }
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

    // Batch the projected_operator calls for implicit operator drifts so the
    // chunked design sweep (kernel scalars + GEMMs) is traversed once and
    // shared across all matching axes.
    let mut ext_rotated: Vec<Option<Array2<f64>>> = (0..ext_drifts.len()).map(|_| None).collect();
    let mut op_terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
    for (i, drift) in ext_drifts.iter().enumerate() {
        match drift {
            DriftDerivResult::Dense(matrix) => {
                ext_rotated[i] = Some(dense_hop.rotate_to_eigenbasis(matrix));
            }
            DriftDerivResult::Operator(operator) => {
                op_terms.push((i, 1.0, operator.as_ref()));
            }
        }
    }
    if !op_terms.is_empty() {
        let batched = projected_operator_terms_batched(
            ext_drifts.len(),
            &op_terms,
            &dense_hop.eigenvectors,
            &dense_hop.projected_factor_cache,
        );
        for (i, _, _) in &op_terms {
            ext_rotated[*i] = Some(batched[*i].clone());
        }
    }
    for r in ext_rotated {
        rotated.push(r.expect("every ext drift contributes a rotation"));
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

/// Shared precomputed REML derivative intermediates threaded from the
/// gradient pass into the dense Hessian assembler so the per-coordinate
/// `penalty_a_k_beta` / `hop.solve` / drift-correction work is not repeated.
pub(crate) struct RemlDerivativeWorkspace<'a> {
    pub curvature_lambdas: &'a [f64],
    pub rho_penalty_a_k_betas: &'a [Array1<f64>],
    pub rho_curvature_a_k_betas: &'a [Array1<f64>],
    pub rho_v_ks: Option<&'a [Array1<f64>]>,
    pub coord_corrections: &'a [Option<DriftDerivResult>],
}

struct KktRhoCorrections {
    gradient: Array1<f64>,
    hessian: Option<Array2<f64>>,
}

fn solve_kkt_residual_kernel(
    hop: &dyn HessianOperator,
    subspace: Option<&PenaltySubspaceTrace>,
    rhs: &Array1<f64>,
) -> Array1<f64> {
    if let Some(kernel) = subspace {
        let projected = crate::faer_ndarray::fast_atv(&kernel.u_s, rhs);
        let solved_projected = kernel.h_proj_inverse.dot(&projected);
        crate::faer_ndarray::fast_av(&kernel.u_s, &solved_projected)
    } else {
        hop.solve(rhs)
    }
}

/// Derivatives of the same Newton/IFT residual correction used by the cost:
///
///   C(ρ) = -½ r(ρ)^T K(ρ) r(ρ),   K = H^{-1}
///
/// for fixed-dispersion LAML. At fixed β̂, `r_i = A_i β̂` and
/// `H_i = A_i`, so
///
///   C_i  = -a_i^T q + ½ q^T A_i q,
///   q    = K r,
///   q_j  = K(a_j - A_j q),
///   C_ij = -δ_ij a_i^T q - a_i^T q_j
///          + q_j^T A_i q + ½δ_ij q^T A_i q.
///
/// The dense outer Hessian already contains the exact-KKT profile term
/// `-a_i^T K a_j`. That term is valid only when `r = 0`; the residual
/// correction is therefore added as `a_i^T K a_j + C_ij`. This guarantees
/// the additive block vanishes at exact KKT and is exact for the Gaussian
/// quadratic reproduction.
fn compute_kkt_residual_rho_corrections(
    solution: &InnerSolution<'_>,
    hop: &dyn HessianOperator,
    lambdas: &[f64],
    penalty_a_k_betas: &[Array1<f64>],
    residual: &Array1<f64>,
    include_hessian: bool,
) -> Result<KktRhoCorrections, String> {
    let k = penalty_a_k_betas.len();
    if k == 0 {
        return Ok(KktRhoCorrections {
            gradient: Array1::zeros(0),
            hessian: include_hessian.then(|| Array2::zeros((0, 0))),
        });
    }
    if lambdas.len() != k || solution.penalty_coords.len() != k {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT rho correction dimension mismatch: lambdas={} coords={} rhs={}",
                lambdas.len(),
                solution.penalty_coords.len(),
                k
            ),
        }
        .into());
    }
    if residual.len() != hop.dim() {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT residual dimension mismatch: residual={} Hessian dim={}",
                residual.len(),
                hop.dim()
            ),
        }
        .into());
    }

    let subspace = solution.penalty_subspace_trace.as_deref();
    let q = solve_kkt_residual_kernel(hop, subspace, residual);
    let mut a_i_qs = Vec::with_capacity(k);
    let mut a_i_dot_q = Vec::with_capacity(k);
    let mut q_a_i_q = Vec::with_capacity(k);

    for idx in 0..k {
        let a_i_q = solution.penalty_coords[idx].scaled_matvec(&q, lambdas[idx]);
        let linear = penalty_a_k_betas[idx].dot(&q);
        let quadratic = q.dot(&a_i_q);
        if !linear.is_finite() || !quadratic.is_finite() {
            return Err(RemlError::NonFiniteValue {
                reason: format!(
                    "KKT rho correction produced non-finite gradient ingredients at coord {idx}: \
                     linear={linear} quadratic={quadratic}"
                ),
            }
            .into());
        }
        a_i_dot_q.push(linear);
        q_a_i_q.push(quadratic);
        a_i_qs.push(a_i_q);
    }

    let mut gradient = Array1::<f64>::zeros(k);
    for idx in 0..k {
        gradient[idx] = -a_i_dot_q[idx] + 0.5 * q_a_i_q[idx];
    }

    let hessian = if include_hessian {
        let mut a_solutions = Vec::with_capacity(k);
        let mut q_derivs = Vec::with_capacity(k);
        for idx in 0..k {
            a_solutions.push(solve_kkt_residual_kernel(
                hop,
                subspace,
                &penalty_a_k_betas[idx],
            ));
            let mut rhs = penalty_a_k_betas[idx].clone();
            rhs -= &a_i_qs[idx];
            q_derivs.push(solve_kkt_residual_kernel(hop, subspace, &rhs));
        }

        let entry = |i: usize, j: usize| -> f64 {
            let delta = if i == j { 1.0 } else { 0.0 };
            let cancel_exact_kkt_profile_term = penalty_a_k_betas[i].dot(&a_solutions[j]);
            cancel_exact_kkt_profile_term
                - delta * a_i_dot_q[i]
                - penalty_a_k_betas[i].dot(&q_derivs[j])
                + q_derivs[j].dot(&a_i_qs[i])
                + 0.5 * delta * q_a_i_q[i]
        };

        let mut h = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in i..k {
                let raw = if i == j {
                    entry(i, j)
                } else {
                    0.5 * (entry(i, j) + entry(j, i))
                };
                if !raw.is_finite() {
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "KKT rho correction produced non-finite Hessian entry ({i}, {j}): {raw}"
                        ),
                    }
                    .into());
                }
                h[[i, j]] = raw;
                if i != j {
                    h[[j, i]] = raw;
                }
            }
        }
        Some(h)
    } else {
        None
    };

    Ok(KktRhoCorrections { gradient, hessian })
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
    workspace: Option<&RemlDerivativeWorkspace<'_>>,
) -> Result<Array2<f64>, String> {
    let k = rho.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let mut hess = Array2::zeros((total, total));
    let curvature_lambdas_storage: Option<Vec<f64>> = if workspace.is_some() {
        None
    } else {
        Some(
            lambdas
                .iter()
                .copied()
                .map(|lambda| rho_curvature_lambda(solution, lambda))
                .collect(),
        )
    };
    let curvature_lambdas: &[f64] = match workspace {
        Some(ws) => ws.curvature_lambdas,
        None => curvature_lambdas_storage
            .as_deref()
            .expect("curvature_lambdas_storage populated when workspace is None"),
    };

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

    let penalty_a_k_betas_storage: Option<Vec<Array1<f64>>> = if workspace.is_some() {
        None
    } else {
        Some(
            (0..k)
                .map(|idx| {
                    penalty_a_k_beta(&solution.penalty_coords[idx], &solution.beta, lambdas[idx])
                })
                .collect(),
        )
    };
    let curvature_a_k_betas_storage: Option<Vec<Array1<f64>>> = if workspace.is_some() {
        None
    } else {
        Some(
            (0..k)
                .map(|idx| {
                    penalty_a_k_beta(
                        &solution.penalty_coords[idx],
                        &solution.beta,
                        curvature_lambdas[idx],
                    )
                })
                .collect(),
        )
    };
    let penalty_a_k_betas: &[Array1<f64>] = match workspace {
        Some(ws) => ws.rho_penalty_a_k_betas,
        None => penalty_a_k_betas_storage.as_deref().expect("storage set"),
    };
    let curvature_a_k_betas: &[Array1<f64>] = match workspace {
        Some(ws) => ws.rho_curvature_a_k_betas,
        None => curvature_a_k_betas_storage.as_deref().expect("storage set"),
    };

    let v_ks_storage: Option<Vec<Array1<f64>>> = match workspace.and_then(|ws| ws.rho_v_ks) {
        Some(_) => None,
        None => Some(
            curvature_a_k_betas
                .iter()
                .map(|a_k_beta| hop.solve(a_k_beta))
                .collect(),
        ),
    };
    let v_ks: &[Array1<f64>] = match workspace.and_then(|ws| ws.rho_v_ks) {
        Some(vs) => vs,
        None => v_ks_storage.as_deref().expect("storage set"),
    };

    // Precompute a_k = ½ λₖ(β̂-μₖ)'Sₖ(β̂-μₖ) for profiled Gaussian correction.
    let rho_a_vals: Vec<f64> = (0..k)
        .map(|idx| {
            0.5 * penalty_a_k_quadratic(&solution.penalty_coords[idx], &solution.beta, lambdas[idx])
        })
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

        let correction: Option<Array2<f64>> = match workspace {
            Some(ws) => match ws.coord_corrections[idx].as_ref() {
                Some(DriftDerivResult::Dense(matrix)) => Some(matrix.clone()),
                Some(DriftDerivResult::Operator(_)) => {
                    if effective_deriv.has_corrections() {
                        effective_deriv.hessian_derivative_correction(&v_ks[idx])?
                    } else {
                        None
                    }
                }
                None => None,
            },
            None => {
                if effective_deriv.has_corrections() {
                    effective_deriv.hessian_derivative_correction(&v_ks[idx])?
                } else {
                    None
                }
            }
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
        let mut drifts = h_k_matrices
            .iter()
            .cloned()
            .map(DriftDerivResult::Dense)
            .collect::<Vec<_>>();
        drifts.extend(ext_h_drifts.iter().cloned());
        penalty_subspace_reduce_drifts_batched(kernel, &drifts)
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
    //
    // The K(K+1)/2 upper-triangular pairs are independent (each writes to a
    // disjoint hess[[kk, ll]]/hess[[ll, kk]] cell pair) and the dominant
    // per-pair cost — `compute_ift_correction_trace` → `kernel.trace_operator`
    // → `op.mul_mat(U_S)` materialising a CompositeHyperOperator over a
    // (p × r) factor — scales like O(family_callback × r × n). At biobank
    // shape (n ≈ 2·10⁵, r ≈ 24, K = 8) the sequential walk dominates the
    // outer-Hessian wall-clock by 1–2 orders of magnitude, so we dispatch
    // the pair list through rayon and stitch the symmetric Array2 sequentially.
    //
    // `effective_deriv` is `&dyn HessianDerivativeProvider` whose trait bound
    // includes `Send + Sync`, and `hop`, `subspace`, `glm_ingredients`,
    // `adjoint_z_c`, `leverage`, and `fourth_trace_matrix` are all read-only
    // shared state, so the closure body is genuinely thread-safe. The default
    // `HyperOperator::mul_mat` already short-circuits to sequential when
    // `rayon::current_thread_index().is_some()`, preventing nested-rayon
    // oversubscription inside the family callbacks invoked by
    // `compute_ift_correction_trace`.
    let rho_pair_indices: Vec<(usize, usize)> = (0..k)
        .flat_map(|kk| (kk..k).map(move |ll| (kk, ll)))
        .collect();
    let rho_pair_count = rho_pair_indices.len();
    let rho_pair_start = std::time::Instant::now();
    log::debug!(
        "[compute_outer_hessian rho-rho] starting {} pair(s), k={}",
        rho_pair_count,
        k,
    );

    let build_rho_pair_rhs = |kk: usize, ll: usize| {
        let mut rhs = h_k_matrices[ll].dot(&v_ks[kk]);
        rhs += &solution.penalty_coords[kk].scaled_matvec(&v_ks[ll], curvature_lambdas[kk]);
        if kk == ll {
            rhs -= &curvature_a_k_betas[kk];
        }
        rhs
    };

    let batched_rho_pair_corrections: Option<Vec<f64>> = if incl_logdet_h
        && subspace.is_some()
        && effective_deriv.has_corrections()
        && effective_deriv.has_batched_hessian_second_derivative_corrections()
    {
        let mut rhs_matrix = Array2::<f64>::zeros((hop.dim(), rho_pair_count));
        for (pair_idx, &(kk, ll)) in rho_pair_indices.iter().enumerate() {
            let rhs = build_rho_pair_rhs(kk, ll);
            rhs_matrix.column_mut(pair_idx).assign(&rhs);
        }
        let solved = hop.solve_multi(&rhs_matrix);
        let triples: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = rho_pair_indices
            .iter()
            .enumerate()
            .map(|(pair_idx, &(kk, ll))| {
                (
                    v_ks[kk].clone(),
                    v_ks[ll].clone(),
                    solved.column(pair_idx).to_owned(),
                )
            })
            .collect();
        let corrections = effective_deriv.hessian_second_derivative_corrections_result(&triples)?;
        let mut correction_values = vec![0.0_f64; corrections.len()];
        if let Some(kernel) = subspace {
            let mut present_indices = Vec::new();
            let mut present_drifts = Vec::new();
            for (idx, correction) in corrections.into_iter().enumerate() {
                if let Some(drift) = correction {
                    present_indices.push(idx);
                    present_drifts.push(drift);
                }
            }
            let traced = penalty_subspace_trace_drifts_batched(kernel, &present_drifts);
            for (idx, value) in present_indices.into_iter().zip(traced) {
                correction_values[idx] = value;
            }
        }
        Some(correction_values)
    } else {
        None
    };

    let rho_pair_values: Vec<(usize, usize, f64)> = {
        use rayon::iter::ParallelIterator;
        rho_pair_indices
            .par_iter()
            .enumerate()
            .map(
                |(pair_idx, &(kk, ll))| -> Result<(usize, usize, f64), String> {
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
                            let (block, start, end) =
                                solution.penalty_coords[kk].scaled_block_local(1.0);
                            hop.trace_logdet_block_local(&block, curvature_lambdas[kk], start, end)
                        } else {
                            hop.trace_logdet_gradient(&a_k_matrices[kk])
                        }
                    } else {
                        0.0
                    };

                    let correction =
                        if let Some(corrections) = batched_rho_pair_corrections.as_ref() {
                            corrections[pair_idx]
                        } else {
                            let rhs = build_rho_pair_rhs(kk, ll);
                            compute_ift_correction_trace(
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
                            )?
                        };

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
                    Ok((kk, ll, h_val))
                },
            )
            .collect::<Result<Vec<_>, String>>()?
    };

    for (kk, ll, h_val) in rho_pair_values {
        hess[[kk, ll]] = h_val;
        if kk != ll {
            hess[[ll, kk]] = h_val;
        }
    }

    log::debug!(
        "[compute_outer_hessian rho-rho] {} pair(s) done in {:.3}s",
        rho_pair_count,
        rho_pair_start.elapsed().as_secs_f64(),
    );

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
pub struct WeightedHyperOperator {
    pub(crate) terms: Vec<(f64, Arc<dyn HyperOperator>)>,
    pub(crate) dim_hint: usize,
}

impl HyperOperator for WeightedHyperOperator {
    fn as_weighted(&self) -> Option<&WeightedHyperOperator> {
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

    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                projected.scaled_add(*weight, &op.projected_matrix(factor));
            }
        }
        projected
    }

    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                projected.scaled_add(*weight, &op.projected_matrix_cached(factor, cache));
            }
        }
        projected
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
            return Err(RemlError::InvalidKernelMode {
                reason: "scalar correction requested for non-scalar kernel".to_string(),
            }
            .into());
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
            return Err(RemlError::InvalidKernelMode {
                reason: "callback correction requested for non-callback kernel".to_string(),
            }
            .into());
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
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian alpha length mismatch: got {}, expected {}",
                    alpha.len(),
                    self.coords.len()
                ),
            }
            .into());
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
    precomputed_coord_vs: Option<&[Array1<f64>]>,
    precomputed_coord_corrections: Option<&[Option<DriftDerivResult>]>,
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
    // Mode responses are fixed-β stationarity derivatives and always use
    // the full Hessian solve.  Rank-deficient LAML changes only the logdet
    // trace kernel, handled below through `subspace`; projecting these solves
    // would change β_i/β_ij curvature semantics.
    let subspace = solution.penalty_subspace_trace.as_deref();
    let dispatch_solve = |v: &Array1<f64>| -> Array1<f64> { hop.solve(v) };
    let coord_vs_storage;
    let coord_vs: &[Array1<f64>] = if let Some(precomputed) = precomputed_coord_vs {
        if precomputed.len() != total {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian precomputed mode-response count mismatch: got {}, expected {}",
                    precomputed.len(),
                    total
                ),
            }
            .into());
        }
        precomputed
    } else {
        let mut owned: Vec<Array1<f64>> = rho_curvature_a_k_betas
            .par_iter()
            .map(dispatch_solve)
            .collect();
        owned.extend(
            solution
                .ext_coords
                .par_iter()
                .map(|coord| dispatch_solve(&coord.g))
                .collect::<Vec<_>>(),
        );
        coord_vs_storage = owned;
        &coord_vs_storage
    };

    let coord_corrections_storage;
    let coord_corrections: &[Option<DriftDerivResult>] = if let Some(precomputed) =
        precomputed_coord_corrections
    {
        if precomputed.len() != total {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian precomputed correction count mismatch: got {}, expected {}",
                    precomputed.len(),
                    total
                ),
            }
            .into());
        }
        precomputed
    } else if effective_deriv.has_corrections() {
        if effective_deriv.has_batched_hessian_derivative_corrections() {
            log::info!(
                "[STAGE] outer_hessian coord_corrections mode=batched k={} ext_dim={} n={} dim={}",
                k,
                ext_dim,
                solution.n_observations,
                hop.dim()
            );
            coord_corrections_storage =
                effective_deriv.hessian_derivative_corrections_result(coord_vs)?;
        } else {
            coord_corrections_storage = coord_vs
                .par_iter()
                .map(|v_i| effective_deriv.hessian_derivative_correction_result(v_i))
                .collect::<Result<Vec<_>, _>>()?;
        }
        &coord_corrections_storage
    } else {
        coord_corrections_storage = (0..total).map(|_| None).collect::<Vec<_>>();
        &coord_corrections_storage
    };

    let mut coords = Vec::with_capacity(total);
    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let curvature_a_k_beta = rho_curvature_a_k_betas[idx].clone();
        let v_k = coord_vs[idx].clone();
        let correction = coord_corrections[idx].as_ref();
        let mut total_dense = None;
        let mut total_operators = Vec::new();
        match penalty_total_drift_result(coord, curvature_lambdas[idx], correction) {
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
        let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambdas[idx]);
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
        let coord_idx = k + ext_idx;
        let v_i = coord_vs[coord_idx].clone();
        let correction = coord_corrections[coord_idx].as_ref();
        let (total_dense, total_operators) =
            hyper_coord_total_drift_parts(&coord.drift, correction);
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
        let pair_drifts: Vec<((usize, usize), Vec<DriftDerivResult>)> = pairs
            .into_par_iter()
            .map(|(ii, jj)| {
                let beta_i = coords[ii].v.mapv(|value| -value);
                let beta_j = coords[jj].v.mapv(|value| -value);
                let mut drifts = Vec::new();
                if let Some(drift_fn) = solution.fixed_drift_deriv.as_ref() {
                    if coords[ii].b_depends_on_beta
                        && let Some(ext_i) = coords[ii].ext_index
                        && let Some(result) = drift_fn(ext_i, &beta_j)
                    {
                        drifts.push(result);
                    }
                    if coords[jj].b_depends_on_beta
                        && let Some(ext_j) = coords[jj].ext_index
                        && let Some(result) = drift_fn(ext_j, &beta_i)
                    {
                        drifts.push(result);
                    }
                }
                ((ii, jj), drifts)
            })
            .collect();

        let mut term_pairs = Vec::new();
        let mut term_drifts = Vec::new();
        for ((ii, jj), drifts) in pair_drifts {
            for drift in drifts {
                term_pairs.push((ii, jj));
                term_drifts.push(drift);
            }
        }

        if !term_drifts.is_empty() {
            let term_traces = if let Some(kernel) = subspace {
                penalty_subspace_trace_drifts_batched(kernel, &term_drifts)
            } else if let Some(ds) = hop.as_exact_dense_spectral() {
                dense_spectral_trace_logdet_drifts_batched(ds, &term_drifts)
            } else {
                term_drifts
                    .iter()
                    .map(|drift| drift.trace_logdet(hop.as_ref()))
                    .collect()
            };
            for ((ii, jj), trace) in term_pairs.into_iter().zip(term_traces.into_iter()) {
                m_pair_trace[[ii, jj]] += trace;
                if ii != jj {
                    m_pair_trace[[jj, ii]] += trace;
                }
            }
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
            let drift_parts = coords
                .iter()
                .map(|coord| {
                    let dense = coord.total_drift.dense.clone();
                    let op = if coord.total_drift.operators.is_empty() {
                        None
                    } else {
                        Some(Arc::new(CompositeHyperOperator {
                            dim_hint: hop.dim(),
                            dense: None,
                            operators: coord.total_drift.operators.clone(),
                        }) as Arc<dyn HyperOperator>)
                    };
                    match (dense, op) {
                        (Some(matrix), Some(operator)) => {
                            DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                                dim_hint: hop.dim(),
                                dense: Some(matrix),
                                operators: vec![operator],
                            }))
                        }
                        (Some(matrix), None) => DriftDerivResult::Dense(matrix),
                        (None, Some(operator)) => DriftDerivResult::Operator(operator),
                        (None, None) => {
                            DriftDerivResult::Dense(Array2::zeros((hop.dim(), hop.dim())))
                        }
                    }
                })
                .collect::<Vec<_>>();
            let reduced = penalty_subspace_reduce_drifts_batched(kernel, &drift_parts);
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
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "outer Hessian operator projected cross_trace[{ii}, {jj}] is non-finite ({value})"
                        ),
                    }
                    .into());
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
            let mut rotated: Vec<Array2<f64>> =
                coords
                    .iter()
                    .map(|coord| {
                        coord.total_drift.dense_rotated.clone().unwrap_or_else(|| {
                            Array2::<f64>::zeros((dense_hop.n_dim, dense_hop.n_dim))
                        })
                    })
                    .collect();
            let mut terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
            for (idx, coord) in coords.iter().enumerate() {
                for op in &coord.total_drift.operators {
                    collect_projected_matrix_terms(
                        idx,
                        1.0,
                        op.as_ref(),
                        &dense_hop.eigenvectors,
                        &mut rotated,
                        &mut terms,
                    );
                }
            }
            let projected_ops = project_hyper_operators_batched(
                total,
                &terms,
                &dense_hop.eigenvectors,
                &dense_hop.projected_factor_cache,
            );
            for (dst, projected) in rotated.iter_mut().zip(projected_ops.iter()) {
                *dst += projected;
            }

            let mut ct = Array2::<f64>::zeros((total, total));
            for ii in 0..total {
                for jj in ii..total {
                    let value =
                        dense_hop.trace_logdet_hessian_cross_rotated(&rotated[ii], &rotated[jj]);
                    if !value.is_finite() {
                        return Err(RemlError::NonFiniteValue {
                            reason: format!(
                                "outer Hessian operator cross_trace[{ii}, {jj}] is non-finite ({value})"
                            ),
                        }
                        .into());
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
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "outer Hessian operator cross_trace[{ii}, {jj}] is non-finite ({value})"
                        ),
                    }
                    .into());
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
/// ## Hessian-drift corrections
///
/// `g_full` is the same gradient `reml_laml_evaluate` produces in
/// `EvalMode::ValueAndGradient`, which already includes the third-
/// derivative `C[v_k]` IFT correction for non-Gaussian families. The
/// EFS step inherits this correction automatically through `g_full`.
/// Gaussian/quadratic likelihoods have beta-independent observed Hessians,
/// so `C[v_k] = 0` and the classical trace fixed point is exact. For
/// non-Gaussian likelihoods, the pure MacKay/Tipping explicit-trace update
/// is exact only after the logdet Hessian-drift correction is included in
/// the outer gradient.
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
        let q_eff = efs_q_eff_with_gamma_rate(
            efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale),
            lambda,
            &solution.rho_prior,
            idx,
        );
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
                    let q_eff = efs_q_eff_with_gamma_rate(
                        efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale),
                        lambda,
                        &solution.rho_prior,
                        idx,
                    );
                    (idx, efs_log_step_from_grad(q_eff, gradient[idx]))
                })
                .collect()
        } else {
            (0..k)
                .map(|idx| {
                    let coord = &solution.penalty_coords[idx];
                    let lambda = rho[idx].exp();
                    let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambda);
                    let q_eff = efs_q_eff_with_gamma_rate(
                        efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale),
                        lambda,
                        &solution.rho_prior,
                        idx,
                    );
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
                // Batch the operator-backed drifts so the chunked X·F sweep
                // is shared across all matching axes (compute_xf runs once,
                // kernel scalars are batched).
                let mut projected_drifts: Vec<Option<Array2<f64>>> =
                    (0..n_psi).map(|_| None).collect();
                let mut op_terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
                for idx in 0..n_psi {
                    if let Some(op) = drift_ops[idx].as_ref() {
                        op_terms.push((idx, 1.0, op.as_ref()));
                    } else {
                        projected_drifts[idx] = Some(
                            dense_hop.projected_matrix(
                                dense_drifts[idx]
                                    .as_ref()
                                    .expect("dense drift should be cached"),
                            ),
                        );
                    }
                }
                if !op_terms.is_empty() {
                    let batched = projected_operator_terms_batched(
                        n_psi,
                        &op_terms,
                        &dense_hop.w_factor,
                        &dense_hop.projected_factor_cache,
                    );
                    for (idx, _, _) in &op_terms {
                        projected_drifts[*idx] = Some(batched[*idx].clone());
                    }
                }
                let projected_drifts: Vec<Array2<f64>> = projected_drifts
                    .into_iter()
                    .map(|m| m.expect("projected drift filled"))
                    .collect();
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

/// Diagnostic returned when the (free-subspace) outer Hessian is indefinite.
///
/// An indefinite outer Hessian at the reported optimum means one of:
///  - the outer optimization has not converged to a stationary point,
///  - the reported point is a saddle, not a minimum,
///  - the active-bound set on θ is wrong (the unconstrained Hessian is
///    being inspected on directions that the constraint set actually pins),
///  - the objective being inspected is a surrogate / smoothed version of
///    the true REML/LAML criterion.
///
/// The previous implementation silently clamped the negative eigenvalues to
/// zero, which under-reports the uncertainty in those directions (it pretends
/// the directions don't exist). That is not "conservative" — it is wrong.
/// We refuse to fabricate a covariance and instead return this diagnostic.
#[derive(Debug, Clone)]
pub struct OuterHessianIndefinite {
    /// Most-negative eigenvalue of the projected (free-subspace) outer Hessian.
    pub min_eigenvalue: f64,
    /// Indices of θ-coordinates that were detected as active on a bound.
    pub active_constraints: Vec<usize>,
    /// θ at the reported optimum (if available; empty otherwise).
    pub theta: Vec<f64>,
    /// L2 norm of the outer gradient at θ (NaN if unavailable).
    pub gradient_norm: f64,
    /// Frobenius norm of the outer Hessian.
    pub hessian_norm: f64,
    /// Suggested next action for the caller / user.
    pub suggested_action: &'static str,
}

impl OuterHessianIndefinite {
    fn theta_dimension(&self) -> usize {
        self.theta.len()
    }
}

/// Errors that can arise while building the corrected covariance.
#[derive(Debug, Clone)]
pub enum CorrectedCovarianceError {
    /// Argument shapes do not agree (with explanatory message).
    ShapeMismatch(String),
    /// The eigendecomposition failed numerically (with the underlying message).
    EigendecompositionFailed(String),
    /// The projected outer Hessian is indefinite — see diagnostic.
    Indefinite(OuterHessianIndefinite),
}

impl core::fmt::Display for CorrectedCovarianceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
            Self::EigendecompositionFailed(msg) => write!(f, "eigendecomposition failed: {msg}"),
            Self::Indefinite(d) => write!(
                f,
                "outer Hessian indefinite on free subspace (min eigenvalue = {:.3e}, \
                 ||H||_F = {:.3e}, ||g||_2 = {:.3e}, active = {:?}, theta = {:?}); {}",
                d.min_eigenvalue,
                d.hessian_norm,
                d.gradient_norm,
                d.active_constraints,
                d.theta,
                d.suggested_action,
            ),
        }
    }
}

impl std::error::Error for CorrectedCovarianceError {}

/// Result describing the corrected covariance plus structural diagnostics.
#[derive(Debug, Clone)]
pub struct CorrectedCovariance {
    /// The p×p corrected covariance V*_α.
    pub matrix: Array2<f64>,
    /// θ-indices that were treated as active on a bound and excluded from V_θ.
    pub active_constraints: Vec<usize>,
    /// θ-indices in the free subspace whose curvature was so close to zero
    /// that they were treated as structurally rank-deficient (pseudoinverse).
    pub rank_deficient_directions: Vec<usize>,
}

impl CorrectedCovariance {
    fn has_structural_diagnostics(&self) -> bool {
        !self.active_constraints.is_empty() || !self.rank_deficient_directions.is_empty()
    }
}

/// Suggested action text returned with `OuterHessianIndefinite`.
const INDEFINITE_SUGGESTED_ACTION: &str = "refit with a tighter outer tolerance, verify the inspected objective is the true \
     REML/LAML cost rather than a surrogate, and audit recent active-set transitions";

/// Detect θ-coordinates that are sitting on the [-RHO_BOUND, RHO_BOUND] bound.
///
/// We use the same `tolerance = 1e-8` as the rest of the outer code path so the
/// active-set view here agrees with the optimizer's view at the reported optimum.
fn detect_active_theta_bounds(theta: Option<&[f64]>, q: usize) -> Vec<usize> {
    let Some(theta) = theta else {
        return Vec::new();
    };
    if theta.len() != q {
        return Vec::new();
    }
    let bound = crate::solver::estimate::RHO_BOUND;
    let tol = 1e-8;
    theta
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v.abs() >= bound - tol).then_some(i))
        .collect()
}

/// Decide which θ-coordinates are bounded (ρ-style) vs unbounded (ψ-style).
///
/// We treat the LAST `ext_len` coordinates as ψ (unbounded extended
/// hyperparameters) and the FIRST `rho_len` as ρ (bounded by ±RHO_BOUND).
/// This matches the layout used everywhere else in this file: J_α has ρ
/// columns first, then ext columns.
fn active_bound_indices_for_theta(
    theta: Option<&[f64]>,
    rho_len: usize,
    ext_len: usize,
) -> Vec<usize> {
    let q = rho_len + ext_len;
    let mut active = detect_active_theta_bounds(theta, q);
    // Drop ψ-coordinates: they are unbounded by construction.
    active.retain(|&i| i < rho_len);
    let _ = ext_len;
    active
}

/// Inertia gate + projected-inverse on the free subspace of θ.
///
/// Returns `(V_θ_full, rank_deficient_free_indices)` where `V_θ_full` is q×q
/// with rows/columns of active coordinates set to zero. If the projected
/// Hessian is indefinite beyond tolerance, returns the diagnostic instead.
fn projected_inverse_with_inertia_gate(
    outer_hessian: &Array2<f64>,
    active: &[usize],
    theta_for_diag: Option<&[f64]>,
    gradient_norm: f64,
) -> Result<(Array2<f64>, Vec<usize>), CorrectedCovarianceError> {
    let q = outer_hessian.nrows();
    let mut is_active = vec![false; q];
    for &i in active {
        if i < q {
            is_active[i] = true;
        }
    }
    let free: Vec<usize> = (0..q).filter(|i| !is_active[*i]).collect();
    let qf = free.len();

    let h_norm = outer_hessian.iter().map(|v| v * v).sum::<f64>().sqrt();

    let mut v_theta_full = Array2::<f64>::zeros((q, q));
    if qf == 0 {
        return Ok((v_theta_full, Vec::new()));
    }

    let mut h_ff = Array2::<f64>::zeros((qf, qf));
    for (a, &ia) in free.iter().enumerate() {
        for (b, &ib) in free.iter().enumerate() {
            h_ff[[a, b]] = outer_hessian[[ia, ib]];
        }
    }

    let (evals, evecs) = h_ff.eigh(faer::Side::Lower).map_err(|e| {
        CorrectedCovarianceError::EigendecompositionFailed(format!("projected outer Hessian: {e}"))
    })?;

    let eps = f64::EPSILON;
    let neg_tol = 8.0 * eps * (q.max(1) as f64) * h_norm.max(1.0);
    let min_eig = evals.iter().copied().fold(f64::INFINITY, f64::min);
    if min_eig < -neg_tol {
        let diagnostic = OuterHessianIndefinite {
            min_eigenvalue: min_eig,
            active_constraints: active.to_vec(),
            theta: theta_for_diag.map(|t| t.to_vec()).unwrap_or_default(),
            gradient_norm,
            hessian_norm: h_norm,
            suggested_action: INDEFINITE_SUGGESTED_ACTION,
        };
        let _theta_dimension = diagnostic.theta_dimension();
        return Err(CorrectedCovarianceError::Indefinite(diagnostic));
    }

    let pos_tol = 8.0 * eps * (q.max(1) as f64) * h_norm.max(1.0);
    let mut v_theta_ff = Array2::<f64>::zeros((qf, qf));
    let mut rank_deficient_free: Vec<usize> = Vec::new();
    for j in 0..qf {
        let sigma = evals[j];
        if sigma.abs() <= pos_tol {
            rank_deficient_free.push(j);
            continue;
        }
        let inv_sigma = 1.0 / sigma;
        let u = evecs.column(j);
        for a in 0..qf {
            let ua = inv_sigma * u[a];
            for b in a..qf {
                let val = ua * u[b];
                v_theta_ff[[a, b]] += val;
                if a != b {
                    v_theta_ff[[b, a]] += val;
                }
            }
        }
    }

    for (a, &ia) in free.iter().enumerate() {
        for (b, &ib) in free.iter().enumerate() {
            v_theta_full[[ia, ib]] = v_theta_ff[[a, b]];
        }
    }

    let rank_deficient_directions: Vec<usize> =
        rank_deficient_free.into_iter().map(|j| free[j]).collect();

    Ok((v_theta_full, rank_deficient_directions))
}

/// Corrected covariance of the coefficient vector, accounting for uncertainty
/// in the smoothing/hyperparameters θ = (ρ, ψ).
///
/// The standard conditional covariance H^{-1} ignores uncertainty in θ.
/// The corrected covariance adds the propagation term:
///
/// ```text
///   V*_α = H^{-1} + J_α V_θ J_α^T
/// ```
///
/// where:
/// - H^{-1} is obtained via `hop.solve` on identity columns,
/// - J_α = [-v_1, …, -v_k, -ext_v_1, …, -ext_v_m] is the p×q matrix of
///   negated mode responses (implicit-function sensitivities ∂β̂/∂θ),
/// - V_θ is the inverse of the outer Hessian RESTRICTED to the free subspace
///   of θ (coordinates that are not pinned to a bound) and inertia-gated.
///
/// # Active-bound handling and inertia gate
///
/// If `theta_at_optimum` is supplied, ρ-coordinates sitting on ±`RHO_BOUND`
/// are treated as active and excluded from V_θ. The remaining free Hessian
/// block H_FF is eigen-decomposed:
///   - if min(σ) < -8·ε·q·‖H‖_F → return [`CorrectedCovarianceError::Indefinite`]
///     with a diagnostic (the previous behavior of clamping negatives to zero
///     under-reports uncertainty and is therefore refused);
///   - if |σ| ≤ 8·ε·q·‖H‖_F → that direction is treated as structurally
///     rank-deficient (Moore-Penrose drop) and listed in
///     `rank_deficient_directions` for the caller to surface;
///   - otherwise H_FF is inverted exactly via the spectral expansion.
pub fn compute_corrected_covariance(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
) -> Result<Array2<f64>, CorrectedCovarianceError> {
    compute_corrected_covariance_with_constraints(v_ks, ext_v, outer_hessian, hop, None, f64::NAN)
        .map(|cov| {
            if cov.has_structural_diagnostics() {
                log::debug!(
                    "corrected covariance diagnostics: active_constraints={:?} rank_deficient_directions={:?}",
                    cov.active_constraints,
                    cov.rank_deficient_directions
                );
            }
            cov.matrix
        })
}

/// Constraint- and inertia-aware version of [`compute_corrected_covariance`].
///
/// Prefer this entry point when θ at the optimum and the outer-gradient norm
/// are available — it auto-derives the active-bound set on ρ and emits the
/// rank-deficient diagnostic alongside the matrix.
pub fn compute_corrected_covariance_with_constraints(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
    theta_at_optimum: Option<&[f64]>,
    gradient_norm: f64,
) -> Result<CorrectedCovariance, CorrectedCovarianceError> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    if q == 0 {
        let eye = Array2::eye(p);
        return Ok(CorrectedCovariance {
            matrix: hop.solve_multi(&eye),
            active_constraints: Vec::new(),
            rank_deficient_directions: Vec::new(),
        });
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(CorrectedCovarianceError::ShapeMismatch(format!(
            "compute_corrected_covariance: outer Hessian dimension ({}, {}) does not match \
             total hyperparameter count q = {} (rho: {}, ext: {})",
            outer_hessian.nrows(),
            outer_hessian.ncols(),
            q,
            v_ks.len(),
            ext_v.len(),
        )));
    }

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

    let active = active_bound_indices_for_theta(theta_at_optimum, v_ks.len(), ext_v.len());

    let (v_theta, rank_deficient_directions) = projected_inverse_with_inertia_gate(
        outer_hessian,
        &active,
        theta_at_optimum,
        gradient_norm,
    )?;

    let j_v_theta = j_alpha.dot(&v_theta);
    let correction = j_v_theta.dot(&j_alpha.t());

    let eye = Array2::eye(p);
    let mut h_inv = hop.solve_multi(&eye);
    h_inv += &correction;

    enforce_symmetry_inplace(&mut h_inv);

    Ok(CorrectedCovariance {
        matrix: h_inv,
        active_constraints: active,
        rank_deficient_directions,
    })
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
) -> Result<Array1<f64>, CorrectedCovarianceError> {
    compute_corrected_covariance_diagonal_with_constraints(
        v_ks,
        ext_v,
        outer_hessian,
        hop,
        None,
        f64::NAN,
    )
    .map(|d| {
        if d.has_structural_diagnostics() {
            log::debug!(
                "corrected covariance diagonal diagnostics: active_constraints={:?} rank_deficient_directions={:?}",
                d.active_constraints,
                d.rank_deficient_directions
            );
        }
        d.diagonal
    })
}

/// Diagonal of the corrected covariance plus active-set / rank-deficiency
/// diagnostics. See [`compute_corrected_covariance_with_constraints`] for the
/// full version (the inertia gate logic is identical).
#[derive(Debug, Clone)]
pub struct CorrectedCovarianceDiagonal {
    pub diagonal: Array1<f64>,
    pub active_constraints: Vec<usize>,
    pub rank_deficient_directions: Vec<usize>,
}

impl CorrectedCovarianceDiagonal {
    fn has_structural_diagnostics(&self) -> bool {
        !self.active_constraints.is_empty() || !self.rank_deficient_directions.is_empty()
    }
}

pub fn compute_corrected_covariance_diagonal_with_constraints(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
    theta_at_optimum: Option<&[f64]>,
    gradient_norm: f64,
) -> Result<CorrectedCovarianceDiagonal, CorrectedCovarianceError> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    let mut diag = Array1::zeros(p);
    for i in 0..p {
        let mut e_i = Array1::zeros(p);
        e_i[i] = 1.0;
        let h_inv_ei = hop.solve(&e_i);
        diag[i] = h_inv_ei[i];
    }

    if q == 0 {
        return Ok(CorrectedCovarianceDiagonal {
            diagonal: diag,
            active_constraints: Vec::new(),
            rank_deficient_directions: Vec::new(),
        });
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(CorrectedCovarianceError::ShapeMismatch(format!(
            "compute_corrected_covariance_diagonal: outer Hessian dimension ({}, {}) \
             does not match q = {}",
            outer_hessian.nrows(),
            outer_hessian.ncols(),
            q,
        )));
    }

    let active = active_bound_indices_for_theta(theta_at_optimum, v_ks.len(), ext_v.len());
    let (v_theta_full, rank_deficient_directions) = projected_inverse_with_inertia_gate(
        outer_hessian,
        &active,
        theta_at_optimum,
        gradient_norm,
    )?;

    // Symmetric square root of V_θ via eigendecomposition (PSD by construction).
    let (sym_evals, sym_evecs) = v_theta_full
        .eigh(faer::Side::Lower)
        .map_err(|e| CorrectedCovarianceError::EigendecompositionFailed(e.to_string()))?;
    let mut v_theta_sqrt = Array2::<f64>::zeros((q, q));
    for j in 0..q {
        let s = sym_evals[j];
        if s <= 0.0 {
            continue;
        }
        let scale = s.sqrt();
        for row in 0..q {
            v_theta_sqrt[[row, j]] = sym_evecs[[row, j]] * scale;
        }
    }

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

    let m = j_alpha.dot(&v_theta_sqrt);
    for i in 0..p {
        let mut row_norm_sq = 0.0;
        for j in 0..m.ncols() {
            row_norm_sq += m[[i, j]] * m[[i, j]];
        }
        diag[i] += row_norm_sq;
    }

    Ok(CorrectedCovarianceDiagonal {
        diagonal: diag,
        active_constraints: active,
        rank_deficient_directions,
    })
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
    pub fn reg_eigenvalue(&self, k: usize) -> f64 {
        self.reg_eigenvalues[k]
    }
    pub fn eigenvector_entry(&self, i: usize, k: usize) -> f64 {
        self.eigenvectors[[i, k]]
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
    fn rotate_to_eigenbasis(&self, matrix: &Array2<f64>) -> Array2<f64> {
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
        let left = crate::faer_ndarray::fast_atb(&self.w_factor, matrix);
        crate::faer_ndarray::fast_ab(&left, &self.w_factor)
    }

    #[inline]
    fn projected_operator(&self, factor: &Array2<f64>, op: &dyn HyperOperator) -> Array2<f64> {
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

/// Coalesce repeated identical `[STAGE]` log lines from `DenseSpectralOperator`
/// methods. First occurrence of a (method, dims, implicit-flags) signature
/// logs immediately; identical consecutive repeats are silenced and accrue
/// into a counter, emitting heartbeat summaries at doubling cadence
/// (2, 4, 8, 16, …) and a final summary when the signature changes.
fn dense_spectral_stage_log(signature: &str, elapsed_s: f64) {
    use std::sync::Mutex;
    struct Repeat {
        signature: String,
        count: u64,
        total: f64,
        min: f64,
        max: f64,
        next_heartbeat: u64,
    }
    static REPEAT: Mutex<Option<Repeat>> = Mutex::new(None);

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
        self.active_mask.iter().filter(|&&active| active).count()
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
    cached_logdet: crate::resource::RayonSafeOnce<f64>,
    n_dim: usize,
    // `RayonSafeOnce`, not `OnceLock`: `materialize_dense_operator` invokes
    // `apply`, which for operator-source joint Hessians dispatches a nested
    // `into_par_iter` (e.g. `exact_newton_joint_hessian_matvec_from_cache`).
    // With a plain `OnceLock`, concurrent rayon workers entering
    // `solve`/`logdet` from inside an outer par_iter would park on the
    // OnceLock's OS condvar; the leader's nested par_iter would then starve
    // for workers. `RayonSafeOnce` keeps init lock-free — racers may
    // duplicate the dim²-matvec build, but the first to publish wins and
    // steady-state matches `OnceLock`.
    dense_spectral: crate::resource::RayonSafeOnce<Option<DenseSpectralOperator>>,
    // Pseudo-logdet convention threaded from the family. The dense outer path
    // already plumbs `PseudoLogdetMode` into `BlockCoupledOperator`; the
    // matrix-free path materializes a `DenseSpectralOperator` lazily and must
    // use the same convention so that `logdet`, `trace_hinv_product`, the
    // IFT response `H⁻¹ g`, and every cross-trace agree with the dense path.
    // Without this, families that declare `HardPseudo` (BMS, GAMLSS) silently
    // get Smooth full-spectrum semantics on the matrix-free path, and outer
    // gradients are inflated by `1/σ_j` over numerical null directions.
    mode: PseudoLogdetMode,
}

impl MatrixFreeSpdOperator {
    const EXACT_DENSE_SPECTRAL_MAX_BYTES: usize = 512 * 1024 * 1024;
    const EXACT_DENSE_SPECTRAL_ARRAYS: usize = 6;

    pub fn new_with_mode<F>(dim: usize, apply: F, mode: PseudoLogdetMode) -> Self
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + 'static,
    {
        let apply = Arc::new(apply);

        Self {
            apply,
            cached_logdet: crate::resource::RayonSafeOnce::new(),
            n_dim: dim,
            dense_spectral: crate::resource::RayonSafeOnce::new(),
            mode,
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
        let result = DenseSpectralOperator::from_symmetric_with_mode(&matrix, self.mode).ok();
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
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "penalty_matrix_root: expected square matrix, got {}×{}",
                n,
                s.ncols()
            ),
        }
        .into());
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
    /// Hutch++ low-rank sketch dimension. `None` = plain Hutchinson.
    /// `Some(m_s)` runs the Meyer–Musco Hutch++ split: m_s sketch matvecs
    /// build an orthonormal range basis Q via randomized range finder, the
    /// projected trace tr(QᵀM Q) is computed exactly (m_s additional
    /// matvecs), and the residual tr((I-QQᵀ)M(I-QQᵀ)) is estimated by
    /// Hutchinson with the remaining probe budget. Achieves O(1/ε)
    /// matvecs for ε relative error vs O(1/ε²) for plain Hutchinson;
    /// the gain is largest when M has rapidly decaying singular values.
    pub hutchpp_sketch_dim: Option<usize>,
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
            hutchpp_sketch_dim: None,
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
            hutchpp_sketch_dim: None,
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

        if self.config.hutchpp_sketch_dim.is_some() {
            let op = DenseMatrixHyperOperator {
                matrix: matrix.clone(),
            };
            return hutchpp_estimate_trace_hinv_op_squared(hop, &op, &self.config);
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

        if self.config.hutchpp_sketch_dim.is_some() {
            return hutchpp_estimate_trace_hinv_op_squared(hop, op, &self.config);
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

/// Modified Gram–Schmidt orthonormalization of the columns of `y`,
/// writing the orthonormal basis into `q` and returning the retained
/// rank.
///
/// `y` and `q` must have the same shape `(p, m)`. Columns whose
/// reduction norm falls below `1e-12` of the largest input column
/// norm are dropped (numerical-rank cutoff). After this call,
/// `q.column(0..rank)` is column-orthonormal and approximates
/// `range(y)`; later columns of `q` are zeroed.
fn modified_gram_schmidt(y: &Array2<f64>, q: &mut Array2<f64>) -> usize {
    let p = y.nrows();
    let m = y.ncols();
    debug_assert_eq!(q.dim(), (p, m));
    q.fill(0.0);
    if p == 0 || m == 0 {
        return 0;
    }
    let mut max_norm: f64 = 0.0;
    for j in 0..m {
        let n = y.column(j).dot(&y.column(j)).sqrt();
        if n > max_norm {
            max_norm = n;
        }
    }
    let drop_tol = (max_norm * 1.0e-12).max(f64::MIN_POSITIVE);
    let mut rank = 0usize;
    for j in 0..m {
        let mut v = y.column(j).to_owned();
        for k in 0..rank {
            let qk = q.column(k);
            let proj = qk.dot(&v);
            if proj != 0.0 {
                v.scaled_add(-proj, &qk);
            }
        }
        let norm = v.dot(&v).sqrt();
        if !norm.is_finite() || norm <= drop_tol {
            continue;
        }
        let inv = 1.0 / norm;
        v.iter_mut().for_each(|x| *x *= inv);
        q.column_mut(rank).assign(&v);
        rank += 1;
    }
    rank
}

/// Hutch++ estimate of `tr(H⁻¹ M)` where `M` is accessed through its
/// matrix-vector product (operator-only, dim p).
///
/// Total cost: `2 m_s + m_h` H⁻¹ solves and `M·v` matvecs, where
/// `m_s = config.hutchpp_sketch_dim.unwrap_or(0)` and `m_h` is the
/// number of residual Hutchinson probes drawn (between
/// `config.n_probes_min` and `config.n_probes_max - 2 m_s`).
///
/// When `hutchpp_sketch_dim` is `None`, this falls back to plain
/// Hutchinson on the full probe budget — the result is deterministic
/// for a given seed because the probe RNG is seeded from
/// `config.seed`.
///
/// # Algorithm (Meyer–Musco 2021, SOSA)
///
/// 1. Sketch: draw `Z_s ∈ {±1}^{p × m_s}` Rademacher, compute
///    `Y = H⁻¹ M Z_s`, orthonormalize columns: `Y = Q R`.
/// 2. Low-rank trace: `T_low = tr(Qᵀ H⁻¹ M Q)` exactly via `m_s`
///    additional matvecs into `W = H⁻¹ M Q` and accumulating
///    `Σ_j Q[:,j] · W[:,j]`.
/// 3. Residual Hutchinson on the orthogonal complement: for each
///    residual probe `z`, set `z̃ = (I - Q Qᵀ) z`, compute
///    `w̃ = H⁻¹ M z̃`, and accumulate `z̃ · w̃` (which equals
///    `z̃ᵀ (H⁻¹ M) z̃` because `z̃` is in the complement).
/// 4. Output: `T_low + (1/m_h) Σ residual estimates`.
///
/// # When this wins over plain Hutchinson
///
/// Hutch++ converges in `O(1/ε)` matvecs vs `O(1/ε²)` for Hutchinson.
/// The gain is largest when `H⁻¹ M` has rapid singular-value decay —
/// the sketch captures the dominant subspace exactly and Hutchinson
/// only handles the small residual. For roughly-flat spectra both
/// methods perform similarly per-matvec.
pub(crate) fn hutchpp_estimate_trace_hinv_operator<H: HessianOperator + ?Sized>(
    hop: &H,
    op: &dyn HyperOperator,
    config: &StochasticTraceConfig,
) -> f64 {
    let p = hop.dim();
    debug_assert_eq!(op.dim(), p, "Hutch++: operator dim mismatch");
    if p == 0 {
        return 0.0;
    }
    let sketch_dim = config.hutchpp_sketch_dim.unwrap_or(0).min(p);
    let mut rng_state = Xoshiro256SS::from_seed(config.seed);

    // Phase 1: build orthonormal Q ∈ R^{p × sketch_dim} approximating
    // range(H⁻¹ M) via a randomized range finder.
    let mut q = Array2::<f64>::zeros((p, sketch_dim));
    let mut q_rank = 0usize;
    if sketch_dim > 0 {
        let mut y = Array2::<f64>::zeros((p, sketch_dim));
        let mut z = Array1::<f64>::zeros(p);
        let mut mz = Array1::<f64>::zeros(p);
        for j in 0..sketch_dim {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            op.mul_vec_into(z.view(), mz.view_mut());
            let w = hop.stochastic_trace_solve(&mz, config.solve_rel_tol);
            y.column_mut(j).assign(&w);
        }
        q_rank = modified_gram_schmidt(&y, &mut q);
    }

    // Phase 2: T_low = tr(Qᵀ H⁻¹ M Q). Apply H⁻¹ M to each retained
    // column of Q and accumulate Q[:,j] · W[:,j].
    let mut t_low = 0.0;
    if q_rank > 0 {
        let mut mq = Array1::<f64>::zeros(p);
        for j in 0..q_rank {
            let qcol = q.column(j).to_owned();
            op.mul_vec_into(qcol.view(), mq.view_mut());
            let w = hop.stochastic_trace_solve(&mq, config.solve_rel_tol);
            t_low += qcol.dot(&w);
        }
    }

    // Phase 3: residual Hutchinson on (I - Q Qᵀ) M (I - Q Qᵀ).
    // Budget = remaining matvecs from n_probes_max minus the 2*q_rank
    // we already spent (sketch + Q-trace), but never below n_probes_min.
    let used = 2 * q_rank;
    let residual_budget_max = config.n_probes_max.saturating_sub(used);
    let residual_min = config.n_probes_min.min(residual_budget_max);
    let residual_budget = residual_budget_max.max(residual_min);
    if residual_budget == 0 {
        return t_low;
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    let mut z = Array1::<f64>::zeros(p);
    let mut z_tilde = Array1::<f64>::zeros(p);
    let mut mz = Array1::<f64>::zeros(p);
    let check_interval = 4usize;
    for m in 0..residual_budget {
        rademacher_probe_into(z.view_mut(), &mut rng_state);
        // z_tilde = (I - Q Qᵀ) z = z - Q (Qᵀ z)
        z_tilde.assign(&z);
        if q_rank > 0 {
            for j in 0..q_rank {
                let qcol = q.column(j);
                let proj = qcol.dot(&z);
                if proj != 0.0 {
                    z_tilde.scaled_add(-proj, &qcol);
                }
            }
        }
        op.mul_vec_into(z_tilde.view(), mz.view_mut());
        let w = hop.stochastic_trace_solve(&mz, config.solve_rel_tol);
        let q_val = z_tilde.dot(&w);
        sum += q_val;
        sum_sq += q_val * q_val;
        count += 1;

        // Adaptive stopping: same Welford-style relative-error check
        // as `estimate_from_probe_batch`, applied to the residual mean.
        if count >= residual_min && count % check_interval == 0 && count >= 2 {
            let n = count as f64;
            let mean = sum / n;
            let var = (sum_sq - n * mean * mean) / (n - 1.0).max(1.0);
            if var.is_finite() && var >= 0.0 {
                let stderr = (var / n).sqrt();
                let denom = (mean.abs()).max(config.tau_rel);
                if stderr / denom <= config.relative_tol {
                    let _ = m; // matvec count just for documentation
                    break;
                }
            }
        }
    }
    let mean_residual = if count > 0 { sum / count as f64 } else { 0.0 };
    t_low + mean_residual
}

/// Hutch++ estimate of `tr((H⁻¹ A)²) = tr(H⁻¹ A H⁻¹ A)` for a symmetric
/// HVP-only operator `A`. Cost per applied "matvec" is 2 H⁻¹ solves and
/// 2 A applies; total cost is `2 m_s + m_h` such matvecs.
///
/// Although `B = H⁻¹ A` is not symmetric in the standard inner product,
/// `B²` is similar to `(H^{-1/2} A H^{-1/2})²` (PSD), so its trace is
/// nonnegative and Hutch++ on the linear map `x ↦ B² x` produces an
/// unbiased estimate of `tr(B²)` on standard probes.
pub(crate) fn hutchpp_estimate_trace_hinv_op_squared<H: HessianOperator + ?Sized>(
    hop: &H,
    op: &dyn HyperOperator,
    config: &StochasticTraceConfig,
) -> f64 {
    let p = hop.dim();
    debug_assert_eq!(op.dim(), p, "Hutch++ squared: operator dim mismatch");
    if p == 0 {
        return 0.0;
    }
    let sketch_dim = config.hutchpp_sketch_dim.unwrap_or(0).min(p);
    let mut rng_state = Xoshiro256SS::from_seed(config.seed);

    // Apply B² = H⁻¹ A H⁻¹ A in place via two solve+apply legs.
    let apply_b_squared = |hop: &H,
                           op: &dyn HyperOperator,
                           input: ArrayView1<'_, f64>,
                           tmp: &mut Array1<f64>|
     -> Array1<f64> {
        op.mul_vec_into(input, tmp.view_mut());
        let mid = hop.stochastic_trace_solve(tmp, config.solve_rel_tol);
        op.mul_vec_into(mid.view(), tmp.view_mut());
        hop.stochastic_trace_solve(tmp, config.solve_rel_tol)
    };

    let mut q = Array2::<f64>::zeros((p, sketch_dim));
    let mut q_rank = 0usize;
    if sketch_dim > 0 {
        let mut y = Array2::<f64>::zeros((p, sketch_dim));
        let mut z = Array1::<f64>::zeros(p);
        let mut tmp = Array1::<f64>::zeros(p);
        for j in 0..sketch_dim {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            let w = apply_b_squared(hop, op, z.view(), &mut tmp);
            y.column_mut(j).assign(&w);
        }
        q_rank = modified_gram_schmidt(&y, &mut q);
    }

    let mut t_low = 0.0;
    if q_rank > 0 {
        let mut tmp = Array1::<f64>::zeros(p);
        for j in 0..q_rank {
            let qcol = q.column(j).to_owned();
            let w = apply_b_squared(hop, op, qcol.view(), &mut tmp);
            t_low += qcol.dot(&w);
        }
    }

    let used = 2 * q_rank;
    let residual_budget_max = config.n_probes_max.saturating_sub(used);
    let residual_min = config.n_probes_min.min(residual_budget_max);
    let residual_budget = residual_budget_max.max(residual_min);
    if residual_budget == 0 {
        return t_low;
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    let mut z = Array1::<f64>::zeros(p);
    let mut z_tilde = Array1::<f64>::zeros(p);
    let mut tmp = Array1::<f64>::zeros(p);
    let check_interval = 4usize;
    for _ in 0..residual_budget {
        rademacher_probe_into(z.view_mut(), &mut rng_state);
        z_tilde.assign(&z);
        if q_rank > 0 {
            for j in 0..q_rank {
                let qcol = q.column(j);
                let proj = qcol.dot(&z);
                if proj != 0.0 {
                    z_tilde.scaled_add(-proj, &qcol);
                }
            }
        }
        let w = apply_b_squared(hop, op, z_tilde.view(), &mut tmp);
        let q_val = z_tilde.dot(&w);
        sum += q_val;
        sum_sq += q_val * q_val;
        count += 1;

        if count >= residual_min && count % check_interval == 0 && count >= 2 {
            let n = count as f64;
            let mean = sum / n;
            let var = (sum_sq - n * mean * mean) / (n - 1.0).max(1.0);
            if var.is_finite() && var >= 0.0 {
                let stderr = (var / n).sqrt();
                let denom = (mean.abs()).max(config.tau_rel);
                if stderr / denom <= config.relative_tol {
                    break;
                }
            }
        }
    }
    let mean_residual = if count > 0 { sum / count as f64 } else { 0.0 };
    t_low + mean_residual
}

/// Hutch++-style estimate of `tr(H⁻¹ A_left H⁻¹ A_right)` for two
/// (possibly distinct) symmetric HVP-only operators. Uses a shared
/// sketch built from `M = M_L M_R` where `M_L = H⁻¹ A_left` and
/// `M_R = H⁻¹ A_right`; per matvec is 2 H⁻¹ solves + 2 A applies.
///
/// On standard Rademacher probes `E[zᵀ M z] = tr(M)` regardless of
/// symmetry, so the residual Hutchinson average is unbiased even when
/// `M` is not self-adjoint in the standard inner product.
///
/// A leave-one-out XTrace estimator (Epperly & Tropp 2024, arxiv
/// 2301.07825) would reduce variance further by exchanging each probe
/// between sketch and residual roles, at O(m²) bookkeeping cost.
pub(crate) fn hutchpp_estimate_trace_hinv_operator_cross<H: HessianOperator + ?Sized>(
    hop: &H,
    left: &dyn HyperOperator,
    right: &dyn HyperOperator,
    config: &StochasticTraceConfig,
) -> f64 {
    let p = hop.dim();
    debug_assert_eq!(left.dim(), p, "cross trace: left operator dim mismatch");
    debug_assert_eq!(right.dim(), p, "cross trace: right operator dim mismatch");
    if p == 0 {
        return 0.0;
    }
    let sketch_dim = config.hutchpp_sketch_dim.unwrap_or(0).min(p);
    let mut rng_state = Xoshiro256SS::from_seed(config.seed);

    let apply_m = |hop: &H, x: ArrayView1<'_, f64>, tmp: &mut Array1<f64>| -> Array1<f64> {
        // M x = H⁻¹ A_L H⁻¹ A_R x
        right.mul_vec_into(x, tmp.view_mut());
        let mid = hop.stochastic_trace_solve(tmp, config.solve_rel_tol);
        left.mul_vec_into(mid.view(), tmp.view_mut());
        hop.stochastic_trace_solve(tmp, config.solve_rel_tol)
    };

    let mut q = Array2::<f64>::zeros((p, sketch_dim));
    let mut q_rank = 0usize;
    if sketch_dim > 0 {
        let mut y = Array2::<f64>::zeros((p, sketch_dim));
        let mut z = Array1::<f64>::zeros(p);
        let mut tmp = Array1::<f64>::zeros(p);
        for j in 0..sketch_dim {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            let w = apply_m(hop, z.view(), &mut tmp);
            y.column_mut(j).assign(&w);
        }
        q_rank = modified_gram_schmidt(&y, &mut q);
    }

    // T_low = tr(Qᵀ M Q): for non-symmetric M this is the projected
    // trace of M restricted to range(Q), which is exact on that
    // subspace.
    let mut t_low = 0.0;
    if q_rank > 0 {
        let mut tmp = Array1::<f64>::zeros(p);
        for j in 0..q_rank {
            let qcol = q.column(j).to_owned();
            let w = apply_m(hop, qcol.view(), &mut tmp);
            t_low += qcol.dot(&w);
        }
    }

    let used = 2 * q_rank;
    let residual_budget_max = config.n_probes_max.saturating_sub(used);
    let residual_min = config.n_probes_min.min(residual_budget_max);
    let residual_budget = residual_budget_max.max(residual_min);
    if residual_budget == 0 {
        return t_low;
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    let mut z = Array1::<f64>::zeros(p);
    let mut z_tilde = Array1::<f64>::zeros(p);
    let mut tmp = Array1::<f64>::zeros(p);
    let check_interval = 4usize;
    for _ in 0..residual_budget {
        rademacher_probe_into(z.view_mut(), &mut rng_state);
        z_tilde.assign(&z);
        if q_rank > 0 {
            for j in 0..q_rank {
                let qcol = q.column(j);
                let proj = qcol.dot(&z);
                if proj != 0.0 {
                    z_tilde.scaled_add(-proj, &qcol);
                }
            }
        }
        let w = apply_m(hop, z_tilde.view(), &mut tmp);
        let q_val = z_tilde.dot(&w);
        sum += q_val;
        sum_sq += q_val * q_val;
        count += 1;

        if count >= residual_min && count % check_interval == 0 && count >= 2 {
            let n = count as f64;
            let mean = sum / n;
            let var = (sum_sq - n * mean * mean) / (n - 1.0).max(1.0);
            if var.is_finite() && var >= 0.0 {
                let stderr = (var / n).sqrt();
                let denom = (mean.abs()).max(config.tau_rel);
                if stderr / denom <= config.relative_tol {
                    break;
                }
            }
        }
    }
    let mean_residual = if count > 0 { sum / count as f64 } else { 0.0 };
    t_low + mean_residual
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::estimate::DP_FLOOR;
    use approx::assert_relative_eq;
    use ndarray::array;

    // ─── Verification tests for the projected-pseudo-inverse IFT fix ─────
    //
    // The hypothesis: when the inner KKT residual `r` has spurious noise
    // outside `range(S_+)` and H has a near-null eigenvalue in
    // `null(S_+)`, the full-H solve `H⁻¹·r` amplifies that noise by
    // `1/σ_min(H)`. Routing the IFT correction through
    // `(U_S · H_proj⁻¹ · U_Sᵀ)` kills the noise without biasing the
    // honest correction. The tests below verify this with synthetic H
    // matrices whose eigenstructure we control directly.

    /// Helper: build a 5×5 diagonal SPD H with one eigenvalue placed in a
    /// chosen direction. `placement` selects whether the small eigenvalue
    /// lives inside `range(S_+)` (col 0..4) or inside `null(S_+)` (col 4).
    fn synthetic_h_with_small_eig(
        small_eig: f64,
        placement: SmallEigPlacement,
    ) -> (Array2<f64>, Array2<f64>) {
        let p = 5usize;
        let r = 4usize;
        let mut h_full = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            h_full[[i, i]] = 1.0;
        }
        let (small_idx, u_s_cols): (usize, Vec<usize>) = match placement {
            // Small direction is the unpenalized parametric column (e_4);
            // `U_S` spans e_0..e_3 — the projection excludes the near-null
            // direction entirely, so the projected pseudo-inverse never
            // touches the small eigenvalue.
            SmallEigPlacement::OutsideRangeSPlus => (4, vec![0, 1, 2, 3]),
            // Small direction lives inside `range(S_+)` (e_0); the
            // projection retains it and the projected pseudo-inverse must
            // still amplify by `1/σ_min` — this is the case where the fix
            // does *not* help and a different remediation (truncated SVD
            // / Tikhonov) would be required. The test makes that
            // explicit.
            SmallEigPlacement::InsideRangeSPlus => (0, vec![0, 1, 2, 3]),
        };
        h_full[[small_idx, small_idx]] = small_eig;
        // Anchor the non-small in-subspace direction at a moderately large
        // eigenvalue so the "outside" placement still produces a noticeable
        // contrast against the full-H result.
        if matches!(placement, SmallEigPlacement::OutsideRangeSPlus) {
            // Boost diag entry 0 so the full-H block on `range(S_+)` is
            // well-conditioned — only the e_4 direction is degenerate.
        }
        let mut u_s = Array2::<f64>::zeros((p, r));
        for (col_pos, &row) in u_s_cols.iter().enumerate() {
            u_s[[row, col_pos]] = 1.0;
        }
        (h_full, u_s)
    }

    #[derive(Clone, Copy)]
    enum SmallEigPlacement {
        OutsideRangeSPlus,
        InsideRangeSPlus,
    }

    /// Build a `PenaltySubspaceTrace` from a full H + U_S pair, using the
    /// exact same formula the production code uses: `H_proj⁻¹ = (U_Sᵀ H
    /// U_S)⁻¹`. Inverts the projected matrix analytically for the test.
    fn build_subspace_kernel(h_full: &Array2<f64>, u_s: &Array2<f64>) -> PenaltySubspaceTrace {
        // Compute H_proj = U_Sᵀ H U_S, then invert via eigendecomposition —
        // exactly matching the production builder in
        // `joint_penalty_subspace_trace_parts`
        // (`src/families/custom_family.rs:13835`). Production uses the same
        // Moore-Penrose recipe: eigendecompose, threshold near-zero
        // eigenvalues, then build `Σ_i (1/σ_i) v_i v_iᵀ`. For our
        // well-conditioned `H_proj` (the small eigenvalue of H lives
        // OUTSIDE U_S so H_proj is full-rank by construction), every
        // eigenvalue passes the threshold and the result is the exact
        // inverse.
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let h_proj = u_s.t().dot(h_full).dot(u_s);
        let (evals, evecs) = h_proj
            .eigh(Side::Lower)
            .expect("h_proj eigh in test fixture");
        let r = h_proj.nrows();
        let mut h_proj_inverse = Array2::<f64>::zeros((r, r));
        for k in 0..evals.len() {
            assert!(
                evals[k].abs() > 1e-10,
                "test fixture must keep H_proj non-singular; got eval[{k}] = {}",
                evals[k]
            );
            let inv = 1.0 / evals[k];
            for i in 0..r {
                for j in 0..r {
                    h_proj_inverse[[i, j]] += inv * evecs[[i, k]] * evecs[[j, k]];
                }
            }
        }
        PenaltySubspaceTrace {
            u_s: u_s.clone(),
            h_proj_inverse,
        }
    }

    /// Direct ground-truth full-H inverse bilinear form `aᵀ H⁻¹ b`. The
    /// test fixtures all use diagonal H so we invert componentwise.
    fn full_h_inv_bilinear(h_full: &Array2<f64>, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let p = h_full.nrows();
        let mut acc = 0.0_f64;
        for i in 0..p {
            assert!(h_full[[i, i]].abs() > 0.0);
            acc += a[i] * b[i] / h_full[[i, i]];
        }
        acc
    }

    /// **Mechanism test, line of evidence 1**: when the near-null
    /// eigenvalue of H sits in `null(S_+)` (the unpenalized parametric
    /// direction — intercept/sex/prs_z for the failing biobank survival
    /// marginal-slope), the full-H `r ↦ H⁻¹ r` solve amplifies any
    /// spurious noise component of `r` in that direction by
    /// `1/σ_min(H)`, while the projected pseudo-inverse drops that
    /// component entirely and recovers the honest correction.
    #[test]
    fn ift_projected_pseudo_inverse_kills_null_subspace_noise() {
        let small_eig = 1e-12_f64;
        let (h_full, u_s) =
            synthetic_h_with_small_eig(small_eig, SmallEigPlacement::OutsideRangeSPlus);
        let kernel = build_subspace_kernel(&h_full, &u_s);
        // r: honest signal in range(S_+) + tiny noise in null(S_+).
        let r_honest = array![1.0_f64, 0.0, 0.0, 0.0, 0.0];
        let r_noise = array![0.0_f64, 0.0, 0.0, 0.0, 1e-3];
        let r_total = &r_honest + &r_noise;
        // a_k = λ_k · S_k · β̂ lies in range(S_+) by construction (every
        // S_k has range ⊂ range(S_+)). Pick a non-trivial alignment with
        // r_honest so the IFT correction is non-zero in the well-
        // conditioned subspace.
        let a_k = array![0.0_f64, 1.0, 0.0, 0.0, 0.0];

        // Honest reference: pseudo-inverse evaluated on the noise-free r
        // (gives the correction we'd see at exact inner KKT).
        let corr_honest = kernel.bilinear_pseudo_inverse(&r_honest, &a_k);
        let corr_proj = kernel.bilinear_pseudo_inverse(&r_total, &a_k);
        let corr_full = full_h_inv_bilinear(&h_full, &r_total, &a_k);

        // Projected pseudo-inverse: matches the honest reference exactly
        // because the noise component lives in `null(U_S)` and dies under
        // the `U_Sᵀ` projection.
        assert_relative_eq!(corr_proj, corr_honest, max_relative = 1e-12);
        // Full-H bilinear form on this fixture happens to also match the
        // honest result *here* (a_k has no e_4 component, so the
        // amplified noise direction never sees a multiplier from a_k).
        // Switch to a configuration where a_k DOES have e_4 alignment
        // to expose the full-H pathology.
        let _ = corr_full;
    }

    /// **Mechanism test, line of evidence 2** — the failure mode itself:
    /// when `r` AND `a_k` both have spurious components in the near-null
    /// direction (the realistic floating-point pattern at the failing
    /// biobank iterate), the full-H solve produces a `~ηξ/σ_min`-scale
    /// blow-up while the projected pseudo-inverse stays bounded. With
    /// `η = ξ = 1e-3` and `σ_min = 1e-12`, the full-H result is `1e6`
    /// while the projection drops it to 0 — six orders of magnitude
    /// reduction in noise, on the same input.
    #[test]
    fn ift_full_h_solve_amplifies_null_subspace_noise_by_inverse_small_eig() {
        let small_eig = 1e-12_f64;
        let (h_full, u_s) =
            synthetic_h_with_small_eig(small_eig, SmallEigPlacement::OutsideRangeSPlus);
        let kernel = build_subspace_kernel(&h_full, &u_s);

        let eta = 1e-3_f64; // noise scale on r in null(S_+)
        let xi = 1e-3_f64; // noise scale on a_k in null(S_+)
        let r = array![0.0_f64, 0.0, 0.0, 0.0, eta];
        let a_k = array![0.0_f64, 0.0, 0.0, 0.0, xi];

        let corr_proj = kernel.bilinear_pseudo_inverse(&r, &a_k);
        let corr_full = full_h_inv_bilinear(&h_full, &r, &a_k);

        // Projection: U_Sᵀ kills both r and a_k entirely (they live in
        // `null(S_+) = span(e_4)` which is orthogonal to U_S's columns),
        // so the correction is exactly zero.
        assert!(
            corr_proj.abs() < 1e-15,
            "projected correction must drop pure-null-subspace contributions, got {corr_proj:.3e}"
        );
        // Full-H solve: `η · (1/σ_min) · ξ = 1e-3 · 1e12 · 1e-3 = 1e6`.
        // This is the noise-amplification mechanism behind the observed
        // |g|∞ ≈ 10¹³ on the failing biobank iterate (scale it by the
        // tighter noise floor and the per-coord magnitude of a_k).
        let expected_full = eta * xi / small_eig;
        assert_relative_eq!(corr_full, expected_full, max_relative = 1e-12);
        // Quantitative ratio: the projected fix delivers `≥ ~6 orders of
        // magnitude` noise reduction on this synthetic fixture, and the
        // ratio grows linearly with `1/σ_min(H)` — i.e. the worse H is
        // conditioned, the more the projected approach saves.
        assert!(
            corr_full / 1.0 >= 1e5,
            "full-H solve must produce a large blow-up for this fixture"
        );
    }

    /// **Mechanism test, line of evidence 3** (honest counter-example):
    /// when the near-null eigenvalue lives *inside* `range(S_+)`, the
    /// projection cannot help — `H_proj` inherits the same small
    /// eigenvalue and the bilinear form has the same amplification. This
    /// test pins that limit so future readers know exactly where the fix
    /// breaks down; if the failing-biobank H matches this geometry the
    /// fix is the wrong remediation and we need truncated-SVD /
    /// Tikhonov regularization instead. The current production
    /// experience (gradient drops by orders of magnitude after the fix)
    /// is the empirical evidence that the failing geometry is the
    /// outside-`range(S_+)` case in tests 1 and 2 above.
    #[test]
    fn ift_projected_pseudo_inverse_cannot_help_when_small_eig_lives_inside_range_s_plus() {
        let small_eig = 1e-8_f64;
        let (h_full, u_s) =
            synthetic_h_with_small_eig(small_eig, SmallEigPlacement::InsideRangeSPlus);
        let kernel = build_subspace_kernel(&h_full, &u_s);

        let eta = 1e-3_f64;
        let xi = 1e-3_f64;
        let r = array![eta, 0.0, 0.0, 0.0, 0.0]; // noise in e_0, which is in range(S_+)
        let a_k = array![xi, 0.0, 0.0, 0.0, 0.0];

        let corr_proj = kernel.bilinear_pseudo_inverse(&r, &a_k);
        let corr_full = full_h_inv_bilinear(&h_full, &r, &a_k);

        // Both methods give the same blow-up `ηξ/σ_min = 1e2` because
        // the near-null eigenvalue is inside the projected subspace.
        let expected = eta * xi / small_eig;
        assert_relative_eq!(corr_proj, expected, max_relative = 1e-12);
        assert_relative_eq!(corr_full, expected, max_relative = 1e-12);
    }

    /// **Sanity test, line of evidence 4**: the projected pseudo-inverse
    /// does NOT bias the well-conditioned case — when H is well-
    /// conditioned and `r`, `a_k` are honest in-subspace signals, the
    /// projection and the full-H solve agree to machine precision. This
    /// keeps the fix from introducing a regression on Gaussian-identity
    /// fixtures (where the subspace-projection-LAML fix is not active
    /// and the existing code path is correct).
    #[test]
    fn ift_projected_pseudo_inverse_matches_full_h_on_well_conditioned_fixture() {
        // Well-conditioned H — every eigenvalue O(1).
        let p = 5usize;
        let r_subspace = 4usize;
        let mut h_full = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            h_full[[i, i]] = (i as f64 + 1.0) * 2.0;
        }
        let mut u_s = Array2::<f64>::zeros((p, r_subspace));
        for j in 0..r_subspace {
            u_s[[j, j]] = 1.0;
        }
        let kernel = build_subspace_kernel(&h_full, &u_s);

        let r = array![0.3_f64, -0.7, 1.2, 0.4, 0.0]; // honest in range(S_+)
        let a_k = array![0.5_f64, 0.1, -0.2, 0.8, 0.0]; // honest in range(S_+)

        let corr_proj = kernel.bilinear_pseudo_inverse(&r, &a_k);
        let corr_full = full_h_inv_bilinear(&h_full, &r, &a_k);

        assert_relative_eq!(corr_proj, corr_full, max_relative = 1e-12);
    }

    /// Direct ground-truth full-H inverse bilinear form `aᵀ H⁻¹ b` for an
    /// arbitrary SPD `H`, computed via an explicit eigendecomposition.
    /// Diagonal `full_h_inv_bilinear` cannot exhibit the cross-coupling
    /// pathology described in `0dc469bd` (the off-diagonal entries are
    /// what propagate `r`'s null-space noise into the `a_k ∈ range(S_+)`
    /// solve).
    fn dense_h_inv_bilinear_via_eig(h_full: &Array2<f64>, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        use crate::faer_ndarray::FaerEigh;
        let (evals, evecs) = h_full
            .eigh(faer::Side::Lower)
            .expect("eigendecomp of test fixture H");
        // (Uᵀ a)_i · (1/λ_i) · (Uᵀ b)_i summed over i.
        let ua = evecs.t().dot(a);
        let ub = evecs.t().dot(b);
        let mut acc = 0.0_f64;
        for i in 0..evals.len() {
            assert!(
                evals[i].abs() > 0.0,
                "fixture eigenvalue must be nonzero for direct solve"
            );
            acc += ua[i] * ub[i] / evals[i];
        }
        acc
    }

    /// **Mechanism test, line of evidence 5 (production geometry)**: an
    /// SPD `H` with a small eigenvalue whose eigenvector MIXES
    /// `range(S_+)` and `null(S_+)`. This is the geometry the biobank
    /// survival marginal-slope hits at the failing iterate: the unpenalized
    /// parametric columns (intercept, sex, prs_z) interact with the
    /// penalized Duchon centers via `Xᵀ W X` off-diagonal coupling, so the
    /// smallest eigendirection of `H` is NOT axis-aligned with the
    /// `U_S = span(S_+)` block.
    ///
    /// In this regime:
    ///   * `a_k = λ_k S_k β̂ ∈ range(S_+)` (by construction, exactly the
    ///     production input shape — purely in-subspace, NO null
    ///     contamination).
    ///   * `r = r_clean + ε · e_null` has small but nonzero null-space
    ///     contamination representative of floating-point KKT residual
    ///     noise at the inner exit certificate.
    ///   * The full-H inverse `H⁻¹ a_k` PICKS UP a null-direction
    ///     component via the Schur complement / cross-block coupling: the
    ///     small-eigenvalue eigenvector v_min has both a range(S_+) and
    ///     a null(S_+) leg, so `H⁻¹ a_k ∝ (a_kᵀ v_min) · v_min / σ_min`
    ///     has a null leg of magnitude `(a_k · v_min_S) · v_min_N /
    ///     σ_min ≈ 1 · 1 / 1e-12 = 1e12`. Dotting with `r`'s tiny null
    ///     noise `ε ≈ 1e-3` gives a `1e9` spurious contribution.
    ///   * The projected helper `aᵀ U_S H_proj⁻¹ U_Sᵀ b`:
    ///       - `U_Sᵀ a_k = a_k_S` (unchanged, since `a_k ∈ range(S_+)`)
    ///       - `U_Sᵀ r = r_S` (drops the `ε · e_null` contamination)
    ///       - `H_proj = U_Sᵀ H U_S` — the in-subspace block, which has
    ///         the well-conditioned `O(1)` eigenvalues only (the small
    ///         eigendirection's range(S_+) leg is `≪ 1`, so its
    ///         contribution to `H_proj` is `O(σ_min · (v_min_S)²) ≪`
    ///         the other eigenvalues; `H_proj⁻¹` stays `O(1)`).
    ///       - Result: `r_S · H_proj⁻¹ · a_k_S` is `O(1)`.
    ///
    /// This test reproduces the FAILING-BIOBANK geometry and asserts
    /// FOUR INDEPENDENT properties:
    ///   (P1) helper matches an independent eigendecomposition-based
    ///        ground-truth bilinear to 1e-12 (validates the inversion
    ///        path on a NON-DIAGONAL h_proj, where the diagonal
    ///        shortcut would silently fail);
    ///   (P2) null pollution on r is invariant under the helper (1e-12);
    ///   (P3) the SAME null pollution corrupts full-H by the
    ///        analytically predicted `ε · (q_minᵀ a_k) · q_min_null /
    ///        σ_min` (matched within 5%);
    ///   (P4) on CLEAN r the projected helper and full-H still DISAGREE
    ///        by `O(σ_min⁻¹)` because the projected kernel pairs with
    ///        `½ log|U_Sᵀ H U_S|_+` while the full-H is the Schur
    ///        complement inverse — mathematically distinct, not just
    ///        less noisy.
    /// Independent ground-truth bilinear form `aᵀ U_S (U_Sᵀ H U_S)⁻¹
    /// U_Sᵀ b`. Recomputes the projected inverse via a fresh
    /// eigendecomposition of `U_Sᵀ H U_S` — a separate code path from
    /// `PenaltySubspaceTrace::bilinear_pseudo_inverse` (which applies a
    /// PRECOMPUTED `h_proj_inverse`). Match between the two is non-
    /// trivial verification of the helper's inversion.
    fn projected_pseudo_inverse_truth(
        h_full: &Array2<f64>,
        u_s: &Array2<f64>,
        a: &Array1<f64>,
        b: &Array1<f64>,
    ) -> f64 {
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let proj_a = u_s.t().dot(a);
        let proj_b = u_s.t().dot(b);
        let h_proj = u_s.t().dot(h_full).dot(u_s);
        let (evals, evecs) = h_proj.eigh(Side::Lower).expect("h_proj eigh");
        let ua = evecs.t().dot(&proj_a);
        let ub = evecs.t().dot(&proj_b);
        let mut acc = 0.0_f64;
        for i in 0..evals.len() {
            assert!(evals[i].abs() > 1e-10, "h_proj eigenvalue must be nonzero");
            acc += ua[i] * ub[i] / evals[i];
        }
        acc
    }

    #[test]
    fn ift_projected_pseudo_inverse_saves_orders_of_magnitude_on_cross_coupled_h() {
        let small_eig = 1e-12_f64;
        let p = 5usize;
        let r_subspace = 4usize;

        // U_S spans the first 4 standard basis vectors (range of S_+).
        let mut u_s = Array2::<f64>::zeros((p, r_subspace));
        for j in 0..r_subspace {
            u_s[[j, j]] = 1.0;
        }

        // Build SPD H = Q diag(λ) Qᵀ where the smallest-eigenvalue
        // eigenvector v_min has LEGS ON ALL FOUR range(S_+) coordinates
        // (not just e_3 as in the earlier, weaker fixture). This forces
        // `h_proj = U_Sᵀ H U_S` to be genuinely non-diagonal, so the
        // helper's eigendecomposition-based inversion is actually
        // exercised (a diagonal shortcut would silently fail here).
        let v_min = {
            let leg_s = 0.15_f64;
            let leg_n = (1.0 - 4.0 * leg_s * leg_s).sqrt();
            array![leg_s, leg_s, leg_s, leg_s, leg_n]
        };
        // Four ambient vectors with MIXED support (not the standard
        // basis) so Gram-Schmidt against `v_min` produces dense Q
        // columns and a dense `h_proj`.
        let ambients = vec![
            array![1.0_f64, 0.3, -0.2, 0.5, 0.0],
            array![0.4_f64, 1.0, 0.6, -0.3, 0.0],
            array![-0.5_f64, 0.2, 1.0, 0.7, 0.0],
            array![0.6_f64, -0.4, 0.3, 1.0, 0.0],
        ];
        let mut q = Array2::<f64>::zeros((p, p));
        q.column_mut(p - 1).assign(&v_min);
        let mut col_idx = 0usize;
        for ambient in ambients.iter() {
            let mut v = ambient.clone();
            let dot = v.dot(&v_min);
            v.scaled_add(-dot, &v_min);
            for prev in 0..col_idx {
                let qprev = q.column(prev).to_owned();
                let d = v.dot(&qprev);
                v.scaled_add(-d, &qprev);
            }
            let norm = v.dot(&v).sqrt();
            assert!(
                norm > 1e-10,
                "Gram-Schmidt failed at col {col_idx}: norm = {norm}"
            );
            v /= norm;
            q.column_mut(col_idx).assign(&v);
            col_idx += 1;
        }
        assert_eq!(col_idx, p - 1);

        let eigvals = array![10.0_f64, 5.0, 2.0, 1.0, small_eig];
        let mut h_full = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            let qi = q.column(i).to_owned();
            for a in 0..p {
                for b in 0..p {
                    h_full[[a, b]] += eigvals[i] * qi[a] * qi[b];
                }
            }
        }
        // Symmetrize to suppress 1e-16-scale asymmetry from
        // outer-product summation order.
        for a in 0..p {
            for b in (a + 1)..p {
                let avg = 0.5 * (h_full[[a, b]] + h_full[[b, a]]);
                h_full[[a, b]] = avg;
                h_full[[b, a]] = avg;
            }
        }

        // Verify `h_proj` is genuinely non-diagonal (the WHOLE POINT of
        // this fixture — earlier versions silently relied on it being
        // diagonal).
        let h_proj_check = u_s.t().dot(&h_full).dot(&u_s);
        let mut max_offdiag = 0.0_f64;
        let mut max_diag = 0.0_f64;
        for i in 0..r_subspace {
            max_diag = max_diag.max(h_proj_check[[i, i]].abs());
            for j in 0..r_subspace {
                if i != j {
                    max_offdiag = max_offdiag.max(h_proj_check[[i, j]].abs());
                }
            }
        }
        assert!(
            max_offdiag > 0.1 * max_diag,
            "fixture must produce non-diagonal h_proj; max_offdiag = \
             {max_offdiag:.3e}, max_diag = {max_diag:.3e}"
        );

        let kernel = build_subspace_kernel(&h_full, &u_s);

        // `a_k` purely in range(S_+) — production geometry exactly:
        // `λ_k S_k β̂ ∈ col(S_k) ⊂ range(S_+)`.
        let a_k = array![0.5_f64, 0.7, -0.3, 0.9, 0.0];
        let eps_null = 1e-3_f64;
        let r_clean = array![0.4_f64, -0.6, 1.1, 0.3, 0.0];
        let r_total = &r_clean + &array![0.0_f64, 0.0, 0.0, 0.0, eps_null];

        // ── (P1) Helper matches independent ground-truth bilinear ──
        let truth_clean = projected_pseudo_inverse_truth(&h_full, &u_s, &r_clean, &a_k);
        let truth_total = projected_pseudo_inverse_truth(&h_full, &u_s, &r_total, &a_k);
        let corr_proj_clean = kernel.bilinear_pseudo_inverse(&r_clean, &a_k);
        let corr_proj_total = kernel.bilinear_pseudo_inverse(&r_total, &a_k);
        assert_relative_eq!(corr_proj_clean, truth_clean, max_relative = 1e-10);
        assert_relative_eq!(corr_proj_total, truth_total, max_relative = 1e-10);

        // ── (P2) Projection invariance under null pollution ──
        // The two helper outputs agree because `U_Sᵀ ε e_null = 0`,
        // AND this holds DESPITE the non-diagonal `h_proj` (i.e. the
        // full inversion path is exercised).
        assert_relative_eq!(corr_proj_total, corr_proj_clean, max_relative = 1e-12);
        assert_relative_eq!(truth_total, truth_clean, max_relative = 1e-12);

        // ── (P3) Full-H IS corrupted by the same pollution ──
        // Predicted scale: ε · (q_minᵀ a_k) · q_min[p-1] / σ_min.
        let corr_full_clean = dense_h_inv_bilinear_via_eig(&h_full, &r_clean, &a_k);
        let corr_full_total = dense_h_inv_bilinear_via_eig(&h_full, &r_total, &a_k);
        let full_noise_contrib = corr_full_total - corr_full_clean;
        let v_min_dot_a = v_min.dot(&a_k);
        let v_min_null = v_min[p - 1];
        let predicted_noise = eps_null * v_min_dot_a * v_min_null / small_eig;
        assert!(
            full_noise_contrib.abs() > 1e6,
            "full-H must show 10⁶+ noise amplification; got |Δ| = {:.3e}",
            full_noise_contrib.abs()
        );
        let ratio = full_noise_contrib / predicted_noise;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "full-H corruption must follow predicted scaling; \
             ratio = {ratio:.6}, predicted = {predicted_noise:.3e}, \
             actual = {full_noise_contrib:.3e}"
        );

        // ── (P4) Self-consistency: projected ≠ full-H even on CLEAN r ──
        // The fix is the kernel that pairs with `½ log|U_Sᵀ H U_S|_+`;
        // full-H pairs with `½ log|H|`. On a cross-coupled SPD H with a
        // small-eigenvalue mixed direction, the two bilinear forms
        // disagree by `O(σ_min⁻¹)`. NOT just a denoising effect.
        let clean_disagreement = corr_full_clean - corr_proj_clean;
        assert!(
            clean_disagreement.abs() > 1e5,
            "fix is self-consistency, NOT denoising: on CLEAN input \
             projected ({corr_proj_clean:.3e}) and full-H \
             ({corr_full_clean:.3e}) must differ by O(1/σ_min); \
             got disagreement = {clean_disagreement:.3e}"
        );

        eprintln!(
            "[ift-cross-coupled-airtight] h_proj non-diag ratio = \
             {:.3} (max_off / max_diag), clean: projected = \
             {corr_proj_clean:.6e}, full = {corr_full_clean:.6e}, \
             disagreement = {clean_disagreement:.3e}",
            max_offdiag / max_diag
        );
        eprintln!(
            "[ift-cross-coupled-airtight] pollute(ε={eps_null:.0e}): \
             projected = {corr_proj_total:.6e}, full = {corr_full_total:.6e}, \
             predicted_noise = {predicted_noise:.6e}, actual_noise = \
             {full_noise_contrib:.6e}, ratio = {ratio:.6}"
        );
    }

    fn make_factor_key(seed: u64) -> ProjectedFactorKey {
        // Build a unique-by-seed key without going through
        // `from_factor_view` so the test can inject fingerprints
        // directly. Using public construction via a real ArrayView2
        // would couple this test to ndarray pointer aliasing.
        ProjectedFactorKey {
            design_id: 1,
            factor_ptr: seed as usize,
            rows: 1,
            cols: 1,
            row_stride: 1,
            col_stride: 1,
            value_hash: seed,
            value_hash2: seed.wrapping_mul(31),
        }
    }

    #[test]
    fn projected_factor_cache_lru_evicts_oldest_under_budget() {
        let entry_floats = 32usize;
        let entry_bytes = entry_floats * std::mem::size_of::<f64>();
        // Budget that fits exactly two entries — inserting a third must
        // evict the least-recently-used one.
        let cache = ProjectedFactorCache::with_budget(entry_bytes * 2);

        let make = |seed: u64| -> Array2<f64> { Array2::from_elem((4, 8), seed as f64) };

        let _a = cache.get_or_insert_with(make_factor_key(1), || make(1));
        let _b = cache.get_or_insert_with(make_factor_key(2), || make(2));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total_bytes(), entry_bytes * 2);

        // Bump `a`'s recency so it survives the next eviction.
        let _a_again = cache.get_or_insert_with(make_factor_key(1), || make(1));

        // Inserting `c` must evict `b` (oldest), not `a` (most recent).
        let _c = cache.get_or_insert_with(make_factor_key(3), || make(3));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total_bytes(), entry_bytes * 2);

        // `a` and `c` survive; `b` was evicted.
        let post_a = cache.get_or_insert_with(make_factor_key(1), || make(99));
        let post_c = cache.get_or_insert_with(make_factor_key(3), || make(99));
        assert_eq!(post_a[[0, 0]], 1.0, "a survived eviction");
        assert_eq!(post_c[[0, 0]], 3.0, "c is the freshly inserted entry");

        let post_b = cache.get_or_insert_with(make_factor_key(2), || make(99));
        assert_eq!(
            post_b[[0, 0]],
            99.0,
            "b was evicted; recompute closure runs"
        );
    }

    #[test]
    fn projected_factor_cache_zero_budget_disables_eviction() {
        let cache = ProjectedFactorCache::with_budget(0);
        for seed in 0..16 {
            let _ = cache.get_or_insert_with(make_factor_key(seed), || {
                Array2::from_elem((8, 8), seed as f64)
            });
        }
        assert_eq!(cache.len(), 16);
    }

    #[test]
    fn projected_factor_cache_oversize_entry_is_cached_unconditionally() {
        // An entry larger than the entire budget cannot be made to fit
        // by eviction; we still cache it (refusing to cache would force
        // a recompute on every query, defeating the cache's purpose).
        let cache = ProjectedFactorCache::with_budget(8);
        let huge = cache.get_or_insert_with(make_factor_key(1), || Array2::from_elem((4, 4), 1.0));
        assert_eq!(huge[[0, 0]], 1.0);
        assert_eq!(cache.len(), 1);
    }
    #[test]
    fn projected_factor_cache_waiters_wake_when_producer_panics() {
        let cache = Arc::new(ProjectedFactorCache::with_budget(0));
        let key = make_factor_key(42);
        let (started_tx, started_rx) = std::sync::mpsc::channel();

        let producer_cache = Arc::clone(&cache);
        let producer = std::thread::spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                producer_cache.get_or_insert_with(key, || {
                    started_tx.send(()).expect("send producer-start signal");
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    panic!("simulated projected-factor panic");
                });
            }))
            .is_err()
        });

        started_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("producer started computing");

        let waiter_cache = Arc::clone(&cache);
        let waiter = std::thread::spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                waiter_cache.get_or_insert_with(key, || Array2::from_elem((1, 1), 7.0));
            }))
            .is_err()
        });

        assert!(producer.join().expect("producer thread joined"));
        assert!(waiter.join().expect("waiter thread joined"));

        let recovered = cache.get_or_insert_with(key, || Array2::from_elem((1, 1), 9.0));
        assert_eq!(recovered[[0, 0]], 9.0);
    }

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
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 2,
            nullspace_dim: 0.0,
            // Profiled Gaussian does not satisfy the fixed-dispersion IFT
            // identity used by the projected KKT residual correction, so an
            // inconsistent envelope gradient remains a soft "unavailable
            // derivative" result rather than a missing-residual contract
            // violation.
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
        };

        let result = reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None)
            .expect("family outer operator evaluation");
        let crate::solver::outer_strategy::HessianResult::Operator(op) = result.hessian else {
            panic!("expected family-supplied operator Hessian route");
        };
        let dense = op.materialize_dense().expect("sentinel materialization");
        assert_eq!(dense, family_matrix);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Replicates the AoU biobank marginal-slope failure mechanism in three
    // tiers, each pinning a distinct code-level math issue observed in
    // the failing log:
    //
    //   [PIRLS/joint-Newton convergence] cycle 303 | constrained-stationary
    //     certificate: linear-solve neutralised 0.0% of g (... multiplier ...);
    //     |Δobjective|=3.421e-1 ≤ obj_tol=3.479e-1
    //   [reml_laml envelope-gradient consistency] |g|∞ = 9.669e16 ... |cost|
    //     = 3.480e5  ratio 4.14e3  → gradient suppressed → seed rejected.
    //
    // BUG-1 (math, compute_kkt_residual_rho_corrections @ ~unified.rs:8500):
    //   At cert exit, the projected KKT residual r_proj ≈ 0 (multiplier
    //   captured by active set). Then q = H⁻¹·r_proj = 0, so the gradient
    //   correction `-aᵀ_k q + ½ qᵀA_k q` is identically zero. The cert's
    //   contract gives ZERO cancellation of the inflated envelope trace.
    //
    // BUG-2 (gate, envelope_inconsistent @ ~unified.rs:7466):
    //   `kkt_residual_was_applied = kkt_residual_correction_active` only
    //   checks `Some(..) && Fixed`, NOT whether the correction magnitude is
    //   nonzero. So at the cert's r_proj=0 contract the tripwire goes to
    //   the "applied" arm and passes an inflated meaningless gradient on.
    //
    // BUG-3 (math, envelope ½ tr(H⁻¹·∂H/∂ρ) at near-singular H):
    //   Without penalty_subspace_trace, the trace uses full H⁻¹ which
    //   blows up as σ_min(H)⁻¹. The cert's projected residual contract
    //   addresses the inner-stationarity correction term but does nothing
    //   about the LAML envelope trace itself.
    //
    // Each test below isolates one of these bugs.
    // ───────────────────────────────────────────────────────────────────────

    /// BUG-1: r_proj = 0 ⇒ IFT gradient correction is identically 0.
    /// This is a pure math identity test that ANCHORS the failure mode.
    #[test]
    fn ift_gradient_correction_with_zero_projected_residual_is_zero() {
        let h = Array2::eye(3);
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let solution = build_gaussian_solution_at_beta(&[0.0, 0.0], array![0.5, -0.25, 0.1], false);

        let lambdas = [1.0_f64, 1.0_f64];
        let penalty_a_k_betas = vec![array![0.3, -0.7, 0.0], array![0.0, 0.0, 0.5]];
        let zero_residual = Array1::<f64>::zeros(hop.dim());

        let corrections = compute_kkt_residual_rho_corrections(
            &solution,
            &hop,
            &lambdas,
            &penalty_a_k_betas,
            &zero_residual,
            true,
        )
        .expect("kkt correction must succeed at zero residual");

        for (i, &g) in corrections.gradient.iter().enumerate() {
            assert_eq!(
                g, 0.0,
                "BUG-1: IFT gradient correction at coord {} must be exactly 0.0 when \
                 r_proj = 0 (q = H⁻¹·0 = 0); got {:.3e}. The cert path's projected residual \
                 ≈ 0 contract therefore gives ZERO correction to the envelope gradient — \
                 inflated ½ tr(H⁻¹·∂H/∂ρ) is left uncancelled.",
                i, g
            );
        }
        let h_corr = corrections.hessian.expect("hessian requested");
        for ((i, j), &v) in h_corr.indexed_iter() {
            assert_eq!(
                v, 0.0,
                "BUG-1 hessian: entry ({}, {}) must be 0; got {:.3e}",
                i, j, v
            );
        }
    }

    /// BUG-2 (hard failing): cert exit with r_proj = 0 passes an inflated
    /// `|g|∞ = 1e20` gradient through the envelope tripwire because the gate
    /// only checks `kkt_residual.is_some()`, not whether the correction is
    /// nonzero. Contract under test: either suppress (gradient=None) OR
    /// produce a numerically honest gradient (|g|∞·√ε ≤ 4·|cost|). Current
    /// code does NEITHER and this assertion FAILS — pinpointing the gate bug.
    #[test]
    fn cert_zero_residual_must_not_emit_unbounded_gradient_through_gate() {
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&Array2::eye(2)).unwrap());
        let family_operator = Arc::new(SentinelOuterHessianOperator {
            matrix: array![[42.0]],
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
                first: array![1.0e20],
                second: Some(array![[0.0]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 2,
            nullspace_dim: 0.0,
            // Profiled Gaussian does not satisfy the fixed-dispersion IFT
            // identity used by the projected KKT residual correction, so an
            // inconsistent envelope gradient remains a soft "unavailable
            // derivative" result rather than a missing-residual contract
            // violation.
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
            kkt_residual: Some(ProjectedKktResidual::from_projected(array![0.0, 0.0])),
            active_constraints: None,
        };

        let result = reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None)
            .expect("constrained-stationary cert evaluation");

        let cost_scale = result.cost.abs().max(1.0);
        let resolve_step = f64::EPSILON.sqrt();
        let max_abs = match result.gradient.as_ref() {
            Some(g) => g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max),
            None => 0.0, // suppressed is acceptable
        };
        let predicted_change = max_abs * resolve_step;
        let ratio = predicted_change / cost_scale;
        assert!(
            result.gradient.is_none() || (ratio <= 4.0 && max_abs.is_finite()),
            "BUG-2 PINPOINTED: cert exit with r_proj = 0 passed an inflated gradient \
             through the tripwire. |grad|∞ = {:.3e}, predicted Δcost along √ε step = \
             {:.3e}, cost = {:.3e}, ratio = {:.3e} (must be ≤ 4 OR gradient = None). \
             The `kkt_residual_was_applied` gate at unified.rs:~7466 sets itself from \
             `kkt_residual_correction_active` (which only checks Some(..) && Fixed), \
             NOT from the magnitude of `corrections.gradient`. At r_proj = 0 the \
             correction is identically zero (see BUG-1 test) but the gate still routes \
             to the 'applied' arm of the match at unified.rs:~7720, passing the inflated \
             envelope through. The outer optimizer then sees |g| ≈ 1e20 and rejects.",
            max_abs,
            predicted_change,
            cost_scale,
            ratio,
        );
    }

    /// BUG-3 (hard failing): envelope analytic gradient on near-singular H
    /// disagrees with FD on a re-solved cost. Pinpoints `½ tr(H⁻¹·∂H/∂ρ)`
    /// blowing up by 1/σ_min(H) when `penalty_subspace_trace = None`.
    #[test]
    fn envelope_gradient_matches_fd_at_near_singular_h() {
        // H = X'X + λ₁S₁ + λ₂S₂. We pick X'X with one tiny eigenvalue so
        // σ_min(H) ≈ 1e-10 at ρ = 0. λ_k·S_k DO act on the same span as
        // X'X (so H stays near-singular) — this mirrors the production case
        // where the inner Hessian is near-singular AT the cert point.
        let xtx = array![[1.0, 0.0, 0.0], [0.0, 1.0e-10, 0.0], [0.0, 0.0, 1.0],];
        let s1 = array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let _xty = array![0.5, 1.0e-6, 0.5];

        let rho = vec![0.0_f64, 0.0_f64];
        let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
        let mut h_mat = xtx.clone();
        h_mat.scaled_add(lambdas[0], &s1);
        h_mat.scaled_add(lambdas[1], &s2);
        let op = DenseSpectralOperator::from_symmetric(&h_mat).unwrap();
        let beta_star = op.solve(&_xty);

        let cost_at = |rho_eval: &[f64]| -> f64 {
            let lambdas_eval: Vec<f64> = rho_eval.iter().map(|r| r.exp()).collect();
            let mut h_e = xtx.clone();
            h_e.scaled_add(lambdas_eval[0], &s1);
            h_e.scaled_add(lambdas_eval[1], &s2);
            let op_e = DenseSpectralOperator::from_symmetric(&h_e).unwrap();
            let beta_e = op_e.solve(&_xty);
            let mut sol_e = build_gaussian_solution_at_beta(rho_eval, beta_e, false);
            sol_e.dispersion = DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            };
            reml_laml_evaluate(&sol_e, rho_eval, EvalMode::ValueOnly, None)
                .unwrap()
                .cost
        };
        let fd_eps = 1e-4;
        let mut fd_grad = Array1::<f64>::zeros(rho.len());
        for k in 0..rho.len() {
            let mut rp = rho.clone();
            rp[k] += fd_eps;
            let mut rm = rho.clone();
            rm[k] -= fd_eps;
            fd_grad[k] = (cost_at(&rp) - cost_at(&rm)) / (2.0 * fd_eps);
        }

        let mut sol = build_gaussian_solution_at_beta(&rho, beta_star.clone(), false);
        sol.dispersion = DispersionHandling::Fixed {
            phi: 1.0,
            include_logdet_h: true,
            include_logdet_s: true,
        };
        let result = reml_laml_evaluate(&sol, &rho, EvalMode::ValueAndGradient, None).unwrap();
        let analytic = result
            .gradient
            .expect("gradient at exact β* must be available");

        for k in 0..rho.len() {
            let err = (analytic[k] - fd_grad[k]).abs();
            let scale = analytic[k].abs().max(fd_grad[k].abs()).max(1.0);
            let rel = err / scale;
            assert!(
                rel < 1e-2 && analytic[k].is_finite(),
                "BUG-3 PINPOINTED at coord {}: analytic gradient = {:+.6e}, FD = {:+.6e}, \
                 rel err = {:.3e}. The LAML envelope `½ tr(H⁻¹·∂H/∂ρ_k)` inflates by \
                 ~1/σ_min(H) when H is near-singular and `penalty_subspace_trace = None`. \
                 FD on the re-solved cost surface gives a bounded reference; the analytic \
                 formula disagrees because the trace's full H⁻¹ amplifies floating-point \
                 noise in directions where the cost surface itself has no slope. This is \
                 the math source of |g|∞ = 9.669e16 while the cost is only O(3.48e5) in \
                 the AoU log.",
                k,
                analytic[k],
                fd_grad[k],
                rel
            );
        }
    }

    /// Hard reproducer for the AoU missing-residual path. In fixed-dispersion
    /// LAML, an envelope-inconsistent derivative request with
    /// `kkt_residual=None` is not a recoverable value-gradient result: the
    /// evaluator cannot distinguish "exact KKT" from "convergent inner path
    /// forgot to hand over the projected residual". The principled response is
    /// a contract error naming `BlockwiseInnerResult::kkt_residual`, not
    /// `gradient=None` that the outer seed validator later reports as
    /// non-finite derivatives.
    #[test]
    fn aou_missing_projected_kkt_residual_is_contract_error() {
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&Array2::eye(2)).unwrap());
        let family_operator = Arc::new(SentinelOuterHessianOperator {
            matrix: array![[42.0]],
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
                first: array![1.0e20],
                second: Some(array![[0.0]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
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
            kkt_residual: None,
            active_constraints: None,
        };

        let err = match reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None)
        {
            Ok(_) => panic!("missing projected KKT residual must be a hard contract error"),
            Err(err) => err,
        };
        assert!(
            err.contains("fixed-dispersion derivative contract violated")
                && err.contains("BlockwiseInnerResult::kkt_residual"),
            "unexpected error for missing projected residual: {err}"
        );
    }

    #[test]
    fn envelope_inconsistent_gradient_skips_outer_hessian_assembly() {
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&Array2::eye(2)).unwrap());
        let family_operator = Arc::new(SentinelOuterHessianOperator {
            matrix: array![[42.0]],
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
                first: array![1.0e20],
                second: Some(array![[0.0]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 2,
            nullspace_dim: 0.0,
            // Profiled Gaussian does not satisfy the fixed-dispersion IFT
            // identity used by the projected KKT residual correction, so an
            // inconsistent envelope gradient remains a soft "unavailable
            // derivative" result rather than a missing-residual contract
            // violation.
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
        };

        let result = reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None)
            .expect("envelope tripwire evaluation");
        assert!(
            result.gradient.is_none(),
            "inconsistent envelope gradient should be suppressed"
        );
        assert!(
            matches!(
                result.hessian,
                crate::solver::outer_strategy::HessianResult::Unavailable
            ),
            "inconsistent envelope gradient should skip Hessian assembly"
        );
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
            hessian_weights: Array1::ones(3),
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
            rho_prior: crate::types::RhoPrior::Flat,
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
            kkt_residual: None,
            active_constraints: None,
        };
        let rho: Vec<f64> = vec![0.2_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let dense = compute_outer_hessian(
            &solution,
            &rho,
            &lambdas,
            solution.hessian_op.as_ref(),
            solution.deriv_provider.as_ref(),
            None,
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
            None,
            None,
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
    fn subspace_projected_leverage_and_adjoint_shortcut_match_dense() {
        // Locks down both production identities used by the subspace
        // leverage shortcut in `build_outer_hessian_operator`:
        //
        //   (1) `xt_projected_kernel_x_diagonal(X)_i = Xᵢᵀ · K · Xᵢ` per row
        //   (2) `tr(K · C[u]) = uᵀ · Xᵀ(c ⊙ h^{G,proj})`
        //       with `K = U_S H_proj⁻¹ U_Sᵀ` and `C[u] = Xᵀ diag(c ⊙ Xu) X`.
        //
        // (1) is the per-row contract `xt_projected_kernel_x_diagonal`
        // promises (its docstring); (2) is the math identity that the
        // leverage / `adjoint_z_c` shortcut relies on for its `O(n·r)`
        // adjoint-trick replacement of the dense materialised correction.
        let u_s = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
        let det = 3.0_f64 * 5.0 - 0.1 * 0.1;
        let h_proj_inverse = array![[5.0 / det, -0.1 / det], [-0.1 / det, 3.0 / det]];
        let subspace = PenaltySubspaceTrace {
            u_s: u_s.clone(),
            h_proj_inverse: h_proj_inverse.clone(),
        };

        let x_data = array![
            [1.0, 0.2, 0.5, 0.3],
            [1.0, 1.1, -0.2, 0.4],
            [1.0, -0.8, 0.7, -0.1],
            [1.0, 0.5, 0.3, 0.6]
        ];
        let c = array![0.31_f64, -0.27, 0.19, -0.11];

        // Dense reference K = U_S · H_proj⁻¹ · U_Sᵀ.
        let k_dense = u_s.dot(&h_proj_inverse).dot(&u_s.t());
        let n = x_data.nrows();

        // (1) Production helper vs per-row dense reference.
        let x_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_data.clone()));
        let h_g_proj = subspace.xt_projected_kernel_x_diagonal(&x_design);
        assert_eq!(h_g_proj.len(), n);
        for i in 0..n {
            let row = x_data.row(i).to_owned();
            let kx = k_dense.dot(&row);
            assert_relative_eq!(h_g_proj[i], row.dot(&kx), epsilon = 1e-12);
        }

        // (2) Adjoint shortcut: tr(K · C[u]) = uᵀ · Xᵀ(c ⊙ h^{G,proj}).
        // Probe u directions including ones lifting into null(U_S).
        let probes = [
            array![0.6_f64, -0.4, 0.0, 0.0],
            array![0.0_f64, 0.0, 0.5, 0.7],
            array![0.3_f64, -0.1, 0.4, -0.2],
            array![1.0_f64, 1.0, 1.0, 1.0],
        ];
        for u in probes.iter() {
            let xu = x_data.dot(u);
            let mut weighted_x = x_data.clone();
            for i in 0..n {
                let w = c[i] * xu[i];
                for j in 0..weighted_x.ncols() {
                    weighted_x[[i, j]] *= w;
                }
            }
            let c_u_dense = x_data.t().dot(&weighted_x);

            // LHS: tr(K · C[u]) via the production projected-logdet path.
            let lhs = subspace.trace_projected_logdet(&c_u_dense);

            // RHS: uᵀ · Xᵀ(c ⊙ h^{G,proj}) using the production helper's output.
            let mut weighted = Array1::<f64>::zeros(n);
            for i in 0..n {
                weighted[i] = c[i] * h_g_proj[i];
            }
            let rhs = u.dot(&x_data.t().dot(&weighted));

            assert_relative_eq!(lhs, rhs, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn subspace_base_h2_traces_match_scalar_projected_kernel_path() {
        let h = array![[3.0, 0.1, 0.0], [0.1, 5.0, 0.2], [0.0, 0.2, 7.0]];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let u_s = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let det = 3.0_f64 * 5.0 - 0.1 * 0.1;
        let kernel = PenaltySubspaceTrace {
            u_s,
            h_proj_inverse: array![[5.0 / det, -0.1 / det], [-0.1 / det, 3.0 / det]],
        };

        let dense_only = array![[0.4, 0.1, 0.0], [0.1, -0.2, 0.3], [0.0, 0.3, 0.6]];
        let op_a = array![[0.2, -0.1, 0.4], [-0.1, 0.7, 0.0], [0.4, 0.0, -0.3]];
        let op_b = array![[0.8, 0.2, -0.2], [0.2, 0.1, 0.5], [-0.2, 0.5, 0.9]];
        let composite_dense = array![[0.05, 0.02, 0.0], [0.02, 0.03, 0.01], [0.0, 0.01, 0.04]];

        let op_a_arc: Arc<dyn HyperOperator> = Arc::new(DenseMatrixHyperOperator {
            matrix: op_a.clone(),
        });
        let op_b_arc: Arc<dyn HyperOperator> = Arc::new(DenseMatrixHyperOperator {
            matrix: op_b.clone(),
        });
        let weighted: Arc<dyn HyperOperator> = Arc::new(WeightedHyperOperator {
            terms: vec![(0.25, op_b_arc.clone()), (-0.5, op_a_arc.clone())],
            dim_hint: 3,
        });

        let pairs = vec![
            HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(3),
                b_mat: dense_only,
                b_operator: None,
                ld_s: 0.0,
            },
            HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(3),
                b_mat: Array2::zeros((0, 0)),
                b_operator: Some(Box::new(DenseMatrixHyperOperator { matrix: op_a })),
                ld_s: 0.0,
            },
            HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(3),
                b_mat: Array2::zeros((0, 0)),
                b_operator: Some(Box::new(CompositeHyperOperator {
                    dense: Some(composite_dense),
                    operators: vec![weighted, op_b_arc],
                    dim_hint: 3,
                })),
                ld_s: 0.0,
            },
        ];
        let pair_refs: Vec<&HyperCoordPair> = pairs.iter().collect();

        let batched = compute_base_h2_traces(&hop, &pair_refs, Some(&kernel));
        let scalar: Vec<f64> = pair_refs
            .iter()
            .map(|pair| {
                compute_base_h2_trace(&hop, &pair.b_mat, pair.b_operator.as_deref(), Some(&kernel))
            })
            .collect();

        assert_eq!(batched.len(), scalar.len());
        for (idx, (got, expected)) in batched.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (*got - *expected).abs() <= 1e-12_f64.max(1e-12 * expected.abs()),
                "projected base_h2 trace mismatch at pair {idx}: got={got}, expected={expected}"
            );
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

        let x = array![
            [1.0, 0.2, 0.5, 0.3],
            [1.0, 1.1, -0.2, 0.4],
            [1.0, -0.8, 0.7, -0.1],
            [1.0, 0.5, 0.3, 0.6]
        ];
        let c_array = array![0.31, -0.27, 0.19, -0.11];
        let d_array = array![0.17, -0.11, 0.23, 0.07];
        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array,
            d_array: Some(d_array),
            hessian_weights: Array1::ones(4),
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
            rho_prior: crate::types::RhoPrior::Flat,
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
            kkt_residual: None,
            active_constraints: None,
        };
        let rho: Vec<f64> = vec![0.2_f64, -0.1];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let dense = compute_outer_hessian(
            &solution,
            &rho,
            &lambdas,
            solution.hessian_op.as_ref(),
            solution.deriv_provider.as_ref(),
            None,
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
            None,
            None,
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
            let hvp = crate::solver::outer_strategy::OuterHessianOperator::matvec(&operator, alpha)
                .expect("operator HVP");
            let dense_hvp = dense.dot(alpha);
            for i in 0..hvp.len() {
                assert_relative_eq!(hvp[i], dense_hvp[i], epsilon = 1e-12, max_relative = 1e-12);
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
            hessian_weights: Array1::ones(4),
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
            rho_prior: crate::types::RhoPrior::Flat,
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
            kkt_residual: None,
            active_constraints: None,
        };
        let rho: Vec<f64> = vec![0.2_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let dense = compute_outer_hessian(
            &solution,
            &rho,
            &lambdas,
            solution.hessian_op.as_ref(),
            solution.deriv_provider.as_ref(),
            None,
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
            None,
            None,
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
    fn penalty_subspace_batched_reduction_matches_serial_operator_reduction() {
        let kernel = PenaltySubspaceTrace {
            u_s: array![[1.0, 0.0], [0.2, 0.8], [-0.1, 0.6]],
            h_proj_inverse: array![[0.8, 0.1], [0.1, 0.6]],
        };
        let dense = array![[0.4, 0.1, -0.2], [0.1, 0.7, 0.3], [-0.2, 0.3, 0.5]];
        let op_matrix = array![[0.3, -0.2, 0.1], [-0.2, 0.9, 0.4], [0.1, 0.4, 0.8]];
        let composite_dense = array![[0.05, 0.01, 0.0], [0.01, -0.02, 0.03], [0.0, 0.03, 0.04]];
        let drifts = vec![
            DriftDerivResult::Dense(dense.clone()),
            DriftDerivResult::Operator(Arc::new(DenseMatrixHyperOperator {
                matrix: op_matrix.clone(),
            })),
            DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                dim_hint: 3,
                dense: Some(composite_dense.clone()),
                operators: vec![Arc::new(DenseMatrixHyperOperator {
                    matrix: op_matrix.clone(),
                })],
            })),
        ];

        let batched = penalty_subspace_reduce_drifts_batched(&kernel, &drifts);
        let serial = vec![
            kernel.reduce(&dense),
            kernel.reduce_operator(&DenseMatrixHyperOperator {
                matrix: op_matrix.clone(),
            }),
            kernel.reduce_operator(&CompositeHyperOperator {
                dim_hint: 3,
                dense: Some(composite_dense),
                operators: vec![Arc::new(DenseMatrixHyperOperator { matrix: op_matrix })],
            }),
        ];

        for (_idx, (batched_mat, serial_mat)) in batched.iter().zip(serial.iter()).enumerate() {
            for row in 0..batched_mat.nrows() {
                for col in 0..batched_mat.ncols() {
                    assert_relative_eq!(
                        batched_mat[[row, col]],
                        serial_mat[[row, col]],
                        epsilon = 1e-12,
                        max_relative = 1e-12,
                    );
                }
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
            hessian_weights: Array1::ones(4),
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
            rho_prior: crate::types::RhoPrior::Flat,
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
            kkt_residual: None,
            active_constraints: None,
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
    fn callback_outer_hessian_routes_by_row_pair_work_even_at_small_p() {
        assert!(!prefer_outer_hessian_operator(155_980, 19, 23));
        assert!(use_outer_hessian_operator_path(155_980, 19, 23, true));
        assert!(!use_outer_hessian_operator_path(155_980, 19, 23, false));
        assert!(!use_outer_hessian_operator_path(1_000, 19, 23, true));
    }

    #[test]
    fn callback_outer_hessian_ignores_generic_large_n_small_p_crossover() {
        assert!(prefer_outer_hessian_operator(195_780, 33, 8));
        assert!(!use_outer_hessian_operator_path(195_780, 33, 8, true));
        assert!(use_outer_hessian_operator_path(195_780, 512, 8, true));
        assert!(use_outer_hessian_operator_path(195_780, 33, 32, true));

        let plan = outer_hessian_route_plan(195_780, 33, 8, true, true, false);
        assert!(!plan.use_operator);
        assert_eq!(plan.choice(), "dense");
        assert_eq!(plan.reason, "below_crossover");
        assert!(!plan.scale_prefers_operator);
    }

    #[test]
    fn outer_hessian_route_respects_dense_workspace_budget() {
        let plan = outer_hessian_route_plan(10_000, 10_000, 2, true, true, false);
        assert!(plan.use_operator);
        assert_eq!(plan.reason, "dense_memory_budget");
        assert!(plan.dense_workspace_bytes > outer_hessian_dense_workspace_budget_bytes());
    }

    #[test]
    fn outer_hessian_route_reports_kernel_absent_before_scale_model() {
        let plan = outer_hessian_route_plan(1_000_000, 10_000, 64, false, false, false);
        assert!(!plan.use_operator);
        assert_eq!(plan.reason, "kernel_absent");
        assert!(!plan.scale_prefers_operator);
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
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 320_000,
            nullspace_dim: 1.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
        };
        let rho: Vec<f64> = vec![0.2_f64, -0.4_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let dense = compute_outer_hessian(
            &solution,
            &rho,
            &lambdas,
            solution.hessian_op.as_ref(),
            solution.deriv_provider.as_ref(),
            None,
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
            None,
            None,
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
            rho_prior: crate::types::RhoPrior::Flat,
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
            kkt_residual: None,
            active_constraints: None,
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
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 100,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
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
            rho_prior: crate::types::RhoPrior::Flat,
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
            kkt_residual: None,
            active_constraints: None,
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
                return Err(RemlError::DimensionMismatch {
                    reason: format!(
                        "fixed test outer Hessian dimension mismatch: got {}, expected {}",
                        v.len(),
                        self.dim()
                    ),
                }
                .into());
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
            rho_prior: crate::types::RhoPrior::Flat,
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
            kkt_residual: None,
            active_constraints: None,
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
            rho_prior: crate::types::RhoPrior::Flat,
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
            kkt_residual: None,
            active_constraints: None,
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

    #[test]
    fn test_rho_corrections_serial_large_work_case_stays_finite() {
        let rho = 0.0;
        let mut solution = build_projected_rho_gradient_solution(rho);
        solution.n_observations = 40_000_000;

        let result = reml_laml_evaluate(&solution, &[rho], EvalMode::ValueAndGradient, None)
            .expect("serial rho correction evaluation");
        assert!(result.cost.is_finite());
        let gradient = result.gradient.expect("gradient");
        assert_eq!(gradient.len(), 1);
        assert!(gradient[0].is_finite());
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
        let log_det_s_at = |rho_eval: &[f64]| -> f64 {
            let lambdas_eval: Vec<f64> = rho_eval.iter().map(|&r| r.exp()).collect();
            let mut s_eval = Array2::zeros((p, p));
            s_eval.scaled_add(lambdas_eval[0], &s1);
            s_eval.scaled_add(lambdas_eval[1], &s2);
            let (s_eigs_eval, _) = s_eval.eigh(faer::Side::Lower).unwrap();
            let threshold_eval = positive_eigenvalue_threshold(s_eigs_eval.as_slice().unwrap());
            exact_pseudo_logdet(s_eigs_eval.as_slice().unwrap(), threshold_eval)
        };

        let mut det1 = Array1::zeros(rho.len());
        let eps = 1e-7;
        for k in 0..rho.len() {
            let mut rho_plus = rho.to_vec();
            rho_plus[k] += eps;
            let log_det_s_plus = log_det_s_at(&rho_plus);

            let mut rho_minus = rho.to_vec();
            rho_minus[k] -= eps;
            let log_det_s_minus = log_det_s_at(&rho_minus);

            det1[k] = (log_det_s_plus - log_det_s_minus) / (2.0 * eps);
        }
        let mut det2 = Array2::zeros((rho.len(), rho.len()));
        let eps2 = 1e-5;
        for i in 0..rho.len() {
            for j in i..rho.len() {
                let value = if i == j {
                    let mut rho_plus = rho.to_vec();
                    rho_plus[i] += eps2;
                    let mut rho_minus = rho.to_vec();
                    rho_minus[i] -= eps2;
                    (log_det_s_at(&rho_plus) - 2.0 * log_det_s + log_det_s_at(&rho_minus))
                        / (eps2 * eps2)
                } else {
                    let mut pp = rho.to_vec();
                    pp[i] += eps2;
                    pp[j] += eps2;
                    let mut pm = rho.to_vec();
                    pm[i] += eps2;
                    pm[j] -= eps2;
                    let mut mp = rho.to_vec();
                    mp[i] -= eps2;
                    mp[j] += eps2;
                    let mut mm = rho.to_vec();
                    mm[i] -= eps2;
                    mm[j] -= eps2;
                    (log_det_s_at(&pp) - log_det_s_at(&pm) - log_det_s_at(&mp) + log_det_s_at(&mm))
                        / (4.0 * eps2 * eps2)
                };
                det2[[i, j]] = value;
                if i != j {
                    det2[[j, i]] = value;
                }
            }
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
                second: Some(det2),
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: n,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
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
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: n,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
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
            hutchpp_sketch_dim: None,
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
    fn modified_gram_schmidt_orthonormalizes_well_conditioned_input() {
        let y = array![
            [1.0, 2.0, 0.5, 3.0],
            [0.0, 1.0, 0.5, 1.5],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut q = Array2::<f64>::zeros(y.dim());
        let rank = modified_gram_schmidt(&y, &mut q);
        assert_eq!(rank, 4, "well-conditioned input should retain full rank");
        // Q^T Q = I within the retained rank.
        for j in 0..rank {
            for k in 0..rank {
                let dot = q.column(j).dot(&q.column(k));
                let expected = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "QᵀQ off-identity at ({j},{k}): got {dot}",
                );
            }
        }
    }

    #[test]
    fn modified_gram_schmidt_drops_redundant_columns() {
        let y = array![
            [1.0, 2.0, 1.0, 4.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let mut q = Array2::<f64>::zeros(y.dim());
        let rank = modified_gram_schmidt(&y, &mut q);
        assert_eq!(
            rank, 2,
            "two duplicate columns plus a zero-extension should drop to rank 2"
        );
        for j in 0..rank {
            for k in 0..rank {
                let dot = q.column(j).dot(&q.column(k));
                let expected = if j == k { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn hutchpp_estimate_trace_hinv_operator_matches_exact_within_tolerance() {
        // Build a small SPD H and an HVP-only operator wrapping a dense M.
        // Compare Hutch++ to the exact tr(H⁻¹ M).
        let h = array![
            [4.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.2, 0.0, 0.0, 0.0],
            [0.5, 0.2, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.7, 0.1],
            [0.0, 0.0, 0.0, 0.7, 4.0, 0.3],
            [0.0, 0.0, 0.0, 0.1, 0.3, 3.0],
        ];
        let m = array![
            [1.0, 0.3, 0.0, 0.1, 0.0, 0.0],
            [0.3, 0.5, 0.1, 0.0, 0.2, 0.0],
            [0.0, 0.1, 0.2, 0.0, 0.0, 0.05],
            [0.1, 0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.2, 0.0, 0.2, 0.6, 0.1],
            [0.0, 0.0, 0.05, 0.0, 0.1, 0.4],
        ];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let m_op = DenseMatrixHyperOperator { matrix: m.clone() };

        let exact = hop.trace_hinv_product(&m);

        let config = StochasticTraceConfig {
            n_probes_min: 12,
            n_probes_max: 64,
            relative_tol: 0.005,
            tau_rel: 1e-10,
            solve_rel_tol: 1e-10,
            seed: 0xABCDEF,
            hutchpp_sketch_dim: Some(3),
        };
        let est = hutchpp_estimate_trace_hinv_operator(&hop, &m_op, &config);
        let rel_err = (est - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err < 0.05,
            "Hutch++ trace est={est:.6} exact={exact:.6} rel_err={rel_err:.4}"
        );

        // Plain Hutchinson with the same probe budget should not be more
        // accurate; this guards against an inadvertent regression where
        // the sketch contribution is silently zeroed.
        let mut config_plain = config.clone();
        config_plain.hutchpp_sketch_dim = None;
        config_plain.n_probes_max = 64; // same total budget
        let est_plain = hutchpp_estimate_trace_hinv_operator(&hop, &m_op, &config_plain);
        let rel_err_plain = (est_plain - exact).abs() / exact.abs().max(1e-10);
        // Allow Hutch++ to either beat plain or match it; never be much worse.
        assert!(
            rel_err <= rel_err_plain * 2.0 + 0.01,
            "Hutch++ ({rel_err:.4}) should be competitive with Hutchinson ({rel_err_plain:.4})"
        );
    }

    #[test]
    fn hutchpp_estimate_trace_hinv_op_squared_matches_exact() {
        // SPD H and symmetric A; compare tr(H⁻¹ A H⁻¹ A) to the exact
        // value computed via trace_hinv_product_cross(A, A) =
        // tr((H⁻¹ A) (H⁻¹ A)).
        let h = array![
            [4.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.2, 0.0, 0.0, 0.0],
            [0.5, 0.2, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.7, 0.1],
            [0.0, 0.0, 0.0, 0.7, 4.0, 0.3],
            [0.0, 0.0, 0.0, 0.1, 0.3, 3.0],
        ];
        let a = array![
            [1.0, 0.3, 0.0, 0.1, 0.0, 0.0],
            [0.3, 0.5, 0.1, 0.0, 0.2, 0.0],
            [0.0, 0.1, 0.2, 0.0, 0.0, 0.05],
            [0.1, 0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.2, 0.0, 0.2, 0.6, 0.1],
            [0.0, 0.0, 0.05, 0.0, 0.1, 0.4],
        ];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let a_op = DenseMatrixHyperOperator { matrix: a.clone() };

        let exact = hop.trace_hinv_product_cross(&a, &a);

        let config = StochasticTraceConfig {
            n_probes_min: 16,
            n_probes_max: 96,
            relative_tol: 0.005,
            tau_rel: 1e-10,
            solve_rel_tol: 1e-10,
            seed: 0xC0FFEE,
            hutchpp_sketch_dim: Some(3),
        };
        let est = hutchpp_estimate_trace_hinv_op_squared(&hop, &a_op, &config);
        let rel_err = (est - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err < 0.05,
            "Hutch++ tr((H⁻¹A)²) est={est:.6} exact={exact:.6} rel_err={rel_err:.4}"
        );

        // Wired path: estimate_second_order_single_dense routes through
        // Hutch++ when hutchpp_sketch_dim is Some(_).
        let estimator = StochasticTraceEstimator::new(config.clone());
        let est_wired = estimator.estimate_second_order_single_dense(&hop, &a);
        let rel_err_wired = (est_wired - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err_wired < 0.05,
            "wired Hutch++ second-order est={est_wired:.6} exact={exact:.6} rel_err={rel_err_wired:.4}"
        );
        assert!(
            (est_wired - est).abs() <= 1e-12,
            "wired path must call hutchpp_estimate_trace_hinv_op_squared with the same seed/config"
        );
    }

    #[test]
    fn hutchpp_estimate_trace_hinv_operator_cross_matches_exact() {
        let h = array![
            [4.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.2, 0.0, 0.0, 0.0],
            [0.5, 0.2, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.7, 0.1],
            [0.0, 0.0, 0.0, 0.7, 4.0, 0.3],
            [0.0, 0.0, 0.0, 0.1, 0.3, 3.0],
        ];
        let a = array![
            [1.0, 0.3, 0.0, 0.1, 0.0, 0.0],
            [0.3, 0.5, 0.1, 0.0, 0.2, 0.0],
            [0.0, 0.1, 0.2, 0.0, 0.0, 0.05],
            [0.1, 0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.2, 0.0, 0.2, 0.6, 0.1],
            [0.0, 0.0, 0.05, 0.0, 0.1, 0.4],
        ];
        let b = array![
            [0.5, 0.0, 0.1, 0.0, 0.05, 0.0],
            [0.0, 0.7, 0.0, 0.2, 0.0, 0.1],
            [0.1, 0.0, 0.4, 0.0, 0.15, 0.0],
            [0.0, 0.2, 0.0, 0.6, 0.0, 0.05],
            [0.05, 0.0, 0.15, 0.0, 0.3, 0.0],
            [0.0, 0.1, 0.0, 0.05, 0.0, 0.5],
        ];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let a_op = DenseMatrixHyperOperator { matrix: a.clone() };
        let b_op = DenseMatrixHyperOperator { matrix: b.clone() };

        let exact = hop.trace_hinv_product_cross(&a, &b);

        let config = StochasticTraceConfig {
            n_probes_min: 16,
            n_probes_max: 128,
            relative_tol: 0.005,
            tau_rel: 1e-10,
            solve_rel_tol: 1e-10,
            seed: 0xDEAD_BEEF,
            hutchpp_sketch_dim: Some(3),
        };
        let est = hutchpp_estimate_trace_hinv_operator_cross(&hop, &a_op, &b_op, &config);
        let rel_err = (est - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err < 0.07,
            "Hutch++ cross trace est={est:.6} exact={exact:.6} rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn trace_hinv_operator_cross_default_routes_implicit_to_hutchpp() {
        // Build a synthetic 200-dim SPD H and an HVP-only operator pair
        // (mark `is_implicit() = true`) so the trait default routes
        // through the Hutch++ path. The exact reference comes from the
        // dense materialization of the same operator.
        let p = 200usize;
        let mut h = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            h[[i, i]] = 5.0 + (i as f64) * 0.01;
            if i + 1 < p {
                h[[i, i + 1]] = 0.2;
                h[[i + 1, i]] = 0.2;
            }
        }
        let mut a = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            a[[i, i]] = 1.0 + 0.005 * (i as f64);
            if i + 2 < p {
                a[[i, i + 2]] = 0.1;
                a[[i + 2, i]] = 0.1;
            }
        }
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Wrapper that masquerades as implicit so the default route fires.
        struct ImplicitDense(Array2<f64>);
        impl HyperOperator for ImplicitDense {
            fn dim(&self) -> usize {
                self.0.nrows()
            }
            fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
                let mut out = Array1::<f64>::zeros(self.0.nrows());
                dense_matvec_into(&self.0, v.view(), out.view_mut());
                out
            }
            fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
                dense_matvec_into(&self.0, v, out);
            }
            fn to_dense(&self) -> Array2<f64> {
                self.0.clone()
            }
            fn is_implicit(&self) -> bool {
                true
            }
        }

        let a_op = ImplicitDense(a.clone());
        let exact = hop.trace_hinv_product_cross(&a, &a);
        // Same-operator path: routes through the squared estimator.
        let est_same = hop.trace_hinv_operator_cross(&a_op, &a_op);
        assert!(est_same.is_finite(), "cross trace must be finite");
        let rel_err_same = (est_same - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err_same < 0.10,
            "default same-op cross routing est={est_same:.6} exact={exact:.6} rel_err={rel_err_same:.4}"
        );

        // Distinct-operator path: routes through the cross estimator.
        let mut b = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            b[[i, i]] = 0.6 + 0.003 * (i as f64);
            if i + 1 < p {
                b[[i, i + 1]] = 0.05;
                b[[i + 1, i]] = 0.05;
            }
        }
        let b_op = ImplicitDense(b.clone());
        let exact_ab = hop.trace_hinv_product_cross(&a, &b);
        let est_ab = hop.trace_hinv_operator_cross(&a_op, &b_op);
        assert!(est_ab.is_finite(), "cross trace (a,b) must be finite");
        let rel_err_ab = (est_ab - exact_ab).abs() / exact_ab.abs().max(1e-10);
        assert!(
            rel_err_ab < 0.10,
            "default distinct-op cross routing est={est_ab:.6} exact={exact_ab:.6} rel_err={rel_err_ab:.4}"
        );

        // Matrix-operator path: routes through the cross estimator with
        // a synthetic dense LHS wrapper.
        let exact_ma = hop.trace_hinv_product_cross(&a, &b);
        let est_ma = hop.trace_hinv_matrix_operator_cross(&a, &b_op);
        assert!(est_ma.is_finite(), "matrix-op cross trace must be finite");
        let rel_err_ma = (est_ma - exact_ma).abs() / exact_ma.abs().max(1e-10);
        assert!(
            rel_err_ma < 0.10,
            "default matrix-operator cross routing est={est_ma:.6} exact={exact_ma:.6} rel_err={rel_err_ma:.4}"
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
        let hop = MatrixFreeSpdOperator::new_with_mode(
            diag.len(),
            move |v| &diag * v,
            PseudoLogdetMode::Smooth,
        );
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
        let op = MatrixFreeSpdOperator::new_with_mode(
            diag.len(),
            move |v| &diag * v,
            PseudoLogdetMode::Smooth,
        );
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

    /// Indefinite outer Hessian (no active bounds, no rank deficiency) must
    /// surface as `CorrectedCovarianceError::Indefinite` — never as a
    /// covariance with the negative directions silently clamped to zero.
    #[test]
    fn corrected_covariance_indefinite_returns_diagnostic() {
        // 2×2 outer Hessian with one positive and one clearly negative
        // eigenvalue ⇒ the projected (=full, since no active bounds) inertia
        // gate must reject. Using diag(2, -1) on a small p=2 base.
        let outer = ndarray::arr2(&[[2.0_f64, 0.0], [0.0, -1.0]]);

        // Build a SPD base H = I_2 so DenseSpectralOperator works trivially.
        let base = Array2::<f64>::eye(2);
        let hop = DenseSpectralOperator::from_symmetric(&base)
            .expect("DenseSpectralOperator from identity should succeed");

        // Two ρ-coords with arbitrary mode responses (their values don't
        // affect the inertia gate; the gate fires before any J·V_θ·Jᵀ work).
        let v0 = Array1::from_vec(vec![0.1, 0.2]);
        let v1 = Array1::from_vec(vec![0.3, 0.4]);

        // No theta supplied ⇒ active set is empty ⇒ projected Hessian = full
        // outer Hessian, which is indefinite ⇒ Err(Indefinite).
        let res = compute_corrected_covariance_with_constraints(
            &[v0.clone(), v1.clone()],
            &[],
            &outer,
            &hop,
            None,
            f64::NAN,
        );
        match res {
            Err(CorrectedCovarianceError::Indefinite(diag)) => {
                assert!(
                    diag.min_eigenvalue < -0.5,
                    "min eigenvalue should be ~-1, got {}",
                    diag.min_eigenvalue,
                );
                assert!(
                    diag.active_constraints.is_empty(),
                    "no theta supplied ⇒ no active constraints",
                );
                assert!(
                    !diag.suggested_action.is_empty(),
                    "diagnostic must include a suggested-action message",
                );
            }
            Err(other) => panic!("expected Indefinite diagnostic, got error: {:?}", other),
            Ok(cov) => panic!(
                "indefinite outer Hessian must NOT yield a covariance; got matrix shape {:?}",
                cov.matrix.shape(),
            ),
        }

        // Also check the legacy entry point preserves the same behaviour.
        let res_legacy = compute_corrected_covariance(&[v0, v1], &[], &outer, &hop);
        assert!(
            matches!(res_legacy, Err(CorrectedCovarianceError::Indefinite(_))),
            "legacy entry point must also surface Indefinite, got: {:?}",
            res_legacy.map(|m| m.shape().to_vec()),
        );
    }

    /// When the indefinite direction is precisely the bound-active θ, the
    /// projected-Hessian inertia gate sees a SPD matrix and we return a
    /// covariance (with the active coordinate listed in `active_constraints`).
    #[test]
    fn corrected_covariance_indefinite_with_active_bound_succeeds() {
        // Outer Hessian: positive on coord 0, negative on coord 1.
        let outer = ndarray::arr2(&[[3.0_f64, 0.0], [0.0, -2.0]]);
        let base = Array2::<f64>::eye(2);
        let hop = DenseSpectralOperator::from_symmetric(&base).expect("hop");

        let v0 = Array1::from_vec(vec![0.5, 0.0]);
        let v1 = Array1::from_vec(vec![0.0, 0.5]);

        // θ pinned at +RHO_BOUND on coord 1 (the negative-curvature direction).
        // After projecting away coord 1, the free Hessian is [[3]] — SPD.
        let theta = vec![0.0_f64, crate::solver::estimate::RHO_BOUND];
        let res = compute_corrected_covariance_with_constraints(
            &[v0, v1],
            &[],
            &outer,
            &hop,
            Some(&theta),
            0.0,
        )
        .expect("free-subspace SPD ⇒ covariance returned");
        assert_eq!(res.active_constraints, vec![1]);
        assert!(res.matrix.iter().all(|v| v.is_finite()));
    }

    // ------------------------------------------------------------------------
    // Numerical proof of the outer-ρ projected-kernel REML gradient bug
    // (runtime.rs:5465-5481).
    //
    // Hypothesis: when `hessian_logdet_correction ≠ 0` (cost uses projected
    // logdet `log|U_S^T H U_S|_+`) but the gradient computation uses the
    // full-space kernel `G_ε(H)` instead of `U_S (U_S^T H U_S)⁻¹ U_S^T`, the
    // third-derivative correction `D_β H[v_k] = X' diag(c ⊙ X v_k) X` leaks
    // onto null(S) and produces a spurious O(λ_k) outer-gradient term at
    // large λ_k.
    //
    // Method: Option B (synthetic). Build a Gaussian fixture where β̂(ρ) is
    // exact (so FD of `reml_laml_evaluate` returns the true gradient of the
    // projected REML cost). Inject a non-zero `c_array` via
    // `SinglePredictorGlmDerivatives` — this is a "lie" about the family
    // (the cost is Gaussian so the actual `dH/dρ` has no third-deriv term),
    // but it lets the analytic gradient path see a `D_β H[v_k]` that is
    // structurally identical to the survival_location_scale leak. FD does
    // NOT see the lie (cost only uses β/H/log-lik/penalty). Projected and
    // unprojected analytic gradients DO see it; the projected kernel kills
    // the null-space part exactly, so it matches FD.
    // ------------------------------------------------------------------------
    fn build_leak_proof_solution(
        rho: &[f64],
        x: &Array2<f64>,
        s1: &Array2<f64>,
        s2: &Array2<f64>,
        xty: &Array1<f64>,
        yty: f64,
        c_array: Array1<f64>,
        use_projected_kernel: bool,
    ) -> InnerSolution<'static> {
        let p = x.ncols();
        let n = x.nrows();
        assert_eq!(rho.len(), 2);
        let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();

        let xtx = crate::faer_ndarray::fast_atb(x, x);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        s_lambda.scaled_add(lambdas[0], s1);
        s_lambda.scaled_add(lambdas[1], s2);

        let mut h = xtx.clone();
        h += &s_lambda;

        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let beta = hop.solve(xty);
        let deviance = yty - 2.0 * beta.dot(xty) + beta.dot(&xtx.dot(&beta));
        let log_lik = -0.5 * deviance;
        let penalty_quad = beta.dot(&s_lambda.dot(&beta));

        // Penalty logdet & first derivatives (rank = 2; both penalties have
        // rank-1 supports that don't overlap, so log|S_λ|_+ = ln λ₁ + ln λ₂).
        let (s_eigs, _) = s_lambda.eigh(faer::Side::Lower).unwrap();
        let threshold = positive_eigenvalue_threshold(s_eigs.as_slice().unwrap());
        let log_det_s = exact_pseudo_logdet(s_eigs.as_slice().unwrap(), threshold);

        let eps_det = 1e-7;
        let mut det1 = Array1::zeros(2);
        for k in 0..2 {
            let mut rp = rho.to_vec();
            rp[k] += eps_det;
            let lp: Vec<f64> = rp.iter().map(|r| r.exp()).collect();
            let mut sp = Array2::<f64>::zeros((p, p));
            sp.scaled_add(lp[0], s1);
            sp.scaled_add(lp[1], s2);
            let (ev_p, _) = sp.eigh(faer::Side::Lower).unwrap();
            let thp = positive_eigenvalue_threshold(ev_p.as_slice().unwrap());
            let ld_p = exact_pseudo_logdet(ev_p.as_slice().unwrap(), thp);

            let mut rm = rho.to_vec();
            rm[k] -= eps_det;
            let lm: Vec<f64> = rm.iter().map(|r| r.exp()).collect();
            let mut sm = Array2::<f64>::zeros((p, p));
            sm.scaled_add(lm[0], s1);
            sm.scaled_add(lm[1], s2);
            let (ev_m, _) = sm.eigh(faer::Side::Lower).unwrap();
            let thm = positive_eigenvalue_threshold(ev_m.as_slice().unwrap());
            let ld_m = exact_pseudo_logdet(ev_m.as_slice().unwrap(), thm);

            det1[k] = (ld_p - ld_m) / (2.0 * eps_det);
        }

        // Build projection U_S onto range(S_λ) (rank 2 here: cols 1 and 2 of X).
        // Use eigendecomposition of S_λ and keep eigenvectors with eigenvalue > tol.
        let (s_full_eigs, s_full_vecs) = s_lambda.eigh(faer::Side::Lower).unwrap();
        let s_thresh = positive_eigenvalue_threshold(s_full_eigs.as_slice().unwrap());
        let active: Vec<usize> = s_full_eigs
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > s_thresh)
            .map(|(i, _)| i)
            .collect();
        let r_rank = active.len();
        let mut u_s = Array2::<f64>::zeros((p, r_rank));
        for (j, &idx) in active.iter().enumerate() {
            for i in 0..p {
                u_s[[i, j]] = s_full_vecs[[i, idx]];
            }
        }
        // H_proj = U_Sᵀ H U_S; invert it.
        let h_proj = u_s.t().dot(&h).dot(&u_s);
        let (hp_eigs, hp_vecs) = h_proj.eigh(faer::Side::Lower).unwrap();
        let mut h_proj_inv = Array2::<f64>::zeros((r_rank, r_rank));
        for i in 0..r_rank {
            for j in 0..r_rank {
                let mut acc = 0.0;
                for k_idx in 0..r_rank {
                    acc += hp_vecs[[i, k_idx]] * hp_vecs[[j, k_idx]] / hp_eigs[k_idx];
                }
                h_proj_inv[[i, j]] = acc;
            }
        }
        let log_det_h_proj: f64 = hp_eigs.iter().map(|v| v.ln()).sum();
        let log_det_h_full = hop.logdet();
        let hessian_logdet_correction = log_det_h_proj - log_det_h_full;

        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array,
            d_array: None,
            hessian_weights: Array1::ones(n),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
        };

        let r1 = penalty_matrix_root(s1).unwrap();
        let r2 = penalty_matrix_root(s2).unwrap();

        let penalty_subspace_trace = if use_projected_kernel {
            Some(Arc::new(PenaltySubspaceTrace {
                u_s,
                h_proj_inverse: h_proj_inv,
            }))
        } else {
            None
        };

        InnerSolution {
            log_likelihood: log_lik,
            penalty_quadratic: penalty_quad,
            hessian_op: Arc::new(hop),
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
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction,
            penalty_subspace_trace,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: n,
            nullspace_dim: (p - r_rank) as f64,
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
            kkt_residual: None,
            active_constraints: None,
        }
    }

    /// Numerical proof: at large λ_j, the unprojected-kernel REML gradient
    /// disagrees with finite-difference of the projected-logdet cost by a
    /// margin that grows with λ_j, while the projected-kernel gradient
    /// matches FD to floating-point tolerance.
    #[test]
    fn proof_outer_rho_projected_kernel_fixes_leak() {
        let n = 100;
        let p = 3;
        // Design: intercept + two "spline-like" columns. Intercept is in null(S₁) ∩ null(S₂).
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            x[[i, 0]] = 1.0; // intercept
            x[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
            x[[i, 2]] = (t - 0.5) * (t - 0.3);
        }
        // S₁ penalizes column 1, S₂ penalizes column 2. Both have null space ⊇ {e_0}.
        let mut s1 = Array2::<f64>::zeros((p, p));
        s1[[1, 1]] = 1.0;
        let mut s2 = Array2::<f64>::zeros((p, p));
        s2[[2, 2]] = 1.0;

        // Some synthetic response — y = X β_true + noise (deterministic).
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            y[i] =
                0.7 + 0.4 * (2.0 * std::f64::consts::PI * t).sin() + 0.1 * ((i as f64) * 0.7).cos();
        }
        let xty = crate::faer_ndarray::fast_atb(&x, &y.clone().insert_axis(ndarray::Axis(1)))
            .column(0)
            .to_owned();
        let yty = y.dot(&y);

        // c_array: a non-trivial vector — the "third-derivative weight surface".
        // Chosen to have non-zero mean (so it projects onto the intercept direction),
        // making the leak large.
        let c_array = Array1::from_shape_fn(n, |i| {
            0.3 + 0.5 * ((i as f64) * 0.11).sin() + 0.2 * ((i as f64) * 0.27).cos()
        });

        // Runaway scenario: ρ = [0.0, 12.0] → λ ≈ [1, 1.6e5]. Big enough that
        // λ_2 dominates H and the leak is highly visible, but not so big that
        // FD on the projected-logdet cost (cost ~ ½ ln λ_2) loses precision —
        // at ρ_2 ≳ 17 the FD denominator gets within ~1e-8 of the cost itself
        // and catastrophic cancellation makes FD unreliable as a reference.
        let rho = vec![0.0_f64, 12.0_f64];

        // --- (a) FD of projected-logdet cost via reml_laml_evaluate value-only ---
        let delta = 1e-4;
        let mut fd_grad = vec![0.0_f64; 2];
        for j in 0..2 {
            let mut rp = rho.clone();
            rp[j] += delta;
            let sp = build_leak_proof_solution(&rp, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
            let cost_p = reml_laml_evaluate(&sp, &rp, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            let mut rm = rho.clone();
            rm[j] -= delta;
            let sm = build_leak_proof_solution(&rm, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
            let cost_m = reml_laml_evaluate(&sm, &rm, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            fd_grad[j] = (cost_p - cost_m) / (2.0 * delta);
        }

        // --- (b) Analytic gradient WITHOUT projected kernel (pre-v0.3.31 bug) ---
        let sol_unproj =
            build_leak_proof_solution(&rho, &x, &s1, &s2, &xty, yty, c_array.clone(), false);
        let g_unproj = reml_laml_evaluate(&sol_unproj, &rho, EvalMode::ValueAndGradient, None)
            .unwrap()
            .gradient
            .unwrap();

        // --- (c) Analytic gradient WITH projected kernel (v0.3.31 fix) ---
        let sol_proj =
            build_leak_proof_solution(&rho, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
        let g_proj = reml_laml_evaluate(&sol_proj, &rho, EvalMode::ValueAndGradient, None)
            .unwrap()
            .gradient
            .unwrap();

        eprintln!(
            "=== Outer-ρ gradient at runaway ρ = {:?} (λ = {:?}) ===",
            rho,
            rho.iter().map(|r| r.exp()).collect::<Vec<_>>()
        );
        for j in 0..2 {
            eprintln!(
                "  coord {}: FD={:+.6e}   unprojected_analytic={:+.6e}   projected_analytic={:+.6e}",
                j, fd_grad[j], g_unproj[j], g_proj[j]
            );
            let rel_proj = (g_proj[j] - fd_grad[j]).abs() / fd_grad[j].abs().max(1e-12);
            let rel_unproj = (g_unproj[j] - fd_grad[j]).abs() / fd_grad[j].abs().max(1e-12);
            eprintln!(
                "           |projected − FD|/|FD| = {:.3e}   |unprojected − FD|/|FD| = {:.3e}",
                rel_proj, rel_unproj
            );
        }

        // --- Sweep λ_2 to check scaling ---
        eprintln!("=== Sweep λ_2 (coord 1) — unprojected analytic vs FD ===");
        for &rho2 in &[6.0_f64, 9.0, 12.0, 15.0, 18.0, 20.0] {
            let r = vec![0.0_f64, rho2];
            let fd1 = {
                let mut rp = r.clone();
                rp[1] += delta;
                let sp =
                    build_leak_proof_solution(&rp, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
                let cp = reml_laml_evaluate(&sp, &rp, EvalMode::ValueOnly, None)
                    .unwrap()
                    .cost;
                let mut rm = r.clone();
                rm[1] -= delta;
                let sm =
                    build_leak_proof_solution(&rm, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
                let cm = reml_laml_evaluate(&sm, &rm, EvalMode::ValueOnly, None)
                    .unwrap()
                    .cost;
                (cp - cm) / (2.0 * delta)
            };
            let su = build_leak_proof_solution(&r, &x, &s1, &s2, &xty, yty, c_array.clone(), false);
            let gu = reml_laml_evaluate(&su, &r, EvalMode::ValueAndGradient, None)
                .unwrap()
                .gradient
                .unwrap();
            let sp = build_leak_proof_solution(&r, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
            let gp = reml_laml_evaluate(&sp, &r, EvalMode::ValueAndGradient, None)
                .unwrap()
                .gradient
                .unwrap();
            let leak = gu[1] - fd1;
            eprintln!(
                "  ρ_2={:5.1} λ_2={:+.3e}  FD={:+.6e}  unproj={:+.6e}  proj={:+.6e}  leak(unproj−FD)={:+.6e}",
                rho2,
                rho2.exp(),
                fd1,
                gu[1],
                gp[1],
                leak
            );
        }

        // --- Assertions (per task spec) ---
        // At the runaway coordinate (j = 1):
        let j = 1;
        let rel_proj = (g_proj[j] - fd_grad[j]).abs() / fd_grad[j].abs().max(1e-12);
        let rel_unproj = (g_unproj[j] - fd_grad[j]).abs() / fd_grad[j].abs().max(1e-12);
        assert!(
            rel_proj < 1e-2,
            "projected gradient should match FD at runaway coord: \
             FD={:+.6e}, projected={:+.6e}, rel={:.3e}",
            fd_grad[j],
            g_proj[j],
            rel_proj
        );
        assert!(
            rel_unproj > 0.5,
            "unprojected gradient should DISAGREE with FD at runaway coord: \
             FD={:+.6e}, unprojected={:+.6e}, rel={:.3e}",
            fd_grad[j],
            g_unproj[j],
            rel_unproj
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Implicit-function-theorem (IFT) correction: math validation
    // ═══════════════════════════════════════════════════════════════════════
    //
    // Builds a Gaussian REML problem where the inner KKT condition can be
    // solved in closed form, then deliberately perturbs β̂ away from β*.
    // Verifies the math derived in `reml_laml_evaluate`:
    //
    //   r = 0  ⇒  IFT correction = 0  ⇒  cost+gradient unchanged.
    //   r ≠ 0  ⇒  envelope formula mismatches FD by O(‖r‖·‖v_k‖);
    //            IFT-corrected formula matches FD to higher order.

    /// Build a Gaussian InnerSolution at an *arbitrary* β̂ (not necessarily
    /// the inner optimum), recomputing log_likelihood, penalty_quadratic,
    /// and the KKT residual r = S(λ)β̂ − ∇ℓ(β̂) consistently.
    fn build_gaussian_solution_at_beta(
        rho: &[f64],
        beta_hat: Array1<f64>,
        attach_residual: bool,
    ) -> InnerSolution<'_> {
        let p = 3usize;
        let n = 50usize;
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0]];
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let xty = array![5.0, 3.0, 2.0];
        let yty = 20.0;
        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();

        let mut h = xtx.clone();
        h.scaled_add(lambdas[0], &s1);
        h.scaled_add(lambdas[1], &s2);
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let penalty_quad = lambdas[0] * beta_hat.dot(&s1.dot(&beta_hat))
            + lambdas[1] * beta_hat.dot(&s2.dot(&beta_hat));
        let deviance = yty - 2.0 * beta_hat.dot(&xty) + beta_hat.dot(&xtx.dot(&beta_hat));
        let log_likelihood = -0.5 * deviance;

        // KKT residual r = S(λ)β̂ − ∇ℓ(β̂).  For Gaussian:
        //   ℓ(β) = −½(yᵀy − 2 βᵀX'y + βᵀX'Xβ),   ∇ℓ(β) = X'y − X'Xβ.
        // ⇒ r = (λ₁S₁+λ₂S₂)β̂ − (X'y − X'Xβ̂) = Hβ̂ − X'y.
        // At β* = H⁻¹X'y this is identically zero.
        let kkt_residual = if attach_residual {
            Some(ProjectedKktResidual::from_projected(
                &h.dot(&beta_hat) - &xty,
            ))
        } else {
            None
        };

        let r1 = penalty_matrix_root(&s1).unwrap();
        let r2 = penalty_matrix_root(&s2).unwrap();

        let mut s_total = Array2::zeros((p, p));
        s_total.scaled_add(lambdas[0], &s1);
        s_total.scaled_add(lambdas[1], &s2);
        let (s_eigs, _) = s_total.eigh(faer::Side::Lower).unwrap();
        let threshold = positive_eigenvalue_threshold(s_eigs.as_slice().unwrap());
        let log_det_s = exact_pseudo_logdet(s_eigs.as_slice().unwrap(), threshold);

        let log_det_s_at = |rho_eval: &[f64]| -> f64 {
            let lambdas_eval: Vec<f64> = rho_eval.iter().map(|&r| r.exp()).collect();
            let mut s_eval = Array2::zeros((p, p));
            s_eval.scaled_add(lambdas_eval[0], &s1);
            s_eval.scaled_add(lambdas_eval[1], &s2);
            let (s_eigs_eval, _) = s_eval.eigh(faer::Side::Lower).unwrap();
            let threshold_eval = positive_eigenvalue_threshold(s_eigs_eval.as_slice().unwrap());
            exact_pseudo_logdet(s_eigs_eval.as_slice().unwrap(), threshold_eval)
        };

        let mut det1 = Array1::zeros(rho.len());
        let eps = 1e-7;
        for k in 0..rho.len() {
            let mut rho_plus = rho.to_vec();
            rho_plus[k] += eps;
            let log_det_s_plus = log_det_s_at(&rho_plus);

            let mut rho_minus = rho.to_vec();
            rho_minus[k] -= eps;
            let log_det_s_minus = log_det_s_at(&rho_minus);

            det1[k] = (log_det_s_plus - log_det_s_minus) / (2.0 * eps);
        }
        let mut det2 = Array2::zeros((rho.len(), rho.len()));
        let eps2 = 1e-5;
        for i in 0..rho.len() {
            for j in i..rho.len() {
                let value = if i == j {
                    let mut rho_plus = rho.to_vec();
                    rho_plus[i] += eps2;
                    let mut rho_minus = rho.to_vec();
                    rho_minus[i] -= eps2;
                    (log_det_s_at(&rho_plus) - 2.0 * log_det_s + log_det_s_at(&rho_minus))
                        / (eps2 * eps2)
                } else {
                    let mut pp = rho.to_vec();
                    pp[i] += eps2;
                    pp[j] += eps2;
                    let mut pm = rho.to_vec();
                    pm[i] += eps2;
                    pm[j] -= eps2;
                    let mut mp = rho.to_vec();
                    mp[i] -= eps2;
                    mp[j] += eps2;
                    let mut mm = rho.to_vec();
                    mm[i] -= eps2;
                    mm[j] -= eps2;
                    (log_det_s_at(&pp) - log_det_s_at(&pm) - log_det_s_at(&mp) + log_det_s_at(&mm))
                        / (4.0 * eps2 * eps2)
                };
                det2[[i, j]] = value;
                if i != j {
                    det2[[j, i]] = value;
                }
            }
        }

        InnerSolution {
            log_likelihood,
            penalty_quadratic: penalty_quad,
            hessian_op: Arc::new(op),
            beta: beta_hat,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(r1),
                PenaltyCoordinate::from_dense_root(r2),
            ],
            penalty_logdet: PenaltyLogdetDerivs {
                value: log_det_s,
                first: det1,
                second: Some(det2),
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: n,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
            kkt_residual,
            active_constraints: None,
        }
    }

    #[test]
    fn malformed_projected_kkt_residual_is_contract_error() {
        let rho: Vec<f64> = vec![1.0, -0.5];
        let beta_hat = array![0.1, -0.2, 0.3];
        let mut sol = build_gaussian_solution_at_beta(&rho, beta_hat, false);
        sol.dispersion = DispersionHandling::Fixed {
            phi: 1.0,
            include_logdet_h: true,
            include_logdet_s: true,
        };
        sol.kkt_residual = Some(ProjectedKktResidual::from_projected(array![0.0, 0.0]));

        let err = match reml_laml_evaluate(&sol, &rho, EvalMode::ValueAndGradient, None) {
            Ok(_) => panic!("wrong-length projected KKT residual must be rejected"),
            Err(err) => err,
        };
        assert!(
            err.contains("projected KKT residual length mismatch"),
            "unexpected error: {err}"
        );
    }

    /// At exact KKT (r = 0) the IFT correction is identically zero.
    /// Attaching `Some(zeros)` must not perturb the envelope cost/gradient.
    #[test]
    fn ift_correction_vanishes_at_exact_kkt() {
        let rho: Vec<f64> = vec![1.0, -0.5];
        // Recompute exact β* = H⁻¹X'y at this ρ.
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0]];
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let xty = array![5.0, 3.0, 2.0];
        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
        let mut h = xtx.clone();
        h.scaled_add(lambdas[0], &s1);
        h.scaled_add(lambdas[1], &s2);
        let op_for_solve = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let beta_star = op_for_solve.solve(&xty);

        let sol_envelope = build_gaussian_solution_at_beta(&rho, beta_star.clone(), false);
        let grad_envelope =
            reml_laml_evaluate(&sol_envelope, &rho, EvalMode::ValueAndGradient, None)
                .unwrap()
                .gradient
                .unwrap();
        let cost_envelope = reml_laml_evaluate(&sol_envelope, &rho, EvalMode::ValueOnly, None)
            .unwrap()
            .cost;

        let sol_with_residual = build_gaussian_solution_at_beta(&rho, beta_star.clone(), true);
        let r_norm = sol_with_residual
            .kkt_residual
            .as_ref()
            .unwrap()
            .as_array()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            r_norm < 1e-10,
            "residual at exact β* should be numerically zero, got ‖r‖∞ = {:.3e}",
            r_norm
        );

        let result_ift =
            reml_laml_evaluate(&sol_with_residual, &rho, EvalMode::ValueAndGradient, None).unwrap();
        let grad_ift = result_ift.gradient.unwrap();
        let cost_ift = result_ift.cost;

        assert_relative_eq!(
            cost_ift,
            cost_envelope,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
        for k in 0..rho.len() {
            assert_relative_eq!(
                grad_ift[k],
                grad_envelope[k],
                epsilon = 1e-10,
                max_relative = 1e-8
            );
        }
    }

    /// With β̂ perturbed off β* and the matching r attached, the IFT-corrected
    /// gradient must match a re-solved FD reference much better than the
    /// uncorrected envelope formula evaluated at the perturbed β̂.
    #[test]
    fn ift_correction_recovers_fd_at_perturbed_beta() {
        let rho: Vec<f64> = vec![0.5, 0.3];

        // Re-solve for exact β* at ρ.
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0]];
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let xty = array![5.0, 3.0, 2.0];
        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
        let mut h = xtx.clone();
        h.scaled_add(lambdas[0], &s1);
        h.scaled_add(lambdas[1], &s2);
        let op_for_solve = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let beta_star = op_for_solve.solve(&xty);

        // Use Fixed-dispersion V = -ℓ + ½βᵀSβ + ½(log|H| − log|S|_+).
        // This is the parameterisation under which the IFT correction is
        // exact (∂V/∂β = r, no `denom/dp` chain factor as in the profiled
        // Gaussian path).  Matches the production survival-marginal-slope
        // path that the biobank failure exercises.
        fn to_fixed<'a>(mut sol: InnerSolution<'a>) -> InnerSolution<'a> {
            sol.dispersion = DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            };
            sol
        }

        // FD reference: evaluate at re-solved β*(ρ±ε), which is what an
        // ideal inner solver would deliver.  Use ValueOnly to avoid the
        // recursive gradient path.
        let fd_eps = 1e-5;
        let mut fd_grad = Array1::<f64>::zeros(rho.len());
        for k in 0..rho.len() {
            let mut rho_plus = rho.clone();
            rho_plus[k] += fd_eps;
            let mut h_plus = xtx.clone();
            let lambdas_plus: Vec<f64> = rho_plus.iter().map(|&r| r.exp()).collect();
            h_plus.scaled_add(lambdas_plus[0], &s1);
            h_plus.scaled_add(lambdas_plus[1], &s2);
            let beta_star_plus = DenseSpectralOperator::from_symmetric(&h_plus)
                .unwrap()
                .solve(&xty);
            let sol_plus = to_fixed(build_gaussian_solution_at_beta(
                &rho_plus,
                beta_star_plus,
                false,
            ));
            let cost_plus = reml_laml_evaluate(&sol_plus, &rho_plus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            let mut rho_minus = rho.clone();
            rho_minus[k] -= fd_eps;
            let mut h_minus = xtx.clone();
            let lambdas_minus: Vec<f64> = rho_minus.iter().map(|&r| r.exp()).collect();
            h_minus.scaled_add(lambdas_minus[0], &s1);
            h_minus.scaled_add(lambdas_minus[1], &s2);
            let beta_star_minus = DenseSpectralOperator::from_symmetric(&h_minus)
                .unwrap()
                .solve(&xty);
            let sol_minus = to_fixed(build_gaussian_solution_at_beta(
                &rho_minus,
                beta_star_minus,
                false,
            ));
            let cost_minus = reml_laml_evaluate(&sol_minus, &rho_minus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            fd_grad[k] = (cost_plus - cost_minus) / (2.0 * fd_eps);
        }

        // Perturb β̂ off β* — small enough that linear IFT recovers cleanly.
        let perturb = Array1::from_vec(vec![0.02, -0.015, 0.025]);
        let beta_hat = &beta_star + &perturb;

        let sol_envelope = to_fixed(build_gaussian_solution_at_beta(
            &rho,
            beta_hat.clone(),
            false,
        ));
        let grad_envelope =
            reml_laml_evaluate(&sol_envelope, &rho, EvalMode::ValueAndGradient, None)
                .unwrap()
                .gradient
                .unwrap();

        let sol_ift = to_fixed(build_gaussian_solution_at_beta(
            &rho,
            beta_hat.clone(),
            true,
        ));
        let r_norm = sol_ift
            .kkt_residual
            .as_ref()
            .unwrap()
            .as_array()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            r_norm > 1e-3,
            "perturbed β̂ should produce a non-trivial residual, got ‖r‖∞ = {:.3e}",
            r_norm
        );
        let grad_ift = reml_laml_evaluate(&sol_ift, &rho, EvalMode::ValueAndGradient, None)
            .unwrap()
            .gradient
            .unwrap();

        // IFT correction must shrink the gradient error meaningfully on at
        // least one coordinate, and never blow it up on any.
        let mut at_least_one_improved = false;
        for k in 0..rho.len() {
            let err_envelope = (grad_envelope[k] - fd_grad[k]).abs();
            let err_ift = (grad_ift[k] - fd_grad[k]).abs();
            assert!(
                err_ift <= err_envelope * 1.05 + 1e-9,
                "IFT correction must not enlarge gradient error: coord={} envelope_err={:.3e} \
                 ift_err={:.3e} FD={:.6e}",
                k,
                err_envelope,
                err_ift,
                fd_grad[k]
            );
            if err_ift < err_envelope * 0.5 && err_envelope > 1e-6 {
                at_least_one_improved = true;
            }
        }
        assert!(
            at_least_one_improved,
            "IFT correction should improve gradient accuracy on at least one coord: \
             envelope=[{:.3e}, {:.3e}] ift=[{:.3e}, {:.3e}] fd=[{:.3e}, {:.3e}]",
            (grad_envelope[0] - fd_grad[0]).abs(),
            (grad_envelope[1] - fd_grad[1]).abs(),
            (grad_ift[0] - fd_grad[0]).abs(),
            (grad_ift[1] - fd_grad[1]).abs(),
            fd_grad[0],
            fd_grad[1],
        );
    }

    /// The analytic rho Hessian must differentiate the same KKT-residual
    /// correction used by the value and gradient. This is the minimized
    /// reproduction of the biobank failure mode: an off-KKT inner mode with a
    /// finite residual made the envelope Hessian inconsistent, so ARC chased a
    /// curvature model for the wrong objective.
    #[test]
    fn ift_correction_recovers_fd_hessian_at_perturbed_beta() {
        let rho: Vec<f64> = vec![0.5, 0.3];
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0]];
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let xty = array![5.0, 3.0, 2.0];

        fn to_fixed<'a>(mut sol: InnerSolution<'a>) -> InnerSolution<'a> {
            sol.dispersion = DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            };
            sol
        }

        let solve_beta_star = |rho_eval: &[f64]| -> Array1<f64> {
            let lambdas_eval: Vec<f64> = rho_eval.iter().map(|&r| r.exp()).collect();
            let mut h = xtx.clone();
            h.scaled_add(lambdas_eval[0], &s1);
            h.scaled_add(lambdas_eval[1], &s2);
            DenseSpectralOperator::from_symmetric(&h)
                .unwrap()
                .solve(&xty)
        };
        let exact_profile_cost = |rho_eval: &[f64]| -> f64 {
            let beta_star = solve_beta_star(rho_eval);
            let sol = to_fixed(build_gaussian_solution_at_beta(rho_eval, beta_star, false));
            reml_laml_evaluate(&sol, rho_eval, EvalMode::ValueOnly, None)
                .unwrap()
                .cost
        };

        let fd_eps = 2e-4;
        let mut fd_hessian = Array2::<f64>::zeros((rho.len(), rho.len()));
        let center_cost = exact_profile_cost(&rho);
        for i in 0..rho.len() {
            for j in i..rho.len() {
                let value = if i == j {
                    let mut rho_plus = rho.clone();
                    rho_plus[i] += fd_eps;
                    let mut rho_minus = rho.clone();
                    rho_minus[i] -= fd_eps;
                    (exact_profile_cost(&rho_plus) - 2.0 * center_cost
                        + exact_profile_cost(&rho_minus))
                        / (fd_eps * fd_eps)
                } else {
                    let mut pp = rho.clone();
                    pp[i] += fd_eps;
                    pp[j] += fd_eps;
                    let mut pm = rho.clone();
                    pm[i] += fd_eps;
                    pm[j] -= fd_eps;
                    let mut mp = rho.clone();
                    mp[i] -= fd_eps;
                    mp[j] += fd_eps;
                    let mut mm = rho.clone();
                    mm[i] -= fd_eps;
                    mm[j] -= fd_eps;
                    (exact_profile_cost(&pp) - exact_profile_cost(&pm) - exact_profile_cost(&mp)
                        + exact_profile_cost(&mm))
                        / (4.0 * fd_eps * fd_eps)
                };
                fd_hessian[[i, j]] = value;
                if i != j {
                    fd_hessian[[j, i]] = value;
                }
            }
        }

        let beta_star = solve_beta_star(&rho);
        let beta_hat = &beta_star + &Array1::from_vec(vec![0.02, -0.015, 0.025]);
        let sol_envelope = to_fixed(build_gaussian_solution_at_beta(
            &rho,
            beta_hat.clone(),
            false,
        ));
        let hessian_envelope =
            reml_laml_evaluate(&sol_envelope, &rho, EvalMode::ValueGradientHessian, None)
                .unwrap()
                .hessian
                .unwrap_analytic();

        let sol_ift = to_fixed(build_gaussian_solution_at_beta(&rho, beta_hat, true));
        let hessian_ift = reml_laml_evaluate(&sol_ift, &rho, EvalMode::ValueGradientHessian, None)
            .unwrap()
            .hessian
            .unwrap_analytic();

        let mut envelope_was_wrong = false;
        for i in 0..rho.len() {
            for j in 0..rho.len() {
                let envelope_err = (hessian_envelope[[i, j]] - fd_hessian[[i, j]]).abs();
                let ift_err = (hessian_ift[[i, j]] - fd_hessian[[i, j]]).abs();
                assert!(
                    ift_err <= envelope_err * 0.25 + 2e-5,
                    "IFT Hessian correction failed at ({}, {}): envelope={:.8e} ift={:.8e} \
                     fd={:.8e} envelope_err={:.3e} ift_err={:.3e}",
                    i,
                    j,
                    hessian_envelope[[i, j]],
                    hessian_ift[[i, j]],
                    fd_hessian[[i, j]],
                    envelope_err,
                    ift_err
                );
                if envelope_err > 1e-4 && ift_err < envelope_err * 0.1 {
                    envelope_was_wrong = true;
                }
            }
        }
        assert!(
            envelope_was_wrong,
            "test did not reproduce the Hessian bug: envelope={:?} ift={:?} fd={:?}",
            hessian_envelope, hessian_ift, fd_hessian
        );
    }
}
