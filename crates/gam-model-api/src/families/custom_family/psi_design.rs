//! The ψ (hyperparameter) design-derivative operators: the
//! `CustomFamilyPsiDerivativeOperator` trait and every concrete operator
//! (implicit / embedded-implicit / zero / embedded-dense / rowwise-Kronecker),
//! the ψ design/second-design actions and linear-map refs, the joint ψ operator,
//! and the exact-Newton joint-ψ term carriers + workspace traits.

use crate::families::custom_family::family_trait::ExactNewtonJointGradientEvaluation;
use gam_problem::{CustomFamilyError, DenseMatrixHyperOperator, EvalMode, HyperOperator};
use ndarray::{Array1, Array2};
use std::sync::Arc;

// The neutral ψ-derivative carriers and operator traits live in
// `gam-problem`. Only the trait that couples to the `CustomFamily` evaluation
// carrier (`ExactNewtonJointHessianWorkspace`) stays local below.
pub use gam_problem::{
    CustomFamilyBlockPsiDerivative, CustomFamilyPsiDerivativeOperator,
    JointHessianSourcePreference, MaterializablePsiDerivativeOperator, MaterializationIntent,
    SharedDerivativeBlocks,
};
use gam_problem::{
    ExactNewtonJointPsiSecondOrderContracted, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
};

pub trait ExactNewtonJointHessianWorkspace: Send + Sync {
    /// Pre-build any per-row jet caches the workspace will hand to the
    /// outer-eval directional-derivative path. Called once when the
    /// `compute_dh` / `compute_d2h` closures are wired up at top-level
    /// rayon, *before* the outer ext-coordinate `par_iter` enters. The
    /// alternative — letting the cache materialise lazily on first call
    /// from inside the outer `par_iter` — collapses the build's own
    /// `par_iter` to a single worker (the seven other workers are parked
    /// on the cache's `OnceLock`). Default impl is a no-op for workspaces
    /// with no per-row jet cache.
    ///
    /// Deliberately not called from PIRLS-side workspaces (which never
    /// invoke `directional_derivative_operator` and would pay the prime
    /// cost without ever consuming the cache).
    ///
    /// `eval_mode` bounds *which* caches are worth priming for the eval that
    /// is about to run (gam#979 — large-scale margslope line-search /
    /// seed-screen cost):
    ///   * [`EvalMode::ValueOnly`] consumes **no** directional cache — the
    ///     objective is read straight off the converged inner mode — so a
    ///     value-only probe (line search, seed screen, continuation pre-warm)
    ///     should prime nothing.
    ///   * [`EvalMode::ValueAndGradient`] consumes only the **first** (third-
    ///     derivative) directional cache, which feeds the REML/LAML gradient's
    ///     `coord_corrections` IFT-drift trace.
    ///   * [`EvalMode::ValueGradientHessian`] additionally consumes the
    ///     **second** (fourth-derivative) directional cache used by the outer
    ///     Hessian's second-directional pass.
    /// Priming a cache the mode never reads is pure wasted O(n) work, so under-
    /// priming is always safe: every cache is a lazy `get_or_compute`, so a
    /// later consumer (if any) still builds it on demand — just without the
    /// top-level-rayon fan-out this hook would have given it.
    fn warm_up_outer_caches_for_mode(&self, eval_mode: EvalMode) -> Result<(), String> {
        // Legacy default: prime everything (matches the historic
        // `warm_up_outer_caches` contract) for any workspace that has not
        // opted into mode-aware priming. Every mode falls through to the same
        // mode-blind prime; mode-aware workspaces override this method to skip
        // caches the requested mode never reads.
        match eval_mode {
            EvalMode::ValueOnly | EvalMode::ValueAndGradient | EvalMode::ValueGradientHessian => {
                self.warm_up_outer_caches()
            }
        }
    }

    /// Mode-blind warm-up retained for back-compat: primes every directional
    /// cache the workspace keeps. Production call sites should prefer
    /// [`Self::warm_up_outer_caches_for_mode`] so value-only probes skip the
    /// prime entirely. Default impl is a no-op for workspaces with no per-row
    /// jet cache.
    fn warm_up_outer_caches(&self) -> Result<(), String> {
        Ok(())
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Preferred representation for callers that can consume either the dense
    /// coefficient Hessian or the matrix-free HVP source.
    fn hessian_source_preference(&self) -> JointHessianSourcePreference {
        JointHessianSourcePreference::Dense
    }

    /// Intent-aware representation choice (#738). Given what the consumer is
    /// about to do with the Hessian ([`MaterializationIntent`]), return the
    /// representation the workspace prefers to hand back. The default keeps the
    /// legacy intent-blind behaviour by delegating to
    /// [`Self::hessian_source_preference`], so existing workspaces are
    /// unchanged. Workspaces with a structural direct-dense build that also
    /// expose a matrix-free HVP override this to answer `Operator` for
    /// [`MaterializationIntent::InnerSolve`] (stream the HVP) and `Dense` for
    /// [`MaterializationIntent::LogdetFactorization`] (the consumer factorizes,
    /// so building the operator wrapper only to re-densify it is pure waste).
    fn hessian_source_preference_for_intent(
        &self,
        intent: MaterializationIntent,
    ) -> JointHessianSourcePreference {
        // Intent-agnostic default: every intent maps to the single legacy
        // preference. Implementors that benefit from per-intent representation
        // (e.g. CTN: dense for logdet, operator for inner solve) override this.
        match intent {
            MaterializationIntent::InnerSolve
            | MaterializationIntent::LogdetFactorization
            | MaterializationIntent::OuterEvaluation
            | MaterializationIntent::OuterGradient => self.hessian_source_preference(),
        }
    }

    /// Forced dense materialization that bypasses any amortization gate the
    /// workspace applies to `hessian_dense`. Callers that genuinely need a
    /// dense matrix (logdet, factorize-based QP solves) use this so they pay
    /// the workspace's structural direct-dense build cost rather than the
    /// caller-side column-basis HVP fallback. Returning `None` means the
    /// workspace has no preferred direct-dense path and the caller should
    /// fall back to column-basis HVP via `hessian_matvec` / `apply`.
    fn hessian_dense_forced(&self) -> Result<Option<Array2<f64>>, String> {
        self.hessian_dense()
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        Ok(None)
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(None)
    }

    /// Whether `hessian_matvec` / `hessian_matvec_into` will return `Some`.
    /// A cheap synchronisation-free flag consulted by
    /// `exact_newton_joint_hessian_source_from_workspace` to decide whether
    /// to construct a matrix-free `JointHessianSource::Operator` variant.
    /// Returning `false` is equivalent to returning `Ok(None)` from
    /// `hessian_matvec` but avoids allocating and running a full HVP sweep
    /// against a zero vector just to discover unavailability.
    /// Default is `false` matching the base-trait `hessian_matvec` returning
    /// `Ok(None)`. Concrete impls that override `hessian_matvec` must also
    /// override this to return `true`.
    fn hessian_matvec_available(&self) -> bool {
        false
    }

    fn hessian_matvec(&self, arr: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Write-into variant of `hessian_matvec`. The default implementation
    /// delegates to the legacy owned-return form and copies the result into
    /// `out`, providing back-compat without per-impl work. Concrete impls in
    /// the inner-Newton large-scale hot path (Bernoulli marginal-slope and
    /// survival marginal-slope) override this to write directly into the
    /// caller-owned buffer, eliminating per-PCG-iter `Array1` allocations.
    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        match self.hessian_matvec(v)? {
            Some(result) => {
                if result.len() != out.len() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "hessian_matvec_into: result length {} != out length {}",
                            result.len(),
                            out.len()
                        ),
                    }
                    .into());
                }
                out.assign(&result);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Batched multi-RHS Hessian apply: writes `H · V` into `out`, where `V`
    /// and `out` are `(total, n_rhs)` with each column an independent
    /// direction. Returns `Ok(true)` when the apply was performed and
    /// `Ok(false)` when the workspace exposes no matrix-free apply (mirroring
    /// `hessian_matvec_into`).
    ///
    /// The default implementation applies `hessian_matvec_into` column by
    /// column, so every existing workspace gets a correct batched apply for
    /// free and the batched result is, column for column, **numerically
    /// identical** to looping the single-vector HVP. Workspaces whose Hessian
    /// is `Σ_i Jᵢᵀ Hᵢ Jᵢ` over a streamed/tiled per-row primary Hessian `Hᵢ`
    /// (Bernoulli marginal-slope) override this to sweep each row tile **once**
    /// and apply its `Hᵢ` to all `n_rhs` columns in that single pass — the
    /// per-tile `Hᵢ` read and the design-row projection are then amortised
    /// across every RHS instead of paid once per column. This is the
    /// representation that makes dense reconstruction of a matrix-free operator
    /// (`H = H · [e_0 | … | e_{p-1}]`) one tile sweep wide instead of `p`.
    fn hessian_apply_mat(
        &self,
        v_cols: &Array2<f64>,
        out: &mut Array2<f64>,
    ) -> Result<bool, String> {
        if v_cols.nrows() != out.nrows() || v_cols.ncols() != out.ncols() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "hessian_apply_mat: v_cols {}x{} != out {}x{}",
                    v_cols.nrows(),
                    v_cols.ncols(),
                    out.nrows(),
                    out.ncols()
                ),
            }
            .into());
        }
        let total = v_cols.nrows();
        let mut col_in = Array1::<f64>::zeros(total);
        let mut col_out = Array1::<f64>::zeros(total);
        for col in 0..v_cols.ncols() {
            col_in.assign(&v_cols.column(col));
            if !self.hessian_matvec_into(&col_in, &mut col_out)? {
                return Ok(false);
            }
            out.column_mut(col).assign(&col_out);
        }
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    /// Exact row-local contractions for
    /// `trace(F^T · D_beta H[d_j] · F)` over many coefficient directions.
    ///
    /// Workspaces that own the current row cache can implement this to avoid
    /// rebuilding row contexts or materializing each `D_beta H[d_j]` as a
    /// coefficient-space operator when the caller only needs its projected
    /// trace against the fixed logdet factor `F`.
    fn projected_directional_derivative_traces(
        &self,
        factor: &Array2<f64>,
        directions: &Array2<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        assert_eq!(
            factor.nrows(),
            directions.nrows(),
            "projected directional derivative traces require shared coefficient dimension"
        );
        Ok(None)
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String>;

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        Ok(self
            .directional_derivative(d_beta_flat)?
            .map(|matrix| Arc::new(DenseMatrixHyperOperator { matrix }) as Arc<dyn HyperOperator>))
    }

    fn directional_derivative_operators(
        &self,
        d_beta_flats: &[Array1<f64>],
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        d_beta_flats
            .iter()
            .map(|d_beta_flat| self.directional_derivative_operator(d_beta_flat))
            .collect()
    }

    fn second_directional_derivative(
        &self,
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        Ok(self
            .second_directional_derivative(d_beta_u, d_beta_v)?
            .map(|matrix| Arc::new(DenseMatrixHyperOperator { matrix }) as Arc<dyn HyperOperator>))
    }

    fn second_directional_derivative_operators(
        &self,
        d_beta_pairs: &[(Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        d_beta_pairs
            .iter()
            .map(|(u, v)| self.second_directional_derivative_operator(u, v))
            .collect()
    }
}
