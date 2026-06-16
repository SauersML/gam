//! # Outer-objective contract (lower shared layer)
//!
//! The interface types that the `families` layer must *name, implement, and
//! return* to participate in outer smoothing-parameter optimization, hosted
//! below both `families` and `solver` so families stop importing *up* into
//! `crate::solver::rho_optimizer` (#1135).
//!
//! What lives here is exactly the **family ↔ solver contract**: the matrix-free
//! [`OuterHessianOperator`] trait that families implement, the [`OuterEval`] /
//! [`HessianResult`] result types they return, the [`EfsEval`] step bundle, and
//! the capability enums ([`Derivative`], [`DeclaredHessianForm`],
//! [`OuterHessianMaterialization`]) plus the operator-shape error
//! ([`OuterStrategyError`]).
//!
//! What does *not* live here is the solver's *use* of the contract — the outer
//! runner, ARC/trust-region planning, seeding, caching, barrier configuration,
//! and `OuterProblem` — all of which stay in `crate::solver::rho_optimizer` and
//! depend downward on this module. `crate::solver::rho_optimizer` re-exports
//! these names so existing `crate::solver::rho_optimizer::*` paths keep working.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};

/// Exact dense-materialization route exposed by an outer Hessian operator.
///
/// The optimizer uses this as a work-model contract before turning a
/// matrix-free analytic Hessian into a dense ARC model. `Unavailable` means
/// callers must stay matrix-free; the remaining variants are all analytic
/// but differ in how much per-column HVP overhead they imply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterHessianMaterialization {
    /// Dense materialization is not part of this operator's contract.
    Unavailable,
    /// Materialization is exact but implemented by cheap repeated HVP probes.
    RepeatedHvp,
    /// Materialization is exact and can apply many HVP directions together.
    BatchedHvp,
    /// Materialization is exact and can be assembled without basis probing.
    Explicit,
}

impl OuterHessianMaterialization {
    pub(crate) fn is_available(self) -> bool {
        !matches!(self, Self::Unavailable)
    }
}

/// Typed error for the outer-strategy Hessian-operator surface.
///
/// All construction sites inside `outer_strategy` build one of these variants
/// instead of an ad-hoc `String`; the historical `Result<_, String>` boundary
/// on the [`OuterHessianOperator`] trait (which has out-of-crate implementors
/// in `families/*` and `solver/reml/*`) is preserved by an explicit
/// `OuterStrategyError -> String` conversion at the leaf return points.
#[derive(Debug, Clone)]
pub enum OuterStrategyError {
    /// Length / shape mismatch raised by an outer Hessian operator
    /// (matvec input/output length, `mul_mat` factor row count,
    /// `materialize_dense` output dimensions, etc.).
    OperatorShape { reason: String },
    /// Dense materialization produced non-finite entries.
    NonFiniteHessian { reason: String },
    /// Shape / dimension violation of a rho-block additive Hessian update.
    RhoBlockShape { reason: String },
}

crate::impl_reason_error_boilerplate! {
    OuterStrategyError {
        OperatorShape,
        NonFiniteHessian,
        RhoBlockShape,
    }
}

/// Matrix-free outer Hessian operator.
///
/// This is the exact outer Hessian action `H_outer * v` evaluated at the
/// current outer point, without requiring dense materialization.
///
/// The trait provides four increasingly materialized primitives:
///
/// - [`matvec`](Self::matvec) — single column, the only one implementors must
///   provide.
/// - [`mul_mat`](Self::mul_mat) — multi-column; the default falls back to
///   column-by-column `matvec`. Implementors override this when they can
///   amortize per-Hv-apply overhead (cached factorizations, parallel matvecs)
///   across many right-hand-sides.
/// - [`materialization_capability`](Self::materialization_capability) — an
///   explicit work-model contract that tells ARC whether dense exact
///   materialization is unavailable, cheap repeated-HVP, batched-HVP, or
///   explicit.
/// - [`materialize_dense`](Self::materialize_dense) — the special case
///   `mul_mat(I_dim)` followed by a symmetric average of the off-diagonals to
///   absorb round-off asymmetry. ARC callers only use this when
///   [`materialization_capability`](Self::materialization_capability) advertises
///   an exact dense route, preserving the no-numerical-Hessian policy.
pub trait OuterHessianOperator: Send + Sync {
    fn dim(&self) -> usize;
    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String>;

    /// Write `out <- H * v` into a caller-supplied buffer. Default
    /// impl wraps `matvec` and copies; backends override for a true
    /// zero-alloc inner-CG path. The matrix-free trust-region adapter
    /// (`OuterToOptHessianOperator`) calls this on every CG step
    /// inside `opt::MatrixFreeTrustRegion`, so an override compounds:
    /// over a 50-outer-iter × 30-CG-iter solve at n=200 the default
    /// path allocates 1500 transient `Array1<f64>` of size 200 that
    /// the override eliminates.
    fn apply_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), String> {
        let result = self.matvec(v)?;
        if result.len() != out.len() {
            return Err(format!(
                "outer Hessian operator matvec produced length {} but expected {}",
                result.len(),
                out.len()
            ));
        }
        out.assign(&result);
        Ok(())
    }

    /// Whether probing all basis columns is cheap enough for dense ARC.
    ///
    /// The default is deliberately conservative. For operator-backed Duchon,
    /// CTN, survival, or other row-streaming kernels, `dim <= 64` does not
    /// imply cheap materialization: each column may trigger a full data pass.
    ///
    /// New implementations should prefer overriding
    /// [`materialization_capability`](Self::materialization_capability) so the
    /// caller can distinguish cheap repeated probes from true batched/explicit
    /// Hessian materialization.
    fn is_cheap_to_materialize(&self) -> bool {
        false
    }

    /// Exact dense-materialization capability for this operator.
    ///
    /// The default preserves the historical work-model hook: operators that
    /// already opted into cheap probing via
    /// [`is_cheap_to_materialize`](Self::is_cheap_to_materialize) are treated
    /// as exact repeated-HVP materializers. Backends that can amortize or avoid
    /// basis probes should override this to return
    /// [`OuterHessianMaterialization::BatchedHvp`] or
    /// [`OuterHessianMaterialization::Explicit`].
    fn materialization_capability(&self) -> OuterHessianMaterialization {
        if self.is_cheap_to_materialize() {
            OuterHessianMaterialization::RepeatedHvp
        } else {
            OuterHessianMaterialization::Unavailable
        }
    }

    /// Apply the operator to all `m` columns of `factor`, returning a
    /// `dim × m` matrix whose `j`th column is `H · factor[:, j]`.
    ///
    /// The default implementation runs the per-column matvecs in parallel
    /// over rayon — each matvec is independent and the K×K basis-probe used
    /// by [`materialize_dense`](Self::materialize_dense) issues exactly `dim`
    /// such calls.  Implementors override when batching is cheaper (cached
    /// factorizations, BLAS-3 kernels). All
    /// [`materialize_dense`](Self::materialize_dense) callers route through
    /// this method, so an override automatically accelerates any
    /// work-model-approved materialization path used by the planner.
    fn mul_mat(&self, factor: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let dim = self.dim();
        if factor.nrows() != dim {
            return Err(OuterStrategyError::OperatorShape {
                reason: format!(
                    "outer Hessian operator factor row count mismatch: got {}, expected {}",
                    factor.nrows(),
                    dim
                ),
            }
            .into());
        }
        let m = factor.ncols();
        let cols: Result<Vec<Array1<f64>>, String> = (0..m)
            .into_par_iter()
            .map(|j| {
                let col = factor.column(j).to_owned();
                let hv = self.matvec(&col)?;
                if hv.len() != dim {
                    return Err(OuterStrategyError::OperatorShape {
                        reason: format!(
                            "outer Hessian operator matvec length mismatch: got {}, expected {}",
                            hv.len(),
                            dim
                        ),
                    }
                    .into());
                }
                Ok(hv)
            })
            .collect();
        let cols = cols?;
        let mut out = Array2::<f64>::zeros((dim, m));
        for (j, hv) in cols.into_iter().enumerate() {
            out.column_mut(j).assign(&hv);
        }
        Ok(out)
    }

    /// Materialize the outer Hessian into a dense `dim × dim` matrix by
    /// applying the operator to the identity in a single
    /// [`mul_mat`](Self::mul_mat) call, then averaging the off-diagonals to
    /// stabilize against round-off asymmetry.
    fn materialize_dense(&self) -> Result<Array2<f64>, String> {
        let dim = self.dim();
        let identity = Array2::<f64>::eye(dim);
        let mut dense = self.mul_mat(identity.view())?;
        if dense.nrows() != dim || dense.ncols() != dim {
            return Err(OuterStrategyError::OperatorShape {
                reason: format!(
                    "outer Hessian operator mul_mat returned {}x{}, expected {}x{}",
                    dense.nrows(),
                    dense.ncols(),
                    dim,
                    dim
                ),
            }
            .into());
        }
        for row in 0..dim {
            for col in (row + 1)..dim {
                let sym = 0.5 * (dense[[row, col]] + dense[[col, row]]);
                dense[[row, col]] = sym;
                dense[[col, row]] = sym;
            }
        }
        if !dense.iter().all(|value| value.is_finite()) {
            return Err(OuterStrategyError::NonFiniteHessian {
                reason: "outer Hessian dense materialization produced non-finite entries"
                    .to_string(),
            }
            .into());
        }
        Ok(dense)
    }
}

/// Whether an analytic derivative is available for a given order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivative {
    /// Exact analytic derivative implemented and available.
    Analytic,
    /// No analytic derivative; must be approximated or skipped.
    Unavailable,
}

/// Capability-time declaration of what shape the outer Hessian takes.
/// Replaces the binary `Derivative` for the Hessian field on
/// `OuterCapability`: callers that know the shape upfront declare
/// it here, and the planner routes between dense ARC and matrix-free
/// trust-region *before* seed evaluation rather than dynamically
/// branching on `seed_eval.hessian` at runtime.
///
/// Variants:
/// - `Dense`: the family always returns `HessianResult::Analytic(_)`.
///   The planner picks dense ARC; matrix-free TR is never engaged.
/// - `Operator { materialization, estimated_materialization_cost }`:
///   the family always returns `HessianResult::Operator(_)`. The
///   planner picks matrix-free TR unless `materialization` advertises
///   `Explicit`/`BatchedHvp` cheaply enough that materializing once
///   per outer iter (opt 0.4.2 `with_materialize_when_cheap`) wins.
///   `estimated_materialization_cost` is reserved for a future cost
///   model; today it is purely informational.
/// - `Either`: the family may return either shape; the runner inspects
///   the seed eval and locks the route then. This is the historical
///   default for code paths where `Derivative::Analytic` made the
///   declaration and the seed loop branched on `seed_eval.hessian`.
/// - `Unavailable`: no analytic Hessian. The planner picks BFGS / EFS
///   per the gradient declaration and the rest of the capability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeclaredHessianForm {
    Dense,
    Operator {
        materialization: OuterHessianMaterialization,
        estimated_materialization_cost: Option<f64>,
    },
    Either,
    Unavailable,
}

impl DeclaredHessianForm {
    /// Coarse "is an analytic Hessian declared?" projection. `true`
    /// for `Dense` / `Operator` / `Either`; `false` for `Unavailable`.
    /// Used by `plan` to keep the existing `Derivative`-based match
    /// arms while richer routing decisions consult the form directly.
    pub const fn is_analytic(self) -> bool {
        !matches!(self, DeclaredHessianForm::Unavailable)
    }

    /// True when the declaration commits to a matrix-free path.
    pub const fn is_operator_only(self) -> bool {
        matches!(self, DeclaredHessianForm::Operator { .. })
    }

    /// True when the declaration commits to a dense path.
    pub const fn is_dense_only(self) -> bool {
        matches!(self, DeclaredHessianForm::Dense)
    }
}

/// Shared outer-objective result used by optimizer-facing objective
/// implementations.
pub struct OuterEval {
    pub cost: f64,
    pub gradient: Array1<f64>,
    pub hessian: HessianResult,
    /// Optional inner-solver iterate at this rho. Families whose inner solve
    /// produces a PIRLS beta populate this so the persistent-cache layer can
    /// store `(rho, beta)` together.
    pub inner_beta_hint: Option<Array1<f64>>,
}

impl OuterEval {
    /// Conventional representation of an infeasible trial point.
    pub fn infeasible(n_params: usize) -> Self {
        Self {
            cost: f64::INFINITY,
            gradient: Array1::zeros(n_params),
            hessian: HessianResult::Unavailable,
            inner_beta_hint: None,
        }
    }

    pub(crate) fn value_only(
        cost: f64,
        n_params: usize,
        inner_beta_hint: Option<Array1<f64>>,
    ) -> Self {
        Self {
            cost,
            gradient: Array1::zeros(n_params),
            hessian: HessianResult::Unavailable,
            inner_beta_hint,
        }
    }
}

/// Explicit Hessian result replacing `Option<Array2<f64>>`.
pub enum HessianResult {
    /// Analytic Hessian was computed and returned.
    Analytic(Array2<f64>),
    /// Analytic Hessian is available as an exact Hessian-vector product.
    Operator(Arc<dyn OuterHessianOperator>),
    /// No analytic Hessian available for this model path.
    Unavailable,
}

impl Clone for OuterEval {
    fn clone(&self) -> Self {
        Self {
            cost: self.cost,
            gradient: self.gradient.clone(),
            hessian: self.hessian.clone(),
            inner_beta_hint: self.inner_beta_hint.clone(),
        }
    }
}

impl std::fmt::Debug for OuterEval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OuterEval")
            .field("cost", &self.cost)
            .field("gradient", &self.gradient)
            .field("hessian", &self.hessian)
            .finish()
    }
}

impl Clone for HessianResult {
    fn clone(&self) -> Self {
        match self {
            Self::Analytic(h) => Self::Analytic(h.clone()),
            Self::Operator(op) => Self::Operator(Arc::clone(op)),
            Self::Unavailable => Self::Unavailable,
        }
    }
}

impl std::fmt::Debug for HessianResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Analytic(h) => f
                .debug_tuple("Analytic")
                .field(&format!("{}x{}", h.nrows(), h.ncols()))
                .finish(),
            Self::Operator(op) => f
                .debug_tuple("Operator")
                .field(&format!("dim={}", op.dim()))
                .finish(),
            Self::Unavailable => f.write_str("Unavailable"),
        }
    }
}

impl HessianResult {
    /// Returns `true` if an analytic Hessian is present in any exact form.
    pub fn is_analytic(&self) -> bool {
        matches!(
            self,
            HessianResult::Analytic(_) | HessianResult::Operator(_)
        )
    }

    pub fn dim(&self) -> Option<usize> {
        match self {
            HessianResult::Analytic(h) => Some(h.nrows()),
            HessianResult::Operator(op) => Some(op.dim()),
            HessianResult::Unavailable => None,
        }
    }

    pub fn materialize_dense(&self) -> Result<Option<Array2<f64>>, String> {
        match self {
            HessianResult::Analytic(h) => Ok(Some(h.clone())),
            HessianResult::Operator(op) => op.materialize_dense().map(Some),
            HessianResult::Unavailable => Ok(None),
        }
    }
}

/// Result bundle returned by the EFS (extended Fellner–Schall) evaluation
/// path. Pure data: families compute the additive step and the optional
/// curvature/gradient diagnostics; the solver consumes them.
#[derive(Clone, Debug)]
pub struct EfsEval {
    /// REML/LAML cost at the current rho (for convergence monitoring and
    /// comparing candidates).
    pub cost: f64,
    /// Additive steps. Length = n_rho + n_ext_coords.
    ///
    /// For pure EFS: steps for non-penalty-like coordinates are 0.0.
    /// For hybrid EFS: ρ-coords get standard EFS multiplicative steps,
    /// ψ-coords get preconditioned gradient steps `Δψ = -α G⁺ g_ψ`.
    pub steps: Vec<f64>,
    /// Current coefficient vector β̂ from the inner P-IRLS solve.
    /// Used by the EFS loop for the runtime barrier-curvature significance
    /// check when monotonicity constraints are present.
    pub beta: Option<Array1<f64>>,
    /// Raw REML/LAML gradient restricted to the ψ block (design-moving coords).
    ///
    /// Present only when the hybrid EFS strategy is active. Used by the
    /// outer iteration for backtracking on the ψ step: if the combined
    /// (ρ-EFS, ψ-gradient) step does not decrease V(θ), the ψ step size
    /// α is halved while keeping the ρ-EFS step fixed.
    ///
    /// This avoids re-evaluating the gradient during backtracking since
    /// the gradient was already computed as part of the hybrid EFS eval.
    pub psi_gradient: Option<Array1<f64>>,
    /// Indices into the full θ vector that correspond to ψ (design-moving)
    /// coordinates. Used by the backtracking logic to selectively scale
    /// only the ψ portion of the step.
    pub psi_indices: Option<Vec<usize>>,
    /// Inner-Hessian curvature scale captured during the EFS eval, used to
    /// condition the ψ preconditioner across outer iterations.
    pub inner_hessian_scale: Option<f64>,
    /// Logdet enclosure gap diagnostic (lower/upper bound spread) captured at
    /// this EFS evaluation when the bounded-logdet path is active.
    pub logdet_enclosure_gap: Option<f64>,
}

impl EfsEval {
    pub fn with_logdet_enclosure_gap(mut self, gap: Option<f64>) -> Self {
        self.logdet_enclosure_gap = gap;
        self
    }
}
