use super::*;

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

impl std::fmt::Display for OuterStrategyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OuterStrategyError::OperatorShape { reason }
            | OuterStrategyError::NonFiniteHessian { reason }
            | OuterStrategyError::RhoBlockShape { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for OuterStrategyError {}

impl From<OuterStrategyError> for String {
    fn from(err: OuterStrategyError) -> String {
        err.to_string()
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

pub(crate) struct RhoBlockAdditiveOuterHessian {
    base: Arc<dyn OuterHessianOperator>,
    rho_block: Array2<f64>,
    dim: usize,
}

impl OuterHessianOperator for RhoBlockAdditiveOuterHessian {
    fn dim(&self) -> usize {
        self.dim
    }

    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.dim {
            return Err(OuterStrategyError::OperatorShape {
                reason: format!(
                    "outer Hessian operator input length mismatch: got {}, expected {}",
                    v.len(),
                    self.dim
                ),
            }
            .into());
        }
        let mut out = self.base.matvec(v)?;
        let k = self.rho_block.nrows();
        if k > 0 {
            let rho_v = v.slice(ndarray::s![..k]).to_owned();
            let rho_out = self.rho_block.dot(&rho_v);
            out.slice_mut(ndarray::s![..k]).scaled_add(1.0, &rho_out);
        }
        Ok(out)
    }

    /// Zero-alloc override for the inner-CG hot path.
    ///
    /// Delegates to `base.apply_into` (which may itself be zero-alloc when the
    /// base overrides) then adds the rho-block correction using a row-dot loop
    /// rather than materialising an intermediate `rho_v.to_owned()` +
    /// `rho_block.dot(&rho_v)` pair.
    fn apply_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), String> {
        if v.len() != self.dim {
            return Err(OuterStrategyError::OperatorShape {
                reason: format!(
                    "outer Hessian operator input length mismatch: got {}, expected {}",
                    v.len(),
                    self.dim
                ),
            }
            .into());
        }
        if out.len() != self.dim {
            return Err(OuterStrategyError::OperatorShape {
                reason: format!(
                    "outer Hessian apply_into output length mismatch: got {}, expected {}",
                    out.len(),
                    self.dim
                ),
            }
            .into());
        }
        self.base.apply_into(v, out)?;
        let k = self.rho_block.nrows();
        if k > 0 {
            let v_top = v.slice(ndarray::s![..k]);
            for i in 0..k {
                out[i] += self.rho_block.row(i).dot(&v_top);
            }
        }
        Ok(())
    }

    /// Batched apply: delegate to the inner operator's `mul_mat` (which may
    /// itself parallelize), then add `rho_block` to the leading `k × k`
    /// block. This propagates the batched-amortization benefit to wrappers
    /// — `materialize_dense` (which goes through `mul_mat(eye)`) and any
    /// future K-column inner-CG batching path.
    fn mul_mat(&self, factor: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let mut out = self.base.mul_mat(factor)?;
        let k = self.rho_block.nrows();
        if k > 0 {
            if k > out.nrows() {
                return Err(OuterStrategyError::RhoBlockShape {
                    reason: format!(
                        "rho-block Hessian update shape mismatch: rho_block is {}x{}, mul_mat output has {} rows",
                        self.rho_block.nrows(),
                        self.rho_block.ncols(),
                        out.nrows()
                    ),
                }
                .into());
            }
            // Update the leading-k rows of `out` by the rho_block contribution
            // to the first k rows of v: out[..k, :] += rho_block · factor[..k, :].
            let factor_top = factor.slice(ndarray::s![..k, ..]);
            let rho_contrib = self.rho_block.dot(&factor_top);
            out.slice_mut(ndarray::s![..k, ..])
                .scaled_add(1.0, &rho_contrib);
        }
        Ok(out)
    }

    fn is_cheap_to_materialize(&self) -> bool {
        self.base.is_cheap_to_materialize()
    }

    fn materialization_capability(&self) -> OuterHessianMaterialization {
        self.base.materialization_capability()
    }
}

/// Upper safety bound for operator materialization after the operator has
/// explicitly declared that dense probing is cheap. Dimension alone is never
/// sufficient: a 50-column operator can still mean 50 full row-streaming CTN,
/// Duchon, or survival passes.
pub(crate) const OUTER_HVP_MATERIALIZE_MAX_DIM: usize = 64;

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
/// [`OuterCapability`]: callers that know the shape upfront declare
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
