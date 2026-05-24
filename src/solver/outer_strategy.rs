//! Central authority for outer smoothing-parameter optimization strategy.
//!
//! Every path that optimizes smoothing parameters (standard REML, link-wiggle,
//! GAMLSS custom family, spatial kappa, etc.) declares its derivative
//! capability here and receives an [`OuterPlan`] that determines which solver
//! and Hessian source to use.
//!
//! # Design invariant
//!
//! The planner never synthesizes numerical Hessians. If a path cannot provide
//! an analytic Hessian, that fact is visible in its
//! [`OuterCapability`] declaration and in the resulting [`OuterPlan`], which
//! falls back to BFGS or an EFS variant instead of synthesizing second-order
//! curvature numerically.

use crate::cache::{LoadSource, Session as CacheSession};
use crate::estimate::EstimationError;
use crate::solver::estimate::reml::unified::BarrierConfig;
use ::opt::{
    Arc as ArcOptimizer, ArcError, Bfgs, BfgsError, Bounds, FallbackPolicy as OptFallbackPolicy,
    FirstOrderObjective, FirstOrderSample, FixedPoint, FixedPointError, FixedPointObjective,
    FixedPointSample, FixedPointStatus, GradientTolerance, HessianFallbackPolicy,
    HessianMaterialization, HessianOperator, HessianValue, MatrixFreeTrustRegion, MaxIterations,
    ObjectiveEvalError, OperatorObjective, OperatorSample, OptimizationStatus, OptimizerObserver,
    SecondOrderObjective, SecondOrderSample, Solution, StepInfo, Tolerance, ZerothOrderObjective,
};
use ndarray::{Array1, Array2, ArrayView2};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const OPERATOR_TRUST_RESTART_RADIUS_FLOOR: f64 = 1.0e-6;

/// Bidirectional inner-PIRLS feedback channel.
///
/// The outer-loop scheduler (BFGS or ARC bridge) writes a coarsened
/// iteration cap into `cap` before each accepted gradient/Hessian eval,
/// and the inner solver (`execute_pirls_if_needed`) writes back into
/// `last_iters` / `last_converged` after each NON-screening solve so the
/// next outer iter's schedule can adapt to the inner solver's actual
/// convergence behavior rather than a hardcoded iter-count tier.
///
/// All atomics are owned by `RemlObjectiveState`; the bridges hold
/// `Arc` clones. `last_iters == 0` means "no inner-Newton signal yet" —
/// the schedule falls back to the coarse iter-count tier for the first
/// outer iter. `ift_residual_bits == 0` means "no IFT-predictor quality
/// signal yet" — the schedule's +margin reverts to the conservative
/// default. The two signals are independent: the IFT residual may be
/// missing even after a successful inner solve (when the predictor was
/// rejected by the |Δρ| cap and a flat warm-start was used instead).
#[derive(Clone, Debug)]
pub struct InnerProgressFeedback {
    pub cap: Arc<AtomicUsize>,
    /// Count of accepted outer steps observed via the
    /// `OuterAcceptObserver` plugged into `opt`'s solver. Replaces
    /// the bridge-side `eval_count / 2` heuristic on routes that
    /// see trial-and-rejection probing (ARC dense, matrix-free TR):
    /// rejection iters used to inflate the schedule's iter index,
    /// lifting the cap too early. With this counter, the schedule
    /// sees the true accepted-step count and the cap relaxes only
    /// when real progress has been made.
    pub accepted_iter: Arc<AtomicUsize>,
    pub last_iters: Arc<AtomicUsize>,
    pub last_converged: Arc<AtomicBool>,
    /// Bit-packed `f64` residual `‖β_converged − β_predicted‖ /
    /// ‖β_converged‖` from the previous IFT-predicted PIRLS solve.
    /// Used to tighten or loosen the cap's `+margin` when the
    /// predictor's empirical faithfulness is known: a small residual
    /// means the inner Newton starts very close to the KKT β and only
    /// needs +1 iter of margin; a large residual means the prediction
    /// collapsed to flat warm-start and the inner Newton has more
    /// recovery work, so +4 is appropriate. `0` means "no signal yet".
    pub ift_residual: Arc<AtomicU64>,
    /// Bit-packed `f64` accepted gain ratio
    /// (`actual_reduction / predicted_reduction`) from the most recent
    /// non-screening PIRLS solve. NaN bits encode "no signal yet"
    /// (matches `ift_residual`'s sentinel discipline). Used by
    /// `first_order_inner_cap_schedule` as a third quality signal
    /// alongside `last_iters` and `last_converged`: a small accept_rho
    /// (model overstating predicted reduction) is a hint the next
    /// iter's inner Newton may need extra margin even when the
    /// previous solve converged in few iters.
    pub accept_rho: Arc<AtomicU64>,
}

impl InnerProgressFeedback {
    /// Snapshot the read-back atomics for the cap schedule. Returns `None`
    /// when no inner solve has reported yet (`last_iters == 0`); the
    /// schedule then falls back to the coarse iter-count tier.
    ///
    /// The IFT residual decoding uses the same NaN-sentinel discipline
    /// as `RemlState::predict_warm_start_beta_ift_with_outcome` — see commit
    /// `748cc066` for the rationale. A residual of exactly 0 (every
    /// β_predicted_i bit-equal to β_converged_i) must NOT be confused
    /// with "no signal yet"; the NaN sentinel + `is_finite()` check
    /// distinguishes the two cleanly. Both ends of the atomic share
    /// `crate::solver::reml::runtime::IFT_RESIDUAL_NO_SIGNAL_BITS`
    /// implicitly via the same bit pattern.
    fn snapshot(&self) -> Option<InnerProgressSnapshot> {
        let iters = self.last_iters.load(Ordering::Relaxed);
        if iters == 0 {
            None
        } else {
            // NaN sentinel + is_finite() check covers three cases in
            // one expression: "no signal yet" (sentinel decodes to NaN,
            // fails is_finite), "corrupted state" (any non-finite or
            // negative residual), and "real signal" (finite non-negative
            // → Some). Matches the IFT predictor's reader semantics.
            let residual_bits = self.ift_residual.load(Ordering::Relaxed);
            let r = f64::from_bits(residual_bits);
            let last_ift_residual = if r.is_finite() && r >= 0.0 {
                Some(r)
            } else {
                None
            };
            let accept_rho_bits = self.accept_rho.load(Ordering::Relaxed);
            let ar = f64::from_bits(accept_rho_bits);
            let last_accept_rho = if ar.is_finite() && ar >= 0.0 {
                Some(ar)
            } else {
                None
            };
            Some(InnerProgressSnapshot {
                last_iters: iters,
                last_converged: self.last_converged.load(Ordering::Relaxed),
                last_ift_residual,
                last_accept_rho,
            })
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct InnerProgressSnapshot {
    last_iters: usize,
    last_converged: bool,
    /// Most-recent IFT predictor residual (see field doc on
    /// `InnerProgressFeedback`). `None` when the predictor has not
    /// reported yet, when the cache was reset, or when the previous
    /// solve fell back to flat warm-start (no IFT prediction
    /// consumed).
    last_ift_residual: Option<f64>,
    /// Most-recent accepted LM gain ratio (see field doc on
    /// `InnerProgressFeedback::accept_rho`). `None` when no step was
    /// accepted in the previous solve (rejection-exhausted) or when
    /// the cache was reset.
    last_accept_rho: Option<f64>,
}

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
    fn is_available(self) -> bool {
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

struct RhoBlockAdditiveOuterHessian {
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

/// Bridge for the partial migration from the legacy `Derivative`-only
/// declaration to the richer [`DeclaredHessianForm`] used by the
/// outer-optimizer planner. Family call sites that still produce
/// `Derivative` (the simple available/unavailable bit) lift through
/// this `From` impl without each rewriting its capability probe:
/// `Analytic` defaults to `Either` so the seed loop inspects the
/// realized seed eval and locks the route at runtime, mirroring the
/// historical seed-eval-branch behavior; `Unavailable` projects to
/// `Unavailable`. New call sites that already know whether the
/// Hessian is materialized as `Dense` or comes through an `Operator`
/// can construct the richer variant directly.
impl From<Derivative> for DeclaredHessianForm {
    fn from(d: Derivative) -> Self {
        match d {
            Derivative::Analytic => DeclaredHessianForm::Either,
            Derivative::Unavailable => DeclaredHessianForm::Unavailable,
        }
    }
}

/// Declares what a specific model path can provide to the outer optimizer.
///
/// Each call site that optimizes smoothing parameters constructs one of these
/// to describe its analytic derivative coverage. The [`plan`] function then
/// selects the optimizer and Hessian strategy.
const SMALL_OUTER_BFGS_MAX_PARAMS: usize = 8;
const SECOND_ORDER_GEOMETRY_PROBE_MAX_PARAMS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OuterThetaLayout {
    pub n_params: usize,
    pub psi_dim: usize,
}

impl OuterThetaLayout {
    pub const fn new(n_params: usize, psi_dim: usize) -> Self {
        Self { n_params, psi_dim }
    }

    pub const fn rho_dim(&self) -> usize {
        self.n_params.saturating_sub(self.psi_dim)
    }

    fn validate_capability(&self, context: &str) -> Result<(), EstimationError> {
        if self.psi_dim > self.n_params {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "{context}: invalid outer theta layout (psi_dim={} exceeds n_params={})",
                self.psi_dim, self.n_params
            )));
        }
        Ok(())
    }

    fn validate_point_len(
        &self,
        theta: &Array1<f64>,
        context: &str,
    ) -> Result<(), ObjectiveEvalError> {
        if theta.len() != self.n_params {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer theta length mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                theta.len(),
                self.n_params,
                self.rho_dim(),
                self.psi_dim
            )));
        }
        Ok(())
    }

    fn validate_gradient_len(
        &self,
        gradient: &Array1<f64>,
        context: &str,
    ) -> Result<(), ObjectiveEvalError> {
        if gradient.len() != self.n_params {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer gradient length mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                gradient.len(),
                self.n_params,
                self.rho_dim(),
                self.psi_dim
            )));
        }
        Ok(())
    }

    fn validate_hessian_shape(
        &self,
        hessian: &Array2<f64>,
        context: &str,
    ) -> Result<(), ObjectiveEvalError> {
        if hessian.nrows() != self.n_params || hessian.ncols() != self.n_params {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer Hessian shape mismatch: got {}x{}, expected {}x{} (rho_dim={}, psi_dim={})",
                hessian.nrows(),
                hessian.ncols(),
                self.n_params,
                self.n_params,
                self.rho_dim(),
                self.psi_dim
            )));
        }
        Ok(())
    }

    fn validate_efs_eval(&self, eval: &EfsEval, context: &str) -> Result<(), ObjectiveEvalError> {
        if eval.steps.len() != self.n_params {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer EFS step length mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                eval.steps.len(),
                self.n_params,
                self.rho_dim(),
                self.psi_dim
            )));
        }
        if let Some(ref psi_gradient) = eval.psi_gradient
            && psi_gradient.len() != self.psi_dim
        {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer EFS psi-gradient length mismatch: got {}, expected {}",
                psi_gradient.len(),
                self.psi_dim
            )));
        }
        if let Some(ref psi_indices) = eval.psi_indices {
            if psi_indices.len() != self.psi_dim {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: outer EFS psi-index count mismatch: got {}, expected {}",
                    psi_indices.len(),
                    self.psi_dim
                )));
            }
            if psi_indices.iter().any(|&idx| idx >= self.n_params) {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: outer EFS psi index out of range for n_params={}",
                    self.n_params
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct OuterCapability {
    pub gradient: Derivative,
    /// Declared shape of the analytic Hessian (or its absence). Replaces
    /// the binary `Derivative` so the planner can route between dense
    /// ARC and matrix-free trust-region *before* seed evaluation. See
    /// [`DeclaredHessianForm`].
    pub hessian: DeclaredHessianForm,
    /// Number of smoothing (+ any auxiliary hyper-) parameters being optimized.
    pub n_params: usize,
    /// Number of ψ (design-moving) coordinates among the extended
    /// hyperparameter coordinates. When 0, all coords are penalty-like and
    /// pure EFS is eligible (given `fixed_point_available`). When > 0,
    /// hybrid EFS is eligible instead: EFS for ρ + preconditioned gradient
    /// for ψ.
    ///
    /// # Hybrid EFS strategy (when `psi_dim > 0`)
    ///
    /// Enabled when `psi_dim > 0`,
    /// `n_params > SMALL_OUTER_BFGS_MAX_PARAMS`, and
    /// `fixed_point_available`.
    /// Combines:
    /// - Standard EFS multiplicative fixed-point updates for ρ coordinates
    /// - Safeguarded preconditioned gradient steps for ψ coordinates:
    ///   `Δψ = -α G⁺ g_ψ` where G is the trace Gram matrix
    ///
    /// Mathematically necessary because no EFS-type fixed-point iteration
    /// exists for indefinite B_ψ (see response.md Section 2). The structural
    /// requirement for EFS is `H^{-1/2} B_d H^{-1/2} ≽ 0` (PSD) plus fixed
    /// nullspace — exactly what penalty-like coords satisfy and design-moving
    /// coords do not.
    ///
    /// The hybrid is O(1) H⁻¹ solves per iteration (same as pure EFS),
    /// compared to O(dim(θ)) for BFGS.
    pub psi_dim: usize,
    /// Whether the objective actually implements `eval_efs()` for fixed-point
    /// plans. Structural eligibility (`psi_dim == 0` / `psi_dim > 0`)
    /// is not sufficient by itself: if this is false, the planner must stay on
    /// Newton/BFGS-style plans even when EFS or Hybrid-EFS would otherwise be
    /// mathematically admissible.
    pub fixed_point_available: bool,
    /// Optional log-barrier configuration for structural monotonicity constraints.
    /// When present, EFS is still eligible at plan time, but the EFS iteration
    /// loop performs a quantitative check each step: if
    /// `barrier_curvature_is_significant(β, ref_diag, threshold)` fires, EFS
    /// is abandoned and the fallback ladder routes to a first-order joint
    /// optimizer.
    ///
    /// Previously this was a binary `barrier_active: bool` that unconditionally
    /// blocked EFS. The quantitative check allows EFS when constraints exist but
    /// the barrier curvature is negligible (coefficients far from their bounds).
    pub barrier_config: Option<BarrierConfig>,
    /// Policy hint for derivative-free auxiliary optimizers only. Primary REML
    /// optimization ignores this flag when an analytic Hessian exists: exact
    /// second-order geometry must not be hidden behind a quasi-Newton policy.
    pub prefer_gradient_only: bool,
    /// Policy hint: even when the objective implements `eval_efs()` and the
    /// coordinate structure is penalty-like, the planner must NOT select
    /// EFS/HybridEfs for this problem.
    ///
    /// Set by the caller for problem classes where the Wood-Fasiolo structural
    /// property (`H^{-1/2} B_k H^{-1/2} ≽ 0` plus parameter-independent
    /// nullspace) is known not to hold — e.g. GAMLSS/location-scale families
    /// where the joint Hessian is β-dependent and cross-block smoothers
    /// induce non-diagonal curvature that the EFS multiplicative fixed-point
    /// cannot resolve. Also set by the automatic fallback cascade when an
    /// EFS/HybridEfs attempt failed to converge, so the next attempt falls
    /// back to analytic-gradient BFGS rather than retrying EFS.
    pub disable_fixed_point: bool,
}

impl OuterCapability {
    pub const fn theta_layout(&self) -> OuterThetaLayout {
        OuterThetaLayout::new(self.n_params, self.psi_dim)
    }

    pub fn validate_layout(&self, context: &str) -> Result<(), EstimationError> {
        self.theta_layout().validate_capability(context)
    }

    /// True when all coordinates are penalty-like (no ψ coords).
    pub const fn all_penalty_like(&self) -> bool {
        self.psi_dim == 0
    }
    /// True when ψ (design-moving) coordinates are present.
    pub const fn has_psi_coords(&self) -> bool {
        self.psi_dim > 0
    }

    fn efs_plan_eligible(&self) -> bool {
        self.fixed_point_available
            && !self.disable_fixed_point
            && self.all_penalty_like()
            && self.n_params > SMALL_OUTER_BFGS_MAX_PARAMS
    }

    fn hybrid_efs_plan_eligible(&self) -> bool {
        self.fixed_point_available
            && !self.disable_fixed_point
            && self.has_psi_coords()
            && self.n_params > SMALL_OUTER_BFGS_MAX_PARAMS
    }

    fn declared_hessian_for_planning(&self) -> Derivative {
        if self.hessian.is_analytic() {
            Derivative::Analytic
        } else {
            Derivative::Unavailable
        }
    }
}

/// Which solver algorithm to use for the outer optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// Adaptive Regularized Cubic; fastest convergence, requires Hessian.
    Arc,
    /// BFGS; gradient only, builds a dense curvature approximation.
    Bfgs,
    /// Extended Fellner-Schall; multiplicative fixed-point iteration.
    /// Only valid when all hyperparameter coordinates are penalty-like.
    /// Needs no gradient or Hessian — only traces tr(H^{-1} A_k) and
    /// Frobenius norms from the inner solution.
    Efs,
    /// Hybrid EFS + preconditioned gradient.
    ///
    /// Used when ψ (design-moving) coordinates are present alongside ρ
    /// (penalty-like) coordinates. Combines:
    /// - Standard EFS multiplicative fixed-point steps for ρ coords
    /// - Safeguarded preconditioned gradient steps for ψ coords:
    ///   `Δψ = -α G⁺ g_ψ` where `G_{de} = tr(H⁻¹ B_d H⁻¹ B_e)`
    ///
    /// This hybrid exists because no EFS-type fixed-point iteration can
    /// guarantee convergence for indefinite B_ψ (proven by counterexample
    /// in response.md Section 2). The key structural property that EFS
    /// needs — `H^{-1/2} B_d H^{-1/2} ≽ 0` plus parameter-independent
    /// nullspace — holds for penalty-like coords but fails for
    /// design-moving coords where B_ψ has mixed inertia.
    ///
    /// The preconditioned gradient uses the same trace Gram matrix that
    /// EFS already computes, so the cost is O(1) H⁻¹ solves per iteration
    /// (same as pure EFS), compared to O(dim(θ)) for full BFGS.
    HybridEfs,
    /// Opportunistic coordinate compass search (positive basis {±e_i} with
    /// step contraction). Derivative-free by construction — no gradient.
    ///
    /// Reserved for genuinely-derivative-free auxiliary searches
    /// (baseline-theta for parametric survival baselines, SAS/BetaLogistic
    /// /Mixture inverse-link parameters) where no analytic
    /// ∂cost/∂θ is available and the dimension is small (≤ ~5).
    ///
    /// The planner only selects this variant when the caller has opted in
    /// via [`SolverClass::AuxiliaryGradientFree`]; it is NEVER selected
    /// for the main REML outer. For the big REML outer, declared-analytic
    /// gradients must converge on their own merits.
    ///
    /// Convergence to a stationary point on any continuously-differentiable
    /// cost bounded below on a compact box follows from
    /// Kolda-Lewis-Torczon, SIAM Review 45:385, 2003, Thm 3.3. The theorem
    /// requires that all 2·dim basis directions are polled before step
    /// contraction; the dispatcher's sweep loop satisfies this by the
    /// `!improved ⇒ step /= 2` branch.
    CompassSearch,
}

/// Declares which "class" of outer optimization the caller is doing.
///
/// The default `Primary` class applies to the main REML outer — the
/// canonical smoothing-parameter optimization — and has access to
/// Arc/Bfgs/Efs/HybridEfs according to declared derivatives.
///
/// `AuxiliaryGradientFree` unlocks `Solver::CompassSearch` for small-dim
/// auxiliary pre-optimizations where no analytic ∂cost/∂θ exists (survival
/// baseline theta, non-standard inverse-link parameters). The planner gates
/// selection of CompassSearch strictly on this flag; REML builders never
/// set it, so REML can never be routed to compass search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SolverClass {
    /// The main REML outer — smoothing parameters and ψ-coords where
    /// analytic gradient (and typically analytic Hessian) is the contract.
    #[default]
    Primary,
    /// A genuinely-derivative-free low-dim auxiliary search (e.g. survival
    /// baseline theta). Opts into `Solver::CompassSearch` when gradient is
    /// Unavailable. Must not be set by REML builders.
    AuxiliaryGradientFree,
}

#[inline]
fn effective_seed_budget(
    requested_budget: usize,
    solver: Solver,
    risk_profile: crate::seeding::SeedRiskProfile,
    screening_enabled: bool,
) -> usize {
    let requested_budget = requested_budget.max(1);
    let capped = match (solver, risk_profile) {
        (Solver::Efs | Solver::HybridEfs, _) => 1,
        (Solver::Arc, crate::seeding::SeedRiskProfile::Survival) => 1,
        (Solver::Arc, crate::seeding::SeedRiskProfile::GeneralizedLinear) if screening_enabled => 1,
        (Solver::Arc, crate::seeding::SeedRiskProfile::GeneralizedLinear) => 2,
        // Aux direct-search is a single-start low-dim local method; restarting
        // from another seed would just re-explore the same basin.
        (Solver::CompassSearch, _) => 1,
        _ => requested_budget,
    };
    requested_budget.min(capped)
}

#[inline]
fn should_screen_seeds(
    config: &OuterConfig,
    solver: Solver,
    generated_seed_count: usize,
    seed_budget: usize,
) -> bool {
    config.screening_cap.is_some()
        && generated_seed_count > seed_budget
        && matches!(
            solver,
            Solver::Arc | Solver::Bfgs | Solver::Efs | Solver::HybridEfs
        )
}

#[inline]
fn expensive_unsuccessful_seed_limit(
    solver: Solver,
    risk_profile: crate::seeding::SeedRiskProfile,
) -> Option<usize> {
    match (solver, risk_profile) {
        (Solver::Efs | Solver::HybridEfs, _) => Some(1),
        (Solver::Arc, crate::seeding::SeedRiskProfile::Survival) => Some(1),
        (Solver::Arc, crate::seeding::SeedRiskProfile::GeneralizedLinear) => Some(2),
        (Solver::CompassSearch, _) => Some(1),
        _ => None,
    }
}

/// Multipliers for the seed-screening cap cascade, applied to the user's
/// `screen_max_inner_iterations`.
///
/// The cascade evaluates seeds at successive caps until at least one
/// produces a finite cost — at which point it ranks them and exits. The
/// geometric ×4 progression keeps each escalation step cheap relative to
/// the next while still letting the cap reach the full inner budget if
/// needed: `initial × {1, 4, 16}` followed by uncapped (`0` interpreted
/// by the inner solver as "use the full `pirls_config.max_iterations`").
///
/// Worst-case extra work bounds: every seed pays at most
/// `initial × (1 + 4 + 16)` = 21 × initial inner iterations across the
/// three capped stages before falling through to the uncapped pass —
/// negligible overhead compared to a full P-IRLS solve, paid only when
/// every cap stage collapsed all seeds to non-finite cost.
const SEED_SCREENING_CASCADE_MULTIPLIERS: [usize; 3] = [1, 4, 16];

/// Sentinel cap value passed to the inner solver to mean "no cap — use
/// the full `pirls_config.max_iterations`". Always the final cascade
/// stage after the geometric escalation exhausts.
const SEED_SCREENING_UNCAPPED: usize = 0;

fn rank_seeds_with_screening(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    seeds: &[Array1<f64>],
) -> Vec<Array1<f64>> {
    let Some(screening_cap) = config.screening_cap.as_ref() else {
        return seeds.to_vec();
    };

    let initial_cap = config.seed_config.screen_max_inner_iterations.max(1);
    let previous_cap = screening_cap.swap(initial_cap, Ordering::Relaxed);

    // Geometric cap cascade: each stage exits the moment any seed produces
    // a finite cost. The original two-stage protocol (initial cap → fully
    // uncapped on every seed) has a degenerate worst case at biobank scale
    // — when every seed at the shallow cap collapses, we re-evaluate every
    // seed at the *full* inner budget, costing `N_seeds × full_pirls_work`
    // just to pick a starting point. The cascade replaces that all-or-
    // nothing jump with a geometric escalation: the typical case stays at
    // the initial cap (one pass), and the rare uniform-failure case pays
    // only `21 × initial` extra inner iterations before the uncapped
    // fallback.
    let cascade_caps = [
        initial_cap.saturating_mul(SEED_SCREENING_CASCADE_MULTIPLIERS[0]),
        initial_cap.saturating_mul(SEED_SCREENING_CASCADE_MULTIPLIERS[1]),
        initial_cap.saturating_mul(SEED_SCREENING_CASCADE_MULTIPLIERS[2]),
        SEED_SCREENING_UNCAPPED,
    ];

    let mut ranked: Vec<(usize, f64)> = Vec::with_capacity(seeds.len());
    let mut rejected = 0usize;
    let mut final_cap_used = initial_cap;
    let mut stages_consumed = 0usize;
    let cascade_start = std::time::Instant::now();

    for (stage, &cap) in cascade_caps.iter().enumerate() {
        screening_cap.store(cap, Ordering::Relaxed);
        ranked.clear();
        rejected = 0;
        for (idx, seed) in seeds.iter().enumerate() {
            obj.reset();
            screening_cap.store(cap, Ordering::Relaxed);
            match obj.eval_screening_proxy(seed) {
                Ok(cost) if cost.is_finite() => ranked.push((idx, cost)),
                Ok(_) | Err(_) => rejected += 1,
            }
        }
        final_cap_used = cap;
        stages_consumed = stage + 1;
        if !ranked.is_empty() {
            if stage > 0 {
                log::info!(
                    "[OUTER] {context}: seed screening cap escalated from {} to {} \
                     (initial cap was too shallow for this problem; {}/{} seeds ranked)",
                    initial_cap,
                    if cap == 0 {
                        "uncapped".to_string()
                    } else {
                        cap.to_string()
                    },
                    ranked.len(),
                    seeds.len(),
                );
            }
            break;
        }
    }

    screening_cap.store(previous_cap, Ordering::Relaxed);
    obj.reset();
    log::info!(
        "[OUTER] {context}: seed screening cascade complete elapsed={:.3}s stages_used={} final_cap={} ranked={}/{}",
        cascade_start.elapsed().as_secs_f64(),
        stages_consumed,
        if final_cap_used == 0 {
            "uncapped".to_string()
        } else {
            final_cap_used.to_string()
        },
        ranked.len(),
        seeds.len(),
    );

    if ranked.is_empty() {
        log::info!(
            "[OUTER] {context}: no finite seed cost even with full inner budget \
             ({} seeds, {} rejected, {} cascade stages tried); keeping heuristic order",
            seeds.len(),
            rejected,
            stages_consumed,
        );
        return seeds.to_vec();
    }

    ranked.sort_by(|(idx_a, cost_a), (idx_b, cost_b)| {
        cost_a.total_cmp(cost_b).then_with(|| idx_a.cmp(idx_b))
    });

    let mut ordered = Vec::with_capacity(seeds.len());
    let mut seen = vec![false; seeds.len()];
    for (idx, _) in ranked {
        seen[idx] = true;
        ordered.push(seeds[idx].clone());
    }
    for (idx, seed) in seeds.iter().enumerate() {
        if !seen[idx] {
            ordered.push(seed.clone());
        }
    }

    log::debug!(
        "[OUTER] {context}: seed screening ranked {}/{} candidates at cap={} \
         (initial cap={}, stages used={}); rejected={}",
        ordered.len() - rejected,
        seeds.len(),
        if final_cap_used == 0 {
            "uncapped".to_string()
        } else {
            final_cap_used.to_string()
        },
        initial_cap,
        stages_consumed,
        rejected,
    );

    ordered
}

#[inline]
fn candidate_improves_best(candidate: &OuterResult, best: Option<&OuterResult>) -> bool {
    match best {
        None => true,
        Some(best) if candidate.converged != best.converged => candidate.converged,
        Some(best) => candidate.final_value < best.final_value,
    }
}

/// How the Hessian will be obtained for the outer optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HessianSource {
    /// Exact analytic Hessian provided by the objective.
    Analytic,
    /// No explicit Hessian; BFGS builds a rank-2 approximation from
    /// gradient history.
    BfgsApprox,
    /// No explicit Hessian or gradient needed. EFS uses traces and
    /// Frobenius norms from the inner solution directly.
    EfsFixedPoint,
    /// Hybrid EFS + preconditioned gradient for ψ coordinates.
    /// EFS traces for ρ coords, trace Gram matrix + gradient for ψ coords.
    HybridEfsFixedPoint,
}

/// Requested derivative order for an outer objective evaluation.
///
/// Cost-only paths continue to use [`OuterObjective::eval_cost`]. This enum is
/// for the shared `eval` bridge where the runner needs either first-order or
/// second-order information depending on the active plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterEvalOrder {
    /// Compute value and gradient only.
    ValueAndGradient,
    /// Compute value, gradient, and analytic Hessian when available.
    ValueGradientHessian,
}

/// The outer optimization plan. Produced by [`plan`], consumed by the runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OuterPlan {
    pub solver: Solver,
    pub hessian_source: HessianSource,
}

pub(crate) const EFS_FIRST_ORDER_FALLBACK_MARKER: &str = "[outer-efs-first-order-fallback]";

/// Whether outer_strategy should automatically derive a retry ladder from the
/// primary capability, or disable retries entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackPolicy {
    /// Centralized retry path chosen from the declared capability.
    Automatic,
    /// No retries; use only the primary plan.
    Disabled,
}

impl std::fmt::Display for OuterPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "solver={:?}, hessian_source={:?}",
            self.solver, self.hessian_source
        )
    }
}

impl OuterPlan {
    /// Stable, grep-friendly routing token for biobank/log regression
    /// assertions. Emits `solver=<Solver>;hessian=<Source>;matrix-free=<bool>`.
    /// Planning alone does not prove the runtime Hessian representation;
    /// matrix-free routing is decided after the seed evaluation returns an
    /// operator Hessian, so the static plan token reports `false`.
    pub fn routing_log_line(&self) -> String {
        let matrix_free = false;
        format!(
            "solver={:?};hessian={:?};matrix-free={}",
            self.solver, self.hessian_source, matrix_free
        )
    }
}

/// Select the outer optimization strategy from the declared capability.
///
/// This is a pure function with no side effects. All policy lives here.
pub fn plan(cap: &OuterCapability) -> OuterPlan {
    use Derivative::*;
    use HessianSource as H;
    use Solver as S;

    match (cap.gradient, cap.declared_hessian_for_planning()) {
        (Analytic, Analytic) => OuterPlan {
            solver: S::Arc,
            hessian_source: H::Analytic,
        },
        // EFS: all penalty-like coords, no analytic Hessian, many params.
        // Multiplicative fixed-point needs only traces — no gradient evals.
        // Much cheaper than BFGS for k=10-50 smoothing parameters.
        //
        // When a log-barrier is present (monotonicity constraints), EFS is
        // still selected here. The EFS iteration loop in `run_outer` performs
        // a quantitative check each step via `barrier_curvature_is_significant`
        // and bails out early if the barrier curvature becomes non-negligible
        // relative to the penalized Hessian diagonal.
        (Analytic, Unavailable) if cap.efs_plan_eligible() => OuterPlan {
            solver: S::Efs,
            hessian_source: H::EfsFixedPoint,
        },
        (Unavailable, Unavailable) if cap.efs_plan_eligible() => OuterPlan {
            solver: S::Efs,
            hessian_source: H::EfsFixedPoint,
        },

        // Hybrid EFS: ψ (design-moving) coords present alongside ρ coords.
        //
        // When ψ coords are present, pure EFS is invalid because B_ψ can be
        // indefinite (see response.md Section 2 for the counterexample). But
        // falling back to full BFGS wastes the cheap EFS structure for ρ coords.
        //
        // The hybrid strategy uses EFS for ρ-coords and a safeguarded
        // preconditioned gradient step for ψ-coords:
        //   Δψ = -α G⁺ g_ψ,  G_{de} = tr(H⁻¹ B_d H⁻¹ B_e)
        //
        // This stays O(1) H⁻¹ solves per iteration (vs O(dim(θ)) for BFGS)
        // and uses the same trace Gram matrix that EFS already computes.
        (Analytic, Unavailable) if cap.hybrid_efs_plan_eligible() => OuterPlan {
            solver: S::HybridEfs,
            hessian_source: H::HybridEfsFixedPoint,
        },
        (Unavailable, Unavailable) if cap.hybrid_efs_plan_eligible() => OuterPlan {
            solver: S::HybridEfs,
            hessian_source: H::HybridEfsFixedPoint,
        },

        // Gradient-only problems should use a gradient-only optimizer.
        (Analytic, Unavailable) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },
        // No analytic gradient: emit a BFGS plan so the error surfaces with
        // context rather than as a panic on an unmatched arm. The runner will
        // reject this path because it requires an analytic gradient.
        (Unavailable, _) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },
    }
}

/// Plan selection with an explicit [`SolverClass`] opt-in.
///
/// For `SolverClass::Primary` this is identical to [`plan`] — the main REML
/// outer dispatch never changes behavior.
///
/// For `SolverClass::AuxiliaryGradientFree` with no declared gradient or
/// Hessian capability, returns a `Solver::CompassSearch` plan. This is the
/// sole path by which compass search can be dispatched; the primary REML
/// builder never sets the aux class, so the direct-search variant cannot
/// leak into the big REML outer or the automatic fallback cascade.
///
/// If the aux class is set but analytic gradient IS available, that is a
/// caller error (the caller should have used `Primary` and let Arc/Bfgs
/// handle it); we defer to the standard `plan` in that case so the caller
/// still gets a well-formed plan rather than a silent mis-dispatch.
pub fn plan_with_class(cap: &OuterCapability, class: SolverClass) -> OuterPlan {
    use Derivative::*;
    if class == SolverClass::AuxiliaryGradientFree
        && cap.gradient == Unavailable
        && cap.declared_hessian_for_planning() == Unavailable
        && !cap.efs_plan_eligible()
        && !cap.hybrid_efs_plan_eligible()
    {
        return OuterPlan {
            solver: Solver::CompassSearch,
            hessian_source: HessianSource::BfgsApprox,
        };
    }
    plan(cap)
}

/// Log the outer optimization plan. Called once per fit at the start of
/// outer optimization so the user can see what strategy was selected and why.
pub fn log_plan(context: &str, cap: &OuterCapability, the_plan: &OuterPlan) {
    let hess_warning = match the_plan.hessian_source {
        HessianSource::BfgsApprox if cap.n_params > 0 => {
            " [no Hessian: BFGS approximation]".to_string()
        }
        _ => String::new(),
    };
    let barrier_note = if cap.barrier_config.is_some() && cap.efs_plan_eligible() {
        " [EFS with runtime barrier-curvature guard]"
    } else {
        ""
    };
    let hybrid_note = if the_plan.solver == Solver::HybridEfs {
        " [hybrid EFS(ρ) + preconditioned-gradient(ψ)]"
    } else {
        ""
    };
    // Promoted to info: this fires once per outer optimization dispatch and
    // tells the user immediately whether ARC, BFGS, EFS, etc. was selected
    // and why. That information is otherwise inferred only from the per-iter
    // log tag prefix once the loop has started.
    log::info!(
        "[OUTER] {context}: n_params={}, gradient={:?}, hessian={:?} -> {} [{}]{hess_warning}{barrier_note}{hybrid_note}",
        cap.n_params,
        cap.gradient,
        cap.hessian,
        the_plan,
        the_plan.routing_log_line(),
    );
}

fn requests_immediate_first_order_fallback(message: &str) -> bool {
    message.contains(EFS_FIRST_ORDER_FALLBACK_MARKER)
}

/// Disable the EFS/HybridEfs planner path, forcing BFGS-class solvers on the
/// next attempt. Returns `None` if fixed-point is already disabled.
fn disable_fixed_point(cap: &OuterCapability) -> Option<OuterCapability> {
    (!cap.disable_fixed_point && (cap.efs_plan_eligible() || cap.hybrid_efs_plan_eligible())).then(
        || {
            let mut degraded = cap.clone();
            degraded.disable_fixed_point = true;
            degraded
        },
    )
}

fn automatic_fallback_attempts(cap: &OuterCapability) -> Vec<OuterCapability> {
    // Production fallback ladder is strictly analytic-gradient.
    //
    // The cascade is:
    //   1. If the primary plan is EFS/HybridEFS AND an analytic gradient is
    //      available, retry with fixed-point disabled so the analytic
    //      derivative declaration is evaluated directly.
    //   2. If the primary plan is Arc (declared (Analytic, Analytic)
    //      capability), do NOT add a degraded fallback. Demoting to
    //      BFGS+BfgsApprox in this case discards the analytic outer Hessian
    //      ARC was using — a strictly weaker geometry — and silently masks
    //      ARC's actual failure mode (e.g. budget exhaustion, indefinite
    //      curvature) under a BFGS Strong-Wolfe plateau on a flat surface.
    //      ARC retries are handled by the per-attempt budget-bump retry
    //      ladder in `run_outer_with_strategy`; once that is exhausted, the
    //      caller surfaces the underlying ARC failure verbatim.
    //   3. Otherwise (e.g. (Analytic, Unavailable) without EFS eligibility,
    //      which is the BFGS primary), there is nothing to degrade further
    //      — the caller surfaces the RemlOptimizationFailed error so the
    //      non-convergence is visible.
    let mut attempts = Vec::new();

    if cap.gradient == Derivative::Analytic
        && matches!(plan(cap).solver, Solver::Efs | Solver::HybridEfs)
        && let Some(no_fp_cap) = disable_fixed_point(cap)
    {
        attempts.push(no_fp_cap.clone());
        return attempts;
    }

    // Arc primary: no lateral demotion to BFGS. The runner's ARC-budget-bump
    // retry covers cases where ARC needed more iterations; if even that is
    // exhausted, the caller sees the genuine analytic-Hessian non-convergence
    // rather than a misleading BFGS-on-flat-surface plateau.
    if matches!(plan(cap).solver, Solver::Arc) {
        return attempts;
    }

    attempts
}

fn disabled_fallback_hybrid_efs_has_standalone_bfgs_primary(
    cap: &OuterCapability,
    config: &OuterConfig,
) -> bool {
    config.solver_class == SolverClass::Primary
        && config.fallback_policy == FallbackPolicy::Disabled
        && cap.gradient == Derivative::Analytic
        && matches!(plan(cap).solver, Solver::HybridEfs)
}

fn primary_capability_for_config(
    mut cap: OuterCapability,
    config: &OuterConfig,
    context: &str,
) -> OuterCapability {
    if disabled_fallback_hybrid_efs_has_standalone_bfgs_primary(&cap, config) {
        // HybridEFS is not a standalone first-order method for ψ coordinates:
        // when ψ backtracking proves non-descent, the bridge intentionally
        // surfaces `EFS_FIRST_ORDER_FALLBACK_MARKER` so the runner can switch
        // to a joint gradient solver that enforces ∇ψ V = 0. With fallback
        // disabled and an analytic gradient available, selecting HybridEFS as
        // the only primary attempt is internally inconsistent; BFGS is the
        // standalone first-order primary for that capability.
        log::info!(
            "[OUTER] {context}: HybridEFS requires the automatic first-order \
             escape path for ψ coordinates; fallback is disabled, so routing the \
             primary attempt to analytic-gradient BFGS"
        );
        cap.disable_fixed_point = true;
    }
    cap
}

/// Result of one outer objective evaluation.
///
/// The Hessian field uses [`HessianResult`] instead of `Option<Array2<f64>>`
/// to make the presence/absence of an analytic Hessian explicit and
/// pattern-matchable.
pub struct OuterEval {
    pub cost: f64,
    pub gradient: Array1<f64>,
    pub hessian: HessianResult,
    /// Optional inner-solver iterate at this ρ. Families whose inner
    /// solve produces a PIRLS β (GAM, custom-family marginal-slope, …)
    /// populate this so the persistent-cache layer can store `(ρ, β)`
    /// together. Empty for callers that have no inner state or don't
    /// expose it; the cache gracefully writes a ρ-only record then.
    pub inner_beta_hint: Option<Array1<f64>>,
}

impl OuterEval {
    /// Conventional representation of an infeasible trial point.
    ///
    /// `opt` translates the non-finite objective into a recoverable trial
    /// failure so trust-region/line-search solvers retreat without the caller
    /// needing to special-case infeasible regions locally.
    pub fn infeasible(n_params: usize) -> Self {
        Self {
            cost: f64::INFINITY,
            gradient: Array1::zeros(n_params),
            hessian: HessianResult::Unavailable,
            inner_beta_hint: None,
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
    /// The runner must use the [`HessianSource`] from the [`OuterPlan`]
    /// to choose a declared first-order or derivative-free strategy.
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
    /// Extract the Hessian matrix, panicking if unavailable.
    ///
    /// Only call this when the [`OuterPlan`] guarantees `HessianSource::Analytic`.
    pub fn unwrap_analytic(self) -> Array2<f64> {
        match self {
            HessianResult::Analytic(h) => h,
            // `unwrap_analytic`'s doc-comment pins its contract: caller must
            // hold an `OuterPlan` whose `HessianSource::Analytic` is dense.
            HessianResult::Operator(_) => {
                // SAFETY: reaching this arm means the plan promised dense
                // analytic but the family returned an operator — contract
                // violation; fail-fast surfaces the upstream plan bug.
                panic!("expected dense analytic Hessian but got HessianResult::Operator")
            }
            HessianResult::Unavailable => {
                // SAFETY: same contract as Operator above — `Unavailable`
                // means the family failed to produce the analytic Hessian
                // its `OuterPlan` declared.
                panic!("expected analytic Hessian but got HessianResult::Unavailable")
            }
        }
    }

    /// Returns `true` if an analytic Hessian is present in any exact form.
    pub fn is_analytic(&self) -> bool {
        matches!(
            self,
            HessianResult::Analytic(_) | HessianResult::Operator(_)
        )
    }

    /// Convert to the optional Hessian shape used by the opt bridge.
    pub fn into_option(self) -> Option<Array2<f64>> {
        match self {
            HessianResult::Analytic(h) => Some(h),
            HessianResult::Operator(_) => None,
            HessianResult::Unavailable => None,
        }
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

    pub fn add_rho_block_dense(&mut self, rho_block: &Array2<f64>) -> Result<(), String> {
        if rho_block.nrows() != rho_block.ncols() {
            return Err(OuterStrategyError::RhoBlockShape {
                reason: format!(
                    "rho-block Hessian update must be square, got {}x{}",
                    rho_block.nrows(),
                    rho_block.ncols()
                ),
            }
            .into());
        }
        match self {
            HessianResult::Analytic(h) => {
                if rho_block.nrows() > h.nrows() || rho_block.ncols() > h.ncols() {
                    return Err(OuterStrategyError::RhoBlockShape {
                        reason: format!(
                            "rho-block Hessian update shape mismatch: got {}x{}, outer Hessian is {}x{}",
                            rho_block.nrows(),
                            rho_block.ncols(),
                            h.nrows(),
                            h.ncols()
                        ),
                    }
                    .into());
                }
                let k = rho_block.nrows();
                let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                sl += rho_block;
                Ok(())
            }
            HessianResult::Operator(op) => {
                let base = Arc::clone(op);
                let dim = base.dim();
                if rho_block.nrows() > dim {
                    return Err(OuterStrategyError::RhoBlockShape {
                        reason: format!(
                            "rho-block Hessian update dimension mismatch: got {}x{}, operator dim is {}",
                            rho_block.nrows(),
                            rho_block.ncols(),
                            dim
                        ),
                    }
                    .into());
                }
                *self = HessianResult::Operator(Arc::new(RhoBlockAdditiveOuterHessian {
                    base,
                    rho_block: rho_block.clone(),
                    dim,
                }));
                Ok(())
            }
            HessianResult::Unavailable => Ok(()),
        }
    }
}

/// Result of an EFS (Extended Fellner-Schall) evaluation at a given rho.
///
/// Contains the REML/LAML cost at the current rho and the additive step
/// vector produced by `compute_efs_update`. The caller applies the step as
/// `rho_new[i] = rho[i] + steps[i]`.
///
/// For the hybrid EFS+preconditioned-gradient strategy, the steps vector
/// contains both EFS steps (for ρ coords) and preconditioned gradient steps
/// (for ψ coords). The `psi_gradient` field carries the raw ψ-block gradient
/// for optional backtracking.
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
    /// Representative curvature scale of the inner Hessian
    /// `H = X'W_HX + S_λ` (+ any barrier perturbation), at β̂.
    ///
    /// Production REML/LAML paths provide the geometric mean of the active
    /// Hessian spectrum, `exp(log|H|_+ / rank(H))`, which is basis-invariant,
    /// has curvature units, and is already available from the Hessian operator
    /// used to evaluate the objective. The EFS barrier check uses it for the
    /// dimensionally correct comparison `max_j τ/Δ_j² > threshold · scale`.
    pub inner_hessian_scale: Option<f64>,
}

/// Common interface for outer smoothing-parameter objectives.
///
/// Every model path that optimizes smoothing parameters implements this trait.
/// The runner function consumes it and handles solver selection,
/// multi-start, and logging while delegating derivative fallback policy to
/// `opt`.
///
/// # Contract
///
/// - `capability()` must be stable (same result across calls).
/// - `eval()` may return `HessianResult::Unavailable` at individual trial
///   points even when `capability().hessian == Analytic`; `opt` degrades that
///   step to first-order behavior instead of requiring the objective to fake a
///   stale or non-finite Hessian.
/// - Use `eval_cost()` / `OuterEval::infeasible()` for infeasible trial points.
///   Return `Err(...)` for genuine evaluation breakdowns so the runner can mark
///   the step as a recoverable solver failure and escalate to the next declared
///   fallback plan if the full attempt still fails.
/// - `eval_cost()` is used only for cost-based optimization paths.
/// - `eval()` is the main evaluation path (cost + gradient + optional Hessian).
/// - `eval_efs()` is used only by the EFS solver. It runs the inner solve,
///   builds the `InnerSolution`, and computes the EFS step vector. The default
///   implementation returns an error; only objectives that support EFS need
///   to override it.
/// - `reset()` restores state to a clean baseline (for multi-start).
pub trait OuterObjective {
    /// Declare what this objective can compute analytically.
    fn capability(&self) -> OuterCapability;

    /// Evaluate cost only for cost-based optimization paths.
    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError>;

    /// Evaluate the seed-screening ranking proxy at this `rho`.
    ///
    /// Used exclusively by the `rank_seeds_with_screening` cascade. The
    /// default delegates to [`OuterObjective::eval_cost`], which preserves
    /// behavior for non-REML objectives.
    ///
    /// Concrete REML-state objectives override this to return the per-seed
    /// minimum penalized deviance observed during the inner P-IRLS solve
    /// (a monotonically descending quantity that remains a meaningful
    /// quality signal even at a 3-iteration screening cap), instead of the
    /// V_LAML criterion (which is dominated by a poorly-conditioned
    /// `0.5·log|H|` term at partial-fit β̂ and ranks seeds little better
    /// than random). The proxy fires *only* in screening mode; outside
    /// screening it must return the regular V_LAML cost so the optimization
    /// objective is unchanged.
    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.eval_cost(rho)
    }

    /// Evaluate cost + gradient + (if capable) Hessian.
    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError>;

    /// Evaluate the outer objective at the order requested by the active plan.
    ///
    /// The default preserves legacy behavior by delegating to
    /// [`OuterObjective::eval`].
    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        match order {
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                self.eval(rho)
            }
        }
    }

    /// Evaluate cost + EFS step vector. Only needed when the plan selects
    /// `Solver::Efs`. The default returns an error indicating EFS is not
    /// supported by this objective.
    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        Err(EstimationError::RemlOptimizationFailed(format!(
            "EFS evaluation not implemented for this objective at rho_dim={}",
            rho.len()
        )))
    }

    /// Restore to a clean baseline for the next multi-start candidate.
    fn reset(&mut self);

    /// Seed the inner-solver iterate before the first eval, e.g. when the
    /// outer-iterate cache restored a `(ρ, β)` pair from a prior run.
    ///
    /// Objectives must make an explicit choice here. Implementations with an
    /// inner β slot install the seed; implementations without one return a
    /// contract error instead of silently degrading a β-bearing checkpoint into
    /// a ρ-only resume.
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<(), EstimationError>;
}

// ─── Persistent warm-start checkpoint plumbing ────────────────────────
//
// `CheckpointingObjective` wraps any `OuterObjective` to write a copy of
// `(rho, cost, eval_id)` to disk on each finite evaluation. The on-disk
// [`crate::cache::Session`] rate-limits writes (≥2 s gap unless this iterate
// strictly improves on the best-so-far) so a tight inner loop never thrashes
// the filesystem. The same checkpoint is also broadcast to optional mirror
// sessions, which lets interrupted exact-key runs seed later related fits via
// their prefix key instead of waiting for a final converged write.

#[derive(serde::Serialize, serde::Deserialize)]
struct IteratePayload {
    /// Bump on incompatible payload changes; decode rejects mismatches.
    schema: u32,
    rho: Vec<f64>,
    /// Inner-solver iterate (PIRLS β) captured alongside ρ. The (ρ, β)
    /// pair lives on the implicit-function manifold β = β*(ρ); restoring
    /// ρ alone forces the next inner solve to reconstruct β from scratch.
    /// For saturated ρ (|ρ_i| near `rho_bound`) the inner Hessian
    /// `X'WX + Σ λ_i S_i` has condition number `≈ e^{2·rho_bound}` — Newton
    /// degrades to O(1/k) descent and the cycle budget exhausts before
    /// KKT. Caching β lets the resume start in Newton's quadratic basin
    /// regardless of where ρ lives. Empty when the family did not surface
    /// an inner-β hint at write time (still useful as a ρ-only seed).
    #[serde(default)]
    beta: Vec<f64>,
    cost: f64,
    eval_id: u64,
}

/// Bumped 1 → 2 (2026-05-21): schema 2 entries carry β alongside ρ.
/// Pre-2 entries — produced by binaries that stored ρ-only — decode to
/// `None` here. This orphans poisoned v1 checkpoints whose resume value
/// at boundary ρ depended on cold β = 0 and could not be re-warmed,
/// which manifested as inner-PIRLS budget exhaustion on subsequent runs.
const ITERATE_PAYLOAD_SCHEMA: u32 = 2;

fn encode_iterate(
    rho: &Array1<f64>,
    beta: Option<&Array1<f64>>,
    cost: f64,
    eval_id: u64,
) -> Option<Vec<u8>> {
    let p = IteratePayload {
        schema: ITERATE_PAYLOAD_SCHEMA,
        rho: rho.to_vec(),
        beta: beta.map(|b| b.to_vec()).unwrap_or_default(),
        cost,
        eval_id,
    };
    serde_json::to_vec(&p).ok()
}

fn decode_iterate(bytes: &[u8], expected_rho_dim: usize) -> Option<IteratePayload> {
    let p: IteratePayload = serde_json::from_slice(bytes).ok()?;
    if p.schema != ITERATE_PAYLOAD_SCHEMA {
        return None;
    }
    if p.rho.len() != expected_rho_dim {
        return None;
    }
    if !p.rho.iter().all(|x| x.is_finite()) || !p.cost.is_finite() {
        return None;
    }
    if !p.beta.iter().all(|x| x.is_finite()) {
        return None;
    }
    Some(p)
}

/// Outcome of inspecting a cache entry as a seed for the outer optimizer.
///
/// The classifier rejects only entries that fail structural validity
/// (wrong dimension, non-finite payload). It does NOT reshape ρ based on
/// saturation: every finite, well-shaped entry is honored as the next
/// run's seed.
///
/// Previously this enum carried `saturated_coords` / `clamped_to` /
/// "all-coords-saturated-poisoned-entry" branches that pulled boundary
/// ρ inward or discarded fully-saturated entries. Those were read-side
/// band-aids over the real bug: the warm-start contract stored ρ but
/// not β, so resuming at boundary ρ forced PIRLS to recompute β from
/// cold-start against a Hessian with condition number `≈ e^{2·rho_bound}`,
/// and Newton degraded to O(1/k) descent that exhausted the cycle budget.
///
/// The contract is now `(ρ, β)`: the schema-2 iterate payload carries
/// both, and [`CheckpointingObjective`] refuses to persist a divergent
/// inner state (non-finite cost or β). Boundary ρ — when written under
/// the new invariant — is a *legitimate* finding (the smoothness wants
/// to be near-null), and the cached β puts the next inner solve at the
/// previously converged iterate where the gradient is already at zero.
/// No clamp or shape-based discard is needed.
#[derive(Debug)]
pub(crate) enum CacheSeedDecision {
    ExactFinal {
        rho: Array1<f64>,
        /// Optional inner β captured at the converged ρ. Empty when the
        /// payload didn't carry one (legacy ρ-only writes or families
        /// that don't surface β).
        beta: Vec<f64>,
        final_value: f64,
        iterations: usize,
        prior_obj_display: f64,
    },
    Seed {
        rho: Array1<f64>,
        /// Optional inner β to prime the next run's inner solver via
        /// [`OuterObjective::seed_inner_state`]. When non-empty, the
        /// dispatcher injects β before the first eval so the inner
        /// PIRLS opens at zero-gradient regardless of where ρ sits in
        /// the box.
        beta: Vec<f64>,
        prior_obj_display: f64,
        iteration: u64,
    },
    Discard {
        reason: &'static str,
        prior_obj_display: f64,
        all_rho_finite: Option<bool>,
    },
}

pub(crate) fn classify_cache_entry_for_outer(
    loaded: &crate::cache::LoadedEntry,
    expected_rho_dim: usize,
) -> CacheSeedDecision {
    let entry = &loaded.entry;
    let Some(payload) = decode_iterate(&entry.payload, expected_rho_dim) else {
        return CacheSeedDecision::Discard {
            reason: "payload-shape-mismatch",
            prior_obj_display: entry.objective.unwrap_or(f64::NAN),
            all_rho_finite: None,
        };
    };
    let cached_rho = Array1::from_vec(payload.rho);
    let prior_obj_display = entry.objective.unwrap_or(f64::NAN);
    if matches!(entry.objective, Some(v) if !v.is_finite()) {
        return CacheSeedDecision::Discard {
            reason: "non-finite-payload",
            prior_obj_display,
            all_rho_finite: Some(cached_rho.iter().all(|v| v.is_finite())),
        };
    }
    if !cached_rho.iter().all(|v| v.is_finite()) {
        return CacheSeedDecision::Discard {
            reason: "non-finite-payload",
            prior_obj_display,
            all_rho_finite: Some(false),
        };
    }
    if loaded.source == LoadSource::Exact && entry.kind == crate::cache::EntryKind::Final {
        return CacheSeedDecision::ExactFinal {
            rho: cached_rho,
            beta: payload.beta,
            final_value: entry.objective.unwrap_or(payload.cost),
            iterations: entry
                .iteration
                .unwrap_or(payload.eval_id)
                .min(usize::MAX as u64) as usize,
            prior_obj_display,
        };
    }
    CacheSeedDecision::Seed {
        rho: cached_rho,
        beta: payload.beta,
        prior_obj_display,
        iteration: entry.iteration.unwrap_or(payload.eval_id),
    }
}

pub(crate) fn cache_entry_would_help_outer(
    loaded: &crate::cache::LoadedEntry,
    expected_rho_dim: usize,
) -> bool {
    matches!(
        classify_cache_entry_for_outer(loaded, expected_rho_dim),
        CacheSeedDecision::ExactFinal { .. } | CacheSeedDecision::Seed { .. }
    )
}

struct CheckpointingObjective<'a> {
    inner: &'a mut dyn OuterObjective,
    session: Arc<CacheSession>,
    mirror_sessions: Vec<Arc<CacheSession>>,
    eval_counter: AtomicU64,
    /// Most-recent inner β surfaced via [`OuterEval::inner_beta_hint`]. The
    /// finalize path reads this so the `kind: Final` write encodes the
    /// (ρ, β) pair that the BFGS optimum was actually fitted at — without
    /// this the finalize would clobber per-eval checkpoint β state with a
    /// ρ-only payload, reintroducing the cold-β resume failure.
    last_inner_beta: std::sync::Mutex<Option<Array1<f64>>>,
}

impl<'a> CheckpointingObjective<'a> {
    fn new(
        inner: &'a mut dyn OuterObjective,
        session: Arc<CacheSession>,
        mirror_sessions: Vec<Arc<CacheSession>>,
    ) -> Self {
        Self {
            inner,
            session,
            mirror_sessions,
            eval_counter: AtomicU64::new(0),
            last_inner_beta: std::sync::Mutex::new(None),
        }
    }

    fn last_inner_beta(&self) -> Option<Array1<f64>> {
        self.last_inner_beta.lock().ok().and_then(|g| g.clone())
    }

    fn note(&self, rho: &Array1<f64>, beta: Option<&Array1<f64>>, cost: f64) {
        if !cost.is_finite() {
            return;
        }
        // If β is provided, require it to be finite; non-finite β is a
        // divergent inner state — persisting it would re-poison the cache.
        if let Some(b) = beta {
            if !b.iter().all(|v| v.is_finite()) {
                return;
            }
            if let Ok(mut guard) = self.last_inner_beta.lock() {
                *guard = Some(b.clone());
            }
        }
        let i = self.eval_counter.fetch_add(1, Ordering::Relaxed);
        if let Some(bytes) = encode_iterate(rho, beta, cost, i) {
            self.session.checkpoint(&bytes, Some(cost), Some(i));
            for mirror in &self.mirror_sessions {
                mirror.checkpoint(&bytes, Some(cost), Some(i));
            }
        }
    }
}

impl<'a> OuterObjective for CheckpointingObjective<'a> {
    fn capability(&self) -> OuterCapability {
        self.inner.capability()
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let v = self.inner.eval_cost(rho)?;
        // `eval_cost` carries no inner-β handle — persist ρ-only.
        self.note(rho, None, v);
        Ok(v)
    }

    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        // Screening proxies run at sub-converged β̂ and aren't a meaningful
        // best-so-far signal; forward without persisting.
        self.inner.eval_screening_proxy(rho)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let r = self.inner.eval(rho)?;
        self.note(rho, r.inner_beta_hint.as_ref(), r.cost);
        Ok(r)
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        let r = self.inner.eval_with_order(rho, order)?;
        self.note(rho, r.inner_beta_hint.as_ref(), r.cost);
        Ok(r)
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        let r = self.inner.eval_efs(rho)?;
        // EfsEval has no inner-β hint surface yet — persist ρ-only.
        self.note(rho, None, r.cost);
        Ok(r)
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<(), EstimationError> {
        // Forward to the wrapped objective, then prime our last-inner-beta
        // cache so a subsequent finalize-write encodes the seeded β if no
        // eval surfaces a fresher β first.
        let result = self.inner.seed_inner_state(beta);
        if result.is_ok()
            && beta.iter().all(|v| v.is_finite())
            && let Ok(mut guard) = self.last_inner_beta.lock()
        {
            *guard = Some(beta.clone());
        }
        result
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Closure-based adapter for [`OuterObjective`].
///
/// This allows any call site to construct an `OuterObjective` from closures
/// without needing to define a wrapper struct or modify the state type.
/// Each call site wraps its existing methods into closures and passes them here.
pub struct ClosureObjective<
    S,
    Fc,
    Fe,
    Fr = fn(&mut S),
    Fefs = fn(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    Feo = fn(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
    Fsp = fn(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
> {
    pub(crate) state: S,
    pub(crate) cap: OuterCapability,
    pub(crate) cost_fn: Fc,
    pub(crate) eval_fn: Fe,
    /// Optional order-aware eval closure. When `None`, `eval_with_order()`
    /// falls back to `eval()`.
    pub(crate) eval_order_fn: Option<Feo>,
    /// Optional reset closure. When `None`, `reset()` is a no-op.
    pub(crate) reset_fn: Option<Fr>,
    /// Optional EFS evaluation closure. When `None`, the default
    /// `OuterObjective::eval_efs` returns an error.
    pub(crate) efs_fn: Option<Fefs>,
    /// Optional seed-screening ranking proxy closure. When `None`,
    /// `eval_screening_proxy()` falls back to `eval_cost()` (the trait
    /// default), preserving legacy behavior for non-REML objectives.
    pub(crate) screening_proxy_fn: Option<Fsp>,
}

impl<S, Fc, Fe, Fr, Fefs, Feo, Fsp> OuterObjective
    for ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp>
where
    Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
    Fr: FnMut(&mut S),
    Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
    Fsp: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
{
    fn capability(&self) -> OuterCapability {
        self.cap.clone()
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        crate::solver::estimate::reml::runtime::record_current_outer_theta_for_ift(rho);
        (self.cost_fn)(&mut self.state, rho)
    }

    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        crate::solver::estimate::reml::runtime::record_current_outer_theta_for_ift(rho);
        match self.screening_proxy_fn.as_mut() {
            Some(f) => f(&mut self.state, rho),
            None => (self.cost_fn)(&mut self.state, rho),
        }
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        crate::solver::estimate::reml::runtime::record_current_outer_theta_for_ift(rho);
        (self.eval_fn)(&mut self.state, rho)
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        crate::solver::estimate::reml::runtime::record_current_outer_theta_for_ift(rho);
        match self.eval_order_fn.as_mut() {
            Some(f) => f(&mut self.state, rho, order),
            None => (self.eval_fn)(&mut self.state, rho),
        }
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        crate::solver::estimate::reml::runtime::record_current_outer_theta_for_ift(rho);
        match self.efs_fn.as_mut() {
            Some(f) => f(&mut self.state, rho),
            None => Err(EstimationError::RemlOptimizationFailed(
                "EFS evaluation not implemented for this objective".to_string(),
            )),
        }
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<(), EstimationError> {
        if beta.is_empty() {
            Ok(())
        } else {
            Err(EstimationError::InvalidInput(format!(
                "cached inner beta has length {}, but this objective does not expose an inner-state seeding hook",
                beta.len()
            )))
        }
    }

    fn reset(&mut self) {
        if let Some(f) = self.reset_fn.as_mut() {
            f(&mut self.state);
        }
    }
}

fn into_objective_error(context: &str, err: EstimationError) -> ObjectiveEvalError {
    ObjectiveEvalError::recoverable(format!("{context}: {err}"))
}

fn finite_cost_or_error(context: &str, cost: f64) -> Result<f64, ObjectiveEvalError> {
    if cost.is_finite() {
        Ok(cost)
    } else {
        Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite cost"
        )))
    }
}

fn finite_outer_eval_or_error(
    context: &str,
    layout: OuterThetaLayout,
    eval: OuterEval,
) -> Result<OuterEval, ObjectiveEvalError> {
    layout.validate_gradient_len(&eval.gradient, context)?;
    if !eval.cost.is_finite() {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite cost"
        )));
    }
    if !eval.gradient.iter().all(|v| v.is_finite()) {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite gradient"
        )));
    }
    match &eval.hessian {
        HessianResult::Analytic(hessian) => {
            layout.validate_hessian_shape(hessian, context)?;
            if !hessian.iter().all(|v| v.is_finite()) {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: objective returned a non-finite Hessian"
                )));
            }
        }
        HessianResult::Operator(op) => {
            if op.dim() != layout.n_params {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: outer Hessian operator dimension mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                    op.dim(),
                    layout.n_params,
                    layout.rho_dim(),
                    layout.psi_dim
                )));
            }
        }
        HessianResult::Unavailable => {}
    }
    Ok(eval)
}

fn finite_outer_first_order_eval_or_error(
    context: &str,
    layout: OuterThetaLayout,
    eval: OuterEval,
) -> Result<OuterEval, ObjectiveEvalError> {
    layout.validate_gradient_len(&eval.gradient, context)?;
    if !eval.cost.is_finite() {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite cost"
        )));
    }
    if !eval.gradient.iter().all(|v| v.is_finite()) {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite gradient"
        )));
    }
    Ok(eval)
}

fn validate_second_order_seed_hessian(
    context: &str,
    layout: OuterThetaLayout,
    eval: &OuterEval,
) -> Result<(), ObjectiveEvalError> {
    if layout.n_params > SECOND_ORDER_GEOMETRY_PROBE_MAX_PARAMS || !eval.hessian.is_analytic() {
        return Ok(());
    }
    if matches!(
        &eval.hessian,
        HessianResult::Operator(op) if !op.materialization_capability().is_available()
    ) {
        return Ok(());
    }

    let Some(hessian) = eval.hessian.materialize_dense().map_err(|message| {
        ObjectiveEvalError::recoverable(format!(
            "{context}: analytic outer Hessian materialization failed during second-order seed validation: {message}"
        ))
    })?
    else {
        return Ok(());
    };

    layout.validate_hessian_shape(&hessian, context)?;
    if !hessian.iter().all(|value| value.is_finite()) {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: analytic outer Hessian probe encountered non-finite entries"
        )));
    }

    Ok(())
}

struct OuterFirstOrderBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
    /// Outer-aware inner-PIRLS cap atomic. When `Some`, the bridge stores
    /// a coarsen-then-tighten cap into it on every accepted gradient eval
    /// (see `first_order_inner_cap_schedule`). The cap is NEVER touched
    /// in `eval_cost` so line-search probes within an outer iter see a
    /// stable inner tolerance — Wolfe conditions assume constant cost
    /// noise within a bracket.
    outer_inner_cap: Option<InnerProgressFeedback>,
    /// Counts gradient evaluations for logging only. Inner-PIRLS scheduling
    /// uses `InnerProgressFeedback.accepted_iter` so rejected line-search
    /// probes do not relax the inner work budget.
    iter_count: usize,
    /// First observed `‖g‖` from `eval_grad`. Used by the schedule to
    /// compute the gradient-ratio (`last / initial`) — when the ratio
    /// drops, the optimizer is approaching convergence and the inner
    /// cap should lift to full so the cached β is at full tolerance.
    g_norm_initial: Option<f64>,
    /// `‖g‖` from the most recent eval. Stale by one outer iter relative
    /// to the cap that consumes it (the cap is set BEFORE the new eval),
    /// but for monotone-decreasing g_norm this is safe — it makes the
    /// cap conservatively LARGER than the truly-needed value, never
    /// smaller.
    last_g_norm: Option<f64>,
}

impl ZerothOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Per-axis line-search step caps now live natively in opt::Bfgs
        // (`with_axis_step_caps`), which shortens the BFGS direction before
        // line search instead of poisoning the Wolfe bracket with a
        // sentinel cost. This entry point can therefore stay honest: any
        // call that lands here is a real line-search probe, not a too-far
        // attempt the bridge needs to swat away.
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let cost = self
            .obj
            .eval_cost(x)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        finite_cost_or_error("outer eval_cost failed", cost)
    }
}

impl FirstOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        // Drive the outer-aware inner-PIRLS cap from accepted outer
        // iterations, BEFORE invoking the inner solve. Cap stays fixed
        // within line-search cost probes (`eval_cost` never touches the
        // atomic). A cap of 0 means "no cap from this source"; the inner
        // solver still honors `pirls_max_iterations` and the screening cap.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let accepted_iter = feedback.accepted_iter.load(Ordering::Relaxed);
            let cap = first_order_inner_cap_schedule(accepted_iter, g_ratio, snapshot);
            let prev = feedback.cap.swap(cap, Ordering::Relaxed);
            if prev != cap {
                let ratio_str = match g_ratio {
                    Some(r) => format!("{:.3e}", r),
                    None => "n/a".to_string(),
                };
                let snap_str = match snapshot {
                    Some(s) => format!(
                        "last_iters={} converged={} ift_residual={} accept_rho={}",
                        s.last_iters,
                        s.last_converged,
                        match s.last_ift_residual {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
                        match s.last_accept_rho {
                            Some(r) => format!("{:.3}", r),
                            None => "n/a".to_string(),
                        },
                    ),
                    None => "no-history".to_string(),
                };
                log::info!(
                    "[OUTER schedule] inner-PIRLS cap transition accepted_iter={} eval_count={} g_ratio={} {} prev={} new={} ({})",
                    accepted_iter,
                    self.iter_count,
                    ratio_str,
                    snap_str,
                    prev,
                    cap,
                    if cap == 0 { "uncapped" } else { "capped" }
                );
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueAndGradient dim={} (first-order bridge, iter={})",
            x.len(),
            self.iter_count
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        let gradient = eval.gradient;
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        log::info!(
            "[STAGE] outer eval end order=ValueAndGradient elapsed={:.3}s cost={:.6e} |g|={:.3e} (first-order bridge, iter={})",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
            self.iter_count,
        );
        // Push the (cost, ‖g‖) sample so the live progress chart shows the
        // BFGS outer descent. Recorded as a trial; `OuterAcceptObserver`
        // promotes the latest trial into the accepted series when BFGS's
        // Wolfe line-search accepts the step. Cheap: throttled internally.
        crate::solver::visualizer::record_outer_eval(eval.cost, g_norm);
        self.iter_count = self.iter_count.saturating_add(1);
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient,
        })
    }
}

/// Adaptive inner-PIRLS cap schedule. Replaces the older hardcoded
/// iter-tier (3/5/10/20) and ratio-tier (0.50/0.20/0.05/0.01) schedule
/// with a cap driven by the inner solver's actual convergence behavior
/// — Eisenstat-Walker style for the inner Newton.
///
/// Inputs:
/// - `iter_count`: outer iter index, used only as a fallback when no
///   inner-progress feedback has arrived yet (first 1-2 outer iters).
/// - `g_ratio`: outer gradient-norm decay `‖g_now‖ / ‖g_initial‖`. When
///   this drops below 1% the outer is essentially converged; we lift
///   the cap fully so the cached β is at full inner tolerance and the
///   convergence guard does not have to re-pay a full inner solve.
/// - `last`: snapshot from `InnerProgressFeedback`. When present and
///   the previous solve converged, we set the cap to `last_iters + 2`
///   (a small margin in case ρ moved enough to need a couple more
///   iters); when the previous solve hit the cap, we double — a
///   geometric backoff that recovers from too-tight a cap without
///   thrashing.
///
/// A cap of 0 means "no cap from this source"; the inner solver still
/// honors `pirls_max_iterations` and the screening cap. The cap is
/// floored at 3 (anything less is below noise) and ceilinged at 64
/// (the inner noise floor at biobank scale; further iters would be
/// pure waste).
fn first_order_inner_cap_schedule(
    iter_count: usize,
    g_ratio: Option<f64>,
    last: Option<InnerProgressSnapshot>,
) -> usize {
    // Convergence override: when the outer is essentially converged the
    // cached β must be at full inner tolerance. This belt-and-suspenders
    // path is independent of inner-progress history because the outer
    // re-evaluation guard pays a full inner solve anyway — uncapping
    // here just avoids one wasted iter at low cap before the guard.
    if matches!(g_ratio, Some(r) if r < 0.01) {
        return 0;
    }

    // Adaptive path: drive the cap from the inner solver's prior iter
    // count rather than a hardcoded tier.
    if let Some(snap) = last {
        let next = if snap.last_converged {
            // Converged in `last_iters` last time; pick a small margin
            // for ρ-step variability. The IFT predictor's residual
            // tells us how close the warm-start was to the KKT point:
            //   residual < 0.01  → next solve starts essentially AT the
            //                      KKT β, so +1 iter of margin suffices.
            //   residual < 0.10  → +2 (default, current behavior).
            //   residual ≥ 0.10  → predictor was poor (or fell back to
            //                      flat); the inner Newton has more
            //                      recovery work, so +4 to be safe.
            //   None             → no signal yet → +2 (default).
            // This wires the [IFT-QUALITY] feedback directly into the
            // adaptive cap, replacing the previous fixed +2.
            let mut margin = match snap.last_ift_residual {
                Some(r) if r < 0.01 => 1usize,
                Some(r) if r >= 0.10 => 4usize,
                _ => 2usize,
            };
            // LM model fidelity (commit 6445c079): if the previous
            // solve's accepted gain ratio was poor (model overstating
            // predicted reduction), the inner Newton's quadratic model
            // is unreliable. Bump margin by +2 — even a fast-converged
            // previous iter (small `last_iters`) provides weaker
            // evidence about the next solve's required effort when the
            // model is mis-calibrated. Threshold 0.5 is the textbook
            // "good agreement" cutoff for trust-region gain ratios.
            if matches!(snap.last_accept_rho, Some(r) if r < 0.5) {
                margin = margin.saturating_add(2);
            }
            snap.last_iters.saturating_add(margin)
        } else {
            // Hit the cap. Geometric backoff so we don't thrash on a
            // marginally-too-tight cap, but enforce floor of
            // last_iters+4 to actually grow.
            //
            // LM-fidelity escalation: if the previous solve's accepted
            // gain ratio was VERY poor (`accept_rho < 0.3`), the LM
            // model is severely mis-calibrated — doubling the cap may
            // not give the inner Newton enough headroom to find a
            // usable trust radius. Triple instead of doubling so we
            // don't waste another cycle hitting the cap. The 0.3
            // threshold is tighter than the +2-margin trigger (0.5)
            // because here we ALREADY know the iter budget was
            // insufficient AND the model was poor — both signals
            // pointing the same way.
            let multiplier = if matches!(snap.last_accept_rho, Some(r) if r < 0.3) {
                3
            } else {
                2
            };
            snap.last_iters
                .saturating_mul(multiplier)
                .max(snap.last_iters.saturating_add(4))
        };
        return next.clamp(3, 64);
    }

    // No feedback yet (first outer iter, or right after a screening
    // bundle reset). Coarse iter-count fallback for the first 1-2
    // outer iters so the cold-start cap is shallow even before the
    // adaptive signal kicks in.
    match iter_count {
        0 => 3,
        1 => 5,
        _ => 10,
    }
}

#[cfg(test)]
mod outer_inner_cap_schedule_tests {
    use super::{InnerProgressSnapshot, first_order_inner_cap_schedule};

    fn snap(last_iters: usize, last_converged: bool) -> Option<InnerProgressSnapshot> {
        Some(InnerProgressSnapshot {
            last_iters,
            last_converged,
            last_ift_residual: None,
            last_accept_rho: None,
        })
    }

    fn snap_with_accept_rho(
        last_iters: usize,
        last_converged: bool,
        accept_rho: f64,
    ) -> Option<InnerProgressSnapshot> {
        Some(InnerProgressSnapshot {
            last_iters,
            last_converged,
            last_ift_residual: None,
            last_accept_rho: Some(accept_rho),
        })
    }

    fn snap_with_residual(
        last_iters: usize,
        last_converged: bool,
        residual: f64,
    ) -> Option<InnerProgressSnapshot> {
        Some(InnerProgressSnapshot {
            last_iters,
            last_converged,
            last_ift_residual: Some(residual),
            last_accept_rho: None,
        })
    }

    /// The bridge's snapshot reader must distinguish "no signal yet"
    /// (NaN sentinel, encoded as `IFT_RESIDUAL_NO_SIGNAL_BITS`) from
    /// "residual was 0.0" (a real signal). Previously the bridge used
    /// `bits == 0` to detect no-signal, which collided with
    /// `f64::to_bits(0.0) == 0`. This test pins down the new
    /// NaN-sentinel discipline at the bridge layer.
    #[test]
    fn snapshot_distinguishes_zero_residual_from_no_signal() {
        use super::InnerProgressFeedback;
        use crate::solver::estimate::reml::runtime::IFT_RESIDUAL_NO_SIGNAL_BITS;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};

        // Helper to build a feedback channel with concrete values.
        let make_feedback =
            |iters: usize, converged: bool, residual_bits: u64| InnerProgressFeedback {
                cap: Arc::new(AtomicUsize::new(0)),
                accepted_iter: Arc::new(AtomicUsize::new(0)),
                last_iters: Arc::new(AtomicUsize::new(iters)),
                last_converged: Arc::new(AtomicBool::new(converged)),
                ift_residual: Arc::new(AtomicU64::new(residual_bits)),
                accept_rho: Arc::new(AtomicU64::new(
                    crate::solver::estimate::reml::runtime::IFT_RESIDUAL_NO_SIGNAL_BITS,
                )),
            };

        // Sentinel → no IFT signal (last_ift_residual = None).
        let fb = make_feedback(5, true, IFT_RESIDUAL_NO_SIGNAL_BITS);
        let snap = fb.snapshot().expect("iters > 0, snapshot present");
        assert!(
            snap.last_ift_residual.is_none(),
            "sentinel must decode to None"
        );

        // 0.0 residual → genuine signal (last_ift_residual = Some(0.0)).
        // This is the bug: previously the reader treated `bits == 0` as
        // no-signal, dropping the genuine 0.0 residual.
        let fb = make_feedback(5, true, 0.0_f64.to_bits());
        let snap = fb.snapshot().expect("iters > 0, snapshot present");
        assert_eq!(
            snap.last_ift_residual,
            Some(0.0),
            "residual of exactly 0.0 must round-trip as a real signal, \
             not be confused with the no-signal sentinel",
        );

        // Modest finite residual round-trips.
        let fb = make_feedback(5, true, 0.05_f64.to_bits());
        let snap = fb.snapshot().expect("snapshot present");
        assert_eq!(snap.last_ift_residual, Some(0.05));

        // last_iters == 0 → entire snapshot is None (no inner-Newton
        // signal yet at all). Sentinel residual irrelevant.
        let fb = make_feedback(0, false, IFT_RESIDUAL_NO_SIGNAL_BITS);
        assert!(fb.snapshot().is_none());
    }

    #[test]
    fn schedule_falls_back_to_iter_tier_without_feedback() {
        // No inner-progress history yet → coarse iter-count fallback so
        // the cold-start cap is shallow even before the adaptive signal
        // arrives.
        assert_eq!(first_order_inner_cap_schedule(0, None, None), 3);
        assert_eq!(first_order_inner_cap_schedule(1, None, None), 5);
        assert_eq!(first_order_inner_cap_schedule(2, None, None), 10);
        assert_eq!(first_order_inner_cap_schedule(20, None, None), 10);
    }

    #[test]
    fn schedule_uses_last_iters_plus_margin_when_converged() {
        // Inner converged in 4 iters last time → cap = 4+2 = 6.
        assert_eq!(first_order_inner_cap_schedule(2, None, snap(4, true)), 6);
        // Inner converged in 12 → cap = 14.
        assert_eq!(first_order_inner_cap_schedule(5, None, snap(12, true)), 14);
    }

    #[test]
    fn schedule_geometric_backoff_when_last_hit_cap() {
        // Last hit cap at 5 → 2*5=10, max(10, 5+4=9) = 10.
        assert_eq!(first_order_inner_cap_schedule(2, None, snap(5, false)), 10);
        // Last hit cap at 1 → 2*1=2, max(2, 1+4=5) = 5.
        assert_eq!(first_order_inner_cap_schedule(2, None, snap(1, false)), 5);
        // Last hit cap at 30 → would be 60 but ceiling is 64, so 60.
        assert_eq!(first_order_inner_cap_schedule(2, None, snap(30, false)), 60);
    }

    #[test]
    fn schedule_clamps_floor_and_ceiling() {
        // Last converged in 0 (degenerate; should never happen because
        // the producer only writes nonzero, but defensively check the
        // floor of 3).
        assert_eq!(first_order_inner_cap_schedule(2, None, snap(0, true)), 3);
        // Last converged in 100 → ceiling 64.
        assert_eq!(first_order_inner_cap_schedule(2, None, snap(100, true)), 64);
    }

    #[test]
    fn schedule_uncaps_when_outer_converged() {
        // g_ratio < 1% trumps everything: cached β must be at full
        // inner tolerance for the convergence guard.
        assert_eq!(first_order_inner_cap_schedule(0, Some(0.0001), None), 0);
        assert_eq!(
            first_order_inner_cap_schedule(0, Some(0.005), snap(4, true)),
            0
        );
        assert_eq!(
            first_order_inner_cap_schedule(20, Some(0.001), snap(50, false)),
            0
        );
    }

    #[test]
    fn schedule_ignores_modest_g_ratio_decay() {
        // Old schedule had tiered ratio caps at 0.50/0.20/0.05; the new
        // schedule only special-cases the deep-convergence threshold
        // (<1%). Modest decay no longer overrides the adaptive cap.
        assert_eq!(
            first_order_inner_cap_schedule(2, Some(0.30), snap(4, true)),
            6
        );
        assert_eq!(
            first_order_inner_cap_schedule(2, Some(0.05), snap(4, true)),
            6
        );
    }

    #[test]
    fn schedule_uses_ift_residual_to_pick_margin() {
        // Excellent IFT prediction (residual < 0.01): warm-start lands
        // essentially AT the KKT β, so +1 of margin suffices.
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.005)),
            5
        );
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.0001)),
            5
        );
        // Default zone (0.01 ≤ residual < 0.10): +2, current behavior.
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.05)),
            6
        );
        // Poor IFT prediction (residual ≥ 0.10): +4, the inner Newton
        // has more recovery work after a near-flat warm-start.
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.20)),
            8
        );
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.80)),
            8
        );
        // Margin policy is monotone non-decreasing in residual: a worse
        // predictor never produces a tighter cap than a better one.
        let residuals = [0.001, 0.05, 0.30];
        let caps: Vec<usize> = residuals
            .iter()
            .map(|&r| first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, r)))
            .collect();
        for w in caps.windows(2) {
            assert!(
                w[0] <= w[1],
                "ift-residual margin policy regressed monotonicity: {caps:?}"
            );
        }
    }

    #[test]
    fn schedule_bumps_margin_on_poor_lm_accept_rho() {
        // Healthy LM model fidelity (accept_rho ≥ 0.5): margin
        // unchanged from the no-accept-rho baseline (+2 default).
        // last_iters=4, default margin=2 → cap=6.
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.95)),
            6
        );
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.5)),
            6
        );
        // Poor LM model fidelity (accept_rho < 0.5): +2 margin bump
        // beyond the IFT-residual base. last_iters=4, default base=2,
        // accept_rho<0.5 bump=+2 → margin=4 → cap=8.
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.4)),
            8
        );
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.1)),
            8
        );
        // accept_rho saturation guard: `r < 0.5` is the strict
        // textbook "good agreement" cutoff for trust-region gain
        // ratios. Boundary at 0.5 admits, just below 0.5 bumps.
        assert_eq!(
            first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.49)),
            8
        );
    }

    #[test]
    fn schedule_escalates_geometric_backoff_on_very_poor_accept_rho() {
        // Cap-hit (last_converged=false) with VERY poor LM model
        // (accept_rho < 0.3): triple instead of double the cap, so the
        // next solve has materially more iter budget when the model is
        // both insufficient (cap-hit) AND mis-calibrated (poor rho).
        // last_iters=4 → 4*3 = 12, vs 4*2=8 with doubling.
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: false,
            last_ift_residual: None,
            last_accept_rho: Some(0.15),
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 12);
        // Cap-hit with moderately-poor accept_rho (0.3 ≤ r < 0.5):
        // standard doubling. The threshold for escalation is 0.3, not
        // 0.5, because the +2-margin path (commit 04b30163) already
        // covers the 0.3-0.5 case for the converged branch.
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: false,
            last_ift_residual: None,
            last_accept_rho: Some(0.4),
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 8);
        // Cap-hit with healthy accept_rho ≥ 0.5: standard doubling.
        // The previous solve hit the cap because it needed more iters,
        // not because the LM was mis-calibrated.
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: false,
            last_ift_residual: None,
            last_accept_rho: Some(0.9),
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 8);
        // Cap-hit with no accept_rho signal: standard doubling. No
        // escalation when we don't have evidence of LM trouble.
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: false,
            last_ift_residual: None,
            last_accept_rho: None,
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 8);
        // Boundary at exactly 0.3: NOT escalated (`< 0.3` is strict).
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: false,
            last_ift_residual: None,
            last_accept_rho: Some(0.3),
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 8);
    }

    #[test]
    fn schedule_skips_lm_accept_rho_bump_when_signal_absent() {
        // None for last_accept_rho means "no signal yet" — the schedule
        // must NOT bump the margin in that case (otherwise a fresh
        // surface with the NaN sentinel would get penalty cap inflation
        // for no reason). last_iters=4, last_ift_residual=None →
        // default base margin=2 → cap=6, regardless of accept_rho being
        // unset.
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: true,
            last_ift_residual: None,
            last_accept_rho: None,
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 6);
        // Regression-lock the boundary: accept_rho exactly at 0.5
        // (textbook good-agreement cutoff) does NOT bump (`< 0.5` is
        // strict). cap = 4 + 2 = 6.
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: true,
            last_ift_residual: None,
            last_accept_rho: Some(0.5),
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 6);
        // accept_rho = 1.0 is the textbook "perfect agreement" — never
        // bumps. cap = 4 + 2 = 6.
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: true,
            last_ift_residual: None,
            last_accept_rho: Some(1.0),
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 6);
    }

    #[test]
    fn schedule_combines_ift_residual_and_lm_accept_rho() {
        // When BOTH signals fire (poor IFT prediction AND poor LM
        // accept_rho), the bumps compose: IFT base = 4, accept_rho
        // bump = +2 → total margin = 6, cap = last_iters + 6.
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: true,
            last_ift_residual: Some(0.30),
            last_accept_rho: Some(0.20),
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 10);
        // When only LM accept_rho is poor (IFT residual is excellent),
        // the bumps still compose: IFT base = 1 (excellent), accept_rho
        // bump = +2 → margin = 3, cap = 4 + 3 = 7.
        let snap = Some(InnerProgressSnapshot {
            last_iters: 4,
            last_converged: true,
            last_ift_residual: Some(0.005),
            last_accept_rho: Some(0.30),
        });
        assert_eq!(first_order_inner_cap_schedule(2, None, snap), 7);
    }
}

struct OuterSecondOrderBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
    hessian_source: HessianSource,
    /// When the evaluator returns `HessianResult::Operator(op)` and the
    /// operator advertises an exact dense route, the bridge may materialize the
    /// operator into a dense K×K matrix so the dense ARC path can run an exact
    /// factorization instead of operator-CG.
    materialize_operator_max_dim: usize,
    /// Counts gradient/Hessian evaluations so that progress is visible even
    /// when the upstream `opt` solver does not emit per-iteration logs of its
    /// own. Emitted at INFO from `eval_grad` and `eval_hessian` (the calls
    /// that gate one optimizer step); skipped on `eval_cost` so linesearch
    /// trial points do not flood the log. Also drives the outer-aware
    /// inner-PIRLS cap schedule (see `first_order_inner_cap_schedule`).
    eval_count: usize,
    /// Outer-aware inner-PIRLS cap atomic. When `Some`, the bridge stores
    /// a coarsen-then-tighten cap into it on every accepted eval_grad /
    /// eval_hessian call. Mirrors the BFGS-side wiring in
    /// `OuterFirstOrderBridge`. Cap is NEVER touched in `eval_cost` so
    /// line-search probes within an outer iter see a stable inner
    /// tolerance (Wolfe / trust-region acceptance both assume constant
    /// cost noise within a bracket).
    outer_inner_cap: Option<InnerProgressFeedback>,
    /// First observed `‖g‖` from `eval_grad`/`eval_hessian`. Used by the
    /// schedule's gradient-ratio gate so the cap lifts when the optimizer
    /// is approaching convergence, not just when iter count says so.
    g_norm_initial: Option<f64>,
    /// `‖g‖` from the most recent eval. See `OuterFirstOrderBridge` for
    /// the staleness rationale: monotone-decreasing g_norm means the cap
    /// is conservatively LARGER than truly needed, never smaller.
    last_g_norm: Option<f64>,
}

impl ZerothOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let cost = self
            .obj
            .eval_cost(x)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        finite_cost_or_error("outer eval_cost failed", cost)
    }
}

impl FirstOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            // The ARC bridge increments `eval_count` in BOTH `eval_grad` and
            // `eval_hessian`. ARC calls both per outer iter, so `eval_count
            // / 2` is the correct iter index for the schedule. Without this
            // divisor the schedule would lift to full inner-cap at ARC iter
            // 3 instead of iter 6.
            // Use the observer-fed accepted-iter counter (opt 0.5.0
            // OptimizerObserver) instead of `eval_count / 2`; the
            // observer increments only on rho-accepted steps, so the
            // schedule no longer relaxes the cap on rejected trials.
            let arc_iter = feedback.accepted_iter.load(Ordering::Relaxed);
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let cap = first_order_inner_cap_schedule(arc_iter, g_ratio, snapshot);
            let prev = feedback.cap.swap(cap, Ordering::Relaxed);
            if prev != cap {
                let ratio_str = match g_ratio {
                    Some(r) => format!("{:.3e}", r),
                    None => "n/a".to_string(),
                };
                let snap_str = match snapshot {
                    Some(s) => format!(
                        "last_iters={} converged={} ift_residual={} accept_rho={}",
                        s.last_iters,
                        s.last_converged,
                        match s.last_ift_residual {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
                        match s.last_accept_rho {
                            Some(r) => format!("{:.3}", r),
                            None => "n/a".to_string(),
                        },
                    ),
                    None => "no-history".to_string(),
                };
                log::info!(
                    "[OUTER schedule] inner-PIRLS cap transition (ARC bridge) arc_iter={} g_ratio={} {} prev={} new={} ({})",
                    arc_iter,
                    ratio_str,
                    snap_str,
                    prev,
                    cap,
                    if cap == 0 { "uncapped" } else { "capped" }
                );
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueAndGradient dim={}",
            x.len()
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        self.eval_count += 1;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        log::info!(
            "[STAGE] outer eval end order=ValueAndGradient elapsed={:.3}s cost={:.6e} |g|={:.3e}",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
        );
        log::info!(
            "[OUTER] eval#{n} (grad) cost={cost:.6e} |g|={gnorm:.3e} rho=[{rho}]",
            n = self.eval_count,
            cost = eval.cost,
            gnorm = g_norm,
            rho = x
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        // Live-chart trial sample (ARC bridge first-order entry). Mirrors
        // the eval_hessian site below; both run once per outer iter, so the
        // chart's x-coord progresses on every accepted-or-rejected eval and
        // the accepted line moves only on rho-acceptance.
        crate::solver::visualizer::record_outer_eval(eval.cost, g_norm);
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

impl SecondOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            // Use the observer-fed accepted-iter counter (opt 0.5.0
            // OptimizerObserver) instead of `eval_count / 2`; the
            // observer increments only on rho-accepted steps, so the
            // schedule no longer relaxes the cap on rejected trials.
            let arc_iter = feedback.accepted_iter.load(Ordering::Relaxed);
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let cap = first_order_inner_cap_schedule(arc_iter, g_ratio, snapshot);
            let prev = feedback.cap.swap(cap, Ordering::Relaxed);
            if prev != cap {
                let ratio_str = match g_ratio {
                    Some(r) => format!("{:.3e}", r),
                    None => "n/a".to_string(),
                };
                let snap_str = match snapshot {
                    Some(s) => format!(
                        "last_iters={} converged={} ift_residual={} accept_rho={}",
                        s.last_iters,
                        s.last_converged,
                        match s.last_ift_residual {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
                        match s.last_accept_rho {
                            Some(r) => format!("{:.3}", r),
                            None => "n/a".to_string(),
                        },
                    ),
                    None => "no-history".to_string(),
                };
                log::info!(
                    "[OUTER schedule] inner-PIRLS cap transition (ARC bridge) arc_iter={} g_ratio={} {} prev={} new={} ({})",
                    arc_iter,
                    ratio_str,
                    snap_str,
                    prev,
                    cap,
                    if cap == 0 { "uncapped" } else { "capped" }
                );
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueGradientHessian dim={}",
            x.len()
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueGradientHessian)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_eval_or_error("outer eval failed", self.layout, eval)?;
        self.eval_count += 1;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        log::info!(
            "[STAGE] outer eval end order=ValueGradientHessian elapsed={:.3}s cost={:.6e} |g|={:.3e}",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
        );
        log::info!(
            "[OUTER] eval#{n} (hess) cost={cost:.6e} |g|={gnorm:.3e} rho=[{rho}]",
            n = self.eval_count,
            cost = eval.cost,
            gnorm = g_norm,
            rho = x
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        let hessian = build_bridge_hessian_for_source(
            self.hessian_source,
            eval.hessian,
            self.materialize_operator_max_dim,
        )?;
        Ok(SecondOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
            hessian,
        })
    }
}

// =====================================================================
// opt 0.4 matrix-free TR adapter (Phase 6)
// =====================================================================
//
// `OuterToOptHessianOperator` wraps gam's `OuterHessianOperator` so it
// can be passed to `opt::MatrixFreeTrustRegion` via
// `opt::HessianValue::Operator`. The two traits have nearly identical
// surfaces — the adapter is just shape/error translation:
//
//   gam::OuterHessianOperator              opt::HessianOperator
//     dim()                       <-->       dim()
//     matvec(v) -> Array1         <-->       apply_into(v, &mut out)
//     mul_mat(X) -> Array2        <-->       apply_mat(X)
//     materialization_capability  <-->       materialization
//     materialize_dense           <-->       materialize_dense
//
// gam errors are `String`; opt errors are `ObjectiveEvalError`. We
// promote everything to `ObjectiveEvalError::Fatal` because operator
// failures inside a solver step are not generally recoverable —
// shrinking the trust radius would not fix a dimension mismatch.
//
// `OuterOperatorBridge` is the bridge that implements
// `opt::OperatorObjective` for `gam`'s outer objective — parallel to
// `OuterSecondOrderBridge` but produces `OperatorSample` whose
// Hessian is `HessianValue::Operator(_)` (or `Dense(_)` when the
// operator declares an exact materialization route).

/// `opt::OptimizerObserver` that increments
/// `InnerProgressFeedback.accepted_iter` on every accepted outer
/// step. Replaces the bridge-side `eval_count / 2` heuristic on
/// routes that see trial-and-rejection probing (ARC dense,
/// matrix-free TR). The bridge's inner-cap schedule reads
/// `accepted_iter` from the feedback channel instead of inferring
/// it from raw eval counts.
struct OuterAcceptObserver {
    feedback: InnerProgressFeedback,
}

impl OptimizerObserver for OuterAcceptObserver {
    fn on_step_accepted(&mut self, info: &StepInfo) {
        log::trace!(
            "outer step accepted iter={} step_norm={:.3e} predicted_decrease={:.3e} actual_decrease={:.3e}",
            info.iter,
            info.step_norm,
            info.predicted_decrease,
            info.actual_decrease,
        );
        self.feedback.accepted_iter.fetch_add(1, Ordering::Relaxed);
    }
}

struct OuterToOptHessianOperator(Arc<dyn OuterHessianOperator>);

impl HessianOperator for OuterToOptHessianOperator {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn apply_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), ObjectiveEvalError> {
        // Forward to gam's `OuterHessianOperator::apply_into` (default
        // impl wraps `matvec`; backends with a native into-buffer
        // kernel override for true zero-alloc CG iterations).
        self.0
            .apply_into(v, out)
            .map_err(|message| ObjectiveEvalError::Fatal {
                message: format!("outer Hessian operator apply_into failed: {message}"),
            })
    }

    fn apply_mat(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, ObjectiveEvalError> {
        self.0
            .mul_mat(x)
            .map_err(|message| ObjectiveEvalError::Fatal {
                message: format!("outer Hessian operator mul_mat failed: {message}"),
            })
    }

    fn materialization(&self) -> HessianMaterialization {
        match self.0.materialization_capability() {
            OuterHessianMaterialization::Unavailable => HessianMaterialization::Unavailable,
            OuterHessianMaterialization::RepeatedHvp => HessianMaterialization::RepeatedHvp,
            OuterHessianMaterialization::BatchedHvp => HessianMaterialization::BatchedHvp,
            OuterHessianMaterialization::Explicit => HessianMaterialization::Explicit,
        }
    }

    fn materialize_dense(&self) -> Result<Array2<f64>, ObjectiveEvalError> {
        self.0
            .materialize_dense()
            .map_err(|message| ObjectiveEvalError::Fatal {
                message: format!("outer Hessian operator materialization failed: {message}"),
            })
    }
}

/// Translate a gam `HessianResult` into an `opt::HessianValue` for
/// consumption by `MatrixFreeTrustRegion`. `Analytic` becomes
/// `Dense`; `Operator` is wrapped in the adapter; `Unavailable` is
/// preserved (the solver's `HessianFallbackPolicy` decides what
/// happens then).
fn hessian_result_to_value(hessian: HessianResult) -> HessianValue {
    match hessian {
        HessianResult::Analytic(h) => HessianValue::Dense(h),
        HessianResult::Operator(op) => {
            HessianValue::Operator(Arc::new(OuterToOptHessianOperator(op)))
        }
        HessianResult::Unavailable => HessianValue::Unavailable,
    }
}

/// Bridge that exposes gam's outer objective as an
/// `opt::OperatorObjective`. Used on the matrix-free trust-region
/// route; the dense-Hessian / first-order routes still use
/// `OuterSecondOrderBridge` / `OuterFirstOrderBridge`.
struct OuterOperatorBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
    /// Inner-PIRLS cap atomic, mirroring the BFGS / ARC bridges.
    outer_inner_cap: Option<InnerProgressFeedback>,
    /// Counts gradient/Hessian evaluations for the inner-cap schedule
    /// and progress logs.
    eval_count: usize,
    /// First observed `‖g‖`. Used by the inner-cap schedule's
    /// gradient-ratio gate.
    g_norm_initial: Option<f64>,
    /// `‖g‖` from the most recent eval.
    last_g_norm: Option<f64>,
}

impl ZerothOrderObjective for OuterOperatorBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let cost = self
            .obj
            .eval_cost(x)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        finite_cost_or_error("outer eval_cost failed", cost)
    }
}

impl FirstOrderObjective for OuterOperatorBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

impl OperatorObjective for OuterOperatorBridge<'_> {
    fn eval_value_grad_op(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<OperatorSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        // Drive the outer-aware inner-PIRLS cap, mirroring
        // OuterSecondOrderBridge::eval_grad / eval_hessian. Each
        // accepted outer iter calls eval_value_grad_op exactly once
        // (the matrix-free TR's inner CG uses HVPs, not full
        // evaluations), so we increment per call without the /2 the
        // ARC bridge needs.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let cap = first_order_inner_cap_schedule(self.eval_count, g_ratio, snapshot);
            let previous_cap = feedback.cap.swap(cap, Ordering::Relaxed);
            if previous_cap != cap {
                log::trace!("outer operator bridge updated inner cap from {previous_cap} to {cap}");
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueGradientHessian dim={} (operator bridge)",
            x.len(),
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueGradientHessian)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_eval_or_error("outer eval failed", self.layout, eval)?;
        self.eval_count += 1;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        log::info!(
            "[STAGE] outer eval end elapsed={:.3}s cost={:.6e} |g|={:.3e} (operator bridge)",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
        );
        Ok(OperatorSample {
            value: eval.cost,
            gradient: eval.gradient,
            hessian: hessian_result_to_value(eval.hessian),
        })
    }
}

// Helpers preserved across the Phase 6 rewrite. Both were previously
// shared with `run_operator_trust_region` (now deleted in favor of
// `opt::MatrixFreeTrustRegion`), but they remain in use by the dense
// ARC and BFGS arms of the seed loop.

#[inline]
fn project_to_bounds(x: &Array1<f64>, bounds: Option<&(Array1<f64>, Array1<f64>)>) -> Array1<f64> {
    match bounds {
        Some((lower, upper)) => {
            let mut out = x.clone();
            for idx in 0..out.len() {
                out[idx] = out[idx].clamp(lower[idx], upper[idx]);
            }
            out
        }
        None => x.clone(),
    }
}

/// Translate an `OuterEval`'s Hessian into the `Option<Array2<f64>>`
/// shape expected by `opt::SecondOrderSample`, enforcing the contract
/// implied by the planner's `HessianSource`.
///
/// For `HessianSource::Analytic` (the exact second-order route) a missing
/// or non-materializable Hessian is FATAL: returning `None` here would
/// invite `opt::SecondOrderCache::finite_difference_hessian` to silently
/// estimate the Hessian by finite-differencing the gradient, which (a)
/// throws away the analytic structure the route was selected for, and
/// (b) costs O(K) full outer evaluations per ARC iteration — at biobank
/// scale, hours of work per silently-mis-routed step. The right
/// behavior on a planner/runtime mismatch is to surface it loudly so
/// the seed loop can either retry, demote the plan, or fail the seed.
///
/// Operator Hessians that *are* cheaply materializable (the operator's
/// `materialization_capability` reports `Explicit` / `BatchedHvp` and the
/// dimension is below `materialize_operator_max_dim`) are converted to
/// dense in-place so dense ARC can run an exact factorization. Operator
/// Hessians that are NOT cheaply materializable should never arrive
/// here: the seed loop routes those to `run_operator_trust_region`
/// before constructing the bridge. Reaching this branch on the analytic
/// route means the runtime contradicted the seed-time decision, which
/// is the same kind of mismatch we treat as fatal.
///
/// For `HessianSource::BfgsApprox`, `EfsFixedPoint`, and
/// `HybridEfsFixedPoint` we deliberately return `None`: those routes do
/// not consume an analytic Hessian and feed the Hessian into a
/// quasi-Newton/fixed-point update instead. (Today these `HessianSource`
/// variants don't actually drive `opt`'s second-order solvers, but the
/// match preserves the original behavior in case a future routing
/// reuses this bridge.)
fn build_bridge_hessian_for_source(
    source: HessianSource,
    hessian: HessianResult,
    materialize_operator_max_dim: usize,
) -> Result<Option<Array2<f64>>, ObjectiveEvalError> {
    match source {
        HessianSource::Analytic => match hessian {
            HessianResult::Analytic(h) => Ok(Some(h)),
            HessianResult::Operator(op)
                if op.materialization_capability().is_available()
                    && op.dim() <= materialize_operator_max_dim =>
            {
                op.materialize_dense()
                    .map(Some)
                    .map_err(|message| ObjectiveEvalError::Fatal {
                        message: format!(
                            "outer Hessian operator materialization failed: {message}"
                        ),
                    })
            }
            HessianResult::Operator(op) => Err(ObjectiveEvalError::Fatal {
                message: format!(
                    "outer plan declared HessianSource::Analytic but the runtime returned a \
                     non-materializable Hessian operator (dim={}, materialization={:?}); \
                     finite-difference Hessian estimation is not permitted on the analytic route",
                    op.dim(),
                    op.materialization_capability(),
                ),
            }),
            HessianResult::Unavailable => Err(ObjectiveEvalError::Fatal {
                message: "outer plan declared HessianSource::Analytic but the runtime returned \
                          HessianResult::Unavailable; finite-difference Hessian estimation is \
                          not permitted on the analytic route"
                    .to_string(),
            }),
        },
        HessianSource::BfgsApprox
        | HessianSource::EfsFixedPoint
        | HessianSource::HybridEfsFixedPoint => Ok(None),
    }
}

struct OuterFixedPointBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
    barrier_config: Option<BarrierConfig>,
    fixed_point_tolerance: f64,
    /// Consecutive HybridEFS iterations whose ψ block was zeroed after
    /// exhausting backtracking. When this reaches
    /// [`MAX_CONSECUTIVE_PSI_STAGNATION`], the bridge surfaces the
    /// [`EFS_FIRST_ORDER_FALLBACK_MARKER`] error so the runner aborts the
    /// HybridEFS attempt and the fallback ladder routes to a joint
    /// gradient-based solver where ψ stationarity ∇_ψ V = 0 can be enforced.
    consecutive_psi_zero_iters: usize,
}

impl OuterFixedPointBridge<'_> {
    fn reject_nonstationary_tiny_psi_step(
        &self,
        step: &Array1<f64>,
        psi_indices: Option<&[usize]>,
        psi_gradient: Option<&Array1<f64>>,
        cost: f64,
    ) -> Result<(), ObjectiveEvalError> {
        let Some(psi_indices) = psi_indices else {
            return Ok(());
        };
        let Some(psi_gradient) = psi_gradient else {
            return Ok(());
        };
        let psi_step_inf = psi_indices
            .iter()
            .map(|&idx| step[idx].abs())
            .fold(0.0_f64, f64::max);
        let psi_grad_inf = psi_gradient.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if psi_step_inf <= self.fixed_point_tolerance && psi_grad_inf > self.fixed_point_tolerance {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{} HybridEFS ψ nonstationary: ||Δψ||∞={:.3e} <= tol={:.3e} \
                 but raw ||gψ||∞={:.3e} (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                EFS_FIRST_ORDER_FALLBACK_MARKER,
                psi_step_inf,
                self.fixed_point_tolerance,
                psi_grad_inf,
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                cost,
            )));
        }
        Ok(())
    }
}

/// Maximum number of α halvings for the cost line search wrapping the EFS
/// step.
///
/// The Wood–Fasiolo paper proves that the EFS update direction is an *ascent
/// direction* for REML/LAML on penalty-like coordinates, but full-step
/// monotonicity is not guaranteed — both the original Fellner–Schall paper
/// and the extension recommend step-length control. We backtrack the entire
/// θ vector by halving α ∈ {1, 1/2, …, 1/2⁸ ≈ 0.004}, accepting the first
/// trial point with a strictly lower cost. With 8 halvings the smallest
/// trial step is ≈ 0.4% of the raw EFS step in every coordinate, which is
/// enough to clear pathologies near the identifiability boundary while
/// staying inside one cache-warm Hessian factorization budget.
const MAX_EFS_BACKTRACK: usize = 8;

/// Step components below this threshold (in θ-space) are treated as zero
/// for backtracking purposes — there is no point line-searching a step of
/// magnitude `1e-12`, and skipping the trial keeps the convergence path
/// numerically clean (no spurious cost decreases from ULP noise).
const EFS_NEGLIGIBLE_STEP: f64 = 1e-12;

/// Maximum infinity-norm of the EFS step (in θ-space) at which we skip the
/// cost line search and trust the multiplicative formula's quadratic
/// convergence. Above this, we always backtrack.
///
/// At small step magnitudes the canonical formula `Δρ = log((d−t)/q_eff)`
/// is itself a Newton step on the REML stationarity equation, with
/// quadratic local convergence. Under Wood–Fasiolo's Loewner-order
/// assumptions on the penalty derivative, sufficiently small steps are
/// always descent on `V`, so the line search would add an inner P-IRLS
/// solve per outer iteration with essentially zero chance of finding a
/// halving that beats the full step. The threshold is set to ~exp(0.5)
/// ≈ 1.65× change in any single λ_i (well inside the local-convergence
/// regime) and gates only the line-search call — the step itself is
/// applied unchanged, so correctness is preserved.
const EFS_LINESEARCH_THRESHOLD: f64 = 0.5;

/// Relative tolerance for the descent condition `c < current_cost` during
/// EFS backtracking. Without this, ULP-level cost noise near a fixed point
/// can cause spurious backtracking even when the step is mathematically
/// correct. We accept any trial whose cost is within
/// `EFS_COST_DESCENT_TOL · |current_cost|` of the current value.
const EFS_COST_DESCENT_TOL: f64 = 1e-12;

/// Maximum number of consecutive HybridEFS iterations whose ψ block was
/// zeroed before the bridge bails out and triggers a solver switch.
///
/// On hard problems (Matérn additive at biobank scale, Duchon60, anisotropic
/// joint penalties) a single zeroed-ψ iteration after exhausted backtracking
/// is already strong evidence the EFS ψ direction is not descent-correlated
/// at the current iterate; continuing on ρ alone with Δψ = 0 cannot enforce
/// ∇_ψ V = 0 and burns outer iterations on a non-stationary direction.
/// Bail out immediately so the fallback ladder routes to a joint
/// gradient-based solver (BFGS / L-BFGS) where ψ stationarity is part of
/// the optimality condition.
const MAX_CONSECUTIVE_PSI_STAGNATION: usize = 1;

impl FixedPointObjective for OuterFixedPointBridge<'_> {
    fn eval_step(&mut self, x: &Array1<f64>) -> Result<FixedPointSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer EFS eval failed")?;
        let eval = match self.obj.eval_efs(x) {
            Ok(eval) => eval,
            Err(err @ EstimationError::GradientUnavailable { .. })
                if requests_immediate_first_order_fallback(&err.to_string()) =>
            {
                log::warn!(
                    "[STAGE] EFS -> gradient fallback: gradient unavailable at \
                     fixed-point dispatch; retrying with fixed-point disabled \
                     (rho_dim={}, psi_dim={}, n_params={})",
                    self.layout.rho_dim(),
                    self.layout.psi_dim,
                    self.layout.n_params,
                );
                return Err(ObjectiveEvalError::recoverable(format!(
                    "outer EFS eval failed: {err}"
                )));
            }
            Err(err) => return Err(into_objective_error("outer EFS eval failed", err)),
        };
        self.layout
            .validate_efs_eval(&eval, "outer EFS eval failed")?;
        if !eval.cost.is_finite() {
            return Err(ObjectiveEvalError::recoverable(
                "outer EFS eval failed: objective returned a non-finite cost".to_string(),
            ));
        }
        // Reject non-finite EFS step components at the bridge boundary with
        // full diagnostic context (which coord, its value, and whether it is
        // a ρ or ψ coord). Without this, a NaN/Inf step flows into the
        // hybrid-EFS backtrack loop, which halves it via `NaN * 0.5^k = NaN`
        // until backtracking exhausts, then silently zeros the ψ block and
        // applies only the ρ step — masking the analytic-gradient bug that
        // produced the NaN. The opt crate's FixedPoint::run also detects
        // this downstream (opt 0.2.2 lib.rs:4949) but surfaces only the bare
        // `NonFiniteStep` variant with no context, which is not actionable.
        if let Some((idx, value)) = eval.steps.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            let psi_indices = eval.psi_indices.as_deref();
            let coord_kind = match psi_indices {
                Some(indices) if indices.contains(&idx) => "ψ",
                Some(_) => "ρ/τ",
                None => "ρ",
            };
            return Err(ObjectiveEvalError::recoverable(format!(
                "outer EFS eval failed: non-finite {coord_kind} step at coord {idx} \
                 (step[{idx}]={value}, rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                eval.cost,
            )));
        }
        if let Some(ref barrier_cfg) = self.barrier_config
            && let Some(ref beta) = eval.beta
        {
            // Scale-free precondition check for EFS. Wood–Fasiolo's
            // multiplicative log-λ update is derived under the
            // assumption that the inner Hessian is ≈ X'WX + S. A log
            // barrier adds τ/(β_j−l_j)² to the Hessian diagonal at the
            // constrained coords; when the tightest slack is much
            // smaller than the typical slack, that diagonal becomes
            // locally dominant and the EFS direction is no longer
            // guaranteed-ascent. Comparing slack *ratios* is
            // dimensionless — independent of τ, β scale, and the
            // inner-Hessian magnitude — which is exactly the regime
            // change EFS cannot represent. The earlier criterion
            // `barrier_curvature_is_significant(β, ref_diag=1.0, 0.01)`
            // was dimensionful and depended on three quantities the
            // bridge has no way to set correctly.
            //
            // Two principled triggers, each catching a distinct
            // failure mode of the EFS precondition:
            //  • `ratio = 0.1`        — asymmetric concentration:
            //    the worst slack is ≥10× tighter than the median.
            //    Catches the common "one coefficient hits its bound
            //    while others stay healthy" case.
            //  • `saturation = 1.0`   — absolute saturation:
            //    `max_j τ/Δ_j² ≥ 1`, i.e. at least one barrier-
            //    diagonal entry has reached the natural unit penalty
            //    scale. Catches the symmetric near-boundary regime
            //    that ratio-only checks would let through (median Δ
            //    also small, so min/median ratio stays near 1, but
            //    EFS's "ignore the barrier diagonal" assumption is
            //    still violated everywhere on the active set).
            const LOCAL_CONCENTRATION_RATIO: f64 = 0.1;
            const BARRIER_CURVATURE_SATURATION: f64 = 1.0;
            const BARRIER_CURVATURE_RELATIVE_THRESHOLD: f64 = 0.05;
            if let Some(hessian_scale) = eval.inner_hessian_scale
                && hessian_scale.is_finite()
                && hessian_scale > 0.0
                && barrier_cfg.barrier_curvature_is_significant(
                    beta,
                    hessian_scale,
                    BARRIER_CURVATURE_RELATIVE_THRESHOLD,
                )
            {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{} EFS barrier curvature significant relative to inner Hessian \
                         (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e}, ref_diag={:.3e})",
                    EFS_FIRST_ORDER_FALLBACK_MARKER,
                    self.layout.rho_dim(),
                    self.layout.psi_dim,
                    self.layout.n_params,
                    eval.cost,
                    hessian_scale,
                )));
            }
            if barrier_cfg.barrier_curvature_locally_concentrated(
                beta,
                LOCAL_CONCENTRATION_RATIO,
                BARRIER_CURVATURE_SATURATION,
            ) {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{} EFS barrier curvature locally concentrated \
                         (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                    EFS_FIRST_ORDER_FALLBACK_MARKER,
                    self.layout.rho_dim(),
                    self.layout.psi_dim,
                    self.layout.n_params,
                    eval.cost,
                )));
            }
        }
        let status = FixedPointStatus::Continue;

        let raw_step = Array1::from_vec(eval.steps);
        let psi_indices = eval.psi_indices.clone();
        self.reject_nonstationary_tiny_psi_step(
            &raw_step,
            psi_indices.as_deref(),
            eval.psi_gradient.as_ref(),
            eval.cost,
        )?;
        let max_step_abs = raw_step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        let current_cost = eval.cost;
        if self.fixed_point_step_converged(x, &raw_step, psi_indices.as_deref()) {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status: FixedPointStatus::Stop,
            });
        }

        // Negligible raw step — the iteration is at (or numerically
        // indistinguishable from) a fixed point. Pass it through so the
        // outer step-norm convergence check fires; no point evaluating the
        // cost at x + 1e-30·s to chase ULP-level "improvements".
        if max_step_abs < EFS_NEGLIGIBLE_STEP {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status,
            });
        }

        // Small-step fast path. The canonical Wood–Fasiolo formula is
        // locally quadratically convergent, so once we are inside the
        // multiplicative-Newton basin (`||Δθ||∞ < EFS_LINESEARCH_THRESHOLD`)
        // a halving is essentially never accepted over the full step. Skip
        // the inner P-IRLS solve we'd otherwise burn on backtracking. For
        // hybrid runs we still need to reset the ψ-stagnation counter.
        if max_step_abs < EFS_LINESEARCH_THRESHOLD {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status,
            });
        }

        // ── Stage 1: full-vector cost backtracking ──
        //
        // Wood–Fasiolo gives ascent in the EFS direction but not full-step
        // monotonicity, so backtrack α ∈ {1, 1/2, …} on the *whole* step
        // vector (not just ψ). This is a uniform requirement: even on the
        // pure-ρ path, the additive log-λ formula is exact only at the
        // fixed point and is otherwise just a Newton-flavoured Wood–Fasiolo
        // surrogate that benefits from line search at large iterations.
        if let Some(scaled) = self.efs_backtrack(x, &raw_step, current_cost, MAX_EFS_BACKTRACK)? {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: scaled,
                status,
            });
        }

        // ── Stage 2 (hybrid only): ψ-zeroed retry ──
        //
        // Full-vector backtracking exhausted means *every* α we tried gave
        // a worse cost. On the hybrid path, the most common cause is a
        // bad ψ direction polluting an otherwise-good ρ step (preconditioned
        // gradient step on a near-singular ψ-ψ Gram matrix overshoots).
        // Try the ρ/τ block alone with the same backtracking schedule. If
        // that succeeds, we make progress on ρ this iteration; the ψ
        // stagnation counter advances and triggers the joint-solver
        // fallback once it crosses MAX_CONSECUTIVE_PSI_STAGNATION.
        if let Some(psi_idx) = psi_indices.as_ref() {
            let mut rho_only = raw_step.clone();
            for &i in psi_idx {
                rho_only[i] = 0.0;
            }
            let max_rho_abs = rho_only.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
            if max_rho_abs >= EFS_NEGLIGIBLE_STEP
                && let Some(scaled) =
                    self.efs_backtrack(x, &rho_only, current_cost, MAX_EFS_BACKTRACK)?
            {
                self.consecutive_psi_zero_iters = self.consecutive_psi_zero_iters.saturating_add(1);
                log::info!(
                    "[HYBRID-EFS] full-vector backtrack exhausted; ρ/τ-only step \
                         accepted. Consecutive ψ-zero iters = {}",
                    self.consecutive_psi_zero_iters,
                );
                if self.consecutive_psi_zero_iters >= MAX_CONSECUTIVE_PSI_STAGNATION {
                    log::info!(
                        "[STAGE] HybridEFS -> joint gradient (BFGS/L-BFGS) fallback: \
                             {} consecutive ψ-zero iterations after exhausted backtracking \
                             (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                        self.consecutive_psi_zero_iters,
                        self.layout.rho_dim(),
                        self.layout.psi_dim,
                        self.layout.n_params,
                        current_cost,
                    );
                    return Err(ObjectiveEvalError::recoverable(format!(
                        "{} HybridEFS ψ stagnation: {} consecutive iterations \
                             exhausted backtracking and zeroed ψ step \
                             (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                        EFS_FIRST_ORDER_FALLBACK_MARKER,
                        self.consecutive_psi_zero_iters,
                        self.layout.rho_dim(),
                        self.layout.psi_dim,
                        self.layout.n_params,
                        current_cost,
                    )));
                }
                return Ok(FixedPointSample {
                    value: current_cost,
                    step: scaled,
                    status,
                });
            }
            // ρ/τ-only backtracking also failed — surface the joint-solver
            // fallback marker so the runner abandons EFS for this attempt.
            log::info!(
                "[STAGE] HybridEFS -> joint gradient fallback: ρ/τ-only step also \
                 failed all {} halvings (rho_dim={}, psi_dim={}, n_params={}, \
                 cost={:.6e})",
                MAX_EFS_BACKTRACK,
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                current_cost,
            );
            return Err(ObjectiveEvalError::recoverable(format!(
                "{} HybridEFS step rejected after {} halvings on full vector \
                 and {} halvings on ρ/τ-only fallback \
                 (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                EFS_FIRST_ORDER_FALLBACK_MARKER,
                MAX_EFS_BACKTRACK,
                MAX_EFS_BACKTRACK,
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                current_cost,
            )));
        }

        // Pure-EFS path with full backtracking exhausted: there is no ψ
        // block to escape to. Surface the same fallback marker so the
        // runner switches to a gradient-based solver instead of looping.
        log::info!(
            "[STAGE] EFS -> gradient fallback: no α ∈ {{1, …, 2^-{}}} decreased the \
             cost (rho_dim={}, n_params={}, cost={:.6e})",
            MAX_EFS_BACKTRACK,
            self.layout.rho_dim(),
            self.layout.n_params,
            current_cost,
        );
        Err(ObjectiveEvalError::recoverable(format!(
            "{} EFS step rejected after {} halvings on pure-ρ vector \
             (rho_dim={}, n_params={}, cost={:.6e})",
            EFS_FIRST_ORDER_FALLBACK_MARKER,
            MAX_EFS_BACKTRACK,
            self.layout.rho_dim(),
            self.layout.n_params,
            current_cost,
        )))
    }
}

impl OuterFixedPointBridge<'_> {
    /// Backtrack the cost along `raw_step` by halving α ∈ {1, 1/2, …, 2^-k}
    /// up to `max_halvings` times. Returns `Some(α·raw_step)` for the first
    /// α that yields a strictly lower finite cost, `None` if every trial
    /// failed or evaluation errored. Eval errors at trial points are
    /// treated as step rejection (a common pathology in inner solves at
    /// over-aggressive λ jumps), not propagated.
    fn efs_backtrack(
        &mut self,
        x: &Array1<f64>,
        raw_step: &Array1<f64>,
        current_cost: f64,
        max_halvings: usize,
    ) -> Result<Option<Array1<f64>>, ObjectiveEvalError> {
        // Relaxed Armijo: accept any trial within ULP noise of the current
        // cost. Pure `<` rejects ULP-noise dithering on flat regions of V
        // and forces unnecessary halvings.
        let cost_floor = current_cost + EFS_COST_DESCENT_TOL * current_cost.abs().max(1.0);
        let mut alpha = 1.0_f64;
        for bt in 0..=max_halvings {
            let trial_step = raw_step * alpha;
            let trial = x + &trial_step;
            match self.obj.eval_cost(&trial) {
                Ok(c) if c.is_finite() && c <= cost_floor => {
                    if bt > 0 {
                        log::debug!(
                            "[EFS] backtrack accepted at α=2^-{bt}={alpha:.4e} \
                             after {bt} halvings (cost: {current_cost:.6e} → {c:.6e})"
                        );
                    }
                    return Ok(Some(trial_step));
                }
                Ok(c) => {
                    log::trace!(
                        "[EFS] backtrack α=2^-{bt}={alpha:.4e}: trial cost {c:.6e} \
                         not below current {current_cost:.6e}, halving"
                    );
                }
                Err(err) => {
                    log::trace!(
                        "[EFS] backtrack α=2^-{bt}={alpha:.4e}: trial eval failed \
                         ({err}), halving"
                    );
                }
            }
            alpha *= 0.5;
        }
        Ok(None)
    }

    fn fixed_point_step_converged(
        &self,
        x: &Array1<f64>,
        step: &Array1<f64>,
        psi_indices: Option<&[usize]>,
    ) -> bool {
        if x.len() != step.len() {
            return false;
        }
        for idx in 0..step.len() {
            let scale = match psi_indices {
                Some(indices) if indices.contains(&idx) => x[idx].abs().max(1.0),
                _ => 1.0,
            };
            let normalized = step[idx].abs() / scale;
            if !normalized.is_finite() || normalized > self.fixed_point_tolerance {
                return false;
            }
        }
        true
    }
}

/// Outcome of an auxiliary compass-search run.
enum CompassSearchOutcome {
    /// The step length contracted below tolerance with no further improvement
    /// — i.e. the iterate is a step-minimizer over the positive basis
    /// {±step·e_i} at scale < step_tol. By Kolda-Lewis-Torczon Thm 3.3 this
    /// implies first-order stationarity up to the step-tol grid.
    Converged {
        point: Array1<f64>,
        cost: f64,
        polls: usize,
    },
    /// The poll budget was exhausted before step contraction reached the
    /// tolerance. Return the best-seen iterate; caller treats as
    /// non-converged so log/diagnostics surface the truncation.
    BudgetExhausted {
        point: Array1<f64>,
        cost: f64,
        polls: usize,
    },
}

/// Coordinate compass search with bound clamping.
///
/// Why this method is correct for derivative-free aux optimization:
/// the algorithm only compares cost values at polled points. It never builds
/// derivative approximations and never feeds approximations into a
/// gradient-based optimizer. For any continuously differentiable cost
/// bounded below on the compact box [lower, upper], compass search
/// converges to a stationary point (Kolda-Lewis-Torczon, SIAM Review
/// 45:385, 2003, Thm 3.3). The theorem's polling requirement — that
/// all 2·dim directions ±step·e_i are evaluated before the step
/// contracts — is satisfied explicitly by the `!improved ⇒ step /= 2`
/// branch below: if no coordinate probe improved, every probe was
/// evaluated and rejected.
///
/// Error policy: `obj.eval_cost` errors at a probe are treated as
/// infeasible (the search simply does not accept that point). A genuine
/// error at the seed itself is surfaced by the caller via the initial
/// `eval_cost` check, so the helper only runs against a finite seed cost.
fn compass_search_outer(
    obj: &mut dyn OuterObjective,
    mut x: Array1<f64>,
    mut best_cost: f64,
    lower: ndarray::ArrayView1<'_, f64>,
    upper: ndarray::ArrayView1<'_, f64>,
    init_step: f64,
    step_tol: f64,
    max_polls: usize,
) -> CompassSearchOutcome {
    for i in 0..x.len() {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }
    let mut step = init_step;
    let mut polls: usize = 0;
    while step > step_tol && polls < max_polls {
        let mut improved = false;
        'sweep: for i in 0..x.len() {
            for &sign in &[1.0, -1.0] {
                if polls >= max_polls {
                    break 'sweep;
                }
                polls += 1;
                let candidate_i = (x[i] + sign * step).clamp(lower[i], upper[i]);
                if (candidate_i - x[i]).abs() < step_tol {
                    continue;
                }
                let mut candidate = x.clone();
                candidate[i] = candidate_i;
                let probe = obj.eval_cost(&candidate).ok().filter(|v| v.is_finite());
                if let Some(c) = probe
                    && c < best_cost
                {
                    x = candidate;
                    best_cost = c;
                    improved = true;
                    break 'sweep;
                }
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    if step <= step_tol {
        CompassSearchOutcome::Converged {
            point: x,
            cost: best_cost,
            polls,
        }
    } else {
        CompassSearchOutcome::BudgetExhausted {
            point: x,
            cost: best_cost,
            polls,
        }
    }
}

fn solution_into_outer_result(
    solution: Solution,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    OuterResult {
        rho: solution.final_point,
        final_value: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm: solution.final_gradient_norm,
        final_gradient: solution.final_gradient,
        final_hessian: solution.final_hessian,
        converged,
        plan_used,
        operator_trust_radius: None,
        operator_stop_reason: None,
    }
}

use crate::inference::diagnostics::format_top_abs as format_top_abs_components;

fn bfgs_line_search_failure_message(
    context: &str,
    solution: &Solution,
    max_attempts: usize,
    failure_reason: impl std::fmt::Debug,
) -> String {
    let grad_norm = solution
        .final_gradient_norm
        .or_else(|| {
            solution
                .final_gradient
                .as_ref()
                .map(|gradient| gradient.iter().map(|v| v * v).sum::<f64>().sqrt())
        })
        .unwrap_or(f64::NAN);
    let gradient_detail = solution
        .final_gradient
        .as_ref()
        .map(|gradient| format_top_abs_components(gradient, "top_abs_gradient", 6))
        .unwrap_or_else(|| "top_abs_gradient=<unavailable>".to_string());
    format!(
        "{context}: BFGS line search failed; reason={failure_reason:?} \
         max_attempts={max_attempts} iterations={} final_value={:.6e} \
         |g|={:.3e} func_evals={} grad_evals={} {} {}",
        solution.iterations,
        solution.final_value,
        grad_norm,
        solution.func_evals,
        solution.grad_evals,
        format_top_abs_components(&solution.final_point, "top_abs_rho", 6),
        gradient_detail,
    )
}

/// Configuration for the outer optimization runner.
#[derive(Clone, Debug)]
struct OuterConfig {
    tolerance: f64,
    max_iter: usize,
    bounds: Option<(Array1<f64>, Array1<f64>)>,
    seed_config: crate::seeding::SeedConfig,
    rho_bound: f64,
    heuristic_lambdas: Option<Vec<f64>>,
    initial_rho: Option<Array1<f64>>,
    fallback_policy: FallbackPolicy,
    screening_cap: Option<Arc<AtomicUsize>>,
    screen_initial_rho: bool,
    /// Outer-aware inner-PIRLS iteration cap (sibling of `screening_cap`).
    /// When set, the BFGS bridge drives this atomic on every accepted
    /// gradient eval to coarsen the inner Newton solve at early outer iters
    /// (when ρ is far from converged) and lift it back to full as
    /// convergence approaches. Distinct from `screening_cap` in that it
    /// does NOT suppress cache writes / warm-start updates / KKT
    /// enforcement; it is purely a budget. See
    /// `RemlObjectiveState::outer_inner_cap` for dual-cap semantics.
    outer_inner_cap: Option<InnerProgressFeedback>,
    solver_class: SolverClass,
    operator_initial_trust_radius: Option<f64>,
    arc_initial_regularization: Option<f64>,
    /// Optional scale factor for the objective's natural magnitude.
    /// Used to widen the absolute gradient-norm floor on objectives whose
    /// gradient lives on a non-unit scale (e.g. Gaussian-identity REML at
    /// large `n`, whose ∂/∂logλ inherits the O(n) likelihood constant).
    /// `None` falls back to the bare `tolerance` floor.
    objective_scale: Option<f64>,
    /// BFGS line-search infinity-norm cap applied to the leading `rho_dim`
    /// outer parameters (log-λ axes). Documented natural step for
    /// `log(lambda)` is ≈ 5 (`e^5 ≈ 148`-fold smoothing-parameter change
    /// per accepted outer iter — matches typical quasi-Newton direction
    /// magnitude on flat REML surfaces). Setting this `None` disables the
    /// rho-axis cap entirely.
    bfgs_step_cap: Option<f64>,
    /// BFGS line-search infinity-norm cap applied to the trailing `psi_dim`
    /// outer parameters (kappa / aniso-log-scale axes). Required because
    /// the kernel scale axes need much tighter control (`e^1 ≈ 2.7`-fold
    /// per iter is plenty) — using the rho-axis cap here lets the optimizer
    /// jump kappa by orders of magnitude per step and oscillate. Setting
    /// this `None` disables the psi-axis cap.
    bfgs_step_cap_psi: Option<f64>,
    /// Optional persistent-cache session. When `Some`, every finite objective
    /// evaluation is written through to disk (rate-limited, atomic-rename)
    /// and the best on-disk rho is prepended as a seed at the start of each
    /// plan attempt. Defaulted off so test-only paths skip filesystem I/O.
    cache_session: Option<Arc<CacheSession>>,
    /// Optional mirror cache sessions. Checkpoints and successful finalize
    /// writes are also written to each of these sessions (different keys,
    /// shared store). Used for hierarchical broadcast: the current best ρ is
    /// written to the exact-key (primary) AND the data-independent
    /// seed-prefix key so the next fit with related structure can warm-start
    /// from this one, even after an interrupted run.
    cache_mirror_sessions: Vec<Arc<CacheSession>>,
}

impl Default for OuterConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            max_iter: 200,
            bounds: None,
            seed_config: crate::seeding::SeedConfig::default(),
            rho_bound: 30.0,
            heuristic_lambdas: None,
            initial_rho: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
            screen_initial_rho: false,
            outer_inner_cap: None,
            solver_class: SolverClass::Primary,
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            objective_scale: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
        }
    }
}

// ─── OuterProblem builder ─────────────────────────────────────────────
//
// Declarative builder for outer optimization problems.  Derives
// OuterCapability flags from high-level inputs (gradient/hessian
// availability, psi dimension, EFS eligibility) so call sites never
// hand-copy capability flags.

/// Declarative outer-problem builder.  Produces both the
/// [`OuterCapability`] (what the objective can provide) and the
/// [`OuterConfig`] (how the runner should behave) from a small set
/// of high-level declarations.
pub struct OuterProblem {
    n_params: usize,
    gradient: Derivative,
    hessian: DeclaredHessianForm,
    prefer_gradient_only: bool,
    disable_fixed_point: bool,
    psi_dim: usize,
    barrier_config: Option<BarrierConfig>,
    tolerance: f64,
    max_iter: usize,
    bounds: Option<(Array1<f64>, Array1<f64>)>,
    rho_bound: f64,
    seed_config: crate::seeding::SeedConfig,
    heuristic_lambdas: Option<Vec<f64>>,
    initial_rho: Option<Array1<f64>>,
    fallback_policy: FallbackPolicy,
    screening_cap: Option<Arc<AtomicUsize>>,
    screen_initial_rho: bool,
    outer_inner_cap: Option<InnerProgressFeedback>,
    solver_class: SolverClass,
    operator_initial_trust_radius: Option<f64>,
    arc_initial_regularization: Option<f64>,
    objective_scale: Option<f64>,
    bfgs_step_cap: Option<f64>,
    bfgs_step_cap_psi: Option<f64>,
    cache_session: Option<Arc<CacheSession>>,
    cache_mirror_sessions: Vec<Arc<CacheSession>>,
}

impl OuterProblem {
    pub fn new(n_params: usize) -> Self {
        Self {
            n_params,
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            prefer_gradient_only: false,
            disable_fixed_point: false,
            psi_dim: 0,
            barrier_config: None,
            tolerance: 1e-5,
            max_iter: 200,
            bounds: None,
            rho_bound: 30.0,
            seed_config: crate::seeding::SeedConfig::default(),
            heuristic_lambdas: None,
            initial_rho: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
            screen_initial_rho: false,
            outer_inner_cap: None,
            solver_class: SolverClass::Primary,
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            objective_scale: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
        }
    }

    pub fn with_gradient(mut self, d: Derivative) -> Self {
        self.gradient = d;
        self
    }
    pub fn with_hessian(mut self, form: DeclaredHessianForm) -> Self {
        self.hessian = form;
        self
    }
    pub fn with_prefer_gradient_only(mut self, prefer_gradient_only: bool) -> Self {
        self.prefer_gradient_only = prefer_gradient_only;
        self
    }
    /// Forbid the planner from selecting EFS/HybridEfs, even when the
    /// objective implements `eval_efs()` and the coordinate structure would
    /// otherwise make pure/hybrid EFS eligible.
    ///
    /// Callers use this for families where the Wood-Fasiolo structural
    /// property is known not to hold (e.g. GAMLSS/location-scale with
    /// β-dependent joint Hessian), so EFS would stagnate and burn budget
    /// before the automatic cascade falls back to gradient-based BFGS.
    pub fn with_disable_fixed_point(mut self, disable: bool) -> Self {
        self.disable_fixed_point = disable;
        self
    }
    pub fn with_psi_dim(mut self, dim: usize) -> Self {
        self.psi_dim = dim;
        self
    }
    pub fn with_barrier(mut self, cfg: Option<BarrierConfig>) -> Self {
        self.barrier_config = cfg;
        self
    }
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    pub fn with_bounds(mut self, lo: Array1<f64>, hi: Array1<f64>) -> Self {
        self.bounds = Some((lo, hi));
        self
    }
    pub fn with_rho_bound(mut self, b: f64) -> Self {
        self.rho_bound = b;
        self
    }
    pub fn with_seed_config(mut self, sc: crate::seeding::SeedConfig) -> Self {
        self.seed_config = sc;
        self
    }
    pub fn with_heuristic_lambdas(mut self, h: Vec<f64>) -> Self {
        self.heuristic_lambdas = Some(h);
        self
    }
    pub fn with_initial_rho(mut self, rho: Array1<f64>) -> Self {
        self.initial_rho = Some(rho);
        self
    }
    pub fn with_screening_cap(mut self, screening_cap: Arc<AtomicUsize>) -> Self {
        self.screening_cap = Some(screening_cap);
        self
    }
    /// Allow seed screening to rank the explicit initial rho against generated
    /// candidates even when the effective seed budget is one. The default keeps
    /// a user-provided initial point authoritative and avoids a separate
    /// screening pass.
    pub fn with_screen_initial_rho(mut self, screen_initial_rho: bool) -> Self {
        self.screen_initial_rho = screen_initial_rho;
        self
    }
    /// Wire the bidirectional inner-PIRLS feedback channel.
    ///
    /// The outer bridge writes a coarsened iteration cap into
    /// `feedback.cap` on every accepted gradient/Hessian eval; the inner
    /// solver writes back into `feedback.last_iters` /
    /// `feedback.last_converged` after each non-screening solve so the
    /// next outer iter's schedule can adapt to the inner solver's
    /// actual convergence behavior. Typical caller passes
    /// `InnerProgressFeedback {
    ///     cap: Arc::clone(&reml_state.outer_inner_cap),
    ///     last_iters: Arc::clone(&reml_state.last_inner_iters),
    ///     last_converged: Arc::clone(&reml_state.last_inner_converged),
    /// }` so the inner and outer observe the same atomics.
    pub fn with_outer_inner_cap(mut self, feedback: InnerProgressFeedback) -> Self {
        self.outer_inner_cap = Some(feedback);
        self
    }
    /// Opt into a specific solver class. The default is
    /// [`SolverClass::Primary`] (the main REML outer). Setting
    /// [`SolverClass::AuxiliaryGradientFree`] unlocks
    /// [`Solver::CompassSearch`] dispatch for small-dim problems with no
    /// analytic gradient (survival baseline theta, inverse-link params).
    /// REML builders must not set this.
    pub fn with_solver_class(mut self, class: SolverClass) -> Self {
        self.solver_class = class;
        self
    }

    pub fn with_operator_initial_trust_radius(mut self, radius: Option<f64>) -> Self {
        self.operator_initial_trust_radius = sanitized_operator_trust_restart_radius(radius);
        self
    }

    /// Override the ARC initial cubic-regularization parameter sigma
    /// (default in `opt`: 1.0). Smaller sigma → less cubic penalty on the
    /// first step → larger first move on benign objectives. The matrix-
    /// free Newton-TR analog is `with_operator_initial_trust_radius`.
    ///
    /// Used by Gaussian-identity REML at biobank n: the objective is
    /// quadratic-like in log-λ near the optimum (sigma is the right
    /// scale), and log-λ moves of 2–4 units in the early iters
    /// otherwise burn 4–8 iters of trust-region expansion before the
    /// model trusts the analytic Hessian.
    pub fn with_arc_initial_regularization(mut self, sigma: Option<f64>) -> Self {
        self.arc_initial_regularization = sigma.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Set the objective's natural magnitude scale, used to derive an
    /// `n`-aware absolute gradient-norm floor. When set to `Some(s)`,
    /// the runner uses `abs_floor = max(tol, s * 1e-9)` for the
    /// projected-gradient convergence check.
    ///
    /// Rationale: a fixed `abs = tol` (e.g. 1e-6) is appropriate when the
    /// objective and its gradient live on a unit scale, but Gaussian-
    /// identity REML carries an O(n) likelihood constant that flows into
    /// ∂/∂logλ. At biobank n the floor becomes binding even when the
    /// relative-from-seed component (`rel_initial_grad * ‖g0‖`) declared
    /// convergence iters earlier — chasing sub-ULP changes in log-λ at
    /// the cost of repeated k²·n·p² analytic-Hessian assemblies.
    pub fn with_objective_scale(mut self, scale: Option<f64>) -> Self {
        self.objective_scale = scale.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Cap the infinity-norm displacement of BFGS cost-only line-search probes
    /// on the **rho axes** (the first `n_params - psi_dim` outer parameters,
    /// = log-λ). Also scales the initial inverse metric so the first trial
    /// direction respects the same local budget coordinate-wise. Documented
    /// natural step on log-λ is ≈ 5; tighter values throttle BFGS and starve
    /// convergence on flat REML valleys.
    pub fn with_bfgs_step_cap(mut self, cap: Option<f64>) -> Self {
        self.bfgs_step_cap = cap.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Cap the infinity-norm displacement of BFGS cost-only line-search probes
    /// on the **psi axes** (the trailing `psi_dim` outer parameters, = kappa
    /// or anisotropic log-scales). Mirrors [`Self::with_bfgs_step_cap`] but
    /// scoped to kernel-scale parameters whose natural step is much smaller
    /// than log-λ (≈ ln 2 per iter keeps kappa from oscillating). Without
    /// this split, a uniform rho-scale cap lets psi explode while a uniform
    /// psi-scale cap throttles rho — both fail the survival-marginal-slope
    /// path at biobank scale, where rho needs |d|≈5 while psi wants |d|≤1.
    pub fn with_bfgs_step_cap_psi(mut self, cap: Option<f64>) -> Self {
        self.bfgs_step_cap_psi = cap.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    pub fn with_cache_session(mut self, session: Arc<CacheSession>) -> Self {
        self.cache_session = Some(session);
        self
    }

    /// Attach mirror cache sessions that receive a broadcast copy of
    /// the final-result finalize write. See
    /// [`OuterConfig::cache_mirror_sessions`].
    pub fn with_cache_mirror_sessions(mut self, sessions: Vec<Arc<CacheSession>>) -> Self {
        self.cache_mirror_sessions = sessions;
        self
    }

    /// Override the fallback policy. Default is [`FallbackPolicy::Automatic`].
    ///
    /// Set [`FallbackPolicy::Disabled`] when the caller requires the primary
    /// plan to stand on its own. Exact-Hessian objectives use this to ensure
    /// failures surface on the analytic geometry instead of being reinterpreted
    /// by a different optimizer class.
    pub fn with_fallback_policy(mut self, policy: FallbackPolicy) -> Self {
        self.fallback_policy = policy;
        self
    }

    /// Derive the capability flags from the builder state.
    /// `fixed_point_available` is set to `false` here; `build_objective`
    /// overrides it based on whether an EFS closure is actually provided.
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: self.gradient,
            hessian: self.hessian,
            prefer_gradient_only: self.prefer_gradient_only,
            disable_fixed_point: self.disable_fixed_point,
            n_params: self.n_params,
            psi_dim: self.psi_dim,
            fixed_point_available: false,
            barrier_config: self.barrier_config.clone(),
        }
    }

    /// Derive the runner configuration from the builder state.
    fn config(&self) -> OuterConfig {
        OuterConfig {
            tolerance: self.tolerance,
            max_iter: self.max_iter,
            bounds: self.bounds.clone(),
            seed_config: self.seed_config,
            rho_bound: self.rho_bound,
            heuristic_lambdas: self.heuristic_lambdas.clone(),
            initial_rho: self.initial_rho.clone(),
            fallback_policy: self.fallback_policy,
            screening_cap: self.screening_cap.clone(),
            screen_initial_rho: self.screen_initial_rho,
            outer_inner_cap: self.outer_inner_cap.clone(),
            solver_class: self.solver_class,
            operator_initial_trust_radius: self.operator_initial_trust_radius,
            arc_initial_regularization: self.arc_initial_regularization,
            objective_scale: self.objective_scale,
            bfgs_step_cap: self.bfgs_step_cap,
            bfgs_step_cap_psi: self.bfgs_step_cap_psi,
            cache_session: self.cache_session.clone(),
            cache_mirror_sessions: self.cache_mirror_sessions.clone(),
        }
    }

    /// Construct a [`ClosureObjective`] with capability flags derived from the
    /// builder state **and** the closures actually provided.
    ///
    /// `fixed_point_available` is set to `true` when `efs_fn` is `Some`,
    /// regardless of whether `.with_efs()` was called.  This is the canonical
    /// way to create production objectives — it eliminates the drift risk of
    /// manually entering capability flags.
    pub fn build_objective<S, Fc, Fe, Fr, Fefs>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    {
        let mut cap = self.capability();
        // Derive fixed_point_available from whether the caller actually
        // provided an EFS hook, rather than relying on manual flags.
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: None,
            reset_fn,
            efs_fn,
            screening_proxy_fn: None::<fn(&mut S, &Array1<f64>) -> Result<f64, EstimationError>>,
        }
    }

    /// Construct a [`ClosureObjective`] with an order-aware evaluation hook.
    ///
    /// This lets the runner request first-order vs second-order work based on
    /// the active outer plan while preserving the legacy eager `eval_fn`.
    pub fn build_objective_with_eval_order<S, Fc, Fe, Feo, Fr, Fefs>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        eval_order_fn: Feo,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    {
        let mut cap = self.capability();
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: Some(eval_order_fn),
            reset_fn,
            efs_fn,
            screening_proxy_fn: None::<fn(&mut S, &Array1<f64>) -> Result<f64, EstimationError>>,
        }
    }

    /// Construct a [`ClosureObjective`] with both an order-aware evaluation
    /// hook and a custom seed-screening ranking proxy. The proxy fires only
    /// when the cascade in `rank_seeds_with_screening` calls it; outside
    /// screening the regular cost path is unaffected.
    pub fn build_objective_with_screening_proxy<S, Fc, Fe, Feo, Fr, Fefs, Fsp>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        eval_order_fn: Feo,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
        screening_proxy_fn: Fsp,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        Fsp: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    {
        let mut cap = self.capability();
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: Some(eval_order_fn),
            reset_fn,
            efs_fn,
            screening_proxy_fn: Some(screening_proxy_fn),
        }
    }

    /// Run the outer optimization with a given objective.
    pub fn run(
        &self,
        obj: &mut dyn OuterObjective,
        context: &str,
    ) -> Result<OuterResult, EstimationError> {
        let mut config = self.config();
        let Some(session) = config.cache_session.clone() else {
            return run_outer(obj, &config, context);
        };
        let key_hex = session.key().to_hex();
        let short_key = &key_hex[..8.min(key_hex.len())];
        let mut had_hit = false;
        let mut cached_inner_beta: Option<Array1<f64>> = None;
        if let Some(loaded) = session.try_load_with_source() {
            match classify_cache_entry_for_outer(&loaded, self.n_params) {
                CacheSeedDecision::ExactFinal {
                    rho,
                    beta: _beta_final,
                    final_value,
                    iterations,
                    prior_obj_display,
                } => {
                    let cap = primary_capability_for_config(obj.capability(), &config, context);
                    let plan_used = plan_with_class(&cap, config.solver_class);
                    log::info!(
                        "[CACHE] final-hit key={}.. context={} rho_dim={} prior_obj={:.6e} iter={} action=skip-outer-validation",
                        short_key,
                        context,
                        rho.len(),
                        prior_obj_display,
                        iterations,
                    );
                    return Ok(OuterResult {
                        rho,
                        final_value,
                        iterations,
                        final_grad_norm: None,
                        final_gradient: None,
                        final_hessian: None,
                        converged: true,
                        plan_used,
                        operator_trust_radius: None,
                        operator_stop_reason: None,
                    });
                }
                CacheSeedDecision::Seed {
                    rho,
                    beta,
                    prior_obj_display,
                    iteration,
                } => {
                    let beta_len = beta.len();
                    let beta_arr = if beta.is_empty() {
                        None
                    } else {
                        Some(Array1::from_vec(beta))
                    };
                    if config
                        .initial_rho
                        .as_ref()
                        .is_none_or(|initial| initial != rho)
                    {
                        log::info!(
                            "[CACHE] hit  key={}.. context={} rho_dim={} beta_dim={} prior_obj={:.6e} iter={}",
                            short_key,
                            context,
                            rho.len(),
                            beta_len,
                            prior_obj_display,
                            iteration,
                        );
                        config.initial_rho = Some(rho);
                        config.screen_initial_rho = false;
                        had_hit = true;
                    } else {
                        log::info!(
                            "[CACHE] hit  key={}.. context={} rho_dim={} beta_dim={} already-aligned prior_obj={:.6e}",
                            short_key,
                            context,
                            rho.len(),
                            beta_len,
                            prior_obj_display,
                        );
                        had_hit = true;
                    }
                    cached_inner_beta = beta_arr;
                }
                CacheSeedDecision::Discard {
                    reason: "payload-shape-mismatch",
                    ..
                } => {
                    log::info!(
                        "[CACHE] skip key={}.. context={} reason=payload-shape-mismatch n_params={}",
                        short_key,
                        context,
                        self.n_params,
                    );
                }
                CacheSeedDecision::Discard {
                    reason,
                    prior_obj_display,
                    all_rho_finite,
                } => {
                    log::info!(
                        "[CACHE] skip key={}.. context={} reason={} prior_obj={:.6e} all_rho_finite={}",
                        short_key,
                        context,
                        reason,
                        prior_obj_display,
                        all_rho_finite.unwrap_or(false),
                    );
                }
            }
        } else {
            log::info!(
                "[CACHE] miss key={}.. context={} reason=fresh-fingerprint n_params={}",
                short_key,
                context,
                self.n_params,
            );
        }
        let mut checkpointing = CheckpointingObjective::new(
            obj,
            Arc::clone(&session),
            config.cache_mirror_sessions.clone(),
        );
        // Inject the cached inner β (when present) so the family's PIRLS
        // opens at the prior converged iterate. Families that don't expose
        // a β slot inherit the trait's no-op default and silently ignore
        // the hint — that's a ρ-only resume, identical to the pre-β-cache
        // behavior, but never a regression. Families that DO expose β
        // (PIRLS-based GAMs, custom-family marginal slope, …) override
        // `seed_inner_state` to install β before the first eval.
        if let Some(beta) = cached_inner_beta.as_ref() {
            match checkpointing.seed_inner_state(beta) {
                Ok(()) => log::info!(
                    "[CACHE] beta-warm key={}.. context={} beta_dim={} action=installed",
                    short_key,
                    context,
                    beta.len(),
                ),
                Err(err) => log::warn!(
                    "[CACHE] beta-warm key={}.. context={} beta_dim={} action=skip err={}",
                    short_key,
                    context,
                    beta.len(),
                    err,
                ),
            }
        }
        let result = run_outer(&mut checkpointing, &config, context);
        // Pull the most-recent inner β surfaced by the inner solver so the
        // finalize write encodes the (ρ, β) pair the BFGS optimum was
        // actually fitted at, not a ρ-only seed that resumes at cold β.
        let final_beta = checkpointing.last_inner_beta();
        if let Ok(result) = result.as_ref()
            && result.final_value.is_finite()
            && result.converged
            && let Some(bytes) = encode_iterate(
                &result.rho,
                final_beta.as_ref(),
                result.final_value,
                result.iterations as u64,
            )
        {
            let saved = session.finalize(
                &bytes,
                Some(result.final_value),
                Some(result.iterations as u64),
            );
            if saved {
                log::info!(
                    "[CACHE] save key={}.. context={} final_obj={:.6e} iter={} resumed={}",
                    short_key,
                    context,
                    result.final_value,
                    result.iterations,
                    had_hit,
                );
            }
            // Broadcast finalize to mirror keys. The seed-prefix mirror
            // exists so future fits with related-but-not-identical
            // structure can warm-start from this run via the dispatcher's
            // prefix lookup.
            for mirror in &config.cache_mirror_sessions {
                let mirror_saved = mirror.finalize(
                    &bytes,
                    Some(result.final_value),
                    Some(result.iterations as u64),
                );
                if mirror_saved {
                    let mirror_hex = mirror.key().to_hex();
                    log::info!(
                        "[CACHE] save key={}.. context={} mirror final_obj={:.6e} iter={}",
                        &mirror_hex[..8.min(mirror_hex.len())],
                        context,
                        result.final_value,
                        result.iterations,
                    );
                }
            }
        }
        result
    }
}

/// Result of a completed outer optimization.
#[derive(Clone, Debug)]
pub struct OuterResult {
    /// Optimized log-smoothing parameters.
    pub rho: Array1<f64>,
    /// Final objective value.
    pub final_value: f64,
    /// Total outer iterations across all solver restarts.
    pub iterations: usize,
    /// Final gradient norm, when the solver computed an actual gradient.
    pub final_grad_norm: Option<f64>,
    /// Final gradient when the solver is gradient-based.
    pub final_gradient: Option<Array1<f64>>,
    /// Final Hessian when the solver tracks one.
    pub final_hessian: Option<Array2<f64>>,
    /// Whether the optimizer converged to a stationary point.
    pub converged: bool,
    /// Which plan was actually used (may differ from initial if fallback fired).
    pub plan_used: OuterPlan,
    /// Final trust radius for the internal operator trust-region solver.
    ///
    /// A non-converged operator-ARC attempt may be restarted by the budget
    /// ladder. Restarting only from the last θ but resetting the trust radius
    /// is not a warm start: it replays the same rejected large trial steps.
    /// Carry this globalization state so retries resume from the scale the
    /// previous attempt already learned.
    pub operator_trust_radius: Option<f64>,
    /// Why the internal operator trust-region solver stopped.
    pub operator_stop_reason: Option<OperatorTrustRegionStopReason>,
}

impl OuterResult {
    /// Human-readable rendering of `final_grad_norm` for diagnostics. Returns
    /// `"n/a"` when no gradient was measured (gradient-free / cache-hit paths).
    pub fn final_grad_norm_report(&self) -> String {
        match self.final_grad_norm {
            Some(g) => format!("{g:.3e}"),
            None => "n/a".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorTrustRegionStopReason {
    Converged,
    RejectFloor,
    IterationBudget,
    /// Family returned a non-operator Hessian mid-flight after routing into
    /// the operator path. Best-effort `x_k` returned with this reason; the
    /// caller should consider re-fitting under a different solver class
    /// (e.g. BFGS gradient-only) instead of trusting the partial result.
    RoutingMismatch,
}

/// Run the outer smoothing-parameter optimization.
///
/// This is the single entry point that replaces the scattered optimizer wiring
/// across estimate.rs, joint.rs, and custom_family.rs. It:
///
/// 1. Queries and canonicalizes the objective's capability declaration.
/// 2. Calls `plan()` to select solver + hessian source.
/// 3. Logs the plan and the analytic derivative capabilities it will consume.
/// 4. Generates seed candidates.
/// 5. Runs the chosen solver on candidates in heuristic order up to budget.
/// 6. If the configured fallback policy allows it, re-plans with degraded
///    capabilities chosen centrally inside outer_strategy and retries.
/// 7. Returns the best result (including which plan was actually used).
///
/// Do not wrap `run_outer` calls in try/catch with ad-hoc solver recovery.
/// Callers should declare only the primary capability and, at most, whether
/// automatic fallback is enabled at all.
fn run_outer(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<OuterResult, EstimationError> {
    let cap = primary_capability_for_config(obj.capability(), config, context);
    cap.validate_layout(context)?;
    if let Some(initial_rho) = config.initial_rho.as_ref() {
        cap.theta_layout()
            .validate_point_len(initial_rho, "initial outer seed")
            .map_err(|err| match err {
                ObjectiveEvalError::Recoverable { message }
                | ObjectiveEvalError::Fatal { message } => {
                    EstimationError::RemlOptimizationFailed(format!("{context}: {message}"))
                }
            })?;
    }
    crate::solver::estimate::reml::runtime::clear_outer_ift_residual_energy_for_fit();

    if cap.n_params == 0 {
        let cost = obj.eval_cost(&Array1::zeros(0))?;
        let the_plan = plan_with_class(&cap, config.solver_class);
        return Ok(OuterResult {
            rho: Array1::zeros(0),
            final_value: cost,
            iterations: 0,
            final_grad_norm: Some(0.0),
            final_gradient: None,
            final_hessian: None,
            converged: true,
            plan_used: the_plan,
            operator_trust_radius: None,
            operator_stop_reason: None,
        });
    }

    // Build the ordered list of capabilities to attempt: primary first, then
    // any centrally-derived degraded capabilities. Aux direct-search has no
    // degraded ladder — a single attempt either succeeds or the failure is
    // surfaced to the caller.
    let fallback_attempts = match (config.fallback_policy, config.solver_class) {
        (FallbackPolicy::Automatic, SolverClass::Primary) => automatic_fallback_attempts(&cap),
        (FallbackPolicy::Automatic, SolverClass::AuxiliaryGradientFree)
        | (FallbackPolicy::Disabled, _) => Vec::new(),
    };
    let mut attempts: Vec<OuterCapability> = Vec::with_capacity(1 + fallback_attempts.len());
    attempts.push(cap.clone());
    for degraded in fallback_attempts {
        attempts.push(degraded);
    }

    let mut last_error: Option<EstimationError> = None;

    for (attempt_idx, attempt_cap) in attempts.iter().enumerate() {
        let the_plan = plan_with_class(attempt_cap, config.solver_class);
        if attempt_idx > 0 {
            log::debug!("[OUTER] {context}: primary plan failed; falling back to {the_plan}");
        }
        log_plan(context, attempt_cap, &the_plan);

        obj.reset();

        // ARC budget-exhaustion retry: when an Arc attempt runs out of
        // outer iterations, reseed a fresh Arc run from the previous
        // attempt's last ρ and trust radius. Inner caches (PIRLS LRU,
        // eval bundle, warm-start predictor, adaptive signals) are wiped
        // by `obj.reset()`; the operator-TR's Cauchy/Newton/CG state has
        // no resume API and is not preserved. The lever that changes for
        // the resumed run is the inner-PIRLS cap (uncapped via the
        // feedback handle), not `max_iter` — empirically the prior stall
        // was an inner-tolerance / model-fidelity issue, not an outer
        // budget shortfall, and doubling `max_iter` only replays the
        // same trajectory byte-for-byte. The retry is gated on observed
        // `‖g‖` progress so trajectories that made no headway fall
        // through to the degraded plan instead of replaying.
        let mut arc_retries_left: u32 = if matches!(the_plan.solver, Solver::Arc) {
            2
        } else {
            0
        };
        let mut retry_config: Option<OuterConfig> = None;
        // Tracks the previous ARC attempt's terminal `‖g‖`. The retry
        // gate compares attempt-over-attempt: if a retry didn't move
        // the gradient norm, the trajectory replayed (same seed, same
        // trust radius, cold caches, deterministic optimizer) and
        // further retries cannot help. First retry is unconditional
        // (no prior attempt to compare against).
        let mut prev_attempt_grad_norm: Option<f64> = None;

        let outcome = loop {
            // Bind the active config by cloning into a local owned value so
            // subsequent retry-config assignment does not collide with the
            // borrow used inside this iteration body.
            let active_config_owned: OuterConfig =
                retry_config.clone().unwrap_or_else(|| config.clone());
            let active_config: &OuterConfig = &active_config_owned;
            match run_outer_with_plan(obj, active_config, context, attempt_cap, &the_plan) {
                Ok(result) => {
                    if result.converged
                        || arc_retries_left == 0
                        || matches!(
                            result.operator_stop_reason,
                            Some(OperatorTrustRegionStopReason::RejectFloor)
                        )
                    {
                        break Ok(result);
                    }
                    // Gate the retry on attempt-over-attempt `‖g‖`
                    // progress. The first retry is unconditional (no
                    // prior attempt). Subsequent retries fall through
                    // to the degraded plan when the gradient norm did
                    // not materially shrink — the deterministic
                    // optimizer with the same seed and trust radius
                    // would replay the same trajectory.
                    let Some(cur_grad_norm) = result.final_grad_norm else {
                        log::info!(
                            "[OUTER] {context}: ARC attempt exhausted budget at \
                             iter={} cost={:.6e} without a final gradient norm; \
                             falling through to degraded plan",
                            result.iterations,
                            result.final_value,
                        );
                        break Ok(result);
                    };
                    if let Some(prev_g) = prev_attempt_grad_norm {
                        let progressed = cur_grad_norm.is_finite()
                            && prev_g.is_finite()
                            && cur_grad_norm < 0.5 * prev_g;
                        if !progressed {
                            log::info!(
                                "[OUTER] {context}: ARC retry stalled at \
                                 iter={} cost={:.6e} |g|={:.6e} (prev |g|={:.6e}); \
                                 deterministic replay suspected, falling through \
                                 to degraded plan",
                                result.iterations,
                                result.final_value,
                                cur_grad_norm,
                                prev_g,
                            );
                            break Ok(result);
                        }
                    }
                    let next_trust_radius =
                        sanitized_operator_trust_restart_radius(result.operator_trust_radius);
                    log::info!(
                        "[OUTER] {context}: ARC attempt exhausted budget at \
                         iter={} cost={:.6e} |g|={:.6e}; resuming from last \
                         rho + trust_radius={:?}, inner-PIRLS uncapped \
                         (objective caches wiped; operator-TR Cauchy/Newton \
                         state is not resumable)",
                        result.iterations,
                        result.final_value,
                        cur_grad_norm,
                        next_trust_radius,
                    );
                    // Snapshot the cap-feedback handle before we
                    // reassign `retry_config` (which currently backs
                    // `active_config`'s borrow). `InnerProgressFeedback`
                    // is an Arc-wrapper bundle, so the clone is cheap.
                    let cap_feedback = active_config.outer_inner_cap.clone();
                    let mut next = active_config.clone();
                    prev_attempt_grad_norm = Some(cur_grad_norm);
                    next.initial_rho = Some(result.rho.clone());
                    next.operator_initial_trust_radius = next_trust_radius;
                    retry_config = Some(next);
                    arc_retries_left -= 1;
                    obj.reset();
                    // Lift any inner-PIRLS cap for the resumed run. The
                    // schedule's cold-start ladder (3/5/10) would
                    // re-coarsen exactly the inner solves whose tolerance
                    // is suspected to have starved the prior trajectory.
                    // The next outer iter consumes ρ near a near-stationary
                    // point where exact β / gradient / Hessian is the
                    // load-bearing input to the operator-TR geometry.
                    if let Some(feedback) = cap_feedback.as_ref() {
                        feedback.cap.store(0, Ordering::Relaxed);
                    }
                }
                Err(e) => break Err(e),
            }
        };

        match outcome {
            Ok(result) => {
                if result.converged || attempt_idx + 1 == attempts.len() {
                    if !result.converged {
                        log::warn!(
                            "[OUTER warning] {context}: final outer attempt returned without convergence \
                             (plan={the_plan}, iterations={}, final_value={:.6e}, |g|={})",
                            result.iterations,
                            result.final_value,
                            result.final_grad_norm_report(),
                        );
                    }
                    return Ok(result);
                }

                let message = format!(
                    "{context}: attempt {} (plan={the_plan}) exhausted without convergence",
                    attempt_idx + 1
                );
                log::debug!("[OUTER] {message}; trying degraded fallback plan");
                last_error = Some(EstimationError::RemlOptimizationFailed(message));
            }
            Err(e) => {
                log::debug!(
                    "[OUTER] {context}: attempt {} (plan={the_plan}) failed: {e}",
                    attempt_idx + 1
                );
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        EstimationError::RemlOptimizationFailed(format!("all plan attempts exhausted ({context})"))
    }))
}

fn outer_bounds(lo: &Array1<f64>, hi: &Array1<f64>) -> Result<Bounds, EstimationError> {
    Bounds::new(lo.clone(), hi.clone(), 1e-6).map_err(|err| {
        EstimationError::InvalidInput(format!("outer rho bounds are invalid: {err}"))
    })
}

fn outer_tolerance(value: f64) -> Result<Tolerance, EstimationError> {
    Tolerance::new(value)
        .map_err(|err| EstimationError::InvalidInput(format!("outer tolerance is invalid: {err}")))
}

fn outer_gradient_tolerance(config: &OuterConfig) -> GradientTolerance {
    let abs = config
        .objective_scale
        .map(|scale| config.tolerance.max(scale * 1.0e-9))
        .unwrap_or(config.tolerance);
    GradientTolerance {
        abs,
        rel_initial_grad: None,
        rel_cost: Some(config.tolerance),
        projected: true,
    }
}

fn outer_max_iterations(value: usize) -> Result<MaxIterations, EstimationError> {
    MaxIterations::new(value)
        .map_err(|err| EstimationError::InvalidInput(format!("outer max_iter is invalid: {err}")))
}

fn sanitized_operator_trust_restart_radius(radius: Option<f64>) -> Option<f64> {
    radius
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| value.max(OPERATOR_TRUST_RESTART_RADIUS_FLOOR))
}

fn bfgs_axis_step_caps(config: &OuterConfig, layout: OuterThetaLayout) -> Option<Array1<f64>> {
    if config.bfgs_step_cap.is_none() && config.bfgs_step_cap_psi.is_none() {
        return None;
    }
    let mut caps = Array1::from_elem(layout.n_params, f64::INFINITY);
    if let Some(cap) = config.bfgs_step_cap {
        for i in 0..layout.rho_dim() {
            caps[i] = cap;
        }
    }
    if let Some(cap) = config.bfgs_step_cap_psi {
        for i in layout.rho_dim()..layout.n_params {
            caps[i] = cap;
        }
    }
    Some(caps)
}

/// Execute a single plan attempt (seed generation → solver loop → best result).
fn run_outer_with_plan(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    cap: &OuterCapability,
    the_plan: &OuterPlan,
) -> Result<OuterResult, EstimationError> {
    let mut seeds = {
        let generated = crate::seeding::generate_rho_candidates(
            cap.n_params,
            config.heuristic_lambdas.as_deref(),
            &config.seed_config,
        );
        if generated.is_empty() {
            Vec::new()
        } else {
            generated
        }
    };
    if let Some(initial_rho) = config.initial_rho.as_ref()
        && !seeds.iter().any(|seed| seed == initial_rho)
    {
        seeds.insert(0, initial_rho.clone());
    }
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "no seeds generated for outer optimization ({context})"
        )));
    }

    let (lower, upper) = config.bounds.clone().unwrap_or_else(|| {
        (
            Array1::<f64>::from_elem(cap.n_params, -config.rho_bound),
            Array1::<f64>::from_elem(cap.n_params, config.rho_bound),
        )
    });
    let bounds_template = (lower, upper);
    let mut projected_seeds = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let projected = project_to_bounds(&seed, Some(&bounds_template));
        if !projected_seeds.contains(&projected) {
            projected_seeds.push(projected);
        }
    }
    seeds = projected_seeds;
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "no bounded seeds generated for outer optimization ({context})"
        )));
    }

    let screening_enabled = config.screening_cap.is_some();
    let seed_budget = effective_seed_budget(
        config.seed_config.seed_budget,
        the_plan.solver,
        config.seed_config.risk_profile,
        screening_enabled,
    )
    .min(seeds.len());
    let explicit_initial_rho_owns_single_seed_budget = config.initial_rho.is_some()
        && seed_budget == 1
        && seeds.len() > 1
        && !config.screen_initial_rho;
    if !explicit_initial_rho_owns_single_seed_budget
        && should_screen_seeds(config, the_plan.solver, seeds.len(), seed_budget)
    {
        seeds = rank_seeds_with_screening(obj, config, context, &seeds);
    }
    log::debug!(
        "[OUTER] {context}: trying generated seeds directly (generated={}, budget={})",
        seeds.len(),
        seed_budget,
    );
    if seed_budget < config.seed_config.seed_budget.max(1) {
        log::debug!(
            "[OUTER] {context}: capped requested seed budget {} -> {} for {:?} ({:?})",
            config.seed_config.seed_budget.max(1),
            seed_budget,
            the_plan.solver,
            config.seed_config.risk_profile,
        );
    }
    if seeds.len() > seed_budget {
        log::debug!(
            "[OUTER] {context}: trying up to {seed_budget}/{} generated seeds in heuristic order",
            seeds.len(),
        );
    }

    let mut best: Option<OuterResult> = None;
    // Accumulate every per-seed rejection with its 0-based seed index and the
    // phase that rejected it (validation vs solver run). When all seeds fail
    // systematically (bad analytic gradient, rank-deficient penalty, etc.) the
    // first rejection's rho + error is often the most diagnostic.
    let mut rejection_reasons: Vec<(usize, &'static str, String)> = Vec::new();
    let layout = cap.theta_layout();
    let mut started_seeds = 0usize;
    let expensive_seed_limit =
        expensive_unsuccessful_seed_limit(the_plan.solver, config.seed_config.risk_profile);
    let mut unsuccessful_expensive_seeds = 0usize;
    // Tracks whether the loop broke out early due to
    // `expensive_unsuccessful_seed_limit` so the aggregate error can
    // distinguish "all generated seeds tried" from "stopped early".
    let mut stopped_early_due_to_limit = false;

    'seed_attempts: for (seed_idx, seed) in seeds.iter().enumerate() {
        if started_seeds == seed_budget {
            break;
        }
        crate::solver::estimate::reml::runtime::record_current_outer_iter_for_ift(0);
        obj.reset();
        let t_seed_start = std::time::Instant::now();
        let seed_slot;
        let result: Result<OuterResult, EstimationError> = match the_plan.solver {
            Solver::Arc => {
                let seed_eval = obj
                    .eval_with_order(seed, OuterEvalOrder::ValueGradientHessian)
                    .map_err(|err| into_objective_error("outer eval failed", err));
                let seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        let err = match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        };
                        if requests_immediate_first_order_fallback(&err.to_string()) {
                            return Err(err);
                        }
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                };
                let seed_eval = finite_outer_eval_or_error("outer eval failed", layout, seed_eval)
                    .map_err(|err| match err {
                        ObjectiveEvalError::Recoverable { message }
                        | ObjectiveEvalError::Fatal { message } => {
                            EstimationError::RemlOptimizationFailed(message)
                        }
                    });
                let mut seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                };
                validate_second_order_seed_hessian(context, layout, &seed_eval).map_err(|err| {
                    match err {
                        ObjectiveEvalError::Recoverable { message }
                        | ObjectiveEvalError::Fatal { message } => {
                            EstimationError::RemlOptimizationFailed(message)
                        }
                    }
                })?;
                started_seeds += 1;
                seed_slot = started_seeds;

                let cheap_materializable_operator = matches!(
                    seed_eval.hessian,
                    HessianResult::Operator(ref op)
                        if op.materialization_capability().is_available()
                            && op.dim() <= OUTER_HVP_MATERIALIZE_MAX_DIM
                );
                if cheap_materializable_operator {
                    // The operator's own work model says probing every column
                    // is cheap; convert the seed Hessian to dense in-place.
                    // Subsequent bridge evaluations apply the same predicate.
                    if let HessianResult::Operator(op) = &seed_eval.hessian {
                        match op.materialize_dense() {
                            Ok(dense) => {
                                seed_eval.hessian = HessianResult::Analytic(dense);
                            }
                            Err(message) => {
                                let err = EstimationError::RemlOptimizationFailed(format!(
                                    "outer Hessian operator materialization failed: {message}"
                                ));
                                log::warn!(
                                    "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                                );
                                rejection_reasons.push((seed_idx, "validation", err.to_string()));
                                continue 'seed_attempts;
                            }
                        }
                    }
                }
                if matches!(seed_eval.hessian, HessianResult::Operator(_)) {
                    log::debug!(
                        "[OUTER] {context}: analytic Hessian provided as Hv operator; \
                        routing to opt::MatrixFreeTrustRegion (Steihaug-Toint CG)"
                    );
                    let (lo, hi) = &bounds_template;
                    let bounds_obj = outer_bounds(lo, hi)?;
                    // Scale-aware tolerance via opt 0.5.0:
                    // `relative_to_cost(τ)` = `τ * (1 + |f|)` resolved
                    // at run time from the seed cost and initial grad
                    // norm. Replaces the previous gam-side
                    // precomputed `outer_scaled_tolerance` hack.
                    let grad_tol = outer_gradient_tolerance(config);
                    let max_iter = outer_max_iterations(config.max_iter)?;

                    // Translate the seed_eval into an opt::OperatorSample
                    // so the matrix-free TR solver can serve its first
                    // call from cache without redoing the full outer
                    // eval. The Hessian translation goes through the
                    // gam->opt operator adapter when the seed Hessian is
                    // an Hv operator; Analytic seeds become Dense.
                    let initial_op_sample = OperatorSample {
                        value: seed_eval.cost,
                        gradient: seed_eval.gradient.clone(),
                        hessian: hessian_result_to_value(seed_eval.hessian.clone()),
                    };

                    let bridge_obj = OuterOperatorBridge {
                        obj,
                        layout,
                        outer_inner_cap: config.outer_inner_cap.clone(),
                        eval_count: 0,
                        g_norm_initial: None,
                        last_g_norm: None,
                    };

                    let mut solver = MatrixFreeTrustRegion::new(seed.clone(), bridge_obj)
                        .with_bounds(bounds_obj)
                        .with_gradient_tolerance(grad_tol)
                        .with_max_iterations(max_iter)
                        .with_initial_sample(seed.clone(), initial_op_sample)
                        // Looser Eisenstat–Walker forcing factor on the
                        // inner Steihaug–Toint CG (default 0.1 → 0.5). The
                        // matrix-free route is reached only after
                        // `prefer_outer_hessian_operator` says Hv is
                        // expensive (large k, n·p crossover, or wide
                        // basis), which is exactly the regime where the
                        // standard inexact-Newton-Krylov 0.5 forcing
                        // factor wins: one extra outer-TR iter is cheap
                        // versus halving the number of inner Hv applies
                        // per outer iter. At biobank shape (n=300 K,
                        // ~64 outer-TR iters × ~30 trace_logdet calls per
                        // Hv) this halves the dominant per-fit work.
                        .with_cg_tolerance(0.5)
                        // The matrix-free route is exclusively for
                        // exact analytic Hessians; an `Unavailable`
                        // here is a routing/contract violation.
                        .with_hessian_fallback_policy(HessianFallbackPolicy::Error);
                    if let Some(feedback) = config.outer_inner_cap.as_ref() {
                        solver = solver.with_observer(OuterAcceptObserver {
                            feedback: feedback.clone(),
                        });
                    }
                    if let Some(r) = sanitized_operator_trust_restart_radius(
                        config.operator_initial_trust_radius,
                    ) {
                        solver = solver.with_initial_trust_radius(r);
                    }

                    let mf_start = std::time::Instant::now();
                    let report = solver.run_report();
                    let mf_elapsed = mf_start.elapsed().as_secs_f64();
                    let final_radius = report.diagnostics.final_trust_radius;
                    log::info!(
                        "[OUTER summary] matrix-free TR finished status={:?} in {} iters \
                         elapsed={:.3}s final_value={:.6e} final_trust_radius={}",
                        report.status,
                        report.solution.iterations,
                        mf_elapsed,
                        report.solution.final_value,
                        match final_radius {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
                    );
                    // Translate the structured report into an `OuterResult`.
                    // `operator_stop_reason` wiring (read by the gam-side
                    // retry orchestrator in `run_outer_with_plan`) maps
                    // directly from `OptimizationStatus`. opt 0.4.1
                    // populates `final_trust_radius` so the
                    // `operator_trust_radius` warm-start hook now works
                    // for matrix-free retries: the budget-bumped retry
                    // resumes from the geometry the previous attempt
                    // already learned instead of redoing the trust-radius
                    // adaptation from the configured initial radius.
                    match report.status {
                        OptimizationStatus::Converged
                        | OptimizationStatus::NumericallyConverged => {
                            let mut result =
                                solution_into_outer_result(report.solution, true, *the_plan);
                            result.operator_stop_reason =
                                Some(OperatorTrustRegionStopReason::Converged);
                            result.operator_trust_radius = final_radius;
                            Ok(result)
                        }
                        OptimizationStatus::MaxIterations => {
                            log::warn!(
                                "[OUTER warning] {context}: matrix-free TR hit max_iter={} at final_value={:.6e} |g|={:.3e} final_trust_radius={}",
                                config.max_iter,
                                report.solution.final_value,
                                report.solution.final_gradient_norm.unwrap_or(f64::NAN),
                                match final_radius {
                                    Some(r) => format!("{:.3e}", r),
                                    None => "n/a".to_string(),
                                },
                            );
                            let mut result =
                                solution_into_outer_result(report.solution, false, *the_plan);
                            result.operator_stop_reason =
                                Some(OperatorTrustRegionStopReason::IterationBudget);
                            result.operator_trust_radius = final_radius;
                            Ok(result)
                        }
                        OptimizationStatus::TrustRegionRejectFloor => {
                            log::warn!(
                                "[OUTER warning] {context}: matrix-free TR reached trust-radius reject floor at final_value={:.6e} |g|={:.3e} final_trust_radius={}",
                                report.solution.final_value,
                                report.solution.final_gradient_norm.unwrap_or(f64::NAN),
                                match final_radius {
                                    Some(r) => format!("{:.3e}", r),
                                    None => "n/a".to_string(),
                                },
                            );
                            let mut result =
                                solution_into_outer_result(report.solution, false, *the_plan);
                            result.operator_stop_reason =
                                Some(OperatorTrustRegionStopReason::RejectFloor);
                            result.operator_trust_radius = final_radius;
                            Ok(result)
                        }
                        OptimizationStatus::ObjectiveFailed
                        | OptimizationStatus::NumericalFailure
                        | OptimizationStatus::LineSearchFailed => {
                            Err(EstimationError::RemlOptimizationFailed(format!(
                                "matrix-free TR solver failed with status={:?}",
                                report.status
                            )))
                        }
                    }
                } else {
                    let hessian_source = the_plan.hessian_source;
                    let (lo, hi) = &bounds_template;
                    let bounds = outer_bounds(lo, hi)?;
                    let grad_tol = outer_gradient_tolerance(config);
                    let max_iter = outer_max_iterations(config.max_iter)?;

                    let objective = OuterSecondOrderBridge {
                        obj,
                        layout,
                        hessian_source,
                        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
                        eval_count: 0,
                        outer_inner_cap: config.outer_inner_cap.clone(),
                        g_norm_initial: None,
                        last_g_norm: None,
                    };

                    // Build the opt seed sample from the precomputed
                    // outer evaluation. The Hessian translation goes
                    // through `build_bridge_hessian_for_source` so the
                    // analytic-route contract (no None Hessian on
                    // `HessianSource::Analytic`) applies at seed time
                    // too, not just inside the bridge's live path.
                    let seed_hessian = build_bridge_hessian_for_source(
                        hessian_source,
                        seed_eval.hessian.clone(),
                        OUTER_HVP_MATERIALIZE_MAX_DIM,
                    )
                    .map_err(|err| match err {
                        ObjectiveEvalError::Recoverable { message }
                        | ObjectiveEvalError::Fatal { message } => {
                            EstimationError::RemlOptimizationFailed(message)
                        }
                    })?;
                    let initial_sample = SecondOrderSample {
                        value: seed_eval.cost,
                        gradient: seed_eval.gradient.clone(),
                        hessian: seed_hessian,
                    };

                    let mut optimizer = ArcOptimizer::new(seed.clone(), objective)
                        .with_bounds(bounds)
                        .with_gradient_tolerance(grad_tol)
                        .with_max_iterations(max_iter)
                        .with_initial_sample(seed.clone(), initial_sample);
                    if let Some(sigma) = config.arc_initial_regularization {
                        optimizer = optimizer.with_initial_regularization(sigma);
                    }
                    if let Some(feedback) = config.outer_inner_cap.as_ref() {
                        optimizer = optimizer.with_observer(OuterAcceptObserver {
                            feedback: feedback.clone(),
                        });
                    }
                    // On the exact-Hessian ARC route, forbid both (a)
                    // finite-difference Hessian estimation if the
                    // objective ever returns
                    // `SecondOrderSample { hessian: None }` and (b)
                    // `opt`'s internal AutoBfgs demotion on step
                    // failure. `HessianFallbackPolicy::Error` plus
                    // `FallbackPolicy::Never` is the precise
                    // expression of "stay inside analytic-Hessian
                    // geometry; surface mismatches loudly". opt 0.3.0
                    // API; previously this was approximated by the
                    // coarse `Profile::Deterministic` knob (which also
                    // tightens unrelated `eta_accept` / history caps).
                    if matches!(hessian_source, HessianSource::Analytic) {
                        optimizer = optimizer
                            .with_hessian_fallback_policy(HessianFallbackPolicy::Error)
                            .with_fallback_policy(OptFallbackPolicy::Never);
                    }
                    match optimizer.run() {
                        Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                        Err(ArcError::MaxIterationsReached { last_solution, .. }) => {
                            log::warn!(
                                "[OUTER warning] {context}: ARC hit max_iter={} at final_value={:.6e} |g|={:.3e}",
                                config.max_iter,
                                last_solution.final_value,
                                last_solution.final_gradient_norm.unwrap_or(f64::NAN),
                            );
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "Arc solver failed: {e:?}"
                        ))),
                    }
                }
            }
            Solver::Bfgs => {
                // Production invariant: the outer BFGS runner requires an
                // analytic gradient capability. Fail loudly at the top of the
                // seed loop so the caller surfaces the underlying
                // capability/plan mismatch instead of degrading correctness
                // behind the scenes.
                if cap.gradient != Derivative::Analytic {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "{context}: outer BFGS requires an analytic gradient capability; \
                         no non-analytic fallback is available (plan={the_plan}, \
                         declared gradient={:?})",
                        cap.gradient,
                    )));
                }
                let seed_eval = obj
                    .eval_with_order(seed, OuterEvalOrder::ValueAndGradient)
                    .map_err(|err| into_objective_error("outer eval failed", err));
                let seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        let err = match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        };
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                };
                let seed_eval = match finite_outer_first_order_eval_or_error(
                    "outer eval failed",
                    layout,
                    seed_eval,
                )
                .map_err(|err| match err {
                    ObjectiveEvalError::Recoverable { message }
                    | ObjectiveEvalError::Fatal { message } => {
                        EstimationError::RemlOptimizationFailed(message)
                    }
                }) {
                    Ok(eval) => eval,
                    Err(err) => {
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                };
                started_seeds += 1;
                seed_slot = started_seeds;
                let (lo, hi) = &bounds_template;
                let bounds = outer_bounds(lo, hi)?;
                let grad_tol = outer_gradient_tolerance(config);
                let max_iter = outer_max_iterations(config.max_iter)?;
                let objective = OuterFirstOrderBridge {
                    obj,
                    layout,
                    outer_inner_cap: config.outer_inner_cap.clone(),
                    iter_count: 0,
                    g_norm_initial: None,
                    last_g_norm: None,
                };
                // Hand the precomputed (cost, gradient) seed eval to
                // `opt::Bfgs` so its first internal `eval_grad` call is
                // served from cache instead of re-running the outer
                // objective. Inner P-IRLS solves dominate outer cost
                // at biobank scale; skipping one re-eval at the seed
                // is one of the cheapest wins available. (opt 0.3.0
                // API; before that this was implemented via a
                // gam-side cache on the bridge.)
                let initial_sample = FirstOrderSample {
                    value: seed_eval.cost,
                    gradient: seed_eval.gradient.clone(),
                };
                let mut optimizer = Bfgs::new(seed.clone(), objective)
                    .with_initial_sample(seed.clone(), initial_sample)
                    .with_bounds(bounds)
                    .with_gradient_tolerance(grad_tol)
                    .with_max_iterations(max_iter);
                if let Some(caps) = bfgs_axis_step_caps(config, layout) {
                    optimizer = optimizer.with_axis_step_caps(caps);
                }
                if let Some(feedback) = config.outer_inner_cap.as_ref() {
                    optimizer = optimizer.with_observer(OuterAcceptObserver {
                        feedback: feedback.clone(),
                    });
                }
                let bfgs_start = std::time::Instant::now();
                let outcome = optimizer.run();
                let bfgs_elapsed = bfgs_start.elapsed().as_secs_f64();
                match &outcome {
                    Ok(sol) => log::info!(
                        "[OUTER summary] BFGS converged in {} iters elapsed={:.3}s final_value={:.6e}",
                        sol.iterations,
                        bfgs_elapsed,
                        sol.final_value
                    ),
                    Err(BfgsError::MaxIterationsReached { last_solution }) => log::warn!(
                        // Include `in N iters` for symmetry with the
                        // converged log line — the runner aggregator
                        // (commit afd66d6a) reads the optional iters
                        // group to build `bfgs_iters_p50/_max` across
                        // both successful and cap-hit runs. Without
                        // this, the iter-count distribution would be
                        // biased toward fast-converged runs.
                        "[OUTER summary] BFGS hit max_iter in {} iters elapsed={:.3}s final_value={:.6e}",
                        last_solution.iterations,
                        bfgs_elapsed,
                        last_solution.final_value
                    ),
                    Err(BfgsError::LineSearchFailed {
                        last_solution,
                        max_attempts,
                        failure_reason,
                    }) => log::info!(
                        // Same rationale as the MaxIterationsReached
                        // arm: surface `in N iters` so the runner can
                        // include line-search-failed runs in the
                        // iter-count distribution. A line-search
                        // failure at iter 1 (cold start collapses
                        // immediately) is a different signal from
                        // failure at iter 50 (the optimizer made
                        // substantial progress before stalling).
                        "[OUTER summary] BFGS line-search failed in {} iters elapsed={:.3}s final_value={:.6e} reason={:?} max_attempts={} |g|={:.3e}",
                        last_solution.iterations,
                        bfgs_elapsed,
                        last_solution.final_value,
                        failure_reason,
                        max_attempts,
                        last_solution.final_gradient_norm.unwrap_or(f64::NAN),
                    ),
                    Err(e) => log::info!(
                        "[OUTER summary] BFGS failed elapsed={:.3}s err={:?}",
                        bfgs_elapsed,
                        e
                    ),
                }
                match outcome {
                    Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                    Err(BfgsError::MaxIterationsReached { last_solution }) => {
                        Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                    }
                    Err(BfgsError::LineSearchFailed {
                        last_solution,
                        max_attempts,
                        failure_reason,
                    }) => {
                        if last_solution.final_value.is_finite()
                            && last_solution.final_point.iter().all(|v| v.is_finite())
                            && last_solution
                                .final_gradient
                                .as_ref()
                                .is_none_or(|g| g.iter().all(|v| v.is_finite()))
                        {
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        } else {
                            Err(EstimationError::RemlOptimizationFailed(
                                bfgs_line_search_failure_message(
                                    context,
                                    &last_solution,
                                    max_attempts,
                                    failure_reason,
                                ),
                            ))
                        }
                    }
                    Err(BfgsError::ObjectiveFailed { message }) => {
                        Err(EstimationError::RemlOptimizationFailed(format!(
                            "BFGS solver failed: ObjectiveFailed {{ message: {message:?} }}"
                        )))
                    }
                    Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                        "BFGS solver failed: {e:?}"
                    ))),
                }
            }
            Solver::Efs => {
                let mut objective = OuterFixedPointBridge {
                    obj,
                    layout,
                    barrier_config: cap.barrier_config.clone(),
                    fixed_point_tolerance: config.tolerance,
                    consecutive_psi_zero_iters: 0,
                };
                match objective.eval_step(seed) {
                    Ok(_) => {}
                    Err(err) => {
                        let err = match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        };
                        if requests_immediate_first_order_fallback(&err.to_string()) {
                            return Err(err);
                        }
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                };
                started_seeds += 1;
                seed_slot = started_seeds;
                let (lo, hi) = &bounds_template;
                let bounds = outer_bounds(lo, hi)?;
                let tol = outer_tolerance(config.tolerance)?;
                let max_iter = outer_max_iterations(config.max_iter)?;
                let mut optimizer = FixedPoint::new(seed.clone(), objective)
                    .with_bounds(bounds)
                    .with_tolerance(tol)
                    .with_max_iterations(max_iter);
                match optimizer.run() {
                    Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                    Err(FixedPointError::MaxIterationsReached { last_solution }) => {
                        log::warn!(
                            "[OUTER warning] {context}: EFS hit max_iter={} at final_value={:.6e} step_norm={:.3e}",
                            config.max_iter,
                            last_solution.final_value,
                            last_solution.final_gradient_norm.unwrap_or(f64::NAN),
                        );
                        Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                    }
                    Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                        "fixed-point solver failed: {e:?}"
                    ))),
                }
            }
            Solver::HybridEfs => {
                let mut objective = OuterFixedPointBridge {
                    obj,
                    layout,
                    barrier_config: cap.barrier_config.clone(),
                    fixed_point_tolerance: config.tolerance,
                    consecutive_psi_zero_iters: 0,
                };
                match objective.eval_step(seed) {
                    Ok(_) => {}
                    Err(err) => {
                        let err = match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        };
                        if requests_immediate_first_order_fallback(&err.to_string()) {
                            return Err(err);
                        }
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                };
                started_seeds += 1;
                seed_slot = started_seeds;
                let (lo, hi) = &bounds_template;
                let bounds = outer_bounds(lo, hi)?;
                let tol = outer_tolerance(config.tolerance)?;
                let max_iter = outer_max_iterations(config.max_iter)?;
                let mut optimizer = FixedPoint::new(seed.clone(), objective)
                    .with_bounds(bounds)
                    .with_tolerance(tol)
                    .with_max_iterations(max_iter);
                match optimizer.run() {
                    Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                    Err(FixedPointError::MaxIterationsReached { last_solution }) => {
                        log::warn!(
                            "[OUTER warning] {context}: HybridEFS hit max_iter={} at final_value={:.6e} step_norm={:.3e}",
                            config.max_iter,
                            last_solution.final_value,
                            last_solution.final_gradient_norm.unwrap_or(f64::NAN),
                        );
                        Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                    }
                    Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                        "hybrid EFS solver failed: {e:?}"
                    ))),
                }
            }
            Solver::CompassSearch => {
                // Aux direct-search: uses cost values only, never queries
                // gradient or Hessian. config.tolerance is the step-length
                // floor, config.max_iter is the total-poll budget.
                let projected_seed = project_to_bounds(seed, Some(&bounds_template));
                let seed_cost = obj.eval_cost(&projected_seed).map_err(|err| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "aux direct-search seed cost failed ({context}): {err}"
                    ))
                })?;
                if !seed_cost.is_finite() {
                    rejection_reasons.push((
                        seed_idx,
                        "validation",
                        format!("aux direct-search rejects non-finite seed cost ({seed_cost})"),
                    ));
                    continue 'seed_attempts;
                }
                started_seeds += 1;
                seed_slot = started_seeds;
                let (lo, hi) = &bounds_template;
                let outcome = compass_search_outer(
                    obj,
                    projected_seed,
                    seed_cost,
                    lo.view(),
                    hi.view(),
                    1.0,
                    config.tolerance,
                    config.max_iter,
                );
                match outcome {
                    CompassSearchOutcome::Converged { point, cost, polls } => Ok(OuterResult {
                        rho: point,
                        final_value: cost,
                        iterations: polls,
                        final_grad_norm: None,
                        final_gradient: None,
                        final_hessian: None,
                        converged: true,
                        plan_used: *the_plan,
                        operator_trust_radius: None,
                        operator_stop_reason: None,
                    }),
                    CompassSearchOutcome::BudgetExhausted { point, cost, polls } => {
                        log::warn!(
                            "[OUTER warning] {context}: compass search exhausted max_polls={} at best_cost={:.6e}",
                            config.max_iter,
                            cost,
                        );
                        Ok(OuterResult {
                            rho: point,
                            final_value: cost,
                            iterations: polls,
                            final_grad_norm: None,
                            final_gradient: None,
                            final_hessian: None,
                            converged: false,
                            plan_used: *the_plan,
                            operator_trust_radius: None,
                            operator_stop_reason: None,
                        })
                    }
                }
            }
        };

        let seed_elapsed = t_seed_start.elapsed().as_secs_f64();
        match result {
            Ok(candidate) => {
                let candidate_converged = candidate.converged;
                log::debug!(
                    "[outer-timing] seed {}/{} ({:?}): {:.3}s  cost={:.6e}  converged={}",
                    seed_slot,
                    seed_budget,
                    the_plan.solver,
                    seed_elapsed,
                    candidate.final_value,
                    candidate.converged,
                );
                if candidate_improves_best(&candidate, best.as_ref()) {
                    best = Some(candidate);
                }
                let quality_compare_remaining_gaussian_seeds = matches!(
                    config.seed_config.risk_profile,
                    crate::seeding::SeedRiskProfile::Gaussian
                ) && seed_budget > 1
                    && started_seeds < seed_budget;
                if best.as_ref().is_some_and(|b| b.converged)
                    && !quality_compare_remaining_gaussian_seeds
                {
                    break;
                }
                if !candidate_converged && matches!(expensive_seed_limit, Some(limit) if limit > 0)
                {
                    unsuccessful_expensive_seeds += 1;
                    if let Some(limit) = expensive_seed_limit
                        && unsuccessful_expensive_seeds >= limit
                    {
                        log::info!(
                            "[OUTER] {context}: stopping expensive multi-start after {} non-converged {:?} seed(s)",
                            unsuccessful_expensive_seeds,
                            the_plan.solver,
                        );
                        stopped_early_due_to_limit = true;
                        break;
                    }
                }
            }
            Err(e) => {
                if requests_immediate_first_order_fallback(&e.to_string()) {
                    return Err(e);
                }
                log::debug!(
                    "[outer-timing] seed {}/{} ({:?}): {:.3}s  FAILED: {}",
                    seed_slot,
                    seed_budget,
                    the_plan.solver,
                    seed_elapsed,
                    e,
                );
                rejection_reasons.push((seed_idx, "solver", e.to_string()));
                if let Some(limit) = expensive_seed_limit {
                    unsuccessful_expensive_seeds += 1;
                    if unsuccessful_expensive_seeds >= limit {
                        log::info!(
                            "[OUTER] {context}: stopping expensive multi-start after {} failed {:?} seed(s)",
                            unsuccessful_expensive_seeds,
                            the_plan.solver,
                        );
                        stopped_early_due_to_limit = true;
                        break;
                    }
                }
            }
        }
    }

    best.ok_or_else(|| {
        // Build a compact breakdown of why every attempted seed failed so the
        // caller sees root causes, not just the last-written reason. Earlier
        // behaviour stored only `Option<String>` and overwrote on every reject,
        // which erased diagnostic context for the first k-1 seeds — a silent
        // drift especially bad when analytic-gradient or penalty-rank bugs
        // systematically break every seed with the same class of error, because
        // then only the LAST occurrence was visible to the caller.
        let n_generated = seeds.len();
        let n_attempted = n_generated.min(seed_budget);
        let n_rejected = rejection_reasons.len();
        let breakdown = if rejection_reasons.is_empty() {
            String::new()
        } else {
            let joined = rejection_reasons
                .iter()
                .map(|(idx, phase, msg)| format!("seed {idx} ({phase}): {msg}"))
                .collect::<Vec<_>>()
                .join("; ");
            format!("; reasons: [{joined}]")
        };
        let early_stop_note = if stopped_early_due_to_limit {
            format!(
                "; stopped early after {unsuccessful_expensive_seeds} consecutive \
                 non-converged {:?} seed(s) (expensive_unsuccessful_seed_limit)",
                the_plan.solver
            )
        } else {
            String::new()
        };
        if started_seeds == 0 {
            EstimationError::RemlOptimizationFailed(format!(
                "no candidate seeds passed outer startup validation ({context}); \
                 generated={n_generated}, attempted={n_attempted}, rejected={n_rejected}{breakdown}"
            ))
        } else {
            EstimationError::RemlOptimizationFailed(format!(
                "all {started_seeds} seed candidates failed ({context}); \
                 generated={n_generated}, attempted={n_attempted}, \
                 started_in_solver={started_seeds}, rejected={n_rejected}\
                 {early_stop_note}{breakdown}"
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::opt::FixedPointObjective;
    use ndarray::array;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    // The two `outer_scaled_tolerance_*` tests that lived here have
    // been removed: the helper is gone in favor of opt 0.5.0's
    // `GradientTolerance::relative_to_cost(τ)`. Equivalent threshold
    // coverage now lives upstream as
    // `opt::tests::gradient_tolerance_relative_to_cost_matches_textbook_form`.

    struct FailingSeedMaterializationOperator {
        dim: usize,
    }

    impl OuterHessianOperator for FailingSeedMaterializationOperator {
        fn dim(&self) -> usize {
            self.dim
        }

        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            Ok(v.clone())
        }

        fn is_cheap_to_materialize(&self) -> bool {
            true
        }

        fn materialize_dense(&self) -> Result<Array2<f64>, String> {
            Err("seed materialization failed".to_string())
        }
    }

    #[test]
    fn materialize_dense_uses_single_batched_mul_mat() {
        struct BatchedOnlyHessian {
            matrix: Array2<f64>,
            matvec_calls: Arc<AtomicUsize>,
            mul_mat_calls: Arc<AtomicUsize>,
            rhs_columns: Arc<AtomicUsize>,
        }

        impl OuterHessianOperator for BatchedOnlyHessian {
            fn dim(&self) -> usize {
                self.matrix.nrows()
            }

            fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
                self.matvec_calls.fetch_add(1, Ordering::Relaxed);
                Ok(self.matrix.dot(v))
            }

            fn mul_mat(&self, factor: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
                self.mul_mat_calls.fetch_add(1, Ordering::Relaxed);
                self.rhs_columns
                    .fetch_add(factor.ncols(), Ordering::Relaxed);
                Ok(self.matrix.dot(&factor))
            }
        }

        let matvec_calls = Arc::new(AtomicUsize::new(0));
        let mul_mat_calls = Arc::new(AtomicUsize::new(0));
        let rhs_columns = Arc::new(AtomicUsize::new(0));
        let op = BatchedOnlyHessian {
            matrix: array![[2.0, 0.25, -0.5], [0.5, 3.0, 1.0], [-0.25, 2.0, 4.0]],
            matvec_calls: Arc::clone(&matvec_calls),
            mul_mat_calls: Arc::clone(&mul_mat_calls),
            rhs_columns: Arc::clone(&rhs_columns),
        };

        let dense = op
            .materialize_dense()
            .expect("batched dense materialization");
        let expected = array![[2.0, 0.375, -0.375], [0.375, 3.0, 1.5], [-0.375, 1.5, 4.0]];
        assert_eq!(dense, expected);
        assert_eq!(
            mul_mat_calls.load(Ordering::Relaxed),
            1,
            "dense materialization must batch all identity columns into one mul_mat call"
        );
        assert_eq!(
            rhs_columns.load(Ordering::Relaxed),
            3,
            "the single batched materialization call must include every identity RHS"
        );
        assert_eq!(
            matvec_calls.load(Ordering::Relaxed),
            0,
            "operators with batched mul_mat must not be probed column-by-column"
        );
    }

    #[test]
    fn plan_analytic_hessian_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_prefer_gradient_only_does_not_hide_analytic_hessian() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 3,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_survival_baseline_exact_hessian_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_no_hessian_few_params_selects_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_no_hessian_many_params_selects_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 12,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_boundary_8_params_uses_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: SMALL_OUTER_BFGS_MAX_PARAMS,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_boundary_9_params_uses_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: SMALL_OUTER_BFGS_MAX_PARAMS + 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_selected_for_penalty_like_many_params() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_penalty_like_without_fixed_point_stays_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_not_selected_few_params_even_if_penalty_like() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 5,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_not_selected_with_analytic_hessian() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 20,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        // Arc is always preferred when analytic Hessian is available.
        assert_eq!(p.solver, Solver::Arc);
    }

    #[test]
    fn plan_efs_with_no_gradient_penalty_like_many_params() {
        // Even without analytic gradient, EFS works because it doesn't
        // need the gradient at all.
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 20,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_efs_allowed_with_barrier_config() {
        // When barrier_config is present (monotonicity constraints), EFS is
        // still selected at plan time. The runtime barrier-curvature guard
        // in the EFS loop handles safety.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0, 1],
            lower_bounds: vec![0.0, 0.0],
            bound_signs: vec![1.0, 1.0],
        };
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: Some(barrier),
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_efs_allowed_with_barrier_config_no_gradient() {
        // Even without analytic gradient, EFS is selected when all coords
        // are penalty-like and the problem is above the small-problem
        // BFGS cutoff, regardless of barrier presence.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0],
            lower_bounds: vec![0.0],
            bound_signs: vec![1.0],
        };
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 20,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: Some(barrier),
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn barrier_curvature_significant_blocks_efs_at_runtime() {
        // Verify that barrier_curvature_is_significant correctly detects
        // when coefficients are near their bounds.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0],
            lower_bounds: vec![0.0],
            bound_signs: vec![1.0],
        };
        // β very close to bound → curvature is large
        let beta_near = Array1::from_vec(vec![0.001]);
        assert!(barrier.barrier_curvature_is_significant(&beta_near, 1.0, 0.01));

        // β far from bound → curvature is negligible
        let beta_far = Array1::from_vec(vec![10.0]);
        assert!(!barrier.barrier_curvature_is_significant(&beta_far, 1.0, 0.01));
    }

    #[test]
    fn barrier_curvature_locally_concentrated_covers_both_failure_modes() {
        // τ = 1e-6 (BarrierConfig default).
        // For the dimensional check τ/Δ² ≥ saturation_threshold:
        //   • Δ = 1e-3 ⇒ τ/Δ² = 1.0 (right at saturation = 1.0)
        //   • Δ = 1e-2 ⇒ τ/Δ² = 1e-2 (well below)
        //   • Δ = 1e-4 ⇒ τ/Δ² = 100 (well above)
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0, 1],
            lower_bounds: vec![0.0, 0.0],
            bound_signs: vec![1.0, 1.0],
        };

        // Mode (b) symmetric near-boundary: slacks uniform & both small.
        // With saturation = 1.0, Δ = 1e-2 stays under the saturation
        // wall and ratio is healthy → not concentrated. Δ = 1e-4
        // saturates absolutely → concentrated.
        let mild_uniform = Array1::from_vec(vec![1.0e-2, 1.0e-2]);
        assert!(!barrier.barrier_curvature_locally_concentrated(&mild_uniform, 0.1, 1.0));
        let tight_uniform = Array1::from_vec(vec![1.0e-4, 1.0e-4]);
        assert!(barrier.barrier_curvature_locally_concentrated(&tight_uniform, 0.1, 1.0));

        // Mode (b) is gated by saturation_threshold: with a very large
        // threshold (effectively disabling (b)), tight uniform stops
        // tripping until you also relax (a) — the asymmetric ratio
        // check — which on uniform slacks is necessarily false.
        assert!(!barrier.barrier_curvature_locally_concentrated(&tight_uniform, 0.1, 1.0e9));

        // Large uniform slacks: neither mode trips.
        let large_uniform = Array1::from_vec(vec![10.0, 10.0]);
        assert!(!barrier.barrier_curvature_locally_concentrated(&large_uniform, 0.1, 1.0));

        // Mode (a) asymmetric concentration: one slack 100× tighter
        // than the other, all in a regime where mode (b) DOESN'T fire.
        // Δ_min = 1e-2 ⇒ τ/Δ² = 1e-2 ≪ 1.0 saturation. So only the
        // ratio check is doing work here.
        let imbalanced = Array1::from_vec(vec![1.0e-2, 1.0]);
        assert!(barrier.barrier_curvature_locally_concentrated(&imbalanced, 0.1, 1.0));
        // With a permissive ratio (1e-3) and mode (b) effectively off
        // (huge threshold), neither check trips.
        assert!(!barrier.barrier_curvature_locally_concentrated(&imbalanced, 1.0e-3, 1.0e9));

        // Infeasible (β ≤ l) → conservatively concentrated.
        let infeasible = Array1::from_vec(vec![-0.5, 1.0]);
        assert!(barrier.barrier_curvature_locally_concentrated(&infeasible, 0.1, 1.0));
    }

    #[test]
    fn hessian_result_unwrap_analytic() {
        let h = Array2::<f64>::eye(3);
        let result = HessianResult::Analytic(h.clone());
        assert!(result.is_analytic());
        let extracted = result.unwrap_analytic();
        assert_eq!(extracted, h);
    }

    #[test]
    #[should_panic(expected = "expected analytic Hessian")]
    fn hessian_result_unwrap_unavailable_panics() {
        let result = HessianResult::Unavailable;
        result.unwrap_analytic();
    }

    #[test]
    fn zero_params_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 0,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn hessian_result_into_option() {
        let h = Array2::<f64>::eye(2);
        let result = HessianResult::Analytic(h.clone());
        assert_eq!(result.into_option(), Some(h));

        let result = HessianResult::Unavailable;
        assert_eq!(result.into_option(), None);
    }

    #[test]
    fn closure_objective_delegates() {
        let mut obj = ClosureObjective {
            state: 42_i32,
            cap: OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Unavailable,
                n_params: 1,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            },
            cost_fn: |_: &mut i32, _: &Array1<f64>| Ok(1.0),
            eval_fn: |_: &mut i32, _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 1.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            eval_order_fn: None::<
                fn(&mut i32, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
            >,
            reset_fn: Some(|st: &mut i32| {
                *st = 42;
            }),
            efs_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
            screening_proxy_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<f64, EstimationError>>,
        };
        assert_eq!(obj.capability().n_params, 1);
        assert_eq!(obj.eval_cost(&Array1::zeros(1)).unwrap(), 1.0);
    }

    #[test]
    fn hybrid_efs_backtracking_uses_half_step_after_first_rejection() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 12,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let mut obj = ClosureObjective {
            state: (),
            cap: cap.clone(),
            cost_fn: |_: &mut (), theta: &Array1<f64>| {
                let psi = theta[11];
                let cost = if (psi - 0.0).abs() < 1e-12 {
                    1.0
                } else if (psi - 0.5).abs() < 1e-12 {
                    0.5
                } else {
                    2.0
                };
                Ok(cost)
            },
            eval_fn: |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[11].abs(),
                    gradient: Array1::zeros(theta.len()),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            eval_order_fn: None::<
                fn(&mut (), &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
            >,
            reset_fn: None::<fn(&mut ())>,
            efs_fn: Some(|_: &mut (), theta: &Array1<f64>| {
                let mut steps = vec![0.0; theta.len()];
                steps[11] = 1.0;
                Ok(EfsEval {
                    cost: 1.0,
                    steps,
                    beta: None,
                    psi_gradient: Some(array![1.0]),
                    psi_indices: Some(vec![11]),
                    inner_hessian_scale: None,
                })
            }),
            screening_proxy_fn: None::<fn(&mut (), &Array1<f64>) -> Result<f64, EstimationError>>,
        };
        let mut bridge = OuterFixedPointBridge {
            obj: &mut obj,
            layout: cap.theta_layout(),
            barrier_config: None,
            fixed_point_tolerance: 1e-8,
            consecutive_psi_zero_iters: 0,
        };

        let sample = bridge
            .eval_step(&Array1::zeros(cap.n_params))
            .expect("hybrid EFS step should backtrack cleanly");

        assert_eq!(sample.status, FixedPointStatus::Continue);
        assert_eq!(sample.step.len(), cap.n_params);
        assert_eq!(sample.step[11], 0.5);
        assert!(
            sample
                .step
                .iter()
                .enumerate()
                .all(|(idx, &value)| idx == 11 || value == 0.0)
        );
    }

    #[test]
    fn run_bfgs_mode_aware_eval_skips_hessian_work() {
        let seen_orders = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_initial_rho(array![1.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective_with_eval_order(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput(
                    "legacy eager eval should not run on BFGS".to_string(),
                ))
            },
            {
                let seen_orders = Arc::clone(&seen_orders);
                move |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
                    seen_orders.lock().unwrap().push(order);
                    Ok(OuterEval {
                        cost: theta[0] * theta[0],
                        gradient: array![2.0 * theta[0]],
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "mode-aware bfgs first order")
            .expect("BFGS should use the order-aware first-order bridge");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        let seen_orders = seen_orders.lock().unwrap();
        assert!(
            !seen_orders.is_empty(),
            "mode-aware eval hook should have been used"
        );
        assert!(
            seen_orders
                .iter()
                .all(|order| *order == OuterEvalOrder::ValueAndGradient),
            "BFGS should request only value+gradient, saw {seen_orders:?}"
        );
    }

    // The historical bridge-side `rejects_oversized_bfgs_cost_probe_before_objective`
    // test exercised a mechanism (returning `BFGS_LINE_SEARCH_REJECT_COST`
    // from `eval_cost` on overreach) that has been retired in favor of
    // `opt::Bfgs::with_axis_step_caps` — the line-search direction is now
    // shortened up front by opt itself, so the bridge never sees an
    // oversized probe in the first place. The equivalent invariant now
    // lives in opt's `with_axis_step_caps` test surface.

    #[test]
    fn first_order_bridge_keeps_true_gradient_on_repeated_flat_cost() {
        let eval_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(1000.0),
            {
                let eval_calls = Arc::clone(&eval_calls);
                move |_: &mut (), _: &Array1<f64>| {
                    let call = eval_calls.fetch_add(1, Ordering::Relaxed);
                    let cost = match call {
                        0 => 999.9995,
                        1 => 999.9990,
                        _ => 999.9987,
                    };
                    Ok(OuterEval {
                        cost,
                        gradient: array![4.0],
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut bridge = OuterFirstOrderBridge {
            obj: &mut obj,
            layout: OuterThetaLayout::new(1, 0),
            outer_inner_cap: None,
            iter_count: 0,
            g_norm_initial: None,
            last_g_norm: None,
        };

        let first = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
            .expect("first flat-cost eval should expose the true gradient");
        let second = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
            .expect("second flat-cost eval should expose the true gradient");
        let third = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
            .expect("third flat-cost eval should expose the true gradient");
        let fourth = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
            .expect("fourth flat-cost eval should expose the true gradient");

        assert_eq!(first.gradient[0], 4.0);
        assert_eq!(second.gradient[0], 4.0);
        assert_eq!(third.gradient[0], 4.0);
        assert_eq!(fourth.gradient[0], 4.0);
        assert_eq!(bridge.last_g_norm, Some(4.0));
        assert_eq!(eval_calls.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn outer_second_order_bridge_separates_first_and_second_order_requests() {
        let seen_orders = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either);
        let mut obj = problem.build_objective_with_eval_order(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput(
                    "legacy eager eval should not run".to_string(),
                ))
            },
            {
                let seen_orders = Arc::clone(&seen_orders);
                move |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
                    seen_orders.lock().unwrap().push(order);
                    Ok(OuterEval {
                        cost: theta[0] * theta[0],
                        gradient: array![2.0 * theta[0]],
                        hessian: match order {
                            OuterEvalOrder::ValueAndGradient => HessianResult::Unavailable,
                            OuterEvalOrder::ValueGradientHessian => {
                                HessianResult::Analytic(array![[2.0]])
                            }
                        },
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut bridge = OuterSecondOrderBridge {
            obj: &mut obj,
            layout: OuterThetaLayout::new(1, 0),
            hessian_source: HessianSource::Analytic,
            materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
            eval_count: 0,
            outer_inner_cap: None,
            g_norm_initial: None,
            last_g_norm: None,
        };
        let grad_sample =
            FirstOrderObjective::eval_grad(&mut bridge, &array![1.0]).expect("grad eval");
        assert_eq!(grad_sample.value, 1.0);
        assert_eq!(grad_sample.gradient, array![2.0]);
        let hess_sample =
            SecondOrderObjective::eval_hessian(&mut bridge, &array![1.0]).expect("hessian eval");
        assert_eq!(hess_sample.value, 1.0);
        assert_eq!(hess_sample.gradient, array![2.0]);
        assert_eq!(hess_sample.hessian, Some(array![[2.0]]));
        let seen_orders = seen_orders.lock().unwrap();
        assert!(
            *seen_orders
                == vec![
                    OuterEvalOrder::ValueAndGradient,
                    OuterEvalOrder::ValueGradientHessian
                ],
            "second-order bridge should split first-order and second-order requests, saw {seen_orders:?}"
        );
    }

    /// Phase 1.1 — On `HessianSource::Analytic` the bridge MUST surface a
    /// fatal error rather than producing `SecondOrderSample { hessian: None }`
    /// when the runtime returns `HessianResult::Unavailable`. A `None` here
    /// would let `opt::SecondOrderCache::finite_difference_hessian` silently
    /// estimate the Hessian by finite-differencing the gradient — at biobank
    /// scale, hours of work per silently-mis-routed step. The seed loop
    /// should retry, demote, or fail loudly instead.
    #[test]
    fn analytic_route_unavailable_hessian_is_fatal() {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either);
        let mut obj = problem.build_objective_with_eval_order(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput(
                    "legacy eager eval should not run".to_string(),
                ))
            },
            move |_: &mut (), theta: &Array1<f64>, _order: OuterEvalOrder| {
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut bridge = OuterSecondOrderBridge {
            obj: &mut obj,
            layout: OuterThetaLayout::new(1, 0),
            hessian_source: HessianSource::Analytic,
            materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
            eval_count: 0,
            outer_inner_cap: None,
            g_norm_initial: None,
            last_g_norm: None,
        };
        let err = SecondOrderObjective::eval_hessian(&mut bridge, &array![1.0])
            .expect_err("Analytic route must reject Unavailable Hessian, not pass None to opt");
        match err {
            ObjectiveEvalError::Fatal { message } => {
                assert!(
                    message.contains("HessianSource::Analytic") && message.contains("Unavailable"),
                    "fatal message should explain the analytic-route mismatch, saw: {message}"
                );
            }
            ObjectiveEvalError::Recoverable { message } => panic!(
                "Analytic-route Hessian violations must be Fatal (FD estimation is forbidden); \
                 got Recoverable: {message}"
            ),
        }
    }

    // Phase 5 (Cargo dep at opt 0.3) replaces the gam-side bridge
    // seed cache with `opt::{Bfgs, Arc, NewtonTrustRegion}::with_initial_sample`.
    // The two cache tests that lived here have been removed;
    // equivalent integration coverage now lives upstream as
    // `opt::tests::with_initial_sample_serves_first_call_from_cache`
    // and `opt::tests::bfgs_with_initial_sample_serves_first_call_from_cache`.
    // The fatal-on-Analytic-route contract (Phase 1.1) is still tested
    // here since it lives in gam's `build_bridge_hessian_for_source`.

    #[test]
    fn outer_config_default() {
        let cfg = OuterConfig::default();
        assert_eq!(cfg.tolerance, 1e-5);
        assert_eq!(cfg.max_iter, 200);
        assert_eq!(cfg.rho_bound, 30.0);
    }

    #[test]
    fn plan_hybrid_efs_selected_for_psi_coords_many_params() {
        // When ψ (design-moving) coords are present and the problem is above
        // the small-problem BFGS cutoff, the planner should select HybridEfs
        // instead of falling back to BFGS.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::HybridEfs);
        assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
    }

    #[test]
    fn plan_psi_without_fixed_point_stays_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 1,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_hybrid_efs_no_gradient_selected_for_psi_coords() {
        // Even without analytic gradient, hybrid EFS works because the
        // gradient is computed internally by the unified evaluator.
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::HybridEfs);
        assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
    }

    // ----------------------------------------------------------------------
    // Routing regression tests (spec section 12).
    //
    // Post-#1 (compute-budget failure paths removed) and #2 (Hessian
    // cost-gating in custom_family.rs removed), the planner no longer
    // downgrades `(Analytic, Analytic)` to BFGS at any problem size. The
    // contract is:
    //
    //   high dense work + analytic+analytic     → ARC + Analytic
    //                                             (runtime then chooses
    //                                              operator HVP per family)
    //   high dense work + analytic + Unavailable → BFGS + BfgsApprox
    //                                             (matrix-free not advertised
    //                                              by the family — BFGS is
    //                                              still the right choice)
    //
    // `routing_log_line()` exposes a stable token that biobank log
    // regressions in tests/bench_biobank_scale_runner_test.py pin against.
    // ----------------------------------------------------------------------

    fn cap_for_routing(
        gradient: Derivative,
        hessian: DeclaredHessianForm,
        n_params: usize,
    ) -> OuterCapability {
        OuterCapability {
            gradient,
            hessian,
            n_params,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    #[test]
    fn routing_analytic_analytic_stays_arc_at_biobank_scale() {
        // Biobank-scale standard GAM (n=320K, p=65, k=6) used to trigger the
        // aggregate `k·n·p²` cost-driven downgrade. Post-#1 the planner has
        // no scale-driven downgrade, so `(Analytic, Analytic)` must stay on
        // ARC + Analytic regardless of the problem dimensions.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 6);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_analytic_analytic_stays_arc_at_dense_work_scale() {
        // n=3·10⁵, p=300 used to trigger the per-inner-solve `n·p²` downgrade
        // (`2.7·10¹⁰ ≫ 5·10⁹`). Post-#1, no work-hint API exists; ARC stays.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 3);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_unavailable_hessian_routes_to_bfgs() {
        // Spec section 12: when the family cannot provide a second derivative
        // (matrix-free or otherwise), BFGS is the correct route.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Unavailable, 8);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn routing_explicit_prefer_gradient_only_does_not_override_exact_hessian() {
        // The primary REML outer must never hide an analytic Hessian behind a
        // quasi-Newton route. Auxiliary gradient-only optimizers are separate
        // solver classes; this flag is ignored for Analytic+Analytic primary
        // capabilities.
        let mut cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 6);
        cap.prefer_gradient_only = true;
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_log_line_arc_analytic_does_not_advertise_matrix_free() {
        // Token pinned by tests/bench_biobank_scale_runner_test.py. Renaming
        // any of these substrings is a log-regression and breaks downstream
        // grep patterns.
        let p = OuterPlan {
            solver: Solver::Arc,
            hessian_source: HessianSource::Analytic,
        };
        let line = p.routing_log_line();
        assert!(line.contains("solver=Arc"), "got {line}");
        assert!(line.contains("hessian=Analytic"), "got {line}");
        assert!(line.contains("matrix-free=false"), "got {line}");
    }

    #[test]
    fn routing_log_line_bfgs_reports_no_matrix_free() {
        let p = OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        };
        let line = p.routing_log_line();
        assert!(line.contains("solver=Bfgs"), "got {line}");
        assert!(line.contains("hessian=BfgsApprox"), "got {line}");
        assert!(line.contains("matrix-free=false"), "got {line}");
    }

    #[test]
    fn routing_log_line_efs_reports_no_matrix_free() {
        // EFS variants don't expose a Hessian operator either, so the
        // matrix-free token is `false`.
        for source in [
            HessianSource::EfsFixedPoint,
            HessianSource::HybridEfsFixedPoint,
        ] {
            let p = OuterPlan {
                solver: Solver::Efs,
                hessian_source: source,
            };
            assert!(
                p.routing_log_line().contains("matrix-free=false"),
                "{:?} should not advertise matrix-free",
                source
            );
        }
    }

    // ----------------------------------------------------------------------
    // Per-family routing regression tests.
    //
    // Each family that gains matrix-free Hessian operators must, at the
    // OuterProblem build site, declare both derivatives `Analytic` so the
    // planner stays on ARC + Analytic. These tests pin that contract from
    // the planner side. The runtime's choice between dense-Hessian-assembly
    // and operator-HVPs is independent of the planner; a separate per-family
    // test (in the family's own module) should pin that.
    //
    // ----------------------------------------------------------------------

    #[test]
    fn routing_custom_family_gamlss_stays_on_arc_when_both_derivs_analytic() {
        // Post-#5/#12, GAMLSS advertises matrix-free directional operators
        // for the joint Hessian; the OuterProblem build site must declare
        // both derivatives Analytic so ARC + Analytic stays in effect.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 4);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_matern_iso_kappa_stays_on_arc_when_both_derivs_analytic() {
        // Post-#7, Matern/TPS spatial κ/τ derivative drifts ship as
        // HyperOperators; planner contract: (Analytic, Analytic) → ARC.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 5);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_matern_iso_large_kappa_dim_stays_on_arc_with_analytic_hessian() {
        // Spatial isotropic κ no longer declares Hessian unavailable when
        // kappa_dim > 30.  Large κ blocks are represented by exact HVP
        // operators at evaluation time, so the planner must keep second-order
        // ARC instead of selecting HybridEFS.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 37,
            psi_dim: 31,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_marginal_slope_stays_on_arc_when_both_derivs_analytic() {
        // Bernoulli/survival marginal-slope: the planner contract is the
        // same — (Analytic, Analytic) → ARC + Analytic. Runtime selects
        // operator HVPs via `use_joint_matrix_free_path`.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 3);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_hybrid_efs_not_selected_few_params() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 5,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_exact_hvp_capability_selects_arc_even_when_fixed_point_is_available() {
        // Large spatial/custom-family problems may also expose EFS/HybridEFS
        // fixed-point traces, but an explicit dense Hessian or exact HVP
        // operator is stronger geometry. The planner must therefore select
        // ARC + Analytic rather than cost-demoting to BFGS/EFS when the
        // evaluator advertises second-order capability.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 64,
            psi_dim: 16,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_hybrid_efs_not_selected_with_analytic_hessian() {
        // Arc is always preferred when analytic Hessian is available,
        // even with ψ coordinates.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 20,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
    }

    #[test]
    fn plan_pure_efs_not_hybrid_when_all_penalty_like() {
        // When all coords are penalty-like (no ψ), pure EFS is selected
        // even if has_psi_coords is false.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn automatic_fallbacks_preserve_analytic_hessian_for_arc_primary() {
        // For an (Analytic, Analytic) capability the planner emits ARC. The
        // cascade MUST NOT add a BFGS+BfgsApprox demotion: doing so discards
        // the analytic outer Hessian ARC was using, replaces it with a
        // strictly weaker rank-2 approximation, and silently masks ARC's
        // actual failure mode (budget exhaustion, indefinite curvature)
        // under a BFGS Strong-Wolfe plateau. ARC budget exhaustion is
        // handled by the per-attempt retry ladder in
        // `run_outer_with_strategy`; once that is exhausted, the caller
        // sees the genuine analytic-Hessian non-convergence verbatim.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 12,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::Arc);
        let attempts = automatic_fallback_attempts(&cap);
        assert!(
            attempts.is_empty(),
            "ARC primary must not lateral-demote to BFGS+BfgsApprox; \
             ARC budget retries live in the runner",
        );
    }

    #[test]
    fn automatic_fallbacks_from_efs_prefer_analytic_bfgs_over_fd() {
        // When the primary plan is EFS, the first fallback must keep the
        // analytic gradient and just disable the fixed-point path so the
        // planner picks gradient-based BFGS. Silently downgrading to finite
        // differences here was the long-standing production bug we are
        // guarding against.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::Efs);

        let attempts = automatic_fallback_attempts(&cap);
        assert!(!attempts.is_empty(), "EFS failure must have a fallback");
        assert_eq!(attempts[0].gradient, Derivative::Analytic);
        assert_eq!(attempts[0].hessian, DeclaredHessianForm::Unavailable);
        assert!(attempts[0].disable_fixed_point);
        assert_eq!(plan(&attempts[0]).solver, Solver::Bfgs);

        assert!(
            attempts.iter().all(|c| c.gradient == Derivative::Analytic),
            "fallback cascade must stay on analytic-gradient attempts",
        );
    }

    #[test]
    fn automatic_fallbacks_from_hybrid_efs_prefer_analytic_bfgs_over_fd() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 2,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::HybridEfs);

        let attempts = automatic_fallback_attempts(&cap);
        assert!(!attempts.is_empty());
        assert_eq!(attempts[0].gradient, Derivative::Analytic);
        assert!(attempts[0].disable_fixed_point);
        assert_eq!(plan(&attempts[0]).solver, Solver::Bfgs);
    }

    #[test]
    fn disabled_fallback_hybrid_efs_capability_routes_to_bfgs_primary() {
        // Production Matérn60 exact adaptive regularization at biobank scale:
        // rho_dim=3 retained quadratic penalties, psi_dim=6 adaptive λ/ε
        // coordinates, n_params=9, analytic gradient, and exact outer Hessian
        // cost-gated unavailable. Structurally this is HybridEFS-shaped, but
        // HybridEFS with ψ coordinates is not a standalone primary solver: its
        // ψ backtracking path can legitimately request the first-order escape
        // ladder. If that ladder is disabled, the runner must route the primary
        // attempt directly to BFGS instead of relying on call sites to remember
        // `.with_disable_fixed_point(true)`.
        let trapped_cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 9,
            psi_dim: 6,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&trapped_cap).solver, Solver::HybridEfs);

        let disabled_config = OuterConfig {
            fallback_policy: FallbackPolicy::Disabled,
            ..OuterConfig::default()
        };
        let primary_cap = primary_capability_for_config(
            trapped_cap.clone(),
            &disabled_config,
            "biobank exact adaptive",
        );
        assert!(primary_cap.disable_fixed_point);
        assert_eq!(plan(&primary_cap).solver, Solver::Bfgs);

        let pure_efs_cap = OuterCapability {
            psi_dim: 0,
            ..trapped_cap.clone()
        };
        assert_eq!(plan(&pure_efs_cap).solver, Solver::Efs);
        let pure_primary_cap =
            primary_capability_for_config(pure_efs_cap.clone(), &disabled_config, "pure EFS");
        assert!(!pure_primary_cap.disable_fixed_point);
        assert_eq!(plan(&pure_primary_cap).solver, Solver::Efs);

        let no_gradient_cap = OuterCapability {
            gradient: Derivative::Unavailable,
            ..trapped_cap.clone()
        };
        assert_eq!(plan(&no_gradient_cap).solver, Solver::HybridEfs);
        let no_gradient_primary_cap = primary_capability_for_config(
            no_gradient_cap.clone(),
            &disabled_config,
            "gradient-unavailable hybrid EFS",
        );
        assert!(!no_gradient_primary_cap.disable_fixed_point);
        assert_eq!(plan(&no_gradient_primary_cap).solver, Solver::HybridEfs);

        let automatic_config = OuterConfig::default();
        let automatic_cap = primary_capability_for_config(
            trapped_cap.clone(),
            &automatic_config,
            "biobank exact adaptive",
        );
        assert!(!automatic_cap.disable_fixed_point);
        assert_eq!(plan(&automatic_cap).solver, Solver::HybridEfs);

        let automatic_attempts = automatic_fallback_attempts(&trapped_cap);
        assert!(!automatic_attempts.is_empty());
        assert!(automatic_attempts[0].disable_fixed_point);
        assert_eq!(plan(&automatic_attempts[0]).solver, Solver::Bfgs);
    }

    #[test]
    fn disabled_fallback_hybrid_efs_problem_uses_bfgs_without_calling_efs() {
        let efs_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(9)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_psi_dim(6)
            .with_fallback_policy(FallbackPolicy::Disabled)
            .with_initial_rho(Array1::zeros(9))
            .with_max_iter(5);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * theta.dot(theta),
                    gradient: theta.clone(),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            {
                let efs_calls = Arc::clone(&efs_calls);
                Some(move |_: &mut (), _: &Array1<f64>| {
                    efs_calls.fetch_add(1, Ordering::Relaxed);
                    Err(EstimationError::RemlOptimizationFailed(format!(
                        "{} synthetic biobank adaptive HybridEFS escape",
                        EFS_FIRST_ORDER_FALLBACK_MARKER,
                    )))
                })
            },
        );

        let result = problem
            .run(&mut obj, "disabled fallback marker")
            .expect("disabled-fallback HybridEFS-shaped problem should route directly to BFGS");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(
            efs_calls.load(Ordering::Relaxed),
            0,
            "central primary-capability canonicalization should avoid the EFS hook entirely"
        );
    }

    #[test]
    fn automatic_fallbacks_without_gradient_stop_at_fixed_point_status() {
        for (psi_dim, expected_solver) in [(0, Solver::Efs), (2, Solver::HybridEfs)] {
            let cap = OuterCapability {
                gradient: Derivative::Unavailable,
                hessian: DeclaredHessianForm::Unavailable,
                n_params: 15,
                psi_dim,
                fixed_point_available: true,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            };
            assert_eq!(plan(&cap).solver, expected_solver);
            assert!(
                automatic_fallback_attempts(&cap).is_empty(),
                "gradient-unavailable fixed-point capabilities must not fabricate a BFGS fallback",
            );
        }
    }

    #[test]
    fn automatic_fallbacks_do_not_repeat_arc_when_fixed_point_is_irrelevant() {
        // The contract here is that the cascade does not lateral-hop ARC
        // through the EFS planner arm when `fixed_point_available=true` is
        // incidentally set on an (Analytic, Analytic) capability that the
        // planner already chose ARC for. Combined with the
        // analytic-Hessian-preservation contract enforced by
        // `automatic_fallbacks_preserve_analytic_hessian_for_arc_primary`,
        // the ARC primary now has zero degraded fallbacks — the runner's
        // ARC budget-bump retry ladder owns recovery.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::Arc);

        let attempts = automatic_fallback_attempts(&cap);
        assert!(
            attempts.is_empty(),
            "ARC primary with incidental fixed_point_available must not \
             cascade through the EFS arm or lateral-demote to BFGS",
        );
    }

    #[test]
    fn plan_disable_fixed_point_forces_bfgs_even_when_efs_eligible() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn run_malformed_gradient_seed_surfaces_as_error() {
        // A capability that declares Analytic gradient but returns a malformed
        // one must fail loudly. The previous numerical-gradient fallback masked
        // the underlying bug by silently spinning a cost-only BFGS; that path is
        // disabled in production.
        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_initial_rho(Array1::zeros(2))
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let err = problem
            .run(&mut obj, "test gradient mismatch")
            .expect_err("malformed analytic gradient must surface as error");
        assert!(
            matches!(err, EstimationError::RemlOptimizationFailed(_)),
            "unexpected error variant: {err:?}",
        );
    }

    #[test]
    fn run_bfgs_ignores_malformed_hessian_payload() {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_initial_rho(array![0.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    // First-order paths must ignore Hessian payload quality.
                    hessian: HessianResult::Analytic(array![[f64::NAN, 0.0], [0.0, 1.0]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "bfgs should ignore malformed hessian payload")
            .expect("valid first-order data should be enough for BFGS");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(result.plan_used.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn finite_outer_eval_reports_gradient_length_mismatch() {
        let err = finite_outer_eval_or_error(
            "test gradient mismatch",
            OuterThetaLayout::new(2, 0),
            OuterEval {
                cost: 0.0,
                gradient: Array1::zeros(1),
                hessian: HessianResult::Unavailable,
                inner_beta_hint: None,
            },
        )
        .expect_err("gradient mismatch should be rejected");
        let message = match err {
            ObjectiveEvalError::Recoverable { message } | ObjectiveEvalError::Fatal { message } => {
                message
            }
        };
        assert!(
            message.contains("outer gradient length mismatch"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn run_with_initial_seed_still_considers_generated_candidates() {
        let generated = crate::seeding::generate_rho_candidates(
            1,
            None,
            &crate::seeding::SeedConfig::default(),
        );
        let valid_seed = generated
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let expected_seed = valid_seed.clone();
        let initial_seed = array![9.0];
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(initial_seed)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    if theta == valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(f64::INFINITY)
                    }
                }
            },
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == valid_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: Array1::zeros(1),
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "generated seed should remain reachable")
            .expect("generated seed should still be eligible when an initial seed is provided");
        assert_eq!(result.rho, expected_seed);
    }

    #[test]
    fn run_indefinite_analytic_seed_stays_on_arc() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_initial_rho(array![0.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: array![0.0],
                    hessian: HessianResult::Analytic(array![[-1.0]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "indefinite seed geometry")
            .expect("indefinite analytic seed geometry should stay on the second-order plan");
        assert_eq!(result.plan_used.solver, Solver::Arc);
        assert_eq!(result.plan_used.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn run_seed_materialization_failure_surfaces_arc_error_verbatim() {
        // Under the budget-bump retry ladder (commit c96c4233), an ARC
        // primary with `(Analytic, Analytic)` capability has zero degraded
        // fallbacks. A seed-materialization failure surfaces as `Err`
        // verbatim — there is no lateral demote to BFGS+BfgsApprox that
        // would silently discard the analytic outer Hessian. Materialization
        // failures are deterministic w.r.t. rho, so the budget-bump retry
        // ladder cannot rescue them; the operator returns the same Err on
        // every retry. Hence the runner returns the original Err.
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_initial_rho(array![0.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: array![0.0],
                    hessian: HessianResult::Operator(Arc::new(
                        FailingSeedMaterializationOperator { dim: 1 },
                    )),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let err = problem
            .run(&mut obj, "seed materialization failure")
            .expect_err(
                "ARC primary must surface the materialization failure verbatim — \
                 no lateral demote to BFGS+BfgsApprox",
            );
        let msg = err.to_string();
        assert!(
            msg.contains("seed materialization failed"),
            "error must propagate the underlying materialization message; got: {msg}"
        );
    }

    #[test]
    fn run_nonconverged_arc_stays_on_arc_after_budget_retry_ladder() {
        // When an ARC primary exhausts its iteration budget, the runner
        // reseeds a fresh ARC attempt from the previous attempt's last
        // ρ and trust radius (up to two retries) and uncaps the inner
        // PIRLS cap for the resumed run via the InnerProgressFeedback
        // handle. Retries are gated on attempt-over-attempt `‖g‖`
        // halving so a deterministic-replay trajectory falls through.
        // The objective's analytic outer Hessian is preserved across
        // every attempt — no lateral demote to BFGS+BfgsApprox. After
        // the retries are exhausted (or the gate fires), the runner
        // returns the final `Ok(OuterResult{converged:false})` from
        // the last ARC attempt; the plan stays ARC + Analytic Hessian.
        //
        // We use `cost = x^4`, `grad = 4 x^3`, `hess = 12 x^2` from
        // `initial_rho = [5.0]` with `max_iter = 1`. Newton-style ARC
        // steps on x^4 contract the gradient by ~3× per attempt, so
        // the halving gate passes and both retries proceed; ARC still
        // cannot reach the optimum in three single-iter attempts.
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let (_d, session) = tmp_cache_session("nonconverged-arc-cache");
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_initial_rho(array![5.0])
            .with_max_iter(1)
            .with_cache_session(Arc::clone(&session));
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0].powi(4)),
            |_: &mut (), theta: &Array1<f64>| {
                let x = theta[0];
                Ok(OuterEval {
                    cost: x.powi(4),
                    gradient: array![4.0 * x.powi(3)],
                    hessian: HessianResult::Analytic(array![[12.0 * x.powi(2)]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "nonconverged arc should stay on arc")
            .expect(
                "ARC ladder must surface the last non-converged ARC result rather than \
                 demoting to BFGS+BfgsApprox",
            );
        assert_eq!(
            result.plan_used.solver,
            Solver::Arc,
            "ARC primary must not lateral-demote after budget exhaustion"
        );
        assert_eq!(
            result.plan_used.hessian_source,
            HessianSource::Analytic,
            "analytic outer Hessian must be preserved across the budget-bump retry ladder"
        );
        assert!(
            !result.converged,
            "test fixture is engineered so the ladder cannot converge; \
             converged=true would mean the fixture stopped exercising the ladder"
        );
    }

    #[test]
    fn candidate_selection_prefers_lower_cost_within_same_convergence_class() {
        let nonconverged_hi = OuterResult {
            rho: array![0.0],
            final_value: 9.0,
            iterations: 1,
            final_grad_norm: Some(1.0),
            final_gradient: None,
            final_hessian: None,
            converged: false,
            plan_used: OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
            operator_trust_radius: None,
            operator_stop_reason: None,
        };
        let nonconverged_lo = OuterResult {
            rho: array![1.0],
            final_value: 1.0,
            iterations: 1,
            final_grad_norm: Some(1.0),
            final_gradient: None,
            final_hessian: None,
            converged: false,
            plan_used: OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
            operator_trust_radius: None,
            operator_stop_reason: None,
        };
        let converged = OuterResult {
            rho: array![2.0],
            final_value: 5.0,
            iterations: 1,
            final_grad_norm: Some(0.0),
            final_gradient: None,
            final_hessian: None,
            converged: true,
            plan_used: OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
            operator_trust_radius: None,
            operator_stop_reason: None,
        };

        assert!(candidate_improves_best(&nonconverged_hi, None));
        assert!(candidate_improves_best(
            &nonconverged_lo,
            Some(&nonconverged_hi)
        ));
        assert!(!candidate_improves_best(
            &nonconverged_hi,
            Some(&nonconverged_lo)
        ));
        assert!(candidate_improves_best(&converged, Some(&nonconverged_lo)));
        assert!(!candidate_improves_best(&nonconverged_lo, Some(&converged)));
    }

    #[test]
    fn gaussian_multistart_compares_converged_seed_costs() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 2;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::Gaussian;
        let started = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_max_iter(4);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(if theta[0] < -1.0 { 0.0 } else { 10.0 }),
            {
                let started = Arc::clone(&started);
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    Ok(OuterEval {
                        cost: if theta[0] < -1.0 { 0.0 } else { 10.0 },
                        gradient: array![0.0],
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "Gaussian quality multistart")
            .expect("Gaussian multistart should compare both converged seeds");
        let starts = started.lock().unwrap();
        assert!(
            starts.len() >= 2,
            "Gaussian quality mode should not stop at the first converged seed"
        );
        assert!(
            result.rho[0] < -1.0,
            "lower-cost converged Gaussian seed should win"
        );
        assert_eq!(result.final_value, 0.0);
    }

    #[test]
    fn run_starts_solver_with_direct_startup_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let calls = Arc::clone(&calls);
                move |_: &mut (), theta: &Array1<f64>| {
                    calls.lock().unwrap().push("cost");
                    Ok(theta[0] * theta[0])
                }
            },
            {
                let calls = Arc::clone(&calls);
                move |_: &mut (), theta: &Array1<f64>| {
                    calls.lock().unwrap().push("eval");
                    Ok(OuterEval {
                        cost: theta[0] * theta[0],
                        gradient: array![2.0 * theta[0]],
                        hessian: HessianResult::Analytic(array![[2.0]]),
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        problem
            .run(&mut obj, "solver should start from a direct startup eval")
            .expect("analytic plans should start with a direct full evaluation");
        let calls = calls.lock().unwrap();
        let first_eval_idx = calls
            .iter()
            .position(|call| *call == "eval")
            .expect("solver should eventually request a full eval");
        assert!(
            first_eval_idx == 0,
            "startup should not perform a separate cost-screening pass first: {calls:?}"
        );
    }

    #[test]
    fn run_screening_reorders_expensive_generated_seeds_before_full_startup_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 4;
        seed_config.seed_budget = 2;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::GeneralizedLinear;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
            .last()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let started = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    if theta == valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(1000.0)
                    }
                }
            },
            {
                let valid_seed = valid_seed.clone();
                let started = Arc::clone(&started);
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    if theta == valid_seed {
                        Ok(OuterEval {
                            cost: 0.0,
                            gradient: array![0.0],
                            hessian: HessianResult::Analytic(array![[1.0]]),
                            inner_beta_hint: None,
                        })
                    } else {
                        Ok(OuterEval::infeasible(theta.len()))
                    }
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "screening should reorder expensive seeds")
            .expect("screened startup should reach the best generated seed");
        assert_eq!(result.rho, valid_seed);
        assert_eq!(
            started.lock().unwrap().first().cloned(),
            Some(valid_seed),
            "screening should move the lowest-cost seed to the front before full startup eval",
        );
        assert_eq!(screening_cap.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn initial_rho_with_single_seed_budget_skips_expensive_screening() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 4;
        seed_config.seed_budget = 1;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::GeneralizedLinear;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let screening_calls = Arc::new(AtomicUsize::new(0));
        let initial_seed = array![9.0];
        let started = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_initial_rho(initial_seed.clone())
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let screening_calls = Arc::clone(&screening_calls);
                move |_: &mut (), _theta: &Array1<f64>| {
                    screening_calls.fetch_add(1, Ordering::Relaxed);
                    Ok(0.0)
                }
            },
            {
                let started = Arc::clone(&started);
                let initial_seed = initial_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    if theta == initial_seed {
                        Ok(OuterEval {
                            cost: 0.0,
                            gradient: array![0.0],
                            hessian: HessianResult::Analytic(array![[1.0]]),
                            inner_beta_hint: None,
                        })
                    } else {
                        Ok(OuterEval::infeasible(theta.len()))
                    }
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "initial rho should be authoritative")
            .expect("initial-rho startup should not spend seed-screening solves");
        assert_eq!(result.rho, initial_seed);
        assert_eq!(
            screening_calls.load(Ordering::Relaxed),
            0,
            "explicit initial rho plus seed_budget=1 should skip screening"
        );
        assert_eq!(
            started.lock().unwrap().first().cloned(),
            Some(initial_seed),
            "solver should start from the explicit initial rho"
        );
        assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn run_screening_reorders_bfgs_seeds_before_full_startup_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::Gaussian;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let initial_seed = array![9.0];
        let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let started = Arc::new(Mutex::new(Vec::new()));
        let screening_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_initial_rho(initial_seed)
            .with_screen_initial_rho(true)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let valid_seed = valid_seed.clone();
                let screening_calls = Arc::clone(&screening_calls);
                move |_: &mut (), theta: &Array1<f64>| {
                    screening_calls.fetch_add(1, Ordering::Relaxed);
                    if theta == valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(1000.0)
                    }
                }
            },
            {
                let valid_seed = valid_seed.clone();
                let started = Arc::clone(&started);
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    if theta == valid_seed {
                        Ok(OuterEval {
                            cost: 0.0,
                            gradient: array![0.0],
                            hessian: HessianResult::Unavailable,
                            inner_beta_hint: None,
                        })
                    } else {
                        Ok(OuterEval::infeasible(theta.len()))
                    }
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "BFGS screening should reorder expensive seeds")
            .expect("screened BFGS startup should reach the best generated seed");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(result.rho, valid_seed);
        assert_eq!(
            started.lock().unwrap().first().cloned(),
            Some(valid_seed),
            "BFGS screening should move the lowest-cost seed to the front before full startup eval",
        );
        assert!(
            screening_calls.load(Ordering::Relaxed) > 1,
            "BFGS seed screening should rank candidates with cost-only probes first",
        );
        assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn screening_cap_survives_per_seed_reset_before_proxy_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 3;
        seed_config.seed_budget = 1;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::Gaussian;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let proxy_saw_cap = Arc::new(AtomicBool::new(false));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_max_iter(1);
        let mut obj = problem.build_objective_with_screening_proxy(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[0].abs(),
                    gradient: array![0.0],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            |_: &mut (), theta: &Array1<f64>, _: OuterEvalOrder| {
                Ok(OuterEval {
                    cost: theta[0].abs(),
                    gradient: array![0.0],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            {
                let screening_cap = Arc::clone(&screening_cap);
                Some(move |_: &mut ()| {
                    screening_cap.store(0, Ordering::Relaxed);
                })
            },
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
            {
                let screening_cap = Arc::clone(&screening_cap);
                let proxy_saw_cap = Arc::clone(&proxy_saw_cap);
                move |_: &mut (), theta: &Array1<f64>| {
                    let cap = screening_cap.load(Ordering::Relaxed);
                    if cap > 0 {
                        proxy_saw_cap.store(true, Ordering::Relaxed);
                        Ok(theta[0].abs())
                    } else {
                        Err(EstimationError::RemlOptimizationFailed(
                            "screening proxy ran without an active cap".to_string(),
                        ))
                    }
                }
            },
        );
        problem
            .run(&mut obj, "screening cap reset regression")
            .expect("screening cap should be restored after each per-seed reset");
        assert!(
            proxy_saw_cap.load(Ordering::Relaxed),
            "screening proxy should observe a nonzero cap"
        );
        assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn rank_seeds_cascade_escalates_when_initial_cap_collapses_all() {
        // When every seed's cost is non-finite at the initial screening cap
        // we must NOT jump straight to a fully uncapped re-evaluation on
        // every seed (the original two-stage protocol). Instead the cap
        // should escalate geometrically (initial → 4× → 16× → uncapped),
        // exiting the moment any cap stage produces a finite cost. This
        // test forces a cost function that returns non-finite for cap < 12
        // and finite for cap ≥ 12, then asserts the cascade exits at the
        // 4× stage with a meaningful ranking — never reaching the uncapped
        // pass.
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        seed_config.screen_max_inner_iterations = 3;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let initial_seed = array![5.0];
        let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let max_cap_seen = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_initial_rho(initial_seed.clone())
            .with_screen_initial_rho(true)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let screening_cap = Arc::clone(&screening_cap);
                let max_cap_seen = Arc::clone(&max_cap_seen);
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    let cap = screening_cap.load(Ordering::Relaxed);
                    max_cap_seen.fetch_max(cap, Ordering::Relaxed);
                    // Mimic an inner solver that needs ≥ 12 iterations of
                    // budget to certify a finite cost; below that it returns
                    // a non-finite "could not converge" signal.
                    if cap > 0 && cap < 12 {
                        return Ok(f64::NAN);
                    }
                    if theta == valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(1000.0)
                    }
                }
            },
            {
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    if theta == valid_seed {
                        Ok(OuterEval {
                            cost: 0.0,
                            gradient: array![0.0],
                            hessian: HessianResult::Analytic(array![[1.0]]),
                            inner_beta_hint: None,
                        })
                    } else {
                        Ok(OuterEval::infeasible(theta.len()))
                    }
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        problem
            .run(&mut obj, "cascade should escalate")
            .expect("cascade should reach a finite cost at the 4× cap stage");
        // The cascade is [3, 12, 48, 0]; the 4× stage (cap=12) is the first
        // stage that produces a finite cost, so the cascade must exit there
        // and never escalate to 48 or to the uncapped (0) stage.
        let max_cap = max_cap_seen.load(Ordering::Relaxed);
        assert_eq!(
            max_cap, 12,
            "cascade should stop at the 4× cap stage; observed max cap = {max_cap}"
        );
        assert_eq!(
            screening_cap.load(Ordering::Relaxed),
            0,
            "screening cap must be restored to its previous value after cascade"
        );
    }

    #[test]
    fn run_efs_skips_global_cost_screening() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 6;
        seed_config.seed_budget = 1;
        let screening_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(15)
            .with_gradient(Derivative::Unavailable)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let screening_calls = Arc::clone(&screening_calls);
                move |_: &mut (), _: &Array1<f64>| {
                    screening_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(0.0)
                }
            },
            |_: &mut (), theta: &Array1<f64>| Ok(OuterEval::infeasible(theta.len())),
            None::<fn(&mut ())>,
            Some(|_: &mut (), theta: &Array1<f64>| {
                Ok(EfsEval {
                    cost: 0.0,
                    steps: vec![0.0; theta.len()],
                    beta: None,
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                })
            }),
        );
        problem
            .run(
                &mut obj,
                "EFS should not use a separate global cost-screening pass",
            )
            .expect("first generated EFS seed should be sufficient");
        assert_eq!(
            screening_calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "EFS startup should not call eval_cost just to screen seeds"
        );
    }

    #[test]
    fn run_efs_skips_invalid_leading_seed_without_spending_budget() {
        let generated = crate::seeding::generate_rho_candidates(
            15,
            None,
            &crate::seeding::SeedConfig::default(),
        );
        let valid_seed = generated
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let invalid_seed = Array1::from_elem(15, 9.0);
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(15)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(invalid_seed)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), theta: &Array1<f64>| Ok(OuterEval::infeasible(theta.len())),
            None::<fn(&mut ())>,
            {
                let valid_seed = valid_seed.clone();
                Some(move |_: &mut (), theta: &Array1<f64>| {
                    if theta == valid_seed {
                        Ok(EfsEval {
                            cost: 0.0,
                            steps: vec![0.0; theta.len()],
                            beta: None,
                            psi_gradient: None,
                            psi_indices: None,
                            inner_hessian_scale: None,
                        })
                    } else {
                        Err(EstimationError::RemlOptimizationFailed(
                            "invalid EFS seed".to_string(),
                        ))
                    }
                })
            },
        );
        let result = problem
            .run(&mut obj, "efs generated seed should remain reachable")
            .expect("invalid startup seeds should not consume the only EFS seed slot");
        assert_eq!(result.rho, valid_seed);
        assert_eq!(result.plan_used.solver, Solver::Efs);
    }

    #[test]
    fn run_efs_runtime_fallback_marker_degrades_to_bfgs_immediately() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 2;
        let efs_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(12)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(Array1::zeros(12))
            .with_max_iter(5);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * theta.dot(theta),
                    gradient: theta.clone(),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            {
                let efs_calls = Arc::clone(&efs_calls);
                Some(move |_: &mut (), _: &Array1<f64>| {
                    efs_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Err(EstimationError::RemlOptimizationFailed(format!(
                        "{} synthetic runtime escape hatch",
                        EFS_FIRST_ORDER_FALLBACK_MARKER,
                    )))
                })
            },
        );
        let result = problem
            .run(&mut obj, "efs runtime fallback marker")
            .expect("runtime EFS escape hatch should degrade to BFGS");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(
            efs_calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "runtime fallback marker should abort the EFS attempt immediately"
        );
    }

    #[test]
    fn run_rejects_invalid_theta_layout() {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_psi_dim(2)
            .with_initial_rho(Array1::zeros(1))
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let err = problem
            .run(&mut obj, "test invalid layout")
            .expect_err("invalid theta layout should fail cleanly");
        assert!(
            err.to_string().contains("invalid outer theta layout"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn effective_seed_budget_caps_expensive_solver_retries() {
        assert_eq!(
            effective_seed_budget(
                4,
                Solver::Efs,
                crate::seeding::SeedRiskProfile::GeneralizedLinear,
                false,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                4,
                Solver::HybridEfs,
                crate::seeding::SeedRiskProfile::Survival,
                false,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                3,
                Solver::Arc,
                crate::seeding::SeedRiskProfile::GeneralizedLinear,
                true,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                3,
                Solver::Arc,
                crate::seeding::SeedRiskProfile::Survival,
                false,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                3,
                Solver::Bfgs,
                crate::seeding::SeedRiskProfile::Survival,
                false,
            ),
            3
        );
    }

    // ─── Gated SolverClass / CompassSearch dispatch ──────────────────────

    fn aux_cap_unavailable(n_params: usize) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    #[test]
    fn plan_with_class_primary_is_identical_to_plan_for_unavailable_grad() {
        // Unavailable-grad capability must still route to BFGS for the
        // primary class (the runner will then hard-error on the missing
        // analytic gradient — that behavior is tested elsewhere).
        let cap = aux_cap_unavailable(3);
        assert_eq!(plan_with_class(&cap, SolverClass::Primary), plan(&cap));
    }

    #[test]
    fn plan_with_class_aux_unavailable_routes_to_compass_search() {
        let cap = aux_cap_unavailable(3);
        let p = plan_with_class(&cap, SolverClass::AuxiliaryGradientFree);
        assert_eq!(p.solver, Solver::CompassSearch);
    }

    #[test]
    fn plan_with_class_aux_analytic_grad_defers_to_primary_plan() {
        // Aux class + analytic gradient is a misuse: the caller should
        // have used Primary. We defer to the standard plan so the caller
        // still gets a well-formed result rather than silently being
        // routed to direct search when a derivative-based solver exists.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan_with_class(&cap, SolverClass::AuxiliaryGradientFree);
        assert_eq!(p.solver, Solver::Arc);
    }

    #[test]
    fn plan_with_class_aux_efs_eligible_defers_to_primary() {
        // If the coordinate structure is EFS-eligible, use EFS even if
        // the caller set Auxiliary — EFS is strictly better than compass
        // search whenever it applies.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 12,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan_with_class(&cap, SolverClass::AuxiliaryGradientFree);
        assert_eq!(p.solver, Solver::Efs);
    }

    #[test]
    fn automatic_fallback_never_includes_compass_search() {
        // The fallback cascade must not introduce direct-search for the
        // primary REML path. Aux direct-search is a single-attempt
        // method; its dispatch is orthogonal to the fallback ladder.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 5,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let attempts = automatic_fallback_attempts(&cap);
        for attempt_cap in &attempts {
            let p = plan_with_class(attempt_cap, SolverClass::Primary);
            assert_ne!(p.solver, Solver::CompassSearch);
        }
    }

    #[test]
    fn compass_search_budget_accounts_for_single_seed() {
        // Aux direct-search is intrinsically a single-seed local method;
        // generating extra seeds would just duplicate cost.
        let b = effective_seed_budget(
            8,
            Solver::CompassSearch,
            crate::seeding::SeedRiskProfile::Survival,
            false,
        );
        assert_eq!(b, 1);
    }

    #[test]
    fn run_aux_compass_projects_seed_before_seed_cost() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_solver_class(SolverClass::AuxiliaryGradientFree)
            .with_bounds(array![0.0], array![1.0])
            .with_initial_rho(array![2.0])
            .with_max_iter(64);
        let mut obj = problem.build_objective(
            (),
            {
                let seen = Arc::clone(&seen);
                move |_: &mut (), theta: &Array1<f64>| {
                    seen.lock().unwrap().push(theta.clone());
                    Ok((theta[0] - 2.0).powi(2))
                }
            },
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput(
                    "aux direct-search test should not call eval".to_string(),
                ))
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "aux direct-search seed projection")
            .expect("aux direct-search should evaluate the projected seed");
        assert_eq!(result.plan_used.solver, Solver::CompassSearch);
        assert_eq!(result.rho, array![1.0]);
        assert_eq!(result.final_value, 1.0);
        assert_eq!(
            seen.lock().unwrap().first().cloned(),
            Some(array![1.0]),
            "aux direct-search must project the seed before evaluating its cost",
        );
    }

    #[test]
    fn run_arc_projects_seed_before_seed_validation_eval() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 1;
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_bounds(array![0.0], array![1.0])
            .with_initial_rho(array![2.0])
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
            {
                let seen = Arc::clone(&seen);
                move |_: &mut (), theta: &Array1<f64>| {
                    seen.lock().unwrap().push(theta.clone());
                    Ok(OuterEval {
                        cost: (theta[0] - 0.25).powi(2),
                        gradient: array![2.0 * (theta[0] - 0.25)],
                        hessian: HessianResult::Analytic(array![[2.0]]),
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        problem
            .run(&mut obj, "arc seed projection")
            .expect("arc should evaluate the projected seed");
        assert_eq!(
            seen.lock().unwrap().first().cloned(),
            Some(array![1.0]),
            "Arc must project the seed before validating the initial sample",
        );
    }

    #[test]
    fn run_bfgs_projects_seed_before_seed_validation_eval() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 1;
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_bounds(array![0.0], array![1.0])
            .with_initial_rho(array![2.0])
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
            {
                let seen = Arc::clone(&seen);
                move |_: &mut (), theta: &Array1<f64>| {
                    seen.lock().unwrap().push(theta.clone());
                    Ok(OuterEval {
                        cost: (theta[0] - 0.25).powi(2),
                        gradient: array![2.0 * (theta[0] - 0.25)],
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        problem
            .run(&mut obj, "bfgs seed projection")
            .expect("BFGS should evaluate the projected seed");
        assert_eq!(
            seen.lock().unwrap().first().cloned(),
            Some(array![1.0]),
            "BFGS must project the seed before validating the initial sample",
        );
    }

    fn tmp_cache_session(label: &str) -> (tempfile::TempDir, Arc<CacheSession>) {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::cache::WarmStartStore::open(
            dir.path().to_path_buf(),
            crate::cache::StoreOptions {
                size_budget_bytes: 1024 * 1024,
                ttl: std::time::Duration::from_secs(60),
            },
        )
        .unwrap();
        let mut fp = crate::cache::Fingerprinter::new();
        fp.absorb_str(b"outer-test", label);
        let key = fp.finalize();
        (dir, Arc::new(CacheSession::open(store, key)))
    }

    #[test]
    fn checkpointing_objective_persists_finite_evals() {
        let (_d, session) = tmp_cache_session("ckpt-persist");
        let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
        let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput("eval not used".into()))
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
        // Initial: nothing on disk.
        assert!(session.try_load().is_none());
        // First eval persists.
        let v0 = wrapped.eval_cost(&array![3.0]).unwrap();
        assert!((v0 - 9.0).abs() < 1e-12);
        let on_disk = session.try_load().expect("first eval should checkpoint");
        let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
        assert!((payload.cost - 9.0).abs() < 1e-12);
        assert_eq!(payload.rho, vec![3.0]);
        // Strictly improving eval must bypass the 2-second rate limit.
        let _v1 = wrapped.eval_cost(&array![0.5]).unwrap();
        let on_disk = session
            .try_load()
            .expect("improving eval should checkpoint");
        let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
        assert!((payload.cost - 0.25).abs() < 1e-12);
        assert_eq!(payload.rho, vec![0.5]);
        // Non-finite values must not corrupt the on-disk best-known iterate.
        let v_inf = wrapped.eval_cost(&array![f64::NAN]);
        assert!(v_inf.is_ok() || v_inf.is_err());
        let on_disk = session.try_load().expect("prior best preserved");
        let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
        assert!((payload.cost - 0.25).abs() < 1e-12);
    }

    #[test]
    fn checkpointing_objective_rejects_wrong_dim_on_decode() {
        // A payload from a 3-dim fit is invalid input for a 5-dim resume.
        let bytes = encode_iterate(&array![1.0, 2.0, 3.0], None, 0.5, 0).expect("encode");
        assert!(decode_iterate(&bytes, 3).is_some());
        assert!(decode_iterate(&bytes, 5).is_none());
    }

    #[test]
    fn iterate_payload_v1_entries_orphaned_by_schema_bump() {
        // Pre-2026-05 the outer-iterate payload stored ρ-only with schema=1.
        // Loading such an entry into a binary that ships the v2 (ρ,β) schema
        // returned a ρ-only seed and forced the inner solve to reconstruct β
        // from cold start; for saturated ρ (|ρ_i| near rho_bound) this gave
        // inner-Hessian κ ≈ e^{2·rho_bound} and Newton degraded to O(1/k)
        // descent, exhausting the cycle budget before KKT. Bumping the
        // schema 1 → 2 makes those poisoned entries unreadable so the next
        // run cold-starts cleanly.
        #[derive(serde::Serialize)]
        struct LegacyV1Payload {
            schema: u32,
            rho: Vec<f64>,
            cost: f64,
            eval_id: u64,
        }
        let v1_bytes = serde_json::to_vec(&LegacyV1Payload {
            schema: 1,
            rho: vec![10.0; 8],
            cost: 3.459864e5,
            eval_id: 3,
        })
        .expect("encode v1");
        assert!(
            decode_iterate(&v1_bytes, 8).is_none(),
            "v1 payloads must be rejected so the runner falls through to cold start",
        );
    }

    #[test]
    fn iterate_payload_round_trips_beta() {
        // Every persisted entry that comes with an inner-β hint round-trips
        // (ρ, β) together — that pair lets a resume open inner PIRLS in the
        // basin of quadratic attraction regardless of where ρ sits.
        let rho = array![10.0, -10.0, 5.0];
        let beta = array![0.12, -0.34, 0.56, 7.89];
        let bytes = encode_iterate(&rho, Some(&beta), 1.0, 7).expect("encode");
        let decoded = decode_iterate(&bytes, rho.len()).expect("decode");
        assert_eq!(decoded.rho, rho.to_vec());
        assert_eq!(decoded.beta, beta.to_vec());
        // ρ-only writes (β = None) still encode but with an empty beta slot.
        let ro_bytes = encode_iterate(&rho, None, 1.0, 7).expect("encode-rho-only");
        let ro = decode_iterate(&ro_bytes, rho.len()).expect("decode-rho-only");
        assert!(ro.beta.is_empty());
    }

    #[test]
    fn note_persists_inner_beta_hint_from_eval() {
        // Write-side proof of the principled fix: when the inner solver
        // surfaces β via OuterEval::inner_beta_hint, CheckpointingObjective
        // captures it on every accepted eval AND exposes it for finalize.
        let (_d, session) = tmp_cache_session("note-persists-beta");
        let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
        let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(1.0),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: Some(array![1.5, 2.5, 3.5]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
        drop(wrapped.eval(&array![0.5]).expect("eval ok"));
        let on_disk = session
            .try_load()
            .expect("eval with finite β must persist a (ρ,β) checkpoint");
        let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
        assert_eq!(payload.beta, vec![1.5, 2.5, 3.5]);
        let captured = wrapped.last_inner_beta().expect("β was captured");
        assert_eq!(captured.to_vec(), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn note_rejects_nonfinite_inner_beta() {
        // A divergent inner state must NOT poison the cache: persisting a
        // non-finite β would re-create the failure mode the schema bump fixed.
        let (_d, session) = tmp_cache_session("note-rejects-bad-beta");
        let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
        let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(1.0),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: Some(array![f64::NAN, 0.5]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
        drop(wrapped.eval(&array![0.5]).expect("eval ok"));
        assert!(
            session.try_load().is_none(),
            "non-finite β must abort the checkpoint write, not poison the cache",
        );
        assert!(
            wrapped.last_inner_beta().is_none(),
            "non-finite β must not be exposed via last_inner_beta()",
        );
    }

    #[test]
    fn classify_extracts_beta_from_v2_payload() {
        // The classifier propagates `beta` from the v2 payload onto its
        // Seed/ExactFinal decisions so the dispatcher can hand it to
        // OuterObjective::seed_inner_state. Without this, the (ρ, β) payload
        // would write β but never resurface it on resume.
        let rho = array![1.0, 2.0];
        let beta = array![10.0, 20.0, 30.0];
        let payload = encode_iterate(&rho, Some(&beta), 1.0, 0).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(0),
                kind: crate::cache::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Preloaded,
        };
        let CacheSeedDecision::Seed {
            beta: decoded_beta, ..
        } = classify_cache_entry_for_outer(&loaded, 2)
        else {
            panic!("expected Seed decision");
        };
        assert_eq!(decoded_beta, beta.to_vec());

        // ρ-only payload (legacy or family-without-β) decodes to empty beta.
        let payload = encode_iterate(&rho, None, 1.0, 0).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(0),
                kind: crate::cache::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Preloaded,
        };
        let CacheSeedDecision::Seed {
            beta: decoded_beta, ..
        } = classify_cache_entry_for_outer(&loaded, 2)
        else {
            panic!("expected Seed decision");
        };
        assert!(
            decoded_beta.is_empty(),
            "ρ-only payload must produce an empty beta so the dispatcher skips seed_inner_state"
        );
    }

    #[test]
    fn run_calls_seed_inner_state_with_cached_beta() {
        // End-to-end read-side wiring: a cache hit carrying β must call
        // OuterObjective::seed_inner_state(&beta) *before* the first BFGS
        // eval. We verify this by routing through a custom OuterObjective
        // that records the β it was seeded with.
        struct RecordingObj {
            seeded: Arc<Mutex<Option<Array1<f64>>>>,
            eval_count: Arc<Mutex<usize>>,
        }
        impl OuterObjective for RecordingObj {
            fn capability(&self) -> OuterCapability {
                // Analytic gradient AND analytic Hessian so the planner picks
                // the same Hessian-bearing path a real fit takes; using
                // Unavailable here would test a degenerate plan.
                OuterCapability {
                    gradient: Derivative::Analytic,
                    hessian: DeclaredHessianForm::Dense,
                    n_params: 2,
                    psi_dim: 0,
                    fixed_point_available: false,
                    barrier_config: None,
                    prefer_gradient_only: false,
                    disable_fixed_point: false,
                }
            }
            fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
                Ok(theta.dot(theta))
            }
            fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
                *self.eval_count.lock().unwrap() += 1;
                // f(θ) = ‖θ‖² → ∇f = 2θ, ∇²f = 2I.
                Ok(OuterEval {
                    cost: theta.dot(theta),
                    gradient: 2.0 * theta,
                    hessian: HessianResult::Analytic(2.0 * Array2::<f64>::eye(theta.len())),
                    inner_beta_hint: None,
                })
            }
            fn reset(&mut self) {}
            fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<(), EstimationError> {
                *self.seeded.lock().unwrap() = Some(beta.clone());
                Ok(())
            }
        }

        let (_d, session) = tmp_cache_session("seed-inner-state-call");
        let bytes = encode_iterate(&array![1.0, 2.0], Some(&array![7.5, 8.5, 9.5]), 5.0, 3)
            .expect("encode");
        session.checkpoint(&bytes, Some(5.0), Some(3));

        let seeded: Arc<Mutex<Option<Array1<f64>>>> = Arc::new(Mutex::new(None));
        let eval_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let mut obj = RecordingObj {
            seeded: Arc::clone(&seeded),
            eval_count: Arc::clone(&eval_count),
        };

        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_max_iter(1)
            .with_cache_session(Arc::clone(&session));
        drop(problem.run(&mut obj, "seed-inner-state-call"));

        let observed = seeded.lock().unwrap().clone();
        assert_eq!(
            observed,
            Some(array![7.5, 8.5, 9.5]),
            "dispatcher must call seed_inner_state with the cached β before run_outer",
        );
    }

    #[test]
    fn run_skips_seed_inner_state_when_payload_has_no_beta() {
        // Symmetric guard: a ρ-only cache entry must NOT invoke
        // seed_inner_state — calling it with an empty / zero / garbage β
        // would silently degrade a family that has a non-trivial inner
        // default into one started at zeros.
        struct CountingObj {
            seed_calls: Arc<Mutex<usize>>,
        }
        impl OuterObjective for CountingObj {
            fn capability(&self) -> OuterCapability {
                // Analytic gradient AND analytic Hessian so the planner picks
                // the same Hessian-bearing path a real fit takes; using
                // Unavailable here would test a degenerate plan.
                OuterCapability {
                    gradient: Derivative::Analytic,
                    hessian: DeclaredHessianForm::Dense,
                    n_params: 2,
                    psi_dim: 0,
                    fixed_point_available: false,
                    barrier_config: None,
                    prefer_gradient_only: false,
                    disable_fixed_point: false,
                }
            }
            fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
                Ok(theta.dot(theta))
            }
            fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
                // f(θ) = ‖θ‖² → ∇f = 2θ, ∇²f = 2I.
                Ok(OuterEval {
                    cost: theta.dot(theta),
                    gradient: 2.0 * theta,
                    hessian: HessianResult::Analytic(2.0 * Array2::<f64>::eye(theta.len())),
                    inner_beta_hint: None,
                })
            }
            fn reset(&mut self) {}
            fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<(), EstimationError> {
                *self.seed_calls.lock().unwrap() += beta.len().max(1);
                Ok(())
            }
        }

        let (_d, session) = tmp_cache_session("seed-inner-state-skip");
        // ρ-only payload — no β.
        let bytes = encode_iterate(&array![1.0, 2.0], None, 5.0, 3).expect("encode");
        session.checkpoint(&bytes, Some(5.0), Some(3));

        let seed_calls: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let mut obj = CountingObj {
            seed_calls: Arc::clone(&seed_calls),
        };

        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_max_iter(1)
            .with_cache_session(Arc::clone(&session));
        drop(problem.run(&mut obj, "seed-inner-state-skip"));

        assert_eq!(
            *seed_calls.lock().unwrap(),
            0,
            "seed_inner_state must not fire when the cached payload carries no β",
        );
    }

    #[test]
    fn cache_entry_classifier_honors_finite_seeds_regardless_of_saturation() {
        // The classifier no longer reshapes ρ based on shape. Any finite,
        // correctly-dimensioned payload is honored as the next run's seed.
        // Boundary-saturated entries written under the v2 (ρ, β) invariant
        // are a *legitimate* finding — the smoothness wants to be near-null
        // — and the persisted β puts the next inner solve at zero-gradient,
        // making the cold-β failure mode impossible to re-create from cache.
        for rho_seed in [array![9.0, 0.0], array![10.0, -10.0], array![-10.0, 10.0]] {
            let payload = encode_iterate(&rho_seed, None, 1.0, 0).expect("encode");
            let loaded = crate::cache::LoadedEntry {
                entry: crate::cache::CachedEntry {
                    payload,
                    objective: Some(1.0),
                    iteration: Some(0),
                    kind: crate::cache::EntryKind::Checkpoint,
                    written_unix_secs: 0,
                },
                source: crate::cache::LoadSource::Preloaded,
            };

            assert!(cache_entry_would_help_outer(&loaded, 2));
            let CacheSeedDecision::Seed { rho, .. } = classify_cache_entry_for_outer(&loaded, 2)
            else {
                panic!(
                    "finite seed {:?} must be honored unchanged; the read-side clamp / \
                     all-saturated-discard branches were band-aids over the missing β cache",
                    rho_seed
                );
            };
            assert_eq!(rho, rho_seed, "ρ must round-trip without reshaping");
        }
    }

    #[test]
    fn cache_entry_classifier_rejects_only_structural_failures() {
        // Only structural failures discard: payload shape (wrong rho_dim,
        // non-finite payload internals → decode None → "payload-shape-mismatch")
        // and non-finite cache metadata → "non-finite-payload". Saturation,
        // β presence, and schema-1 mismatches are NOT discards here:
        // saturation is honored under v2; schema-1 returns None from decode
        // and surfaces as "payload-shape-mismatch" via the decode rejection.

        // Non-finite metadata objective: decode succeeds (finite payload
        // cost), but the entry-level objective is NaN — discard as
        // non-finite-payload.
        let payload = encode_iterate(&array![0.5, 0.5], None, 1.0, 0).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(f64::NAN),
                iteration: Some(0),
                kind: crate::cache::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Preloaded,
        };
        assert!(matches!(
            classify_cache_entry_for_outer(&loaded, 2),
            CacheSeedDecision::Discard {
                reason: "non-finite-payload",
                ..
            }
        ));

        // Dimension mismatch: 2-D payload viewed as a 3-D problem → decode
        // rejects shape → "payload-shape-mismatch".
        let payload = encode_iterate(&array![0.5, 0.5], None, 1.0, 0).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(0),
                kind: crate::cache::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Preloaded,
        };
        assert!(matches!(
            classify_cache_entry_for_outer(&loaded, 3),
            CacheSeedDecision::Discard {
                reason: "payload-shape-mismatch",
                ..
            }
        ));
    }

    #[test]
    fn exact_final_cache_hit_is_helpful_even_at_boundary() {
        let payload = encode_iterate(&array![10.0, -10.0], None, 1.0, 3).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(3),
                kind: crate::cache::EntryKind::Final,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Exact,
        };

        assert!(cache_entry_would_help_outer(&loaded, 2));
        assert!(matches!(
            classify_cache_entry_for_outer(&loaded, 2),
            CacheSeedDecision::ExactFinal { iterations: 3, .. }
        ));
    }

    #[test]
    fn checkpointing_objective_mirrors_checkpoints() {
        let (_primary_dir, primary) = tmp_cache_session("ckpt-primary");
        let (_mirror_dir, mirror) = tmp_cache_session("ckpt-mirror");
        let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
        let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput("eval not used".into()))
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut wrapped = CheckpointingObjective::new(
            &mut inner,
            Arc::clone(&primary),
            vec![Arc::clone(&mirror)],
        );

        let value = wrapped.eval_cost(&array![4.0]).unwrap();
        assert_eq!(value, 16.0);

        let primary_payload =
            decode_iterate(&primary.try_load().expect("primary checkpoint").payload, 1)
                .expect("primary decode");
        let mirror_payload =
            decode_iterate(&mirror.try_load().expect("mirror checkpoint").payload, 1)
                .expect("mirror decode");
        assert_eq!(primary_payload.rho, vec![4.0]);
        assert_eq!(mirror_payload.rho, vec![4.0]);
        assert_eq!(primary_payload.cost, mirror_payload.cost);
    }

    #[test]
    fn cached_rho_is_prepended_as_first_seed() {
        // Whitebox: pre-seed the session with a known iterate, then run
        // an OuterProblem with a deliberately-different `initial_rho`.
        // The runner must visit the cached rho before the configured
        // `initial_rho` because `try_load` overrode it.
        let (_d, session) = tmp_cache_session("seed-prepend");
        // Hand-write the cached checkpoint: rho = [2.5], cost = 0.25.
        // Final exact hits return immediately; checkpoints still exercise the
        // regular seed-prepend path.
        let payload = encode_iterate(&array![2.5], None, 0.25, 0).expect("encode");
        session.checkpoint(&payload, Some(0.25), Some(0));
        assert!(
            session.try_load().is_some(),
            "precondition: cache populated"
        );

        let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
        // Use the AuxiliaryGradientFree class so a no-gradient problem
        // routes to compass search (`run_aux_compass_projects_seed_before_seed_cost`
        // above uses the same pattern). Bounds must contain the cached rho
        // so the projector doesn't snap it away.
        let problem = OuterProblem::new(1)
            .with_solver_class(SolverClass::AuxiliaryGradientFree)
            .with_bounds(array![-5.0], array![5.0])
            .with_initial_rho(array![-3.0]) // deliberately not 2.5
            .with_max_iter(8)
            .with_cache_session(Arc::clone(&session));
        let mut obj = problem.build_objective(
            seen.clone(),
            |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok((theta[0] - 2.5).powi(2))
            },
            |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, _: &Array1<f64>| {
                Err(EstimationError::InvalidInput("eval not used".into()))
            },
            None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
            None::<
                fn(
                    &mut Arc<Mutex<Vec<Array1<f64>>>>,
                    &Array1<f64>,
                ) -> Result<EfsEval, EstimationError>,
            >,
        );
        drop(problem.run(&mut obj, "seed-prepend"));
        // The cached rho (2.5) must appear in the eval trace, and it must
        // appear no later than the configured initial_rho (−3.0). Both
        // are inside the bounds so the projector cannot rewrite them.
        let evals = seen.lock().unwrap();
        let pos_cached = evals.iter().position(|r| (r[0] - 2.5).abs() < 1e-9);
        let pos_initial = evals.iter().position(|r| (r[0] + 3.0).abs() < 1e-9);
        assert!(
            pos_cached.is_some(),
            "cached rho must be evaluated; saw {:?}",
            *evals
        );
        if let (Some(c), Some(i)) = (pos_cached, pos_initial) {
            assert!(
                c <= i,
                "cached rho (idx {c}) must precede initial_rho (idx {i})",
            );
        }
    }

    #[test]
    fn all_saturated_cached_rho_is_honored_as_seed() {
        // Inverse of the prior `all_saturated_cached_rho_is_discarded_before_seed_validation`
        // test. Under v1 the cache stored ρ-only, so resuming at boundary ρ
        // forced PIRLS to cold-start β against a Hessian with condition
        // number `≈ e^{2·rho_bound}` — Newton degraded to O(1/k) descent
        // that exhausted the cycle budget. The "discard if all-saturated"
        // branch was a read-side band-aid; it suppressed a legitimate
        // resume signal in exchange for tolerating the broken contract.
        //
        // Under v2 the iterate payload carries (ρ, β). When β is persisted
        // alongside boundary ρ the next inner solve opens at zero gradient,
        // and the conditioning is no longer a barrier. Therefore the
        // classifier no longer reshapes ρ based on saturation: every
        // finite, correctly-dimensioned entry is used as the seed. This
        // test pins that contract.
        let (_d, session) = tmp_cache_session("all-saturated-honored");
        let payload = encode_iterate(&array![10.0, -10.0], None, 1.0, 0).expect("encode");
        session.checkpoint(&payload, Some(1.0), Some(0));
        assert!(
            session.try_load().is_some(),
            "precondition: cache populated"
        );

        let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 4;
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(array![0.0, 0.0])
            .with_rho_bound(10.0)
            .with_max_iter(1)
            .with_cache_session(Arc::clone(&session));

        let mut obj = problem.build_objective(
            seen.clone(),
            |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| Ok(theta.dot(theta)),
            |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: theta.dot(theta),
                    gradient: theta.clone(),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
            None::<
                fn(
                    &mut Arc<Mutex<Vec<Array1<f64>>>>,
                    &Array1<f64>,
                ) -> Result<EfsEval, EstimationError>,
            >,
        );

        drop(problem.run(&mut obj, "all-saturated-honored"));
        let evals = seen.lock().unwrap();
        assert!(
            evals.iter().any(|rho| rho == array![10.0, -10.0]),
            "cached saturated ρ must be evaluated unchanged under v2 (ρ, β) invariant; saw {:?}",
            *evals
        );
    }

    #[test]
    fn exact_final_cache_hit_skips_outer_validation() {
        let (_d, session) = tmp_cache_session("final-skip");
        let payload = encode_iterate(&array![2.5], None, 0.25, 7).expect("encode");
        session.finalize(&payload, Some(0.25), Some(7));

        let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_solver_class(SolverClass::AuxiliaryGradientFree)
            .with_bounds(array![-5.0], array![5.0])
            .with_initial_rho(array![-3.0])
            .with_max_iter(8)
            .with_cache_session(Arc::clone(&session));
        let mut obj = problem.build_objective(
            seen.clone(),
            |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok((theta[0] - 2.5).powi(2))
            },
            |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, _: &Array1<f64>| {
                Err(EstimationError::InvalidInput("eval not used".into()))
            },
            None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
            None::<
                fn(
                    &mut Arc<Mutex<Vec<Array1<f64>>>>,
                    &Array1<f64>,
                ) -> Result<EfsEval, EstimationError>,
            >,
        );

        let result = problem
            .run(&mut obj, "final-skip")
            .expect("final exact hit should return cached outer result");
        assert_eq!(result.rho, array![2.5]);
        assert_eq!(result.final_value, 0.25);
        assert_eq!(result.iterations, 7);
        assert!(result.converged);
        assert!(
            seen.lock().unwrap().is_empty(),
            "exact final hit should not evaluate the outer objective"
        );
    }
}
