use crate::cache::{LoadSource, Session as CacheSession};

use crate::estimate::EstimationError;

use crate::solver::estimate::reml::unified::BarrierConfig;

use crate::solver::priority_selection::{
    PriorityBudgetStage, PriorityStageSummary, rank_indices_with_budget_cascade,
};

use crate::solver::startup_stats::{
    SeedRejection, StartupStats, format_no_seeds_passed, uniform_structural_key,
};

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

use std::sync::Mutex;

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};


const OPERATOR_TRUST_RESTART_RADIUS_FLOOR: f64 = 1.0e-6;


fn outer_strategy_contract_panic(message: impl Into<String>) -> ! {
    std::panic::panic_any(message.into())
}


/// Per-θ component of an outer-gradient finite-difference audit.
#[derive(Clone, Debug)]
pub struct OuterGradientFdComponent {
    /// Human label for the block this coordinate belongs to (e.g. "timewiggle").
    pub block: String,
    /// Flat θ index.
    pub index: usize,
    /// Analytic ∂V/∂θ_i returned by the family evaluator.
    pub analytic: f64,
    /// Central finite-difference of the outer criterion in θ_i.
    pub fd: f64,
}


impl OuterGradientFdComponent {
    /// Absolute analytic−FD gap.
    pub fn abs_gap(&self) -> f64 {
        (self.analytic - self.fd).abs()
    }
    /// analytic/fd ratio (None when fd≈0). A clean −1 signals a sign
    /// convention; a stable constant ≠1 signals a dropped/extra additive term.
    pub fn ratio(&self) -> Option<f64> {
        if self.fd.abs() > 1e-12 {
            Some(self.analytic / self.fd)
        } else {
            None
        }
    }
}


/// Result of a component-by-component finite-difference audit of an outer
/// REML/LAML gradient at a fixed θ, plus the outer-Hessian eigenvalues.
///
/// This is the discriminating diagnostic that forks the two failure modes of a
/// non-terminating outer loop: an **objective↔gradient desync** (analytic ≠ FD
/// on some component → the trust region chases a phantom descent direction
/// forever) versus **weak identifiability** (analytic ≈ FD everywhere but a
/// near-zero outer-Hessian eigenvalue → a genuinely flat valley the optimizer
/// crawls along). It is family-agnostic: any path that exposes an outer
/// evaluator closure `θ ↦ (V, ∇V, H)` can call it.
#[derive(Clone, Debug)]
pub struct OuterGradientFdAudit {
    /// Outer criterion value at θ₀.
    pub value: f64,
    /// Per-coordinate analytic-vs-FD comparison.
    pub components: Vec<OuterGradientFdComponent>,
    /// Eigenvalues of the (symmetrized) outer Hessian at θ₀, ascending. Empty
    /// when no analytic/operator Hessian was available.
    pub hessian_eigenvalues: Vec<f64>,
}


impl OuterGradientFdAudit {
    /// Per-block L2 norm of the analytic gradient.
    pub fn analytic_block_norms(&self) -> Vec<(String, f64)> {
        let mut order: Vec<String> = Vec::new();
        let mut acc: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for c in &self.components {
            if !acc.contains_key(&c.block) {
                order.push(c.block.clone());
            }
            *acc.entry(c.block.clone()).or_insert(0.0) += c.analytic * c.analytic;
        }
        order
            .into_iter()
            .map(|b| {
                let v = acc.get(&b).copied().unwrap_or(0.0).sqrt();
                (b, v)
            })
            .collect()
    }

    /// Worst per-coordinate analytic−FD gap and its component.
    pub fn worst_component(&self) -> Option<&OuterGradientFdComponent> {
        self.components.iter().max_by(|a, b| {
            a.abs_gap()
                .partial_cmp(&b.abs_gap())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Smallest-magnitude outer-Hessian eigenvalue (flatness proxy).
    pub fn min_abs_eigenvalue(&self) -> Option<f64> {
        self.hessian_eigenvalues
            .iter()
            .map(|e| e.abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Emit a single human-readable verdict block to the log.
    pub fn log_verdict(&self, context: &str) {
        log::warn!("[OUTER-FD-AUDIT/{context}] value={:.6e}", self.value);
        for (block, norm) in self.analytic_block_norms() {
            log::warn!("[OUTER-FD-AUDIT/{context}] block={block} |g_analytic|={norm:.6e}");
        }
        for c in &self.components {
            let ratio = c
                .ratio()
                .map(|r| format!("{r:.4}"))
                .unwrap_or_else(|| "n/a".to_string());
            log::warn!(
                "[OUTER-FD-AUDIT/{context}] block={} i={} analytic={:.6e} fd={:.6e} gap={:.3e} ratio={}",
                c.block,
                c.index,
                c.analytic,
                c.fd,
                c.abs_gap(),
                ratio
            );
        }
        if !self.hessian_eigenvalues.is_empty() {
            let evs: Vec<String> = self
                .hessian_eigenvalues
                .iter()
                .map(|e| format!("{e:.4e}"))
                .collect();
            log::warn!(
                "[OUTER-FD-AUDIT/{context}] hessian_eigenvalues=[{}] min_abs={:.4e}",
                evs.join(", "),
                self.min_abs_eigenvalue().unwrap_or(f64::NAN)
            );
        }
        match self.worst_component() {
            Some(w) if w.abs_gap() > 1e-3 && w.abs_gap() > 1e-3 * w.fd.abs().max(1.0) => {
                log::warn!(
                    "[OUTER-FD-AUDIT/{context}] VERDICT=DESYNC worst_block={} worst_i={} gap={:.3e} (analytic gradient disagrees with FD of the criterion: fix the derivative)",
                    w.block,
                    w.index,
                    w.abs_gap()
                );
            }
            _ => {
                let flat = self.min_abs_eigenvalue().map(|m| m < 1e-6).unwrap_or(false);
                if flat {
                    log::warn!(
                        "[OUTER-FD-AUDIT/{context}] VERDICT=FLATNESS min_abs_eig={:.3e} (analytic≈FD but the outer Hessian is near-singular: weak identifiability, fix termination not the gradient)",
                        self.min_abs_eigenvalue().unwrap_or(f64::NAN)
                    );
                } else {
                    log::warn!(
                        "[OUTER-FD-AUDIT/{context}] VERDICT=CLEAN analytic≈FD and outer Hessian well-conditioned at this θ"
                    );
                }
            }
        }
    }
}


/// Run a component-by-component central finite-difference audit of an outer
/// REML/LAML gradient at a fixed θ₀.
///
/// `eval` is the family's outer evaluator: `θ, mode ↦ (V, ∇V, H)` where the
/// gradient is honored at `ValueAndGradient`/`ValueGradientHessian` and `H` at
/// `ValueGradientHessian`. `block_for_index` labels each flat θ coordinate
/// (used only to group the report). `h` is the FD step.
///
/// Cost: one `ValueGradientHessian` eval at θ₀ plus `2·len(θ)` `ValueOnly`
/// evals. The caller is responsible for only invoking this on a
/// diagnostic-sized problem (it is not part of the production hot loop).
pub fn outer_gradient_fd_audit<EvalF>(
    theta0: &Array1<f64>,
    h: f64,
    block_for_index: impl Fn(usize) -> String,
    mut eval: EvalF,
) -> Result<OuterGradientFdAudit, String>
where
    EvalF: FnMut(
        &Array1<f64>,
        crate::solver::estimate::reml::unified::EvalMode,
    ) -> Result<(f64, Array1<f64>, HessianResult), String>,
{
    use crate::solver::estimate::reml::unified::EvalMode;
    let (value, analytic_grad, hess) = eval(theta0, EvalMode::ValueGradientHessian)?;
    if analytic_grad.len() != theta0.len() {
        return Err(format!(
            "outer_gradient_fd_audit: analytic gradient length {} != theta length {}",
            analytic_grad.len(),
            theta0.len()
        ));
    }
    let mut components = Vec::with_capacity(theta0.len());
    for i in 0..theta0.len() {
        let mut tp = theta0.clone();
        tp[i] += h;
        let mut tm = theta0.clone();
        tm[i] -= h;
        let (vp, _, _) = eval(&tp, EvalMode::ValueOnly)?;
        let (vm, _, _) = eval(&tm, EvalMode::ValueOnly)?;
        let fd = (vp - vm) / (2.0 * h);
        components.push(OuterGradientFdComponent {
            block: block_for_index(i),
            index: i,
            analytic: analytic_grad[i],
            fd,
        });
    }
    let hessian_eigenvalues = match hess.materialize_dense() {
        Ok(Some(mut hmat)) => {
            // Symmetrize defensively before the self-adjoint solve.
            let n = hmat.nrows();
            if n == hmat.ncols() && n > 0 {
                for r in 0..n {
                    for c in (r + 1)..n {
                        let avg = 0.5 * (hmat[[r, c]] + hmat[[c, r]]);
                        hmat[[r, c]] = avg;
                        hmat[[c, r]] = avg;
                    }
                }
                match crate::linalg::faer_ndarray::FaerEigh::eigh(&hmat, faer::Side::Lower) {
                    Ok((vals, _)) => {
                        let mut v: Vec<f64> = vals.to_vec();
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        v
                    }
                    Err(_) => Vec::new(),
                }
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    };
    Ok(OuterGradientFdAudit {
        value,
        components,
        hessian_eigenvalues,
    })
}


/// One recorded ContinuationPath demotion: a structural defect that, for a
/// continuation-entry objective, routes a seed to a heavier path regime instead
/// of disqualifying it. Carried in the seed-loop ledger so the startup stats
/// surface a heavier-regime re-entry (with its reason) rather than a vanished
/// candidate. Never fatal.
#[derive(Clone, Debug)]
struct PathDemotionRecord {
    /// 0-based seed index whose structural defect triggered the demotion.
    seed_idx: usize,
    /// The path regime the seed was re-entered at after the demotion.
    regime: crate::solver::continuation_path::PathRegime,
    /// Human-readable reason (the underlying structural diagnosis message).
    reason: String,
}


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
        Ok::<(), _>(())
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
        Ok::<(), _>(())
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
        Ok::<(), _>(())
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
        Ok::<(), _>(())
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
    // uncapped on every seed) has a degenerate worst case at large scale
    // — when every seed at the shallow cap collapses, we re-evaluate every
    // seed at the *full* inner budget, costing `N_seeds × full_pirls_work`
    // just to pick a starting point. The cascade replaces that all-or-
    // nothing jump with a geometric escalation: the typical case stays at
    // the initial cap (one pass), and the rare uniform-failure case pays
    // only `21 × initial` extra inner iterations before the uncapped
    // fallback.
    let cascade_caps = [
        PriorityBudgetStage {
            cap: initial_cap.saturating_mul(SEED_SCREENING_CASCADE_MULTIPLIERS[0]),
        },
        PriorityBudgetStage {
            cap: initial_cap.saturating_mul(SEED_SCREENING_CASCADE_MULTIPLIERS[1]),
        },
        PriorityBudgetStage {
            cap: initial_cap.saturating_mul(SEED_SCREENING_CASCADE_MULTIPLIERS[2]),
        },
        PriorityBudgetStage {
            cap: SEED_SCREENING_UNCAPPED,
        },
    ];

    let cascade_start = std::time::Instant::now();
    log::info!(
        "[STAGE] {context}: seed screening cascade start seeds={} initial_cap={} stages={}",
        seeds.len(),
        initial_cap,
        cascade_caps.len(),
    );

    let cascade_result = rank_indices_with_budget_cascade(
        seeds.len(),
        &cascade_caps,
        |stage, cap, idx| {
            screening_cap.store(cap, Ordering::Relaxed);
            obj.reset();
            screening_cap.store(cap, Ordering::Relaxed);
            let seed_started = std::time::Instant::now();
            let result = obj.eval_screening_proxy(&seeds[idx]);
            let seed_elapsed = seed_started.elapsed().as_secs_f64();
            match result {
                Ok(cost) if cost.is_finite() => {
                    log::info!(
                        "[STAGE] {context}: seed-screen stage={} seed={}/{} cap={} elapsed={:.3}s cost={:.6e}",
                        stage,
                        idx + 1,
                        seeds.len(),
                        if cap == 0 {
                            "uncapped".to_string()
                        } else {
                            cap.to_string()
                        },
                        seed_elapsed,
                        cost,
                    );
                    Ok(cost)
                }
                Ok(cost) => {
                    log::info!(
                        "[STAGE] {context}: seed-screen stage={} seed={}/{} cap={} elapsed={:.3}s cost=non-finite ({:.3e})",
                        stage,
                        idx + 1,
                        seeds.len(),
                        if cap == 0 {
                            "uncapped".to_string()
                        } else {
                            cap.to_string()
                        },
                        seed_elapsed,
                        cost,
                    );
                    Ok(cost)
                }
                Err(_) => {
                    log::info!(
                        "[STAGE] {context}: seed-screen stage={} seed={}/{} cap={} elapsed={:.3}s rejected (error)",
                        stage,
                        idx + 1,
                        seeds.len(),
                        if cap == 0 {
                            "uncapped".to_string()
                        } else {
                            cap.to_string()
                        },
                        seed_elapsed,
                    );
                    Err(())
                }
            }
        },
        |PriorityStageSummary {
             stage,
             cap,
             ranked,
             rejected,
         }| {
            log::info!(
                "[STAGE] {context}: seed-screen stage={} cap={} elapsed={:.3}s ranked={} rejected={}",
                stage,
                if cap == 0 {
                    "uncapped".to_string()
                } else {
                    cap.to_string()
                },
                cascade_start.elapsed().as_secs_f64(),
                ranked,
                rejected,
            );
            if ranked > 0 && stage > 0 {
                let final_cap = if cap == 0 {
                    "uncapped".to_string()
                } else {
                    cap.to_string()
                };
                log::info!(
                    "[OUTER] {context}: seed screening cap escalated from {} to {} \
                     (initial cap was too shallow for this problem; {}/{} seeds ranked)",
                    initial_cap,
                    final_cap,
                    ranked,
                    seeds.len(),
                );
            }
        },
    );

    let rejected = cascade_result.rejected;
    let final_cap_used = cascade_result.final_cap;
    let stages_consumed = cascade_result.stages_consumed;
    let ranked = cascade_result.ranked_indices;

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

    let mut ordered = Vec::with_capacity(seeds.len());
    let mut seen = vec![false; seeds.len()];
    for idx in ranked {
        seen[idx] = true;
        ordered.push(seeds[idx].clone());
    }
    for (idx, seed) in seeds.iter().enumerate() {
        if !seen[idx] {
            ordered.push(seed.clone());
        }
    }

    // Demote over-smoothing boundary seeds below every interior seed.
    //
    // The seed-screening cost is a *marginal-likelihood* proxy fit at a
    // capped inner-iteration budget. For a separation-stability seed pinned
    // at the ρ upper bound (`Array1::from_elem(k, bounds.1)`), that proxy is
    // systematically the cheapest: the penalized coefficients are shrunk
    // into the penalty null space, the capped inner solve converges
    // trivially, and the LAML/REML value is locally flat. So screening
    // ranks the boundary seed *first*. But the boundary is a degenerate
    // descent origin: ∂V/∂ρ → 0 there (nothing left to penalize), so a
    // trust-region / Newton outer solver started at the boundary certifies
    // box-constraint stationarity at iteration 0 and never reaches the
    // interior — which, for a location-scale model, is frequently
    // *anisotropic* (the well-determined mean wants heavy shrinkage while
    // the second-moment scale block wants far less). Starting the descent
    // from any interior seed instead lets the optimizer climb back up to
    // the bound coordinate-wise when the data truly want it, while still
    // resolving the coordinates whose optimum is interior. The boundary is
    // reachable by ascent but inescapable once it is the start, so it must
    // never out-rank an interior seed. We keep it at the tail as a
    // stability fallback: if every interior seed fails its full-budget
    // solve (genuine separation), the seed loop still falls through to it.
    // (#686/#687/#688: Gaussian location-scale was pinned at ρ=bound,
    // over-smoothing the log-σ envelope and wrecking held-out calibration.)
    let rho_dim = obj.capability().theta_layout().rho_dim();
    if rho_dim > 0 && ordered.len() > 1 {
        let upper: Vec<f64> = match config.bounds.as_ref() {
            Some((_, hi)) => hi.to_vec(),
            None => vec![config.rho_bound; rho_dim],
        };
        let (interior, boundary): (Vec<Array1<f64>>, Vec<Array1<f64>>) = ordered
            .into_iter()
            .partition(|seed| !seed_is_oversmoothing_boundary(seed, rho_dim, &upper));
        if !interior.is_empty() && !boundary.is_empty() {
            log::info!(
                "[OUTER] {context}: demoted {} over-smoothing boundary seed(s) below {} \
                 interior seed(s) so the outer descent does not originate on the flat \
                 ρ=bound plateau",
                boundary.len(),
                interior.len(),
            );
        }
        ordered = interior;
        ordered.extend(boundary);
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


/// ρ margin (in log-λ units) within which a smoothing coordinate counts as
/// sitting on the over-smoothing upper bound. The separation-stability seed is
/// generated *exactly* at the bound, so a small margin suffices; it is kept
/// loose enough to absorb a `project_to_bounds` round-trip without catching a
/// genuinely interior candidate (the next-densest generated seed is several
/// log-λ units below any realistic bound).
const OVERSMOOTH_BOUNDARY_MARGIN: f64 = 0.5;


/// Whether `seed` is pinned at the over-smoothing ρ upper bound in *every*
/// smoothing coordinate — the degenerate plateau where the penalized
/// coefficients collapse into the penalty null space and the REML/LAML
/// gradient ∂V/∂ρ vanishes. Only the leading `rho_dim` (smoothing) coordinates
/// are inspected; trailing ψ/auxiliary coordinates have their own geometry and
/// never make ρ a flat plateau. Used to keep such seeds from becoming the outer
/// optimizer's descent origin (see `rank_seeds_with_screening`).
fn seed_is_oversmoothing_boundary(seed: &Array1<f64>, rho_dim: usize, upper: &[f64]) -> bool {
    if rho_dim == 0 || seed.len() < rho_dim {
        return false;
    }
    (0..rho_dim).all(|i| {
        let hi = upper.get(i).copied().unwrap_or(f64::INFINITY);
        hi.is_finite() && seed[i] >= hi - OVERSMOOTH_BOUNDARY_MARGIN
    })
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
/// This enum is for the shared `eval` bridge where the runner needs value-only,
/// first-order, or second-order information depending on the active plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterEvalOrder {
    /// Compute only the objective value.
    Value,
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
    /// Stable, grep-friendly routing token for large-scale/log regression
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
        // No analytic gradient (with or without a declared Hessian), and the
        // EFS/HybridEFS fixed-point lane ruled out above. Every outer objective
        // in the tree now supplies an analytic gradient, so a cost-only
        // capability is a programming error. Emit a BFGS plan so it surfaces
        // loudly with context: the runner rejects it because BFGS requires the
        // analytic gradient this capability declares is absent. We deliberately
        // do NOT invent a working primary here — a cost-only objective has no
        // solver, by design.
        (Unavailable, _) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },
    }
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
    config.fallback_policy == FallbackPolicy::Disabled
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
                outer_strategy_contract_panic(
                    "expected dense analytic Hessian but got HessianResult::Operator",
                )
            }
            HessianResult::Unavailable => {
                // SAFETY: same contract as Operator above — `Unavailable`
                // means the family failed to produce the analytic Hessian
                // its `OuterPlan` declared.
                outer_strategy_contract_panic(
                    "expected analytic Hessian but got HessianResult::Unavailable",
                )
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
    /// `Some(gap)` when the `½log|H|` term in `cost` was produced as a CERTIFIED
    /// TWO-SIDED ENCLOSURE (the #1011 block-preconditioned border-Schur bound)
    /// rather than an exact logdet — `gap` is the enclosure width that `cost`
    /// inherits. The EFS engine consults the decision-margin contract against
    /// its step tolerance: a step is only allowed to declare convergence when
    /// the cost it converged on is resolved more tightly than that tolerance.
    /// `None` (the default for every exact-logdet objective) preserves today's
    /// behavior bit-for-bit.
    pub logdet_enclosure_gap: Option<f64>,
}


impl EfsEval {
    /// Attach a certified logdet-enclosure gap to this eval (the #1011 contract).
    /// The EFS engine then refuses to declare convergence on a cost the
    /// enclosure does not pin down below the step tolerance, escalating instead.
    #[must_use]
    pub fn with_logdet_enclosure_gap(mut self, gap: Option<f64>) -> Self {
        self.logdet_enclosure_gap = gap;
        self
    }
}


/// Outcome of [`OuterObjective::seed_inner_state`].
///
/// Distinguishes two non-error outcomes that callers handle differently:
///
/// - [`SeedOutcome::Installed`] — the objective owns an inner-β slot and the
///   provided β has been stored there. The next `eval*` will warm-start from
///   this β.
/// - [`SeedOutcome::NoSlot`] — the objective has no inner-β slot at all. The
///   provided β is silently discarded. This is the contract reply for
///   objectives whose inner iterate is conceptually empty (e.g. line-search
///   bridges, screening proxies, fixed-spec objectives).
///
/// Genuine seeding failures (wrong dimension when a slot exists, internal
/// allocation faults, …) are reported via `Err(EstimationError)`.
///
/// The two non-error variants exist because the two real callers want
/// opposite behavior on the no-slot path:
///
/// - The outer cache warm-start path (`OuterProblem::run`) reads a `(ρ, β)`
///   pair from disk; if the objective has no β slot it must log loudly
///   ("β-bearing checkpoint silently degraded to ρ-only resume") so cache
///   provenance is auditable.
/// - The continuation walk (`prime_outer_seed`) forwards `inner_beta_hint`
///   from the previous step; if the objective has no β slot the walk
///   simply proceeds cold — no log, no error.
///
/// Encoding the distinction in the return type lets each caller branch on
/// the variant without inspecting error message strings (the previous
/// brittle approach, see git history for `is_no_hook` in continuation.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedOutcome {
    /// The objective installed the provided β into its inner-β slot.
    Installed,
    /// The objective has no inner-β slot; the β was discarded.
    NoSlot,
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
    ///
    /// # Why the `eval_cost` default is correct for everyone else (#969)
    ///
    /// The partial-fit pathology is CAUSED by the screening cap: it is the
    /// `0.5·log|H|` term evaluated at a β̂ whose inner solve was truncated
    /// by `screening_max_inner_iterations`. An objective only suffers it if
    /// it (a) consumes that cap atomic AND (b) ranks on a curvature-bearing
    /// criterion at the truncated iterate — which is exactly the REML/LAML
    /// state-objective family, all of which override this method (or are
    /// built via `build_objective_with_screening_proxy`). Objectives that
    /// never wire the cap pay the full inner solve during screening, so
    /// their screened cost IS the true criterion — slower, but a correct
    /// ranking by definition, and a proxy could only degrade it. Any future
    /// objective that starts honoring the screening cap on a
    /// curvature-bearing criterion must override this with its own
    /// monotonically-descending inner quantity (the penalized-deviance
    /// pattern above generalizes: rank on the best inner merit seen, never
    /// on a curvature term at a truncated iterate).
    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.eval_cost(rho)
    }

    /// Evaluate cost + gradient + (if capable) Hessian.
    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError>;

    /// Evaluate the outer objective at the order requested by the active plan.
    ///
    /// The default preserves legacy behavior by delegating value-only requests
    /// to [`OuterObjective::eval_cost`] and derivative requests to
    /// [`OuterObjective::eval`].
    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        match order {
            OuterEvalOrder::Value => {
                let cost = self.eval_cost(rho)?;
                Ok(OuterEval {
                    cost,
                    gradient: Array1::zeros(rho.len()),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            }
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
    /// outer-iterate cache restored a `(ρ, β)` pair from a prior run, or
    /// when the continuation walk forwards `OuterEval::inner_beta_hint`
    /// from the previous step.
    ///
    /// Objectives make an explicit choice via the [`SeedOutcome`] return:
    /// implementations with an inner β slot return [`SeedOutcome::Installed`]
    /// after storing β; implementations without one return
    /// [`SeedOutcome::NoSlot`]. Genuine seeding failures (wrong dimension
    /// when a slot exists, etc.) are reported via `Err(EstimationError)`.
    ///
    /// Callers that need to distinguish "no slot" from "installed" (the
    /// outer cache warm-start path, which logs cache provenance) branch on
    /// the variant. Callers that don't care (the continuation walk, which
    /// only proceeds-cold when the hint is unusable) ignore it and only
    /// propagate `Err`.
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError>;

    /// Whether the objective can benefit from continuation pre-warm before
    /// the first solver eval at a candidate seed.
    ///
    /// Pre-warm is only correct for objectives with a real writable inner
    /// state slot: it evaluates an oversmoothed rho path before the seed and
    /// forwards non-empty `inner_beta_hint`s between steps. Generic synthetic
    /// objectives and rho-only cache probes must start at the chosen seed
    /// directly, otherwise the pre-warm becomes an observable extra eval and
    /// can clobber seed-dispatch bookkeeping with empty beta seeds.
    fn allow_continuation_prewarm(&self) -> bool {
        false
    }

    /// Optional opt-in to the device-resident outer REML BFGS-over-ρ driver
    /// (`crate::solver::gpu::reml_outer::run_reml_outer_on_device`). Returns
    /// `Some(adm)` when the objective is a REML evaluator whose
    /// `(spec, n, p, num_rho)` admission predicate accepts the device path,
    /// and `None` otherwise.
    ///
    /// The default returns `None` so non-REML objectives (line-search-only
    /// inner bridges, screening proxies, the EFS / hybrid-EFS sub-objectives)
    /// keep the host BFGS branch unconditionally — only the concrete
    /// REML-state objectives override this to consult
    /// [`crate::solver::estimate::reml::runtime::outer_reml_device_admission`].
    fn outer_device_admission(&self) -> Option<crate::gpu::policy::RemlOuterAdmission> {
        None
    }

    /// Whether every joint fit of this objective must ENTER through the
    /// [`crate::solver::continuation_path::ContinuationPath`] (heavy-smoothing
    /// entry) rather than being solved cold at the seed ρ*.
    ///
    /// The SAE-manifold joint objective overrides this to `true`: its joint
    /// `(logits, t, β)` block has a combinatorial active-set component that a
    /// cold solve can collapse, so it is entered at a heavy-smoothing regime
    /// and annealed down. Crucially, this flips the seed cascade's structural
    /// failure handling from REJECT to **DEMOTE-WITH-REASON**: a "cold"
    /// structural defect (rank/alias/active-set diagnosis from the seed
    /// pre-warm or the uniform-structural early-exit) is not a disqualification
    /// but a signal to RE-ENTER the same seed at a *heavier* ContinuationPath
    /// regime. The candidate set therefore never empties on a structural
    /// diagnosis — every demotion is recorded with its reason and routed to a
    /// heavier regime.
    ///
    /// The default `false` preserves the existing contract for every other
    /// objective: pre-warm stays an optimization (never a feasibility gate),
    /// and a uniform structural rejection still short-circuits the cascade.
    fn requires_continuation_path_entry(&self) -> bool {
        false
    }

    /// Run the objective's certified curvature-homotopy entry leg, if it has
    /// one, leaving the inner state warm at the real (`η = 1`) objective.
    ///
    /// An objective with a *certified anchor* — a point known by construction to
    /// be the global optimum of a relaxed problem — can replace the blind
    /// multi-seed multistart with a single predictor-corrector walk from that
    /// anchor to the true objective (#1007). The SAE-manifold objective
    /// overrides this: its `η = 0` Eckart-Young linear relaxation is convex and
    /// its optimum is certified by `linear_span_anchor`, so the walk in `η`
    /// tracks the unique optimal branch to `η = 1`. The walk monitors the
    /// arrow-factor min-pivot and halves the `η` step when it shrinks; a pivot
    /// collapse below tolerance is a DETECTED bifurcation (recorded on the fit
    /// payload, never silent), at which point the objective falls back to the
    /// documented multi-seed cascade.
    ///
    /// Returns:
    ///   * `None` — no certified anchor; use the standard seed cascade
    ///     (the default for every other objective).
    ///   * `Some(Ok(true))` — the walk arrived; the inner state is warm at the
    ///     certified `η = 1` solution and the seed cascade is bypassed.
    ///   * `Some(Ok(false))` — the anchor degenerated or the walk detected a
    ///     bifurcation; fall back to the multi-seed cascade (the report is
    ///     recorded on the objective for the fit payload).
    ///   * `Some(Err(_))` — a hard failure constructing the anchor.
    fn curvature_homotopy_entry(
        &mut self,
        rho: &Array1<f64>,
    ) -> Option<Result<bool, EstimationError>> {
        // Default: no certified anchor — but a non-finite seed is reported
        // here rather than silently handed to the seed cascade, mirroring the
        // hard-failure contract of the overriding implementations.
        if let Some(idx) = rho.iter().position(|v| !v.is_finite()) {
            return Some(Err(EstimationError::RemlOptimizationFailed(format!(
                "curvature-homotopy entry received non-finite rho[{idx}]"
            ))));
        }
        None
    }

    /// Let an objective declare that a seed is already a terminal outer result.
    /// Used for objectives with a certified high-quality construction seed where
    /// the generic rho optimizer can only degrade the fitted state.
    fn accept_seed_without_outer_iterations(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<Option<f64>, EstimationError> {
        if rho.is_empty() {
            return Ok(None);
        }
        Ok(None)
    }

    /// Re-install the selected outer result into the mutable objective before
    /// callers consume objective-owned fitted state. Optimizers may evaluate
    /// rejected trial points after the best point was found; without this final
    /// synchronization, stateful objectives can report the last trial fit rather
    /// than the returned `OuterResult::rho`.
    fn finalize_outer_result(
        &mut self,
        rho: &Array1<f64>,
        plan: &OuterPlan,
    ) -> Result<(), EstimationError> {
        log::debug!(
            "[OUTER] finalize: re-installing best rho into the objective (solver {:?})",
            plan.solver
        );
        self.eval_cost(rho).map(|_| ())
    }
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


/// Entries with a different schema id are rejected by `decode_iterate`
/// so incompatible on-disk payloads fall through to cold start instead
/// of seeding the inner solve with a malformed iterate.
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

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // Forward to the wrapped objective, then prime our last-inner-beta
        // cache so a subsequent finalize-write encodes the seeded β if no
        // eval surfaces a fresher β first. Only prime on actual install —
        // `NoSlot` means the inner solver will not see β, so the cache
        // entry would be a lie.
        let result = self.inner.seed_inner_state(beta);
        if matches!(result, Ok(SeedOutcome::Installed))
            && beta.iter().all(|v| v.is_finite())
            && let Ok(mut guard) = self.last_inner_beta.lock()
        {
            *guard = Some(beta.clone());
        }
        result
    }

    fn allow_continuation_prewarm(&self) -> bool {
        self.inner.allow_continuation_prewarm()
    }

    fn requires_continuation_path_entry(&self) -> bool {
        self.inner.requires_continuation_path_entry()
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
    Fseed = fn(&mut S, &Array1<f64>) -> Result<(), EstimationError>,
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
    /// Optional inner-state seeding closure. Objectives with PIRLS / Newton
    /// inner state install cached β here before the first outer eval.
    pub(crate) seed_fn: Option<Fseed>,
    /// Whether a seed hook should also opt the objective into generic
    /// continuation pre-warm. High-dimensional REML keeps the seed hook for
    /// cache/warm-start replay but declines the expensive rho-anneal pre-pass.
    pub(crate) continuation_prewarm: bool,
}


impl<S, Fc, Fe, Fr, Fefs, Feo, Fsp, Fseed> OuterObjective
    for ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp, Fseed>
where
    Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
    Fr: FnMut(&mut S),
    Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
    Fsp: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fseed: FnMut(&mut S, &Array1<f64>) -> Result<(), EstimationError>,
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

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // Empty β: by convention, "no warm-start available" — treat as a
        // no-op install. Distinct from `NoSlot` because the objective may
        // very well have a slot; the caller just didn't supply a β to fill
        // it. Reporting `Installed` is correct: the slot's pre-existing
        // state (cold default) is the post-seed state.
        if beta.is_empty() {
            return Ok(SeedOutcome::Installed);
        }
        match self.seed_fn.as_mut() {
            Some(f) => f(&mut self.state, beta).map(|()| SeedOutcome::Installed),
            // No hook installed — the objective owns no inner-β slot.
            // The caller decides whether this is a loud cache-provenance
            // event or a silent continuation-walk degradation.
            None => Ok(SeedOutcome::NoSlot),
        }
    }

    fn allow_continuation_prewarm(&self) -> bool {
        self.continuation_prewarm && self.seed_fn.is_some()
    }

    fn reset(&mut self) {
        if let Some(f) = self.reset_fn.as_mut() {
            f(&mut self.state);
        }
    }
}


impl<S, Fc, Fe, Fr, Fefs, Feo, Fsp> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp>
where
    Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
    Fr: FnMut(&mut S),
    Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
    Fsp: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
{
    pub fn with_seed_inner_state<Fseed>(
        self,
        seed_fn: Fseed,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp, Fseed>
    where
        Fseed: FnMut(&mut S, &Array1<f64>) -> Result<(), EstimationError>,
    {
        ClosureObjective {
            state: self.state,
            cap: self.cap,
            cost_fn: self.cost_fn,
            eval_fn: self.eval_fn,
            eval_order_fn: self.eval_order_fn,
            reset_fn: self.reset_fn,
            efs_fn: self.efs_fn,
            screening_proxy_fn: self.screening_proxy_fn,
            seed_fn: Some(seed_fn),
            continuation_prewarm: self.continuation_prewarm,
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
    /// (see `first_order_inner_cap_schedule`).
    ///
    /// The cap is a perf optimization for the GRADIENT inner solve only: at
    /// the accepted ρ the warm-start is excellent, so a small cap converges
    /// the inner Newton and a still-non-converged result is honestly rejected
    /// as infeasible. But the line-search COST probe (`eval_cost`) evaluates a
    /// DIFFERENT trial ρ whose warm-start is worse; the same small cap can stop
    /// the inner solve short of its fixed point, returning a non-converged
    /// `f64::INFINITY` cost for a point that is actually feasible. With every
    /// trial step then reporting `∞`, no Wolfe/ARC step satisfies descent, the
    /// optimizer never leaves the accepted ρ, and the gradient re-evaluated
    /// there is identical iter after iter — the frozen-|g| outer stall in
    /// gam#787 (bernoulli matern marginal-slope) and gam#808 (survival
    /// marginal-slope). The line-search cost MUST be the same converged-inner
    /// objective the analytic envelope gradient differentiates; a capped
    /// surrogate is a different objective. So `eval_cost` UNCAPS the inner solve
    /// (stores `0` = full `pirls_config.max_iterations`) before delegating, and
    /// `eval_grad`/`eval_hessian` restore the scheduled cap on the next call.
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
    /// Most recent derivative-evaluation point. Value-only line-search probes
    /// log their distance from this reference so hidden backtracking work is
    /// visible in STAGE traces.
    last_value_grad_rho: Option<Array1<f64>>,
    /// Exact memo for recent line-search value probes. BFGS can re-query the
    /// same rejected trial when switching Wolfe strategies; the SAE inner solve
    /// behind a Value probe is deterministic, so serving an identical rho from
    /// this memo preserves the objective while avoiding duplicate refinement
    /// work.
    value_probe_cache: Vec<ValueProbeCacheEntry>,
    /// Gradient-independent cost-stall convergence guard. `opt::Bfgs` only
    /// terminates on a small *projected gradient norm* (its stall exit ANDs
    /// gradient-smallness with cost-smallness), so on a fully-penalized
    /// (double-penalty) REML surface with a shallow, weakly-identified valley —
    /// where the REML score flatlines while `‖∇_ρ V‖` plateaus *above*
    /// tolerance — no opt-side exit ever fires and BFGS burns its entire
    /// `max_iterations` budget (each iteration spending many line-search +
    /// coordinate-rescue + jiggle probes) on every seed. That is the #1089
    /// pathology: a trivial n≈30..120 Gaussian fit emitting ~850k cost-only
    /// evaluations until a wall-clock budget kills it. This guard adds the
    /// missing mgcv-style score-change stop: it watches the accepted-iterate
    /// REML objective and, once it stops improving by more than a relative
    /// tolerance over a window of consecutive accepted outer steps, publishes
    /// the best-so-far iterate and signals BFGS to stop. The runner then
    /// classifies the run as *converged at the flat-valley floor* rather than
    /// non-converged — the remaining gradient lies along weakly-identified ρ
    /// directions that do not reduce the objective.
    cost_stall: Option<CostStallGuard>,
    /// Count of consecutive `eval_cost` calls that returned `Recoverable`
    /// without a single success in between. When every trial step in every
    /// search direction is infeasible (the inner solve refuses to converge at
    /// any neighboring ρ), BFGS would otherwise spend its full
    /// `max_iterations × line_search_budget` budget doing inner solves that
    /// all fail — the non-termination reported in issue #NaN-outer-loop.
    ///
    /// Once this counter exceeds [`PROBE_REFUSAL_FATAL_THRESHOLD`] and no
    /// gradient evaluation has ever been accepted on this seed (`iter_count ==
    /// 0`), the bridge escalates to `Fatal` so BFGS exits immediately via
    /// `ObjectiveFailed`. The seed loop treats that outcome as a rejected seed
    /// and moves on, keeping the cascade bounded.
    ///
    /// Reset to 0 on any successful cost evaluation so normal line-search
    /// noise (a few recoverable probes followed by an accepted step) never
    /// trips this guard.
    consecutive_probe_refusals: usize,
}


const VALUE_PROBE_CACHE_CAPACITY: usize = 256;

const VALUE_PROBE_REJECT_COST_FLOOR: f64 = 1.0e11;


/// Number of consecutive recoverable `eval_cost` failures (every line-search
/// probe infeasible) before the bridge escalates to `Fatal` and forces an
/// immediate BFGS exit. This guard fires only before the first accepted
/// gradient step (`iter_count == 0`): once BFGS has accepted at least one
/// outer iteration the current ρ is feasible and isolated probe refusals are
/// normal line-search noise, not a stuck loop.
///
/// The threshold covers one full StrongWolfe attempt (up to 20 probes)
/// plus one backtracking fallback (up to 50 probes) with a small margin,
/// so a SINGLE failed direction does not fire the guard. Two consecutive
/// direction failures (120 probes) always does — once both Wolfe and
/// backtracking exhausted two complete directions with no success, the
/// neighborhood is globally infeasible and further BFGS iterations are
/// pure waste.
const PROBE_REFUSAL_FATAL_THRESHOLD: usize = 150;


/// Tighter probe-refusal threshold used when the bridge has never seen a
/// `eval_grad` call of its own — i.e. the seed (cost, gradient) was supplied
/// via `with_initial_sample` so `last_value_grad_rho` is `None` and every
/// `trial_rho_distance` prints as NaN.  In this case the seed gradient is
/// already confirmed feasible externally; if even the first line-search
/// direction exhausts its Wolfe probes without success (≈ 20 probes), the
/// neighborhood IS globally infeasible and further iterations just repeat
/// the same expensive inner solve 150 more times.  One generous Wolfe
/// budget (25 probes) is enough to confirm the failure; 13 seeds ×
/// 150 probes × ~3 s each would otherwise cause an observed ~97 min hang.
const PROBE_REFUSAL_FATAL_THRESHOLD_NAN_SEED: usize = 25;


/// Sentinel prefix embedded in the [`ObjectiveEvalError::Fatal`] message the
/// bridge returns when [`PROBE_REFUSAL_FATAL_THRESHOLD`] fires. The seed-loop
/// runner matches this prefix and routes the failed seed to
/// `rejection_reasons` rather than propagating a fatal error.
const PROBE_REFUSAL_FATAL_SENTINEL: &str = "OUTER_PROBE_REFUSAL_FATAL";


/// Sentinel embedded in the [`ObjectiveEvalError::Fatal`] message the bridge
/// returns when [`CostStallGuard`] halts BFGS on a cost stall. `opt::Bfgs`
/// preserves the message verbatim in [`BfgsError::ObjectiveFailed`]; the
/// seed-loop runner recognizes this sentinel and rebuilds an outer result from
/// the published best iterate. Whether that result is reported `converged` is
/// NOT decided here — it is carried on the published [`CostStallExit`], gated on
/// the projected gradient norm at the best iterate clearing the same outer
/// gradient tolerance the genuine convergence path uses. A cost stall whose
/// residual gradient still exceeds that tolerance is a flat-valley stall, not a
/// stationary optimum, and is reported `converged = false`.
const COST_STALL_CONVERGED_SENTINEL: &str = "OUTER_COST_STALL_CONVERGED";


/// Verdict produced by folding one accepted outer iterate into
/// [`CostStallGuard::observe`].
enum CostStallVerdict {
    /// The objective is still improving (or the no-improvement window has not
    /// yet filled). Keep descending.
    Continue,
    /// The objective has stopped improving over the window AND the projected
    /// gradient norm at the best iterate clears the outer gradient tolerance:
    /// a genuine stationary optimum on a (legitimately) flat REML surface.
    Converged,
    /// The objective has stopped improving over the window but the projected
    /// gradient norm at the best iterate is still above the outer gradient
    /// tolerance: a weakly-identified flat-valley FLOOR with residual
    /// non-stationarity. Halting here is correct (no further cost progress is
    /// available), but the iterate is NOT a stationary optimum and must be
    /// reported `converged = false`.
    FlatValleyStall { residual_grad_norm: f64 },
}


/// Number of consecutive accepted outer iterates with negligible relative
/// objective improvement required before the cost-stall guard declares
/// convergence. Matches the spirit of `opt`'s own `StallPolicy { window: 3 }`
/// but, crucially, is gated on the cost alone (not on gradient smallness),
/// which is the condition `opt` never checks in isolation.
const COST_STALL_WINDOW: usize = 6;


/// Best iterate captured by a cost-stall convergence, handed from the bridge
/// (which is moved into `opt::Bfgs`) back to the seed-loop runner via the
/// guard's shared cell.
#[derive(Clone)]
struct CostStallExit {
    rho: Array1<f64>,
    value: f64,
    grad_norm: f64,
    /// Accepted outer iterates observed when the stall fired (for the runner's
    /// `OuterResult.iterations` field and logging).
    iterations: usize,
    /// Whether the best iterate is a genuine stationary optimum: `true` only
    /// when its projected gradient norm cleared the outer gradient tolerance
    /// (legitimately-flat REML surface). `false` for a flat-valley stall whose
    /// residual gradient remains above tolerance — the runner reports the
    /// rebuilt outer result as non-converged in that case.
    converged: bool,
}


/// Tracks the monotone best accepted-iterate REML objective and a
/// no-improvement streak, firing a gradient-independent convergence once the
/// objective has effectively stopped decreasing. See the `cost_stall` field
/// doc on [`OuterFirstOrderBridge`] for the full rationale (#1089).
struct CostStallGuard {
    /// Relative improvement floor: an accepted step counts as "no improvement"
    /// when `(best - cost) <= rel_tol * (1 + |best|)`. Derived from the outer
    /// convergence tolerance so it tracks the configured precision rather than
    /// a free-standing magic constant.
    rel_tol: f64,
    /// Consecutive accepted-step window with no improvement before declaring
    /// convergence.
    window: usize,
    /// Projected outer gradient-norm threshold that the best iterate must clear
    /// for a cost stall to count as a genuine stationary optimum. This is the
    /// SAME threshold the normal BFGS convergence path uses
    /// (`outer_gradient_tolerance(config).threshold(seed_cost, ‖g_0‖)`),
    /// evaluated once at seed. A cost stall above this threshold is a
    /// flat-valley stall, reported `converged = false`.
    grad_threshold: f64,
    best_value: f64,
    best_rho: Option<Array1<f64>>,
    best_grad_norm: f64,
    no_improve_streak: usize,
    accepted_iters: usize,
    /// Shared publication slot read by the seed-loop runner after
    /// `optimizer.run()` returns the sentinel error.
    exit: Arc<Mutex<Option<CostStallExit>>>,
}


impl CostStallGuard {
    fn new(
        rel_tol: f64,
        window: usize,
        grad_threshold: f64,
        exit: Arc<Mutex<Option<CostStallExit>>>,
    ) -> Self {
        Self {
            rel_tol,
            window,
            grad_threshold,
            best_value: f64::INFINITY,
            best_rho: None,
            best_grad_norm: f64::INFINITY,
            no_improve_streak: 0,
            accepted_iters: 0,
            exit,
        }
    }

    /// Fold one accepted-iterate `(ρ, cost, ‖g‖)` into the guard. Returns a
    /// [`CostStallVerdict`]: `Continue` while the score is still improving,
    /// `Converged` when the score has stalled AND the projected gradient norm
    /// at the best iterate clears the outer gradient tolerance (a genuine
    /// stationary optimum on a flat REML surface), or `FlatValleyStall` when
    /// the score has stalled but the residual gradient remains above tolerance
    /// (a weakly-identified flat valley that is NOT stationary). Either stalled
    /// verdict publishes the best iterate to the shared cell, tagged with its
    /// `converged` status.
    fn observe(&mut self, rho: &Array1<f64>, value: f64, grad_norm: f64) -> CostStallVerdict {
        if !value.is_finite() {
            // A non-finite accepted objective is the inner-solver's problem,
            // not a stall; reset so a later real descent is not falsely
            // credited as a no-improvement step.
            self.no_improve_streak = 0;
            return CostStallVerdict::Continue;
        }
        self.accepted_iters = self.accepted_iters.saturating_add(1);
        let improvement = self.best_value - value;
        let floor = self.rel_tol * (1.0 + self.best_value.abs());
        if value < self.best_value {
            self.best_value = value;
            self.best_rho = Some(rho.clone());
            self.best_grad_norm = grad_norm;
        }
        if improvement <= floor {
            self.no_improve_streak = self.no_improve_streak.saturating_add(1);
        } else {
            self.no_improve_streak = 0;
        }
        if self.no_improve_streak < self.window {
            return CostStallVerdict::Continue;
        }
        // Publish the best iterate. Prefer the recorded best; fall back to the
        // current point if (pathologically) none was stored.
        let best_rho = self.best_rho.clone().unwrap_or_else(|| rho.clone());
        let best_value = if self.best_value.is_finite() {
            self.best_value
        } else {
            value
        };
        let best_grad_norm = if self.best_grad_norm.is_finite() {
            self.best_grad_norm
        } else {
            grad_norm
        };
        // Convergence is STATIONARITY, not cost-flatness: a cost stall counts
        // as a converged optimum only when the projected gradient norm at the
        // best iterate clears the same outer gradient tolerance the genuine
        // BFGS convergence path checks. Otherwise it is a flat-valley floor
        // with residual non-stationarity, reported `converged = false`.
        let converged = best_grad_norm.is_finite() && best_grad_norm <= self.grad_threshold;
        if let Ok(mut slot) = self.exit.lock() {
            *slot = Some(CostStallExit {
                rho: best_rho,
                value: best_value,
                grad_norm: best_grad_norm,
                iterations: self.accepted_iters,
                converged,
            });
        }
        if converged {
            CostStallVerdict::Converged
        } else {
            CostStallVerdict::FlatValleyStall {
                residual_grad_norm: best_grad_norm,
            }
        }
    }
}


#[derive(Clone)]
struct ValueProbeCacheEntry {
    rho: Array1<f64>,
    outcome: CachedValueProbeOutcome,
}


#[derive(Clone)]
enum CachedValueProbeOutcome {
    Cost(f64),
    Recoverable(String),
    Fatal(String),
}


fn trial_rho_distance(reference: Option<&Array1<f64>>, trial: &Array1<f64>) -> f64 {
    let Some(reference) = reference else {
        return f64::NAN;
    };
    if reference.len() != trial.len() {
        return f64::NAN;
    }
    reference
        .iter()
        .zip(trial.iter())
        .map(|(a, b)| {
            let d = b - a;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}


fn same_outer_point(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}


fn cached_value_probe_result(outcome: &CachedValueProbeOutcome) -> Result<f64, ObjectiveEvalError> {
    match outcome {
        CachedValueProbeOutcome::Cost(cost) => Ok(*cost),
        CachedValueProbeOutcome::Recoverable(message) => {
            Err(ObjectiveEvalError::recoverable(message.clone()))
        }
        CachedValueProbeOutcome::Fatal(message) => Err(ObjectiveEvalError::Fatal {
            message: message.clone(),
        }),
    }
}


fn cache_value_probe_result(result: &Result<f64, ObjectiveEvalError>) -> CachedValueProbeOutcome {
    match result {
        Ok(cost) => CachedValueProbeOutcome::Cost(*cost),
        Err(ObjectiveEvalError::Recoverable { message }) => {
            CachedValueProbeOutcome::Recoverable(message.clone())
        }
        Err(ObjectiveEvalError::Fatal { message }) => {
            CachedValueProbeOutcome::Fatal(message.clone())
        }
    }
}


fn value_probe_outcome_label(outcome: &CachedValueProbeOutcome) -> &'static str {
    match outcome {
        CachedValueProbeOutcome::Cost(_) => "cost",
        CachedValueProbeOutcome::Recoverable(_) => "recoverable",
        CachedValueProbeOutcome::Fatal(_) => "fatal",
    }
}


fn value_probe_reject_outcome(outcome: &CachedValueProbeOutcome) -> bool {
    match outcome {
        CachedValueProbeOutcome::Cost(cost) => *cost >= VALUE_PROBE_REJECT_COST_FLOOR,
        CachedValueProbeOutcome::Recoverable(_) | CachedValueProbeOutcome::Fatal(_) => true,
    }
}


fn remember_value_probe(
    cache: &mut Vec<ValueProbeCacheEntry>,
    rho: &Array1<f64>,
    outcome: CachedValueProbeOutcome,
) {
    if let Some(entry) = cache
        .iter_mut()
        .find(|entry| same_outer_point(&entry.rho, rho))
    {
        entry.outcome = outcome;
        return;
    }
    if cache.len() == VALUE_PROBE_CACHE_CAPACITY {
        cache.remove(0);
    }
    cache.push(ValueProbeCacheEntry {
        rho: rho.clone(),
        outcome,
    });
}


impl ZerothOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Per-axis line-search step caps now live natively in opt::Bfgs
        // (`with_axis_step_caps`), which shortens the BFGS direction before
        // line search instead of poisoning the Wolfe bracket with a
        // sentinel cost. This entry point can therefore stay honest: any
        // call that lands here is a real line-search probe, not a too-far
        // attempt the bridge needs to swat away.
        //
        // Uncap the inner solve for the line-search cost probe (see the field
        // doc on `outer_inner_cap`): the deciding cost MUST be the true
        // converged-inner objective the analytic gradient differentiates, not
        // the scheduled gradient-path cap which can stop a trial-ρ inner solve
        // short of its fixed point and report a spurious `∞`. `eval_grad`
        // restores the scheduled cap on the next call.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            feedback
                .cap
                .store(SEED_SCREENING_UNCAPPED, Ordering::Relaxed);
        }
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let trial_rho_distance = trial_rho_distance(self.last_value_grad_rho.as_ref(), x);
        let stage_start = std::time::Instant::now();
        if let Some(entry) = self
            .value_probe_cache
            .iter()
            .find(|entry| same_outer_point(&entry.rho, x))
        {
            let outcome_label = value_probe_outcome_label(&entry.outcome);
            log::info!(
                "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e} (first-order bridge, iter={}, cached=true)",
                x.len(),
                trial_rho_distance,
                self.iter_count
            );
            match &entry.outcome {
                CachedValueProbeOutcome::Cost(cost) => log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e} (first-order bridge, iter={}, cached=true)",
                    stage_start.elapsed().as_secs_f64(),
                    cost,
                    trial_rho_distance,
                    self.iter_count
                ),
                CachedValueProbeOutcome::Recoverable(_) | CachedValueProbeOutcome::Fatal(_) => {
                    log::info!(
                        "[STAGE] outer eval end order=Value elapsed={:.3}s outcome={} trial_rho_distance={:.3e} (first-order bridge, iter={}, cached=true)",
                        stage_start.elapsed().as_secs_f64(),
                        outcome_label,
                        trial_rho_distance,
                        self.iter_count
                    );
                }
            }
            return cached_value_probe_result(&entry.outcome);
        }
        log::info!(
            "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e} (first-order bridge, iter={})",
            x.len(),
            trial_rho_distance,
            self.iter_count
        );
        let result = self
            .obj
            .eval_with_order(x, OuterEvalOrder::Value)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))
            .and_then(|eval| finite_cost_or_error("outer eval_cost failed", eval.cost));
        let cached_outcome = cache_value_probe_result(&result);
        remember_value_probe(&mut self.value_probe_cache, x, cached_outcome);
        match &result {
            Ok(cost) => {
                // A successful probe resets the consecutive-refusal counter: the
                // current ρ neighbourhood has at least one feasible point, so
                // isolated refusals on other directions are normal line-search
                // noise, not a globally-infeasible neighbourhood.
                self.consecutive_probe_refusals = 0;
                log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e} (first-order bridge, iter={})",
                    stage_start.elapsed().as_secs_f64(),
                    cost,
                    trial_rho_distance,
                    self.iter_count
                );
            }
            Err(ObjectiveEvalError::Recoverable { .. }) => {
                log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s outcome=recoverable trial_rho_distance={:.3e} (first-order bridge, iter={})",
                    stage_start.elapsed().as_secs_f64(),
                    trial_rho_distance,
                    self.iter_count
                );
                // Non-termination guard (#NaN-outer-loop): when every
                // line-search probe is infeasible and BFGS has never
                // accepted a gradient step (`iter_count == 0`), the
                // neighbourhood around the seed is globally degenerate.
                // BFGS would otherwise spend its entire max_iterations ×
                // line_search_budget doing inner solves that all fail.
                // Escalate to Fatal so BFGS exits immediately; the seed
                // loop routes it as a rejected seed.
                self.consecutive_probe_refusals =
                    self.consecutive_probe_refusals.saturating_add(1);
                // When the bridge seed (cost, gradient) was supplied via
                // `with_initial_sample` the bridge's own `eval_grad` is
                // never called, so `last_value_grad_rho` stays `None` and
                // every `trial_rho_distance` prints as NaN.  The seed IS
                // feasible (it was evaluated externally), but if every
                // line-search probe is Recoverable from the very first
                // direction, the neighbourhood is globally infeasible.
                // Use the tighter NaN-seed threshold so the guard fires
                // after one generous Wolfe budget instead of 150 probes
                // (which, at ~3 s each × 13 seeds, would produce an
                // observed ~97 min hang on real D=5120 LLM activations).
                let threshold = if self.last_value_grad_rho.is_none() {
                    PROBE_REFUSAL_FATAL_THRESHOLD_NAN_SEED
                } else {
                    PROBE_REFUSAL_FATAL_THRESHOLD
                };
                if self.iter_count == 0 && self.consecutive_probe_refusals >= threshold {
                    log::warn!(
                        "[OUTER] probe-refusal non-termination guard fired after {} consecutive \
                         infeasible cost probes with no accepted gradient step \
                         (nan_seed={}); escalating to Fatal to abort this seed \
                         (first-order bridge, iter={})",
                        self.consecutive_probe_refusals,
                        self.last_value_grad_rho.is_none(),
                        self.iter_count,
                    );
                    return Err(ObjectiveEvalError::Fatal {
                        message: format!(
                            "{PROBE_REFUSAL_FATAL_SENTINEL}: {consecutive} consecutive \
                             infeasible probes with no accepted outer step",
                            consecutive = self.consecutive_probe_refusals,
                        ),
                    });
                }
            }
            Err(ObjectiveEvalError::Fatal { .. }) => {
                log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s outcome=fatal trial_rho_distance={:.3e} (first-order bridge, iter={})",
                    stage_start.elapsed().as_secs_f64(),
                    trial_rho_distance,
                    self.iter_count
                );
            }
        }
        result
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
        self.last_value_grad_rho = Some(x.clone());
        // A successful gradient evaluation means the current ρ is feasible;
        // reset the consecutive-probe-refusal counter so the guard only fires
        // when ALL probes in EVERY subsequent direction fail.
        self.consecutive_probe_refusals = 0;
        self.value_probe_cache
            .retain(|entry| value_probe_reject_outcome(&entry.outcome));
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
        // Cost-stall halt (#1089). `eval_grad` is invoked by `opt::Bfgs` at
        // each accepted iterate (line-search COST probes go through `eval_cost`,
        // not here), so folding the objective in here counts accepted outer
        // steps. When the REML score has stopped improving over
        // `COST_STALL_WINDOW` consecutive accepted steps, halt BFGS by returning
        // a sentinel `Fatal` (an observer cannot stop `opt::Bfgs`; an error is
        // the only in-band way to halt it). The runner rebuilds the outer result
        // from the published best iterate — but whether that result is reported
        // CONVERGED is decided by the guard's STATIONARITY test, not by
        // cost-flatness alone: a stall whose projected gradient still exceeds the
        // outer gradient tolerance is a flat-valley floor (`converged = false`),
        // a stationary one is a real optimum (`converged = true`). Both share the
        // sentinel; the verdict rides on the published `CostStallExit.converged`.
        if let Some(guard) = self.cost_stall.as_mut() {
            match guard.observe(x, eval.cost, g_norm) {
                CostStallVerdict::Continue => {}
                CostStallVerdict::Converged => {
                    log::info!(
                        "[OUTER] cost-stall convergence: REML objective improved < {:.3e} \
                         (relative) over {} consecutive accepted outer steps AND the projected \
                         gradient cleared the outer tolerance (|g|={:.3e} <= {:.3e}); accepting \
                         best-so-far as a stationary optimum (value={:.6e}).",
                        guard.rel_tol,
                        guard.window,
                        guard.best_grad_norm,
                        guard.grad_threshold,
                        guard.best_value,
                    );
                    return Err(ObjectiveEvalError::Fatal {
                        message: COST_STALL_CONVERGED_SENTINEL.to_string(),
                    });
                }
                CostStallVerdict::FlatValleyStall { residual_grad_norm } => {
                    log::warn!(
                        "[OUTER] cost-stall FLAT-VALLEY STALL: REML objective improved < {:.3e} \
                         (relative) over {} consecutive accepted outer steps but the projected \
                         gradient is still ABOVE the outer tolerance (|g|={:.3e} > {:.3e}); \
                         halting on a weakly-identified ρ valley floor and reporting NON-CONVERGED \
                         (residual outer non-stationarity, value={:.6e}).",
                        guard.rel_tol,
                        guard.window,
                        residual_grad_norm,
                        guard.grad_threshold,
                        guard.best_value,
                    );
                    return Err(ObjectiveEvalError::Fatal {
                        message: COST_STALL_CONVERGED_SENTINEL.to_string(),
                    });
                }
            }
        }
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient,
        })
    }
}


/// Outer gradient-decay ratio `‖g_now‖/‖g_initial‖` below which the outer is
/// treated as essentially converged: the inner cap is lifted entirely so the
/// cached β reaches full inner tolerance before the convergence guard runs.
const INNER_CAP_CONVERGENCE_OVERRIDE_RATIO: f64 = 0.01;


/// Floor on the adaptive inner-PIRLS cap. Any cap below this is below the
/// inner-Newton noise level and would reject usable warm-started steps.
const INNER_CAP_FLOOR: usize = 3;


/// Ceiling on the adaptive inner-PIRLS cap, set at the inner-Newton noise
/// floor at large scale; further iterations are pure waste once the warm
/// start is close.
const INNER_CAP_CEILING: usize = 64;


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
/// (the inner noise floor at large scale; further iters would be
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
    if matches!(g_ratio, Some(r) if r < INNER_CAP_CONVERGENCE_OVERRIDE_RATIO) {
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
        return next.clamp(INNER_CAP_FLOOR, INNER_CAP_CEILING);
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
    /// Most recent derivative-evaluation point, used to log value-probe
    /// displacement in line-search / trial-acceptance STAGE traces.
    last_value_grad_rho: Option<Array1<f64>>,
}


impl ZerothOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Uncap the inner solve for the ARC line-search / trial-acceptance cost
        // probe. Identical rationale to `OuterFirstOrderBridge::eval_cost`: the
        // deciding cost must be the true converged-inner objective the analytic
        // gradient/Hessian differentiate, never the scheduled gradient-path cap
        // (which at a trial ρ can stop the inner solve short and report a
        // spurious `∞`, freezing the ARC at constant cost / |g| — gam#808
        // survival marginal-slope, gam#787 bernoulli matern marginal-slope).
        // `eval_grad`/`eval_hessian` restore the scheduled cap on the next call.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            feedback
                .cap
                .store(SEED_SCREENING_UNCAPPED, Ordering::Relaxed);
        }
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let trial_rho_distance = trial_rho_distance(self.last_value_grad_rho.as_ref(), x);
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e}",
            x.len(),
            trial_rho_distance
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::Value)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        let cost = finite_cost_or_error("outer eval_cost failed", eval.cost)?;
        log::info!(
            "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e}",
            stage_start.elapsed().as_secs_f64(),
            cost,
            trial_rho_distance
        );
        Ok(cost)
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
        self.last_value_grad_rho = Some(x.clone());
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
        self.last_value_grad_rho = Some(x.clone());
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
    /// Most recent derivative-evaluation point, used to log value-probe
    /// displacement in line-search STAGE traces.
    last_value_grad_rho: Option<Array1<f64>>,
}


impl ZerothOrderObjective for OuterOperatorBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Uncap the inner solve for the matrix-free TR line-search cost probe.
        // Identical rationale to the BFGS / ARC bridges: the deciding cost must
        // be the true converged-inner objective the analytic gradient/operator
        // Hessian differentiate, never the scheduled gradient-path cap (which at
        // a trial ρ can stop the inner solve short and report a spurious `∞`,
        // freezing the TR at constant cost / |g|). This is the route the
        // ψ-bearing matern bernoulli marginal-slope fit takes (gam#787);
        // `eval_value_grad_op` restores the scheduled cap on the next call.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            feedback
                .cap
                .store(SEED_SCREENING_UNCAPPED, Ordering::Relaxed);
        }
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let trial_rho_distance = trial_rho_distance(self.last_value_grad_rho.as_ref(), x);
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e} (operator bridge)",
            x.len(),
            trial_rho_distance
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::Value)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        let cost = finite_cost_or_error("outer eval_cost failed", eval.cost)?;
        log::info!(
            "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e} (operator bridge)",
            stage_start.elapsed().as_secs_f64(),
            cost,
            trial_rho_distance
        );
        Ok(cost)
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
        self.last_value_grad_rho = Some(x.clone());
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
        self.last_value_grad_rho = Some(x.clone());
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
/// (b) costs O(K) full outer evaluations per ARC iteration — at large-scale
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
/// On hard problems (Matérn additive at large scale, Duchon60, anisotropic
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
        // the inner P-IRLS solve we'd otherwise burn on backtracking. When a
        // barrier is configured, every accepted rho-step must still pass
        // through the barrier-aware cost because feasibility can change even
        // under a small smoothing-parameter move. For hybrid runs we still
        // need to reset the ψ-stagnation counter.
        if self.barrier_config.is_none() && max_step_abs < EFS_LINESEARCH_THRESHOLD {
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


fn solution_into_outer_result(
    solution: Solution,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let mut result = OuterResult::new(
        solution.final_point,
        solution.final_value,
        solution.iterations,
        converged,
        plan_used,
    );
    result.final_grad_norm = solution.final_gradient_norm;
    result.final_gradient = solution.final_gradient;
    result.final_hessian = solution.final_hessian;
    result
}


fn outer_result_with_gradient_norm(
    rho: Array1<f64>,
    final_value: f64,
    iterations: usize,
    final_grad_norm: Option<f64>,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let mut result = OuterResult::new(rho, final_value, iterations, converged, plan_used);
    result.final_grad_norm = final_grad_norm;
    result
}


fn outer_result_with_gradient(
    rho: Array1<f64>,
    final_value: f64,
    iterations: usize,
    final_grad_norm: Option<f64>,
    final_gradient: Option<Array1<f64>>,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let mut result = outer_result_with_gradient_norm(
        rho,
        final_value,
        iterations,
        final_grad_norm,
        converged,
        plan_used,
    );
    result.final_gradient = final_gradient;
    result
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
    rho_uncertainty_problem_size: crate::inference::rho_uncertainty::RhoUncertaintyProblemSize,
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
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            objective_scale: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            rho_uncertainty_problem_size:
                crate::inference::rho_uncertainty::RhoUncertaintyProblemSize::default(),
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
    operator_initial_trust_radius: Option<f64>,
    arc_initial_regularization: Option<f64>,
    objective_scale: Option<f64>,
    bfgs_step_cap: Option<f64>,
    bfgs_step_cap_psi: Option<f64>,
    cache_session: Option<Arc<CacheSession>>,
    cache_mirror_sessions: Vec<Arc<CacheSession>>,
    rho_uncertainty_problem_size: crate::inference::rho_uncertainty::RhoUncertaintyProblemSize,
    continuation_prewarm: bool,
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
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            objective_scale: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            rho_uncertainty_problem_size:
                crate::inference::rho_uncertainty::RhoUncertaintyProblemSize::default(),
            continuation_prewarm: true,
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
    // MEASURE-JET ψ REGISTRATION: the engine below is already complete for a
    // 3-coordinate measure-jet ψ group (s, α, ln τ) — `psi_dim` is generic,
    // `with_bounds` carries the s ∈ (0, 2) box (the same convention matern κ
    // uses for its log-κ window; no logistic reparameterization exists or is
    // needed in-house), `with_bfgs_step_cap_psi` caps per-iteration ψ moves,
    // and `DirectionalHyperParam::new_compact` (solver/reml/mod.rs) carries
    // penalty-only first/second/cross jets with `is_penalty_like`
    // auto-derived from the identically-zero design drift (∂X/∂ψ ≡ 0).
    // Every remaining registration arm is formula-layer dispatch in
    // src/terms/smooth.rs (eligibility in
    // `spatial_term_supports_hyper_optimization`, dims in
    // `spatial_dims_per_term`, seed/bounds/write-back on
    // `SpatialLogKappaCoords`, the per-trial rebuild in
    // `apply_log_kappa_to_term`, and the derivative bundle in
    // `try_build_spatial_term_log_kappa_derivative`, which currently returns
    // `Ok(None)` for `SmoothBasisSpec::MeasureJet`) plus the
    // `build_measure_jet_basis_psi_derivatives` producer in
    // src/terms/basis/measure_jet_smooth.rs; both are owned by the
    // measure-jet terms actor. Registration stays gated on those arms — do
    // NOT add measure-jet-specific branches to this engine.
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
    /// Toggle the generic rho-continuation seed pre-warm. This does not affect
    /// objectives that require an explicit continuation path; it only controls
    /// the cheap-by-default pre-pass gated by `allow_continuation_prewarm()`.
    pub fn with_continuation_prewarm(mut self, enabled: bool) -> Self {
        self.continuation_prewarm = enabled;
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
    pub fn with_operator_initial_trust_radius(mut self, radius: Option<f64>) -> Self {
        self.operator_initial_trust_radius = sanitized_operator_trust_restart_radius(radius);
        self
    }

    /// Override the ARC initial cubic-regularization parameter sigma
    /// (default in `opt`: 1.0). Smaller sigma → less cubic penalty on the
    /// first step → larger first move on benign objectives. The matrix-
    /// free Newton-TR analog is `with_operator_initial_trust_radius`.
    ///
    /// Used by Gaussian-identity REML at large-scale n: the objective is
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
    /// ∂/∂logλ. At large-scale n the floor becomes binding even when the
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
    /// path at large scale, where rho needs |d|≈5 while psi wants |d|≤1.
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

    pub fn with_problem_size(mut self, n_obs: usize, p_coefficients: usize) -> Self {
        self.rho_uncertainty_problem_size =
            crate::inference::rho_uncertainty::RhoUncertaintyProblemSize {
                n_obs: Some(n_obs),
                p_coefficients: Some(p_coefficients),
            };
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
            operator_initial_trust_radius: self.operator_initial_trust_radius,
            arc_initial_regularization: self.arc_initial_regularization,
            objective_scale: self.objective_scale,
            bfgs_step_cap: self.bfgs_step_cap,
            bfgs_step_cap_psi: self.bfgs_step_cap_psi,
            cache_session: self.cache_session.clone(),
            cache_mirror_sessions: self.cache_mirror_sessions.clone(),
            rho_uncertainty_problem_size: self.rho_uncertainty_problem_size,
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
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<(), EstimationError>>,
            continuation_prewarm: self.continuation_prewarm,
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
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<(), EstimationError>>,
            continuation_prewarm: self.continuation_prewarm,
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
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<(), EstimationError>>,
            continuation_prewarm: self.continuation_prewarm,
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
                    let plan_used = plan(&cap);
                    log::info!(
                        "[CACHE] final-hit key={}.. context={} rho_dim={} prior_obj={:.6e} iter={} action=skip-outer-validation",
                        short_key,
                        context,
                        rho.len(),
                        prior_obj_display,
                        iterations,
                    );
                    let mut result =
                        OuterResult::new(rho, final_value, iterations, true, plan_used);
                    result.rho_uncertainty_diagnostic = Some(compute_rho_uncertainty_diagnostic(
                        obj, &config, context, &result,
                    ));
                    return Ok(result);
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
                Ok(SeedOutcome::Installed) => log::info!(
                    "[CACHE] beta-warm key={}.. context={} beta_dim={} action=installed",
                    short_key,
                    context,
                    beta.len(),
                ),
                Ok(SeedOutcome::NoSlot) => log::warn!(
                    "[CACHE] beta-warm key={}.. context={} beta_dim={} action=skip \
                     reason=objective_has_no_inner_beta_slot",
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
    /// First-order optimality self-audit at the returned point (#934).
    ///
    /// `None` when no analytic gradient was measured at termination
    /// (gradient-free solvers, cache-hit short-circuits, per-atom EFS) or
    /// when an audit probe failed to evaluate. Populated once by
    /// [`run_outer`] after the solver ladder returns, outside all hot loops.
    pub criterion_certificate: Option<CriterionCertificate>,
    /// Post-fit PSIS diagnostic for whether sampled smoothing-parameter weights
    /// show evidence that plug-in REML/LAML intervals are unreliable. Populated
    /// once by [`run_outer`] when the exact rho Hessian is cheap enough to use.
    pub rho_uncertainty_diagnostic:
        Option<crate::inference::rho_uncertainty::RhoUncertaintyDiagnostic>,
}


impl OuterResult {
    pub fn new(
        rho: Array1<f64>,
        final_value: f64,
        iterations: usize,
        converged: bool,
        plan_used: OuterPlan,
    ) -> Self {
        Self {
            rho,
            final_value,
            iterations,
            final_grad_norm: None,
            final_gradient: None,
            final_hessian: None,
            converged,
            plan_used,
            operator_trust_radius: None,
            operator_stop_reason: None,
            criterion_certificate: None,
            rho_uncertainty_diagnostic: None,
        }
    }

    /// Human-readable rendering of `final_grad_norm` for diagnostics. Returns
    /// `"n/a"` when no gradient was measured (gradient-free / cache-hit paths).
    pub fn final_grad_norm_report(&self) -> String {
        match self.final_grad_norm {
            Some(g) => format!("{g:.3e}"),
            None => "n/a".to_string(),
        }
    }
}


// ─── First-order optimality certificate (#934) ────────────────────────
//
// The objective↔gradient desync bug genus (#748, #752, #808, #901, …) has a
// universal signature: at the returned "optimum" the analytic gradient says
// converged while a finite difference of the ACTUAL criterion value says
// otherwise (or the optimizer stalls and rails λ). Every such bug was
// diagnosed by a human running exactly that FD comparison by hand. The
// certificate makes the engine run it on itself, once, at θ̂, on every fit:
// two central-difference pairs of the VALUE path along one deterministic
// random direction, compared against ∇F(θ̂)·v from the analytic path, plus
// the two ancillary facts every desync postmortem asks for (is the outer
// curvature PD here; did any λ rail to a bound). It is the runtime
// enforcement layer for the criterion-atom architecture (#931): atoms make
// desync structurally hard, the certificate makes any residue observable.
//
// Cost discipline: at most four value-path evaluations at the single final
// point, outside every hot loop. The value path is evaluated through
// `eval_cost` at θ̂±hv — points the gradient path never visited, so the
// existing ρ-keyed caches naturally miss and the true value code runs.
// Disagreement does not fail the fit: it names the broken criterion loudly
// in the result, the log, and the report.

/// Standardized-disagreement gate: the audit flags inconsistency when the
/// analytic and FD directional derivatives differ by more than this many FD
/// error bars (and also fail the relative gate).
const CERTIFICATE_Z_GATE: f64 = 4.0;


/// Relative agreement gate: differences below this fraction of the larger
/// directional derivative are consistent regardless of the (possibly
/// underestimated) FD error bar.
const CERTIFICATE_RELATIVE_GATE: f64 = 1e-3;


/// ρ margin (in log-λ units) within which an outer smoothing coordinate
/// counts as railed against its box bound — the #752 signature (λ → ∞ when
/// a curvature term goes missing from the criterion). Same spirit as
/// [`OVERSMOOTH_BOUNDARY_MARGIN`], which classifies seed *starts*; this one
/// classifies returned *optima*.
const CERTIFICATE_RAIL_MARGIN: f64 = 0.5;


/// First-order optimality certificate: gradient-vs-objective FD audit at the
/// returned optimum (#934).
///
/// Answers, machine-checkably, the three questions every objective↔gradient
/// desync postmortem asks: does the analytic gradient match the actual
/// criterion value HERE ([`Self::first_order_consistent`]); is the outer
/// curvature positive definite HERE (`hessian_pd`); did any smoothing
/// coordinate rail to a box bound (`lambdas_railed`).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CriterionCertificate {
    /// ‖∇F(θ̂)‖₂ from the analytic gradient path at the returned point.
    pub grad_norm: f64,
    /// Analytic directional derivative ∇F(θ̂)·v along the audit direction.
    pub analytic_directional: f64,
    /// Richardson-extrapolated central difference of the criterion VALUE
    /// path along the same direction: (4·D_h − D_2h)/3 from the h and 2h
    /// central-difference pairs.
    pub fd_directional: f64,
    /// Error bar on `fd_directional`: the Richardson residual |D_h − D_2h|
    /// (which absorbs both truncation and inner-solve value noise) floored
    /// by the central-difference roundoff bound ε·|F|/h.
    pub fd_error: f64,
    /// |analytic − fd| / fd_error — standardized disagreement.
    pub agreement_z: f64,
    /// Base central-difference step h along the unit direction.
    pub fd_step: f64,
    /// Whether the final outer Hessian is positive definite at θ̂, when the
    /// solver tracked one (`None` when no final Hessian was available).
    pub hessian_pd: Option<bool>,
    /// Leading smoothing coordinates (ρ block) pinned within
    /// [`CERTIFICATE_RAIL_MARGIN`] of either box bound at the optimum.
    pub lambdas_railed: Vec<usize>,
}


impl CriterionCertificate {
    /// Whether the analytic directional derivative agrees with the finite
    /// difference of the actual criterion value at the optimum.
    ///
    /// Two gates, either suffices: within [`CERTIFICATE_Z_GATE`] FD error
    /// bars (the principled test), or within [`CERTIFICATE_RELATIVE_GATE`]
    /// of the larger derivative (guards against an underestimated error bar
    /// flagging two derivatives that agree to 0.1%).
    pub fn first_order_consistent(&self) -> bool {
        let diff = (self.analytic_directional - self.fd_directional).abs();
        let scale = self
            .analytic_directional
            .abs()
            .max(self.fd_directional.abs());
        diff <= (CERTIFICATE_Z_GATE * self.fd_error).max(CERTIFICATE_RELATIVE_GATE * scale)
    }

    /// Whether every audited fact is clean: gradient matches objective, no
    /// definiteness failure, no railed smoothing coordinate.
    pub fn is_clean(&self) -> bool {
        self.first_order_consistent()
            && self.hessian_pd != Some(false)
            && self.lambdas_railed.is_empty()
    }

    /// One-line human-readable rendering for logs and reports.
    pub fn summary(&self) -> String {
        format!(
            "grad·v={:.6e} fd·v={:.6e}±{:.1e} z={:.2} |g|={:.3e} hessian_pd={} railed={:?} → {}",
            self.analytic_directional,
            self.fd_directional,
            self.fd_error,
            self.agreement_z,
            self.grad_norm,
            match self.hessian_pd {
                Some(true) => "yes",
                Some(false) => "NO",
                None => "n/a",
            },
            self.lambdas_railed,
            if self.first_order_consistent() {
                "consistent"
            } else {
                "GRADIENT-OBJECTIVE DESYNC"
            },
        )
    }
}


/// Deterministic unit direction on the θ sphere for the certificate audit.
///
/// Seeded from the problem fingerprint (context string + θ̂ bits) via FNV-1a
/// and expanded with SplitMix64 + Box–Muller — no clock, no global RNG, so
/// the audit direction is reproducible across runs of the same fit.
fn certificate_audit_direction(theta: &Array1<f64>, context: &str) -> Array1<f64> {
    let mut seed: u64 = 0xcbf2_9ce4_8422_2325;
    let mut fnv = |byte: u8| {
        seed ^= u64::from(byte);
        seed = seed.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for byte in context.bytes() {
        fnv(byte);
    }
    for &x in theta.iter() {
        for byte in x.to_bits().to_le_bytes() {
            fnv(byte);
        }
    }
    let mut state = seed;
    let mut next_unit = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // Uniform in (0, 1): 53 mantissa bits, nudged off zero for the log.
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut direction = Array1::<f64>::zeros(theta.len());
    let mut i = 0;
    while i < direction.len() {
        let (u1, u2) = (next_unit(), next_unit());
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        direction[i] = radius * angle.cos();
        if i + 1 < direction.len() {
            direction[i + 1] = radius * angle.sin();
        }
        i += 2;
    }
    let norm = direction.dot(&direction).sqrt();
    if norm.is_finite() && norm > f64::EPSILON {
        direction.mapv_inplace(|v| v / norm);
        direction
    } else {
        // Degenerate draw (probability ~0): fall back to the first axis.
        let mut fallback = Array1::<f64>::zeros(theta.len());
        fallback[0] = 1.0;
        fallback
    }
}


/// Plain Cholesky positive-definiteness probe for the (small, outer-dim)
/// final Hessian. Returns `None` when the matrix is empty, non-square, or
/// non-finite; `Some(false)` on any non-positive pivot.
fn certificate_hessian_is_pd(hessian: &Array2<f64>) -> Option<bool> {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || hessian.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut chol = hessian.clone();
    for j in 0..n {
        for k in 0..j {
            let l_jk = chol[[j, k]];
            for i in j..n {
                chol[[i, j]] -= chol[[i, k]] * l_jk;
            }
        }
        let pivot = chol[[j, j]];
        if !(pivot > 0.0) || !pivot.is_finite() {
            return Some(false);
        }
        let inv_sqrt = 1.0 / pivot.sqrt();
        for i in j..n {
            chol[[i, j]] *= inv_sqrt;
        }
    }
    Some(true)
}


/// Smoothing coordinates (leading ρ block) railed against the outer box.
fn certificate_railed_lambdas(
    rho: &Array1<f64>,
    rho_dim: usize,
    config: &OuterConfig,
) -> Vec<usize> {
    (0..rho_dim.min(rho.len()))
        .filter(|&k| {
            let (lo, hi) = match config.bounds.as_ref() {
                Some((lo, hi)) if k < lo.len() && k < hi.len() => (lo[k], hi[k]),
                Some(_) => return false,
                None => (-config.rho_bound, config.rho_bound),
            };
            (rho[k] - lo).abs() <= CERTIFICATE_RAIL_MARGIN
                || (hi - rho[k]).abs() <= CERTIFICATE_RAIL_MARGIN
        })
        .collect()
}


/// Perform the randomized first-order self-audit at the returned optimum.
///
/// Requires an analytic final gradient (the thing being audited); returns
/// `None` — never an error — when the gradient is absent/non-finite or when
/// any of the four value probes fails to evaluate, so the audit can never
/// fail a fit that the optimizer accepted.
fn audit_first_order_optimality(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &OuterResult,
) -> Option<CriterionCertificate> {
    let gradient = result.final_gradient.as_ref()?;
    if gradient.is_empty()
        || gradient.len() != result.rho.len()
        || gradient.iter().any(|g| !g.is_finite())
        || result.rho.iter().any(|r| !r.is_finite())
    {
        return None;
    }

    let theta = &result.rho;
    let direction = certificate_audit_direction(theta, context);
    // Central-difference step on the optimal ε^(1/3) scale, sized to the
    // iterate so saturated ρ (|ρ| up to rho_bound) keeps θ̂±2hv resolvable.
    let theta_scale = theta.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let step = f64::EPSILON.cbrt() * (1.0 + theta_scale);

    let mut probe = |scale: f64| -> Option<f64> {
        let point = theta + &(scale * &direction);
        match obj.eval_cost(&point) {
            Ok(value) if value.is_finite() => Some(value),
            Ok(value) => {
                log::debug!(
                    "[CERTIFICATE] {context}: audit probe at θ̂{scale:+.3e}·v returned \
                     non-finite criterion value {value}; certificate skipped"
                );
                None
            }
            Err(err) => {
                log::debug!(
                    "[CERTIFICATE] {context}: audit probe at θ̂{scale:+.3e}·v failed ({err}); \
                     certificate skipped"
                );
                None
            }
        }
    };
    let f_plus_h = probe(step)?;
    let f_minus_h = probe(-step)?;
    let f_plus_2h = probe(2.0 * step)?;
    let f_minus_2h = probe(-2.0 * step)?;

    let d_h = (f_plus_h - f_minus_h) / (2.0 * step);
    let d_2h = (f_plus_2h - f_minus_2h) / (4.0 * step);
    let fd_directional = (4.0 * d_h - d_2h) / 3.0;
    // Error bar: the Richardson residual measures truncation + value-path
    // noise (inner-solve tolerance) empirically; the roundoff bound floors
    // it when the residual is accidentally tiny.
    let value_scale = f_plus_h
        .abs()
        .max(f_minus_h.abs())
        .max(f_plus_2h.abs())
        .max(f_minus_2h.abs());
    let roundoff = f64::EPSILON * (1.0 + value_scale) / step;
    let fd_error = (d_h - d_2h).abs().max(roundoff);

    let analytic_directional = gradient.dot(&direction);
    let grad_norm = gradient.dot(gradient).sqrt();
    let agreement_z = (analytic_directional - fd_directional).abs() / fd_error;

    let rho_dim = obj.capability().theta_layout().rho_dim();
    let certificate = CriterionCertificate {
        grad_norm,
        analytic_directional,
        fd_directional,
        fd_error,
        agreement_z,
        fd_step: step,
        hessian_pd: result
            .final_hessian
            .as_ref()
            .and_then(certificate_hessian_is_pd),
        lambdas_railed: certificate_railed_lambdas(theta, rho_dim, config),
    };
    if certificate.is_clean() {
        log::info!("[CERTIFICATE] {context}: {}", certificate.summary());
    } else {
        log::warn!(
            "[CERTIFICATE warning] {context}: optimality self-audit flagged the returned \
             optimum — {}",
            certificate.summary(),
        );
    }
    Some(certificate)
}


fn compute_rho_uncertainty_diagnostic(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &OuterResult,
) -> crate::inference::rho_uncertainty::RhoUncertaintyDiagnostic {
    let cap = obj.capability();
    let layout = cap.theta_layout();
    let rho_dim = layout.rho_dim();
    let gate = crate::inference::rho_uncertainty::RhoUncertaintyCostGate {
        sample_count: 32,
        problem_size: config.rho_uncertainty_problem_size,
    };
    if let Err(reason) = crate::inference::rho_uncertainty::cost_gate_allows(rho_dim, gate) {
        return crate::inference::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(reason, 0);
    }
    if result.rho.len() != layout.n_params {
        return crate::inference::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
            format!(
                "final outer point length {} does not match objective dimension {}",
                result.rho.len(),
                layout.n_params
            ),
            0,
        );
    }

    let final_eval = match obj.eval_with_order(&result.rho, OuterEvalOrder::ValueGradientHessian) {
        Ok(eval) => eval,
        Err(err) => {
            return crate::inference::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
                format!("final exact Hessian evaluation failed: {err}"),
                1,
            );
        }
    };
    let hessian = match final_eval.hessian.materialize_dense() {
        Ok(Some(hessian)) => hessian,
        Ok(None) => {
            return crate::inference::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
                "exact outer Hessian unavailable at fitted rho",
                1,
            );
        }
        Err(message) => {
            return crate::inference::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
                format!("exact outer Hessian materialization failed: {message}"),
                1,
            );
        }
    };
    if hessian.nrows() != layout.n_params || hessian.ncols() != layout.n_params {
        return crate::inference::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
            format!(
                "exact outer Hessian shape {}x{} does not match objective dimension {}",
                hessian.nrows(),
                hessian.ncols(),
                layout.n_params
            ),
            1,
        );
    }
    let mut hessian_rho = Array2::<f64>::zeros((rho_dim, rho_dim));
    for row in 0..rho_dim {
        for col in 0..rho_dim {
            hessian_rho[[row, col]] = hessian[[row, col]];
        }
    }
    let rho_hat = result.rho.slice(ndarray::s![..rho_dim]).to_owned();
    let theta_hat = result.rho.clone();
    let cost_hat = final_eval.cost;
    let final_beta_hint = final_eval.inner_beta_hint.clone();
    let diagnostic = {
        let mut served_hat_cost = false;
        let mut criterion = |rho: &Array1<f64>| -> Option<f64> {
            let is_hat = rho.len() == rho_hat.len()
                && rho
                    .iter()
                    .zip(rho_hat.iter())
                    .all(|(&left, &right)| left.to_bits() == right.to_bits());
            if is_hat && !served_hat_cost {
                served_hat_cost = true;
                return Some(cost_hat);
            }
            let mut theta = theta_hat.clone();
            for idx in 0..rho_dim {
                theta[idx] = rho[idx];
            }
            if let Some(beta) = final_beta_hint.as_ref()
                && obj.seed_inner_state(beta).is_err()
            {
                return None;
            }
            obj.eval_cost(&theta).ok()
        };
        crate::inference::rho_uncertainty::rho_uncertainty_diagnostic(
            &rho_hat,
            &hessian_rho,
            gate,
            &mut criterion,
        )
    };
    if let Some(beta) = final_beta_hint.as_ref()
        && let Err(err) = obj.seed_inner_state(beta)
    {
        log::debug!(
            "[RHO uncertainty] {context}: final inner-state restore skipped after diagnostic ({err})"
        );
    }
    match &diagnostic.status {
        crate::inference::rho_uncertainty::RhoUncertaintyStatus::NoEvidenceOfHeavyTails => {
            log::info!(
                "[RHO uncertainty] {context}: no heavy-tail evidence at sampled rho proposals k_hat={:.3} evals={}",
                diagnostic.k_hat.unwrap_or(f64::NAN),
                diagnostic.n_evaluations,
            );
        }
        crate::inference::rho_uncertainty::RhoUncertaintyStatus::HeavyTailsDetected { k_hat } => {
            log::warn!(
                "[RHO uncertainty] {context}: heavy rho-importance tail detected k_hat={:.3} evals={}",
                k_hat,
                diagnostic.n_evaluations,
            );
        }
        crate::inference::rho_uncertainty::RhoUncertaintyStatus::Skipped { reason } => {
            log::info!("[RHO uncertainty] {context}: skipped ({reason})");
        }
    }
    diagnostic
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
    let mut result = run_outer_uncertified(obj, config, context)?;
    // First-order optimality self-audit (#934): once, at the returned θ̂,
    // outside all hot loops, for every entry point of the solver ladder
    // (dense, device, per-atom EFS, fallback plans). Probes evaluate the
    // value path at θ̂±hv AFTER the solve, so the only state they perturb
    // is warm-start residue O(h) from the optimum — every caller recovers
    // its fitted state from `result.rho`, not from last-eval residue.
    result.criterion_certificate = audit_first_order_optimality(obj, config, context, &result);
    result.rho_uncertainty_diagnostic = Some(compute_rho_uncertainty_diagnostic(
        obj, config, context, &result,
    ));
    Ok(result)
}


/// The solver ladder behind [`run_outer`], without the #934 self-audit.
fn run_outer_uncertified(
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

    // Frontier ρ-scaling auto-switch (#986): at per-atom-EFS-eligible frontier
    // rho dimension the decoupled per-atom fixed point is the primary outer
    // iteration; everything else falls through to the dense / standard path
    // below. Routed here so every entry point inherits it (magic by default).
    if let Some(result) = run_per_atom_efs_if_frontier(obj, config, context)? {
        return Ok(result);
    }

    if cap.n_params == 0 {
        let cost = obj.eval_cost(&Array1::zeros(0))?;
        let the_plan = plan(&cap);
        return Ok(outer_result_with_gradient_norm(
            Array1::zeros(0),
            cost,
            0,
            Some(0.0),
            true,
            the_plan,
        ));
    }

    // Build the ordered list of capabilities to attempt: primary first, then
    // any centrally-derived degraded capabilities. Aux direct-search has no
    // degraded ladder — a single attempt either succeeds or the failure is
    // surfaced to the caller.
    let fallback_attempts = match config.fallback_policy {
        FallbackPolicy::Automatic => automatic_fallback_attempts(&cap),
        FallbackPolicy::Disabled => Vec::new(),
    };
    let mut attempts: Vec<OuterCapability> = Vec::with_capacity(1 + fallback_attempts.len());
    attempts.push(cap.clone());
    for degraded in fallback_attempts {
        attempts.push(degraded);
    }

    let mut last_error: Option<EstimationError> = None;

    for (attempt_idx, attempt_cap) in attempts.iter().enumerate() {
        let the_plan = plan(attempt_cap);
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


// ─── Frontier ρ-scaling auto-switch (issue #986) ─────────────────────────
//
// ARD-per-atom assigns one smoothing coordinate per dictionary atom, so the
// ρ-vector reaches 10^4–10^5 coordinates. A dense outer quasi-Newton over that
// materializes an O(K²) Hessian and is impossible at scale. When the ρ-dimension
// is frontier-scale AND every coordinate is penalty-like with a working
// fixed-point hook, route the PRIMARY outer iteration to the per-atom decoupled
// EFS path (`crate::solver::estimate::reml::per_atom_efs`) instead of the dense
// ARC/BFGS lane. The decision is auto-derived from the coordinate count alone —
// there is no flag — and it is additive: the dense path is unchanged for small K
// and for any objective that is not per-atom-EFS-eligible.

/// Whether this capability is in the frontier ρ-scaling regime where the
/// per-atom decoupled EFS primary should take over from the dense outer.
///
/// Delegates the eligibility decision to
/// [`crate::solver::estimate::reml::per_atom_efs::per_atom_efs_eligible`], which
/// requires all-penalty-like coordinates, a working `eval_efs` hook,
/// fixed-point not disabled, and a frontier-scale ρ-dimension. This is the
/// single auto-switch predicate; `plan` keeps selecting the
/// dense or standard-EFS solver for everything below the frontier threshold.
pub fn is_per_atom_efs_frontier(cap: &OuterCapability) -> bool {
    crate::solver::estimate::reml::per_atom_efs::per_atom_efs_eligible(cap)
}


/// Auto-switch entry point: when `cap` is frontier-scale per-atom-EFS-eligible,
/// run the per-atom decoupled EFS primary and return its [`OuterResult`];
/// otherwise return `Ok(None)` so the caller falls through to the existing dense
/// / standard-EFS path via [`OuterProblem::run`] / [`run_outer`].
///
/// Builds the same bounded seed and tolerance/budget the standard plan path
/// uses, picks the seed (initial-ρ if supplied, else the first generated
/// candidate — the per-atom fixed point is a contraction near the optimum and
/// does not need the multi-seed cascade the dense path runs for its non-convex
/// quasi-Newton surface), then drives the per-atom EFS loop. The shared-border
/// topology defaults to disjoint (every atom owns a private penalty block — the
/// common ARD-per-atom case); callers with a known arrow-border overlap can run
/// the module's `run_per_atom_efs` directly with a populated
/// `SharedBorderTopology`.
///
/// Additive: this function neither mutates nor bypasses the dense path; it is
/// the pre-dispatch shortcut [`run_outer`] calls before the dense ladder.
fn run_per_atom_efs_if_frontier(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<Option<OuterResult>, EstimationError> {
    let cap = primary_capability_for_config(obj.capability(), config, context);
    cap.validate_layout(context)?;
    if !is_per_atom_efs_frontier(&cap) {
        return Ok(None);
    }

    let the_plan = plan(&cap);
    let rho_dim = cap.theta_layout().rho_dim();

    let (lower, upper) = outer_bounds_template(config, cap.n_params);

    // Seed: cache/explicit initial ρ if present, otherwise the first generated
    // candidate. The per-atom multiplicative fixed point is locally
    // contractive, so a single seed suffices; the heavy multi-seed cascade
    // exists for the dense quasi-Newton's non-convex surface, not for EFS.
    let seed = match config.initial_rho.as_ref() {
        Some(initial) if initial.len() == cap.n_params => initial.clone(),
        _ => {
            let generated = crate::seeding::generate_rho_candidates(
                cap.n_params,
                config.heuristic_lambdas.as_deref(),
                &config.seed_config,
            );
            match generated.into_iter().next() {
                Some(first) => first,
                None => Array1::<f64>::zeros(cap.n_params),
            }
        }
    };

    log::info!(
        "[OUTER] {context}: frontier ρ-scaling (rho_dim={rho_dim}) → per-atom decoupled EFS primary"
    );

    let pa_cfg = crate::solver::estimate::reml::per_atom_efs::PerAtomEfsConfig::new(
        config.tolerance,
        config.max_iter,
        lower,
        upper,
    );
    let topology =
        crate::solver::estimate::reml::per_atom_efs::SharedBorderTopology::disjoint(rho_dim);

    obj.reset();
    let result = crate::solver::estimate::reml::per_atom_efs::run_per_atom_efs(
        obj, &seed, &pa_cfg, &topology,
    )?;
    Ok(Some(result.into_outer_result(the_plan)))
}


fn outer_bounds(lo: &Array1<f64>, hi: &Array1<f64>) -> Result<Bounds, EstimationError> {
    Bounds::new(lo.clone(), hi.clone(), 1e-6).map_err(|err| {
        EstimationError::InvalidInput(format!("outer rho bounds are invalid: {err}"))
    })
}


fn outer_bounds_template(config: &OuterConfig, n: usize) -> (Array1<f64>, Array1<f64>) {
    config.bounds.clone().unwrap_or_else(|| {
        (
            Array1::<f64>::from_elem(n, -config.rho_bound),
            Array1::<f64>::from_elem(n, config.rho_bound),
        )
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


enum FixedPointOuterRunError {
    SeedRejected(EstimationError),
    ImmediateFallback(EstimationError),
    Failed(EstimationError),
}


fn run_fixed_point_outer_solver(
    obj: &mut dyn OuterObjective,
    layout: OuterThetaLayout,
    barrier_config: Option<BarrierConfig>,
    config: &OuterConfig,
    context: &str,
    seed: &Array1<f64>,
    the_plan: OuterPlan,
    label: &str,
    failure_prefix: &str,
) -> Result<OuterResult, FixedPointOuterRunError> {
    let mut objective = OuterFixedPointBridge {
        obj,
        layout,
        barrier_config,
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
                return Err(FixedPointOuterRunError::ImmediateFallback(err));
            }
            return Err(FixedPointOuterRunError::SeedRejected(err));
        }
    };
    let (lo, hi) = outer_bounds_template(config, layout.n_params);
    let bounds = outer_bounds(&lo, &hi).map_err(FixedPointOuterRunError::Failed)?;
    let tol = outer_tolerance(config.tolerance).map_err(FixedPointOuterRunError::Failed)?;
    let max_iter =
        outer_max_iterations(config.max_iter).map_err(FixedPointOuterRunError::Failed)?;
    let mut optimizer = FixedPoint::new(seed.clone(), objective)
        .with_bounds(bounds)
        .with_tolerance(tol)
        .with_max_iterations(max_iter);
    match optimizer.run() {
        Ok(sol) => Ok(solution_into_outer_result(sol, true, the_plan)),
        Err(FixedPointError::MaxIterationsReached { last_solution }) => {
            log::warn!(
                "[OUTER warning] {context}: {label} hit max_iter={} at final_value={:.6e} step_norm={:.3e}",
                config.max_iter,
                last_solution.final_value,
                last_solution.final_gradient_norm.unwrap_or(f64::NAN),
            );
            Ok(solution_into_outer_result(*last_solution, false, the_plan))
        }
        Err(e) => Err(FixedPointOuterRunError::Failed(
            EstimationError::RemlOptimizationFailed(format!("{failure_prefix}: {e:?}")),
        )),
    }
}
