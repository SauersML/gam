//! Family-agnostic identifiability compiler.
//!
//! Single source of truth for cross-block W-metric residualisation across
//! every blockwise family (BMS, SMGS, …). Row-Jacobian compiler that
//! orthogonalises parameter blocks in the *row primary-state* metric `H_i`. Each block
//! exposes a [`RowJacobianOperator`] that maps a coefficient perturbation
//! `δβ ∈ R^p` to its contribution to the per-row primary state
//! `u_i ∈ R^K`. The compiler walks the supplied ordering left-to-right,
//! solves the weighted Gram system against the cumulative anchor, and
//! emits a [`CompiledBlock`] per stage. A post-walk column-pivoted QR
//! audit on the joint primary-state design deterministically drops
//! trailing pivots from the latest block when joint rank is lost.

use std::ops::Range;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, Axis, s};

use faer::Side;
use gam_linalg::decision::{RankDecision, certified_rank, equilibrate_gram};
use gam_linalg::faer_ndarray::{FaerEigh, default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_with_permutation};

/// Slack factor (multiples of machine ε) for the rank-revealing eigenvalue
/// threshold used when pseudo-inverting a Gram matrix or selecting the
/// positive eigenspace of a residual Gram. The retain threshold is
/// `scale · RANK_REVEAL_EPS_SLACK · size · ε`, where `scale` is the dominant
/// eigenvalue (and matrix size accounts for the worst-case roundoff
/// accumulation in the `O(size)` inner products forming each Gram entry). 64×
/// keeps numerically-zero directions out of the kept subspace while preserving
/// every genuinely identified direction at large-scale conditioning.
const RANK_REVEAL_EPS_SLACK: f64 = 64.0;

/// Two-sided multiplicative half-gap for the certified-rank guard band (issue
/// #2337 §9-step-6). A rank decision is host-stable only when no equilibrated
/// eigenvalue lands inside `(τ/(1+gap), τ·(1+gap))`. `gap = 1.0` (a factor-of-2
/// band on each side) is used purely to *observe* Ambiguous frequency in this
/// stage-1, observe-only rollout; the actual retained rank still comes from the
/// unchanged threshold count.
pub(crate) const RANK_DECISION_GAP: f64 = 1.0;

/// Maps a coefficient perturbation `δβ ∈ R^p` for one parameter block into
/// its contribution to the per-row primary state `u_i ∈ R^K`.
///
/// For affine blocks (everything in this compiler), `J_i = ∂u_i/∂β_block` is
/// independent of `β` and equals the transposed row of the block's effective
/// design matrix lifted into `R^K`.
pub trait RowJacobianOperator: Send + Sync {
    /// Dimension of the row primary state (survival marginal-slope: `3 + K`
    /// for `K` score coordinates; Bernoulli: 1).
    fn k(&self) -> usize;

    /// Number of coefficients in this block (= width of `J_i`).
    fn ncols(&self) -> usize;

    /// Number of training rows.
    fn nrows(&self) -> usize;

    /// Apply the row Jacobian: writes `J_i · δβ ∈ R^K` for `row` into `out`.
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]);

    /// Materialise the full operator as an `(n_rows × ncols × K)` tensor.
    fn evaluate_full(&self) -> Array3<f64>;

    /// Build the sqrt(H)-scaled design `W = stack_i sqrt(H_i) · J_i`, flattened
    /// channel-major to `(n_rows·K × ncols)`.
    ///
    /// This is the representation the identifiability *compiler*
    /// ([`compile_with_dual_metric`]) actually consumes — it residualises and
    /// eigendecomposes Grams of `W`, and never indexes the per-row `(n, p, K)`
    /// tensor element-wise. Requesting the scaled design directly lets an
    /// operator with a structured / streaming form supply it without
    /// materialising and cloning the whole `O(n·p·K)` tensor; the default
    /// implementation routes through [`evaluate_full`] so existing operators
    /// remain correct unchanged. (#738: a capability is not a representation —
    /// the compiler asks for the scaled design it needs, not the dense tensor.)
    ///
    /// [`evaluate_full`]: RowJacobianOperator::evaluate_full
    /// [`compile_with_dual_metric`]: crate::families::compiler::compile_with_dual_metric
    fn scaled_design_by_sqrt_h(&self, h_full: &Array3<f64>) -> Array2<f64> {
        scale_block_by_sqrt_h(&self.evaluate_full(), h_full)
    }

    /// Write the channel-flattened column `col` — the `(n_rows · K)` vector
    /// whose entry `i·K + ch` is `J[i, col, ch]` — into `out`.
    ///
    /// This is the representation the identifiability *audit* actually consumes
    /// (per-column leverage statistics and pairwise overlaps), as opposed to the
    /// dense `(n, p, K)` tensor. Requesting a column directly lets an operator
    /// that has a structured / streaming form supply it without materialising
    /// and cloning the whole `O(n·p·K)` tensor on every audit pass; the default
    /// implementation routes through [`evaluate_full`] so existing operators
    /// remain correct unchanged. (#738: a capability is not a representation —
    /// the audit asks for the column view it needs, not the tensor.)
    ///
    /// [`evaluate_full`]: RowJacobianOperator::evaluate_full
    fn channel_flattened_column(&self, col: usize, out: &mut [f64]) {
        let k = self.k();
        let n = self.nrows();
        assert!(
            col < self.ncols(),
            "channel_flattened_column col {col} out of range {}",
            self.ncols()
        );
        assert_eq!(
            out.len(),
            n * k,
            "channel_flattened_column out length {} != n*k = {}*{}",
            out.len(),
            n,
            k
        );
        let full = self.evaluate_full();
        for i in 0..n {
            for ch in 0..k {
                out[i * k + ch] = full[[i, col, ch]];
            }
        }
    }

    /// Write channel-flattened rows for `rows` into `out`.
    ///
    /// `out` has shape `(rows.len() * K, ncols)`, with row
    /// `local_row * K + channel` holding `J[row, :, channel]`. The default
    /// implementation materialises the full tensor for legacy operators; large
    /// construction-time adapters override this to stream row chunks.
    fn channel_flattened_rows(&self, rows: Range<usize>, out: &mut Array2<f64>) {
        let n = self.nrows();
        let start = rows.start.min(n);
        let end = rows.end.min(n);
        let chunk = end - start;
        let k = self.k();
        let p = self.ncols();
        assert_eq!(out.shape(), &[chunk * k, p]);
        let full = self.evaluate_full();
        for local_i in 0..chunk {
            let row = start + local_i;
            for ch in 0..k {
                for col in 0..p {
                    out[[local_i * k + ch, col]] = full[[row, col, ch]];
                }
            }
        }
    }
}

/// Per-row `K × K` PSD Hessian of `−log L_i(u_i)` evaluated at a pilot β.
pub trait RowHessian: Send + Sync {
    fn k(&self) -> usize;
    fn nrows(&self) -> usize;
    /// Fill the `K × K` block at `row` into `out` (row-major).
    fn fill_row(&self, row: usize, out: &mut [f64]);
    /// Materialise full `(n_rows × K × K)` tensor.
    fn evaluate_full(&self) -> Array3<f64>;
}

/// Identity row metric: `K^S_i = I_K` for every row. Default structural
/// metric for [`compile_with_dual_metric`]. Decoupling the
/// "which directions are real structural columns" decision from a
/// possibly rank-deficient pilot curvature `H` prevents the compiler from
/// wrongly dropping columns whose curvature happens to be zero at the
/// pilot β but which would be kept at the optimum.
pub struct IdentityRowHessian {
    n: usize,
    k: usize,
}

impl IdentityRowHessian {
    /// Construct an identity row metric with `n` rows and `K`-channel
    /// row primary state.
    pub fn new(n: usize, k: usize) -> Self {
        Self { n, k }
    }
}

impl RowHessian for IdentityRowHessian {
    fn k(&self) -> usize {
        self.k
    }
    fn nrows(&self) -> usize {
        self.n
    }
    fn fill_row(&self, row: usize, out: &mut [f64]) {
        assert!(
            row < self.n,
            "IdentityRowHessian::fill_row row {row} out of range {n}",
            n = self.n
        );
        assert_eq!(out.len(), self.k * self.k);
        for i in 0..self.k {
            for j in 0..self.k {
                out[i * self.k + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let mut out = Array3::<f64>::zeros((self.n, self.k, self.k));
        for i in 0..self.n {
            for c in 0..self.k {
                out[[i, c, c]] = 1.0;
            }
        }
        out
    }
}

/// One compiled block: reparam matrix `V` (`t_lw`) and the optional anchor
/// correction matrix `M` that downstream blocks consume as a first-class
/// anchor.
pub struct CompiledBlock {
    /// Orthogonal-complement reparam matrix `V ∈ R^{p × p'}` (right-selector).
    pub t_lw: Array2<f64>,
    /// Residualised anchor correction `M ∈ R^{d_raw × p'}` at the compiled
    /// width, expressed in *raw* cumulative-anchor-column coordinates: `d_raw`
    /// is the sum of the raw column counts of every prior block, NOT the
    /// (possibly smaller) count of kept anchor directions. The predict-time
    /// row contribution is `(C(x)·V − A_raw(x)·M)·β`, where `A_raw(x)` is the
    /// raw anchor evaluation. `None` for the first block in the ordering.
    /// Synonymous with `r_lw`.
    pub anchor_correction: Option<Array2<f64>>,
    /// Residualised reparam `R_b = M_b · V_b` — what the residualised row
    /// evaluator uses to subtract the anchor portion. `None` for the first
    /// block in the ordering (no anchor). Equal to `anchor_correction`.
    pub r_lw: Option<Array2<f64>>,
}

/// Output of [`compile`]: one [`CompiledBlock`] per input block plus the
/// joint pre-fit audit verdict.
pub struct CompiledBlocks {
    pub blocks: Vec<CompiledBlock>,
    /// Joint rank reported by the post-walk column-pivoted QR audit.
    pub joint_rank: usize,
    /// Columns deterministically dropped by the audit, as
    /// `(block_idx, local_col)`. The audit drops only from the latest block.
    pub dropped: Vec<(usize, usize)>,
}

/// Structural relationship between one raw penalized block and the higher-priority
/// anchor already accepted by the identifiability compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenalizedDirectionAnnotationKind {
    /// The block kept its full realized-design span; none of its penalized
    /// directions were already represented by a higher-priority block.
    Independent,
    /// Some, but not all, raw directions were absorbed by the higher-priority
    /// anchor. The kept width is the independent residual span.
    PartiallyAbsorbedByHigherPriority,
    /// The entire block was the same realized-design direction/span as the
    /// higher-priority anchor and therefore contributes no independent
    /// coefficients or smoothing parameter directions.
    FullyAbsorbedByHigherPriority,
}

/// Per-block structural annotation emitted by [`orthogonalize_design_blocks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PenalizedDirectionAnnotation {
    pub block_idx: usize,
    pub raw_width: usize,
    pub kept_width: usize,
    pub absorbed_width: usize,
    pub kind: PenalizedDirectionAnnotationKind,
}

/// Errors raised by [`compile`].
#[derive(Debug)]
pub enum CompilerError {
    /// Operator/Hessian/ordering dimensions are inconsistent.
    DimensionMismatch(String),
    /// A supplied row metric is not a finite positive-semidefinite weight.
    InvalidMetric(String),
    /// A block degenerated to zero residual span — fully aliased by the
    /// cumulative anchor in the row metric.
    FullyAliased { block_idx: usize, reason: String },
    /// A linear-algebra step failed (Gram solve, eigendecomposition, QR).
    LinalgFailure(String),
    /// CUDA was configured for this compile, but probing the runtime failed.
    GpuFailure(String),
}

impl std::fmt::Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilerError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            CompilerError::InvalidMetric(msg) => write!(f, "invalid row metric: {msg}"),
            CompilerError::FullyAliased { block_idx, reason } => {
                write!(f, "block {block_idx} fully aliased: {reason}")
            }
            CompilerError::LinalgFailure(msg) => write!(f, "linalg failure: {msg}"),
            CompilerError::GpuFailure(msg) => write!(f, "GPU failure: {msg}"),
        }
    }
}

impl std::error::Error for CompilerError {}

/// Semantic block label. The compiler does not need to know what the block
/// *is*, only its relative order — but downstream consumers (per-family
/// install paths) tag the input operators with these labels so that the
/// compiled output can be routed back to the right runtime slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockOrder {
    Time,
    Marginal,
    Slope,
    ScoreWarp,
    LinkDev,
}

/// Compile a sequence of row-Jacobian operators against a shared row
/// Hessian. Walks `ordering` left-to-right, residualising each block
/// against the cumulative anchor in the `H_i`-weighted row metric, then
/// performs a joint-design audit and emits one [`CompiledBlock`] per
/// input (in the same order as `operators`).
///
/// `ordering` parallels `operators` and supplies the semantic label for
/// each block. The compiler treats `ordering[i]` purely as metadata —
/// the *position* `i` is the residualisation order.
pub fn compile(
    operators: &[Arc<dyn RowJacobianOperator>],
    row_hess: &dyn RowHessian,
    ordering: &[BlockOrder],
) -> Result<CompiledBlocks, CompilerError> {
    compile_protected(operators, row_hess, ordering, &[])
}

/// Variant of [`compile`] that keeps designated blocks at full raw width.
///
/// `protected[b] == true` forces block `b` to retain every raw column,
/// suppressing both the structural and curvature eigenspace drops for that
/// block while still using it as a full-width anchor for later blocks. See
/// [`compile_from_raw_grams_protected`] for the motivation: a block whose
/// effective Jacobian is a fixed nonlinear functional basis (e.g. the survival
/// marginal-slope time-wiggle block) cannot be expressed on a linearly reduced
/// design, so it must not be reparameterised/dropped. `protected` may be
/// shorter than `ordering`; an empty slice reproduces [`compile`] exactly.
pub fn compile_protected(
    operators: &[Arc<dyn RowJacobianOperator>],
    row_hess: &dyn RowHessian,
    ordering: &[BlockOrder],
    protected: &[bool],
) -> Result<CompiledBlocks, CompilerError> {
    // Default structural metric is the per-row identity `K^S_i = I_K`.
    // A pilot-curvature `H` can collapse a direction (zero eigenvalue) at
    // a bad β even though the optimum keeps that direction; routing the
    // rank decision through the structural metric and reserving `H` for
    // *within-kept-subspace* curvature handling prevents that mis-drop.
    let n = row_hess.nrows();
    let k = row_hess.k();
    let id_struct = IdentityRowHessian::new(n, k);
    compile_with_dual_metric_protected(operators, row_hess, &id_struct, ordering, protected)
}

/// Variant of [`compile_with_dual_metric`] that keeps designated blocks at full
/// raw width (see [`compile_protected`] / [`compile_from_raw_grams_protected`]
/// for the motivation). `protected[b] == true` replaces block `b`'s structural
/// and curvature eigenspace drops with identity, so the block emerges at full
/// raw width while still anchoring later blocks. `protected` may be shorter
/// than `ordering`; an empty slice reproduces [`compile_with_dual_metric`].
pub fn compile_with_dual_metric_protected(
    operators: &[Arc<dyn RowJacobianOperator>],
    row_hess: &dyn RowHessian,
    row_structural: &dyn RowHessian,
    ordering: &[BlockOrder],
    protected: &[bool],
) -> Result<CompiledBlocks, CompilerError> {
    if operators.len() != ordering.len() {
        return Err(CompilerError::DimensionMismatch(format!(
            "operators ({}) and ordering ({}) length mismatch",
            operators.len(),
            ordering.len()
        )));
    }
    if operators.is_empty() {
        return Ok(CompiledBlocks {
            blocks: Vec::new(),
            joint_rank: 0,
            dropped: Vec::new(),
        });
    }

    let k = row_hess.k();
    let n = row_hess.nrows();
    if row_structural.k() != k {
        return Err(CompilerError::DimensionMismatch(format!(
            "structural row metric has K={} but curvature row Hessian has K={k}",
            row_structural.k()
        )));
    }
    if row_structural.nrows() != n {
        return Err(CompilerError::DimensionMismatch(format!(
            "structural row metric has nrows={} but curvature row Hessian has nrows={n}",
            row_structural.nrows()
        )));
    }
    for (idx, op) in operators.iter().enumerate() {
        if op.k() != k {
            return Err(CompilerError::DimensionMismatch(format!(
                "operator {idx} has K={} but row Hessian has K={k}",
                op.k()
            )));
        }
        if op.nrows() != n {
            return Err(CompilerError::DimensionMismatch(format!(
                "operator {idx} has nrows={} but row Hessian has nrows={n}",
                op.nrows()
            )));
        }
    }

    // Materialise once per metric. K is tiny (1 or 4) so the K×K
    // symmetric-sqrt cost is dominated by the joint-design audit below.
    let h_full = row_hess.evaluate_full();
    let s_full = row_structural.evaluate_full();

    // Request each block's sqrt(H)-scaled design directly through the intent
    // accessor — the `(n·K, p)` representation the compiler actually consumes —
    // instead of first materialising the dense `(n, p, K)` per-row tensor and
    // scaling it. The default `scaled_design_by_sqrt_h` impl still routes
    // through `evaluate_full()`, so operators without a structured form stay
    // correct unchanged; a streaming operator (e.g. `BlockJacobianAsRowOp`)
    // overrides it to scale straight out of its stored layout, dropping the
    // `O(n·p·K)` tensor clone that `evaluate_full()` performs per block at
    // large-scale `n`. (#738: a capability is not a representation — the compiler
    // asks for the scaled design it needs, never the dense tensor.)
    let scaled_h: Vec<Array2<f64>> = operators
        .iter()
        .map(|op| op.scaled_design_by_sqrt_h(&h_full))
        .collect();
    let scaled_s: Vec<Array2<f64>> = operators
        .iter()
        .map(|op| op.scaled_design_by_sqrt_h(&s_full))
        .collect();

    let mut compiled: Vec<CompiledBlock> = Vec::with_capacity(operators.len());
    // Demotions that happen *inside* the per-block walk (a structurally-kept
    // block losing all its directions to a higher-priority anchor in the
    // structural or curvature pass) are recorded here, one entry per demoted
    // raw column, in the same `(block_idx, local_col)` convention that
    // `audit_and_drop_trailing_pivots` emits at the joint-audit step. Without
    // this, a zero-width demotion vanished from `dropped`, breaking the
    // `kept_width + dropped_count == structural_pre_audit_width` accounting.
    let mut walk_demotions: Vec<(usize, usize)> = Vec::new();
    let mut anchor_h: Array2<f64> = Array2::zeros((n * k, 0));
    let mut anchor_s: Array2<f64> = Array2::zeros((n * k, 0));
    // Cumulative *raw* (un-residualised) curvature-scaled anchor: the
    // horizontal stack of `sqrt(H)·J_b` for every block already walked,
    // keeping one column per raw block column. Where `anchor_h` carries the
    // residualised, kept-direction anchor (its width shrinks whenever a block
    // sheds an aliased column), this matrix keeps the full raw column count so
    // the emitted `anchor_correction` can be expressed in raw-anchor-column
    // coordinates — exactly the basis the predict-time subtraction
    // `A_raw(x)·M` evaluates against. See the `M_raw` derivation below.
    let mut raw_anchor_h: Array2<f64> = Array2::zeros((n * k, 0));

    for idx in 0..operators.len() {
        let w_h = &scaled_h[idx];
        let w_s = &scaled_s[idx];
        let p_b = w_h.ncols();
        let block_protected = protected.get(idx).copied().unwrap_or(false);

        // A zero-width block owns no raw columns, so it cannot alias against any
        // anchor and is trivially identifiable. Emit an empty compiled block and
        // skip the structural/curvature passes: their residual Grams are 0×0 and
        // yield no positive eigenspace, which the `anchor_h.ncols() == 0`
        // first-block guards below would otherwise mis-report as `FullyAliased`
        // even though there is nothing to alias. This mirrors the empty block a
        // fully-absorbed later block compiles to, with no demotions to record
        // (there are no columns) and no change to the running anchors.
        if p_b == 0 {
            compiled.push(CompiledBlock {
                t_lw: Array2::<f64>::zeros((0, 0)),
                anchor_correction: Some(Array2::<f64>::zeros((raw_anchor_h.ncols(), 0))),
                r_lw: Some(Array2::<f64>::zeros((raw_anchor_h.ncols(), 0))),
            });
            continue;
        }

        // Pass 1 (structural): residualise W^S_b against cumulative
        // structural anchor; eigendecompose the structural residual Gram
        // and keep only directions with non-zero structural mass → D
        // (raw-block selector).
        // Only the structural residual is consumed downstream; the
        // structural-metric correction M^S is intentionally discarded —
        // predict-time subtraction uses the curvature metric correction
        // (`M^H_inner` below), not the structural one.
        let (residual_s, _) = residualise_in_metric(&anchor_s, w_s)?;
        let g_s = fast_atb(&residual_s, &residual_s);
        // Scale reference for the kept-eigenspace tolerance: the *original*
        // (pre-residualisation) structural block Gram trace. A fully-absorbed
        // block's residual collapses to ~ε² noise; anchoring tau to that would
        // keep the noise directions and wrongly treat the block as
        // structurally independent. The original-block trace is invariant to
        // absorption, so a near-zero residual is rejected as fully absorbed.
        let g_s_bb = fast_atb(w_s, w_s);
        let g_s_trace: f64 = (0..p_b).map(|i| g_s_bb[[i, i]].max(0.0)).sum();
        // A protected block keeps every raw column: the structural residual
        // eigenfilter is replaced by identity so no within-block direction is
        // dropped. It still anchors later blocks at full raw width.
        let d = if block_protected {
            Array2::<f64>::eye(p_b)
        } else {
            keep_positive_eigenspace(&g_s, n, k, g_s_trace)?
        };
        if d.ncols() == 0 {
            if anchor_h.ncols() == 0 {
                return Err(CompilerError::FullyAliased {
                    block_idx: idx,
                    reason: format!(
                        "structural residual Gram has no positive eigenspace (block of width {p_b} has zero structural span before any anchor exists)"
                    ),
                });
            }
            compiled.push(CompiledBlock {
                t_lw: Array2::<f64>::zeros((p_b, 0)),
                anchor_correction: Some(Array2::<f64>::zeros((raw_anchor_h.ncols(), 0))),
                r_lw: Some(Array2::<f64>::zeros((raw_anchor_h.ncols(), 0))),
            });
            // The structural pass fully absorbed all `p_b` raw columns into the
            // higher-priority anchor: record each as a drop so the per-block
            // width accounting (kept + dropped == raw width) stays exact.
            for c in 0..p_b {
                walk_demotions.push((idx, c));
            }
            raw_anchor_h = concat_cols(&raw_anchor_h, w_h);
            continue;
        }

        // Pass 2 (curvature): form W^H_b · D and residualise against the
        // cumulative curvature anchor. Eigendecompose the curvature
        // residual Gram and drop curvature-zero directions inside D →
        // T_inner. A direction kept by the structural pass but degenerate
        // here is genuinely curvature-redundant *within* the
        // structurally-kept basis, so dropping it is correct.
        let w_h_d = fast_ab(w_h, &d);
        let (residual_h, m_h_inner_opt) = residualise_in_metric(&anchor_h, &w_h_d)?;
        let g_h = fast_atb(&residual_h, &residual_h);
        let p_d = d.ncols();
        // Scale reference: the *unresidualised* curvature block Gram trace of
        // `W^H_b · D` (the same convention the closed-form `compile_from_raw_grams`
        // path uses with `d_t_kh_d`). Anchoring to the residual trace would
        // collapse to ~ε² when the block is fully curvature-absorbed and keep
        // its noise directions.
        let g_h_dd = fast_atb(&w_h_d, &w_h_d);
        let g_h_trace: f64 = (0..p_d).map(|i| g_h_dd[[i, i]].max(0.0)).sum();
        // Protected block: retain every structurally-kept direction (identity
        // curvature span) instead of dropping curvature-degenerate ones; its own
        // penalty nullspace regularises the conditioning downstream.
        let t_inner = if block_protected {
            Array2::<f64>::eye(p_d)
        } else {
            keep_positive_eigenspace(&g_h, n, k, g_h_trace)?
        };
        if t_inner.ncols() == 0 {
            if anchor_h.ncols() == 0 {
                return Err(CompilerError::FullyAliased {
                    block_idx: idx,
                    reason: format!(
                        "curvature residual Gram has no positive eigenspace within structurally-kept basis (block of width {p_b}, structural-kept {p_d}) before any anchor exists"
                    ),
                });
            }
            compiled.push(CompiledBlock {
                t_lw: Array2::<f64>::zeros((p_b, 0)),
                anchor_correction: Some(Array2::<f64>::zeros((raw_anchor_h.ncols(), 0))),
                r_lw: Some(Array2::<f64>::zeros((raw_anchor_h.ncols(), 0))),
            });
            // The structural pass kept `p_d` directions, but the curvature pass
            // absorbed all of them into the higher-priority anchor. Record each
            // structurally-kept-but-curvature-demoted direction as a drop so the
            // pre-audit structural width is fully accounted for.
            for c in 0..p_d {
                walk_demotions.push((idx, c));
            }
            raw_anchor_h = concat_cols(&raw_anchor_h, w_h);
            continue;
        }

        // Compose V = D · T_inner (raw-block → kept).
        let v = fast_ab(&d, &t_inner);

        // `m_h_inner_opt` was residualised against `anchor_h` as it stands
        // *here*, i.e. the cumulative kept-direction anchor of all PRIOR
        // blocks. Snapshot that pre-append anchor and its raw counterpart
        // before this block's residual columns are appended below; the
        // change-of-basis for this block's correction must be expressed
        // against the prior-block anchor that `m` is indexed against, not the
        // post-append anchor that already carries this block's own columns.
        let prior_anchor_h = anchor_h.clone();
        let prior_raw_anchor_h = raw_anchor_h.clone();

        // Append residual-V columns to both cumulative anchors so future
        // blocks see the structurally-orthogonal and curvature-orthogonal
        // residual designs of this block, never the raw scaled block.
        let residual_h_t = fast_ab(&residual_h, &t_inner);
        anchor_h = concat_cols(&anchor_h, &residual_h_t);
        // The structural anchor needs the structural-residual restricted
        // to the kept directions: residual_s · v gives (W^S_b − A^S · M^S)·V.
        let residual_s_v = fast_ab(&residual_s, &v);
        anchor_s = concat_cols(&anchor_s, &residual_s_v);

        // Compiled anchor correction lives in the curvature metric — the
        // predict-time row contribution is `(C(x) · V − A(x) · M)·β`, where
        // the subtraction makes residuals H-orthogonal at training and `A(x)`
        // is the *raw* anchor evaluation (one column per raw anchor column).
        //
        // `m_h_inner_opt · t_inner` (call it `M_kept`) lives in the
        // *kept-direction* anchor coordinates of the PRIOR-block anchor
        // `prior_anchor_h` (the value `anchor_h` held when `m` was produced at
        // `residualise_in_metric` above, before this block's residual columns
        // were appended). Its row count is `prior_anchor_h.ncols()`, which
        // equals the prior-block raw anchor width only when no upstream block
        // shed an aliased column. The predict path multiplies by the raw
        // anchor matrix `A_raw` (one column per raw anchor column of the prior
        // blocks), so we must re-express `M_kept` in raw-anchor-column
        // coordinates.
        //
        // `prior_anchor_h` and `prior_raw_anchor_h` span the same column space
        // in the curvature metric (the residualisation/rotation only drops
        // directions that lie inside that span), so there is an exact `Z` with
        // `prior_raw_anchor_h · Z = prior_anchor_h`. Then
        //   `prior_anchor_h · M_kept = prior_raw_anchor_h · (Z · M_kept)`,
        // and the raw-coordinate correction is `M_raw = Z · M_kept`, with row
        // count `prior_raw_anchor_h.ncols()` = the sum of prior raw anchor
        // block widths. `Z = (Aᵀ A)⁺ Aᵀ prior_anchor_h` (with
        // `A = prior_raw_anchor_h`) is the metric-exact least-squares change of
        // basis (`solve_psd_system`).
        let m_compiled = match m_h_inner_opt.as_ref() {
            Some(m) => {
                let m_kept = fast_ab(m, &t_inner);
                if m_kept.nrows() != prior_anchor_h.ncols() {
                    return Err(CompilerError::DimensionMismatch(format!(
                        "anchor correction must be indexed by prior-block kept anchor directions: \
                         m_kept has {} rows but prior_anchor_h has {} columns",
                        m_kept.nrows(),
                        prior_anchor_h.ncols()
                    )));
                }
                let g_raw = fast_atb(&prior_raw_anchor_h, &prior_raw_anchor_h);
                let z_rhs = fast_atb(&prior_raw_anchor_h, &prior_anchor_h);
                let z = solve_psd_system(&g_raw, &z_rhs)?;
                Some(fast_ab(&z, &m_kept))
            }
            None => None,
        };
        compiled.push(CompiledBlock {
            t_lw: v,
            anchor_correction: m_compiled.clone(),
            r_lw: m_compiled,
        });

        // Append this block's raw curvature-scaled columns to the raw anchor
        // accumulator so the *next* block's `M_raw` is expressed against the
        // full raw column set of all blocks walked so far.
        raw_anchor_h = concat_cols(&raw_anchor_h, w_h);
    }

    // Joint-design audit on the curvature-scaled cumulative anchor: the
    // identifiability question the fit cares about is curvature-rank.
    let audit_dropped = audit_and_drop_trailing_pivots(&anchor_h, &mut compiled)?;
    // Combine in-walk demotions (structural / curvature full absorption of a
    // block) with the joint-audit trailing-pivot drops so `dropped` accounts
    // for *every* column the compiler removed, not just the joint-audit ones.
    let mut dropped = walk_demotions;
    dropped.extend(audit_dropped);
    let joint_rank: usize = compiled.iter().map(|b| b.t_lw.ncols()).sum();

    Ok(CompiledBlocks {
        blocks: compiled,
        joint_rank,
        dropped,
    })
}

/// Build `W_b = stack_i sqrt(H_i) · J_b,i` flattened to `(n*K, ncols)` without
/// ever requiring a materialised `(n, p, K)` tensor.
///
/// The Jacobian entries are pulled through the `jac` closure
/// (`jac(i, a, c) = J_b,i[a, c]`), so a structured operator that stores its
/// Jacobian in a compact / streaming form can supply the sqrt(H)-scaled design
/// directly — the representation the compiler actually consumes — rather than
/// being forced to clone a dense `(n, p, K)` tensor first. (#738: a capability
/// is not a representation — the compiler asks for the scaled `(n·K, p)` design
/// it needs, not the dense per-row tensor.)
///
/// `K` is tiny (1 or 4), so the per-row symmetric sqrt is negligible relative
/// to the overall compile.
pub fn scale_jacobian_by_sqrt_h_with(
    n: usize,
    p: usize,
    k: usize,
    h_full: &Array3<f64>,
    jac: impl Fn(usize, usize, usize) -> f64,
) -> Array2<f64> {
    assert_eq!(h_full.shape(), &[n, k, k]);
    let mut out = Array2::<f64>::zeros((n * k, p));
    let mut sqrt_h = Array2::<f64>::zeros((k, k));
    let mut scratch_jrow = Array2::<f64>::zeros((p, k));
    for i in 0..n {
        // Symmetric square root of H_i via eigendecomposition.
        let h_i = h_full.index_axis(Axis(0), i).to_owned();
        sqrt_h.fill(0.0);
        symmetric_sqrt_into(&h_i, &mut sqrt_h);
        // scratch_jrow[a, c] = J_b,i[a, c] (transpose-friendly layout for
        // the GEMV below: we want (p × k) · (k,) = (p,) for each column of
        // sqrt_h, but we batch by writing out[(i*k+c), a] = (sqrt_h · J_b,iᵀ)[c, a].
        for a in 0..p {
            for c in 0..k {
                scratch_jrow[[a, c]] = jac(i, a, c);
            }
        }
        for c in 0..k {
            for a in 0..p {
                let mut acc = 0.0;
                for cp in 0..k {
                    acc += sqrt_h[[c, cp]] * scratch_jrow[[a, cp]];
                }
                out[[i * k + c, a]] = acc;
            }
        }
    }
    out
}

/// Symmetric matrix square root via eigendecomposition with negative
/// eigenvalues clamped to zero (PSD projection guard).
pub(crate) fn symmetric_sqrt_into(m: &Array2<f64>, out: &mut Array2<f64>) {
    let k = m.nrows();
    assert_eq!(m.ncols(), k);
    assert_eq!(out.shape(), &[k, k]);
    if k == 1 {
        out[[0, 0]] = m[[0, 0]].max(0.0).sqrt();
        return;
    }
    let (evals, evecs) = match m.eigh(Side::Lower) {
        Ok(pair) => pair,
        Err(_) => {
            // Fall back to clipped diagonal — extremely defensive for the
            // K=4 row Hessian which is already PSD-clamped by the caller.
            out.fill(0.0);
            for i in 0..k {
                out[[i, i]] = m[[i, i]].max(0.0).sqrt();
            }
            return;
        }
    };
    // out = U · diag(sqrt(max(0, λ))) · Uᵀ
    let mut scaled = evecs.clone();
    for j in 0..k {
        let s = evals[j].max(0.0).sqrt();
        for i in 0..k {
            scaled[[i, j]] *= s;
        }
    }
    out.assign(&fast_atb(&evecs.t().to_owned(), &scaled.t().to_owned()));
    // The above fast_atb computed (Uᵀ)ᵀ · (Uᵀ·diag(s)) = U · diag(s) · Uᵀ
    // when the inputs are owned. To be safe and avoid layout surprises,
    // re-do the small multiplication explicitly for K ≤ 4.
    out.fill(0.0);
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for l in 0..k {
                acc += evecs[[i, l]] * evals[l].max(0.0).sqrt() * evecs[[j, l]];
            }
            out[[i, j]] = acc;
        }
    }
}

/// Solve `Aᵀ A · M = Aᵀ B` and return `(B − A·M, Some(M))`. With `A`
/// having zero columns, returns `(B, None)` — the first block needs no
/// anchor correction.
fn residualise_in_metric(
    a_scaled: &Array2<f64>,
    b_scaled: &Array2<f64>,
) -> Result<(Array2<f64>, Option<Array2<f64>>), CompilerError> {
    let d = a_scaled.ncols();
    if d == 0 {
        return Ok((b_scaled.clone(), None));
    }
    let g_aa = fast_atb(a_scaled, a_scaled);
    let g_ab = fast_atb(a_scaled, b_scaled);
    let m = solve_psd_system(&g_aa, &g_ab)?;
    let a_m = fast_ab(a_scaled, &m);
    let residual = b_scaled - &a_m;
    Ok((residual, Some(m)))
}

/// Solve a PSD linear system `G · M = R` for `M`. Tries the eigen-based
/// pseudoinverse with a relative threshold and falls back to a damped
/// solve if the spectrum is ill-conditioned beyond what the threshold
/// can clean.
fn solve_psd_system(g: &Array2<f64>, r: &Array2<f64>) -> Result<Array2<f64>, CompilerError> {
    let n = g.nrows();
    if n == 0 {
        return Ok(Array2::zeros((0, r.ncols())));
    }
    let (evals, evecs) = g
        .eigh(Side::Lower)
        .map_err(|err| CompilerError::LinalgFailure(format!("Gram eigh failed: {err:?}")))?;
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max).max(0.0);
    let tol = lambda_max * RANK_REVEAL_EPS_SLACK * (n.max(1) as f64) * f64::EPSILON;
    // M = U · diag(1/λ_kept) · Uᵀ · R
    let u_t_r = fast_atb(&evecs, r);
    let mut scaled = u_t_r.clone();
    for i in 0..n {
        let lam = evals[i];
        let inv = if lam > tol { 1.0 / lam } else { 0.0 };
        for j in 0..scaled.ncols() {
            scaled[[i, j]] *= inv;
        }
    }
    let m = fast_ab(&evecs, &scaled);
    Ok(m)
}

/// Eigendecompose the residual Gram `G̃` and return `V` made of the
/// eigenvectors whose eigenvalues exceed
/// `τ = max(λ_max(G̃), tr(G_BB)) · RANK_REVEAL_EPS_SLACK · n · K · ε`.
fn keep_positive_eigenspace(
    g_tilde: &Array2<f64>,
    n: usize,
    k: usize,
    g_bb_trace: f64,
) -> Result<Array2<f64>, CompilerError> {
    let p = g_tilde.nrows();
    if p == 0 {
        return Ok(Array2::zeros((0, 0)));
    }
    // A block whose UNRESIDUALISED diagonal trace is zero owns no positive
    // eigenspace (an all-zero residual Gram): rank 0.
    if g_bb_trace <= 0.0 {
        return Ok(Array2::zeros((p, 0)));
    }
    let (evals, evecs) = g_tilde.eigh(Side::Lower).map_err(|err| {
        CompilerError::LinalgFailure(format!("residual Gram eigh failed: {err:?}"))
    })?;

    // WEIGHT-INVARIANT RANK COUNT. The rank tolerance is relative to the dominant
    // eigenvalue, so a single stiff residual direction inflates it until
    // well-conditioned independent directions are dropped. The marginal-slope
    // effective Jacobian carries the per-row chain weight c_i = sqrt(1+(s·g_i)²);
    // it produced a residual Gram spectrum σ² of [7.5e15, 1.3e3, …, 2.3e-3] — one
    // stiff direction and eleven absolutely well-conditioned ones — and the
    // lambda_max-relative cutoff dropped all eleven, reporting range_rank 1/12 on
    // a fully identified time surface. Identifiability is invariant to a positive
    // per-column scaling (a diagonal congruence D^{-1/2}·G·D^{-1/2} preserves rank
    // and inertia), so take the rank COUNT from the diagonally-equilibrated
    // residual Gram, whose cutoff sees true residual correlation rather than
    // scale. Return the RAW eigenvectors (top-`rank` by descending raw
    // eigenvalue), so blocks already ranked correctly are byte-identical — only
    // stiff-direction-mislabeled blocks gain their true rank.
    // Problem-size factor shared by the equilibrated rank cutoff and the raw
    // absorption floor below, so the two stay in the same currency.
    let nk = (n.saturating_mul(k)).max(p).max(1) as f64;

    // BLOCK-LEVEL FULL ABSORPTION, decided in the RAW gauge before any
    // equilibration. This is a different question from "which directions
    // survive", and it is the only one a first-order tolerance can answer safely.
    //
    // Callers hand this function residual Grams built by different arithmetic.
    // `orthogonalize_design_blocks` residualises the weighted DESIGN and then
    // squares it, so an absorbed block's residual Gram is `O(ε²·tr(G_BB))`.
    // `compile_from_raw_grams` instead forms a Schur complement of Grams,
    // `G_bb − G_abᵀ·G_aa⁺·G_ab`, which is ONE cancellation between computed
    // `O(tr)` quantities through a pseudo-inverse: an absorbed block leaves
    // `O(κ(G_AA)·ε·tr(G_BB))`, first order in ε. A per-direction floor at that
    // first-order level is NOT safe — the marginal-slope effective Jacobian's
    // smallest genuine direction sits at `2.3e-3` of `7.5e15 ≈ 3e-19`, five
    // orders below it, and `b58bd1909` records that this is exactly the route
    // (`compile_from_raw_grams → keep_positive_eigenspace`) that regression
    // travelled.
    //
    // But FULL absorption is a statement about the whole residual, not about one
    // direction: for an exact alias `B = A·L` the Schur complement is
    // identically zero in exact arithmetic, so EVERY eigenvalue is noise,
    // `λ_max` included. Testing `λ_max` therefore separates the two regimes with
    // no ambiguity — the stiff case keeps its block alive on `λ_max = 7.5e15`
    // however small its other directions are, and cannot be touched by this
    // branch. Partial absorption is left entirely to the equilibrated count
    // below, which is where it belongs.
    let lambda_max_raw = evals.iter().cloned().fold(0.0_f64, f64::max).max(0.0);
    if lambda_max_raw <= g_bb_trace * RANK_REVEAL_EPS_SLACK * nk * f64::EPSILON {
        return Ok(Array2::zeros((p, 0)));
    }

    let rank = {
        // Diagonally equilibrate into the column-scale gauge (Sylvester's law of
        // inertia: the congruence preserves rank), then take the count from the
        // equilibrated spectrum. See `gam_linalg::decision::equilibrate_gram`.
        let (g_eq, _) = equilibrate_gram(g_tilde);
        let (evals_eq, _) = g_eq.eigh(Side::Lower).map_err(|err| {
            CompilerError::LinalgFailure(format!("equilibrated residual Gram eigh failed: {err:?}"))
        })?;
        let lambda_max_eq = evals_eq.iter().cloned().fold(0.0_f64, f64::max).max(0.0);
        let tau_eq = lambda_max_eq * RANK_REVEAL_EPS_SLACK * nk * f64::EPSILON;
        // Threshold count the pipeline has always acted on: the decision we must
        // preserve exactly.
        let threshold_count = evals_eq.iter().filter(|&&e| e > tau_eq).count();
        // Two-stage rollout (#2337 §9-step-6). STAGE 1 — OBSERVE ONLY: classify
        // the same decision against a two-sided guard band. When the band is
        // clean the certified rank equals `threshold_count` by construction (no
        // eigenvalue lies in `(τ/(1+gap), τ·(1+gap))`, so `#{e ≥ high}` =
        // `#{e > τ}`). When a value sits inside the band the decision is
        // host-unstable; we do NOT refuse here — we log the payload so we can
        // measure Ambiguous frequency before enforcing a refusal path in stage 2
        // — and fall back to the preserved threshold count.
        match certified_rank(
            evals_eq.as_slice().unwrap_or(&[]),
            tau_eq,
            RANK_DECISION_GAP,
        ) {
            RankDecision::Certified { rank, .. } => rank,
            RankDecision::Ambiguous {
                rank_floor,
                rank_ceil,
                sigma_in_band,
                tol,
                gap,
            } => {
                log::warn!(
                    "keep_positive_eigenspace: ambiguous equilibrated rank (observe-only, \
                     #2337 stage 1): rank_floor={rank_floor}, rank_ceil={rank_ceil}, \
                     sigma_in_band={sigma_in_band:.3e}, tol={tol:.3e}, gap={gap}, \
                     falling back to threshold_count={threshold_count}"
                );
                threshold_count
            }
        }
    };

    // ABSORPTION FLOOR, in the RAW gauge. Equilibration is scale-invariant by
    // construction — that is exactly why it fixes the stiff-direction case — and
    // that same invariance makes it blind to a block that carries no residual at
    // all. When block `b` is fully absorbed by a higher-priority anchor its
    // residual Gram is pure roundoff, `O(ε²·tr(G_BB))` in EVERY direction;
    // dividing each column by its own `√diag` turns that noise into a
    // correlation matrix with unit diagonal and `O(1)` eigenvalues, so a cutoff
    // relative to `λ_max(G_eq)` admits all of it and the block is reported
    // `Independent`. This function's own contract already says the tolerance is
    // relative to `max(λ_max(G̃), tr(G_BB))`; the `tr(G_BB)` half was what the
    // equilibrated count dropped.
    //
    // Restore it as an absolute admission test rather than by inflating `τ_eq`:
    // a direction is genuine only if its residual NORM clears the roundoff of
    // the ORIGINAL block norm, i.e. `√(λ_raw / tr(G_BB)) > SLACK·n·K·ε`. The
    // floor is therefore the SQUARE of the usual rank tolerance, because a Gram
    // is the square of the residual it is built from. That separates the two
    // regimes by orders of magnitude in both directions and does not re-open the
    // stiff case: a fully absorbed block sits at `λ_raw/tr ≈ ε² ≈ 5e-32`, while
    // the marginal-slope effective Jacobian's smallest GENUINE direction sat at
    // `2.3e-3 / 7.5e15 ≈ 3e-19` — seven orders above this floor and seven below
    // where a raw λ_max-relative cutoff would have killed it.
    //
    // Blocks where the floor does not bind are untouched, so every already-ranked
    // block keeps its exact previous basis.
    let raw_absorption_floor = {
        let rel = RANK_REVEAL_EPS_SLACK * nk * f64::EPSILON;
        g_bb_trace * rel * rel
    };

    // Top-`rank` RAW eigenvectors by descending raw eigenvalue (stable order),
    // restricted to those clearing the absorption floor.
    let mut kept: Vec<usize> = (0..p).collect();
    kept.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    kept.retain(|&i| evals[i] > raw_absorption_floor);
    kept.truncate(rank);
    let mut v = Array2::<f64>::zeros((p, kept.len()));
    for (out_col, &src_col) in kept.iter().enumerate() {
        for row in 0..p {
            v[[row, out_col]] = evecs[[row, src_col]];
        }
    }
    Ok(v)
}

/// Concatenate two matrices column-wise. Both must have the same row count.
fn concat_cols(left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
    let nrows = left.nrows().max(right.nrows());
    let lc = left.ncols();
    let rc = right.ncols();
    let mut out = Array2::<f64>::zeros((nrows, lc + rc));
    if lc > 0 {
        out.slice_mut(s![.., ..lc]).assign(left);
    }
    if rc > 0 {
        out.slice_mut(s![.., lc..]).assign(right);
    }
    out
}

/// Post-walk audit: column-pivoted QR on the cumulative scaled design.
/// If rank < p_total, deterministically drop trailing pivots from the
/// latest block's `V`. Earlier blocks are never modified.
fn audit_and_drop_trailing_pivots(
    w_joint: &Array2<f64>,
    compiled: &mut [CompiledBlock],
) -> Result<Vec<(usize, usize)>, CompilerError> {
    let p_total: usize = compiled.iter().map(|b| b.t_lw.ncols()).sum();
    if p_total == 0 || w_joint.nrows() == 0 {
        return Ok(Vec::new());
    }

    // RRQR rank with the codebase's default α.
    let rrqr = rrqr_with_permutation(w_joint, default_rrqr_rank_alpha())
        .map_err(|err| CompilerError::LinalgFailure(format!("audit RRQR failed: {err:?}")))?;
    let rank = rrqr.rank;
    if rank >= p_total {
        return Ok(Vec::new());
    }

    // Trailing pivots are the redundant columns. Attribute every demoted
    // global column to the *latest* block by truncating its V; earlier
    // blocks keep their full V. The demoted suffix is sorted only by
    // pivot order, but we drop deterministically: take the count of
    // demoted columns and truncate that many trailing columns of the
    // latest block.
    let drop_count = p_total - rank;
    let latest_idx = compiled.len() - 1;
    let latest = &mut compiled[latest_idx];
    let kept_local = latest.t_lw.ncols().saturating_sub(drop_count);
    let dropped_locals: Vec<(usize, usize)> = (kept_local..latest.t_lw.ncols())
        .map(|c| (latest_idx, c))
        .collect();
    // Truncate ALL kept-direction-indexed matrices in lockstep so the
    // shape contract (`anchor_correction: d_total × k_kept`, `r_lw:
    // d_total × k_kept`, `t_lw: p_raw × k_kept`) holds after the audit
    // drops trailing pivots. Forgetting these two left
    // `anchor_correction.ncols() == pre_truncation_k_kept` while
    // `t_lw.ncols() == post_truncation_k_kept`, surfaced downstream as
    // `cross-block identifiability: anchor_correction shape D×P does
    // not match expected d_total=D × k_kept=K`.
    latest.t_lw = latest.t_lw.slice(s![.., ..kept_local]).to_owned();
    if let Some(m) = latest.anchor_correction.as_ref() {
        latest.anchor_correction = Some(m.slice(s![.., ..kept_local]).to_owned());
    }
    if let Some(r) = latest.r_lw.as_ref() {
        latest.r_lw = Some(r.slice(s![.., ..kept_local]).to_owned());
    }
    Ok(dropped_locals)
}

/// Channel-pair decomposition of every parameter block's row Jacobian.
///
/// For families with `K` primary-state channels (survival: K=4), each block
/// `b` contributes a (n × p_b) channel matrix `X_b^(c)` per channel `c` that
/// it touches. Blocks that do not contribute to a channel store `None` in
/// that slot. The closed-form Gram compiler consumes this view directly to
/// build the joint Gram `K^H` without ever materialising the full
/// `(n·K) × p_total` weighted design `W = sqrt(H) · J`.
pub struct PrimaryChannelBlocks {
    /// Outer index: block. Inner index: channel `c ∈ 0..K`. `None` means the
    /// block does not contribute to that channel.
    pub blocks: Vec<Vec<Option<Array2<f64>>>>,
}

/// Closed-form Gram-based compile output: a single `p_raw × p_compiled`
/// reparam matrix `T` mapping compiled coordinates back to raw width.
/// `T · θ` lifts a fitted compiled-width β back to raw width; predict-time
/// row contribution is `X_raw · T · θ` where `X_raw` is the full raw design.
///
/// `compiled_block_ranges[b]` gives the column range inside `T` (and inside
/// the compiled-width coefficient vector) attributable to raw block `b`.
/// `raw_block_ranges[b]` gives the corresponding raw-width column range.
#[derive(Debug)]
pub struct CompiledMap {
    /// `(p_raw × p_compiled)` raw-from-compiled reparam matrix.
    pub raw_from_compiled: Array2<f64>,
    /// Per-block compiled-width column ranges, parallel to
    /// `raw_block_ranges`. Same length as the input `ordering`.
    pub compiled_block_ranges: Vec<std::ops::Range<usize>>,
    /// Per-block raw-width column ranges (copied through from input).
    pub raw_block_ranges: Vec<std::ops::Range<usize>>,
}

/// Neutral view of this compiled reparametrisation for the gauge layer
/// (#1521): `Gauge::from_compiled_map` lives DOWN in `gam-problem` and
/// names only the `CompiledBlockMap` trait, never the concrete
/// `CompiledMap` (which lives ABOVE `gam-problem`). This `impl` supplies
/// the inverted dependency edge.
impl gam_problem::gauge::CompiledBlockMap for CompiledMap {

    fn compiled_block_ranges(&self) -> &[std::ops::Range<usize>] {
        &self.compiled_block_ranges
    }
    fn raw_from_compiled(&self) -> &Array2<f64> {
        &self.raw_from_compiled
    }
    fn raw_block_ranges(&self) -> &[std::ops::Range<usize>] {
        &self.raw_block_ranges
    }
}

/// Closed-form Gram-based identifiability compile.
///
/// Sequential algorithm operating purely on the raw-width Grams
/// `K^H = Σ_i J_iᵀ H_i J_i` (curvature) and `K^S = Σ_i J_iᵀ J_i`
/// (structural). Walks `ordering` left-to-right; for each block `b` with
/// raw-width selector `P_b` (columns of the identity selecting that
/// block) and cumulative compiled map `T = [T_0, …, T_{b-1}]`:
///
/// 1. Structural rank step (drop true gauges):
///    `G^S_AA = Tᵀ K^S T`, `G^S_Ab = Tᵀ K^S P_b`, `G^S_bb = P_bᵀ K^S P_b`,
///    `R_S = (G^S_AA)^+ G^S_Ab`, `G^S_res = G^S_bb − G^S_Abᵀ R_S`.
///    Eigendecompose `G^S_res`; keep positive eigvecs `Q+`. Then
///    `D = (P_b − T R_S) · Q+` (raw-space cols, structurally independent
///    of `T`).
/// 2. Curvature step (within-block conditioning):
///    `G^H_AA = Tᵀ K^H T`, `G^H_AD = Tᵀ K^H D`,
///    `R_H = (G^H_AA)^+ G^H_AD`, `E = D − T R_H` (raw-space).
///    Curvature Gram `G^H_res = Dᵀ K^H D − G^H_ADᵀ R_H`. Eigendecompose
///    and keep positive eigvecs `U`. Then `T_b = E · U`.
/// 3. Append: `T ← [T, T_b]`.
///
/// Returns [`CompilerError::FullyAliased`] only when the first block has no
/// usable structural/curvature span. Later fully absorbed blocks compile to a
/// zero-width block range, which is the reduced-coordinate representation of
/// the lower-priority block owning no degrees of freedom.
pub fn compile_from_raw_grams(
    gram_h: &Array2<f64>,
    gram_struct: &Array2<f64>,
    raw_block_ranges: &[std::ops::Range<usize>],
    ordering: &[BlockOrder],
) -> Result<CompiledMap, CompilerError> {
    compile_from_raw_grams_protected(gram_h, gram_struct, raw_block_ranges, ordering, &[])
}

/// Variant of [`compile_from_raw_grams`] that keeps designated blocks at full
/// raw width instead of dropping their near-null structural/curvature
/// directions.
///
/// `protected[b] == true` forces block `b` to retain **all** of its raw
/// columns: the structural and curvature eigenspace filters that would drop
/// weak directions are replaced by identity, so `T_b` embeds the full raw
/// block (orthogonalised against earlier anchors) rather than a reduced
/// section. The block still serves as a full-width anchor for every later
/// (unprotected) block, so cross-block aliasing against it is removed exactly
/// as before — only the protected block's own within-block reparameterisation
/// is suppressed.
///
/// This exists for blocks whose effective Jacobian is a **fixed nonlinear
/// functional basis** rather than a plain linear design (e.g. the survival
/// marginal-slope monotone time-wiggle block). Such a block's chain-rule
/// Jacobian recomputes its basis at the raw coefficient width on every
/// evaluation and therefore cannot be expressed on a linearly recombined /
/// reduced design; reparameterising it silently corrupts — and can index out
/// of bounds in — that basis evaluation. Keeping it at raw width lets its own
/// penalty nullspace regularise its conditioning, which is the correct
/// treatment for a within-block (as opposed to cross-block) rank deficiency.
///
/// `protected` may be shorter than `ordering` (missing entries default to
/// `false`); an empty slice reproduces [`compile_from_raw_grams`] exactly.
pub fn compile_from_raw_grams_protected(
    gram_h: &Array2<f64>,
    gram_struct: &Array2<f64>,
    raw_block_ranges: &[std::ops::Range<usize>],
    ordering: &[BlockOrder],
    protected: &[bool],
) -> Result<CompiledMap, CompilerError> {
    if raw_block_ranges.len() != ordering.len() {
        return Err(CompilerError::DimensionMismatch(format!(
            "raw_block_ranges ({}) and ordering ({}) length mismatch",
            raw_block_ranges.len(),
            ordering.len()
        )));
    }
    let p_raw = raw_block_ranges.last().map(|r| r.end).unwrap_or(0);
    if gram_h.shape() != [p_raw, p_raw] {
        return Err(CompilerError::DimensionMismatch(format!(
            "gram_h shape {:?} != [p_raw={p_raw}, p_raw={p_raw}]",
            gram_h.shape()
        )));
    }
    if gram_struct.shape() != [p_raw, p_raw] {
        return Err(CompilerError::DimensionMismatch(format!(
            "gram_struct shape {:?} != [p_raw={p_raw}, p_raw={p_raw}]",
            gram_struct.shape()
        )));
    }
    if raw_block_ranges.is_empty() {
        return Ok(CompiledMap {
            raw_from_compiled: Array2::<f64>::zeros((0, 0)),
            compiled_block_ranges: Vec::new(),
            raw_block_ranges: Vec::new(),
        });
    }
    // Validate contiguous ranges from 0.
    let mut expected_start = 0usize;
    for (b, r) in raw_block_ranges.iter().enumerate() {
        if r.start != expected_start {
            return Err(CompilerError::DimensionMismatch(format!(
                "raw_block_ranges must be contiguous from 0; block {b} starts at {} expected {expected_start}",
                r.start
            )));
        }
        expected_start = r.end;
    }

    // Cumulative raw-from-compiled map. Starts empty (zero compiled cols).
    let mut t_cum: Array2<f64> = Array2::<f64>::zeros((p_raw, 0));
    let mut compiled_block_ranges: Vec<std::ops::Range<usize>> =
        Vec::with_capacity(raw_block_ranges.len());

    for (idx, range_b) in raw_block_ranges.iter().enumerate() {
        let p_b = range_b.end - range_b.start;
        let block_protected = protected.get(idx).copied().unwrap_or(false);
        // A zero-width block owns no raw columns. It contributes no compiled
        // degrees of freedom and — having no columns — cannot alias against any
        // anchor, so it is trivially identifiable. Emit an empty compiled range
        // and skip the structural/curvature analysis: a 0×0 residual Gram has no
        // positive eigenspace, which the first-block guard below would otherwise
        // mis-report as `FullyAliased` even though there is literally nothing to
        // alias. This mirrors the empty range a fully-absorbed later block
        // already compiles to (see the `q_plus.ncols() == 0` / `u_mat.ncols() == 0`
        // branches), keeping `kept_width + dropped_count == raw_width` exact.
        if p_b == 0 {
            let at = t_cum.ncols();
            compiled_block_ranges.push(at..at);
            continue;
        }
        // Slice gram columns/rows by raw block range. P_bᵀ K X = rows
        // range_b of K X. K^S T and K^H T are full-rows products.
        // 1) Structural rank step.
        // K^S · T (p_raw × p_compiled)
        let ks_t = fast_ab(gram_struct, &t_cum);
        // G^S_AA = Tᵀ K^S T (p_compiled × p_compiled)
        let g_s_aa = fast_atb(&t_cum, &ks_t);
        // G^S_Ab = Tᵀ K^S P_b = Tᵀ · K^S[:, range_b] (p_compiled × p_b)
        let ks_pb = gram_struct
            .slice(s![.., range_b.start..range_b.end])
            .to_owned();
        let g_s_ab = fast_atb(&t_cum, &ks_pb);
        // G^S_bb = P_bᵀ K^S P_b = K^S[range_b, range_b] (p_b × p_b)
        let g_s_bb = gram_struct
            .slice(s![range_b.start..range_b.end, range_b.start..range_b.end])
            .to_owned();
        // R_S = (G^S_AA)^+ G^S_Ab (p_compiled × p_b)
        let r_s = solve_psd_system(&g_s_aa, &g_s_ab)?;
        // G^S_res = G^S_bb − G^S_Abᵀ R_S (p_b × p_b), symmetrise.
        let g_s_res_raw = &g_s_bb - &fast_atb(&g_s_ab, &r_s);
        let g_s_res = symmetrise(&g_s_res_raw);
        // Trace of the unresidualised diagonal block (scale ref).
        let g_s_bb_trace: f64 = (0..p_b).map(|i| g_s_bb[[i, i]].max(0.0)).sum();
        // p_raw stands in as the "n*K" scale for the closed-form tolerance.
        // A protected block keeps every raw column (identity structural span);
        // the residual-Gram eigenfilter that would drop weak directions is
        // suppressed so the block emerges at full raw width.
        let q_plus = if block_protected {
            Array2::<f64>::eye(p_b)
        } else {
            keep_positive_eigenspace(&g_s_res, p_raw, 1, g_s_bb_trace)?
        };
        if q_plus.ncols() == 0 {
            if t_cum.ncols() == 0 {
                return Err(CompilerError::FullyAliased {
                    block_idx: idx,
                    reason: format!(
                        "structural residual Gram has no positive eigenspace (block of width {p_b} has zero structural span before any anchor exists)"
                    ),
                });
            }
            let at = t_cum.ncols();
            compiled_block_ranges.push(at..at);
            continue;
        }
        // D = (P_b − T R_S) · Q+ (p_raw × k_kept). Build (P_b − T R_S)
        // explicitly as a p_raw × p_b matrix: columns of P_b are columns
        // range_b of I_p_raw, so (P_b − T R_S) places −T R_S in all rows
        // and adds the identity on rows range_b.
        let mut diff = Array2::<f64>::zeros((p_raw, p_b));
        if t_cum.ncols() > 0 {
            // diff = −T · R_S
            let t_rs = fast_ab(&t_cum, &r_s);
            for i in 0..p_raw {
                for j in 0..p_b {
                    diff[[i, j]] = -t_rs[[i, j]];
                }
            }
        }
        for j in 0..p_b {
            diff[[range_b.start + j, j]] += 1.0;
        }
        let d_mat = fast_ab(&diff, &q_plus);

        // 2) Curvature step.
        // K^H · T (p_raw × p_compiled), K^H · D (p_raw × k_kept)
        let kh_t = fast_ab(gram_h, &t_cum);
        let g_h_aa = fast_atb(&t_cum, &kh_t);
        let kh_d = fast_ab(gram_h, &d_mat);
        let g_h_ad = fast_atb(&t_cum, &kh_d);
        let r_h = solve_psd_system(&g_h_aa, &g_h_ad)?;
        // G^H_res = Dᵀ K^H D − G^H_ADᵀ R_H (k_kept × k_kept)
        let d_t_kh_d = fast_atb(&d_mat, &kh_d);
        let g_h_res_raw = &d_t_kh_d - &fast_atb(&g_h_ad, &r_h);
        let g_h_res = symmetrise(&g_h_res_raw);
        let k_kept = q_plus.ncols();
        let g_h_dd_trace: f64 = (0..k_kept).map(|i| d_t_kh_d[[i, i]].max(0.0)).sum();
        // A protected block also retains every structurally-kept curvature
        // direction (identity curvature span), so no within-block conditioning
        // drop occurs; its own penalty nullspace regularises the fit instead.
        let u_mat = if block_protected {
            Array2::<f64>::eye(k_kept)
        } else {
            keep_positive_eigenspace(&g_h_res, p_raw, 1, g_h_dd_trace)?
        };
        if u_mat.ncols() == 0 {
            if t_cum.ncols() == 0 {
                return Err(CompilerError::FullyAliased {
                    block_idx: idx,
                    reason: format!(
                        "curvature residual Gram has no positive eigenspace within structurally-kept basis (block of width {p_b}, structural-kept {k_kept}) before any anchor exists"
                    ),
                });
            }
            let at = t_cum.ncols();
            compiled_block_ranges.push(at..at);
            continue;
        }
        // E = D − T · R_H (p_raw × k_kept); T_b = E · U.
        let mut e_mat = d_mat.clone();
        if t_cum.ncols() > 0 {
            let t_rh = fast_ab(&t_cum, &r_h);
            e_mat = &e_mat - &t_rh;
        }
        let t_b = fast_ab(&e_mat, &u_mat);

        let start = t_cum.ncols();
        let end = start + t_b.ncols();
        compiled_block_ranges.push(start..end);
        t_cum = concat_cols(&t_cum, &t_b);
    }

    // Finite check.
    for v in t_cum.iter() {
        if !v.is_finite() {
            return Err(CompilerError::LinalgFailure(
                "compile_from_raw_grams produced non-finite entry in raw_from_compiled".to_string(),
            ));
        }
    }

    Ok(CompiledMap {
        raw_from_compiled: t_cum,
        compiled_block_ranges,
        raw_block_ranges: raw_block_ranges.to_vec(),
    })
}

impl CompiledMap {
    /// Raw coefficient width (`p_raw`).
    pub fn p_raw(&self) -> usize {
        self.raw_from_compiled.nrows()
    }

}

/// Per-block exact orthogonal reparameterisation of structural confounds.
///
/// `block_transforms[b]` is a dense `(p_b × r_b)` reparam `V_b` mapping raw
/// block-`b` coefficients to reduced coordinates: the orthogonalised block
/// design is `X_b · V_b`, and a fitted reduced coefficient lifts back to raw
/// space exactly via `β_b_raw = V_b · θ_b`. `r_b ≤ p_b`; `r_b < p_b` exactly
/// when block `b` carries `p_b − r_b` directions already spanned (in the
/// pilot W-metric) by the cumulative anchor of all higher-priority blocks —
/// those directions are removed (not penalised), so the joint design
/// `[X_0 V_0 | X_1 V_1 | …]` has the overlap excised exactly.
pub struct BlockOrthogonalization {
    /// `block_transforms[b]`: the `(p_b × r_b)` reparam `V_b` for raw block `b`,
    /// in the **original block order** (parallel to the `block_designs` input).
    pub block_transforms: Vec<Array2<f64>>,
    /// `(block_idx, local_raw_col_count_dropped)` for every block whose
    /// reduced width is strictly smaller than its raw width — i.e. the blocks
    /// that shed overlap directions against the anchor. Empty when no block
    /// overlapped (every `V_b` is then a `p_b × p_b` rotation/identity).
    pub dropped: Vec<(usize, usize)>,
    /// One structural annotation per input block, in original block order.
    ///
    /// This is the explicit "same direction vs independent direction" verdict:
    /// `Independent` means the block kept its full realized-design rank, while
    /// `PartiallyAbsorbed...` / `FullyAbsorbed...` mean the lower-priority block
    /// shared realized-design directions with the cumulative anchor and those
    /// directions were removed rather than assigned a separate penalty.
    pub direction_annotations: Vec<PenalizedDirectionAnnotation>,
}

/// Build per-block exact W-metric orthogonalising reparameterisations.
///
/// `block_designs[b]` is the raw `(n × p_b)` design of block `b`.
/// `priority[b]` is the block's gauge priority — blocks are residualised in
/// **descending** priority order, so the highest-priority block keeps its full
/// column span and lower-priority blocks shed only the directions already
/// explained by the cumulative higher-priority anchor. `weight` is the pilot
/// W-metric row weight `w_i ≥ 0` (the diagonal of the working GLM/GAM Hessian
/// at the pilot β); pass an all-ones vector for the plain Euclidean metric.
///
/// The returned `block_transforms` are in the **original** block order. For a
/// block whose columns are all W-orthogonal to the anchor, `V_b` is a square
/// `p_b × p_b` orthonormal rotation (rank preserved, round-trip exact). For a
/// block with an overlap of dimension `d`, `V_b` is `p_b × (p_b − d)` and the
/// `d` overlap directions are removed exactly.
///
/// Exactness / round-trip: `X_b · V_b` is the reduced design and
/// `β_b_raw = V_b · θ_b` lifts a reduced fit back to raw coordinates. `V_b` has
/// orthonormal columns (eigenvectors of the residual Gram), so the lift is the
/// minimum-norm raw representative of the reduced fit.
pub fn orthogonalize_design_blocks(
    block_designs: &[Array2<f64>],
    priority: &[u32],
    weight: &[f64],
) -> Result<BlockOrthogonalization, CompilerError> {
    if block_designs.len() != priority.len() {
        return Err(CompilerError::DimensionMismatch(format!(
            "block_designs ({}) and priority ({}) length mismatch",
            block_designs.len(),
            priority.len()
        )));
    }
    if block_designs.is_empty() {
        return Ok(BlockOrthogonalization {
            block_transforms: Vec::new(),
            dropped: Vec::new(),
            direction_annotations: Vec::new(),
        });
    }
    let n = block_designs[0].nrows();
    for (b, x) in block_designs.iter().enumerate() {
        if x.nrows() != n {
            return Err(CompilerError::DimensionMismatch(format!(
                "block {b} design has {} rows but block 0 has {n}",
                x.nrows()
            )));
        }
    }
    if weight.len() != n {
        return Err(CompilerError::DimensionMismatch(format!(
            "weight length {} != n {n}",
            weight.len()
        )));
    }
    // sqrt(W) row scale. The pilot Hessian is PSD-clamped upstream; accepting
    // a negative or non-finite value here would silently change the requested
    // metric and can turn an aliased direction into an apparently independent
    // one. Reject the invalid mathematical object at the boundary.
    let mut sqrt_w = Array1::<f64>::zeros(n);
    for i in 0..n {
        let wi = weight[i];
        if !wi.is_finite() || wi < 0.0 {
            return Err(CompilerError::InvalidMetric(format!(
                "weight[{i}] must be finite and non-negative; got {wi}"
            )));
        }
        sqrt_w[i] = wi.sqrt();
    }

    // Descending-priority visitation order over the original block indices.
    // Stable on ties (preserves input order) so the anchor build is
    // deterministic.
    let mut order: Vec<usize> = (0..block_designs.len()).collect();
    order.sort_by(|&a, &b| priority[b].cmp(&priority[a]));

    // Cumulative weighted anchor `A = sqrt(W) · [kept block designs]`.
    let mut anchor: Array2<f64> = Array2::<f64>::zeros((n, 0));

    // Output transforms indexed by ORIGINAL block index (filled out of order).
    let mut block_transforms: Vec<Option<Array2<f64>>> = vec![None; block_designs.len()];
    let mut direction_annotations: Vec<Option<PenalizedDirectionAnnotation>> =
        vec![None; block_designs.len()];
    let mut dropped: Vec<(usize, usize)> = Vec::new();

    for &b in order.iter() {
        let x_b = &block_designs[b];
        let p_b = x_b.ncols();
        // Weighted block design `W_b = sqrt(W) · X_b`.
        let mut w_b = x_b.clone();
        for i in 0..n {
            let s = sqrt_w[i];
            for j in 0..p_b {
                w_b[[i, j]] *= s;
            }
        }
        // Residualise `W_b` against the cumulative anchor in the W-metric and
        // eigendecompose the residual Gram. Eigenvectors with positive
        // eigenvalues span block `b`'s W-orthogonal-to-anchor column space;
        // the zero-eigenvalue directions are exactly the overlap with the
        // anchor and are removed.
        let (residual, _correction) = residualise_in_metric(&anchor, &w_b)?;
        let g_res = symmetrise(&fast_atb(&residual, &residual));
        // Scale reference for `keep_positive_eigenspace` must be the
        // *original* (pre-residualisation) weighted block Gram trace, NOT the
        // residual's. When `b` is fully absorbed by a higher-priority anchor
        // the residual collapses to floating-point noise (~ε² of the original
        // O(1) data); anchoring tau to that noise floor would keep the noise
        // eigenvalues and misreport a fully-absorbed block as `Independent`.
        // The original-block trace is invariant to absorption, so a near-zero
        // residual is correctly rejected as fully absorbed.
        let g_bb = fast_atb(&w_b, &w_b);
        let g_bb_trace: f64 = (0..p_b).map(|i| g_bb[[i, i]].max(0.0)).sum();
        let v_b = keep_positive_eigenspace(&g_res, n, 1, g_bb_trace)?;
        let r_b = v_b.ncols();
        let absorbed_width = p_b - r_b;
        let kind = if absorbed_width == 0 {
            PenalizedDirectionAnnotationKind::Independent
        } else if r_b == 0 {
            PenalizedDirectionAnnotationKind::FullyAbsorbedByHigherPriority
        } else {
            PenalizedDirectionAnnotationKind::PartiallyAbsorbedByHigherPriority
        };
        direction_annotations[b] = Some(PenalizedDirectionAnnotation {
            block_idx: b,
            raw_width: p_b,
            kept_width: r_b,
            absorbed_width,
            kind,
        });
        if absorbed_width > 0 {
            dropped.push((b, absorbed_width));
        }
        // Append this block's kept, W-orthogonalised weighted columns to the
        // anchor so lower-priority blocks residualise against them too. The
        // residual (already anchor-orthogonal) projected onto the kept basis
        // is `residual · V_b` — these are mutually orthogonal in the W-metric
        // by construction of `keep_positive_eigenspace`.
        let kept_weighted = fast_ab(&residual, &v_b);
        anchor = concat_cols(&anchor, &kept_weighted);
        block_transforms[b] = Some(v_b);
    }

    let block_transforms: Vec<Array2<f64>> = block_transforms
        .into_iter()
        .enumerate()
        .map(|(b, t)| {
            t.ok_or_else(|| {
                CompilerError::LinalgFailure(format!(
                    "orthogonalize_design_blocks: block {b} transform was never assigned"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let direction_annotations: Vec<PenalizedDirectionAnnotation> = direction_annotations
        .into_iter()
        .enumerate()
        .map(|(b, annotation)| {
            annotation.ok_or_else(|| {
                CompilerError::LinalgFailure(format!(
                    "orthogonalize_design_blocks: block {b} direction annotation was never assigned"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Finite check on every transform.
    for (b, v) in block_transforms.iter().enumerate() {
        for value in v.iter() {
            if !value.is_finite() {
                return Err(CompilerError::LinalgFailure(format!(
                    "orthogonalize_design_blocks: block {b} transform has a non-finite entry"
                )));
            }
        }
    }

    Ok(BlockOrthogonalization {
        block_transforms,
        dropped,
        direction_annotations,
    })
}

/// Symmetrise a (nearly-symmetric) matrix by averaging with its transpose.
fn symmetrise(m: &Array2<f64>) -> Array2<f64> {
    let (r, c) = m.dim();
    assert_eq!(r, c, "symmetrise expects square matrix");
    let mut out = Array2::<f64>::zeros((r, c));
    for i in 0..r {
        for j in 0..c {
            out[[i, j]] = 0.5 * (m[[i, j]] + m[[j, i]]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Convenience: wrap a dense `(n × p)` block design as a `K=1`
    /// row-Jacobian operator. Used by tests; production families ship their
    /// own concrete operators.
    struct DenseScalarOperator {
        design: Array2<f64>,
    }

    impl DenseScalarOperator {
        fn new(design: Array2<f64>) -> Self {
            Self { design }
        }
    }

    impl RowJacobianOperator for DenseScalarOperator {
        fn k(&self) -> usize {
            1
        }
        fn ncols(&self) -> usize {
            self.design.ncols()
        }
        fn nrows(&self) -> usize {
            self.design.nrows()
        }
        fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
            assert_eq!(out.len(), 1);
            let mut acc = 0.0;
            for (j, &b) in delta_beta.iter().enumerate() {
                acc += self.design[[row, j]] * b;
            }
            out[0] = acc;
        }
        fn evaluate_full(&self) -> Array3<f64> {
            let n = self.design.nrows();
            let p = self.design.ncols();
            let mut out = Array3::<f64>::zeros((n, p, 1));
            for i in 0..n {
                for j in 0..p {
                    out[[i, j, 0]] = self.design[[i, j]];
                }
            }
            out
        }
    }

    // `IdentityRowHessian` is re-exported from the parent module's `use
    // super::*;` above (now a public struct so the dual-metric API can
    // share the default structural metric with callers).

    /// Diagonal row Hessian with per-row scalar weights (K=1 case).
    struct DiagonalScalarRowHessian {
        w: Array1<f64>,
    }

    impl DiagonalScalarRowHessian {
        fn new(w: Array1<f64>) -> Self {
            Self { w }
        }
    }

    impl RowHessian for DiagonalScalarRowHessian {
        fn k(&self) -> usize {
            1
        }
        fn nrows(&self) -> usize {
            self.w.len()
        }
        fn fill_row(&self, row: usize, out: &mut [f64]) {
            assert_eq!(out.len(), 1);
            out[0] = self.w[row];
        }
        fn evaluate_full(&self) -> Array3<f64> {
            let n = self.w.len();
            let mut out = Array3::<f64>::zeros((n, 1, 1));
            for i in 0..n {
                out[[i, 0, 0]] = self.w[i];
            }
            out
        }
    }

    fn op(design: Array2<f64>) -> Arc<dyn RowJacobianOperator> {
        Arc::new(DenseScalarOperator::new(design))
    }

    /// §10 test #1: two affine blocks, identity row Hessian. The compiled
    /// second-block design must be orthogonal to the first block under the
    /// (identity) row metric to machine epsilon.
    #[test]
    fn compile_two_block_orthogonalises_under_metric() {
        let n = 50;
        let a = Array2::from_shape_fn((n, 3), |(i, j)| ((i + 1) as f64).sin().powi((j + 1) as i32));
        // B partly aliases A's first column.
        let b = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.5 * a[[i, 0]] + ((i as f64) * 0.13 + j as f64).cos()
        });
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a.clone()), op(b.clone())];
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Slope])
            .expect("compile should succeed");
        // Build A's design (no rotation) and B's compiled design B·V − A·M.
        let v_b = &compiled.blocks[1].t_lw;
        let m_b = compiled.blocks[1]
            .anchor_correction
            .as_ref()
            .expect("second block must carry an anchor correction");
        let b_v = b.dot(v_b);
        let a_m = a.dot(m_b);
        let b_compiled = &b_v - &a_m;
        // <A, B_compiled>_I = Aᵀ · B_compiled should be ≈ 0.
        let cross = a.t().dot(&b_compiled);
        let max_err = cross.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            max_err < 1e-10,
            "orthogonality residual too large: {max_err:e}"
        );
    }

    /// §10 test #2: three-block chain with sequential aliases.
    #[test]
    fn compile_three_block_chain() {
        let n = 80;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 * 0.1 + j as f64).sin());
        let b = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.3 * a[[i, 0]] + (j as f64) * (i as f64).cos()
        });
        let c = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.2 * a[[i, 1]] + 0.4 * b[[i, 0]] + ((i + j) as f64).tan().min(5.0).max(-5.0)
        });
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a), op(b), op(c)];
        let compiled = compile(
            &ops,
            &hess,
            &[
                BlockOrder::Marginal,
                BlockOrder::Slope,
                BlockOrder::LinkDev,
            ],
        )
        .expect("compile should succeed");
        let total: usize = compiled.blocks.iter().map(|b| b.t_lw.ncols()).sum();
        assert_eq!(
            compiled.joint_rank, total,
            "audit must report full rank on synthetic full-rank design"
        );
    }

    /// `compile_protected` keeps a rank-deficient protected first block at full
    /// raw width (identity V) while the unprotected path drops its null
    /// direction, and later blocks still orthogonalise against the full anchor.
    /// Mirrors the `compile_from_raw_grams_protected` guard for the operator
    /// (per-term) reduction path used by the survival time-wiggle time block.
    #[test]
    fn compile_protected_keeps_rank_deficient_first_block_full_width() {
        let n = 40;
        // Block A: two identical columns → structural rank 1 (one within-block
        // null the unprotected filter drops).
        let a = Array2::from_shape_fn((n, 2), |(i, _)| ((i + 1) as f64 * 0.31).sin());
        let b = Array2::from_shape_fn((n, 2), |(i, j)| ((i as f64) * 0.17 + j as f64).cos());
        let hess = IdentityRowHessian::new(n, 1);
        let ordering = [BlockOrder::Time, BlockOrder::Marginal];

        let unprotected = compile(&[op(a.clone()), op(b.clone())], &hess, &ordering)
            .expect("unprotected compile");
        assert_eq!(
            unprotected.blocks[0].t_lw.ncols(),
            1,
            "unprotected first block drops its duplicate column"
        );

        let protected = compile_protected(
            &[op(a.clone()), op(b.clone())],
            &hess,
            &ordering,
            &[true, false],
        )
        .expect("protected compile");
        let v_a = &protected.blocks[0].t_lw;
        assert_eq!(
            v_a.ncols(),
            2,
            "protected first block retains its full raw width"
        );
        // V_a is the 2×2 identity: raw coords == compiled coords for the
        // protected first block.
        for i in 0..2 {
            for j in 0..2 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v_a[[i, j]] - expect).abs() <= 1e-12,
                    "protected first block V must be identity, got [{i},{j}]={}",
                    v_a[[i, j]]
                );
            }
        }
    }

    /// §10 test #3: non-identity row Hessian. With K=1 and weights `w`,
    /// the projection of a 1-col block `b` onto a 1-col block `a` is
    /// `Σ w·a·b / Σ w·a²`. Verify the Gram solve recovers this scalar.
    #[test]
    fn compile_weighted_metric_nontrivial() {
        let n = 32;
        let a: Array2<f64> = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 + 1.0).sqrt());
        let b: Array2<f64> =
            Array2::from_shape_fn((n, 1), |(i, _)| 0.7 * a[[i, 0]] + (i as f64 * 0.05).cos());
        let w = Array1::from_shape_fn(n, |i| 0.5 + (i as f64 * 0.2).sin().abs());
        let hess = DiagonalScalarRowHessian::new(w.clone());
        let ops = vec![op(a.clone()), op(b.clone())];
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Slope])
            .expect("compile should succeed");
        let m = compiled.blocks[1]
            .anchor_correction
            .as_ref()
            .expect("anchor correction present");
        let analytic_num: f64 = (0..n).map(|i| w[i] * a[[i, 0]] * b[[i, 0]]).sum();
        let analytic_den: f64 = (0..n).map(|i| w[i] * a[[i, 0]] * a[[i, 0]]).sum();
        let analytic = analytic_num / analytic_den;
        assert!(m.dim() == (1, 1));
        assert!(
            (m[[0, 0]] - analytic).abs() < 1e-10,
            "weighted projection mismatch: got {got}, analytic {analytic}",
            got = m[[0, 0]]
        );
    }

    /// Regression for #372: an anchor block that internally sheds an aliased
    /// column makes the residualised kept-anchor width (`anchor_h.ncols()`)
    /// strictly smaller than the raw anchor width (`d_total`). The emitted
    /// `anchor_correction` must be expressed in *raw* anchor-column
    /// coordinates so the predict-time / install-time subtraction
    /// `A_raw(x)·M` is dimensionally and metrically correct. Previously the
    /// correction was indexed by kept directions, producing a (d_total−1)×k
    /// matrix and the failure
    /// `anchor_correction shape 36x6 does not match d_total=37`.
    #[test]
    fn compile_emits_anchor_correction_in_raw_column_coordinates() {
        let n = 64;
        // Anchor block A has 3 raw columns but only rank 2: col 2 is an exact
        // linear combination of cols 0 and 1, so the compiler keeps just two
        // anchor directions (kept width 2 < raw width 3).
        let a: Array2<f64> = Array2::from_shape_fn((n, 3), |(i, j)| {
            let c0 = (i as f64 * 0.07 + 1.0).ln();
            let c1 = (i as f64 * 0.13).sin();
            match j {
                0 => c0,
                1 => c1,
                _ => 2.0 * c0 - 0.5 * c1,
            }
        });
        // Candidate block C: partly aliases A's span plus genuine signal.
        let c: Array2<f64> = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.4 * a[[i, 0]] + (j as f64) * (i as f64 * 0.05).cos() + (i as f64 * 0.011).tanh()
        });
        let w = Array1::from_shape_fn(n, |i| 0.3 + (i as f64 * 0.17).sin().abs());
        let hess = DiagonalScalarRowHessian::new(w.clone());
        let ops = vec![op(a.clone()), op(c.clone())];
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::LinkDev])
            .expect("compile should succeed");

        let v = &compiled.blocks[1].t_lw;
        let m = compiled.blocks[1]
            .anchor_correction
            .as_ref()
            .expect("candidate block must carry an anchor correction");
        let k_kept = v.ncols();
        assert!(k_kept >= 1, "candidate must keep at least one direction");

        // The off-by-one the issue tripped on: M must have one row per *raw*
        // anchor column (3), not per kept anchor direction (2).
        assert_eq!(
            m.nrows(),
            a.ncols(),
            "anchor_correction must be indexed by raw anchor columns (d_total), \
             got {} rows for {} raw anchor columns",
            m.nrows(),
            a.ncols(),
        );
        assert_eq!(m.ncols(), k_kept, "anchor_correction width must match V");

        // Metric correctness: the raw-coordinate subtraction A_raw·M must make
        // the compiled candidate design W-orthogonal to the full raw anchor
        // span. C̃ = C·V − A·M; require Aᵀ W C̃ ≈ 0 column-wise.
        let c_v = c.dot(v);
        let a_m = a.dot(m);
        let c_tilde = &c_v - &a_m;
        let mut max_cross = 0.0_f64;
        for ac in 0..a.ncols() {
            for cc in 0..c_tilde.ncols() {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += w[i] * a[[i, ac]] * c_tilde[[i, cc]];
                }
                max_cross = max_cross.max(acc.abs());
            }
        }
        assert!(
            max_cross < 1e-9,
            "raw-coordinate anchor correction must W-orthogonalise the candidate \
             against the raw anchor span; max |Aᵀ W C̃| = {max_cross:e}"
        );
    }

    /// §10 test #4: deliberately rank-deficient joint design. The trailing
    /// pivot drop must come from the *latest* block in the ordering.
    #[test]
    fn compile_drops_trailing_pivots_from_latest_block() {
        let n = 40;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 + 1.0).ln() * (j as f64 + 1.0));
        // c is exactly a's first column → after residualising c against a,
        // the residual span is zero in that direction, but a non-zero
        // independent column also exists. Add an extra exact-alias column
        // to force trailing-pivot drop at the audit stage.
        let c = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                a[[i, 0]]
            } else {
                (i as f64 * 0.1).cos()
            }
        });
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a), op(c)];
        // Manually inject a known alias: pass a second block whose
        // residualised columns will themselves be linearly dependent on
        // the first block after metric projection — already covered by the
        // eigenvalue threshold inside `compile`. Verify either drop path
        // (eigen-threshold or audit) attributes loss to block index 1.
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Slope])
            .expect("compile should succeed");
        // Either the eigen-threshold dropped a column from block 1, or
        // the audit did. In both cases block 1's V must have fewer than
        // its 2 input columns.
        let v1_cols = compiled.blocks[1].t_lw.ncols();
        assert!(
            v1_cols < 2 || !compiled.dropped.is_empty(),
            "expected rank loss attributed to block 1, got v1_cols={v1_cols}, dropped={dropped:?}",
            dropped = compiled.dropped
        );
        for (block_idx, _) in &compiled.dropped {
            assert_eq!(
                *block_idx, 1,
                "audit drops must come from the latest block only"
            );
        }
    }

    /// Regression: when `audit_and_drop_trailing_pivots` truncates the
    /// latest block's `t_lw`, the sibling `anchor_correction` and `r_lw`
    /// matrices must be truncated to the same `k_kept` so the trailing-
    /// block install path sees a coherent
    /// `t_lw.ncols() == anchor_correction.ncols() == r_lw.ncols()` shape.
    ///
    /// Pre-fix bug: only `t_lw` got truncated. Downstream callers
    /// asserting `anchor_correction.ncols() == k_kept` then failed with
    /// `cross-block identifiability: anchor_correction shape D×P does
    /// not match expected d_total=D × k_kept=K` — surfaced via the
    /// large-scale V+M repro test.
    #[test]
    fn audit_truncation_keeps_t_lw_and_anchor_correction_in_lockstep() {
        let n = 40;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 + 1.0).ln() * (j as f64 + 1.0));
        let c = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                a[[i, 0]]
            } else {
                (i as f64 * 0.1).cos()
            }
        });
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a), op(c)];
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Slope])
            .expect("compile should succeed");
        for (idx, block) in compiled.blocks.iter().enumerate() {
            let k_kept = block.t_lw.ncols();
            if let Some(m) = block.anchor_correction.as_ref() {
                assert_eq!(
                    m.ncols(),
                    k_kept,
                    "block {idx}: anchor_correction.ncols()={ac} must equal t_lw.ncols()={k_kept} \
                     after audit truncation",
                    ac = m.ncols(),
                );
            }
            if let Some(r) = block.r_lw.as_ref() {
                assert_eq!(
                    r.ncols(),
                    k_kept,
                    "block {idx}: r_lw.ncols()={r_cols} must equal t_lw.ncols()={k_kept} \
                     after audit truncation",
                    r_cols = r.ncols(),
                );
            }
        }
    }

    /// §10 test #5: regression test for the deleted FlexEvaluation skip
    /// bug. A flex anchor (represented by a dense scalar operator with the
    /// same column span as the parametric reference) must receive the same
    /// residualisation as the parametric anchor.
    #[test]
    fn compile_flex_anchor_is_first_class() {
        let n = 60;
        // Two parametric blocks A, B; a third "flex" block C whose
        // operator is dense (modelling a compiled flex anchor's column
        // span). All-parametric reference vs. mixed parametric+flex must
        // produce identical compiled blocks B (residualised against A)
        // because the compiler treats every input as a `RowJacobianOperator`.
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 * 0.07 + j as f64).sin());
        let b = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.4 * a[[i, 0]] + (j as f64) * (i as f64 + 1.0).ln()
        });
        let hess = IdentityRowHessian::new(n, 1);

        let ops_param = vec![op(a.clone()), op(b.clone())];
        let compiled_param = compile(
            &ops_param,
            &hess,
            &[BlockOrder::Marginal, BlockOrder::Slope],
        )
        .expect("compile should succeed");

        // Now wrap A's design behind a mock anchor evaluator and feed it
        // to the compiler as a `DenseScalarOperator` with the same span.
        // The B-block result must match the parametric reference.
        let ops_flex = vec![op(a.clone()), op(b.clone())];
        let compiled_flex = compile(
            &ops_flex,
            &hess,
            &[BlockOrder::ScoreWarp, BlockOrder::LinkDev],
        )
        .expect("compile should succeed");

        let m_param = compiled_param.blocks[1].anchor_correction.as_ref().unwrap();
        let m_flex = compiled_flex.blocks[1].anchor_correction.as_ref().unwrap();
        assert_eq!(m_param.dim(), m_flex.dim());
        let max_diff = (m_param - m_flex)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            max_diff < 1e-12,
            "flex vs parametric anchor correction mismatch: {max_diff:e}"
        );
    }

    /// §10 test #7: Bernoulli row Hessian = IRLS weight. Verified at the
    /// trait level — a `DiagonalScalarRowHessian` round-trips through
    /// `evaluate_full` to the same per-row scalar.
    #[test]
    fn bernoulli_row_hessian_matches_irls_weight() {
        let w = Array1::from(vec![0.1, 0.5, 0.9, 0.25, 0.75]);
        let hess = DiagonalScalarRowHessian::new(w.clone());
        let full = hess.evaluate_full();
        assert_eq!(full.shape(), &[5, 1, 1]);
        for i in 0..5 {
            assert_eq!(full[[i, 0, 0]], w[i]);
            let mut buf = [0.0_f64; 1];
            hess.fill_row(i, &mut buf);
            assert_eq!(buf[0], w[i]);
        }
    }

    /// §10 test #8: predict-path roundtrip. With the parametric setting,
    /// the row-application of `(C(x)·V − A(x)·M)` at training rows must
    /// equal the in-metric residual computed during `compile`.
    #[test]
    fn compiler_predict_path_roundtrip() {
        let n = 24;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 * 0.21).cos() + j as f64);
        let b = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.3 * a[[i, 0]] + (i as f64 + j as f64).sqrt()
        });
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a.clone()), op(b.clone())];
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Slope])
            .expect("compile should succeed");
        let v_b = &compiled.blocks[1].t_lw;
        let m_b = compiled.blocks[1].anchor_correction.as_ref().unwrap();
        // Training-time residual: B · V − A · M.
        let predict_design = b.dot(v_b) - a.dot(m_b);
        // Compare to the algebraic in-metric residual: same expression
        // (identity row Hessian collapses sqrt(H) = I), so this is a
        // self-consistency / shape check ensuring V and M compose to the
        // promised predict-time operator.
        assert_eq!(predict_design.nrows(), n);
        assert_eq!(predict_design.ncols(), v_b.ncols());
        // Finite-value gate.
        for &val in predict_design.iter() {
            assert!(val.is_finite(), "predict design produced non-finite entry");
        }
    }

    /// `r_lw` and `anchor_correction` are populated on every non-first
    /// block as `M_b · V_b` at compiled width. The first block carries
    /// `None`. Also verifies the H-orthogonality invariant that the
    /// cumulative anchor for the next iteration is orthogonal (in the row
    /// metric) to the prior block's design.
    #[test]
    fn compile_exposes_r_lw_equal_to_m_dot_v() {
        let n = 40;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 * 0.17 + j as f64).sin());
        // B partially aliases A's first column, so anchor correction is non-trivial.
        let b = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.6 * a[[i, 0]] + ((i as f64) * 0.11 + j as f64).cos()
        });
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a.clone()), op(b.clone())];
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Slope])
            .expect("compile should succeed");

        // First block: no anchor → both fields None.
        assert!(compiled.blocks[0].r_lw.is_none());
        assert!(compiled.blocks[0].anchor_correction.is_none());

        // Second block: r_lw and anchor_correction must both equal M·V at
        // compiled width (p_a_kept × p_b_kept).
        let v_a = &compiled.blocks[0].t_lw;
        let v_b = &compiled.blocks[1].t_lw;
        let m_compiled = compiled.blocks[1]
            .anchor_correction
            .as_ref()
            .expect("second block must carry an anchor correction");
        let r_lw = compiled.blocks[1]
            .r_lw
            .as_ref()
            .expect("second block must expose r_lw");
        let p_a_kept = v_a.ncols();
        let p_b_kept = v_b.ncols();
        assert_eq!(
            m_compiled.dim(),
            (p_a_kept, p_b_kept),
            "anchor_correction must be at compiled width"
        );
        assert_eq!(r_lw.dim(), (p_a_kept, p_b_kept));
        // r_lw and anchor_correction are synonymous.
        let diff = r_lw - m_compiled;
        let max_diff = diff.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        assert!(
            max_diff == 0.0,
            "r_lw and anchor_correction must be identical"
        );

        // H-orthogonality (identity row metric): the residualised
        // compiled B-design `B·V − A·(M·V)` must be orthogonal to A in
        // the column-inner-product sense. This validates that the
        // cumulative anchor build uses `(W_b − A·M)·V` rather than `W_b·V`.
        let b_compiled = b.dot(v_b) - a.dot(m_compiled);
        let cross = a.t().dot(&b_compiled);
        let max_cross = cross.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        assert!(
            max_cross < 1e-10,
            "compiled B-design must be H-orthogonal to A: max cross = {max_cross:e}"
        );
    }

    // Per-row Hessian (K=1) sourced from an arbitrary positive vector —
    // used by the dual-metric sanity test to drive both structural and
    // curvature passes with the *same* non-identity weights.

    // ---- compile_from_raw_grams tests ----

    #[test]
    fn orthogonalization_annotates_independent_and_fully_absorbed_blocks() {
        let n = 18;
        let anchor = Array2::from_shape_fn((n, 2), |(i, j)| {
            ((i + 1) as f64 * (0.19 + j as f64 * 0.07)).sin()
        });
        let duplicate = anchor.clone();
        let independent = Array2::from_shape_fn((n, 1), |(i, _)| ((i + 1) as f64 * 0.43).cos());
        let weight = vec![1.0; n];
        let ortho = orthogonalize_design_blocks(
            &[anchor, duplicate, independent],
            &[200, 100, 50],
            &weight,
        )
        .expect("structural annotation compile");

        assert_eq!(
            ortho.direction_annotations[0].kind,
            PenalizedDirectionAnnotationKind::Independent
        );
        assert_eq!(ortho.direction_annotations[0].absorbed_width, 0);
        assert_eq!(
            ortho.direction_annotations[1].kind,
            PenalizedDirectionAnnotationKind::FullyAbsorbedByHigherPriority,
            "a duplicated lower-priority block is the same realized-design direction"
        );
        assert_eq!(ortho.direction_annotations[1].raw_width, 2);
        assert_eq!(ortho.direction_annotations[1].kept_width, 0);
        assert_eq!(ortho.direction_annotations[1].absorbed_width, 2);
        assert_eq!(
            ortho.direction_annotations[2].kind,
            PenalizedDirectionAnnotationKind::Independent,
            "a genuinely new realized-design direction keeps its own penalty block"
        );
        assert_eq!(ortho.direction_annotations[2].raw_width, 1);
        assert_eq!(ortho.direction_annotations[2].kept_width, 1);
        assert_eq!(ortho.dropped, vec![(1, 2)]);
    }

    #[test]
    fn orthogonalization_rejects_invalid_row_metric_weights() {
        let design = Array2::from_shape_fn((4, 1), |(row, _)| row as f64 + 1.0);
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            let result = orthogonalize_design_blocks(
                std::slice::from_ref(&design),
                &[1],
                &[1.0, invalid, 1.0, 1.0],
            );
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("an invalid W-metric must be rejected, not silently clamped"),
            };
            assert!(
                matches!(error, CompilerError::InvalidMetric(_)),
                "unexpected error for weight {invalid}: {error}"
            );
        }
    }

}

/// Build `W_b = stack_i sqrt(H_i) · J_b,i` flattened to `(n*K, ncols)` from a
/// materialised `(n, p, K)` tensor. Thin wrapper over
/// [`scale_jacobian_by_sqrt_h_with`] that reads the tensor element-wise.
fn scale_block_by_sqrt_h(jb: &Array3<f64>, h_full: &Array3<f64>) -> Array2<f64> {
    let n = jb.shape()[0];
    let p = jb.shape()[1];
    let k = jb.shape()[2];
    scale_jacobian_by_sqrt_h_with(n, p, k, h_full, |i, a, c| jb[[i, a, c]])
}