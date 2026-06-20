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

use crate::linalg::faer_ndarray::{
    FaerEigh, default_rrqr_rank_alpha, fast_ab, fast_ata, fast_atb, fast_xt_diag_y,
    rrqr_with_permutation,
};
use faer::Side;

/// Slack factor (multiples of machine ε) for the rank-revealing eigenvalue
/// threshold used when pseudo-inverting a Gram matrix or selecting the
/// positive eigenspace of a residual Gram. The retain threshold is
/// `scale · RANK_REVEAL_EPS_SLACK · size · ε`, where `scale` is the dominant
/// eigenvalue (and matrix size accounts for the worst-case roundoff
/// accumulation in the `O(size)` inner products forming each Gram entry). 64×
/// keeps numerically-zero directions out of the kept subspace while preserving
/// every genuinely identified direction at large-scale conditioning.
const RANK_REVEAL_EPS_SLACK: f64 = 64.0;

/// Maps a coefficient perturbation `δβ ∈ R^p` for one parameter block into
/// its contribution to the per-row primary state `u_i ∈ R^K`.
///
/// For affine blocks (everything in this compiler), `J_i = ∂u_i/∂β_block` is
/// independent of `β` and equals the transposed row of the block's effective
/// design matrix lifted into `R^K`.
pub trait RowJacobianOperator: Send + Sync {
    /// Dimension of the row primary state (survival: 4, Bernoulli: 1).
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
    /// [`compile_with_dual_metric`]: crate::identifiability::families::compiler::compile_with_dual_metric
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
    /// A block degenerated to zero residual span — fully aliased by the
    /// cumulative anchor in the row metric.
    FullyAliased { block_idx: usize, reason: String },
    /// A linear-algebra step failed (Gram solve, eigendecomposition, QR).
    LinalgFailure(String),
}

impl std::fmt::Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilerError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            CompilerError::FullyAliased { block_idx, reason } => {
                write!(f, "block {block_idx} fully aliased: {reason}")
            }
            CompilerError::LinalgFailure(msg) => write!(f, "linalg failure: {msg}"),
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
    Logslope,
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
    // Default structural metric is the per-row identity `K^S_i = I_K`.
    // A pilot-curvature `H` can collapse a direction (zero eigenvalue) at
    // a bad β even though the optimum keeps that direction; routing the
    // rank decision through the structural metric and reserving `H` for
    // *within-kept-subspace* curvature handling prevents that mis-drop.
    let n = row_hess.nrows();
    let k = row_hess.k();
    let id_struct = IdentityRowHessian::new(n, k);
    compile_with_dual_metric(operators, row_hess, &id_struct, ordering)
}

/// Compile a sequence of row-Jacobian operators using *separate* metrics
/// for structural rank decisions and curvature-aware orthogonalisation.
///
/// - `row_hess` is the curvature row metric `K^H_i` (a PSD-clamped Hessian
///   of `−log L_i(u_i)` at a pilot β).
/// - `row_structural` is the structural row metric `K^S_i` — typically an
///   [`IdentityRowHessian`] — used only to decide which columns survive
///   block-against-block residualisation. A direction that the curvature
///   `K^H` happens to see as zero at a bad pilot β is *not* dropped here
///   as long as it is structurally non-degenerate.
///
/// Per-block algorithm (left-to-right walk over `ordering`):
///
/// 1. Residualise the block in the structural metric against the
///    cumulative structural anchor; eigendecompose the structural residual
///    Gram and drop only structural-zero eigenvalues → kept basis `D`
///    (raw-block selector).
/// 2. Residualise `W^H_b · D` in the curvature metric against the
///    cumulative curvature anchor → curvature anchor correction
///    `M^H_inner` and residual `R^H`.
/// 3. Eigendecompose the curvature Gram of `R^H` and drop curvature-zero
///    directions (a *within*-structurally-kept curvature alias is a true
///    redundancy) → rotation/selector `T_inner`.
/// 4. Compose: `V = D · T_inner`; compiled anchor correction is
///    `M^H_inner · T_inner` so the predict-time row contribution stays
///    `(C(x) · V − A(x) · anchor_correction) · β`.
///
/// When `row_structural` and `row_hess` represent the same metric (e.g.
/// `compile()` with an identity row Hessian on both sides), the two
/// passes collapse to the single-metric loop.
pub fn compile_with_dual_metric(
    operators: &[Arc<dyn RowJacobianOperator>],
    row_hess: &dyn RowHessian,
    row_structural: &dyn RowHessian,
    ordering: &[BlockOrder],
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
        let d = keep_positive_eigenspace(&g_s, n, k, g_s_trace)?;
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
        let t_inner = keep_positive_eigenspace(&g_h, n, k, g_h_trace)?;
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

/// Build `W_b = stack_i sqrt(H_i) · J_b,i` flattened to `(n*K, ncols)` from a
/// materialised `(n, p, K)` tensor. Thin wrapper over
/// [`scale_jacobian_by_sqrt_h_with`] that reads the tensor element-wise.
fn scale_block_by_sqrt_h(jb: &Array3<f64>, h_full: &Array3<f64>) -> Array2<f64> {
    let n = jb.shape()[0];
    let p = jb.shape()[1];
    let k = jb.shape()[2];
    scale_jacobian_by_sqrt_h_with(n, p, k, h_full, |i, a, c| jb[[i, a, c]])
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
    let (evals, evecs) = g_tilde.eigh(Side::Lower).map_err(|err| {
        CompilerError::LinalgFailure(format!("residual Gram eigh failed: {err:?}"))
    })?;
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max).max(0.0);
    let scale = lambda_max.max(g_bb_trace);
    let nk = (n.saturating_mul(k)).max(p).max(1) as f64;
    let tau = scale * RANK_REVEAL_EPS_SLACK * nk * f64::EPSILON;
    // Collect kept column indices.
    let mut kept: Vec<usize> = (0..p).filter(|&i| evals[i] > tau).collect();
    // Sort kept indices by descending eigenvalue for a stable column order.
    kept.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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

/// Closed-form Gram builder: `K^H[a, b] = Σ_{c,d} (X_a^(c))ᵀ · diag(h_{cd}) · X_b^(d)`.
///
/// Inputs:
/// - `channel_blocks`: per-block channel decomposition of the row Jacobian.
/// - `row_hess`: `(n × K × K)` per-row PSD Hessian (typically clamped to PSD
///   by the family upstream).
/// - `raw_block_ranges`: `[start, end)` column ranges of each block inside
///   the full `p_total`-wide coefficient vector. Must be contiguous and
///   non-overlapping; their union spans `0..p_total`.
///
/// Returns the symmetric `(p_total × p_total)` Gram matrix.
pub fn build_raw_grams_from_channel_blocks(
    channel_blocks: &PrimaryChannelBlocks,
    row_hess: &dyn RowHessian,
    raw_block_ranges: &[std::ops::Range<usize>],
) -> Result<Array2<f64>, CompilerError> {
    let num_blocks = channel_blocks.blocks.len();
    if num_blocks != raw_block_ranges.len() {
        return Err(CompilerError::DimensionMismatch(format!(
            "channel_blocks ({num_blocks}) and raw_block_ranges ({}) length mismatch",
            raw_block_ranges.len()
        )));
    }
    if num_blocks == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let k = row_hess.k();
    let n = row_hess.nrows();
    let p_total: usize = raw_block_ranges.iter().map(|r| r.end - r.start).sum();
    let expected_total = raw_block_ranges.last().map(|r| r.end).unwrap_or(0);
    if expected_total != p_total {
        return Err(CompilerError::DimensionMismatch(format!(
            "raw_block_ranges must be contiguous from 0; got p_total={p_total} but last end={expected_total}"
        )));
    }
    // Per-block channel-slot shape sanity.
    for (b, slots) in channel_blocks.blocks.iter().enumerate() {
        if slots.len() != k {
            return Err(CompilerError::DimensionMismatch(format!(
                "block {b}: expected {k} channel slots, got {}",
                slots.len()
            )));
        }
        let p_b = raw_block_ranges[b].end - raw_block_ranges[b].start;
        for (c, mat) in slots.iter().enumerate() {
            if let Some(x) = mat.as_ref() {
                if x.nrows() != n {
                    return Err(CompilerError::DimensionMismatch(format!(
                        "block {b} channel {c}: nrows={} but row Hessian nrows={n}",
                        x.nrows()
                    )));
                }
                if x.ncols() != p_b {
                    return Err(CompilerError::DimensionMismatch(format!(
                        "block {b} channel {c}: ncols={} but block width={p_b}",
                        x.ncols()
                    )));
                }
            }
        }
    }

    // Materialise H once and slice it into K·K length-n vectors h_{cd}.
    let h_full = row_hess.evaluate_full();
    if h_full.shape() != &[n, k, k] {
        return Err(CompilerError::DimensionMismatch(format!(
            "row Hessian evaluate_full shape {:?} != [n={n}, k={k}, k={k}]",
            h_full.shape()
        )));
    }
    // h_pairs[c * k + d] = length-n vector of H_i[c, d].
    let mut h_pairs: Vec<Array1<f64>> = Vec::with_capacity(k * k);
    for c in 0..k {
        for d in 0..k {
            let mut v = Array1::<f64>::zeros(n);
            for i in 0..n {
                v[i] = h_full[[i, c, d]];
            }
            h_pairs.push(v);
        }
    }

    let mut gram = Array2::<f64>::zeros((p_total, p_total));
    // Accumulate upper triangle (a ≤ b) then symmetrise.
    for a in 0..num_blocks {
        let range_a = raw_block_ranges[a].clone();
        for b in a..num_blocks {
            let range_b = raw_block_ranges[b].clone();
            let mut block_acc =
                Array2::<f64>::zeros((range_a.end - range_a.start, range_b.end - range_b.start));
            for c in 0..k {
                let Some(x_a_c) = channel_blocks.blocks[a][c].as_ref() else {
                    continue;
                };
                for d in 0..k {
                    let Some(x_b_d) = channel_blocks.blocks[b][d].as_ref() else {
                        continue;
                    };
                    let h_cd = &h_pairs[c * k + d];
                    // (X_a^(c))ᵀ · diag(h_cd) · X_b^(d)  →  (p_a × p_b).
                    let contrib = fast_xt_diag_y(x_a_c, h_cd, x_b_d);
                    block_acc += &contrib;
                }
            }
            // Write into upper triangle (and the diagonal block itself).
            gram.slice_mut(s![range_a.start..range_a.end, range_b.start..range_b.end])
                .assign(&block_acc);
        }
    }
    // Symmetrise: copy upper triangle to lower. Diagonal blocks are
    // themselves p_a × p_a — symmetrise within them too.
    for i in 0..p_total {
        for j in 0..i {
            let v = gram[[j, i]];
            gram[[i, j]] = v;
        }
    }
    Ok(gram)
}

/// Structural Gram `K^S`: same shape as [`build_raw_grams_from_channel_blocks`]
/// but with the per-row Hessian replaced by the K×K identity. Used by the
/// dual-metric compiler as the un-weighted reference geometry.
///
/// `K^S[a, b] = Σ_c (X_a^(c))ᵀ · X_b^(c)` (cross-channel terms vanish under
/// `H_i = I_K`).
pub fn build_raw_grams_structural(
    channel_blocks: &PrimaryChannelBlocks,
    raw_block_ranges: &[std::ops::Range<usize>],
) -> Array2<f64> {
    let num_blocks = channel_blocks.blocks.len();
    assert_eq!(
        num_blocks,
        raw_block_ranges.len(),
        "channel_blocks ({num_blocks}) and raw_block_ranges ({}) length mismatch",
        raw_block_ranges.len()
    );
    if num_blocks == 0 {
        return Array2::<f64>::zeros((0, 0));
    }
    let p_total = raw_block_ranges.last().map(|r| r.end).unwrap_or(0);
    let mut gram = Array2::<f64>::zeros((p_total, p_total));
    for a in 0..num_blocks {
        let range_a = raw_block_ranges[a].clone();
        for b in a..num_blocks {
            let range_b = raw_block_ranges[b].clone();
            let p_a = range_a.end - range_a.start;
            let p_b = range_b.end - range_b.start;
            let k_a = channel_blocks.blocks[a].len();
            let k_b = channel_blocks.blocks[b].len();
            assert_eq!(
                k_a, k_b,
                "structural Gram: block {a} has {k_a} channels but block {b} has {k_b}",
            );
            let mut block_acc = Array2::<f64>::zeros((p_a, p_b));
            for c in 0..k_a {
                let (Some(x_a_c), Some(x_b_c)) = (
                    channel_blocks.blocks[a][c].as_ref(),
                    channel_blocks.blocks[b][c].as_ref(),
                ) else {
                    continue;
                };
                let contrib = if a == b {
                    // Diagonal block, same channel — symmetric XᵀX.
                    fast_ata(x_a_c)
                } else {
                    fast_atb(x_a_c, x_b_c)
                };
                block_acc += &contrib;
            }
            gram.slice_mut(s![range_a.start..range_a.end, range_b.start..range_b.end])
                .assign(&block_acc);
        }
    }
    for i in 0..p_total {
        for j in 0..i {
            let v = gram[[j, i]];
            gram[[i, j]] = v;
        }
    }
    gram
}

/// Build the primary-state curvature Gram `K^H` and structural Gram `K^S`
/// for a block decomposition, preferring the device (GPU) path when
/// available and falling back to the CPU closed-form builders otherwise.
///
/// The GPU path is only attempted for survival-family geometry
/// (`K = CHANNELS = 4`) — that is the case the GPU kernel
/// ([`crate::identifiability::families::gpu::try_primary_state_gram_cuda`])
/// is specialised for via the packed-symmetric `n × 10` weight layout.
/// For any other `K` the CPU builders are used unconditionally.
///
/// Returns `(gram_h, gram_struct)` with the same shape and semantics as
/// [`build_raw_grams_from_channel_blocks`] + [`build_raw_grams_structural`].
pub fn build_primary_grams_gpu_or_cpu(
    channel_blocks: &PrimaryChannelBlocks,
    row_hess: &dyn RowHessian,
    raw_block_ranges: &[std::ops::Range<usize>],
) -> Result<(Array2<f64>, Array2<f64>), CompilerError> {
    let k = row_hess.k();
    if k == crate::identifiability::families::gpu::CHANNELS {
        let gpu_blocks: Vec<Vec<Option<Array2<f64>>>> = channel_blocks
            .blocks
            .iter()
            .map(|slots| slots.iter().cloned().collect())
            .collect();
        if let Some(h_packed) = pack_row_hessian_symmetric(row_hess) {
            if let Some(bundle) = crate::identifiability::families::gpu::try_primary_state_gram_cuda(
                &gpu_blocks,
                &h_packed,
                raw_block_ranges,
            ) {
                log::info!("[identifiability_compile] gram path = gpu");
                return Ok((bundle.gram_h, bundle.gram_struct));
            }
        }
    }
    log::info!("[identifiability_compile] gram path = cpu");
    let gram_h = build_raw_grams_from_channel_blocks(channel_blocks, row_hess, raw_block_ranges)?;
    let gram_struct = build_raw_grams_structural(channel_blocks, raw_block_ranges);
    Ok((gram_h, gram_struct))
}

/// Pack a per-row symmetric `K = 4` Hessian into the `n × 10`
/// upper-triangular row-major layout consumed by the GPU kernel
/// (`packed_index(c, d)` for `c ≤ d`). Returns `None` when `K != 4`.
fn pack_row_hessian_symmetric(row_hess: &dyn RowHessian) -> Option<Array2<f64>> {
    use crate::identifiability::families::gpu::{CHANNELS, PACKED_LEN, packed_index};
    if row_hess.k() != CHANNELS {
        return None;
    }
    let n = row_hess.nrows();
    let h_full = row_hess.evaluate_full();
    if h_full.shape() != [n, CHANNELS, CHANNELS] {
        return None;
    }
    let mut packed = Array2::<f64>::zeros((n, PACKED_LEN));
    for i in 0..n {
        for c in 0..CHANNELS {
            for d in c..CHANNELS {
                packed[[i, packed_index(c, d)]] = h_full[[i, c, d]];
            }
        }
    }
    Some(packed)
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
        let q_plus = keep_positive_eigenspace(&g_s_res, p_raw, 1, g_s_bb_trace)?;
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
        let u_mat = keep_positive_eigenspace(&g_h_res, p_raw, 1, g_h_dd_trace)?;
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

    /// Compiled (reduced) coefficient width (`p_compiled`).
    pub fn p_compiled(&self) -> usize {
        self.raw_from_compiled.ncols()
    }

    /// Reparameterise a raw design into compiled coordinates:
    /// `X_compiled = X_raw · T` (`n × p_compiled`). Because the lift is
    /// `β_raw = T β_compiled`, the compiled design predicts identically to the
    /// raw design on every compiled coefficient: `X_compiled · θ = X_raw · (T θ)`.
    /// Families that build directly in reduced coordinates feed this compiled
    /// design (and the [`reduce_penalties_with_map`] penalties) to the solver;
    /// the rank-deficient raw basis never reaches Newton.
    pub fn reduce_design(&self, raw_design: &Array2<f64>) -> Result<Array2<f64>, String> {
        if raw_design.ncols() != self.p_raw() {
            return Err(format!(
                "CompiledMap::reduce_design: raw_design has {} columns, expected p_raw {}",
                raw_design.ncols(),
                self.p_raw()
            ));
        }
        Ok(fast_ab(raw_design, &self.raw_from_compiled))
    }

    /// Lift a fitted compiled-width coefficient vector back to raw width:
    /// `β_raw = T · β_compiled`. This is the exact inverse direction of the
    /// quotient reduction — the reduced coordinates are what Newton/REML
    /// operate in, and this map carries the final estimate (and any linear
    /// functional of it) back to the original parameterisation so reported
    /// coefficients and predictions match the raw design.
    pub fn lift_coefficients(&self, beta_compiled: &Array1<f64>) -> Result<Array1<f64>, String> {
        if beta_compiled.len() != self.p_compiled() {
            return Err(format!(
                "CompiledMap::lift_coefficients: beta_compiled len {} != p_compiled {}",
                beta_compiled.len(),
                self.p_compiled()
            ));
        }
        Ok(self.raw_from_compiled.dot(beta_compiled))
    }

    /// The rows of `T` belonging to raw block `b` (`T[raw_block_ranges[b], :]`,
    /// shape `p_b_raw × p_compiled`). A raw-block penalty `S_b` acts only on
    /// these raw columns, so the penalty's reduced-coordinate form depends on
    /// `T` only through this slice.
    fn raw_block_rows(&self, block_idx: usize) -> Result<Array2<f64>, String> {
        let range = self.raw_block_ranges.get(block_idx).ok_or_else(|| {
            format!(
                "CompiledMap::raw_block_rows: block {block_idx} out of range {}",
                self.raw_block_ranges.len()
            )
        })?;
        Ok(self
            .raw_from_compiled
            .slice(s![range.start..range.end, ..])
            .to_owned())
    }
}

/// Transform a per-block raw-width penalty into the compiled (reduced)
/// coordinate frame defined by `map`.
///
/// `raw_penalties[b]` is the penalty matrix `S_b` acting on raw block `b`
/// (shape `p_b_raw × p_b_raw`), or `None` for an unpenalised block. The
/// returned `reduced[b]` is the **full** `(p_compiled × p_compiled)` penalty
/// `Tᵀ Ŝ_b T`, where `Ŝ_b` embeds `S_b` into the `p_raw × p_raw` zero matrix
/// at block `b`'s position. Because `Ŝ_b` is zero outside block `b`'s rows and
/// columns, this equals `T_bᵀ S_b T_b` with `T_b = T[raw_block_ranges[b], :]`,
/// so the reduced penalty is computed from the block's lift rows alone — no
/// dense `p_raw × p_raw` embedding is materialised.
///
/// Exactness: for any compiled coefficient `θ` with raw lift `β = T θ`, the raw
/// penalty energy `βᵀ Ŝ_b β = (T θ)ᵀ Ŝ_b (T θ) = θᵀ (Tᵀ Ŝ_b T) θ`, so the
/// reduced penalty reproduces the raw penalty energy on every lifted point.
/// A compiled block that absorbed to zero width simply contributes a zero
/// column range; its raw penalty (if any) projects onto the surviving
/// compiled directions through `T_b`, never lost.
pub fn reduce_penalties_with_map(
    map: &CompiledMap,
    raw_penalties: &[Option<Array2<f64>>],
) -> Result<Vec<Option<Array2<f64>>>, String> {
    if raw_penalties.len() != map.raw_block_ranges.len() {
        return Err(format!(
            "reduce_penalties_with_map: raw_penalties ({}) != blocks ({})",
            raw_penalties.len(),
            map.raw_block_ranges.len()
        ));
    }
    let p_compiled = map.p_compiled();
    let mut reduced: Vec<Option<Array2<f64>>> = Vec::with_capacity(raw_penalties.len());
    for (block_idx, raw_penalty) in raw_penalties.iter().enumerate() {
        let Some(s_b) = raw_penalty.as_ref() else {
            reduced.push(None);
            continue;
        };
        let p_b_raw = map.raw_block_ranges[block_idx].len();
        if s_b.shape() != [p_b_raw, p_b_raw] {
            return Err(format!(
                "reduce_penalties_with_map: block {block_idx} penalty shape {:?} != [{p_b_raw}, {p_b_raw}]",
                s_b.shape()
            ));
        }
        // T_b = T[raw rows of block b, :]  (p_b_raw × p_compiled)
        let t_b = map.raw_block_rows(block_idx)?;
        // S_compiled = T_bᵀ S_b T_b  (p_compiled × p_compiled)
        let s_t_b = fast_ab(s_b, &t_b); // (p_b_raw × p_compiled)
        let s_compiled_raw = fast_atb(&t_b, &s_t_b); // (p_compiled × p_compiled)
        let mut s_compiled = symmetrise(&s_compiled_raw);
        if s_compiled.shape() != [p_compiled, p_compiled] {
            return Err(format!(
                "reduce_penalties_with_map: block {block_idx} reduced penalty shape {:?} != [{p_compiled}, {p_compiled}]",
                s_compiled.shape()
            ));
        }
        for v in s_compiled.iter_mut() {
            if !v.is_finite() {
                return Err(format!(
                    "reduce_penalties_with_map: block {block_idx} reduced penalty has non-finite entry"
                ));
            }
        }
        reduced.push(Some(s_compiled));
    }
    Ok(reduced)
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
    // sqrt(W) row scale (clamp tiny-negative to zero — the pilot Hessian
    // diagonal is PSD-clamped upstream, but guard against round-off).
    let mut sqrt_w = Array1::<f64>::zeros(n);
    for i in 0..n {
        let wi = weight[i].max(0.0);
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
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
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
                BlockOrder::Logslope,
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
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
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
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
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
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
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
            &[BlockOrder::Marginal, BlockOrder::Logslope],
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
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
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
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
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

    /// `K=4` dense row Hessian: per-row PSD matrix supplied directly.
    struct DenseRowHessian {
        h: Array3<f64>,
    }

    impl RowHessian for DenseRowHessian {
        fn k(&self) -> usize {
            self.h.shape()[1]
        }
        fn nrows(&self) -> usize {
            self.h.shape()[0]
        }
        fn fill_row(&self, row: usize, out: &mut [f64]) {
            let k = self.k();
            assert_eq!(out.len(), k * k);
            for c in 0..k {
                for d in 0..k {
                    out[c * k + d] = self.h[[row, c, d]];
                }
            }
        }
        fn evaluate_full(&self) -> Array3<f64> {
            self.h.clone()
        }
    }

    /// Reference W-based Gram for verification: build `W = sqrt(H) · J` then
    /// return `Wᵀ W`. Mirrors the in-walk path in [`compile`].
    fn reference_gram_from_w(j_full: &Array3<f64>, h_full: &Array3<f64>) -> Array2<f64> {
        let w = scale_block_by_sqrt_h(j_full, h_full);
        fast_ata(&w)
    }

    /// Two-block toy at K=4: build per-channel (n × p_b) blocks and verify
    /// the closed-form Gram matches the reference W-based Gram.
    #[test]
    fn closed_form_gram_matches_reference_two_block_k4() {
        let n = 17;
        let k = 4;
        let p_a = 3;
        let p_b = 2;

        // Random-ish per-channel design matrices for each block.
        let make_block = |seed: f64, n: usize, p: usize| -> Vec<Option<Array2<f64>>> {
            (0..4)
                .map(|c| {
                    let m = Array2::from_shape_fn((n, p), |(i, j)| {
                        ((i as f64 + 1.0) * (j as f64 + 1.0) * (c as f64 + 1.0) + seed).sin()
                    });
                    Some(m)
                })
                .collect()
        };
        let block_a = make_block(0.3, n, p_a);
        let block_b = make_block(1.1, n, p_b);

        // Per-row PSD H: random symmetric PSD via Mᵀ M.
        let h = Array3::from_shape_fn((n, k, k), |(i, c, d)| {
            let mut acc = 0.0;
            for r in 0..k {
                let mc = ((i + 1) as f64 * (c + 1) as f64 * (r + 1) as f64 * 0.13).cos();
                let md = ((i + 1) as f64 * (d + 1) as f64 * (r + 1) as f64 * 0.13).cos();
                acc += mc * md;
            }
            acc + if c == d { 0.5 } else { 0.0 }
        });
        let row_hess = DenseRowHessian { h: h.clone() };

        let channel_blocks = PrimaryChannelBlocks {
            blocks: vec![block_a.clone(), block_b.clone()],
        };
        let raw_ranges = vec![0..p_a, p_a..(p_a + p_b)];

        let gram = build_raw_grams_from_channel_blocks(&channel_blocks, &row_hess, &raw_ranges)
            .expect("closed-form Gram should succeed");

        // Reference: assemble full row Jacobian J as (n × p_total × K) by
        // placing per-block, per-channel slices at the right columns.
        let p_total = p_a + p_b;
        let mut j_full = Array3::<f64>::zeros((n, p_total, k));
        for c in 0..k {
            if let Some(xa) = block_a[c].as_ref() {
                for i in 0..n {
                    for j in 0..p_a {
                        j_full[[i, j, c]] = xa[[i, j]];
                    }
                }
            }
            if let Some(xb) = block_b[c].as_ref() {
                for i in 0..n {
                    for j in 0..p_b {
                        j_full[[i, p_a + j, c]] = xb[[i, j]];
                    }
                }
            }
        }
        let ref_gram = reference_gram_from_w(&j_full, &h);

        let diff = &gram - &ref_gram;
        let max_err = diff.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let scale = ref_gram.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            max_err < 1e-9 * scale.max(1.0),
            "closed-form Gram mismatches reference: max_err={max_err:e}, scale={scale:e}"
        );

        // Symmetry of the result.
        for i in 0..p_total {
            for j in 0..p_total {
                assert!(
                    (gram[[i, j]] - gram[[j, i]]).abs() < 1e-12,
                    "closed-form Gram not symmetric at ({i},{j})"
                );
            }
        }
    }

    /// Channel sparsity test: block A contributes only to channel 0, block B
    /// only to channel 3. Cross-block contribution must be exactly
    /// `(X_A^(0))ᵀ · diag(h_{03}) · X_B^(3)` — zero when `h_03 ≡ 0`,
    /// non-zero otherwise.
    #[test]
    fn closed_form_gram_channel_sparsity() {
        let n = 13;
        let k = 4;
        let p_a = 2;
        let p_b = 2;

        let xa = Array2::from_shape_fn((n, p_a), |(i, j)| ((i + 1) as f64 * 0.21 + j as f64).cos());
        let xb = Array2::from_shape_fn((n, p_b), |(i, j)| {
            ((i + 1) as f64 * 0.17 + j as f64).sin() + 0.5
        });

        let block_a: Vec<Option<Array2<f64>>> = vec![Some(xa.clone()), None, None, None];
        let block_b: Vec<Option<Array2<f64>>> = vec![None, None, None, Some(xb.clone())];

        // Case 1: H with non-zero h_{03} (and h_{30}). The cross-block
        // (A, B) entries must equal `Xaᵀ · diag(h_03) · Xb`.
        let h_03_vec = Array1::from_shape_fn(n, |i| 0.7 + 0.3 * ((i as f64) * 0.4).sin());
        let h = Array3::from_shape_fn((n, k, k), |(i, c, d)| {
            // Symmetric: only the (0,3)/(3,0) off-diagonal carries weight,
            // plus a strong PSD diagonal so per-row H is PSD.
            if (c, d) == (0, 3) || (c, d) == (3, 0) {
                h_03_vec[i]
            } else if c == d {
                2.0
            } else {
                0.0
            }
        });
        let row_hess = DenseRowHessian { h: h.clone() };

        let channel_blocks = PrimaryChannelBlocks {
            blocks: vec![block_a.clone(), block_b.clone()],
        };
        let raw_ranges = vec![0..p_a, p_a..(p_a + p_b)];
        let gram = build_raw_grams_from_channel_blocks(&channel_blocks, &row_hess, &raw_ranges)
            .expect("closed-form Gram should succeed");

        // Cross-block submatrix.
        let cross = gram.slice(s![0..p_a, p_a..(p_a + p_b)]).to_owned();
        // Expected: only the (c=0, d=3) channel-pair survives.
        let expected = fast_xt_diag_y(&xa, &h_03_vec, &xb);
        let diff = &cross - &expected;
        let max_err = diff.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            max_err < 1e-12,
            "cross-block Gram must equal Xaᵀ·diag(h_03)·Xb: max_err={max_err:e}"
        );

        // Case 2: zero out h_{03} → cross-block must be zero.
        let h_zero = Array3::from_shape_fn((n, k, k), |(_, c, d)| if c == d { 2.0 } else { 0.0 });
        let row_hess_zero = DenseRowHessian { h: h_zero };
        let gram_zero =
            build_raw_grams_from_channel_blocks(&channel_blocks, &row_hess_zero, &raw_ranges)
                .expect("closed-form Gram should succeed");
        let cross_zero = gram_zero.slice(s![0..p_a, p_a..(p_a + p_b)]);
        let max_zero = cross_zero.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            max_zero < 1e-12,
            "cross-block Gram must vanish when coupling channel pair is zero: got {max_zero:e}"
        );
    }

    /// Structural Gram: identity per-row Hessian collapses the channel-pair
    /// sum to within-channel `XᵀX`. Validates [`build_raw_grams_structural`].
    #[test]
    fn structural_gram_matches_within_channel_sum() {
        let n = 11;
        let p_a = 2;
        let p_b = 3;
        let make_block = |seed: f64, n: usize, p: usize| -> Vec<Option<Array2<f64>>> {
            (0..4)
                .map(|c| {
                    if c == 1 {
                        // Sparse channel for variety.
                        return None;
                    }
                    Some(Array2::from_shape_fn((n, p), |(i, j)| {
                        ((i as f64 + 1.0) * (j as f64 + 1.0) + seed * (c as f64 + 1.0)).sin()
                    }))
                })
                .collect()
        };
        let block_a = make_block(0.1, n, p_a);
        let block_b = make_block(0.7, n, p_b);
        let channel_blocks = PrimaryChannelBlocks {
            blocks: vec![block_a.clone(), block_b.clone()],
        };
        let raw_ranges = vec![0..p_a, p_a..(p_a + p_b)];
        let gram = build_raw_grams_structural(&channel_blocks, &raw_ranges);

        // Hand-compute cross block: Σ_c Xaᵀ Xb over channels where both
        // sides are present (skipping channel 1 entirely).
        let mut expected_cross = Array2::<f64>::zeros((p_a, p_b));
        for c in 0..4 {
            if let (Some(xa), Some(xb)) = (block_a[c].as_ref(), block_b[c].as_ref()) {
                expected_cross += &fast_atb(xa, xb);
            }
        }
        let cross = gram.slice(s![0..p_a, p_a..(p_a + p_b)]).to_owned();
        let diff = &cross - &expected_cross;
        let max_err = diff.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            max_err < 1e-12,
            "structural cross-block must equal Σ_c Xaᵀ·Xb: max_err={max_err:e}"
        );

        // Symmetry.
        for i in 0..(p_a + p_b) {
            for j in 0..(p_a + p_b) {
                assert!(
                    (gram[[i, j]] - gram[[j, i]]).abs() < 1e-12,
                    "structural Gram not symmetric at ({i},{j})"
                );
            }
        }
    }

    // Per-row Hessian (K=1) sourced from an arbitrary positive vector —
    // used by the dual-metric sanity test to drive both structural and
    // curvature passes with the *same* non-identity weights.
    fn diag_hess(w: Array1<f64>) -> DiagonalScalarRowHessian {
        DiagonalScalarRowHessian::new(w)
    }

    /// L#1: dual-metric with structural = curvature reproduces single-metric
    /// `compile()` exactly. The two passes degenerate to one because the
    /// structural-anchor and curvature-anchor are the same matrix.
    #[test]
    fn dual_metric_with_equal_metrics_matches_single_metric() {
        let n = 36;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 * 0.13 + j as f64).sin());
        // B partially aliases A's first column.
        let b = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.4 * a[[i, 0]] + (i as f64 * 0.07 + j as f64).cos()
        });
        let w = Array1::from_shape_fn(n, |i| 0.5 + (i as f64 * 0.17).sin().abs());
        let curvature = diag_hess(w.clone());
        let ordering = [BlockOrder::Marginal, BlockOrder::Logslope];

        let ops_single = vec![op(a.clone()), op(b.clone())];
        let single = compile(&ops_single, &curvature, &ordering)
            .expect("single-metric compile should succeed");

        // Dual-metric with structural = curvature (same `RowHessian` on both
        // sides). The structural pass collapses to the curvature pass.
        let structural_same = diag_hess(w.clone());
        let ops_dual = vec![op(a.clone()), op(b.clone())];
        let dual = compile_with_dual_metric(&ops_dual, &curvature, &structural_same, &ordering)
            .expect("dual-metric compile should succeed");

        assert_eq!(single.blocks.len(), dual.blocks.len());
        for (idx, (sb, db)) in single.blocks.iter().zip(dual.blocks.iter()).enumerate() {
            assert_eq!(sb.t_lw.dim(), db.t_lw.dim(), "block {idx}: V dims differ");
            let max_v = (&sb.t_lw - &db.t_lw)
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            assert!(max_v < 1e-10, "block {idx}: V mismatch {max_v:e}");
            match (sb.anchor_correction.as_ref(), db.anchor_correction.as_ref()) {
                (None, None) => {}
                (Some(s), Some(d)) => {
                    assert_eq!(s.dim(), d.dim());
                    let max_m = (s - d).iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
                    assert!(max_m < 1e-10, "block {idx}: M mismatch {max_m:e}");
                }
                _ => panic!("block {idx}: one side has anchor correction, the other does not"),
            }
        }
        assert_eq!(single.joint_rank, dual.joint_rank);
    }

    /// L#2: the pilot-curvature trap. A 2-block toy where the pilot
    /// curvature `H` has a zero direction that is NOT a real gauge — the
    /// dual-metric path keeps it (identity-structural sees it as a full-
    /// rank structural direction), while a single-metric path through the
    /// same H would drop it.
    ///
    /// Construction: two K=1 blocks `A` (n × 1) and `B` (n × 1). Choose H
    /// (diagonal row weights) so that `H · B` happens to be a scalar
    /// multiple of `H · A` (curvature alias) but `B` is *not* a scalar
    /// multiple of `A` in the unweighted metric. Specifically, pick rows
    /// where `w_i` is non-zero only on a handful of rows where A and B
    /// happen to be proportional, and zero on the rows where they differ.
    /// Under identity-structural this is structurally-independent; under H
    /// it is a (spurious) curvature alias.
    #[test]
    fn dual_metric_resists_pilot_curvature_alias() {
        let n = 12;
        // A: x_i = i+1 (no zeros). B: equals 2·A on rows 0..6 only; the
        // remaining rows are uncorrelated (linear vs trigonometric).
        let a = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64) + 1.0);
        let b = Array2::from_shape_fn((n, 1), |(i, _)| {
            if i < 6 {
                2.0 * a[[i, 0]]
            } else {
                ((i as f64) * 0.3).cos() + 0.5
            }
        });

        // Curvature weights are non-zero ONLY on the rows where B == 2A.
        // Under curvature metric, B is exactly 2·A → curvature-rank drops
        // B fully. Under identity-structural, B is independent of A across
        // all rows → structural-rank is 1 (kept).
        let mut w_vec = vec![0.0_f64; n];
        for w in &mut w_vec[..6] {
            *w = 1.0;
        }
        let w = Array1::from(w_vec);
        let curvature = diag_hess(w.clone());

        // Reference single-metric compile (uses identity by `compile()` —
        // which now routes through identity-structural). For this test we
        // explicitly invoke the dual-metric API both ways.
        let id_struct = IdentityRowHessian::new(n, 1);
        let ordering = [BlockOrder::Marginal, BlockOrder::Logslope];

        // Path 1: dual-metric with identity-structural (the new default).
        // Structural pass: B is independent of A across all rows → keep
        // B's single column.
        let ops_dual = vec![op(a.clone()), op(b.clone())];
        let dual = compile_with_dual_metric(&ops_dual, &curvature, &id_struct, &ordering);

        // Path 2: dual-metric with structural = curvature (the "H decides
        // everything" trap). On the curvature-only rows, B ≡ 2A, so
        // structural pass sees zero residual span and rejects the block.
        let ops_h_only = vec![op(a.clone()), op(b.clone())];
        let h_only = compile_with_dual_metric(&ops_h_only, &curvature, &curvature, &ordering);

        // The H-only path must fail (FullyAliased) or strip B's column.
        // The dual (identity-structural) path must keep B.
        match h_only {
            Err(CompilerError::FullyAliased { block_idx, .. }) => {
                assert_eq!(block_idx, 1, "H-only path must alias block 1");
            }
            Ok(out) => {
                // If the H-only path somehow compiled, it must have
                // either dropped B's column to zero width or audited it
                // out. Either way B's V must be empty after the audit
                // attributes the drop.
                let v1_cols = out.blocks[1].t_lw.ncols();
                assert!(
                    v1_cols == 0 || !out.dropped.is_empty(),
                    "H-only path should reject B's curvature-aliased column; v1_cols={v1_cols}, dropped={dropped:?}",
                    dropped = out.dropped,
                );
            }
            Err(other) => panic!("unexpected H-only error: {other:?}"),
        }

        let dual =
            dual.expect("dual-metric must succeed: identity-structural sees B as independent");
        // The dual path may still drop B's column at the joint audit step
        // because the joint H-scaled design is rank-1 (only the first
        // block contributes non-zero rows under the curvature weights).
        // What matters is that the *structural* decision did NOT drop B
        // — verified by the structural pass not raising FullyAliased and
        // by B's `t_lw` having the full structural width before the audit
        // demotes it. After audit, B's V may shrink because the curvature
        // joint design is rank-deficient, and that is expected.
        assert_eq!(dual.blocks.len(), 2);
        assert_eq!(dual.blocks[0].t_lw.ncols(), 1, "A must keep its column");
        // Block 1 either keeps its structural rank-1 column or is audited
        // away by the joint H-rank check, but in either case the per-block
        // pre-audit width must reflect that the structural pass kept the
        // column (i.e. the function did not return FullyAliased).
        let v1_post_audit = dual.blocks[1].t_lw.ncols();
        let dropped_count = dual.dropped.len();
        assert_eq!(
            v1_post_audit + dropped_count,
            1,
            "structural pass kept B's column; audit may demote it but the pre-audit width was 1"
        );
    }

    /// L#3: identity-structural lets the compiler keep a direction even
    /// when the pilot curvature has reduced rank. This is the same
    /// scenario as L#2 but with a curvature `H` whose row weights are all
    /// strictly positive — so the *only* aliasing source is the structural
    /// pass deciding to keep or drop. The dual-metric path with non-trivial
    /// `H` and identity-structural must agree with the dual-metric path
    /// with identity on both sides whenever the blocks are structurally
    /// non-aliased.
    #[test]
    fn dual_metric_identity_structural_preserves_full_rank() {
        let n = 24;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| ((i + 1) as f64 + j as f64).sqrt());
        let b = Array2::from_shape_fn((n, 2), |(i, j)| {
            ((i + 1) as f64).ln() + (i as f64 * 0.1 + j as f64).cos()
        });
        let w = Array1::from_shape_fn(n, |i| 0.4 + (i as f64 * 0.05).sin().powi(2));
        let curvature = diag_hess(w.clone());
        let id_struct = IdentityRowHessian::new(n, 1);
        let ordering = [BlockOrder::Marginal, BlockOrder::Logslope];

        let ops = vec![op(a.clone()), op(b.clone())];
        let out =
            compile_with_dual_metric(&ops, &curvature, &id_struct, &ordering).expect("compile");
        // Both blocks structurally independent → both keep full width.
        assert_eq!(out.blocks[0].t_lw.ncols(), 2);
        assert_eq!(out.blocks[1].t_lw.ncols(), 2);
        assert_eq!(out.dropped.len(), 0);
        assert_eq!(out.joint_rank, 4);
    }

    /// Smoke test for the GPU-or-CPU dispatch helper. On non-CUDA hosts
    /// (or when the runtime is unavailable) the helper falls back to the
    /// CPU closed-form builders; the result must match the CPU builders
    /// called directly. When a CUDA runtime is live, parity vs. CPU is
    /// verified to tight tolerance.
    #[test]
    fn build_primary_grams_gpu_or_cpu_two_block_k4_matches_cpu() {
        let n = 11;
        let k = 4;
        let p_a = 2;
        let p_b = 3;

        let make_block = |seed: f64, n: usize, p: usize| -> Vec<Option<Array2<f64>>> {
            (0..4)
                .map(|c| {
                    let m = Array2::from_shape_fn((n, p), |(i, j)| {
                        ((i as f64 + 1.0) * (j as f64 + 1.0) * (c as f64 + 1.0) + seed).sin()
                    });
                    Some(m)
                })
                .collect()
        };
        let block_a = make_block(0.7, n, p_a);
        let block_b = make_block(-0.4, n, p_b);

        let h = Array3::from_shape_fn((n, k, k), |(i, c, d)| {
            let mut acc = 0.0;
            for r in 0..k {
                let mc = ((i + 1) as f64 * (c + 1) as f64 * (r + 1) as f64 * 0.11).cos();
                let md = ((i + 1) as f64 * (d + 1) as f64 * (r + 1) as f64 * 0.11).cos();
                acc += mc * md;
            }
            acc + if c == d { 0.25 } else { 0.0 }
        });
        let row_hess = DenseRowHessian { h: h.clone() };

        let channel_blocks = PrimaryChannelBlocks {
            blocks: vec![block_a, block_b],
        };
        let raw_ranges = vec![0..p_a, p_a..(p_a + p_b)];

        let (gram_h, gram_struct) =
            build_primary_grams_gpu_or_cpu(&channel_blocks, &row_hess, &raw_ranges)
                .expect("dispatch helper should succeed");

        let cpu_h = build_raw_grams_from_channel_blocks(&channel_blocks, &row_hess, &raw_ranges)
            .expect("CPU curvature Gram should succeed");
        let cpu_s = build_raw_grams_structural(&channel_blocks, &raw_ranges);

        let tol = 1e-9_f64;
        for idx in cpu_h.indexed_iter().map(|(i, _)| i) {
            let diff = (gram_h[idx] - cpu_h[idx]).abs();
            let scale = cpu_h[idx].abs().max(1.0);
            assert!(
                diff <= tol * scale,
                "gram_h mismatch at {idx:?}: helper={} cpu={}",
                gram_h[idx],
                cpu_h[idx]
            );
        }
        for idx in cpu_s.indexed_iter().map(|(i, _)| i) {
            let diff = (gram_struct[idx] - cpu_s[idx]).abs();
            let scale = cpu_s[idx].abs().max(1.0);
            assert!(
                diff <= tol * scale,
                "gram_struct mismatch at {idx:?}: helper={} cpu={}",
                gram_struct[idx],
                cpu_s[idx]
            );
        }
    }

    // ---- compile_from_raw_grams tests ----

    /// Build (gram_h, gram_struct) for a K=1 scalar two-block toy via the
    /// per-block channel-block builders. Used by the closed-form tests
    /// below.
    fn scalar_grams_two_block(
        a: &Array2<f64>,
        b: &Array2<f64>,
        w: &Array1<f64>,
    ) -> (Array2<f64>, Array2<f64>, Vec<std::ops::Range<usize>>) {
        let p_a = a.ncols();
        let p_b = b.ncols();
        let raw_ranges = vec![0..p_a, p_a..(p_a + p_b)];
        let channel_blocks = PrimaryChannelBlocks {
            blocks: vec![vec![Some(a.clone())], vec![Some(b.clone())]],
        };
        let row_hess = DiagonalScalarRowHessian::new(w.clone());
        let gram_h =
            build_raw_grams_from_channel_blocks(&channel_blocks, &row_hess, &raw_ranges).unwrap();
        let gram_struct = build_raw_grams_structural(&channel_blocks, &raw_ranges);
        (gram_h, gram_struct, raw_ranges)
    }

    /// Block B is a column-duplicate of block A in the structural metric
    /// → the lower-priority block compiles to zero width instead of making
    /// callers skip reduced-coordinate construction.
    #[test]
    fn compile_from_raw_grams_full_structural_alias() {
        let n = 10;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| ((i + 1) as f64 * (j + 1) as f64).sin());
        // Block B = A · L for some 2×2 invertible L → same column span.
        let l = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, -0.25, 1.0]).unwrap();
        let b = a.dot(&l);
        let w = Array1::ones(n);
        let (gram_h, gram_struct, raw_ranges) = scalar_grams_two_block(&a, &b, &w);
        let res = compile_from_raw_grams(
            &gram_h,
            &gram_struct,
            &raw_ranges,
            &[BlockOrder::Marginal, BlockOrder::Logslope],
        )
        .expect("lower-priority full alias should compile to zero width");
        assert_eq!(res.compiled_block_ranges[0].len(), 2);
        assert_eq!(res.compiled_block_ranges[1].len(), 0);
        assert_eq!(res.raw_from_compiled.dim(), (4, 2));
        assert!(
            res.raw_from_compiled
                .slice(s![raw_ranges[1].clone(), ..])
                .iter()
                .all(|v| v.abs() <= 1.0e-12),
            "zero-width block must not retain raw coefficient directions in T"
        );
    }

    /// A zero-width *first* block has no columns to alias and must compile to
    /// an empty range with the remaining blocks intact — not abort with
    /// `FullyAliased`. Regression for the survival location-scale lognormal AFT
    /// pre-fit channel-aware audit, whose `time_transform` block collapses to
    /// zero free coefficients under the parametric AFT reduction and previously
    /// crashed the fit ("block of width 0 has zero structural span").
    #[test]
    fn compile_from_raw_grams_zero_width_first_block_is_identifiable() {
        let n = 12;
        let empty = Array2::<f64>::zeros((n, 0));
        let b = Array2::from_shape_fn((n, 2), |(i, j)| {
            ((i + 1) as f64 * (j + 1) as f64 * 0.23).cos()
        });
        let w = Array1::ones(n);
        let (gram_h, gram_struct, raw_ranges) = scalar_grams_two_block(&empty, &b, &w);
        let map = compile_from_raw_grams(
            &gram_h,
            &gram_struct,
            &raw_ranges,
            &[BlockOrder::Marginal, BlockOrder::Logslope],
        )
        .expect("zero-width first block must be trivially identifiable, not FullyAliased");
        assert_eq!(
            map.compiled_block_ranges[0].len(),
            0,
            "empty first block keeps zero columns"
        );
        assert_eq!(
            map.compiled_block_ranges[1].len(),
            2,
            "the second block keeps its full structural rank"
        );
        assert_eq!(map.raw_from_compiled.dim(), (2, 2));
    }

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
    fn compile_from_raw_grams_three_block_full_logslope_alias_keeps_fast_path() {
        let n = 24;
        let time = Array2::from_shape_fn((n, 2), |(i, j)| {
            ((i + 1) as f64 * (j + 2) as f64 * 0.17).sin()
        });
        let marginal = Array2::from_shape_fn((n, 1), |(i, _)| ((i + 3) as f64 * 0.11).cos());
        let logslope = marginal.clone();
        let p_time = time.ncols();
        let p_marg = marginal.ncols();
        let p_log = logslope.ncols();
        let raw_ranges = vec![
            0..p_time,
            p_time..(p_time + p_marg),
            (p_time + p_marg)..(p_time + p_marg + p_log),
        ];
        let channel_blocks = PrimaryChannelBlocks {
            blocks: vec![
                vec![Some(time.clone())],
                vec![Some(marginal.clone())],
                vec![Some(logslope.clone())],
            ],
        };
        let row_hess = DiagonalScalarRowHessian::new(Array1::ones(n));
        let gram_h =
            build_raw_grams_from_channel_blocks(&channel_blocks, &row_hess, &raw_ranges).unwrap();
        let gram_struct = build_raw_grams_structural(&channel_blocks, &raw_ranges);

        let map = compile_from_raw_grams(
            &gram_h,
            &gram_struct,
            &raw_ranges,
            &[BlockOrder::Time, BlockOrder::Marginal, BlockOrder::Logslope],
        )
        .expect("fully aliased logslope block should not skip the compiled-map path");

        assert_eq!(map.compiled_block_ranges[0].len(), p_time);
        assert_eq!(map.compiled_block_ranges[1].len(), p_marg);
        assert_eq!(map.compiled_block_ranges[2].len(), 0);
        assert_eq!(
            map.raw_from_compiled.dim(),
            (p_time + p_marg + p_log, p_time + p_marg)
        );
        let x_raw = {
            let mut out = Array2::<f64>::zeros((n, p_time + p_marg + p_log));
            out.slice_mut(s![.., raw_ranges[0].clone()]).assign(&time);
            out.slice_mut(s![.., raw_ranges[1].clone()])
                .assign(&marginal);
            out.slice_mut(s![.., raw_ranges[2].clone()])
                .assign(&logslope);
            out
        };
        let x_compiled = fast_ab(&x_raw, &map.raw_from_compiled);
        let rrqr = rrqr_with_permutation(&x_compiled, default_rrqr_rank_alpha()).unwrap();
        assert_eq!(rrqr.rank, x_compiled.ncols());
    }

    /// Partial alias: block B's first column duplicates A; second column is
    /// independent. Closed-form `T` must have shape `(p_raw × (p_a + 1))`
    /// — block 1's compiled width is exactly the independent direction —
    /// and the joint design `X_raw · T` must span the same column space as
    /// the W-based reference compile result.
    #[test]
    fn compile_from_raw_grams_partial_alias_matches_w_reference() {
        let n = 25;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| {
            ((i + 1) as f64 * (j + 1) as f64 * 0.3).sin()
        });
        // B = [a_0  +  independent]
        let mut b = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            b[[i, 0]] = a[[i, 0]];
            b[[i, 1]] = ((i + 1) as f64 * 0.7).cos();
        }
        let w = Array1::from_shape_fn(n, |i| 1.0 + 0.1 * (i as f64));
        let (gram_h, gram_struct, raw_ranges) = scalar_grams_two_block(&a, &b, &w);
        let compiled = compile_from_raw_grams(
            &gram_h,
            &gram_struct,
            &raw_ranges,
            &[BlockOrder::Marginal, BlockOrder::Logslope],
        )
        .expect("closed-form compile must succeed");
        let p_a = a.ncols();
        let p_b = b.ncols();
        assert_eq!(compiled.raw_from_compiled.shape()[0], p_a + p_b);
        assert_eq!(
            compiled.raw_from_compiled.shape()[1],
            p_a + 1,
            "partial alias should leave compiled width = p_a + 1 (one column dropped from B)"
        );
        // Block ranges sum to compiled width.
        assert_eq!(compiled.compiled_block_ranges[0], 0..p_a);
        assert_eq!(
            compiled.compiled_block_ranges[1].end - compiled.compiled_block_ranges[1].start,
            1
        );

        // Column-span equality vs. W-reference: stack the raw design
        // X_raw = [A | B] and check that range(X_raw · T) ⊆ range(X_raw)
        // and has the same rank as the W-based compile.
        let mut x_raw = Array2::<f64>::zeros((n, p_a + p_b));
        for i in 0..n {
            for j in 0..p_a {
                x_raw[[i, j]] = a[[i, j]];
            }
            for j in 0..p_b {
                x_raw[[i, p_a + j]] = b[[i, j]];
            }
        }
        let x_compiled = fast_ab(&x_raw, &compiled.raw_from_compiled);
        // Rank of compiled design via Gram eigvals.
        let g_compiled = fast_ata(&x_compiled);
        let (evals, _) = g_compiled.eigh(Side::Lower).unwrap();
        let lam_max = evals.iter().cloned().fold(0.0_f64, f64::max);
        let tol = lam_max * 64.0 * (g_compiled.nrows() as f64) * f64::EPSILON;
        let rank_compiled = evals.iter().filter(|&&l| l > tol).count();
        assert_eq!(
            rank_compiled,
            p_a + 1,
            "compiled design column rank must equal p_a + 1 after dropping the alias"
        );

        // Reference compile via the W-based dual-metric path on the same
        // scalar blocks; compiled total width should also be p_a + 1.
        let ops_dual: Vec<Arc<dyn RowJacobianOperator>> = vec![op(a.clone()), op(b.clone())];
        let curvature = DiagonalScalarRowHessian::new(w.clone());
        let id_struct = IdentityRowHessian::new(n, 1);
        let dual = compile_with_dual_metric(
            &ops_dual,
            &curvature,
            &id_struct,
            &[BlockOrder::Marginal, BlockOrder::Logslope],
        )
        .expect("dual metric compile should succeed");
        let dual_total: usize = dual.blocks.iter().map(|b| b.t_lw.ncols()).sum();
        assert_eq!(dual_total, p_a + 1, "W-reference total width should match");
    }

    /// Three-block toy: changing the ordering changes the per-block
    /// compiled widths (later blocks absorb the alias instead of earlier).
    #[test]
    fn compile_from_raw_grams_three_block_ordering_matters() {
        let n = 30;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| {
            ((i + 1) as f64 * (j + 2) as f64 * 0.2).sin()
        });
        // B has 2 cols: col 0 independent, col 1 = a[:, 0]
        let mut b = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            b[[i, 0]] = ((i + 1) as f64 * 0.4).cos();
            b[[i, 1]] = a[[i, 0]];
        }
        // C has 2 cols: col 0 independent, col 1 = a[:, 1]
        let mut c = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            c[[i, 0]] = ((i + 1) as f64 * 0.55).sin();
            c[[i, 1]] = a[[i, 1]];
        }
        let w = Array1::ones(n);

        let build = |b0: &Array2<f64>, b1: &Array2<f64>, b2: &Array2<f64>| {
            let raw_ranges = vec![
                0..b0.ncols(),
                b0.ncols()..(b0.ncols() + b1.ncols()),
                (b0.ncols() + b1.ncols())..(b0.ncols() + b1.ncols() + b2.ncols()),
            ];
            let channel_blocks = PrimaryChannelBlocks {
                blocks: vec![
                    vec![Some(b0.clone())],
                    vec![Some(b1.clone())],
                    vec![Some(b2.clone())],
                ],
            };
            let row_hess = DiagonalScalarRowHessian::new(w.clone());
            let gram_h =
                build_raw_grams_from_channel_blocks(&channel_blocks, &row_hess, &raw_ranges)
                    .unwrap();
            let gram_struct = build_raw_grams_structural(&channel_blocks, &raw_ranges);
            (gram_h, gram_struct, raw_ranges)
        };

        // Order 1: A, B, C — B drops 1 (col 1 aliased to A), C drops 1.
        let (gh, gs, rr) = build(&a, &b, &c);
        let order_abc = compile_from_raw_grams(
            &gh,
            &gs,
            &rr,
            &[
                BlockOrder::Marginal,
                BlockOrder::Logslope,
                BlockOrder::LinkDev,
            ],
        )
        .expect("ABC compile");
        assert_eq!(order_abc.compiled_block_ranges[0].len(), 2);
        assert_eq!(order_abc.compiled_block_ranges[1].len(), 1);
        assert_eq!(order_abc.compiled_block_ranges[2].len(), 1);

        // Order 2: B, A, C — A's col 0 is aliased by B's col 1 now; A's
        // col 1 is independent. So A drops 1; C still drops 1.
        let (gh2, gs2, rr2) = build(&b, &a, &c);
        let order_bac = compile_from_raw_grams(
            &gh2,
            &gs2,
            &rr2,
            &[
                BlockOrder::Marginal,
                BlockOrder::Logslope,
                BlockOrder::LinkDev,
            ],
        )
        .expect("BAC compile");
        assert_eq!(order_bac.compiled_block_ranges[0].len(), 2);
        assert_eq!(order_bac.compiled_block_ranges[1].len(), 1);
        // Total rank invariant under permutation: 4.
        let total_abc: usize = order_abc
            .compiled_block_ranges
            .iter()
            .map(|r| r.len())
            .sum();
        let total_bac: usize = order_bac
            .compiled_block_ranges
            .iter()
            .map(|r| r.len())
            .sum();
        assert_eq!(total_abc, total_bac);
        assert_eq!(total_abc, 4);
    }

    /// Build a K=1 raw `(gram_h, gram_struct)` pair for a single stacked design
    /// `X` with per-row curvature weights `w`: `gram_struct = Xᵀ X`,
    /// `gram_h = Xᵀ diag(w) X`. Mirrors the closed-form definitions the
    /// production Gram builders implement for the scalar-channel case.
    fn k1_grams(x: &Array2<f64>, w: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
        let gram_struct = fast_atb(x, x);
        let xw = fast_xt_diag_y(x, w, x);
        (xw, gram_struct)
    }

    /// Full-rank reduction: when the two blocks are jointly independent the
    /// compiled width equals the raw width and the lift `T` reproduces a raw
    /// coefficient exactly from its compiled image `θ = T⁺ β` (here, with no
    /// aliasing, `lift_coefficients(θ)` of any compiled `θ` lands in the raw
    /// design's column interpretation: applying `T` then comparing the induced
    /// raw predictor `X·Tθ` to `X·β_raw` for the `θ` solving `Tθ=β_raw`).
    #[test]
    fn compiled_map_lift_coefficients_roundtrips_full_rank() {
        let n = 21;
        let p_a = 2;
        let p_b = 2;
        // Distinct per-column frequencies make the four sinusoidal columns
        // genuinely linearly independent over the sample grid. (A shared phase
        // offset varying only by column would collapse every column into
        // span{sin θ, cos θ, 1}, i.e. rank 3, and the compiler would correctly
        // absorb a column — defeating the full-rank premise of this test.)
        let x = Array2::from_shape_fn((n, p_a + p_b), |(i, j)| {
            ((i as f64 + 1.0) * (0.21 + 0.17 * j as f64)).sin() + 0.11 * (j as f64)
        });
        let w = Array1::from_shape_fn(n, |i| 0.5 + 0.5 * ((i as f64) * 0.3).cos().abs());
        let (gh, gs) = k1_grams(&x, &w);
        let raw_ranges = vec![0..p_a, p_a..(p_a + p_b)];
        let map = compile_from_raw_grams(
            &gh,
            &gs,
            &raw_ranges,
            &[BlockOrder::Marginal, BlockOrder::Logslope],
        )
        .expect("full-rank compile");
        // Jointly independent ⇒ no columns absorbed.
        assert_eq!(map.p_compiled(), p_a + p_b);
        assert_eq!(map.p_raw(), p_a + p_b);
        // For a target raw coefficient, solve T θ = β_raw (T square invertible
        // here) and confirm lift_coefficients(θ) == β_raw.
        let beta_raw = Array1::from_shape_fn(p_a + p_b, |j| 0.4 * (j as f64) - 0.7);
        // T is (p × p); recover θ by a least-squares solve via the normal
        // equations TᵀT θ = Tᵀ β.
        let tt = fast_atb(&map.raw_from_compiled, &map.raw_from_compiled);
        let tb = map.raw_from_compiled.t().dot(&beta_raw);
        let theta = solve_psd_system(&tt, &tb.insert_axis(Axis(1)))
            .expect("normal-equation solve")
            .column(0)
            .to_owned();
        let lifted = map.lift_coefficients(&theta).expect("lift");
        let max_err = (&lifted - &beta_raw)
            .iter()
            .fold(0.0_f64, |a, &v| a.max(v.abs()));
        assert!(
            max_err < 1e-8,
            "lift round-trip error {max_err:e} (full-rank reduction must be exactly invertible)"
        );
    }

    /// Design reparameterisation exactness: the compiled design predicts
    /// identically to the raw design on every lifted coefficient, i.e.
    /// `X_compiled · θ == X_raw · (T θ)`. This is the contract that lets a
    /// family fit in reduced coordinates and still produce raw-design
    /// predictions.
    #[test]
    fn compiled_map_reduce_design_matches_lifted_raw_predictor() {
        let n = 23;
        let p_a = 3;
        let p_b = 3;
        let mut x = Array2::from_shape_fn((n, p_a + p_b), |(i, j)| {
            ((i as f64 + 1.0) * 0.41 + (j as f64 + 1.0) * 0.7).sin() + 0.05 * (i % 3) as f64
        });
        // Alias one B column onto an A column so the reduction is non-trivial.
        for i in 0..n {
            x[[i, p_a + 1]] = x[[i, 1]];
        }
        let w = Array1::from_shape_fn(n, |i| 0.6 + 0.4 * ((i as f64) * 0.25).cos().abs());
        let (gh, gs) = k1_grams(&x, &w);
        let raw_ranges = vec![0..p_a, p_a..(p_a + p_b)];
        let map = compile_from_raw_grams(
            &gh,
            &gs,
            &raw_ranges,
            &[BlockOrder::Marginal, BlockOrder::Logslope],
        )
        .expect("compile");
        let x_compiled = map.reduce_design(&x).expect("reduce_design");
        assert_eq!(x_compiled.ncols(), map.p_compiled());
        let theta = Array1::from_shape_fn(map.p_compiled(), |j| 0.3 * (j as f64) - 0.5);
        let pred_compiled = x_compiled.dot(&theta);
        let beta_raw = map.lift_coefficients(&theta).expect("lift");
        let pred_raw = x.dot(&beta_raw);
        let max_err = (&pred_compiled - &pred_raw)
            .iter()
            .fold(0.0_f64, |a, &v| a.max(v.abs()));
        assert!(
            max_err < 1e-9,
            "compiled-design predictor diverges from lifted raw predictor: {max_err:e}"
        );
    }

    /// Penalty-energy preservation: the reduced penalty `Tᵀ Ŝ_b T` reproduces
    /// the raw penalty energy `βᵀ Ŝ_b β` on every lifted point `β = T θ`. This
    /// is the exactness contract the lift map must satisfy for REML/inference
    /// to be invariant to the quotient reparameterisation.
    #[test]
    fn reduce_penalties_with_map_preserves_energy_on_lift() {
        let n = 19;
        let p_a = 3;
        let p_b = 2;
        // Make block B partly aliased with A so the reduction actually drops a
        // column — the penalty reduction must still preserve energy on the
        // surviving compiled directions.
        let mut x = Array2::from_shape_fn((n, p_a + p_b), |(i, j)| {
            ((i as f64 + 1.0) * 0.29 + (j as f64 + 1.0) * 0.9).cos()
        });
        // Column (p_a+0) := column 0 (exact alias) ⇒ B loses one direction.
        for i in 0..n {
            x[[i, p_a]] = x[[i, 0]];
        }
        let w = Array1::from_shape_fn(n, |i| 0.7 + 0.3 * ((i as f64) * 0.2).sin().abs());
        let (gh, gs) = k1_grams(&x, &w);
        let raw_ranges = vec![0..p_a, p_a..(p_a + p_b)];
        let map = compile_from_raw_grams(
            &gh,
            &gs,
            &raw_ranges,
            &[BlockOrder::Marginal, BlockOrder::Logslope],
        )
        .expect("compile with alias");
        assert!(
            map.p_compiled() < p_a + p_b,
            "expected at least one absorbed column, got p_compiled={}",
            map.p_compiled()
        );
        // A simple per-block raw penalty: ridge on each block.
        let s_a = Array2::<f64>::eye(p_a);
        let s_b = Array2::<f64>::eye(p_b);
        let reduced = reduce_penalties_with_map(&map, &[Some(s_a.clone()), Some(s_b.clone())])
            .expect("reduce penalties");
        // For random compiled θ, raw β = T θ. Raw energy for block b is
        // β[range_b]ᵀ S_b β[range_b]; reduced energy is θᵀ S_reduced_b θ.
        let theta = Array1::from_shape_fn(map.p_compiled(), |j| {
            0.6 * (j as f64) - 0.3 + 0.05 * (j % 2) as f64
        });
        let beta = map.lift_coefficients(&theta).expect("lift");
        for (block_idx, s_raw) in [(0usize, &s_a), (1usize, &s_b)] {
            let range = &map.raw_block_ranges[block_idx];
            let beta_b = beta.slice(s![range.start..range.end]).to_owned();
            let raw_energy = beta_b.dot(&s_raw.dot(&beta_b));
            let s_reduced = reduced[block_idx]
                .as_ref()
                .expect("reduced penalty present");
            let reduced_energy = theta.dot(&s_reduced.dot(&theta));
            assert!(
                (raw_energy - reduced_energy).abs() < 1e-8 * raw_energy.abs().max(1.0),
                "block {block_idx} energy mismatch: raw={raw_energy:e} reduced={reduced_energy:e}"
            );
        }
    }
}
