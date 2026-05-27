//! Survival marginal-slope concrete impls for the family-agnostic
//! identifiability compiler (`crate::families::identifiability_compiler`).
//!
//! Phase 2 architecture: `docs/identifiability_compiler.md` §1, §2.
//!
//! Survival's row primary state is the 4-vector `u_i = (q0, q1, qd1, g)`,
//! so `K = 4`. The row Hessian is the 4×4 second-derivative block of the
//! per-row neg-log-likelihood kernel `row_primary_closed_form` at a pilot
//! `β`, PSD-clamped via eigendecomposition (negative eigenvalues projected
//! to zero) to handle pilot points far from the optimum.
//!
//! Each block exposes its row Jacobian as the contribution of `δβ_block`
//! to the row primary-state vector:
//!
//! - **TimeBlockOperator**: `(δq0, δq1, δqd1, 0)` from `design_entry`,
//!   `design_exit`, `design_derivative_exit` rows.
//! - **MarginalBlockOperator**: `(δq, δq, δqd_marginal, 0)` from the
//!   marginal design row (shared by q0 and q1; qd contribution zero unless
//!   timewiggle is active — captured by an explicit derivative row matrix).
//! - **LogslopeBlockOperator**: `(0, 0, 0, δg)` from the logslope design.
//! - **ScoreWarpBlockOperator**: `(δq, δq, δqd_warp, 0)` from the warp
//!   basis (shifts q at entry/exit; chain rule via dq0_seed/dt for qd1).
//! - **LinkDevBlockOperator**: `(δq, δq, δqd_link, 0)` from the link-dev
//!   basis on the rigid/pilot q-seed.
//!
//! Phase 4a delivery: trait impls + an input-builder helper. Phase 4b
//! threads these through SMGS's construction site and the migrated pilot
//! β; Phase 4c deletes the legacy
//! `install_compiled_flex_block_into_runtime` path.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use crate::families::custom_family::PenaltyMatrix;
use crate::families::identifiability_compiler::{
    AnchorRowEvaluator, BlockOrder, RowHessian, RowJacobianOperator,
};
use crate::linalg::faer_ndarray::{FaerEigh, fast_ab};
use crate::linalg::matrix::{
    CoefficientTransformOperator, DenseDesignMatrix, DesignMatrix, ResidualisedDesignOperator,
};
use faer::Side;

const K_SURVIVAL: usize = 4;

/// Per-row 4×4 row Hessian for the survival marginal-slope likelihood at a
/// pilot `β`. The pilot supplies the primary-state vector
/// `(q0_i, q1_i, qd1_i, g_i)` and the per-row sample weight + event
/// indicator + z + probit scale. The 4×4 block is evaluated via the
/// existing `row_primary_closed_form` kernel (which already returns the
/// full Hessian in `(q0, q1, qd1, g)` order) and PSD-clamped per row.
pub struct SurvivalRowHessian {
    /// PSD-projected per-row 4×4 Hessian, stored row-major as
    /// `(n × 4 × 4)`.
    h: Array3<f64>,
}

impl SurvivalRowHessian {
    /// Construct from explicit per-row pilot primary-state and the row
    /// data needed by `row_primary_closed_form`. Negative eigenvalues are
    /// projected to zero before storage so the matrix is PSD.
    pub fn from_pilot_primary_state(
        q0: &Array1<f64>,
        q1: &Array1<f64>,
        qd1: &Array1<f64>,
        g: &Array1<f64>,
        z: &Array1<f64>,
        weights: &Array1<f64>,
        event: &Array1<f64>,
        derivative_guard: f64,
        probit_scale: f64,
    ) -> Result<Self, String> {
        let n = q0.len();
        if [q1.len(), qd1.len(), g.len(), z.len(), weights.len(), event.len()]
            .iter()
            .any(|&l| l != n)
        {
            return Err(format!(
                "SurvivalRowHessian: length mismatch \
                 q0={n}, q1={}, qd1={}, g={}, z={}, weights={}, event={}",
                q1.len(),
                qd1.len(),
                g.len(),
                z.len(),
                weights.len(),
                event.len()
            ));
        }
        let mut h_full = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            let (_, _grad, hess) = crate::families::survival_marginal_slope::row_primary_for_compiler(
                q0[i],
                q1[i],
                qd1[i],
                g[i],
                z[i],
                weights[i],
                event[i],
                derivative_guard,
                probit_scale,
            )?;
            // PSD-clamp via eigendecomposition: project negative eigvals to 0.
            let mut h_i = Array2::<f64>::zeros((K_SURVIVAL, K_SURVIVAL));
            for a in 0..K_SURVIVAL {
                for b in 0..K_SURVIVAL {
                    h_i[[a, b]] = hess[a][b];
                }
            }
            let clamped = psd_clamp_4x4(&h_i);
            for a in 0..K_SURVIVAL {
                for b in 0..K_SURVIVAL {
                    h_full[[i, a, b]] = clamped[[a, b]];
                }
            }
        }
        Ok(Self { h: h_full })
    }

    /// Construct from an already-PSD per-row tensor. Used by callers that
    /// have computed the Hessian via a different route.
    pub fn from_full(h: Array3<f64>) -> Self {
        assert_eq!(h.shape()[1], K_SURVIVAL);
        assert_eq!(h.shape()[2], K_SURVIVAL);
        Self { h }
    }
}

impl RowHessian for SurvivalRowHessian {
    fn k(&self) -> usize {
        K_SURVIVAL
    }
    fn nrows(&self) -> usize {
        self.h.shape()[0]
    }
    fn fill_row(&self, row: usize, out: &mut [f64]) {
        assert_eq!(out.len(), K_SURVIVAL * K_SURVIVAL);
        for a in 0..K_SURVIVAL {
            for b in 0..K_SURVIVAL {
                out[a * K_SURVIVAL + b] = self.h[[row, a, b]];
            }
        }
    }
    fn evaluate_full(&self) -> Array3<f64> {
        self.h.clone()
    }
}

/// Project a 4×4 symmetric matrix onto the PSD cone: zero negative
/// eigenvalues. If the eigendecomposition fails (extremely defensive —
/// `row_primary_closed_form` already guarantees finite entries), return
/// the diagonal with negatives clamped.
fn psd_clamp_4x4(m: &Array2<f64>) -> Array2<f64> {
    let k = m.nrows();
    let (evals, evecs) = match m.eigh(Side::Lower) {
        Ok(pair) => pair,
        Err(_) => {
            let mut out = Array2::<f64>::zeros((k, k));
            for i in 0..k {
                out[[i, i]] = m[[i, i]].max(0.0);
            }
            return out;
        }
    };
    let mut out = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for l in 0..k {
                acc += evecs[[i, l]] * evals[l].max(0.0) * evecs[[j, l]];
            }
            out[[i, j]] = acc;
        }
    }
    out
}

/// Row Jacobian operator for the survival time block. Channels (q0, q1,
/// qd1) come from the three time designs; the g channel is zero.
pub struct TimeBlockOperator {
    dq0: Array2<f64>,
    dq1: Array2<f64>,
    dqd1: Array2<f64>,
}

impl TimeBlockOperator {
    pub fn new(dq0: Array2<f64>, dq1: Array2<f64>, dqd1: Array2<f64>) -> Self {
        assert_eq!(dq0.dim(), dq1.dim());
        assert_eq!(dq0.dim(), dqd1.dim());
        Self { dq0, dq1, dqd1 }
    }
}

impl RowJacobianOperator for TimeBlockOperator {
    fn k(&self) -> usize {
        K_SURVIVAL
    }
    fn ncols(&self) -> usize {
        self.dq0.ncols()
    }
    fn nrows(&self) -> usize {
        self.dq0.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), K_SURVIVAL);
        assert_eq!(delta_beta.len(), self.dq0.ncols());
        let mut acc = [0.0_f64; K_SURVIVAL];
        for (j, &b) in delta_beta.iter().enumerate() {
            acc[0] += self.dq0[[row, j]] * b;
            acc[1] += self.dq1[[row, j]] * b;
            acc[2] += self.dqd1[[row, j]] * b;
        }
        out.copy_from_slice(&acc);
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.dq0.nrows();
        let p = self.dq0.ncols();
        let mut out = Array3::<f64>::zeros((n, p, K_SURVIVAL));
        for i in 0..n {
            for j in 0..p {
                out[[i, j, 0]] = self.dq0[[i, j]];
                out[[i, j, 1]] = self.dq1[[i, j]];
                out[[i, j, 2]] = self.dqd1[[i, j]];
            }
        }
        out
    }
}

/// Row Jacobian operator for a block whose contribution flows into the
/// q-channels (q0 and q1 identically) and optionally the qd1 channel.
/// Covers the survival marginal, score-warp, and link-dev blocks (all
/// three share the structural property `δq0 = δq1 = basis·δβ`, `δg = 0`).
pub struct QChannelBlockOperator {
    dq: Array2<f64>,
    dqd1: Array2<f64>,
}

impl QChannelBlockOperator {
    pub fn new(dq: Array2<f64>, dqd1: Array2<f64>) -> Self {
        assert_eq!(dq.dim(), dqd1.dim());
        Self { dq, dqd1 }
    }
}

impl RowJacobianOperator for QChannelBlockOperator {
    fn k(&self) -> usize {
        K_SURVIVAL
    }
    fn ncols(&self) -> usize {
        self.dq.ncols()
    }
    fn nrows(&self) -> usize {
        self.dq.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), K_SURVIVAL);
        assert_eq!(delta_beta.len(), self.dq.ncols());
        let mut dq_acc = 0.0;
        let mut dqd_acc = 0.0;
        for (j, &b) in delta_beta.iter().enumerate() {
            dq_acc += self.dq[[row, j]] * b;
            dqd_acc += self.dqd1[[row, j]] * b;
        }
        out[0] = dq_acc;
        out[1] = dq_acc;
        out[2] = dqd_acc;
        out[3] = 0.0;
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.dq.nrows();
        let p = self.dq.ncols();
        let mut out = Array3::<f64>::zeros((n, p, K_SURVIVAL));
        for i in 0..n {
            for j in 0..p {
                let v = self.dq[[i, j]];
                out[[i, j, 0]] = v;
                out[[i, j, 1]] = v;
                out[[i, j, 2]] = self.dqd1[[i, j]];
            }
        }
        out
    }
}

/// Row Jacobian operator for the survival logslope block: contribution
/// lives entirely on the g channel.
pub struct LogslopeBlockOperator {
    dg: Array2<f64>,
}

impl LogslopeBlockOperator {
    pub fn new(dg: Array2<f64>) -> Self {
        Self { dg }
    }
}

impl RowJacobianOperator for LogslopeBlockOperator {
    fn k(&self) -> usize {
        K_SURVIVAL
    }
    fn ncols(&self) -> usize {
        self.dg.ncols()
    }
    fn nrows(&self) -> usize {
        self.dg.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), K_SURVIVAL);
        assert_eq!(delta_beta.len(), self.dg.ncols());
        let mut acc = 0.0;
        for (j, &b) in delta_beta.iter().enumerate() {
            acc += self.dg[[row, j]] * b;
        }
        out[0] = 0.0;
        out[1] = 0.0;
        out[2] = 0.0;
        out[3] = acc;
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.dg.nrows();
        let p = self.dg.ncols();
        let mut out = Array3::<f64>::zeros((n, p, K_SURVIVAL));
        for i in 0..n {
            for j in 0..p {
                out[[i, j, 3]] = self.dg[[i, j]];
            }
        }
        out
    }
}

/// Predict-time anchor row evaluator for a parametric survival block.
/// Returns the supplied design as-is (already materialised at the requested
/// rows). Phase 4b's predict-path migration introduces the variant that
/// recomputes from a constructor at predict rows.
pub struct ParametricAnchorEvaluator {
    design: Array2<f64>,
}

impl ParametricAnchorEvaluator {
    pub fn new(design: Array2<f64>) -> Self {
        Self { design }
    }
}

impl AnchorRowEvaluator for ParametricAnchorEvaluator {
    fn anchor_rows(&self, predict_arg: &Array1<f64>) -> Result<Array2<f64>, String> {
        if predict_arg.len() != self.design.nrows() {
            return Err(format!(
                "ParametricAnchorEvaluator: predict_arg length {} must match \
                 materialised design rows {}",
                predict_arg.len(),
                self.design.nrows()
            ));
        }
        Ok(self.design.clone())
    }
    fn ncols(&self) -> usize {
        self.design.ncols()
    }
}

/// Predict-time anchor row evaluator for a compiled survival flex block.
/// Composes `C(x)·V − A(x)·M` exactly as the Bernoulli analogue does.
pub struct CompiledFlexAnchorEvaluator {
    raw_basis: Arc<dyn Fn(&Array1<f64>) -> Result<Array2<f64>, String> + Send + Sync>,
    t_lw: Array2<f64>,
    anchor_correction: Option<Array2<f64>>,
    parent: Option<Arc<dyn AnchorRowEvaluator>>,
}

impl CompiledFlexAnchorEvaluator {
    pub fn new(
        raw_basis: Arc<dyn Fn(&Array1<f64>) -> Result<Array2<f64>, String> + Send + Sync>,
        t_lw: Array2<f64>,
        anchor_correction: Option<Array2<f64>>,
        parent: Option<Arc<dyn AnchorRowEvaluator>>,
    ) -> Self {
        Self {
            raw_basis,
            t_lw,
            anchor_correction,
            parent,
        }
    }
}

impl AnchorRowEvaluator for CompiledFlexAnchorEvaluator {
    fn anchor_rows(&self, predict_arg: &Array1<f64>) -> Result<Array2<f64>, String> {
        let raw = (self.raw_basis)(predict_arg)?;
        let rotated = fast_ab(&raw, &self.t_lw);
        match (&self.anchor_correction, &self.parent) {
            (Some(m), Some(parent)) => {
                let anchor = parent.anchor_rows(predict_arg)?;
                let correction = fast_ab(&anchor, m);
                Ok(&rotated - &correction)
            }
            (None, _) | (_, None) => Ok(rotated),
        }
    }
    fn ncols(&self) -> usize {
        self.t_lw.ncols()
    }
}

/// Inputs assembled for the survival fit driver to feed `compile()`. The
/// ordering follows `gauge_priority` descending (time=200 → marginal=150 →
/// logslope=120 → score_warp=80 → link_dev=60).
pub struct SurvivalCompilerInputs {
    pub operators: Vec<Arc<dyn RowJacobianOperator>>,
    pub ordering: Vec<BlockOrder>,
}

/// Per-block V reparameterisation matrices for the three parametric
/// survival blocks emitted by [`compile_survival_parametric_designs`].
/// Each `v_*` is a `(p_block_raw × p_block_kept)` selection-or-rotation
/// matrix that maps a `β_kept` coefficient vector to its `β_raw`
/// equivalent: `β_raw = V · β_kept`. The construction site applies these
/// to the raw block designs (`design_raw · V → design_compiled`) and
/// to the penalties (`Vᵀ S V`) before building `ParameterBlockSpec`s
/// and passing the compiled designs into `make_family`.
///
/// Phase-4b architecture: this is the seam where the family-agnostic
/// row-Jacobian compiler hands control back to the family-specific
/// construction site. Each `v_*` width equals the corresponding
/// `CompiledBlocks::blocks[i].t_lw.ncols()` — i.e., the kept-direction
/// count after sqrt-H-metric residualisation and post-walk RRQR
/// trailing-pivot drop.
pub struct SurvivalParametricCompiled {
    pub v_time: Array2<f64>,
    pub v_marginal: Array2<f64>,
    pub v_logslope: Array2<f64>,
    /// Per-block dropped raw-column count, indexed
    /// `(time_dropped, marginal_dropped, logslope_dropped)`. Equal to
    /// `(p_raw − v.ncols())` for each block. Useful for logging the
    /// gauge-attribution summary at the construction site.
    pub drops_by_block: (usize, usize, usize),
}

/// Survival parametric block designs and penalties after applying the
/// per-block V reparameterisation matrices from
/// [`compile_survival_parametric_designs`]. Each design is wrapped via
/// [`CoefficientTransformOperator`] so the operator interface is
/// preserved (sparse / lazy inner designs stay sparse / lazy; the V
/// multiplication is applied lazily per row chunk with an Arc-cached
/// dense materialisation when affordable).
///
/// **Time block**: three designs (entry, exit, derivative_exit) share a
/// single β, so they each get the same `V_time` applied. Their
/// penalties are pulled back jointly because the time penalty matrices
/// are over the shared β coordinate.
///
/// **Marginal / logslope**: one design + their respective penalty list,
/// each independently V-transformed and Vᵀ-S-V-pulled-back.
///
/// The construction site replaces the raw `marginal_design`,
/// `logslope_design`, and time-block triplet with these compiled
/// variants, and uses the pulled-back penalty matrices in the
/// `ParameterBlockSpec` list. The family's captured
/// `marginal_design` / `logslope_design` / time triplet then carry the
/// compiled widths too — so `evaluate_blockwise_exact_newton`'s
/// `syr_row_into_view` / `row_outer_into_view` assertions remain
/// width-consistent without further family-level changes.
pub struct CompiledSurvivalDesigns {
    pub time_design_entry: DesignMatrix,
    pub time_design_exit: DesignMatrix,
    pub time_design_derivative_exit: DesignMatrix,
    pub marginal_design: DesignMatrix,
    pub logslope_design: DesignMatrix,
    pub time_penalties: Vec<PenaltyMatrix>,
    pub marginal_penalties: Vec<PenaltyMatrix>,
    pub logslope_penalties: Vec<PenaltyMatrix>,
}

/// Apply `compiled.v_*` to the raw survival parametric designs and pull
/// back the per-block penalties as `Vᵀ S V`. Returns
/// [`CompiledSurvivalDesigns`] ready to thread through
/// `make_family` / `build_blocks` at the SMGS construction site.
///
/// Sparse designs are wrapped through `CoefficientTransformOperator`,
/// which composes lazily by default and materialises the `(n × p_kept)`
/// dense block on first hot use (gated by
/// `CoefficientTransformOperator::MATERIALIZE_MAX_BYTES = 1 GiB`).
/// For biobank-scale survival shapes (`n ≈ 320 k`, `p_kept ≤ 50`) this
/// is ≤ 130 MiB — well within budget and reused across PIRLS / outer
/// iterations.
///
/// The penalty pullback `Vᵀ S V` is exact for selection-T (V is a
/// column selector, so Vᵀ S V is just the slice of S to the kept
/// rows / cols) and for rotation-V (V is a general orthogonal-
/// complement basis from the compiler's eigendecomposition).
pub fn apply_survival_parametric_compile_to_designs(
    compiled: &SurvivalParametricCompiled,
    time_design_entry: DesignMatrix,
    time_design_exit: DesignMatrix,
    time_design_derivative_exit: DesignMatrix,
    marginal_design: DesignMatrix,
    logslope_design: DesignMatrix,
    time_penalties: &[PenaltyMatrix],
    marginal_penalties: &[PenaltyMatrix],
    logslope_penalties: &[PenaltyMatrix],
) -> Result<CompiledSurvivalDesigns, String> {
    Ok(CompiledSurvivalDesigns {
        time_design_entry: wrap_design_with_transform(
            time_design_entry,
            &compiled.v_time,
            "survival time block design_entry",
        )?,
        time_design_exit: wrap_design_with_transform(
            time_design_exit,
            &compiled.v_time,
            "survival time block design_exit",
        )?,
        time_design_derivative_exit: wrap_design_with_transform(
            time_design_derivative_exit,
            &compiled.v_time,
            "survival time block design_derivative_exit",
        )?,
        marginal_design: wrap_design_with_transform(
            marginal_design,
            &compiled.v_marginal,
            "survival marginal block design",
        )?,
        logslope_design: wrap_design_with_transform(
            logslope_design,
            &compiled.v_logslope,
            "survival logslope block design",
        )?,
        time_penalties: pull_back_penalties(time_penalties, &compiled.v_time),
        marginal_penalties: pull_back_penalties(marginal_penalties, &compiled.v_marginal),
        logslope_penalties: pull_back_penalties(logslope_penalties, &compiled.v_logslope),
    })
}

fn wrap_design_with_transform(
    raw: DesignMatrix,
    v: &Array2<f64>,
    context: &str,
) -> Result<DesignMatrix, String> {
    if raw.ncols() != v.nrows() {
        return Err(format!(
            "{context}: raw design has {} cols but V has {} rows (V is {}×{})",
            raw.ncols(),
            v.nrows(),
            v.nrows(),
            v.ncols(),
        ));
    }
    let inner_dense = match raw {
        DesignMatrix::Dense(d) => d,
        DesignMatrix::Sparse(_) => {
            let dense = raw
                .try_to_dense_by_chunks(&format!("{context} sparse→dense for V apply"))
                .map_err(|reason| format!("{context}: densify failed: {reason}"))?;
            DenseDesignMatrix::from(dense)
        }
    };
    let op = CoefficientTransformOperator::new(inner_dense, v.clone())
        .map_err(|reason| format!("{context}: CoefficientTransformOperator::new: {reason}"))?;
    Ok(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op))))
}

fn pull_back_penalties(penalties: &[PenaltyMatrix], v: &Array2<f64>) -> Vec<PenaltyMatrix> {
    penalties
        .iter()
        .map(|p| {
            let label = p.precision_label().map(|s| s.to_string());
            let s_dense = p.as_dense_cow();
            // Vᵀ S V. With V as a selection matrix this collapses to
            // S[kept, kept]; with V as a rotation (general orthogonal
            // complement) this is the full pullback. Either way the
            // result is (p_kept × p_kept) symmetric (modulo numerical
            // noise — symmetrise explicitly).
            let s_view = s_dense.view();
            let s_v = fast_ab(&s_view.to_owned(), v);
            let vt_s_v = fast_ab(&v.t().to_owned(), &s_v);
            let mut sym = Array2::<f64>::zeros(vt_s_v.dim());
            for i in 0..sym.nrows() {
                for j in 0..sym.ncols() {
                    sym[[i, j]] = 0.5 * (vt_s_v[[i, j]] + vt_s_v[[j, i]]);
                }
            }
            let base = PenaltyMatrix::Dense(sym);
            match label {
                Some(lbl) => base.with_precision_label(lbl),
                None => base,
            }
        })
        .collect()
}

/// Per-term V reparameterisation matrices for the three parametric
/// survival blocks. Each block's full V is the block-diagonal assembly
/// of its per-term V's (one entry per element of the input
/// `*_partition`). Preserves per-term penalty structure: applying
/// `V_b = block_diag(V_term1, ..., V_termM)` to a per-term BlockwisePenalty
/// pulls each penalty back only via its OWN term's V, so what was a
/// per-term λ tunable in REML stays per-term tunable.
pub struct SurvivalParametricCompiledPerTerm {
    pub v_time_per_term: Vec<Array2<f64>>,
    pub v_marginal_per_term: Vec<Array2<f64>>,
    pub v_logslope_per_term: Vec<Array2<f64>>,
    /// Per-term residualised reparam `R_b = M_b · V_b` from the
    /// identifiability compiler, in the same global compile order
    /// (time terms, then marginal terms, then logslope terms). `None`
    /// for the very first compiled block (no anchor). Used by the
    /// V+M-exact apply path to emit residualised rows
    /// `C_b·V_b − A_{<b}·R_b` and to assemble the full triangular T.
    pub r_lw_per_term: Vec<Option<Array2<f64>>>,
    /// Per-block drops (raw_cols − sum(kept_cols across terms)).
    pub drops_by_block: (usize, usize, usize),
}

impl SurvivalParametricCompiledPerTerm {
    /// Block-diagonal V for the time block.
    pub fn v_time_block_diag(&self) -> Array2<f64> {
        block_diag_from(&self.v_time_per_term)
    }
    /// Block-diagonal V for the marginal block.
    pub fn v_marginal_block_diag(&self) -> Array2<f64> {
        block_diag_from(&self.v_marginal_per_term)
    }
    /// Block-diagonal V for the logslope block.
    pub fn v_logslope_block_diag(&self) -> Array2<f64> {
        block_diag_from(&self.v_logslope_per_term)
    }
}

/// Stack a list of per-term V matrices into a single block-diagonal V.
/// Output has rows = sum of raw widths, cols = sum of kept widths.
fn block_diag_from(v_per_term: &[Array2<f64>]) -> Array2<f64> {
    let total_rows: usize = v_per_term.iter().map(|v| v.nrows()).sum();
    let total_cols: usize = v_per_term.iter().map(|v| v.ncols()).sum();
    let mut out = Array2::<f64>::zeros((total_rows, total_cols));
    let mut row_off = 0usize;
    let mut col_off = 0usize;
    for v in v_per_term {
        let r = v.nrows();
        let c = v.ncols();
        if r > 0 && c > 0 {
            out.slice_mut(ndarray::s![row_off..row_off + r, col_off..col_off + c])
                .assign(v);
        }
        row_off += r;
        col_off += c;
    }
    out
}

/// Per-term-aware compile: residualise each block's TERMS individually
/// in priority order so the emitted V is block-diagonal on term
/// boundaries. This preserves the per-term penalty structure that
/// REML's per-λ accounting depends on.
///
/// Each `*_partition` is a list of disjoint contiguous column ranges
/// covering `[0..p_block)`. For the marginal/logslope blocks the
/// natural source is the union of `BlockwisePenalty::col_range` values
/// (one per smoothness penalty / term) plus the complement
/// (unpenalised parametric columns).
///
/// Order of residualisation: time terms first (in their partition
/// order), then marginal terms, then logslope terms. Within each
/// block, terms are residualised against ALL prior anchor columns
/// (terms from earlier blocks + earlier terms within this block).
/// Aliased directions land in the lowest-priority block that contains
/// them, in the natural term order within that block — matching the
/// gauge-priority ownership contract.
pub fn compile_survival_parametric_designs_per_term(
    time_dq0: Array2<f64>,
    time_dq1: Array2<f64>,
    time_dqd1: Array2<f64>,
    time_partition: &[std::ops::Range<usize>],
    marginal_dq: Array2<f64>,
    marginal_dqd1: Array2<f64>,
    marginal_partition: &[std::ops::Range<usize>],
    logslope_dg: Array2<f64>,
    logslope_partition: &[std::ops::Range<usize>],
    row_hess: &dyn RowHessian,
) -> Result<SurvivalParametricCompiledPerTerm, String> {
    use crate::families::identifiability_compiler::compile;

    let p_time = time_dq0.ncols();
    let p_marg = marginal_dq.ncols();
    let p_log = logslope_dg.ncols();
    validate_partition(time_partition, p_time, "time")?;
    validate_partition(marginal_partition, p_marg, "marginal")?;
    validate_partition(logslope_partition, p_log, "logslope")?;

    // Build per-term operators. Each term gets its own RowJacobianOperator
    // restricted to its column slice; the operator type matches the
    // block's K-channel signature (Time, QChannel, Logslope).
    let mut operators: Vec<Arc<dyn RowJacobianOperator>> = Vec::new();
    let mut ordering: Vec<BlockOrder> = Vec::new();
    for range in time_partition {
        let dq0 = time_dq0.slice(ndarray::s![.., range.clone()]).to_owned();
        let dq1 = time_dq1.slice(ndarray::s![.., range.clone()]).to_owned();
        let dqd1 = time_dqd1.slice(ndarray::s![.., range.clone()]).to_owned();
        operators.push(Arc::new(TimeBlockOperator::new(dq0, dq1, dqd1)));
        ordering.push(BlockOrder::Time);
    }
    for range in marginal_partition {
        let dq = marginal_dq.slice(ndarray::s![.., range.clone()]).to_owned();
        let dqd1 = marginal_dqd1.slice(ndarray::s![.., range.clone()]).to_owned();
        operators.push(Arc::new(QChannelBlockOperator::new(dq, dqd1)));
        ordering.push(BlockOrder::Marginal);
    }
    for range in logslope_partition {
        let dg = logslope_dg.slice(ndarray::s![.., range.clone()]).to_owned();
        operators.push(Arc::new(LogslopeBlockOperator::new(dg)));
        ordering.push(BlockOrder::Logslope);
    }

    let compiled = compile(&operators, row_hess, &ordering)
        .map_err(|e| format!("identifiability_compiler::compile (per-term) failed: {e}"))?;
    let blocks = compiled.blocks;
    let n_time = time_partition.len();
    let n_marg = marginal_partition.len();
    let n_log = logslope_partition.len();
    if blocks.len() != n_time + n_marg + n_log {
        return Err(format!(
            "per-term compile: expected {} compiled blocks (time={}, marg={}, log={}), got {}",
            n_time + n_marg + n_log,
            n_time,
            n_marg,
            n_log,
            blocks.len(),
        ));
    }
    let mut iter = blocks.into_iter();
    let mut v_time_per_term: Vec<Array2<f64>> = Vec::with_capacity(n_time);
    let mut r_time_per_term: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_time);
    for _ in 0..n_time {
        let blk = iter.next().unwrap();
        v_time_per_term.push(blk.t_lw);
        r_time_per_term.push(blk.r_lw);
    }
    let mut v_marginal_per_term: Vec<Array2<f64>> = Vec::with_capacity(n_marg);
    let mut r_marginal_per_term: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_marg);
    for _ in 0..n_marg {
        let blk = iter.next().unwrap();
        v_marginal_per_term.push(blk.t_lw);
        r_marginal_per_term.push(blk.r_lw);
    }
    let mut v_logslope_per_term: Vec<Array2<f64>> = Vec::with_capacity(n_log);
    let mut r_logslope_per_term: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_log);
    for _ in 0..n_log {
        let blk = iter.next().unwrap();
        v_logslope_per_term.push(blk.t_lw);
        r_logslope_per_term.push(blk.r_lw);
    }
    let mut r_lw_per_term: Vec<Option<Array2<f64>>> =
        Vec::with_capacity(n_time + n_marg + n_log);
    r_lw_per_term.extend(r_time_per_term);
    r_lw_per_term.extend(r_marginal_per_term);
    r_lw_per_term.extend(r_logslope_per_term);
    let drops_time: usize = time_partition
        .iter()
        .zip(v_time_per_term.iter())
        .map(|(r, v)| r.len().saturating_sub(v.ncols()))
        .sum();
    let drops_marg: usize = marginal_partition
        .iter()
        .zip(v_marginal_per_term.iter())
        .map(|(r, v)| r.len().saturating_sub(v.ncols()))
        .sum();
    let drops_log: usize = logslope_partition
        .iter()
        .zip(v_logslope_per_term.iter())
        .map(|(r, v)| r.len().saturating_sub(v.ncols()))
        .sum();
    Ok(SurvivalParametricCompiledPerTerm {
        v_time_per_term,
        v_marginal_per_term,
        v_logslope_per_term,
        r_lw_per_term,
        drops_by_block: (drops_time, drops_marg, drops_log),
    })
}

fn validate_partition(
    partition: &[std::ops::Range<usize>],
    p_block: usize,
    label: &str,
) -> Result<(), String> {
    if partition.is_empty() {
        if p_block == 0 {
            return Ok(());
        }
        return Err(format!(
            "{label} partition empty but block has p={p_block} columns"
        ));
    }
    if partition[0].start != 0 {
        return Err(format!(
            "{label} partition must start at 0, got start={}",
            partition[0].start
        ));
    }
    if partition.last().unwrap().end != p_block {
        return Err(format!(
            "{label} partition must cover [0, {p_block}); last range ends at {}",
            partition.last().unwrap().end
        ));
    }
    for w in partition.windows(2) {
        if w[0].end != w[1].start {
            return Err(format!(
                "{label} partition has gap/overlap between [{}..{}) and [{}..{})",
                w[0].start, w[0].end, w[1].start, w[1].end
            ));
        }
        if w[0].is_empty() {
            return Err(format!(
                "{label} partition has empty range [{}..{})",
                w[0].start, w[0].end
            ));
        }
    }
    if partition.last().unwrap().is_empty() {
        return Err(format!(
            "{label} partition's final range is empty",
        ));
    }
    Ok(())
}

/// Derive a disjoint contiguous partition of `[0..p_block)` from a
/// list of BlockwisePenalty col_ranges. Distinct penalty ranges define
/// term boundaries; gaps between them (unpenalised columns) become
/// their own single-column partitions. Multiple penalties with the
/// SAME col_range (e.g. tensor anisotropy axes) coalesce to one term.
pub fn extract_term_partition_from_penalty_ranges(
    p_block: usize,
    penalty_ranges: &[std::ops::Range<usize>],
) -> Vec<std::ops::Range<usize>> {
    use std::collections::BTreeSet;
    let mut starts: BTreeSet<usize> = BTreeSet::new();
    starts.insert(0);
    starts.insert(p_block);
    for r in penalty_ranges {
        starts.insert(r.start.min(p_block));
        starts.insert(r.end.min(p_block));
    }
    let v: Vec<usize> = starts.into_iter().collect();
    v.windows(2).filter_map(|w| if w[0] < w[1] { Some(w[0]..w[1]) } else { None }).collect()
}

/// Pull back a BlockwisePenalty under a block-diagonal V whose per-term
/// V's are listed in `v_per_term`, partitioned by `raw_partition`.
/// The penalty's `col_range` must lie within a single partition entry
/// (i.e. the penalty belongs to one term). Returns a new BlockwisePenalty
/// with `local = V_term^T · local · V_term` and `col_range` shifted to
/// compiled coordinates.
pub fn pull_back_blockwise_penalty_per_term(
    pen: &crate::terms::smooth::BlockwisePenalty,
    raw_partition: &[std::ops::Range<usize>],
    v_per_term: &[Array2<f64>],
) -> Result<crate::terms::smooth::BlockwisePenalty, String> {
    use crate::terms::smooth::BlockwisePenalty;
    if raw_partition.len() != v_per_term.len() {
        return Err(format!(
            "pull_back_blockwise_penalty_per_term: partition len {} != v_per_term len {}",
            raw_partition.len(),
            v_per_term.len()
        ));
    }
    // Find which term the penalty belongs to.
    let mut term_idx = None;
    for (idx, range) in raw_partition.iter().enumerate() {
        if pen.col_range.start >= range.start && pen.col_range.end <= range.end {
            term_idx = Some(idx);
            break;
        }
    }
    let term_idx = term_idx.ok_or_else(|| {
        format!(
            "pull_back_blockwise_penalty_per_term: penalty col_range {}..{} does not fit \
             within any term partition (partition entries: {:?})",
            pen.col_range.start, pen.col_range.end, raw_partition
        )
    })?;
    let v_term = &v_per_term[term_idx];
    let term_range = &raw_partition[term_idx];
    // The penalty's local matrix is sized `pen.col_range.len()` and
    // covers a *sub-region* of the term (some smooths emit a single
    // penalty over the whole term; tensor smooths can emit multiple
    // penalties at the same col_range; anisotropic constructions can
    // emit a penalty over a sub-range of the term). We need a V slice
    // that lines up with pen.col_range relative to term_range.start.
    let local_off_start = pen.col_range.start - term_range.start;
    // Per-term V is (term_p × term_p_kept). Slicing the rows of V along
    // pen.col_range's offset within the term gives the sub-block to use
    // for pullback. Equivalently: build an embedded V_full at term width
    // and multiply V_full^T · embed(local) · V_full.
    //
    // For correctness when the penalty doesn't cover the full term, we
    // embed the local matrix to the full term width, then pull back via
    // the whole V_term, then keep all kept-cols of the result. This
    // matches the canonical Vᵀ embed(S) V on the term subspace.
    let term_p = term_range.len();
    let mut embedded = Array2::<f64>::zeros((term_p, term_p));
    for i in 0..pen.col_range.len() {
        for j in 0..pen.col_range.len() {
            embedded[[local_off_start + i, local_off_start + j]] = pen.local[[i, j]];
        }
    }
    // Vᵀ · embedded · V
    let temp = embedded.dot(v_term);
    let pulled = v_term.t().dot(&temp);
    // Symmetrise to wash out floating noise.
    let r = pulled.nrows();
    let mut sym = Array2::<f64>::zeros((r, r));
    for i in 0..r {
        for j in 0..r {
            sym[[i, j]] = 0.5 * (pulled[[i, j]] + pulled[[j, i]]);
        }
    }
    // Compiled term col_range: offsets in the compiled (block-diagonal)
    // coordinate are the cumulative sum of v_per_term[k].ncols() up to
    // term_idx.
    let mut compiled_start = 0usize;
    for v in v_per_term.iter().take(term_idx) {
        compiled_start += v.ncols();
    }
    let compiled_end = compiled_start + v_term.ncols();
    // The pen.local previously covered a sub-region of the term; after
    // pullback we drop that sub-region distinction and the new penalty
    // covers the WHOLE compiled term (since the embedded-then-pulled-back
    // form is over all term-kept cols).
    Ok(BlockwisePenalty::new(compiled_start..compiled_end, sym))
}

/// Assemble the block-upper-triangular reparameterisation T that maps a
/// compiled coefficient vector θ (concatenation of per-term θ_b in
/// `kept_p_b` widths) to the raw coefficient vector γ (concatenation of
/// per-term γ_a in `raw_p_a` widths) under the V+M-exact residualised
/// emitted design `L_b = C_b V_b − A_{<b} R_b`.
///
/// On the diagonal: block (b, b) = `V_b` of shape `raw_p_b × kept_p_b`.
/// In the strictly upper triangle: block (a, b) for `a < b` =
/// `−R_{a→b}` of shape `raw_p_a × kept_p_b`. Each `r_per_term[b]` is
/// `Some(R_b)` where the rows of `R_b` are the vertical stack of the
/// `R_{a→b}` over `a = 0..b` (so `R_b.nrows() == Σ_{a<b} raw_p_a`).
/// `r_per_term[0]` should be `None` (or a zero-row matrix) since no
/// earlier blocks exist.
///
/// Returns T of shape `(Σ raw_p_b) × (Σ kept_p_b)`.
pub fn build_full_t_matrix(
    v_per_term: &[Array2<f64>],
    r_per_term: &[Option<Array2<f64>>],
) -> Array2<f64> {
    assert_eq!(
        v_per_term.len(),
        r_per_term.len(),
        "build_full_t_matrix: v_per_term len {} != r_per_term len {}",
        v_per_term.len(),
        r_per_term.len(),
    );
    let raw_widths: Vec<usize> = v_per_term.iter().map(|v| v.nrows()).collect();
    let kept_widths: Vec<usize> = v_per_term.iter().map(|v| v.ncols()).collect();
    let row_offsets: Vec<usize> = {
        let mut o = Vec::with_capacity(raw_widths.len() + 1);
        o.push(0);
        for w in &raw_widths {
            o.push(o.last().copied().unwrap_or(0) + w);
        }
        o
    };
    let col_offsets: Vec<usize> = {
        let mut o = Vec::with_capacity(kept_widths.len() + 1);
        o.push(0);
        for w in &kept_widths {
            o.push(o.last().copied().unwrap_or(0) + w);
        }
        o
    };
    let total_rows = row_offsets.last().copied().unwrap_or(0);
    let total_cols = col_offsets.last().copied().unwrap_or(0);
    let mut t = Array2::<f64>::zeros((total_rows, total_cols));
    // Diagonal: place V_b on (b, b).
    for (b, v) in v_per_term.iter().enumerate() {
        let r = v.nrows();
        let c = v.ncols();
        if r > 0 && c > 0 {
            t.slice_mut(ndarray::s![
                row_offsets[b]..row_offsets[b] + r,
                col_offsets[b]..col_offsets[b] + c
            ])
            .assign(v);
        }
    }
    // Upper triangle: for each b ≥ 1, place −R_{a→b} at (a, b) for a < b.
    // `r_per_term[b]` stacks the R_{a→b} blocks row-wise in order
    // a = 0, 1, …, b-1; each block has `raw_widths[a]` rows and
    // `kept_widths[b]` cols.
    for b in 1..v_per_term.len() {
        let Some(r_stack) = r_per_term[b].as_ref() else {
            continue;
        };
        let kept_b = kept_widths[b];
        assert_eq!(
            r_stack.ncols(),
            kept_b,
            "build_full_t_matrix: r_per_term[{b}] has {} cols, expected {}",
            r_stack.ncols(),
            kept_b,
        );
        let expected_rows: usize = raw_widths.iter().take(b).sum();
        assert_eq!(
            r_stack.nrows(),
            expected_rows,
            "build_full_t_matrix: r_per_term[{b}] has {} rows, expected {} (sum of raw_widths[0..{}])",
            r_stack.nrows(),
            expected_rows,
            b,
        );
        let mut local_row = 0usize;
        for a in 0..b {
            let r_a = raw_widths[a];
            if r_a == 0 || kept_b == 0 {
                local_row += r_a;
                continue;
            }
            let block = r_stack.slice(ndarray::s![local_row..local_row + r_a, ..]);
            let mut dst = t.slice_mut(ndarray::s![
                row_offsets[a]..row_offsets[a] + r_a,
                col_offsets[b]..col_offsets[b] + kept_b
            ]);
            for i in 0..r_a {
                for j in 0..kept_b {
                    dst[[i, j]] = -block[[i, j]];
                }
            }
            local_row += r_a;
        }
    }
    t
}

/// Pull a single raw per-term penalty back through the full-width
/// reparameterisation T. The penalty's `local` is `block_p × block_p`
/// where `block_p == pen.col_range.len()`; `anchor_offset` is the start
/// (in the joint raw coordinate) of the term whose partition contains
/// `pen.col_range`, so the penalty is embedded into the joint raw
/// matrix at rows/cols `anchor_offset + pen.col_range`. The returned
/// matrix is the full-width compiled penalty `T^T · embed(S) · T`,
/// stored as `PenaltyMatrix::Dense`.
///
/// Unlike `pull_back_blockwise_penalty_per_term`, the result is NOT a
/// `BlockwisePenalty`: residualisation off-diagonal couples θ_b through
/// every earlier block's θ_a, so the pulled-back penalty is generally
/// dense across the full compiled width.
pub fn pull_back_penalty_through_t(
    pen: &crate::terms::smooth::BlockwisePenalty,
    anchor_offset: usize,
    t: &Array2<f64>,
) -> PenaltyMatrix {
    let raw_total = t.nrows();
    let compiled_total = t.ncols();
    let block_p = pen.col_range.len();
    let embed_start = anchor_offset + pen.col_range.start;
    let embed_end = embed_start + block_p;
    assert!(
        embed_end <= raw_total,
        "pull_back_penalty_through_t: embed range {}..{} exceeds raw total {}",
        embed_start, embed_end, raw_total,
    );
    let mut embedded = Array2::<f64>::zeros((raw_total, raw_total));
    if block_p > 0 {
        let mut dst = embedded.slice_mut(ndarray::s![
            embed_start..embed_end,
            embed_start..embed_end
        ]);
        for i in 0..block_p {
            for j in 0..block_p {
                dst[[i, j]] = pen.local[[i, j]];
            }
        }
    }
    let temp = embedded.dot(t);
    let pulled = t.t().dot(&temp);
    let mut sym = Array2::<f64>::zeros((compiled_total, compiled_total));
    for i in 0..compiled_total {
        for j in 0..compiled_total {
            sym[[i, j]] = 0.5 * (pulled[[i, j]] + pulled[[j, i]]);
        }
    }
    PenaltyMatrix::Dense(sym)
}

/// Per-term-aware compiled designs + penalties + block-diagonal V
/// matrices for the result-time β lift. The construction site swaps
/// raw designs/penalties for these compiled versions before building
/// `ParameterBlockSpec`s; at fit result the per-block β is lifted via
/// `v_time · β_time_compiled` (and similarly for marginal/logslope) to
/// produce raw-width β that predict-time consumes unchanged.
pub struct CompiledSurvivalDesignsPerTerm {
    pub time_design_entry: DesignMatrix,
    pub time_design_exit: DesignMatrix,
    pub time_design_derivative_exit: DesignMatrix,
    pub marginal_design: DesignMatrix,
    pub logslope_design: DesignMatrix,
    pub time_penalties: Vec<crate::terms::smooth::BlockwisePenalty>,
    pub marginal_penalties: Vec<crate::terms::smooth::BlockwisePenalty>,
    pub logslope_penalties: Vec<crate::terms::smooth::BlockwisePenalty>,
    /// Block-diagonal V for each block: rows = raw block width,
    /// cols = compiled block width. Used by `lift_smgs_block_betas_to_raw`
    /// at fit result to map compiled β back to raw β before serialise.
    pub v_time: Array2<f64>,
    pub v_marginal: Array2<f64>,
    pub v_logslope: Array2<f64>,
}

/// Per-term apply: produce compiled designs + per-term-pulled-back
/// penalties + the block-diagonal V matrices needed for the result-
/// time lift.
#[allow(clippy::too_many_arguments)]
pub fn apply_per_term_survival_parametric_compile_to_designs(
    compiled: &SurvivalParametricCompiledPerTerm,
    time_partition: &[std::ops::Range<usize>],
    marginal_partition: &[std::ops::Range<usize>],
    logslope_partition: &[std::ops::Range<usize>],
    time_design_entry: DesignMatrix,
    time_design_exit: DesignMatrix,
    time_design_derivative_exit: DesignMatrix,
    marginal_design: DesignMatrix,
    logslope_design: DesignMatrix,
    time_penalties: &[crate::terms::smooth::BlockwisePenalty],
    marginal_penalties: &[crate::terms::smooth::BlockwisePenalty],
    logslope_penalties: &[crate::terms::smooth::BlockwisePenalty],
) -> Result<CompiledSurvivalDesignsPerTerm, String> {
    let v_time = compiled.v_time_block_diag();
    let v_marginal = compiled.v_marginal_block_diag();
    let v_logslope = compiled.v_logslope_block_diag();
    let pull_set = |pens: &[crate::terms::smooth::BlockwisePenalty],
                    partition: &[std::ops::Range<usize>],
                    v_per_term: &[Array2<f64>]|
     -> Result<Vec<crate::terms::smooth::BlockwisePenalty>, String> {
        pens.iter()
            .map(|p| pull_back_blockwise_penalty_per_term(p, partition, v_per_term))
            .collect()
    };
    Ok(CompiledSurvivalDesignsPerTerm {
        time_design_entry: wrap_design_with_transform(
            time_design_entry,
            &v_time,
            "smgs per-term apply: time entry",
        )?,
        time_design_exit: wrap_design_with_transform(
            time_design_exit,
            &v_time,
            "smgs per-term apply: time exit",
        )?,
        time_design_derivative_exit: wrap_design_with_transform(
            time_design_derivative_exit,
            &v_time,
            "smgs per-term apply: time derivative_exit",
        )?,
        marginal_design: wrap_design_with_transform(
            marginal_design,
            &v_marginal,
            "smgs per-term apply: marginal",
        )?,
        logslope_design: wrap_design_with_transform(
            logslope_design,
            &v_logslope,
            "smgs per-term apply: logslope",
        )?,
        time_penalties: pull_set(time_penalties, time_partition, &compiled.v_time_per_term)?,
        marginal_penalties: pull_set(
            marginal_penalties,
            marginal_partition,
            &compiled.v_marginal_per_term,
        )?,
        logslope_penalties: pull_set(
            logslope_penalties,
            logslope_partition,
            &compiled.v_logslope_per_term,
        )?,
        v_time,
        v_marginal,
        v_logslope,
    })
}

/// Per-block V matrices for the SMGS result-time β lift. The block
/// order is: index 0 = time, 1 = marginal, 2 = logslope, 3+ = flex
/// (score_warp_dev, link_dev) which are NOT compiled by this path —
/// they go through `identifiability_compiler::compile` independently
/// at construction time via `install_compiled_flex_block_into_runtime`
/// and need no result-time lift.
#[derive(Debug, Clone)]
pub struct SmgsLiftPerBlockV {
    pub v_per_block: Vec<Array2<f64>>,
}

impl SmgsLiftPerBlockV {
    /// Lift `block_betas[i]` from compiled width to raw via
    /// `v_per_block[i] · block_betas[i]` when `i < v_per_block.len()`
    /// and the V's shape matches; otherwise pass through unchanged
    /// (so flex blocks and any other consumers stay untouched).
    pub fn lift_block_betas(&self, block_betas: &mut [Array1<f64>]) {
        for (i, beta) in block_betas.iter_mut().enumerate() {
            if let Some(v) = self.v_per_block.get(i) {
                if v.ncols() == beta.len() && v.nrows() != v.ncols() {
                    let raw = v.dot(&*beta);
                    *beta = raw;
                }
                // If v is square (identity case) we still apply for
                // correctness, but the result equals input modulo
                // floating noise.
                else if v.ncols() == beta.len() && v.nrows() == v.ncols() {
                    // Square V — likely identity. Apply for correctness;
                    // the identity case is a no-op modulo float noise.
                    let raw = v.dot(&*beta);
                    *beta = raw;
                }
            }
        }
    }
}

/// Triangular block-upper-triangular lift `T` for the V+M-exact path.
///
/// Whereas [`SmgsLiftPerBlockV`] performs a strictly per-block lift
/// `β_b_raw = V_b · θ_b`, the V+M-exact path requires a joint lift
/// `β_raw = T · θ_full` where `T` is block-upper-triangular: the
/// diagonal blocks are the per-block `V_b` matrices and the
/// strictly-upper off-diagonal blocks are `−R_{a,b}` (the
/// residualisation reparameterisations of earlier-priority block `a`
/// against later block `b`'s compiled kept directions).
///
/// Mathematically, partitioning θ_full into per-block compiled vectors
/// `θ = (θ_1, …, θ_B)` (concatenated in priority order) and the raw
/// β_full into per-block raw vectors `β_raw = (β_1_raw, …, β_B_raw)`:
///
/// ```text
///   β_a_raw = V_a · θ_a  −  Σ_{b > a} R_{a,b} · θ_b
/// ```
///
/// `T` packages this as a single dense block-upper-triangular matrix
/// of shape `(Σ p_b_raw) × (Σ p_b_compiled)`. When all `R_{a,b}` are
/// zero (no cross-block residualisation), `T` reduces to the
/// block-diagonal of `V`s and `lift_block_betas_via_t` agrees with
/// applying [`SmgsLiftPerBlockV`] per block.
#[derive(Debug, Clone)]
pub struct SmgsLiftViaT {
    pub t_full: Array2<f64>,
    pub block_starts_compiled: Vec<usize>,
    pub block_starts_raw: Vec<usize>,
}

impl SmgsLiftViaT {
    /// Build `T_full` from per-block diagonal `V_b` matrices and the
    /// strictly-upper off-diagonal `R_{a,b}` matrices.
    ///
    /// `r_per_term[b]` (when `Some`) packs ALL strictly-upper
    /// off-diagonal columns for block `b` stacked row-wise across all
    /// earlier-priority blocks `a < b`. It must have:
    ///
    /// - `nrows = Σ_{a < b} v_per_term[a].nrows()` (sum of raw widths
    ///   of all earlier blocks), and
    /// - `ncols = v_per_term[b].ncols()` (compiled width of block `b`).
    ///
    /// `r_per_term[0]` is unused (the first block has no earlier
    /// blocks to residualise against) and should be `None`.
    pub fn from_v_and_r(
        v_per_term: &[Array2<f64>],
        r_per_term: &[Option<Array2<f64>>],
    ) -> Self {
        assert_eq!(
            v_per_term.len(),
            r_per_term.len(),
            "SmgsLiftViaT::from_v_and_r: v_per_term and r_per_term must have the same length"
        );
        let n_blocks = v_per_term.len();

        let mut block_starts_compiled = Vec::with_capacity(n_blocks + 1);
        let mut block_starts_raw = Vec::with_capacity(n_blocks + 1);
        block_starts_compiled.push(0);
        block_starts_raw.push(0);
        for v in v_per_term {
            let prev_c = *block_starts_compiled.last().unwrap();
            let prev_r = *block_starts_raw.last().unwrap();
            block_starts_compiled.push(prev_c + v.ncols());
            block_starts_raw.push(prev_r + v.nrows());
        }
        let total_raw = *block_starts_raw.last().unwrap();
        let total_compiled = *block_starts_compiled.last().unwrap();

        let mut t_full = Array2::<f64>::zeros((total_raw, total_compiled));
        for (a, v) in v_per_term.iter().enumerate() {
            let r0 = block_starts_raw[a];
            let r1 = block_starts_raw[a + 1];
            let c0 = block_starts_compiled[a];
            let c1 = block_starts_compiled[a + 1];
            if v.nrows() > 0 && v.ncols() > 0 {
                t_full.slice_mut(ndarray::s![r0..r1, c0..c1]).assign(v);
            }
        }
        for b in 1..n_blocks {
            let r_b = match r_per_term[b].as_ref() {
                Some(r) => r,
                None => continue,
            };
            let c0 = block_starts_compiled[b];
            let c1 = block_starts_compiled[b + 1];
            assert_eq!(
                r_b.ncols(),
                c1 - c0,
                "SmgsLiftViaT::from_v_and_r: r_per_term[{b}] has {} cols, expected compiled width {}",
                r_b.ncols(),
                c1 - c0,
            );
            assert_eq!(
                r_b.nrows(),
                block_starts_raw[b],
                "SmgsLiftViaT::from_v_and_r: r_per_term[{b}] has {} rows, expected {} (raw width sum of earlier blocks)",
                r_b.nrows(),
                block_starts_raw[b],
            );
            for a in 0..b {
                let ra0 = block_starts_raw[a];
                let ra1 = block_starts_raw[a + 1];
                if ra1 == ra0 || c1 == c0 {
                    continue;
                }
                let slab = r_b.slice(ndarray::s![ra0..ra1, ..]);
                let mut dest = t_full.slice_mut(ndarray::s![ra0..ra1, c0..c1]);
                for i in 0..(ra1 - ra0) {
                    for j in 0..(c1 - c0) {
                        dest[[i, j]] = -slab[[i, j]];
                    }
                }
            }
        }

        Self {
            t_full,
            block_starts_compiled,
            block_starts_raw,
        }
    }

    /// Apply the triangular lift to per-block compiled betas. Inputs
    /// are concatenated into θ_full, multiplied by `T_full` to give
    /// β_full, and split at `block_starts_raw` into per-block raw βs.
    pub fn lift_block_betas_via_t(
        &self,
        compiled_block_betas: &[Array1<f64>],
    ) -> Vec<Array1<f64>> {
        let n_blocks = self.block_starts_compiled.len().saturating_sub(1);
        assert_eq!(
            compiled_block_betas.len(),
            n_blocks,
            "SmgsLiftViaT::lift_block_betas_via_t: got {} compiled block betas, expected {}",
            compiled_block_betas.len(),
            n_blocks,
        );
        for (b, beta) in compiled_block_betas.iter().enumerate() {
            let expected = self.block_starts_compiled[b + 1] - self.block_starts_compiled[b];
            assert_eq!(
                beta.len(),
                expected,
                "SmgsLiftViaT::lift_block_betas_via_t: block {b} has β of len {}, expected compiled width {}",
                beta.len(),
                expected,
            );
        }
        let total_compiled = *self.block_starts_compiled.last().unwrap_or(&0);
        let mut theta_full = Array1::<f64>::zeros(total_compiled);
        for (b, beta) in compiled_block_betas.iter().enumerate() {
            let c0 = self.block_starts_compiled[b];
            let c1 = self.block_starts_compiled[b + 1];
            theta_full.slice_mut(ndarray::s![c0..c1]).assign(beta);
        }
        let beta_full = self.t_full.dot(&theta_full);
        let mut out = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let r0 = self.block_starts_raw[b];
            let r1 = self.block_starts_raw[b + 1];
            out.push(beta_full.slice(ndarray::s![r0..r1]).to_owned());
        }
        out
    }
}

/// Run the identifiability compiler on the three survival parametric
/// blocks (time, marginal, logslope) at a pilot β and return the per-
/// block V reparameterisation matrices.
///
/// `row_hess` must be a PSD per-row 4×4 Hessian of `−log L_i(u_i)` at
/// the pilot β (see [`SurvivalRowHessian::from_pilot_primary_state`]).
/// The compiler residualises blocks left-to-right in priority order
/// (time → marginal → logslope) in the sqrt-H-metric so any aliased
/// direction lands in the lower-priority block, then runs a post-walk
/// column-pivoted QR on the cumulative anchor and drops trailing
/// pivots from the latest block. The returned V matrices are ready to
/// be applied to each block's raw design and penalty before the
/// `ParameterBlockSpec` list is assembled.
///
/// On `FullyAliased` from `compile()` (a block fully absorbed by its
/// cumulative anchor) this returns `Err`. The construction site should
/// surface that as a structured user-facing diagnostic — the model is
/// asking the compiler to assign zero degrees of freedom to a named
/// parametric block, which is a model-spec bug not a numerical one.
///
/// Sibling Phase-4b wiring (`bernoulli_marginal_slope::install_compiled_flex_block_into_runtime`)
/// already calls `compile()` for the flex blocks. This helper extends
/// that contract to the parametric blocks by giving the SMGS
/// construction site a one-line entry point — it does NOT yet apply
/// the V transforms to the family's captured designs (the captured-
/// design update is the remaining integration step that touches the
/// family's row-Hessian assembly assertions).
pub fn compile_survival_parametric_designs(
    time_dq0: Array2<f64>,
    time_dq1: Array2<f64>,
    time_dqd1: Array2<f64>,
    marginal_dq: Array2<f64>,
    marginal_dqd1: Array2<f64>,
    logslope_dg: Array2<f64>,
    row_hess: &dyn RowHessian,
) -> Result<SurvivalParametricCompiled, String> {
    use crate::families::identifiability_compiler::compile;

    let p_time_raw = time_dq0.ncols();
    let p_marg_raw = marginal_dq.ncols();
    let p_log_raw = logslope_dg.ncols();

    let inputs = build_survival_compiler_inputs(
        time_dq0,
        time_dq1,
        time_dqd1,
        marginal_dq,
        marginal_dqd1,
        logslope_dg,
        None,
        None,
    );
    if inputs.operators.len() != 3 {
        return Err(format!(
            "compile_survival_parametric_designs: expected exactly 3 parametric operators \
             (time, marginal, logslope); got {}",
            inputs.operators.len(),
        ));
    }
    let compiled = compile(&inputs.operators, row_hess, &inputs.ordering)
        .map_err(|e| format!("identifiability_compiler::compile failed: {e}"))?;
    if compiled.blocks.len() != 3 {
        return Err(format!(
            "compile_survival_parametric_designs: compiler emitted {} blocks; expected 3",
            compiled.blocks.len(),
        ));
    }
    let v_time = compiled.blocks[0].t_lw.clone();
    let v_marginal = compiled.blocks[1].t_lw.clone();
    let v_logslope = compiled.blocks[2].t_lw.clone();
    let drops_by_block = (
        p_time_raw.saturating_sub(v_time.ncols()),
        p_marg_raw.saturating_sub(v_marginal.ncols()),
        p_log_raw.saturating_sub(v_logslope.ncols()),
    );
    Ok(SurvivalParametricCompiled {
        v_time,
        v_marginal,
        v_logslope,
        drops_by_block,
    })
}

/// Build the operator stack from already-materialised dense designs.
///
/// `time_dq0/dq1/dqd1` are the time block's three primary-state Jacobians
/// at training rows. `marginal_dq` and `marginal_dqd1` are the marginal
/// block's contributions to q (shared between q0 and q1) and to qd1
/// (typically zero unless timewiggle interacts). `logslope_dg` is the
/// logslope block's contribution to g.
///
/// `score_warp_(dq, dqd1)` / `link_dev_(dq, dqd1)` are present only when
/// the corresponding flex block is active. The returned `ordering` parallels
/// `operators` so the caller can route compiled outputs back to runtime slots.
pub fn build_survival_compiler_inputs(
    time_dq0: Array2<f64>,
    time_dq1: Array2<f64>,
    time_dqd1: Array2<f64>,
    marginal_dq: Array2<f64>,
    marginal_dqd1: Array2<f64>,
    logslope_dg: Array2<f64>,
    score_warp_dq_dqd1: Option<(Array2<f64>, Array2<f64>)>,
    link_dev_dq_dqd1: Option<(Array2<f64>, Array2<f64>)>,
) -> SurvivalCompilerInputs {
    let mut operators: Vec<Arc<dyn RowJacobianOperator>> = Vec::with_capacity(5);
    let mut ordering: Vec<BlockOrder> = Vec::with_capacity(5);

    operators.push(Arc::new(TimeBlockOperator::new(
        time_dq0, time_dq1, time_dqd1,
    )));
    ordering.push(BlockOrder::Time);

    operators.push(Arc::new(QChannelBlockOperator::new(
        marginal_dq,
        marginal_dqd1,
    )));
    ordering.push(BlockOrder::Marginal);

    operators.push(Arc::new(LogslopeBlockOperator::new(logslope_dg)));
    ordering.push(BlockOrder::Logslope);

    if let Some((dq, dqd1)) = score_warp_dq_dqd1 {
        operators.push(Arc::new(QChannelBlockOperator::new(dq, dqd1)));
        ordering.push(BlockOrder::ScoreWarp);
    }
    if let Some((dq, dqd1)) = link_dev_dq_dqd1 {
        operators.push(Arc::new(QChannelBlockOperator::new(dq, dqd1)));
        ordering.push(BlockOrder::LinkDev);
    }

    SurvivalCompilerInputs {
        operators,
        ordering,
    }
}

/// Per-term-aware V+M-exact compiled designs + penalties + the full
/// triangular T matrix for result-time β lift. The construction site
/// swaps raw designs/penalties for these compiled versions before
/// building `ParameterBlockSpec`s; at fit result the joint compiled β
/// is lifted to raw via `T · θ` (where T is block-upper-triangular with
/// V_b on the diagonal and `-R_{a→b}` off-diagonal).
///
/// Difference from [`CompiledSurvivalDesignsPerTerm`]:
/// - emitted designs use [`ResidualisedDesignOperator`] (the exact
///   `C_b·V_b − A_{<b}·R_b` row form), not the V-only
///   `CoefficientTransformOperator`;
/// - per-term penalties are pulled back through the FULL triangular T
///   (not just V_b), so they are full-width `PenaltyMatrix::Dense`
///   matrices that may couple across blocks via the off-diagonal
///   `-R_{a→b}` entries of T;
/// - the joint result-time lift uses `t_full` rather than the per-block
///   V_b matrices on their own.
pub struct CompiledSurvivalDesignsVMExact {
    pub time_design_entry: DesignMatrix,
    pub time_design_exit: DesignMatrix,
    pub time_design_derivative_exit: DesignMatrix,
    pub marginal_design: DesignMatrix,
    pub logslope_design: DesignMatrix,
    /// Per-term penalties, each pulled back through the full triangular
    /// T. The result is a full-width `PenaltyMatrix::Dense` (joint
    /// p_compiled × p_compiled) — cross-block coupling can be nonzero
    /// when `R_{a→b}` is nonzero.
    pub time_penalties: Vec<PenaltyMatrix>,
    pub marginal_penalties: Vec<PenaltyMatrix>,
    pub logslope_penalties: Vec<PenaltyMatrix>,
    /// Full triangular T matrix: rows = raw joint width, cols = compiled
    /// joint width. Diagonal blocks are V_b; upper off-diagonal blocks
    /// at `(a, b)` for `a < b` are `-R_{a→b}` (residualised reparam
    /// against the earlier block). Used at result time to lift
    /// `θ_compiled → β_raw = T · θ_compiled`.
    pub t_full: Array2<f64>,
}

/// Channel for the V+M-exact apply path. Identifies which channel a
/// given compiled-block's anchor design belongs to — earlier blocks
/// from a different channel contribute zero anchor rows because the
/// compiler's row-primary-state subspace they residualise against does
/// not project to this channel's emitted-row column space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VmChannel {
    TimeEntry,
    TimeExit,
    TimeDerivativeExit,
    Marginal,
    Logslope,
}

/// Per-term V+M-exact apply: produce compiled designs assembled via
/// `ResidualisedDesignOperator`, per-term penalties pulled back through
/// the full triangular T, and `t_full` for result-time lift.
///
/// Cross-channel anchors (e.g., a time-term anchor when emitting the
/// marginal design) contribute zero — the time block's row Jacobian has
/// no marginal-channel component, so its anchor design at the marginal
/// design's row coordinates is a zero matrix. We express this by
/// passing a zero `DenseDesignMatrix` of the appropriate shape as the
/// anchor design for cross-channel anchor entries.
#[allow(clippy::too_many_arguments)]
pub fn apply_per_term_vm_exact(
    compiled: &SurvivalParametricCompiledPerTerm,
    time_partition: &[std::ops::Range<usize>],
    marginal_partition: &[std::ops::Range<usize>],
    logslope_partition: &[std::ops::Range<usize>],
    time_design_entry: DesignMatrix,
    time_design_exit: DesignMatrix,
    time_design_derivative_exit: DesignMatrix,
    marginal_design: DesignMatrix,
    logslope_design: DesignMatrix,
    time_penalties: &[crate::terms::smooth::BlockwisePenalty],
    marginal_penalties: &[crate::terms::smooth::BlockwisePenalty],
    logslope_penalties: &[crate::terms::smooth::BlockwisePenalty],
) -> Result<CompiledSurvivalDesignsVMExact, String> {
    let n_time = time_partition.len();
    let n_marg = marginal_partition.len();
    let n_log = logslope_partition.len();
    let n_total = n_time + n_marg + n_log;

    if compiled.v_time_per_term.len() != n_time
        || compiled.v_marginal_per_term.len() != n_marg
        || compiled.v_logslope_per_term.len() != n_log
    {
        return Err(format!(
            "apply_per_term_vm_exact: compiled per-term V counts \
             (time={}, marg={}, log={}) do not match partition counts \
             (time={}, marg={}, log={})",
            compiled.v_time_per_term.len(),
            compiled.v_marginal_per_term.len(),
            compiled.v_logslope_per_term.len(),
            n_time,
            n_marg,
            n_log,
        ));
    }
    if compiled.r_lw_per_term.len() != n_total {
        return Err(format!(
            "apply_per_term_vm_exact: compiled r_lw count {} != total compiled-blocks {}",
            compiled.r_lw_per_term.len(),
            n_total,
        ));
    }

    // Flatten per-term V's in global compile order: time-terms, then
    // marginal-terms, then logslope-terms. Mirrors the order in
    // `compile_survival_parametric_designs_per_term` so r_lw_per_term
    // indexing matches.
    let mut v_per_block: Vec<Array2<f64>> = Vec::with_capacity(n_total);
    let mut block_channel: Vec<VmChannel> = Vec::with_capacity(n_total);
    let mut block_term_index_within_channel: Vec<usize> = Vec::with_capacity(n_total);
    for (i, v) in compiled.v_time_per_term.iter().enumerate() {
        v_per_block.push(v.clone());
        block_channel.push(VmChannel::TimeEntry);
        block_term_index_within_channel.push(i);
    }
    for (i, v) in compiled.v_marginal_per_term.iter().enumerate() {
        v_per_block.push(v.clone());
        block_channel.push(VmChannel::Marginal);
        block_term_index_within_channel.push(i);
    }
    for (i, v) in compiled.v_logslope_per_term.iter().enumerate() {
        v_per_block.push(v.clone());
        block_channel.push(VmChannel::Logslope);
        block_term_index_within_channel.push(i);
    }

    // Build the full triangular T for result-time lift and penalty pullback.
    let t_full = build_full_t_matrix(&v_per_block, &compiled.r_lw_per_term);

    let p_compiled_total: usize = v_per_block.iter().map(|v| v.ncols()).sum();

    let channel_class_matches = |design_channel: VmChannel, bc: VmChannel| -> bool {
        let is_time_class = |c: VmChannel| {
            matches!(
                c,
                VmChannel::TimeEntry | VmChannel::TimeExit | VmChannel::TimeDerivativeExit
            )
        };
        match design_channel {
            VmChannel::TimeEntry | VmChannel::TimeExit | VmChannel::TimeDerivativeExit => {
                is_time_class(bc)
            }
            VmChannel::Marginal => bc == VmChannel::Marginal,
            VmChannel::Logslope => bc == VmChannel::Logslope,
        }
    };

    let raw_width_for_block = |i: usize| -> usize {
        match block_channel[i] {
            VmChannel::TimeEntry | VmChannel::TimeExit | VmChannel::TimeDerivativeExit => {
                time_partition[block_term_index_within_channel[i]].len()
            }
            VmChannel::Marginal => marginal_partition[block_term_index_within_channel[i]].len(),
            VmChannel::Logslope => logslope_partition[block_term_index_within_channel[i]].len(),
        }
    };
    let term_range_for_block = |i: usize| -> std::ops::Range<usize> {
        match block_channel[i] {
            VmChannel::TimeEntry | VmChannel::TimeExit | VmChannel::TimeDerivativeExit => {
                time_partition[block_term_index_within_channel[i]].clone()
            }
            VmChannel::Marginal => {
                marginal_partition[block_term_index_within_channel[i]].clone()
            }
            VmChannel::Logslope => {
                logslope_partition[block_term_index_within_channel[i]].clone()
            }
        }
    };

    let assemble_channel_design = |raw_design: DesignMatrix,
                                   channel: VmChannel,
                                   context: &str|
     -> Result<DesignMatrix, String> {
        let dense_full = match raw_design {
            DesignMatrix::Dense(d) => d.to_dense(),
            DesignMatrix::Sparse(_) => raw_design
                .try_to_dense_by_chunks(&format!("{context}: sparse→dense for V+M apply"))
                .map_err(|e| format!("{context}: densify failed: {e}"))?,
        };
        let n_rows = dense_full.nrows();

        // Build per-block raw slices in global compile order. For
        // cross-channel blocks we emit a zero slice of the right shape
        // so anchor-subtraction is a no-op. For same-class blocks we
        // slice `dense_full` at the term's raw column range.
        let mut raw_slices: Vec<Array2<f64>> = Vec::with_capacity(n_total);
        for i in 0..n_total {
            let raw_w = raw_width_for_block(i);
            if channel_class_matches(channel, block_channel[i]) {
                let term_range = term_range_for_block(i);
                if term_range.end > dense_full.ncols() {
                    return Err(format!(
                        "{context}: term range {:?} exceeds raw design cols {}",
                        term_range,
                        dense_full.ncols(),
                    ));
                }
                raw_slices.push(
                    dense_full
                        .slice(ndarray::s![.., term_range])
                        .to_owned(),
                );
            } else {
                raw_slices.push(Array2::<f64>::zeros((n_rows, raw_w)));
            }
        }

        let mut per_block_designs: Vec<DesignMatrix> = Vec::with_capacity(n_total);
        for g in 0..n_total {
            let inner_dense = DenseDesignMatrix::from(raw_slices[g].clone());
            let v_b = v_per_block[g].clone();
            let mut anchors: Vec<(DesignMatrix, Array2<f64>)> = Vec::new();
            if g > 0 {
                let r_full = compiled
                    .r_lw_per_term
                    .get(g)
                    .ok_or_else(|| {
                        format!("{context}: missing r_lw entry for compiled-block {g}")
                    })?
                    .as_ref();
                let mut row_off = 0usize;
                for a in 0..g {
                    let kept_a = v_per_block[a].ncols();
                    let r_a_g = match r_full {
                        Some(r) => r
                            .slice(ndarray::s![row_off..row_off + kept_a, ..])
                            .to_owned(),
                        None => Array2::<f64>::zeros((kept_a, v_b.ncols())),
                    };
                    // Anchor design at kept-width: wrap raw_slices[a]
                    // through V_a so `anchor_chunk · r_a_g` has the
                    // expected shape (chunk_rows × kept_b).
                    let inner_a = DenseDesignMatrix::from(raw_slices[a].clone());
                    let v_a = v_per_block[a].clone();
                    let kept_anchor_design = {
                        let op = CoefficientTransformOperator::new(inner_a, v_a)
                            .map_err(|e| {
                                format!(
                                    "{context}: CoefficientTransformOperator anchor block {a}: {e}",
                                )
                            })?;
                        DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op)))
                    };
                    anchors.push((kept_anchor_design, r_a_g));
                    row_off += kept_a;
                }
            }
            let op = ResidualisedDesignOperator::new(inner_dense, v_b, anchors)
                .map_err(|e| {
                    format!("{context}: ResidualisedDesignOperator::new (block {g}): {e}")
                })?;
            per_block_designs.push(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op))));
        }

        let same_class_designs: Vec<DesignMatrix> = per_block_designs
            .into_iter()
            .enumerate()
            .filter_map(|(i, d)| {
                if channel_class_matches(channel, block_channel[i]) {
                    Some(d)
                } else {
                    None
                }
            })
            .collect();
        if same_class_designs.is_empty() {
            return Err(format!(
                "{context}: no compiled-blocks belong to this channel class",
            ));
        }
        DesignMatrix::hstack(same_class_designs)
            .map_err(|e| format!("{context}: hstack: {e}"))
    };

    let time_entry_out = assemble_channel_design(
        time_design_entry,
        VmChannel::TimeEntry,
        "vm-exact: time entry",
    )?;
    let time_exit_out = assemble_channel_design(
        time_design_exit,
        VmChannel::TimeExit,
        "vm-exact: time exit",
    )?;
    let time_deriv_out = assemble_channel_design(
        time_design_derivative_exit,
        VmChannel::TimeDerivativeExit,
        "vm-exact: time derivative_exit",
    )?;
    let marginal_out = assemble_channel_design(
        marginal_design,
        VmChannel::Marginal,
        "vm-exact: marginal",
    )?;
    let logslope_out = assemble_channel_design(
        logslope_design,
        VmChannel::Logslope,
        "vm-exact: logslope",
    )?;

    // Pull each per-term penalty back through the full triangular T.
    // The penalty's `col_range` is interpreted within the channel's
    // raw block. We add the channel's raw offset (within the joint raw
    // width) to get the anchor_offset used by
    // `pull_back_penalty_through_t`.
    let p_time_raw: usize = time_partition.iter().map(|r| r.len()).sum();
    let p_marg_raw: usize = marginal_partition.iter().map(|r| r.len()).sum();
    let time_offset = 0usize;
    let marg_offset = p_time_raw;
    let log_offset = p_time_raw + p_marg_raw;

    let pull_set = |pens: &[crate::terms::smooth::BlockwisePenalty],
                    anchor_offset: usize|
     -> Result<Vec<PenaltyMatrix>, String> {
        pens.iter()
            .map(|p| {
                pull_back_penalty_through_t(p, anchor_offset, &t_full).map_err(|e| {
                    format!("vm-exact: pull_back_penalty_through_t: {e}")
                })
            })
            .collect()
    };

    if t_full.ncols() != p_compiled_total {
        return Err(format!(
            "vm-exact: t_full has {} cols but expected compiled total {p_compiled_total}",
            t_full.ncols(),
        ));
    }

    Ok(CompiledSurvivalDesignsVMExact {
        time_design_entry: time_entry_out,
        time_design_exit: time_exit_out,
        time_design_derivative_exit: time_deriv_out,
        marginal_design: marginal_out,
        logslope_design: logslope_out,
        time_penalties: pull_set(time_penalties, time_offset)?,
        marginal_penalties: pull_set(marginal_penalties, marg_offset)?,
        logslope_penalties: pull_set(logslope_penalties, log_offset)?,
        t_full,
    })
}

/// Project a raw-space warm start β_raw into compiled coordinates θ via the
/// Gram-aware least-squares formula
///
///     θ = (Tᵀ K^S T)^+ · Tᵀ K^S · β_raw
///
/// where K^S is the structural Gram on raw space. This is the correct
/// projection whenever T's columns are not Euclidean-orthonormal (e.g. the
/// V+M-exact compiled basis), because Vᵀ·β is only an orthogonal projector
/// in the Euclidean inner product, not in the K^S-induced inner product
/// the compiled coordinates live under.
pub fn project_raw_beta_to_compiled(
    t_full: &Array2<f64>,
    k_struct: &Array2<f64>,
    beta_raw: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let p_raw = t_full.nrows();
    let p_comp = t_full.ncols();
    if k_struct.nrows() != p_raw || k_struct.ncols() != p_raw {
        return Err(format!(
            "project_raw_beta_to_compiled: K^S shape {}x{} mismatches T rows {}",
            k_struct.nrows(),
            k_struct.ncols(),
            p_raw,
        ));
    }
    if beta_raw.len() != p_raw {
        return Err(format!(
            "project_raw_beta_to_compiled: beta_raw length {} mismatches T rows {}",
            beta_raw.len(),
            p_raw,
        ));
    }
    if p_comp == 0 {
        return Ok(Array1::<f64>::zeros(0));
    }
    if p_raw == 0 {
        return Ok(Array1::<f64>::zeros(p_comp));
    }

    // K^S · T  (p_raw × p_comp)
    let ks_t = fast_ab(k_struct, t_full);
    // M = Tᵀ K^S T  (p_comp × p_comp); compute as Tᵀ · (K^S T).
    let m = crate::linalg::faer_ndarray::fast_atb(t_full, &ks_t);
    // Symmetrize M before eigendecomposition.
    let mut m_sym = Array2::<f64>::zeros((p_comp, p_comp));
    for i in 0..p_comp {
        for j in 0..p_comp {
            m_sym[[i, j]] = 0.5 * (m[[i, j]] + m[[j, i]]);
        }
    }

    // Tᵀ K^S β_raw  (length p_comp): compute K^S β_raw, then Tᵀ · that.
    let ks_beta = crate::linalg::faer_ndarray::fast_av(k_struct, beta_raw);
    let rhs = crate::linalg::faer_ndarray::fast_atv(t_full, &ks_beta);

    let (evals, evecs) = m_sym
        .eigh(Side::Lower)
        .map_err(|e| format!("project_raw_beta_to_compiled: eigh failed: {e:?}"))?;

    let max_abs = evals.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = (p_comp as f64) * f64::EPSILON * max_abs.max(1.0);

    // Apply M^+ via spectral decomposition: M^+ = U · diag(1/λ_i for λ_i>tol) · Uᵀ.
    let ut_rhs = crate::linalg::faer_ndarray::fast_atv(&evecs, &rhs);
    let mut scaled = Array1::<f64>::zeros(p_comp);
    for i in 0..p_comp {
        if evals[i].abs() > tol {
            scaled[i] = ut_rhs[i] / evals[i];
        }
    }
    let theta = crate::linalg::faer_ndarray::fast_av(&evecs, &scaled);
    Ok(theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psd_clamp_zeros_negative_eigenvalues() {
        // Construct M = U diag(2, -1, 0.5, -0.25) Uᵀ for a fixed U from
        // a small rotation, verify the clamped matrix has eigenvalues
        // (2, 0, 0.5, 0).
        let mut m = Array2::<f64>::zeros((4, 4));
        // Diagonal with mixed signs is sufficient for the test: the
        // eigenvalues equal the diagonal and the eigenvectors are e_i.
        m[[0, 0]] = 2.0;
        m[[1, 1]] = -1.0;
        m[[2, 2]] = 0.5;
        m[[3, 3]] = -0.25;
        let clamped = psd_clamp_4x4(&m);
        assert!((clamped[[0, 0]] - 2.0).abs() < 1e-12);
        assert!(clamped[[1, 1]].abs() < 1e-12);
        assert!((clamped[[2, 2]] - 0.5).abs() < 1e-12);
        assert!(clamped[[3, 3]].abs() < 1e-12);
    }

    #[test]
    fn time_block_operator_evaluate_full_shape() {
        let n = 6;
        let p = 3;
        let dq0 = Array2::from_shape_fn((n, p), |(i, j)| (i + j) as f64);
        let dq1 = Array2::from_shape_fn((n, p), |(i, j)| (i as f64) * 2.0 + j as f64);
        let dqd1 = Array2::from_shape_fn((n, p), |(i, j)| 0.5 * ((i * j) as f64));
        let op = TimeBlockOperator::new(dq0.clone(), dq1.clone(), dqd1.clone());
        let full = op.evaluate_full();
        assert_eq!(full.shape(), &[n, p, K_SURVIVAL]);
        for i in 0..n {
            for j in 0..p {
                assert_eq!(full[[i, j, 0]], dq0[[i, j]]);
                assert_eq!(full[[i, j, 1]], dq1[[i, j]]);
                assert_eq!(full[[i, j, 2]], dqd1[[i, j]]);
                assert_eq!(full[[i, j, 3]], 0.0);
            }
        }
    }

    #[test]
    fn q_channel_block_apply_row_shares_q0_q1() {
        let n = 5;
        let p = 2;
        let dq = Array2::from_shape_fn((n, p), |(i, j)| (i as f64) * (j as f64 + 1.0));
        let dqd1 = Array2::from_shape_fn((n, p), |(i, j)| (j as f64) - (i as f64));
        let op = QChannelBlockOperator::new(dq.clone(), dqd1.clone());
        let mut out = [0.0_f64; K_SURVIVAL];
        let delta = [1.0_f64, -0.5];
        op.apply_row(3, &delta, &mut out);
        let want_q = dq[[3, 0]] * 1.0 + dq[[3, 1]] * (-0.5);
        let want_qd = dqd1[[3, 0]] * 1.0 + dqd1[[3, 1]] * (-0.5);
        assert!((out[0] - want_q).abs() < 1e-12);
        assert!((out[1] - want_q).abs() < 1e-12);
        assert!((out[2] - want_qd).abs() < 1e-12);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn logslope_block_writes_only_g_channel() {
        let n = 4;
        let p = 2;
        let dg = Array2::from_shape_fn((n, p), |(i, j)| (i as f64) + 0.1 * (j as f64));
        let op = LogslopeBlockOperator::new(dg.clone());
        let mut out = [0.0_f64; K_SURVIVAL];
        let delta = [2.0_f64, -1.0];
        op.apply_row(1, &delta, &mut out);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
        let want = dg[[1, 0]] * 2.0 + dg[[1, 1]] * (-1.0);
        assert!((out[3] - want).abs() < 1e-12);
    }

    #[test]
    fn extract_term_partition_simple_cases() {
        // No penalties: whole block is one term.
        let part = extract_term_partition_from_penalty_ranges(5, &[]);
        assert_eq!(part, vec![0..5]);
        // One penalty covering the whole block.
        let part = extract_term_partition_from_penalty_ranges(5, &[0..5]);
        assert_eq!(part, vec![0..5]);
        // Two penalties with a gap: produces three terms (pen1, gap, pen2).
        let part = extract_term_partition_from_penalty_ranges(10, &[0..3, 6..10]);
        assert_eq!(part, vec![0..3, 3..6, 6..10]);
        // Duplicate penalty ranges coalesce.
        let part = extract_term_partition_from_penalty_ranges(6, &[0..3, 0..3, 3..6]);
        assert_eq!(part, vec![0..3, 3..6]);
        // Empty block.
        let part = extract_term_partition_from_penalty_ranges(0, &[]);
        assert!(part.is_empty());
    }

    #[test]
    fn block_diag_from_assembles_correctly() {
        let v1 = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| (i * 2 + j + 1) as f64);
        let v2 = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| (10 + i * 2 + j) as f64);
        let bd = block_diag_from(&[v1.clone(), v2.clone()]);
        assert_eq!(bd.dim(), (5, 4));
        // Top-left = v1, bottom-right = v2, off-blocks = zero.
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(bd[[i, j]], v1[[i, j]]);
                assert_eq!(bd[[i, 2 + j]], 0.0);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(bd[[3 + i, 2 + j]], v2[[i, j]]);
                assert_eq!(bd[[3 + i, j]], 0.0);
            }
        }
    }

    #[test]
    fn pull_back_blockwise_penalty_per_term_full_term_identity_v() {
        use crate::terms::smooth::BlockwisePenalty;
        // Single-term partition, identity V → pullback returns input.
        let v_term = Array2::<f64>::eye(3);
        let v_per_term = vec![v_term];
        let partition = vec![0..3];
        let local = Array2::<f64>::from_shape_fn((3, 3), |(i, j)| (i + j) as f64);
        let pen = BlockwisePenalty::new(0..3, local.clone());
        let out = pull_back_blockwise_penalty_per_term(&pen, &partition, &v_per_term)
            .expect("identity-V pullback must succeed");
        assert_eq!(out.col_range, 0..3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((out.local[[i, j]] - 0.5 * (local[[i, j]] + local[[j, i]])).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pull_back_blockwise_penalty_per_term_drops_one_column() {
        use crate::terms::smooth::BlockwisePenalty;
        // 3-column term, V drops one column (3 → 2): V is the 3×2
        // matrix with columns e_1 and e_2 (drop the constant).
        let mut v_term = Array2::<f64>::zeros((3, 2));
        v_term[[1, 0]] = 1.0;
        v_term[[2, 1]] = 1.0;
        let partition = vec![0..3];
        let local = Array2::<f64>::from_shape_fn((3, 3), |(i, j)| (i + j + 1) as f64);
        let pen = BlockwisePenalty::new(0..3, local.clone());
        let out = pull_back_blockwise_penalty_per_term(&pen, &partition, &[v_term])
            .expect("selection-V pullback must succeed");
        assert_eq!(out.col_range, 0..2);
        // Expected: pullback is the (1..3, 1..3) sub-block of the
        // symmetric part of `local`.
        let sym = |i: usize, j: usize| 0.5 * (local[[i, j]] + local[[j, i]]);
        assert!((out.local[[0, 0]] - sym(1, 1)).abs() < 1e-12);
        assert!((out.local[[0, 1]] - sym(1, 2)).abs() < 1e-12);
        assert!((out.local[[1, 0]] - sym(2, 1)).abs() < 1e-12);
        assert!((out.local[[1, 1]] - sym(2, 2)).abs() < 1e-12);
    }

    #[test]
    fn pull_back_blockwise_penalty_per_term_routes_to_correct_term() {
        use crate::terms::smooth::BlockwisePenalty;
        // Two-term partition: term0 = 0..2, term1 = 2..5.
        // V_term0 = 2×2 identity; V_term1 drops 1 col (3 → 2).
        let v0 = Array2::<f64>::eye(2);
        let mut v1 = Array2::<f64>::zeros((3, 2));
        v1[[0, 0]] = 1.0;
        v1[[2, 1]] = 1.0;
        let partition = vec![0..2, 2..5];
        let v_per_term = vec![v0, v1];
        // Penalty over term1 only (col_range 2..5).
        let local1 = Array2::<f64>::from_shape_fn((3, 3), |(i, j)| (10 + i + j) as f64);
        let pen1 = BlockwisePenalty::new(2..5, local1.clone());
        let out1 = pull_back_blockwise_penalty_per_term(&pen1, &partition, &v_per_term)
            .expect("term1 pullback must succeed");
        // Compiled col_range for term1 = (0+2)..(0+2)+v1.ncols() = 2..4.
        assert_eq!(out1.col_range, 2..4);
        // Pullback expected: V1ᵀ · sym(local1) · V1, which picks rows/
        // cols (0, 2) of sym(local1).
        let sym = |i: usize, j: usize| 0.5 * (local1[[i, j]] + local1[[j, i]]);
        assert!((out1.local[[0, 0]] - sym(0, 0)).abs() < 1e-12);
        assert!((out1.local[[0, 1]] - sym(0, 2)).abs() < 1e-12);
        assert!((out1.local[[1, 0]] - sym(2, 0)).abs() < 1e-12);
        assert!((out1.local[[1, 1]] - sym(2, 2)).abs() < 1e-12);
    }

    #[test]
    fn build_full_t_matrix_identity_when_v_eye_and_r_none() {
        let v_a = Array2::<f64>::eye(2);
        let v_b = Array2::<f64>::eye(2);
        let t = build_full_t_matrix(&[v_a, v_b], &[None, None]);
        assert_eq!(t.dim(), (4, 4));
        let eye4 = Array2::<f64>::eye(4);
        for i in 0..4 {
            for j in 0..4 {
                assert!((t[[i, j]] - eye4[[i, j]]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn build_full_t_matrix_with_drops_and_nonzero_r() {
        let mut v_a = Array2::<f64>::zeros((3, 2));
        v_a[[0, 0]] = 1.0;
        v_a[[1, 0]] = 0.5;
        v_a[[2, 1]] = 1.0;
        let v_b = Array2::<f64>::eye(2);
        let r_ab = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| 1.0 + (i as f64) + 0.25 * (j as f64));
        let t = build_full_t_matrix(&[v_a.clone(), v_b.clone()], &[None, Some(r_ab.clone())]);
        assert_eq!(t.dim(), (5, 4));
        for i in 0..3 {
            for j in 0..2 {
                assert!((t[[i, j]] - v_a[[i, j]]).abs() < 1e-14);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((t[[3 + i, 2 + j]] - v_b[[i, j]]).abs() < 1e-14);
            }
        }
        for i in 0..3 {
            for j in 0..2 {
                assert!((t[[i, 2 + j]] + r_ab[[i, j]]).abs() < 1e-14);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(t[[3 + i, j]], 0.0);
            }
        }
    }

    #[test]
    fn pull_back_penalty_through_t_identity_returns_zero_embedded_raw() {
        use crate::terms::smooth::BlockwisePenalty;
        let v_a = Array2::<f64>::eye(2);
        let v_b = Array2::<f64>::eye(2);
        let t = build_full_t_matrix(&[v_a, v_b], &[None, None]);
        let local = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| (i + 2 * j + 1) as f64);
        let pen = BlockwisePenalty::new(0..2, local.clone());
        let anchor_offset = 2usize;
        let out = pull_back_penalty_through_t(&pen, anchor_offset, &t);
        let PenaltyMatrix::Dense(dense) = out else {
            panic!("expected PenaltyMatrix::Dense");
        };
        assert_eq!(dense.dim(), (4, 4));
        let sym_local = |i: usize, j: usize| 0.5 * (local[[i, j]] + local[[j, i]]);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i >= 2 && j >= 2 {
                    sym_local(i - 2, j - 2)
                } else {
                    0.0
                };
                assert!(
                    (dense[[i, j]] - want).abs() < 1e-14,
                    "mismatch at ({i},{j}): got {}, want {}",
                    dense[[i, j]],
                    want,
                );
            }
        }
    }

    #[test]
    fn pull_back_penalty_through_t_nontrivial_t_has_off_block_coupling() {
        use crate::terms::smooth::BlockwisePenalty;
        let v_a = Array2::<f64>::eye(2);
        let v_b = Array2::<f64>::eye(2);
        let r_ab = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| 1.0 + (i + j) as f64);
        let t = build_full_t_matrix(&[v_a, v_b], &[None, Some(r_ab.clone())]);
        let mut local = Array2::<f64>::zeros((2, 2));
        local[[0, 0]] = 2.0;
        local[[1, 1]] = 3.0;
        let pen = BlockwisePenalty::new(0..2, local.clone());
        let out = pull_back_penalty_through_t(&pen, 0, &t);
        let PenaltyMatrix::Dense(dense) = out else {
            panic!("expected PenaltyMatrix::Dense");
        };
        assert_eq!(dense.dim(), (4, 4));
        let mut any_nonzero = false;
        for i in 0..2 {
            for j in 0..2 {
                if dense[[i, 2 + j]].abs() > 1e-10 {
                    any_nonzero = true;
                }
            }
        }
        assert!(any_nonzero, "expected nonzero off-block coupling");
        let want = local.dot(&r_ab).map(|v| -v);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (dense[[i, 2 + j]] - want[[i, j]]).abs() < 1e-12,
                    "off-block (a,b) mismatch at ({i},{j}): got {}, want {}",
                    dense[[i, 2 + j]],
                    want[[i, j]],
                );
                assert!((dense[[2 + j, i]] - want[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pull_back_penalty_through_t_round_trip_quadratic_form() {
        use crate::terms::smooth::BlockwisePenalty;
        let v_a = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| {
            ((i + 1) as f64).sin() + 0.3 * (j as f64)
        });
        let v_b = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| 1.0 + 0.1 * ((i * 2 + j) as f64));
        let r_ab = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| 0.5 - 0.2 * (i as f64) + 0.1 * (j as f64));
        let t = build_full_t_matrix(&[v_a.clone(), v_b.clone()], &[None, Some(r_ab.clone())]);
        let raw_local = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| {
            if i == j { 2.5 + i as f64 } else { 0.4 }
        });
        let pen = BlockwisePenalty::new(0..2, raw_local.clone());
        let anchor = 3usize;
        let out = pull_back_penalty_through_t(&pen, anchor, &t);
        let PenaltyMatrix::Dense(s_compiled) = out else {
            panic!("expected Dense");
        };
        let raw_total = t.nrows();
        let mut s_raw_emb = Array2::<f64>::zeros((raw_total, raw_total));
        for i in 0..2 {
            for j in 0..2 {
                s_raw_emb[[anchor + i, anchor + j]] = raw_local[[i, j]];
            }
        }
        let mut s_raw_sym = Array2::<f64>::zeros((raw_total, raw_total));
        for i in 0..raw_total {
            for j in 0..raw_total {
                s_raw_sym[[i, j]] = 0.5 * (s_raw_emb[[i, j]] + s_raw_emb[[j, i]]);
            }
        }
        let theta = Array1::<f64>::from_shape_fn(t.ncols(), |k| ((k as f64) * 0.7 - 0.3).cos());
        let gamma = t.dot(&theta);
        let lhs = theta.dot(&s_compiled.dot(&theta));
        let rhs = gamma.dot(&s_raw_sym.dot(&gamma));
        assert!(
            (lhs - rhs).abs() < 1e-10,
            "round-trip mismatch: lhs={lhs}, rhs={rhs}",
        );
    }

    #[test]
    fn validate_partition_rejects_bad_partitions() {
        // Doesn't start at 0.
        assert!(validate_partition(&[1..5], 5, "test").is_err());
        // Doesn't cover the block.
        assert!(validate_partition(&[0..3], 5, "test").is_err());
        // Has a gap.
        assert!(validate_partition(&[0..2, 3..5], 5, "test").is_err());
        // Has overlap.
        assert!(validate_partition(&[0..3, 2..5], 5, "test").is_err());
        // Has empty range.
        assert!(validate_partition(&[0..0, 0..5], 5, "test").is_err());
        // Empty block + empty partition OK.
        assert!(validate_partition(&[], 0, "test").is_ok());
        // Valid partition.
        assert!(validate_partition(&[0..2, 2..5], 5, "test").is_ok());
        assert!(validate_partition(&[0..5], 5, "test").is_ok());
    }

    /// Phase-4b application step: take the V matrices from
    /// `compile_survival_parametric_designs` and apply them to raw
    /// designs + penalties via
    /// `apply_survival_parametric_compile_to_designs`. Verify the
    /// produced `CompiledSurvivalDesigns` has consistent widths
    /// across the time triplet, the marginal/logslope singletons,
    /// and their pulled-back penalty matrices.
    #[test]
    fn apply_compile_produces_width_consistent_designs_and_penalties() {
        use crate::families::custom_family::PenaltyMatrix;
        use crate::linalg::matrix::DenseDesignMatrix;

        let n = 16;
        let p_time = 3;
        let p_marginal = 3;
        let p_logslope = 2;
        let x: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0)).collect();
        let mut time_dq0 = Array2::<f64>::zeros((n, p_time));
        let mut time_dq1 = Array2::<f64>::zeros((n, p_time));
        let mut time_dqd1 = Array2::<f64>::zeros((n, p_time));
        let mut marg_dq = Array2::<f64>::zeros((n, p_marginal));
        let marg_dqd1 = Array2::<f64>::zeros((n, p_marginal));
        let mut log_dg = Array2::<f64>::zeros((n, p_logslope));
        for i in 0..n {
            time_dq0[[i, 0]] = 1.0;
            time_dq0[[i, 1]] = x[i];
            time_dq0[[i, 2]] = x[i] * x[i];
            time_dq1[[i, 0]] = 1.0;
            time_dq1[[i, 1]] = x[i];
            time_dq1[[i, 2]] = x[i] * x[i];
            time_dqd1[[i, 0]] = 0.0;
            time_dqd1[[i, 1]] = 1.0;
            time_dqd1[[i, 2]] = 2.0 * x[i];
            marg_dq[[i, 0]] = 1.0;
            marg_dq[[i, 1]] = x[i] * x[i] * x[i];
            marg_dq[[i, 2]] = x[i].sin();
            log_dg[[i, 0]] = (2.0 * x[i]).cos();
            log_dg[[i, 1]] = x[i].tanh();
        }
        let mut h_full = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            for k in 0..K_SURVIVAL {
                h_full[[i, k, k]] = 1.0;
            }
        }
        let row_hess = SurvivalRowHessian::from_full(h_full);
        let compiled = compile_survival_parametric_designs(
            time_dq0.clone(),
            time_dq1.clone(),
            time_dqd1.clone(),
            marg_dq.clone(),
            marg_dqd1.clone(),
            log_dg.clone(),
            &row_hess,
        )
        .expect("compile must succeed");

        // Build raw DesignMatrix wrappers around the same dense data
        // for the apply step (in production these come from the
        // family's design accumulation; here we re-use the dense
        // matrices we already built for the operator construction).
        let raw_time_entry = DesignMatrix::Dense(DenseDesignMatrix::from(time_dq0.clone()));
        let raw_time_exit = DesignMatrix::Dense(DenseDesignMatrix::from(time_dq1.clone()));
        let raw_time_deriv = DesignMatrix::Dense(DenseDesignMatrix::from(time_dqd1.clone()));
        let raw_marg = DesignMatrix::Dense(DenseDesignMatrix::from(marg_dq.clone()));
        let raw_log = DesignMatrix::Dense(DenseDesignMatrix::from(log_dg.clone()));

        // Penalties: simple diagonal placeholders at raw width so we
        // can verify the pulled-back result has the expected shape.
        let time_pens = vec![PenaltyMatrix::Dense(Array2::<f64>::from_shape_fn(
            (p_time, p_time),
            |(i, j)| if i == j { (i + 1) as f64 } else { 0.0 },
        ))];
        let marg_pens = vec![PenaltyMatrix::Dense(Array2::<f64>::from_shape_fn(
            (p_marginal, p_marginal),
            |(i, j)| if i == j { (i + 1) as f64 } else { 0.0 },
        ))];
        let log_pens = vec![PenaltyMatrix::Dense(Array2::<f64>::from_shape_fn(
            (p_logslope, p_logslope),
            |(i, j)| if i == j { (i + 1) as f64 } else { 0.0 },
        ))];

        let out = apply_survival_parametric_compile_to_designs(
            &compiled,
            raw_time_entry,
            raw_time_exit,
            raw_time_deriv,
            raw_marg,
            raw_log,
            &time_pens,
            &marg_pens,
            &log_pens,
        )
        .expect("apply must succeed");

        // Time triplet: all three designs share V_time, so all three
        // have the same compiled width = V_time.ncols() = p_time (no
        // drops on time block in this scenario).
        assert_eq!(out.time_design_entry.ncols(), compiled.v_time.ncols());
        assert_eq!(out.time_design_exit.ncols(), compiled.v_time.ncols());
        assert_eq!(out.time_design_derivative_exit.ncols(), compiled.v_time.ncols());

        // Marginal / logslope: widths equal their V's column count.
        assert_eq!(out.marginal_design.ncols(), compiled.v_marginal.ncols());
        assert_eq!(out.logslope_design.ncols(), compiled.v_logslope.ncols());

        // Penalty pullbacks: each penalty matrix is (p_kept × p_kept).
        for s in &out.time_penalties {
            let dense = s.as_dense_cow();
            assert_eq!(dense.dim(), (compiled.v_time.ncols(), compiled.v_time.ncols()));
        }
        for s in &out.marginal_penalties {
            let dense = s.as_dense_cow();
            assert_eq!(
                dense.dim(),
                (compiled.v_marginal.ncols(), compiled.v_marginal.ncols())
            );
        }
        for s in &out.logslope_penalties {
            let dense = s.as_dense_cow();
            assert_eq!(
                dense.dim(),
                (compiled.v_logslope.ncols(), compiled.v_logslope.ncols())
            );
        }

        // Row count of every design must equal n.
        assert_eq!(out.time_design_entry.nrows(), n);
        assert_eq!(out.marginal_design.nrows(), n);
        assert_eq!(out.logslope_design.nrows(), n);
    }

    /// Top-level Phase-4b API test for the SMGS parametric path:
    /// call `compile_survival_parametric_designs` on a shared-constant
    /// alias between time and marginal, with an identity row Hessian.
    /// Verify the returned `v_*` matrices have the expected widths
    /// (time keeps all 3, marginal loses 1, logslope keeps both) and
    /// `drops_by_block` reports `(0, 1, 0)`.
    #[test]
    fn compile_survival_parametric_designs_helper_attributes_drop_to_marginal() {
        let n = 24;
        let p_time = 3;
        let p_marginal = 3;
        let p_logslope = 2;
        let x: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0)).collect();
        let mut time_dq0 = Array2::<f64>::zeros((n, p_time));
        let mut time_dq1 = Array2::<f64>::zeros((n, p_time));
        let mut time_dqd1 = Array2::<f64>::zeros((n, p_time));
        let mut marg_dq = Array2::<f64>::zeros((n, p_marginal));
        let marg_dqd1 = Array2::<f64>::zeros((n, p_marginal));
        let mut log_dg = Array2::<f64>::zeros((n, p_logslope));
        for i in 0..n {
            time_dq0[[i, 0]] = 1.0;
            time_dq0[[i, 1]] = x[i];
            time_dq0[[i, 2]] = x[i] * x[i];
            time_dq1[[i, 0]] = 1.0;
            time_dq1[[i, 1]] = x[i];
            time_dq1[[i, 2]] = x[i] * x[i];
            time_dqd1[[i, 0]] = 0.0;
            time_dqd1[[i, 1]] = 1.0;
            time_dqd1[[i, 2]] = 2.0 * x[i];
            marg_dq[[i, 0]] = 1.0; // alias with time col 0
            marg_dq[[i, 1]] = x[i] * x[i] * x[i];
            marg_dq[[i, 2]] = x[i].sin();
            log_dg[[i, 0]] = (2.0 * x[i]).cos();
            log_dg[[i, 1]] = x[i].tanh();
        }
        let mut h_full = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            for k in 0..K_SURVIVAL {
                h_full[[i, k, k]] = 1.0;
            }
        }
        let row_hess = SurvivalRowHessian::from_full(h_full);
        let out = compile_survival_parametric_designs(
            time_dq0, time_dq1, time_dqd1, marg_dq, marg_dqd1, log_dg, &row_hess,
        )
        .expect("Phase-4b parametric compile must succeed on single-direction alias");
        assert_eq!(out.v_time.ncols(), p_time, "time keeps all columns");
        assert_eq!(
            out.v_marginal.ncols(),
            p_marginal - 1,
            "marginal loses exactly the shared-constant direction"
        );
        assert_eq!(out.v_logslope.ncols(), p_logslope, "logslope is clean");
        assert_eq!(
            out.drops_by_block,
            (0, 1, 0),
            "attribution: zero from time/logslope, one from marginal",
        );
    }

    /// End-to-end Phase-4b smoke test: build the full 3-block survival
    /// parametric operator stack (time + marginal + logslope) with a
    /// shared-constant alias seeded between the time and marginal
    /// blocks, feed it into `compile()` with an identity 4×4 row
    /// Hessian on every row, and verify the compiler:
    ///
    ///   (1) returns a [`CompiledBlocks`] with one block per input;
    ///   (2) preserves all 3 columns of the highest-priority `Time`
    ///       block in `t_lw` (the time block enters first in the
    ///       ordering, so its full column span survives);
    ///   (3) drops exactly one direction from `Marginal` (the
    ///       constant aliased with the time intercept), leaving its
    ///       remaining columns in `t_lw`;
    ///   (4) reports `joint_rank` = (raw_total - 1).
    ///
    /// This validates the Phase-4b construction-time orthogonalisation
    /// path on the survival K=4 row primary state without requiring
    /// the SMGS construction site to be wired yet (which is what the
    /// active sibling integration handles). The shape of the result
    /// is the contract the SMGS wiring will assert against.
    #[test]
    fn compile_survival_three_block_with_shared_constant_drops_one_direction() {
        use crate::families::identifiability_compiler::compile;

        let n = 32;
        let p_time = 3;
        let p_marginal = 3;
        let p_logslope = 2;

        // Time block:
        //   col 0 = ones (the shared constant — aliases marginal col 0);
        //   col 1 = linear x;
        //   col 2 = quadratic x².
        // q0/q1 share the same design (so the alias surfaces in both
        // the entry and exit primary channels); qd1 is the derivative
        // of the design w.r.t. time at the exit point, which for the
        // constant column is exactly zero (the gauge identity that
        // makes the constant a true null direction under (q0, q1, qd1)
        // joint).
        let x: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0)).collect();
        let mut time_dq0 = Array2::<f64>::zeros((n, p_time));
        let mut time_dq1 = Array2::<f64>::zeros((n, p_time));
        let mut time_dqd1 = Array2::<f64>::zeros((n, p_time));
        for i in 0..n {
            time_dq0[[i, 0]] = 1.0;
            time_dq0[[i, 1]] = x[i];
            time_dq0[[i, 2]] = x[i] * x[i];
            time_dq1[[i, 0]] = 1.0;
            time_dq1[[i, 1]] = x[i];
            time_dq1[[i, 2]] = x[i] * x[i];
            // d/dt of a constant = 0; d/dt of x ≡ 1; d/dt of x² ≡ 2x.
            time_dqd1[[i, 0]] = 0.0;
            time_dqd1[[i, 1]] = 1.0;
            time_dqd1[[i, 2]] = 2.0 * x[i];
        }

        // Marginal block (q-channel only; qd1 contribution zero — no
        // timewiggle in this scenario):
        //   col 0 = ones (the shared constant);
        //   col 1 = x³;
        //   col 2 = sin(x).
        let mut marg_dq = Array2::<f64>::zeros((n, p_marginal));
        let marg_dqd1 = Array2::<f64>::zeros((n, p_marginal));
        for i in 0..n {
            marg_dq[[i, 0]] = 1.0;
            marg_dq[[i, 1]] = x[i] * x[i] * x[i];
            marg_dq[[i, 2]] = x[i].sin();
        }

        // Logslope block (g-channel only):
        //   col 0 = cos(2x);
        //   col 1 = tanh(x).  (no shared constant — logslope is clean)
        let mut log_dg = Array2::<f64>::zeros((n, p_logslope));
        for i in 0..n {
            log_dg[[i, 0]] = (2.0 * x[i]).cos();
            log_dg[[i, 1]] = x[i].tanh();
        }

        let inputs = build_survival_compiler_inputs(
            time_dq0,
            time_dq1,
            time_dqd1,
            marg_dq,
            marg_dqd1,
            log_dg,
            None,
            None,
        );

        // Identity 4×4 row Hessian on every row. With H_i = I the
        // sqrt-H metric collapses to the standard Frobenius metric,
        // so the compiler's residualisation is ordinary least-squares
        // projection — exactly what we want for verifying the
        // structural rank-deficiency attribution.
        let mut h_full = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            for k in 0..K_SURVIVAL {
                h_full[[i, k, k]] = 1.0;
            }
        }
        let row_hess = SurvivalRowHessian::from_full(h_full);

        let compiled = compile(&inputs.operators, &row_hess, &inputs.ordering)
            .expect("survival 3-block compile must succeed; aliasing is single-direction");

        // (1) One CompiledBlock per input.
        assert_eq!(compiled.blocks.len(), 3, "expected 3 CompiledBlocks");

        // (2) Time enters first; under sqrt-I metric every column of
        // the time block is residual-vs-empty-anchor and therefore
        // survives the eigendecomposition with positive eigenvalue.
        // V_time has p_time columns.
        let v_time = &compiled.blocks[0].t_lw;
        assert_eq!(
            v_time.ncols(),
            p_time,
            "time block (first in ordering) must retain all {p_time} of its columns; V_time={:?}",
            v_time.dim(),
        );

        // (3) Marginal enters second. Its constant column is aliased
        // with time's constant column in (q0, q1) and contributes zero
        // to qd1. After residualising against the time anchor in the
        // K=4 stacked metric, the residual Gram has rank
        // p_marginal − 1 (one direction collapsed by the alias). So
        // V_marginal has exactly (p_marginal − 1) columns.
        let v_marg = &compiled.blocks[1].t_lw;
        assert_eq!(
            v_marg.ncols(),
            p_marginal - 1,
            "marginal block must lose exactly the shared-constant direction; \
             V_marginal cols = {}, expected {}",
            v_marg.ncols(),
            p_marginal - 1,
        );

        // (4) Logslope enters third and carries no shared direction
        // with time or marginal in the g-channel. Both columns survive.
        let v_log = &compiled.blocks[2].t_lw;
        assert_eq!(
            v_log.ncols(),
            p_logslope,
            "logslope block (no shared direction) must retain all {p_logslope} columns",
        );

        // (5) Joint rank consistency: sum of compiled column counts
        // equals raw_total minus the one aliased direction.
        let raw_total = p_time + p_marginal + p_logslope;
        let kept_total: usize = compiled.blocks.iter().map(|b| b.t_lw.ncols()).sum();
        assert_eq!(
            kept_total,
            raw_total - 1,
            "joint kept = raw_total − aliased; got {kept_total}, expected {}",
            raw_total - 1,
        );
        assert_eq!(
            compiled.joint_rank, kept_total,
            "CompiledBlocks::joint_rank must match the sum of per-block t_lw widths",
        );
    }

    /// `T = I` case: per-block V = identity, R = None. The triangular
    /// lift must be the identity on each block.
    #[test]
    fn smgs_lift_via_t_identity_passes_through() {
        let v0 = Array2::<f64>::eye(3);
        let v1 = Array2::<f64>::eye(2);
        let v_per_term = vec![v0, v1];
        let r_per_term: Vec<Option<Array2<f64>>> = vec![None, None];
        let lift = SmgsLiftViaT::from_v_and_r(&v_per_term, &r_per_term);
        assert_eq!(lift.t_full.dim(), (5, 5));
        assert_eq!(lift.block_starts_compiled, vec![0, 3, 5]);
        assert_eq!(lift.block_starts_raw, vec![0, 3, 5]);
        for i in 0..5 {
            for j in 0..5 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((lift.t_full[[i, j]] - want).abs() < 1e-14);
            }
        }
        let theta_0 = Array1::from(vec![1.0_f64, -2.0, 3.5]);
        let theta_1 = Array1::from(vec![-0.5_f64, 7.0]);
        let lifted = lift.lift_block_betas_via_t(&[theta_0.clone(), theta_1.clone()]);
        assert_eq!(lifted.len(), 2);
        for (a, b) in theta_0.iter().zip(lifted[0].iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        for (a, b) in theta_1.iter().zip(lifted[1].iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    /// Two-block toy: V_a = I_3, V_b drops the middle column, R is a
    /// non-trivial residualised reparam. Verify β_a_raw = θ_a − R · θ_b
    /// and β_b_raw = V_b · θ_b.
    #[test]
    fn smgs_lift_via_t_two_block_with_residualisation() {
        let v_a = Array2::<f64>::eye(3);
        let mut v_b = Array2::<f64>::zeros((3, 2));
        v_b[[0, 0]] = 1.0;
        v_b[[2, 1]] = 1.0;
        let mut r_b = Array2::<f64>::zeros((3, 2));
        r_b[[0, 0]] = 0.4;
        r_b[[0, 1]] = -0.1;
        r_b[[1, 0]] = 0.7;
        r_b[[1, 1]] = 1.3;
        r_b[[2, 0]] = -0.2;
        r_b[[2, 1]] = 0.5;
        let lift = SmgsLiftViaT::from_v_and_r(
            &[v_a.clone(), v_b.clone()],
            &[None, Some(r_b.clone())],
        );
        assert_eq!(lift.t_full.dim(), (6, 5));
        assert_eq!(lift.block_starts_compiled, vec![0, 3, 5]);
        assert_eq!(lift.block_starts_raw, vec![0, 3, 6]);

        let theta_a = Array1::from(vec![1.0_f64, 2.0, -1.5]);
        let theta_b = Array1::from(vec![0.5_f64, -0.25]);
        let lifted = lift.lift_block_betas_via_t(&[theta_a.clone(), theta_b.clone()]);
        let r_theta_b = r_b.dot(&theta_b);
        let expected_a = &theta_a - &r_theta_b;
        assert_eq!(lifted[0].len(), 3);
        for (got, want) in lifted[0].iter().zip(expected_a.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
        assert_eq!(lifted[1].len(), 3);
        assert!((lifted[1][0] - theta_b[0]).abs() < 1e-12);
        assert!(lifted[1][1].abs() < 1e-12);
        assert!((lifted[1][2] - theta_b[1]).abs() < 1e-12);
    }

    /// When all R's are None, lift_block_betas_via_t must equal the
    /// per-block V · θ lift produced by `SmgsLiftPerBlockV`.
    #[test]
    fn smgs_lift_via_t_zero_r_matches_per_block_v_lift() {
        let mut v_a = Array2::<f64>::zeros((3, 2));
        v_a[[0, 0]] = 0.6;
        v_a[[1, 0]] = -0.8;
        v_a[[1, 1]] = 0.3;
        v_a[[2, 1]] = 0.9;
        let mut v_b = Array2::<f64>::zeros((4, 3));
        v_b[[0, 0]] = 1.0;
        v_b[[1, 1]] = -0.4;
        v_b[[2, 0]] = 0.2;
        v_b[[2, 2]] = 0.7;
        v_b[[3, 2]] = -1.1;
        let v_per_term = vec![v_a.clone(), v_b.clone()];
        let lift = SmgsLiftViaT::from_v_and_r(&v_per_term, &[None, None]);
        let theta_a = Array1::from(vec![0.3_f64, -1.4]);
        let theta_b = Array1::from(vec![2.1_f64, 0.0, -0.7]);
        let via_t = lift.lift_block_betas_via_t(&[theta_a.clone(), theta_b.clone()]);
        let ref_a = v_a.dot(&theta_a);
        let ref_b = v_b.dot(&theta_b);
        assert_eq!(via_t[0].len(), ref_a.len());
        for (g, w) in via_t[0].iter().zip(ref_a.iter()) {
            assert!((g - w).abs() < 1e-12);
        }
        assert_eq!(via_t[1].len(), ref_b.len());
        for (g, w) in via_t[1].iter().zip(ref_b.iter()) {
            assert!((g - w).abs() < 1e-12);
        }

        let per_block = SmgsLiftPerBlockV { v_per_block: v_per_term };
        let mut block_betas = vec![theta_a, theta_b];
        per_block.lift_block_betas(&mut block_betas);
        for (got, want) in via_t[0].iter().zip(block_betas[0].iter()) {
            assert!((got - want).abs() < 1e-12);
        }
        for (got, want) in via_t[1].iter().zip(block_betas[1].iter()) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    /// 2-block synthetic V+M-exact end-to-end test. Construct a small
    /// per-term compiled struct (1 time term + 1 marginal term, no
    /// logslope), invoke `apply_per_term_vm_exact`, and verify that
    /// the produced designs have the expected compiled widths and
    /// `t_full` is shape-consistent (rows = raw joint width, cols =
    /// compiled joint width) and block-decomposes with V_b on the
    /// diagonal.
    #[test]
    fn apply_per_term_vm_exact_produces_consistent_widths_and_t_full() {
        use crate::families::custom_family::PenaltyMatrix;
        use crate::linalg::matrix::DenseDesignMatrix;
        use crate::terms::smooth::BlockwisePenalty;
        use ndarray::Array2;
        use std::ops::Range;
        let n = 8usize;
        let p_time = 2usize;
        let p_marg = 2usize;
        let p_log = 0usize;
        let time_partition: Vec<Range<usize>> = vec![0..p_time];
        let marg_partition: Vec<Range<usize>> = vec![0..p_marg];
        let log_partition: Vec<Range<usize>> = Vec::new();

        // V_time = identity (2×2); V_marginal = drops one column (2×1).
        let v_time = Array2::<f64>::eye(p_time);
        let mut v_marg = Array2::<f64>::zeros((p_marg, 1));
        v_marg[[1, 0]] = 1.0;
        // r_lw for the first block is None (it's first); for the
        // second compiled-block (marginal-term-0) we supply a small
        // (p_time_kept × p_marg_kept) = (2 × 1) residualised reparam.
        let mut r_marg = Array2::<f64>::zeros((2, 1));
        r_marg[[0, 0]] = 0.25;
        r_marg[[1, 0]] = -0.5;
        let compiled = SurvivalParametricCompiledPerTerm {
            v_time_per_term: vec![v_time.clone()],
            v_marginal_per_term: vec![v_marg.clone()],
            v_logslope_per_term: Vec::new(),
            r_lw_per_term: vec![None, Some(r_marg.clone())],
            drops_by_block: (0, 1, 0),
        };

        // Raw designs (small).
        let mk = |p: usize, seed: usize| -> DesignMatrix {
            let m = Array2::<f64>::from_shape_fn((n, p), |(i, j)| {
                ((seed + 1) * (i + 1) + (j + 1) * 3) as f64 * 0.125
            });
            DesignMatrix::Dense(DenseDesignMatrix::from(m))
        };
        let time_entry = mk(p_time, 1);
        let time_exit = mk(p_time, 2);
        let time_deriv = mk(p_time, 3);
        let marg = mk(p_marg, 4);
        let log = if p_log == 0 {
            DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((n, 0))))
        } else {
            mk(p_log, 5)
        };

        // A simple identity-shaped per-term penalty over time and marginal.
        let time_pens = vec![BlockwisePenalty::new(0..p_time, Array2::<f64>::eye(p_time))];
        let marg_pens = vec![BlockwisePenalty::new(0..p_marg, Array2::<f64>::eye(p_marg))];
        let log_pens: Vec<BlockwisePenalty> = Vec::new();

        let out = apply_per_term_vm_exact(
            &compiled,
            &time_partition,
            &marg_partition,
            &log_partition,
            time_entry,
            time_exit,
            time_deriv,
            marg,
            log,
            &time_pens,
            &marg_pens,
            &log_pens,
        )
        .expect("apply_per_term_vm_exact must succeed on the 2-block synthetic");

        // Compiled widths:
        // - time-class designs have width = v_time.ncols() = 2 (single time term).
        // - marginal design has width = v_marg.ncols() = 1.
        // - logslope design has width = 0 (no terms).
        assert_eq!(out.time_design_entry.ncols(), v_time.ncols());
        assert_eq!(out.time_design_exit.ncols(), v_time.ncols());
        assert_eq!(out.time_design_derivative_exit.ncols(), v_time.ncols());
        assert_eq!(out.marginal_design.ncols(), v_marg.ncols());
        assert_eq!(out.logslope_design.ncols(), 0);

        // t_full shape: rows = raw joint width = p_time + p_marg + p_log = 4;
        // cols = compiled joint width = 2 + 1 + 0 = 3.
        let p_raw_total = p_time + p_marg + p_log;
        let p_compiled_total = v_time.ncols() + v_marg.ncols();
        assert_eq!(out.t_full.dim(), (p_raw_total, p_compiled_total));

        // T diagonal blocks should be V_b. With V_time=I (2×2) and
        // V_marg = (0, 1)^T (2×1), the diagonal blocks of t_full are:
        // - rows [0..2], cols [0..2] = V_time = I
        // - rows [2..4], cols [2..3] = V_marg
        for i in 0..p_time {
            for j in 0..v_time.ncols() {
                assert!(
                    (out.t_full[[i, j]] - v_time[[i, j]]).abs() < 1e-12,
                    "T diagonal block (time) mismatch at [{i},{j}]",
                );
            }
        }
        for i in 0..p_marg {
            let row_full = p_time + i;
            for j in 0..v_marg.ncols() {
                let col_full = v_time.ncols() + j;
                assert!(
                    (out.t_full[[row_full, col_full]] - v_marg[[i, j]]).abs() < 1e-12,
                    "T diagonal block (marginal) mismatch at [{row_full},{col_full}]",
                );
            }
        }

        // Penalty widths: each pulled-back penalty must be
        // p_compiled_total × p_compiled_total (full-width dense).
        for pen in out.time_penalties.iter().chain(out.marginal_penalties.iter()) {
            match pen {
                PenaltyMatrix::Dense(m) => {
                    assert_eq!(m.dim(), (p_compiled_total, p_compiled_total));
                }
                other => panic!("expected Dense full-width penalty, got {other:?}"),
            }
        }
    }
}
