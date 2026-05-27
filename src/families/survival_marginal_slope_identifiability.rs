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
use crate::linalg::matrix::{CoefficientTransformOperator, DenseDesignMatrix, DesignMatrix};
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
}
