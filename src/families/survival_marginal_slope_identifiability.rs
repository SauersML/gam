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
//! `enforce_cross_block_identifiability_for_flex_block` path.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use crate::families::identifiability_compiler::{
    AnchorRowEvaluator, BlockOrder, RowHessian, RowJacobianOperator,
};
use crate::linalg::faer_ndarray::{FaerEigh, fast_ab};
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
    fn anchor_rows(&self, _predict_arg: &Array1<f64>) -> Result<Array2<f64>, String> {
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
}
