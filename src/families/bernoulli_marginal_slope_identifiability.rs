//! Bernoulli marginal-slope concrete impls for the family-agnostic
//! identifiability compiler (`crate::families::identifiability_compiler`).
//!
//! Bernoulli's row primary state is the scalar linear predictor `η_i`, so
//! `K = 1` throughout. Every block's row Jacobian is the row of its dense
//! design matrix; the row Hessian is the standard probit IRLS weight
//!     `W_i = w_i · φ(η_i)² / (Φ(η_i) · Φ(−η_i))`
//! evaluated at the pilot η.
//!
//! Phase 4a delivery: trait impls + an input-builder. Phase 4b migrates the
//! BMS fit driver onto the compiler; Phase 4c deletes the legacy
//! `install_compiled_flex_block_into_runtime` path.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use crate::families::custom_family::FamilyChannelHessian;
use crate::families::identifiability_compiler::{
    AnchorRowEvaluator, BlockOrder, RowHessian, RowJacobianOperator, scale_jacobian_by_sqrt_h_with,
};
use crate::linalg::faer_ndarray::fast_ab;

/// Standard normal pdf.
#[inline]
fn phi(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (std::f64::consts::TAU).sqrt()
}

/// Standard normal cdf. Wrapper for the codebase's `normal_cdf`.
#[inline]
fn cdf(x: f64) -> f64 {
    crate::inference::probability::normal_cdf(x)
}

/// Probit IRLS row weight `w_i · φ(η)² / (Φ(η) · Φ(−η))`. Clamped strictly
/// positive: the residualised Gram must remain PSD even at extreme η pilots
/// (probit saturation collapses both `Φ(η)` and `Φ(−η)` toward zero).
fn probit_irls_weight(eta: f64, sample_weight: f64) -> f64 {
    let p = cdf(eta).clamp(f64::MIN_POSITIVE, 1.0 - f64::MIN_POSITIVE);
    let one_m = (1.0 - p).max(f64::MIN_POSITIVE);
    let phi_eta = phi(eta);
    let denom = (p * one_m).max(f64::MIN_POSITIVE);
    sample_weight * phi_eta * phi_eta / denom
}

/// Row Hessian for Bernoulli's K=1 row primary state. The "Hessian" is the
/// scalar IRLS weight per row at the pilot η.
pub struct BernoulliRowHessian {
    w: Array1<f64>,
}

impl BernoulliRowHessian {
    pub fn from_eta_pilot(eta_pilot: &Array1<f64>, sample_weights: &Array1<f64>) -> Self {
        assert_eq!(
            eta_pilot.len(),
            sample_weights.len(),
            "BernoulliRowHessian: eta_pilot length {} must match sample_weights length {}",
            eta_pilot.len(),
            sample_weights.len(),
        );
        let w = Array1::from_iter(
            eta_pilot
                .iter()
                .zip(sample_weights.iter())
                .map(|(&eta, &w)| probit_irls_weight(eta, w)),
        );
        Self { w }
    }

    /// Construct directly from a pre-computed row-weight vector (e.g. the
    /// existing `pilot_irls_hessian_row_metric_at_eta` output).
    pub fn from_row_weights(w: Array1<f64>) -> Self {
        Self { w }
    }

    /// Borrow the underlying per-row weight vector.
    pub fn row_weights(&self) -> &Array1<f64> {
        &self.w
    }
}

impl RowHessian for BernoulliRowHessian {
    fn k(&self) -> usize {
        1
    }
    fn nrows(&self) -> usize {
        self.w.len()
    }
    fn fill_row(&self, row: usize, out: &mut [f64]) {
        assert_eq!(out.len(), 1, "BernoulliRowHessian::fill_row expects K=1");
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

/// `FamilyChannelHessian` for Bernoulli marginal-slope.
///
/// BMS has a single output channel (K=1). The per-subject channel Hessian
/// W_i is the scalar probit IRLS weight:
///
/// ```text
/// W_i = w_i · φ(η_i)² / (Φ(η_i) · (1 − Φ(η_i)))
/// ```
///
/// This is exactly the 1×1 scalar stored in `BernoulliRowHessian::w`.
/// Since K=1, the scalar fast path is used and cross-channel curvature
/// is vacuous. Families that genuinely have a single output channel
/// (Gaussian, Binomial, Poisson, etc.) all use this 1×1 identity path.
impl FamilyChannelHessian for BernoulliRowHessian {
    fn n_outputs(&self) -> usize {
        1
    }

    fn n_subjects(&self) -> usize {
        self.w.len()
    }

    fn fill_subject(&self, i: usize, out: &mut [f64]) {
        assert_eq!(
            out.len(),
            1,
            "BernoulliRowHessian::fill_subject expects K=1"
        );
        out[0] = self.w[i];
    }

    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        let n = self.w.len();
        let mut out = ndarray::Array3::<f64>::zeros((n, 1, 1));
        for i in 0..n {
            out[[i, 0, 0]] = self.w[i];
        }
        out
    }
}

/// Row Jacobian operator backed by a dense design matrix. K=1 — the only
/// channel is `δη = design.row(i) · δβ`. Covers BMS's marginal, logslope,
/// score-warp, and link-deviation blocks uniformly.
pub struct BernoulliDenseDesignOperator {
    design: Array2<f64>,
}

impl BernoulliDenseDesignOperator {
    pub fn new(design: Array2<f64>) -> Self {
        Self { design }
    }
}

impl RowJacobianOperator for BernoulliDenseDesignOperator {
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
        assert_eq!(delta_beta.len(), self.design.ncols());
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
    fn scaled_design_by_sqrt_h(&self, h_full: &Array3<f64>) -> Array2<f64> {
        // K=1: the only channel is `δη = design.row(i)·δβ`. Scale straight from
        // the stored `(n, p)` design rather than reshaping it into a `(n, p, 1)`
        // tensor first. (#738: a capability is not a representation.)
        let n = self.design.nrows();
        let p = self.design.ncols();
        scale_jacobian_by_sqrt_h_with(n, p, 1, h_full, |i, a, c| {
            debug_assert_eq!(c, 0, "K=1 operator has only channel 0");
            self.design[[i, a]]
        })
    }
}

/// Predict-time anchor row evaluator for a parametric block. Returns the
/// supplied dense design as-is — the predict-time argument is unused
/// because the design is already materialised at the requested rows.
///
/// Phase 4b will introduce a "build-from-predict-arg" variant that recomputes
/// the parametric design at predict rows via the family's design constructor.
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

/// Predict-time anchor row evaluator for a compiled flex block. Composes
/// the residualised row operator `C(x)·V − A(x)·M` where `C(x)` is the raw
/// span-basis at the predict argument, `V` is `t_lw`, `A(x)` is the parent
/// anchor evaluator's output, and `M` is `anchor_correction`.
///
/// Phase 4a delivery: the constructor takes a raw-basis closure plus a
/// parent evaluator and the compiled block's V/M. Phase 4b wires this up to
/// `DeviationRuntime::span_basis_at` so flex anchors compose naturally in
/// downstream block residualisation.
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

/// Build a stack of row-Jacobian operators from raw dense designs in the
/// order the BMS fit driver presents them. Mirrors the `[marginal, logslope,
/// score_warp, link_dev]` ordering implied by `gauge_priority` (parametric
/// blocks at 100 ahead of the flex bases at lower priorities).
///
/// `score_warp_design` / `link_dev_design` are `None` when the corresponding
/// flex block is inactive. Operators are returned in the same order as the
/// returned `BlockOrder` tags; Phase 4b's call-site update will route the
/// compiled outputs back to the right runtime slot via these tags.
pub fn build_bernoulli_compiler_inputs(
    marginal_design: Array2<f64>,
    logslope_design: Array2<f64>,
    score_warp_design: Option<Array2<f64>>,
    link_dev_design: Option<Array2<f64>>,
) -> (Vec<Arc<dyn RowJacobianOperator>>, Vec<BlockOrder>) {
    let mut ops: Vec<Arc<dyn RowJacobianOperator>> = Vec::with_capacity(4);
    let mut order: Vec<BlockOrder> = Vec::with_capacity(4);

    ops.push(Arc::new(BernoulliDenseDesignOperator::new(marginal_design)));
    order.push(BlockOrder::Marginal);
    ops.push(Arc::new(BernoulliDenseDesignOperator::new(logslope_design)));
    order.push(BlockOrder::Logslope);
    if let Some(sw) = score_warp_design {
        ops.push(Arc::new(BernoulliDenseDesignOperator::new(sw)));
        order.push(BlockOrder::ScoreWarp);
    }
    if let Some(ld) = link_dev_design {
        ops.push(Arc::new(BernoulliDenseDesignOperator::new(ld)));
        order.push(BlockOrder::LinkDev);
    }
    (ops, order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_row_hessian_matches_probit_irls_weight() {
        let eta = Array1::from(vec![-1.5_f64, 0.0, 0.75, 2.0]);
        let w = Array1::from(vec![1.0_f64; 4]);
        let hess = BernoulliRowHessian::from_eta_pilot(&eta, &w);
        for i in 0..eta.len() {
            let want = probit_irls_weight(eta[i], 1.0);
            let got = hess.row_weights()[i];
            assert!(
                (got - want).abs() < 1e-14,
                "row {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn dense_design_operator_evaluate_full_shape() {
        let design = Array2::from_shape_fn((5, 3), |(i, j)| (i as f64) * 0.1 + (j as f64));
        let op = BernoulliDenseDesignOperator::new(design.clone());
        let full = op.evaluate_full();
        assert_eq!(full.shape(), &[5, 3, 1]);
        for i in 0..5 {
            for j in 0..3 {
                assert_eq!(full[[i, j, 0]], design[[i, j]]);
            }
        }
    }

    #[test]
    fn parametric_anchor_evaluator_returns_design_verbatim() {
        let design = Array2::from_shape_fn((4, 2), |(i, j)| (i + j) as f64);
        let ev = ParametricAnchorEvaluator::new(design.clone());
        let predict_arg = Array1::from(vec![0.0_f64; 4]);
        let rows = ev.anchor_rows(&predict_arg).expect("anchor_rows ok");
        assert_eq!(rows, design);
    }
}
