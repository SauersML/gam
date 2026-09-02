//! Bernoulli marginal-slope concrete impls for the family-agnostic
//! identifiability compiler (`crate::families::compiler`).
//!
//! Bernoulli's row primary state is the scalar linear predictor `η_i`, so
//! `K = 1` throughout. Every block's row Jacobian is the row of its dense
//! design matrix; the row Hessian is the standard probit IRLS weight
//!     `W_i = w_i · φ(η_i)² / (Φ(η_i) · Φ(−η_i))`
//! evaluated at the pilot η.
//!
//! These concrete impls (`BernoulliRowHessian`, `BernoulliDenseDesignOperator`)
//! feed the live BMS fit driver via `bms::install_flex`, whose
//! `install_compiled_flex_block_into_runtime` is the entry point that
//! residualises each flex block against the compiled parametric anchors.

use ndarray::{Array1, Array2, Array3};

use crate::families::compiler::{RowHessian, RowJacobianOperator, scale_jacobian_by_sqrt_h_with};
use gam_problem::FamilyChannelHessian;

/// Row Hessian for Bernoulli's K=1 row primary state. The "Hessian" is the
/// scalar IRLS weight per row at the pilot η.
pub struct BernoulliRowHessian {
    w: Array1<f64>,
}

impl BernoulliRowHessian {

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

    fn fill_subject(&self, i: usize, out: &mut [f64]) {
        assert_eq!(
            out.len(),
            1,
            "BernoulliRowHessian::fill_subject expects K=1"
        );
        out[0] = self.w[i];
    }

    fn n_subjects(&self) -> usize {
        self.w.len()
    }
    fn n_outputs(&self) -> usize {
        1
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
/// channel is `δη = design.row(i) · δβ`. Covers BMS's marginal, slope,
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
            assert_eq!(c, 0, "K=1 operator has only channel 0");
            self.design[[i, a]]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
