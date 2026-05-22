//! Operator-form penalty interface.
//!
//! Defines the `PenaltyOp` trait that abstracts a square symmetric PSD penalty
//! operator. Two concrete implementations live alongside:
//!   * `Array2<f64>` (via blanket `impl PenaltyOp for Array2<f64>`) — adapts a
//!     materialized dense penalty into the operator interface.
//!   * `ClosedFormPenaltyOperator` — implements the trait with analytic,
//!     streaming pair-kernel matvecs and only materializes when `as_dense()` is
//!     explicitly requested.
//!
//! See `matrix_free_penalty_integration_assessment.md` for why the operator
//! does not have a "true matrix-free" backing implementation in our K range
//! and why this trait is still worth threading through PIRLS/REML: the
//! wallclock win lives at the *Hessian* level (PCG-against-implicit-H). The
//! closed-form Duchon operator is also matrix-free so large K paths avoid
//! accidental dense Gram construction in matvec/log-det probes.

use std::sync::Arc;

use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

use crate::linalg::faer_ndarray::{FaerEigh, fast_av_view_into};
use crate::terms::closed_form_operator::ClosedFormPenaltyOperator;

/// Square symmetric PSD penalty operator.
///
/// Implementations may be backed by a materialized `Array2<f64>` or by a
/// closed-form operator that builds (and caches) its dense form lazily. All
/// methods must be consistent with the same underlying matrix `S`:
/// `matvec(w) = S w`, `diag()[i] = S[i,i]`, etc.
pub trait PenaltyOp: Send + Sync {
    /// Side length of the (square) operator.
    fn dim(&self) -> usize;

    /// Apply the operator: `out = S w`.
    fn matvec(&self, w: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>);

    /// Diagonal entries `S[i,i]`.
    fn diag(&self) -> Array1<f64>;

    /// Trace `tr(S) = Σ_i S[i,i]`.
    fn trace(&self) -> f64 {
        self.diag().sum()
    }

    /// Exact `log det(S + λI)` for `λ > 0`.
    /// `S` is allowed to be rank-deficient; the regularization makes the
    /// regularized operator strictly positive definite.
    fn log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String>;

    /// Symmetric eigendecomposition `S = V diag(λ) V^T`. The default
    /// implementation materializes via `as_dense` and runs the same
    /// `FaerEigh` path the existing dense pipeline uses, which preserves
    /// numerical agreement with `analyze_penalty_block`. Implementations
    /// that have a faster path (Lanczos top-k for very large K) may
    /// override.
    fn eigendecompose(&self) -> Result<(Array1<f64>, Array2<f64>), String> {
        let dense = self.as_dense();
        dense
            .eigh(Side::Lower)
            .map_err(|e| format!("PenaltyOp::eigendecompose: {e}"))
    }

    /// Materialize the operator as a dense matrix. Required for the
    /// existing `analyze_penalty_block` path and for callers that need a
    /// `&Array2` view (Cholesky factorization, etc.). Implementations that
    /// already hold a dense form should return it cheaply.
    fn as_dense(&self) -> Array2<f64>;
}

impl PenaltyOp for Array2<f64> {
    fn dim(&self) -> usize {
        debug_assert_eq!(
            self.nrows(),
            self.ncols(),
            "PenaltyOp matrix must be square"
        );
        self.nrows()
    }

    fn matvec(&self, w: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
        fast_av_view_into(self, &w, out);
    }

    fn diag(&self) -> Array1<f64> {
        let n = self.nrows();
        let mut d = Array1::<f64>::zeros(n);
        for i in 0..n {
            d[i] = self[[i, i]];
        }
        d
    }

    fn log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String> {
        assert!(lambda > 0.0, "log_det_plus_lambda_i requires λ > 0");
        let n = <Self as PenaltyOp>::dim(self);
        let mut regularized = self.clone();
        for i in 0..n {
            regularized[[i, i]] += lambda;
        }
        let (evals, _) = regularized.eigh(Side::Lower).map_err(|e| {
            format!("PenaltyOp::log_det_plus_lambda_i eigendecomposition failed: {e}")
        })?;
        let mut logdet = 0.0;
        for (idx, &ev) in evals.iter().enumerate() {
            if !ev.is_finite() || ev <= 0.0 {
                return Err(format!(
                    "PenaltyOp::log_det_plus_lambda_i expected SPD S+λI, \
                     eigenvalue {idx} is {ev:.3e}"
                ));
            }
            logdet += ev.ln();
        }
        Ok(logdet)
    }

    fn as_dense(&self) -> Array2<f64> {
        self.clone()
    }
}

impl PenaltyOp for ClosedFormPenaltyOperator {
    fn dim(&self) -> usize {
        self.dim()
    }

    fn matvec(&self, w: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
        self.matvec(w, out)
    }

    fn diag(&self) -> Array1<f64> {
        self.diag()
    }

    fn trace(&self) -> f64 {
        self.trace()
    }

    fn log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String> {
        self.log_det_plus_lambda_i(lambda)
    }

    fn as_dense(&self) -> Array2<f64> {
        self.dense_form()
    }
}

/// Wrap any `PenaltyOp` with a scalar multiplier. Useful when the dense
/// `PenaltyCandidate.matrix` has been normalized by a Frobenius factor `norm`
/// and we need an operator whose `as_dense()` matches it bit-for-bit. The
/// adapter divides every matvec / diag / trace result by `norm` (equivalently:
/// scales by `1/norm`).
pub struct ScaledPenaltyOp {
    inner: Arc<dyn PenaltyOp>,
    scale: f64,
}

impl ScaledPenaltyOp {
    pub fn new(inner: Arc<dyn PenaltyOp>, scale: f64) -> Self {
        Self { inner, scale }
    }
}

impl PenaltyOp for ScaledPenaltyOp {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn matvec(&self, w: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        self.inner.matvec(w, out.view_mut());
        out.mapv_inplace(|v| v * self.scale);
    }

    fn diag(&self) -> Array1<f64> {
        let mut d = self.inner.diag();
        d.mapv_inplace(|v| v * self.scale);
        d
    }

    fn trace(&self) -> f64 {
        self.inner.trace() * self.scale
    }

    fn log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String> {
        // log det(scale * S + λ I) cannot be derived from log det(S + (λ/scale) I)
        // by a uniform shift unless we materialize. Materialize via as_dense and
        // call the exact Array2 implementation on the scaled dense matrix.
        let dense = self.as_dense();
        <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
    }

    fn as_dense(&self) -> Array2<f64> {
        let mut m = self.inner.as_dense();
        m.mapv_inplace(|v| v * self.scale);
        m
    }
}

/// Concrete carrier of a penalty within the candidate / canonical-block
/// pipeline. The `Dense` variant carries an owned `Array2<f64>` and is the
/// default; the `Operator` variant carries an `Arc<dyn PenaltyOp>` so the
/// operator may be reused across consumers (eigendecomposition, PCG matvec,
/// log-det). Construction sites that build penalties from closed-form factory
/// routines (`operator_penalty_candidates_closed_form{,_pure}`) may emit
/// `Operator` form when the size threshold warrants it; below threshold the
/// dense form is preserved.
#[derive(Clone)]
pub enum PenaltyForm {
    Dense(Array2<f64>),
    Operator(Arc<dyn PenaltyOp>),
}

impl PenaltyForm {
    pub fn dim(&self) -> usize {
        match self {
            PenaltyForm::Dense(m) => {
                debug_assert_eq!(m.nrows(), m.ncols());
                m.nrows()
            }
            PenaltyForm::Operator(op) => op.dim(),
        }
    }

    /// Materialize the underlying matrix as `Array2<f64>`. Cheap for `Dense`
    /// (clone), and may be expensive for `Operator` (full assembly), but is
    /// always the same numerical result as the operator's matvec contract.
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            PenaltyForm::Dense(m) => m.clone(),
            PenaltyForm::Operator(op) => op.as_dense(),
        }
    }
}

impl std::fmt::Debug for PenaltyForm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PenaltyForm::Dense(m) => f
                .debug_tuple("PenaltyForm::Dense")
                .field(&format_args!("{}×{}", m.nrows(), m.ncols()))
                .finish(),
            PenaltyForm::Operator(op) => f
                .debug_tuple("PenaltyForm::Operator")
                .field(&format_args!("dim={}", op.dim()))
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    fn psd_fixture() -> Array2<f64> {
        // Symmetric PSD: A = B^T B with random-ish B.
        let b = Array::from_shape_vec(
            (3, 4),
            vec![
                1.0, -0.3, 0.7, 0.1, 0.2, 1.1, -0.5, 0.4, -0.1, 0.6, 0.9, -0.2,
            ],
        )
        .unwrap();
        b.t().dot(&b)
    }

    #[test]
    fn array2_impl_matvec_matches_dot() {
        let s = psd_fixture();
        let v = Array1::from_vec(vec![0.7, -0.4, 0.2, 1.1]);
        let mut out = Array1::<f64>::zeros(4);
        s.matvec(v.view(), out.view_mut());
        let want = s.dot(&v);
        for i in 0..4 {
            assert_abs_diff_eq!(out[i], want[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn array2_impl_diag_and_trace() {
        let s = psd_fixture();
        let d = <Array2<f64> as PenaltyOp>::diag(&s);
        for i in 0..4 {
            assert_abs_diff_eq!(d[i], s[[i, i]], epsilon = 0.0);
        }
        let tr = <Array2<f64> as PenaltyOp>::trace(&s);
        assert_abs_diff_eq!(tr, s.diag().sum(), epsilon = 0.0);
    }

    #[test]
    fn array2_impl_eigendecompose_matches_eigh() {
        let s = psd_fixture();
        let (evals_op, evecs_op) = <Array2<f64> as PenaltyOp>::eigendecompose(&s).expect("eigh");
        let (evals_ref, evecs_ref) = s.eigh(Side::Lower).expect("eigh ref");
        for i in 0..evals_op.len() {
            assert_abs_diff_eq!(evals_op[i], evals_ref[i], epsilon = 1e-12);
        }
        // Sign of eigenvectors is gauge-free; compare V V^T for a stable check.
        let p_op = evecs_op.dot(&evecs_op.t());
        let p_ref = evecs_ref.dot(&evecs_ref.t());
        for i in 0..p_op.nrows() {
            for j in 0..p_op.ncols() {
                assert_abs_diff_eq!(p_op[[i, j]], p_ref[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn penalty_form_dim_and_to_dense_round_trip() {
        let s = psd_fixture();
        let form = PenaltyForm::Dense(s.clone());
        assert_eq!(form.dim(), 4);
        let m = form.to_dense();
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(m[[i, j]], s[[i, j]], epsilon = 0.0);
            }
        }

        let arc: Arc<dyn PenaltyOp> = Arc::new(s.clone());
        let op_form = PenaltyForm::Operator(arc);
        assert_eq!(op_form.dim(), 4);
        let m2 = op_form.to_dense();
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(m2[[i, j]], s[[i, j]], epsilon = 0.0);
            }
        }
    }
}
