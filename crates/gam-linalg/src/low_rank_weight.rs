//! Structured low-rank working weight: `W = D + U Vᵀ`.
//!
//! The PIRLS inner solve assembles a working weight `W ∈ ℝ^{n×n}` and forms
//! the normal equations `(XᵀWX + S) β = XᵀWz`. Most working models supply a
//! diagonal `W` (one scalar per observation), but Fisher-Rao / behavioral
//! metrics that come from an *external* backward pass through a downstream
//! model are not diagonal — they are diagonal **plus a structured low-rank
//! correction** `U Vᵀ`, where `U, V ∈ ℝ^{n×r}` are tall-skinny and the rank
//! `r ≪ n`. For symmetric metrics (the usual case) `U == V`, but the API
//! does not assume it so we can support nonsymmetric weighting in IRLS
//! corrections too.
//!
//! Crucially: the metric is supplied by the caller. This module never
//! estimates a covariance internally.
//!
//! ## Composition with the existing signed-Gram API
//!
//! The diagonal part `D` flows through the existing `xt_diag_x_signed` /
//! `xt_diag_x_psd` kernels exactly as before, so the rank-0 specialisation
//! coincides with the legacy diagonal path. The low-rank correction adds
//! `(XᵀU)(VᵀX)` — a `p × p` outer product of two tall-skinny matrices —
//! computed in `O(n · p · r)` and *never* materialising an `n × n` weight.
//!
//! ## Woodbury / matrix-determinant lemma
//!
//! Solving with `W` directly:
//!   (D + U Vᵀ)⁻¹ = D⁻¹ − D⁻¹U (I_r + Vᵀ D⁻¹ U)⁻¹ VᵀD⁻¹
//!
//! Solving with the *Gram* `A + Û V̂ᵀ` where `Â = XᵀDX + S`, `Û = XᵀU`,
//! `V̂ = XᵀV`:
//!   (A + Û V̂ᵀ)⁻¹ b = A⁻¹ b − A⁻¹ Û (I_r + V̂ᵀ A⁻¹ Û)⁻¹ V̂ᵀ A⁻¹ b
//!
//! The latter is the form PIRLS uses: one factorisation of the diagonal-W
//! penalised system `A` (Cholesky, as today), then a rank-r capacitance
//! solve of size `r × r`. The dimensionality of the corrected normal
//! equation is `p × p`; nothing blows up to `n`.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::faer_ndarray::{fast_atv, fast_av};

/// `W = diag(diag) + u · vᵀ`. Rows: `n`. Rank of correction: `u.ncols()`.
///
/// For a symmetric metric (the default in Fisher-Rao fits) the caller
/// supplies `u == v` (use [`LowRankWeight::symmetric`]). Asymmetric
/// weights are supported because observed-information IRLS can produce
/// signed off-diagonal contributions.
#[derive(Clone, Copy, Debug)]
pub struct LowRankWeight<'a> {
    pub diag: ArrayView1<'a, f64>,
    pub u: ArrayView2<'a, f64>,
    pub v: ArrayView2<'a, f64>,
}

impl<'a> LowRankWeight<'a> {
    /// Construct a low-rank weight, validating shapes.
    pub fn new(
        diag: ArrayView1<'a, f64>,
        u: ArrayView2<'a, f64>,
        v: ArrayView2<'a, f64>,
    ) -> Result<Self, String> {
        let n = diag.len();
        if u.nrows() != n {
            return Err(format!(
                "LowRankWeight: u has {} rows but diag has {} entries",
                u.nrows(),
                n
            ));
        }
        if v.nrows() != n {
            return Err(format!(
                "LowRankWeight: v has {} rows but diag has {} entries",
                v.nrows(),
                n
            ));
        }
        if u.ncols() != v.ncols() {
            return Err(format!(
                "LowRankWeight: u has rank {} but v has rank {}",
                u.ncols(),
                v.ncols()
            ));
        }
        Ok(LowRankWeight { diag, u, v })
    }

    /// Symmetric metric: `W = D + U Uᵀ`.
    pub fn symmetric(diag: ArrayView1<'a, f64>, u: ArrayView2<'a, f64>) -> Result<Self, String> {
        Self::new(diag, u, u)
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.diag.len()
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.u.ncols()
    }

    #[inline]
    pub fn is_rank_zero(&self) -> bool {
        self.rank() == 0
    }

    /// `W · x` without materialising the `n × n` weight.
    ///
    /// Cost: `O(n) + O(n · r)`.
    pub fn apply(&self, x: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = self.nrows();
        assert_eq!(
            x.len(),
            n,
            "LowRankWeight::apply: x has {} entries but W has {} rows",
            x.len(),
            n
        );
        // diag(D) · x
        let mut out = Array1::<f64>::from_iter((0..n).map(|i| self.diag[i] * x[i]));
        if self.is_rank_zero() {
            return out;
        }
        // U (Vᵀ x): Vᵀ x is r-dim, then U times r-vector. The `fast_*`
        // kernels are generic over `Data<Elem = f64>`, so we pass views
        // directly — no `.to_owned()` copies of `u`, `v`, or `x`.
        let vtx = fast_atv(&self.v, &x);
        let uvtx = fast_av(&self.u, &vtx);
        out += &uvtx;
        out
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_ndarray::fast_ab;
    use crate::matrix::{DesignMatrix, FiniteSignedWeightsView, LinearOperator};
    use ndarray::array;

    /// Build a tiny dense design and a corresponding `DesignMatrix::Dense`.
    fn small_design() -> DesignMatrix {
        let x = array![
            [1.0, 0.5, -0.2],
            [0.3, 1.2, 0.4],
            [-0.1, 0.7, 1.0],
            [0.6, -0.3, 0.8],
            [0.2, 0.9, -0.5],
        ];
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x))
    }

    /// Reference dense weight `W = diag(d) + u vᵀ`.
    fn dense_w(d: &Array1<f64>, u: &Array2<f64>, v: &Array2<f64>) -> Array2<f64> {
        let n = d.len();
        let mut w = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            w[[i, i]] = d[i];
        }
        let uvt = fast_ab(u, &v.t().to_owned());
        w + &uvt
    }

    #[test]
    fn apply_matches_dense() {
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = array![
            [0.1, -0.2],
            [0.4, 0.3],
            [-0.1, 0.5],
            [0.2, 0.1],
            [0.0, -0.3]
        ];
        let v = array![[0.2, 0.1], [0.0, 0.4], [0.3, -0.2], [-0.1, 0.6], [0.5, 0.0]];
        let lr = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();
        let x = array![1.0, -2.0, 0.5, 0.3, -1.0];
        let got = lr.apply(x.view());
        let w = dense_w(&d, &u, &v);
        let want = w.dot(&x);
        for i in 0..got.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-12,
                "row {}: {} vs {}",
                i,
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn xtwx_correction_matches_dense() {
        let design = small_design();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = array![
            [0.1, -0.2],
            [0.4, 0.3],
            [-0.1, 0.5],
            [0.2, 0.1],
            [0.0, -0.3]
        ];
        let v = array![[0.2, 0.1], [0.0, 0.4], [0.3, -0.2], [-0.1, 0.6], [0.5, 0.0]];
        let lr = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();

        let mut xtwx = design
            .xt_diag_x_signed_op(FiniteSignedWeightsView::try_from_array(&d).unwrap())
            .unwrap();
        lr.add_low_rank_xtwx_correction(&design, &mut xtwx).unwrap();

        // Reference: Xᵀ W X with the full dense W.
        let xdense = design.as_dense().unwrap().to_owned();
        let w = dense_w(&d, &u, &v);
        let want = xdense.t().dot(&w).dot(&xdense);

        for i in 0..xtwx.nrows() {
            for j in 0..xtwx.ncols() {
                assert!(
                    (xtwx[[i, j]] - want[[i, j]]).abs() < 1e-10,
                    "({},{}): {} vs {}",
                    i,
                    j,
                    xtwx[[i, j]],
                    want[[i, j]]
                );
            }
        }
    }

    #[test]
    fn xtwy_matches_dense() {
        let design = small_design();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = array![
            [0.1, -0.2],
            [0.4, 0.3],
            [-0.1, 0.5],
            [0.2, 0.1],
            [0.0, -0.3]
        ];
        let v = array![[0.2, 0.1], [0.0, 0.4], [0.3, -0.2], [-0.1, 0.6], [0.5, 0.0]];
        let lr = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();
        let y = array![0.7, -1.2, 0.3, 0.9, -0.4];
        let got = lr.xtw_y(&design, y.view()).unwrap();

        let xdense = design.as_dense().unwrap().to_owned();
        let w = dense_w(&d, &u, &v);
        let want = xdense.t().dot(&w.dot(&y));
        for i in 0..got.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-10,
                "i={}: {} vs {}",
                i,
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn rank_zero_reduces_to_diagonal() {
        let design = small_design();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = Array2::<f64>::zeros((5, 0));
        let v = Array2::<f64>::zeros((5, 0));
        let lr = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();
        assert!(lr.is_rank_zero());

        let mut xtwx = design
            .xt_diag_x_signed_op(FiniteSignedWeightsView::try_from_array(&d).unwrap())
            .unwrap();
        let baseline = xtwx.clone();
        lr.add_low_rank_xtwx_correction(&design, &mut xtwx).unwrap();
        // Correction must be the zero matrix when r=0.
        let diff = (&xtwx - &baseline).mapv(f64::abs).sum();
        assert!(diff < 1e-14, "rank-0 correction left {}", diff);

        // xtw_y should equal Xᵀ (D y) exactly.
        let y = array![0.7, -1.2, 0.3, 0.9, -0.4];
        let got = lr.xtw_y(&design, y.view()).unwrap();
        let dy: Array1<f64> = (&d) * (&y);
        let want = design.transpose_vector_multiply(&dy);
        let diff = (&got - &want).mapv(f64::abs).sum();
        assert!(diff < 1e-14);
    }

    #[test]
    fn woodbury_row_capacitance_inverts_w() {
        // (D + UVᵀ)⁻¹ from row-space Woodbury must match dense inverse.
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = array![
            [0.1, -0.2],
            [0.4, 0.3],
            [-0.1, 0.5],
            [0.2, 0.1],
            [0.0, -0.3]
        ];
        let v = array![[0.2, 0.1], [0.0, 0.4], [0.3, -0.2], [-0.1, 0.6], [0.5, 0.0]];
        let lr = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();

        let cap = lr.row_capacitance();
        // Invert cap by a small LU via solving against I.
        let cap_inv = small_inverse(&cap);

        // Compute Woodbury inverse acting on a probe vector b:
        //   W⁻¹ b = D⁻¹ b − D⁻¹ U cap⁻¹ Vᵀ D⁻¹ b
        let b = array![1.0, -2.0, 0.5, 0.3, -1.0];
        let dinv_b: Array1<f64> = (&b) / (&d);
        let vt_dinvb = fast_atv(&v, &dinv_b);
        let cap_inv_vt = cap_inv.dot(&vt_dinvb);
        let u_correction = u.dot(&cap_inv_vt);
        let dinv_u_correction: Array1<f64> = (&u_correction) / (&d);
        let w_inv_b = &dinv_b - &dinv_u_correction;

        // Reference: dense (D + UVᵀ) solved against b.
        let w = dense_w(&d, &u, &v);
        let w_inv = small_inverse(&w);
        let want = w_inv.dot(&b);

        for i in 0..w_inv_b.len() {
            assert!(
                (w_inv_b[i] - want[i]).abs() < 1e-9,
                "i={}: woodbury {} vs dense {}",
                i,
                w_inv_b[i],
                want[i]
            );
        }
    }

    #[test]
    fn woodbury_gram_capacitance_consistency() {
        // The penalised-Gram Woodbury at p × p: with A = XᵀDX + S (S=0
        // here for simplicity) and Û=XᵀU, V̂=XᵀV, the corrected solve
        //   (A + Û V̂ᵀ)⁻¹ b = A⁻¹ b − A⁻¹ Û cap⁻¹ V̂ᵀ A⁻¹ b
        // must match the dense `(XᵀWX)⁻¹ b`.
        let design = small_design();
        let xdense = design.as_dense().unwrap().to_owned();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = array![
            [0.1, -0.2],
            [0.4, 0.3],
            [-0.1, 0.5],
            [0.2, 0.1],
            [0.0, -0.3]
        ];
        let v = array![[0.2, 0.1], [0.0, 0.4], [0.3, -0.2], [-0.1, 0.6], [0.5, 0.0]];
        let lr = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();

        let a = design
            .xt_diag_x_signed_op(FiniteSignedWeightsView::try_from_array(&d).unwrap())
            .unwrap(); // p × p
        let a_inv = small_inverse(&a);
        let uhat = lr.project_u(&design).unwrap(); // p × r
        let vhat = lr.project_v(&design).unwrap(); // p × r
        let a_inv_uhat = a_inv.dot(&uhat);
        let cap = LowRankWeight::gram_capacitance(&a_inv_uhat, &vhat).unwrap();
        let cap_inv = small_inverse(&cap);

        let b = array![0.5, -1.0, 0.4];
        let a_inv_b = a_inv.dot(&b);
        let vt_ainv_b = vhat.t().dot(&a_inv_b);
        let cap_inv_v = cap_inv.dot(&vt_ainv_b);
        let correction = a_inv_uhat.dot(&cap_inv_v);
        let got = &a_inv_b - &correction;

        // Reference: dense (XᵀWX)⁻¹ b.
        let w = dense_w(&d, &u, &v);
        let xtwx_full = xdense.t().dot(&w).dot(&xdense);
        let want = small_inverse(&xtwx_full).dot(&b);

        for i in 0..got.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-8,
                "i={}: gram-woodbury {} vs dense {}",
                i,
                got[i],
                want[i]
            );
        }
    }

    /// Tiny Gauss-Jordan inverse, used only by these unit tests.
    fn small_inverse(a: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        assert_eq!(a.ncols(), n);
        let mut aug = Array2::<f64>::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n + i]] = 1.0;
        }
        for i in 0..n {
            // pivot
            let mut piv = i;
            let mut best = aug[[i, i]].abs();
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > best {
                    best = aug[[k, i]].abs();
                    piv = k;
                }
            }
            if piv != i {
                for j in 0..(2 * n) {
                    let tmp = aug[[i, j]];
                    aug[[i, j]] = aug[[piv, j]];
                    aug[[piv, j]] = tmp;
                }
            }
            let pivval = aug[[i, i]];
            for j in 0..(2 * n) {
                aug[[i, j]] /= pivval;
            }
            for k in 0..n {
                if k == i {
                    continue;
                }
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
        let mut inv = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = aug[[i, n + j]];
            }
        }
        inv
    }
}
