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

