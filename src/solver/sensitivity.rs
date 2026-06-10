//! ONE sensitivity operator (#935): every "how does the fit move?"
//! question is the same solve.
//!
//! At a penalized optimum the stationarity condition `g(β̂; t) = 0` makes
//! every sensitivity of the fit one object — the factored fitted curvature
//! applied to a perturbation of the score:
//!
//! ```text
//!   ∂β̂/∂t = −H⁻¹ · ∂g/∂t
//! ```
//!
//! for ANY perturbation channel t: smoothing parameters (the REML outer
//! gradient), case weights (ALO / leave-one-out / Cook's distance),
//! responses (data attribution), stage-1 nuisances (the #461 influence
//! absorber). The tree computed this in independent dialects with
//! independent factorizations — `AloFactoredHessian` (runtime.rs),
//! `ift_dbeta_drho_from_solver` (evidence.rs), the outer-assembly mode
//! responses and the projected pseudo-inverse of the rank-deficient LAML
//! kernel (unified.rs) — and each site had to independently answer the
//! question that actually causes bugs: **which inverse is "H⁻¹"?** The
//! biobank fix 0dc469bd and the #901 layer-2 investigation are both
//! incidents of two sites answering differently.
//!
//! [`FitSensitivity`] is the single answer. It is built once at the
//! optimum from whichever factored form the solver already has — a faer
//! Cholesky factor, a raw lower-triangular factor, or the projected
//! pseudo-inverse `U · M⁻¹ · Uᵀ` (the #752/#901 intrinsic-quotient
//! convention) — and every consumer asks it, never a factor directly.
//! Consumers therefore cannot disagree about the inverse, and every
//! batching/cone/caching improvement made inside [`FitSensitivity::apply_multi`]
//! is inherited by all of them at once (#779's cone-of-influence becomes
//! an optimization here instead of a ρ-path-only feature).
//!
//! Migration ladder (each step DELETES a factorization site):
//! 1. (this commit) `AloFactoredHessian` holds a `FitSensitivity` instead
//!    of a bare factor; `ift_dbeta_drho` routes through
//!    [`FitSensitivity::mode_response`].
//! 2. unified.rs mode-response stacks + `PenaltySubspaceTrace`
//!    pseudo-inverse construct the [`FittedInverse::Projected`] variant.
//! 3. #461 `score_influence_jacobian` sites.

use ndarray::{Array1, Array2, ArrayView2};

use crate::linalg::faer_ndarray::FaerCholeskyFactor;

/// The fitted curvature in whichever factored form the solver produced —
/// the SINGLE place that knows how to invert it.
pub enum FittedInverse<'a> {
    /// Cholesky factor of the (stabilized) penalized Hessian: the
    /// full-rank convention (PIRLS / ALO path).
    FaerCholesky(&'a FaerCholeskyFactor),
    /// Raw lower-triangular Cholesky factor `L` with `H = L·Lᵀ` (the
    /// arrow-Schur reduced factor in evidence.rs).
    LowerTriangular(&'a Array2<f64>),
    /// Projected (pseudo-)inverse `U · M⁻¹ · Uᵀ` over a column basis `U`
    /// (p × r) with reduced inverse `M⁻¹` (r × r) — the rank-deficient
    /// LAML convention (#752/0dc469bd/#901): the inverse acts on
    /// range(U) and annihilates its complement.
    Projected {
        basis: &'a Array2<f64>,
        reduced_inverse: &'a Array2<f64>,
    },
}

/// The one sensitivity operator built at the optimum. See module docs.
pub struct FitSensitivity<'a> {
    inverse: FittedInverse<'a>,
    dim: usize,
}

impl<'a> FitSensitivity<'a> {
    pub fn from_faer_cholesky(factor: &'a FaerCholeskyFactor, dim: usize) -> Self {
        Self {
            inverse: FittedInverse::FaerCholesky(factor),
            dim,
        }
    }

    pub fn from_lower_triangular(factor: &'a Array2<f64>) -> Self {
        let dim = factor.nrows();
        Self {
            inverse: FittedInverse::LowerTriangular(factor),
            dim,
        }
    }

    pub fn from_projected(basis: &'a Array2<f64>, reduced_inverse: &'a Array2<f64>) -> Self {
        let dim = basis.nrows();
        Self {
            inverse: FittedInverse::Projected {
                basis,
                reduced_inverse,
            },
            dim,
        }
    }

    /// Coefficient dimension `p` the operator acts on.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `H⁻¹ · rhs` (pseudo-inverse action for the projected variant).
    pub fn apply(&self, rhs: &Array1<f64>) -> Array1<f64> {
        assert_eq!(rhs.len(), self.dim, "FitSensitivity rhs dimension");
        match &self.inverse {
            FittedInverse::FaerCholesky(factor) => factor.solvevec(rhs),
            FittedInverse::LowerTriangular(factor) => {
                crate::linalg::triangular::cholesky_solve_vector(factor, rhs)
            }
            FittedInverse::Projected {
                basis,
                reduced_inverse,
            } => {
                let reduced = basis.t().dot(rhs);
                basis.dot(&reduced_inverse.dot(&reduced))
            }
        }
    }

    /// `H⁻¹ · RHS` for a (p × m) block of right-hand sides — the batched
    /// form every multi-channel consumer should use (outer ρ-pair solves,
    /// ALO's `H⁻¹Xᵀ` leverage block) so the factor is traversed once per
    /// block instead of once per column.
    pub fn apply_multi(&self, rhs: ArrayView2<'_, f64>) -> Array2<f64> {
        assert_eq!(rhs.nrows(), self.dim, "FitSensitivity RHS dimension");
        match &self.inverse {
            FittedInverse::FaerCholesky(factor) => {
                let mut out = rhs.to_owned();
                factor.solve_mat_in_place(&mut out);
                out
            }
            FittedInverse::LowerTriangular(factor) => {
                crate::linalg::triangular::cholesky_solve_matrix(*factor, rhs)
            }
            FittedInverse::Projected {
                basis,
                reduced_inverse,
            } => {
                let reduced = crate::linalg::faer_ndarray::fast_atb(basis, &rhs.to_owned());
                crate::linalg::faer_ndarray::fast_ab(basis, &reduced_inverse.dot(&reduced))
            }
        }
    }

    /// The IFT mode response `∂β̂/∂t = −H⁻¹ · ∂g/∂t` for a (p × m) block
    /// of score perturbations — THE object of #935.
    ///
    /// Returns `None` if any solved entry is non-finite (the factored
    /// curvature was unusable for this channel); callers must not
    /// substitute an approximation, matching the contract of the deleted
    /// `ift_dbeta_drho_from_solver`.
    pub fn mode_response(&self, dg_dt: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        if dg_dt.nrows() != self.dim {
            return None;
        }
        let mut out = self.apply_multi(dg_dt);
        if out.iter().any(|value| !value.is_finite()) {
            return None;
        }
        out.mapv_inplace(|value| -value);
        Some(out)
    }

    /// `H⁻¹Xᵀ` (p × n) — the shared leverage/case-sensitivity block: its
    /// column i is simultaneously ALO's per-observation solve, the case-
    /// weight channel `∂g/∂w_i ∝ x_i`, and the response channel
    /// `∂g/∂y_i ∝ x_i`. One blocked solve serves all three diagnostics.
    pub fn leverage_block(&self, design: &Array2<f64>) -> Array2<f64> {
        assert_eq!(design.ncols(), self.dim, "FitSensitivity design width");
        self.apply_multi(design.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::faer_ndarray::FaerCholesky;
    use faer::Side;
    use ndarray::array;

    /// Textbook 3×3 lower-Cholesky, written out so the LowerTriangular
    /// variant is tested against an independently constructed factor.
    fn lower_cholesky_3x3(h: &Array2<f64>) -> Array2<f64> {
        let mut l = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..=i {
                let mut acc = h[[i, j]];
                for k in 0..j {
                    acc -= l[[i, k]] * l[[j, k]];
                }
                if i == j {
                    l[[i, j]] = acc.sqrt();
                } else {
                    l[[i, j]] = acc / l[[j, j]];
                }
            }
        }
        l
    }

    /// All three factored forms of the SAME matrix must agree on every
    /// entry point — this is the "no site can pick a different inverse"
    /// guarantee, checked directly.
    #[test]
    fn all_variants_agree_on_the_same_curvature() {
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let faer = h.cholesky(Side::Lower).expect("SPD factor");
        let lower = lower_cholesky_3x3(&h);
        // Projected variant with full basis = the plain inverse.
        let eye = Array2::eye(3);
        let h_inv = {
            let mut out: Array2<f64> = Array2::eye(3);
            faer.solve_mat_in_place(&mut out);
            out
        };
        let s_faer = FitSensitivity::from_faer_cholesky(&faer, 3);
        let s_tri = FitSensitivity::from_lower_triangular(&lower);
        let s_proj = FitSensitivity::from_projected(&eye, &h_inv);

        let rhs = array![0.7, -1.2, 0.4];
        let block = array![[0.7, 1.0], [-1.2, 0.0], [0.4, -2.0]];
        let a = s_faer.apply(&rhs);
        let b = s_tri.apply(&rhs);
        let c = s_proj.apply(&rhs);
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() <= 1e-12, "faer vs triangular [{i}]");
            assert!((a[i] - c[i]).abs() <= 1e-12, "faer vs projected [{i}]");
        }
        let ma = s_faer.apply_multi(block.view());
        let mb = s_tri.apply_multi(block.view());
        let mc = s_proj.apply_multi(block.view());
        let resp = s_faer.mode_response(block.view()).expect("finite");
        for i in 0..3 {
            for j in 0..2 {
                assert!((ma[[i, j]] - mb[[i, j]]).abs() <= 1e-12);
                assert!((ma[[i, j]] - mc[[i, j]]).abs() <= 1e-12);
                assert!((resp[[i, j]] + ma[[i, j]]).abs() <= 1e-15, "mode response sign");
            }
        }
        // H · apply(rhs) must reproduce rhs (it is a true inverse here).
        let back = h.dot(&a);
        for i in 0..3 {
            assert!((back[i] - rhs[i]).abs() <= 1e-12, "inverse residual [{i}]");
        }
    }

    #[test]
    fn mode_response_refuses_non_finite_channels() {
        let h = array![[2.0, 0.0], [0.0, 1.0]];
        let faer = h.faer_cholesky().expect("SPD factor");
        let s = FitSensitivity::from_faer_cholesky(&faer, 2);
        let bad = array![[1.0], [f64::NAN]];
        assert!(s.mode_response(bad.view()).is_none());
        let wrong_dim = array![[1.0], [0.0], [0.0]];
        assert!(s.mode_response(wrong_dim.view()).is_none());
    }
}
