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
//! for ANY perturbation channel `t`: smoothing parameters (the REML outer
//! gradient), case weights (ALO / leave-one-out / Cook's distance),
//! responses (data attribution). One identity, read off in whichever
//! direction a diagnostic needs it.
//!
//! Before this, the tree computed `H⁻¹·` in independent dialects with
//! independent factorizations — an
//! `ift_dbeta_drho_from_solver` solve-closure and a separate coned variant
//! (evidence.rs), and the projected pseudo-inverse of the rank-deficient
//! LAML kernel (unified.rs) — so each site had to answer on its own the
//! question that actually causes bugs: **which inverse is "H⁻¹"?** The
//! large-scale fix 0dc469bd and the #901 layer-2 investigation are both
//! incidents of two sites answering differently.
//!
//! [`FitSensitivity`] is the single answer. It is built once at the optimum
//! from whichever factored form the solver already has — a faer Cholesky
//! factor, a raw lower-triangular (arrow-Schur) factor, or the projected
//! pseudo-inverse `U · M⁻¹ · Uᵀ` (the #752/#901 intrinsic-quotient
//! convention) — and every consumer asks it, never a factor directly.
//! Consumers therefore cannot disagree about the inverse, and every
//! batching/cone improvement made inside [`FitSensitivity::apply_multi`] is
//! inherited by all of them at once.
//!
//! The channels, each a one-line restatement of the identity above:
//!
//! - [`mode_response`](FitSensitivity::mode_response) — `−H⁻¹ ∂g/∂t`, the
//!   REML outer gradient's `∂β̂/∂ρ` (evidence `ift_dbeta_drho`).
//! - [`mode_response_coned`](FitSensitivity::mode_response_coned) — the same
//!   response confined to its cone of influence (#779); the lazy/local form
//!   the smoothing-correction IFT uses.
//! - [`leverage_block`](FitSensitivity::leverage_block) — `H⁻¹Xᵀ`, whose
//!   column `i` is at once ALO's per-row solve and the case/response channel.
//! - [`case_deletion`](FitSensitivity::case_deletion) — dfbetas + Cook's
//!   distance, the leave-one-out channel, one scaled column of `H⁻¹Xᵀ` each.
//!
//! What is deliberately NOT folded in: the matrix-free `hop.solve_multi`
//! (PCG/GPU), the constrained kernel `K_T = K_S − K_S Aᵀ(A K_S Aᵀ)⁻¹A K_S`,
//! and `alo.rs`'s zero-copy `StableSolver` loop. Those are distinct inverse
//! *representations*, not duplicate spellings of the same factored inverse —
//! routing them through here would regress performance and couple unrelated
//! concerns rather than remove the bug class.

use ndarray::{Array1, Array2, ArrayView2};

use gam_linalg::faer_ndarray::FaerCholeskyFactor;

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
                gam_linalg::triangular::cholesky_solve_vector(factor.view(), rhs.view())
            }
            FittedInverse::Projected {
                basis,
                reduced_inverse,
            } => {
                // `U · (M⁻¹ · (Uᵀ · a))` via faer SIMD contractions — the
                // single spelling of the projected (rank-deficient LAML)
                // inverse, shared with `PenaltySubspaceTrace`.
                let proj = gam_linalg::faer_ndarray::fast_atv(basis, rhs);
                let reduced = reduced_inverse.dot(&proj);
                gam_linalg::faer_ndarray::fast_av(basis, &reduced)
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
                gam_linalg::triangular::cholesky_solve_matrix(*factor, rhs)
            }
            FittedInverse::Projected {
                basis,
                reduced_inverse,
            } => {
                let reduced = gam_linalg::faer_ndarray::fast_atb(basis, &rhs.to_owned());
                gam_linalg::faer_ndarray::fast_ab(basis, &reduced_inverse.dot(&reduced))
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

    /// Cone-of-influence mode response (#779), the lazy/local form of
    /// [`Self::mode_response`]. Each perturbation column `∂g/∂t_a` is
    /// structurally supported only within `col_supports[a]`, so its response
    /// `−H⁻¹ ∂g/∂t_a` is exactly zero outside the coupling component of
    /// `hessian` containing that support. Columns whose support is empty (a
    /// structurally inactive channel) are skipped with no solve; the active
    /// columns are solved as ONE batched block through [`Self::apply_multi`]
    /// — strictly better than the per-column BLAS-2 loop this replaces — and
    /// each result confined to its cone. On a fully coupled `hessian` every
    /// cone is the whole space and the result equals [`Self::mode_response`]
    /// bit-for-bit.
    ///
    /// `hessian` must be the same curvature this operator inverts; a
    /// dimension mismatch (or any non-finite solved entry) returns `None`
    /// rather than silently substituting an approximation.
    pub fn mode_response_coned(
        &self,
        hessian: ArrayView2<'_, f64>,
        dg_dt: ArrayView2<'_, f64>,
        col_supports: &[std::ops::Range<usize>],
    ) -> Option<Array2<f64>> {
        let p = self.dim;
        let r = dg_dt.ncols();
        if dg_dt.nrows() != p
            || hessian.nrows() != p
            || hessian.ncols() != p
            || col_supports.len() != r
        {
            return None;
        }
        let labels = crate::evidence::coupling_components(hessian);
        if labels.len() != p {
            return None;
        }

        // Active columns + their cones; structurally inactive columns (empty
        // support → empty cone) contribute an identically-zero sensitivity
        // and are skipped entirely (no solve).
        let mut active: Vec<(usize, Vec<usize>)> = Vec::new();
        for a in 0..r {
            let sr = &col_supports[a];
            let support: Vec<usize> = (sr.start..sr.end)
                .filter(|idx| *idx < p)
                .filter(|idx| dg_dt[[*idx, a]] != 0.0)
                .collect();
            let cone = crate::evidence::cone_of_influence(&labels, &support);
            if !cone.is_empty() {
                active.push((a, cone));
            }
        }

        let mut out = Array2::<f64>::zeros((p, r));
        if active.is_empty() {
            return Some(out);
        }
        // One batched solve over only the active columns.
        let mut rhs = Array2::<f64>::zeros((p, active.len()));
        for (j, (a, _)) in active.iter().enumerate() {
            rhs.column_mut(j).assign(&dg_dt.column(*a));
        }
        let solved = self.apply_multi(rhs.view());
        if solved.iter().any(|value| !value.is_finite()) {
            return None;
        }
        for (j, (a, cone)) in active.iter().enumerate() {
            for &row in cone {
                out[[row, *a]] = -solved[[row, j]];
            }
        }
        Some(out)
    }

}

/// Exact (Gaussian) / one-step (GLM) case-deletion influence produced by
/// [`FitSensitivity::case_deletion`]. See that method for the identities.
pub struct CaseDeletionInfluence {
    /// `dfbeta[[i, j]]` = change in coefficient `j` when observation `i` is
    /// left out, `β̂_j − β̂₍ᵢ₎_j`.
    pub dfbeta: Array2<f64>,
    /// Leverage (hat value) `h_ii = w_i x_iᵀ H⁻¹ x_i` per observation.
    pub leverage: Array1<f64>,
    /// Cook's distance per observation.
    pub cooks_distance: Array1<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;
    use ndarray::array;

    #[test]
    fn mode_response_refuses_non_finite_channels() {
        let h = array![[2.0, 0.0], [0.0, 1.0]];
        let faer = h.cholesky(Side::Lower).expect("SPD factor");
        let s = FitSensitivity::from_faer_cholesky(&faer, 2);
        let bad = array![[1.0], [f64::NAN]];
        assert!(s.mode_response(bad.view()).is_none());
        let wrong_dim = array![[1.0], [0.0], [0.0]];
        assert!(s.mode_response(wrong_dim.view()).is_none());
    }
}
