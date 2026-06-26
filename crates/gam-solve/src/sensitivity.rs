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
//! independent factorizations — `AloFactoredHessian` (runtime.rs), an
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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

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

    /// `H⁻¹Xᵀ` (p × n) — the shared leverage/case-sensitivity block: its
    /// column i is simultaneously ALO's per-observation solve, the case-
    /// weight channel `∂g/∂w_i ∝ x_i`, and the response channel
    /// `∂g/∂y_i ∝ x_i`. One blocked solve serves all three diagnostics.
    pub fn leverage_block(&self, design: &Array2<f64>) -> Array2<f64> {
        assert_eq!(design.ncols(), self.dim, "FitSensitivity design width");
        self.apply_multi(design.t())
    }

    /// Data attribution `∂β̂/∂y` (p × n) — how each fitted coefficient
    /// responds to each response value, the `t = y` channel of the one
    /// identity `∂β̂/∂t = −H⁻¹ ∂g/∂t`.
    ///
    /// The response enters the penalized score only through the working
    /// residual, so `∂g/∂y_i = −w_i x_i` and therefore
    ///
    /// ```text
    ///   ∂β̂/∂y_i = w_i · H⁻¹ x_i,
    /// ```
    /// i.e. column `i` of [`Self::leverage_block`] scaled by the working
    /// weight `w_i`. Contracting back through the design recovers the
    /// smoother/hat matrix `A = X (∂β̂/∂y) = X H⁻¹ Xᵀ W`, whose diagonal is
    /// the leverage already reported elsewhere. For a Gaussian penalized fit
    /// `β̂ = H⁻¹ Xᵀ y`, so this Jacobian is exact (and weight-free); for a GLM
    /// it is the one-step attribution at the fitted working weights.
    ///
    /// Returns `None` on a shape mismatch.
    pub fn response_jacobian(
        &self,
        design: &Array2<f64>,
        working_weights: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>> {
        let n = design.nrows();
        if design.ncols() != self.dim || working_weights.len() != n {
            return None;
        }
        // Column i is H⁻¹ x_i; scale it by w_i to get ∂β̂/∂y_i.
        let mut dbeta_dy = self.leverage_block(design);
        for i in 0..n {
            let w_i = working_weights[i];
            dbeta_dy.column_mut(i).mapv_inplace(|v| w_i * v);
        }
        Some(dbeta_dy)
    }

    /// Case-deletion influence (dfbetas + Cook's distance) for every
    /// observation, built from the one sensitivity operator — the
    /// "leave-one-out" channel #935 was designed to unify.
    ///
    /// Deleting observation `i` perturbs the penalized score by exactly its
    /// own contribution, so the IFT mode response gives the coefficient
    /// change in closed form (the penalized Sherman–Morrison identity):
    ///
    /// ```text
    ///   β̂ − β̂₍ᵢ₎ = (w_i r_i / (1 − h_ii)) · H⁻¹ x_i
    /// ```
    ///
    /// where `x_i` is row `i` of `design`, `w_i = working_weights[i]` the
    /// IRLS working weight, `r_i = working_residual[i]` the working residual
    /// `z_i − x_iᵀβ̂`, and `h_ii = w_i x_iᵀ H⁻¹ x_i` the leverage. Column `i`
    /// of [`Self::leverage_block`] **is** `H⁻¹ x_i`, so each dfbeta is one
    /// scaled column — no per-observation refit, no second factorization.
    /// For a Gaussian penalized fit the identity is exact; for a GLM it is
    /// the standard one-step (ALO) approximation, consistent with the
    /// leverage already reported by `AloDiagnostics`.
    ///
    /// Cook's distance uses the metric the fit actually moves in,
    /// `H = XᵀWX + S`:
    ///
    /// ```text
    ///   D_i = (β̂−β̂₍ᵢ₎)ᵀ H (β̂−β̂₍ᵢ₎) / (p · φ)
    ///       = scale_i² · (x_iᵀ H⁻¹ x_i) / (p · φ),   scale_i = w_i r_i / (1 − h_ii),
    /// ```
    ///
    /// the second form following from `(H⁻¹x_i)ᵀ H (H⁻¹x_i) = x_iᵀ H⁻¹ x_i`,
    /// so the single quadratic form `x_iᵀ H⁻¹ x_i` gates the leverage, the
    /// deletion denominator, and Cook's distance alike — no separate `H` apply.
    ///
    /// This is an *opt-in* diagnostic: `dfbeta` is `n × p` and is never
    /// materialized on the default fit path (it would be ruinous at large-scale
    /// scale). Returns `None` on a shape mismatch or if any leverage reaches
    /// `1` (a point the deletion identity cannot resolve).
    pub fn case_deletion(
        &self,
        design: &Array2<f64>,
        working_weights: ArrayView1<'_, f64>,
        working_residual: ArrayView1<'_, f64>,
        phi: f64,
    ) -> Option<CaseDeletionInfluence> {
        let n = design.nrows();
        let p = design.ncols();
        if p != self.dim
            || working_weights.len() != n
            || working_residual.len() != n
            || !(phi.is_finite() && phi > 0.0)
            || p == 0
        {
            return None;
        }
        // Column i of H⁻¹Xᵀ is H⁻¹ x_i — one blocked solve for all n.
        let h_inv_xt = self.leverage_block(design);

        let mut dfbeta = Array2::<f64>::zeros((n, p));
        let mut leverage = Array1::<f64>::zeros(n);
        let mut cooks = Array1::<f64>::zeros(n);
        let p_phi = p as f64 * phi;
        for i in 0..n {
            // hinv_xi = H⁻¹x_i is column i of the leverage block; the single
            // quadratic form x_iᵀ H⁻¹ x_i gates everything below.
            let hinv_xi = h_inv_xt.column(i);
            let xhx = design.row(i).dot(&hinv_xi);
            let h_ii = working_weights[i] * xhx;
            let denom = 1.0 - h_ii;
            // Leverage 1 pins the row to its own fit: the closed-form
            // deletion is singular there, so we refuse rather than emit ∞.
            if !denom.is_finite() || denom.abs() < f64::EPSILON {
                return None;
            }
            // β̂ − β̂₍ᵢ₎ = scale · H⁻¹x_i — one scaled column, no refit.
            let scale = working_weights[i] * working_residual[i] / denom;
            dfbeta.row_mut(i).assign(&(&hinv_xi * scale));
            leverage[i] = h_ii;
            cooks[i] = scale * scale * xhx / p_phi;
        }
        Some(CaseDeletionInfluence {
            dfbeta,
            leverage,
            cooks_distance: cooks,
        })
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
    use gam_linalg::faer_ndarray::FaerCholesky;
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
                assert!(
                    (resp[[i, j]] + ma[[i, j]]).abs() <= 1e-15,
                    "mode response sign"
                );
            }
        }
        // H · apply(rhs) must reproduce rhs (it is a true inverse here).
        let back = h.dot(&a);
        for i in 0..3 {
            assert!((back[i] - rhs[i]).abs() <= 1e-12, "inverse residual [{i}]");
        }
    }

    /// Case-deletion dfbetas from the operator must equal the EXACT
    /// leave-one-out refit of a ridge-penalized Gaussian fit — brute-force
    /// dropping each row and re-solving — to machine precision. This is the
    /// penalized Sherman–Morrison identity, analytic ground truth, no
    /// external tool. Cook's distance is checked against its own definition
    /// `(β−β₍ᵢ₎)ᵀ H (β−β₍ᵢ₎)/(p·φ)` using the exact refit.
    #[test]
    fn case_deletion_matches_exact_loo_refit() {
        // Small over-determined ridge problem: H = XᵀX + S, β = H⁻¹ Xᵀ y,
        // working weights w_i = 1 and working residual r_i = y_i − x_iᵀβ
        // (the Gaussian identity-link IRLS reduction).
        let x = array![
            [1.0, 0.2, -0.5],
            [1.0, -1.1, 0.3],
            [1.0, 0.7, 1.4],
            [1.0, 2.0, -0.8],
            [1.0, -0.4, 0.9],
            [1.0, 1.3, 0.1],
        ];
        let y = array![0.5, -1.2, 2.1, 0.3, -0.7, 1.0];
        let n = x.nrows();
        let p = x.ncols();
        // Penalty S (a mild ridge on the two slopes, intercept unpenalized).
        let mut s = Array2::<f64>::zeros((p, p));
        s[[1, 1]] = 0.4;
        s[[2, 2]] = 0.4;

        let xtx = x.t().dot(&x);
        let h = &xtx + &s;
        let h_inv = {
            let f = h.cholesky(Side::Lower).expect("SPD");
            let mut out: Array2<f64> = Array2::eye(p);
            f.solve_mat_in_place(&mut out);
            out
        };
        let xty = x.t().dot(&y);
        let beta = h_inv.dot(&xty);

        let w = Array1::<f64>::ones(n);
        let resid = &y - &x.dot(&beta);

        let faer = h.cholesky(Side::Lower).expect("SPD");
        let op = FitSensitivity::from_faer_cholesky(&faer, p);
        let infl = op
            .case_deletion(&x, w.view(), resid.view(), 1.0)
            .expect("case deletion");

        for i in 0..n {
            // Exact refit with row i removed: H₍ᵢ₎ = H − x_i x_iᵀ (penalty S
            // is a prior, unchanged by data deletion), rhs = Xᵀy − x_i y_i.
            let x_i = x.row(i).to_owned();
            let mut h_del = h.clone();
            for a in 0..p {
                for b in 0..p {
                    h_del[[a, b]] -= x_i[a] * x_i[b];
                }
            }
            let rhs_del = &xty - &(&x_i * y[i]);
            let h_del_inv = {
                let f = h_del.cholesky(Side::Lower).expect("SPD deleted");
                let mut out: Array2<f64> = Array2::eye(p);
                f.solve_mat_in_place(&mut out);
                out
            };
            let beta_del = h_del_inv.dot(&rhs_del);
            let exact_dfbeta = &beta - &beta_del;

            for j in 0..p {
                assert!(
                    (infl.dfbeta[[i, j]] - exact_dfbeta[j]).abs() < 1e-9,
                    "dfbeta[{i},{j}]: operator {} vs exact refit {}",
                    infl.dfbeta[[i, j]],
                    exact_dfbeta[j]
                );
            }
            // Cook's distance against its definition with the exact refit.
            let cook_exact = exact_dfbeta.dot(&h.dot(&exact_dfbeta)) / (p as f64 * 1.0);
            assert!(
                (infl.cooks_distance[i] - cook_exact).abs() < 1e-9,
                "cooks[{i}]: {} vs {}",
                infl.cooks_distance[i],
                cook_exact
            );
            // Leverage must match the hat value x_iᵀ H⁻¹ x_i (w_i = 1).
            let h_ii = x_i.dot(&h_inv.dot(&x_i));
            assert!((infl.leverage[i] - h_ii).abs() < 1e-12, "leverage[{i}]");
        }
    }

    /// Data attribution `∂β̂/∂y` must equal the actual change in the fitted
    /// coefficients when a response value is perturbed and the ridge fit is
    /// re-solved — exact for the Gaussian penalized model (`β̂ = H⁻¹Xᵀy`).
    #[test]
    fn response_jacobian_matches_refit_perturbation() {
        let x = array![
            [1.0, 0.2, -0.5],
            [1.0, -1.1, 0.3],
            [1.0, 0.7, 1.4],
            [1.0, 2.0, -0.8],
            [1.0, -0.4, 0.9],
        ];
        let y = array![0.5, -1.2, 2.1, 0.3, -0.7];
        let n = x.nrows();
        let p = x.ncols();
        let mut s = Array2::<f64>::zeros((p, p));
        s[[1, 1]] = 0.4;
        s[[2, 2]] = 0.4;
        let h = &x.t().dot(&x) + &s;
        let faer = h.cholesky(Side::Lower).expect("SPD");

        let solve = |rhs: &Array1<f64>| faer.solvevec(rhs);
        let beta = solve(&x.t().dot(&y));

        let op = FitSensitivity::from_faer_cholesky(&faer, p);
        let w = Array1::<f64>::ones(n);
        let dbeta_dy = op.response_jacobian(&x, w.view()).expect("jacobian");

        // Exact (linear) refit: bump y_j, re-solve, compare (β'−β)/ε to col j.
        let eps = 1e-6;
        for j in 0..n {
            let mut yp = y.clone();
            yp[j] += eps;
            let beta_p = solve(&x.t().dot(&yp));
            for c in 0..p {
                let fd = (beta_p[c] - beta[c]) / eps;
                assert!(
                    (dbeta_dy[[c, j]] - fd).abs() < 1e-7,
                    "∂β̂_{c}/∂y_{j}: analytic {} vs refit {}",
                    dbeta_dy[[c, j]],
                    fd
                );
            }
        }
    }

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
