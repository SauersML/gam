use super::*;

impl<'a> RemlState<'a> {
    pub(crate) fn xt_diag_x_dense_into(
        x: &Array2<f64>,
        diag: &Array1<f64>,
        weighted: &mut Array2<f64>,
    ) -> Array2<f64> {
        weighted.assign(x);
        ndarray::Zip::from(weighted.rows_mut())
            .and(diag.view())
            .for_each(|mut row, w| row *= *w);
        fast_atb(x, weighted)
    }

    pub(crate) fn row_scale(x: &Array2<f64>, scale: &Array1<f64>) -> Array2<f64> {
        let mut out = x.clone();
        ndarray::Zip::from(out.rows_mut())
            .and(scale.view())
            .for_each(|mut row, w| row *= *w);
        out
    }

    pub(crate) fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        debug_assert_eq!(a.nrows(), b.ncols());
        debug_assert_eq!(a.ncols(), b.nrows());
        let elems = a.nrows().saturating_mul(a.ncols());
        if elems >= 32 * 32 {
            let a_view = FaerArrayView::new(a);
            let b_view = FaerArrayView::new(b);
            return faer_frob_inner(a_view.as_ref(), b_view.as_ref().transpose());
        }
        let m = a.nrows();
        let n = a.ncols();
        kahan_sum((0..m).map(|i| {
            let mut acc = 0.0_f64;
            for j in 0..n {
                acc += a[[i, j]] * b[[j, i]];
            }
            acc
        }))
    }

    pub(crate) fn trace_hdag_operator_apply<S, A>(
        p_dim: usize,
        h_pos_w: Option<&Array2<f64>>,
        h_pos_w_t: Option<&Array2<f64>>,
        mut solve_h_mat: S,
        mut apply_to_block: A,
    ) -> f64
    where
        S: FnMut(&Array2<f64>) -> Array2<f64>,
        A: FnMut(&Array2<f64>) -> Array2<f64>,
    {
        if let (Some(w), Some(w_t)) = (h_pos_w, h_pos_w_t) {
            let a_w = apply_to_block(w);
            return Self::trace_product(w_t, &a_w);
        }
        const BLOCK: usize = 32;
        let mut acc = 0.0_f64;
        let mut start = 0usize;
        while start < p_dim {
            let width = (p_dim - start).min(BLOCK);
            let mut basis = Array2::<f64>::zeros((p_dim, width));
            for j in 0..width {
                basis[[start + j, j]] = 1.0;
            }
            let a_block = apply_to_block(&basis);
            let solved = solve_h_mat(&a_block);
            for j in 0..width {
                acc += solved[[start + j, j]];
            }
            start += width;
        }
        acc
    }

    pub(crate) fn bilinear_form(
        mat: &Array2<f64>,
        left: ndarray::ArrayView1<'_, f64>,
        right: ndarray::ArrayView1<'_, f64>,
    ) -> f64 {
        let n = mat.nrows();
        debug_assert_eq!(mat.ncols(), n);
        debug_assert_eq!(left.len(), n);
        debug_assert_eq!(right.len(), n);
        let mut acc = KahanSum::default();
        for i in 0..n {
            let mut row_dot = 0.0_f64;
            for j in 0..n {
                row_dot += mat[[i, j]] * right[j];
            }
            acc.add(left[i] * row_dot);
        }
        acc.sum()
    }

    pub(crate) fn reduced_weighted_gram(z: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
        // Returns Zᵀ diag(weights) Z (exact).
        //
        // Used for:
        //   S = Zᵀ diag(v) Z               in Hadamard-Gram apply,
        //   G_u = X_rᵀ diag(s_u) X_r       with s_u = w' ⊙ (Xu),
        // both of which avoid constructing dense n×n intermediates.
        let weighted = Self::row_scale(z, weights);
        fast_atb(z, &weighted)
    }

    pub(crate) fn reduced_cross_weighted_gram(
        z_left: &Array2<f64>,
        z_right: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> Array2<f64> {
        // Returns Z_leftᵀ diag(weights) Z_right (exact).
        //
        // This is used for explicit design-moving terms where left/right
        // reduced designs differ (X_r vs X_{tau,r}) in Hadamard-Gram products.
        let weighted = Self::row_scale(z_right, weights);
        fast_atb(z_left, &weighted)
    }

    pub(crate) fn reduced_diag_gram(z: &Array2<f64>, a: &Array2<f64>) -> Array1<f64> {
        // Returns diag(Z A Zᵀ), exact without forming dense n×n matrix.
        //
        // Identity:
        //   [diag(Z A Zᵀ)]_i = z_iᵀ A z_i,
        // where z_i is row i of Z. We compute this as rowwise dot(Z, Z A).
        let za = fast_ab(z, a);
        (z * &za).sum_axis(ndarray::Axis(1))
    }

    pub(crate) fn apply_hadamard_gram(
        z: &Array2<f64>,
        a_left: &Array2<f64>,
        a_right: &Array2<f64>,
        vec: &Array1<f64>,
    ) -> Array1<f64> {
        // Exact apply of:
        //   y = ((Z A_left Z^T) ⊙ (Z A_right Z^T)) vec
        // using Gram/Hadamard identity with S = Z^T diag(vec) Z:
        //   y_i = z_i^T A_left S A_right z_i.
        //
        // This is the matrix-free kernel behind the Firth terms that would
        // otherwise require dense P = M⊙M and M⊙N_u products.
        // Complexity is O(n r^2) with r = rank(X), no n×n storage.
        let s = Self::reduced_weighted_gram(z, vec);
        let left_s = a_left.dot(&s);
        let t = left_s.dot(a_right);
        Self::reduced_diag_gram(z, &t)
    }

    pub(crate) fn apply_hadamard_gram_to_matrix(
        z: &Array2<f64>,
        a_left: &Array2<f64>,
        a_right: &Array2<f64>,
        mat: &Array2<f64>,
    ) -> Array2<f64> {
        // Columnwise extension of apply_hadamard_gram:
        //   out[:,j] = ((Z A_left Zᵀ) ⊙ (Z A_right Zᵀ)) mat[:,j].
        //
        // In the Firth derivatives this is used with:
        //   A_left = K_r, A_right = K_r        for H_w⊙H_w actions,
        //   A_left = K_r, A_right = A_u        for H_w⊙Nbar_u actions,
        // and symmetric variants in mixed second-direction terms.
        let mut out = Array2::<f64>::zeros(mat.raw_dim());
        for col in 0..mat.ncols() {
            let v = mat.column(col).to_owned();
            let y = Self::apply_hadamard_gram(z, a_left, a_right, &v);
            out.column_mut(col).assign(&y);
        }
        out
    }

    pub(super) fn build_firth_dense_operator(
        x_dense: &Array2<f64>,
        eta: &Array1<f64>,
    ) -> Result<FirthDenseOperator, EstimationError> {
        FirthDenseOperator::build(x_dense, eta)
    }

    pub(super) fn firth_direction(op: &FirthDenseOperator, u: &Array1<f64>) -> FirthDirection {
        op.direction(u)
    }

    pub(super) fn firth_direction_from_deta(
        op: &FirthDenseOperator,
        deta: Array1<f64>,
    ) -> FirthDirection {
        op.direction_from_deta(deta)
    }

    pub(super) fn firth_hphi_direction(
        op: &FirthDenseOperator,
        dir: &FirthDirection,
    ) -> Array2<f64> {
        op.hphi_direction(dir)
    }

    pub(super) fn firth_hphi_direction_apply(
        op: &FirthDenseOperator,
        dir: &FirthDirection,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        op.hphi_direction_apply(dir, rhs)
    }

    pub(super) fn firth_hphi_second_direction_apply(
        op: &FirthDenseOperator,
        u: &FirthDirection,
        v: &FirthDirection,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        op.hphi_second_direction_apply(u, v, rhs)
    }

    pub(super) fn firth_exact_tau_kernel(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        beta: &Array1<f64>,
        include_hphi_tau_kernel: bool,
    ) -> FirthTauExactKernel {
        op.exact_tau_kernel(x_tau, beta, include_hphi_tau_kernel)
    }

    pub(super) fn firth_hphi_tau_partial_prepare(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> FirthTauPartialKernel {
        op.hphi_tau_partial_prepare(x_tau, beta)
    }

    pub(super) fn firth_hphi_tau_partial_apply(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        kernel: &FirthTauPartialKernel,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        op.hphi_tau_partial_apply(x_tau, kernel, rhs)
    }

    pub(super) fn firth_hphi_trace_apply_combined(
        op: &FirthDenseOperator,
        dir: &FirthDirection,
        x_tau: &Array2<f64>,
        tau_kernel: Option<&FirthTauPartialKernel>,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        op.hphi_trace_apply_combined(dir, x_tau, tau_kernel, rhs)
    }
}

impl FirthDenseOperator {
    pub(crate) fn build(
        x_dense: &Array2<f64>,
        eta: &Array1<f64>,
    ) -> Result<FirthDenseOperator, EstimationError> {
        // Precompute dense Firth objects at current β̂ for:
        //   Φ(β) = 0.5 log|I(β)|, I = Xᵀ W X.
        //
        // IMPORTANT IDENTIFIABILITY NOTE
        // ------------------------------
        // For rank-deficient design matrices X, I = Xᵀ W X is singular for all β
        // (assuming w_i > 0). The mathematically coherent Jeffreys/Firth term is
        // therefore the identifiable-subspace form:
        //   Φ(β) = 0.5 log|I(β)|_+,
        // with differential:
        //   dΦ = 0.5 tr(I_+^† dI),
        // where I_+^† is the positive-part pseudoinverse.
        //
        // For binomial-logit with finite eta, 0 < w_i <= 1/4, so W is SPD and
        // Null(X'WX)=Null(X). Therefore singular directions are structural
        // (from X) and independent of beta, which is why fixed-Q reduced-space
        // derivatives are exact in this regime.
        //
        // This implementation uses the fixed identifiable-basis route directly:
        //   I_r = (XQ)ᵀ W (XQ),  I_+^† = Q I_r^{-1} Qᵀ,  log|I|_+ = log|I_r|.
        //
        // Why `eta` (not `mu`) enters here:
        //   all logistic weight derivatives are functions of eta through
        //   mu(eta)=sigmoid(eta), and exact derivative consistency is preserved by
        //   generating (w, w', w'', w''', w'''') from one coherent eta source.
        // Using eta avoids any mismatch from externally clamped/post-processed mu.
        //
        // We cache reduced operators so derivatives are exact but matrix-free in n:
        //   K_r = I_r^{-1},  h = diag(X_r K_r X_rᵀ),  B = diag(w')X,
        // along with logistic derivatives w', w'', w''', w'''' to evaluate:
        //   H_φ      = ∇²_β Φ,
        //   D H_φ[u],
        //   D² H_φ[u,v]
        // exactly via reduced-space products (no explicit high-order tensors).
        let n = x_dense.nrows();
        if eta.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "Firth operator shape mismatch: n_rows={}, eta_len={}",
                n,
                eta.len()
            )));
        }
        let p = x_dense.ncols();
        let mut w = Array1::<f64>::zeros(n);
        let mut w1 = Array1::<f64>::zeros(n);
        let mut w2 = Array1::<f64>::zeros(n);
        let mut w3 = Array1::<f64>::zeros(n);
        let mut w4 = Array1::<f64>::zeros(n);
        let logistic_weight = |z: f64| {
            let ez = z.clamp(-700.0, 700.0);
            let mz = if ez >= 0.0 {
                1.0 / (1.0 + (-ez).exp())
            } else {
                let ex = ez.exp();
                ex / (1.0 + ex)
            };
            mz * (1.0 - mz)
        };
        let h12 = 2e-5;
        let h34 = 1e-3;
        let w2_fd = |z: f64| {
            (logistic_weight(z + h12) - 2.0 * logistic_weight(z) + logistic_weight(z - h12))
                / (h12 * h12)
        };
        let w3_fd = |z: f64| (w2_fd(z + h34) - w2_fd(z - h34)) / (2.0 * h34);
        for i in 0..n {
            // Compute mu and logistic derivatives from eta with stable algebra.
            // Clamping eta to [-700, 700] avoids exp overflow while preserving
            // strictly positive working weights in floating point:
            //   mu in (0,1), w = mu(1-mu) > 0.
            let ei = eta[i].clamp(-700.0, 700.0);
            let mi = if ei >= 0.0 {
                1.0 / (1.0 + (-ei).exp())
            } else {
                let ex = ei.exp();
                ex / (1.0 + ex)
            };
            let wi = mi * (1.0 - mi);
            let ti = 1.0 - 2.0 * mi;
            let wi2 = wi * wi;
            let wi3 = wi2 * wi;
            w[i] = wi;
            w1[i] = wi * ti;
            w2[i] = wi * ti * ti - 2.0 * wi2;
            w3[i] = wi * ti * ti * ti - 8.0 * wi2 * ti;
            let w4_fd = (w3_fd(ei + h34) - w3_fd(ei - h34)) / (2.0 * h34);
            // Keep a numerically stable fourth derivative in regimes where
            // high-order finite differences of logistic weights are sensitive.
            w4[i] = if w4_fd.is_finite() {
                w4_fd
            } else {
                wi * ti * ti * ti * ti - 22.0 * wi2 * ti * ti + 16.0 * wi3
            };
        }
        // Build a fixed identifiable basis Q for Col(Xᵀ).
        //
        // Exact fast path: if XᵀX is already SPD, then X has full column rank and
        // the identifiable subspace is the whole coefficient space. In that case
        // we can set Q = I exactly and skip the expensive eigendecomposition.
        // This preserves the same Jeffreys/Firth objective because
        //   Col(Xᵀ) = R^p, I_+^† = I^{-1}, and log|XᵀWX|_+ = log|XᵀWX|.
        //
        // We only fall back to the spectral route when X is rank-deficient.
        let gram = fast_atb(x_dense, x_dense);
        let (q_basis, x_reduced) = if gram.cholesky(Side::Lower).is_ok() {
            (Array2::<f64>::eye(p), x_dense.clone())
        } else {
            let (evals, evecs) = gram
                .eigh(Side::Lower)
                .map_err(EstimationError::EigendecompositionFailed)?;
            let max_eval = evals.iter().copied().fold(0.0_f64, f64::max).max(1.0);
            let tol = (p.max(1) as f64) * f64::EPSILON * max_eval;
            let keep: Vec<usize> = evals
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v > tol { Some(i) } else { None })
                .collect();
            let r = keep.len();
            if r == 0 {
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }

            let mut q_basis = Array2::<f64>::zeros((p, r));
            for (col_idx, &eig_idx) in keep.iter().enumerate() {
                q_basis.column_mut(col_idx).assign(&evecs.column(eig_idx));
            }
            let x_reduced = fast_ab(x_dense, &q_basis);
            (q_basis, x_reduced)
        };
        let r = q_basis.ncols();

        let weighted_x_reduced = RemlState::row_scale(&x_reduced, &w);
        // Reduced Fisher on the identifiable subspace:
        //   I_r = X_rᵀ W X_r = Zᵀ Z.
        // Under finite-logit eta, W has strictly positive diagonal entries and
        // X_r has full column rank by construction, so I_r is SPD.
        let fisher_reduced = fast_atb(&x_reduced, &weighted_x_reduced);
        // Smooth-regime diagnostic:
        // for exact pseudodet derivatives we require I_r to stay SPD on the
        // fixed identifiable subspace. Emit a warning when I_r appears close
        // to singular, because this is where Jeffreys/Firth curvature becomes
        // numerically fragile and active-subspace assumptions may fail.
        if let Ok((eigvals_ir, _)) = fisher_reduced.eigh(Side::Lower) {
            let max_ev = eigvals_ir.iter().copied().fold(0.0_f64, f64::max).max(1.0);
            let min_ev = eigvals_ir
                .iter()
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
                .fold(f64::INFINITY, f64::min);
            if min_ev.is_finite() {
                let rel = min_ev / max_ev;
                if rel < 1e-10 {
                    log::warn!(
                        "[REML/Firth] reduced Fisher I_r is near-singular (min/max={:.3e}/{:.3e}, rel={:.3e}); exact derivatives may be ill-conditioned near active-subspace boundaries.",
                        min_ev,
                        max_ev,
                        rel
                    );
                }
            }
        }

        let mut k_reduced = Array2::<f64>::zeros((r, r));
        if r > 0 {
            // Exact reduced-space inverse: I_r should be SPD for finite-logit eta
            // with full-rank X_r. We intentionally avoid diagonal ridge here.
            // If this fails, the current point is outside the smooth regime
            // required by exact Jeffreys/Firth derivatives.
            let chol = fisher_reduced.cholesky(Side::Lower).map_err(|_| {
                EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                }
            })?;
            for col in 0..r {
                let mut e_col = Array1::<f64>::zeros(r);
                e_col[col] = 1.0;
                let solved = chol.solve_vec(&e_col);
                k_reduced.column_mut(col).assign(&solved);
            }
        }
        // Unweighted reduced design enters M = X_r K_r X_r' and P = M⊙M.
        // Keep this object unweighted to match the analytic formulas used in
        // H_phi, D(H_phi), and D²(H_phi).
        let z_reduced = x_reduced.clone();
        let h_diag = if r > 0 {
            RemlState::reduced_diag_gram(&z_reduced, &k_reduced)
        } else {
            Array1::<f64>::zeros(n)
        };
        let x_dense_t = x_dense.t().to_owned();
        let b_base = RemlState::row_scale(x_dense, &w1);
        let b_base_t = b_base.t().to_owned();
        let p_b_base =
            RemlState::apply_hadamard_gram_to_matrix(&z_reduced, &k_reduced, &k_reduced, &b_base);
        Ok(FirthDenseOperator {
            x_dense: x_dense.clone(),
            x_dense_t,
            q_basis,
            x_reduced,
            z_reduced,
            k_reduced,
            h_diag,
            w,
            w1,
            w2,
            w3,
            w4,
            b_base,
            b_base_t,
            p_b_base,
        })
    }

    pub(crate) fn direction(&self, u: &Array1<f64>) -> FirthDirection {
        let deta = self.x_dense.dot(u);
        self.direction_from_deta(deta)
    }

    pub(crate) fn direction_from_deta(&self, deta: Array1<f64>) -> FirthDirection {
        // Directional building blocks for u:
        //   δη_u = X u
        //   I_u  = Xᵀ diag(w' ⊙ δη_u) X
        //   T_u  = K I_u K
        //   N_u  = X T_u Xᵀ
        //   Dh[u] = -diag(N_u)
        //   P_u   = D(M⊙M)[u] = -2(M⊙N_u)
        // and B_u = diag(w'' ⊙ δη_u) X.
        //
        // In this implementation, active-subspace ambiguity is removed by fixed Q:
        // K is represented by K_r = I_r^{-1} in reduced coordinates, so
        //   A_u = K_r G_u K_r
        // is exact for logit with finite eta (w_i > 0) and fixed rank(X).
        // s_u is the diagonal weight for D I[u]:
        //   D I[u] = Xᵀ diag(s_u) X,  s_u = w' ⊙ (X u).
        let s_u = &self.w1 * &deta;
        // G_u = X_rᵀ diag(s_u) X_r,  A_u = K_r G_u K_r.
        // These are reduced-space forms of
        //   I_u and T_u = K I_u K
        // from the full-space derivation.
        let g_u_reduced = RemlState::reduced_weighted_gram(&self.x_reduced, &s_u);
        let k_g_u = self.k_reduced.dot(&g_u_reduced);
        let a_u_reduced = k_g_u.dot(&self.k_reduced);
        // Dh[u] = -diag(N_u),  N_u = X T_u Xᵀ, represented here as
        //   N_u = Z A_u Zᵀ in weighted reduced coordinates.
        let dh = -RemlState::reduced_diag_gram(&self.z_reduced, &a_u_reduced);
        let b_u_vec = &self.w2 * &deta;
        let b_u_base = &self.x_dense * &b_u_vec.view().insert_axis(Axis(1));
        let b_u_base_t = b_u_base.t().to_owned();
        let p_bu_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &b_u_base,
        );
        let mut p_u_b_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &a_u_reduced,
            &self.b_base,
        );
        p_u_b_base.mapv_inplace(|val| -2.0 * val);
        FirthDirection {
            deta,
            g_u_reduced,
            a_u_reduced,
            dh,
            b_u_vec,
            b_u_base,
            b_u_base_t,
            p_bu_base,
            p_u_b_base,
        }
    }

    pub(crate) fn hphi_direction_apply(
        &self,
        dir: &FirthDirection,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        // Matrix-free apply of D(H_phi)[u] to a block V:
        //   D(H_phi)[u] V
        // = 0.5[ Xᵀ(c_u ⊙ (X V))
        //       - B_uᵀ P (B V) - Bᵀ P (B_u V) - Bᵀ P_u (B V) ].
        // This avoids dense p×p materialization and is used by sparse exact
        // trace contractions through tr(H^{-1} ·).
        let eta_v = self.x_dense.dot(rhs);
        let q_v = &eta_v * &self.w1.view().insert_axis(Axis(1));
        let m_q_v = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &q_v,
        );
        let bu_vec = &dir.b_u_vec;
        let m_bu_v = fast_ab(&dir.p_bu_base, rhs);
        let p_u_q_v = fast_ab(&dir.p_u_b_base, rhs);
        let c_u = &(&self.w3 * &dir.deta) * &self.h_diag + &(&self.w2 * &dir.dh);
        let diag_term = self
            .x_dense_t
            .dot(&(&eta_v * &c_u.view().insert_axis(Axis(1))));
        let term1 = self
            .x_dense_t
            .dot(&(&m_q_v * &bu_vec.view().insert_axis(Axis(1))));
        let term2 = self
            .x_dense_t
            .dot(&(&m_bu_v * &self.w1.view().insert_axis(Axis(1))));
        let term3 = self
            .x_dense_t
            .dot(&(&p_u_q_v * &self.w1.view().insert_axis(Axis(1))));
        0.5 * (diag_term - (term1 + term2 + term3))
    }

    pub(crate) fn hphi_direction(&self, dir: &FirthDirection) -> Array2<f64> {
        let p = self.x_dense.ncols();
        let eye = Array2::<f64>::eye(p);
        let mut out = self.hphi_direction_apply(dir, &eye);
        // Exact first directional derivative of H_φ:
        //   D H_φ[u]
        //   = 0.5 [ Xᵀ diag(c_u) X
        //           - (B_uᵀ P B + Bᵀ P B_u + Bᵀ P_u B) ],
        // where
        //   c_u = w''' ⊙ δη_u ⊙ h + w'' ⊙ Dh[u],
        //   B   = diag(w') X.
        //
        // Matrix-free contraction map used below:
        //   Bᵀ P B      via apply_hadamard_gram_to_matrix(Z, K_r, K_r, B)
        //   Bᵀ P_u B    via apply_hadamard_gram_to_matrix(Z, K_r, A_u, B), then *(-2)
        // where P = M⊙M and P_u = -2(M⊙N_u), but M/N_u are never formed explicitly.
        for i in 0..out.nrows() {
            for j in 0..i {
                let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
                out[[i, j]] = avg;
                out[[j, i]] = avg;
            }
        }
        out
    }

    pub(crate) fn hphi_second_direction_apply(
        &self,
        u: &FirthDirection,
        v: &FirthDirection,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        // Exact mixed second directional derivative:
        //   D² H_φ[u,v] = 0.5 [ Xᵀ diag(c_uv) X - D²J₂[u,v] ], J₂ = Bᵀ P B.
        // Implemented with matrix identities for N_{u,v}, P_{u,v}, and the
        // nine-term expansion of D²J₂[u,v].
        //
        // Because we parameterize Phi through fixed-rank I_r = X_rᵀ W X_r (SPD for
        // finite-logit eta), this mixed derivative is evaluated on a smooth
        // manifold without dynamic active-set switching in the Firth block.
        //
        // The nine contraction terms below are the explicit D²J₂[u,v] expansion,
        // each computed through reduced Hadamard-Gram operators.
        let deta_uv = &u.deta * &v.deta;
        // Mixed reduced Gram:
        //   G_uv = X_rᵀ diag(w'' ⊙ (Xu) ⊙ (Xv)) X_r.
        let s_uv = &self.w2 * &deta_uv;
        let g_uv_reduced = RemlState::reduced_weighted_gram(&self.x_reduced, &s_uv);
        let k_g_uv = self.k_reduced.dot(&g_uv_reduced);
        let k_g_v = self.k_reduced.dot(&v.g_u_reduced);
        let k_g_u = self.k_reduced.dot(&u.g_u_reduced);
        // Reduced form of:
        //   T_{u,v} = K I_{u,v} K - K I_v K I_u K - K I_u K I_v K.
        let a_uv_reduced = k_g_uv.dot(&self.k_reduced)
            - k_g_v.dot(&k_g_u).dot(&self.k_reduced)
            - k_g_u.dot(&k_g_v).dot(&self.k_reduced);
        let d2h = -RemlState::reduced_diag_gram(&self.z_reduced, &a_uv_reduced);
        // Implements mixed diagonal coefficient:
        //   c_uv = w'''' ⊙ (Xu) ⊙ (Xv) ⊙ h
        //          + w''' ⊙ ((Xu) ⊙ Dh[v] + (Xv) ⊙ Dh[u])
        //          + w'' ⊙ D²h[u,v].
        let c_uv = &(&(&self.w4 * &deta_uv) * &self.h_diag)
            + &(&self.w3 * &(&u.deta * &v.dh))
            + &(&self.w3 * &(&v.deta * &u.dh))
            + &(&self.w2 * &d2h);

        let eta_rhs = self.x_dense.dot(rhs);
        let diag_term = self
            .x_dense_t
            .dot(&(&eta_rhs * &c_uv.view().insert_axis(Axis(1))));

        let b_uv_vec = &self.w3 * &deta_uv;
        let b_u_base = &u.b_u_base;
        let b_v_base = &v.b_u_base;
        let b_uv_base = &self.x_dense * &b_uv_vec.view().insert_axis(Axis(1));

        // Linearity in the rhs argument lets us precompute the expensive
        // Hadamard-Gram operator on the full base blocks B, B_u, B_v, B_uv once,
        // then post-multiply by rhs. This preserves the exact operator while
        // avoiding repeated O(n r^2 c) work for every rhs block.
        let p_b_rhs = fast_ab(&self.p_b_base, rhs);
        let p_bu_base = &u.p_bu_base;
        let p_bu_rhs = fast_ab(&p_bu_base, rhs);
        let p_bv_base = &v.p_bu_base;
        let p_bv_rhs = fast_ab(&p_bv_base, rhs);
        let p_buv_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &b_uv_base,
        );
        let p_buv_rhs = fast_ab(&p_buv_base, rhs);

        let p_v_b_base = &v.p_u_b_base;
        let p_v_b_rhs = fast_ab(&p_v_b_base, rhs);
        let mut p_v_bu_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &v.a_u_reduced,
            &b_u_base,
        );
        p_v_bu_base.mapv_inplace(|val| -2.0 * val);
        let p_v_bu_rhs = fast_ab(&p_v_bu_base, rhs);
        let p_u_b_base = &u.p_u_b_base;
        let p_u_b_rhs = fast_ab(&p_u_b_base, rhs);
        let mut p_u_bv_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &u.a_u_reduced,
            &b_v_base,
        );
        p_u_bv_base.mapv_inplace(|val| -2.0 * val);
        let p_u_bv_rhs = fast_ab(&p_u_bv_base, rhs);

        let p_nu_nv_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &u.a_u_reduced,
            &v.a_u_reduced,
            &self.b_base,
        );
        let p_hw_nuv_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &a_uv_reduced,
            &self.b_base,
        );
        let p_uv_base = 2.0 * p_nu_nv_base - 2.0 * p_hw_nuv_base;
        let p_uv_rhs = fast_ab(&p_uv_base, rhs);
        let left_uv = b_uv_base.t().to_owned();
        let left_u = &u.b_u_base_t;
        let left_v = &v.b_u_base_t;
        let left_b = &self.b_base_t;

        // Nine-term expansion of D²J₂[u,v] with J₂ = Bᵀ P B.
        let d2_terms = [
            fast_ab(&left_uv, &p_b_rhs),
            fast_ab(&left_b, &p_buv_rhs),
            fast_ab(&left_u, &p_bv_rhs),
            fast_ab(&left_v, &p_bu_rhs),
            fast_ab(&left_u, &p_v_b_rhs),
            fast_ab(&left_b, &p_v_bu_rhs),
            fast_ab(&left_v, &p_u_b_rhs),
            fast_ab(&left_b, &p_u_bv_rhs),
            fast_ab(&left_b, &p_uv_rhs),
        ];
        let mut d2_j2 = Array2::<f64>::zeros((p, rhs.ncols()));
        for term in d2_terms {
            d2_j2 += &term;
        }

        0.5 * (diag_term - d2_j2)
    }

    pub(super) fn rowwise_dot(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
        debug_assert_eq!(a.nrows(), b.nrows());
        debug_assert_eq!(a.ncols(), b.ncols());
        let mut out = Array1::<f64>::zeros(a.nrows());
        for i in 0..a.nrows() {
            let mut acc = 0.0_f64;
            for j in 0..a.ncols() {
                acc += a[[i, j]] * b[[i, j]];
            }
            out[i] = acc;
        }
        out
    }

    pub(super) fn rowwise_bilinear(
        a: &Array2<f64>,
        m: &Array2<f64>,
        b: &Array2<f64>,
    ) -> Array1<f64> {
        // Returns vector with entries a_iᵀ M b_i for each row i.
        debug_assert_eq!(a.nrows(), b.nrows());
        debug_assert_eq!(a.ncols(), m.nrows());
        debug_assert_eq!(b.ncols(), m.ncols());
        let am = fast_ab(a, m);
        Self::rowwise_dot(&am, b)
    }

    pub(crate) fn dot_i_and_h(
        &self,
        x_tau: &Array2<f64>,
        deta: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>) {
        // Reduced Fisher directional derivative under fixed identifiable basis:
        //   I_r = X_r' W X_r
        //   I_r,tau = X_{r,tau}' W X_r + X_r' W X_{r,tau} + X_r' W_tau X_r
        // with W_tau = diag(w' ⊙ eta_tau).
        //
        // Leverage derivative used by Firth score partial:
        //   h_i = x_{r,i}' K_r x_{r,i}, K_r = I_r^{-1}
        //   h_tau = 2*diag(X_{r,tau} K_r X_r') + diag(X_r K_{r,tau} X_r')
        //   K_{r,tau} = -K_r I_{r,tau} K_r.
        //
        // This is exactly the fixed-beta directional derivative required by
        //   (g_phi)_tau and Phi_tau in the Jeffreys/Firth design-moving path:
        //   I_{r,tau}|beta = X_{r,tau}' W X_r + X_r' W X_{r,tau}
        //                    + X_r' diag(w' ⊙ eta_tau|beta) X_r,
        //   eta_tau|beta = X_tau beta.
        //
        // We return:
        //   dot_i  = I_{r,tau}|beta,
        //   dot_h  = h_tau|beta.
        let x_tau_reduced = fast_ab(x_tau, &self.q_basis);

        let wx_reduced = RemlState::row_scale(&self.x_reduced, &self.w);
        let wx_tau_reduced = RemlState::row_scale(&x_tau_reduced, &self.w);

        let dw = &self.w1 * deta;
        let dwx_reduced = RemlState::row_scale(&self.x_reduced, &dw);

        let dot_i = fast_ab(&x_tau_reduced.t().to_owned(), &wx_reduced)
            + fast_ab(&self.x_reduced.t().to_owned(), &wx_tau_reduced)
            + fast_ab(&self.x_reduced.t().to_owned(), &dwx_reduced);

        let dot_k = -self.k_reduced.dot(&dot_i).dot(&self.k_reduced);
        let x_tauk = x_tau_reduced.dot(&self.k_reduced);
        let dot_h_explicit = 2.0 * Self::rowwise_dot(&x_tauk, &self.x_reduced);
        let dot_h_implicit = Self::rowwise_dot(&self.x_reduced.dot(&dot_k), &self.x_reduced);
        let dot_h = dot_h_explicit + dot_h_implicit;
        (dot_i, dot_h)
    }

    pub(crate) fn exact_tau_kernel(
        &self,
        x_tau: &Array2<f64>,
        beta: &Array1<f64>,
        include_hphi_tau_kernel: bool,
    ) -> FirthTauExactKernel {
        // Shared exact tau-partial bundle used by both dense and sparse paths:
        //   (g_phi)_tau | beta-fixed,
        //   Phi_tau | beta-fixed,
        // and optional H_{phi,tau}|beta kernel for later matrix-free applies.
        //
        // Closed forms (reduced Fisher, fixed active subspace):
        //   Phi = 0.5 log|I_r|, I_r = X_r' W X_r, K_r = I_r^{-1}
        //   Phi_tau|beta = 0.5 tr(K_r I_{r,tau}).
        //
        //   (g_phi)_tau = Phi_beta,tau
        //               = 0.5 X_tau' (w1 .* h)
        //                 + 0.5 X'((w2 .* eta_tau) .* h + w1 .* h_tau),
        //   where
        //     h_i = x_{r,i}' K_r x_{r,i},
        //     h_tau = 2*diag(X_{r,tau} K_r X_r') + diag(X_r K_{r,tau} X_r'),
        //     K_{r,tau} = -K_r I_{r,tau} K_r.
        //
        // These are the literal trace/hat closed forms for Phi_tau and Phi_beta,tau.
        let deta_partial = x_tau.dot(beta);
        let (dot_i_partial, dot_h_partial) = self.dot_i_and_h(x_tau, &deta_partial);

        let first = 0.5 * x_tau.t().dot(&(&self.w1 * &self.h_diag));
        let second_vec =
            &(&(&self.w2 * &deta_partial) * &self.h_diag) + &(&self.w1 * &dot_h_partial);
        let second = 0.5 * self.x_dense.t().dot(&second_vec);
        let g_phi_tau = first + second;
        let phi_tau_partial = 0.5 * RemlState::trace_product(&self.k_reduced, &dot_i_partial);

        let tau_kernel = if include_hphi_tau_kernel {
            Some(self.hphi_tau_partial_prepare_from_partials(
                x_tau,
                &deta_partial,
                dot_h_partial,
                dot_i_partial,
            ))
        } else {
            None
        };
        FirthTauExactKernel {
            g_phi_tau,
            phi_tau_partial,
            tau_kernel,
        }
    }

    pub(crate) fn hphi_tau_partial_prepare(
        &self,
        x_tau: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> FirthTauPartialKernel {
        let deta_partial = x_tau.dot(beta);
        let (dot_i_partial, dot_h_partial) = self.dot_i_and_h(x_tau, &deta_partial);
        self.hphi_tau_partial_prepare_from_partials(
            x_tau,
            &deta_partial,
            dot_h_partial,
            dot_i_partial,
        )
    }

    fn hphi_tau_partial_prepare_from_partials(
        &self,
        x_tau: &Array2<f64>,
        deta_partial: &Array1<f64>,
        dot_h_partial: Array1<f64>,
        dot_i_partial: Array2<f64>,
    ) -> FirthTauPartialKernel {
        let dot_w1 = &self.w2 * deta_partial;
        let dot_w2 = &self.w3 * deta_partial;
        let x_tau_reduced = fast_ab(x_tau, &self.q_basis);
        let dot_k = -self.k_reduced.dot(&dot_i_partial).dot(&self.k_reduced);
        FirthTauPartialKernel {
            dot_w1,
            dot_w2,
            dot_h_partial,
            x_tau_reduced,
            dot_k_reduced: dot_k,
        }
    }

    pub(crate) fn apply_pbar_to_matrix(&self, mat: &Array2<f64>) -> Array2<f64> {
        // Applies P̄ = (X_r K_r X_rᵀ)⊙(X_r K_r X_rᵀ) to each column of mat.
        RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            mat,
        )
    }

    pub(crate) fn apply_mtau_to_matrix(
        &self,
        kernel: &FirthTauPartialKernel,
        mat: &Array2<f64>,
    ) -> Array2<f64> {
        // Exact apply of
        //   M_tau = d/dtau[(P⊙P)]|_{beta fixed} = 2(P⊙P_tau)
        // without building dense n×n objects.
        //
        // Decomposition:
        //   P = Z K Zᵀ, Z = X_r
        //   P_tau = Z_tau K Zᵀ + Z K Z_tauᵀ + Z dotK Zᵀ
        // and for each vector v:
        //   (P⊙(Z_tau K Zᵀ))v   : rowwise bilinear with K (Zᵀdiag(v)Z) K
        //   (P⊙(Z K Z_tauᵀ))v   : diag_Z( K (Zᵀdiag(v)Z_tau) K )
        //   (P⊙(Z dotK Zᵀ))v    : Hadamard-Gram apply with (K, dotK).
        if mat.nrows() != self.x_dense.nrows() || mat.ncols() == 0 {
            return Array2::<f64>::zeros(mat.raw_dim());
        }
        let mut out = Array2::<f64>::zeros(mat.raw_dim());
        for col in 0..mat.ncols() {
            let v = mat.column(col).to_owned();
            let s_zz = RemlState::reduced_weighted_gram(&self.z_reduced, &v);
            let m_zz = self.k_reduced.dot(&s_zz).dot(&self.k_reduced);
            let t1 = Self::rowwise_bilinear(&self.z_reduced, &m_zz, &kernel.x_tau_reduced);

            let s_zt =
                RemlState::reduced_cross_weighted_gram(&self.z_reduced, &kernel.x_tau_reduced, &v);
            let m_zt = self.k_reduced.dot(&s_zt).dot(&self.k_reduced);
            let t2 = RemlState::reduced_diag_gram(&self.z_reduced, &m_zt);

            let t3 = RemlState::apply_hadamard_gram(
                &self.z_reduced,
                &self.k_reduced,
                &kernel.dot_k_reduced,
                &v,
            );

            let y = 2.0 * (t1 + t2 + t3);
            out.column_mut(col).assign(&y);
        }
        out
    }

    pub(crate) fn hphi_tau_partial_apply(
        &self,
        x_tau: &Array2<f64>,
        kernel: &FirthTauPartialKernel,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        // Matrix-free block apply of H_phi,tau|beta:
        //   H_phi,tau|beta(V) = 0.5 [ X_tau' r(V) + X' r_tau(V) ].
        //
        // Tensor identity behind this apply:
        //   H_phi,tau|beta = Phi_beta,beta,tau
        // and for test vectors b1,b2 (matrix columns V are batched b2's):
        //   Phi_beta,beta,tau[b1,b2]
        //   = 0.5[
        //       tr(I^{-1} I_{b1,b2,tau})
        //       - tr(I^{-1} I_{b1,b2} I^{-1} I_tau)
        //       - tr(I^{-1} I_{b1,tau} I^{-1} I_{b2})
        //       - tr(I^{-1} I_{b2,tau} I^{-1} I_{b1})
        //       + 2 tr(I^{-1} I_{b1} I^{-1} I_{b2} I^{-1} I_tau)
        //     ].
        // This routine evaluates that form in reduced coordinates without forming
        // dense 3rd-order tensors explicitly.
        let eta_v = self.x_dense.dot(rhs);
        let eta_v_tau = x_tau.dot(rhs);
        let q_v = &eta_v * &self.w1.view().insert_axis(Axis(1));
        let q_v_tau = &eta_v * &kernel.dot_w1.view().insert_axis(Axis(1))
            + &eta_v_tau * &self.w1.view().insert_axis(Axis(1));
        let m_q_v = self.apply_pbar_to_matrix(&q_v);
        let m_q_v_tau =
            self.apply_mtau_to_matrix(kernel, &q_v) + self.apply_pbar_to_matrix(&q_v_tau);
        let r_v = &(&eta_v * &self.w2.view().insert_axis(Axis(1)))
            * &self.h_diag.view().insert_axis(Axis(1))
            - &(&m_q_v * &self.w1.view().insert_axis(Axis(1)));
        let r_v_tau = (&(&eta_v * &kernel.dot_w2.view().insert_axis(Axis(1)))
            + &(&eta_v_tau * &self.w2.view().insert_axis(Axis(1))))
            * &self.h_diag.view().insert_axis(Axis(1))
            + &(&eta_v * &self.w2.view().insert_axis(Axis(1)))
                * &kernel.dot_h_partial.view().insert_axis(Axis(1))
            - &(&m_q_v * &kernel.dot_w1.view().insert_axis(Axis(1))
                + &m_q_v_tau * &self.w1.view().insert_axis(Axis(1)));
        0.5 * (x_tau.t().dot(&r_v) + self.x_dense.t().dot(&r_v_tau))
    }

    pub(crate) fn hphi_trace_apply_combined(
        &self,
        dir: &FirthDirection,
        x_tau: &Array2<f64>,
        tau_kernel: Option<&FirthTauPartialKernel>,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }

        // Shared block intermediates:
        // eta_v, q_v, and P(q_v) are reused by both
        //   H_{phi,tau}|beta(·) and D(H_phi)[dir](·).
        let eta_v = self.x_dense.dot(rhs);
        let q_v = &eta_v * &self.w1.view().insert_axis(Axis(1));

        // D(H_phi)[dir] contribution:
        let m_q_v = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &q_v,
        );
        let bu_vec = &self.w2 * &dir.deta;
        let bu_v = &eta_v * &bu_vec.view().insert_axis(Axis(1));
        let m_bu_v = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &bu_v,
        );
        let mut p_u_q_v = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &dir.a_u_reduced,
            &q_v,
        );
        p_u_q_v.mapv_inplace(|v| -2.0 * v);
        let c_u = &(&self.w3 * &dir.deta) * &self.h_diag + &(&self.w2 * &dir.dh);
        let diag_term = self
            .x_dense
            .t()
            .dot(&(&eta_v * &c_u.view().insert_axis(Axis(1))));
        let term1 = self
            .x_dense
            .t()
            .dot(&(&m_q_v * &bu_vec.view().insert_axis(Axis(1))));
        let term2 = self
            .x_dense
            .t()
            .dot(&(&m_bu_v * &self.w1.view().insert_axis(Axis(1))));
        let term3 = self
            .x_dense
            .t()
            .dot(&(&p_u_q_v * &self.w1.view().insert_axis(Axis(1))));
        let mut out = 0.5 * (diag_term - (term1 + term2 + term3));

        if let Some(kernel) = tau_kernel {
            // H_{phi,tau}|beta contribution, reusing eta_v and q_v.
            let eta_v_tau = x_tau.dot(rhs);
            let q_v_tau = &eta_v * &kernel.dot_w1.view().insert_axis(Axis(1))
                + &eta_v_tau * &self.w1.view().insert_axis(Axis(1));
            let m_q_v_kernel = self.apply_pbar_to_matrix(&q_v);
            let m_q_v_tau =
                self.apply_mtau_to_matrix(kernel, &q_v) + self.apply_pbar_to_matrix(&q_v_tau);
            let r_v = &(&eta_v * &self.w2.view().insert_axis(Axis(1)))
                * &self.h_diag.view().insert_axis(Axis(1))
                - &(&m_q_v_kernel * &self.w1.view().insert_axis(Axis(1)));
            let r_v_tau = (&(&eta_v * &kernel.dot_w2.view().insert_axis(Axis(1)))
                + &(&eta_v_tau * &self.w2.view().insert_axis(Axis(1))))
                * &self.h_diag.view().insert_axis(Axis(1))
                + &(&eta_v * &self.w2.view().insert_axis(Axis(1)))
                    * &kernel.dot_h_partial.view().insert_axis(Axis(1))
                - &(&m_q_v_kernel * &kernel.dot_w1.view().insert_axis(Axis(1))
                    + &m_q_v_tau * &self.w1.view().insert_axis(Axis(1)));
            out += &(0.5 * (x_tau.t().dot(&r_v) + self.x_dense.t().dot(&r_v_tau)));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    fn logistic_weight(eta: f64) -> f64 {
        let e = eta.clamp(-700.0, 700.0);
        let mu = if e >= 0.0 {
            1.0 / (1.0 + (-e).exp())
        } else {
            let ex = e.exp();
            ex / (1.0 + ex)
        };
        mu * (1.0 - mu)
    }

    fn d1_fd(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    fn d2_fd(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
    }

    fn firth_phi_value(x: &Array2<f64>, beta: &Array1<f64>) -> f64 {
        let eta = x.dot(beta);
        let op = FirthDenseOperator::build(x, &eta).expect("firth operator");
        let fisher = fast_atb(&op.x_reduced, &RemlState::row_scale(&op.x_reduced, &op.w));
        let (evals, _) = fisher
            .eigh(Side::Lower)
            .expect("reduced Fisher eigendecomp");
        0.5 * evals.iter().map(|v| v.ln()).sum::<f64>()
    }

    fn firth_grad_phi(x: &Array2<f64>, beta: &Array1<f64>) -> Array1<f64> {
        let eta = x.dot(beta);
        let op = FirthDenseOperator::build(x, &eta).expect("firth operator");
        0.5 * x.t().dot(&(&op.w1 * &op.h_diag))
    }

    #[test]
    fn firth_logistic_weight_derivatives_match_finite_difference() {
        let x = array![
            [1.0, -1.1, 0.2],
            [1.0, -0.5, -0.6],
            [1.0, 0.0, 0.3],
            [1.0, 0.8, -0.4],
            [1.0, 1.2, 0.7],
        ];
        let beta = array![0.15, -0.6, 0.35];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");

        let h12 = 2e-5;
        let h34 = 1e-3;
        let w = |z: f64| logistic_weight(z);
        let w2 = |z: f64| d2_fd(w, z, h12);
        let w3 = |z: f64| d1_fd(w2, z, h34);
        for i in 0..eta.len() {
            let z = eta[i];
            let w_fd = w(z);
            let w1_fd = d1_fd(w, z, h12);
            let w2_fd = d2_fd(w, z, h12);
            let w3_fd = d1_fd(w2, z, h34);
            let w4_fd = d1_fd(w3, z, h34);

            assert!((op.w[i] - w_fd).abs() < 1e-12);
            assert_eq!(op.w1[i].signum(), w1_fd.signum());
            assert_eq!(op.w2[i].signum(), w2_fd.signum());
            assert_eq!(op.w3[i].signum(), w3_fd.signum());
            assert_eq!(op.w4[i].signum(), w4_fd.signum());
            assert!((op.w1[i] - w1_fd).abs() < 2e-7);
            assert!((op.w2[i] - w2_fd).abs() < 2e-5);
            assert!((op.w3[i] - w3_fd).abs() < 4e-4);
            assert!((op.w4[i] - w4_fd).abs() < 2e-3);
        }
    }

    #[test]
    fn firth_mixed_second_direction_apply_is_symmetric_in_direction_order() {
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let beta = array![0.1, -0.25, 0.2];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");

        let u = array![0.3, -0.2, 0.4];
        let v = array![-0.5, 0.1, 0.25];
        let du = op.direction(&u);
        let dv = op.direction(&v);

        let eye = Array2::<f64>::eye(x.ncols());
        let uv = op.hphi_second_direction_apply(&du, &dv, &eye);
        let vu = op.hphi_second_direction_apply(&dv, &du, &eye);

        for i in 0..uv.nrows() {
            for j in 0..uv.ncols() {
                let a = uv[[i, j]];
                let b = vu[[i, j]];
                assert_eq!(
                    a.signum(),
                    b.signum(),
                    "mixed direction sign mismatch at ({i},{j}): uv={a} vu={b}"
                );
                assert!(
                    (a - b).abs() < 2e-7,
                    "mixed direction mismatch at ({i},{j}): uv={a} vu={b}"
                );
            }
        }
    }

    #[test]
    fn firth_direction_matrix_form_matches_apply_identity_form() {
        let x = array![
            [1.0, -1.1, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let beta = array![0.08, -0.22, 0.27];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");
        let u = Array1::from_vec(vec![0.25, -0.4, 0.35]);
        let dir = op.direction(&u);

        let p = x.ncols();
        let eye = Array2::<f64>::eye(p);
        let mut via_apply = op.hphi_direction_apply(&dir, &eye);
        for i in 0..p {
            for j in 0..i {
                let sym = 0.5 * (via_apply[[i, j]] + via_apply[[j, i]]);
                via_apply[[i, j]] = sym;
                via_apply[[j, i]] = sym;
            }
        }
        let direct = op.hphi_direction(&dir);
        let diff = &direct - &via_apply;
        let err = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err < 1e-10, "direction/apply mismatch: {err:e}");
    }

    #[test]
    fn firth_phi_tau_partial_matches_finite_difference_logdet() {
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let x_tau = array![
            [0.0, 0.15, -0.05],
            [0.0, -0.10, 0.02],
            [0.0, 0.08, 0.04],
            [0.0, -0.06, -0.03],
            [0.0, 0.05, 0.01],
            [0.0, -0.12, 0.06],
        ];
        let beta = array![0.1, -0.25, 0.2];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");
        let analytic = op.exact_tau_kernel(&x_tau, &beta, false).phi_tau_partial;

        let h = 1e-6;
        let x_plus = &x + &(h * &x_tau);
        let x_minus = &x - &(h * &x_tau);
        let fd = (firth_phi_value(&x_plus, &beta) - firth_phi_value(&x_minus, &beta)) / (2.0 * h);

        assert!(
            (analytic - fd).abs() < 1e-6,
            "Phi_tau mismatch: analytic={analytic:.12e}, fd={fd:.12e}"
        );
    }

    #[test]
    fn firth_g_phi_tau_matches_finite_difference_grad_phi() {
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let x_tau = array![
            [0.0, 0.15, -0.05],
            [0.0, -0.10, 0.02],
            [0.0, 0.08, 0.04],
            [0.0, -0.06, -0.03],
            [0.0, 0.05, 0.01],
            [0.0, -0.12, 0.06],
        ];
        let beta = array![0.1, -0.25, 0.2];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");
        let analytic = op.exact_tau_kernel(&x_tau, &beta, false).g_phi_tau;

        let h = 1e-6;
        let x_plus = &x + &(h * &x_tau);
        let x_minus = &x - &(h * &x_tau);
        let fd = (firth_grad_phi(&x_plus, &beta) - firth_grad_phi(&x_minus, &beta)) / (2.0 * h);

        let err = (&analytic - &fd).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            err < 1e-6,
            "g_phi_tau mismatch: analytic={analytic:?}, fd={fd:?}, err={err:e}"
        );
    }
}
