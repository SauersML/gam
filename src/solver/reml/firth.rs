use super::*;

impl<'a> RemlState<'a> {
    pub(super) fn xt_diag_x_dense_into(
        x: &Array2<f64>,
        diag: &Array1<f64>,
        weighted: &mut Array2<f64>,
    ) -> Array2<f64> {
        let n = x.nrows();
        weighted.assign(x);
        for i in 0..n {
            let w = diag[i];
            for j in 0..x.ncols() {
                weighted[[i, j]] *= w;
            }
        }
        fast_atb(x, weighted)
    }

    pub(super) fn row_scale(x: &Array2<f64>, scale: &Array1<f64>) -> Array2<f64> {
        let mut out = x.clone();
        for i in 0..out.nrows() {
            let w = scale[i];
            for j in 0..out.ncols() {
                out[[i, j]] *= w;
            }
        }
        out
    }

    pub(super) fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
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

    pub(super) fn trace_hdag_operator_apply<S, A>(
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

    pub(super) fn bilinear_form(
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

    pub(super) fn reduced_weighted_gram(z: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
        // Returns Zᵀ diag(weights) Z (exact).
        //
        // Used for:
        //   S = Zᵀ diag(v) Z               in Hadamard-Gram apply,
        //   G_u = X_rᵀ diag(s_u) X_r       with s_u = w' ⊙ (Xu),
        // both of which avoid constructing dense n×n intermediates.
        let mut weighted = z.clone();
        for i in 0..weighted.nrows() {
            let scale = weights[i];
            for j in 0..weighted.ncols() {
                weighted[[i, j]] *= scale;
            }
        }
        fast_atb(z, &weighted)
    }

    pub(super) fn reduced_cross_weighted_gram(
        z_left: &Array2<f64>,
        z_right: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> Array2<f64> {
        // Returns Z_leftᵀ diag(weights) Z_right (exact).
        //
        // This is used for explicit design-moving terms where left/right
        // reduced designs differ (X_r vs X_{tau,r}) in Hadamard-Gram products.
        let mut weighted = z_right.clone();
        for i in 0..weighted.nrows() {
            let scale = weights[i];
            for j in 0..weighted.ncols() {
                weighted[[i, j]] *= scale;
            }
        }
        fast_atb(z_left, &weighted)
    }

    pub(super) fn reduced_diag_gram(z: &Array2<f64>, a: &Array2<f64>) -> Array1<f64> {
        // Returns diag(Z A Zᵀ), exact without forming dense n×n matrix.
        //
        // Identity:
        //   [diag(Z A Zᵀ)]_i = z_iᵀ A z_i,
        // where z_i is row i of Z. We compute this as rowwise dot(Z, Z A).
        let za = fast_ab(z, a);
        let mut out = Array1::<f64>::zeros(z.nrows());
        for i in 0..z.nrows() {
            let mut acc = 0.0_f64;
            for j in 0..z.ncols() {
                acc += z[[i, j]] * za[[i, j]];
            }
            out[i] = acc;
        }
        out
    }

    pub(super) fn apply_hadamard_gram(
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

    pub(super) fn apply_hadamard_gram_to_matrix(
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
            w4[i] = wi * ti * ti * ti * ti - 22.0 * wi2 * ti * ti + 16.0 * wi3;
        }
        // Build a fixed identifiable basis Q for Col(Xᵀ) from XᵀX eigenvectors.
        // This gives a pseudoinverse-compatible representation:
        //   I_+^† = Q I_r^{-1} Qᵀ,   I_r = (XQ)ᵀ W (XQ).
        let gram = fast_atb(x_dense, x_dense);
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

        let mut q_basis = Array2::<f64>::zeros((p, r));
        for (col_idx, &eig_idx) in keep.iter().enumerate() {
            q_basis.column_mut(col_idx).assign(&evecs.column(eig_idx));
        }
        if r == 0 {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        let x_reduced = fast_ab(x_dense, &q_basis);

        let mut weighted_x_reduced = x_reduced.clone();
        for i in 0..n {
            let wi = w[i];
            for j in 0..r {
                weighted_x_reduced[[i, j]] *= wi;
            }
        }
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
            Self::reduced_diag_gram(&z_reduced, &k_reduced)
        } else {
            Array1::<f64>::zeros(n)
        };
        let mut b_base = x_dense.clone();
        for i in 0..n {
            let scale = w1[i];
            for j in 0..p {
                b_base[[i, j]] *= scale;
            }
        }
        let p_b_base =
            Self::apply_hadamard_gram_to_matrix(&z_reduced, &k_reduced, &k_reduced, &b_base);
        Ok(FirthDenseOperator {
            x_dense: x_dense.clone(),
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
            p_b_base,
        })
    }

    pub(super) fn firth_direction(op: &FirthDenseOperator, u: &Array1<f64>) -> FirthDirection {
        let deta = op.x_dense.dot(u);
        Self::firth_direction_from_deta(op, deta)
    }

    pub(super) fn firth_direction_from_deta(
        op: &FirthDenseOperator,
        deta: Array1<f64>,
    ) -> FirthDirection {
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
        let s_u = &op.w1 * &deta;
        // G_u = X_rᵀ diag(s_u) X_r,  A_u = K_r G_u K_r.
        // These are reduced-space forms of
        //   I_u and T_u = K I_u K
        // from the full-space derivation.
        let g_u_reduced = Self::reduced_weighted_gram(&op.x_reduced, &s_u);
        let k_g_u = op.k_reduced.dot(&g_u_reduced);
        let a_u_reduced = k_g_u.dot(&op.k_reduced);
        // Dh[u] = -diag(N_u),  N_u = X T_u Xᵀ, represented here as
        //   N_u = Z A_u Zᵀ in weighted reduced coordinates.
        let dh = -Self::reduced_diag_gram(&op.z_reduced, &a_u_reduced);
        FirthDirection {
            deta,
            g_u_reduced,
            a_u_reduced,
            dh,
        }
    }

    pub(super) fn firth_hphi_direction_apply(
        op: &FirthDenseOperator,
        dir: &FirthDirection,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = op.x_dense.ncols();
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
        let eta_v = op.x_dense.dot(rhs);
        let q_v = &eta_v * &op.w1.view().insert_axis(Axis(1));
        let m_q_v =
            Self::apply_hadamard_gram_to_matrix(&op.z_reduced, &op.k_reduced, &op.k_reduced, &q_v);
        let bu_vec = &op.w2 * &dir.deta;
        let bu_v = &eta_v * &bu_vec.view().insert_axis(Axis(1));
        let m_bu_v =
            Self::apply_hadamard_gram_to_matrix(&op.z_reduced, &op.k_reduced, &op.k_reduced, &bu_v);
        let mut p_u_q_v = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &dir.a_u_reduced,
            &q_v,
        );
        p_u_q_v.mapv_inplace(|v| -2.0 * v);
        let c_u = &(&op.w3 * &dir.deta) * &op.h_diag + &(&op.w2 * &dir.dh);
        let diag_term = op
            .x_dense
            .t()
            .dot(&(&eta_v * &c_u.view().insert_axis(Axis(1))));
        let term1 = op
            .x_dense
            .t()
            .dot(&(&m_q_v * &bu_vec.view().insert_axis(Axis(1))));
        let term2 = op
            .x_dense
            .t()
            .dot(&(&m_bu_v * &op.w1.view().insert_axis(Axis(1))));
        let term3 = op
            .x_dense
            .t()
            .dot(&(&p_u_q_v * &op.w1.view().insert_axis(Axis(1))));
        0.5 * (diag_term - (term1 + term2 + term3))
    }

    pub(super) fn firth_hphi_direction(
        op: &FirthDenseOperator,
        dir: &FirthDirection,
    ) -> Array2<f64> {
        let p = op.x_dense.ncols();
        let eye = Array2::<f64>::eye(p);
        let mut out = Self::firth_hphi_direction_apply(op, dir, &eye);
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

    pub(super) fn firth_hphi_second_direction_apply(
        op: &FirthDenseOperator,
        u: &FirthDirection,
        v: &FirthDirection,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = op.x_dense.ncols();
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
        let s_uv = &op.w2 * &deta_uv;
        let g_uv_reduced = Self::reduced_weighted_gram(&op.x_reduced, &s_uv);
        let k_g_uv = op.k_reduced.dot(&g_uv_reduced);
        let k_g_v = op.k_reduced.dot(&v.g_u_reduced);
        let k_g_u = op.k_reduced.dot(&u.g_u_reduced);
        // Reduced form of:
        //   T_{u,v} = K I_{u,v} K - K I_v K I_u K - K I_u K I_v K.
        let a_uv_reduced = k_g_uv.dot(&op.k_reduced)
            - k_g_v.dot(&k_g_u).dot(&op.k_reduced)
            - k_g_u.dot(&k_g_v).dot(&op.k_reduced);
        let d2h = -Self::reduced_diag_gram(&op.z_reduced, &a_uv_reduced);
        // Implements mixed diagonal coefficient:
        //   c_uv = w'''' ⊙ (Xu) ⊙ (Xv) ⊙ h
        //          + w''' ⊙ ((Xu) ⊙ Dh[v] + (Xv) ⊙ Dh[u])
        //          + w'' ⊙ D²h[u,v].
        let c_uv = &(&(&op.w4 * &deta_uv) * &op.h_diag)
            + &(&op.w3 * &(&u.deta * &v.dh))
            + &(&op.w3 * &(&v.deta * &u.dh))
            + &(&op.w2 * &d2h);

        let eta_rhs = op.x_dense.dot(rhs);
        let diag_term = op
            .x_dense
            .t()
            .dot(&(&eta_rhs * &c_uv.view().insert_axis(Axis(1))));

        let b_uv_vec = &op.w3 * &deta_uv;
        let b_rhs = &eta_rhs * &op.w1.view().insert_axis(Axis(1));
        let b_u_rhs = &eta_rhs * &(&op.w2 * &u.deta).view().insert_axis(Axis(1));
        let b_v_rhs = &eta_rhs * &(&op.w2 * &v.deta).view().insert_axis(Axis(1));
        let b_uv_rhs = &eta_rhs * &b_uv_vec.view().insert_axis(Axis(1));

        let p_b_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &op.k_reduced,
            &b_rhs,
        );
        let p_bu_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &op.k_reduced,
            &b_u_rhs,
        );
        let p_bv_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &op.k_reduced,
            &b_v_rhs,
        );
        let p_buv_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &op.k_reduced,
            &b_uv_rhs,
        );

        let mut p_v_b_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &v.a_u_reduced,
            &b_rhs,
        );
        p_v_b_rhs.mapv_inplace(|val| -2.0 * val);
        let mut p_v_bu_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &v.a_u_reduced,
            &b_u_rhs,
        );
        p_v_bu_rhs.mapv_inplace(|val| -2.0 * val);
        let mut p_u_b_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &u.a_u_reduced,
            &b_rhs,
        );
        p_u_b_rhs.mapv_inplace(|val| -2.0 * val);
        let mut p_u_bv_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &u.a_u_reduced,
            &b_v_rhs,
        );
        p_u_bv_rhs.mapv_inplace(|val| -2.0 * val);

        let p_nu_nv_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &u.a_u_reduced,
            &v.a_u_reduced,
            &b_rhs,
        );
        let p_hw_nuv_rhs = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &a_uv_reduced,
            &b_rhs,
        );
        let p_uv_rhs = 2.0 * p_nu_nv_rhs - 2.0 * p_hw_nuv_rhs;
        let b_uv_t = op.x_dense.t().to_owned();
        let b_u_t = op.x_dense.t().to_owned();
        let b_v_t = op.x_dense.t().to_owned();
        let b_base_t = op.x_dense.t().to_owned();
        let left_uv = &b_uv_t * &b_uv_vec.view().insert_axis(Axis(0));
        let left_u = &b_u_t * &(&op.w2 * &u.deta).view().insert_axis(Axis(0));
        let left_v = &b_v_t * &(&op.w2 * &v.deta).view().insert_axis(Axis(0));
        let left_b = &b_base_t * &op.w1.view().insert_axis(Axis(0));

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

    pub(super) fn firth_dot_i_and_h(
        op: &FirthDenseOperator,
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
        let x_tau_reduced = fast_ab(x_tau, &op.q_basis);
        let n = op.x_reduced.nrows();

        let mut wx_reduced = op.x_reduced.clone();
        let mut wx_tau_reduced = x_tau_reduced.clone();
        for i in 0..n {
            let wi = op.w[i];
            for j in 0..wx_reduced.ncols() {
                wx_reduced[[i, j]] *= wi;
            }
            for j in 0..wx_tau_reduced.ncols() {
                wx_tau_reduced[[i, j]] *= wi;
            }
        }

        let dw = &op.w1 * deta;
        let mut dwx_reduced = op.x_reduced.clone();
        for i in 0..n {
            let dwi = dw[i];
            for j in 0..dwx_reduced.ncols() {
                dwx_reduced[[i, j]] *= dwi;
            }
        }

        let dot_i = fast_ab(&x_tau_reduced.t().to_owned(), &wx_reduced)
            + fast_ab(&op.x_reduced.t().to_owned(), &wx_tau_reduced)
            + fast_ab(&op.x_reduced.t().to_owned(), &dwx_reduced);

        let dot_k = -op.k_reduced.dot(&dot_i).dot(&op.k_reduced);
        let x_tauk = x_tau_reduced.dot(&op.k_reduced);
        let dot_h_explicit = 2.0 * Self::rowwise_dot(&x_tauk, &op.x_reduced);
        let dot_h_implicit = Self::rowwise_dot(&op.x_reduced.dot(&dot_k), &op.x_reduced);
        let dot_h = dot_h_explicit + dot_h_implicit;
        (dot_i, dot_h)
    }

    pub(super) fn firth_exact_tau_kernel(
        op: &FirthDenseOperator,
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
        //   Phi = -0.5 log|I_r|, I_r = X_r' W X_r, K_r = I_r^{-1}
        //   Phi_tau|beta = -0.5 tr(K_r I_{r,tau}).
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
        let (dot_i_partial, dot_h_partial) = Self::firth_dot_i_and_h(op, x_tau, &deta_partial);

        let first = 0.5 * x_tau.t().dot(&(&op.w1 * &op.h_diag));
        let second_vec = &(&(&op.w2 * &deta_partial) * &op.h_diag) + &(&op.w1 * &dot_h_partial);
        let second = 0.5 * op.x_dense.t().dot(&second_vec);
        let g_phi_tau = first + second;
        let phi_tau_partial = -0.5 * Self::trace_product(&op.k_reduced, &dot_i_partial);

        let tau_kernel = if include_hphi_tau_kernel {
            Some(Self::firth_hphi_tau_partial_prepare_from_partials(
                op,
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

    #[allow(dead_code)]
    pub(super) fn firth_hphi_tau_partial(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> Array2<f64> {
        let p = op.x_dense.ncols();
        let eye = Array2::<f64>::eye(p);
        let kernel = Self::firth_hphi_tau_partial_prepare(op, x_tau, beta);
        let out_fast = Self::firth_hphi_tau_partial_apply(op, x_tau, &kernel, &eye);
        let mut out = out_fast.clone();
        // Auto policy (no user-facing toggle):
        // 1) Fast reduced-operator kernel is default.
        // 2) Promote explicit trace expansion only when either:
        //    - problem is small (cheap exact oracle), or
        //    - reduced Fisher system is poorly conditioned and p is still moderate.
        // 3) If fast vs explicit mismatch is above tolerance, switch to explicit
        //    for this step and emit a warning.
        let cond_k = Self::firth_reduced_condition_number(&op.k_reduced);
        let run_explicit_oracle = p <= 24 || (p <= 64 && cond_k.is_some_and(|c| c > 1e10));
        if run_explicit_oracle {
            let explicit = Self::firth_hphi_tau_partial_trace_expansion(op, x_tau, beta);
            let diff = (&out_fast - &explicit).mapv(|v| v.abs()).sum();
            let scale = explicit.mapv(|v| v.abs()).sum().max(1.0);
            let rel_l1 = diff / scale;
            if rel_l1 > 5e-7 {
                log::warn!(
                    "[REML/Firth] switching to explicit H_phi,tau expansion (rel_l1={:.3e}, p={}, cond(K_r)={:.3e})",
                    rel_l1,
                    p,
                    cond_k.unwrap_or(f64::NAN)
                );
                out = explicit;
            } else {
                out = out_fast;
            }
            #[cfg(debug_assertions)]
            {
                debug_assert!(
                    rel_l1 < 5e-7,
                    "firth_hphi_tau_partial mismatch fast vs explicit: rel_l1={:.3e}",
                    rel_l1
                );
            }
        }
        for i in 0..p {
            for j in 0..i {
                let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
                out[[i, j]] = avg;
                out[[j, i]] = avg;
            }
        }
        out
    }

    #[allow(dead_code)]
    pub(super) fn firth_reduced_condition_number(k_reduced: &Array2<f64>) -> Option<f64> {
        if k_reduced.nrows() == 0 || k_reduced.ncols() == 0 {
            return None;
        }
        let (eigvals, _) = k_reduced.eigh(Side::Lower).ok()?;
        let max_ev = eigvals
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0_f64, f64::max);
        let min_ev = eigvals
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(f64::INFINITY, f64::min);
        if !max_ev.is_finite() || !min_ev.is_finite() || min_ev <= 0.0 {
            return None;
        }
        Some(max_ev / min_ev.max(1e-300))
    }

    pub(super) fn firth_hphi_tau_partial_prepare(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> FirthTauPartialKernel {
        let deta_partial = x_tau.dot(beta);
        let (dot_i_partial, dot_h_partial) = Self::firth_dot_i_and_h(op, x_tau, &deta_partial);
        Self::firth_hphi_tau_partial_prepare_from_partials(
            op,
            x_tau,
            &deta_partial,
            dot_h_partial,
            dot_i_partial,
        )
    }

    fn firth_hphi_tau_partial_prepare_from_partials(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        deta_partial: &Array1<f64>,
        dot_h_partial: Array1<f64>,
        dot_i_partial: Array2<f64>,
    ) -> FirthTauPartialKernel {
        let dot_w1 = &op.w2 * deta_partial;
        let dot_w2 = &op.w3 * deta_partial;
        let x_tau_reduced = fast_ab(x_tau, &op.q_basis);
        let dot_k = -op.k_reduced.dot(&dot_i_partial).dot(&op.k_reduced);
        FirthTauPartialKernel {
            dot_w1,
            dot_w2,
            dot_h_partial,
            x_tau_reduced,
            dot_k_reduced: dot_k,
        }
    }

    pub(super) fn firth_apply_pbar_to_matrix(
        op: &FirthDenseOperator,
        mat: &Array2<f64>,
    ) -> Array2<f64> {
        // Applies P̄ = (X_r K_r X_rᵀ)⊙(X_r K_r X_rᵀ) to each column of mat.
        Self::apply_hadamard_gram_to_matrix(&op.z_reduced, &op.k_reduced, &op.k_reduced, mat)
    }

    pub(super) fn firth_apply_mtau_to_matrix(
        op: &FirthDenseOperator,
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
        if mat.nrows() != op.x_dense.nrows() || mat.ncols() == 0 {
            return Array2::<f64>::zeros(mat.raw_dim());
        }
        let mut out = Array2::<f64>::zeros(mat.raw_dim());
        for col in 0..mat.ncols() {
            let v = mat.column(col).to_owned();
            let s_zz = Self::reduced_weighted_gram(&op.z_reduced, &v);
            let m_zz = op.k_reduced.dot(&s_zz).dot(&op.k_reduced);
            let t1 = Self::rowwise_bilinear(&op.z_reduced, &m_zz, &kernel.x_tau_reduced);

            let s_zt = Self::reduced_cross_weighted_gram(&op.z_reduced, &kernel.x_tau_reduced, &v);
            let m_zt = op.k_reduced.dot(&s_zt).dot(&op.k_reduced);
            let t2 = Self::reduced_diag_gram(&op.z_reduced, &m_zt);

            let t3 =
                Self::apply_hadamard_gram(&op.z_reduced, &op.k_reduced, &kernel.dot_k_reduced, &v);

            let y = 2.0 * (t1 + t2 + t3);
            out.column_mut(col).assign(&y);
        }
        out
    }

    pub(super) fn firth_hphi_tau_partial_apply(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        kernel: &FirthTauPartialKernel,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = op.x_dense.ncols();
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
        let eta_v = op.x_dense.dot(rhs);
        let eta_v_tau = x_tau.dot(rhs);
        let q_v = &eta_v * &op.w1.view().insert_axis(Axis(1));
        let q_v_tau = &eta_v * &kernel.dot_w1.view().insert_axis(Axis(1))
            + &eta_v_tau * &op.w1.view().insert_axis(Axis(1));
        let m_q_v = Self::firth_apply_pbar_to_matrix(op, &q_v);
        let m_q_v_tau = Self::firth_apply_mtau_to_matrix(op, kernel, &q_v)
            + Self::firth_apply_pbar_to_matrix(op, &q_v_tau);
        let r_v = &(&eta_v * &op.w2.view().insert_axis(Axis(1)))
            * &op.h_diag.view().insert_axis(Axis(1))
            - &(&m_q_v * &op.w1.view().insert_axis(Axis(1)));
        let r_v_tau = (&(&eta_v * &kernel.dot_w2.view().insert_axis(Axis(1)))
            + &(&eta_v_tau * &op.w2.view().insert_axis(Axis(1))))
            * &op.h_diag.view().insert_axis(Axis(1))
            + &(&eta_v * &op.w2.view().insert_axis(Axis(1)))
                * &kernel.dot_h_partial.view().insert_axis(Axis(1))
            - &(&m_q_v * &kernel.dot_w1.view().insert_axis(Axis(1))
                + &m_q_v_tau * &op.w1.view().insert_axis(Axis(1)));
        0.5 * (x_tau.t().dot(&r_v) + op.x_dense.t().dot(&r_v_tau))
    }

    pub(super) fn firth_hphi_trace_apply_combined(
        op: &FirthDenseOperator,
        dir: &FirthDirection,
        x_tau: &Array2<f64>,
        tau_kernel: Option<&FirthTauPartialKernel>,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = op.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }

        // Shared block intermediates:
        // eta_v, q_v, and P(q_v) are reused by both
        //   H_{phi,tau}|beta(·) and D(H_phi)[dir](·).
        let eta_v = op.x_dense.dot(rhs);
        let q_v = &eta_v * &op.w1.view().insert_axis(Axis(1));

        // D(H_phi)[dir] contribution:
        let m_q_v =
            Self::apply_hadamard_gram_to_matrix(&op.z_reduced, &op.k_reduced, &op.k_reduced, &q_v);
        let bu_vec = &op.w2 * &dir.deta;
        let bu_v = &eta_v * &bu_vec.view().insert_axis(Axis(1));
        let m_bu_v =
            Self::apply_hadamard_gram_to_matrix(&op.z_reduced, &op.k_reduced, &op.k_reduced, &bu_v);
        let mut p_u_q_v = Self::apply_hadamard_gram_to_matrix(
            &op.z_reduced,
            &op.k_reduced,
            &dir.a_u_reduced,
            &q_v,
        );
        p_u_q_v.mapv_inplace(|v| -2.0 * v);
        let c_u = &(&op.w3 * &dir.deta) * &op.h_diag + &(&op.w2 * &dir.dh);
        let diag_term = op
            .x_dense
            .t()
            .dot(&(&eta_v * &c_u.view().insert_axis(Axis(1))));
        let term1 = op
            .x_dense
            .t()
            .dot(&(&m_q_v * &bu_vec.view().insert_axis(Axis(1))));
        let term2 = op
            .x_dense
            .t()
            .dot(&(&m_bu_v * &op.w1.view().insert_axis(Axis(1))));
        let term3 = op
            .x_dense
            .t()
            .dot(&(&p_u_q_v * &op.w1.view().insert_axis(Axis(1))));
        let mut out = 0.5 * (diag_term - (term1 + term2 + term3));

        if let Some(kernel) = tau_kernel {
            // H_{phi,tau}|beta contribution, reusing eta_v and q_v.
            let eta_v_tau = x_tau.dot(rhs);
            let q_v_tau = &eta_v * &kernel.dot_w1.view().insert_axis(Axis(1))
                + &eta_v_tau * &op.w1.view().insert_axis(Axis(1));
            let m_q_v_kernel = Self::firth_apply_pbar_to_matrix(op, &q_v);
            let m_q_v_tau = Self::firth_apply_mtau_to_matrix(op, kernel, &q_v)
                + Self::firth_apply_pbar_to_matrix(op, &q_v_tau);
            let r_v = &(&eta_v * &op.w2.view().insert_axis(Axis(1)))
                * &op.h_diag.view().insert_axis(Axis(1))
                - &(&m_q_v_kernel * &op.w1.view().insert_axis(Axis(1)));
            let r_v_tau = (&(&eta_v * &kernel.dot_w2.view().insert_axis(Axis(1)))
                + &(&eta_v_tau * &op.w2.view().insert_axis(Axis(1))))
                * &op.h_diag.view().insert_axis(Axis(1))
                + &(&eta_v * &op.w2.view().insert_axis(Axis(1)))
                    * &kernel.dot_h_partial.view().insert_axis(Axis(1))
                - &(&m_q_v_kernel * &kernel.dot_w1.view().insert_axis(Axis(1))
                    + &m_q_v_tau * &op.w1.view().insert_axis(Axis(1)));
            out += &(0.5 * (x_tau.t().dot(&r_v) + op.x_dense.t().dot(&r_v_tau)));
        }
        out
    }

    #[allow(dead_code)]
    pub(super) fn firth_hphi_tau_partial_trace_expansion(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> Array2<f64> {
        let x = &op.x_dense;
        let n = x.nrows();
        let p = x.ncols();
        let deta = x_tau.dot(beta);
        let dotw = &op.w1 * &deta;
        let dotw1 = &op.w2 * &deta;
        let dotw2 = &op.w3 * &deta;

        let k_full = op
            .q_basis
            .dot(&op.k_reduced)
            .dot(&op.q_basis.t().to_owned());
        let mut weighted = Array2::<f64>::zeros(x.raw_dim());
        let mut weighted_tau = Array2::<f64>::zeros(x_tau.raw_dim());
        for i in 0..n {
            let wi = op.w[i];
            for j in 0..p {
                weighted[[i, j]] = wi * x[[i, j]];
                weighted_tau[[i, j]] = wi * x_tau[[i, j]];
            }
        }
        let mut dotw_x = x.clone();
        for i in 0..n {
            let dwi = dotw[i];
            for j in 0..p {
                dotw_x[[i, j]] *= dwi;
            }
        }
        let dot_f = x_tau.t().dot(&weighted) + x.t().dot(&weighted_tau) + x.t().dot(&dotw_x);
        let dot_k = -k_full.dot(&dot_f).dot(&k_full);

        let mut f_j = Vec::<Array2<f64>>::with_capacity(p);
        let mut dot_f_j = Vec::<Array2<f64>>::with_capacity(p);
        for j in 0..p {
            let xj = x.column(j).to_owned();
            let xtauj = x_tau.column(j).to_owned();
            let d_j = &op.w1 * &xj;
            let mut f_j_mat = Self::xt_diag_x_dense_into(x, &d_j, &mut weighted);

            let d_j_tau = &(&dotw1 * &xj) + &(&op.w1 * &xtauj);
            let term_outer = x_tau.t().dot(&{
                let mut m = x.clone();
                for i in 0..n {
                    let di = d_j[i];
                    for c in 0..p {
                        m[[i, c]] *= di;
                    }
                }
                m
            });
            let term_outer_t = x.t().dot(&{
                let mut m = x_tau.clone();
                for i in 0..n {
                    let di = d_j[i];
                    for c in 0..p {
                        m[[i, c]] *= di;
                    }
                }
                m
            });
            let term_diag = Self::xt_diag_x_dense_into(x, &d_j_tau, &mut weighted);
            let mut dot_f_j_mat = term_outer + term_outer_t + term_diag;
            for a in 0..p {
                for b in 0..a {
                    let avgf = 0.5 * (f_j_mat[[a, b]] + f_j_mat[[b, a]]);
                    f_j_mat[[a, b]] = avgf;
                    f_j_mat[[b, a]] = avgf;
                    let avgd = 0.5 * (dot_f_j_mat[[a, b]] + dot_f_j_mat[[b, a]]);
                    dot_f_j_mat[[a, b]] = avgd;
                    dot_f_j_mat[[b, a]] = avgd;
                }
            }
            f_j.push(f_j_mat);
            dot_f_j.push(dot_f_j_mat);
        }

        let mut out = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let xj = x.column(j).to_owned();
            let xtauj = x_tau.column(j).to_owned();
            for k in j..p {
                let xk = x.column(k).to_owned();
                let xtauk = x_tau.column(k).to_owned();
                let d_jk = &(&op.w2 * &xj) * &xk;
                let f_jk = Self::xt_diag_x_dense_into(x, &d_jk, &mut weighted);

                let d_jk_tau = &(&(&dotw2 * &xj) * &xk)
                    + &(&(&op.w2 * &xtauj) * &xk)
                    + &(&(&op.w2 * &xj) * &xtauk);
                let term_outer = x_tau.t().dot(&{
                    let mut m = x.clone();
                    for i in 0..n {
                        let di = d_jk[i];
                        for c in 0..p {
                            m[[i, c]] *= di;
                        }
                    }
                    m
                });
                let term_outer_t = x.t().dot(&{
                    let mut m = x_tau.clone();
                    for i in 0..n {
                        let di = d_jk[i];
                        for c in 0..p {
                            m[[i, c]] *= di;
                        }
                    }
                    m
                });
                let dot_f_jk = term_outer
                    + term_outer_t
                    + Self::xt_diag_x_dense_into(x, &d_jk_tau, &mut weighted);

                let f_k = &f_j[k];
                let f_jm = &f_j[j];
                let dot_f_k = &dot_f_j[k];
                let dot_f_jm = &dot_f_j[j];

                let term1 = Self::trace_product(&dot_k, &f_jk);
                let term2 = Self::trace_product(&k_full, &dot_f_jk);
                let term3 = Self::trace_product(&dot_k.dot(f_k).dot(&k_full), f_jm);
                let term4 = Self::trace_product(&k_full.dot(dot_f_k).dot(&k_full), f_jm);
                let term5 = Self::trace_product(&k_full.dot(f_k).dot(&dot_k), f_jm);
                let term6 = Self::trace_product(&k_full.dot(f_k).dot(&k_full), dot_f_jm);
                let val = 0.5 * (term1 + term2 - term3 - term4 - term5 - term6);
                out[[j, k]] = val;
                out[[k, j]] = val;
            }
        }
        out
    }
}
