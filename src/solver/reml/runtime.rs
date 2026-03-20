use super::*;
use crate::construction::{
    create_balanced_penalty_root_from_canonical, precompute_reparam_invariant_from_canonical,
};
use crate::linalg::sparse_exact::build_sparse_penalty_blocks_from_canonical;
use crate::linalg::utils::{boundary_hit_indices, symmetric_spectrum_condition_number};
use crate::pirls::PirlsWorkspace;
use crate::types::{InverseLink, LinkFunction, SasLinkState};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

const TK_BLOCK_SIZE: usize = 128;
const TK_MAX_OBSERVATIONS: usize = 20_000;
const TK_MAX_COEFFICIENTS: usize = 2_000;
const TK_MAX_DENSE_WORK: usize = 5_000_000;

struct TkCorrectionTerms {
    value: f64,
    gradient: Option<Array1<f64>>,
    hessian: Option<Array2<f64>>,
}

impl<'a> RemlState<'a> {
    fn tk_disabled_terms(rho: &Array1<f64>, mode: super::unified::EvalMode) -> TkCorrectionTerms {
        TkCorrectionTerms {
            value: 0.0,
            gradient: if mode != super::unified::EvalMode::ValueOnly {
                Some(Array1::zeros(rho.len()))
            } else {
                None
            },
            hessian: if mode == super::unified::EvalMode::ValueGradientHessian {
                Some(Array2::zeros((rho.len(), rho.len())))
            } else {
                None
            },
        }
    }

    pub(super) fn sparse_exact_beta_original(&self, pirls_result: &PirlsResult) -> Array1<f64> {
        match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                pirls_result.beta_transformed.as_ref().clone()
            }
            pirls::PirlsCoordinateFrame::TransformedQs => pirls_result
                .reparam_result
                .qs
                .dot(pirls_result.beta_transformed.as_ref()),
        }
    }

    fn bundle_matrix_in_original_basis(
        &self,
        pirls_result: &PirlsResult,
        matrix: &Array2<f64>,
    ) -> Array2<f64> {
        match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => matrix.clone(),
            pirls::PirlsCoordinateFrame::TransformedQs => {
                let qs = &pirls_result.reparam_result.qs;
                let tmp = qs.dot(matrix);
                tmp.dot(&qs.t())
            }
        }
    }

    pub(crate) fn last_ridge_used(&self) -> Option<f64> {
        self.cache_manager
            .current_eval_bundle
            .read()
            .unwrap()
            .as_ref()
            .map(|bundle| bundle.ridge_passport.delta)
    }

    fn dense_penalty_logdet_derivs(
        &self,
        rho: &Array1<f64>,
        e_for_logdet: &Array2<f64>,
        penalty_roots: &[Array2<f64>],
        ridge_passport: RidgePassport,
        mode: super::unified::EvalMode,
    ) -> Result<(usize, super::unified::PenaltyLogdetDerivs), EstimationError> {
        let (penalty_rank, log_det_s) =
            self.fixed_subspace_penalty_rank_and_logdet(e_for_logdet, ridge_passport)?;
        let lambdas = rho.mapv(f64::exp);

        // Use block-local path from canonical penalties (basis-invariant logdet).
        // The penalty logdet log|Σ λ_k S_k|₊ is invariant under orthogonal
        // transformation, so the original-basis canonical penalties give the
        // same result as transformed-basis roots.
        let (det1, det2_full) = if !self.canonical_penalties.is_empty()
            && self.canonical_penalties.len() == rho.len()
        {
            self.structural_penalty_logdet_derivatives_block_local(
                &lambdas,
                ridge_passport.penalty_logdet_ridge(),
            )?
        } else if !penalty_roots.is_empty() {
            self.structural_penalty_logdet_derivatives(
                penalty_roots,
                &lambdas,
                ridge_passport.penalty_logdet_ridge(),
            )?
        } else {
            (
                Array1::zeros(rho.len()),
                Array2::zeros((rho.len(), rho.len())),
            )
        };

        let det2 = if mode == super::unified::EvalMode::ValueGradientHessian {
            Some(det2_full)
        } else {
            None
        };
        Ok((
            penalty_rank,
            super::unified::PenaltyLogdetDerivs {
                value: log_det_s,
                first: det1,
                second: det2,
            },
        ))
    }

    fn tk_sanitized_array(values: &Array1<f64>) -> Array1<f64> {
        values.mapv(|v| if v.is_finite() { v } else { 0.0 })
    }

    /// Analytic TK value + gradient given Z = H⁻¹ X^T.
    ///
    /// Gradient formula (holding c, d fixed, differentiating through Σ = H⁻¹):
    ///
    ///   ∂TK/∂ρ_k = tr(Ḣ_k · P_total)
    ///
    /// where P_total is computed once from value intermediates:
    ///
    ///   P_total = ¼ P_Q - ¼ P_{T1} - ¼ P_{T2a} - ⅛ P_{T2b}
    ///   P_Q     = Z diag(d⊙h) Z^T
    ///   P_{T1}  = Z · [(cc^T) ⊙ G²] · Z^T
    ///   P_{T2a} = Z diag(c⊙(Xy)) Z^T
    ///   P_{T2b} = y y^T
    ///
    /// and Ḣ_k = A_k + X^T diag(c⊙Xv_k) X is the total Hessian drift.
    /// Analytic TK value + gradient given Z = H⁻¹ X^T.
    ///
    /// This is the shared core for both dense and sparse paths.
    /// The caller is responsible for computing Z from whichever factorization
    /// is available (dense Cholesky/eigen, or sparse Cholesky).
    ///
    /// For the gradient, also needs a solver for H⁻¹ v (used to compute
    /// mode responses v_k = H⁻¹(A_k β̂)). The `h_inv_solve` closure
    /// provides this.
    /// Compute the TK gradient given Z, precomputed v_ks and x_vks.
    ///
    /// Returns the k-vector of gradient entries. This helper is called by the
    /// main analytic core for the base gradient, and again with perturbed
    /// inputs to build the Hessian via matrix-level differentiation.
    fn tk_gradient_from_intermediates(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        tk_penalties: &[crate::construction::CanonicalPenalty],
        lambdas: &[f64],
        x_vks: &[Array1<f64>],
        h_inv_solve: &dyn Fn(&Array1<f64>) -> Array1<f64>,
        penalty_coords: Option<&[&super::unified::PenaltyCoordinate]>,
    ) -> Array1<f64> {
        let n = x_dense.nrows();
        let p = x_dense.ncols();
        let k = tk_penalties.len();
        let xt = x_dense.t();

        // Hat leverages
        let mut h_diag = Array1::<f64>::zeros(n);
        for i in 0..n {
            let val = x_dense.row(i).dot(&z.column(i));
            h_diag[i] = if val.is_finite() { val } else { 0.0 };
        }

        // y = H⁻¹ X^T(c⊙h)
        let m_vec = c_array * &h_diag;
        let x_m = xt.dot(&m_vec);
        let y = h_inv_solve(&x_m);
        let x_y = x_dense.dot(&y);

        // P_total (Q + T2a combined, then T2b, then T1)
        let mut diag_combined = Array1::<f64>::zeros(n);
        for i in 0..n {
            diag_combined[i] = d_array[i] * h_diag[i] - c_array[i] * x_y[i];
        }
        let mut p_total = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let wi = diag_combined[i];
            if wi == 0.0 {
                continue;
            }
            let z_i = z.column(i);
            for a in 0..p {
                let wa = wi * z_i[a];
                for b in a..p {
                    let val = wa * z_i[b];
                    p_total[[a, b]] += val;
                    if a != b {
                        p_total[[b, a]] += val;
                    }
                }
            }
        }
        p_total.mapv_inplace(|v| 0.25 * v);
        for a in 0..p {
            for b in 0..p {
                p_total[[a, b]] -= 0.125 * y[a] * y[b];
            }
        }

        // T1: subtract ¼ Z M Z^T
        for j0 in (0..n).step_by(TK_BLOCK_SIZE) {
            let j1 = (j0 + TK_BLOCK_SIZE).min(n);
            let z_block_j = z.slice(s![.., j0..j1]);
            let c_j = c_array.slice(s![j0..j1]);
            for i0 in (0..=j0).step_by(TK_BLOCK_SIZE) {
                let i1 = (i0 + TK_BLOCK_SIZE).min(n);
                let x_block_i = x_dense.slice(s![i0..i1, ..]);
                let c_i = c_array.slice(s![i0..i1]);
                let gram = x_block_i.dot(&z_block_j.to_owned());
                for bi in 0..(i1 - i0) {
                    let ci = c_i[bi];
                    if ci == 0.0 {
                        continue;
                    }
                    for bj in 0..(j1 - j0) {
                        let cj = c_j[bj];
                        if cj == 0.0 {
                            continue;
                        }
                        let gij = gram[[bi, bj]];
                        let weight = ci * cj * gij * gij;
                        let ii = i0 + bi;
                        let jj = j0 + bj;
                        let sym_factor = if ii == jj { 1.0 } else { 2.0 };
                        let scale = -0.25 * weight * sym_factor;
                        if ii == jj {
                            let z_i = z.column(ii);
                            for a in 0..p {
                                for b in a..p {
                                    let val = scale * z_i[a] * z_i[b];
                                    p_total[[a, b]] += val;
                                    if a != b {
                                        p_total[[b, a]] += val;
                                    }
                                }
                            }
                        } else {
                            let z_ii = z.column(ii);
                            let z_jj = z.column(jj);
                            let half_scale = 0.5 * scale;
                            for a in 0..p {
                                for b in 0..p {
                                    p_total[[a, b]] +=
                                        half_scale * (z_ii[a] * z_jj[b] + z_jj[a] * z_ii[b]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Compute leverage from P and gradient entries
        let xp = x_dense.dot(&p_total);
        let mut lev_p = Array1::<f64>::zeros(n);
        for i in 0..n {
            lev_p[i] = xp.row(i).dot(&x_dense.row(i).to_owned());
        }

        let mut gradient = Array1::<f64>::zeros(k);
        for idx in 0..k {
            let trace_ak_p = if let Some(coords) = penalty_coords {
                coords[idx].trace_with_dense(&p_total, lambdas[idx])
            } else {
                // Block-local: tr(λ_k S_k P) = λ_k tr(R_k P[block,block] R_k^T)
                let cp = &tk_penalties[idx];
                let r = &cp.col_range;
                let p_block = p_total.slice(s![r.start..r.end, r.start..r.end]);
                let rk_p = cp.root.dot(&p_block);
                lambdas[idx]
                    * (0..cp.rank())
                        .map(|row| rk_p.row(row).dot(&cp.root.row(row).to_owned()))
                        .sum::<f64>()
            };
            let correction_trace: f64 = (0..n).map(|i| c_array[i] * x_vks[idx][i] * lev_p[i]).sum();
            gradient[idx] = trace_ak_p + correction_trace;
        }

        for g in gradient.iter_mut() {
            if !g.is_finite() {
                *g = 0.0;
            }
        }
        gradient
    }

    fn tierney_kadane_analytic_core(
        &self,
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        tk_penalties: &[crate::construction::CanonicalPenalty],
        lambdas: &[f64],
        beta: &Array1<f64>,
        compute_gradient: bool,
        compute_hessian: bool,
        h_inv_solve: &dyn Fn(&Array1<f64>) -> Array1<f64>,
    ) -> Result<TkCorrectionTerms, EstimationError> {
        let n = x_dense.nrows();
        let p = x_dense.ncols();
        let k = tk_penalties.len();
        let xt = x_dense.t();

        // ── Hat leverages h_i = x_i^T H⁻¹ x_i ──
        let mut h_diag = Array1::<f64>::zeros(n);
        for i in 0..n {
            let val = x_dense.row(i).dot(&z.column(i));
            h_diag[i] = if val.is_finite() { val } else { 0.0 };
        }

        // ── Q term: -⅛ Σ_i d_i h_ii² ──
        let q_term = -0.125
            * d_array
                .iter()
                .zip(h_diag.iter())
                .map(|(&d_i, &h_i)| d_i * h_i * h_i)
                .sum::<f64>();

        // ── T2 term: ⅛ ‖w‖² where w = X^T(c⊙h), y = H⁻¹ w ──
        let m_vec = c_array * &h_diag; // c ⊙ h (n-vector)
        let x_m = xt.dot(&m_vec); // X^T(c⊙h) (p-vector)
        let y = h_inv_solve(&x_m);
        let t2_term = 0.125 * x_m.dot(&y);

        // ── T1 term: 1/12 Σ_{i,j} c_i c_j G_ij³ (blocked) ──
        let mut t1_sum = 0.0_f64;
        for j0 in (0..n).step_by(TK_BLOCK_SIZE) {
            let j1 = (j0 + TK_BLOCK_SIZE).min(n);
            let z_block = z.slice(s![.., j0..j1]).to_owned();
            let c_j = c_array.slice(s![j0..j1]);
            for i0 in (0..=j0).step_by(TK_BLOCK_SIZE) {
                let i1 = (i0 + TK_BLOCK_SIZE).min(n);
                let x_block = x_dense.slice(s![i0..i1, ..]);
                let c_i = c_array.slice(s![i0..i1]);
                let gram = x_block.dot(&z_block);
                let mut block_sum = 0.0_f64;
                for bi in 0..(i1 - i0) {
                    let ci = c_i[bi];
                    if ci == 0.0 {
                        continue;
                    }
                    for bj in 0..(j1 - j0) {
                        let cj = c_j[bj];
                        if cj == 0.0 {
                            continue;
                        }
                        let kij = gram[[bi, bj]];
                        block_sum += ci * cj * kij * kij * kij;
                    }
                }
                t1_sum += if i0 == j0 { block_sum } else { 2.0 * block_sum };
            }
        }

        let value = q_term + t1_sum / 12.0 + t2_term;
        let value = if value.is_finite() { value } else { 0.0 };

        if !compute_gradient {
            return Ok(TkCorrectionTerms {
                value,
                gradient: None,
                hessian: None,
            });
        }

        // ══════════════════════════════════════════════════════════════════
        // Precompute v_k = H⁻¹(A_k β̂) and x_vk = X v_k for all k.
        // ══════════════════════════════════════════════════════════════════
        let mut v_ks: Vec<Array1<f64>> = Vec::with_capacity(k);
        let mut x_vks: Vec<Array1<f64>> = Vec::with_capacity(k);
        for idx in 0..k {
            // Block-local: A_k β = λ_k R^T (R β[block]), embedded into p-vector.
            let cp = &tk_penalties[idx];
            let r = &cp.col_range;
            let beta_block = beta.slice(s![r.start..r.end]);
            let r_beta = cp.root.dot(&beta_block);
            let mut s_k_beta = Array1::<f64>::zeros(p);
            for a in 0..cp.block_dim() {
                s_k_beta[r.start + a] = (0..cp.rank())
                    .map(|row| cp.root[[row, a]] * r_beta[row])
                    .sum::<f64>();
            }
            let a_k_beta = &s_k_beta * lambdas[idx];
            let v_k = h_inv_solve(&a_k_beta);
            let x_vk = x_dense.dot(&v_k);
            v_ks.push(v_k);
            x_vks.push(x_vk);
        }

        // ══════════════════════════════════════════════════════════════════
        // Gradient via helper
        // ══════════════════════════════════════════════════════════════════
        let gradient = Self::tk_gradient_from_intermediates(
            x_dense,
            z,
            c_array,
            d_array,
            tk_penalties,
            lambdas,
            &x_vks,
            h_inv_solve,
            None,
        );

        // ══════════════════════════════════════════════════════════════════
        // Hessian via analytic matrix-level perturbation
        //
        // For each l, the first-order perturbation of H⁻¹ w.r.t. ρ_l is:
        //   δH⁻¹ = -H⁻¹ Ḣ_l H⁻¹
        // so:
        //   δZ = -H⁻¹ Ḣ_l Z         (perturbation of Z = H⁻¹ X^T)
        //   δv_k = -H⁻¹ Ḣ_l v_k + δ_kl v_k   (perturbation of v_k)
        //
        // We compute the gradient at Z ± ε δZ_l with v_k ± ε δv_k_l,
        // then Hessian[:,l] = (g⁺ - g⁻) / (2ε).
        // ══════════════════════════════════════════════════════════════════
        let hessian = if compute_hessian && k > 0 {
            let eps = 1e-5_f64;
            let mut hess = Array2::<f64>::zeros((k, k));

            for l in 0..k {
                // Compute Ḣ_l Z = A_l Z + X^T diag(c ⊙ x_vl) X Z, column by column.
                // Then δZ_l = H⁻¹(Ḣ_l Z).
                let cp_l = &tk_penalties[l];
                let rl = &cp_l.col_range;
                let x_vl = &x_vks[l];

                // A_l Z: block-local A_l z_j = λ_l R^T (R z_j[block])
                // C_l Z: for each column j, X^T(c ⊙ x_vl ⊙ (X z_j))
                let mut h_dot_l_z = Array2::<f64>::zeros((p, n));
                for j in 0..n {
                    let z_j = z.column(j).to_owned();
                    // A_l z_j — block-local
                    let z_j_block = z_j.slice(s![rl.start..rl.end]);
                    let rl_zj = cp_l.root.dot(&z_j_block);
                    let mut col = Array1::<f64>::zeros(p);
                    for a in 0..cp_l.block_dim() {
                        col[rl.start + a] = lambdas[l]
                            * (0..cp_l.rank())
                                .map(|row| cp_l.root[[row, a]] * rl_zj[row])
                                .sum::<f64>();
                    }
                    // C_l z_j = X^T(c ⊙ x_vl ⊙ (X z_j))
                    let x_zj = x_dense.dot(&z_j);
                    let mut weights = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        weights[i] = c_array[i] * x_vl[i] * x_zj[i];
                    }
                    col += &x_dense.t().dot(&weights);
                    h_dot_l_z.column_mut(j).assign(&col);
                }

                // δZ_l = H⁻¹(Ḣ_l Z)
                let mut delta_z_l = Array2::<f64>::zeros((p, n));
                for j in 0..n {
                    let col = h_dot_l_z.column(j).to_owned();
                    let solved = h_inv_solve(&col);
                    delta_z_l.column_mut(j).assign(&solved);
                }

                // δv_k = -H⁻¹(Ḣ_l v_k) + δ_kl v_k for each k
                let mut delta_v_ks: Vec<Array1<f64>> = Vec::with_capacity(k);
                for kk in 0..k {
                    let v_kk = &v_ks[kk];
                    // Ḣ_l v_k = A_l v_k + X^T(c ⊙ x_vl ⊙ X v_k)
                    let v_kk_block = v_kk.slice(s![rl.start..rl.end]);
                    let rl_vk = cp_l.root.dot(&v_kk_block);
                    let mut h_dot_l_vk = Array1::<f64>::zeros(p);
                    for a in 0..cp_l.block_dim() {
                        h_dot_l_vk[rl.start + a] = lambdas[l]
                            * (0..cp_l.rank())
                                .map(|row| cp_l.root[[row, a]] * rl_vk[row])
                                .sum::<f64>();
                    }
                    let x_vkk = &x_vks[kk];
                    let mut w = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        w[i] = c_array[i] * x_vl[i] * x_vkk[i];
                    }
                    h_dot_l_vk += &x_dense.t().dot(&w);

                    let mut dv = h_inv_solve(&h_dot_l_vk);
                    dv.mapv_inplace(|v| -v); // -H⁻¹(Ḣ_l v_k)
                    if kk == l {
                        dv += v_kk; // + δ_kl v_k
                    }
                    delta_v_ks.push(dv);
                }

                // Perturbed inputs: Z± = Z ∓ ε δZ_l, v_k± = v_k ± ε δv_k
                let mut z_plus = z.to_owned();
                let mut z_minus = z.to_owned();
                z_plus.scaled_add(-eps, &delta_z_l);
                z_minus.scaled_add(eps, &delta_z_l);

                let mut v_ks_plus: Vec<Array1<f64>> = Vec::with_capacity(k);
                let mut v_ks_minus: Vec<Array1<f64>> = Vec::with_capacity(k);
                let mut x_vks_plus: Vec<Array1<f64>> = Vec::with_capacity(k);
                let mut x_vks_minus: Vec<Array1<f64>> = Vec::with_capacity(k);
                for kk in 0..k {
                    let mut vp = v_ks[kk].clone();
                    vp.scaled_add(eps, &delta_v_ks[kk]);
                    let mut vm = v_ks[kk].clone();
                    vm.scaled_add(-eps, &delta_v_ks[kk]);
                    x_vks_plus.push(x_dense.dot(&vp));
                    x_vks_minus.push(x_dense.dot(&vm));
                    v_ks_plus.push(vp);
                    v_ks_minus.push(vm);
                }

                // Perturbed lambdas: λ_l(ρ+ε) = e^{ρ_l+ε} ≈ λ_l(1+ε)
                let mut lambdas_plus: Vec<f64> = lambdas.to_vec();
                let mut lambdas_minus: Vec<f64> = lambdas.to_vec();
                lambdas_plus[l] = (lambdas[l].ln() + eps).exp();
                lambdas_minus[l] = (lambdas[l].ln() - eps).exp();

                let grad_plus = Self::tk_gradient_from_intermediates(
                    x_dense,
                    &z_plus,
                    c_array,
                    d_array,
                    tk_penalties,
                    &lambdas_plus,
                    &x_vks_plus,
                    h_inv_solve,
                    None,
                );
                let grad_minus = Self::tk_gradient_from_intermediates(
                    x_dense,
                    &z_minus,
                    c_array,
                    d_array,
                    tk_penalties,
                    &lambdas_minus,
                    &x_vks_minus,
                    h_inv_solve,
                    None,
                );

                let inv_2eps = 0.5 / eps;
                for kk in 0..k {
                    hess[[kk, l]] = (grad_plus[kk] - grad_minus[kk]) * inv_2eps;
                }
            }

            // Symmetrize (should be symmetric already, but enforce numerically)
            for kk in 0..k {
                for ll in (kk + 1)..k {
                    let avg = 0.5 * (hess[[kk, ll]] + hess[[ll, kk]]);
                    hess[[kk, ll]] = avg;
                    hess[[ll, kk]] = avg;
                }
            }

            // Sanitize
            for val in hess.iter_mut() {
                if !val.is_finite() {
                    *val = 0.0;
                }
            }

            Some(hess)
        } else {
            None
        };

        Ok(TkCorrectionTerms {
            value,
            gradient: Some(gradient),
            hessian,
        })
    }

    fn tierney_kadane_terms(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<TkCorrectionTerms, EstimationError> {
        if self.config.link_function() == LinkFunction::Identity {
            return Ok(TkCorrectionTerms {
                value: 0.0,
                gradient: None,
                hessian: None,
            });
        }

        let pirls_result = bundle.pirls_result.as_ref();
        let (c_array, d_array) = self.hessian_cd_arrays(pirls_result)?;
        let c_array = Self::tk_sanitized_array(&c_array);
        let d_array = Self::tk_sanitized_array(&d_array);
        if c_array.is_empty() || d_array.is_empty() {
            return Ok(TkCorrectionTerms {
                value: 0.0,
                gradient: if mode != super::unified::EvalMode::ValueOnly {
                    Some(Array1::zeros(rho.len()))
                } else {
                    None
                },
                hessian: None,
            });
        }

        let compute_gradient = mode != super::unified::EvalMode::ValueOnly;
        let compute_hessian = mode == super::unified::EvalMode::ValueGradientHessian;
        let n_x = self.x().nrows();
        let p_x = self.x().ncols();
        let dense_work = n_x.saturating_mul(p_x);
        if n_x > TK_MAX_OBSERVATIONS || p_x > TK_MAX_COEFFICIENTS || dense_work > TK_MAX_DENSE_WORK
        {
            log::warn!(
                "disabling exact Tierney-Kadane correction for large model (n={}, p={}, n*p={}): exact TK is small-model-only",
                n_x,
                p_x,
                dense_work
            );
            return Ok(Self::tk_disabled_terms(rho, mode));
        }

        // Build Z = H⁻¹ X^T and solve closure, then call shared analytic core.
        if let Some(sparse) = bundle.sparse_exact.as_ref() {
            // Sparse path: Z and solver from sparse Cholesky factor.
            let x_dense = self
                .x()
                .try_to_dense_arc("exact TK correction requires dense design access")
                .map_err(EstimationError::InvalidInput)?;
            let xt = x_dense.t().to_owned();
            let z_mat =
                crate::linalg::sparse_exact::solve_sparse_spdmulti(sparse.factor.as_ref(), &xt)?;
            let factor_ref = sparse.factor.clone();
            let h_inv_solve = |rhs: &Array1<f64>| -> Array1<f64> {
                crate::linalg::sparse_exact::solve_sparse_spd(&factor_ref, rhs)
                    .unwrap_or_else(|_| Array1::zeros(rhs.len()))
            };

            let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
            let beta = self.sparse_exact_beta_original(pirls_result);

            return self.tierney_kadane_analytic_core(
                x_dense.as_ref(),
                &z_mat,
                &c_array,
                &d_array,
                &self.canonical_penalties,
                &lambdas,
                &beta,
                compute_gradient,
                compute_hessian,
                &h_inv_solve,
            );
        }

        // Dense path: Z and solver from dense Cholesky/eigendecomposition.
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let h_eff_eval = if let Some(z) = free_basis_opt.as_ref() {
            Self::projectwith_basis(bundle.h_eff.as_ref(), z)
        } else {
            bundle.h_eff.as_ref().clone()
        };
        let x_eff_dense = if let Some(z) = free_basis_opt.as_ref() {
            pirls_result.x_transformed.to_dense().dot(z)
        } else {
            pirls_result.x_transformed.to_dense()
        };

        let xt = x_eff_dense.t().to_owned();
        let p = x_eff_dense.ncols();
        let n = x_eff_dense.nrows();

        // Factor H once and reuse for both Z computation and the solve closure.
        enum HFactor {
            Cholesky(crate::linalg::faer_ndarray::FaerCholeskyFactor),
            Eigh {
                evals: Array1<f64>,
                evecs: Array2<f64>,
            },
            None(usize),
        }
        let h_factor = if let Ok(chol) = h_eff_eval.cholesky(Side::Lower) {
            HFactor::Cholesky(chol)
        } else if let Ok((evals, evecs)) = h_eff_eval.eigh(Side::Lower) {
            HFactor::Eigh { evals, evecs }
        } else {
            HFactor::None(p)
        };

        // Build Z = H⁻¹ Xᵀ using the pre-computed factorization.
        let z_mat = match &h_factor {
            HFactor::Cholesky(chol) => {
                let mut solved = xt.clone();
                chol.solve_mat_in_place(&mut solved);
                solved
            }
            HFactor::Eigh { evals, evecs } => {
                let floor = 1e-12;
                let mut solved = Array2::<f64>::zeros((p, n));
                for m in 0..evals.len() {
                    let ev = evals[m];
                    if ev <= floor {
                        continue;
                    }
                    let u = evecs.column(m).to_owned();
                    let coeffs = xt.t().dot(&u).mapv(|v| v / ev);
                    for row in 0..p {
                        let u_row = u[row];
                        for col in 0..n {
                            solved[[row, col]] += u_row * coeffs[col];
                        }
                    }
                }
                solved
            }
            HFactor::None(_) => Array2::<f64>::zeros((p, n)),
        };

        // Reuse the same factorization for the solve closure.
        let h_inv_solve = move |rhs: &Array1<f64>| -> Array1<f64> {
            match &h_factor {
                HFactor::Cholesky(chol) => chol.solvevec(rhs),
                HFactor::Eigh { evals, evecs } => {
                    let floor = 1e-12;
                    let mut sol = Array1::<f64>::zeros(rhs.len());
                    for m in 0..evals.len() {
                        if evals[m] > floor {
                            let u = evecs.column(m);
                            let coeff = u.dot(rhs) / evals[m];
                            sol.scaled_add(coeff, &u.to_owned());
                        }
                    }
                    sol
                }
                HFactor::None(p) => Array1::zeros(*p),
            }
        };

        // Use pre-built canonical penalties in the transformed coordinate frame.
        // When constraint projection is active, project the roots into the free subspace.
        let p_eff = x_eff_dense.ncols();
        let tk_penalties: Vec<crate::construction::CanonicalPenalty> = if let Some(z) =
            free_basis_opt.as_ref()
        {
            pirls_result
                .reparam_result
                .canonical_transformed
                .iter()
                .map(|cp| {
                    crate::construction::CanonicalPenalty::from_dense_root(cp.root.dot(z), p_eff)
                })
                .collect()
        } else {
            pirls_result.reparam_result.canonical_transformed.clone()
        };
        let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
        let beta = if let Some(z) = free_basis_opt.as_ref() {
            z.t()
                .dot(&pirls_result.beta_transformed.as_ref().to_owned())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };

        self.tierney_kadane_analytic_core(
            &x_eff_dense,
            &z_mat,
            &c_array,
            &d_array,
            &tk_penalties,
            &lambdas,
            &beta,
            compute_gradient,
            compute_hessian,
            &h_inv_solve,
        )
    }

    fn apply_tk_to_result(
        &self,
        mut result: super::unified::RemlLamlResult,
        rho: &Array1<f64>,
        tk_terms: TkCorrectionTerms,
    ) -> super::unified::RemlLamlResult {
        result.cost += tk_terms.value;
        if let (Some(ref mut grad), Some(tk_grad)) = (result.gradient.as_mut(), tk_terms.gradient) {
            let k = rho.len().min(grad.len()).min(tk_grad.len());
            {
                let mut sl = grad.slice_mut(s![..k]);
                sl += &tk_grad.slice(s![..k]);
            }
        }
        if let (Some(ref mut hess), Some(tk_hess)) = (result.hessian.as_mut(), tk_terms.hessian) {
            let k = rho
                .len()
                .min(hess.nrows())
                .min(hess.ncols())
                .min(tk_hess.nrows());
            let mut sl = hess.slice_mut(s![..k, ..k]);
            sl += &tk_hess.slice(s![..k, ..k]);
        }
        result
    }

    fn apply_tk_cost_to_efs(
        &self,
        mut eval: crate::solver::outer_strategy::EfsEval,
        tk_value: f64,
    ) -> crate::solver::outer_strategy::EfsEval {
        eval.cost += tk_value;
        eval
    }

    pub(super) fn should_compute_hot_diagnostics(&self, eval_idx: u64) -> bool {
        // Keep expensive diagnostics out of the hot path unless they can
        // be surfaced. This has zero effect on optimization math.
        (log::log_enabled!(log::Level::Info) || log::log_enabled!(log::Level::Warn))
            && (eval_idx == 1 || eval_idx % 200 == 0)
    }

    fn invalidate_link_dependent_state(&self) {
        self.cache_manager.clear_eval_and_factor_caches();
        self.cache_manager.pirls_cache.write().unwrap().clear();
        self.warm_start_beta.write().unwrap().take();
    }

    pub(crate) fn set_link_states(
        &mut self,
        mixture_link_state: Option<crate::types::MixtureLinkState>,
        sas_link_state: Option<SasLinkState>,
    ) {
        let changed = self.runtime_mixture_link_state != mixture_link_state
            || self.runtime_sas_link_state != sas_link_state;
        if !changed {
            return;
        }
        self.runtime_mixture_link_state = mixture_link_state;
        self.runtime_sas_link_state = sas_link_state;
        self.invalidate_link_dependent_state();
    }

    /// Returns the eta-derivative carriers (c = dW/deta, d = d^2W/deta^2) for
    /// the exact Hessian surface that PIRLS accepted at the mode.
    ///
    /// For canonical links these coincide with the Fisher arrays; for
    /// non-canonical links they carry the clamped **observed-information**
    /// curvature used on the PIRLS left-hand side, including the residual-
    /// dependent corrections:
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    ///
    /// These flow into the outer REML gradient's C[v] correction term and
    /// the outer Hessian's Q[v_k, v_l] correction term. Using the observed
    /// versions is required for the exact Laplace approximation; Fisher
    /// versions would yield a PQL-type surrogate. See response.md Section 3.
    fn hessian_cd_arrays(
        &self,
        pirls_result: &PirlsResult,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        Ok((
            pirls_result.solve_c_array.clone(),
            pirls_result.solve_d_array.clone(),
        ))
    }

    /// Compute soft prior cost without needing workspace
    pub(super) fn compute_soft_priorcost(&self, rho: &Array1<f64>) -> f64 {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return 0.0;
        }

        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        let mut cost = 0.0;
        for &ri in rho.iter() {
            let scaled = sharp * ri * inv_bound;
            cost += scaled.cosh().ln();
        }

        cost * RHO_SOFT_PRIOR_WEIGHT
    }

    /// Compute soft prior gradient without workspace mutation.
    pub(super) fn compute_soft_priorgrad(&self, rho: &Array1<f64>) -> Array1<f64> {
        let len = rho.len();
        let mut grad = Array1::<f64>::zeros(len);
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return grad;
        }
        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        for (g, &ri) in grad.iter_mut().zip(rho.iter()) {
            let scaled = sharp * ri * inv_bound;
            *g = sharp * inv_bound * scaled.tanh() * RHO_SOFT_PRIOR_WEIGHT;
        }
        grad
    }

    /// Add the exact Hessian of the soft rho prior in place.
    ///
    /// Prior definition per coordinate:
    ///   C_i(rho_i) = w * log(cosh(a * rho_i)),
    ///   a = rho_soft_prior_sharpness / rho_bound,
    ///   w = rho_soft_prior_weight.
    ///
    /// Then:
    ///   dC_i/drho_i   = w * a * tanh(a * rho_i),
    ///   d²C_i/drho_i² = w * a² * sech²(a * rho_i)
    ///                = w * a² * (1 - tanh²(a * rho_i)).
    ///
    /// The prior is separable across coordinates, so off-diagonals are zero.
    pub(super) fn add_soft_priorhessian_in_place(&self, rho: &Array1<f64>, hess: &mut Array2<f64>) {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return;
        }
        let a = RHO_SOFT_PRIOR_SHARPNESS / RHO_BOUND;
        let prefactor = RHO_SOFT_PRIOR_WEIGHT * a * a;
        for i in 0..len {
            let t = (a * rho[i]).tanh();
            hess[[i, i]] += prefactor * (1.0 - t * t);
        }
    }

    /// Compute the soft prior Hessian as a standalone matrix.
    ///
    /// This is the diagonal matrix of second derivatives of the soft prior penalty.
    /// Returns `None` when the prior contributes zero curvature (empty ρ or zero weight).
    pub(super) fn compute_soft_priorhess(&self, rho: &Array1<f64>) -> Option<Array2<f64>> {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return None;
        }
        let mut hess = Array2::<f64>::zeros((len, len));
        self.add_soft_priorhessian_in_place(rho, &mut hess);
        if hess.iter().any(|&v| v != 0.0) {
            Some(hess)
        } else {
            None
        }
    }

    /// Returns the effective Hessian and the ridge value used (if any).
    /// Uses the same Hessian matrix in both cost and gradient calculations.
    ///
    /// PIRLS folds any stabilization ridge directly into the penalized objective:
    ///   l_p(beta; rho) = l(beta) - 0.5 * beta^T (S_lambda + ridge I) beta.
    /// Therefore the curvature used in LAML is
    ///   H_eff = X' W_H X + S_lambda + ridge I,
    /// and adding another ridge here places the Laplace expansion on a different surface.
    ///
    /// IMPORTANT: W_H here is the **observed-information** weight for non-canonical
    /// links (or Fisher weight for canonical links, where the two coincide). The
    /// `stabilizedhessian_transformed` from PIRLS was assembled with the Hessian-side
    /// weights (`lasthessian_weights`), which are observed when PIRLS accepted that
    /// curvature. This ensures log|H_eff| in the outer REML uses the correct Laplace
    /// Hessian. See response.md Section 3 for the mathematical justification.
    ///
    pub(super) fn effectivehessian(
        &self,
        pr: &PirlsResult,
    ) -> Result<(Array2<f64>, RidgePassport), EstimationError> {
        // Verify PD via SymmetricMatrix::factorize() (sparse-aware),
        // then densify for the spectral evaluator which needs eigh().
        let h = &pr.stabilizedhessian_transformed;
        if h.factorize().is_ok() {
            return Ok((h.to_dense(), pr.ridge_passport));
        }

        Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })
    }

    pub(crate) fn newwith_offset<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        offset: ArrayView1<'_, f64>,
        canonical_penalties: Vec<crate::construction::CanonicalPenalty>,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        Self::newwith_offset_shared(
            y,
            x,
            weights,
            offset,
            Arc::new(canonical_penalties),
            p,
            config,
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
        )
    }

    pub(crate) fn newwith_offset_shared<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        offset: ArrayView1<'_, f64>,
        canonical_penalties: Arc<Vec<crate::construction::CanonicalPenalty>>,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        let x = x.into();

        let expected_len = canonical_penalties.len();
        let nullspace_dims = match nullspace_dims {
            Some(dims) => {
                if dims.len() != expected_len {
                    return Err(EstimationError::InvalidInput(format!(
                        "nullspace_dims length {} does not match penalties {}",
                        dims.len(),
                        expected_len
                    )));
                }
                dims
            }
            None => vec![0; expected_len],
        };

        let balanced_penalty_root =
            create_balanced_penalty_root_from_canonical(&canonical_penalties, p)?;
        let reparam_invariant =
            precompute_reparam_invariant_from_canonical(&canonical_penalties, p)?;

        let sparse_penalty_blocks =
            build_sparse_penalty_blocks_from_canonical(canonical_penalties.as_ref(), p)?
                .map(Arc::new);

        Ok(Self {
            y,
            x,
            weights,
            offset: offset.to_owned(),
            canonical_penalties,
            balanced_penalty_root,
            reparam_invariant,
            sparse_penalty_blocks,
            p,
            config,
            runtime_mixture_link_state: config.link_kind.mixture_state().cloned(),
            runtime_sas_link_state: config.link_kind.sas_state().copied(),
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
            penalty_shrinkage_floor: None,
            cache_manager: EvalCacheManager::new(),
            arena: RemlArena::new(),
            warm_start_beta: RwLock::new(None),
            warm_start_enabled: AtomicBool::new(true),
            screening_max_inner_iterations: Arc::new(AtomicUsize::new(0)),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        })
    }

    /// Inject Kronecker penalty system metadata for tensor-product smooth terms.
    ///
    /// When set, the REML evaluator will use O(∏q_j) logdet instead of O(p³)
    /// eigendecomposition.  Also stores the full factored basis so that P-IRLS
    /// can use factored reparameterization (Qs = U_1 ⊗ ... ⊗ U_d).
    pub(crate) fn set_kronecker_penalty_system(
        &mut self,
        system: crate::smooth::KroneckerPenaltySystem,
    ) {
        self.kronecker_penalty_system = Some(system);
    }

    /// Inject the full Kronecker factored basis for P-IRLS factored reparameterization.
    pub(crate) fn set_kronecker_factored(
        &mut self,
        factored: crate::basis::KroneckerFactoredBasis,
    ) {
        self.kronecker_factored = Some(factored);
    }

    /// Sets the shrinkage floor for penalized block eigenvalues.
    /// The ridge magnitude will be `epsilon * max_balanced_eigenvalue` (rho-independent).
    /// This prevents barely-penalized directions from causing pathological
    /// non-Gaussianity in the posterior. Typical value: `Some(1e-6)`.
    pub(crate) fn set_penalty_shrinkage_floor(&mut self, floor: Option<f64>) {
        self.penalty_shrinkage_floor = floor;
    }

    /// Creates a sanitized cache key from rho values.
    /// Returns None if any component is NaN, in which case caching is skipped.
    /// Maps -0.0 to 0.0 to ensure consistency in caching.
    pub(super) fn rhokey_sanitized(&self, rho: &Array1<f64>) -> Option<Vec<u64>> {
        EvalCacheManager::sanitized_rhokey(rho)
    }

    pub(super) fn prepare_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let decision = self.select_reml_geometry(rho);
        match decision.geometry {
            RemlGeometry::SparseExactSpd => {
                match self.prepare_sparse_eval_bundlewithkey(rho, key.clone()) {
                    Ok(bundle) => {
                        log::info!(
                            "[reml-geometry] sparse_exact_spd reason={} p={} nnz_x={} nnz_h_est={} density_h_est={}",
                            decision.reason,
                            decision.p,
                            decision.nnz_x,
                            decision
                                .nnz_h_upper_est
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "na".to_string()),
                            decision
                                .density_h_upper_est
                                .map(|v| format!("{v:.4}"))
                                .unwrap_or_else(|| "na".to_string()),
                        );
                        Ok(bundle)
                    }
                    Err(err) => {
                        log::warn!(
                            "[reml-geometry] sparse_exact_spd failed ({}); falling back to dense spectral",
                            err
                        );
                        self.prepare_dense_eval_bundlewithkey(rho, key)
                    }
                }
            }
            RemlGeometry::DenseSpectral => self.prepare_dense_eval_bundlewithkey(rho, key),
        }
    }

    pub(crate) fn obtain_eval_bundle(
        &self,
        rho: &Array1<f64>,
    ) -> Result<EvalShared, EstimationError> {
        let key = self.rhokey_sanitized(rho);
        if let Some(existing) = self.cache_manager.cached_eval_bundle(&key) {
            return Ok(existing.clone());
        }
        let bundle = self.prepare_eval_bundlewithkey(rho, key)?;
        self.cache_manager.store_eval_bundle(bundle.clone());
        Ok(bundle)
    }

    pub(crate) fn objective_innerhessian(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        Ok(bundle.h_total.as_ref().clone())
    }

    pub(super) fn active_constraint_free_basis(&self, pr: &PirlsResult) -> Option<Array2<f64>> {
        let lin = pr.linear_constraints_transformed.as_ref()?;
        let active_tol = 1e-8;
        let beta_t = pr.beta_transformed.as_ref();
        let mut activerows: Vec<Array1<f64>> = Vec::new();
        for i in 0..lin.a.nrows() {
            let slack = lin.a.row(i).dot(beta_t) - lin.b[i];
            if slack <= active_tol {
                activerows.push(lin.a.row(i).to_owned());
            }
        }
        if activerows.is_empty() {
            return None;
        }

        let p_t = lin.a.ncols();
        let mut a_t = Array2::<f64>::zeros((p_t, activerows.len()));
        for (j, row) in activerows.iter().enumerate() {
            for k in 0..p_t {
                a_t[[k, j]] = row[k];
            }
        }

        let qrow = Self::orthonormalize_columns(&a_t, 1e-10); // basis for active row-space^T
        let rank = qrow.ncols();
        if rank == 0 {
            return None;
        }
        if rank >= p_t {
            return Some(Array2::<f64>::zeros((p_t, 0)));
        }

        // Build orthonormal basis for null(A_active) as complement of row-space.
        let mut z = Array2::<f64>::zeros((p_t, p_t - rank));
        let mut kept = 0usize;
        for j in 0..p_t {
            let mut v = Array1::<f64>::zeros(p_t);
            v[j] = 1.0;
            for t in 0..rank {
                let qt = qrow.column(t);
                let proj = qt.dot(&v);
                v -= &qt.mapv(|x| x * proj);
            }
            for t in 0..kept {
                let zt = z.column(t);
                let proj = zt.dot(&v);
                v -= &zt.mapv(|x| x * proj);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > 1e-10 {
                z.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                kept += 1;
                if kept == p_t - rank {
                    break;
                }
            }
        }
        Some(z.slice(ndarray::s![.., 0..kept]).to_owned())
    }

    /// Construct a `BarrierConfig` from linear inequality constraints `A β ≥ b`
    /// by extracting rows that represent simple coordinate bounds (β_j ≥ b_i).
    ///
    /// Delegates to `BarrierConfig::from_constraints` and logs a diagnostic
    /// barrier-curvature check at a test point near the bounds.
    fn barrier_config_from_constraints(
        constraints: &crate::pirls::LinearInequalityConstraints,
    ) -> Option<super::unified::BarrierConfig> {
        let config = super::unified::BarrierConfig::from_constraints(Some(constraints))?;
        // Diagnostic: check curvature significance at a test point near bounds.
        {
            let max_idx = config
                .constrained_indices
                .iter()
                .max()
                .copied()
                .unwrap_or(0);
            let mut beta_test = Array1::<f64>::zeros(max_idx + 1);
            for (&idx, &lb) in config
                .constrained_indices
                .iter()
                .zip(config.lower_bounds.iter())
            {
                beta_test[idx] = lb + 0.01; // slightly above constraint
            }
            let significant = config.barrier_curvature_is_significant(&beta_test, 1.0, 0.05);
            log::trace!(
                "[barrier] curvature significant={significant} (tau={:.2e}, n_constrained={})",
                config.tau,
                config.constrained_indices.len(),
            );
        }
        Some(config)
    }

    pub(super) fn enforce_constraint_kkt(&self, pr: &PirlsResult) -> Result<(), EstimationError> {
        let Some(kkt) = pr.constraint_kkt.as_ref() else {
            return Ok(());
        };
        let tol_primal = 1e-7;
        let tol_dual = 1e-7;
        let tol_comp = 1e-7;
        let tol_stat = 5e-6;
        if kkt.primal_feasibility > tol_primal
            || kkt.dual_feasibility > tol_dual
            || kkt.complementarity > tol_comp
            || kkt.stationarity > tol_stat
        {
            let mut worstrow_msg = String::new();
            if let Some(lin) = pr.linear_constraints_transformed.as_ref() {
                let mut worst = 0.0_f64;
                let mut worstrow = 0usize;
                for i in 0..lin.a.nrows() {
                    let slack = lin.a.row(i).dot(&pr.beta_transformed.0) - lin.b[i];
                    let viol = (-slack).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worstrow = i;
                    }
                }
                if worst > 0.0 {
                    worstrow_msg = format!("; worstrow={} worstviolation={:.3e}", worstrow, worst);
                }
            }
            return Err(EstimationError::ParameterConstraintViolation(format!(
                "KKT residuals exceed tolerance: primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}; active={}/{}{}",
                kkt.primal_feasibility,
                kkt.dual_feasibility,
                kkt.complementarity,
                kkt.stationarity,
                kkt.n_active,
                kkt.n_constraints,
                worstrow_msg
            )));
        }
        Ok(())
    }

    pub(super) fn projectwith_basis(matrix: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
        let zt_m = z.t().dot(matrix);
        zt_m.dot(z)
    }

    /// Compute the structural penalty rank and exact pseudo-logdet log|S|₊.
    ///
    /// Uses the exact pseudo-logdet on the positive eigenspace:
    ///   L(S) = Σ_{σ_i > ε} log σ_i
    /// where ε is a relative threshold identifying the structural nullspace
    /// directly from the eigenspectrum. No δ-regularization or nullity metadata.
    ///
    /// For S(ρ) = Σ exp(ρ_k) S_k with S_k ⪰ 0, the nullspace N(S) = ∩_k N(S_k)
    /// is structurally fixed, so this is C∞ in ρ.
    pub(super) fn fixed_subspace_penalty_rank_and_logdet(
        &self,
        e_transformed: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<(usize, f64), EstimationError> {
        let structural_rank = e_transformed.nrows().min(e_transformed.ncols());
        if structural_rank == 0 {
            return Ok((0, 0.0));
        }

        let mut s_lambda = e_transformed.t().dot(e_transformed);
        let ridge = ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..s_lambda.nrows() {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (evals, _) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Exact pseudo-logdet on the positive eigenspace: L = Σ_{σ_i > ε} log σ_i.
        // The structural nullspace is identified directly from the eigenspectrum.
        let threshold = super::unified::positive_eigenvalue_threshold(evals.as_slice().unwrap());
        let log_det = super::unified::exact_pseudo_logdet(evals.as_slice().unwrap(), threshold);

        Ok((structural_rank, log_det))
    }

    /// Compute tr(S⁺ S_direction) — the first derivative of log|S|₊
    /// in the direction S_direction.
    ///
    /// Uses the exact pseudoinverse S⁺ restricted to the positive eigenspace.
    /// Only eigenvectors with eigenvalues above the structural threshold
    /// participate.
    pub(super) fn fixed_subspace_penalty_trace(
        &self,
        e_transformed: &Array2<f64>,
        s_direction: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<f64, EstimationError> {
        let (structural_rank, _) =
            self.fixed_subspace_penalty_rank_and_logdet(e_transformed, ridge_passport)?;
        let p_dim = e_transformed.ncols();
        if structural_rank == 0 || p_dim == 0 {
            return Ok(0.0);
        }

        let mut s_lambda = e_transformed.t().dot(e_transformed);
        let ridge = ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..p_dim {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (evals, evecs) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Exact pseudoinverse trace: tr(S⁺ S_direction) on the positive eigenspace.
        let threshold = super::unified::positive_eigenvalue_threshold(evals.as_slice().unwrap());

        let mut trace = 0.0;
        for idx in 0..p_dim {
            let ev = evals[idx];
            if ev > threshold {
                let u = evecs.column(idx).to_owned();
                let spsi_u = s_direction.dot(&u);
                trace += u.dot(&spsi_u) / ev;
            }
        }
        Ok(trace)
    }

    pub(super) fn updatewarm_start_from(&self, pr: &PirlsResult) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        match pr.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                let beta_original = match pr.coordinate_frame {
                    pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                        pr.beta_transformed.as_ref().clone()
                    }
                    pirls::PirlsCoordinateFrame::TransformedQs => {
                        pr.reparam_result.qs.dot(pr.beta_transformed.as_ref())
                    }
                };
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta_original));
            }
            _ => {
                self.warm_start_beta.write().unwrap().take();
            }
        }
    }

    pub(crate) fn setwarm_start_original_beta(&self, beta_original: Option<ArrayView1<'_, f64>>) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        match beta_original {
            Some(beta) if beta.len() == self.p => {
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta.to_owned()));
            }
            _ => {
                self.warm_start_beta.write().unwrap().take();
            }
        }
    }

    // Accessor methods for private fields
    pub(crate) fn x(&self) -> &DesignMatrix {
        &self.x
    }

    pub(crate) fn balanced_penalty_root(&self) -> &Array2<f64> {
        &self.balanced_penalty_root
    }

    pub(crate) fn canonical_penalties(&self) -> &[crate::construction::CanonicalPenalty] {
        &self.canonical_penalties
    }

    /// Compute the exact pseudo-logdet for the sparse penalty path.
    ///
    /// For each sparse penalty block k with precomputed positive eigenvalues
    /// {σ_1, ..., σ_r}, the exact pseudo-logdet of λ_k S_k is:
    ///
    ///   L(λ_k S_k) = Σ_i log(λ_k σ_i) = r ρ_k + Σ_i log σ_i
    ///
    /// The first derivative is:
    ///   ∂/∂ρ_k L = r  (the rank, i.e. number of positive eigenvalues)
    pub(super) fn sparse_penalty_logdet_runtime(
        &self,
        rho: &Array1<f64>,
        blocks: &[SparsePenaltyBlock],
    ) -> (f64, Array1<f64>) {
        let mut logdet = 0.0_f64;
        let mut det1 = Array1::<f64>::zeros(rho.len());
        for block in blocks {
            let lambda_k = rho[block.term_index].exp();

            // L(λ_k S_k) = Σ_{positive} log(λ_k σ_i).
            for &eig in block.positive_eigenvalues.iter() {
                logdet += (lambda_k * eig).ln();
            }

            // ∂/∂ρ_k L = rank (number of positive eigenvalues).
            // Since ∂/∂ρ_k log(λ_k σ_i) = ∂/∂ρ_k (ρ_k + log σ_i) = 1.
            if block.term_index < det1.len() {
                det1[block.term_index] = block.positive_eigenvalues.len() as f64;
            }
        }
        (logdet, det1)
    }

    pub(super) fn prepare_dense_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        let (h_eff, ridge_passport) = self.effectivehessian(pirls_result.as_ref())?;

        let mut h_total = h_eff.clone();
        let mut firth_dense_operator: Option<Arc<FirthDenseOperator>> = None;
        if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            // Guard: Firth operator requires dense design materialization.
            let firth_n = pirls_result.x_transformed.nrows();
            let firth_p = pirls_result.x_transformed.ncols();
            const FIRTH_MAX_DENSE_WORK: usize = 50_000_000;
            if firth_n.saturating_mul(firth_p) > FIRTH_MAX_DENSE_WORK {
                log::warn!(
                    "disabling Firth bias reduction for large model (n={}, p={}): \
                     Firth operator requires dense design and is a small-model-only feature",
                    firth_n,
                    firth_p,
                );
            } else {
                let x_dense = pirls_result
                .x_transformed
                .try_to_dense_arc(
                    "dense REML eval bundle requires dense transformed design for Firth operator",
                )
                .map_err(EstimationError::InvalidInput)?;
                let firth_build_start = std::time::Instant::now();
                let firth_op = Arc::new(Self::build_firth_dense_operator(
                    x_dense.as_ref(),
                    &pirls_result.final_eta,
                    self.weights,
                )?);
                log::info!(
                    "[Firth-op] build n={} p={} r={} half_logdet={:.3e} elapsed={:.3}s",
                    firth_op.x_dense.nrows(),
                    firth_op.x_dense.ncols(),
                    firth_op.k_reduced.nrows(),
                    firth_op.half_log_det,
                    firth_build_start.elapsed().as_secs_f64(),
                );
                // Firth-adjusted inner Jacobian for implicit differentiation:
                //   H_total = Xᵀ W X + S - H_φ,
                //   H_φ     = ∇²_β Φ
                //          = 0.5 [ Xᵀ diag(w'' ⊙ h) X - Bᵀ P B ].
                // This keeps B_k/B_{kl} solves on the same objective surface as
                // the Firth-augmented stationarity system.
                //
                // Conceptually Φ is the pseudodeterminant term 0.5 log|XᵀWX|_+.
                // The hphi block below is therefore the curvature of that
                // identifiable-subspace Jeffreys penalty, represented in the
                // current transformed basis.
                let mut weighted_xtdx = Array2::<f64>::zeros((0, 0));
                let diag_term = Self::xt_diag_x_dense_into(
                    &firth_op.x_dense,
                    &(&firth_op.w2 * &firth_op.h_diag),
                    &mut weighted_xtdx,
                );
                let bpb = fast_ab(&firth_op.b_base.t().to_owned(), &firth_op.p_b_base);
                let mut hphi = 0.5 * (diag_term - bpb);
                // Numerical symmetry guard.
                for i in 0..hphi.nrows() {
                    for j in 0..i {
                        let avg = 0.5 * (hphi[[i, j]] + hphi[[j, i]]);
                        hphi[[i, j]] = avg;
                        hphi[[j, i]] = avg;
                    }
                }
                // Keep tiny numerical noise from making the solve surface less stable.
                if hphi.iter().all(|v| v.is_finite()) {
                    h_total -= &hphi;
                }
                firth_dense_operator = Some(firth_op);
            } // else (not too large for Firth)
        }

        // Add log-barrier Hessian diagonal for monotonicity-constrained coefficients.
        // This augments the penalized Hessian before the spectral decomposition so
        // that logdet, trace, and solve operations all reflect the barrier curvature.
        if let Some(ref lin) = pirls_result.linear_constraints_transformed {
            if let Some(barrier_cfg) = Self::barrier_config_from_constraints(lin) {
                let beta_t = pirls_result.beta_transformed.as_ref();
                if let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut h_total, beta_t) {
                    log::warn!("Barrier Hessian diagonal skipped: {e}");
                }
            }
        }

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::DenseSpectral,
            h_eff: Arc::new(h_eff),
            h_total: Arc::new(h_total),
            sparse_exact: None,
            firth_dense_operator,
            firth_dense_operator_original: None,
        })
    }

    pub(super) fn prepare_sparse_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        if !matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::OriginalSparseNative
        ) {
            return Err(EstimationError::InvalidInput(
                "sparse exact geometry requires sparse-native PIRLS coordinates".to_string(),
            ));
        }
        let ridge_passport = pirls_result.ridge_passport;
        let x_sparse = self.x().as_sparse().ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse exact geometry requires sparse original design".to_string(),
            )
        })?;
        let penalty_blocks = self
            .sparse_penalty_blocks
            .as_ref()
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "sparse exact geometry requires block-separable penalties".to_string(),
                )
            })?
            .clone();

        let lambdas = rho.mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, cp) in self.canonical_penalties.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                cp.accumulate_weighted(&mut s_lambda, lambdas[k]);
            }
        }
        // Add log-barrier Hessian diagonal for monotonicity-constrained
        // coefficients (sparse path uses original coordinates).
        if let Some(ref lin) = self.linear_constraints {
            if let Some(barrier_cfg) = Self::barrier_config_from_constraints(lin) {
                let beta_orig = self.sparse_exact_beta_original(pirls_result.as_ref());
                if let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut s_lambda, &beta_orig)
                {
                    log::warn!("Sparse barrier Hessian diagonal skipped: {e}");
                }
            }
        }

        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        let sparse_system = assemble_and_factor_sparse_penalized_system(
            &mut workspace,
            x_sparse,
            &pirls_result.finalweights,
            &s_lambda,
            ridge_passport.delta,
        )?;
        let (logdet_s_pos, det1_values) =
            self.sparse_penalty_logdet_runtime(rho, penalty_blocks.as_ref());
        let firth_dense_operator_original = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            // Guard: Firth operator requires dense design materialization.
            let firth_n = self.x().nrows();
            let firth_p = self.x().ncols();
            const FIRTH_MAX_DENSE_WORK: usize = 50_000_000;
            if firth_n.saturating_mul(firth_p) > FIRTH_MAX_DENSE_WORK {
                log::warn!(
                    "disabling Firth bias reduction for large model (n={}, p={}): \
                     Firth operator requires dense design and is a small-model-only feature",
                    firth_n,
                    firth_p,
                );
                None
            } else {
                let x_dense = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact REML runtime requires dense design for Firth operator",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Some(Arc::new(Self::build_firth_dense_operator(
                    x_dense.as_ref(),
                    &pirls_result.final_eta,
                    self.weights,
                )?))
            }
        } else {
            None
        };

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::SparseExactSpd,
            h_eff: Arc::new(Array2::zeros((0, 0))),
            h_total: Arc::new(Array2::zeros((0, 0))),
            sparse_exact: Some(Arc::new({
                let factor = Arc::new(sparse_system.factor);
                // Compute Takahashi selected inverse from simplicial factorization.
                // This precomputes H^{-1} entries on the filled pattern of L, enabling
                // O(nnz) trace computations instead of O(p) column solves.
                let takahashi =
                    crate::linalg::sparse_exact::factorize_simplicial(&sparse_system.h_sparse)
                        .ok()
                        .map(|sfactor| {
                            Arc::new(crate::linalg::sparse_exact::TakahashiInverse::compute(
                                &sfactor,
                            ))
                        });
                SparseExactEvalData {
                    factor,
                    takahashi,
                    logdet_h: sparse_system.logdet_h,
                    logdet_s_pos,
                    det1_values: Arc::new(det1_values),
                }
            })),
            firth_dense_operator: None,
            firth_dense_operator_original,
        })
    }

    /// Runs the inner P-IRLS loop, caching the result.
    pub(super) fn execute_pirls_if_needed(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Arc<PirlsResult>, EstimationError> {
        let screening_active = self.screening_max_inner_iterations.load(Ordering::Relaxed) > 0;
        let use_cache = !screening_active
            && self
                .cache_manager
                .pirls_cache_enabled
                .load(Ordering::Relaxed);
        // Use sanitized key to handle NaN and -0.0 vs 0.0 issues
        let key_opt = self.rhokey_sanitized(rho);
        if use_cache
            && let Some(key) = &key_opt
            && let Some(cached) = self.cache_manager.pirls_cache.write().unwrap().get(key)
        {
            // Do not overwrite the current warm start from cache hits.
            // Line search / multi-eval outer loops revisit older rho keys and
            // replacing a recent nearby beta with an older cached mode can
            // materially slow subsequent PIRLS convergence.
            if cached.cache_compacted {
                let mut pirls_config = self.config.as_pirls_config();
                pirls_config.link_kind =
                    if let Some(state) = self.runtime_mixture_link_state.clone() {
                        InverseLink::Mixture(state)
                    } else if let Some(state) = self.runtime_sas_link_state {
                        if matches!(self.config.link_function(), LinkFunction::BetaLogistic) {
                            InverseLink::BetaLogistic(state)
                        } else {
                            InverseLink::Sas(state)
                        }
                    } else {
                        InverseLink::Standard(self.config.link_function())
                    };
                return Ok(Arc::new(cached.rehydrate_after_reml_cache(
                    self.x(),
                    self.y,
                    self.weights,
                    self.offset.view(),
                    pirls_config.likelihood,
                    &pirls_config.link_kind,
                )?));
            }
            return Ok(cached);
        }

        // Run P-IRLS with original matrices to perform fresh reparameterization
        // The returned result will include the transformation matrix qs
        let pirls_result = {
            let warm_start_holder = self.warm_start_beta.read().unwrap();
            let warm_start_ref = if self.warm_start_enabled.load(Ordering::Relaxed) {
                warm_start_holder.as_ref()
            } else {
                None
            };
            let mut pirls_config = self.config.as_pirls_config();
            // Apply screening cap when active (seed screening uses cheap partial PIRLS).
            let screen_cap = self.screening_max_inner_iterations.load(Ordering::Relaxed);
            if screen_cap > 0 {
                pirls_config.max_iterations = pirls_config.max_iterations.min(screen_cap);
            }
            pirls_config.link_kind = if let Some(state) = self.runtime_mixture_link_state.clone() {
                InverseLink::Mixture(state)
            } else if let Some(state) = self.runtime_sas_link_state {
                if matches!(self.config.link_function(), LinkFunction::BetaLogistic) {
                    InverseLink::BetaLogistic(state)
                } else {
                    InverseLink::Sas(state)
                }
            } else {
                InverseLink::Standard(self.config.link_function())
            };
            let problem = pirls::PirlsProblem {
                x: &self.x,
                offset: self.offset.view(),
                y: self.y,
                priorweights: self.weights,
                covariate_se: None,
            };
            let penalty = pirls::PenaltyConfig {
                canonical_penalties: &self.canonical_penalties,
                balanced_penalty_root: Some(&self.balanced_penalty_root),
                reparam_invariant: Some(&self.reparam_invariant),
                p: self.p,
                coefficient_lower_bounds: self.coefficient_lower_bounds.as_ref(),
                linear_constraints_original: self.linear_constraints.as_ref(),
                penalty_shrinkage_floor: self.penalty_shrinkage_floor,
                kronecker_factored: self.kronecker_factored.as_ref(),
            };
            let pirls_start = std::time::Instant::now();
            let result = pirls::fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view()),
                problem,
                penalty,
                &pirls_config,
                warm_start_ref,
            );
            let pirls_elapsed = pirls_start.elapsed();
            let screen_cap = self.screening_max_inner_iterations.load(Ordering::Relaxed);
            if let Ok((ref res, ref wm)) = result {
                log::info!(
                    "[PIRLS-timing] iters={} status={:?} max_eta={:.1} jeffreys_logdet={} elapsed={:.3}s screening={}",
                    wm.iterations,
                    res.status,
                    res.max_abs_eta,
                    res.jeffreys_logdet()
                        .map(|v| format!("{v:.3e}"))
                        .unwrap_or_else(|| "none".to_string()),
                    pirls_elapsed.as_secs_f64(),
                    if screen_cap > 0 {
                        format!("cap={screen_cap}")
                    } else {
                        "off".to_string()
                    },
                );
            }
            result
        };

        if let Err(e) = &pirls_result {
            log::warn!("[GAM COST]   -> P-IRLS INNER LOOP FAILED. Error: {e:?}");
            // Keep the previous successful warm start even when a trial point
            // fails. Outer line search commonly probes unstable candidates and
            // then returns to nearby feasible rho values where the prior warm
            // beta remains the best initializer.
        }

        let (pirls_result, _) = pirls_result?; // Propagate error if it occurred
        let pirls_result = Arc::new(pirls_result);
        self.enforce_constraint_kkt(pirls_result.as_ref())?;

        // Check the status returned by the P-IRLS routine.
        match pirls_result.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                self.updatewarm_start_from(pirls_result.as_ref());
                // This is a successful fit. Cache only if key is valid (not NaN).
                if use_cache && let Some(key) = key_opt {
                    self.cache_manager
                        .pirls_cache
                        .write()
                        .unwrap()
                        .insert(key, Arc::new(pirls_result.compact_for_reml_cache()));
                }
                Ok(pirls_result)
            }
            pirls::PirlsStatus::Unstable => {
                // The fit was unstable. This is where we throw our specific, user-friendly error.
                // Pass the diagnostic info into the error
                Err(EstimationError::PerfectSeparationDetected {
                    iteration: pirls_result.iteration,
                    max_abs_eta: pirls_result.max_abs_eta,
                })
            }
            pirls::PirlsStatus::MaxIterationsReached => {
                // Envelope-theorem outer gradients require the inner solve to be near
                // stationarity; a loose max-iter acceptance threshold (e.g. 1.0) causes
                // persistent KKT/envelope diagnostic failures and inaccurate hyper-gradients.
                let acceptable_kkt = if self.runtime_sas_link_state.is_some() {
                    // SAS-link outer sweeps can stall at boundary-heavy rho configurations
                    // before satisfying strict default KKT tolerances; allow a wider
                    // near-stationary band to avoid false hard failures in those regimes.
                    (self.config.convergence_tolerance * 50.0).max(1e-2)
                } else {
                    (self.config.convergence_tolerance * 10.0).max(1e-4)
                };
                if pirls_result.lastgradient_norm > acceptable_kkt {
                    // The fit timed out and gradient is still too large for reliable outer
                    // derivatives, so fail fast and let the caller backtrack.
                    log::error!(
                        "P-IRLS failed convergence check: gradient norm {:.3e} > {:.3e} (iter {})",
                        pirls_result.lastgradient_norm,
                        acceptable_kkt,
                        pirls_result.iteration
                    );
                    Err(EstimationError::PirlsDidNotConverge {
                        max_iterations: pirls_result.iteration,
                        last_change: pirls_result.lastgradient_norm,
                    })
                } else {
                    // Near-stationary despite max iterations; treat as usable.
                    log::warn!(
                        "P-IRLS reached max iterations but gradient norm {:.3e} <= {:.3e}; accepting near-stationary fit.",
                        pirls_result.lastgradient_norm,
                        acceptable_kkt
                    );
                    self.updatewarm_start_from(pirls_result.as_ref());
                    Ok(pirls_result)
                }
            }
        }
    }
}

impl<'a> RemlState<'a> {
    /// Compute the objective function for BFGS optimization.
    ///
    /// Full objective reference.
    /// This function returns the scalar outer cost minimized over ρ.
    ///
    /// Non-Gaussian branch (negative LAML form used by optimizer):
    ///   V_LAML(ρ) =
    ///      -ℓ(β̂(ρ))
    ///      + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const
    ///
    /// where:
    ///   S(ρ) = Σ_k exp(ρ_k) S_k + δI
    ///   H(ρ) = -∇²ℓ(β̂(ρ)) + S(ρ)
    ///
    /// Gaussian identity-link REML branch:
    ///   V_REML(ρ, φ) =
    ///      D_p(ρ)/(2φ)
    ///      + (n_r/2) log φ
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const
    ///
    /// with profiled φ:
    ///   φ̂(ρ) = D_p(ρ)/n_r
    ///   V_REML,prof(ρ) =
    ///      (n_r/2) log D_p(ρ)
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const.
    ///
    /// Consistency rule enforced throughout:
    ///   The same stabilized matrices/factorizations are used for
    ///   objective and gradient/Hessian terms. Mixing different H/S variants
    ///   causes deterministic gradient mismatch and unstable outer optimization.
    ///
    /// Determinant conventions:
    ///   - log|H| may use positive-part/stabilized spectrum conventions when needed.
    ///   - log|S|_+ follows fixed-rank pseudo-determinant conventions in the
    ///     transformed penalty basis, optionally including ridge policy.
    /// These conventions are mirrored in gradient code via corresponding trace terms.
    pub fn compute_cost(&self, p: &Array1<f64>) -> Result<f64, EstimationError> {
        let cost_call_idx = {
            let mut calls = self.arena.cost_eval_count.write().unwrap();
            *calls += 1;
            *calls
        };
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                // Inner linear algebra says "too singular" — treat as barrier.
                log::warn!(
                    "P-IRLS flagged ill-conditioning for current rho; returning +inf cost to retreat."
                );
                // Diagnostics: which rho are at bounds
                let (at_lower, at_upper) = boundary_hit_indices(p.view(), RHO_BOUND, 1e-8);
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    log::debug!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower,
                        at_upper
                    );
                }
                return Ok(f64::INFINITY);
            }
            Err(EstimationError::PerfectSeparationDetected { .. })
            | Err(EstimationError::PirlsDidNotConverge { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                log::warn!(
                    "P-IRLS separation/non-convergence at current rho; returning +inf cost to retreat."
                );
                return Ok(f64::INFINITY);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                let (at_lower, at_upper) = boundary_hit_indices(p.view(), RHO_BOUND, 1e-8);
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    log::debug!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower,
                        at_upper
                    );
                }
                return Err(e);
            }
        };
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            let result =
                self.evaluate_unified_sparse(p, &bundle, super::unified::EvalMode::ValueOnly)?;
            return Ok(result.cost);
        }
        {
            // Validation and diagnostics (before delegating to unified evaluator).
            let pirls_result = bundle.pirls_result.as_ref();
            let ridge_used = bundle.ridge_passport.delta;

            if !p.is_empty() {
                let k_lambda = p.len();
                let k_r = pirls_result.reparam_result.canonical_transformed.len();
                let k_d = pirls_result.reparam_result.det1.len();
                if !(k_lambda == k_r && k_r == k_d) {
                    return Err(EstimationError::LayoutError(format!(
                        "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                        k_lambda, k_r, k_d
                    )));
                }
                if self.nullspace_dims.len() != k_lambda {
                    return Err(EstimationError::LayoutError(format!(
                        "Nullspace dimension mismatch: expected {} entries, got {}",
                        k_lambda,
                        self.nullspace_dims.len()
                    )));
                }
            }

            const MIN_ACCEPTABLE_HESSIAN_EIGENVALUE: f64 = 1e-12;
            let want_hot_diag = self.should_compute_hot_diagnostics(cost_call_idx);
            if ridge_used > 0.0 && want_hot_diag {
                // Eigenvalue diagnostics require dense; only pay the cost when
                // hot diagnostics are requested and ridge was applied.
                let pht_dense = pirls_result.penalized_hessian_transformed.to_dense();
                if let Ok((eigs, _)) = pht_dense.eigh(Side::Lower) {
                    if let Some(min_eig) = eigs.iter().cloned().reduce(f64::min) {
                        if should_emit_h_min_eig_diag(min_eig) {
                            log::debug!(
                                "[Diag] H min_eig={:.3e} (ridge={:.3e})",
                                min_eig,
                                ridge_used
                            );
                        }
                        if min_eig <= 0.0 {
                            log::warn!(
                                "Penalized Hessian not PD (min eig <= 0) before stabilization; proceeding with ridge {:.3e}.",
                                ridge_used
                            );
                        }
                        if !min_eig.is_finite() || min_eig <= MIN_ACCEPTABLE_HESSIAN_EIGENVALUE {
                            let condition_number = symmetric_spectrum_condition_number(&pht_dense);
                            log::warn!(
                                "Penalized Hessian extremely ill-conditioned (cond={:.3e}); continuing with stabilized Hessian.",
                                condition_number
                            );
                        }
                    }
                }
            }
        }
        // Delegate to the unified evaluator for the actual formula computation.
        // This ensures cost and gradient share the exact same formula.
        let result = self.evaluate_unified(p, &bundle, super::unified::EvalMode::ValueOnly)?;
        Ok(result.cost)
    }

    ///
    /// Exact non-Laplace evidence identities (reference comments; not runtime path)
    /// We optimize a Laplace-style outer objective for scalability, but the exact
    /// marginal likelihood for non-Gaussian models can be written analytically as:
    ///
    ///   L(ρ) = ∫ exp(l(β) - 0.5 βᵀ S(ρ) β) dβ,   S(ρ)=Σ_k exp(ρ_k) S_k.
    ///
    /// Universal exact gradient identity (when differentiation under the integral
    /// is justified and L(ρ) < ∞):
    ///
    ///   ∂_{ρ_k} log L(ρ)
    ///   = -0.5 * exp(ρ_k) * E_{π(β|y,ρ)}[ βᵀ S_k β ].
    ///
    /// Laplace bridge to implemented terms:
    /// - If π(β|y,ρ) is approximated locally by N(β̂, H^{-1}), then
    ///     E[βᵀ S_k β] ≈ β̂ᵀ S_k β̂ + tr(H^{-1} S_k),
    ///   giving the familiar quadratic + trace structure.
    /// - In this code those appear as:
    ///     0.5 * β̂ᵀ S_k^ρ β̂,
    ///     -0.5 * tr(S^+ S_k^ρ),
    ///     +0.5 * tr(H^{-1} H_k).
    ///
    /// Why this does NOT collapse to only tr(H^{-1}S_k):
    /// - The exact identity differentiates the true integral measure.
    /// - LAML differentiates a moving approximation:
    ///     V_LAML(ρ) = -ℓ(β̂(ρ)) + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
    ///                 + 0.5 log|H(ρ)| - 0.5 log|S(ρ)|_+.
    /// - Here both center β̂(ρ) and curvature H(ρ) move with ρ.
    /// - For non-Gaussian families, H_k includes the third-derivative tensor path
    ///   through β̂(ρ), i.e. H_k != S_k^ρ. These are the explicit dH/dρ_k terms
    ///   retained below to differentiate the Laplace objective exactly.
    ///
    /// For Bernoulli-logit, an exact Pólya-Gamma augmentation gives:
    ///
    ///   L(ρ) = 2^{-n} (2π)^{p/2}
    ///          E_{ω_i ~ PG(1,0)} [ |Q(ω,ρ)|^{-1/2} exp(0.5 bᵀ Q^{-1} b) ],
    ///   Q(ω,ρ)=S(ρ)+XᵀΩX, b=Xᵀ(y-1/2).
    ///
    /// and
    ///
    ///   ∂_{ρ_k} log L
    ///   = -0.5 * exp(ρ_k) *
    ///     E_{ω|y,ρ}[ tr(S_k Q^{-1}) + μᵀ S_k μ ],  μ=Q^{-1}b.
    /// Equivalently, since β|ω,y,ρ ~ N(μ,Q^{-1}):
    ///   E[βᵀS_kβ | ω,y,ρ] = tr(S_k Q^{-1}) + μᵀS_kμ.
    ///
    /// yielding exact (but high-dimensional) contour integrals / series after
    /// analytically integrating β.
    ///
    /// Practical note:
    /// - These are exact equalities but generally not polynomial-time tractable
    ///   for arbitrary dense (X, n, p).
    /// - This code therefore uses deterministic Laplace/implicit-differentiation
    ///   machinery for the main optimizer path, with exact tensor terms where
    ///   feasible (H_k, H_{kℓ}, c/d arrays), and scalable trace backends.
    ///
    /// Full outer-derivative reference (exact system, sign convention used here).
    /// This optimizer minimizes an outer cost V(ρ).
    ///
    /// Common definitions:
    ///   λ_k = exp(ρ_k)
    ///   S(ρ) = Σ_k λ_k S_k + δI
    ///   A_k = ∂S/∂ρ_k = λ_k S_k
    ///   A_{kℓ} = ∂²S/(∂ρ_k∂ρ_ℓ) = δ_{kℓ} A_k
    ///
    /// Inner mode (β̂):
    ///   ∇_β ℓ(β̂) - S(ρ) β̂ = 0
    ///
    /// Curvature:
    ///   H(ρ) = -∇²_β ℓ(β̂(ρ)) + S(ρ)
    ///
    ///   w_i = -∂²ℓ_i/∂η_i²
    ///   d_i = -∂³ℓ_i/∂η_i³
    ///   e_i = -∂⁴ℓ_i/∂η_i⁴
    ///
    /// Then:
    ///   H_k = A_k + Xᵀ diag(d ⊙ u_k) X,     u_k := X B_k
    ///   H_{kℓ} = δ_{kℓ}A_k + Xᵀ diag(e ⊙ u_k ⊙ u_ℓ + d ⊙ u_{kℓ}) X
    ///
    /// with implicit derivatives:
    ///   B_k := ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
    ///   H B_{kℓ} = -(H_ℓ B_k + δ_{kℓ}A_k β̂ + A_k B_ℓ)
    ///
    /// Non-Gaussian negative LAML cost:
    ///   V(ρ) = -ℓ(β̂) + 0.5 β̂ᵀSβ̂ + 0.5 log|H| - 0.5 log|S|_+
    ///
    /// Exact gradient:
    ///   g_k = 0.5 β̂ᵀA_kβ̂ + 0.5 tr(H^{-1}H_k) - 0.5 ∂_k log|S|_+
    ///
    /// Exact Hessian decomposition:
    ///   ∂²V/(∂ρ_k∂ρ_ℓ) = Q_{kℓ} + L_{kℓ} + P_{kℓ}
    ///
    ///   Q_{kℓ} = 0.5 δ_{kℓ} β̂ᵀA_kβ̂ - B_ℓᵀ H B_k
    ///
    ///   L_{kℓ} = 0.5 [ tr(H^{-1}H_{kℓ}) - tr(H^{-1}H_k H^{-1}H_ℓ) ]
    ///
    ///   P_{kℓ} = -0.5 ∂²_{kℓ} log|S|_+
    ///
    /// Here, this function computes the exact gradient terms (including dH/dρ_k via d_i).
    /// The full exact Hessian is not assembled in this loop because it requires B_{kℓ}
    /// solves and fourth-derivative terms for every (k,ℓ) pair.
    ///
    /// Gaussian REML note:
    ///   In identity-link Gaussian, d=e=0 so H_k=A_k and H_{kℓ}=δ_{kℓ}A_k.
    ///   With profiled φ, use either:
    ///   - explicit profiled objective derivatives, or
    ///   - Schur complement in (ρ, log φ):
    ///       H_prof = H_{ρρ} - H_{ρα} H_{αα}^{-1} H_{αρ}.
    ///
    /// Pseudo-determinant note:
    ///   The code uses fixed-rank/stabilized conventions for log|S|_+ to keep objective
    ///   derivatives smooth and consistent with the transformed penalty basis used by PIRLS.
    ///
    /// This is the core of the outer optimization loop and provides the search direction for the BFGS algorithm.
    /// The calculation differs significantly between the Gaussian (REML) and non-Gaussian (LAML) cases.
    ///
    /// # Mathematical Basis (Gaussian/REML Case)
    ///
    /// For Gaussian models (Identity link), we minimize the negative REML log-likelihood, which serves as our cost function.
    /// From Wood (2011, JRSSB, Eq. 4), the cost function to minimize is:
    ///
    ///   Cost(ρ) = -l_r(ρ) = D_p / (2φ) + (1/2)log|XᵀWX + S(ρ)| - (1/2)log|S(ρ)|_+
    ///
    /// where D_p is the penalized deviance, H = XᵀWX + S(ρ) is the penalized Hessian, S(ρ) is the total
    /// penalty matrix, and |S(ρ)|_+ is the pseudo-determinant.
    ///
    /// The gradient ∇Cost(ρ) is computed term-by-term. A key simplification for the Gaussian case is the
    /// **envelope theorem**: at the P-IRLS optimum for β̂, the derivative of the cost function with respect to β̂ is zero.
    /// This means we only need the *partial* derivatives with respect to ρ, and the complex indirect derivatives
    /// involving ∂β̂/∂ρ can be ignored.
    ///
    /// # Mathematical Basis (Non-Gaussian/LAML Case)
    ///
    /// For non-Gaussian models, the envelope theorem does not apply because the weight matrix W depends on β̂.
    /// The gradient requires calculating the full derivative, including the indirect term (∂V/∂β̂)ᵀ(∂β̂/∂ρ).
    /// This leads to a different final formula involving derivatives of the weight matrix, as detailed in
    /// Wood (2011, Appendix D).
    ///
    /// This method handles two distinct statistical criteria for marginal likelihood optimization:
    ///
    /// - For Gaussian models (Identity link), this calculates the exact REML gradient
    ///   (Restricted Maximum Likelihood).
    /// - For non-Gaussian GLMs, this calculates the LAML gradient (Laplace Approximate
    ///   Marginal Likelihood) as derived in Wood (2011, Appendix C & D).
    ///
    /// # Mathematical Theory
    ///
    /// The gradient calculation requires careful application of the chain rule and envelope theorem
    /// due to the nested optimization structure of GAMs:
    ///
    /// - The inner loop (P-IRLS) finds coefficients β̂ that maximize the penalized log-likelihood
    ///   for a fixed set of smoothing parameters ρ.
    /// - The outer loop (BFGS) finds smoothing parameters ρ that maximize the marginal likelihood.
    ///
    /// Since β̂ is an implicit function of ρ, the total derivative is:
    ///
    ///    dV_R/dρ_k = (∂V_R/∂β̂)ᵀ(∂β̂/∂ρ_k) + ∂V_R/∂ρ_k
    ///
    /// By the envelope theorem, (∂V_R/∂β̂) = 0 at the optimum β̂, so the first term vanishes.
    ///
    /// # Key Distinction Between REML and LAML Gradients
    ///
    /// - Gaussian (REML): by the envelope theorem the indirect β̂ terms vanish. The deviance
    ///   contribution reduces to the penalty-only derivative, yielding the familiar
    ///   (β̂ᵀS_kβ̂)/σ² piece in the gradient.
    /// - Non-Gaussian (LAML): there is no cancellation of the penalty derivative within the
    ///   deviance component. The derivative of the penalized deviance contains both
    ///   d(D)/dρ_k and d(βᵀSβ)/dρ_k. Our implementation follows mgcv’s gdi1: we add the penalty
    ///   derivative to the deviance derivative before applying the 1/2 factor.
    // Stage: Start with the chain rule for any λₖ,
    //     dV/dλₖ = ∂V/∂λₖ  (holding β̂ fixed)  +  (∂V/∂β̂)ᵀ · (∂β̂/∂λₖ).
    //     The first summand is called the direct part, the second the indirect part.
    //
    // Stage: Note the two outer criteria—Gaussian likelihood maximizes REML, while non-Gaussian likelihood
    //     maximizes a Laplace approximation to the marginal likelihood (LAML). These objectives respond differently to β̂.
    //
    //     2.1  Gaussian case, REML.
    //          The REML construction integrates the fixed effects out of the likelihood.  At the optimum
    //          the partial derivative ∂V/∂β̂ is exactly zero.  The indirect part therefore vanishes.
    //          What remains is the direct derivative of the penalty and determinant terms.  The penalty
    //          contribution is found by differentiating −½ β̂ᵀ S_λ β̂ / σ² with respect to λₖ; this yields
    //          −½ β̂ᵀ Sₖ β̂ / σ².  No opposing term exists, so the quantity stays in the REML gradient.
    //          The code path selected by LinkFunction::Identity therefore computes
    //          beta_term = β̂ᵀ Sₖ β̂ and places it inside
    //          gradient[k] = 0.5 * λₖ * (beta_term / σ² − trace_term).
    //
    //     2.2  Non-Gaussian case, LAML.
    //          The Laplace objective contains −½ log |H_p| with H_p = Xᵀ W(β̂) X + S_λ.  Because W
    //          depends on β̂, the total derivative includes dW/dλₖ via β̂.  Differentiating the
    //          optimality condition for β̂ gives
    //          ∂β̂/∂λₖ = −λₖ H_p⁻¹ Sₖ β̂.  The penalized log-likelihood L(β̂, λ) still obeys the
    //          envelope theorem, so dL/dλₖ = −½ β̂ᵀ Sₖ β̂ (no implicit term).
    //          The resulting cost gradient combines four pieces:
    //            +½ λₖ β̂ᵀ Sₖ β̂
    //            +½ λₖ tr(H_p⁻¹ Sₖ)
    //            +½ tr(H_p⁻¹ Xᵀ ∂W/∂λₖ X)
    //            −½ λₖ tr(S_λ⁺ Sₖ)
    //
    // Stage: Remember that the sign of ∂β̂/∂λₖ matters; from the implicit-function theorem the linear solve reads
    //     −H_p (∂β̂/∂λₖ) = λₖ Sₖ β̂, giving the minus sign used above.  With that sign the indirect and
    //     direct quadratic pieces are exact negatives, which is what the algebra requires.
    /// Build the exact Jeffreys operator in the same coefficient basis used by
    /// the outer Hessian operator and derivative provider.
    ///
    /// If the outer solve is projected to a free subspace `β = Z β_free`, the
    /// Jeffreys term must also be evaluated on `X_eff = X Z`. Reusing the
    /// cached full-basis operator in that situation would recreate the original
    /// value/derivative mismatch this code is meant to eliminate.
    fn build_dense_firth_operator_for_outer_basis(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
        free_basis_opt: &Option<Array2<f64>>,
    ) -> Result<Option<std::sync::Arc<super::FirthDenseOperator>>, EstimationError> {
        if !(self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit))
        {
            return Ok(None);
        }

        if let Some(z) = free_basis_opt.as_ref() {
            let x_projected = pirls_result.x_transformed.to_dense().dot(z);
            return Ok(Some(std::sync::Arc::new(Self::build_firth_dense_operator(
                &x_projected,
                &pirls_result.final_eta,
                self.weights,
            )?)));
        }

        if let Some(cached) = bundle.firth_dense_operator.clone() {
            return Ok(Some(cached));
        }

        let x_dense = pirls_result.x_transformed.to_dense();
        Ok(Some(std::sync::Arc::new(Self::build_firth_dense_operator(
            &x_dense,
            &pirls_result.final_eta,
            self.weights,
        )?)))
    }

    /// Build the derivative provider, dispersion handling, zeroed TK placeholders,
    /// exact Firth operator, log-likelihood, and barrier config
    /// that are common to all dense evaluation paths.
    ///
    /// # Arguments
    /// - `pirls_result`: the inner-loop solution
    /// - `bundle`: the evaluation bundle (carries h_eff, firth operator, etc.)
    /// - `free_basis_opt`: optional constraint-projection basis
    /// - `include_firth_derivs`: whether to wrap the derivative provider with
    ///   Firth-aware derivatives
    ///
    /// The actual Tierney-Kadane correction is added outside the unified
    /// evaluator so value, gradient, and Hessian can all come from the same
    /// invariant scalar correction.
    fn build_dense_derivative_context(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
        free_basis_opt: &Option<Array2<f64>>,
        include_firth_derivs: bool,
    ) -> Result<
        (
            Box<dyn super::unified::HessianDerivativeProvider>,
            super::unified::DispersionHandling,
            f64,                                               // tk_correction
            Option<Array1<f64>>,                               // tk_gradient
            f64,                                               // log_likelihood
            Option<std::sync::Arc<super::FirthDenseOperator>>, // firth_op
            Option<super::unified::BarrierConfig>,
        ),
        EstimationError,
    > {
        use super::unified::{
            DispersionHandling, GaussianDerivatives, SinglePredictorGlmDerivatives,
        };

        let is_gaussian_identity = self.config.link_function() == LinkFunction::Identity;
        let firth_op =
            self.build_dense_firth_operator_for_outer_basis(pirls_result, bundle, free_basis_opt)?;

        // Derivative provider.
        let firth_active_for_derivs = include_firth_derivs && firth_op.is_some();
        let deriv_provider: Box<dyn super::unified::HessianDerivativeProvider> =
            if is_gaussian_identity {
                Box::new(GaussianDerivatives)
            } else {
                // Differentiate the same Hessian-side curvature arrays that the
                // inner PIRLS solve accepted at the mode.
                let (c_array, d_array) = self.hessian_cd_arrays(pirls_result)?;
                let x_transformed = if let Some(z) = free_basis_opt.as_ref() {
                    // Project the design: X_proj = X Z
                    let x_dense = pirls_result.x_transformed.to_dense();
                    crate::linalg::matrix::DesignMatrix::Dense(
                        crate::matrix::DenseDesignMatrix::from(x_dense.dot(z)),
                    )
                } else {
                    pirls_result.x_transformed.clone()
                };
                let base = SinglePredictorGlmDerivatives {
                    c_array,
                    d_array: Some(d_array),
                    x_transformed,
                };
                if firth_active_for_derivs {
                    if let Some(firth_op) = firth_op.clone() {
                        Box::new(super::unified::FirthAwareGlmDerivatives { base, firth_op })
                    } else {
                        Box::new(base)
                    }
                } else {
                    Box::new(base)
                }
            };

        // Dispersion handling.
        let dispersion = if is_gaussian_identity {
            DispersionHandling::ProfiledGaussian
        } else {
            DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            }
        };

        let (tk_correction, tk_gradient) = (0.0, None);

        let log_likelihood = crate::pirls::calculate_loglikelihood_omitting_constants(
            self.y,
            &pirls_result.finalmu,
            self.config.likelihood(),
            self.weights,
        );

        // Construct barrier config for monotonicity constraints when no
        // active-set projection is in effect (barrier indices are in the
        // full transformed-coefficient space).
        let barrier_config = if free_basis_opt.is_none() {
            pirls_result
                .linear_constraints_transformed
                .as_ref()
                .and_then(Self::barrier_config_from_constraints)
        } else {
            None
        };

        Ok((
            deriv_provider,
            dispersion,
            tk_correction,
            tk_gradient,
            log_likelihood,
            firth_op,
            barrier_config,
        ))
    }

    /// Build the dispersion handling, derivative provider, log-likelihood,
    /// exact Firth operator, and barrier config that are common
    /// to both sparse evaluation paths (`evaluate_unified_sparse` and
    /// `evaluate_unified_sparse_for_efs`).
    ///
    /// TK correction is NOT included here because the sparse paths compute
    /// it via sparse Cholesky solves with completely different logic.
    fn build_sparse_derivative_context(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
    ) -> Result<
        (
            super::unified::DispersionHandling,
            Box<dyn super::unified::HessianDerivativeProvider>,
            f64,                                               // log_likelihood
            Option<std::sync::Arc<super::FirthDenseOperator>>, // firth_op
            Option<super::unified::BarrierConfig>,
        ),
        EstimationError,
    > {
        use super::unified::{
            DispersionHandling, FirthAwareGlmDerivatives, GaussianDerivatives,
            SinglePredictorGlmDerivatives,
        };

        let is_gaussian_identity = self.config.link_function() == LinkFunction::Identity;

        // Sparse exact still uses the same dense Jeffreys operator; only the
        // H^{-1} applications move to the sparse Cholesky operator.
        let firth_op = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            if let Some(cached) = bundle.firth_dense_operator_original.clone() {
                Some(cached)
            } else {
                let x_dense = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact REML runtime requires dense design for Firth operator",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Some(std::sync::Arc::new(Self::build_firth_dense_operator(
                    x_dense.as_ref(),
                    &pirls_result.final_eta,
                    self.weights,
                )?))
            }
        } else {
            None
        };

        // Dispersion and derivative provider depend on family.
        let (dispersion, deriv_provider): (_, Box<dyn super::unified::HessianDerivativeProvider>) =
            if is_gaussian_identity {
                (
                    DispersionHandling::ProfiledGaussian,
                    Box::new(GaussianDerivatives),
                )
            } else {
                // Differentiate the same Hessian-side curvature arrays that the
                // inner PIRLS solve accepted at the mode.
                let (c_array, d_array) = self.hessian_cd_arrays(pirls_result)?;
                (
                    DispersionHandling::Fixed {
                        phi: 1.0,
                        include_logdet_h: true,
                        include_logdet_s: true,
                    },
                    {
                        let base = SinglePredictorGlmDerivatives {
                            c_array,
                            d_array: Some(d_array),
                            x_transformed: self.x().clone(),
                        };
                        // Match the dense exact path: when Firth-logit is
                        // active, the directional log|H_total| drift includes
                        // -D(H_phi)[B_k]. The sparse backend differs only in how
                        // B_k = H^{-1} rhs is solved.
                        if let Some(firth_op) = firth_op.clone() {
                            Box::new(FirthAwareGlmDerivatives { base, firth_op })
                        } else {
                            Box::new(base)
                        }
                    },
                )
            };

        let log_likelihood = crate::pirls::calculate_loglikelihood_omitting_constants(
            self.y,
            &pirls_result.finalmu,
            self.config.likelihood(),
            self.weights,
        );

        // Construct barrier config for monotonicity constraints.
        // The sparse path operates in the original (non-reparameterized)
        // coordinate system, so use the original linear constraints.
        let barrier_config = self
            .linear_constraints
            .as_ref()
            .and_then(Self::barrier_config_from_constraints);

        Ok((
            dispersion,
            deriv_provider,
            log_likelihood,
            firth_op,
            barrier_config,
        ))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified inner-solution assembly
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build penalty coordinates from canonical penalties, with Kronecker
    /// fast-path when available and active.
    fn build_penalty_coords(&self) -> Vec<super::unified::PenaltyCoordinate> {
        if let Some(ref kron) = self.kronecker_penalty_system {
            if self.kronecker_factored.is_some() {
                let d = kron.ndim();
                let total_dim = kron.p_total();
                let eigenvalues: Vec<ndarray::Array1<f64>> = kron
                    .marginal_eigensystems
                    .iter()
                    .map(|(evals, _)| evals.clone())
                    .collect();
                let mut coords = Vec::with_capacity(kron.num_penalties());
                for k in 0..d {
                    coords.push(super::unified::PenaltyCoordinate::KroneckerMarginal {
                        eigenvalues: eigenvalues.clone(),
                        dim_index: k,
                        marginal_dims: kron.marginal_dims.clone(),
                        total_dim,
                    });
                }
                if kron.has_double_penalty {
                    let identity_root = ndarray::Array2::<f64>::eye(total_dim);
                    coords.push(super::unified::PenaltyCoordinate::from_dense_root(
                        identity_root,
                    ));
                }
                return coords;
            }
        }
        self.canonical_penalties
            .iter()
            .map(|cp| cp.to_penalty_coordinate())
            .collect()
    }

    /// Build an `InnerAssembly` using the dense-transformed backend.
    ///
    /// Shared ingredient builder for all dense-spectral evaluation paths
    /// (normal, ext-coord, and EFS). Callers may set `ext_coords` and pair
    /// functions on the returned assembly before passing it to
    /// `assemble_and_evaluate` or `assemble_and_evaluate_efs`.
    fn build_dense_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        use super::unified::DenseSpectralOperator;

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let (h_for_operator, e_for_logdet) = if let Some(z) = free_basis_opt.as_ref() {
            (
                Self::projectwith_basis(bundle.h_total.as_ref(), z),
                pirls_result.reparam_result.e_transformed.dot(z),
            )
        } else {
            (
                bundle.h_total.as_ref().clone(),
                pirls_result.reparam_result.e_transformed.clone(),
            )
        };

        let hessian_op = Box::new(
            DenseSpectralOperator::from_symmetric(&h_for_operator).map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "DenseSpectralOperator from PIRLS Hessian: {e}"
                ))
            })?,
        );

        let (penalty_rank, penalty_logdet) =
            self.dense_penalty_logdet_derivs(rho, &e_for_logdet, &[], ridge_passport, mode)?;

        let beta = if let Some(z) = free_basis_opt.as_ref() {
            z.t().dot(&pirls_result.beta_transformed.as_ref().to_owned())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };

        let nullspace_dim = h_for_operator.ncols().saturating_sub(penalty_rank) as f64;

        let (deriv_provider, dispersion, tk_correction, tk_gradient, log_likelihood, firth_op, barrier_config) =
            self.build_dense_derivative_context(pirls_result, bundle, &free_basis_opt, true)?;

        Ok(super::assembly::InnerAssembly {
            log_likelihood,
            penalty_quadratic: pirls_result.stable_penalty_term,
            beta,
            n_observations: self.y.len(),
            hessian_op,
            penalty_coords: self.build_penalty_coords(),
            penalty_logdet,
            dispersion,
            deriv_provider: Some(deriv_provider),
            tk_correction,
            tk_gradient,
            firth: firth_op,
            nullspace_dim: Some(nullspace_dim),
            barrier_config,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
        })
    }

    /// Build an `InnerAssembly` using the sparse-exact backend.
    ///
    /// Shared ingredient builder for all sparse Cholesky evaluation paths.
    fn build_sparse_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        use super::unified::{HessianOperator, PenaltyLogdetDerivs, SparseCholeskyOperator};

        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();

        let beta = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta.len();
        let hessian_op: Box<dyn HessianOperator> = {
            let mut op = SparseCholeskyOperator::new(sparse.factor.clone(), sparse.logdet_h, p_dim);
            if let Some(ref taka) = sparse.takahashi {
                op = op.with_takahashi(taka.clone());
            }
            Box::new(op)
        };

        log::trace!(
            "SparseCholeskyOperator: dim={}, active_rank={}",
            hessian_op.dim(),
            hessian_op.active_rank()
        );

        let mp = self.nullspace_dims.iter().copied().sum::<usize>() as f64;
        let det2 = if mode == super::unified::EvalMode::ValueGradientHessian {
            let lambdas = rho.mapv(f64::exp);
            let (_, det2) = self.structural_penalty_logdet_derivatives_block_local(
                &lambdas,
                bundle.ridge_passport.penalty_logdet_ridge(),
            )?;
            Some(det2)
        } else {
            None
        };
        let penalty_logdet = PenaltyLogdetDerivs {
            value: sparse.logdet_s_pos,
            first: sparse.det1_values.as_ref().clone(),
            second: det2,
        };

        let (dispersion, deriv_provider, log_likelihood, firth_op, barrier_config) =
            self.build_sparse_derivative_context(pirls_result, bundle)?;

        Ok(super::assembly::InnerAssembly {
            log_likelihood,
            penalty_quadratic: pirls_result.stable_penalty_term,
            beta,
            n_observations: self.y.len(),
            hessian_op,
            penalty_coords: self.build_penalty_coords(),
            penalty_logdet,
            dispersion,
            deriv_provider: Some(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: firth_op,
            nullspace_dim: Some(mp),
            barrier_config,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
        })
    }

    /// Build the soft prior tuple for the given mode.
    fn build_prior(
        &self,
        rho: &Array1<f64>,
        mode: super::unified::EvalMode,
    ) -> Option<(f64, Array1<f64>, Option<Array2<f64>>)> {
        super::assembly::soft_prior_for_mode(
            rho,
            mode,
            |r| self.compute_soft_priorcost(r),
            |r| self.compute_soft_priorgrad(r),
            |r| self.compute_soft_priorhess(r),
        )
    }

    /// Single assembly point: evaluate an `InnerAssembly`, compute prior, and
    /// apply TK correction.
    ///
    /// All `evaluate_unified*` methods funnel through here.
    fn assemble_and_evaluate(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        assembly: super::assembly::InnerAssembly<'static>,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let prior = self.build_prior(rho, mode);
        let result = assembly
            .evaluate(rho.as_slice().unwrap(), mode, prior)
            .map_err(EstimationError::InvalidInput)?;
        let tk_terms = self.tierney_kadane_terms(rho, bundle, mode)?;
        Ok(self.apply_tk_to_result(result, rho, tk_terms))
    }

    /// Single assembly point for EFS: build `InnerSolution` from an assembly,
    /// compute cost (and gradient when ψ coords are present), return EFS step
    /// vector with TK cost correction applied.
    fn assemble_and_evaluate_efs(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        assembly: super::assembly::InnerAssembly<'static>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        use super::unified::{compute_efs_update, compute_hybrid_efs_update};

        let beta_for_barrier = assembly.beta.clone();
        let inner_solution = assembly.build();

        let has_psi = inner_solution.ext_coords.iter().any(|c| !c.is_penalty_like);
        let eval_mode = if has_psi {
            super::unified::EvalMode::ValueAndGradient
        } else {
            super::unified::EvalMode::ValueOnly
        };

        let prior = self.build_prior(rho, eval_mode);
        let cost_result = super::assembly::evaluate_solution(
            &inner_solution, rho.as_slice().unwrap(), eval_mode, prior,
        ).map_err(EstimationError::InvalidInput)?;

        let efs_eval = if has_psi {
            let gradient = cost_result.gradient.as_ref().expect(
                "hybrid EFS requires gradient (ValueAndGradient mode should have computed it)",
            );
            let hybrid = compute_hybrid_efs_update(
                &inner_solution,
                rho.as_slice().unwrap(),
                gradient.as_slice().unwrap(),
            );
            let psi_gradient = if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(ndarray::Array1::from_vec(hybrid.psi_gradient))
            };
            let psi_indices = if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(hybrid.psi_indices)
            };
            crate::solver::outer_strategy::EfsEval {
                cost: cost_result.cost,
                steps: hybrid.steps,
                beta: Some(beta_for_barrier),
                psi_gradient,
                psi_indices,
            }
        } else {
            let steps = compute_efs_update(&inner_solution, rho.as_slice().unwrap());
            crate::solver::outer_strategy::EfsEval {
                cost: cost_result.cost,
                steps,
                beta: Some(beta_for_barrier),
                psi_gradient: None,
                psi_indices: None,
            }
        };

        let tk_terms =
            self.tierney_kadane_terms(rho, bundle, super::unified::EvalMode::ValueOnly)?;
        Ok(self.apply_tk_cost_to_efs(efs_eval, tk_terms.value))
    }

    /// Evaluate the REML/LAML objective (and optionally gradient) through the
    /// unified evaluator, using a pre-obtained eval bundle.
    ///
    /// This is the single bridge from the existing PIRLS infrastructure to the
    /// unified `reml_laml_evaluate` function. Both `compute_cost` and
    /// `compute_gradient` ultimately delegate here, ensuring that cost and
    /// gradient share the exact same formula.
    pub fn evaluate_unified(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        self.assemble_and_evaluate(rho, bundle, mode,
            self.build_dense_assembly(rho, bundle, mode)?)
    }

    /// Sparse-exact bridge: delegates to `build_sparse_assembly` + `assemble_and_evaluate`.
    pub fn evaluate_unified_sparse(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let assembly = self.build_sparse_assembly(rho, bundle, mode)?;
        self.assemble_and_evaluate(rho, bundle, mode, assembly)
    }

    /// Sparse-exact EFS bridge: delegates to `build_sparse_assembly` +
    /// `assemble_and_evaluate_efs`.
    fn evaluate_unified_sparse_for_efs(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        let mut assembly = self.build_sparse_assembly(
            rho, bundle, super::unified::EvalMode::ValueOnly,
        )?;
        // EFS doesn't use TK gradient.
        assembly.tk_gradient = None;
        self.assemble_and_evaluate_efs(rho, bundle, assembly)
    }

    /// Evaluate the unified REML/LAML objective with anisotropic ψ ext_coords
    /// injected into the `InnerSolution`.
    ///
    /// This is the primary bridge for joint [ρ, ψ] optimization of anisotropic
    /// spatial smooths. It:
    /// 1. Obtains the PIRLS bundle at the current ρ (with the design already
    ///    rebuilt for the current ψ).
    /// 2. Builds `HyperCoord` ext_coords from the supplied `DirectionalHyperParam`
    ///    list via the dense/transformed or sparse/native τ builders, depending
    ///    on the active backend.
    /// 3. Builds the full `InnerSolution` (mirroring `evaluate_unified`), injects
    ///    the ext_coords, and calls `reml_laml_evaluate`.
    ///
    /// The caller is responsible for rebuilding the `RemlState` with the design
    /// X(ψ) and penalties S(ψ) corresponding to the current ψ values before
    /// calling this method. The unified evaluator then preserves the backend:
    /// dense spectral paths stay dense, and sparse exact paths stay sparse.
    pub fn evaluate_unified_with_psi_ext(
        &self,
        rho: &Array1<f64>,
        mode: super::unified::EvalMode,
        hyper_dirs: &[crate::estimate::reml::DirectionalHyperParam],
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;

        let (ext_coords, ext_pair_fn, rho_ext_pair_fn) = if !hyper_dirs.is_empty() {
            if mode == super::unified::EvalMode::ValueGradientHessian {
                let (coords, epf, repf) =
                    self.build_tau_unified_objects_from_bundle(rho, &bundle, hyper_dirs)?;
                (coords, Some(epf), Some(repf))
            } else if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
                (
                    self.build_tau_hyper_coords_sparse_exact(rho, &bundle, hyper_dirs)?,
                    None,
                    None,
                )
            } else if matches!(
                bundle.pirls_result.coordinate_frame,
                pirls::PirlsCoordinateFrame::TransformedQs
            ) && self
                .active_constraint_free_basis(bundle.pirls_result.as_ref())
                .is_none()
            {
                (
                    self.build_tau_hyper_coords_original_basis(rho, &bundle, hyper_dirs)?,
                    None,
                    None,
                )
            } else {
                (
                    self.build_tau_hyper_coords(rho, &bundle, hyper_dirs)?,
                    None,
                    None,
                )
            }
        } else {
            (Vec::new(), None, None)
        };

        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            self.evaluate_unified_sparse_with_ext_from_bundle(
                rho,
                &bundle,
                mode,
                ext_coords,
                ext_pair_fn,
                rho_ext_pair_fn,
            )
        } else {
            self.evaluate_unified_with_ext_from_bundle(
                rho,
                &bundle,
                mode,
                ext_coords,
                ext_pair_fn,
                rho_ext_pair_fn,
            )
        }
    }

    /// Prepare ingredients for the dense-reparam evaluation path.
    ///
    /// Shared between the normal, ext-coord, and EFS dense paths.
    /// Returns an `InnerAssembly` without ext_coords — the caller injects
    /// those before evaluating.
    fn prepare_dense_reparam_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        penalty_logdet_mode: super::unified::EvalMode,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        use super::unified::DenseSpectralOperator;

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let (h_for_operator, e_for_logdet) = if let Some(z) = free_basis_opt.as_ref() {
            (
                Self::projectwith_basis(bundle.h_total.as_ref(), z),
                pirls_result.reparam_result.e_transformed.dot(z),
            )
        } else {
            (
                bundle.h_total.as_ref().clone(),
                pirls_result.reparam_result.e_transformed.clone(),
            )
        };

        let hessian_op = Box::new(
            DenseSpectralOperator::from_symmetric(&h_for_operator).map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "DenseSpectralOperator from PIRLS Hessian: {e}"
                ))
            })?,
        );

        let (penalty_rank, penalty_logdet) = self.dense_penalty_logdet_derivs(
            rho, &e_for_logdet, &[], ridge_passport, penalty_logdet_mode,
        )?;

        let beta = if let Some(z) = free_basis_opt.as_ref() {
            z.t().dot(&pirls_result.beta_transformed.as_ref().to_owned())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };

        let p_eff_dim = h_for_operator.ncols();
        let nullspace_dim = p_eff_dim.saturating_sub(penalty_rank) as f64;

        let (deriv_provider, dispersion, tk_correction, tk_gradient, log_likelihood, firth_op, barrier_config) =
            self.build_dense_derivative_context(pirls_result, bundle, &free_basis_opt, true)?;

        Ok(super::assembly::InnerAssembly {
            log_likelihood,
            penalty_quadratic: pirls_result.stable_penalty_term,
            beta,
            n_observations: self.y.len(),
            hessian_op,
            penalty_coords: self.build_penalty_coords(),
            penalty_logdet,
            dispersion,
            deriv_provider: Some(deriv_provider),
            tk_correction,
            tk_gradient,
            firth: firth_op,
            nullspace_dim: Some(nullspace_dim),
            barrier_config,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
        })
    }

    /// Dense-transformed bridge with optional ext_coords. Dispatches to the
    /// original-basis path when QS-transformed but unconstrained.
    fn evaluate_unified_with_ext_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        ext_coords: Vec<super::unified::HyperCoord>,
        ext_coord_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
        rho_ext_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let pirls_result = bundle.pirls_result.as_ref();
        if matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::TransformedQs
        ) && self.active_constraint_free_basis(pirls_result).is_none()
        {
            return self.evaluate_unified_dense_original_with_ext_from_bundle(
                rho, bundle, mode, ext_coords, ext_coord_pair_fn, rho_ext_pair_fn,
            );
        }

        let mut assembly = self.build_dense_assembly(rho, bundle, mode)?;
        assembly.ext_coords = ext_coords;
        assembly.ext_coord_pair_fn = ext_coord_pair_fn;
        assembly.rho_ext_pair_fn = rho_ext_pair_fn;
        self.assemble_and_evaluate(rho, bundle, mode, assembly)
    }

    /// Dense original-basis bridge with ext_coords.
    fn evaluate_unified_dense_original_with_ext_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        ext_coords: Vec<super::unified::HyperCoord>,
        ext_coord_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
        rho_ext_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let mut assembly = self.build_dense_original_assembly(rho, bundle, mode)?;
        assembly.ext_coords = ext_coords;
        assembly.ext_coord_pair_fn = ext_coord_pair_fn;
        assembly.rho_ext_pair_fn = rho_ext_pair_fn;
        self.assemble_and_evaluate(rho, bundle, mode, assembly)
    }

    /// Sparse-exact bridge for unified evaluation with extended τ/ψ coordinates.
    ///
    /// This mirrors `evaluate_unified_sparse` but injects ext-coordinates into
    /// the `InnerSolution` so the unified evaluator can compute joint [ρ, τ/ψ]
    /// value/gradient/Hessian while still using the sparse Cholesky operator.
    fn evaluate_unified_sparse_with_ext_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        ext_coords: Vec<super::unified::HyperCoord>,
        ext_coord_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
        rho_ext_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let mut assembly = self.build_sparse_assembly(rho, bundle, mode)?;
        assembly.ext_coords = ext_coords;
        assembly.ext_coord_pair_fn = ext_coord_pair_fn;
        assembly.rho_ext_pair_fn = rho_ext_pair_fn;
        self.assemble_and_evaluate(rho, bundle, mode, assembly)
    }

    /// Compute EFS (Extended Fellner-Schall) evaluation: runs the inner solve
    /// at the given rho, computes the REML/LAML cost, and returns the EFS
    /// step vector from `compute_efs_update`.
    ///
    /// This is the entry point called by the EFS branch of `run_outer` via
    /// the `OuterObjective::eval_efs` method.
    pub fn compute_efs_steps(
        &self,
        p: &Array1<f64>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(EstimationError::RemlOptimizationFailed(
                    "inner solve ill-conditioned during EFS evaluation".to_string(),
                ));
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };

        // Dispatch to the appropriate backend.
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            return self.evaluate_unified_sparse_for_efs(p, &bundle);
        }

        // Reuse the same setup as evaluate_unified but call the EFS tail.
        self.evaluate_unified_for_efs(p, &bundle)
    }

    /// Builds the InnerSolution from an eval bundle and returns EFS steps.
    /// Dense EFS bridge: reuses `prepare_dense_reparam_assembly`.
    fn evaluate_unified_for_efs(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        let assembly = self.prepare_dense_reparam_assembly(
            rho, bundle, super::unified::EvalMode::ValueOnly,
        )?;
        self.assemble_and_evaluate_efs(rho, bundle, assembly)
    }

    pub fn compute_gradient(&self, p: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.arena
            .lastgradient_used_stochastic_fallback
            .store(false, Ordering::Relaxed);
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(err @ EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(err);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            let result = self.evaluate_unified_sparse(
                p,
                &bundle,
                super::unified::EvalMode::ValueAndGradient,
            )?;
            return Ok(result.gradient.unwrap());
        }
        let result =
            self.evaluate_unified(p, &bundle, super::unified::EvalMode::ValueAndGradient)?;
        Ok(result.gradient.unwrap())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified REML evaluation with link-parameter ext_coords
    // ═══════════════════════════════════════════════════════════════════════════

    /// Evaluate the unified REML/LAML objective with SAS or mixture link
    /// parameters injected as ext_coords into the `InnerSolution`.
    ///
    /// This replaces the manual gradient computation in the outer optimizer's
    /// eval_fn closure. The link parameter derivatives (du/dθ, dW/dθ, dℓ/dθ)
    /// are wrapped as `HyperCoord` objects and passed through the same unified
    /// evaluator that handles ρ coordinates, giving a single Newton/BFGS step
    /// over `[ρ, θ_link]` jointly.
    ///
    /// # Parameters
    ///
    /// - `rho`: log-smoothing parameters (penalty coordinates only, length k)
    /// - `mode`: evaluation mode (cost-only, cost+gradient, cost+gradient+Hessian)
    ///
    /// # Returns
    ///
    /// `RemlLamlResult` whose gradient (when requested) has length `k + aux_dim`,
    /// with the link parameters appended after the ρ coordinates. The caller is
    /// responsible for:
    /// - Applying SAS epsilon reparameterization chain rule (`grad[k] *= d_eps/d_raw`)
    /// - Adding SAS ridge/barrier gradient contributions to `grad[k+1]`
    /// - Adding SAS ridge/barrier cost contributions to `result.cost`
    pub fn evaluate_unified_with_link_ext(
        &self,
        rho: &Array1<f64>,
        mode: super::unified::EvalMode,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        // Firth bias reduction modifies the score equations in ways that the
        // link ext_coord construction does not yet account for. Reject early.
        if self.config.firth_bias_reduction {
            return Err(EstimationError::InvalidInput(
                "link-parameter ext_coord optimization is incompatible with \
                 Firth-adjusted outer gradients"
                    .to_string(),
            ));
        }

        let bundle = self.obtain_eval_bundle(rho)?;

        // Build link ext_coords from the current runtime link state.
        let ext_coords = if let Some(sas_state) = &self.runtime_sas_link_state {
            let is_beta_logistic = matches!(
                self.config.link_function(),
                crate::types::LinkFunction::BetaLogistic
            );
            self.build_sas_link_ext_coords(&bundle, sas_state, is_beta_logistic)?
        } else if let Some(mix_state) = &self.runtime_mixture_link_state {
            self.build_mixture_link_ext_coords(&bundle, mix_state)?
        } else {
            Vec::new()
        };

        if ext_coords.is_empty() {
            // No link parameters to optimize — fall back to standard path.
            return self.evaluate_unified(rho, &bundle, mode);
        }

        // Delegate to the existing ext_coords injection path.
        // No pair callbacks for link parameters: the outer Hessian for link
        // coords is left to BFGS (HessianResult::Unavailable). This matches
        // the current behavior where link parameters have gradient-only.
        self.evaluate_unified_with_ext_from_bundle(rho, &bundle, mode, ext_coords, None, None)
    }

    /// EFS evaluation with link-parameter ext_coords injected.
    ///
    /// This is the hybrid EFS counterpart of [`evaluate_unified_with_link_ext`].
    /// The ρ coordinates receive standard EFS multiplicative updates while the
    /// link parameters (ψ) receive preconditioned gradient steps via the Gram
    /// matrix `G_{de} = tr(H⁻¹ B_d H⁻¹ B_e)`.
    ///
    /// The returned `EfsEval` has `steps` of length `k + aux_dim` where
    /// `k` is the number of ρ coordinates and `aux_dim` is the number of link
    /// parameters. The caller is responsible for applying SAS reparameterization
    /// chain rules to the ψ gradient/steps as needed.
    pub fn compute_efs_steps_with_link_ext(
        &self,
        rho: &Array1<f64>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        if self.config.firth_bias_reduction {
            return Err(EstimationError::InvalidInput(
                "link-parameter ext_coord optimization is incompatible with \
                 Firth-adjusted outer gradients"
                    .to_string(),
            ));
        }

        let bundle = self.obtain_eval_bundle(rho)?;

        // Build link ext_coords from the current runtime link state.
        let ext_coords = if let Some(sas_state) = &self.runtime_sas_link_state {
            let is_beta_logistic = matches!(
                self.config.link_function(),
                crate::types::LinkFunction::BetaLogistic
            );
            self.build_sas_link_ext_coords(&bundle, sas_state, is_beta_logistic)?
        } else if let Some(mix_state) = &self.runtime_mixture_link_state {
            self.build_mixture_link_ext_coords(&bundle, mix_state)?
        } else {
            Vec::new()
        };

        if ext_coords.is_empty() {
            // No link parameters — fall back to standard EFS.
            return self.compute_efs_steps(rho);
        }

        // Dense EFS with link ext_coords injected.
        self.evaluate_unified_for_efs_with_ext(rho, &bundle, ext_coords)
    }

    /// Dense EFS bridge with ext_coords: mirrors `evaluate_unified_for_efs` but
    /// injects ext_coords into the `InnerAssembly`.
    fn evaluate_unified_for_efs_with_ext(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        ext_coords: Vec<super::unified::HyperCoord>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        use super::unified::DenseSpectralOperator;

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let (h_for_operator, e_for_logdet) = if let Some(z) = free_basis_opt.as_ref() {
            (
                Self::projectwith_basis(bundle.h_total.as_ref(), z),
                pirls_result.reparam_result.e_transformed.dot(z),
            )
        } else {
            (
                bundle.h_total.as_ref().clone(),
                pirls_result.reparam_result.e_transformed.clone(),
            )
        };

        let hessian_op = Box::new(
            DenseSpectralOperator::from_symmetric(&h_for_operator).map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "DenseSpectralOperator from PIRLS Hessian (hybrid EFS link): {e}"
                ))
            })?,
        );

        let (penalty_rank, ..) =
            self.fixed_subspace_penalty_rank_and_logdet(&e_for_logdet, ridge_passport)?;

        let (_, penalty_logdet) = self.dense_penalty_logdet_derivs(
            rho,
            &e_for_logdet,
            &[],
            ridge_passport,
            super::unified::EvalMode::ValueOnly,
        )?;

        let beta = if let Some(z) = free_basis_opt.as_ref() {
            z.t()
                .dot(&pirls_result.beta_transformed.as_ref().to_owned())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };

        let p_eff_dim = h_for_operator.ncols();
        let nullspace_dim = p_eff_dim.saturating_sub(penalty_rank) as f64;

        let (
            deriv_provider,
            dispersion,
            tk_correction,
            _,
            log_likelihood,
            firth_op,
            barrier_config,
        ) = self.build_dense_derivative_context(pirls_result, bundle, &free_basis_opt, true)?;

        let assembly = super::assembly::InnerAssembly {
            log_likelihood,
            penalty_quadratic: pirls_result.stable_penalty_term,
            beta,
            n_observations: self.y.len(),
            hessian_op,
            penalty_coords: self.build_penalty_coords(),
            penalty_logdet,
            dispersion,
            deriv_provider: Some(deriv_provider),
            tk_correction,
            tk_gradient: None,
            firth: firth_op,
            nullspace_dim: Some(nullspace_dim),
            barrier_config,
            ext_coords,
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
        };
        self.assemble_and_evaluate_efs(rho, bundle, assembly)
    }
}
