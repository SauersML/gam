use super::*;

impl<'a> RemlState<'a> {
    pub(super) fn validate_joint_hyper_inputs(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<usize, EstimationError> {
        if rho_dim > theta.len() {
            return Err(EstimationError::InvalidInput(format!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            )));
        }
        let psi_dim = theta.len() - rho_dim;
        if hyper_dirs.len() != psi_dim {
            return Err(EstimationError::InvalidInput(format!(
                "joint hyper-gradient derivative count mismatch: psi_dim={}, hyper_dirs={}",
                psi_dim,
                hyper_dirs.len(),
            )));
        }
        Ok(psi_dim)
    }

    pub(super) fn build_joint_perturbed_state(
        &self,
        psi: &Array1<f64>,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<RemlState<'a>, EstimationError> {
        if psi.len() != hyper_dirs.len() {
            return Err(EstimationError::InvalidInput(format!(
                "psi/hyper_dirs mismatch: psi={}, hyper_dirs={}",
                psi.len(),
                hyper_dirs.len()
            )));
        }
        let has_design_drift = hyper_dirs
            .iter()
            .enumerate()
            .any(|(j, dir)| psi[j] != 0.0 && dir.x_tau_original.iter().any(|v| *v != 0.0));
        let mut x_mod_dense = if has_design_drift {
            Some(self.x().to_dense_arc().as_ref().clone())
        } else {
            None
        };
        let mut s_mod = self.s_full_list.clone();
        for (j, dir) in hyper_dirs.iter().enumerate() {
            let amp = psi[j];
            if amp == 0.0 {
                continue;
            }
            if let Some(x_mod) = x_mod_dense.as_ref()
                && dir.x_tau_original.raw_dim() != x_mod.raw_dim()
            {
                return Err(EstimationError::InvalidInput(format!(
                    "joint perturbation X_tau shape mismatch: expected {}x{}, got {}x{}",
                    x_mod.nrows(),
                    x_mod.ncols(),
                    dir.x_tau_original.nrows(),
                    dir.x_tau_original.ncols()
                )));
            }
            if dir.s_tau_original.nrows() != self.p || dir.s_tau_original.ncols() != self.p {
                return Err(EstimationError::InvalidInput(format!(
                    "joint perturbation S_tau shape mismatch: expected {}x{}, got {}x{}",
                    self.p,
                    self.p,
                    dir.s_tau_original.nrows(),
                    dir.s_tau_original.ncols()
                )));
            }
            if let Some(x_mod) = x_mod_dense.as_mut() {
                x_mod.scaled_add(amp, &dir.x_tau_original);
            }
            for s_k in s_mod.iter_mut() {
                s_k.scaled_add(amp, &dir.s_tau_original);
            }
        }
        let x_mod = x_mod_dense
            .map(DesignMatrix::from)
            .unwrap_or_else(|| self.x().clone());
        RemlState::new_with_offset(
            self.y,
            x_mod,
            self.weights,
            self.offset.view(),
            s_mod,
            self.p,
            self.config,
            Some(self.nullspace_dims.clone()),
            self.coefficient_lower_bounds.clone(),
            self.linear_constraints.clone(),
        )
    }

    pub(super) fn compute_multi_psi_gradient_with_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Array1<f64>, EstimationError> {
        let mut psi_grad = Array1::<f64>::zeros(hyper_dirs.len());
        for (i, hyper_dir) in hyper_dirs.iter().enumerate() {
            psi_grad[i] =
                self.compute_directional_hyper_gradient_with_bundle(rho, bundle, hyper_dir)?;
        }
        Ok(psi_grad)
    }

    pub(crate) fn compute_joint_hyper_gradient(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Array1<f64>, EstimationError> {
        let psi_dim = self.validate_joint_hyper_inputs(theta, rho_dim, hyper_dirs)?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        let psi = theta.slice(s![rho_dim..]).to_owned();
        let (rho_grad, psi_grad) = if psi_dim > 0 {
            let pert_state = self.build_joint_perturbed_state(&psi, hyper_dirs)?;
            let bundle = pert_state.obtain_eval_bundle(&rho)?;
            (
                pert_state.compute_gradient_with_bundle(&rho, &bundle)?,
                pert_state.compute_multi_psi_gradient_with_bundle(&rho, &bundle, hyper_dirs)?,
            )
        } else {
            let bundle = self.obtain_eval_bundle(&rho)?;
            (
                self.compute_gradient_with_bundle(&rho, &bundle)?,
                Array1::<f64>::zeros(0),
            )
        };
        let mut out = Array1::<f64>::zeros(theta.len());
        out.slice_mut(s![..rho_dim]).assign(&rho_grad);
        out.slice_mut(s![rho_dim..]).assign(&psi_grad);
        Ok(out)
    }

    #[allow(dead_code)]
    pub(crate) fn compute_joint_hyper_cost_gradient(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<(f64, Array1<f64>), EstimationError> {
        let psi_dim = self.validate_joint_hyper_inputs(theta, rho_dim, hyper_dirs)?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        let psi = theta.slice(s![rho_dim..]).to_owned();
        let (cost, rho_grad, psi_grad) = if psi_dim > 0 {
            let pert_state = self.build_joint_perturbed_state(&psi, hyper_dirs)?;
            let bundle = pert_state.obtain_eval_bundle(&rho)?;
            (
                pert_state.compute_cost(&rho)?,
                pert_state.compute_gradient_with_bundle(&rho, &bundle)?,
                pert_state.compute_multi_psi_gradient_with_bundle(&rho, &bundle, hyper_dirs)?,
            )
        } else {
            let bundle = self.obtain_eval_bundle(&rho)?;
            (
                self.compute_cost(&rho)?,
                self.compute_gradient_with_bundle(&rho, &bundle)?,
                Array1::<f64>::zeros(0),
            )
        };
        let mut out = Array1::<f64>::zeros(theta.len());
        out.slice_mut(s![..rho_dim]).assign(&rho_grad);
        out.slice_mut(s![rho_dim..]).assign(&psi_grad);
        Ok((cost, out))
    }

    pub(super) fn compute_mixed_rho_tau_block(
        &self,
        rho: &Array1<f64>,
        psi: &Array1<f64>,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Array2<f64>, EstimationError> {
        let k_count = rho.len();
        let psi_dim = hyper_dirs.len();
        let mut mixed = Array2::<f64>::zeros((k_count, psi_dim));
        if k_count == 0 || psi_dim == 0 {
            return Ok(mixed);
        }
        // Analytic mixed block assembly for each directional hyperparameter tau_j:
        //
        //   V_{k,tau}
        //   = beta^T A_k beta_tau + 0.5 beta^T A_{k,tau} beta
        //     + 0.5[ tr(H^{-1} H_{k,tau}) - tr(H^{-1} H_tau H^{-1} H_k) ]
        //     - 0.5[ tr(S^+ A_{k,tau}) - tr(S^+ S_tau S^+ A_k) ].
        //
        // with coupled IFT solves
        //   H beta_tau  = g_tau,
        //   H B_k       = -A_k beta,
        //   H B_{k,tau} = -(H_tau B_k + A_k beta_tau + A_{k,tau} beta).
        //
        // This removes the previous FD-in-(tau amplitude) mixed block path.
        let pert_state = self.build_joint_perturbed_state(psi, hyper_dirs)?;
        let bundle = pert_state.obtain_eval_bundle(rho)?;
        for j in 0..psi_dim {
            let col = pert_state.compute_mixed_rho_tau_column_analytic_with_bundle(
                rho,
                &bundle,
                &hyper_dirs[j],
            )?;
            mixed.column_mut(j).assign(&col);
        }
        Ok(mixed)
    }

    pub(super) fn compute_mixed_rho_tau_column_analytic_with_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dir: &DirectionalHyperParam,
    ) -> Result<Array1<f64>, EstimationError> {
        let k_count = rho.len();
        let mut out = Array1::<f64>::zeros(k_count);
        if k_count == 0 {
            return Ok(out);
        }

        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        let mut rs_eval = reparam_result.rs_transformed.clone();
        let mut x_eval = pirls_result
            .x_transformed
            .to_dense_arc()
            .as_ref()
            .to_owned();
        let mut h_eff_eval = bundle.h_eff.as_ref().clone();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut x_tau_t = hyper_dir.x_tau_original.dot(&reparam_result.qs);
        let mut s_tau_t = {
            let tmp = reparam_result.qs.t().dot(&hyper_dir.s_tau_original);
            tmp.dot(&reparam_result.qs)
        };

        if let Some(z) = free_basis_opt.as_ref() {
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            rs_eval = reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect();
            x_eval = x_eval.dot(z);
            h_eff_eval = Self::project_with_basis(bundle.h_eff.as_ref(), z);
            h_total_eval = Self::project_with_basis(bundle.h_total.as_ref(), z);
            x_tau_t = x_tau_t.dot(z);
            s_tau_t = {
                let tmp = z.t().dot(&s_tau_t);
                tmp.dot(z)
            };
        }

        let p_dim = beta_eval.len();
        if p_dim == 0 {
            return Ok(out);
        }
        if x_eval.ncols() != p_dim || x_tau_t.ncols() != p_dim {
            return Err(EstimationError::InvalidInput(format!(
                "mixed rho-tau analytic shape mismatch: X={}x{}, X_tau={}x{}, p={}",
                x_eval.nrows(),
                x_eval.ncols(),
                x_tau_t.nrows(),
                x_tau_t.ncols(),
                p_dim
            )));
        }

        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);
        let h_solve_eval = if firth_logit_active {
            &h_total_eval
        } else {
            &h_eff_eval
        };
        // Smart solve backend:
        // 1) Prefer objective-consistent positive-subspace solve H_+^dagger = W W^T.
        // 2) Fall back to stabilized direct factor solve only when active-subspace diagnostics
        //    are unstable or H_+ factor construction fails in current coordinates.
        let prefer_h_pos = !bundle.active_subspace_unstable;
        let h_pos_w_for_solve = if prefer_h_pos {
            if free_basis_opt.is_none() && bundle.h_pos_factor_w.nrows() == p_dim {
                Some(bundle.h_pos_factor_w.as_ref().clone())
            } else {
                match Self::positive_part_factor_w(h_solve_eval) {
                    Ok(w) => Some(w),
                    Err(err) => {
                        log::warn!(
                            "Mixed rho-tau analytic solve using stabilized direct fallback (failed H_+ projector build: {}).",
                            err
                        );
                        None
                    }
                }
            }
        } else {
            log::warn!(
                "Mixed rho-tau analytic solve using stabilized direct fallback (active subspace unstable)."
            );
            None
        };
        let h_pos_w_for_solve_t = h_pos_w_for_solve.as_ref().map(|w| w.t().to_owned());
        let h_factor = if h_pos_w_for_solve.is_none() {
            Some(self.factorize_faer(h_solve_eval))
        } else {
            None
        };
        let solve_h_vec = |rhs: &Array1<f64>| -> Array1<f64> {
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let tmp = w_t.dot(rhs);
                return w.dot(&tmp);
            }
            let mut rhs_mat = Array2::<f64>::zeros((rhs.len(), 1));
            rhs_mat.column_mut(0).assign(rhs);
            let mut rhs_view = array2_to_mat_mut(&mut rhs_mat);
            if let Some(f) = h_factor.as_ref() {
                f.solve_in_place(rhs_view.as_mut());
            }
            rhs_mat.column(0).to_owned()
        };
        let solve_h_mat = |rhs: &Array2<f64>| -> Array2<f64> {
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let wt_rhs = fast_ab(w_t, rhs);
                return fast_ab(w, &wt_rhs);
            }
            let mut out = rhs.clone();
            let mut out_view = array2_to_mat_mut(&mut out);
            if let Some(f) = h_factor.as_ref() {
                f.solve_in_place(out_view.as_mut());
            }
            out
        };
        let trace_hdag = |a: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let wt_a = fast_ab(w_t, a);
                let g = fast_ab(&wt_a, w);
                g.diag().sum()
            } else {
                let solved = solve_h_mat(a);
                solved.diag().sum()
            }
        };
        // Matrix-free trace contraction for D²(H_phi)[u,v]:
        //   tr(H_+^dagger D²H_phi[u,v]) = tr(W^T (D²H_phi[u,v] W))
        // when the positive-subspace solve projector W is available.
        // Falls back to dense assembly only when W is unavailable.
        let trace_hdag_firth_second =
            |op: &FirthDenseOperator, u_dir: &FirthDirection, v_dir: &FirthDirection| -> f64 {
                Self::trace_hdag_operator_apply(
                    p_dim,
                    h_pos_w_for_solve.as_ref(),
                    h_pos_w_for_solve_t.as_ref(),
                    |a| solve_h_mat(a),
                    |basis| Self::firth_hphi_second_direction_apply(op, u_dir, v_dir, basis),
                )
            };
        let trace_hdag_firth_first = |op: &FirthDenseOperator, u_dir: &FirthDirection| -> f64 {
            Self::trace_hdag_operator_apply(
                p_dim,
                h_pos_w_for_solve.as_ref(),
                h_pos_w_for_solve_t.as_ref(),
                |a| solve_h_mat(a),
                |basis| Self::firth_hphi_direction_apply(op, u_dir, basis),
            )
        };
        let trace_hdag_b_hdag_c = |b: &Array2<f64>, c_mat: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let gb = fast_ab(&fast_ab(w_t, b), w);
                let gc = fast_ab(&fast_ab(w_t, c_mat), w);
                Self::trace_product(&gb, &gc)
            } else {
                let left = solve_h_mat(b);
                let right = solve_h_mat(c_mat);
                Self::trace_product(&left, &right)
            }
        };

        let lambdas = rho.mapv(f64::exp);
        let mut a_k_mats = Vec::<Array2<f64>>::with_capacity(k_count);
        let mut a_k_beta = Vec::<Array1<f64>>::with_capacity(k_count);
        let mut rhs_bk = Array2::<f64>::zeros((p_dim, k_count));
        for k in 0..k_count {
            let r_k = &rs_eval[k];
            let s_k = r_k.t().dot(r_k);
            let a_k = s_k.mapv(|v| lambdas[k] * v);
            let a_kb = a_k.dot(&beta_eval);
            rhs_bk.column_mut(k).assign(&a_kb.mapv(|v| -v));
            a_k_mats.push(a_k);
            a_k_beta.push(a_kb);
        }
        let b_k_mat = solve_h_mat(&rhs_bk);
        let u_k_mat = x_eval.dot(&b_k_mat);

        let u = &pirls_result.solve_weights
            * &(&pirls_result.solve_working_response - &pirls_result.final_eta);
        let x_tau_beta = x_tau_t.dot(&beta_eval);
        let weighted_x_tau_beta = &pirls_result.solve_weights * &x_tau_beta;
        let mut g_tau =
            x_tau_t.t().dot(&u) - x_eval.t().dot(&weighted_x_tau_beta) - s_tau_t.dot(&beta_eval);

        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                Some(Self::build_firth_dense_operator(
                    &x_eval,
                    &pirls_result.final_eta,
                )?)
            }
        } else {
            None
        };
        if let Some(op) = firth_op.as_ref() {
            let (g_phi_tau, _) = Self::firth_partial_score_and_fit_tau(op, &x_tau_t, &beta_eval);
            g_tau -= &g_phi_tau;
        }
        let beta_tau = solve_h_vec(&g_tau);
        let eta_tau = &x_tau_beta + &x_eval.dot(&beta_tau);

        let w_tau_callback = directional_working_curvature_callback(self.config.link_function());
        let w_tau = match w_tau_callback(
            &pirls_result.final_eta,
            self.weights,
            &pirls_result.solve_weights,
            &eta_tau,
        ) {
            DirectionalWorkingCurvature::Diagonal(diag) => diag,
        };

        let c = &pirls_result.solve_c_array;
        let d = &pirls_result.solve_d_array;
        let c_tau = d * &eta_tau;
        let mut weighted = Array2::<f64>::zeros(x_eval.raw_dim());
        let mut h_tau = x_tau_t.t().dot(&{
            let mut wx = x_eval.clone();
            for i in 0..wx.nrows() {
                let wi = pirls_result.solve_weights[i];
                for j in 0..wx.ncols() {
                    wx[[i, j]] *= wi;
                }
            }
            wx
        }) + x_eval.t().dot(&{
            let mut wx_tau = x_tau_t.clone();
            for i in 0..wx_tau.nrows() {
                let wi = pirls_result.solve_weights[i];
                for j in 0..wx_tau.ncols() {
                    wx_tau[[i, j]] *= wi;
                }
            }
            wx_tau
        }) + Self::xt_diag_x_dense_into(&x_eval, &w_tau, &mut weighted)
            + &s_tau_t;
        if let Some(op) = firth_op.as_ref() {
            if x_tau_t.iter().any(|v| *v != 0.0) {
                h_tau -= &Self::firth_hphi_tau_partial(op, &x_tau_t, &beta_eval);
            }
            let dir_tau = Self::firth_direction(op, &beta_tau);
            h_tau -= &Self::firth_hphi_direction(op, &dir_tau);
        }

        let mut h_k = Vec::<Array2<f64>>::with_capacity(k_count);
        let firth_dirs = firth_op.as_ref().map(|op| {
            (0..k_count)
                .map(|k| Self::firth_direction(op, &b_k_mat.column(k).to_owned()))
                .collect::<Vec<_>>()
        });
        for k in 0..k_count {
            let mut hk = a_k_mats[k].clone();
            let diag_like_k = c * &u_k_mat.column(k).to_owned();
            hk += &Self::xt_diag_x_dense_into(&x_eval, &diag_like_k, &mut weighted);
            if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_dirs.as_ref()) {
                hk -= &Self::firth_hphi_direction(op, &dirs[k]);
            }
            h_k.push(hk);
        }

        // Structural penalty pseudoinverse for mixed -0.5 log|S|_+ term.
        let s_eval = rs_eval
            .iter()
            .enumerate()
            .fold(Array2::<f64>::zeros((p_dim, p_dim)), |acc, (k, r)| {
                acc + r.t().dot(r).mapv(|v| lambdas[k] * v)
            });
        let (s_eigs, s_vecs) = s_eval
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let s_max = s_eigs
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()))
            .max(1.0);
        let s_tol = (p_dim.max(1) as f64) * f64::EPSILON * s_max;
        let mut s_dag = Array2::<f64>::zeros((p_dim, p_dim));
        for i in 0..p_dim {
            let ev = s_eigs[i];
            if ev > s_tol {
                let ucol = s_vecs.column(i).to_owned();
                let scale = 1.0 / ev;
                let outer = ucol
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&ucol.view().insert_axis(Axis(0)));
                s_dag += &outer.mapv(|v| scale * v);
            }
        }

        for k in 0..k_count {
            let b_k = b_k_mat.column(k).to_owned();
            let a_k = &a_k_mats[k];
            let a_k_tau = s_tau_t.mapv(|v| lambdas[k] * v);
            let rhs_bktau = -(&h_tau.dot(&b_k) + &a_k.dot(&beta_tau) + &a_k_tau.dot(&beta_eval));
            let b_k_tau = solve_h_vec(&rhs_bktau);

            let u_k = u_k_mat.column(k).to_owned();
            let u_k_tau = x_eval.dot(&b_k_tau);
            let x_tau_bk = x_tau_t.dot(&b_k);
            let diag_q = c * &u_k;
            let diag_q_tau = &(&c_tau * &u_k) + &(c * &x_tau_bk) + &(c * &u_k_tau);
            let mut h_k_tau = a_k_tau.clone();
            h_k_tau += &x_tau_t.t().dot(&Self::row_scale(&x_eval, &diag_q));
            h_k_tau += &x_eval.t().dot(&Self::row_scale(&x_tau_t, &diag_q));
            h_k_tau += &Self::xt_diag_x_dense_into(&x_eval, &diag_q_tau, &mut weighted);
            let mut d1_trace_correction = 0.0_f64;
            let mut d2_trace_correction = 0.0_f64;
            if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_dirs.as_ref()) {
                let dir_ktau = Self::firth_direction(op, &b_k_tau);
                let dir_tau = Self::firth_direction(op, &beta_tau);
                d1_trace_correction = trace_hdag_firth_first(op, &dir_ktau);
                d2_trace_correction = trace_hdag_firth_second(op, &dirs[k], &dir_tau);
            }

            let q_mixed =
                beta_eval.dot(&a_k.dot(&beta_tau)) + 0.5 * beta_eval.dot(&a_k_tau.dot(&beta_eval));
            let t_linear = trace_hdag(&h_k_tau) - d1_trace_correction - d2_trace_correction;
            let t_quad = trace_hdag_b_hdag_c(&h_tau, &h_k[k]);
            let p_mixed = -0.5
                * (Self::trace_product(&s_dag, &a_k_tau)
                    - Self::trace_product(&s_dag.dot(&s_tau_t).dot(&s_dag), a_k));
            out[k] = q_mixed + 0.5 * (t_linear - t_quad) + p_mixed;
        }

        Ok(out)
    }

    pub(super) fn compute_tau_tau_block(
        &self,
        rho: &Array1<f64>,
        psi: &Array1<f64>,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Array2<f64>, EstimationError> {
        let psi_dim = hyper_dirs.len();
        let mut h_tt = Array2::<f64>::zeros((psi_dim, psi_dim));
        if psi_dim == 0 {
            return Ok(h_tt);
        }
        let pert_state = self.build_joint_perturbed_state(psi, hyper_dirs)?;
        let bundle = pert_state.obtain_eval_bundle(rho)?;

        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = pert_state.active_constraint_free_basis(pirls_result);

        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        let mut x_eval = pirls_result
            .x_transformed
            .to_dense_arc()
            .as_ref()
            .to_owned();
        let mut h_eff_eval = bundle.h_eff.as_ref().clone();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut x_tau_t = Vec::<Array2<f64>>::with_capacity(psi_dim);
        let mut s_tau_t = Vec::<Array2<f64>>::with_capacity(psi_dim);
        for dir in hyper_dirs {
            x_tau_t.push(dir.x_tau_original.dot(&reparam_result.qs));
            let tmp = reparam_result.qs.t().dot(&dir.s_tau_original);
            s_tau_t.push(tmp.dot(&reparam_result.qs));
        }
        if let Some(z) = free_basis_opt.as_ref() {
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            x_eval = x_eval.dot(z);
            h_eff_eval = Self::project_with_basis(bundle.h_eff.as_ref(), z);
            h_total_eval = Self::project_with_basis(bundle.h_total.as_ref(), z);
            for j in 0..psi_dim {
                x_tau_t[j] = x_tau_t[j].dot(z);
                let tmp = z.t().dot(&s_tau_t[j]);
                s_tau_t[j] = tmp.dot(z);
            }
        }

        let p_dim = beta_eval.len();
        if p_dim == 0 {
            return Ok(h_tt);
        }
        let firth_logit_active = pert_state.config.firth_bias_reduction
            && matches!(pert_state.config.link_function(), LinkFunction::Logit);
        let h_solve_eval = if firth_logit_active {
            &h_total_eval
        } else {
            &h_eff_eval
        };
        // Single-inverse doctrine for analytic tau-tau:
        // use the same positive-subspace generalized inverse used by log|H|_+.
        // This keeps all IFT solves and trace contractions on one objective-consistent surface.
        let prefer_h_pos = !bundle.active_subspace_unstable;
        let h_pos_w_for_solve = if prefer_h_pos {
            if free_basis_opt.is_none() && bundle.h_pos_factor_w.nrows() == p_dim {
                Some(bundle.h_pos_factor_w.as_ref().clone())
            } else {
                match Self::positive_part_factor_w(h_solve_eval) {
                    Ok(w) => Some(w),
                    Err(err) => {
                        log::warn!(
                            "Tau-tau analytic solve using stabilized direct fallback (failed H_+ projector build: {}).",
                            err
                        );
                        None
                    }
                }
            }
        } else {
            log::warn!(
                "Tau-tau analytic solve using stabilized direct fallback (active subspace unstable)."
            );
            None
        };
        let h_pos_w_for_solve_t = h_pos_w_for_solve.as_ref().map(|w| w.t().to_owned());
        let h_factor = if h_pos_w_for_solve.is_none() {
            Some(pert_state.factorize_faer(h_solve_eval))
        } else {
            None
        };
        let solve_h_vec = |rhs: &Array1<f64>| -> Array1<f64> {
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let tmp = w_t.dot(rhs);
                return w.dot(&tmp);
            }
            let mut rhs_mat = Array2::<f64>::zeros((rhs.len(), 1));
            rhs_mat.column_mut(0).assign(rhs);
            let mut rhs_view = array2_to_mat_mut(&mut rhs_mat);
            if let Some(f) = h_factor.as_ref() {
                f.solve_in_place(rhs_view.as_mut());
            }
            rhs_mat.column(0).to_owned()
        };
        let solve_h_mat = |rhs: &Array2<f64>| -> Array2<f64> {
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let wt_rhs = fast_ab(w_t, rhs);
                return fast_ab(w, &wt_rhs);
            }
            let mut out = rhs.clone();
            let mut out_view = array2_to_mat_mut(&mut out);
            if let Some(f) = h_factor.as_ref() {
                f.solve_in_place(out_view.as_mut());
            }
            out
        };
        let trace_hdag = |a: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let wt_a = fast_ab(w_t, a);
                let g = fast_ab(&wt_a, w);
                g.diag().sum()
            } else {
                let mut solved = a.clone();
                let mut solved_view = array2_to_mat_mut(&mut solved);
                if let Some(f) = h_factor.as_ref() {
                    f.solve_in_place(solved_view.as_mut());
                }
                solved.diag().sum()
            }
        };
        // Matrix-free D²(H_phi) trace contraction, identical to mixed block:
        // use tr(W^T (D²H_phi[u,v] W)) on the active positive solve subspace.
        let trace_hdag_firth_second =
            |op: &FirthDenseOperator, u_dir: &FirthDirection, v_dir: &FirthDirection| -> f64 {
                Self::trace_hdag_operator_apply(
                    p_dim,
                    h_pos_w_for_solve.as_ref(),
                    h_pos_w_for_solve_t.as_ref(),
                    |a| solve_h_mat(a),
                    |basis| Self::firth_hphi_second_direction_apply(op, u_dir, v_dir, basis),
                )
            };
        let trace_hdag_firth_first = |op: &FirthDenseOperator, u_dir: &FirthDirection| -> f64 {
            Self::trace_hdag_operator_apply(
                p_dim,
                h_pos_w_for_solve.as_ref(),
                h_pos_w_for_solve_t.as_ref(),
                |a| solve_h_mat(a),
                |basis| Self::firth_hphi_direction_apply(op, u_dir, basis),
            )
        };
        let trace_hdag_firth_tau_partial =
            |op: &FirthDenseOperator, x_tau: &Array2<f64>, kernel: &FirthTauPartialKernel| -> f64 {
                Self::trace_hdag_operator_apply(
                    p_dim,
                    h_pos_w_for_solve.as_ref(),
                    h_pos_w_for_solve_t.as_ref(),
                    |a| solve_h_mat(a),
                    |basis| Self::firth_hphi_tau_partial_apply(op, x_tau, kernel, basis),
                )
            };
        let trace_hdag_b_hdag_c = |b: &Array2<f64>, c_mat: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let gb = fast_ab(&fast_ab(w_t, b), w);
                let gc = fast_ab(&fast_ab(w_t, c_mat), w);
                Self::trace_product(&gb, &gc)
            } else {
                let mut left = b.clone();
                let mut right = c_mat.clone();
                let mut left_view = array2_to_mat_mut(&mut left);
                let mut right_view = array2_to_mat_mut(&mut right);
                if let Some(f) = h_factor.as_ref() {
                    f.solve_in_place(left_view.as_mut());
                    f.solve_in_place(right_view.as_mut());
                }
                Self::trace_product(&left, &right)
            }
        };

        // Build first directional solves for all tau directions.
        let u = &pirls_result.solve_weights
            * &(&pirls_result.solve_working_response - &pirls_result.final_eta);
        let c = &pirls_result.solve_c_array;
        let d = &pirls_result.solve_d_array;
        let w_diag = &pirls_result.solve_weights;
        let lambdas = rho.mapv(f64::exp);
        let lambda_sum = lambdas.sum();
        let a_tau: Vec<Array2<f64>> = s_tau_t.iter().map(|s| s.mapv(|v| lambda_sum * v)).collect();
        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                Some(Self::build_firth_dense_operator(
                    &x_eval,
                    &pirls_result.final_eta,
                )?)
            }
        } else {
            None
        };

        let mut beta_tau = vec![Array1::<f64>::zeros(p_dim); psi_dim];
        let mut eta_tau = vec![Array1::<f64>::zeros(x_eval.nrows()); psi_dim];
        let mut h_tau = vec![Array2::<f64>::zeros((p_dim, p_dim)); psi_dim];
        let mut firth_tau_dirs: Option<Vec<FirthDirection>> = firth_op
            .as_ref()
            .map(|_| Vec::<FirthDirection>::with_capacity(psi_dim));
        let mut firth_tau_kernels: Option<Vec<FirthTauPartialKernel>> = firth_op
            .as_ref()
            .map(|_| Vec::<FirthTauPartialKernel>::with_capacity(psi_dim));
        let mut x_tau_beta = vec![Array1::<f64>::zeros(x_eval.nrows()); psi_dim];
        let mut weighted = Array2::<f64>::zeros(x_eval.raw_dim());
        for j in 0..psi_dim {
            x_tau_beta[j] = x_tau_t[j].dot(&beta_eval);
            let weighted_x_tau_beta = w_diag * &x_tau_beta[j];
            let mut g_tau = x_tau_t[j].t().dot(&u)
                - x_eval.t().dot(&weighted_x_tau_beta)
                - a_tau[j].dot(&beta_eval);
            if let Some(op) = firth_op.as_ref() {
                let (g_phi_tau, _) =
                    Self::firth_partial_score_and_fit_tau(op, &x_tau_t[j], &beta_eval);
                g_tau -= &g_phi_tau;
            }
            beta_tau[j] = solve_h_vec(&g_tau);
            eta_tau[j] = &x_tau_beta[j] + &x_eval.dot(&beta_tau[j]);

            let w_tau_callback =
                directional_working_curvature_callback(pert_state.config.link_function());
            let w_tau_j = match w_tau_callback(
                &pirls_result.final_eta,
                pert_state.weights,
                &pirls_result.solve_weights,
                &eta_tau[j],
            ) {
                DirectionalWorkingCurvature::Diagonal(diag) => diag,
            };
            let mut h_j = x_tau_t[j].t().dot(&Self::row_scale(&x_eval, w_diag));
            h_j += &x_eval.t().dot(&Self::row_scale(&x_tau_t[j], w_diag));
            h_j += &Self::xt_diag_x_dense_into(&x_eval, &w_tau_j, &mut weighted);
            h_j += &a_tau[j];
            if let Some(op) = firth_op.as_ref() {
                if x_tau_t[j].iter().any(|v| *v != 0.0) {
                    let k = Self::firth_hphi_tau_partial_prepare(op, &x_tau_t[j], &beta_eval);
                    h_j -= &Self::firth_hphi_tau_partial_apply(
                        op,
                        &x_tau_t[j],
                        &k,
                        &Array2::<f64>::eye(p_dim),
                    );
                    if let Some(v) = firth_tau_kernels.as_mut() {
                        v.push(k);
                    }
                } else if let Some(v) = firth_tau_kernels.as_mut() {
                    v.push(Self::firth_hphi_tau_partial_prepare(
                        op,
                        &x_tau_t[j],
                        &beta_eval,
                    ));
                }
                let dir = Self::firth_direction(op, &beta_tau[j]);
                h_j -= &Self::firth_hphi_direction(op, &dir);
                if let Some(v) = firth_tau_dirs.as_mut() {
                    v.push(dir);
                }
            }
            h_tau[j] = h_j;
        }

        // Structural S pseudo-inverse for +0.5 tr(S^+ A_j S^+ A_i) block.
        let mut s_eval = Array2::<f64>::zeros((p_dim, p_dim));
        for (k, r_k) in reparam_result.rs_transformed.iter().enumerate() {
            let r_proj = if let Some(z) = free_basis_opt.as_ref() {
                r_k.dot(z)
            } else {
                r_k.clone()
            };
            s_eval += &r_proj.t().dot(&r_proj).mapv(|v| lambdas[k] * v);
        }
        let (s_eigs, s_vecs) = s_eval
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let s_max = s_eigs
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()))
            .max(1.0);
        let s_tol = (p_dim.max(1) as f64) * f64::EPSILON * s_max;
        let mut s_dag = Array2::<f64>::zeros((p_dim, p_dim));
        for i in 0..p_dim {
            let ev = s_eigs[i];
            if ev > s_tol {
                let ucol = s_vecs.column(i).to_owned();
                let outer = ucol
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&ucol.view().insert_axis(Axis(0)));
                s_dag += &outer.mapv(|v| v / ev);
            }
        }
        let sdag_a: Vec<Array2<f64>> = a_tau.iter().map(|a| s_dag.dot(a)).collect();

        for i in 0..psi_dim {
            for j in i..psi_dim {
                // Coupled IFT second derivative:
                //   H * beta_{tau_i,tau_j} = -(H_{tau_j} beta_{tau_i} + g_{tau_i,tau_j})
                // with
                //   g_{tau_i,tau_j} = d/dtau_j [ g_{tau_i} ].
                let u_tau_j = -(w_diag * &eta_tau[j]);
                let term1 = x_tau_t[i].t().dot(&u_tau_j);
                let term2a = x_tau_t[j].t().dot(&(w_diag * &x_tau_beta[i]));
                let term2b = x_eval.t().dot(&((&(c * &eta_tau[j])) * &x_tau_beta[i]));
                let term2c = x_eval.t().dot(&(w_diag * &x_tau_t[i].dot(&beta_tau[j])));
                let mut g_ij = term1 - term2a - term2b - term2c - a_tau[i].dot(&beta_tau[j]);
                if let (Some(op), Some(kernels)) = (firth_op.as_ref(), firth_tau_kernels.as_ref()) {
                    let mut btj = Array2::<f64>::zeros((p_dim, 1));
                    btj.column_mut(0).assign(&beta_tau[j]);
                    let gphi_ij =
                        Self::firth_hphi_tau_partial_apply(op, &x_tau_t[i], &kernels[i], &btj)
                            .column(0)
                            .to_owned();
                    g_ij -= &gphi_ij;
                }
                let rhs_ij = -(&h_tau[j].dot(&beta_tau[i]) + &g_ij);
                let beta_ij = solve_h_vec(&rhs_ij);
                let eta_ij = x_tau_t[i].dot(&beta_tau[j])
                    + x_tau_t[j].dot(&beta_tau[i])
                    + x_eval.dot(&beta_ij);

                let diag_i = c * &eta_tau[i];
                let diag_j = c * &eta_tau[j];
                let c_tau_j = d * &eta_tau[j];
                let diag_ij = &(&c_tau_j * &eta_tau[i]) + &(c * &eta_ij);

                let mut h_ij = Array2::<f64>::zeros((p_dim, p_dim));
                h_ij += &x_tau_t[i].t().dot(&Self::row_scale(&x_eval, &diag_j));
                h_ij += &x_eval.t().dot(&Self::row_scale(&x_tau_t[i], &diag_j));
                h_ij += &x_tau_t[i]
                    .t()
                    .dot(&Self::row_scale(&x_tau_t[j], &pirls_result.solve_weights));
                h_ij += &x_tau_t[j]
                    .t()
                    .dot(&Self::row_scale(&x_tau_t[i], &pirls_result.solve_weights));
                h_ij += &x_tau_t[j].t().dot(&Self::row_scale(&x_eval, &diag_i));
                h_ij += &x_eval.t().dot(&Self::row_scale(&x_tau_t[j], &diag_i));
                h_ij += &Self::xt_diag_x_dense_into(&x_eval, &diag_ij, &mut weighted);

                let mut d1_trace_correction = 0.0_f64;
                let mut d2_trace_correction = 0.0_f64;
                let mut dtau_trace_correction = 0.0_f64;
                if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_tau_dirs.as_ref()) {
                    let dir_ij = Self::firth_direction(op, &beta_ij);
                    d1_trace_correction = trace_hdag_firth_first(op, &dir_ij);
                    d2_trace_correction = trace_hdag_firth_second(op, &dirs[i], &dirs[j]);
                    if x_tau_t[i].iter().any(|v| *v != 0.0) {
                        let k_i_jtau =
                            Self::firth_hphi_tau_partial_prepare(op, &x_tau_t[i], &beta_tau[j]);
                        dtau_trace_correction +=
                            trace_hdag_firth_tau_partial(op, &x_tau_t[i], &k_i_jtau);
                    }
                    if x_tau_t[j].iter().any(|v| *v != 0.0) {
                        let k_j_itau =
                            Self::firth_hphi_tau_partial_prepare(op, &x_tau_t[j], &beta_tau[i]);
                        dtau_trace_correction +=
                            trace_hdag_firth_tau_partial(op, &x_tau_t[j], &k_j_itau);
                    }
                }

                let q = beta_eval.dot(&a_tau[i].dot(&beta_tau[j]));
                let l = 0.5
                    * (trace_hdag(&h_ij)
                        - d1_trace_correction
                        - d2_trace_correction
                        - dtau_trace_correction
                        - trace_hdag_b_hdag_c(&h_tau[j], &h_tau[i]));
                let p_term = 0.5 * Self::trace_product(&sdag_a[j], &sdag_a[i]);
                let val = q + l + p_term;
                h_tt[[i, j]] = val;
                h_tt[[j, i]] = val;
            }
        }
        for i in 0..psi_dim {
            for j in 0..i {
                let avg = 0.5 * (h_tt[[i, j]] + h_tt[[j, i]]);
                h_tt[[i, j]] = avg;
                h_tt[[j, i]] = avg;
            }
        }
        Ok(h_tt)
    }

    #[allow(dead_code)]
    pub(crate) fn compute_joint_hyper_hessian(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Array2<f64>, EstimationError> {
        let psi_dim = self.validate_joint_hyper_inputs(theta, rho_dim, hyper_dirs)?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        let psi = theta.slice(s![rho_dim..]).to_owned();
        let (h_rr, h_rtau, h_tt) = if psi_dim > 0 {
            let pert_state = self.build_joint_perturbed_state(&psi, hyper_dirs)?;
            (
                // Joint block assembly is kept fully analytic/objective-consistent.
                // Do not route through FD fallback policy here.
                pert_state.compute_laml_hessian_exact(&rho)?,
                self.compute_mixed_rho_tau_block(&rho, &psi, hyper_dirs)?,
                self.compute_tau_tau_block(&rho, &psi, hyper_dirs)?,
            )
        } else {
            (
                self.compute_laml_hessian_exact(&rho)?,
                Array2::<f64>::zeros((rho_dim, 0)),
                Array2::<f64>::zeros((0, 0)),
            )
        };
        if h_rr.nrows() != rho_dim || h_rr.ncols() != rho_dim {
            return Err(EstimationError::InvalidInput(format!(
                "rho Hessian shape mismatch in joint assembly: expected {}x{}, got {}x{}",
                rho_dim,
                rho_dim,
                h_rr.nrows(),
                h_rr.ncols()
            )));
        }
        let mut h = Array2::<f64>::zeros((rho_dim + psi_dim, rho_dim + psi_dim));
        h.slice_mut(s![..rho_dim, ..rho_dim]).assign(&h_rr);
        if psi_dim > 0 {
            h.slice_mut(s![..rho_dim, rho_dim..]).assign(&h_rtau);
            h.slice_mut(s![rho_dim.., ..rho_dim])
                .assign(&h_rtau.t().to_owned());
            h.slice_mut(s![rho_dim.., rho_dim..]).assign(&h_tt);
        }
        Ok(h)
    }

    #[allow(dead_code)]
    pub(crate) fn compute_joint_hyper_cost_gradient_hessian(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), EstimationError> {
        let (cost, grad) = self.compute_joint_hyper_cost_gradient(theta, rho_dim, hyper_dirs)?;
        let hess = self.compute_joint_hyper_hessian(theta, rho_dim, hyper_dirs)?;
        Ok((cost, grad, hess))
    }

    #[allow(dead_code)]
    pub(crate) fn compute_directional_hyper_gradient(
        &self,
        rho: &Array1<f64>,
        hyper_dir: &DirectionalHyperParam,
    ) -> Result<f64, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        self.compute_directional_hyper_gradient_with_bundle(rho, &bundle, hyper_dir)
    }

    pub(super) fn compute_directional_hyper_gradient_with_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dir: &DirectionalHyperParam,
    ) -> Result<f64, EstimationError> {
        // Exact directional hyper-gradient for one moving hyperparameter τ with
        // both X(τ) and S(τ) varying, under the same piecewise-smooth
        // fixed-active-subspace conventions used for rho-derivatives.
        //
        // Objective-level identity (inner minimization convention):
        //   V_τ
        //   = ∂_τ L*(β̂,τ)
        //     + 0.5 tr(H_+^dagger H_τ)
        //     - 0.5 tr(S_+^dagger S_τ),
        // where L* = -ℓ + 0.5 β'Sβ - Φ and H = X'WX + S - H_φ on the Firth path.
        //
        // The envelope theorem removes explicit β̂_τ terms from ∂_τ L*(β̂,τ);
        // β̂_τ appears only through total curvature drift H_τ.
        //
        // This routine evaluates all currently implemented exact pieces:
        //   - fit block: -u'X_τβ̂ + 0.5 β̂'S_τβ̂ + Φ_τ|β (Firth),
        //   - log|H|_+ trace on positive spectral subspace when available,
        //   - log|S|_+ structural penalty trace.
        // Dense Firth branch includes design-moving terms in g_τ and fit_part,
        // and evaluates H_τ on the full chain:
        //   H_τ = H_std,τ - (H_φ)_τ|β - D(H_φ)[β_τ],
        // where (H_φ)_τ|β is assembled from reduced-space exact kernels.
        if rho.len() != self.s_full_list.len() {
            return Err(EstimationError::InvalidInput(format!(
                "rho dimension mismatch for psi gradient: expected {}, got {}",
                self.s_full_list.len(),
                rho.len()
            )));
        }
        if Self::geometry_backend_kind(bundle) == GeometryBackendKind::SparseExactSpd {
            return self.compute_directional_hyper_gradient_sparse_exact(rho, bundle, hyper_dir);
        }
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);

        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let mut h_eff_eval = bundle.h_eff.as_ref().clone();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        let mut x_transformed_eval = pirls_result.x_transformed.clone();
        let mut e_eval = reparam_result.e_transformed.clone();

        let mut x_psi_t = hyper_dir.x_tau_original.dot(&reparam_result.qs);
        let mut s_psi_t = {
            let tmp = reparam_result.qs.t().dot(&hyper_dir.s_tau_original);
            tmp.dot(&reparam_result.qs)
        };

        if let Some(z) = free_basis_opt.as_ref() {
            h_eff_eval = Self::project_with_basis(bundle.h_eff.as_ref(), z);
            h_total_eval = Self::project_with_basis(bundle.h_total.as_ref(), z);
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            let x_dense_arc = pirls_result.x_transformed.to_dense_arc();
            x_transformed_eval = DesignMatrix::Dense(x_dense_arc.as_ref().dot(z));
            e_eval = reparam_result.e_transformed.dot(z);
            x_psi_t = x_psi_t.dot(z);
            s_psi_t = {
                let tmp = z.t().dot(&s_psi_t);
                tmp.dot(z)
            };
        }

        if x_psi_t.nrows() != self.y.len() || x_psi_t.ncols() != beta_eval.len() {
            return Err(EstimationError::InvalidInput(format!(
                "X_psi shape mismatch: expected {}x{}, got {}x{}",
                self.y.len(),
                beta_eval.len(),
                x_psi_t.nrows(),
                x_psi_t.ncols()
            )));
        }
        if s_psi_t.nrows() != beta_eval.len() || s_psi_t.ncols() != beta_eval.len() {
            return Err(EstimationError::InvalidInput(format!(
                "S_psi shape mismatch: expected {}x{}, got {}x{}",
                beta_eval.len(),
                beta_eval.len(),
                s_psi_t.nrows(),
                s_psi_t.ncols()
            )));
        }
        let x_dense_arc = x_transformed_eval.to_dense_arc();
        let x_dense = x_dense_arc.as_ref();
        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                Some(Self::build_firth_dense_operator(
                    x_dense,
                    &pirls_result.final_eta,
                )?)
            }
        } else {
            None
        };
        let h_solve_eval = if firth_logit_active {
            &h_total_eval
        } else {
            &h_eff_eval
        };
        // Objective-consistent generalized inverse for implicit solves:
        // when the retained positive eigenspace factor W is available in the
        // current coordinates (no free-basis projection), use
        //   H_+^dagger v = W (W' v)
        // for beta_tau and all stacked linear solves in this routine.
        //
        // This matches the same pseudo-logdet active subspace used in
        // 0.5*tr(H_+^dagger H_tau), avoiding inverse-mismatch between trace and IFT.
        let h_pos_w_for_solve =
            if free_basis_opt.is_none() && bundle.h_pos_factor_w.nrows() == h_solve_eval.nrows() {
                Some(bundle.h_pos_factor_w.as_ref().clone())
            } else {
                None
            };
        let h_pos_w_for_solve_t = h_pos_w_for_solve.as_ref().map(|w| w.t().to_owned());
        enum DirectionalSolveFactor {
            Cached(Arc<FaerFactor>),
            Local(FaerFactor),
        }
        impl DirectionalSolveFactor {
            fn solve_in_place(&self, rhs: faer::MatMut<'_, f64>) {
                match self {
                    DirectionalSolveFactor::Cached(f) => f.solve_in_place(rhs),
                    DirectionalSolveFactor::Local(f) => f.solve_in_place(rhs),
                }
            }
        }
        let h_solve_factor = if h_pos_w_for_solve.is_some() {
            None
        } else if free_basis_opt.is_none() && !firth_logit_active {
            Some(DirectionalSolveFactor::Cached(
                self.get_faer_factor(rho, h_solve_eval),
            ))
        } else {
            Some(DirectionalSolveFactor::Local(
                self.factorize_faer(h_solve_eval),
            ))
        };
        let solve_h_vec = |rhs: &Array1<f64>| -> Array1<f64> {
            if rhs.is_empty() {
                return rhs.clone();
            }
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                // Minimum-norm active-subspace solve:
                //   beta_tau = H_+^dagger rhs = W (W' rhs),  H_+^dagger = W W'.
                let tmp = w_t.dot(rhs);
                return w.dot(&tmp);
            }
            let mut rhs_mat = Array2::<f64>::zeros((rhs.len(), 1));
            rhs_mat.column_mut(0).assign(rhs);
            let mut rhs_view = array2_to_mat_mut(&mut rhs_mat);
            if let Some(f) = h_solve_factor.as_ref() {
                f.solve_in_place(rhs_view.as_mut());
            }
            rhs_mat.column(0).to_owned()
        };
        let solve_h_mat = |rhs: &Array2<f64>| -> Array2<f64> {
            if rhs.ncols() == 0 {
                return rhs.clone();
            }
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let wt_rhs = fast_ab(w_t, rhs);
                return fast_ab(w, &wt_rhs);
            }
            let mut out = rhs.clone();
            let mut out_view = array2_to_mat_mut(&mut out);
            if let Some(f) = h_solve_factor.as_ref() {
                f.solve_in_place(out_view.as_mut());
            }
            out
        };
        // Preferred exact trace evaluator for the log|H|_+ term:
        //   0.5 * tr(H_+^dagger A) = 0.5 * tr(W^T A W), H_+^dagger = W W^T.
        // Fall back to solve-surface contraction only when projected/active
        // coordinates are not aligned with the cached spectral basis.
        let h_pos_w =
            if free_basis_opt.is_none() && bundle.h_pos_factor_w.nrows() == h_solve_eval.nrows() {
                Some(bundle.h_pos_factor_w.as_ref().clone())
            } else {
                None
            };
        let h_pos_w_t = h_pos_w.as_ref().map(|w| w.t().to_owned());
        let half_trace_h_pos = |a: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_pos_w.as_ref(), h_pos_w_t.as_ref()) {
                // Derivation:
                //   d/dτ [0.5 log|H|_+] = 0.5 tr(H_+^dagger H_τ),
                //   H_+^dagger = W W^T on the retained positive eigenspace.
                // Therefore:
                //   0.5 tr(H_+^dagger A) = 0.5 tr(W^T A W).
                let wt_a = fast_ab(w_t, a);
                let g = fast_ab(&wt_a, w);
                0.5 * g.diag().sum()
            } else {
                let h_solved = solve_h_mat(a);
                0.5 * h_solved.diag().sum()
            }
        };

        let u = &pirls_result.solve_weights
            * &(&pirls_result.solve_working_response - &pirls_result.final_eta);
        // `u` is the current predictor-space score contribution g pulled through
        // the PIRLS linearization, so X_τ^T g is represented by x_psi_t^T u.
        let xpsi_beta = x_psi_t.dot(&beta_eval);
        let weighted_xpsi_beta = &pirls_result.solve_weights * &xpsi_beta;
        // Differentiate the stationarity equation
        //   0 = X^T g - S β̂
        // along the solution path β̂(τ):
        //
        //   0 = X_τ^T g + X^T(dg/dτ) - S_τ β̂ - S B,
        //   dg/dτ = -W η̇,
        //   η̇ = X_τ β̂ + X B.
        //
        // Rearranging gives the implicit solve
        //   H_solve B = X_τ^T g - X^T W(X_τ β̂) - S_τ β̂ - (g_phi)_tau|beta,
        // where:
        //   - non-Firth: H_solve = X^T W X + S,
        //   - Firth logit: H_solve = H_total = X^T W X + S - H_phi.
        // This matches the stationarity surface used by the fitted objective.
        //
        // With Firth enabled this is:
        //   g_τ = (g_std)_τ - (g_phi)_τ|_{beta fixed},
        // where (g_phi)_τ comes from reduced Fisher pseudodeterminant calculus:
        //   Phi = -0.5 log|I_r|_+, I_r = X_r^T W X_r.
        // This RHS is exactly `g_psi`, and `solve_h_vec(g_psi)` computes B
        // without ever forming H^{-1} explicitly.
        let mut g_psi =
            x_psi_t.t().dot(&u) - x_dense.t().dot(&weighted_xpsi_beta) - s_psi_t.dot(&beta_eval);
        let mut fit_firth_partial = 0.0_f64;
        if let Some(op) = firth_op.as_ref() {
            // Exact Firth design-moving partial score correction (beta held fixed):
            //   g_tau = (g_std)_tau - (g_phi)_tau,
            // where (g_phi)_tau and Phi_tau|beta are computed from reduced Fisher
            // pseudodeterminant calculus under fixed active subspace.
            //
            // Implemented formulas:
            //   (g_phi)_tau
            //   = 0.5 X_tau' (w' ⊙ h)
            //     + 0.5 X'((w'' ⊙ (X_tau β)) ⊙ h + w' ⊙ h_tau|beta),
            //   Phi_tau|beta = -0.5 tr(I_r^{-1} I_{r,tau}),
            // with I_r = X_r' W X_r and h_i = x_{r,i}' I_r^{-1} x_{r,i}.
            let (g_phi_tau, phi_tau_partial) =
                Self::firth_partial_score_and_fit_tau(op, &x_psi_t, &beta_eval);
            g_psi -= &g_phi_tau;
            fit_firth_partial = phi_tau_partial;
        }
        let beta_psi = solve_h_vec(&g_psi);
        // Total predictor drift:
        //   η̇ = d/dτ(X β̂) = X_τ β̂ + X B.
        let eta_psi = &xpsi_beta + &x_dense.dot(&beta_psi);

        // Penalized-fit block:
        //   d/dτ[-ℓ(β̂,τ) + 0.5 β̂^T S β̂]
        //     = -ℓ_β^T B - ℓ_τ + β̂^T S B + 0.5 β̂^T S_τ β̂.
        //
        // Since stationarity gives ℓ_β = S β̂, the B terms cancel exactly:
        //   -ℓ_β^T B + β̂^T S B = 0,
        // leaving only the explicit τ-dependence
        //   -ℓ_τ + 0.5 β̂^T S_τ β̂ = -g^T X_τ β̂ + 0.5 β̂^T S_τ β̂.
        // Envelope fit block:
        //   d/dtau J(beta_hat,tau) = partial_tau J | beta_hat,
        // i.e. no explicit B term.
        // For Firth-logit:
        //   Phi_tau|beta = -0.5 tr(I_r^+ I_{r,tau}),
        // added via `fit_firth_partial`.
        let fit_block =
            -u.dot(&xpsi_beta) + 0.5 * beta_eval.dot(&s_psi_t.dot(&beta_eval)) + fit_firth_partial;

        let w = &pirls_result.solve_weights;
        let mut wx = x_dense.clone();
        let mut wx_psi = x_psi_t.clone();
        for i in 0..wx.nrows() {
            let wi = w[i];
            for j in 0..wx.ncols() {
                wx[[i, j]] *= wi;
                wx_psi[[i, j]] *= wi;
            }
        }

        let mut h_psi = x_psi_t.t().dot(&wx);
        h_psi += &x_dense.t().dot(&wx_psi);

        match self.config.link_function() {
            LinkFunction::Identity => {
                // Profiled Gaussian REML:
                //   V = D_p/(2φ̂) + 0.5 log|H| - 0.5 log|S|_+ + ((n-M_p)/2) log(2πφ̂),
                //   φ̂ = D_p/(n-M_p).
                //
                // Exact profiled derivative (holding M_p locally fixed):
                //   dV/dτ = D_{p,τ}/(2φ̂)
                //         + 0.5 tr(H^{-1} H_τ)
                //         - 0.5 tr(S^+ S_τ),
                // with
                //   H_τ = X_τ^T W X + X^T W X_τ + S_τ
                // (W_τ = 0 for Gaussian identity) and
                //   D_{p,τ} = 2 * fit_block
                // by stationarity cancellation in the fit block.
                h_psi += &s_psi_t;

                let n = self.y.len() as f64;
                let (penalty_rank, _) = self
                    .fixed_subspace_penalty_rank_and_logdet(&e_eval, pirls_result.ridge_passport)?;
                let p_eff_dim = h_eff_eval.ncols();
                let mp = p_eff_dim.saturating_sub(penalty_rank) as f64;
                let dp = pirls_result.deviance + pirls_result.stable_penalty_term;
                let denom = (n - mp).max(LAML_RIDGE);
                let (dp_c, _) = smooth_floor_dp(dp);
                let phi = dp_c / denom;
                if !phi.is_finite() || phi <= 0.0 {
                    return Err(EstimationError::InvalidInput(
                        "invalid profiled Gaussian dispersion in directional hyper-gradient"
                            .to_string(),
                    ));
                }

                let trace_term = half_trace_h_pos(&h_psi);
                let pseudo_det_trace = self.fixed_subspace_penalty_trace(
                    &e_eval,
                    &s_psi_t,
                    pirls_result.ridge_passport,
                )?;
                let pseudo_det_term = -0.5 * pseudo_det_trace;
                let dp_tau = 2.0 * fit_block;
                let profiled_fit_term = dp_tau / (2.0 * phi);
                Ok(profiled_fit_term + trace_term + pseudo_det_term)
            }
            _ => {
                let w_tau_callback =
                    directional_working_curvature_callback(self.config.link_function());
                let w_direction = w_tau_callback(
                    &pirls_result.final_eta,
                    self.weights,
                    &pirls_result.solve_weights,
                    &eta_psi,
                );
                let mut x_wpsi = x_dense.clone();
                match &w_direction {
                    // Directional curvature derivative:
                    //   W_τ = T[η̇].
                    //
                    // The family-dispatched PIRLS callback returns the operator form of
                    // this derivative. Built-in links are diagonal in observation space,
                    // so the operator is represented by the per-row vector `w_direction`.
                    DirectionalWorkingCurvature::Diagonal(w_direction) => {
                        for i in 0..x_wpsi.nrows() {
                            let wpsi_i = w_direction[i];
                            for j in 0..x_wpsi.ncols() {
                                x_wpsi[[i, j]] *= wpsi_i;
                            }
                        }
                    }
                }
                // Exact total curvature derivative:
                //   H_τ = d/dτ(X^T W X + S)
                //       = X_τ^T W X + X^T W X_τ + X^T W_τ X + S_τ.
                h_psi += &x_dense.t().dot(&x_wpsi);
                h_psi += &s_psi_t;
                if let Some(op) = firth_op.as_ref() {
                    // Firth-adjusted curvature surface:
                    //   H_total = H_std - H_phi.
                    // Exact total Firth contribution in this branch:
                    //   H_{phi,tau} = (H_phi)_tau|beta + D(H_phi)[beta_tau].
                    // Therefore:
                    //   H_total,tau = H_std,tau - H_{phi,tau}
                    //               = H_std,tau
                    //                 - (H_phi)_tau|beta
                    //                 - D(H_phi)[beta_tau].
                    //
                    // We apply both terms explicitly.
                    if x_psi_t.iter().any(|v| *v != 0.0) {
                        let hphi_tau_partial =
                            Self::firth_hphi_tau_partial(op, &x_psi_t, &beta_eval);
                        h_psi -= &hphi_tau_partial;
                    }
                    let firth_dir = Self::firth_direction(op, &beta_psi);
                    h_psi -= &Self::firth_hphi_direction(op, &firth_dir);
                }

                let trace_term = half_trace_h_pos(&h_psi);
                let pseudo_det_trace = self.fixed_subspace_penalty_trace(
                    &e_eval,
                    &s_psi_t,
                    pirls_result.ridge_passport,
                )?;
                let pseudo_det_term = -0.5 * pseudo_det_trace;
                Ok(fit_block + trace_term + pseudo_det_term)
            }
        }
    }
}
