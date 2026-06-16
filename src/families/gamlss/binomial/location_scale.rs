// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

impl BinomialLocationScaleFamily {
    pub const BLOCK_T: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    pub fn parameternames() -> &'static [&'static str] {
        &["threshold", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::InverseLink, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_location_scale",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    pub(crate) fn exact_joint_supported(&self) -> bool {
        self.threshold_design.is_some() && self.log_sigma_design.is_some()
    }

    pub(crate) fn dense_block_designs(
        &self,
    ) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        dense_locscale_block_designs_cached(
            self.threshold_design.as_ref(),
            self.log_sigma_design.as_ref(),
            "BinomialLocationScaleFamily",
            "BinomialLocationScale",
            "threshold",
            &self.policy.material_policy(),
        )
    }

    pub(crate) fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        dense_locscale_block_designs_fromspecs(
            specs,
            2,
            "BinomialLocationScaleFamily",
            "BinomialLocationScale",
            Self::BLOCK_T,
            Self::BLOCK_LOG_SIGMA,
            "threshold",
            &self.policy.material_policy(),
        )
    }

    pub(crate) fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        // The non-wiggle family is structurally capable of exact joint outer
        // rho-derivatives whenever the realized threshold and log-sigma
        // designs are available somewhere. Prefer cached family designs when
        // present, but allow the outer hyper code to recover the exact same
        // joint path from the realized `specs`.
        //
        // This is not a convenience fallback. The coupled profiled derivative
        // is defined in terms of the joint mode system
        //
        //   H u_k = -A_k beta,
        //
        // so if the block specs already determine the realized joint
        // curvature, forcing the code back onto a blockwise surrogate just
        // because the family did not cache duplicate dense designs would be
        // mathematically wrong.
        if self.threshold_design.is_some() && self.log_sigma_design.is_some() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    pub(crate) fn exact_joint_block_designs_owned(
        &self,
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<(DesignMatrix, DesignMatrix)>, String> {
        let designs = if let (Some(x_t), Some(x_ls)) = (
            self.threshold_design.as_ref(),
            self.log_sigma_design.as_ref(),
        ) {
            Some((x_t.clone(), x_ls.clone()))
        } else if let Some(specs) = specs {
            if specs.len() != 2 {
                return Err(GamlssError::DimensionMismatch { reason: format!(
                    "BinomialLocationScaleFamily spec-aware operator path expects 2 specs, got {}",
                    specs.len()
                ) }.into());
            }
            Some((
                specs[Self::BLOCK_T].design.clone(),
                specs[Self::BLOCK_LOG_SIGMA].design.clone(),
            ))
        } else {
            None
        };
        let Some((x_t, x_ls)) = designs else {
            return Ok(None);
        };
        let n = self.y.len();
        if x_t.nrows() != n || x_ls.nrows() != n {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily operator designs have row mismatch: y={}, threshold={}, log_sigma={}",
                n,
                x_t.nrows(),
                x_ls.nrows()
            ) }.into());
        }
        Ok(Some((x_t, x_ls)))
    }

    pub(crate) fn exact_newton_joint_gradient_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &DesignMatrix,
        x_ls: &DesignMatrix,
    ) -> Result<ExactNewtonJointGradientEvaluation, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(
                "BinomialLocationScaleFamily joint gradient input size mismatch".to_string(),
            );
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let mut grad_eta_t_v = vec![0.0_f64; n];
        let mut grad_eta_ls_v = vec![0.0_f64; n];
        let y_slice = self.y.as_slice().expect("y must be contiguous");
        let w_slice = self.weights.as_slice().expect("weights must be contiguous");
        let q0_slice = core.q0.as_slice().expect("q0 must be contiguous");
        let eta_t_slice = eta_t.as_slice().expect("eta_t must be contiguous");
        let eta_ls_slice = eta_ls.as_slice().expect("eta_ls must be contiguous");
        let link_kind = &self.link_kind;
        let gradient_pairs: Result<Vec<(f64, f64)>, String> = (0..n)
            .into_par_iter()
            .map(|i| {
                let tower = binomial_location_scale_nll_tower(
                    y_slice[i],
                    w_slice[i],
                    eta_t_slice[i],
                    eta_ls_slice[i],
                    q0_slice[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                    link_kind,
                    false,
                )?;
                Ok((-tower.g[0], -tower.g[1]))
            })
            .collect();
        for (i, (g_t, g_ls)) in gradient_pairs?.into_iter().enumerate() {
            grad_eta_t_v[i] = g_t;
            grad_eta_ls_v[i] = g_ls;
        }
        let grad_eta_t = Array1::from_vec(grad_eta_t_v);
        let grad_eta_ls = Array1::from_vec(grad_eta_ls_v);
        let grad_t = x_t.transpose_vector_multiply(&grad_eta_t);
        let grad_ls = x_ls.transpose_vector_multiply(&grad_eta_ls);
        let total = grad_t.len() + grad_ls.len();
        let mut gradient = Array1::<f64>::zeros(total);
        gradient.slice_mut(s![0..grad_t.len()]).assign(&grad_t);
        gradient.slice_mut(s![grad_t.len()..total]).assign(&grad_ls);
        Ok(ExactNewtonJointGradientEvaluation {
            log_likelihood: core.log_likelihood,
            gradient,
        })
    }

    pub(crate) fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_block_designs_owned(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_design_matrices(block_states, &x_t, &x_ls)
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    pub(crate) fn expected_joint_information_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily expected information input size mismatch"
                    .to_string(),
            }
            .into());
        }
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let rows: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let q = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
                let (f, _, _) = binomial_expected_q_information_derivatives(
                    self.weights[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                );
                (f * q.q_t * q.q_t, f * q.q_t * q.q_ls, f * q.q_ls * q.q_ls)
            })
            .collect();
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for (i, (tt, tl, ll)) in rows.into_iter().enumerate() {
            coeff_tt[i] = tt;
            coeff_tl[i] = tl;
            coeff_ll[i] = ll;
        }
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        let h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    pub(crate) fn expected_joint_information_directional_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily expected dI input size mismatch".to_string(),
            }
            .into());
        }
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expected dI direction length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let d_eta_t = fast_av(x_t, &d_beta_flat.slice(s![0..pt]));
        let d_eta_ls = fast_av(x_ls, &d_beta_flat.slice(s![pt..total]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let rows: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let q = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
                let u = nonwiggle_q_directional(q, d_eta_t[i], d_eta_ls[i]);
                let (f, f1, _) = binomial_expected_q_information_derivatives(
                    self.weights[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                );
                let tt = f1 * u.delta_q * q.q_t * q.q_t + 2.0 * f * q.q_t * u.delta_q_t;
                let tl = f1 * u.delta_q * q.q_t * q.q_ls
                    + f * (u.delta_q_t * q.q_ls + q.q_t * u.delta_q_ls);
                let ll = f1 * u.delta_q * q.q_ls * q.q_ls + 2.0 * f * q.q_ls * u.delta_q_ls;
                (tt, tl, ll)
            })
            .collect();
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for (i, (tt, tl, ll)) in rows.into_iter().enumerate() {
            coeff_tt[i] = tt;
            coeff_tl[i] = tl;
            coeff_ll[i] = ll;
        }
        let d_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut d_h = Array2::<f64>::zeros((total, total));
        d_h.slice_mut(s![0..pt, 0..pt]).assign(&d_h_tt);
        d_h.slice_mut(s![0..pt, pt..total]).assign(&d_h_tl);
        d_h.slice_mut(s![pt..total, pt..total]).assign(&d_h_ll);
        mirror_upper_to_lower(&mut d_h);
        Ok(Some(d_h))
    }

    pub(crate) fn expected_joint_information_second_directional_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily expected d2I input size mismatch".to_string(),
            }
            .into());
        }
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_u_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily expected d2I u direction length mismatch: got {}, expected {}",
                d_beta_u_flat.len(),
                total
            ) }.into());
        }
        if d_betav_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily expected d2I v direction length mismatch: got {}, expected {}",
                d_betav_flat.len(),
                total
            ) }.into());
        }
        let d_eta_t_u = fast_av(x_t, &d_beta_u_flat.slice(s![0..pt]));
        let d_eta_ls_u = fast_av(x_ls, &d_beta_u_flat.slice(s![pt..total]));
        let d_eta_t_v = fast_av(x_t, &d_betav_flat.slice(s![0..pt]));
        let d_eta_ls_v = fast_av(x_ls, &d_betav_flat.slice(s![pt..total]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let rows: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let q = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
                let (f, f1, f2) = binomial_expected_q_information_derivatives(
                    self.weights[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                );
                binomial_expected_location_scale_second_coefficients(
                    q,
                    f,
                    f1,
                    f2,
                    d_eta_t_u[i],
                    d_eta_ls_u[i],
                    d_eta_t_v[i],
                    d_eta_ls_v[i],
                )
            })
            .collect();
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for (i, (tt, tl, ll)) in rows.into_iter().enumerate() {
            coeff_tt[i] = tt;
            coeff_tl[i] = tl;
            coeff_ll[i] = ll;
        }
        let d2_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d2_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d2_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut d2_h = Array2::<f64>::zeros((total, total));
        d2_h.slice_mut(s![0..pt, 0..pt]).assign(&d2_h_tt);
        d2_h.slice_mut(s![0..pt, pt..total]).assign(&d2_h_tl);
        d2_h.slice_mut(s![pt..total, pt..total]).assign(&d2_h_ll);
        mirror_upper_to_lower(&mut d2_h);
        Ok(Some(d2_h))
    }

    pub(crate) fn expected_joint_contracted_trace_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        trace_weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily expected contracted trace input size mismatch"
                    .to_string(),
            }
            .into());
        }
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if trace_weight.dim() != (total, total) {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expected contracted trace weight shape {:?} == ({total}, {total})",
                    trace_weight.dim()
                ),
            }
            .into());
        }
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let rows: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut trace_tt = 0.0;
                for a in 0..pt {
                    for b in 0..pt {
                        trace_tt += x_t[[i, a]] * trace_weight[[a, b]] * x_t[[i, b]];
                    }
                }
                let mut trace_tl = 0.0;
                for a in 0..pt {
                    for b in 0..pls {
                        trace_tl += x_t[[i, a]]
                            * (trace_weight[[a, pt + b]] + trace_weight[[pt + b, a]])
                            * x_ls[[i, b]];
                    }
                }
                let mut trace_ll = 0.0;
                for a in 0..pls {
                    for b in 0..pls {
                        trace_ll += x_ls[[i, a]] * trace_weight[[pt + a, pt + b]] * x_ls[[i, b]];
                    }
                }
                let q = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
                let (f, f1, f2) = binomial_expected_q_information_derivatives(
                    self.weights[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                );
                let (tt_tt, tt_tl, tt_ll) = binomial_expected_location_scale_second_coefficients(
                    q, f, f1, f2, 1.0, 0.0, 1.0, 0.0,
                );
                let (tl_tt, tl_tl, tl_ll) = binomial_expected_location_scale_second_coefficients(
                    q, f, f1, f2, 1.0, 0.0, 0.0, 1.0,
                );
                let (ll_tt, ll_tl, ll_ll) = binomial_expected_location_scale_second_coefficients(
                    q, f, f1, f2, 0.0, 1.0, 0.0, 1.0,
                );
                (
                    trace_tt * tt_tt + trace_tl * tt_tl + trace_ll * tt_ll,
                    trace_tt * tl_tt + trace_tl * tl_tl + trace_ll * tl_ll,
                    trace_tt * ll_tt + trace_tl * ll_tl + trace_ll * ll_ll,
                )
            })
            .collect();
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for (i, (tt, tl, ll)) in rows.into_iter().enumerate() {
            coeff_tt[i] = tt;
            coeff_tl[i] = tl;
            coeff_ll[i] = ll;
        }
        let h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    pub(crate) fn expected_joint_information_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.expected_joint_information_from_designs(block_states, &x_t, &x_ls)
    }

    pub(crate) fn expected_joint_information_directional_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.expected_joint_information_directional_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_flat,
        )
    }

    pub(crate) fn expected_joint_information_second_directional_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.expected_joint_information_second_directional_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    pub(crate) fn expected_joint_contracted_trace_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        trace_weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.expected_joint_contracted_trace_hessian_from_designs(
            block_states,
            &x_t,
            &x_ls,
            trace_weight,
        )
    }

    pub(crate) fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            &x_t,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &x_t,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &x_t,
            &x_ls,
        )
    }

    /// Compute the rowwise joint curvature coefficients (D_tt, D_tl, D_ll)
    /// shared by the dense joint Hessian path and the matrix-free workspace.
    pub(crate) fn exact_newton_joint_hessian_row_coefficients(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let mut coeff_tt = vec![0.0_f64; n];
        let mut coeff_tl = vec![0.0_f64; n];
        let mut coeff_ll = vec![0.0_f64; n];
        let y_slice = self.y.as_slice().expect("y must be contiguous");
        let w_slice = self.weights.as_slice().expect("weights must be contiguous");
        let q0_slice = core.q0.as_slice().expect("q0 must be contiguous");
        let sigma_slice = core.sigma.as_slice().expect("sigma must be contiguous");
        let dsigma_slice = core
            .dsigma_deta
            .as_slice()
            .expect("dsigma_deta must be contiguous");
        let mu_slice = core.mu.as_slice().expect("mu must be contiguous");
        let dmu_slice = core.dmu_dq.as_slice().expect("dmu_dq must be contiguous");
        let d2mu_slice = core
            .d2mu_dq2
            .as_slice()
            .expect("d2mu_dq2 must be contiguous");
        let d3mu_slice = core
            .d3mu_dq3
            .as_slice()
            .expect("d3mu_dq3 must be contiguous");
        let link_kind = &self.link_kind;
        coeff_tt
            .par_iter_mut()
            .zip(coeff_tl.par_iter_mut())
            .zip(coeff_ll.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((c_tt, c_tl), c_ll))| {
                let q = q0_slice[i];
                let r = 1.0 / sigma_slice[i];
                let kappa = dsigma_slice[i] / sigma_slice[i];
                let (m1, m2, _) = binomial_neglog_q_derivatives_dispatch(
                    y_slice[i],
                    w_slice[i],
                    q,
                    mu_slice[i],
                    dmu_slice[i],
                    d2mu_slice[i],
                    d3mu_slice[i],
                    link_kind,
                );
                *c_tt = m2 * r * r;
                *c_tl = kappa * r * (m1 + q * m2);
                *c_ll = kappa * kappa * q * (m1 + q * m2);
            });
        Ok((
            Array1::from_vec(coeff_tt),
            Array1::from_vec(coeff_tl),
            Array1::from_vec(coeff_ll),
        ))
    }

    /// Exact diagonal-block-only Hessians (h_tt, h_ll) used by `evaluate()`
    /// to populate per-block working sets without ever materializing the
    /// dense p×p joint matrix.
    pub(crate) fn exact_newton_block_diagonal_hessians_from_design_matrices(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &DesignMatrix,
        x_ls: &DesignMatrix,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let (coeff_tt, _coeff_tl, coeff_ll) =
            self.exact_newton_joint_hessian_row_coefficients(block_states)?;
        let h_tt = xt_diag_x_design(x_t, &coeff_tt)?;
        let h_ll = xt_diag_x_design(x_ls, &coeff_ll)?;
        Ok((h_tt, h_ll))
    }

    pub(crate) fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact joint coefficient-space Hessian for the probit, non-wiggle
        // location-scale family.
        //
        // At the fitted mode, the correct joint outer smoothing sensitivity is
        //
        //   H u_k = -g_k,
        //   g_k = A_k beta,
        //
        // so the solve must use the full joint working-curvature matrix `H`.
        // For this family the likelihood is coupled through
        //
        //   q = -eta_t * exp(-eta_ls),
        //
        // so the threshold and log-sigma blocks are not independent even if
        // the penalties are block-diagonal.
        //
        // Write for row i
        //
        //   t_i = x_i^T beta_t,
        //   s_i = z_i^T beta_ls,
        //   r_i = exp(-s_i),
        //   q_i = -t_i r_i,
        //   F_i(q) = -w_i [ y_i log Phi(q) + (1-y_i) log(1-Phi(q)) ].
        //
        // Let
        //
        //   m1_i = F_i'(q_i),
        //   m2_i = F_i''(q_i).
        //
        // The q-derivatives with respect to the two predictors are
        //
        //   q_t  = -r,
        //   q_ls = -q,
        //   q_tt = 0,
        //   q_t,ls = r,
        //   q_ls,ls = q.
        //
        // For any scalar-composition objective G(t,s)=F(q(t,s)), the Hessian
        // coefficients are
        //
        //   G_ab = m2 q_a q_b + m1 q_ab.
        //
        // Therefore the exact rowwise joint curvature in (eta_t, eta_ls) is
        //
        //   coeff_tt = m2 r^2,
        //   coeff_t,ls = r (m1 + q m2),
        //   coeff_ls,ls = q (m1 + q m2),
        //
        // and the full joint coefficient-space Hessian is assembled as
        //
        //   H_tt    = X_t^T diag(coeff_tt)    X_t,
        //   H_t,ls  = X_t^T diag(coeff_t,ls)  X_ls,
        //   H_ls,ls = X_ls^T diag(coeff_ls,ls) X_ls.
        //
        // The off-diagonal block is generally nonzero. That is exactly the
        // coupling term the broken blockwise outer-gradient path was dropping.
        let (coeff_tt, coeff_tl, coeff_ll) =
            self.exact_newton_joint_hessian_row_coefficients(block_states)?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();

        let h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let total = pt + pls;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    pub(crate) fn exact_newton_joint_hessian_from_design_matrices(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &DesignMatrix,
        x_ls: &DesignMatrix,
    ) -> Result<Option<Array2<f64>>, String> {
        if let (Some(x_t_dense), Some(x_ls_dense)) = (x_t.as_dense_ref(), x_ls.as_dense_ref()) {
            return self.exact_newton_joint_hessian_from_designs(
                block_states,
                x_t_dense,
                x_ls_dense,
            );
        }
        let (coeff_tt, coeff_tl, coeff_ll) =
            self.exact_newton_joint_hessian_row_coefficients(block_states)?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();

        let h_tt = xt_diag_x_design(x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_design(x_t, &coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_design(x_ls, &coeff_ll)?;
        let total = pt + pls;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact first directional derivative D_beta H_L[u] of the joint
        // likelihood curvature.
        //
        // Write
        //
        //   t  = X_t beta_t,
        //   ls = X_ls beta_ls,
        //   s  = exp(-ls),
        //   q  = -t .* s.
        //
        // For a full coefficient-space direction
        //
        //   u = (u_t, u_ls),
        //   xi_t  = X_t u_t,
        //   xi_ls = X_ls u_ls,
        //
        // the induced q-direction is
        //
        //   alpha = D q[u] = -s .* xi_t - q .* xi_ls.
        //
        // The joint diagonal-working-curvature likelihood matrix is
        //
        //   H_L = J^T W J,
        //   J_t  = -diag(s) X_t,
        //   J_ls = -diag(q) X_ls.
        //
        // Differentiating once gives
        //
        //   D_beta H_L[u]
        //   = K[u]^T W J
        //     + J^T W K[u]
        //     + J^T diag(nu .* alpha) J,
        //
        // where
        //
        //   K_t[u]  = diag(s .* xi_ls) X_t,
        //   K_ls[u] = diag(s .* xi_t + q .* xi_ls) X_ls,
        //
        // and `nu = d'''(q)` is the third derivative of the scalar row loss.
        // This is exactly the joint curvature drift that enters the profiled
        // derivative through
        //
        //   dot H_k = A_k + D_beta H_L[u_k],
        //   dJ/drho_k
        //   = 0.5 beta^T A_k beta
        //     + 0.5 tr(H^{-1} dot H_k)
        //     - 0.5 tr(S^+ A_k).
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        if d_beta_flat.len() != pt + pls {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    pt + pls
                ),
            }
            .into());
        }
        let d_eta_t = fast_av(x_t, &d_beta_flat.slice(s![0..pt]));
        let d_eta_ls = fast_av(x_ls, &d_beta_flat.slice(s![pt..pt + pls]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let (coeff_tt, coeff_tl, coeff_ll) =
            binomial_location_scale_first_directional_coefficients(
                &self.y,
                &self.weights,
                &core,
                &d_eta_t,
                &d_eta_ls,
                &self.link_kind,
            )?;

        let d_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let total = pt + pls;
        let mut d_h = Array2::<f64>::zeros((total, total));
        d_h.slice_mut(s![0..pt, 0..pt]).assign(&d_h_tt);
        d_h.slice_mut(s![0..pt, pt..total]).assign(&d_h_tl);
        d_h.slice_mut(s![pt..total, pt..total]).assign(&d_h_ll);
        mirror_upper_to_lower(&mut d_h);
        Ok(Some(d_h))
    }

    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact mixed second directional derivative D_beta^2 H_L[u, v].
        //
        // This is the family-specific part of the total second curvature drift
        //
        //   ddot H_{k,l}
        //   = B_{k,l}
        //     + D_beta H_L[u_{k,l}]
        //     + D_beta^2 H_L[u_l, u_k],
        //
        // used in the profiled outer Hessian
        //
        //   d^2J/(drho_k drho_l)
        //   = u_l^T A_k beta
        //     + 0.5 beta^T B_{k,l} beta
        //     + 0.5 tr(H^{-1} ddot H_{k,l})
        //     - 0.5 tr(H^{-1} dot H_l H^{-1} dot H_k)
        //     - 0.5 d^2/drho_k drho_l log|S|_+.
        //
        // For directions
        //
        //   u = (u_t, u_ls),  v = (v_t, v_ls),
        //
        // define the rowwise predictor perturbations
        //
        //   xi_t^(u)  = X_t u_t,    xi_ls^(u)  = X_ls u_ls,
        //   xi_t^(v)  = X_t v_t,    xi_ls^(v)  = X_ls v_ls.
        //
        // With the exact exp sigma link,
        //
        //   s = exp(-eta_ls),
        //   q = -eta_t .* s,
        //
        // the first and second q-drifts are
        //
        //   alpha(u)   = D q[u]   = -s .* xi_t^(u) - q .* xi_ls^(u),
        //   alpha(v)   = D q[v]   = -s .* xi_t^(v) - q .* xi_ls^(v),
        //   alpha(u,v) = D^2 q[u,v]
        //              = s .* (xi_t^(u) .* xi_ls^(v) + xi_t^(v) .* xi_ls^(u))
        //                + q .* xi_ls^(u) .* xi_ls^(v).
        //
        // Differentiating the scalar-composition Hessian coefficients twice
        // yields the rowwise formulas below. Those formulas are exactly the
        // fourth-order beta-curvature contraction needed to make the joint
        // rho-Hessian path consistent with the first-order joint solve.
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_u_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily joint d_beta_u length mismatch: got {}, expected {}",
                d_beta_u_flat.len(),
                total
            ) }.into());
        }
        if d_betav_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily joint d_betav length mismatch: got {}, expected {}",
                d_betav_flat.len(),
                total
            ) }.into());
        }
        let d_eta_t_u = fast_av(x_t, &d_beta_u_flat.slice(s![0..pt]));
        let d_eta_ls_u = fast_av(x_ls, &d_beta_u_flat.slice(s![pt..total]));
        let d_eta_tv = fast_av(x_t, &d_betav_flat.slice(s![0..pt]));
        let d_eta_lsv = fast_av(x_ls, &d_betav_flat.slice(s![pt..total]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let (coeff_tt, coeff_tl, coeff_ll) =
            binomial_location_scalesecond_directional_coefficients(
                &self.y,
                &self.weights,
                &core,
                &d_eta_t_u,
                &d_eta_ls_u,
                &d_eta_tv,
                &d_eta_lsv,
                &self.link_kind,
            )?;

        let d2_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d2_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d2_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut d2_h = Array2::<f64>::zeros((total, total));
        d2_h.slice_mut(s![0..pt, 0..pt]).assign(&d2_h_tt);
        d2_h.slice_mut(s![0..pt, pt..total]).assign(&d2_h_tl);
        d2_h.slice_mut(s![pt..total, pt..total]).assign(&d2_h_ll);
        mirror_upper_to_lower(&mut d2_h);
        Ok(Some(d2_h))
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        let Some(parts) = locscale_joint_psi_direction_parts(
            block_states,
            derivative_blocks,
            psi_index,
            self.y.len(),
            x_t.ncols(),
            x_ls.ncols(),
            Self::BLOCK_T,
            Self::BLOCK_LOG_SIGMA,
            2,
            "BinomialLocationScaleFamily",
            "threshold",
            policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LocationScaleJointPsiDirection {
            block_idx: parts.block_idx,
            local_idx: parts.local_idx,
            x_primary_psi: parts.primary_psi,
            x_ls_psi: parts.log_sigma_psi,
            z_primary_psi: parts.primary_z,
            z_ls_psi: parts.log_sigma_z,
        }))
    }

    pub(crate) fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<LocationScaleJointPsiSecondDrifts, String> {
        locscale_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            LocScalePsiDriftConfig {
                n: self.y.len(),
                p_primary: x_t.ncols(),
                p_log_sigma: x_ls.ncols(),
                primary_block_idx: Self::BLOCK_T,
                log_sigma_block_idx: Self::BLOCK_LOG_SIGMA,
                family_name: "BinomialLocationScaleFamily",
                primary_label: "threshold",
                policy: &self.policy,
            },
        )
    }

    pub(crate) fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        if specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily joint psi terms expect 2 specs and 2 derivative blocks, got {} and {}",
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        // Joint fixed-beta psi terms for the coupled 2-block probit model.
        //
        // We work over the flattened coefficient vector beta = [beta_t; beta_ls]
        // and one realized spatial coordinate psi_a. The exact profiled/Laplace
        // outer calculus needs the family-side explicit objects
        //
        //   V_psi^explicit,  g_psi^explicit,  H_psi^explicit,
        //
        // all in this flattened coefficient space. These are likelihood-only
        // objects:
        //
        //   D_psi, D_{beta psi}, D_{beta beta psi}
        //
        // Generic exact-joint code adds the realized penalty motion
        //
        //   0.5 beta^T S_psi beta,  S_psi beta,  S_psi
        //
        // when forming V_i, g_i, H_i. Keeping the family hook likelihood-only
        // is what makes the unified S(theta) outer calculus correct for both
        // psi-moving designs and psi-moving penalties.
        //
        // Model:
        //   eta_t  = X_t beta_t,
        //   eta_ls = X_ls beta_ls,
        //   r      = exp(-eta_ls),
        //   q      = -eta_t .* r.
        //
        // A single realized psi_a may move either block design, so define the
        // fixed-beta predictor drifts
        //
        //   z_t  = X_{t,psi}  beta_t   (zero if psi_a is not a threshold psi)
        //   z_ls = X_{ls,psi} beta_ls  (zero if psi_a is not a log-sigma psi).
        //
        // Then the explicit q-drift is
        //
        //   q_psi = -r .* z_t - q .* z_ls.
        //
        // Rowwise scalar derivatives of the negative Bernoulli-probit loss are
        //
        //   a = dF/dq,
        //   b = d²F/dq²,
        //   c = d³F/dq³.
        //
        // Predictor-space score pieces:
        //
        //   r_t  = dF/deta_t  = -a r,
        //   r_ls = dF/deta_ls = -a q.
        //
        // Their explicit psi derivatives at fixed beta are
        //
        //   d_psi r_t  = -b q_psi r + a r z_ls,
        //   d_psi r_ls = -(a + q b) q_psi.
        //
        // Hence the exact joint score derivative is
        //
        //   g_psi
        //   = [ X_{t,psi}^T r_t  + X_t^T d_psi r_t,
        //       X_{ls,psi}^T r_ls + X_ls^T d_psi r_ls ].
        //
        // The exact envelope term is
        //
        //   V_psi^explicit = r_t^T z_t + r_ls^T z_ls.
        //
        // For the Laplace trace we also need the explicit Hessian drift. The
        // joint exact Hessian has block coefficients
        //
        //   h_tt = b r²,
        //   h_tl = r (a + q b),
        //   h_ll = q (a + q b),
        //
        // so differentiating those coefficients at fixed beta gives
        //
        //   d_psi h_tt = r² (c q_psi - 2 b z_ls),
        //   d_psi h_tl = r [ (2 b + c q) q_psi - (a + q b) z_ls ],
        //   d_psi h_ll = (a + 3 q b + q² c) q_psi.
        //
        // The full joint explicit Hessian drift is then
        //
        //   H_tt,psi
        //   = X_{t,psi}^T diag(h_tt) X_t
        //     + X_t^T diag(h_tt) X_{t,psi}
        //     + X_t^T diag(d_psi h_tt) X_t,
        //
        //   H_tl,psi
        //   = X_{t,psi}^T diag(h_tl) X_ls
        //     + X_t^T diag(h_tl) X_{ls,psi}
        //     + X_t^T diag(d_psi h_tl) X_ls,
        //
        //   H_ll,psi
        //   = X_{ls,psi}^T diag(h_ll) X_ls
        //     + X_ls^T diag(h_ll) X_{ls,psi}
        //     + X_ls^T diag(d_psi h_ll) X_ls.
        //
        // Even when only one block moves explicitly, the resulting score and
        // Hessian objects are joint because q couples eta_t and eta_ls.
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let (z_t, z_ls) = (&dir_a.z_primary_psi, &dir_a.z_ls_psi);

        // Per-row scalars assembled in parallel. The probit/inverse-link
        // derivatives are O(n) at large scale and are called O(K) times per
        // outer REML gradient (K = number of psi coords), so a parallel pass is
        // worthwhile here.
        struct PsiTermsRow {
            pub(crate) r_t: f64,
            pub(crate) r_ls: f64,
            pub(crate) dr_t: f64,
            pub(crate) dr_ls: f64,
            pub(crate) h_tt: f64,
            pub(crate) h_tl: f64,
            pub(crate) h_ll: f64,
            pub(crate) dh_tt: f64,
            pub(crate) dh_tl: f64,
            pub(crate) dh_ll: f64,
            pub(crate) obj: f64,
        }
        let y_p = self.y.as_slice().expect("y must be contiguous");
        let w_p = self.weights.as_slice().expect("weights must be contiguous");
        let q0_p = core.q0.as_slice().expect("q0 must be contiguous");
        let sigma_p = core.sigma.as_slice().expect("sigma must be contiguous");
        let dsigma_p = core
            .dsigma_deta
            .as_slice()
            .expect("dsigma_deta must be contiguous");
        let mu_p = core.mu.as_slice().expect("mu must be contiguous");
        let dmu_p = core.dmu_dq.as_slice().expect("dmu_dq must be contiguous");
        let d2mu_p = core
            .d2mu_dq2
            .as_slice()
            .expect("d2mu_dq2 must be contiguous");
        let d3mu_p = core
            .d3mu_dq3
            .as_slice()
            .expect("d3mu_dq3 must be contiguous");
        let z_t_p = z_t.as_slice().expect("z_t must be contiguous");
        let z_ls_p = z_ls.as_slice().expect("z_ls must be contiguous");
        let link_kind_p = &self.link_kind;
        let rows: Vec<PsiTermsRow> = (0..n)
            .into_par_iter()
            .map(|i| {
                let q = q0_p[i];
                let r = 1.0 / sigma_p[i];
                let s = dsigma_p[i] / sigma_p[i];
                let sz = s * z_ls_p[i];
                let q_psi = -r * z_t_p[i] - q * sz;
                let (a, b, c) = binomial_neglog_q_derivatives_dispatch(
                    y_p[i],
                    w_p[i],
                    q,
                    mu_p[i],
                    dmu_p[i],
                    d2mu_p[i],
                    d3mu_p[i],
                    link_kind_p,
                );
                let r_t = -a * r;
                let r_ls = -a * q * s;
                PsiTermsRow {
                    r_t,
                    r_ls,
                    dr_t: -b * q_psi * r + a * r * sz,
                    dr_ls: -(a + q * b) * q_psi,
                    h_tt: b * r * r,
                    h_tl: r * (a + q * b),
                    h_ll: q * (a + q * b),
                    dh_tt: r * r * (c * q_psi - 2.0 * b * sz),
                    dh_tl: r * ((2.0 * b + c * q) * q_psi - (a + q * b) * sz),
                    dh_ll: (a + 3.0 * q * b + q * q * c) * q_psi,
                    obj: r_t * z_t_p[i] + r_ls * z_ls_p[i],
                }
            })
            .collect();
        let mut r_t = Array1::<f64>::zeros(n);
        let mut r_ls = Array1::<f64>::zeros(n);
        let mut dr_t = Array1::<f64>::zeros(n);
        let mut dr_ls = Array1::<f64>::zeros(n);
        let mut h_tt = Array1::<f64>::zeros(n);
        let mut h_tl = Array1::<f64>::zeros(n);
        let mut h_ll = Array1::<f64>::zeros(n);
        let mut dh_tt = Array1::<f64>::zeros(n);
        let mut dh_tl = Array1::<f64>::zeros(n);
        let mut dh_ll = Array1::<f64>::zeros(n);
        let mut objective_psi = 0.0_f64;
        for (i, row) in rows.into_iter().enumerate() {
            r_t[i] = row.r_t;
            r_ls[i] = row.r_ls;
            dr_t[i] = row.dr_t;
            dr_ls[i] = row.dr_ls;
            h_tt[i] = row.h_tt;
            h_tl[i] = row.h_tl;
            h_ll[i] = row.h_ll;
            dh_tt[i] = row.dh_tt;
            dh_tl[i] = row.dh_tl;
            dh_ll[i] = row.dh_ll;
            objective_psi += row.obj;
        }

        let hessian_psi_operator = build_two_block_custom_family_joint_psi_operator_from_actions(
            dir_a.x_primary_psi.cloned_first_action(),
            dir_a.x_ls_psi.cloned_first_action(),
            0..pt,
            pt..pt + pls,
            x_t,
            x_ls,
            &h_tt,
            &h_tl,
            &h_ll,
            &dh_tt,
            &dh_tl,
            &dh_ll,
        )?;
        let x_t_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_t = x_t_map.transpose_mul(r_t.view()) + fast_atv(x_t, &dr_t);
        let score_ls = x_ls_map.transpose_mul(r_ls.view()) + fast_atv(x_ls, &dr_ls);
        let mut score_psi = Array1::<f64>::zeros(total);
        score_psi.slice_mut(s![0..pt]).assign(&score_t);
        score_psi.slice_mut(s![pt..pt + pls]).assign(&score_ls);
        let hessian_psi = if hessian_psi_operator.is_some() {
            Array2::zeros((0, 0))
        } else {
            let h_tt_block = weighted_crossprod_psi_maps(
                x_t_map,
                h_tt.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_t),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tt.view(),
                x_t_map,
            )? + &xt_diag_x_dense(x_t, &dh_tt)?;
            let h_tl_block = weighted_crossprod_psi_maps(
                x_t_map,
                h_tl.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tl.view(),
                x_ls_map,
            )? + &xt_diag_y_dense(x_t, &dh_tl, x_ls)?;
            let h_ll_block = weighted_crossprod_psi_maps(
                x_ls_map,
                h_ll.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                h_ll.view(),
                x_ls_map,
            )? + &xt_diag_x_dense(x_ls, &dh_ll)?;

            let mut hessian_psi = Array2::<f64>::zeros((total, total));
            hessian_psi.slice_mut(s![0..pt, 0..pt]).assign(&h_tt_block);
            hessian_psi
                .slice_mut(s![0..pt, pt..pt + pls])
                .assign(&h_tl_block);
            hessian_psi
                .slice_mut(s![pt..pt + pls, pt..pt + pls])
                .assign(&h_ll_block);
            mirror_upper_to_lower(&mut hessian_psi);
            hessian_psi
        };

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator,
        }))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_j) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_i,
                &dir_j,
                x_t,
                x_ls,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_i: &LocationScaleJointPsiDirection,
        dir_j: &LocationScaleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_i,
            dir_j,
            x_t,
            x_ls,
        )?;
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        let x_t_i_map = dir_i.x_primary_psi.as_linear_map_ref();
        let x_t_j_map = dir_j.x_primary_psi.as_linear_map_ref();
        let x_ls_i_map = dir_i.x_ls_psi.as_linear_map_ref();
        let x_ls_j_map = dir_j.x_ls_psi.as_linear_map_ref();
        let x_t_ab_map = second_psi_linear_map(
            second_drifts.x_primary_ab_action.as_ref(),
            second_drifts.x_primary_ab.as_ref(),
            n,
            pt,
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            pls,
        );

        // Exact fixed-beta psi/psi terms for the coupled non-wiggle probit
        // family.
        //
        // For two realized spatial coordinates psi_a, psi_b define
        //
        //   z_t,a  = X_{t,a} beta_t,    z_ls,a  = X_{ls,a} beta_ls,
        //   z_t,b  = X_{t,b} beta_t,    z_ls,b  = X_{ls,b} beta_ls,
        //   z_t,ab = X_{t,ab} beta_t,   z_ls,ab = X_{ls,ab} beta_ls.
        //
        // On the smooth interior branch, with r = exp(-eta_ls) and q = -eta_t r,
        //
        //   q_a  = -r z_t,a - q z_ls,a,
        //   q_b  = -r z_t,b - q z_ls,b,
        //   q_ab = -r z_t,ab
        //          + r(z_t,a z_ls,b + z_t,b z_ls,a)
        //          + q(z_ls,a z_ls,b - z_ls,ab).
        //
        // For scalar row loss derivatives
        //
        //   a = dF/dq,  b = d²F/dq²,  c = d³F/dq³,  d = d⁴F/dq⁴,
        //
        // the exact fixed-beta psi/psi objects are
        //
        //   V_ab = sum [ a q_ab + b q_a q_b ],
        //
        //   g_ab = [ X_{t,ab}^T r_t + X_{t,a}^T d_b r_t + X_{t,b}^T d_a r_t + X_t^T d_ab r_t,
        //            X_{ls,ab}^T r_ls + X_{ls,a}^T d_b r_ls + X_{ls,b}^T d_a r_ls + X_ls^T d_ab r_ls ],
        //
        // where
        //
        //   r_t  = -a r,
        //   r_ls = -a q,
        //
        //   d_a r_t  = -b q_a r + a r z_ls,a,
        //   d_a r_ls = -(a + q b) q_a,
        //
        //   d_ab r_t
        //   = r[
        //       -c q_a q_b - b q_ab
        //       + b(q_a z_ls,b + q_b z_ls,a)
        //       - a z_ls,a z_ls,b
        //       + a z_ls,ab
        //     ],
        //
        //   d_ab r_ls
        //   = -[(2b + q c) q_a q_b + (a + q b) q_ab].
        //
        // The exact Hessian psi/psi drift comes from the second derivatives of
        // the joint Hessian coefficients. In the notation of the unified outer
        // calculus, these rowwise coefficient drifts are precisely the
        // likelihood-side pieces of
        //
        //   D_{beta beta psi_a psi_b},
        //
        // before the generic assembler adds any realized-penalty contribution
        //
        //   S_ab = partial_{psi_a psi_b} S(theta).
        //
        // So this helper returns likelihood-only
        //
        //   D_ab, D_{beta ab}, D_{beta beta ab},
        //
        // and the unified exact assembler in custom_family.rs forms
        //
        //   V_ab = D_ab + 0.5 beta^T S_ab beta,
        //   g_ab = D_{beta ab} + S_ab beta,
        //   H_ab = D_{beta beta ab} + S_ab.
        //
        // Once H_ab is known, the outer assembler combines it with the joint
        // mode responses beta_a, beta_b, beta_ab and the contractions
        //
        //   T_a[beta_b], T_b[beta_a], D_beta H[beta_ab], D_beta^2 H[beta_a, beta_b]
        //
        // to form
        //
        //   ddot H_ab
        //   = H_ab + T_a[beta_b] + T_b[beta_a]
        //     + D_beta H[beta_ab] + D_beta^2 H[beta_a, beta_b].
        //
        // That is why this helper computes only the fixed-beta psi/psi object:
        // the total profiled/Laplace Hessian drift is assembled generically in
        // custom_family.rs after the joint solves.
        //
        // Concretely, the rowwise coefficient identities below are
        //
        //   h_tt = b r²,
        //   h_tl = r(a + q b),
        //   h_ll = q(a + q b),
        //
        // namely
        //
        //   d_ab h_tt
        //   = r²[
        //       d q_a q_b + c q_ab
        //       - 2c(q_b z_ls,a + q_a z_ls,b)
        //       + 4b z_ls,a z_ls,b
        //       - 2b z_ls,ab
        //     ],
        //
        //   d_ab h_tl
        //   = r[
        //       ((3c + q d) q_b) q_a
        //       + (2b + q c) q_ab
        //       - (2b + q c)(q_b z_ls,a + q_a z_ls,b)
        //       + (a + q b)(z_ls,a z_ls,b - z_ls,ab)
        //     ],
        //
        //   d_ab h_ll
        //   = (4b + 5q c + q² d) q_a q_b
        //     + (a + 3q b + q² c) q_ab.
        //
        // Differentiating X^T diag(h) X twice then gives the explicit joint
        // psi/psi Hessian blocks.
        let mut r_t = Array1::<f64>::zeros(n);
        let mut r_ls = Array1::<f64>::zeros(n);
        let mut dr_t_i = Array1::<f64>::zeros(n);
        let mut dr_t_j = Array1::<f64>::zeros(n);
        let mut dr_ls_i = Array1::<f64>::zeros(n);
        let mut dr_ls_j = Array1::<f64>::zeros(n);
        let mut d2r_t = Array1::<f64>::zeros(n);
        let mut d2r_ls = Array1::<f64>::zeros(n);
        let mut h_tt = Array1::<f64>::zeros(n);
        let mut h_tl = Array1::<f64>::zeros(n);
        let mut h_ll = Array1::<f64>::zeros(n);
        let mut dh_tt_i = Array1::<f64>::zeros(n);
        let mut dh_tt_j = Array1::<f64>::zeros(n);
        let mut dh_tl_i = Array1::<f64>::zeros(n);
        let mut dh_tl_j = Array1::<f64>::zeros(n);
        let mut dh_ll_i = Array1::<f64>::zeros(n);
        let mut dh_ll_j = Array1::<f64>::zeros(n);
        let mut d2h_tt = Array1::<f64>::zeros(n);
        let mut d2h_tl = Array1::<f64>::zeros(n);
        let mut d2h_ll = Array1::<f64>::zeros(n);
        let mut objective_psi_psi = 0.0;
        struct PsiSecondRow {
            pub(crate) r_t: f64,
            pub(crate) r_ls: f64,
            pub(crate) dr_t_i: f64,
            pub(crate) dr_t_j: f64,
            pub(crate) dr_ls_i: f64,
            pub(crate) dr_ls_j: f64,
            pub(crate) d2r_t: f64,
            pub(crate) d2r_ls: f64,
            pub(crate) h_tt: f64,
            pub(crate) h_tl: f64,
            pub(crate) h_ll: f64,
            pub(crate) dh_tt_i: f64,
            pub(crate) dh_tt_j: f64,
            pub(crate) dh_tl_i: f64,
            pub(crate) dh_tl_j: f64,
            pub(crate) dh_ll_i: f64,
            pub(crate) dh_ll_j: f64,
            pub(crate) d2h_tt: f64,
            pub(crate) d2h_tl: f64,
            pub(crate) d2h_ll: f64,
            pub(crate) objective: f64,
        }
        let y_p = self.y.as_slice().expect("y must be contiguous");
        let w_p = self.weights.as_slice().expect("weights must be contiguous");
        let q_p = core.q0.as_slice().expect("q0 must be contiguous");
        let sigma_p = core.sigma.as_slice().expect("sigma must be contiguous");
        let mu_p = core.mu.as_slice().expect("mu must be contiguous");
        let dmu_p = core.dmu_dq.as_slice().expect("dmu_dq must be contiguous");
        let d2mu_p = core
            .d2mu_dq2
            .as_slice()
            .expect("d2mu_dq2 must be contiguous");
        let d3mu_p = core
            .d3mu_dq3
            .as_slice()
            .expect("d3mu_dq3 must be contiguous");
        let z_t_i = dir_i
            .z_primary_psi
            .as_slice()
            .expect("z_t_psi_i must be contiguous");
        let z_t_j = dir_j
            .z_primary_psi
            .as_slice()
            .expect("z_t_psi_j must be contiguous");
        let z_ls_i = dir_i
            .z_ls_psi
            .as_slice()
            .expect("z_ls_psi_i must be contiguous");
        let z_ls_j = dir_j
            .z_ls_psi
            .as_slice()
            .expect("z_ls_psi_j must be contiguous");
        let z_t_ab = second_drifts
            .z_primary_ab
            .as_slice()
            .expect("z_t_ab must be contiguous");
        let z_ls_ab = second_drifts
            .z_ls_ab
            .as_slice()
            .expect("z_ls_ab must be contiguous");
        let link_kind_p = &self.link_kind;
        let rows: Result<Vec<PsiSecondRow>, String> = (0..n)
            .into_par_iter()
            .map(|row| {
                let q = q_p[row];
                let r = 1.0 / sigma_p[row];
                let q_i = -r * z_t_i[row] - q * z_ls_i[row];
                let q_j = -r * z_t_j[row] - q * z_ls_j[row];
                let q_ij = -r * z_t_ab[row]
                    + r * (z_t_i[row] * z_ls_j[row] + z_t_j[row] * z_ls_i[row])
                    + q * (z_ls_i[row] * z_ls_j[row] - z_ls_ab[row]);
                let (a, b, c) = binomial_neglog_q_derivatives_dispatch(
                    y_p[row],
                    w_p[row],
                    q,
                    mu_p[row],
                    dmu_p[row],
                    d2mu_p[row],
                    d3mu_p[row],
                    link_kind_p,
                );
                let d = binomial_neglog_q_fourth_derivative_dispatch(
                    y_p[row],
                    w_p[row],
                    q,
                    mu_p[row],
                    dmu_p[row],
                    d2mu_p[row],
                    d3mu_p[row],
                    link_kind_p,
                )?;
                let u = a + q * b;
                let u_i = (2.0 * b + q * c) * q_i;
                let u_j = (2.0 * b + q * c) * q_j;
                Ok(PsiSecondRow {
                    r_t: -a * r,
                    r_ls: -a * q,
                    dr_t_i: -b * q_i * r + a * r * z_ls_i[row],
                    dr_t_j: -b * q_j * r + a * r * z_ls_j[row],
                    dr_ls_i: -u * q_i,
                    dr_ls_j: -u * q_j,
                    d2r_t: r
                        * (-c * q_i * q_j - b * q_ij + b * (q_i * z_ls_j[row] + q_j * z_ls_i[row])
                            - a * z_ls_i[row] * z_ls_j[row]
                            + a * z_ls_ab[row]),
                    d2r_ls: -((2.0 * b + q * c) * q_i * q_j + u * q_ij),
                    h_tt: b * r * r,
                    h_tl: r * u,
                    h_ll: q * u,
                    dh_tt_i: r * r * (c * q_i - 2.0 * b * z_ls_i[row]),
                    dh_tt_j: r * r * (c * q_j - 2.0 * b * z_ls_j[row]),
                    dh_tl_i: r * (u_i - u * z_ls_i[row]),
                    dh_tl_j: r * (u_j - u * z_ls_j[row]),
                    dh_ll_i: (a + 3.0 * q * b + q * q * c) * q_i,
                    dh_ll_j: (a + 3.0 * q * b + q * q * c) * q_j,
                    d2h_tt: r
                        * r
                        * (d * q_i * q_j + c * q_ij
                            - 2.0 * c * (q_j * z_ls_i[row] + q_i * z_ls_j[row])
                            + 4.0 * b * z_ls_i[row] * z_ls_j[row]
                            - 2.0 * b * z_ls_ab[row]),
                    d2h_tl: r
                        * (((3.0 * c + q * d) * q_j) * q_i + (2.0 * b + q * c) * q_ij
                            - (2.0 * b + q * c) * (q_j * z_ls_i[row] + q_i * z_ls_j[row])
                            + u * (z_ls_i[row] * z_ls_j[row] - z_ls_ab[row])),
                    d2h_ll: (4.0 * b + 5.0 * q * c + q * q * d) * q_i * q_j
                        + (a + 3.0 * q * b + q * q * c) * q_ij,
                    objective: a * q_ij + b * q_i * q_j,
                })
            })
            .collect();
        for (row, vals) in rows?.into_iter().enumerate() {
            r_t[row] = vals.r_t;
            r_ls[row] = vals.r_ls;
            dr_t_i[row] = vals.dr_t_i;
            dr_t_j[row] = vals.dr_t_j;
            dr_ls_i[row] = vals.dr_ls_i;
            dr_ls_j[row] = vals.dr_ls_j;
            d2r_t[row] = vals.d2r_t;
            d2r_ls[row] = vals.d2r_ls;
            h_tt[row] = vals.h_tt;
            h_tl[row] = vals.h_tl;
            h_ll[row] = vals.h_ll;
            dh_tt_i[row] = vals.dh_tt_i;
            dh_tt_j[row] = vals.dh_tt_j;
            dh_tl_i[row] = vals.dh_tl_i;
            dh_tl_j[row] = vals.dh_tl_j;
            dh_ll_i[row] = vals.dh_ll_i;
            dh_ll_j[row] = vals.dh_ll_j;
            d2h_tt[row] = vals.d2h_tt;
            d2h_tl[row] = vals.d2h_tl;
            d2h_ll[row] = vals.d2h_ll;
            objective_psi_psi += vals.objective;
        }
        let mut score_psi_psi = Array1::<f64>::zeros(total);
        score_psi_psi.slice_mut(s![0..pt]).assign(
            &(x_t_ab_map.transpose_mul(r_t.view())
                + x_t_i_map.transpose_mul(dr_t_j.view())
                + x_t_j_map.transpose_mul(dr_t_i.view())
                + fast_atv(x_t, &d2r_t)),
        );
        score_psi_psi.slice_mut(s![pt..pt + pls]).assign(
            &(x_ls_ab_map.transpose_mul(r_ls.view())
                + x_ls_i_map.transpose_mul(dr_ls_j.view())
                + x_ls_j_map.transpose_mul(dr_ls_i.view())
                + fast_atv(x_ls, &d2r_ls)),
        );

        let h_tt_block = weighted_crossprod_psi_maps(
            x_t_ab_map,
            h_tt.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_t),
        )? + &weighted_crossprod_psi_maps(x_t_i_map, h_tt.view(), x_t_j_map)?
            + &weighted_crossprod_psi_maps(x_t_j_map, h_tt.view(), x_t_i_map)?
            + &weighted_crossprod_psi_maps(
                x_t_i_map,
                dh_tt_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_t),
            )?
            + &weighted_crossprod_psi_maps(
                x_t_j_map,
                dh_tt_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_t),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tt_i.view(),
                x_t_j_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tt_j.view(),
                x_t_i_map,
            )?
            + &xt_diag_x_dense(x_t, &d2h_tt)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tt.view(),
                x_t_ab_map,
            )?;
        let h_tl_block = weighted_crossprod_psi_maps(
            x_t_ab_map,
            h_tl.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(x_t_i_map, h_tl.view(), x_ls_j_map)?
            + &weighted_crossprod_psi_maps(x_t_j_map, h_tl.view(), x_ls_i_map)?
            + &weighted_crossprod_psi_maps(
                x_t_i_map,
                dh_tl_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                x_t_j_map,
                dh_tl_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tl_i.view(),
                x_ls_j_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tl_j.view(),
                x_ls_i_map,
            )?
            + &xt_diag_y_dense(x_t, &d2h_tl, x_ls)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tl.view(),
                x_ls_ab_map,
            )?;
        let h_ll_block = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            h_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(x_ls_i_map, h_ll.view(), x_ls_j_map)?
            + &weighted_crossprod_psi_maps(x_ls_j_map, h_ll.view(), x_ls_i_map)?
            + &weighted_crossprod_psi_maps(
                x_ls_i_map,
                dh_ll_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                x_ls_j_map,
                dh_ll_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                dh_ll_i.view(),
                x_ls_j_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                dh_ll_j.view(),
                x_ls_i_map,
            )?
            + &xt_diag_x_dense(x_ls, &d2h_ll)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                h_ll.view(),
                x_ls_ab_map,
            )?;

        let mut hessian_psi_psi = Array2::<f64>::zeros((total, total));
        hessian_psi_psi
            .slice_mut(s![0..pt, 0..pt])
            .assign(&h_tt_block);
        hessian_psi_psi
            .slice_mut(s![0..pt, pt..pt + pls])
            .assign(&h_tl_block);
        hessian_psi_psi
            .slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&h_ll_block);
        mirror_upper_to_lower(&mut hessian_psi_psi);

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                x_t,
                x_ls,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily joint psi hessian directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                total
            ) }.into());
        }
        let xi_t = fast_av(x_t, &d_beta_flat.slice(s![0..pt]));
        let xi_ls = fast_av(x_ls, &d_beta_flat.slice(s![pt..pt + pls]));
        let x_t_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();

        // Mixed contraction T_a[u] = D_beta H_{psi_a}[u].
        //
        // In the non-wiggle family the realized design derivatives X_{psi_a}
        // are fixed with respect to beta, so differentiating the explicit
        // Hessian drift H_{psi_a} only moves the rowwise coefficient arrays.
        // This helper therefore returns exactly the likelihood-side mixed drift
        // required by the unified outer Hessian formula
        //
        //   ddot H_{ij}
        //   = H_{ij}
        //     + T_i[beta_j]
        //     + T_j[beta_i]
        //     + D_beta H[beta_ij]
        //     + D_beta^2 H[beta_i, beta_j].
        //
        // For i = psi_a, the generic assembler supplies beta_j and any
        // realized-penalty piece S_{psi_a} itself; this family hook contributes
        // only the exact likelihood-side T_a[beta_j].
        //
        // With
        //   du   = D_beta q[u]   = -r xi_t - q xi_ls,
        //   q_a  = q_{psi_a}     = -r z_t,a - q z_ls,a,
        //   q_au = D_beta q_a[u] = r z_t,a xi_ls - du z_ls,a,
        //
        // the directional derivatives of the first-order Hessian-drift
        // coefficients are the mixed specializations of the exact psi/psi
        // formulas with z_ls,ab = 0 and q_ab = q_au:
        //
        //   D_u(d_a h_tt)
        //   = r²[
        //       d du q_a + c q_au
        //       - 2c(q_a xi_ls + du z_ls,a)
        //       + 4b xi_ls z_ls,a
        //     ],
        //
        //   D_u(d_a h_tl)
        //   = r[
        //       ((3c + q d) q_a) du
        //       + (2b + q c) q_au
        //       - (2b + q c)(q_a xi_ls + du z_ls,a)
        //       + (a + q b) xi_ls z_ls,a
        //     ],
        //
        //   D_u(d_a h_ll)
        //   = (4b + 5q c + q² d) du q_a
        //     + (a + 3q b + q² c) q_au.
        //
        // Since X_t, X_ls, X_{t,psi_a}, X_{ls,psi_a} are all beta-independent
        // here, the full matrix contraction is obtained by replacing the row
        // coefficient arrays in H_{psi_a} by their directional derivatives.
        let mut dh_tt_u = Array1::<f64>::zeros(n);
        let mut dh_tl_u = Array1::<f64>::zeros(n);
        let mut dh_ll_u = Array1::<f64>::zeros(n);
        let mut h_tt_u = Array1::<f64>::zeros(n);
        let mut h_tl_u = Array1::<f64>::zeros(n);
        let mut h_ll_u = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = core.q0[row];
            let r = 1.0 / core.sigma[row];
            let s = core.dsigma_deta[row] / core.sigma[row];
            let xi_ls_s = s * xi_ls[row];
            let z_ls_psi_s = s * dir_a.z_ls_psi[row];
            let du = -r * xi_t[row] - q * xi_ls_s;
            let q_a = -r * dir_a.z_primary_psi[row] - q * z_ls_psi_s;
            let q_au = r * dir_a.z_primary_psi[row] * xi_ls_s - du * z_ls_psi_s;
            let (a, b, c) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let d = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            let u = a + q * b;
            h_tt_u[row] = r * r * (c * du - 2.0 * b * xi_ls_s);
            h_tl_u[row] = r * ((2.0 * b + q * c) * du - u * xi_ls_s);
            h_ll_u[row] = (a + 3.0 * q * b + q * q * c) * du;
            dh_tt_u[row] = r
                * r
                * (d * du * q_a + c * q_au - 2.0 * c * (q_a * xi_ls_s + du * z_ls_psi_s)
                    + 4.0 * b * xi_ls_s * z_ls_psi_s);
            dh_tl_u[row] = r
                * (((3.0 * c + q * d) * q_a) * du + (2.0 * b + q * c) * q_au
                    - (2.0 * b + q * c) * (q_a * xi_ls_s + du * z_ls_psi_s)
                    + u * xi_ls_s * z_ls_psi_s);
            dh_ll_u[row] = (4.0 * b + 5.0 * q * c + q * q * d) * du * q_a
                + (a + 3.0 * q * b + q * q * c) * q_au;
        }

        let tt_block = weighted_crossprod_psi_maps(
            x_t_map,
            h_tt_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_t),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            h_tt_u.view(),
            x_t_map,
        )? + &xt_diag_x_dense(x_t, &dh_tt_u)?;
        let tl_block = weighted_crossprod_psi_maps(
            x_t_map,
            h_tl_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            h_tl_u.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(x_t, &dh_tl_u, x_ls)?;
        let ll_block = weighted_crossprod_psi_maps(
            x_ls_map,
            h_ll_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
            h_ll_u.view(),
            x_ls_map,
        )? + &xt_diag_x_dense(x_ls, &dh_ll_u)?;
        let mut out = Array2::<f64>::zeros((total, total));
        out.slice_mut(s![0..pt, 0..pt]).assign(&tt_block);
        out.slice_mut(s![0..pt, pt..pt + pls]).assign(&tl_block);
        out.slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&ll_block);
        mirror_upper_to_lower(&mut out);
        Ok(out)
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// The two-output map is (η_threshold, η_log_sigma):
    /// - block 0 (threshold):  output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma):  output 0 = zeros, output 1 = design rows
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::families::block_layout::block_jacobian::AdditiveWiggleBlockLayout {
            family: "BinomialLocationScaleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_T, Self::BLOCK_LOG_SIGMA],
            wiggle_block: None,
        }
        .block_effective_jacobian(specs, block_idx)
    }
}

impl CustomFamily for BinomialLocationScaleFamily {
    /// The Binomial location-scale joint Hessian depends on β because the
    /// Hessian blocks are functions of q = -t/σ and the link derivatives,
    /// all of which change when β_t or β_{log σ} move.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: matrix-free workspace applies joint Hv at
        // O(n · (p_t + p_ℓ)); only fall back to the dense build cost when
        // `use_joint_matrix_free_path` declines the operator path.
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        if !self.exact_joint_supported() {
            return Err(
                "BinomialLocationScaleFamily requires exact curvature designs; diagonal fallback has been removed"
                    .to_string(),
            );
        }
        let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleFamily exact path is missing threshold design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleFamily exact path is missing log-sigma design".to_string()
        })?;

        // Per-block gradients from the eta-space score.
        //
        //   score_q = -m1   (m1 = dF/dq, F = -ℓ)
        //   grad_eta_t[i]  = score_q * q_t
        //   grad_eta_ls[i] = score_q * q_ls
        let mut grad_eta_t_v = vec![0.0_f64; n];
        let mut grad_eta_ls_v = vec![0.0_f64; n];
        let y_slice_e = self.y.as_slice().expect("y must be contiguous");
        let w_slice_e = self.weights.as_slice().expect("weights must be contiguous");
        let q0_slice_e = core.q0.as_slice().expect("q0 must be contiguous");
        let eta_t_slice_e = eta_t.as_slice().expect("eta_t must be contiguous");
        let eta_ls_slice_e = eta_ls.as_slice().expect("eta_ls must be contiguous");
        let link_kind_e = &self.link_kind;
        let gradient_pairs: Result<Vec<(f64, f64)>, String> = (0..n)
            .into_par_iter()
            .map(|i| {
                let tower = binomial_location_scale_nll_tower(
                    y_slice_e[i],
                    w_slice_e[i],
                    eta_t_slice_e[i],
                    eta_ls_slice_e[i],
                    q0_slice_e[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                    link_kind_e,
                    false,
                )?;
                Ok((-tower.g[0], -tower.g[1]))
            })
            .collect();
        for (i, (g_t, g_ls)) in gradient_pairs?.into_iter().enumerate() {
            grad_eta_t_v[i] = g_t;
            grad_eta_ls_v[i] = g_ls;
        }
        let grad_eta_t = Array1::from_vec(grad_eta_t_v);
        let grad_eta_ls = Array1::from_vec(grad_eta_ls_v);
        let grad_t = threshold_design.transpose_vector_multiply(&grad_eta_t);
        let grad_ls = log_sigma_design.transpose_vector_multiply(&grad_eta_ls);

        // Per-block Hessians without ever materializing the full p×p joint
        // matrix — the off-diagonal cross block is unused for IRLS-style block
        // working sets and would cost O(p_t * p_ls * n) to form. The diagonal
        // blocks are computed from the same row coefficients as the joint.
        let (h_tt, h_ll) = self.exact_newton_block_diagonal_hessians_from_design_matrices(
            block_states,
            threshold_design,
            log_sigma_design,
        )?;
        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: SymmetricMatrix::Dense(h_tt),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_ls,
                    hessian: SymmetricMatrix::Dense(h_ll),
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        // Zero-allocation O(n) scalar loop — no working sets, no n-vector intermediates.
        binomial_location_scale_ll_only(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )
    }

    /// Outer-only log-likelihood with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// contribute; each row's per-row log-likelihood term is multiplied by
    /// `WeightedOuterRow.weight`, the Horvitz–Thompson inverse-inclusion
    /// factor 1/π_i (uniform or stratified sampling both supported), so the
    /// partial sum is an unbiased estimator of the full-data log-likelihood.
    /// When `None`, this returns the full-data `log_likelihood_only`. Inner
    /// PIRLS line searches never install the subsample option, so they
    /// continue to score the exact full-data log-likelihood.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        use rayon::iter::ParallelIterator;
        let link_kind = &self.link_kind;
        let ll: Result<f64, String> = subsample
            .rows
            .par_iter()
            .try_fold(
                || 0.0_f64,
                |acc, row| -> Result<f64, String> {
                    let i = row.index;
                    let wi = self.weights[i];
                    if wi == 0.0 {
                        return Ok(acc);
                    }
                    let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(eta_ls[i]);
                    let q = binomial_location_scale_q0(eta_t[i], sigma);
                    let mu = if matches!(link_kind, InverseLink::Standard(StandardLink::Probit)) {
                        0.5
                    } else {
                        let jet = inverse_link_jet_for_inverse_link(link_kind, q).map_err(|e| {
                            format!("location-scale inverse-link evaluation failed: {e}")
                        })?;
                        jet.mu
                    };
                    let term =
                        binomial_location_scale_log_likelihood(self.y[i], wi, q, link_kind, mu)?;
                    Ok(acc + row.weight * term)
                },
            )
            .try_reduce(|| 0.0_f64, |a, b| Ok(a + b));
        ll
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        Err(
            "BinomialLocationScaleFamily no longer supports diagonal working weights; exact curvature is required"
                .to_string(),
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            BinomialLocationScaleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        let pt = self
            .threshold_design
            .as_ref()
            .ok_or_else(|| {
                "BinomialLocationScaleFamily exact path is missing threshold design".to_string()
            })?
            .ncols();
        let pls = self
            .log_sigma_design
            .as_ref()
            .ok_or_else(|| {
                "BinomialLocationScaleFamily exact path is missing log-sigma design".to_string()
            })?
            .ncols();
        let total = pt + pls;
        let (start, end, joint_direction) = match block_idx {
            Self::BLOCK_T => {
                if d_beta.len() != pt {
                    return Err(GamlssError::DimensionMismatch { reason: format!(
                        "BinomialLocationScaleFamily threshold d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        pt
                    ) }.into());
                }
                let mut dir = Array1::<f64>::zeros(total);
                dir.slice_mut(s![0..pt]).assign(d_beta);
                (0usize, pt, dir)
            }
            Self::BLOCK_LOG_SIGMA => {
                if d_beta.len() != pls {
                    return Err(GamlssError::DimensionMismatch { reason: format!(
                        "BinomialLocationScaleFamily log-sigma d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        pls
                    ) }.into());
                }
                let mut dir = Array1::<f64>::zeros(total);
                dir.slice_mut(s![pt..pt + pls]).assign(d_beta);
                (pt, pt + pls, dir)
            }
            _ => return Ok(None),
        };
        let joint = self
            .exact_newton_joint_hessian_directional_derivative(block_states, &joint_direction)?
            .ok_or_else(|| {
                format!("missing joint exact-newton directional Hessian for block {block_idx}")
            })?;
        Ok(Some(joint.slice(s![start..end, start..end]).to_owned()))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn joint_jeffreys_information_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_joint_information_for_specs(block_states, Some(specs))
    }

    fn joint_jeffreys_information_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_joint_information_directional_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn joint_jeffreys_information_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_joint_information_second_directional_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn joint_jeffreys_information_contracted_trace_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_joint_contracted_trace_hessian_for_specs(block_states, Some(specs), weight)
    }

    fn joint_jeffreys_information_contracted_trace_hessian_available(&self) -> bool {
        true
    }

    fn joint_jeffreys_information_matches_observed_hessian(&self) -> bool {
        // The Jeffreys information above is the EXPECTED Fisher information,
        // not the observed Hessian: observed-Hessian conditioning certificates
        // ("Jeffreys provably skippable" matvec pre-checks) must not gate the
        // expected-information term off — for probit-class likelihoods the
        // observed information grows on saturated misclassified rows exactly
        // where the expected information collapses and the gate must arm
        // (gam#1020).
        false
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_block_designs_owned(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_gradient_from_designs(block_states, &x_t, &x_ls)
            .map(Some)
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_block_designs_owned(Some(specs))? else {
            return Ok(None);
        };
        let workspace = BinomialLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            x_t,
            x_ls,
        )?;
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-aware joint-Hessian workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_hessian_workspace`. When `Some`, the precomputed
    /// per-row coefficient arrays (`coeff_tt`, `coeff_tl`, `coeff_ll`) — which
    /// every downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) consumes row-linearly via `Xᵀ diag(W) X` — are
    /// replaced by a Horvitz–Thompson mask: each sampled row's coefficient is
    /// multiplied by `WeightedOuterRow.weight` (the inverse-inclusion factor
    /// 1/π_i; uniform or stratified sampling both supported), and non-sampled
    /// rows are zeroed. The resulting joint Hessian is an unbiased estimator
    /// of the full-data joint Hessian. Inner PIRLS never installs the option,
    /// so the inner solve continues to consume the exact full-data Hessian.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_block_designs_owned(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = BinomialLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            x_t,
            x_ls,
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// BinomialLocationScaleFamily overrides
    /// `log_likelihood_only_with_options` and
    /// `exact_newton_joint_hessian_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood and
    /// joint Hessian. The ψ-workspace path is not yet subsample-aware: it
    /// builds the exact full-data ψ Hessian blocks, which are trivially
    /// unbiased; so the outer-score components are a sum of HT-unbiased and
    /// exact-unbiased pieces and the total remains an unbiased estimator of
    /// the full-data outer score. Inner-PIRLS and final-covariance paths
    /// never install the option, so they continue to consume the exact
    /// full-data quantities.
    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // Representation support means the realized two-block designs can be
        // applied as β-space operators. It does not imply that exact
        // second-order outer θ work is cheap.
        if specs.len() != 2 {
            return false;
        }
        let n = self.y.len();
        specs[Self::BLOCK_T].design.nrows() == n && specs[Self::BLOCK_LOG_SIGMA].design.nrows() == n
    }
}

impl CustomFamilyGenerative for BinomialLocationScaleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily generative size mismatch".to_string(),
            }
            .into());
        }
        let mean = gamlss_rowwise_map_result(self.y.len(), |i| {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            let q = binomial_location_scale_q0(eta_t[i], sigma);
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
                .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
            Ok(jet.mu)
        })?;
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}
