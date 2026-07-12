//! Location-scale (sigma) joint-psi evaluation: the options-aware
//! log-likelihood pass, the sigma scale-jet directional NLL, and the
//! first-/second-order sigma joint-psi terms and their directional Hessian.

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Outer-aware variant of `log_likelihood_only`. When
    /// `options.outer_score_subsample` is `None` this iterates over all rows
    /// and matches the legacy full-data implementation. When it is `Some`,
    /// only the sampled rows contribute, with their Horvitz-Thompson
    /// inverse-inclusion weights taken from `OuterScoreSubsample::rows`. Lets outer-only
    /// score/gradient passes scale to large-scale `n` without distorting the
    /// full-data inner-PIRLS or covariance code paths.
    pub(crate) fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let row_iter = outer_weighted_rows(options, self.n).to_vec();
        if flex_active {
            self.validate_exact_monotonicity(block_states)?;
            let total = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                row_iter.len(),
                |range| -> Result<f64, String> {
                    let mut ll = 0.0;
                    for idx in range {
                        let weighted = row_iter[idx];
                        ll -= weighted.weight
                            * self.row_neglog_flex_value(weighted.index, block_states)?;
                    }
                    Ok(ll)
                },
                |left, right| -> Result<f64, String> { Ok(left + right) },
            )
            .map(|opt| opt.unwrap_or(0.0));
            return total;
        }
        // True fast path: K=1 keeps the original scalar closed form; K>1
        // uses the covariance-aware vector likelihood. Rigid derivative
        // paths are derived from the canonical RowProgram expression.
        let guard = self.derivative_guard;
        let probit_scale = self.probit_frailty_scale();
        let score_dim = self.score_dim();
        gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            row_iter.len(),
            |range| -> Result<f64, String> {
                let mut ll = 0.0;
                for idx in range {
                    let weighted = row_iter[idx];
                    let i = weighted.index;
                    let q_geom = self.row_dynamic_q_values(i, block_states)?;
                    if score_dim > 1 {
                        ll -= weighted.weight
                            * self.row_neglog_rigid_vector_value(
                                i,
                                q_geom,
                                block_states,
                                probit_scale,
                            )?;
                        continue;
                    }
                    let g = block_states[2].eta[i];
                    let (nll, _, _) = row_primary_closed_form(
                        q_geom.q0,
                        q_geom.q1,
                        q_geom.qd1,
                        g,
                        self.z[[i, 0]],
                        self.weights[i],
                        self.event[i],
                        guard,
                        probit_scale,
                    )?;
                    ll -= weighted.weight * nll;
                }
                Ok(ll)
            },
            |left, right| -> Result<f64, String> { Ok(left + right) },
        )
        .map(|opt| opt.unwrap_or(0.0))
    }

    pub(crate) fn is_sigma_aux_index(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> bool {
        shared_is_sigma_aux_index(self.gaussian_frailty_sd, derivative_blocks, psi_index)
    }

    pub(crate) fn sigma_scale_jet(
        &self,
        n_dirs: usize,
        first_masks: &[usize],
        second_masks: &[usize],
    ) -> Result<MultiDirJet, String> {
        probit_frailty_scale_multi_dir_jet(
            self.gaussian_frailty_sd,
            "survival marginal-slope log-sigma auxiliary requested without GaussianShift sigma",
            n_dirs,
            first_masks,
            second_masks,
        )
    }

    pub(crate) fn row_neglog_directional_with_scale_jet(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[&Array1<f64>],
        scale_jet: &MultiDirJet,
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival marginal-slope sigma row directional expects 0..=4 directions, got {k}"
                ),
            }
            .into());
        }
        if scale_jet.coeffs.len() != (1usize << k) {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope sigma scale jet dimension mismatch: coeffs={}, dirs={k}",
                    scale_jet.coeffs.len()
                ),
            }
            .into());
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let (z_sum, covariance_ones) = self.exact_shared_score_summary(
            row,
            block_states,
            "row_neglog_directional_with_scale_jet",
        )?;
        let q_geom = self.row_dynamic_q_values(row, block_states)?;

        let first = |idx: usize| -> Vec<f64> { dirs.iter().map(|dir| dir[idx]).collect() };
        let q0_jet = MultiDirJet::linear(k, q_geom.q0, &first(0));
        let q1_jet = MultiDirJet::linear(k, q_geom.q1, &first(1));
        let qd1_jet = MultiDirJet::linear(k, q_geom.qd1, &first(2));
        let g_jet = MultiDirJet::linear(k, block_states[2].eta[row], &first(3));

        let observed_g_jet = g_jet.mul(scale_jet);
        let one_plus_b2 = MultiDirJet::constant(k, 1.0)
            .add(&observed_g_jet.mul(&observed_g_jet).scale(covariance_ones));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));

        let a0_jet = q0_jet.mul(&c_jet);
        let a1_jet = q1_jet.mul(&c_jet);
        let ad1_jet = qd1_jet.mul(&c_jet);
        let z_jet = MultiDirJet::constant(k, z_sum);
        let eta0_jet = a0_jet.add(&observed_g_jet.mul(&z_jet));
        let eta1_jet = a1_jet.add(&observed_g_jet.mul(&z_jet));

        let neg_eta0 = eta0_jet.scale(-1.0);
        let entry_term = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.coeff(0), wi))
            .scale(-1.0);

        let neg_eta1 = eta1_jet.scale(-1.0);
        let exit_term = neg_eta1.compose_unary(unary_derivatives_neglog_phi(
            neg_eta1.coeff(0),
            wi * (1.0 - di),
        ));

        let event_density_term = if di > 0.0 {
            eta1_jet
                .compose_unary(unary_derivatives_log_normal_pdf(eta1_jet.coeff(0)))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        let qd1_lower = self.time_derivative_lower_bound();
        let qd1_val = qd1_jet.coeff(0);
        let ad1_val = ad1_jet.coeff(0);
        if survival_derivative_guard_violated(qd1_val, qd1_lower) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: raw time derivative={qd1_val:.3e} must be at least derivative_guard={qd1_lower:.3e}; transformed time derivative={ad1_val:.3e}"
                ),
            }
            .into());
        }
        let time_deriv_term = if di > 0.0 {
            ad1_jet
                .compose_unary(unary_derivatives_log(ad1_val))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        Ok(exit_term
            .add(&entry_term)
            .add(&event_density_term)
            .add(&time_deriv_term)
            .coeff((1usize << k) - 1))
    }

    pub(crate) fn row_sigma_primary_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        second_sigma: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let primary_dim = N_PRIMARY;
        let zero = zero_primary_direction_ref();
        // The leading prefix is the fixed number of zero primary directions the
        // log-sigma hyperderivative differentiates *through*: one for the first
        // log-sigma derivative, two for the second. The shared
        // `directional_obj_grad_hess` sweep appends the unit primary directions
        // for grad/hess on top of this prefix. `sigma_scale_jet` only depends on
        // the multi-dir spec, not on the row, so each variant is resolved once
        // here and reused across the objective + grad + hess passes.
        let (leading, scales): (Vec<&Array1<f64>>, DirectionalScaleJets) = if second_sigma {
            (
                vec![zero, zero],
                DirectionalScaleJets {
                    obj: Some(self.sigma_scale_jet(2, &[1, 2], &[3])?),
                    grad: self.sigma_scale_jet(3, &[1, 2], &[3])?,
                    hess: self.sigma_scale_jet(4, &[1, 2], &[3])?,
                },
            )
        } else {
            (
                vec![zero],
                DirectionalScaleJets {
                    obj: Some(self.sigma_scale_jet(1, &[1], &[])?),
                    grad: self.sigma_scale_jet(2, &[1], &[])?,
                    hess: self.sigma_scale_jet(3, &[1], &[])?,
                },
            )
        };
        let terms = directional_obj_grad_hess(primary_dim, &leading, &scales, |dirs, scale| {
            self.row_neglog_directional_with_scale_jet(row, block_states, dirs, scale)
        })?;
        Ok((terms.objective, terms.grad, terms.hess))
    }

    pub(crate) fn sigma_exact_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.sigma_exact_joint_psi_terms_with_options(
            block_states,
            specs,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psi_terms`. When
    /// `options.outer_score_subsample` is `None`, iterates all rows and is
    /// bit-for-bit equivalent to the legacy implementation. When `Some`, only
    /// the sampled rows contribute and every row-summed component (objective
    /// scalar, per-block score vectors, Hessian operator blocks) is accumulated
    /// with the row's Horvitz-Thompson inverse-inclusion weight.
    pub(crate) fn sigma_exact_joint_psi_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if specs.len() != block_states.len() {
            return Err(format!(
                "survival marginal-slope sigma psi terms: specs/block_states length mismatch {} vs {}",
                specs.len(),
                block_states.len()
            ));
        }
        if self.flex_active() {
            return Err(
                "survival marginal-slope log-sigma hyperderivatives are implemented for the rigid probit marginal-slope kernel; flex score/link/timewiggle kernels still require the analytic cell-tensor sigma path"
                    .to_string(),
            );
        }
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Bit-deterministic reduction: see `chunked_row_reduction`.
        let (objective_psi, score_t, score_m, score_g, score_h, score_w, acc) =
            chunked_row_reduction(
                row_iter.as_slice(),
                || {
                    (
                        0.0,
                        Array1::zeros(p_t),
                        Array1::zeros(p_m),
                        Array1::zeros(p_g),
                        Array1::zeros(p_h),
                        Array1::zeros(p_w),
                        BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
                    )
                },
                |row, a| -> Result<(), String> {
                    let (mut obj, mut grad, mut hess) =
                        self.row_sigma_primary_terms(row, block_states, false)?;
                    let w = row_weights[row];
                    if w != 1.0 {
                        obj *= w;
                        grad.mapv_inplace(|v| v * w);
                        hess.mapv_inplace(|v| v * w);
                    }
                    a.0 += obj;
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    self.accumulate_score_with_q_geometry(
                        row, &q_geom, &grad, &mut a.1, &mut a.2, &mut a.3,
                    )?;
                    a.6.add_pullback_with_q_geometry(self, row, &q_geom, &grad, &hess)?;
                    Ok(())
                },
                |total, chunk| {
                    total.0 += chunk.0;
                    total.1 += &chunk.1;
                    total.2 += &chunk.2;
                    total.3 += &chunk.3;
                    total.4 += &chunk.4;
                    total.5 += &chunk.5;
                    total.6.add(&chunk.6);
                },
            )?;

        let mut score_psi = Array1::zeros(slices.total);
        score_psi
            .slice_mut(s![slices.time.clone()])
            .assign(&score_t);
        score_psi
            .slice_mut(s![slices.marginal.clone()])
            .assign(&score_m);
        score_psi
            .slice_mut(s![slices.logslope.clone()])
            .assign(&score_g);
        if let Some(range) = slices.score_warp.as_ref() {
            score_psi.slice_mut(s![range.clone()]).assign(&score_h);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            score_psi.slice_mut(s![range.clone()]).assign(&score_w);
        }

        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(Arc::new(acc.into_operator(slices))),
        }))
    }

    pub(crate) fn sigma_exact_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.sigma_exact_joint_psisecond_order_terms_with_options(
            block_states,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psisecond_order_terms`. See
    /// `sigma_exact_joint_psi_terms_with_options` for the row-iter / weighting
    /// contract.
    pub(crate) fn sigma_exact_joint_psisecond_order_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.flex_active() {
            return Ok(None);
        }
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Bit-deterministic reduction: see `chunked_row_reduction`.
        let (objective_psi_psi, score_t, score_m, score_g, score_h, score_w, acc) =
            chunked_row_reduction(
                row_iter.as_slice(),
                || {
                    (
                        0.0,
                        Array1::zeros(p_t),
                        Array1::zeros(p_m),
                        Array1::zeros(p_g),
                        Array1::zeros(p_h),
                        Array1::zeros(p_w),
                        BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
                    )
                },
                |row, a| -> Result<(), String> {
                    let (mut obj, mut grad, mut hess) =
                        self.row_sigma_primary_terms(row, block_states, true)?;
                    let w = row_weights[row];
                    if w != 1.0 {
                        obj *= w;
                        grad.mapv_inplace(|v| v * w);
                        hess.mapv_inplace(|v| v * w);
                    }
                    a.0 += obj;
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    self.accumulate_score_with_q_geometry(
                        row, &q_geom, &grad, &mut a.1, &mut a.2, &mut a.3,
                    )?;
                    a.6.add_pullback_with_q_geometry(self, row, &q_geom, &grad, &hess)?;
                    Ok(())
                },
                |total, chunk| {
                    total.0 += chunk.0;
                    total.1 += &chunk.1;
                    total.2 += &chunk.2;
                    total.3 += &chunk.3;
                    total.4 += &chunk.4;
                    total.5 += &chunk.5;
                    total.6.add(&chunk.6);
                },
            )?;

        let mut score_psi_psi = Array1::zeros(slices.total);
        score_psi_psi
            .slice_mut(s![slices.time.clone()])
            .assign(&score_t);
        score_psi_psi
            .slice_mut(s![slices.marginal.clone()])
            .assign(&score_m);
        score_psi_psi
            .slice_mut(s![slices.logslope.clone()])
            .assign(&score_g);
        if let Some(range) = slices.score_warp.as_ref() {
            score_psi_psi.slice_mut(s![range.clone()]).assign(&score_h);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            score_psi_psi.slice_mut(s![range.clone()]).assign(&score_w);
        }

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(acc.into_operator(slices))),
        }))
    }

    pub(crate) fn sigma_exact_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.sigma_exact_joint_psihessian_directional_derivative_with_options(
            block_states,
            d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psihessian_directional_derivative`.
    /// See `sigma_exact_joint_psi_terms_with_options` for the row-iter /
    /// weighting contract — the returned dense Hessian-derivative matrix is
    /// accumulated with per-row inverse-inclusion weights when a subsample is active.
    pub(crate) fn sigma_exact_joint_psihessian_directional_derivative_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.flex_active() {
            return Ok(None);
        }
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let primary_dim = N_PRIMARY;
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Sigma scale jets and the zero primary direction are constant
        // across rows; resolve once outside the fold instead of rebuilding
        // per-row (and per (a,b) pair) inside it. The shared
        // `directional_obj_grad_hess` sweep differentiates *through* the
        // fixed leading prefix `[zero, row_dir]` (one zero log-sigma slot,
        // the perturbation direction) and appends the grad/hess unit
        // directions; `obj: None` suppresses the zeroth-order pass.
        let scale_grad = self.sigma_scale_jet(3, &[1], &[])?;
        let scale_hess = self.sigma_scale_jet(4, &[1], &[])?;
        let zero = zero_primary_direction_ref();
        // Bit-deterministic reduction: see `chunked_row_reduction`.
        let acc = chunked_row_reduction(
            row_iter.as_slice(),
            || BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
            |row, acc| -> Result<(), String> {
                let row_dir = self.row_primary_direction_from_flat_dynamic(
                    row,
                    block_states,
                    &slices,
                    d_beta_flat,
                )?;
                let scales = DirectionalScaleJets {
                    obj: None,
                    grad: scale_grad.clone(),
                    hess: scale_hess.clone(),
                };
                let terms = directional_obj_grad_hess(
                    primary_dim,
                    &[zero, &row_dir],
                    &scales,
                    |dirs, scale| {
                        self.row_neglog_directional_with_scale_jet(row, block_states, dirs, scale)
                    },
                )?;
                let mut grad = terms.grad;
                let mut hess = terms.hess;
                let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                let w = row_weights[row];
                if w != 1.0 {
                    grad.mapv_inplace(|v| v * w);
                    hess.mapv_inplace(|v| v * w);
                }
                acc.add_pullback_with_q_geometry(self, row, &q_geom, &grad, &hess)?;
                Ok(())
            },
            |total, chunk| {
                total.add(&chunk);
            },
        )?;
        Ok(Some(acc.to_dense(&slices)))
    }
}
