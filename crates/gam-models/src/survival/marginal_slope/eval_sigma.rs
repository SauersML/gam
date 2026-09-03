//! Location-scale (sigma) joint-psi evaluation: the options-aware
//! log-likelihood pass, the sigma scale-jet directional NLL, and the
//! first-/second-order sigma joint-psi terms and their directional Hessian.

use super::*;

struct CompiledSigmaPrimaryTerms {
    objective: f64,
    grad: [f64; 4],
    hess: [[f64; 4]; 4],
}

/// Direct compiler lowering of the canonical rigid feature program through the
/// nonlinear observed-slope map
///
/// `b = scale * g`, `linear = z * b`, `variance = covariance * b²`.
///
/// The row macro owns every feature derivative. This function is only the
/// universal chain rule from that five-feature compiler surface back to
/// `(q0,q1,qd1,g,log_sigma)`: order two plus one contracted third supplies the
/// first auxiliary bundle; the second adds the curvature direction and one
/// contracted fourth. No jet carrier or likelihood algebra is reconstructed.
#[inline]
fn compiled_sigma_primary_terms(
    primaries: [f64; 4],
    scale: crate::survival::lognormal_kernel::ProbitFrailtyScaleJet,
    inputs: &RigidRowInputs,
    second: bool,
) -> Result<CompiledSigmaPrimaryTerms, String> {
    let [q0, q1, qd1, g] = primaries;
    let b = scale.s * g;
    let linear = inputs.z_sum * b;
    let variance = inputs.covariance_ones * b * b;
    // The frailty scale is folded into `b`, so the row program is asked for the
    // unit-scale surface (`probit_scale = 1.0`) and every derivative below is
    // taken with respect to `b` directly.
    let features = static_slope_feature_frame(q0, q1, qd1, linear, variance, 0.0);
    let (_, feature_gradient, feature_hessian, witnesses) =
        rigid_feature_frame_order2(&features, inputs.wi, inputs.di, 1.0, follow_up_varying_flag::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>());
    validate_rigid_row_admission::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
        qd1,
        inputs,
        witnesses[0],
        witnesses[1],
        witnesses[2],
    )?;

    // `∂features/∂b` and `∂²features/∂b²` of the static-slope frame. Both
    // location channels move with `z`, both variance channels with `2·cov·b`,
    // and the three rate channels are identically zero because this geometry's
    // slope does not move along follow-up.
    let tangent = static_slope_feature_frame(
        0.0,
        0.0,
        0.0,
        inputs.z_sum,
        2.0 * inputs.covariance_ones * b,
        0.0,
    );
    let curvature =
        static_slope_feature_frame(0.0, 0.0, 0.0, 0.0, 2.0 * inputs.covariance_ones, 0.0);
    let third_tangent =
        rigid_feature_frame_third_contracted(
            &features,
            inputs.wi,
            inputs.di,
            1.0,
            follow_up_varying_flag::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(),
            &tangent,
        );

    let dot = |left: &[f64; RIGID_FEATURE_DIMENSION], right: &[f64; RIGID_FEATURE_DIMENSION]| {
        let mut total = 0.0;
        for axis in 0..RIGID_FEATURE_DIMENSION {
            total += left[axis] * right[axis];
        }
        total
    };
    let matrix_direction =
        |matrix: &[[f64; RIGID_FEATURE_DIMENSION]; RIGID_FEATURE_DIMENSION],
         row: usize,
         direction: &[f64; RIGID_FEATURE_DIMENSION]| { dot(&matrix[row], direction) };

    let f_b = dot(&feature_gradient, &tangent);
    let h_times_tangent: [f64; RIGID_FEATURE_DIMENSION] =
        std::array::from_fn(|axis| matrix_direction(&feature_hessian, axis, &tangent));
    let f_bb = dot(&h_times_tangent, &tangent) + dot(&feature_gradient, &curvature);
    let f_qb: [f64; 3] = std::array::from_fn(|axis| h_times_tangent[axis]);
    let f_qqb: [[f64; 3]; 3] =
        std::array::from_fn(|axis| std::array::from_fn(|other| third_tangent[axis][other]));
    let f_qbb: [f64; 3] = std::array::from_fn(|axis| {
        matrix_direction(&third_tangent, axis, &tangent)
            + matrix_direction(&feature_hessian, axis, &curvature)
    });
    let f_bbb = dot(
        &std::array::from_fn(|axis| matrix_direction(&third_tangent, axis, &tangent)),
        &tangent,
    ) + 3.0 * dot(&h_times_tangent, &curvature);

    let bt = g * scale.ds;
    if !second {
        let objective = f_b * bt;
        let grad = std::array::from_fn(|axis| {
            if axis < 3 {
                f_qb[axis] * bt
            } else {
                f_bb * scale.s * bt + f_b * scale.ds
            }
        });
        let hess = std::array::from_fn(|axis| {
            std::array::from_fn(|other| match (axis == 3, other == 3) {
                (false, false) => f_qqb[axis][other] * bt,
                (true, true) => f_bbb * scale.s * scale.s * bt + 2.0 * f_bb * scale.s * scale.ds,
                _ => {
                    let primary = if axis == 3 { other } else { axis };
                    f_qbb[primary] * scale.s * bt + f_qb[primary] * scale.ds
                }
            })
        });
        return Ok(CompiledSigmaPrimaryTerms {
            objective,
            grad,
            hess,
        });
    }

    let third_curvature =
        rigid_feature_frame_third_contracted(
            &features,
            inputs.wi,
            inputs.di,
            1.0,
            follow_up_varying_flag::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(),
            &curvature,
        );
    let fourth_tangent = rigid_feature_frame_fourth_contracted(
        &features,
        inputs.wi,
        inputs.di,
        1.0,
        follow_up_varying_flag::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(),
        &tangent,
        &tangent,
    );
    let f_qqbb: [[f64; 3]; 3] = std::array::from_fn(|axis| {
        std::array::from_fn(|other| fourth_tangent[axis][other] + third_curvature[axis][other])
    });
    let f_qbbb: [f64; 3] = std::array::from_fn(|axis| {
        matrix_direction(&fourth_tangent, axis, &tangent)
            + 3.0 * matrix_direction(&third_tangent, axis, &curvature)
    });
    let f_bbbb = dot(
        &std::array::from_fn(|axis| matrix_direction(&fourth_tangent, axis, &tangent)),
        &tangent,
    ) + 6.0
        * dot(
            &std::array::from_fn(|axis| matrix_direction(&third_tangent, axis, &tangent)),
            &curvature,
        )
        + 3.0
            * dot(
                &std::array::from_fn(|axis| matrix_direction(&feature_hessian, axis, &curvature)),
                &curvature,
            );

    let btt = g * scale.d2s;
    let objective = f_bb * bt * bt + f_b * btt;
    let grad = std::array::from_fn(|axis| {
        if axis < 3 {
            f_qbb[axis] * bt * bt + f_qb[axis] * btt
        } else {
            f_bbb * bt * bt * scale.s
                + f_bb * (btt * scale.s + 2.0 * bt * scale.ds)
                + f_b * scale.d2s
        }
    });
    let hess = std::array::from_fn(|axis| {
        std::array::from_fn(|other| match (axis == 3, other == 3) {
            (false, false) => f_qqbb[axis][other] * bt * bt + f_qqb[axis][other] * btt,
            (true, true) => {
                f_bbbb * bt * bt * scale.s * scale.s
                    + f_bbb * (btt * scale.s * scale.s + 4.0 * bt * scale.s * scale.ds)
                    + f_bb * (2.0 * scale.ds * scale.ds + 2.0 * scale.s * scale.d2s)
            }
            _ => {
                let primary = if axis == 3 { other } else { axis };
                f_qbbb[primary] * bt * bt * scale.s
                    + f_qbb[primary] * (btt * scale.s + 2.0 * bt * scale.ds)
                    + f_qb[primary] * scale.d2s
            }
        })
    });
    Ok(CompiledSigmaPrimaryTerms {
        objective,
        grad,
        hess,
    })
}

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
        // True fast path: K=1 uses the packed lowering of the canonical row
        // program; K>1 uses the covariance-aware vector likelihood.
        let guard = self.derivative_guard;
        let probit_scale = self.probit_frailty_scale();
        let score_dim = self.score_dim();
        gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            row_iter.len(),
            |range| -> Result<f64, String> {
                let mut ll = 0.0;
                let mut slope_workspace = self.slope_row_workspace()?;
                let value_workspace = if score_dim > 1 {
                    Some(RigidVectorValueWorkspace::new(&self.score_covariance))
                } else {
                    None
                };
                for idx in range {
                    let weighted = row_iter[idx];
                    let i = weighted.index;
                    // The value the trust region scores a trial on must be the
                    // value of the frame whose gradient and Hessian proposed
                    // the step (`compute_row_primary_gradient_hessian_uncached`
                    // dispatches on the same flag). The time-constant closed
                    // form below reads ONE slope value and a zero rate; on the
                    // follow-up-varying frame the row's `η′₁` carries
                    // `q·c′ + b′ᵀz`, so that form is a different likelihood,
                    // one in which the slope's variation is invisible — which
                    // is what "the slope can't vary" looks like from outside
                    // (gam#2765).
                    if self.slope_is_follow_up_varying() {
                        let inputs = rigid_row_inputs(
                            self,
                            block_states,
                            i,
                            "survival marginal-slope value-only row",
                        )?;
                        let primaries = rigid_row_kernel_primaries::<
                            DYNAMIC_SLOPE_PRIMARIES,
                            DynamicSlopeGeometry,
                        >(self, block_states, i)?;
                        let (nll, _, _) = rigid_row_order2::<
                            DYNAMIC_SLOPE_PRIMARIES,
                            DynamicSlopeGeometry,
                        >(&primaries, &inputs)?;
                        ll -= weighted.weight * nll;
                        continue;
                    }
                    let q_geom = self.row_dynamic_q_values(i, block_states)?;
                    if score_dim > 1 {
                        ll -= weighted.weight
                            * self.row_neglog_rigid_vector_value(
                                i,
                                q_geom,
                                block_states,
                                probit_scale,
                                &mut slope_workspace,
                                value_workspace.as_ref().expect(
                                    "vector value workspace is constructed for multi-score rows",
                                ),
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

    fn sigma_scale_derivatives(
        &self,
    ) -> Result<crate::survival::lognormal_kernel::ProbitFrailtyScaleJet, String> {
        let sigma = self.gaussian_frailty_sd.ok_or_else(|| {
            "survival marginal-slope log-sigma auxiliary requested without GaussianShift sigma"
                .to_string()
        })?;
        Ok(crate::survival::lognormal_kernel::ProbitFrailtyScaleJet::from_log_sigma(sigma.ln()))
    }

    /// Evaluate the canonical rigid row program with its observed slope already
    /// lifted through a jet-valued frailty scale. Passing `probit_scale = 1`
    /// prevents a second scaling inside [`rigid_row_nll`]; probability tails,
    /// event semantics, and monotonicity remain owned by that single source.
    fn row_neglog_canonical_scale_jet<S: gam_math::jet_scalar::JetScalar<N_PRIMARY>>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primaries: &[S; N_PRIMARY],
        scale: &S,
    ) -> Result<S, String> {
        let mut inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope sigma canonical row program",
        )?;
        inputs.probit_scale = 1.0;
        let observed_primaries = [
            primaries[0],
            primaries[1],
            primaries[2],
            primaries[3].mul(scale),
        ];
        rigid_row_nll::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry, _>(
            &observed_primaries,
            &inputs,
        )
    }

    fn row_sigma_primary_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        second_sigma: bool,
    ) -> Result<CompiledSigmaPrimaryTerms, String> {
        let primaries = rigid_row_kernel_primaries::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
            self,
            block_states,
            row,
        )?;
        let scale = self.sigma_scale_derivatives()?;
        let mut inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope sigma compiled row program",
        )?;
        inputs.probit_scale = 1.0;
        compiled_sigma_primary_terms(primaries, scale, &inputs, second_sigma)
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
        let p_g = slices.slope.len();
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
                    let mut terms = self.row_sigma_primary_terms(row, block_states, false)?;
                    let w = row_weights[row];
                    if w != 1.0 {
                        terms.objective *= w;
                        for axis in 0..4 {
                            terms.grad[axis] *= w;
                            for other in 0..4 {
                                terms.hess[axis][other] *= w;
                            }
                        }
                    }
                    a.0 += terms.objective;
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let grad = ndarray::ArrayView1::from(&terms.grad);
                    let hess = ndarray::ArrayView2::from(&terms.hess);
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
            .slice_mut(s![slices.slope.clone()])
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
        let p_g = slices.slope.len();
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
                    let mut terms = self.row_sigma_primary_terms(row, block_states, true)?;
                    let w = row_weights[row];
                    if w != 1.0 {
                        terms.objective *= w;
                        for axis in 0..4 {
                            terms.grad[axis] *= w;
                            for other in 0..4 {
                                terms.hess[axis][other] *= w;
                            }
                        }
                    }
                    a.0 += terms.objective;
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let grad = ndarray::ArrayView1::from(&terms.grad);
                    let hess = ndarray::ArrayView2::from(&terms.hess);
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
            .slice_mut(s![slices.slope.clone()])
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
            hessian_psi_psi_operator: Some(Arc::new(acc.into_operator(slices))),
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
        let p_g = slices.slope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // The frailty-scale stack is common to every row. One TwoSeed row
        // evaluation carries both its sigma direction and the requested
        // coefficient-space direction; its mixed Order2 channel supplies the
        // complete primary gradient and Hessian in one pass.
        let scale = self.sigma_scale_derivatives()?;
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
                let primaries = rigid_row_kernel_primaries::<
                    STATIC_SLOPE_PRIMARIES,
                    StaticSlopeGeometry,
                >(self, block_states, row)?;
                let direction = std::array::from_fn(|axis| row_dir[axis]);
                let terms = first_parameter_directional_order2_terms(
                    primaries,
                    &direction,
                    scale.s,
                    scale.ds,
                    |variables, parameter| {
                        self.row_neglog_canonical_scale_jet(row, block_states, variables, parameter)
                    },
                )?;
                let mut grad = terms.grad;
                let mut hess = terms.hess;
                let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                let w = row_weights[row];
                if w != 1.0 {
                    grad.mapv_inplace(|value| value * w);
                    hess.mapv_inplace(|value| value * w);
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

#[cfg(test)]
mod sigma_parameter_jet_release_tests {
    use super::*;
    use crate::survival::lognormal_kernel::ProbitFrailtyScaleJet;
    use gam_math::jet_scalar::JetScalar;
    use gam_math::jet_tower::{Tower3, Tower4};
    use gam_math::nested_dual::Dual2;
    use gam_math::paired_timing::{SpeedGate, paired_interleaved};

    // One synthetic interior row: finite signed margins and a strictly-positive
    // raw time derivative so the monotonicity guard admits. `probit_scale = 1.0`
    // mirrors `row_neglog_canonical_scale_jet`, which folds the frailty scale
    // into the observed slope primary rather than a second in-kernel scaling.
    fn synthetic_inputs(wi: f64, di: f64, z_sum: f64) -> RigidRowInputs {
        RigidRowInputs {
            row: 0,
            wi,
            di,
            z_sum,
            covariance_ones: 1.0,
            probit_scale: 1.0,
            qd1_lower: 0.0,
        }
    }

    // The exact expression `row_neglog_canonical_scale_jet` evaluates: observe
    // the slope primary through the frailty scale, then the sole rigid row NLL.
    // Generic over the jet scalar so the production seeded `OneSeed`/`TwoSeed`
    // instantiations and the dense-tower racer run bit-for-bit the same program.
    fn eval_scaled<S: JetScalar<4>>(
        primaries: &[S; 4],
        scale: &S,
        inputs: &RigidRowInputs,
    ) -> Result<S, String> {
        let observed = [
            primaries[0],
            primaries[1],
            primaries[2],
            primaries[3].mul(scale),
        ];
        rigid_row_nll::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry, _>(&observed, inputs)
    }

    // Naive dense alternative to `first_parameter_order2_terms`: the SAME row
    // expression through `Dual2<Tower3<4>>` — a fully dense third-order primary
    // tower (the generic forward-mode oracle carrier) with the log-sigma
    // parameter folded in as the outer second-order dual direction. The outer
    // `.g` channel is the first log-sigma derivative of the primary
    // (value, gradient, Hessian) tower, exactly the production output.
    fn racer_first_channel(
        primaries: &[f64; 4],
        scale: &ProbitFrailtyScaleJet,
        inputs: &RigidRowInputs,
    ) -> Result<Tower3<4>, String> {
        let variables: [Dual2<Tower3<4>>; 4] = std::array::from_fn(|axis| Dual2 {
            v: Tower3::variable(primaries[axis], axis),
            g: Tower3::constant(0.0),
            h: Tower3::constant(0.0),
        });
        let scale_jet = Dual2 {
            v: Tower3::constant(scale.s),
            g: Tower3::constant(scale.ds),
            h: Tower3::constant(scale.d2s),
        };
        Ok(eval_scaled(&variables, &scale_jet, inputs)?.g)
    }

    // Naive dense alternative to `second_parameter_order2_terms`: the SAME row
    // expression through `Dual2<Tower4<4>>` — the fully dense fourth-order
    // primary tower with the log-sigma parameter folded in as the outer
    // second-order dual direction. The outer `.h` channel is the second
    // log-sigma derivative of the primary (value, gradient, Hessian) tower.
    fn racer_second_channel(
        primaries: &[f64; 4],
        scale: &ProbitFrailtyScaleJet,
        inputs: &RigidRowInputs,
    ) -> Result<Tower4<4>, String> {
        let variables: [Dual2<Tower4<4>>; 4] = std::array::from_fn(|axis| Dual2 {
            v: Tower4::variable(primaries[axis], axis),
            g: Tower4::constant(0.0),
            h: Tower4::constant(0.0),
        });
        let scale_jet = Dual2 {
            v: Tower4::constant(scale.s),
            g: Tower4::constant(scale.ds),
            h: Tower4::constant(scale.d2s),
        };
        Ok(eval_scaled(&variables, &scale_jet, inputs)?.h)
    }

    type HandBundle = (f64, [f64; 4], [[f64; 4]; 4]);

    #[inline(always)]
    fn inner_derivative(
        q_axis: usize,
        q: f64,
        linear: f64,
        correction: &[f64; 5],
        axes: &[usize],
    ) -> f64 {
        let q_count = axes.iter().filter(|&&axis| axis == q_axis).count();
        if axes.iter().any(|&axis| axis != q_axis && axis != 3) || q_count > 1 {
            return 0.0;
        }
        let slope_count = axes.len() - q_count;
        if q_count == 1 {
            correction[slope_count]
        } else {
            q * correction[slope_count] + if slope_count == 1 { linear } else { 0.0 }
        }
    }

    #[inline(always)]
    fn composed_derivative(
        outer: &[f64; 5],
        inner: impl Fn(&[usize]) -> f64,
        axes: &[usize],
    ) -> f64 {
        match axes {
            [a] => outer[1] * inner(&[*a]),
            [a, b] => outer[2] * inner(&[*a]) * inner(&[*b]) + outer[1] * inner(&[*a, *b]),
            [a, b, c] => {
                outer[3] * inner(&[*a]) * inner(&[*b]) * inner(&[*c])
                    + outer[2]
                        * (inner(&[*a, *b]) * inner(&[*c])
                            + inner(&[*a, *c]) * inner(&[*b])
                            + inner(&[*b, *c]) * inner(&[*a]))
                    + outer[1] * inner(&[*a, *b, *c])
            }
            [a, b, c, d] => {
                outer[4] * inner(&[*a]) * inner(&[*b]) * inner(&[*c]) * inner(&[*d])
                    + outer[3]
                        * (inner(&[*a, *b]) * inner(&[*c]) * inner(&[*d])
                            + inner(&[*a, *c]) * inner(&[*b]) * inner(&[*d])
                            + inner(&[*a, *d]) * inner(&[*b]) * inner(&[*c])
                            + inner(&[*b, *c]) * inner(&[*a]) * inner(&[*d])
                            + inner(&[*b, *d]) * inner(&[*a]) * inner(&[*c])
                            + inner(&[*c, *d]) * inner(&[*a]) * inner(&[*b]))
                    + outer[2]
                        * (inner(&[*a, *b]) * inner(&[*c, *d])
                            + inner(&[*a, *c]) * inner(&[*b, *d])
                            + inner(&[*a, *d]) * inner(&[*b, *c])
                            + inner(&[*a, *b, *c]) * inner(&[*d])
                            + inner(&[*a, *b, *d]) * inner(&[*c])
                            + inner(&[*a, *c, *d]) * inner(&[*b])
                            + inner(&[*b, *c, *d]) * inner(&[*a]))
                    + outer[1] * inner(&[*a, *b, *c, *d])
            }
            _ => 0.0,
        }
    }

    /// Fully expanded, non-jet log-sigma chain for the complete primary
    /// objective/gradient/Hessian bundle. It evaluates the three scalar outer
    /// stacks once and applies the direct carrier-coordinate formulas.
    #[inline(never)]
    fn strongest_hand_sigma_bundle(
        primaries: [f64; 4],
        scale: ProbitFrailtyScaleJet,
        inputs: &RigidRowInputs,
        second: bool,
    ) -> HandBundle {
        let [q0, q1, qd1, g] = primaries;
        let b = scale.s * g;
        let covariance = inputs.covariance_ones;
        let correction_value = (1.0 + covariance * b * b).sqrt();
        let inverse = correction_value.recip();
        let inverse2 = inverse * inverse;
        let inverse3 = inverse2 * inverse;
        let inverse5 = inverse3 * inverse2;
        let inverse7 = inverse5 * inverse2;
        let correction = [
            correction_value,
            covariance * b * inverse,
            covariance * inverse3,
            -3.0 * covariance * covariance * b * inverse5,
            3.0 * covariance * covariance * (4.0 * covariance * b * b - 1.0) * inverse7,
        ];
        let eta0 = q0 * correction[0] + b * inputs.z_sum;
        let eta1 = q1 * correction[0] + b * inputs.z_sum;
        let adjusted = qd1 * correction[0];

        let entry_raw = unary_derivatives_neglog_phi(-eta0, inputs.wi);
        let exit_raw = unary_derivatives_neglog_phi(-eta1, inputs.wi * (1.0 - inputs.di));
        let density_raw = unary_derivatives_log_normal_pdf(eta1);
        let log_raw = unary_derivatives_log(adjusted);
        let entry = std::array::from_fn(|order| {
            -entry_raw[order] * if order % 2 == 0 { 1.0 } else { -1.0 }
        });
        let mut exit =
            std::array::from_fn(|order| exit_raw[order] * if order % 2 == 0 { 1.0 } else { -1.0 });
        let mut log_adjusted = [0.0; 5];
        if inputs.di > 0.0 {
            let event_scale = -inputs.wi * inputs.di;
            for order in 0..5 {
                exit[order] += event_scale * density_raw[order];
                log_adjusted[order] = event_scale * log_raw[order];
            }
        }
        let derivative = |axes: &[usize]| {
            composed_derivative(
                &entry,
                |selected| inner_derivative(0, q0, inputs.z_sum, &correction, selected),
                axes,
            ) + composed_derivative(
                &exit,
                |selected| inner_derivative(1, q1, inputs.z_sum, &correction, selected),
                axes,
            ) + composed_derivative(
                &log_adjusted,
                |selected| inner_derivative(2, qd1, 0.0, &correction, selected),
                axes,
            )
        };

        let f_b = derivative(&[3]);
        let f_bb = derivative(&[3, 3]);
        let f_bbb = derivative(&[3, 3, 3]);
        let f_qb: [f64; 3] = std::array::from_fn(|axis| derivative(&[axis, 3]));
        let f_qbb: [f64; 3] = std::array::from_fn(|axis| derivative(&[axis, 3, 3]));
        let mut f_qqb = [[0.0; 3]; 3];
        for axis in 0..3 {
            for other in axis..3 {
                let value = derivative(&[axis, other, 3]);
                f_qqb[axis][other] = value;
                f_qqb[other][axis] = value;
            }
        }
        let bt = g * scale.ds;
        if !second {
            let objective = f_b * bt;
            let gradient = std::array::from_fn(|axis| {
                if axis < 3 {
                    f_qb[axis] * bt
                } else {
                    f_bb * scale.s * bt + f_b * scale.ds
                }
            });
            let hessian = std::array::from_fn(|axis| {
                std::array::from_fn(|other| match (axis == 3, other == 3) {
                    (false, false) => f_qqb[axis][other] * bt,
                    (true, true) => {
                        f_bbb * scale.s * scale.s * bt + 2.0 * f_bb * scale.s * scale.ds
                    }
                    _ => {
                        let primary = if axis == 3 { other } else { axis };
                        f_qbb[primary] * scale.s * bt + f_qb[primary] * scale.ds
                    }
                })
            });
            return (objective, gradient, hessian);
        }

        let f_bbbb = derivative(&[3, 3, 3, 3]);
        let f_qbbb: [f64; 3] = std::array::from_fn(|axis| derivative(&[axis, 3, 3, 3]));
        let mut f_qqbb = [[0.0; 3]; 3];
        for axis in 0..3 {
            for other in axis..3 {
                let value = derivative(&[axis, other, 3, 3]);
                f_qqbb[axis][other] = value;
                f_qqbb[other][axis] = value;
            }
        }
        let btt = g * scale.d2s;
        let objective = f_bb * bt * bt + f_b * btt;
        let gradient = std::array::from_fn(|axis| {
            if axis < 3 {
                f_qbb[axis] * bt * bt + f_qb[axis] * btt
            } else {
                f_bbb * bt * bt * scale.s
                    + f_bb * (btt * scale.s + 2.0 * bt * scale.ds)
                    + f_b * scale.d2s
            }
        });
        let hessian = std::array::from_fn(|axis| {
            std::array::from_fn(|other| match (axis == 3, other == 3) {
                (false, false) => f_qqbb[axis][other] * bt * bt + f_qqb[axis][other] * btt,
                (true, true) => {
                    f_bbbb * bt * bt * scale.s * scale.s
                        + f_bbb * (btt * scale.s * scale.s + 4.0 * bt * scale.s * scale.ds)
                        + f_bb * (2.0 * scale.ds * scale.ds + 2.0 * scale.s * scale.d2s)
                }
                _ => {
                    let primary = if axis == 3 { other } else { axis };
                    f_qbbb[primary] * bt * bt * scale.s
                        + f_qbb[primary] * (btt * scale.s + 2.0 * bt * scale.ds)
                        + f_qb[primary] * scale.d2s
                }
            })
        });
        (objective, gradient, hessian)
    }

    #[inline(never)]
    fn production_sigma_bundle(
        primaries: [f64; 4],
        scale: ProbitFrailtyScaleJet,
        inputs: &RigidRowInputs,
        second: bool,
    ) -> HandBundle {
        let terms = compiled_sigma_primary_terms(primaries, scale, inputs, second)
            .expect("valid synthetic sigma row");
        (terms.objective, terms.grad, terms.hess)
    }

    fn close(label: &str, actual: f64, expected: f64) {
        let tolerance = 2.0e-11 * actual.abs().max(expected.abs()).max(1.0);
        assert!(
            actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:+.16e}, expected={expected:+.16e}, tolerance={tolerance:.3e}",
        );
    }

    fn close_bundle(label: &str, actual: HandBundle, expected: HandBundle) {
        close(&format!("{label} objective"), actual.0, expected.0);
        for axis in 0..4 {
            close(
                &format!("{label} grad[{axis}]"),
                actual.1[axis],
                expected.1[axis],
            );
            for other in 0..4 {
                close(
                    &format!("{label} hess[{axis},{other}]"),
                    actual.2[axis][other],
                    expected.2[axis][other],
                );
            }
        }
    }

    /// #932 release speed gate for the outer log-sigma hyperparameter compiler
    /// path behind [`SurvivalMarginalSlopeFamily::row_sigma_primary_terms`].
    /// Production composes the macro-emitted order-two, contracted-third, and
    /// contracted-fourth feature derivatives through the observed-slope
    /// Jacobian/curvature. Its first and second log-sigma bundles are
    /// parity-pinned to `2e-11` relative against both a dense generic
    /// `Dual2<Tower3<4>>` / `Dual2<Tower4<4>>` oracle and an independently
    /// expanded analytic schedule.
    ///
    /// The dense-tower ratio is diagnostic. The binding speed gate is the
    /// optimized analytic schedule: it evaluates the scalar outer stacks once,
    /// caches every distinct required mixed partial, reuses Hessian symmetry,
    /// crosses the same outlined ABI, and returns the same complete 21-scalar
    /// bundle. The contenders alternate sampling order and fold every result
    /// channel into the feedback checksum. The MSI release harness fails closed
    /// whenever median strongest-hand time over production time is `<= 1`.
    ///
    /// The feedback barrier (no `std::hint::black_box`) nudges the observed
    /// slope primary by a negligible `1e-18` multiple of the running checksum,
    /// which folds value/gradient/Hessian channels, so the loop-carried
    /// recurrence prevents the optimizer from hoisting or dropping the pure
    /// jet evaluations while keeping the perturbed primary bit-adjacent to the
    /// fixture regime.
    #[test]
    fn release_measure_sigma_parameter_jets_vs_strongest_hand_932() {
        // One ordinary interior row per event branch: censored (d=0) and event
        // (d=1) drive different live derivative stacks, so each is its own cell.
        let cases = [
            (
                -0.7_f64, 0.4_f64, 0.8_f64, -0.3_f64, 1.0_f64, 0.0_f64, 0.6_f64,
            ),
            (
                0.2_f64, -0.5_f64, 1.4_f64, 0.9_f64, 0.8_f64, 1.0_f64, -1.1_f64,
            ),
        ];
        let scale = ProbitFrailtyScaleJet::from_log_sigma((0.85_f64).ln());

        fn unit(state: &mut u64) -> f64 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
        }

        let mut state = 0x932_C0DE_5EED_u64;
        for case in 0..192 {
            let primaries = [
                -1.5 + 3.0 * unit(&mut state),
                -1.5 + 3.0 * unit(&mut state),
                0.2 + 1.8 * unit(&mut state),
                -1.4 + 2.8 * unit(&mut state),
            ];
            let wi = 0.2 + 1.8 * unit(&mut state);
            let di = (case % 2) as f64;
            let z_sum = -2.0 + 4.0 * unit(&mut state);
            let mut inputs = synthetic_inputs(wi, di, z_sum);
            inputs.covariance_ones = 0.05 + 2.45 * unit(&mut state);
            let sigma = 0.2 + 1.8 * unit(&mut state);
            let scale_case = ProbitFrailtyScaleJet::from_log_sigma(sigma.ln());
            for second in [false, true] {
                let production = production_sigma_bundle(primaries, scale_case, &inputs, second);
                let hand = strongest_hand_sigma_bundle(primaries, scale_case, &inputs, second);
                close_bundle(
                    &format!("random case={case} order={}", usize::from(second) + 1),
                    production,
                    hand,
                );
                let dense = if second {
                    let channel = racer_second_channel(&primaries, &scale_case, &inputs)
                        .expect("random second-parameter dense-tower racer");
                    (channel.v, channel.g, channel.h)
                } else {
                    let channel = racer_first_channel(&primaries, &scale_case, &inputs)
                        .expect("random first-parameter dense-tower racer");
                    (channel.v, channel.g, channel.h)
                };
                close_bundle(
                    &format!("random dense case={case} order={}", usize::from(second) + 1),
                    production,
                    dense,
                );
            }
        }

        // Parity below runs in every build; the speed contract only when the
        // gate is open (release profile -- `SpeedGate::open` documents why).
        let mut gate = (!cfg!(debug_assertions)).then(|| SpeedGate::open("SIGMA-PARAM-932"));
        let fold = |(objective, grad, hess): HandBundle| -> f64 {
            objective + grad.iter().sum::<f64>() + hess.iter().flatten().sum::<f64>()
        };
        for &(q0, q1, qd1, g, wi, di, z_sum) in &cases {
            let inputs = synthetic_inputs(wi, di, z_sum);
            let primaries = [q0, q1, qd1, g];

            // Parity: first log-sigma derivative — direct compiler contractions
            // vs dense `Dual2<Tower3<4>>` on objective/gradient/Hessian.
            let production_first = compiled_sigma_primary_terms(primaries, scale, &inputs, false)
                .expect("sigma first-parameter production terms");
            let racer_first = racer_first_channel(&primaries, &scale, &inputs)
                .expect("sigma first-parameter dense-tower racer");
            close(
                &format!("event={di:.0} d/dlogsigma objective"),
                production_first.objective,
                racer_first.v,
            );
            for a in 0..4 {
                close(
                    &format!("event={di:.0} d/dlogsigma grad[{a}]"),
                    production_first.grad[a],
                    racer_first.g[a],
                );
                for b in 0..4 {
                    close(
                        &format!("event={di:.0} d/dlogsigma hess[{a},{b}]"),
                        production_first.hess[a][b],
                        racer_first.h[a][b],
                    );
                }
            }
            let hand_first = strongest_hand_sigma_bundle(primaries, scale, &inputs, false);
            close(
                &format!("event={di:.0} d/dlogsigma hand objective"),
                production_first.objective,
                hand_first.0,
            );
            for a in 0..4 {
                close(
                    &format!("event={di:.0} d/dlogsigma hand grad[{a}]"),
                    production_first.grad[a],
                    hand_first.1[a],
                );
                for b in 0..4 {
                    close(
                        &format!("event={di:.0} d/dlogsigma hand hess[{a},{b}]"),
                        production_first.hess[a][b],
                        hand_first.2[a][b],
                    );
                }
            }

            // Parity: second log-sigma derivative — direct compiler contractions
            // vs dense `Dual2<Tower4<4>>` on objective/gradient/Hessian.
            let production_second = compiled_sigma_primary_terms(primaries, scale, &inputs, true)
                .expect("sigma second-parameter production terms");
            let racer_second = racer_second_channel(&primaries, &scale, &inputs)
                .expect("sigma second-parameter dense-tower racer");
            close(
                &format!("event={di:.0} d2/dlogsigma2 objective"),
                production_second.objective,
                racer_second.v,
            );
            for a in 0..4 {
                close(
                    &format!("event={di:.0} d2/dlogsigma2 grad[{a}]"),
                    production_second.grad[a],
                    racer_second.g[a],
                );
                for b in 0..4 {
                    close(
                        &format!("event={di:.0} d2/dlogsigma2 hess[{a},{b}]"),
                        production_second.hess[a][b],
                        racer_second.h[a][b],
                    );
                }
            }
            let hand_second = strongest_hand_sigma_bundle(primaries, scale, &inputs, true);
            close(
                &format!("event={di:.0} d2/dlogsigma2 hand objective"),
                production_second.objective,
                hand_second.0,
            );
            for a in 0..4 {
                close(
                    &format!("event={di:.0} d2/dlogsigma2 hand grad[{a}]"),
                    production_second.grad[a],
                    hand_second.1[a],
                );
                for b in 0..4 {
                    close(
                        &format!("event={di:.0} d2/dlogsigma2 hand hess[{a},{b}]"),
                        production_second.hess[a][b],
                        hand_second.2[a][b],
                    );
                }
            }

            let Some(gate) = gate.as_mut() else {
                continue;
            };
            // Each order is timed through the same fixed-size 21-scalar bundle
            // against BOTH opponents: the strongest analytic schedule (the
            // binding contract) and the dense generic tower (the specialisation
            // must also beat the substrate it specialises). The nudge perturbs
            // the observed-slope primary.
            for second in [false, true] {
                let order = usize::from(second) + 1;
                let hand = paired_interleaved(
                    15,
                    20_000,
                    0x9320_51_6A ^ order as u64 ^ (di.to_bits() >> 60),
                    |nudge| {
                        fold(production_sigma_bundle(
                            [q0, q1, qd1, g + nudge],
                            scale,
                            &inputs,
                            second,
                        ))
                    },
                    |nudge| {
                        fold(strongest_hand_sigma_bundle(
                            [q0, q1, qd1, g + nudge],
                            scale,
                            &inputs,
                            second,
                        ))
                    },
                );
                gate.faster(
                    &format!("order={order} event={di:.0} opponent=strongest_hand"),
                    &hand,
                    "production",
                    "strongest_hand",
                );
                let dense = paired_interleaved(
                    15,
                    20_000,
                    0x9320_D3_6A ^ order as u64 ^ (di.to_bits() >> 60),
                    |nudge| {
                        fold(production_sigma_bundle(
                            [q0, q1, qd1, g + nudge],
                            scale,
                            &inputs,
                            second,
                        ))
                    },
                    |nudge| {
                        let primaries = [q0, q1, qd1, g + nudge];
                        if second {
                            let channel = racer_second_channel(&primaries, &scale, &inputs)
                                .expect("sigma second-parameter dense-tower racer");
                            fold((channel.v, channel.g, channel.h))
                        } else {
                            let channel = racer_first_channel(&primaries, &scale, &inputs)
                                .expect("sigma first-parameter dense-tower racer");
                            fold((channel.v, channel.g, channel.h))
                        }
                    },
                );
                gate.faster(
                    &format!("order={order} event={di:.0} opponent=dense_tower"),
                    &dense,
                    "production",
                    "dense_tower",
                );
            }
        }
        if let Some(gate) = gate {
            gate.finish();
        }
    }
}
