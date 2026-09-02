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

