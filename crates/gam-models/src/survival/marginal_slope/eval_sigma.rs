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
        // True fast path: K=1 uses the packed lowering of the canonical row
        // program; K>1 uses the covariance-aware vector likelihood.
        let guard = self.derivative_guard;
        let probit_scale = self.probit_frailty_scale();
        let score_dim = self.score_dim();
        gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            row_iter.len(),
            |range| -> Result<f64, String> {
                let mut ll = 0.0;
                let mut logslope_workspace = self.logslope_row_workspace()?;
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
                                &mut logslope_workspace,
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
        rigid_row_nll(&observed_primaries, &inputs)
    }

    pub(crate) fn row_sigma_primary_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        second_sigma: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let primaries = rigid_row_kernel_primaries(self, block_states, row)?;
        let scale = self.sigma_scale_derivatives()?;
        let terms = if second_sigma {
            second_parameter_order2_terms(
                primaries,
                scale.s,
                scale.ds,
                scale.d2s,
                |variables, parameter| {
                    self.row_neglog_canonical_scale_jet(row, block_states, variables, parameter)
                },
            )?
        } else {
            first_parameter_order2_terms(primaries, scale.s, scale.ds, |variables, parameter| {
                self.row_neglog_canonical_scale_jet(row, block_states, variables, parameter)
            })?
        };
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
        let p_g = slices.logslope.len();
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
                let primaries = rigid_row_kernel_primaries(self, block_states, row)?;
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

#[cfg(test)]
mod sigma_parameter_jet_release_tests {
    use super::*;
    use crate::survival::lognormal_kernel::ProbitFrailtyScaleJet;
    use gam_math::jet_scalar::JetScalar;
    use gam_math::jet_tower::{Tower3, Tower4};
    use gam_math::nested_dual::Dual2;
    use std::time::Instant;

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
        rigid_row_nll(&observed, inputs)
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

    fn close(label: &str, actual: f64, expected: f64) {
        let tolerance = 2.0e-11 * actual.abs().max(expected.abs()).max(1.0);
        assert!(
            actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}: production={actual:+.16e}, dense_tower={expected:+.16e}, tolerance={tolerance:.3e}",
        );
    }

    /// #932 release speed gate for the outer log-sigma hyperparameter jet path
    /// behind [`SurvivalMarginalSlopeFamily::row_sigma_primary_terms`]. The
    /// production seeded instantiations — `OneSeed<4>` for the first log-sigma
    /// derivative ([`first_parameter_order2_terms`]) and `TwoSeed<4>` for the
    /// second ([`second_parameter_order2_terms`]) — are timed against the naive
    /// dense alternative that evaluates the same
    /// `row_neglog_canonical_scale_jet` row expression through the generic
    /// forward-mode jet tower (`Dual2<Tower3<4>>` / `Dual2<Tower4<4>>`, the
    /// parameter direction folded into the dense primary tower). Each order is
    /// parity-pinned to `2e-11` relative before timing, and emits one
    /// harness-parsed `hand_over_production` token (dense-tower time over
    /// production time) per event branch; the MSI release harness fails closed
    /// whenever any measured cell is `<= 1`.
    ///
    /// The feedback barrier (no `std::hint::black_box`) nudges the observed
    /// slope primary by a negligible `1e-18` multiple of the running checksum,
    /// which folds value/gradient/Hessian channels, so the loop-carried
    /// recurrence prevents the optimizer from hoisting or dropping the pure
    /// jet evaluations while keeping the perturbed primary bit-adjacent to the
    /// fixture regime.
    #[test]
    fn release_measure_sigma_parameter_jets_vs_generic_tower_932() {
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

        fn best_ns<F>(iterations: usize, base_g: f64, mut evaluate: F) -> f64
        where
            F: FnMut(f64) -> (f64, Array1<f64>, Array2<f64>),
        {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let mut checksum = 0.0_f64;
                let started = Instant::now();
                for _ in 0..iterations {
                    let (objective, grad, hess) = evaluate(base_g + checksum * 1e-18);
                    checksum += objective + grad[0] + hess[[0, 0]];
                }
                assert!(
                    checksum.is_finite(),
                    "sigma-parameter release-measure checksum must stay finite"
                );
                best = best.min(started.elapsed().as_secs_f64());
            }
            best * 1e9 / iterations as f64
        }

        let iterations = 100_000usize;
        for &(q0, q1, qd1, g, wi, di, z_sum) in &cases {
            let inputs = synthetic_inputs(wi, di, z_sum);
            let primaries = [q0, q1, qd1, g];

            // Parity: first log-sigma derivative — production `OneSeed<4>` vs
            // dense `Dual2<Tower3<4>>` on objective/gradient/Hessian.
            let production_first =
                first_parameter_order2_terms(primaries, scale.s, scale.ds, |vars, param| {
                    eval_scaled(vars, param, &inputs)
                })
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
                        production_first.hess[[a, b]],
                        racer_first.h[a][b],
                    );
                }
            }

            // Parity: second log-sigma derivative — production `TwoSeed<4>` vs
            // dense `Dual2<Tower4<4>>` on objective/gradient/Hessian.
            let production_second = second_parameter_order2_terms(
                primaries,
                scale.s,
                scale.ds,
                scale.d2s,
                |vars, param| eval_scaled(vars, param, &inputs),
            )
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
                        production_second.hess[[a, b]],
                        racer_second.h[a][b],
                    );
                }
            }

            // Time each order. Both paths package into the same `Array1`/`Array2`
            // shape, so the ratio isolates the seeded-jet vs dense-tower cost.
            let production_first_ns = best_ns(iterations, g, |perturbed_g| {
                let primaries = [q0, q1, qd1, perturbed_g];
                let terms =
                    first_parameter_order2_terms(primaries, scale.s, scale.ds, |vars, param| {
                        eval_scaled(vars, param, &inputs)
                    })
                    .expect("sigma first-parameter production terms");
                (terms.objective, terms.grad, terms.hess)
            });
            let racer_first_ns = best_ns(iterations, g, |perturbed_g| {
                let primaries = [q0, q1, qd1, perturbed_g];
                let channel = racer_first_channel(&primaries, &scale, &inputs)
                    .expect("sigma first-parameter dense-tower racer");
                let grad = Array1::from_vec(channel.g.to_vec());
                let hess = Array2::from_shape_fn((4, 4), |(i, j)| channel.h[i][j]);
                (channel.v, grad, hess)
            });
            eprintln!(
                "SIGMA-PARAM-JET-932 order=1 event={di:.0} production={production_first_ns:.2} \
                 ns/row dense_tower={racer_first_ns:.2} ns/row hand_over_production={:.6}",
                racer_first_ns / production_first_ns,
            );

            let production_second_ns = best_ns(iterations, g, |perturbed_g| {
                let primaries = [q0, q1, qd1, perturbed_g];
                let terms = second_parameter_order2_terms(
                    primaries,
                    scale.s,
                    scale.ds,
                    scale.d2s,
                    |vars, param| eval_scaled(vars, param, &inputs),
                )
                .expect("sigma second-parameter production terms");
                (terms.objective, terms.grad, terms.hess)
            });
            let racer_second_ns = best_ns(iterations, g, |perturbed_g| {
                let primaries = [q0, q1, qd1, perturbed_g];
                let channel = racer_second_channel(&primaries, &scale, &inputs)
                    .expect("sigma second-parameter dense-tower racer");
                let grad = Array1::from_vec(channel.g.to_vec());
                let hess = Array2::from_shape_fn((4, 4), |(i, j)| channel.h[i][j]);
                (channel.v, grad, hess)
            });
            eprintln!(
                "SIGMA-PARAM-JET-932 order=2 event={di:.0} production={production_second_ns:.2} \
                 ns/row dense_tower={racer_second_ns:.2} ns/row hand_over_production={:.6}",
                racer_second_ns / production_second_ns,
            );
        }
    }
}
