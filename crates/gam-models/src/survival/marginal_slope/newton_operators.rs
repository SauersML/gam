//! Exact-Newton joint-Hessian linear operators built on the family: the
//! psi-Hessian directional operator and the flex no-wiggle joint-Hessian
//! first/second directional-derivative operators.

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn exact_newton_joint_hessian_operator(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<(Arc<dyn HyperOperator>, Array1<f64>, f64, Array1<f64>), String> {
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let p_total = slices.total;
        let make_acc = || BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i);

        // Phase 2d: the same per-row work that yields the block-Hessian pullback
        // also yields the row negative-log-likelihood and the joint gradient.
        // Accumulating all three in one row pass avoids the three separate
        // n-row sweeps the inner Newton previously needed
        // (operator/diagonal here, log-likelihood via `log_likelihood_only_with_options`,
        // joint gradient via `evaluate_exact_newton_joint_gradient_dynamic_q`).
        // The workspace then publishes the cached nll and gradient through
        // `joint_log_likelihood_evaluation` / `joint_gradient_evaluation` so
        // `custom_family.rs` skips its fallback row passes.
        //
        // The per-thread accumulator embeds a `SurvivalMarginalSlopeDynamicRow`
        // workspace so the nine Array2/Array1 buffers are reused across all
        // rows handled by one rayon worker.
        type FusedAcc = (
            BlockHessianAccumulator,
            f64,
            Array1<f64>,
            SurvivalMarginalSlopeDynamicRow,
        );
        let make_fused = || -> FusedAcc {
            (
                make_acc(),
                0.0,
                Array1::<f64>::zeros(p_total),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };
        // Row measure: full-data when `options.outer_score_subsample` is
        // `None`; the stratified mask + HT weights otherwise. Matches the
        // directional-derivative sibling so the inner trust-region ratio
        // evaluates ½δᵀHδ on the same rows as the objective.
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        let flex_active = self.effective_flex_active(block_states)?;
        let identity_blocks = if flex_active {
            flex_identity_block_pairs(&flex_primary_slices(self), &slices)
        } else {
            vec![]
        };
        let primary = if flex_active {
            Some(flex_primary_slices(self))
        } else {
            None
        };
        // The combined-across-blocks accumulator excludes the DynRow
        // workspace: it is per-task scratch, allocated once per base block
        // and never meaningfully combined across blocks (see `make_fused`'s
        // doc comment above).
        type CombinedAcc = (BlockHessianAccumulator, f64, Array1<f64>);
        let final_acc = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            row_iter.len(),
            |range| -> Result<CombinedAcc, String> {
                let (mut state, mut nll_acc, mut grad_acc, mut q_geom) = make_fused();
                for idx in range {
                    let row = row_iter[idx];
                    self.row_dynamic_q_geometry_into(row, block_states, &mut q_geom)?;
                    let (row_nll, mut g, mut h) = if let Some(ref primary) = primary {
                        self.compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
                            primary,
                        )?
                    } else {
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?
                    };
                    let w = row_weights[row];
                    if w != 1.0 {
                        g.mapv_inplace(|v| v * w);
                        h.mapv_inplace(|v| v * w);
                    }
                    // nll: sum over weighted rows. Sign mirrors
                    // `evaluate_exact_newton_joint_dynamic_q_dense` (state.0 -= row_nll).
                    nll_acc -= row_nll * w;
                    // Joint gradient: q-geometry pullback for time/marginal/logslope
                    // primary outputs, plus identity contribution for flex blocks
                    // (score_warp_dev, link_dev). Matches the gradient half of
                    // `accumulate_dynamic_q_joint_row`.
                    self.accumulate_dynamic_q_core_gradient(
                        row,
                        &slices,
                        &q_geom,
                        g.slice(s![0..N_PRIMARY]),
                        &mut grad_acc,
                    )?;
                    for (primary_range, joint_range) in identity_blocks.iter() {
                        for local in 0..primary_range.len() {
                            grad_acc[joint_range.start + local] -= g[primary_range.start + local];
                        }
                    }
                    // Block-Hessian pullback (unchanged).
                    state.add_pullback_with_q_geometry(self, row, &q_geom, &g, &h)?;
                }
                Ok((state, nll_acc, grad_acc))
            },
            |mut a, b| -> Result<CombinedAcc, String> {
                a.0.add(&b.0);
                a.1 += b.1;
                a.2 += &b.2;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| (make_acc(), 0.0, Array1::<f64>::zeros(p_total)));
        // The second field accumulates the *signed* log-likelihood
        // (`state.0 -= row_nll`), mirroring the sign convention in
        // `evaluate_exact_newton_joint_dynamic_q_dense`.
        let acc = final_acc.0;
        let joint_log_likelihood = final_acc.1;
        let joint_gradient = final_acc.2;

        let diagonal = acc.diagonal(&slices);
        Ok((
            Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>,
            diagonal,
            joint_log_likelihood,
            joint_gradient,
        ))
    }

    /// Outer-aware operator builder for the flex-no-wiggle joint-Hessian
    /// directional derivative. The default-options shim is omitted because
    /// the `SurvivalMarginalSlopeExactNewtonJointHessianWorkspace` always
    /// threads its own `BlockwiseFitOptions`. When `options.outer_score_subsample` is
    /// `Some`, only the sampled rows are visited and the accumulator uses
    /// per-row Horvitz-Thompson inverse-inclusion weights before being wrapped
    /// in the `HyperOperator`.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Arc<dyn HyperOperator>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        let make_acc_ws = || {
            (
                BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };
        let acc = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            row_iter.len(),
            |range| -> Result<_, String> {
                let (mut state, mut q_geom) = make_acc_ws();
                for idx in range {
                    let row = row_iter[idx];
                    self.row_dynamic_q_geometry_into(row, block_states, &mut q_geom)?;
                    let h_pi = self
                        .compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
                            &primary,
                        )?
                        .2;
                    let u_d = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_beta_flat,
                    )?;
                    let mut t_ud =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &u_d)?;
                    let mut h_ud = h_pi.dot(&u_d);
                    let w = row_weights[row];
                    if w != 1.0 {
                        h_ud.mapv_inplace(|v| v * w);
                        t_ud.mapv_inplace(|v| v * w);
                    }
                    state.add_pullback_with_q_geometry(self, row, &q_geom, &h_ud, &t_ud)?;
                }
                Ok((state, q_geom))
            },
            |mut a, b| -> Result<_, String> {
                a.0.add(&b.0);
                Ok(a)
            },
        )?
        .unwrap_or_else(make_acc_ws)
        .0;
        Ok(Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>)
    }

    /// Outer-aware operator builder for the flex-no-wiggle joint-Hessian
    /// second directional derivative. The default-options shim is omitted
    /// because the `SurvivalMarginalSlopeExactNewtonJointHessianWorkspace`
    /// always threads its own `BlockwiseFitOptions`. When `options.outer_score_subsample` is
    /// `Some`, only the sampled rows are visited and the accumulator uses
    /// per-row Horvitz-Thompson inverse-inclusion weights before being wrapped
    /// in the `HyperOperator`.
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_operator_flex_no_wiggle_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Arc<dyn HyperOperator>, String> {
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        let make_acc_ws = || {
            (
                BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };
        let acc = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            row_iter.len(),
            |range| -> Result<_, String> {
                let (mut state, mut q_geom) = make_acc_ws();
                for idx in range {
                    let row = row_iter[idx];
                    self.row_dynamic_q_geometry_into(row, block_states, &mut q_geom)?;
                    let ud = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_u,
                    )?;
                    let ue = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_v,
                    )?;
                    let mut q_de =
                        self.row_flex_primary_fourth_contracted_exact(row, block_states, &ud, &ue)?;
                    let t_d =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &ud)?;
                    let mut gamma = t_d.dot(&ue);
                    let w = row_weights[row];
                    if w != 1.0 {
                        gamma.mapv_inplace(|v| v * w);
                        q_de.mapv_inplace(|v| v * w);
                    }
                    state.add_pullback_with_q_geometry(self, row, &q_geom, &gamma, &q_de)?;
                }
                Ok((state, q_geom))
            },
            |mut a, b| -> Result<_, String> {
                a.0.add(&b.0);
                Ok(a)
            },
        )?
        .unwrap_or_else(make_acc_ws)
        .0;
        Ok(Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>)
    }
}
