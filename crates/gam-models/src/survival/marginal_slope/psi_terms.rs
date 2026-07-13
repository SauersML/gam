//! Hyperparameter (psi) score and curvature terms: the score-block
//! accumulation, the time-wiggle psi action, and the first-/second-order
//! psi terms together with their directional-derivative accumulators.

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Accumulate block-local score from a primary-space vector (replaces
    /// pullback_primary_vector + score += for score accumulation).
    pub(crate) fn accumulate_score_blockwise(
        &self,
        row: usize,
        primary: &Array1<f64>,
        score_t: &mut Array1<f64>,
        score_m: &mut Array1<f64>,
        score_g: &mut Array1<f64>,
    ) -> Result<(), String> {
        {
            let mut st = score_t.view_mut();
            self.design_entry
                .axpy_row_into(row, primary[0], &mut st)
                .expect("time entry axpy dim mismatch");
            self.design_exit
                .axpy_row_into(row, primary[1], &mut st)
                .expect("time exit axpy dim mismatch");
            self.design_derivative_exit
                .axpy_row_into(row, primary[2], &mut st)
                .expect("time deriv axpy dim mismatch");
        }
        self.marginal_design.axpy_row_into(
            row,
            primary[0] + primary[1],
            &mut score_m.view_mut(),
        )?;
        self.logslope_layout.coefficient_design()
            .axpy_row_into(row, primary[3], &mut score_g.view_mut())?;
        Ok(())
    }

    pub(crate) fn accumulate_score_identity_blocks(
        &self,
        primary_layout: Option<&FlexPrimarySlices>,
        primary: &Array1<f64>,
        score_h: Option<&mut Array1<f64>>,
        score_w: Option<&mut Array1<f64>>,
    ) {
        if let Some(primary_layout) = primary_layout {
            if let (Some(range), Some(score_h)) = (primary_layout.h.as_ref(), score_h) {
                *score_h = &*score_h + &primary.slice(s![range.clone()]);
            }
            if let (Some(range), Some(score_w)) = (primary_layout.w.as_ref(), score_w) {
                *score_w = &*score_w + &primary.slice(s![range.clone()]);
            }
        }
    }

    /// Score pullback using actual Jacobians from q-geometry (timewiggle-correct).
    pub(crate) fn accumulate_score_with_q_geometry(
        &self,
        row: usize,
        qg: &SurvivalMarginalSlopeDynamicRow,
        primary: &Array1<f64>,
        score_t: &mut Array1<f64>,
        score_m: &mut Array1<f64>,
        score_g: &mut Array1<f64>,
    ) -> Result<(), String> {
        let jt = [&qg.dq0_time, &qg.dq1_time, &qg.dqd1_time];
        let jm = [&qg.dq0_marginal, &qg.dq1_marginal, &qg.dqd1_marginal];
        for q in 0..3 {
            if primary[q] != 0.0 {
                score_t.scaled_add(primary[q], jt[q]);
            }
        }
        for q in 0..3 {
            if primary[q] != 0.0 {
                score_m.scaled_add(primary[q], jm[q]);
            }
        }
        self.logslope_layout.coefficient_design()
            .axpy_row_into(row, primary[3], &mut score_g.view_mut())?;
        Ok(())
    }

    /// U_{iB}^α contribution to score (eq 46, term 1) for timewiggle + marginal ψ.
    pub(crate) fn accumulate_score_timewiggle_psi_u(
        lift: &TimewiggleMarginalPsiRowLift,
        f_pi: &Array1<f64>,
        score_t: &mut Array1<f64>,
        score_m: &mut Array1<f64>,
    ) {
        let ut = [&lift.u_q0_time, &lift.u_q1_time, &lift.u_qd1_time];
        let um = [
            &lift.u_q0_marginal,
            &lift.u_q1_marginal,
            &lift.u_qd1_marginal,
        ];
        for q in 0..3 {
            if f_pi[q] != 0.0 {
                score_t.scaled_add(f_pi[q], ut[q]);
            }
        }
        for q in 0..3 {
            if f_pi[q] != 0.0 {
                score_m.scaled_add(f_pi[q], um[q]);
            }
        }
    }

    /// Compute u_i^{ψε} = D_ψ D_β π_i · d_beta for hessian directional derivative.
    /// With timewiggle + marginal ψ, includes T''(h) cross-terms.
    pub(crate) fn timewiggle_psi_action(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        primary_layout: Option<&FlexPrimarySlices>,
        psi_row: &Array1<f64>,
        beta_psi: &Array1<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let beta_time = &block_states[0].beta;
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let ec = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_psi_action design_entry: {e}"))?;
        let xc = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_psi_action design_exit: {e}"))?;
        let dc = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_psi_action design_derivative_exit: {e}"))?;
        let x_e = ec.row(0).slice(s![..p_base]).to_owned();
        let x_x = xc.row(0).slice(s![..p_base]).to_owned();
        let x_d = dc.row(0).slice(s![..p_base]).to_owned();
        let mc = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_psi_action marginal_design: {e}"))?;
        let x_m = mc.row(0).to_owned();
        let bm = block_states[1].eta[row];
        let h0 = x_e.dot(&beta_time.slice(s![..p_base])) + self.offset_entry[row] + bm;
        let h1 = x_x.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + bm;
        let d_raw = x_d.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
        let eg = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| "missing entry timewiggle for psi action".to_string())?;
        let xg = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| "missing exit timewiggle for psi action".to_string())?;
        let mu = psi_row.dot(beta_psi);
        let dt = d_beta_flat.slice(s![slices.time.clone()]);
        let dm = d_beta_flat.slice(s![slices.marginal.clone()]);
        let dh0 = x_e.dot(&dt.slice(s![..p_base])) + x_m.dot(&dm);
        let dh1 = x_x.dot(&dt.slice(s![..p_base])) + x_m.dot(&dm);
        let dd_raw = x_d.dot(&dt.slice(s![..p_base]));
        let dmu = psi_row.dot(&dm);
        let mut out = Array1::zeros(primary_layout.map_or(N_PRIMARY, |primary| primary.total));
        let q0_idx = primary_layout.map_or(0, |primary| primary.q0);
        let q1_idx = primary_layout.map_or(1, |primary| primary.q1);
        let qd1_idx = primary_layout.map_or(2, |primary| primary.qd1);
        out[q0_idx] = eg.d2q_dq02[0] * mu * dh0 + eg.dq_dq0[0] * dmu;
        out[q1_idx] = xg.d2q_dq02[0] * mu * dh1 + xg.dq_dq0[0] * dmu;
        out[qd1_idx] =
            xg.d3q_dq03[0] * d_raw * mu * dh1 + xg.d2q_dq02[0] * (dd_raw * mu + d_raw * dmu);
        Ok(out)
    }

    pub(crate) fn psi_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms_inner_with_options(
            block_states,
            derivative_blocks,
            psi_index,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `psi_terms_inner`. When
    /// `options.outer_score_subsample` is `None`, iterates all rows and is
    /// bit-for-bit equivalent to the legacy implementation. When `Some`, only
    /// the sampled rows contribute and every row-summed component (objective
    /// scalar, per-block score vectors, Hessian operator blocks) is accumulated
    /// with the row's Horvitz-Thompson inverse-inclusion weight.
    pub(crate) fn psi_terms_inner_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: Option<&EvalCache>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let flex_primary = flex_active.then(|| flex_primary_slices(self));
        let slices = block_slices(self, block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let loading = if let Some(primary) = flex_primary.as_ref() {
            spatial_block_primary_loading_flex(primary, block_idx)?
        } else {
            spatial_block_primary_loading(block_idx)?
        };
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };

        let timewiggle_active = self.flex_timewiggle_active();
        let timewiggle_psi = timewiggle_active && block_idx == 1;

        // NOTE: old dense timewiggle early return removed; now handled by
        // block path with lifted Jacobians below.
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());

        // Build the psi design map once; rowwise loop does direct row_vector(row)
        // calls via the PsiDesignMap API.
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let psi_map = crate::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            self.n,
            p_psi,
            0..self.n,
            psi_label,
            &policy,
        )?;

        // Parallel accumulation: each worker gets its own block-local accumulators.
        type Acc = (
            f64,                     // objective_psi
            Array1<f64>,             // score_t
            Array1<f64>,             // score_m
            Array1<f64>,             // score_g
            Array1<f64>,             // score_h
            Array1<f64>,             // score_w
            BlockHessianAccumulator, // Hessian blocks
        );
        let make_acc = || -> Acc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                Array1::zeros(p_h),
                Array1::zeros(p_w),
                BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
            )
        };

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Process fixed row chunks in parallel and merge local cross-block
        // accumulators in row-chunk order for deterministic timewiggle assembly.
        let (objective_psi, score_t, score_m, score_g, score_h, score_w, acc) =
            chunked_row_reduction(
                row_iter.as_slice(),
                make_acc,
                |row, a| -> Result<(), String> {
                    let psi_row = psi_map
                        .row_vector(row)
                        .map_err(|e| format!("survival rowwise psi map: {e}"))?;

                    let q_geom = if timewiggle_active {
                        Some(self.row_dynamic_q_geometry(row, block_states)?)
                    } else {
                        None
                    };

                    let psi_lift = if timewiggle_psi {
                        Some(self.timewiggle_marginal_psi_row_lift(
                            row,
                            block_states,
                            flex_primary.as_ref(),
                            &psi_row,
                            beta_psi,
                        )?)
                    } else {
                        None
                    };

                    let dir = if let Some(lift) = psi_lift.as_ref() {
                        lift.dir.clone()
                    } else if let Some(primary) = flex_primary.as_ref() {
                        primary_direction_from_psi_row_flex(primary, block_idx, &psi_row, beta_psi)
                    } else {
                        primary_direction_from_psi_row(block_idx, &psi_row, beta_psi)
                    };

                    let q_geom_lazy;
                    let (mut f_pi, mut f_pipi) = if let Some(primary) = flex_primary.as_ref() {
                        let q_ref = match q_geom.as_ref() {
                            Some(q) => q,
                            None => {
                                q_geom_lazy = self.row_dynamic_q_geometry(row, block_states)?;
                                &q_geom_lazy
                            }
                        };
                        let (_, g, h) = self.compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            q_ref,
                            primary,
                        )?;
                        (g, h)
                    } else if let Some(c) = cache {
                        let (g, h) = self.row_primary_gradient_hessian(row, c);
                        (g.clone(), h.clone())
                    } else {
                        let (_, g, h) =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (g, h)
                    };

                    // Third contracted derivative T_i[u^α].
                    let w = row_weights[row];
                    if w != 1.0 {
                        f_pi.mapv_inplace(|v| v * w);
                        f_pipi.mapv_inplace(|v| v * w);
                    }

                    let mut third =
                        self.row_primary_third_contracted_general(row, block_states, &dir)?;
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }

                    // ── Eq (45): objective_psi += f_i^T u_i^α ──
                    a.0 += f_pi.dot(&dir);

                    let s1 = f_pi.dot(&loading);
                    match block_idx {
                        1 => a.2.scaled_add(s1, &psi_row),
                        _ => a.3.scaled_add(s1, &psi_row),
                    }
                    let pb = f_pipi.dot(&dir);
                    if let Some(lift) = psi_lift.as_ref() {
                        Self::accumulate_score_timewiggle_psi_u(lift, &f_pi, &mut a.1, &mut a.2);
                    }
                    if let Some(q) = q_geom.as_ref() {
                        self.accumulate_score_with_q_geometry(
                            row, q, &pb, &mut a.1, &mut a.2, &mut a.3,
                        )?;
                    } else {
                        self.accumulate_score_blockwise(row, &pb, &mut a.1, &mut a.2, &mut a.3)?;
                    }
                    self.accumulate_score_identity_blocks(
                        flex_primary.as_ref(),
                        &pb,
                        Some(&mut a.4),
                        Some(&mut a.5),
                    );

                    let right_primary = f_pipi.dot(&loading);
                    if let Some(q) = q_geom.as_ref() {
                        a.6.add_rank1_psi_cross_with_q_geometry(
                            self,
                            row,
                            q,
                            block_idx,
                            &psi_row,
                            &right_primary,
                        )?;
                    } else {
                        a.6.add_rank1_psi_cross(self, row, block_idx, &psi_row, &right_primary)?;
                    }
                    if let Some(q) = q_geom.as_ref() {
                        let zero_grad = Array1::zeros(third.nrows());
                        a.6.add_pullback_with_q_geometry(self, row, q, &zero_grad, &third)?;
                    } else {
                        a.6.add_pullback(self, row, &third)?;
                    }
                    if let Some(lift) = psi_lift.as_ref() {
                        let q = q_geom.as_ref().unwrap();
                        a.6.add_timewiggle_psi_u_cross(self, row, q, lift, &f_pipi)?;
                        a.6.add_second_pullback_weighted(q, &pb);
                        a.6.add_timewiggle_psi_kappa_alpha(self, lift, &f_pi);
                    }

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

        // Assemble score into flat vector
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
            hessian_psi_operator: Some(std::sync::Arc::new(acc.into_operator(slices))),
        }))
    }

    /// Batched per-axis variant of `psi_terms_inner_with_options`. Performs a
    /// single rayon row pass that fetches the per-row primary gradient and
    /// Hessian once and folds them into every axis in lock-step, instead of
    /// the K serial row passes the per-axis path would issue.
    ///
    /// Returns `Ok(None)` when any branch the fast path does not cover is
    /// active (effective flex, timewiggle, or a sigma-aux index in the
    /// request list); callers fall back to the per-axis path. The simple
    /// spatial-only path (the large-scale survival marginal-slope workload) is
    /// the case this fast path targets.
    pub(crate) fn psi_terms_inner_batched_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_indices: &[usize],
        cache: Option<&EvalCache>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Vec<ExactNewtonJointPsiTerms>>, String> {
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            return Ok(None);
        }
        for &psi_index in psi_indices {
            if self.is_sigma_aux_index(derivative_blocks, psi_index) {
                return Ok(None);
            }
        }

        let k = psi_indices.len();
        if k == 0 {
            return Ok(Some(Vec::new()));
        }
        let slices = block_slices(self, block_states);

        // Per-axis context: psi map, primary-space loading, beta. Resolved
        // once outside the row pass so the row hot loop sees only borrows.
        struct AxisCtx<'a> {
            block_idx: usize,
            psi_map: crate::custom_family::PsiDesignMap,
            loading: Array1<f64>,
            beta_psi: &'a Array1<f64>,
        }
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let mut axes: Vec<AxisCtx<'_>> = Vec::with_capacity(k);
        for &psi_index in psi_indices {
            let Some((block_idx, local_idx, p_psi, psi_label)) =
                self.psi_block_info(derivative_blocks, psi_index)?
            else {
                // psi_block_info returning None means caller passed an index
                // that does not resolve to a known block; defer to per-axis
                // so behaviour matches `first_order_terms(psi_index)`.
                return Ok(None);
            };
            let deriv = &derivative_blocks[block_idx][local_idx];
            let psi_map = crate::custom_family::resolve_custom_family_x_psi_map(
                deriv,
                self.n,
                p_psi,
                0..self.n,
                psi_label,
                &policy,
            )?;
            let loading = spatial_block_primary_loading(block_idx)?;
            let beta_psi: &Array1<f64> = match block_idx {
                1 => &block_states[1].beta,
                _ => &block_states[2].beta,
            };
            axes.push(AxisCtx {
                block_idx,
                psi_map,
                loading,
                beta_psi,
            });
        }

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());

        struct BatchedPsiAxisAcc {
            objective_psi: f64,
            score_t: Array1<f64>,
            score_m: Array1<f64>,
            score_g: Array1<f64>,
            score_h: Array1<f64>,
            score_w: Array1<f64>,
            hessian: BlockHessianAccumulator,
        }
        let make_accs = || -> Vec<BatchedPsiAxisAcc> {
            (0..k)
                .map(|_| BatchedPsiAxisAcc {
                    objective_psi: 0.0,
                    score_t: Array1::zeros(p_t),
                    score_m: Array1::zeros(p_m),
                    score_g: Array1::zeros(p_g),
                    score_h: Array1::zeros(p_h),
                    score_w: Array1::zeros(p_w),
                    hessian: BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
                })
                .collect()
        };

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);

        let folded = chunked_row_reduction(
            row_iter.as_slice(),
            make_accs,
            |row, accs: &mut Vec<BatchedPsiAxisAcc>| -> Result<(), String> {
                let w = row_weights[row];

                // Fetch (f_pi, f_pipi) UNWEIGHTED once per row. Mutating
                // them in place between axes would double-weight every
                // axis after the first; instead each axis applies `w`
                // inline below.
                let (f_pi, f_pipi) = if let Some(c) = cache {
                    let (g, h) = self.row_primary_gradient_hessian(row, c);
                    (g.clone(), h.clone())
                } else {
                    let (_, g, h) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    (g, h)
                };

                for (axis_idx, axis) in axes.iter().enumerate() {
                    let psi_row = axis
                        .psi_map
                        .row_vector(row)
                        .map_err(|e| format!("survival rowwise psi map (batched): {e}"))?;
                    let dir =
                        primary_direction_from_psi_row(axis.block_idx, &psi_row, axis.beta_psi);
                    let mut third =
                        self.row_primary_third_contracted_general(row, block_states, &dir)?;
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }

                    let acc = &mut accs[axis_idx];

                    // objective_psi += w * (f_pi · dir)
                    acc.objective_psi += w * f_pi.dot(&dir);

                    // score_psi += w * (f_pi · loading) * psi_row, routed to
                    // the marginal or logslope block depending on axis.
                    let s1 = w * f_pi.dot(&axis.loading);
                    match axis.block_idx {
                        1 => acc.score_m.scaled_add(s1, &psi_row),
                        _ => acc.score_g.scaled_add(s1, &psi_row),
                    }

                    let mut pb = f_pipi.dot(&dir);
                    if w != 1.0 {
                        pb.mapv_inplace(|v| v * w);
                    }
                    self.accumulate_score_blockwise(
                        row,
                        &pb,
                        &mut acc.score_t,
                        &mut acc.score_m,
                        &mut acc.score_g,
                    )?;
                    self.accumulate_score_identity_blocks(
                        None,
                        &pb,
                        Some(&mut acc.score_h),
                        Some(&mut acc.score_w),
                    );

                    let mut right_primary = f_pipi.dot(&axis.loading);
                    if w != 1.0 {
                        right_primary.mapv_inplace(|v| v * w);
                    }
                    acc.hessian.add_rank1_psi_cross(
                        self,
                        row,
                        axis.block_idx,
                        &psi_row,
                        &right_primary,
                    )?;
                    acc.hessian.add_pullback(self, row, &third)?;
                }
                Ok(())
            },
            |total: &mut Vec<BatchedPsiAxisAcc>, chunk: Vec<BatchedPsiAxisAcc>| {
                for (t, c) in total.iter_mut().zip(chunk.into_iter()) {
                    t.objective_psi += c.objective_psi;
                    t.score_t += &c.score_t;
                    t.score_m += &c.score_m;
                    t.score_g += &c.score_g;
                    t.score_h += &c.score_h;
                    t.score_w += &c.score_w;
                    t.hessian.add(&c.hessian);
                }
            },
        )?;

        let mut out: Vec<ExactNewtonJointPsiTerms> = Vec::with_capacity(k);
        for acc in folded.into_iter() {
            let mut score_psi = Array1::zeros(slices.total);
            score_psi
                .slice_mut(s![slices.time.clone()])
                .assign(&acc.score_t);
            score_psi
                .slice_mut(s![slices.marginal.clone()])
                .assign(&acc.score_m);
            score_psi
                .slice_mut(s![slices.logslope.clone()])
                .assign(&acc.score_g);
            if let Some(range) = slices.score_warp.as_ref() {
                score_psi.slice_mut(s![range.clone()]).assign(&acc.score_h);
            }
            if let Some(range) = slices.link_dev.as_ref() {
                score_psi.slice_mut(s![range.clone()]).assign(&acc.score_w);
            }
            out.push(ExactNewtonJointPsiTerms {
                objective_psi: acc.objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(
                    acc.hessian.into_operator(slices.clone()),
                )),
            });
        }
        Ok(Some(out))
    }

    pub(crate) fn psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms_inner(block_states, derivative_blocks, psi_index, None)
    }

    pub(crate) fn psi_second_order_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms_inner_with_options(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `psi_second_order_terms_inner`. See
    /// `psi_terms_inner_with_options` for the row-iter / weighting contract.
    pub(crate) fn psi_second_order_terms_inner_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: Option<&EvalCache>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let flex_primary = flex_active.then(|| flex_primary_slices(self));
        let slices = block_slices(self, block_states);
        let Some((block_idx_i, local_idx_i, p_psi_i, label_i)) =
            self.psi_block_info(derivative_blocks, psi_i)?
        else {
            return Ok(None);
        };
        let Some((block_idx_j, local_idx_j, p_psi_j, label_j)) =
            self.psi_block_info(derivative_blocks, psi_j)?
        else {
            return Ok(None);
        };
        let deriv_i = &derivative_blocks[block_idx_i][local_idx_i];
        let deriv_j = &derivative_blocks[block_idx_j][local_idx_j];
        let loading_i = if let Some(primary) = flex_primary.as_ref() {
            spatial_block_primary_loading_flex(primary, block_idx_i)?
        } else {
            spatial_block_primary_loading(block_idx_i)?
        };
        let loading_j = if let Some(primary) = flex_primary.as_ref() {
            spatial_block_primary_loading_flex(primary, block_idx_j)?
        } else {
            spatial_block_primary_loading(block_idx_j)?
        };
        let beta_i = match block_idx_i {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let beta_j = match block_idx_j {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };

        let timewiggle_active = self.flex_timewiggle_active();
        let timewiggle_psi_i = timewiggle_active && block_idx_i == 1;
        let timewiggle_psi_j = timewiggle_active && block_idx_j == 1;

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let same_block = block_idx_i == block_idx_j;

        // Build psi design maps once outside the row loop; rowwise calls use
        // the direct row_vector(row) API.
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let psi_map_i = crate::custom_family::resolve_custom_family_x_psi_map(
            deriv_i,
            self.n,
            p_psi_i,
            0..self.n,
            label_i,
            &policy,
        )?;
        let psi_map_j = crate::custom_family::resolve_custom_family_x_psi_map(
            deriv_j,
            self.n,
            p_psi_j,
            0..self.n,
            label_j,
            &policy,
        )?;
        let psi_map_ij = if same_block {
            Some(crate::custom_family::resolve_custom_family_x_psi_psi_map(
                deriv_i,
                deriv_j,
                local_idx_j,
                self.n,
                p_psi_i,
                0..self.n,
                label_i,
                &policy,
            )?)
        } else {
            None
        };

        struct JointPsiSecondOrderAcc {
            objective_psi_psi: f64,
            score_t: Array1<f64>,
            score_m: Array1<f64>,
            score_g: Array1<f64>,
            score_h: Array1<f64>,
            score_w: Array1<f64>,
            hessian: BlockHessianAccumulator,
        }
        let make_acc = || -> JointPsiSecondOrderAcc {
            JointPsiSecondOrderAcc {
                objective_psi_psi: 0.0,
                score_t: Array1::zeros(p_t),
                score_m: Array1::zeros(p_m),
                score_g: Array1::zeros(p_g),
                score_h: Array1::zeros(p_h),
                score_w: Array1::zeros(p_w),
                hessian: BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
            }
        };

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Process fixed row chunks in parallel and merge local cross-block
        // accumulators in row-chunk order for deterministic timewiggle assembly.
        let JointPsiSecondOrderAcc {
            objective_psi_psi,
            score_t,
            score_m,
            score_g,
            score_h,
            score_w,
            hessian,
        } = chunked_row_reduction(
            row_iter.as_slice(),
            make_acc,
            |row, a| -> Result<(), String> {
                // Compute psi design rows once; derive directions from them.
                let psi_row_i = psi_map_i
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                let psi_row_j = psi_map_j
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;

                let q_geom = if timewiggle_active {
                    Some(self.row_dynamic_q_geometry(row, block_states)?)
                } else {
                    None
                };

                let psi_lift_i = if timewiggle_psi_i {
                    Some(self.timewiggle_marginal_psi_row_lift(
                        row,
                        block_states,
                        flex_primary.as_ref(),
                        &psi_row_i,
                        beta_i,
                    )?)
                } else {
                    None
                };
                let psi_lift_j = if timewiggle_psi_j {
                    Some(self.timewiggle_marginal_psi_row_lift(
                        row,
                        block_states,
                        flex_primary.as_ref(),
                        &psi_row_j,
                        beta_j,
                    )?)
                } else {
                    None
                };

                let dir_i = if let Some(lift) = psi_lift_i.as_ref() {
                    lift.dir.clone()
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_direction_from_psi_row_flex(primary, block_idx_i, &psi_row_i, beta_i)
                } else {
                    primary_direction_from_psi_row(block_idx_i, &psi_row_i, beta_i)
                };
                let dir_j = if let Some(lift) = psi_lift_j.as_ref() {
                    lift.dir.clone()
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_direction_from_psi_row_flex(primary, block_idx_j, &psi_row_j, beta_j)
                } else {
                    primary_direction_from_psi_row(block_idx_j, &psi_row_j, beta_j)
                };

                let (psi_row_ij, dir_ij) = if same_block {
                    let r = psi_map_ij
                        .as_ref()
                        .expect("psi_map_ij built when same_block")
                        .row_vector(row)
                        .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                    let d = if let Some(primary) = flex_primary.as_ref() {
                        primary_second_direction_from_psi_row_flex(primary, block_idx_i, &r, beta_i)
                    } else {
                        primary_second_direction_from_psi_row(block_idx_i, &r, beta_i)
                    };
                    (Some(r), d)
                } else {
                    (
                        None,
                        Array1::<f64>::zeros(
                            flex_primary
                                .as_ref()
                                .map_or(N_PRIMARY, |primary| primary.total),
                        ),
                    )
                };
                let has_ij = psi_row_ij
                    .as_ref()
                    .is_some_and(|r| r.iter().any(|v| v.abs() > 0.0));

                let q_geom_lazy;
                let (mut f_pi, mut f_pipi) = if let Some(primary) = flex_primary.as_ref() {
                    let q_ref = match q_geom.as_ref() {
                        Some(q) => q,
                        None => {
                            q_geom_lazy = self.row_dynamic_q_geometry(row, block_states)?;
                            &q_geom_lazy
                        }
                    };
                    let (_, g, h) = self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_ref,
                        primary,
                    )?;
                    (g, h)
                } else if let Some(c) = cache {
                    let (g, h) = self.row_primary_gradient_hessian(row, c);
                    (g.clone(), h.clone())
                } else {
                    let (_, g, h) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    (g, h)
                };
                let w = row_weights[row];
                if w != 1.0 {
                    f_pi.mapv_inplace(|v| v * w);
                    f_pipi.mapv_inplace(|v| v * w);
                }
                let mut third_i =
                    self.row_primary_third_contracted_general(row, block_states, &dir_i)?;
                let mut third_j =
                    self.row_primary_third_contracted_general(row, block_states, &dir_j)?;
                let mut fourth =
                    self.row_primary_fourth_contracted_general(row, block_states, &dir_i, &dir_j)?;
                if w != 1.0 {
                    third_i.mapv_inplace(|v| v * w);
                    third_j.mapv_inplace(|v| v * w);
                    fourth.mapv_inplace(|v| v * w);
                }

                a.objective_psi_psi += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);

                // Score
                if has_ij {
                    let s_ij = f_pi.dot(&loading_i);
                    let psi_ij = psi_row_ij.as_ref().unwrap();
                    match block_idx_i {
                        1 => a.score_m.scaled_add(s_ij, psi_ij),
                        _ => a.score_g.scaled_add(s_ij, psi_ij),
                    }
                }
                let s_i = loading_i.dot(&f_pipi.dot(&dir_j));
                match block_idx_i {
                    1 => a.score_m.scaled_add(s_i, &psi_row_i),
                    _ => a.score_g.scaled_add(s_i, &psi_row_i),
                }
                let s_j = loading_j.dot(&f_pipi.dot(&dir_i));
                match block_idx_j {
                    1 => a.score_m.scaled_add(s_j, &psi_row_j),
                    _ => a.score_g.scaled_add(s_j, &psi_row_j),
                }
                let pb1 = f_pipi.dot(&dir_ij);
                if let Some(q) = q_geom.as_ref() {
                    self.accumulate_score_with_q_geometry(
                        row,
                        q,
                        &pb1,
                        &mut a.score_t,
                        &mut a.score_m,
                        &mut a.score_g,
                    )?;
                } else {
                    self.accumulate_score_blockwise(
                        row,
                        &pb1,
                        &mut a.score_t,
                        &mut a.score_m,
                        &mut a.score_g,
                    )?;
                }
                self.accumulate_score_identity_blocks(
                    flex_primary.as_ref(),
                    &pb1,
                    Some(&mut a.score_h),
                    Some(&mut a.score_w),
                );
                let pb2 = third_i.dot(&dir_j);
                if let Some(q) = q_geom.as_ref() {
                    self.accumulate_score_with_q_geometry(
                        row,
                        q,
                        &pb2,
                        &mut a.score_t,
                        &mut a.score_m,
                        &mut a.score_g,
                    )?;
                } else {
                    self.accumulate_score_blockwise(
                        row,
                        &pb2,
                        &mut a.score_t,
                        &mut a.score_m,
                        &mut a.score_g,
                    )?;
                }
                self.accumulate_score_identity_blocks(
                    flex_primary.as_ref(),
                    &pb2,
                    Some(&mut a.score_h),
                    Some(&mut a.score_w),
                );

                // Hessian
                if has_ij {
                    let rp_ij = f_pipi.dot(&loading_i);
                    if let Some(q) = q_geom.as_ref() {
                        a.hessian.add_rank1_psi_cross_with_q_geometry(
                            self,
                            row,
                            q,
                            block_idx_i,
                            psi_row_ij.as_ref().unwrap(),
                            &rp_ij,
                        )?;
                    } else {
                        a.hessian.add_rank1_psi_cross(
                            self,
                            row,
                            block_idx_i,
                            psi_row_ij.as_ref().unwrap(),
                            &rp_ij,
                        )?;
                    }
                }
                let scalar_ij = loading_i.dot(&f_pipi.dot(&loading_j));
                a.hessian.add_psi_psi_outer(
                    block_idx_i,
                    &psi_row_i,
                    block_idx_j,
                    &psi_row_j,
                    scalar_ij,
                );
                let rp_i = third_j.t().dot(&loading_i);
                if let Some(q) = q_geom.as_ref() {
                    a.hessian.add_rank1_psi_cross_with_q_geometry(
                        self,
                        row,
                        q,
                        block_idx_i,
                        &psi_row_i,
                        &rp_i,
                    )?;
                } else {
                    a.hessian
                        .add_rank1_psi_cross(self, row, block_idx_i, &psi_row_i, &rp_i)?;
                }
                let rp_j = third_i.t().dot(&loading_j);
                if let Some(q) = q_geom.as_ref() {
                    a.hessian.add_rank1_psi_cross_with_q_geometry(
                        self,
                        row,
                        q,
                        block_idx_j,
                        &psi_row_j,
                        &rp_j,
                    )?;
                } else {
                    a.hessian
                        .add_rank1_psi_cross(self, row, block_idx_j, &psi_row_j, &rp_j)?;
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(fourth.nrows());
                    a.hessian
                        .add_pullback_with_q_geometry(self, row, q, &zero_grad, &fourth)?;
                } else {
                    a.hessian.add_pullback(self, row, &fourth)?;
                }
                let mut third_ij =
                    self.row_primary_third_contracted_general(row, block_states, &dir_ij)?;
                if w != 1.0 {
                    third_ij.mapv_inplace(|v| v * w);
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(third_ij.nrows());
                    a.hessian
                        .add_pullback_with_q_geometry(self, row, q, &zero_grad, &third_ij)?;
                } else {
                    a.hessian.add_pullback(self, row, &third_ij)?;
                }

                // Timewiggle psi corrections for ψ_i (terms 1,2,4,5 of eq 47)
                if let Some(lift_i) = psi_lift_i.as_ref() {
                    let q = q_geom.as_ref().unwrap();
                    // U_i^α cross terms with third_j Hessian
                    a.hessian
                        .add_timewiggle_psi_u_cross(self, row, q, lift_i, &third_j)?;
                    // Second pullback weighted by T_j[dir_j] applied to dir_i
                    let hu_i = f_pipi.dot(&dir_i);
                    a.hessian.add_second_pullback_weighted(q, &hu_i);
                    // K^{BC,α_i} weighted by gradient
                    a.hessian
                        .add_timewiggle_psi_kappa_alpha(self, lift_i, &f_pi);
                }
                // Timewiggle psi corrections for ψ_j
                if let Some(lift_j) = psi_lift_j.as_ref() {
                    let q = q_geom.as_ref().unwrap();
                    a.hessian
                        .add_timewiggle_psi_u_cross(self, row, q, lift_j, &third_i)?;
                    let hu_j = f_pipi.dot(&dir_j);
                    a.hessian.add_second_pullback_weighted(q, &hu_j);
                    if psi_lift_i.is_none() {
                        // Only add gradient-weighted K^α for j if we didn't already
                        // add it for i (when both are marginal, it's already covered)
                        a.hessian
                            .add_timewiggle_psi_kappa_alpha(self, lift_j, &f_pi);
                    }
                }

                Ok(())
            },
            |total, chunk| {
                total.objective_psi_psi += chunk.objective_psi_psi;
                total.score_t += &chunk.score_t;
                total.score_m += &chunk.score_m;
                total.score_g += &chunk.score_g;
                total.score_h += &chunk.score_h;
                total.score_w += &chunk.score_w;
                total.hessian.add(&chunk.hessian);
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
            hessian_psi_psi_operator: Some(Box::new(hessian.into_operator(slices))),
        }))
    }

    pub(crate) fn psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms_inner(block_states, derivative_blocks, psi_i, psi_j, None)
    }

    pub(crate) fn psi_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.psi_hessian_directional_derivative_with_options(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `psi_hessian_directional_derivative` that
    /// returns the dense block Hessian directional derivative. When
    /// `options.outer_score_subsample` is `Some`, only the masked rows are
    /// visited and the accumulator uses per-row Horvitz-Thompson
    /// inverse-inclusion weights before being densified.
    /// Shared engine for the per-ψ Hessian directional derivative. Runs the
    /// per-row accumulation once and returns the populated block-Hessian
    /// accumulator together with its block slices; the dense and operator
    /// public variants are thin adapters that scatter or wrap this single
    /// accumulator, so the dense and operator paths can never disagree.
    pub(crate) fn psi_hessian_directional_derivative_accumulator(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<(BlockHessianAccumulator, BlockSlices)>, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let flex_primary = flex_active.then(|| flex_primary_slices(self));
        let slices = block_slices(self, block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let loading = if let Some(primary) = flex_primary.as_ref() {
            spatial_block_primary_loading_flex(primary, block_idx)?
        } else {
            spatial_block_primary_loading(block_idx)?
        };
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let d_beta_block = match block_idx {
            1 => d_beta_flat.slice(s![slices.marginal.clone()]),
            _ => d_beta_flat.slice(s![slices.logslope.clone()]),
        };

        let timewiggle_active = self.flex_timewiggle_active();
        let timewiggle_psi = timewiggle_active && block_idx == 1;

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());

        // Build the psi design map once; rowwise calls use direct row_vector(row).
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let psi_map = crate::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            self.n,
            p_psi,
            0..self.n,
            psi_label,
            &policy,
        )?;

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Process fixed row chunks in parallel and merge local cross-block
        // accumulators in row-chunk order for deterministic timewiggle assembly.
        let acc = chunked_row_reduction(
            row_iter.as_slice(),
            || BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
            |row, acc| -> Result<(), String> {
                let psi_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;

                let q_geom = if timewiggle_active {
                    Some(self.row_dynamic_q_geometry(row, block_states)?)
                } else {
                    None
                };

                let psi_lift = if timewiggle_psi {
                    Some(self.timewiggle_marginal_psi_row_lift(
                        row,
                        block_states,
                        flex_primary.as_ref(),
                        &psi_row,
                        beta_psi,
                    )?)
                } else {
                    None
                };

                let psi_dir = if let Some(lift) = psi_lift.as_ref() {
                    lift.dir.clone()
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_direction_from_psi_row_flex(primary, block_idx, &psi_row, beta_psi)
                } else {
                    primary_direction_from_psi_row(block_idx, &psi_row, beta_psi)
                };
                let psi_action = if psi_lift.is_some() {
                    self.timewiggle_psi_action(
                        row,
                        block_states,
                        &slices,
                        flex_primary.as_ref(),
                        &psi_row,
                        beta_psi,
                        d_beta_flat,
                    )?
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_psi_action_from_psi_row_flex(primary, block_idx, &psi_row, d_beta_block)
                } else {
                    primary_psi_action_from_psi_row(block_idx, &psi_row, d_beta_block)
                };
                let row_dir = self.row_primary_direction_from_flat_dynamic(
                    row,
                    block_states,
                    &slices,
                    d_beta_flat,
                )?;
                let q_geom_lazy;
                let mut h_pi = if let Some(primary) = flex_primary.as_ref() {
                    let q_ref = match q_geom.as_ref() {
                        Some(q) => q,
                        None => {
                            q_geom_lazy = self.row_dynamic_q_geometry(row, block_states)?;
                            &q_geom_lazy
                        }
                    };
                    self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_ref,
                        primary,
                    )?
                    .2
                } else {
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?
                        .2
                };
                let w = row_weights[row];
                if w != 1.0 {
                    h_pi.mapv_inplace(|v| v * w);
                }
                let mut third_beta =
                    self.row_primary_third_contracted_general(row, block_states, &row_dir)?;
                let mut fourth = self.row_primary_fourth_contracted_general(
                    row,
                    block_states,
                    &row_dir,
                    &psi_dir,
                )?;
                if w != 1.0 {
                    third_beta.mapv_inplace(|v| v * w);
                    fourth.mapv_inplace(|v| v * w);
                }

                let right_primary = third_beta.t().dot(&loading);
                if let Some(q) = q_geom.as_ref() {
                    acc.add_rank1_psi_cross_with_q_geometry(
                        self,
                        row,
                        q,
                        block_idx,
                        &psi_row,
                        &right_primary,
                    )?;
                } else {
                    acc.add_rank1_psi_cross(self, row, block_idx, &psi_row, &right_primary)?;
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(fourth.nrows());
                    acc.add_pullback_with_q_geometry(self, row, q, &zero_grad, &fourth)?;
                } else {
                    acc.add_pullback(self, row, &fourth)?;
                }
                let mut third_action =
                    self.row_primary_third_contracted_general(row, block_states, &psi_action)?;
                if w != 1.0 {
                    third_action.mapv_inplace(|v| v * w);
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(third_action.nrows());
                    acc.add_pullback_with_q_geometry(self, row, q, &zero_grad, &third_action)?;
                } else {
                    acc.add_pullback(self, row, &third_action)?;
                }
                // Timewiggle psi corrections
                if let Some(lift) = psi_lift.as_ref() {
                    let q = q_geom.as_ref().unwrap();
                    // U^α cross with third_beta (D_β H term)
                    acc.add_timewiggle_psi_u_cross(self, row, q, lift, &third_beta)?;
                    let second_pullback_weight = third_beta.dot(&psi_dir) + h_pi.dot(&psi_action);
                    acc.add_second_pullback_weighted(q, &second_pullback_weight);
                    let kappa_weight = h_pi.dot(&row_dir);
                    acc.add_timewiggle_psi_kappa_alpha(self, lift, &kappa_weight);
                }
                Ok(())
            },
            |total, chunk| {
                total.add(&chunk);
            },
        )?;

        Ok(Some((acc, slices)))
    }

    /// Outer-aware variant of `psi_hessian_directional_derivative` that
    /// returns the dense block Hessian directional derivative. When
    /// `options.outer_score_subsample` is `Some`, only the masked rows are
    /// visited and the accumulator uses per-row Horvitz-Thompson
    /// inverse-inclusion weights before being densified. Thin adapter over
    /// [`Self::psi_hessian_directional_derivative_accumulator`].
    pub(crate) fn psi_hessian_directional_derivative_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .psi_hessian_directional_derivative_accumulator(
                block_states,
                derivative_blocks,
                psi_index,
                d_beta_flat,
                options,
            )?
            .map(|(acc, slices)| acc.to_dense(&slices)))
    }

    /// Outer-aware operator builder for the per-ψ Hessian directional
    /// derivative. When `options.outer_score_subsample` is `Some`, only the
    /// sampled rows are visited and the accumulator uses per-row
    /// Horvitz-Thompson inverse-inclusion weights before being wrapped in the
    /// `HyperOperator`.
    pub(crate) fn psi_hessian_directional_derivative_operator_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        Ok(self
            .psi_hessian_directional_derivative_accumulator(
                block_states,
                derivative_blocks,
                psi_index,
                d_beta_flat,
                options,
            )?
            .map(|(acc, slices)| Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>))
    }
}
